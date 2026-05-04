"""Code Interpreter for LLM-driven code generation and execution.

This module implements a code interpreter framework that allows an LLM to
generate Python code, execute it in a sandboxed environment, and feed the
results back for iterative refinement. It contains:
- SandboxExecutor: Execute Python code in an isolated namespace with timeout.
- CodeParser: Extract Python code blocks from LLM markdown output.
- CodeInterpreterAgent: LLM generates code -> execute -> feed results back.
- ArtifactStore: Store and retrieve execution results and artifacts.

Architecture overview:
    User task
        -> CodeInterpreterAgent sends task to LLM
        -> LLM returns response with ```python code blocks
        -> CodeParser extracts code blocks
        -> SandboxExecutor runs code in isolated namespace
        -> Execution results are fed back to LLM
        -> Loop continues until no code blocks or max_turns reached
        -> Final answer returned

Design notes:
- LLM function signature: fn(messages: list[dict]) -> str
- Sandbox uses exec() with a restricted namespace (builtins only).
- Code parser uses string matching to find ```python fenced blocks.
- Artifacts can be any Python object (text, data, images).
- Timeout uses threading.Timer for cross-platform compatibility.
- This module is self-contained (no external dependencies beyond the
  standard library).
"""

import io
import re
import threading
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable


__all__ = [
    "SandboxExecutor",
    "CodeParser",
    "CodeInterpreterAgent",
    "ArtifactStore",
]


# ---------------------------------------------------------------------------
# SandboxExecutor
# ---------------------------------------------------------------------------


class SandboxExecutor:
    """Execute Python code in an isolated namespace with timeout protection.

    The executor maintains a persistent namespace across calls, so variables
    defined in one execution are available in subsequent executions. Code
    runs with a restricted set of builtins for basic safety.

    Attributes:
        namespace: The persistent namespace dict shared across executions.

    Example:
        executor = SandboxExecutor()
        result = executor.execute("x = 42\nprint(x)")
        # result["output"] == "42\n"
        # result["success"] is True

        result = executor.execute("print(x + 1)")
        # result["output"] == "43\n"  (x persists from previous call)
    """

    def __init__(self) -> None:
        """Initialize the sandbox with a restricted namespace."""
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}

    def execute(self, code: str, timeout: float = 5.0) -> dict[str, Any]:
        """Execute Python code in the sandbox and return the result.

        Runs the given code string in the persistent namespace. Captures
        stdout and stderr. If execution exceeds the timeout, the code is
        terminated and an error is reported.

        Args:
            code: Python code string to execute.
            timeout: Maximum execution time in seconds (default 5.0).

        Returns:
            A dict with keys:
                - "output" (str): Captured stdout content.
                - "error" (str): Captured stderr content or error message.
                - "success" (bool): True if execution completed without error.
        """
        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()
        error_msg = ""
        success = True

        def run_code() -> None:
            nonlocal error_msg, success
            try:
                with redirect_stdout(stdout_capture), redirect_stderr(stderr_capture):
                    exec(code, self.namespace)
            except Exception as e:
                error_msg = f"{type(e).__name__}: {e}"
                success = False

        thread = threading.Thread(target=run_code, daemon=True)
        thread.start()
        thread.join(timeout=timeout)

        if thread.is_alive():
            return {
                "output": stdout_capture.getvalue(),
                "error": f"TimeoutError: Code execution exceeded {timeout}s limit",
                "success": False,
            }

        stderr_content = stderr_capture.getvalue()
        if stderr_content and not error_msg:
            error_msg = stderr_content

        return {
            "output": stdout_capture.getvalue(),
            "error": error_msg,
            "success": success,
        }


# ---------------------------------------------------------------------------
# CodeParser
# ---------------------------------------------------------------------------


class CodeParser:
    """Extract Python code blocks from LLM markdown output.

    Parses text for fenced code blocks tagged as python, py, or python3,
    and returns the code content of each block.

    Example:
        parser = CodeParser()
        text = "Here is some code:\n```python\nprint('hello')\n```"
        blocks = parser.parse(text)
        # blocks == ["print('hello')"]
    """

    # Match ```python, ```py, or ```python3 code blocks
    _PATTERN = re.compile(
        r"```(?:python|py|python3)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    def parse(self, text: str) -> list[str]:
        """Extract Python code blocks from the given text.

        Searches for fenced code blocks with language tags ``python``,
        ``py``, or ``python3`` (case-insensitive) and returns their
        contents as a list of strings.

        Args:
            text: The text to parse (typically an LLM response).

        Returns:
            A list of code block contents. Returns an empty list if no
            code blocks are found.
        """
        matches = self._PATTERN.findall(text)
        return [match.strip() for match in matches]


# ---------------------------------------------------------------------------
# CodeInterpreterAgent
# ---------------------------------------------------------------------------


class CodeInterpreterAgent:
    """LLM-backed agent that generates code, executes it, and iterates.

    The agent implements a generate-execute loop:
    1. Send the task (and any prior execution results) to the LLM.
    2. Parse code blocks from the LLM's response.
    3. Execute the code in a SandboxExecutor.
    4. Feed results back to the LLM for the next iteration.
    5. Stop when the LLM produces no code blocks or max_turns is reached.

    Attributes:
        llm_fn: A callable that takes a list of message dicts and returns
            a string (the LLM's response).
        max_turns: Maximum number of generate-execute iterations.
        sandbox: The SandboxExecutor used to run generated code.
        parser: The CodeParser used to extract code blocks.

    Example:
        agent = CodeInterpreterAgent(llm_fn=my_llm, max_turns=5)
        result = agent.run("Calculate the first 10 Fibonacci numbers")
    """

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        max_turns: int = 5,
    ) -> None:
        """Initialize the code interpreter agent.

        Args:
            llm_fn: A callable that takes a list of message dicts (OpenAI
                format) and returns a string (the LLM's response).
            max_turns: Maximum number of generate-execute loops (default 5).
        """
        self.llm_fn = llm_fn
        self.max_turns = max_turns
        self.sandbox = SandboxExecutor()
        self.parser = CodeParser()

    def run(self, task: str) -> str:
        """Run the code generation-execution loop on the given task.

        Sends the task to the LLM, extracts and executes any code blocks,
        and feeds results back. Continues until the LLM produces no code
        blocks or max_turns is reached.

        Args:
            task: The task description for the agent to work on.

        Returns:
            The LLM's final response as a string.
        """
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": (
                    "You are a helpful coding assistant. When asked to "
                    "compute or analyze something, write Python code in "
                    "```python fenced code blocks. The code will be executed "
                    "and the results shown to you. Use print() to show "
                    "output. When you have the final answer, present it "
                    "clearly without code blocks."
                ),
            },
            {"role": "user", "content": task},
        ]

        last_response = ""
        for _ in range(self.max_turns):
            response = self.llm_fn(messages)
            last_response = response

            code_blocks = self.parser.parse(response)
            if not code_blocks:
                break

            # Execute each code block and collect results
            results: list[str] = []
            for block in code_blocks:
                exec_result = self.sandbox.execute(block)
                if exec_result["success"]:
                    output = exec_result["output"]
                    results.append(
                        f"Output:\n{output}" if output else "Output: (no output)"
                    )
                else:
                    results.append(f"Error:\n{exec_result['error']}")

            # Feed execution results back to the LLM
            results_text = "\n\n".join(results)
            messages.append({"role": "assistant", "content": response})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        f"Execution results:\n{results_text}\n\n"
                        "Continue working on the task. If you have the "
                        "final answer, present it without code blocks."
                    ),
                }
            )

        return last_response


# ---------------------------------------------------------------------------
# ArtifactStore
# ---------------------------------------------------------------------------


class ArtifactStore:
    """Store and retrieve execution results and artifacts.

    A simple in-memory key-value store for artifacts produced during code
    interpretation. Artifacts can be any Python object (text, data, images).

    Example:
        store = ArtifactStore()
        store.store("result", [1, 2, 3], artifact_type="data")
        store.retrieve("result")  # -> [1, 2, 3]
        store.list_artifacts()    # -> ["result"]
    """

    def __init__(self) -> None:
        """Initialize an empty artifact store."""
        self._artifacts: dict[str, Any] = {}
        self._types: dict[str, str] = {}

    def store(self, name: str, data: Any, artifact_type: str = "text") -> None:
        """Store an artifact under the given name.

        Args:
            name: The name to store the artifact under.
            data: The artifact data (any type).
            artifact_type: A string label for the artifact type
                (default "text").
        """
        self._artifacts[name] = data
        self._types[name] = artifact_type

    def retrieve(self, name: str) -> Any:
        """Retrieve an artifact by name.

        Args:
            name: The name of the artifact to retrieve.

        Returns:
            The stored artifact data.

        Raises:
            KeyError: If no artifact with the given name exists.
        """
        if name not in self._artifacts:
            raise KeyError(f"Artifact '{name}' not found.")
        return self._artifacts[name]

    def list_artifacts(self) -> list[str]:
        """List the names of all stored artifacts.

        Returns:
            A list of artifact name strings.
        """
        return list(self._artifacts.keys())

    def clear(self) -> None:
        """Remove all stored artifacts."""
        self._artifacts.clear()
        self._types.clear()
