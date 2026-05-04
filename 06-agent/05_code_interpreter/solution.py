"""Reference solution for the Code Interpreter exercise."""

import io
import re
import threading
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Callable


class SandboxExecutorSolution:
    """Execute Python code in an isolated namespace with timeout protection."""

    def __init__(self) -> None:
        """Initialize the sandbox with a restricted namespace."""
        self.namespace: dict[str, Any] = {"__builtins__": __builtins__}

    def execute(self, code: str, timeout: float = 5.0) -> dict[str, Any]:
        """Execute Python code in the sandbox and return the result."""
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


class CodeParserSolution:
    """Extract Python code blocks from LLM markdown output."""

    _PATTERN = re.compile(
        r"```(?:python|py|python3)\s*\n(.*?)```",
        re.DOTALL | re.IGNORECASE,
    )

    def parse(self, text: str) -> list[str]:
        """Extract Python code blocks from the given text."""
        matches = self._PATTERN.findall(text)
        return [match.strip() for match in matches]


class CodeInterpreterAgentSolution:
    """LLM-backed agent that generates code, executes it, and iterates."""

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        max_turns: int = 5,
    ) -> None:
        """Initialize the code interpreter agent."""
        self.llm_fn = llm_fn
        self.max_turns = max_turns
        self.sandbox = SandboxExecutorSolution()
        self.parser = CodeParserSolution()

    def run(self, task: str) -> str:
        """Run the code generation-execution loop on the given task."""
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


class ArtifactStoreSolution:
    """Store and retrieve execution results and artifacts."""

    def __init__(self) -> None:
        """Initialize an empty artifact store."""
        self._artifacts: dict[str, Any] = {}
        self._types: dict[str, str] = {}

    def store(self, name: str, data: Any, artifact_type: str = "text") -> None:
        """Store an artifact under the given name."""
        self._artifacts[name] = data
        self._types[name] = artifact_type

    def retrieve(self, name: str) -> Any:
        """Retrieve an artifact by name."""
        if name not in self._artifacts:
            raise KeyError(f"Artifact '{name}' not found.")
        return self._artifacts[name]

    def list_artifacts(self) -> list[str]:
        """List the names of all stored artifacts."""
        return list(self._artifacts.keys())

    def clear(self) -> None:
        """Remove all stored artifacts."""
        self._artifacts.clear()
        self._types.clear()
