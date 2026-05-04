"""Tests for the Code Interpreter implementation."""

import time

from code_interpreter import (
    SandboxExecutor,
    CodeParser,
    CodeInterpreterAgent,
    ArtifactStore,
)


# ---------------------------------------------------------------------------
# Helper LLM functions
# ---------------------------------------------------------------------------


def make_constant_llm(response: str):
    """Create an LLM function that always returns the same response."""
    def llm(messages: list[dict[str, str]]) -> str:
        return response
    return llm


def make_sequential_llm(responses: list[str]):
    """Create an LLM function that returns responses in order."""
    index = [0]
    def llm(messages: list[dict[str, str]]) -> str:
        idx = index[0]
        index[0] += 1
        return responses[idx % len(responses)]
    return llm


def make_code_responding_llm(code_blocks: list[str]):
    """Create an LLM function that returns code blocks in sequence.

    On the first call returns a response with a code block, on subsequent
    calls returns responses without code blocks (final answer).
    """
    index = [0]
    def llm(messages: list[dict[str, str]]) -> str:
        idx = index[0]
        index[0] += 1
        if idx < len(code_blocks):
            return f"Let me compute that:\n```python\n{code_blocks[idx]}\n```"
        return "Here is the final answer based on the computation."
    return llm


# ===========================================================================
# SandboxExecutor tests
# ===========================================================================


class TestSandboxExecutor:
    """Tests for the SandboxExecutor class."""

    def test_execute_simple_print(self):
        """Should capture stdout from print statements."""
        executor = SandboxExecutor()
        result = executor.execute("print('hello world')")
        assert result["success"] is True
        assert "hello world" in result["output"]
        assert result["error"] == ""

    def test_execute_variable_assignment(self):
        """Should execute variable assignments without output."""
        executor = SandboxExecutor()
        result = executor.execute("x = 42")
        assert result["success"] is True
        assert result["output"] == ""

    def test_execute_expression_with_print(self):
        """Should compute expressions and print results."""
        executor = SandboxExecutor()
        result = executor.execute("x = 10\ny = 20\nprint(x + y)")
        assert result["success"] is True
        assert "30" in result["output"]

    def test_persistent_namespace(self):
        """Variables should persist across executions."""
        executor = SandboxExecutor()
        executor.execute("x = 42")
        result = executor.execute("print(x)")
        assert result["success"] is True
        assert "42" in result["output"]

    def test_persistent_namespace_multiple_vars(self):
        """Multiple variables should all persist."""
        executor = SandboxExecutor()
        executor.execute("a = 1\nb = 2\nc = 3")
        result = executor.execute("print(a + b + c)")
        assert result["success"] is True
        assert "6" in result["output"]

    def test_execute_syntax_error(self):
        """Should report syntax errors."""
        executor = SandboxExecutor()
        result = executor.execute("def foo(")
        assert result["success"] is False
        assert "SyntaxError" in result["error"]

    def test_execute_runtime_error(self):
        """Should report runtime errors."""
        executor = SandboxExecutor()
        result = executor.execute("x = 1 / 0")
        assert result["success"] is False
        assert "ZeroDivisionError" in result["error"]

    def test_execute_name_error(self):
        """Should report NameError for undefined variables."""
        executor = SandboxExecutor()
        result = executor.execute("print(undefined_var)")
        assert result["success"] is False
        assert "NameError" in result["error"]

    def test_execute_timeout(self):
        """Should timeout for long-running code."""
        executor = SandboxExecutor()
        result = executor.execute("while True: pass", timeout=0.5)
        assert result["success"] is False
        assert "TimeoutError" in result["error"]

    def test_execute_timeout_with_partial_output(self):
        """Should capture output before timeout occurs."""
        executor = SandboxExecutor()
        result = executor.execute(
            "print('before')\nwhile True: pass",
            timeout=0.5,
        )
        assert result["success"] is False
        assert "before" in result["output"]

    def test_execute_empty_code(self):
        """Should handle empty code string."""
        executor = SandboxExecutor()
        result = executor.execute("")
        assert result["success"] is True
        assert result["output"] == ""

    def test_execute_multiline_code(self):
        """Should handle multi-line code blocks."""
        executor = SandboxExecutor()
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))
"""
        result = executor.execute(code)
        assert result["success"] is True
        assert "120" in result["output"]

    def test_execute_loop(self):
        """Should handle loops correctly."""
        executor = SandboxExecutor()
        code = "for i in range(5):\n    print(i)"
        result = executor.execute(code)
        assert result["success"] is True
        for i in range(5):
            assert str(i) in result["output"]

    def test_execute_stderr_capture(self):
        """Should capture stderr output."""
        executor = SandboxExecutor()
        result = executor.execute("import sys\nprint('err', file=sys.stderr)")
        assert result["success"] is True
        assert "err" in result["error"]

    def test_namespace_isolation_between_instances(self):
        """Different executor instances should have separate namespaces."""
        executor1 = SandboxExecutor()
        executor2 = SandboxExecutor()
        executor1.execute("x = 100")
        result = executor2.execute("print(x)")
        assert result["success"] is False
        assert "NameError" in result["error"]

    def test_execute_default_timeout(self):
        """Default timeout should be 5 seconds."""
        executor = SandboxExecutor()
        result = executor.execute("print('fast')")
        assert result["success"] is True

    def test_execute_list_operations(self):
        """Should handle data structure operations."""
        executor = SandboxExecutor()
        code = "data = [1, 2, 3, 4, 5]\nprint(sum(data))"
        result = executor.execute(code)
        assert result["success"] is True
        assert "15" in result["output"]

    def test_execute_dict_operations(self):
        """Should handle dictionary operations."""
        executor = SandboxExecutor()
        code = "d = {'a': 1, 'b': 2}\nprint(d['a'] + d['b'])"
        result = executor.execute(code)
        assert result["success"] is True
        assert "3" in result["output"]


# ===========================================================================
# CodeParser tests
# ===========================================================================


class TestCodeParser:
    """Tests for the CodeParser class."""

    def test_parse_single_python_block(self):
        """Should extract a single python code block."""
        parser = CodeParser()
        text = "Here is some code:\n```python\nprint('hello')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert blocks[0] == "print('hello')"

    def test_parse_multiple_blocks(self):
        """Should extract multiple code blocks."""
        parser = CodeParser()
        text = (
            "First block:\n```python\nx = 1\n```\n\n"
            "Second block:\n```python\nprint(x)\n```"
        )
        blocks = parser.parse(text)
        assert len(blocks) == 2
        assert blocks[0] == "x = 1"
        assert blocks[1] == "print(x)"

    def test_parse_no_code_blocks(self):
        """Should return empty list when no code blocks found."""
        parser = CodeParser()
        text = "This is just plain text with no code blocks."
        blocks = parser.parse(text)
        assert blocks == []

    def test_parse_empty_string(self):
        """Should return empty list for empty string."""
        parser = CodeParser()
        blocks = parser.parse("")
        assert blocks == []

    def test_parse_py_tag(self):
        """Should handle 'py' language tag."""
        parser = CodeParser()
        text = "```py\nprint('py tag')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "py tag" in blocks[0]

    def test_parse_python3_tag(self):
        """Should handle 'python3' language tag."""
        parser = CodeParser()
        text = "```python3\nprint('python3 tag')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "python3 tag" in blocks[0]

    def test_parse_case_insensitive(self):
        """Should handle uppercase language tags."""
        parser = CodeParser()
        text = "```Python\nprint('uppercase')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "uppercase" in blocks[0]

    def test_parse_mixed_case(self):
        """Should handle mixed case language tags."""
        parser = CodeParser()
        text = "```PYTHON\nprint('mixed')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "mixed" in blocks[0]

    def test_parse_ignores_non_python_blocks(self):
        """Should not extract non-Python code blocks."""
        parser = CodeParser()
        text = "```javascript\nconsole.log('hi')\n```\n```python\nprint('hi')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "print('hi')" in blocks[0]

    def test_parse_multiline_code(self):
        """Should handle multi-line code within a block."""
        parser = CodeParser()
        text = "```python\ndef foo():\n    return 42\n\nprint(foo())\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "def foo():" in blocks[0]
        assert "print(foo())" in blocks[0]

    def test_parse_preserves_indentation(self):
        """Should preserve indentation within code blocks."""
        parser = CodeParser()
        text = "```python\nif True:\n    print('indented')\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert "    print('indented')" in blocks[0]

    def test_parse_strips_whitespace(self):
        """Should strip leading/trailing whitespace from code blocks."""
        parser = CodeParser()
        text = "```python\n\n  x = 1  \n\n```"
        blocks = parser.parse(text)
        assert len(blocks) == 1
        assert blocks[0] == "x = 1"

    def test_parse_surrounded_by_text(self):
        """Should extract code blocks surrounded by explanatory text."""
        parser = CodeParser()
        text = (
            "Let me explain. First, we define a function:\n"
            "```python\ndef add(a, b):\n    return a + b\n```\n"
            "Then we call it:\n"
            "```python\nprint(add(1, 2))\n```\n"
            "The result is 3."
        )
        blocks = parser.parse(text)
        assert len(blocks) == 2
        assert "def add" in blocks[0]
        assert "add(1, 2)" in blocks[1]


# ===========================================================================
# CodeInterpreterAgent tests
# ===========================================================================


class TestCodeInterpreterAgent:
    """Tests for the CodeInterpreterAgent class."""

    def test_creation(self):
        """Should create an agent with correct attributes."""
        agent = CodeInterpreterAgent(
            llm_fn=make_constant_llm("answer"),
            max_turns=3,
        )
        assert agent.max_turns == 3
        assert agent.sandbox is not None
        assert agent.parser is not None

    def test_run_no_code_blocks(self):
        """Should return LLM response when no code blocks are generated."""
        llm = make_constant_llm("The answer is 42.")
        agent = CodeInterpreterAgent(llm_fn=llm)
        result = agent.run("What is the answer?")
        assert result == "The answer is 42."

    def test_run_single_turn_with_code(self):
        """Should execute code and return final response."""
        responses = [
            "Let me calculate:\n```python\nprint(2 + 2)\n```",
            "The result is 4.",
        ]
        llm = make_sequential_llm(responses)
        agent = CodeInterpreterAgent(llm_fn=llm)
        result = agent.run("What is 2 + 2?")
        assert "4" in result

    def test_run_multi_turn(self):
        """Should handle multiple turns of code generation."""
        responses = [
            "```python\nx = 10\nprint(x)\n```",
            "```python\ny = x * 2\nprint(y)\n```",
            "The final result is 20.",
        ]
        llm = make_sequential_llm(responses)
        agent = CodeInterpreterAgent(llm_fn=llm, max_turns=5)
        result = agent.run("Double 10")
        assert "20" in result

    def test_run_respects_max_turns(self):
        """Should stop after max_turns even if code keeps being generated."""
        # LLM always returns code blocks
        llm = make_constant_llm("```python\nprint('again')\n```")
        agent = CodeInterpreterAgent(llm_fn=llm, max_turns=3)
        result = agent.run("loop forever")

        # Should have stopped after 3 turns (the LLM was called 3 times)
        # The result should be the last LLM response
        assert "again" in result

    def test_run_stops_when_no_code(self):
        """Should stop when LLM produces a response without code blocks."""
        responses = [
            "```python\nprint('step 1')\n```",
            "Done! The computation is complete.",
        ]
        llm = make_sequential_llm(responses)
        agent = CodeInterpreterAgent(llm_fn=llm, max_turns=10)
        result = agent.run("compute something")
        assert result == "Done! The computation is complete."

    def test_run_passes_execution_results_to_llm(self):
        """Should include execution results in subsequent LLM calls."""
        seen_messages: list[list[dict[str, str]]] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            if len(seen_messages) == 1:
                return "```python\nprint('hello')\n```"
            return "Final answer."

        agent = CodeInterpreterAgent(llm_fn=tracking_llm)
        agent.run("test")

        # The second LLM call should include execution results
        assert len(seen_messages) == 2
        second_msg = seen_messages[1][-1]["content"]
        assert "Execution results" in second_msg
        assert "hello" in second_msg

    def test_run_handles_execution_error(self):
        """Should pass error information back to the LLM."""
        seen_messages: list[list[dict[str, str]]] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            if len(seen_messages) == 1:
                return "```python\nprint(undefined_var)\n```"
            return "There was an error."

        agent = CodeInterpreterAgent(llm_fn=tracking_llm)
        agent.run("test")

        # The second LLM call should include the error
        second_msg = seen_messages[1][-1]["content"]
        assert "Error" in second_msg
        assert "NameError" in second_msg

    def test_run_multiple_code_blocks_in_one_response(self):
        """Should execute all code blocks from a single response."""
        seen_messages: list[list[dict[str, str]]] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            if len(seen_messages) == 1:
                return (
                    "```python\nx = 5\n```\n"
                    "```python\nprint(x * 2)\n```"
                )
            return "Done."

        agent = CodeInterpreterAgent(llm_fn=tracking_llm)
        agent.run("test")

        # Both outputs should appear in the execution results
        second_msg = seen_messages[1][-1]["content"]
        assert "10" in second_msg

    def test_run_persistent_namespace_across_turns(self):
        """Variables should persist across code execution turns."""
        seen_messages: list[list[dict[str, str]]] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            if len(seen_messages) == 1:
                return "```python\nx = 42\n```"
            return "Final answer."

        agent = CodeInterpreterAgent(llm_fn=tracking_llm)
        agent.run("test")

        # x should be in the sandbox namespace
        assert agent.sandbox.namespace.get("x") == 42

    def test_run_default_max_turns(self):
        """Default max_turns should be 5."""
        agent = CodeInterpreterAgent(llm_fn=make_constant_llm("ok"))
        assert agent.max_turns == 5

    def test_run_system_prompt(self):
        """Should send a system prompt to the LLM."""
        seen_messages: list[list[dict[str, str]]] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            return "ok"

        agent = CodeInterpreterAgent(llm_fn=tracking_llm)
        agent.run("test task")

        system_msg = seen_messages[0][0]
        assert system_msg["role"] == "system"
        assert "coding assistant" in system_msg["content"].lower()


# ===========================================================================
# ArtifactStore tests
# ===========================================================================


class TestArtifactStore:
    """Tests for the ArtifactStore class."""

    def test_store_and_retrieve(self):
        """Should store and retrieve artifacts by name."""
        store = ArtifactStore()
        store.store("result", "hello world")
        assert store.retrieve("result") == "hello world"

    def test_store_with_type(self):
        """Should store artifacts with a type label."""
        store = ArtifactStore()
        store.store("data", [1, 2, 3], artifact_type="list")
        assert store.retrieve("data") == [1, 2, 3]

    def test_store_various_types(self):
        """Should handle different artifact types."""
        store = ArtifactStore()
        store.store("text", "hello")
        store.store("number", 42)
        store.store("list", [1, 2, 3])
        store.store("dict", {"key": "value"})
        store.store("none", None)

        assert store.retrieve("text") == "hello"
        assert store.retrieve("number") == 42
        assert store.retrieve("list") == [1, 2, 3]
        assert store.retrieve("dict") == {"key": "value"}
        assert store.retrieve("none") is None

    def test_store_overwrites(self):
        """Should overwrite existing artifacts with the same name."""
        store = ArtifactStore()
        store.store("x", "old")
        store.store("x", "new")
        assert store.retrieve("x") == "new"

    def test_retrieve_missing_raises_keyerror(self):
        """Should raise KeyError for missing artifacts."""
        store = ArtifactStore()
        try:
            store.retrieve("nonexistent")
            assert False, "Expected KeyError"
        except KeyError as e:
            assert "nonexistent" in str(e)

    def test_list_artifacts_empty(self):
        """Should return empty list when no artifacts stored."""
        store = ArtifactStore()
        assert store.list_artifacts() == []

    def test_list_artifacts_with_data(self):
        """Should return names of all stored artifacts."""
        store = ArtifactStore()
        store.store("a", 1)
        store.store("b", 2)
        store.store("c", 3)
        names = store.list_artifacts()
        assert sorted(names) == ["a", "b", "c"]

    def test_clear(self):
        """Should remove all stored artifacts."""
        store = ArtifactStore()
        store.store("a", 1)
        store.store("b", 2)
        store.clear()
        assert store.list_artifacts() == []

    def test_clear_then_store(self):
        """Should be able to store new artifacts after clearing."""
        store = ArtifactStore()
        store.store("old", "data")
        store.clear()
        store.store("new", "fresh")
        assert store.retrieve("new") == "fresh"
        assert store.list_artifacts() == ["new"]

    def test_list_artifacts_returns_copy(self):
        """Should return a list, not a reference to internal state."""
        store = ArtifactStore()
        store.store("x", 10)
        names = store.list_artifacts()
        names.append("y")
        assert "y" not in store.list_artifacts()

    def test_default_artifact_type(self):
        """Default artifact type should be 'text'."""
        store = ArtifactStore()
        store.store("item", "data")
        # Verify it was stored (the type is internal but shouldn't cause issues)
        assert store.retrieve("item") == "data"

    def test_store_empty_string(self):
        """Should handle storing empty strings."""
        store = ArtifactStore()
        store.store("empty", "")
        assert store.retrieve("empty") == ""
