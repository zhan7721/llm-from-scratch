"""Tests for the Tool Calling implementation."""

import json
from tool_calling import (
    ToolRegistry,
    ToolParser,
    ToolExecutor,
    ToolCallingAgent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_registry() -> ToolRegistry:
    """Create a ToolRegistry with a few sample tools."""
    registry = ToolRegistry()
    registry.register(
        name="add",
        fn=lambda a, b: a + b,
        description="Add two numbers",
        parameters={
            "a": {"type": "number", "description": "First number"},
            "b": {"type": "number", "description": "Second number"},
        },
    )
    registry.register(
        name="greet",
        fn=lambda name: f"Hello, {name}!",
        description="Greet a person",
        parameters={
            "name": {"type": "string", "description": "Person's name"},
        },
    )
    registry.register(
        name="fail",
        fn=lambda: (_ for _ in ()).throw(ValueError("intentional error")),
        description="A tool that always fails",
        parameters={},
    )
    return registry


# ===========================================================================
# ToolRegistry tests
# ===========================================================================


class TestToolRegistry:
    """Tests for the ToolRegistry class."""

    def test_register_and_get(self):
        """Registered tools should be retrievable by name."""
        registry = ToolRegistry()
        fn = lambda x: x
        registry.register("echo", fn, "Echo input", {"x": {"type": "string"}})

        tool = registry.get("echo")
        assert tool is not None
        assert tool["name"] == "echo"
        assert tool["description"] == "Echo input"
        assert tool["fn"] is fn

    def test_get_nonexistent(self):
        """Getting a nonexistent tool should return None."""
        registry = ToolRegistry()
        assert registry.get("missing") is None

    def test_list_tools_empty(self):
        """Empty registry should return empty list."""
        registry = ToolRegistry()
        assert registry.list_tools() == []

    def test_list_tools_sorted(self):
        """list_tools should return sorted tool names."""
        registry = ToolRegistry()
        registry.register("zeta", lambda: None, "z", {})
        registry.register("alpha", lambda: None, "a", {})
        registry.register("mu", lambda: None, "m", {})

        names = registry.list_tools()
        assert names == ["alpha", "mu", "zeta"]

    def test_get_schemas(self):
        """get_schemas should return schemas for all registered tools."""
        registry = ToolRegistry()
        params = {"x": {"type": "number"}}
        registry.register("tool_a", lambda x: x, "Tool A", params)
        registry.register("tool_b", lambda x: x, "Tool B", params)

        schemas = registry.get_schemas()
        assert len(schemas) == 2

        schema_names = {s["name"] for s in schemas}
        assert schema_names == {"tool_a", "tool_b"}

        for schema in schemas:
            assert "description" in schema
            assert "parameters" in schema

    def test_register_overwrites(self):
        """Registering a tool with the same name should overwrite."""
        registry = ToolRegistry()
        registry.register("tool", lambda: "old", "Old tool", {})
        registry.register("tool", lambda: "new", "New tool", {})

        tool = registry.get("tool")
        assert tool["description"] == "New tool"
        assert tool["fn"]() == "new"

    def test_get_schemas_empty(self):
        """Empty registry should return empty schema list."""
        registry = ToolRegistry()
        assert registry.get_schemas() == []


# ===========================================================================
# ToolParser tests
# ===========================================================================


class TestToolParser:
    """Tests for the ToolParser class."""

    def test_parse_single_call(self):
        """Should parse a single tool call from a fenced block."""
        text = 'Some text\n```tool_call\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n```\nMore text'
        calls = ToolParser.parse(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "add"
        assert calls[0]["arguments"] == {"a": 1, "b": 2}

    def test_parse_multiple_calls(self):
        """Should parse multiple tool calls from separate fenced blocks."""
        text = (
            '```tool_call\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n```\n'
            'Some intermediate text\n'
            '```tool_call\n{"name": "greet", "arguments": {"name": "Alice"}}\n```'
        )
        calls = ToolParser.parse(text)

        assert len(calls) == 2
        assert calls[0]["name"] == "add"
        assert calls[1]["name"] == "greet"
        assert calls[1]["arguments"] == {"name": "Alice"}

    def test_parse_no_calls(self):
        """Should return empty list when no tool calls are present."""
        text = "Just a regular response with no tool calls."
        calls = ToolParser.parse(text)
        assert calls == []

    def test_parse_malformed_json(self):
        """Should handle malformed JSON gracefully (return empty list)."""
        text = '```tool_call\n{not valid json}\n```'
        calls = ToolParser.parse(text)
        assert calls == []

    def test_parse_missing_name_key(self):
        """Should skip blocks without a 'name' key."""
        text = '```tool_call\n{"arguments": {"a": 1}}\n```'
        calls = ToolParser.parse(text)
        assert calls == []

    def test_parse_empty_block(self):
        """Should handle empty tool_call blocks gracefully."""
        text = '```tool_call\n\n```'
        calls = ToolParser.parse(text)
        assert calls == []

    def test_parse_no_arguments(self):
        """Should default to empty dict when arguments is missing."""
        text = '```tool_call\n{"name": "noop"}\n```'
        calls = ToolParser.parse(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "noop"
        assert calls[0]["arguments"] == {}

    def test_parse_with_code_blocks(self):
        """Should not confuse regular code blocks with tool_call blocks."""
        text = (
            '```python\nprint("hello")\n```\n'
            '```tool_call\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n```'
        )
        calls = ToolParser.parse(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "add"

    def test_parse_complex_arguments(self):
        """Should handle nested/complex argument structures."""
        args = {"query": "SELECT * FROM users", "params": {"limit": 10, "offset": 0}}
        text = f'```tool_call\n{{"name": "sql_query", "arguments": {json.dumps(args)}}}\n```'
        calls = ToolParser.parse(text)

        assert len(calls) == 1
        assert calls[0]["arguments"]["params"]["limit"] == 10


# ===========================================================================
# ToolExecutor tests
# ===========================================================================


class TestToolExecutor:
    """Tests for the ToolExecutor class."""

    def test_execute_success(self):
        """Should execute a tool and return its result as a string."""
        registry = _make_registry()
        executor = ToolExecutor(registry)

        result = executor.execute({"name": "add", "arguments": {"a": 3, "b": 4}})
        assert result == "7"

    def test_execute_greet(self):
        """Should execute the greet tool correctly."""
        registry = _make_registry()
        executor = ToolExecutor(registry)

        result = executor.execute({"name": "greet", "arguments": {"name": "World"}})
        assert result == "Hello, World!"

    def test_execute_tool_not_found(self):
        """Should return an error message when tool is not found."""
        registry = _make_registry()
        executor = ToolExecutor(registry)

        result = executor.execute({"name": "nonexistent", "arguments": {}})
        assert "Error" in result
        assert "nonexistent" in result

    def test_execute_tool_error(self):
        """Should catch exceptions and return error message."""
        registry = ToolRegistry()
        registry.register(
            "divide",
            lambda a, b: a / b,
            "Divide two numbers",
            {"a": {"type": "number"}, "b": {"type": "number"}},
        )
        executor = ToolExecutor(registry)

        result = executor.execute({"name": "divide", "arguments": {"a": 1, "b": 0}})
        assert "Error" in result
        assert "ZeroDivisionError" in result

    def test_execute_type_error(self):
        """Should handle type errors from wrong argument types."""
        registry = ToolRegistry()
        registry.register(
            "add",
            lambda a, b: a + b,
            "Add two numbers",
            {"a": {"type": "number"}, "b": {"type": "number"}},
        )
        executor = ToolExecutor(registry)

        # Pass a string where a number is expected -- should still work
        # because string + string is valid, but passing wrong number of args
        # should fail gracefully
        result = executor.execute({"name": "add", "arguments": {"a": 1}})
        assert "Error" in result

    def test_format_result_string(self):
        """format_result should return strings as-is."""
        assert ToolExecutor.format_result("hello") == "hello"

    def test_format_result_number(self):
        """format_result should convert non-strings to str."""
        assert ToolExecutor.format_result(42) == "42"
        assert ToolExecutor.format_result(3.14) == "3.14"

    def test_format_result_list(self):
        """format_result should convert lists to string."""
        result = ToolExecutor.format_result([1, 2, 3])
        assert result == "[1, 2, 3]"

    def test_format_result_dict(self):
        """format_result should convert dicts to string."""
        result = ToolExecutor.format_result({"key": "value"})
        assert "key" in result
        assert "value" in result


# ===========================================================================
# ToolCallingAgent tests
# ===========================================================================


class TestToolCallingAgent:
    """Tests for the ToolCallingAgent class."""

    def test_single_turn_no_tools(self):
        """Should return LLM response directly when no tool calls."""
        registry = ToolRegistry()
        llm_fn = lambda messages: "I can help you with that."
        agent = ToolCallingAgent(llm_fn, registry)

        result = agent.run("Hello")
        assert result == "I can help you with that."

    def test_single_tool_call(self):
        """Should execute one tool call and return final response."""
        registry = _make_registry()

        # First call returns a tool call, second call returns final response
        responses = [
            '```tool_call\n{"name": "add", "arguments": {"a": 3, "b": 4}}\n```',
            "The result is 7.",
        ]
        call_count = [0]

        def llm_fn(messages):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        agent = ToolCallingAgent(llm_fn, registry)
        result = agent.run("What is 3 + 4?")

        assert result == "The result is 7."
        assert call_count[0] == 2

    def test_multi_turn_tool_calls(self):
        """Should handle multiple rounds of tool calls."""
        registry = _make_registry()

        responses = [
            '```tool_call\n{"name": "add", "arguments": {"a": 3, "b": 4}}\n```',
            '```tool_call\n{"name": "add", "arguments": {"a": 7, "b": 10}}\n```',
            "The final answer is 17.",
        ]
        call_count = [0]

        def llm_fn(messages):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        agent = ToolCallingAgent(llm_fn, registry)
        result = agent.run("What is 3 + 4 + 10?")

        assert result == "The final answer is 17."
        assert call_count[0] == 3

    def test_max_turns_limit(self):
        """Should stop after max_turns even if LLM keeps requesting tools."""
        registry = _make_registry()

        # Always return a tool call -- agent should stop after max_turns
        def llm_fn(messages):
            return '```tool_call\n{"name": "add", "arguments": {"a": 1, "b": 1}}\n```'

        agent = ToolCallingAgent(llm_fn, registry)
        result = agent.run("Keep adding", max_turns=3)

        # Should return the last response (a tool call string) after 3 turns
        assert "add" in result

    def test_tool_result_in_messages(self):
        """Tool results should appear in subsequent messages to the LLM."""
        registry = _make_registry()
        seen_messages = []

        responses = [
            '```tool_call\n{"name": "greet", "arguments": {"name": "Alice"}}\n```',
            "Done!",
        ]
        call_count = [0]

        def llm_fn(messages):
            seen_messages.append([m.copy() for m in messages])
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        agent = ToolCallingAgent(llm_fn, registry)
        agent.run("Say hello to Alice")

        # On the second call, messages should contain the tool result
        assert len(seen_messages) == 2
        second_call_messages = seen_messages[1]
        # Should have: user, assistant (with tool call), tool (with result)
        assert len(second_call_messages) == 3
        assert second_call_messages[0]["role"] == "user"
        assert second_call_messages[1]["role"] == "assistant"
        assert second_call_messages[2]["role"] == "tool"
        assert "Hello, Alice!" in second_call_messages[2]["content"]

    def test_multiple_tools_in_one_turn(self):
        """Should execute multiple tool calls from a single LLM response."""
        registry = _make_registry()

        # Return two tool calls in one response
        tool_calls_text = (
            '```tool_call\n{"name": "add", "arguments": {"a": 1, "b": 2}}\n```\n'
            '```tool_call\n{"name": "greet", "arguments": {"name": "Bob"}}\n```'
        )
        responses = [tool_calls_text, "Both done!"]
        call_count = [0]

        def llm_fn(messages):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        agent = ToolCallingAgent(llm_fn, registry)
        result = agent.run("Do two things")

        assert result == "Both done!"
        # Second call should have: user, assistant, tool(3), tool(Bob)
        assert call_count[0] == 2

    def test_empty_prompt(self):
        """Should handle empty prompt gracefully."""
        registry = ToolRegistry()
        llm_fn = lambda messages: "Please provide a question."
        agent = ToolCallingAgent(llm_fn, registry)

        result = agent.run("")
        assert result == "Please provide a question."

    def test_agent_with_failing_tool(self):
        """Should handle tool execution errors without crashing."""
        registry = ToolRegistry()
        registry.register(
            "explode",
            lambda: (_ for _ in ()).throw(RuntimeError("boom")),
            "Explode",
            {},
        )

        responses = [
            '```tool_call\n{"name": "explode", "arguments": {}}\n```',
            "Sorry, the tool failed.",
        ]
        call_count = [0]

        def llm_fn(messages):
            idx = call_count[0]
            call_count[0] += 1
            return responses[idx]

        agent = ToolCallingAgent(llm_fn, registry)
        result = agent.run("Try something")

        assert result == "Sorry, the tool failed."
        # The tool error should have been passed back
        assert call_count[0] == 2
