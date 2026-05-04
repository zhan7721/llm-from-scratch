"""Tests for the Model Context Protocol (MCP) implementation."""

import json

from mcp import MCPMessage, MCPServer, MCPClient, MCPAgent


# ===========================================================================
# MCPMessage tests
# ===========================================================================


class TestMCPMessage:
    """Tests for the MCPMessage class."""

    def test_create_user_message(self):
        """Should create a user message with correct fields."""
        msg = MCPMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.tool_calls == []
        assert msg.tool_results == []
        assert msg.metadata == {}

    def test_create_assistant_message(self):
        """Should create an assistant message."""
        msg = MCPMessage(role="assistant", content="Hi there!")
        assert msg.role == "assistant"
        assert msg.content == "Hi there!"

    def test_create_tool_message(self):
        """Should create a tool message with results."""
        results = [{"name": "add", "result": "3"}]
        msg = MCPMessage(role="tool", content="done", tool_results=results)
        assert msg.role == "tool"
        assert msg.tool_results == results

    def test_create_system_message(self):
        """Should create a system message."""
        msg = MCPMessage(role="system", content="You are a helpful assistant.")
        assert msg.role == "system"
        assert msg.content == "You are a helpful assistant."

    def test_create_message_with_tool_calls(self):
        """Should store tool calls."""
        calls = [{"name": "add", "arguments": {"a": 1, "b": 2}}]
        msg = MCPMessage(role="assistant", content="", tool_calls=calls)
        assert msg.tool_calls == calls

    def test_create_message_with_metadata(self):
        """Should store metadata."""
        meta = {"turn": 1, "model": "test"}
        msg = MCPMessage(role="user", content="hi", metadata=meta)
        assert msg.metadata == meta

    def test_invalid_role_raises(self):
        """Should raise ValueError for invalid role."""
        try:
            MCPMessage(role="invalid", content="test")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "invalid" in str(e)

    def test_to_dict(self):
        """Should serialize to dict with all fields."""
        msg = MCPMessage(
            role="user",
            content="test",
            tool_calls=[{"name": "foo"}],
            tool_results=[{"name": "foo", "result": "bar"}],
            metadata={"key": "value"},
        )
        d = msg.to_dict()

        assert d["role"] == "user"
        assert d["content"] == "test"
        assert d["tool_calls"] == [{"name": "foo"}]
        assert d["tool_results"] == [{"name": "foo", "result": "bar"}]
        assert d["metadata"] == {"key": "value"}

    def test_to_dict_minimal(self):
        """Should serialize minimal message correctly."""
        msg = MCPMessage(role="assistant", content="ok")
        d = msg.to_dict()

        assert d["role"] == "assistant"
        assert d["content"] == "ok"
        assert d["tool_calls"] == []
        assert d["tool_results"] == []
        assert d["metadata"] == {}

    def test_from_dict(self):
        """Should deserialize from dict."""
        data = {
            "role": "user",
            "content": "hello",
            "tool_calls": [],
            "tool_results": [],
            "metadata": {"x": 1},
        }
        msg = MCPMessage.from_dict(data)

        assert msg.role == "user"
        assert msg.content == "hello"
        assert msg.metadata == {"x": 1}

    def test_from_dict_missing_role_raises(self):
        """Should raise ValueError when role is missing."""
        try:
            MCPMessage.from_dict({"content": "test"})
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "role" in str(e)

    def test_from_dict_defaults(self):
        """Should use defaults for missing optional fields."""
        data = {"role": "tool"}
        msg = MCPMessage.from_dict(data)

        assert msg.content == ""
        assert msg.tool_calls == []
        assert msg.tool_results == []
        assert msg.metadata == {}

    def test_roundtrip_serialization(self):
        """to_dict -> from_dict should preserve all data."""
        original = MCPMessage(
            role="assistant",
            content="I'll call a tool",
            tool_calls=[{"name": "search", "arguments": {"q": "test"}}],
            tool_results=[],
            metadata={"turn": 2},
        )
        restored = MCPMessage.from_dict(original.to_dict())

        assert restored.role == original.role
        assert restored.content == original.content
        assert restored.tool_calls == original.tool_calls
        assert restored.tool_results == original.tool_results
        assert restored.metadata == original.metadata

    def test_repr(self):
        """Should have a useful repr."""
        msg = MCPMessage(role="user", content="hi")
        r = repr(msg)

        assert "MCPMessage" in r
        assert "user" in r
        assert "hi" in r


# ===========================================================================
# MCPServer tests
# ===========================================================================


class TestMCPServer:
    """Tests for the MCPServer class."""

    def test_register_tool(self):
        """Should register a tool."""
        server = MCPServer()
        server.register_tool(
            name="add",
            fn=lambda a, b: a + b,
            description="Add two numbers",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        )
        schemas = server.get_tool_schemas()

        assert len(schemas) == 1
        assert schemas[0]["name"] == "add"
        assert schemas[0]["description"] == "Add two numbers"

    def test_register_multiple_tools(self):
        """Should register multiple tools."""
        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        server.register_tool("mul", lambda a, b: a * b, "Multiply", {})

        schemas = server.get_tool_schemas()
        names = {s["name"] for s in schemas}
        assert names == {"add", "mul"}

    def test_handle_request_no_tool_calls(self):
        """Should echo back messages without tool calls."""
        server = MCPServer()
        msg = MCPMessage(role="user", content="Hello")
        result = server.handle_request(msg)

        assert result.role == "user"
        assert result.content == "Hello"

    def test_handle_request_with_tool_call(self):
        """Should execute tool calls and return results."""
        server = MCPServer()
        server.register_tool(
            name="add",
            fn=lambda a, b: a + b,
            description="Add",
            parameters={},
        )

        msg = MCPMessage(
            role="assistant",
            content="",
            tool_calls=[{"name": "add", "arguments": {"a": 2, "b": 3}}],
        )
        result = server.handle_request(msg)

        assert result.role == "tool"
        assert len(result.tool_results) == 1
        assert result.tool_results[0]["name"] == "add"
        assert result.tool_results[0]["result"] == "5"

    def test_handle_request_multiple_tool_calls(self):
        """Should execute multiple tool calls."""
        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        server.register_tool("mul", lambda a, b: a * b, "Multiply", {})

        msg = MCPMessage(
            role="assistant",
            content="",
            tool_calls=[
                {"name": "add", "arguments": {"a": 1, "b": 2}},
                {"name": "mul", "arguments": {"a": 3, "b": 4}},
            ],
        )
        result = server.handle_request(msg)

        assert result.role == "tool"
        assert len(result.tool_results) == 2
        assert result.tool_results[0]["result"] == "3"
        assert result.tool_results[1]["result"] == "12"

    def test_handle_request_unknown_tool(self):
        """Should return error for unknown tool."""
        server = MCPServer()
        msg = MCPMessage(
            role="assistant",
            content="",
            tool_calls=[{"name": "nonexistent", "arguments": {}}],
        )
        result = server.handle_request(msg)

        assert result.role == "tool"
        assert "nonexistent" in result.tool_results[0]["result"]
        assert "Error" in result.tool_results[0]["result"]

    def test_handle_request_tool_error(self):
        """Should catch tool execution errors gracefully."""
        def bad_tool():
            raise RuntimeError("something broke")

        server = MCPServer()
        server.register_tool("bad", bad_tool, "Bad tool", {})

        msg = MCPMessage(
            role="assistant",
            content="",
            tool_calls=[{"name": "bad", "arguments": {}}],
        )
        result = server.handle_request(msg)

        assert result.role == "tool"
        assert "Error" in result.tool_results[0]["result"]
        assert "something broke" in result.tool_results[0]["result"]

    def test_handle_request_tool_returns_non_string(self):
        """Should convert non-string tool results to string."""
        server = MCPServer()
        server.register_tool("get_num", lambda: 42, "Get number", {})

        msg = MCPMessage(
            role="assistant",
            content="",
            tool_calls=[{"name": "get_num", "arguments": {}}],
        )
        result = server.handle_request(msg)

        assert result.tool_results[0]["result"] == "42"

    def test_get_tool_schemas_empty(self):
        """Should return empty list when no tools registered."""
        server = MCPServer()
        assert server.get_tool_schemas() == []


# ===========================================================================
# MCPClient tests
# ===========================================================================


class TestMCPClient:
    """Tests for the MCPClient class."""

    def test_send_message(self):
        """Should send message to server and return response."""
        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})

        client = MCPClient(server)
        msg = MCPMessage(
            role="assistant",
            content="",
            tool_calls=[{"name": "add", "arguments": {"a": 1, "b": 2}}],
        )
        response = client.send(msg)

        assert response.role == "tool"
        assert response.tool_results[0]["result"] == "3"

    def test_send_no_tool_calls(self):
        """Should pass through messages without tool calls."""
        server = MCPServer()
        client = MCPClient(server)

        msg = MCPMessage(role="user", content="Hello")
        response = client.send(msg)

        assert response.role == "user"
        assert response.content == "Hello"

    def test_send_and_receive(self):
        """Should create user message and send it."""
        server = MCPServer()
        client = MCPClient(server)

        response = client.send_and_receive("What is the weather?")

        assert response.role == "user"
        assert response.content == "What is the weather?"

    def test_send_and_receive_with_tool(self):
        """send_and_receive should work with server that has tools."""
        server = MCPServer()
        server.register_tool(
            "echo",
            lambda text: f"echo: {text}",
            "Echo text",
            {},
        )
        client = MCPClient(server)

        # send_and_receive creates a user message; no tool calls
        response = client.send_and_receive("Hello")
        assert response.role == "user"
        assert response.content == "Hello"


# ===========================================================================
# MCPAgent tests
# ===========================================================================


class TestMCPAgent:
    """Tests for the MCPAgent class."""

    def test_single_turn_no_tools(self):
        """Should return LLM response directly when no tool calls."""
        def llm(messages):
            return "The answer is 42."

        server = MCPServer()
        agent = MCPAgent(llm, server)
        result = agent.run("What is 6 * 7?")

        assert result == "The answer is 42."

    def test_single_turn_with_tool_call(self):
        """Should execute tool and feed result back to LLM."""
        call_count = [0]

        def llm(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    'I need to calculate.\n'
                    '```tool_call\n'
                    '{"name": "add", "arguments": {"a": 2, "b": 3}}\n'
                    '```'
                )
            return "The result is 5."

        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        agent = MCPAgent(llm, server)
        result = agent.run("What is 2 + 3?")

        assert result == "The result is 5."
        assert call_count[0] == 2

    def test_multi_turn_tool_calls(self):
        """Should handle multiple rounds of tool calls."""
        call_count = [0]

        def llm(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '```tool_call\n'
                    '{"name": "add", "arguments": {"a": 10, "b": 20}}\n'
                    '```'
                )
            if call_count[0] == 2:
                return (
                    '```tool_call\n'
                    '{"name": "mul", "arguments": {"a": 30, "b": 2}}\n'
                    '```'
                )
            return "The final answer is 60."

        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        server.register_tool("mul", lambda a, b: a * b, "Multiply", {})
        agent = MCPAgent(llm, server)
        result = agent.run("Compute (10 + 20) * 2")

        assert result == "The final answer is 60."
        assert call_count[0] == 3

    def test_max_turns_reached(self):
        """Should stop after max_turns even if tools keep being called."""
        def llm(messages):
            return (
                '```tool_call\n'
                '{"name": "noop", "arguments": {}}\n'
                '```'
            )

        server = MCPServer()
        server.register_tool("noop", lambda: None, "No-op", {})
        agent = MCPAgent(llm, server)
        result = agent.run("Loop forever", max_turns=3)

        # Should return the last LLM response after max_turns
        assert "tool_call" in result

    def test_tool_error_handled_gracefully(self):
        """Should handle tool errors without crashing."""
        call_count = [0]

        def llm(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '```tool_call\n'
                    '{"name": "broken", "arguments": {}}\n'
                    '```'
                )
            # The tool result should be in the message history
            for msg in messages:
                if msg.get("role") == "tool" and "Error" in msg.get("content", ""):
                    return "The tool failed, but that's okay."
            return "Something went wrong."

        server = MCPServer()
        server.register_tool(
            "broken",
            lambda: (_ for _ in ()).throw(RuntimeError("broken")),
            "Broken tool",
            {},
        )
        agent = MCPAgent(llm, server)
        result = agent.run("Use the broken tool")

        assert "okay" in result.lower() or "wrong" in result.lower()

    def test_unknown_tool_handled(self):
        """Should handle calls to unregistered tools gracefully."""
        call_count = [0]

        def llm(messages):
            call_count[0] += 1
            if call_count[0] == 1:
                return (
                    '```tool_call\n'
                    '{"name": "ghost", "arguments": {}}\n'
                    '```'
                )
            return "The tool was not found."

        server = MCPServer()
        agent = MCPAgent(llm, server)
        result = agent.run("Call ghost tool")

        assert result == "The tool was not found."

    def test_agent_uses_mcp_client(self):
        """Agent should use MCPClient internally for communication."""
        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        agent = MCPAgent(lambda msgs: "ok", server)

        assert isinstance(agent.client, MCPClient)
        assert agent.client.server is server

    def test_parse_tool_calls_single(self):
        """Should parse a single tool call from text."""
        text = (
            'Let me calculate.\n'
            '```tool_call\n'
            '{"name": "add", "arguments": {"a": 1, "b": 2}}\n'
            '```'
        )
        calls = MCPAgent._parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "add"
        assert calls[0]["arguments"] == {"a": 1, "b": 2}

    def test_parse_tool_calls_multiple(self):
        """Should parse multiple tool calls from text."""
        text = (
            '```tool_call\n'
            '{"name": "add", "arguments": {"a": 1, "b": 2}}\n'
            '```\n'
            'And also:\n'
            '```tool_call\n'
            '{"name": "mul", "arguments": {"a": 3, "b": 4}}\n'
            '```'
        )
        calls = MCPAgent._parse_tool_calls(text)

        assert len(calls) == 2
        assert calls[0]["name"] == "add"
        assert calls[1]["name"] == "mul"

    def test_parse_tool_calls_none(self):
        """Should return empty list when no tool calls present."""
        text = "Just a regular response with no tool calls."
        calls = MCPAgent._parse_tool_calls(text)

        assert calls == []

    def test_parse_tool_calls_malformed_json(self):
        """Should skip malformed JSON blocks."""
        text = (
            '```tool_call\n'
            'not valid json\n'
            '```\n'
            '```tool_call\n'
            '{"name": "add", "arguments": {"a": 1, "b": 2}}\n'
            '```'
        )
        calls = MCPAgent._parse_tool_calls(text)

        assert len(calls) == 1
        assert calls[0]["name"] == "add"

    def test_agent_preserves_message_history(self):
        """Agent should accumulate messages across turns."""
        seen_message_counts = []

        def llm(messages):
            seen_message_counts.append(len(messages))
            if len(messages) <= 1:
                return (
                    '```tool_call\n'
                    '{"name": "add", "arguments": {"a": 1, "b": 2}}\n'
                    '```'
                )
            return "Done."

        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        agent = MCPAgent(llm, server)
        agent.run("test")

        # First call: 1 message (user)
        # Second call: 3 messages (user, assistant, tool)
        assert seen_message_counts[0] == 1
        assert seen_message_counts[1] == 3
