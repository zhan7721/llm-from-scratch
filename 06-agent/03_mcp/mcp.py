"""Model Context Protocol (MCP) implementation for LLM agents.

This module implements a simplified version of the Model Context Protocol,
a standardized way for LLMs to communicate with external tools and services.
It contains:
- MCPMessage: Standardized message format for all communication.
- MCPServer: Manages tools and handles tool execution requests.
- MCPClient: Sends requests to server and handles responses.
- MCPAgent: Agent that uses MCP protocol to interact with tools.

Architecture overview:
    User prompt
        -> MCPAgent creates MCPMessage
        -> MCPClient sends message to MCPServer
        -> MCPServer processes message, executes tools if needed
        -> MCPServer returns MCPMessage with results
        -> MCPClient returns response to agent
        -> Agent decides next action or returns final answer

Design notes:
- All communication goes through MCPMessage objects.
- The server handles tool registration and execution.
- The client handles communication between agent and server.
- The LLM function signature is fn(messages: list[dict]) -> str.
- Messages are serialized to dicts for LLM consumption.
- This module is self-contained (no external dependencies beyond the
  standard library).
- Error handling is graceful: errors are caught and reported, never raised.
"""

import json
from typing import Any, Callable


__all__ = [
    "MCPMessage",
    "MCPServer",
    "MCPClient",
    "MCPAgent",
]


# ---------------------------------------------------------------------------
# MCPMessage
# ---------------------------------------------------------------------------


class MCPMessage:
    """Standardized message format for the Model Context Protocol.

    Every piece of communication in the MCP system is represented as an
    MCPMessage. Messages have a role (who sent them), content (the text),
    and optional tool-related fields.

    Roles:
        - "user": A message from the human user.
        - "assistant": A message from the LLM.
        - "tool": A message containing tool execution results.
        - "system": A system-level instruction.

    Attributes:
        role: The role of the message sender.
        content: The text content of the message.
        tool_calls: List of tool call requests (name + arguments).
        tool_results: List of tool execution results.
        metadata: Additional key-value metadata.

    Example:
        msg = MCPMessage(role="user", content="What is 2 + 2?")
        data = msg.to_dict()
        restored = MCPMessage.from_dict(data)
    """

    VALID_ROLES = ("user", "assistant", "tool", "system")

    def __init__(
        self,
        role: str,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize an MCPMessage.

        Args:
            role: The role of the message sender. Must be one of
                "user", "assistant", "tool", or "system".
            content: The text content of the message.
            tool_calls: Optional list of tool call dicts, each with
                "name" (str) and "arguments" (dict) keys.
            tool_results: Optional list of tool result dicts, each with
                "name" (str) and "result" (str) keys.
            metadata: Optional dict of additional metadata.

        Raises:
            ValueError: If role is not one of the valid roles.
        """
        if role not in self.VALID_ROLES:
            raise ValueError(
                f"Invalid role '{role}'. Must be one of: {self.VALID_ROLES}"
            )
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_results = tool_results or []
        self.metadata = metadata or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message to a dictionary.

        Returns:
            A dict representation of the message with keys: "role",
            "content", "tool_calls", "tool_results", and "metadata".
        """
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPMessage":
        """Deserialize a message from a dictionary.

        Args:
            data: A dict with keys matching those produced by to_dict().

        Returns:
            An MCPMessage instance.

        Raises:
            ValueError: If "role" is missing or invalid.
        """
        if "role" not in data:
            raise ValueError("Message dict must contain a 'role' key.")
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            metadata=data.get("metadata", {}),
        )

    def __repr__(self) -> str:
        """Return a developer-friendly string representation."""
        return (
            f"MCPMessage(role={self.role!r}, content={self.content!r}, "
            f"tool_calls={self.tool_calls!r}, tool_results={self.tool_results!r})"
        )


# ---------------------------------------------------------------------------
# MCPServer
# ---------------------------------------------------------------------------


class MCPServer:
    """Manages tools and handles tool execution requests.

    The server is the central hub of the MCP system. It holds a registry
    of tools and processes incoming messages. When a message contains tool
    calls, the server executes them and returns a message with the results.

    Example:
        server = MCPServer()
        server.register_tool(
            name="add",
            fn=lambda a, b: a + b,
            description="Add two numbers",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        )
    """

    def __init__(self):
        """Initialize the MCP server."""
        self._tools: dict[str, dict[str, Any]] = {}

    def register_tool(
        self,
        name: str,
        fn: Callable,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool with the server.

        Args:
            name: Unique tool name.
            fn: Callable that implements the tool.
            description: Human-readable description of what the tool does.
            parameters: JSON-schema-like dict describing the tool's parameters.
        """
        self._tools[name] = {
            "name": name,
            "fn": fn,
            "description": description,
            "parameters": parameters,
        }

    def handle_request(self, message: MCPMessage) -> MCPMessage:
        """Process an incoming message and execute tool calls if present.

        If the message contains tool calls, each call is executed and the
        results are collected. The server returns a new MCPMessage with
        role "tool" containing the results.

        If the message has no tool calls, it is returned as-is (echo).

        Args:
            message: The incoming MCPMessage to process.

        Returns:
            An MCPMessage with role "tool" containing execution results,
            or the original message if no tool calls were present.
        """
        if not message.tool_calls:
            return message

        results: list[dict[str, Any]] = []
        for call in message.tool_calls:
            name = call.get("name", "")
            arguments = call.get("arguments", {})

            tool = self._tools.get(name)
            if tool is None:
                results.append({
                    "name": name,
                    "result": f"Error: tool '{name}' not found",
                })
                continue

            try:
                result = tool["fn"](**arguments)
                results.append({
                    "name": name,
                    "result": str(result),
                })
            except Exception as e:
                results.append({
                    "name": name,
                    "result": f"Error: {type(e).__name__}: {e}",
                })

        return MCPMessage(
            role="tool",
            content=json.dumps(results),
            tool_results=results,
        )

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        """Return schemas for all registered tools.

        Returns:
            List of dicts with "name", "description", and "parameters" keys.
        """
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for info in self._tools.values()
        ]


# ---------------------------------------------------------------------------
# MCPClient
# ---------------------------------------------------------------------------


class MCPClient:
    """Sends requests to an MCP server and handles responses.

    The client provides a clean interface for communicating with the server.
    It handles message serialization and response parsing.

    Example:
        server = MCPServer()
        client = MCPClient(server)
        response = client.send(MCPMessage(role="user", content="Hello"))
    """

    def __init__(self, server: MCPServer):
        """Initialize the MCP client.

        Args:
            server: The MCPServer instance to communicate with.
        """
        self.server = server

    def send(self, message: MCPMessage) -> MCPMessage:
        """Send a message to the server and return the response.

        Args:
            message: The MCPMessage to send to the server.

        Returns:
            The MCPMessage response from the server.
        """
        return self.server.handle_request(message)

    def send_and_receive(self, text: str) -> MCPMessage:
        """Convenience method: create a user message, send it, and return
        the response.

        Args:
            text: The text content for a user message.

        Returns:
            The MCPMessage response from the server.
        """
        message = MCPMessage(role="user", content=text)
        return self.send(message)


# ---------------------------------------------------------------------------
# MCPAgent
# ---------------------------------------------------------------------------


class MCPAgent:
    """Agent that communicates via the Model Context Protocol.

    Combines an LLM function with an MCP server to implement an agent loop.
    The agent sends messages to the LLM, checks if the LLM wants to call
    tools, executes them via MCP, and repeats until the LLM produces a
    final text response.

    The LLM function should return text that may contain tool call requests
    in JSON format within ```tool_call fenced blocks, following the same
    convention as the ToolCallingAgent.

    Args:
        llm_fn: A callable that takes a list of message dicts (OpenAI format)
            and returns a string (the LLM's response).
        server: An MCPServer instance with registered tools.

    Example:
        server = MCPServer()
        server.register_tool("add", lambda a, b: a + b, "Add", {})
        agent = MCPAgent(my_llm, server)
        result = agent.run("What is 2 + 3?")
    """

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        server: MCPServer,
    ):
        """Initialize the MCP agent."""
        self.llm_fn = llm_fn
        self.server = server
        self.client = MCPClient(server)

    def run(self, prompt: str, max_turns: int = 5) -> str:
        """Run the agent loop until a final response or max turns.

        Args:
            prompt: The user's input prompt.
            max_turns: Maximum number of LLM calls before stopping.

        Returns:
            The final text response from the LLM.
        """
        messages: list[dict[str, str]] = [
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_turns):
            # 1. Generate
            response_text = self.llm_fn(messages)

            # 2. Parse tool calls
            tool_calls = self._parse_tool_calls(response_text)

            # 3. If no tool calls, return the response
            if not tool_calls:
                return response_text

            # 4. Execute tool calls via MCP
            messages.append({"role": "assistant", "content": response_text})

            request_msg = MCPMessage(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls,
            )
            result_msg = self.client.send(request_msg)

            # 5. Add tool results to message history
            for tr in result_msg.tool_results:
                messages.append({
                    "role": "tool",
                    "content": tr.get("result", ""),
                })

        # Max turns reached -- return the last response
        return response_text

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from LLM output text.

        Scans the text for ```tool_call fenced blocks and parses each one
        as a JSON object with "name" and "arguments" keys.

        Args:
            text: The LLM output text to parse.

        Returns:
            List of tool call dicts, each with keys "name" (str) and
            "arguments" (dict). Returns an empty list if no tool calls are
            found.
        """
        tool_calls: list[dict[str, Any]] = []
        marker_start = "```tool_call"
        marker_end = "```"

        remaining = text
        while True:
            start_idx = remaining.find(marker_start)
            if start_idx == -1:
                break

            content_start = remaining.find("\n", start_idx)
            if content_start == -1:
                break
            content_start += 1

            end_idx = remaining.find(marker_end, content_start)
            if end_idx == -1:
                break

            json_str = remaining[content_start:end_idx].strip()
            remaining = remaining[end_idx + len(marker_end):]

            if not json_str:
                continue

            try:
                parsed = json.loads(json_str)
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    })
            except (json.JSONDecodeError, TypeError):
                continue

        return tool_calls
