"""Reference solution for the Model Context Protocol (MCP) exercise."""

import json
from typing import Any, Callable


class MCPMessageSolution:
    """Standardized message format for the Model Context Protocol."""

    VALID_ROLES = ("user", "assistant", "tool", "system")

    def __init__(
        self,
        role: str,
        content: str = "",
        tool_calls: list[dict[str, Any]] | None = None,
        tool_results: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ):
        """Initialize an MCPMessage."""
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
        """Serialize the message to a dictionary."""
        return {
            "role": self.role,
            "content": self.content,
            "tool_calls": self.tool_calls,
            "tool_results": self.tool_results,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPMessageSolution":
        """Deserialize a message from a dictionary."""
        if "role" not in data:
            raise ValueError("Message dict must contain a 'role' key.")
        return cls(
            role=data["role"],
            content=data.get("content", ""),
            tool_calls=data.get("tool_calls", []),
            tool_results=data.get("tool_results", []),
            metadata=data.get("metadata", {}),
        )


class MCPServerSolution:
    """Manages tools and handles tool execution requests."""

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
        """Register a tool with the server."""
        self._tools[name] = {
            "name": name,
            "fn": fn,
            "description": description,
            "parameters": parameters,
        }

    def handle_request(self, message: "MCPMessageSolution") -> "MCPMessageSolution":
        """Process an incoming message and execute tool calls if present."""
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

        return MCPMessageSolution(
            role="tool",
            content=json.dumps(results),
            tool_results=results,
        )


class MCPClientSolution:
    """Sends requests to an MCP server and handles responses."""

    def __init__(self, server: "MCPServerSolution"):
        """Initialize the MCP client."""
        self.server = server

    def send(self, message: "MCPMessageSolution") -> "MCPMessageSolution":
        """Send a message to the server and return the response."""
        return self.server.handle_request(message)

    def send_and_receive(self, text: str) -> "MCPMessageSolution":
        """Create a user message, send it, and return the response."""
        message = MCPMessageSolution(role="user", content=text)
        return self.send(message)


class MCPAgentSolution:
    """Agent that communicates via the Model Context Protocol."""

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        server: "MCPServerSolution",
    ):
        """Initialize the MCP agent."""
        self.llm_fn = llm_fn
        self.server = server
        self.client = MCPClientSolution(server)

    def run(self, prompt: str, max_turns: int = 5) -> str:
        """Run the agent loop until a final response or max turns."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_turns):
            response_text = self.llm_fn(messages)
            tool_calls = self._parse_tool_calls(response_text)

            if not tool_calls:
                return response_text

            messages.append({"role": "assistant", "content": response_text})

            request_msg = MCPMessageSolution(
                role="assistant",
                content=response_text,
                tool_calls=tool_calls,
            )
            result_msg = self.client.send(request_msg)

            for tr in result_msg.tool_results:
                messages.append({
                    "role": "tool",
                    "content": tr.get("result", ""),
                })

        return response_text

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from LLM output text."""
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
