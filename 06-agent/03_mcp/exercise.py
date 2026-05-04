"""Exercise: Implement Model Context Protocol (MCP) for LLM agents.

Complete the TODOs below to build a standardized communication protocol
that allows LLMs to interact with external tools through a clean
client-server architecture.
Run `pytest tests.py` to verify your implementation.
"""

import json
from typing import Any, Callable


class MCPMessageExercise:
    """Standardized message format for the Model Context Protocol.

    TODO: Implement all methods.

    Every piece of communication in the MCP system is represented as an
    MCPMessage. Messages have a role, content, and optional tool fields.

    Example:
        msg = MCPMessageExercise(role="user", content="What is 2 + 2?")
        data = msg.to_dict()
        restored = MCPMessageExercise.from_dict(data)
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
            role: Must be one of "user", "assistant", "tool", or "system".
            content: The text content of the message.
            tool_calls: Optional list of tool call dicts.
            tool_results: Optional list of tool result dicts.
            metadata: Optional dict of additional metadata.

        Raises:
            ValueError: If role is not one of the valid roles.
        """
        # TODO 1: Validate the role
        # Hint: raise ValueError if role not in self.VALID_ROLES

        # TODO 2: Store all fields, using empty defaults for None values
        # Hint: self.tool_calls = tool_calls or []
        pass  # YOUR CODE HERE

    def to_dict(self) -> dict[str, Any]:
        """Serialize the message to a dictionary.

        Returns:
            A dict with keys: "role", "content", "tool_calls",
            "tool_results", and "metadata".
        """
        # TODO 3: Return a dict with all message fields
        return {}  # YOUR CODE HERE

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MCPMessageExercise":
        """Deserialize a message from a dictionary.

        Args:
            data: A dict with keys matching those produced by to_dict().

        Returns:
            An MCPMessageExercise instance.

        Raises:
            ValueError: If "role" is missing.
        """
        # TODO 4: Validate that "role" key exists
        # TODO 5: Create and return a new instance from the dict values
        # Hint: use data.get() for optional fields with defaults
        return cls(role="user")  # YOUR CODE HERE


class MCPServerExercise:
    """Manages tools and handles tool execution requests.

    TODO: Implement all methods.

    Example:
        server = MCPServerExercise()
        server.register_tool(
            name="add",
            fn=lambda a, b: a + b,
            description="Add two numbers",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        )
    """

    def __init__(self):
        """Initialize the MCP server."""
        # TODO 6: Initialize the tools storage
        # Hint: self._tools should be a dict mapping name -> tool info
        pass  # YOUR CODE HERE

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
            description: Human-readable description.
            parameters: JSON-schema-like dict of parameters.
        """
        # TODO 7: Store the tool info in self._tools
        # Hint: store name, fn, description, and parameters
        pass  # YOUR CODE HERE

    def handle_request(self, message: "MCPMessageExercise") -> "MCPMessageExercise":
        """Process an incoming message and execute tool calls if present.

        Args:
            message: The incoming MCPMessageExercise to process.

        Returns:
            An MCPMessageExercise with role "tool" containing results,
            or the original message if no tool calls were present.
        """
        # TODO 8: If no tool calls, return the message as-is
        # TODO 9: For each tool call, look up the tool and execute it
        # TODO 10: Handle unknown tools (return error message)
        # TODO 11: Handle execution errors (catch Exception, return error)
        # TODO 12: Return a new MCPMessageExercise with role "tool" and results
        return message  # YOUR CODE HERE


class MCPClientExercise:
    """Sends requests to an MCP server and handles responses.

    TODO: Implement all methods.

    Example:
        server = MCPServerExercise()
        client = MCPClientExercise(server)
        response = client.send(MCPMessageExercise(role="user", content="Hi"))
    """

    def __init__(self, server: "MCPServerExercise"):
        """Initialize the MCP client.

        Args:
            server: The MCPServerExercise to communicate with.
        """
        # TODO 13: Store the server reference
        pass  # YOUR CODE HERE

    def send(self, message: "MCPMessageExercise") -> "MCPMessageExercise":
        """Send a message to the server and return the response.

        Args:
            message: The MCPMessageExercise to send.

        Returns:
            The MCPMessageExercise response from the server.
        """
        # TODO 14: Delegate to the server's handle_request method
        return message  # YOUR CODE HERE

    def send_and_receive(self, text: str) -> "MCPMessageExercise":
        """Create a user message, send it, and return the response.

        Args:
            text: The text content for a user message.

        Returns:
            The MCPMessageExercise response from the server.
        """
        # TODO 15: Create a user message and send it
        # Hint: use MCPMessageExercise(role="user", content=text)
        return self.send(MCPMessageExercise(role="user", content=text))  # YOUR CODE HERE


class MCPAgentExercise:
    """Agent that communicates via the Model Context Protocol.

    TODO: Implement all methods.

    Args:
        llm_fn: A callable that takes a list of message dicts and returns
            a string (the LLM's response).
        server: An MCPServerExercise instance with registered tools.
    """

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        server: "MCPServerExercise",
    ):
        """Initialize the MCP agent."""
        # TODO 16: Store llm_fn and create an MCPClientExercise
        pass  # YOUR CODE HERE

    def run(self, prompt: str, max_turns: int = 5) -> str:
        """Run the agent loop until a final response or max turns.

        Args:
            prompt: The user's input prompt.
            max_turns: Maximum number of LLM calls before stopping.

        Returns:
            The final text response from the LLM.
        """
        # TODO 17: Initialize message history with user message
        # TODO 18: Loop up to max_turns:
        #   a. Call LLM with current messages
        #   b. Parse tool calls from response
        #   c. If no tool calls, return the response
        #   d. Send tool calls to server via MCPClient
        #   e. Add tool results to message history
        # TODO 19: Return last response if max turns reached
        return ""  # YOUR CODE HERE

    @staticmethod
    def _parse_tool_calls(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from LLM output text.

        Args:
            text: The LLM output text to parse.

        Returns:
            List of tool call dicts with "name" and "arguments" keys.
        """
        # TODO 20: Scan for ```tool_call fenced blocks
        # TODO 21: Parse JSON content from each block
        # TODO 22: Return list of dicts with "name" and "arguments"
        # Hint: look at the ToolParser in 01_tool_calling for reference
        return []  # YOUR CODE HERE
