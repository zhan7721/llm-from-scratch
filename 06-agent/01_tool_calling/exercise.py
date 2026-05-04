"""Exercise: Implement a Tool Calling system for LLM agents.

Complete the TODOs below to build a tool-calling pipeline that lets an LLM
invoke external functions during a conversation.
Run `pytest tests.py` to verify your implementation.
"""

import json
from typing import Any, Callable


class ToolRegistryExercise:
    """Register and manage callable tools with schemas.

    TODO: Implement all methods.

    Each tool has a name, a callable function, a description, and a
    parameter schema.

    Args:
        None (create with default constructor).
    """

    def __init__(self):
        # TODO 1: Initialize the internal storage for tools
        # Hint: use a dict mapping tool name -> tool info dict
        self._tools = None  # YOUR CODE HERE

    def register(
        self,
        name: str,
        fn: Callable,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool.

        Args:
            name: Unique tool name.
            fn: Callable that implements the tool.
            description: Human-readable description of what the tool does.
            parameters: JSON-schema-like dict describing the tool's parameters.
        """
        # TODO 2: Store the tool info in self._tools
        # Hint: self._tools[name] = {"name": name, "fn": fn, ...}
        pass  # YOUR CODE HERE

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a tool by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool dict, or None if not found.
        """
        # TODO 3: Return the tool info for the given name, or None
        return None  # YOUR CODE HERE

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            Sorted list of tool names.
        """
        # TODO 4: Return sorted list of tool names
        return []  # YOUR CODE HERE

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-like schemas for all registered tools.

        Returns:
            List of tool schema dicts with name, description, parameters.
        """
        # TODO 5: Build and return schemas for all tools
        return []  # YOUR CODE HERE


class ToolParserExercise:
    """Parse LLM output to extract tool calls.

    TODO: Implement the parse method.

    Tool calls appear in the following format:

        ```tool_call
        {"name": "tool_name", "arguments": {"arg1": "value1"}}
        ```
    """

    @staticmethod
    def parse(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from LLM output text.

        Args:
            text: The LLM output text to parse.

        Returns:
            List of tool call dicts with "name" and "arguments" keys.
            Empty list if no tool calls found or JSON parsing fails.
        """
        # TODO 6: Find ```tool_call fenced blocks and parse the JSON inside
        # Hints:
        # - Use text.find("```tool_call") to locate blocks
        # - Find the closing ``` after the opening
        # - Use json.loads() to parse the content
        # - Wrap in try/except for malformed JSON
        # - Validate that parsed dict has a "name" key
        return []  # YOUR CODE HERE


class ToolExecutorExercise:
    """Execute tool calls and format results.

    TODO: Implement all methods.

    Args:
        registry: A ToolRegistry containing available tools.
    """

    def __init__(self, registry: ToolRegistryExercise):
        # TODO 7: Store the registry
        self.registry = None  # YOUR CODE HERE

    def execute(self, tool_call: dict[str, Any]) -> str:
        """Execute a single tool call.

        Args:
            tool_call: A dict with "name" and "arguments" keys.

        Returns:
            The formatted result string, or an error message.
        """
        # TODO 8: Look up the tool by name, call it with arguments
        # Hints:
        # - Get name and arguments from tool_call
        # - Look up tool in registry (handle not found)
        # - Call tool["fn"](**arguments)
        # - Wrap in try/except and return error message on failure
        return ""  # YOUR CODE HERE

    @staticmethod
    def format_result(result: Any) -> str:
        """Format a tool result as a string.

        Args:
            result: The raw result from a tool invocation.

        Returns:
            A string representation of the result.
        """
        # TODO 9: Return result as-is if string, otherwise str(result)
        return ""  # YOUR CODE HERE


class ToolCallingAgentExercise:
    """An LLM agent that can call tools in a loop.

    TODO: Implement the run method.

    Args:
        llm_fn: Callable that takes messages list and returns string.
        tools: A ToolRegistry with available tools.
    """

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        tools: ToolRegistryExercise,
    ):
        self.llm_fn = llm_fn
        self.tools = tools
        self.parser = ToolParserExercise()
        self.executor = ToolExecutorExercise(tools)

    def run(self, prompt: str, max_turns: int = 5) -> str:
        """Run the generate-parse-execute loop.

        Args:
            prompt: The user's input prompt.
            max_turns: Maximum number of LLM calls.

        Returns:
            The final text response from the LLM.
        """
        # TODO 10: Implement the agent loop
        # Hints:
        # 1. Start with messages = [{"role": "user", "content": prompt}]
        # 2. For each turn (up to max_turns):
        #    a. Call self.llm_fn(messages) to get a response
        #    b. Parse the response with self.parser.parse(response)
        #    c. If no tool calls, return the response
        #    d. Otherwise, append assistant message and execute tools
        #    e. Append tool results as {"role": "tool", "content": result}
        # 3. Return the last response if max_turns reached
        return ""  # YOUR CODE HERE
