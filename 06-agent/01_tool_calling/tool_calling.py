"""Tool Calling implementation for LLM agents.

This module implements a simple tool-calling system that allows an LLM to
invoke external functions during a conversation. It contains:
- ToolRegistry: Register and manage callable tools with JSON-like schemas.
- ToolParser: Parse LLM output to extract tool calls from fenced code blocks.
- ToolExecutor: Execute tool calls and format results.
- ToolCallingAgent: An LLM + tool loop that generates, parses, and executes
  tool calls until the LLM produces a final text response.

Architecture overview:
    User prompt
        -> LLM generates text (may include ```tool_call blocks)
        -> ToolParser extracts tool calls
        -> ToolExecutor runs each tool
        -> Results appended to message history
        -> Loop until no tool calls or max turns reached
        -> Final text response

Design notes:
- Tool calls are represented as JSON inside ```tool_call fenced blocks.
- The LLM function signature is fn(messages: list[dict]) -> str, where
  messages follow the OpenAI format: [{"role": "user", "content": "..."}].
- This module is self-contained (no external dependencies beyond the
  standard library).
- Error handling is graceful: errors are caught and reported, never raised.
"""

import json
from typing import Any, Callable


__all__ = [
    "ToolRegistry",
    "ToolParser",
    "ToolExecutor",
    "ToolCallingAgent",
]


# ---------------------------------------------------------------------------
# ToolRegistry
# ---------------------------------------------------------------------------


class ToolRegistry:
    """Register and manage callable tools with schemas.

    Each tool has a name, a callable function, a human-readable description,
    and a parameter schema describing the expected arguments.

    Example:
        registry = ToolRegistry()
        registry.register(
            name="add",
            fn=lambda a, b: a + b,
            description="Add two numbers",
            parameters={"a": {"type": "number"}, "b": {"type": "number"}},
        )
    """

    def __init__(self):
        """Initialize the tool registry."""
        self._tools: dict[str, dict[str, Any]] = {}

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
        self._tools[name] = {
            "name": name,
            "fn": fn,
            "description": description,
            "parameters": parameters,
        }

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a tool by name.

        Args:
            name: The tool name to look up.

        Returns:
            The tool dict with keys {name, fn, description, parameters},
            or None if not found.
        """
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names.

        Returns:
            Sorted list of tool names.
        """
        return sorted(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-like schemas for all registered tools.

        Each schema includes the tool's name, description, and parameters,
        suitable for passing to an LLM as a function-calling specification.

        Returns:
            List of tool schema dicts.
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
# ToolParser
# ---------------------------------------------------------------------------


class ToolParser:
    """Parse LLM output to extract tool calls.

    Tool calls are expected in the following format within the LLM output:

        ```tool_call
        {"name": "tool_name", "arguments": {"arg1": "value1"}}
        ```

    Multiple tool calls can appear in a single response, each in its own
    fenced block.
    """

    @staticmethod
    def parse(text: str) -> list[dict[str, Any]]:
        """Extract tool calls from LLM output text.

        Scans the text for ```tool_call fenced blocks and parses each one
        as a JSON object with "name" and "arguments" keys.

        Args:
            text: The LLM output text to parse.

        Returns:
            List of tool call dicts, each with keys "name" (str) and
            "arguments" (dict). Returns an empty list if no tool calls are
            found or if JSON parsing fails.
        """
        tool_calls: list[dict[str, Any]] = []
        marker_start = "```tool_call"
        marker_end = "```"

        remaining = text
        while True:
            # Find the start of a tool_call block
            start_idx = remaining.find(marker_start)
            if start_idx == -1:
                break

            # Find the closing ``` after the opening
            content_start = remaining.find("\n", start_idx)
            if content_start == -1:
                break
            content_start += 1  # skip the newline

            end_idx = remaining.find(marker_end, content_start)
            if end_idx == -1:
                break

            # Extract and parse the JSON content
            json_str = remaining[content_start:end_idx].strip()
            remaining = remaining[end_idx + len(marker_end):]

            if not json_str:
                continue

            try:
                parsed = json.loads(json_str)
                # Validate expected structure
                if isinstance(parsed, dict) and "name" in parsed:
                    tool_calls.append({
                        "name": parsed["name"],
                        "arguments": parsed.get("arguments", {}),
                    })
            except (json.JSONDecodeError, TypeError):
                # Malformed JSON -- skip this block
                continue

        return tool_calls


# ---------------------------------------------------------------------------
# ToolExecutor
# ---------------------------------------------------------------------------


class ToolExecutor:
    """Execute tool calls and format results.

    Uses a ToolRegistry to look up tools by name and run them with the
    provided arguments.

    Example:
        executor = ToolExecutor(registry)
        result = executor.execute({"name": "add", "arguments": {"a": 1, "b": 2}})
    """

    def __init__(self, registry: ToolRegistry):
        """Initialize with a ToolRegistry.

        Args:
            registry: The ToolRegistry containing available tools.
        """
        self.registry = registry

    def execute(self, tool_call: dict[str, Any]) -> str:
        """Execute a single tool call.

        Looks up the tool by name and invokes it with the given arguments.
        Errors are caught and returned as error strings rather than raised.

        Args:
            tool_call: A dict with keys "name" (str) and "arguments" (dict).

        Returns:
            The formatted result string, or an error message if execution fails.
        """
        name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        tool = self.registry.get(name)
        if tool is None:
            return f"Error: tool '{name}' not found"

        try:
            result = tool["fn"](**arguments)
            return self.format_result(result)
        except Exception as e:
            return f"Error: {type(e).__name__}: {e}"

    @staticmethod
    def format_result(result: Any) -> str:
        """Format a tool result as a string.

        Args:
            result: The raw result from a tool invocation.

        Returns:
            A string representation of the result.
        """
        if isinstance(result, str):
            return result
        return str(result)


# ---------------------------------------------------------------------------
# ToolCallingAgent
# ---------------------------------------------------------------------------


class ToolCallingAgent:
    """An LLM agent that can call tools in a loop.

    Combines an LLM function with a tool registry to implement a
    generate-parse-execute loop:
        1. Send messages to the LLM.
        2. Parse the LLM output for tool calls.
        3. If tool calls are found, execute them and append results.
        4. Repeat until no tool calls are found or max turns is reached.
        5. Return the final text response.

    Args:
        llm_fn: A callable that takes a list of messages (OpenAI format)
            and returns a string (the LLM's response).
        tools: A ToolRegistry containing the available tools.
    """

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        tools: ToolRegistry,
    ):
        """Initialize the tool calling agent."""
        self.llm_fn = llm_fn
        self.tools = tools
        self.parser = ToolParser()
        self.executor = ToolExecutor(tools)

    def run(self, prompt: str, max_turns: int = 5) -> str:
        """Run the generate-parse-execute loop.

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
            response = self.llm_fn(messages)

            # 2. Parse
            tool_calls = self.parser.parse(response)

            # 3. If no tool calls, return the response
            if not tool_calls:
                return response

            # 4. Execute tool calls and append results
            messages.append({"role": "assistant", "content": response})

            for tc in tool_calls:
                result = self.executor.execute(tc)
                messages.append({
                    "role": "tool",
                    "content": result,
                })

        # Max turns reached -- return the last response
        return response
