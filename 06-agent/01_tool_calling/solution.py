"""Reference solution for the Tool Calling exercise."""

import json
from typing import Any, Callable


class ToolRegistrySolution:
    """Register and manage callable tools with schemas."""

    def __init__(self):
        self._tools: dict[str, dict[str, Any]] = {}

    def register(
        self,
        name: str,
        fn: Callable,
        description: str,
        parameters: dict[str, Any],
    ) -> None:
        """Register a tool."""
        self._tools[name] = {
            "name": name,
            "fn": fn,
            "description": description,
            "parameters": parameters,
        }

    def get(self, name: str) -> dict[str, Any] | None:
        """Get a tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> list[str]:
        """List all registered tool names."""
        return sorted(self._tools.keys())

    def get_schemas(self) -> list[dict[str, Any]]:
        """Return JSON-like schemas for all registered tools."""
        return [
            {
                "name": info["name"],
                "description": info["description"],
                "parameters": info["parameters"],
            }
            for info in self._tools.values()
        ]


class ToolParserSolution:
    """Parse LLM output to extract tool calls."""

    @staticmethod
    def parse(text: str) -> list[dict[str, Any]]:
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


class ToolExecutorSolution:
    """Execute tool calls and format results."""

    def __init__(self, registry: ToolRegistrySolution):
        self.registry = registry

    def execute(self, tool_call: dict[str, Any]) -> str:
        """Execute a single tool call."""
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
        """Format a tool result as a string."""
        if isinstance(result, str):
            return result
        return str(result)


class ToolCallingAgentSolution:
    """An LLM agent that can call tools in a loop."""

    def __init__(
        self,
        llm_fn: Callable[[list[dict[str, str]]], str],
        tools: ToolRegistrySolution,
    ):
        self.llm_fn = llm_fn
        self.tools = tools
        self.parser = ToolParserSolution()
        self.executor = ToolExecutorSolution(tools)

    def run(self, prompt: str, max_turns: int = 5) -> str:
        """Run the generate-parse-execute loop."""
        messages: list[dict[str, str]] = [
            {"role": "user", "content": prompt},
        ]

        for _ in range(max_turns):
            response = self.llm_fn(messages)

            tool_calls = self.parser.parse(response)

            if not tool_calls:
                return response

            messages.append({"role": "assistant", "content": response})

            for tc in tool_calls:
                result = self.executor.execute(tc)
                messages.append({
                    "role": "tool",
                    "content": result,
                })

        return response
