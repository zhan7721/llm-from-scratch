# Tool Calling

> **Module 06 -- Agents, Chapter 01**

Tool calling allows an LLM to invoke external functions during a conversation. Instead of only generating text, the model can request that a specific tool be run with given arguments, receive the result, and continue reasoning. This is the foundation for building capable AI agents.

---

## Prerequisites

- Python basics: functions, dictionaries, JSON, exception handling
- Understanding of how LLMs generate text (Module 01)
- Familiarity with the OpenAI message format: `[{"role": "user", "content": "..."}]`

## Files

| File | Purpose |
|------|---------|
| `tool_calling.py` | Core tool calling implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A tool-calling agent follows a generate-parse-execute loop:

```
User prompt
    |
    v
LLM generates text (may include ```tool_call blocks)
    |
    v
ToolParser extracts tool calls from the response
    |
    +-- No tool calls --> Return text response
    |
    +-- Tool calls found
        |
        v
    ToolExecutor runs each tool
        |
        v
    Results appended to message history
        |
        v
    Loop back to LLM (repeat until done or max turns)
```

### Key Insight: Structured Output as Communication

The LLM communicates tool requests through structured text -- JSON inside fenced code blocks. This is a simple, model-agnostic approach that works with any LLM that can follow formatting instructions, without requiring special API support for function calling.

---

## Tool Call Format

We use a simple format: JSON inside ` ```tool_call ` fenced blocks.

```
I need to add two numbers.

```tool_call
{"name": "add", "arguments": {"a": 3, "b": 4}}
```
```

Multiple tool calls can appear in a single response:

```
Let me do two things at once.

```tool_call
{"name": "add", "arguments": {"a": 1, "b": 2}}
```

```tool_call
{"name": "greet", "arguments": {"name": "Alice"}}
```
```

---

## Architecture Details

### ToolRegistry

Manages a collection of tools, each with:
- **name**: Unique identifier
- **fn**: The callable function
- **description**: Human-readable explanation
- **parameters**: JSON-schema-like dict of expected arguments

```python
registry = ToolRegistry()
registry.register(
    name="search",
    fn=lambda query: f"Results for: {query}",
    description="Search the web",
    parameters={"query": {"type": "string"}},
)
```

### ToolParser

Extracts tool calls from LLM output text by:
1. Finding ` ```tool_call ` fenced blocks
2. Parsing the JSON content
3. Validating the structure (must have "name" key)
4. Gracefully handling malformed JSON (returns empty list)

### ToolExecutor

Runs tool calls and handles errors:
1. Looks up the tool by name in the registry
2. Calls `tool["fn"](**arguments)`
3. Catches exceptions and returns error strings (never raises)
4. Formats results as strings

### ToolCallingAgent

Orchestrates the full loop:
1. Creates messages with the user prompt
2. Calls the LLM function
3. Parses for tool calls
4. If none found, returns the response
5. If found, executes tools and appends results
6. Repeats until no tool calls or max_turns reached

---

## Message Format

Messages follow the OpenAI chat format:

```python
[
    {"role": "user", "content": "What is 3 + 4?"},
    {"role": "assistant", "content": "I'll use the add tool.\n```tool_call\n...\n```"},
    {"role": "tool", "content": "7"},
    {"role": "assistant", "content": "The answer is 7."},
]
```

---

## Forward Pass Walkthrough

### Single-Turn (No Tools)

```
Input:  "Hello!"
LLM:    "Hi there! How can I help?"
Parse:  [] (no tool calls)
Output: "Hi there! How can I help?"
```

### Multi-Turn (With Tools)

```
Turn 1:
    Input:  "What is 3 + 4?"
    LLM:    "```tool_call\n{"name": "add", "arguments": {"a": 3, "b": 4}}\n```"
    Parse:  [{"name": "add", "arguments": {"a": 3, "b": 4}}]
    Execute: "7"
    Messages: [user, assistant, tool="7"]

Turn 2:
    LLM:    "The result of 3 + 4 is 7."
    Parse:  [] (no tool calls)
    Output: "The result of 3 + 4 is 7."
```

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/01_tool_calling/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 06-agent/01_tool_calling/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the tool calling system yourself.

### Exercise Order

1. **ToolRegistry**: Initialize storage, register tools, get by name, list, and generate schemas
2. **ToolParser**: Find fenced blocks, parse JSON, handle errors
3. **ToolExecutor**: Look up tools, execute with arguments, handle errors, format results
4. **ToolCallingAgent**: Implement the generate-parse-execute loop

### Tips

- Use `dict.get()` with defaults for safe dictionary access.
- Wrap `json.loads()` in `try/except (json.JSONDecodeError, TypeError)` for robustness.
- Tool errors should be caught and returned as strings, never propagated as exceptions.
- The agent loop should stop when no tool calls are found OR max_turns is reached.

---

## Key Takeaways

1. **Tool calling is a structured communication protocol.** The LLM expresses tool requests as JSON in fenced blocks, and the agent parses and executes them.

2. **Graceful error handling is essential.** Tools can fail for many reasons (wrong arguments, network errors, bugs). The executor must catch errors and report them without crashing.

3. **The agent loop is simple but powerful.** Generate, parse, execute, repeat -- this pattern scales to complex multi-step tasks.

4. **Message history maintains context.** By appending tool results to the conversation, the LLM can reason about previous tool outputs in subsequent turns.

5. **Max turns prevents infinite loops.** Without a turn limit, a misbehaving LLM could loop forever requesting tool calls.

---

## Further Reading

- [Function Calling with OpenAI](https://platform.openai.com/docs/guides/function-calling) -- OpenAI's native function calling API
- [Anthropic Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) -- Claude's tool use capabilities
- [ReAct (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) -- Reasoning and Acting with LLMs
- [Toolformer (Schick et al., 2023)](https://arxiv.org/abs/2302.04761) -- LMs that learn to use tools
