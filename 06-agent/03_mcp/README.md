# Model Context Protocol (MCP)

> **Module 06 -- Agents, Chapter 03**

The Model Context Protocol (MCP) is a standardized communication layer between LLM agents and external tools. Instead of ad-hoc function calls, MCP defines a clean message-based protocol where all interactions flow through structured messages, separating concerns between the client (communication), server (tool management), and agent (reasoning loop).

---

## Prerequisites

- Python basics: functions, classes, dictionaries, JSON
- Understanding of tool calling (Module 06, Chapter 01)
- Basic familiarity with client-server architecture

## Files

| File | Purpose |
|------|---------|
| `mcp.py` | Core MCP implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

The MCP architecture separates responsibilities into three layers:

```
User prompt
    |
    v
MCPAgent (reasoning loop)
    |
    +---> MCPClient (communication layer)
    |         |
    |         v
    |     MCPServer (tool management + execution)
    |         |
    |         +---> Tool A
    |         +---> Tool B
    |         +---> Tool C
    |
    v
Final response
```

All communication between layers uses `MCPMessage` objects:

```
MCPMessage
    +-- role: "user" | "assistant" | "tool" | "system"
    +-- content: str
    +-- tool_calls: [{"name": ..., "arguments": ...}]
    +-- tool_results: [{"name": ..., "result": ...}]
    +-- metadata: dict
```

### Key Insight: Protocol Over Plumbing

MCP separates the "what" (tool definitions and logic) from the "how" (communication format). This makes it easy to swap out the transport layer (HTTP, WebSocket, in-process) without changing tool implementations or agent logic.

---

## Architecture Details

### MCPMessage

The universal message format. Every piece of information in the system -- user input, LLM output, tool requests, tool results -- is an MCPMessage. This uniformity makes the system easy to debug, log, and extend.

```python
msg = MCPMessage(role="user", content="What is 2 + 2?")
data = msg.to_dict()  # Serialize for storage/transport
restored = MCPMessage.from_dict(data)  # Deserialize
```

### MCPServer

The server owns the tools. It registers them, looks them up by name, executes them, and catches errors. When a message arrives with tool calls, the server runs each one and returns a message with results.

```python
server = MCPServer()
server.register_tool(
    name="add",
    fn=lambda a, b: a + b,
    description="Add two numbers",
    parameters={"a": {"type": "number"}, "b": {"type": "number"}},
)
```

### MCPClient

A thin communication layer. The client sends messages to the server and returns responses. In this simple implementation it's a direct function call, but the interface could be extended to use HTTP, gRPC, or any other transport.

```python
client = MCPClient(server)
response = client.send(MCPMessage(role="user", content="Hello"))
```

### MCPAgent

The agent ties everything together. It runs a loop:
1. Send messages to the LLM.
2. Parse the LLM output for tool calls.
3. If tool calls are found, send them to the server via the client.
4. Add tool results to the message history.
5. Repeat until no tool calls or max turns reached.

```python
agent = MCPAgent(llm_fn, server)
result = agent.run("What is 2 + 3?")
```

---

## Forward Pass Walkthrough

### Simple Request (No Tools)

```
Input:    "What is 2 + 3?"
LLM:      "The answer is 5."
Output:   "The answer is 5."
```

### Request with Tool Call

```
Input:    "What is 2 + 3?"

Turn 1:
  LLM:    "I need to calculate.
           ```tool_call
           {"name": "add", "arguments": {"a": 2, "b": 3}}
           ```"
  Agent:  Sends tool call to server via client
  Server: Executes add(2, 3) -> "5"
  Result: MCPMessage(role="tool", tool_results=[{name: "add", result: "5"}])

Turn 2:
  LLM:    "The result is 5."
Output:   "The result is 5."
```

### Multi-Turn Tool Chain

```
Input:    "Compute (10 + 20) * 2"

Turn 1:   LLM calls add(10, 20) -> "30"
Turn 2:   LLM calls mul(30, 2) -> "60"
Turn 3:   LLM: "The final answer is 60."
Output:   "The final answer is 60."
```

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/03_mcp/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 06-agent/03_mcp/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the MCP system yourself.

### Exercise Order

1. **MCPMessage**: Implement creation, validation, serialization, deserialization
2. **MCPServer**: Implement tool registration and request handling
3. **MCPClient**: Implement send and send_and_receive
4. **MCPAgent**: Implement the agent loop with tool call parsing

### Tips

- For MCPMessage, validate the role in `__init__` and raise ValueError for invalid roles.
- Use `or []` and `or {}` patterns to handle None defaults cleanly.
- The server should catch all tool execution exceptions and return error strings.
- The agent's `_parse_tool_calls` uses the same fenced-block format as Module 06, Chapter 01.
- Track message history as a list of dicts for LLM consumption.

---

## Key Takeaways

1. **Protocols standardize communication.** By defining a clear message format (MCPMessage), all components can communicate without knowing each other's internals.

2. **Separation of concerns improves maintainability.** The server manages tools, the client handles transport, and the agent focuses on reasoning. Each can be modified independently.

3. **Error handling belongs at the boundary.** The server catches tool errors and converts them to message results, so the agent never crashes from a bad tool call.

4. **The client abstraction enables flexibility.** Today it's a direct function call; tomorrow it could be an HTTP request to a remote service. The agent doesn't need to change.

5. **Message history is the agent's memory.** By accumulating all messages (user, assistant, tool), the agent maintains full context for the LLM to reason about.

---

## Further Reading

- [Model Context Protocol Specification](https://modelcontextprotocol.io/) -- The official MCP specification
- [Anthropic's MCP Announcement](https://www.anthropic.com/news/model-context-protocol) -- MCP introduction
- [Tool Calling for LLMs](https://platform.openai.com/docs/guides/function-calling) -- OpenAI function calling reference
- [ReAct: Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) -- The agent loop pattern
