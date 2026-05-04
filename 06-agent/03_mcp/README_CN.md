# 模型上下文协议 (MCP)

> **模块 06 -- 智能体，第 03 章**

模型上下文协议 (Model Context Protocol, MCP) 是大语言模型智能体与外部工具之间的标准化通信层。MCP 不使用临时的函数调用，而是定义了一套清晰的基于消息的协议，所有交互都通过结构化消息进行，将客户端（通信）、服务器（工具管理）和智能体（推理循环）的关注点分离。

---

## 前置知识

- Python 基础：函数、类、字典、JSON
- 对工具调用的理解（模块 06，第 01 章）
- 基本的客户端-服务器架构知识

## 文件说明

| 文件 | 用途 |
|------|------|
| `mcp.py` | 核心 MCP 实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

MCP 架构将职责分为三层：

```
用户提示
    |
    v
MCPAgent（推理循环）
    |
    +---> MCPClient（通信层）
    |         |
    |         v
    |     MCPServer（工具管理 + 执行）
    |         |
    |         +---> 工具 A
    |         +---> 工具 B
    |         +---> 工具 C
    |
    v
最终响应
```

所有层间通信都使用 `MCPMessage` 对象：

```
MCPMessage
    +-- role: "user" | "assistant" | "tool" | "system"
    +-- content: str
    +-- tool_calls: [{"name": ..., "arguments": ...}]
    +-- tool_results: [{"name": ..., "result": ...}]
    +-- metadata: dict
```

### 核心思想：协议优于管道

MCP 将"做什么"（工具定义和逻辑）与"怎么做"（通信格式）分离开来。这使得更换传输层（HTTP、WebSocket、进程内调用）变得容易，而无需修改工具实现或智能体逻辑。

---

## 架构详解

### MCPMessage

通用消息格式。系统中的每一条信息 -- 用户输入、LLM 输出、工具请求、工具结果 -- 都是一个 MCPMessage。这种统一性使系统易于调试、记录和扩展。

```python
msg = MCPMessage(role="user", content="2 + 2 等于多少？")
data = msg.to_dict()  # 序列化用于存储/传输
restored = MCPMessage.from_dict(data)  # 反序列化
```

### MCPServer

服务器拥有工具。它注册工具、按名称查找工具、执行工具并捕获错误。当收到包含工具调用的消息时，服务器运行每个工具并返回包含结果的消息。

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

一个轻量级的通信层。客户端向服务器发送消息并返回响应。在这个简单实现中，它是一个直接的函数调用，但该接口可以扩展为使用 HTTP、gRPC 或其他传输方式。

```python
client = MCPClient(server)
response = client.send(MCPMessage(role="user", content="你好"))
```

### MCPAgent

智能体将所有组件组合在一起。它运行一个循环：
1. 向 LLM 发送消息。
2. 解析 LLM 输出中的工具调用。
3. 如果找到工具调用，通过客户端发送到服务器。
4. 将工具结果添加到消息历史。
5. 重复直到没有工具调用或达到最大轮数。

```python
agent = MCPAgent(llm_fn, server)
result = agent.run("2 + 3 等于多少？")
```

---

## 前向传播详解

### 简单请求（无工具）

```
输入:    "2 + 3 等于多少？"
LLM:     "答案是 5。"
输出:    "答案是 5。"
```

### 带工具调用的请求

```
输入:    "2 + 3 等于多少？"

第 1 轮:
  LLM:   "我需要计算一下。
          ```tool_call
          {"name": "add", "arguments": {"a": 2, "b": 3}}
          ```"
  智能体: 通过客户端将工具调用发送到服务器
  服务器: 执行 add(2, 3) -> "5"
  结果:   MCPMessage(role="tool", tool_results=[{name: "add", result: "5"}])

第 2 轮:
  LLM:   "结果是 5。"
输出:    "结果是 5。"
```

### 多轮工具链

```
输入:    "计算 (10 + 20) * 2"

第 1 轮:  LLM 调用 add(10, 20) -> "30"
第 2 轮:  LLM 调用 mul(30, 2) -> "60"
第 3 轮:  LLM: "最终答案是 60。"
输出:    "最终答案是 60。"
```

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/03_mcp/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 06-agent/03_mcp/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现 MCP 系统。

### 练习顺序

1. **MCPMessage**：实现创建、验证、序列化、反序列化
2. **MCPServer**：实现工具注册和请求处理
3. **MCPClient**：实现 send 和 send_and_receive
4. **MCPAgent**：实现带工具调用解析的智能体循环

### 提示

- 对于 MCPMessage，在 `__init__` 中验证角色，对无效角色抛出 ValueError。
- 使用 `or []` 和 `or {}` 模式来处理 None 默认值。
- 服务器应捕获所有工具执行异常并返回错误字符串。
- 智能体的 `_parse_tool_calls` 使用与模块 06 第 01 章相同的围栏代码块格式。
- 使用字典列表跟踪消息历史，以便 LLM 消费。

---

## 关键要点

1. **协议标准化通信。** 通过定义清晰的消息格式 (MCPMessage)，所有组件可以在不了解彼此内部实现的情况下进行通信。

2. **关注点分离提高可维护性。** 服务器管理工具，客户端处理传输，智能体专注于推理。每个部分都可以独立修改。

3. **错误处理在边界进行。** 服务器捕获工具错误并将其转换为消息结果，因此智能体永远不会因为错误的工具调用而崩溃。

4. **客户端抽象提供灵活性。** 今天它是直接的函数调用；明天它可以是到远程服务的 HTTP 请求。智能体不需要改变。

5. **消息历史是智能体的记忆。** 通过累积所有消息（用户、助手、工具），智能体为 LLM 保持了完整的推理上下文。

---

## 延伸阅读

- [模型上下文协议规范](https://modelcontextprotocol.io/) -- 官方 MCP 规范
- [Anthropic 的 MCP 公告](https://www.anthropic.com/news/model-context-protocol) -- MCP 介绍
- [LLM 工具调用](https://platform.openai.com/docs/guides/function-calling) -- OpenAI 函数调用参考
- [ReAct：推理与行动 (Yao 等，2022)](https://arxiv.org/abs/2210.03629) -- 智能体循环模式
