# 工具调用

> **模块 06 -- 智能体，第 01 章**

工具调用允许大语言模型在对话过程中调用外部函数。模型不仅能生成文本，还可以请求执行特定工具并接收结果，然后继续推理。这是构建强大 AI 智能体的基础。

---

## 前置知识

- Python 基础：函数、字典、JSON、异常处理
- 对大语言模型如何生成文本的基本理解（模块 01）
- 熟悉 OpenAI 消息格式：`[{"role": "user", "content": "..."}]`

## 文件说明

| 文件 | 用途 |
|------|------|
| `tool_calling.py` | 核心工具调用实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

工具调用智能体遵循一个 生成-解析-执行 循环：

```
用户提示
    |
    v
LLM 生成文本（可能包含 ```tool_call 代码块）
    |
    v
ToolParser 从响应中提取工具调用
    |
    +-- 无工具调用 --> 返回文本响应
    |
    +-- 发现工具调用
        |
        v
    ToolExecutor 执行每个工具
        |
        v
    结果追加到消息历史
        |
        v
    循环回到 LLM（重复直到完成或达到最大轮数）
```

### 核心思想：结构化输出作为通信方式

LLM 通过结构化文本（代码块中的 JSON）来表达工具请求。这是一种简单的、与模型无关的方法，适用于任何能够遵循格式指令的 LLM，无需特殊的函数调用 API 支持。

---

## 工具调用格式

我们使用一种简单格式：` ```tool_call ` 代码块中的 JSON。

```
我需要计算两个数的和。

```tool_call
{"name": "add", "arguments": {"a": 3, "b": 4}}
```
```

单个响应中可以出现多个工具调用：

```
让我同时做两件事。

```tool_call
{"name": "add", "arguments": {"a": 1, "b": 2}}
```

```tool_call
{"name": "greet", "arguments": {"name": "Alice"}}
```
```

---

## 架构详解

### ToolRegistry

管理工具集合，每个工具包含：
- **name**：唯一标识符
- **fn**：可调用的函数
- **description**：人类可读的说明
- **parameters**：类似 JSON schema 的参数描述

```python
registry = ToolRegistry()
registry.register(
    name="search",
    fn=lambda query: f"Results for: {query}",
    description="搜索网页",
    parameters={"query": {"type": "string"}},
)
```

### ToolParser

从 LLM 输出文本中提取工具调用：
1. 查找 ` ```tool_call ` 代码块
2. 解析 JSON 内容
3. 验证结构（必须有 "name" 键）
4. 优雅处理格式错误的 JSON（返回空列表）

### ToolExecutor

执行工具调用并处理错误：
1. 在注册表中按名称查找工具
2. 调用 `tool["fn"](**arguments)`
3. 捕获异常并返回错误字符串（从不抛出异常）
4. 将结果格式化为字符串

### ToolCallingAgent

编排完整循环：
1. 用用户提示创建消息
2. 调用 LLM 函数
3. 解析工具调用
4. 如果没有找到工具调用，返回响应
5. 如果找到，执行工具并追加结果
6. 重复直到没有工具调用或达到最大轮数

---

## 消息格式

消息遵循 OpenAI 聊天格式：

```python
[
    {"role": "user", "content": "3 + 4 等于多少？"},
    {"role": "assistant", "content": "我来用 add 工具计算。\n```tool_call\n...\n```"},
    {"role": "tool", "content": "7"},
    {"role": "assistant", "content": "3 + 4 的结果是 7。"},
]
```

---

## 前向传播详解

### 单轮（无工具调用）

```
输入:  "你好！"
LLM:   "你好！有什么可以帮你的吗？"
解析:  []（无工具调用）
输出:  "你好！有什么可以帮你的吗？"
```

### 多轮（有工具调用）

```
第 1 轮：
    输入:  "3 + 4 等于多少？"
    LLM:   "```tool_call\n{"name": "add", "arguments": {"a": 3, "b": 4}}\n```"
    解析:  [{"name": "add", "arguments": {"a": 3, "b": 4}}]
    执行:  "7"
    消息:  [user, assistant, tool="7"]

第 2 轮：
    LLM:   "3 + 4 的结果是 7。"
    解析:  []（无工具调用）
    输出:  "3 + 4 的结果是 7。"
```

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/01_tool_calling/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 06-agent/01_tool_calling/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现工具调用系统。

### 练习顺序

1. **ToolRegistry**：初始化存储、注册工具、按名称获取、列出、生成 schema
2. **ToolParser**：查找代码块、解析 JSON、处理错误
3. **ToolExecutor**：查找工具、用参数执行、处理错误、格式化结果
4. **ToolCallingAgent**：实现 生成-解析-执行 循环

### 提示

- 使用 `dict.get()` 配合默认值进行安全的字典访问。
- 用 `try/except (json.JSONDecodeError, TypeError)` 包裹 `json.loads()` 以增强健壮性。
- 工具错误应被捕获并以字符串形式返回，永远不要作为异常传播。
- 当没有发现工具调用或达到最大轮数时，智能体循环应停止。

---

## 关键要点

1. **工具调用是一种结构化通信协议。** LLM 在代码块中以 JSON 形式表达工具请求，智能体解析并执行它们。

2. **优雅的错误处理至关重要。** 工具可能因多种原因失败（参数错误、网络错误、bug）。执行器必须捕获错误并报告，而不是崩溃。

3. **智能体循环简单但强大。** 生成、解析、执行、重复——这个模式可以扩展到复杂的多步骤任务。

4. **消息历史维护上下文。** 通过将工具结果追加到对话中，LLM 可以在后续轮次中对之前的工具输出进行推理。

5. **最大轮数防止无限循环。** 没有轮数限制，行为异常的 LLM 可能会永远循环请求工具调用。

---

## 延伸阅读

- [OpenAI 函数调用](https://platform.openai.com/docs/guides/function-calling) -- OpenAI 的原生函数调用 API
- [Anthropic 工具使用](https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview) -- Claude 的工具使用能力
- [ReAct (Yao 等，2022)](https://arxiv.org/abs/2210.03629) -- LLM 的推理与行动
- [Toolformer (Schick 等，2023)](https://arxiv.org/abs/2302.04761) -- 学习使用工具的语言模型
