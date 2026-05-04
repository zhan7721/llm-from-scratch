# 代码解释器

> **模块 06 -- 智能体，第 05 章**

代码解释器允许大语言模型生成 Python 代码，在沙箱环境中执行，并将结果反馈给模型进行迭代优化。这是 ChatGPT 的代码解释器和 Claude 的分析工具背后的核心模式 -- 模型编写代码，运行时执行代码，输出结果指导下一步操作。

---

## 前置知识

- Python 基础：函数、类、exec()、线程
- 理解 LLM 智能体（模块 06，第 01-03 章）
- 熟悉基于消息的 LLM 交互（OpenAI 格式）

## 文件说明

| 文件 | 用途 |
|------|------|
| `code_interpreter.py` | 核心代码解释器实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 整体架构

代码解释器在 LLM 和 Python 运行时之间创建生成-执行循环：

```
用户任务
    |
    v
CodeInterpreterAgent
    |
    +---> LLM 生成包含 ```python 代码块的响应
    |
    +---> CodeParser 提取代码块
    |
    +---> SandboxExecutor 运行代码（隔离命名空间，超时控制）
    |
    +---> 结果反馈给 LLM
    |
    +---> 重复直到没有代码块或达到最大轮次
    |
    v
最终答案
```

### 核心组件

```
SandboxExecutor
    +-- execute(code, timeout)   （运行代码，返回 output/error/success）
    +-- namespace                （跨调用持久化）

CodeParser
    +-- parse(text)              （提取 ```python 代码块）

CodeInterpreterAgent
    +-- run(task)                （生成-执行循环）
    +-- sandbox: SandboxExecutor
    +-- parser: CodeParser

ArtifactStore
    +-- store(name, data, type)  （存储工件）
    +-- retrieve(name)           （获取工件）
    +-- list_artifacts()         （列出所有名称）
    +-- clear()                  （清空）
```

### 核心思想：生成-执行循环

代码解释器的威力来自迭代。LLM 不需要第一次就写对代码 -- 它可以查看执行结果、调试错误并改进方案。这和人类写代码的方式一样：先试试，看输出，再调整。

---

## 架构细节

### SandboxExecutor

在持久化的隔离命名空间中执行 Python 代码。使用 `exec()` 和受限的全局字典（仅包含内置函数）。分别捕获标准输出和标准错误。通过 `threading.Thread` 的 `join(timeout=...)` 实现超时控制。

```python
executor = SandboxExecutor()
result = executor.execute("x = 42\nprint(x)")
# result == {"output": "42\n", "error": "", "success": True}

result = executor.execute("print(x + 1)")  # x 持久存在！
# result == {"output": "43\n", "error": "", "success": True}
```

### CodeParser

使用正则表达式从 LLM 的 markdown 输出中提取 Python 代码。支持 `python`、`py` 和 `python3` 语言标签（不区分大小写）。返回代码字符串列表。

```python
parser = CodeParser()
blocks = parser.parse("```python\nprint('hi')\n```")
# blocks == ["print('hi')"]
```

### CodeInterpreterAgent

将所有组件串联起来。接收一个 LLM 函数和最大轮次限制。运行生成-执行循环：调用 LLM、解析代码块、执行代码、反馈结果、重复。

```python
agent = CodeInterpreterAgent(llm_fn=my_llm, max_turns=5)
result = agent.run("计算前 10 个斐波那契数")
```

### ArtifactStore

用于存储代码解释过程中产生的工件的简单键值存储。工件可以是任何 Python 对象 -- 文本、数据、图片、模型。

```python
store = ArtifactStore()
store.store("result", [1, 2, 3], artifact_type="data")
store.retrieve("result")  # -> [1, 2, 3]
```

---

## 前向传播示例

### 单轮执行

```
任务："2 + 2 等于多少？"

LLM："让我算一下：
      ```python
      print(2 + 2)
      ```"

解析器：提取 "print(2 + 2)"

执行器：运行代码 -> 输出: "4\n"

LLM："结果是 4。"

没有代码块 -> 返回 "结果是 4。"
```

### 多轮调试

```
任务："计算 10 的阶乘"

第 1 轮：
    LLM："```python
          def fact(n):
              return n * fact(n-1) if n > 1 else 1
          print(fact(10))
          ```"
    执行器：RecursionError！
    结果反馈："Error: RecursionError: ..."

第 2 轮：
    LLM："```python
          def fact(n):
              r = 1
              for i in range(2, n+1):
                  r *= i
              return r
          print(fact(10))
          ```"
    执行器：输出: "3628800\n"

第 3 轮：
    LLM："10 的阶乘是 3,628,800。"
    没有代码块 -> 返回答案
```

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/05_code_interpreter/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分，然后验证：

```bash
pytest 06-agent/05_code_interpreter/tests.py -v
```

---

## 练习

打开 `exercise.py` 来亲手实现代码解释器。

### 练习顺序

1. **SandboxExecutor**：实现命名空间初始化和带超时的代码执行
2. **CodeParser**：实现正则表达式模式和代码块提取
3. **CodeInterpreterAgent**：实现生成-执行循环
4. **ArtifactStore**：实现存储、获取、列表和清空

### 提示

- SandboxExecutor 使用守护线程，这样超时时不会阻塞主进程。
- CodeParser 使用 `re.DOTALL`，让正则中的 `.` 能匹配换行符。
- CodeInterpreterAgent 逐步构建消息列表 -- 将助手响应和执行结果作为独立消息追加。
- LLM 函数签名是 `fn(messages: list[dict]) -> str`，其中 messages 遵循 OpenAI 格式。

---

## 核心要点

1. **代码执行扩展了 LLM 的能力。** 通过生成和运行代码，LLM 可以进行精确计算、数据分析和可视化，这些是纯文本生成无法做到的。

2. **生成-执行循环是迭代的。** LLM 不需要第一次就完美。查看执行结果（包括错误）使其能够自我修正，就像人类程序员一样。

3. **沙箱化很重要。** 限制执行命名空间和强制超时可以防止代码失控并限制错误的影响范围。生产系统会增加额外的层级（容器、资源限制）。

4. **持久化状态支持多步工作流。** 跨执行保持的变量让 LLM 可以逐步构建复杂的分析 -- 一轮定义辅助函数，下一轮使用它们。

5. **解析是桥梁。** 代码解析器将非结构化的 LLM 文本转换为可执行代码。支持多种语言标签和处理边界情况（无代码块、多个代码块）使系统更加健壮。

---

## 延伸阅读

- [OpenAI 代码解释器](https://openai.com/index/chatgpt-can-now-see-hear-and-speak/) -- OpenAI 的代码执行能力
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) -- 开源 LLM 代码解释器
- [E2B: AI 代码执行](https://e2b.dev/) -- AI 代码执行的沙箱云环境
- [Python exec()](https://docs.python.org/3/library/functions.html#exec) -- Python exec() 文档
