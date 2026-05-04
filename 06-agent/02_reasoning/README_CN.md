# 思维链推理

> **模块 06 -- 智能体，第 02 章**

思维链 (Chain-of-Thought, CoT) 提示技术鼓励大语言模型在给出最终答案之前逐步推理。模型不是直接跳到结论，而是展示推理过程，这能显著提高多步骤问题的准确率。

---

## 前置知识

- Python 基础：函数、字符串、列表、字典
- 对大语言模型如何生成文本的基本理解（模块 01）
- 基本的提示技术知识

## 文件说明

| 文件 | 用途 |
|------|------|
| `reasoning.py` | 核心思维链推理实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

思维链推理系统的工作流程：

```
用户问题
    |
    v
ChainOfThoughtPrompter 添加 "逐步思考" 指令
    |
    v
LLM 生成推理过程
    |
    v
StepExtractor 将推理过程拆分为独立步骤
    |
    v
ReasoningVerifier 检查推理链
    |
    v
提取最终答案
```

为了提高鲁棒性，SelfConsistency 重复此过程多次：

```
问题
    |
    +---> 路径 1 --> 答案 A
    |
    +---> 路径 2 --> 答案 A
    |
    +---> 路径 3 --> 答案 B
    |
    +---> 路径 4 --> 答案 A
    |
    +---> 路径 5 --> 答案 B
    |
    v
多数投票 --> 答案 A
```

### 核心思想：展示推理过程可以提高准确率

当大语言模型逐步推理时，每个中间步骤都是在前面所有步骤的基础上生成的。这种自回归链将复杂问题分解为更简单的子问题，减少了每个阶段的错误。

---

## 架构详解

### ChainOfThoughtPrompter

用思维链指令格式化问题。支持三种风格：

- **step_by_step**："Let's think step by step."（经典 Kojima 等，2022）
- **think_aloud**："Think out loud, showing your reasoning process."
- **structured**："Break this down into numbered steps and show your work."

还支持带示例的少样本思维链。

```python
prompter = ChainOfThoughtPrompter()
prompt = prompter.format_prompt("15 * 23 等于多少？")
# -> "15 * 23 等于多少？\n\nLet's think step by step."
```

### StepExtractor

将 LLM 的推理输出解析为独立步骤。处理三种格式：

1. **编号格式**："1. 第一步  2. 第二步"
2. **项目符号**："- 第一步  - 第二步"
3. **段落分隔**：用空行分隔的步骤

优先尝试编号格式，然后是项目符号，最后是段落格式。

```python
extractor = StepExtractor()
steps = extractor.extract_steps("1. 计算 15*20=300\n2. 计算 15*3=45\n3. 相加 300+45=345")
# -> ["计算 15*20=300", "计算 15*3=45", "相加 300+45=345"]
```

### SelfConsistency

实现自一致性技术（Wang 等，2022）：

1. 生成 N 条推理路径（使用 temperature > 0 以增加多样性）
2. 从每条路径中提取最终答案
3. 通过多数投票返回最常见的答案

```python
def my_llm(prompt: str) -> str:
    return "第一步：... 答案是 42。"

sc = SelfConsistency(my_llm, num_paths=5)
answer = sc.solve("6 * 7 等于多少？")
```

### ReasoningVerifier

使用启发式规则检查推理链中的问题：

- 空的或缺失的步骤
- 步骤太短，没有意义
- 矛盾的步骤（后面的步骤否定了前面的步骤）
- 答案没有推理步骤的支持

```python
verifier = ReasoningVerifier()
result = verifier.verify_chain(["第一步...", "第二步...", "第三步..."])
# -> {"is_valid": True, "issues": []}
```

---

## 前向传播详解

### 简单思维链

```
输入:   "15 * 23 等于多少？"
提示:   "15 * 23 等于多少？\n\nLet's think step by step."
LLM:    "1. 15 * 20 = 300\n2. 15 * 3 = 45\n3. 300 + 45 = 345\n答案是 345。"
步骤:   ["15 * 20 = 300", "15 * 3 = 45", "300 + 45 = 345"]
答案:   "345"
```

### 自一致性

```
路径 1:  "15 * 20 = 300, 15 * 3 = 45, 300 + 45 = 345。答案：345。"
路径 2:  "15 * 23 = 345。答案：345。"
路径 3:  "15 * 20 = 300... 300 + 45 = 345。答案：345。"
路径 4:  "让我计算 15 * 23... = 345。答案：345。"
路径 5:  "15 * 23 = 345。答案：345。"

多数投票: "345"（5/5 条路径一致）
```

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/02_reasoning/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 06-agent/02_reasoning/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现思维链推理系统。

### 练习顺序

1. **ChainOfThoughtPrompter**：用思维链指令格式化提示，实现少样本示例
2. **StepExtractor**：解析编号步骤、项目符号和段落格式
3. **SelfConsistency**：实现答案提取和多数投票
4. **ReasoningVerifier**：检查空步骤、矛盾和答案支持

### 提示

- 对于答案提取，从文本末尾开始搜索以获取最后出现的匹配。
- 使用 `text.lower().rfind(pattern)` 进行不区分大小写的反向搜索。
- 对于多数投票，先规范化（小写、去空格）再计数，但返回原始大小写。
- 验证器应使用简单的启发式方法，而不是调用另一个大语言模型。

---

## 关键要点

1. **思维链提示是一种简单但强大的技术。** 在提示中添加 "Let's think step by step" 可以在零训练的情况下显著提高推理准确率。

2. **自一致性提高可靠性。** 通过采样多条推理路径并投票，我们减少了单个推理错误的影响。

3. **步骤提取使验证成为可能。** 将推理过程拆分为离散步骤，允许我们独立检查每一步。

4. **启发式验证能捕获明显错误。** 虽然不完美，但检查空步骤、矛盾和不受支持的答案能捕获许多常见的失败模式。

5. **答案提取模式很重要。** 大语言模型经常用 "The answer is" 或 "Therefore" 等短语来表示最终答案。识别这些模式对于可靠的自一致性至关重要。

---

## 延伸阅读

- [思维链提示 (Wei 等，2022)](https://arxiv.org/abs/2201.11903) -- 原始 CoT 论文
- [自一致性 (Wang 等，2022)](https://arxiv.org/abs/2203.11171) -- 采样多条推理路径
- [大语言模型是零样本推理器 (Kojima 等，2022)](https://arxiv.org/abs/2205.11916) -- "Let's think step by step"
- [思维树 (Yao 等，2023)](https://arxiv.org/abs/2305.10601) -- 用树搜索扩展思维链
