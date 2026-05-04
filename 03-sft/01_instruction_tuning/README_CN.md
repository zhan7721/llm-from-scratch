# 指令微调 (Instruction Tuning)

> **模块 03 — 有监督微调，第 01 章**

预训练语言模型了解大量语言知识，但它不知道如何遵循指令。它补全文本，而不是回答问题。指令微调弥合了这一差距：通过在（指令，回复）对上进行微调，我们教会模型将指令视为上下文，并生成有帮助的回复。

本章实现核心机制：Alpaca 风格的提示格式化、标签遮罩（使模型仅从回复中学习），以及尊重遮罩的损失函数。

---

## 前置知识

- Transformer 语言模型基础（模块 01）
- 预训练循环和损失计算（模块 02）
- PyTorch Dataset 和 DataLoader

## 文件说明

| 文件 | 用途 |
|------|------|
| `instruction_tuning.py` | 核心实现：InstructionDataset、format_instruction、compute_instruction_loss |
| `exercise.py` | 填空练习，用于巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 什么是指令微调

### 从文本补全到任务完成

预训练大语言模型的训练目标是预测下一个 token。给定"法国的首都是"，它会产生"巴黎"--不是因为它理解问题，而是因为那是训练数据中最可能的延续。

指令微调改变了这一目标。模型不再是补全任意文本，而是学习响应结构化的指令：

```
指令：将以下句子翻译成法语。
输入：Hello, how are you?
输出：Bonjour, comment allez-vous ?
```

经过指令微调后，模型学会了一个通用模式：给定指令和可选输入，产生适当的输出。这在推理时可以泛化到未见过的指令。

### 为什么不只用提示工程？

提示工程可以说服预训练模型遵循指令，但这很脆弱。措辞的微小变化可能产生截然不同的输出。指令微调使模型更加稳健：它可靠地遵循各种措辞的指令，因为它已被训练这样做。

### Alpaca 格式

斯坦福 Alpaca 项目（2023）推广了一种简单的三部分模板：

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

当没有输入上下文时，省略 `### Input:` 部分。这种格式有三个优点：

1. **清晰的分隔符** -- `###` 标记使程序化查找每个部分变得容易
2. **人类可读** -- 你可以直接检查训练样本
3. **广泛采用** -- 许多开源数据集和模型使用这种精确格式

---

## 标签遮罩：关键技术

### 问题

如果我们在完整文本（指令 + 回复）上训练，模型会浪费容量学习预测指令 token。指令在推理时是给定的 -- 我们不需要模型生成它。我们只希望模型学会产生回复。

### 解决方案

我们创建两个 token 序列：

1. **完整序列**：指令 + 回复（用作 `input_ids`）
2. **仅提示**：不含回复的指令（用于计算遮罩长度）

然后设置 `labels[:prompt_length] = -100`。PyTorch 的 `cross_entropy` 设置 `ignore_index=-100` 会完全跳过这些位置。模型接收完整序列作为输入（因此通过注意力机制看到指令），但仅在回复 token 上计算损失。

```
input_ids:  [tok_1, tok_2, ..., tok_p, tok_p+1, ..., tok_n]
labels:     [-100,  -100,  ..., -100,  tok_p+1, ..., tok_n]
              ^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
              遮罩（指令）              训练（回复）
```

### 为什么这很重要

没有标签遮罩，指令微调会退化为继续预训练。模型将大部分梯度信号用于预测它已经知道的指令 token。有了标签遮罩，100% 的训练信号用于学习回复行为 -- 这就是指令微调数据效率高的原因。

---

## 架构

### InstructionDataset

一个 PyTorch Dataset，功能如下：

1. 存储示例列表，每个示例包含 `instruction`、可选的 `input` 和 `output`
2. 将每个示例格式化为 Alpaca 模板
3. 对完整文本和仅提示文本进行分词
4. 创建用于仅回复训练的遮罩标签

```python
class InstructionDataset(Dataset):
    def __init__(self, examples, tokenizer=None, max_length=512, template="alpaca"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
```

`tokenizer` 参数是可选的。如果为 `None`，使用简单的字符级回退（无需真实分词器即可测试）。

### format_instruction

一个独立函数，将指令示例格式化为提示字符串。支持两种模板：

- **alpaca**：使用 `### Instruction:`、`### Input:`、`### Response:`，以双换行分隔
- **simple**：使用 `Instruction:`、`Input:`、`Output:`，以单换行分隔

```python
def format_instruction(instruction, input_text="", output="", template="alpaca"):
    if template == "alpaca":
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt + output
```

### compute_instruction_loss

计算交叉熵损失，设置 `ignore_index=-100`，使遮罩位置（指令）不参与梯度计算：

```python
def compute_instruction_loss(model, batch):
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100,
    )
    return loss
```

---

## 代码详解

### 步骤 1：格式化示例

```python
full_text = self._format_alpaca(example)
# "### Instruction:\nSummarize\n\n### Response:\nThe summary."
```

完整文本包含指令和回复。这是模型看到的输入。

### 步骤 2：格式化仅提示

```python
prompt_text = self._format_prompt_only(example)
# "### Instruction:\nSummarize\n\n### Response:\n"
```

提示文本在回复标记处结束。其长度告诉我们回复从哪里开始。

### 步骤 3：分词

```python
full_ids = self.tokenizer.encode(full_text)[:self.max_length]
prompt_ids = self.tokenizer.encode(prompt_text)[:self.max_length]
```

两者都被截断到 `max_length`。提示 ID 始终是完整 ID 的前缀，因为提示是完整文本的前缀。

### 步骤 4：创建遮罩标签

```python
input_ids = torch.tensor(full_ids, dtype=torch.long)
labels = input_ids.clone()
labels[:len(prompt_ids)] = -100
```

标签是输入 ID 的副本，但指令部分被替换为 -100。模型看到完整序列，但只从回复中学习。

### 步骤 5：返回批次项

```python
return {
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
}
```

注意力掩码全是 1（每个 token 都被关注）。在实践中，你可能会使用填充，并将填充位置的掩码设为 0。

---

## 训练技巧

### 学习率

指令微调使用比预训练更低的学习率。典型值：

- 全量微调：1e-5 到 5e-5
- LoRA：1e-4 到 3e-4（更高，因为更新的参数更少）

学习率过高会导致灾难性遗忘 -- 模型失去其预训练知识。

### 数据质量优于数量

Alpaca 项目表明，52,000 个高质量示例可以产生强大的指令遵循模型。更多数据有帮助，但质量更重要。优先考虑清晰、多样化的指令和准确的回复。

### 训练轮次

通常 1-3 个 epoch 就足够了。更多轮次有过拟合的风险，尤其是在小数据集上。监控验证损失，如果趋于平稳则提前停止。

### 梯度累积

对于大模型，你可能需要梯度累积来实现超出 GPU 内存容量的有效批量大小：

```python
for i, batch in enumerate(dataloader):
    loss = compute_instruction_loss(model, batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### 混合精度

使用 `torch.amp` 进行混合精度训练，以减少内存使用并加速计算：

```python
with torch.amp.autocast("cuda"):
    loss = compute_instruction_loss(model, batch)
```

---

## 真实世界的指令微调数据集

| 数据集 | 规模 | 来源 |
|--------|------|------|
| Stanford Alpaca | 52K | GPT-4 生成 |
| ShareGPT | ~90K | 用户分享的 ChatGPT 对话 |
| OpenAssistant | ~160K | 人工编写的对话 |
| Dolly 2.0 | 15K | Databricks 员工 |
| LIMA | 1K | 精心策划（研究性质） |

LIMA 论文（"Less Is More for Alignment"，2023）表明，仅 1,000 个精心策划的示例就可以产生有竞争力的模型。数据质量胜过数据数量。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 03-sft/01_instruction_tuning/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分。然后验证：

```bash
pytest 03-sft/01_instruction_tuning/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习自己实现指令微调。

### 练习顺序

1. **`format_instruction_exercise`** -- 将指令示例格式化为 Alpaca 模板
2. **`InstructionDatasetExercise.__getitem__`** -- 分词、创建遮罩标签、返回批次项
3. **`compute_instruction_loss_exercise`** -- 计算带标签遮罩的交叉熵损失

### 提示

- 从 `format_instruction` 开始。这是一个字符串格式化练习 -- 不涉及张量。
- 对于 `__getitem__`，关键洞察是：分别对完整文本和仅提示文本进行分词，然后遮罩提示部分的标签。
- 对于 `compute_instruction_loss`，记住 `ignore_index=-100` 是跳过指令 token 损失的魔法。

---

## 关键要点

1. **指令微调教会模型遵循指令。** 通过在（指令，回复）对上进行微调，模型学会将指令视为上下文并生成有帮助的回复。

2. **标签遮罩是核心技术。** 设置 `labels[:prompt_length] = -100` 确保模型仅从回复中学习，而非指令。这使训练高效且专注。

3. **Alpaca 格式简单有效。** 三个部分（指令、输入、回复）配以清晰的分隔符。输入部分是可选的。

4. **数据质量比数量更重要。** 1K-52K 个高质量示例可以产生强大的指令遵循模型。重点关注多样化、准确的示例。

5. **损失函数是带遮罩的标准交叉熵。** `F.cross_entropy(logits, labels, ignore_index=-100)` 完成所有工作。遮罩发生在标签中，而非损失函数中。

---

## 延伸阅读

- [Stanford Alpaca (Taori et al., 2023)](https://github.com/tatsu-lab/stanford_alpaca) -- 数据集和训练方案
- [LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206) -- 1,000 个示例可能就够了
- [Self-Instruct (Wang et al., 2023)](https://arxiv.org/abs/2212.10560) -- 从模型本身生成指令数据
- [Scaling Data-Constrained Language Models (Muennighoff et al., 2023)](https://arxiv.org/abs/2305.16264) -- 多少数据才够
- [The Flan Collection (Longpre et al., 2023)](https://arxiv.org/abs/2301.13688) -- 大规模指令微调数据集
