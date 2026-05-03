# 数据管道

> **模块 02 -- 预训练，第 01 章**

语言模型在能够学习之前，需要一个结构良好的训练数据流。数据管道就是将原始 token ID 转换为可供 GPU 使用的批量张量的"管道系统"。设计不当的管道会在填充上浪费算力、错误地打乱数据顺序，或产生形状不匹配的批次。本章从零开始构建数据管道。

---

## 前置知识

- 基础 PyTorch（`Dataset`、`DataLoader`、`torch.tensor`）
- 理解分词（模块 01，第 01 章）

## 文件说明

| 文件 | 用途 |
|------|------|
| `data_pipeline.py` | 核心实现：数据集、整理、打包 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | 正确性 pytest 测试 |
| `test_exercise.py` | 练习版本的 pytest 测试 |

---

## 为什么数据管道很重要

预训练 LLM 意味着处理数十亿个 token。即使是一个小小的低效——比如当平均长度只有 300 时，将每个序列填充到固定的 2048 个 token——在数十亿步中都会被放大，浪费大量算力。数据管道必须：

1. **将 token 分割成固定长度的块**，使每个样本具有相同的形状。
2. **创建输入/目标对**，带有一个 token 的偏移（模型学习预测下一个 token）。
3. **高效地组成批次**，不在填充上浪费 GPU 周期。
4. **打乱数据**，使模型不会记忆训练语料的顺序。

---

## 数据集设计：连续序列

最简单的方法是 `PretrainDataset`。它接受一个平坦的 token ID 列表，将其分割成长度为 `seq_len` 的不重叠块。

```
Tokens: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, ...]
                         seq_len = 4

块 0: input = [0, 1, 2, 3],  label = [1, 2, 3, 4]
块 1: input = [4, 5, 6, 7],  label = [5, 6, 7, 8]
块 2: input = [8, 9, 10, 11], label = [9, 10, 11, 12]
```

关键细节：

- **截断**：我们丢弃不能填满完整块的尾部 token。这保证所有样本长度相同。
- **输入/目标偏移**：`labels` 是 `input_ids` 向右偏移一个位置。这是标准的自回归语言建模目标——给定位置 `t` 之前的 token，预测位置 `t+1` 的 token。
- **无重叠**：块是连续且不重叠的。这是最简单的策略。重叠（滑动窗口）也是可行的，但会使内存使用量翻倍。

```python
class PretrainDataset(Dataset):
    def __init__(self, token_ids, seq_len):
        self.seq_len = seq_len
        n_tokens = (len(token_ids) // seq_len) * seq_len
        self.data = torch.tensor(token_ids[:n_tokens], dtype=torch.long)

    def __len__(self):
        return len(self.data) // self.seq_len

    def __getitem__(self, idx):
        start = idx * self.seq_len
        chunk = self.data[start : start + self.seq_len + 1]
        return {
            "input_ids": chunk[:-1],
            "labels": chunk[1:],
        }
```

---

## 动态填充

当批次中的序列长度不同时，必须将它们填充到相同长度，才能堆叠成单个张量。有两种策略：

### 固定填充（浪费）

将每个序列填充到全局最大值（例如 2048）。如果大多数序列为 300 个 token，你会浪费 85% 的算力在填充上。

### 动态填充（高效）

将每个批次填充到该批次中最长序列的长度。如果一个批次有序列长度 [300, 280, 310, 295]，你填充到 310 而不是 2048。

```python
def dynamic_pad_collate(batch, pad_id=0):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    # ... 将每个序列填充到 max_len ...
```

整理函数还产生：

- **`attention_mask`**：二值张量（真实 token 为 1，填充为 0）。模型使用它来避免关注填充位置。
- **标签为 -100**：PyTorch 的 `CrossEntropyLoss` 会忽略标签为 -100 的位置。我们将填充位置设为 -100，使损失仅在真实 token 上计算。

---

## 打包：完全消除填充浪费

动态填充有所帮助，但即使在一个批次内，较短的序列仍然浪费一些空间。**打包**更进一步，完全消除填充。

思路：将所有文档连接成一个长 token 流，然后分割成固定长度的块。

```
文档 A: [a1, a2, a3, a4, a5]
文档 B: [b1, b2, b3]

打包流: [a1, a2, a3, a4, a5, SEP, b1, b2, b3, ...]

块 0 (seq_len=4): input = [a1, a2, a3, a4], label = [a2, a3, a4, a5]
块 1 (seq_len=4): input = [a5, SEP, b1, b2], label = [SEP, b1, b2, b3]
```

### 文档边界

一个微妙的问题：如果文档 A 在位置 3 结束，文档 B 在同一块的位置 4 开始，注意力机制将跨越边界进行关注。模型在预测文档 A 的最后一个 token 时会看到文档 B 的 token。

解决方案：

1. **分隔符 token**：在文档之间插入特殊 token。模型学习到在分隔符之后，上下文重置。
2. **注意力掩码修改**：使用块对角注意力掩码来防止跨文档注意力。这更复杂但更原则性。
3. **接受泄漏**：实际上，在大规模预训练中，边界处的少量跨文档注意力影响最小。

我们的实现使用方法 1（分隔符 token），以保持简单。

---

## DataLoader 配置

`create_pretrain_dataloader` 函数将所有内容包装在一起：

```python
DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,         # 每个 epoch 随机化样本顺序
    collate_fn=...,       # 动态填充（打包时为 None）
    drop_last=True,       # 丢弃最后不完整的批次
)
```

### `shuffle=True`

预训练数据是平坦的 token 流。打乱顺序防止模型学习文档顺序中的虚假模式。每个 epoch 以不同的顺序查看相同的数据。

### `drop_last=True`

最后一个批次可能小于 `batch_size`。丢弃它可以：
- 避免在无法饱和 GPU 的小批次上浪费算力。
- 避免与批归一化相关的潜在形状问题（transformer 中不使用，但是一个好习惯）。

### `num_workers`

对于生产训练，设置 `num_workers > 0` 以在并行进程中加载数据。这在 CPU 准备下一个批次时保持 GPU 繁忙。对于学习和调试，`num_workers=0`（默认值）就可以了。

---

## 代码演练

### 第 1 步：PretrainDataset

```python
dataset = PretrainDataset(token_ids, seq_len=512)
```

接受一个平坦的 token ID 列表（例如，整个训练语料的分词结果），将其分割成 512 个 token 的块。每个 `__getitem__` 调用返回一个包含 `input_ids`（512 个 token）和 `labels`（相同的 512 个 token 偏移一个位置）的字典。

### 第 2 步：dynamic_pad_collate

```python
collate_fn = lambda b: dynamic_pad_collate(b, pad_id=0)
```

当 DataLoader 收集一个批次的样本时，此函数将它们填充到相同长度。它还创建注意力掩码，并将标签填充设为 -100。

### 第 3 步：PackedDataset

```python
dataset = PackedDataset([doc1_tokens, doc2_tokens, ...], seq_len=512)
```

接受多个文档（每个是 token ID 列表），用分隔符连接它们，并分割成块。不需要填充——每个块正好是 `seq_len` 个 token。

### 第 4 步：create_pretrain_dataloader

```python
loader = create_pretrain_dataloader(
    token_ids,
    seq_len=512,
    batch_size=8,
    use_packing=False,
)
```

一个函数搞定一切。根据 `use_packing` 创建 `PretrainDataset` 或 `PackedDataset`，用适当的整理函数将其包装在 DataLoader 中，并返回一个可迭代的训练加载器。

---

## 练习

打开 `exercise.py` 并实现：

1. **`PretrainDataset.__getitem__`**：提取一个块并创建输入/标签偏移。
2. **`dynamic_pad_collate`**：将批次填充到最大长度，创建注意力掩码，将标签填充设为 -100。
3. **`PackedDataset.__getitem__`**：与 PretrainDataset 相同，但使用打包的块大小。

运行练习测试：

```bash
pytest test_exercise.py -v
```

在实现每个方法时，删除对应的 `@pytest.mark.skip` 装饰器。

---

## 运行测试

```bash
# 测试主实现
pytest tests.py -v

# 测试练习版本
pytest test_exercise.py -v
```

---

## 总结

| 概念 | 解决的问题 |
|------|-----------|
| `PretrainDataset` | 将平坦的 token 分割成固定长度的输入/标签对 |
| `dynamic_pad_collate` | 将批次填充到批次中的最大长度，而不是全局最大值 |
| `PackedDataset` | 通过连接文档完全消除填充 |
| `attention_mask` | 告诉模型哪些位置是真实 token，哪些是填充 |
| `labels = -100` | 告诉损失函数忽略填充位置 |
| `drop_last=True` | 避免在小的最终批次上浪费算力 |

---

## 下一步

- **第 02 章（数据工程）**：原始文本的来源、清洗、去重和质量过滤。
- **第 03 章（训练循环）**：如何使用此数据管道实际训练模型。
