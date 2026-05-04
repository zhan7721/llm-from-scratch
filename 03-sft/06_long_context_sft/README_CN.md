# 长上下文有监督微调 (Long Context SFT)

> **模块 03 -- 有监督微调，第 06 章**

大多数语言模型在固定的上下文窗口（如 2048 或 4096 个 token）中进行预训练。在推理时，它们无法关注超出该窗口的信息。长上下文有监督微调通过在更长的序列上进行微调来扩展有效上下文长度，使模型能够处理超过原始限制的文档、对话和代码库。

本章实现核心技术：位置插值、NTK 感知缩放、使用滑动窗口的长上下文数据处理，以及分块损失计算。

---

## 前置知识

- Transformer 语言模型基础（模块 01）
- 预训练循环和损失计算（模块 02）
- 注意力机制和位置编码
- PyTorch Dataset 和 DataLoader

## 文件说明

| 文件 | 用途 |
|------|------|
| `long_context_sft.py` | 核心实现：PositionInterpolation、NTKAwareScaling、LongContextDataset、compute_long_context_loss、LongContextTrainer |
| `exercise.py` | 填空练习，用于巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 问题：固定上下文窗口

在长度 L 的序列上训练的 Transformer 模型无法在推理时泛化到长度超过 L 的序列。位置编码——无论是学习的、正弦的还是旋转的（RoPE）——只在位置 0 到 L-1 之间被见过。当模型遇到位置 L、L+1 等时，嵌入是分布外的，模型会产生不可靠的输出。

这是一个根本性限制。在 2048 个 token 序列上训练的模型无法读取 10,000 个 token 的文档。上下文窗口是一个硬边界。

### 为什么不直接在更长的序列上训练？

可以，但代价很高：

1. **二次注意力成本**：自注意力的计算复杂度是 O(n^2)。将上下文长度加倍，计算量增加四倍。
2. **内存**：长序列需要相应更多的 GPU 内存用于激活值、KV 缓存和梯度。
3. **数据**：你需要足够的长文档来填充训练批次。

长上下文 SFT 提供了一个折中方案：取一个已在较短长度上预训练的模型，通过相对较少的额外训练将其扩展到处理更长的序列。

---

## 位置插值 (PI)

### 核心思想

位置插值（Chen 等人，2023）是最简单的方法。如果模型在最大位置 L_orig 上预训练，你想将其扩展到 L_target，则将所有位置索引线性缩放 L_orig / L_target 倍。

```
原始位置：  0, 1, 2, ..., 2047         （范围 [0, 2048)）
插值后：    0, 0.25, 0.5, ..., 2047.75  （范围 [0, 2048)）
```

模型看到的位置范围与其训练时相同，但位置被压缩了。原来在位置 4096 的信息现在映射到位置 1024——模型可以关注它，因为它在原始范围内。

### 为什么有效

关键洞察是模型的位置嵌入是连续的。例如，RoPE 根据位置索引应用旋转矩阵。如果你给它位置 0.5，它会给出位置 0 和位置 1 之间一半的旋转——这是模型在预训练期间见过的。插值位置始终在训练分布内。

### 权衡

- **简单**：只需将位置除以一个缩放因子。
- **有效**：可以通过最少的微调将上下文扩展 2-4 倍。
- **有损**：压缩位置意味着模型失去了一些区分相邻 token 的能力。原来相距 1 的两个位置现在相距 0.25。

---

## NTK 感知缩放

### 核心思想

NTK 感知缩放（bloc97，2023）采用了不同的方法。它不缩放位置，而是缩放 RoPE 的基频率。

RoPE 使用一组不同频率的正弦函数。低频捕获局部关系（相邻 token），高频捕获全局结构（远距离 token）。位置插值等比例压缩所有频率，这会损害高频分量。

NTK 感知缩放调整基频率，使得低频分量被缩放（扩展上下文窗口），而高频分量被保留（保持局部区分能力）。

### 数学原理

标准 RoPE 计算逆频率：

```
inv_freq[i] = 1.0 / (base^(2i/d))
```

NTK 感知缩放替换基：

```
scaled_base = base * scale_factor
inv_freq[i] = 1.0 / (scaled_base^(2i/d))
```

更高的基意味着低频被更多压缩（扩展上下文），而高频受影响较小（保留局部结构）。

### YaRN：结合 NTK 与注意力温度

YaRN（Yet another RoPE extensioN）通过调整注意力温度来扩展 NTK 感知缩放。注意力中的 softmax 除以 sqrt(d)，YaRN 根据序列长度用温度因子缩放它。这有助于模型在更长序列上校准其注意力分布。

---

## 长上下文数据处理

### 文档打包

短文档可以打包在一起来填充序列：

```
[doc1_tokens] [SEP] [doc2_tokens] [SEP] [padding]
```

这通过避免浪费的填充来最大化 GPU 利用率。

### 滑动窗口

超过上下文窗口的长文档使用滑动窗口处理：

```
文档：    [t0, t1, t2, ..., t1000]
窗口 1：  [t0, t1, ..., t511]
窗口 2：  [t256, t257, ..., t767]
窗口 3：  [t512, t513, ..., t1000]
```

步长控制窗口之间的重叠。更多重叠意味着模型在更多上下文中看到每个 token，但需要更多训练步骤。

### 分块损失计算

对于不适合单次前向传播的超长序列，可以分块计算损失：

1. 将序列分成大小为 `chunk_size` 的块。
2. 独立计算每个块的损失。
3. 对所有块的损失取平均值。

这是一种近似——模型无法跨块边界进行注意力计算——但它允许在远超内存容量的序列上进行训练。

---

## 架构

### PositionInterpolation

```python
class PositionInterpolation:
    def __init__(self, original_max_seq_len, target_max_seq_len):
        self.scale_factor = target_max_seq_len / original_max_seq_len

    def rescale_positions(self, positions):
        return positions / self.scale_factor
```

### NTKAwareScaling

```python
class NTKAwareScaling:
    def __init__(self, original_max_seq_len, target_max_seq_len, base=10000.0):
        self.scale_factor = target_max_seq_len / original_max_seq_len
        self.scaled_base = base * (self.scale_factor ** (2 * pi / (2 * pi)))

    def get_scaled_inv_freq(self, d_model, device):
        inv_freq = 1.0 / (self.scaled_base ** (arange(0, d_model, 2) / d_model))
        return inv_freq
```

### LongContextDataset

```python
class LongContextDataset(Dataset):
    def __init__(self, documents, seq_len, separator_id=0, stride=None):
        self.chunks = []
        for doc in documents:
            if len(doc) <= seq_len:
                # 填充短文档
                self.chunks.append(padded_doc[:seq_len + 1])
            else:
                # 长文档使用滑动窗口
                for start in range(0, len(doc) - seq_len, stride):
                    self.chunks.append(doc[start:start + seq_len + 1])

    def __getitem__(self, idx):
        chunk = self.chunks[idx]
        return {"input_ids": chunk[:-1], "labels": chunk[1:]}
```

### LongContextTrainer

训练器将所有功能组合在一起：位置缩放、梯度累积和梯度裁剪。

```python
class LongContextTrainer:
    def __init__(self, model, optimizer, original_max_seq_len=2048,
                 target_max_seq_len=8192, scaling_method="pi",
                 gradient_accumulation_steps=1, max_grad_norm=1.0):
        ...
```

---

## 代码详解

### 步骤 1：位置插值

```python
pi = PositionInterpolation(original_max_seq_len=2048, target_max_seq_len=8192)
positions = torch.arange(8192)
rescaled = pi.rescale_positions(positions)
# rescaled[8191] = 8191 / 4 = 2047.75  （在原始范围内）
```

### 步骤 2：NTK 感知缩放

```python
ntk = NTKAwareScaling(original_max_seq_len=2048, target_max_seq_len=8192)
inv_freq = ntk.get_scaled_inv_freq(d_model=64, device=torch.device("cpu"))
# inv_freq 使用 scaled_base 而非 10000.0
```

### 步骤 3：数据集构建

```python
documents = [list(range(100)), list(range(200))]
dataset = LongContextDataset(documents, seq_len=50)
# 短文档被填充；长文档通过滑动窗口分割
item = dataset[0]
# item["input_ids"].shape = (50,)
# item["labels"].shape = (50,)  （偏移 1 位）
```

### 步骤 4：分块损失

```python
model = DummyModel()
batch = {"input_ids": torch.randint(0, 100, (2, 100)),
         "labels": torch.randint(0, 100, (2, 100))}
loss = compute_long_context_loss(model, batch, chunk_size=50)
# 将 100 个 token 的序列分成两个 50 的块处理
```

### 步骤 5：训练循环

```python
trainer = LongContextTrainer(model, optimizer,
                             original_max_seq_len=64,
                             target_max_seq_len=128)
batch = {"input_ids": ..., "labels": ...}
result = trainer.train_step(batch, step=0)
# 返回 {"loss": <标量>}
```

---

## 训练注意事项

### 梯度累积

长序列每批次消耗更多内存。梯度累积允许你模拟更大的有效批量大小：

```python
trainer = LongContextTrainer(
    model, optimizer,
    gradient_accumulation_steps=4,  # 有效批量 = 4 * 微批量
)
```

训练器将损失除以累积步数，累积梯度，仅每 N 步更新一次权重。

### 梯度裁剪

长序列可能产生大梯度，特别是在训练初期模型调整到新上下文长度时。训练器将梯度裁剪到 `max_grad_norm`（默认 1.0）以稳定训练。

### 学习率调度

长上下文微调通常使用比预训练更低的学习率：

- 以较短的预热开始（100-500 步）
- 使用余弦衰减到最终学习率
- 典型范围是 1e-5 到 5e-5

### 数据混合

不要专门在长文档上训练。混合较短的文档以防止短上下文能力的灾难性遗忘。常见比例是 50% 长文档，50% 短文档。

### 渐进式扩展

与其一步从 2K 跳到 32K，不如逐步扩展：

```
2K -> 4K -> 8K -> 16K -> 32K
```

每个阶段使用较小的缩放因子，使模型更容易适应。

---

## 方法比较

| 方法 | 改变什么 | 优点 | 缺点 |
|------|----------|------|------|
| 位置插值 | 线性缩放位置 | 简单，2-4 倍扩展有效 | 等比例压缩所有频率 |
| NTK 感知缩放 | 缩放 RoPE 基频率 | 保留高频分量 | 更复杂，需要调参 |
| YaRN | NTK + 注意力温度 | 最佳质量 | 实现最复杂 |
| 直接训练 | 从头在长长度上训练 | 无近似 | 昂贵，需要长数据 |

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 03-sft/06_long_context_sft/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分。然后验证：

```bash
pytest 03-sft/06_long_context_sft/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习自己实现长上下文 SFT。

### 练习顺序

1. **`PositionInterpolationExercise.rescale_positions`** -- 将位置除以缩放因子
2. **`NTKAwareScalingExercise.get_scaled_inv_freq`** -- 计算缩放基和逆频率
3. **`LongContextDatasetExercise.__getitem__`** -- 从块返回偏移的 input_ids 和 labels
4. **`compute_long_context_loss_exercise`** -- 对长序列分块计算损失

### 提示

- 从位置插值开始。这是一个简单的除法运算。
- 对于 NTK 感知缩放，关键是计算 `scaled_base = base * scale_factor`。
- 对于 `__getitem__`，记住语言建模使用偏移标签：input_ids = chunk[:-1]，labels = chunk[1:]。
- 对于分块损失，以 chunk_size 为步长循环遍历序列，对每个块的损失取平均值。

---

## 关键要点

1. **位置插值是最简单的扩展方法。** 将所有位置索引除以缩放因子，将其映射回原始训练范围。

2. **NTK 感知缩放保留高频分量。** 不缩放位置，而是缩放 RoPE 基频率。这在扩展上下文窗口的同时保持局部 token 区分能力。

3. **长文档需要特殊处理。** 使用带重叠的滑动窗口处理超过上下文窗口的文档。将短文档打包在一起以高效填充批次。

4. **分块损失计算避免 OOM。** 对于不适合内存的序列，分块计算损失并取平均值。这是一种近似，但可以在超长序列上进行训练。

5. **梯度累积至关重要。** 长序列意味着每批次更少的样本。梯度累积在不增加内存的情况下模拟更大的有效批量大小。

6. **逐步扩展。** 一步从 2K 到 32K 很困难。渐进式扩展（2K -> 4K -> 8K -> ...）更稳定，需要更少的微调数据。

---

## 延伸阅读

- [Extending Context Window of Large Language Models via Position Interpolation (Chen 等人, 2023)](https://arxiv.org/abs/2306.15595) -- 位置插值论文
- [Reddit/bloc97 NTK-aware scaling](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/) -- NTK 感知 RoPE 扩展
- [YaRN: Efficient Context Window Extension of Large Language Models (Peng 等人, 2023)](https://arxiv.org/abs/2309.00071) -- 结合 NTK + 温度缩放
- [LongLoRA: Efficient Fine-tuning of Long-Context Large Language Models (Chen 等人, 2023)](https://arxiv.org/abs/2309.12307) -- 长上下文 LoRA 微调
- [Scaling Laws for RoPE (Liu 等人, 2023)](https://arxiv.org/abs/2309.16739) -- 位置编码缩放分析
