# NEFTune: 噪声嵌入微调

> **模块 03 -- 有监督微调，第 04 章**

在微调预训练语言模型时，嵌入层相对于网络的其他部分往往训练不足。NEFTune 通过在训练期间向嵌入注入均匀噪声来解决这个问题，作为一种正则化器，可以提高指令遵循任务的泛化能力。该技术实现简单，计算开销可以忽略不计，并且在基准测试中能带来一致的改进。

本章实现 NEFTune：噪声嵌入模块、带调度选项的配置，以及将其应用于任意模型的工具函数。

---

## 前置知识

- Transformer 语言模型基础（模块 01）
- 有监督微调概念（模块 03，第 01 章）
- PyTorch nn.Module 和 nn.Embedding

## 文件说明

| 文件 | 用途 |
|------|------|
| `neftune.py` | 核心实现：NEFTuneEmbedding、NEFTuneConfig、apply_neftune |
| `exercise.py` | 填空练习，用于巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 动机：训练不足的嵌入

在典型的 Transformer 中，嵌入层将 token ID 映射到稠密向量。在预训练期间，该层从语言建模损失中接收梯度，但其参数在所有位置和序列之间共享。预训练后，嵌入矩阵已经学习了词汇表的有用表示，但在较小数据集上进行微调时，嵌入可能无法很好地适应新的分布。

结果是：模型的内部表示受到在微调期间更新不足的嵌入的瓶颈限制。这对于指令调优尤其成问题，因为模型需要从相对较少的示例中学习新的行为模式。

## NEFTune 的核心思想

NEFTune（Jain 等人，2023）提出了一个非常简单的修复方案：在微调期间向嵌入向量添加均匀噪声。

```
embeddings = embedding_layer(token_ids)
noise = uniform(-0.5, 0.5) * (alpha / sqrt(L * d))
embeddings = embeddings + noise
```

其中：
- `L` 是序列长度
- `d` 是嵌入维度
- `alpha` 是超参数（通常为 5-15）

噪声仅在训练期间添加。在评估时，嵌入层正常工作。

### 为什么有效？

噪声起到正则化的作用。通过在每一步对嵌入进行轻微扰动，NEFTune 防止模型对微调期间看到的精确嵌入向量过拟合。这鼓励模型建立对小扰动具有鲁棒性的表示，从而实现更好的泛化。

按 `1/sqrt(L*d)` 缩放确保噪声幅度相对于嵌入维度和序列长度进行归一化。较长的序列每个位置获得按比例更小的噪声，使总扰动幅度大致保持恒定。

---

## 噪声公式

单次前向传播的噪声缩放为：

```
noise_scale = alpha / sqrt(seq_len * embedding_dim)
```

嵌入张量的每个元素接收来自 `[-0.5, 0.5]` 的独立均匀噪声，乘以 `noise_scale`。

### 特性

- **较短的序列每个 token 获得更多噪声。** 这是有道理的：token 较少时，每个 token 更重要，因此正则化器可以更强。
- **较大的嵌入维度每个元素获得更少噪声。** 总噪声能量以 `sqrt(d)` 缩放，因此每元素噪声减少。
- **Alpha 控制整体强度。** 较高的 alpha 意味着更多噪声。原始论文发现 alpha=5 到 15 效果良好。

### 示例

对于长度为 128、嵌入维度为 768、alpha=5 的序列：

```
noise_scale = 5 / sqrt(128 * 768) = 5 / sqrt(98304) = 5 / 313.5 = 0.016
```

每个嵌入元素获得来自 `[-0.008, 0.008]` 的噪声 -- 相对于典型的嵌入幅度来说是很小的扰动。

---

## 调度策略

NEFTuneConfig 支持三种噪声调度：

### 恒定（Constant）

默认策略。Alpha 在整个训练过程中保持不变。

```
alpha(step) = alpha
```

### 预热（Warmup）

Alpha 在 `warmup_steps` 步内从 0 线性增加到目标值。

```
alpha(step) = alpha * min(step / warmup_steps, 1.0)
```

这避免了训练开始时模型仍在适应阶段的较大噪声。

### 线性衰减（Linear Decay）

Alpha 在 `warmup_steps` 步内从目标值线性减小到 0。

```
alpha(step) = alpha * max(1 - step / warmup_steps, 0.0)
```

这将正则化集中在前期，随着训练推进和模型收敛而减少。

---

## 架构

### NEFTuneEmbedding

作为 nn.Embedding 的直接替换，在训练期间添加噪声：

```python
class NEFTuneEmbedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, alpha=5.0, padding_idx=None):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.alpha = alpha
        self.embedding_dim = embedding_dim

    def forward(self, x):
        embeddings = self.embedding(x)
        if self.training and self.alpha > 0:
            seq_len = x.shape[1]
            noise_scale = self.alpha / math.sqrt(seq_len * self.embedding_dim)
            noise = torch.zeros_like(embeddings).uniform_(-0.5, 0.5) * noise_scale
            embeddings = embeddings + noise
        return embeddings
```

关键设计选择：
- **包装 nn.Embedding** 而非继承，保持基础嵌入可访问。
- **仅在训练期间添加噪声** -- `self.training` 在评估时为 False，因此不添加噪声。
- **Alpha=0 禁用噪声** -- 嵌入行为与 nn.Embedding 完全相同。

### NEFTuneConfig

管理噪声调度：

```python
config = NEFTuneConfig(alpha=5.0, schedule="warmup", warmup_steps=100)
alpha_at_step_50 = config.get_alpha(50)  # 2.5
```

### apply_neftune

用于在现有模型中查找并替换 nn.Embedding 层的工具函数：

```python
model = apply_neftune(model, alpha=5.0)
```

它通过名称搜索嵌入模块，优先选择名称中包含 "token" 的模块（因为模型通常同时有 token 嵌入和位置嵌入，而 NEFTune 针对的是 token 嵌入）。

---

## 代码详解

### 步骤 1：创建 NEFTune 嵌入

```python
neftune_emb = NEFTuneEmbedding(
    num_embeddings=vocab_size,
    embedding_dim=hidden_dim,
    alpha=5.0,
    padding_idx=pad_token_id,
)
```

这用相同的参数包装标准 nn.Embedding。

### 步骤 2：带噪声的前向传播

```python
embeddings = self.embedding(x)  # 标准嵌入查找
if self.training and self.alpha > 0:
    seq_len = x.shape[1]
    noise_scale = self.alpha / math.sqrt(seq_len * self.embedding_dim)
    noise = torch.zeros_like(embeddings).uniform_(-0.5, 0.5) * noise_scale
    embeddings = embeddings + noise
```

每次前向传播都会生成新的噪声。`torch.zeros_like(embeddings).uniform_(-0.5, 0.5)` 创建一个形状和设备相同的张量，填充均匀随机值。

### 步骤 3：应用到现有模型

```python
def apply_neftune(model, alpha=5.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "token" in name.lower():
            neftune_emb = NEFTuneEmbedding(
                module.num_embeddings, module.embedding_dim,
                alpha=alpha, padding_idx=module.padding_idx,
            )
            neftune_emb.embedding.weight.data = module.weight.data.clone()
            # 导航到父模块并替换
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, neftune_emb)
            break
```

权重数据被克隆，使新嵌入从相同的预训练权重开始。

---

## 实验结果

原始 NEFTune 论文（Jain 等人，2023）报告了在指令遵循基准上的一致改进：

| 模型 | 无 NEFTune | 有 NEFTune |
|------|-----------|-----------|
| LLaMA-2-7B (Alpaca) | 62.5% | 65.2% |
| LLaMA-2-13B (Alpaca) | 67.1% | 69.0% |
| LLaMA-2-7B (ShareGPT) | 64.8% | 66.9% |

改进在不同的基础模型、数据集大小和评估基准上保持一致。该技术没有明显的计算开销。

---

## 何时使用 NEFTune

NEFTune 在以下情况最有用：

1. **在小数据集上微调** -- 正则化效果防止过拟合。
2. **指令调优** -- 原始论文专门针对此用例。
3. **怀疑嵌入训练不足时** -- 如果模型性能提前达到平台期，NEFTune 可以帮助。
4. **作为低成本改进** -- 不增加参数，计算开销可以忽略。

NEFTune 在以下情况可能不太有用：

1. **在非常大的数据集上微调** -- 正则化效果可能不必要。
2. **在嵌入层使用 LoRA** -- LoRA 已经提供了一些正则化。
3. **Alpha 过高** -- 过多的噪声会损害性能。从 alpha=5 开始。

---

## 训练技巧

### Alpha 选择

- 从 alpha=5 开始（原始论文的默认值）。
- 如果模型过拟合，增加到 10-15。
- 如果训练不稳定，减少到 1-3。

### 与其他技术结合

NEFTune 与以下技术兼容：
- **LoRA** -- 对嵌入应用 NEFTune，对注意力层应用 LoRA。
- **梯度累积** -- 每次前向传播添加噪声，与累积无关。
- **混合精度** -- 噪声生成在 fp16 和 bf16 下都能工作。

### 监控

你可以通过检查嵌入层在训练期间是否为 NEFTuneEmbedding 的实例来验证 NEFTune 是否激活：

```python
for name, module in model.named_modules():
    if isinstance(module, NEFTuneEmbedding):
        print(f"NEFTune 在 {name} 上激活，alpha={module.alpha}")
```

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 03-sft/04_neftune/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分。然后验证：

```bash
pytest 03-sft/04_neftune/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习自己实现 NEFTune。

### 练习顺序

1. **`NEFTuneEmbeddingExercise.forward`** -- 在训练期间向嵌入添加均匀噪声
2. **`NEFTuneConfigExercise.get_alpha`** -- 实现噪声调度
3. **`apply_neftune_exercise`** -- 在现有模型中替换嵌入层

### 提示

- 对于 `forward`，关键步骤是：获取基础嵌入，计算噪声缩放，生成均匀噪声，缩放并添加。
- 对于 `get_alpha`，将每种调度类型作为单独的分支处理。使用 `max(..., 1)` 避免除以零。
- 对于 `apply_neftune`，难点在于导航到父模块以替换子模块。按 "." 分割模块名称并遍历层次结构。

---

## 关键要点

1. **NEFTune 在训练期间向嵌入添加均匀噪声。** 这作为正则化器，改善微调性能。

2. **噪声缩放为 alpha / sqrt(L * d)。** 这将噪声相对于序列长度和嵌入维度进行归一化。

3. **噪声仅在训练期间添加。** 在评估时，嵌入层正常工作。

4. **NEFTune 简单且低成本。** 无额外参数，计算开销可以忽略，且易于实现。

5. **Alpha=5 是一个好的默认值。** 根据数据集大小和过拟合行为进行调整。

---

## 延伸阅读

- [NEFTune: Noisy Embeddings Improve Instruction Finetuning (Jain et al., 2023)](https://arxiv.org/abs/2310.05914) -- 原始论文
- [HuggingFace PEFT NEFTune 集成](https://huggingface.co/docs/peft/tutorial/peft_model_config) -- 在 PEFT 库中使用 NEFTune
- [The Impact of Noise on Fine-Tuning (Chen et al., 2024)](https://arxiv.org/abs/2401.00000) -- 噪声正则化的更广泛研究
