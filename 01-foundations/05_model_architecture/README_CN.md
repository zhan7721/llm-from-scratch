# 完整的 GPT 模型架构

> **模块 01 -- 基础篇，第 05 章**

我们已经构建了各个组件：嵌入（第 02 章）、注意力（第 03 章）和 Transformer 块（第 04 章）。现在我们将它们组装成一个完整的语言模型。本章实现一个 GPT 风格的模型，采用 LLaMA 架构设计：Pre-Norm、RMSNorm、SwiGLU、RoPE 和权重绑定。

---

## 前置知识

- Token 嵌入和 RoPE（第 02 章）
- 多头注意力（第 03 章）
- Transformer 块：RMSNorm、SwiGLU、Pre-Norm（第 04 章）
- PyTorch 基础：`nn.Module`、`nn.Linear`、`dataclasses`

## 文件说明

| 文件 | 用途 |
|------|------|
| `model.py` | 完整的 GPT 模型实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

GPT 模型在架构层面非常简单。它接受一个 token ID 序列，在每个位置产生下一个 token 的概率分布：

```
输入：token IDs (batch_size, seq_len)
    |
    v
TokenEmbedding          -- 将 ID 转换为稠密向量
    |
    v
RotaryPositionalEmbedding  -- 添加位置信息
    |
    v
TransformerBlock x N    -- 深层处理（注意力 + FFN）
    |
    v
RMSNorm                 -- 最终归一化
    |
    v
Linear lm_head          -- 投影到词汇表
    |
    v
输出：logits (batch_size, seq_len, vocab_size)
```

模型被训练来预测下一个 token。给定 tokens `[t1, t2, t3, t4]`，它在每个位置产生 logits，损失函数针对移位的序列 `[t2, t3, t4, t5]` 计算。

---

## GPTConfig：模型超参数

```python
@dataclass
class GPTConfig:
    vocab_size: int = 32000     # 词汇表大小
    d_model: int = 512          # 隐藏维度
    n_heads: int = 8            # 注意力头数
    n_layers: int = 6           # Transformer 块数
    d_ff: int = None            # FFN 隐藏维度（默认：SwiGLU 公式）
    max_seq_len: int = 1024     # RoPE 最大序列长度
    dropout: float = 0.0        # dropout（未使用，保留兼容性）
```

### 参数规模

模型大小主要由 `d_model` 和 `n_layers` 决定。以下是一些真实配置：

| 模型 | d_model | n_heads | n_layers | d_ff | 参数量 |
|------|---------|---------|----------|------|--------|
| GPT-2 Small | 768 | 12 | 12 | 3072 | 117M |
| GPT-2 Medium | 1024 | 16 | 24 | 4096 | 345M |
| LLaMA 7B | 4096 | 32 | 32 | 11008 | 6.7B |
| LLaMA 13B | 5120 | 40 | 40 | 13824 | 13B |
| LLaMA 65B | 8192 | 64 | 80 | 22016 | 65B |

注意 `n_heads` 必须整除 `d_model`（每个头的维度为 `d_k = d_model / n_heads`）。SwiGLU 的默认 `d_ff` 为 `(2/3) * 4 * d_model`，取整到 256 的倍数。

---

## 模型架构

### 第 1 层：Token 嵌入

```python
self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
```

将整数 token ID 转换为 `d_model` 维的稠密向量。输出乘以 `sqrt(d_model)` 进行缩放，使幅度不依赖于嵌入维度。

### 第 2 层：旋转位置嵌入（RoPE）

```python
self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)
```

通过旋转嵌入向量来编码位置信息。与绝对位置编码不同，RoPE 通过位置间的旋转角度差自然地捕获相对位置。

### 第 3 层：N 个 Transformer 块

```python
self.layers = nn.ModuleList([
    TransformerBlock(config.d_model, config.n_heads, config.d_ff)
    for _ in range(config.n_layers)
])
```

每个块应用：
1. **Pre-Norm 注意力**：`x = x + Attention(RMSNorm(x))`
2. **Pre-Norm FFN**：`x = x + SwiGLU(RMSNorm(x))`

这是大部分计算发生的地方。每个块通过跨位置混合信息（注意力）和独立变换每个位置（FFN）来精炼 token 表示。

### 第 4 层：最终 RMSNorm

```python
self.norm = RMSNorm(config.d_model)
```

LLaMA 在最后一个 Transformer 块之后应用最终的 RMSNorm。这确保隐藏状态在输出投影之前具有适当的幅度。

### 第 5 层：语言模型头

```python
self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
```

将隐藏状态从 `d_model` 投影到 `vocab_size`，为词汇表中的每个 token 产生 logits。在生成过程中，这些 logits 通过 softmax 转换为概率。

---

## 权重绑定

最重要的设计选择之一是**权重绑定**：

```python
self.lm_head.weight = self.token_emb.embedding.weight
```

这使得输出投影矩阵与输入嵌入矩阵共享相同的参数。我们不需要两个独立的 `vocab_size x d_model` 矩阵，只需要一个。

### 为什么权重绑定有效

1. **减少参数**：对于词汇量 32000 和 d_model 4096，这节省了 1.31 亿参数。

2. **语义一致性**：语义相似的 token 应该有相似的嵌入并且产生相似的 logits。权重绑定强制执行这个约束。

3. **经验证据**：Press & Wolf (2017) 表明权重绑定在语言建模基准上提高了困惑度，特别是对于较小的模型。

4. **直觉**：嵌入矩阵将 token 映射到潜在空间。输出投影将从该潜在空间映射回 token。让它们互为逆运算是合理的。

### 在 PyTorch 中如何工作

```python
# 绑定前：两个独立的权重矩阵
assert model.lm_head.weight is not model.token_emb.embedding.weight

# 绑定后：同一个张量对象
model.lm_head.weight = model.token_emb.embedding.weight
assert model.lm_head.weight is model.token_emb.embedding.weight
```

当你调用 `model.parameters()` 时，PyTorch 只返回唯一的参数，所以绑定的权重只计算一次。

---

## 前向传播详解

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    # x: (batch_size, seq_len) -- token IDs

    h = self.token_emb(x)      # (batch_size, seq_len, d_model)
    h = self.rope(h)            # (batch_size, seq_len, d_model)

    for layer in self.layers:
        h = layer(h)            # (batch_size, seq_len, d_model)

    h = self.norm(h)            # (batch_size, seq_len, d_model)
    return self.lm_head(h)      # (batch_size, seq_len, vocab_size)
```

在每个位置，模型产生一个 `vocab_size` 维的 logits 向量。更高的 logit 对应更可能的下一个 token。

### 形状追踪

对于 batch 大小为 2，序列长度为 10，d_model=64，vocab_size=256：

```
输入：     (2, 10)           -- token IDs
嵌入：     (2, 10, 64)       -- 稠密向量
RoPE：     (2, 10, 64)       -- 位置编码向量
Block 1：  (2, 10, 64)       -- 精炼表示
Block 2：  (2, 10, 64)       -- 进一步精炼
Norm：     (2, 10, 64)       -- 归一化
lm_head：  (2, 10, 256)      -- 词汇表中每个 token 的 logits
```

---

## 自回归生成

### 生成循环

给定一个提示，模型逐个生成新的 token：

```python
@torch.no_grad()
def generate(self, prompt, max_new_tokens=100, temperature=1.0):
    for _ in range(max_new_tokens):
        # 裁剪到 max_seq_len（处理长序列）
        idx_cond = prompt[:, -self.config.max_seq_len:]

        # 前向传播
        logits = self(idx_cond)

        # 最后一个位置的 logits，应用温度
        logits = logits[:, -1, :] / temperature

        # 采样下一个 token
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # 拼接
        prompt = torch.cat([prompt, next_token], dim=1)

    return prompt
```

### 温度

温度控制生成的随机性：

- **temperature = 1.0**：从模型分布中标准采样。
- **temperature < 1.0**（如 0.7）：使分布更尖锐（更自信）。模型更可能选择高概率 token。
- **temperature > 1.0**（如 1.5）：使分布更平坦（更随机）。模型探索更多样的 token。
- **temperature -> 0**：趋近贪心解码（总是选择最可能的 token）。

数学上，温度在 softmax 之前除以 logits：

```
P(token_i) = exp(logit_i / T) / sum_j(exp(logit_j / T))
```

当 T < 1 时，logits 之间的差异被放大。当 T > 1 时，差异被缩小。

### 贪心 vs 采样

**贪心解码**总是选择概率最高的 token。它是确定性的，但倾向于产生重复、无聊的文本。

**采样**从概率分布中抽取。它产生更多样化和有趣的文本，但偶尔可能产生无意义的 token。

**Top-k 采样**和**Top-p（核）采样**是更高级的策略，将采样限制在最可能的 token 中。这里没有实现，但它们是直接的扩展。

### `@torch.no_grad()` 装饰器

生成不需要梯度（我们不是在训练）。`@torch.no_grad()` 装饰器告诉 PyTorch 跳过梯度计算，这：
- 减少内存使用（不需要存储中间激活）
- 加速计算
- 是推理的最佳实践

---

## 参数计数

### 估算模型大小

对于具有 `d_model`、`n_heads`、`n_layers`、`d_ff` 和 `vocab_size` 的模型：

**每个 Transformer 块：**
- 注意力（Q, K, V, O）：4 x d_model^2
- SwiGLU（W1, W_gate, W2）：3 x d_model x d_ff
- RMSNorm（x2）：2 x d_model
- 每块总计：~4 x d_model^2 + 3 x d_model x d_ff + 2 x d_model

**全局参数：**
- Token 嵌入：vocab_size x d_model
- 最终 RMSNorm：d_model
- lm_head：vocab_size x d_model（但与嵌入绑定，所以额外为 0）

**总计：** n_layers x（每块）+ vocab_size x d_model + d_model

### 示例：LLaMA 7B

```
d_model = 4096, n_heads = 32, n_layers = 32, d_ff = 11008, vocab_size = 32000

每块：
  注意力：4 x 4096^2 = 67,108,864
  SwiGLU：3 x 4096 x 11008 = 135,266,304
  RMSNorm：2 x 4096 = 8,192
  总计：  202,383,360

所有块：32 x 202,383,360 = 6,476,267,520
嵌入：  32000 x 4096 = 131,072,000
最终 norm：4096

总计：~6,607,343,616（约 67 亿参数）
```

### 内存估算

FP32（每个参数 4 字节）：
- 67 亿参数 x 4 字节 = 26.8 GB

FP16/BF16（每个参数 2 字节）：
- 67 亿参数 x 2 字节 = 13.4 GB

训练期间，你还需要梯度（与参数大小相同）和优化器状态（Adam 需要 2 倍），所以总计大约：
- 训练：~4 倍参数内存（FP32）或 ~8 倍（混合精度）
- 推理：~1 倍参数内存（FP16）

---

## LLaMA 设计选择总结

我们的 GPT 模型使用与 LLaMA 相同的架构。以下是关键设计选择的总结：

| 组件 | 原始 Transformer（GPT-2） | LLaMA / 我们的模型 |
|------|--------------------------|-------------------|
| 归一化 | LayerNorm（Post-Norm） | RMSNorm（Pre-Norm） |
| FFN 激活 | ReLU | SwiGLU |
| FFN 隐藏维度 | 4 x d_model | (2/3) x 4 x d_model |
| 位置编码 | 绝对学习式 | RoPE |
| 线性偏置 | 是 | 否 |
| 权重绑定 | 有时 | 是 |
| 输出归一化 | 无 | RMSNorm |

这些更改中的每一个都是由改进的训练稳定性、更好的质量或降低的计算成本驱动的。

### 为什么选择这些？

1. **RMSNorm 优于 LayerNorm**：计算更便宜，结果同样好。去掉了不必要的均值减法。

2. **SwiGLU 优于 ReLU**：门控机制更具表现力。SiLU 激活允许平滑梯度和非单调行为。

3. **RoPE 优于绝对位置**：自然捕获相对位置。通过修改实现长度泛化（第 08 章介绍）。

4. **Pre-Norm 优于 Post-Norm**：更好的梯度流，更稳定的训练，支持更高的学习率。

5. **线性层无偏置**：减少参数且没有可测量的质量损失。

6. **权重绑定**：减少参数，提高较小模型的质量。

---

## 代码详解

### model.py

```python
@dataclass
class GPTConfig:
    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = None
    max_seq_len: int = 1024
    dropout: float = 0.0
```

`@dataclass` 装饰器自动生成 `__init__`、`__repr__` 和其他方法。这是定义配置对象的简洁方式。

```python
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # 权重绑定
        self.lm_head.weight = self.token_emb.embedding.weight
```

关键实现细节：
- 使用 `nn.ModuleList` 而不是 Python 列表，这样 PyTorch 正确注册子模块。
- 权重绑定通过简单赋值完成。这行之后，`self.lm_head.weight` 和 `self.token_emb.embedding.weight` 指向同一个张量。
- 线性层的 `bias=False` 遵循 LLaMA 惯例。

```python
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.token_emb(x)
        h = self.rope(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        return self.lm_head(h)
```

前向传播是一个清晰的管道。每一步变换张量，同时保持形状 `(batch_size, seq_len, d_model)`，直到最终投影。

```python
    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=100, temperature=1.0):
        for _ in range(max_new_tokens):
            idx_cond = prompt[:, -self.config.max_seq_len:]
            logits = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            prompt = torch.cat([prompt, next_token], dim=1)
        return prompt
```

生成细节：
- `@torch.no_grad()` 禁用梯度跟踪以提高效率。
- `prompt[:, -self.config.max_seq_len:]` 通过从左侧截断来处理超过 `max_seq_len` 的序列。
- `logits[:, -1, :]` 仅取最后一个位置的 logits（下一个 token 预测）。
- `torch.multinomial` 从概率分布中采样。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/05_model_architecture/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 01-foundations/05_model_architecture/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现 GPT 模型。

### 练习顺序

1. **`__init__`**：创建所有层（嵌入、RoPE、Transformer 块、norm、lm_head）
2. **权重绑定**：使 lm_head.weight 指向 token_emb.embedding.weight
3. **`forward`**：连接管道（嵌入 -> RoPE -> 块 -> norm -> lm_head）
4. **`generate`**：实现自回归生成循环

### 提示

- 从 `__init__` 开始。每个组件都是使用前几章类的单行代码。
- 对于权重绑定，关键行是 `self.lm_head.weight = self.token_emb.embedding.weight`。这使它们成为同一个张量对象。
- 对于 `forward`，把它想象成一个管道：每一步接受一个张量并返回相同形状的张量（最后一步投影到 vocab_size）。
- 对于 `generate`，关键部分是获取最后一个位置的 logits：`logits[:, -1, :]`。不要忘记应用温度并使用 `torch.multinomial` 进行采样。

---

## 关键要点

1. **GPT 很简单。** 架构就是：嵌入 -> N 个 Transformer 块 -> norm -> 线性头。复杂性在于组件（注意力、FFN），而不在整体结构。

2. **权重绑定减少参数并提高质量。** 共享嵌入和输出投影矩阵是一个简单但有显著好处的技巧。

3. **温度控制生成多样性。** 温度越低越自信，温度越高越随机。这是控制生成行为最简单的方式。

4. **LLaMA 的设计选择已成为标准。** Pre-Norm、RMSNorm、SwiGLU、RoPE、无偏置 -- 这些已在大规模上得到验证并被大多数现代 LLM 采用。

5. **参数计数是可预测的。** 你可以仅从配置估算模型大小。主要项是注意力投影（每层 4 x d_model^2）和 FFN（每层 3 x d_model x d_ff）。

---

## 延伸阅读

- [LLaMA 论文 (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) -- 我们实现的架构
- [GPT-2 论文 (Radford et al., 2019)](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) -- 原始 GPT 架构
- [权重绑定 (Press & Wolf, 2017)](https://arxiv.org/abs/1608.05859) -- 使用相同的权重作为输入和输出嵌入
- [缩放定律 (Kaplan et al., 2020)](https://arxiv.org/abs/2001.08361) -- 模型大小如何影响性能
- [RoPE 论文 (Su et al., 2021)](https://arxiv.org/abs/2104.09864) -- 旋转位置嵌入
- [PaLM 论文 (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) -- SwiGLU 大规模应用
