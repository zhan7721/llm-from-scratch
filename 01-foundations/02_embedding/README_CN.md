# Token Embedding 和旋转位置编码 (RoPE)

> **模块 01 — 基础，第 02 章**

神经网络处理的是数字，而不是 token。分词将文本转换为整数 ID 后，我们需要一种方法将这些 ID 表示为有意义的向量。**Embedding** 就是这座桥梁。在本章中，我们从零开始构建两个基础的 embedding 层：

1. **Token Embedding** — 将 token ID 转换为稠密向量的查找表
2. **旋转位置编码 (RoPE)** — 一种基于旋转的位置编码方法

它们是每个 Transformer 模型的前两层。每个 token 在到达注意力层之前，都会经过这两个层。

---

## 前置要求

- 基本的 Python 和 PyTorch 知识（张量、`nn.Module`）
- 理解分词的输出（整数 ID）
- 可选：线性代数基础（向量、点积、旋转矩阵）

## 文件说明

| 文件 | 用途 |
|------|------|
| `embedding.py` | TokenEmbedding 和 RotaryPositionalEmbedding 的核心实现 |
| `exercise.py` | 填空练习，用于加深理解 |
| `solution.py` | 练习的参考答案 |
| `tests.py` | 正确性测试 |

---

## 为什么需要 Embedding？

分词后，一个句子如 `"The cat sat"` 变成一串整数：

```
[464, 3797, 3332]
```

但神经网络处理的是连续向量，不是离散整数。我们需要一个函数，将每个整数映射到高维空间中的一个向量。这正是 embedding 的作用。

Embedding 是一个查找表。它为词汇表中的每个 token 存储一个可学习的向量。当你传入 token ID 464 时，它返回第 464 行存储的向量。

```
Token ID 464  →  [0.12, -0.45, 0.78, ..., 0.33]  (d_model 维)
Token ID 3797 →  [-0.23, 0.56, 0.01, ..., -0.89]
```

这些向量在训练过程中通过反向传播不断更新。相似的 token 会拥有相似的向量，从而捕获语义关系。

---

## Token Embedding

### 查找表

最简单的 embedding 是 `nn.Embedding(vocab_size, d_model)`。它创建一个形状为 `(vocab_size, d_model)` 的矩阵，初始值为随机数。在训练过程中，这些值会通过反向传播更新。

```python
self.embedding = nn.Embedding(vocab_size, d_model)
```

当你传入一个 token ID 张量时，它直接索引到这个矩阵：

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.embedding(x)  # 形状: (batch, seq_len, d_model)
```

### 乘以 sqrt(d_model) 进行缩放

在原始 Transformer 论文（Vaswani 等人，2017）中，embedding 会乘以模型维度的平方根：

```python
return self.embedding(x) * math.sqrt(self.d_model)
```

为什么？不缩放的话，embedding 的大小取决于 `d_model`。维度越大，注意力机制中的点积就越大，导致 softmax 分布更尖锐（更集中）。乘以 `sqrt(d_model)` 可以让点积保持在一致的范围内，不受 embedding 维度的影响。

这是一个简单但重要的细节。很多实现会忽略它，但它对训练稳定性很重要。

---

## 位置编码

### 问题：排列不变性

Transformer 并行处理所有 token。与 RNN 不同，它们没有内在的顺序感。没有位置信息，模型看到 `"The cat sat"` 和 `"sat cat The"` 是完全一样的 —— 只是一堆 token 的集合。

我们需要注入位置信息。主要有两种方法：

1. **绝对位置编码**：为每个位置的 embedding 添加一个独特的向量。
2. **相对位置编码**：编码成对位置之间的距离。

### 绝对位置编码（正弦余弦）

原始 Transformer 使用正弦余弦函数：

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

每个位置获得一个独特的正弦余弦模式。这些模式在第一个注意力层之前被添加到 token embedding 中。

**优点：** 简单，没有可学习参数，适用于任意序列长度。
**缺点：** 被添加到 embedding 中，所以位置信息在第一层就与 token 信息混合。

### 相对位置编码

相对方法编码位置之间的距离，而不是绝对位置。直觉是：重要的不是"这是位置 5"，而是"这个 token 离那个 token 有 3 个位置的距离"。

**RoPE** 是一种通过旋转向量工作的相对位置编码。它被用于 LLaMA、Mistral、Qwen 等现代大语言模型。

---

## 旋转位置编码 (RoPE)

### 直觉：旋转向量

想象你有一个 2D 向量。你可以用它与 x 轴的夹角来表示它的方向。现在想象两个不同位置的向量。它们之间的角度差编码了它们的相对位置。

RoPE 将这个想法推广到高维。对于一个维度为 `d_model` 的向量，我们将它看作 `d_model/2` 个独立的 2D 向量（成对的维度）。每一对都会根据位置旋转一个角度。

```
位置 0: 旋转 0 * theta
位置 1: 旋转 1 * theta
位置 2: 旋转 2 * theta
...
```

当你计算两个旋转向量的点积时，结果只取决于相对位置（旋转角度的差），而不是绝对位置。

### 数学原理

#### 步骤 1：计算频率

每对维度有不同的旋转频率：

```
theta_i = 1 / (base^(2i/d_model))    其中 i = 0, 1, ..., d_model/2 - 1
```

其中 `base` 通常是 10000.0。较低的维度对旋转更快（频率较高），较高的维度对旋转较慢（频率较低）。

```python
inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
```

#### 步骤 2：计算旋转角度

对于每个位置 `m` 和维度对 `i`，旋转角度是：

```
angle(m, i) = m * theta_i
```

这通过外积计算：

```python
t = torch.arange(seq_len)           # 位置: [0, 1, 2, ...]
freqs = torch.outer(t, inv_freq)    # 形状: (seq_len, d_model/2)
```

#### 步骤 3：复制并计算三角函数

将频率复制到所有 `d_model` 维度：

```python
emb = torch.cat([freqs, freqs], dim=-1)  # 形状: (seq_len, d_model)
cos = emb.cos()
sin = emb.sin()
```

#### 步骤 4：应用旋转

对于位置 `m` 的每个向量 `x`：

```
x_rotated = x * cos(m * theta) + rotate_half(x) * sin(m * theta)
```

其中 `rotate_half(x)` 交换并取反成对的维度：

```
rotate_half([x1, x2, x3, x4]) = [-x2, x1, -x4, x3]
```

这等价于对每一对应用 2D 旋转矩阵：

```
[cos(theta)  -sin(theta)] [x1]
[sin(theta)   cos(theta)] [x2]
```

### 为什么 RoPE 有效

关键性质：当你计算位置 `m` 和 `n` 的两个 RoPE 旋转向量的点积时，结果只取决于 `m - n`（相对位置），而不是 `m` 和 `n` 各自的值。

这意味着：
- 模型可以自然地关注相对位置
- 不需要学习的位置参数
- 模型可以泛化到比训练时更长的序列

### 代码走读

这是完整的 `RotaryPositionalEmbedding` 类：

```python
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048, base=10000.0):
        super().__init__()
        self.d_model = d_model
        # 1. 计算逆频率
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def _build_cache(self, seq_len, device):
        # 2. 计算所有位置的角度
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)  # (seq_len, d_model/2)
        emb = torch.cat([freqs, freqs], dim=-1)  # (seq_len, d_model)
        return emb.cos(), emb.sin()

    def _rotate_half(self, x):
        # 3. 交换并取反: [x1, x2] -> [-x2, x1]
        x1 = x[..., :x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2:]
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, x):
        seq_len = x.shape[1]
        cos, sin = self._build_cache(seq_len, x.device)
        cos = cos[:seq_len].unsqueeze(0)  # 添加 batch 维度
        sin = sin[:seq_len].unsqueeze(0)
        # 4. 应用旋转
        return x * cos + self._rotate_half(x) * sin
```

**关键观察：**
- `_build_cache` 在这个简单实现中每次前向传播都会调用。在生产环境中，你会缓存 cos/sin 张量。
- `_rotate_half` 实现了每对维度的 90 度旋转。
- forward 方法应用旋转公式：`x * cos + rotate(x) * sin`。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/02_embedding/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分。然后验证：

```bash
pytest 01-foundations/02_embedding/tests.py -v -k "test"
```

---

## 练习

打开 `exercise.py`，自己动手实现 embedding。

### 练习顺序

1. **`TokenEmbeddingExercise.__init__`** — 创建 embedding 查找表
2. **`TokenEmbeddingExercise.forward`** — 查找 embedding 并乘以 sqrt(d_model)
3. **`RotaryPositionalEmbeddingExercise.__init__`** — 计算逆频率
4. **`RotaryPositionalEmbeddingExercise._build_cache`** — 预计算 cos/sin 值
5. **`RotaryPositionalEmbeddingExercise._rotate_half`** — 实现 90 度旋转
6. **`RotaryPositionalEmbeddingExercise.forward`** — 应用旋转公式

### 提示

- 从 `TokenEmbedding` 开始。它只是一个带缩放的查找表。
- 对于 RoPE，先理解 `_rotate_half`。它是核心操作。
- `_build_cache` 方法预计算所有旋转角度。把它想象成一个按位置和维度索引的 cos/sin 值表。
- forward 方法组合所有内容：`x * cos + rotate_half(x) * sin`。

---

## 核心要点

1. **Embedding 将离散 token 转换为连续向量。** Embedding 层是一个可学习的查找表，将 token ID 映射到稠密向量。

2. **乘以 sqrt(d_model) 可以稳定训练。** 它让 embedding 的大小保持一致，不受模型维度影响。

3. **位置信息对 Transformer 至关重要。** 没有它，模型无法区分相同 token 的不同排列。

4. **RoPE 通过旋转编码位置。** 每对维度根据位置旋转一个角度。两个旋转向量的点积只取决于它们的相对位置。

5. **RoPE 是相对的，不是绝对的。** 与正弦位置编码（被添加到 embedding 中）不同，RoPE 应用在注意力机制中，自然地捕获相对距离。

---

## 延伸阅读

- [RoPE 论文 (Su 等人, 2021)](https://arxiv.org/abs/2104.09864) — 旋转位置编码的原始论文
- [Attention Is All You Need (Vaswani 等人, 2017)](https://arxiv.org/abs/1706.03762) — 使用正弦位置编码的原始 Transformer 论文
- [LLaMA 论文 (Touvron 等人, 2023)](https://arxiv.org/abs/2302.13971) — 在现代大语言模型中使用 RoPE
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — Embedding 和位置编码的可视化解释
