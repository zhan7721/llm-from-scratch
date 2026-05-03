# Transformer Block: Pre-Norm, SwiGLU, and RMSNorm

> **Module 01 -- Foundations, Chapter 04**

Transformer 模型由多个相同的 **transformer block** 堆叠而成。每个 block 接收一个 token 表示序列，通过两个子层进行精炼：attention（跨位置混合信息）和 feed-forward network（独立变换每个位置）。本章我们实现一个现代 transformer block，采用 LLaMA、PaLM 等前沿模型的设计选择：

1. **RMSNorm** -- 比 LayerNorm 更简单、更快的替代方案
2. **SwiGLU** -- 性能优于 ReLU FFN 的门控前馈网络
3. **Pre-Norm 残差结构** -- 在每个子层之前进行归一化，提升训练稳定性

---

## 前置知识

- 理解 attention 机制（Chapter 03）
- 理解 embedding（Chapter 02）
- PyTorch 基础：`nn.Module`、`nn.Linear`、`nn.Parameter`

## 文件说明

| 文件 | 用途 |
|------|------|
| `transformer_block.py` | RMSNorm、SwiGLU 和 TransformerBlock 的核心实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 整体架构

单个 transformer block 的数据流如下：

```
输入: x (batch_size, seq_len, d_model)

    x = x + Attention(RMSNorm(x))      # 子层 1: 自注意力
    x = x + SwiGLU(RMSNorm(x))         # 子层 2: 前馈网络

输出: x (batch_size, seq_len, d_model)
```

`+` 号表示**残差连接** -- 将输入直接加到子层的输出上。这对训练深层网络至关重要。

在完整的 LLM 中，我们会堆叠多个这样的 block（例如 LLaMA 7B 有 32 层），并在输入处添加 RoPE 位置编码。

---

## RMSNorm：均方根归一化

### 功能

RMSNorm 将输入向量归一化，使其"能量"（均方根）约等于 1，然后应用可学习的缩放因子：

```
RMSNorm(x) = x / RMS(x) * weight
其中 RMS(x) = sqrt(mean(x^2) + eps)
```

### 与 LayerNorm 的对比

LayerNorm（原始 Transformer 和 GPT-2 使用）分两步：
1. **中心化**：减去均值
2. **缩放**：除以标准差

```
LayerNorm(x) = (x - mean(x)) / std(x) * weight + bias
```

RMSNorm 跳过第 1 步（不减均值），使用 RMS 代替标准差：

| 属性 | LayerNorm | RMSNorm |
|------|-----------|---------|
| 减均值 | 是 | 否 |
| 分母 | 标准差 | 均方根 |
| 偏置参数 | 有 | 无 |
| 可学习缩放 | 有 | 有 |
| 计算开销 | 较高 | 较低 |

### 为什么 RMSNorm 有效

原始论文（Zhang & Sennrich, 2019）证明 LayerNorm 中的中心化操作并非必需。真正重要的是重新缩放 -- 控制激活值的幅度。RMSNorm 以更低的计算成本实现了这一点。

实验表明，RMSNorm 在许多任务上的表现与 LayerNorm 相当或更优。LLaMA、PaLM、Gemma 等现代 LLM 均采用 RMSNorm。

### 代码解析

```python
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))  # 可学习缩放因子
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight
```

关键细节：
- `x.pow(2).mean(-1, keepdim=True)` 沿最后一个维度（特征维度）计算平方值的均值。`keepdim=True` 保留维度以便广播。
- 在开方前加 `eps` 防止除以零。
- `self.weight` 初始化为全 1，因此训练开始时 RMSNorm 近似于恒等函数。

---

## SwiGLU：门控前馈网络

### 标准 FFN

在原始 Transformer 中，每个 block 有一个逐位置前馈网络：

```
FFN(x) = W2 * ReLU(W1 * x + b1) + b2
```

将输入投影到更高维度（通常 4 倍），应用非线性激活，再投影回来。隐藏维度通常为 `d_ff = 4 * d_model`。

### ReLU 的问题

ReLU 将所有负值设为零。这意味着对于任何给定输入，一半的隐藏单元是"死"的。虽然这种稀疏性有时有用，但它限制了网络的表达能力。

### SwiGLU：门控更好

SwiGLU（Shazeer, 2020）用**门控机制**替换了简单的 ReLU 激活：

```
SwiGLU(x) = W2 * (SiLU(W_gate * x) * W1 * x)
```

其中：
- `W1 * x` 是"值" -- 要传递的信息
- `W_gate * x` 是"门" -- 控制多少值可以通过
- `SiLU`（也叫 Swish）是激活函数：`SiLU(x) = x * sigmoid(x)`
- `*` 是逐元素乘法（门控操作）

### 为什么用 SiLU 而不是 ReLU？

SiLU 相比 ReLU 有以下优势：
- **平滑**：在零点没有尖角，有利于基于梯度的优化
- **非单调**：在上升前略微为负，允许小的负输出
- **自门控**：`SiLU(x) = x * sigmoid(x)` -- 输入本身充当自己的门

### 隐藏维度公式的由来

使用 SwiGLU 时，默认隐藏维度为：

```
d_ff = (2/3) * 4 * d_model    （四舍五入到 256 的倍数）
```

这是因为 SwiGLU 有**三个**权重矩阵（W1、W_gate、W2），而标准 FFN 只有两个（W1、W2）。如果仍用 `d_ff = 4 * d_model`，FFN 的参数量会多 50%。`(2/3)` 因子使参数量大致回到与标准 FFN 相同的水平。

例如，`d_model = 4096` 时：
- 标准 FFN：`d_ff = 16384`，2 个权重矩阵 -> `2 * 4096 * 16384 = 1.34 亿`参数
- SwiGLU FFN：`d_ff = 10922`（四舍五入为 11008），3 个权重矩阵 -> `3 * 4096 * 11008 = 1.35 亿`参数

参数量几乎相同，但 SwiGLU 效果更好。

### 代码解析

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int = None):
        super().__init__()
        if d_ff is None:
            d_ff = int(2 * (4 * d_model) / 3)
            d_ff = ((d_ff + 255) // 256) * 256  # 四舍五入到 256 的倍数

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))
```

前向传播过程：
1. `self.w_gate(x)` 生成门，经 SiLU 激活
2. `self.w1(x)` 生成值
3. 逐元素相乘：`SiLU(门) * 值`
4. 用 `self.w2` 投影回 `d_model`

注意：现代 LLM 所有线性层都使用 `bias=False`。这减少了参数量，且对质量没有可测量的影响。

---

## Pre-Norm vs Post-Norm

### Post-Norm（原始 Transformer）

原始 Transformer 论文（2017）在每个子层**之后**应用归一化：

```
x = LayerNorm(x + Attention(x))
x = LayerNorm(x + FFN(x))
```

### Pre-Norm（GPT-2、LLaMA 及之后）

现代 LLM 在每个子层**之前**应用归一化：

```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### 为什么 Pre-Norm 更好

#### 1. 梯度流

在 Post-Norm 中，梯度必须在每一层经过 LayerNorm。对于 L 层的网络，梯度被乘以 L 次归一化操作，每次都可能缩小或扭曲梯度。这使得深层模型的训练不稳定。

在 Pre-Norm 中，残差连接提供了一条**直接的梯度路径**，完全绕过子层。梯度可以不受阻碍地从最后一层流到第一层：

```
d_output / d_input = 1 + d_sublayer / d_input
                     ^            ^
                     |            |
                直接路径      子层的贡献
```

`1` 确保梯度至少与上游梯度一样大。

#### 2. 训练稳定性

Post-Norm 模型对以下因素敏感：
- **学习率**：太高会发散，太低浪费算力
- **预热**：学习率预热是必需的
- **初始化**：差的初始化会导致训练失败

Pre-Norm 模型宽容得多。它们可以用更高的学习率、更少的预热（或不需要）、标准初始化来训练。

#### 3. 权衡

Pre-Norm 有一个微妙的缺点：残差流可能"主导"子层的贡献。在很深的网络中，子层对输出的贡献可能相对较小，因为残差连接承载了大部分信号。这有时被称为"表示坍缩"问题。

然而在实践中，这是可控的，训练稳定性的收益远大于此。所有主流 LLM（LLaMA、PaLM、GPT-3、Gemma）都使用 Pre-Norm。

### 代码：Pre-Norm 模式

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, causal=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff)

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))   # Pre-Norm attention
        x = x + self.ffn(self.ffn_norm(x))      # Pre-Norm FFN
        return x
```

每行遵循以下模式：
1. **归一化**输入：`self.attn_norm(x)`
2. 通过子层**变换**：`self.attn(...)`
3. **加上**残差：`x + ...`

---

## 残差连接

### 为什么重要

残差连接（He et al., 2015）是训练深层网络最重要的架构创新。它通过将输入直接加到输出来实现：

```
output = x + F(x)
```

而非仅仅：

```
output = F(x)
```

### 三个关键好处

#### 1. 梯度高速公路

在反向传播中，损失对输入的梯度包含一个简单的 `1` 项（恒等映射）。这意味着梯度可以直接从输出流到输入，不被子层衰减。

没有残差连接时，梯度必须经过每一层的权重和激活函数。在 32 层网络中，这意味着 32 次矩阵乘法，每次都可能缩小（或放大）梯度。有了残差连接，梯度有了捷径。

#### 2. 更容易优化

子层只需要学习**残差** `F(x) = output - x` -- 期望输出与输入的*差值*。学习小的修正比从头学习整个变换要容易得多。

#### 3. 恒等初始化

训练开始时，使用零初始化（或接近零）的权重，`F(x) = 0`，因此 `output = x`。网络从恒等函数开始，逐渐学习有意义的变换。这比随机初始化好得多。

### 在 Transformer Block 中

每个 transformer block 有两个残差连接：
1. 围绕 attention 子层
2. 围绕 FFN 子层

这意味着对于 32 层模型，梯度有 64 条从输出到输入的直接路径。

---

## 综合运用

### 完整的 TransformerBlock

```python
class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, causal=True):
        super().__init__()
        self.attn_norm = RMSNorm(d_model)       # attention 前归一化
        self.attn = MultiHeadAttention(d_model, n_heads, causal=causal)
        self.ffn_norm = RMSNorm(d_model)         # FFN 前归一化
        self.ffn = SwiGLU(d_model, d_ff)         # 门控 FFN

    def forward(self, x):
        x = x + self.attn(self.attn_norm(x))    # attention + 残差
        x = x + self.ffn(self.ffn_norm(x))      # FFN + 残差
        return x
```

### 数据流

对于位置 `t` 的单个 token：

```
1. 输入: x_t（d_model 维向量）

2. Attention 子层：
   a. 归一化: h = RMSNorm(x_t)
   b. Attention: 模型计算位置 t 应该关注
      其他每个位置（包括自身）的程度，使用 Q/K/V 投影
   c. 残差: x_t = x_t + attention_output

3. FFN 子层：
   a. 归一化: h = RMSNorm(x_t)
   b. 门: g = SiLU(W_gate @ h)
   c. 值: v = W1 @ h
   d. 组合: x_t = W2 @ (g * v)
   e. 残差: x_t = x_t + ffn_output

4. 输出: x_t（精炼后的表示）
```

### 每个 Block 的参数量

对于 `d_model` 和 `n_heads` 的 block：

| 组件 | 参数量 |
|------|--------|
| RMSNorm (x2) | 2 * d_model |
| Attention (Q, K, V, O) | 4 * d_model^2 |
| SwiGLU (W1, W_gate, W2) | 3 * d_model * d_ff |
| **总计** | ~4 * d_model^2 + 3 * d_model * d_ff + 2 * d_model |

对于 LLaMA 7B（d_model=4096, d_ff=11008, n_heads=32）：
- Attention: 4 * 4096^2 = 6700 万
- SwiGLU: 3 * 4096 * 11008 = 1.35 亿
- 每个 block 总计：约 2.02 亿

---

## 现代 LLM 的设计选择

| 组件 | 原始 Transformer | LLaMA / 现代 LLM |
|------|-----------------|------------------|
| 归一化 | LayerNorm (Post-Norm) | RMSNorm (Pre-Norm) |
| FFN 激活 | ReLU | SwiGLU |
| FFN 隐藏维度 | 4 * d_model | (2/3) * 4 * d_model |
| 线性层偏置 | 有 | 无 |
| 位置编码 | 绝对正弦 | RoPE |

每个改变的动机要么是提升训练稳定性，要么是提高质量，要么是降低计算成本。

---

## 运行方式

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/04_transformer_block/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分，然后验证：

```bash
pytest 01-foundations/04_transformer_block/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己实现 transformer block。

### 练习顺序

1. **`RMSNorm.forward`** -- 实现 RMS 归一化：计算 RMS、除法、乘以权重
2. **`SwiGLU.forward`** -- 实现门控 FFN：SiLU 门乘以线性值，然后投影
3. **`TransformerBlock.forward`** -- 组装 Pre-Norm 残差结构

### 提示

- 从 RMSNorm 开始。公式很简单：`x / sqrt(mean(x^2) + eps) * weight`。计算均值时使用 `keepdim=True` 以便正确广播。
- 对于 SwiGLU，关键是有两条并行路径：门（`W_gate`）和值（`W1`）。对门应用 SiLU 后，通过逐元素乘法组合。
- 对于 TransformerBlock，每个子层遵循相同模式：归一化、变换、加残差。只需两行代码。

---

## 核心要点

1. **RMSNorm 更简单且同样有效。** 通过去掉减均值操作，在不牺牲质量的前提下节省计算。所有主流 LLM 均已采用。

2. **SwiGLU 用门控机制替代 ReLU。** 对一个线性投影应用 SiLU 激活作为门，控制另一个线性投影的通过量。这比 ReLU 更具表现力，已成为现代 LLM 的标准 FFN。

3. **Pre-Norm 稳定训练。** 在每个子层之前（而非之后）归一化，梯度通过残差连接更顺畅地流动。这使得更深的模型能以更高的学习率训练。

4. **残差连接不可或缺。** 它们提供梯度高速公路，使优化更容易，并允许网络从恒等函数开始。每个现代深度网络都使用它们。

5. **Transformer block 是基本构建单元。** 完整的 LLM 只是这些 block 的堆叠，输入端有 embedding，输出端有语言模型头。

---

## 延伸阅读

- [LLaMA Paper (Touvron et al., 2023)](https://arxiv.org/abs/2302.13971) -- 使用 RMSNorm、SwiGLU、Pre-Norm、RoPE
- [PaLM Paper (Chowdhery et al., 2022)](https://arxiv.org/abs/2204.02311) -- 大规模使用 SwiGLU
- [RMSNorm Paper (Zhang & Sennrich, 2019)](https://arxiv.org/abs/1910.07467) -- 均方根层归一化
- [GLU Variants Paper (Shazeer, 2020)](https://arxiv.org/abs/2002.05202) -- SwiGLU 及其他门控 FFN 变体
- [On Layer Normalization in the Transformer (Xiong et al., 2020)](https://arxiv.org/abs/2002.04745) -- Pre-Norm vs Post-Norm 分析
- [Deep Residual Learning (He et al., 2015)](https://arxiv.org/abs/1512.03385) -- 原始残差连接论文
