# 多头注意力和分组查询注意力

> **模块 01 — 基础，第 03 章**

注意力机制是 Transformer 的核心。它允许每个 token 查看序列中的所有其他 token，并决定对每个 token "关注"多少。在本章中，我们从零开始构建两种注意力变体：

1. **多头注意力 (MHA)** — 来自 "Attention Is All You Need" 的标准注意力机制
2. **分组查询注意力 (GQA)** — LLaMA 2/3、Mistral 等现代大语言模型使用的高效变体

这些层是 Transformer 中真正进行"思考"的部分。其他一切 — embedding、层归一化、前馈网络 — 都是为了支持它们而存在的。

---

## 前置要求

- 理解 embedding（第 02 章）
- 线性代数基础：矩阵乘法、转置、softmax
- 基本的 PyTorch 知识：`nn.Linear`、张量重塑

## 文件说明

| 文件 | 用途 |
|------|------|
| `attention.py` | MultiHeadAttention 和 GroupedQueryAttention 的核心实现 |
| `exercise.py` | 填空练习，用于加深理解 |
| `solution.py` | 练习的参考答案 |
| `tests.py` | 正确性测试 |

---

## 直觉

想象你正在读这句话："那只很老的猫坐在垫子上。"

当你读到"坐"时，你需要知道*谁*坐了。答案是"猫"，它在几个词之外。注意力给了模型一种机制，可以直接将"坐"与"猫"连接起来，不管它们之间的距离有多远。

更正式地说，注意力让每个 token 计算序列中所有其他 token 的加权和，其中权重是基于每个 token 的相关性学习得到的。

---

## 缩放点积注意力

### 公式

核心操作是：

```
Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
```

其中：
- **Q** (查询): "我在找什么？"
- **K** (键): "我包含什么？"
- **V** (值): "我提供什么信息？"
- **d_k**: 键向量的维度

### 逐步分解

#### 步骤 1：计算 Q, K, V

每个输入 token 向量 `x` 被投影成三个不同的向量：

```
Q = x @ W_q    (这个 token 在寻找什么)
K = x @ W_k    (这个 token 作为键提供什么)
V = x @ W_v    (这个 token 提供什么信息)
```

这些是可学习的线性投影。相同的权重应用于每个位置。

#### 步骤 2：计算注意力分数

位置 i 的查询和位置 j 的键之间的分数是它们的点积：

```
score(i, j) = Q[i] @ K[j]^T
```

我们通过矩阵乘法一次计算所有分数：

```
scores = Q @ K^T    # 形状: (seq_len, seq_len)
```

每一行 i 告诉我们位置 i 应该关注每个其他位置的程度。

#### 步骤 3：缩放

我们除以 `sqrt(d_k)`：

```
scores = scores / sqrt(d_k)
```

**为什么要缩放？** 不缩放的话，随着 `d_k` 增大，点积的幅度会增大。大的值会把 softmax 推到梯度极小的区域，使训练不稳定。除以 `sqrt(d_k)` 可以让点积的方差保持在大约 1，不受 `d_k` 影响。

这是一个数学结果：如果 Q 和 K 的元素独立且均值为 0、方差为 1，那么 `Q @ K^T` 的方差是 `d_k`。除以 `sqrt(d_k)` 将方差归一化回 1。

#### 步骤 4：应用 Softmax

将分数转换为概率：

```
weights = softmax(scores, dim=-1)    # 每行之和为 1
```

现在 `weights[i, j]` 表示位置 i 对位置 j 的关注程度。

#### 步骤 5：值的加权和

将注意力权重乘以 V 得到输出：

```
output = weights @ V    # 形状: (seq_len, d_k)
```

每个输出位置是所有值向量的加权组合，权重由注意力分数决定。

### 代码

```python
scores = Q @ K.transpose(-2, -1) / math.sqrt(d_k)
weights = F.softmax(scores, dim=-1)
output = weights @ V
```

三行代码。这就是注意力的核心。

---

## 多头注意力

### 为什么需要多个头？

单个注意力头一次只能学习一种类型的注意力。有了多个头，模型可以同时关注不同类型的信息：

- 一个头可能学习语法关系（主语-谓语）
- 另一个可能学习语义相似性（猫-动物）
- 另一个可能学习位置模式（相邻的词）

### 工作原理

1. **分割**：将 d_model 维度分成 `n_heads` 组，每组维度为 `d_k = d_model / n_heads`
2. **独立计算注意力**：在每个头中独立运行缩放点积注意力
3. **拼接**：拼接所有头的输出
4. **投影**：应用最终的线性投影来组合各头的结果

```python
# 分割成多个头
Q = Q.view(B, T, n_heads, d_k).transpose(1, 2)  # (B, n_heads, T, d_k)
K = K.view(B, T, n_heads, d_k).transpose(1, 2)
V = V.view(B, T, n_heads, d_k).transpose(1, 2)

# 在每个头中计算注意力（批量矩阵乘法一次处理所有头）
scores = Q @ K.transpose(-2, -1) / sqrt(d_k)
weights = softmax(scores)
output = weights @ V  # (B, n_heads, T, d_k)

# 拼接并投影
output = output.transpose(1, 2).contiguous().view(B, T, d_model)
output = output @ W_o
```

### 参数数量

每个头有三个大小为 `(d_model, d_k)` 的投影矩阵。有 `n_heads` 个头，Q、K、V 的总参数量为：

```
3 * d_model * d_model = 3 * d_model^2
```

这与单个大投影相同 — "分割成头"只是一个重塑技巧。实际参数是形状为 `(d_model, d_model)` 的完整 `W_q`、`W_k`、`W_v` 矩阵。

---

## 因果掩码

### 问题

在自回归模型（如 GPT）中，我们一次生成一个 token。在预测下一个 token 时，模型应该只看到之前的 token，不能看到未来的。

在训练期间，为了效率我们一次处理整个序列。但我们需要确保位置 i 不能关注位置 > i 的 token。

### 解决方案

在应用 softmax 之前，我们将所有未来位置设为负无穷：

```python
causal_mask = torch.triu(torch.ones(T, T), diagonal=1).bool()
scores = scores.masked_fill(causal_mask, float("-inf"))
```

softmax 之后，`exp(-inf) = 0`，所以这些位置对注意力输出没有贡献。

```
scores = [[ 0.5,  -inf,  -inf]
          [ 0.3,   0.7,  -inf]
          [ 0.2,   0.4,   0.9]]

weights = [[1.0,  0.0,  0.0]
           [0.3,  0.7,  0.0]
           [0.1,  0.2,  0.7]]
```

位置 0 只能关注自己。位置 1 可以关注位置 0 和 1。位置 2 可以关注所有位置。这形成了一个下三角的注意力模式。

---

## 分组查询注意力 (GQA)

### 动机：KV 缓存问题

在自回归推理过程中，我们需要存储所有之前 token 的 K 和 V 向量（"KV 缓存"）。对于一个有 `n_heads` 个头、每个头 `d_k` 维度、序列长度为 T 的模型，每层的缓存大小为：

```
KV 缓存 = 2 * n_heads * d_k * T * 每个元素的字节数
```

对于一个 70B 参数的模型，有 64 个头、每个头 128 维、4K 上下文长度，这需要数 GB 的内存。KV 缓存通常是推理吞吐量的瓶颈。

### 核心思想

GQA 减少了 KV 头的数量。我们不再有 `n_heads` 个 KV 头（每个查询头一个），而是有 `n_kv_heads` 个 KV 头，其中 `n_kv_heads < n_heads`。每个 KV 头在一组查询头之间共享。

```
MHA:  8 个查询头, 8 个 KV 头   (每个查询头有自己的 KV)
GQA:  8 个查询头, 2 个 KV 头   (每 4 个查询头共享一个 KV 头)
MQA:  8 个查询头, 1 个 KV 头   (所有查询头共享一个 KV 头)
```

### 工作原理

1. **投影 Q** 到 `n_heads` 个头（与 MHA 相同）
2. **投影 K、V** 到 `n_kv_heads` 个头（比 MHA 少）
3. **重复** 每个 KV 头 `n_rep = n_heads / n_kv_heads` 次以匹配查询头数量
4. **照常计算注意力**

```python
# 投影
Q = W_q(x).view(B, T, n_heads, d_k)       # (B, T, 8, d_k)
K = W_k(x).view(B, T, n_kv_heads, d_k)     # (B, T, 2, d_k)
V = W_v(x).view(B, T, n_kv_heads, d_k)     # (B, T, 2, d_k)

# 重复 KV 头
K = repeat_kv(K)  # (B, T, 2, d_k) -> (B, T, 8, d_k)
V = repeat_kv(V)  # (B, T, 2, d_k) -> (B, T, 8, d_k)

# 之后是标准注意力
scores = Q @ K^T / sqrt(d_k)
```

### 参数节省

使用 `n_kv_heads` 个 KV 头而非 `n_heads` 个：

```
MHA KV 参数: 2 * d_model * d_model
GQA KV 参数: 2 * d_model * (n_kv_heads / n_heads) * d_model
```

对于 LLaMA 2 70B：`n_heads=64, n_kv_heads=8`，因此 KV 参数减少了 8 倍。

### 质量权衡

GQA 的 `n_kv_heads` 介于 1（MQA）和 `n_heads`（MHA）之间，提供了良好的平衡：

| 方法 | KV 头数 | 质量 | KV 缓存大小 |
|------|---------|------|-------------|
| MHA  | n_heads | 最佳 | 最大 |
| GQA  | 8       | 接近 MHA | 小 8 倍 |
| MQA  | 1       | 较差 | 最小 |

LLaMA 2 70B 使用 8 个 KV 头的 GQA，以显著的推理加速实现了与 MHA 相当的质量。

---

## 比较：MHA vs MQA vs GQA

### 多头注意力 (MHA)
- **KV 头数**: `n_heads`（每个查询头一个）
- **优点**: 最佳质量，最具表达力
- **缺点**: KV 缓存大，推理慢
- **用于**: GPT-3，原始 Transformer

### 多查询注意力 (MQA)
- **KV 头数**: 1（所有查询头共享）
- **优点**: 最小的 KV 缓存，最快的推理
- **缺点**: 质量下降，尤其是大模型
- **用于**: PaLM, Falcon

### 分组查询注意力 (GQA)
- **KV 头数**: `n_kv_heads`（折中方案）
- **优点**: 接近 MHA 的质量，KV 缓存小得多
- **缺点**: 实现稍复杂
- **用于**: LLaMA 2/3, Mistral, Qwen

---

## 代码走读

### MultiHeadAttention

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, causal=False):
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, mask=None):
        B, T, C = x.shape

        # 投影并重塑为多头格式
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # 缩放点积注意力
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)

        # 掩码
        if self.causal:
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask, float("-inf"))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float("-inf"))

        # 注意力
        weights = F.softmax(scores, dim=-1)
        output = weights @ V

        # 拼接各头并投影
        output = output.transpose(1, 2).contiguous().view(B, T, C)
        return self.W_o(output)
```

### GroupedQueryAttention

```python
class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, causal=False):
        self.W_q = nn.Linear(d_model, n_heads * self.d_k, bias=False)
        self.W_k = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # 更少！
        self.W_v = nn.Linear(d_model, n_kv_heads * self.d_k, bias=False)  # 更少！
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def _repeat_kv(self, x):
        """复制 KV 头以匹配查询头数量。"""
        B, n_kv, T, d_k = x.shape
        if self.n_rep == 1:
            return x
        return x[:, :, None, :, :].expand(B, n_kv, self.n_rep, T, d_k).reshape(
            B, self.n_heads, T, d_k
        )

    def forward(self, x, mask=None):
        B, T, C = x.shape

        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_kv_heads, self.d_k).transpose(1, 2)

        # 扩展 KV 头
        K = self._repeat_kv(K)
        V = self._repeat_kv(V)

        # 之后是相同的注意力计算
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        # ... 掩码、softmax、输出 ...
```

**关键区别**：K 和 V 投影输出 `n_kv_heads * d_k` 而非 `n_heads * d_k`。`_repeat_kv` 方法将它们扩展回 `n_heads` 以进行注意力计算。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/03_attention/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分。然后验证：

```bash
pytest 01-foundations/03_attention/tests.py -v
```

---

## 练习

打开 `exercise.py`，自己动手实现注意力机制。

### 练习顺序

1. **`MultiHeadAttentionExercise.__init__`** — 创建 W_q、W_k、W_v、W_o 线性层
2. **`MultiHeadAttentionExercise.forward`** — 实现完整的 MHA 前向传播
3. **`GroupedQueryAttentionExercise.__init__`** — 创建更少 KV 头的投影
4. **`GroupedQueryAttentionExercise._repeat_kv`** — 实现 KV 头重复
5. **`GroupedQueryAttentionExercise.forward`** — 实现完整的 GQA 前向传播

### 提示

- 从 MHA 开始。前向传播的流程是：投影 -> 重塑 -> 分数 -> 掩码 -> softmax -> 输出 -> 投影。
- `.view(B, T, n_heads, d_k).transpose(1, 2)` 这个模式是将平坦向量分割成多个头的方式。理解为什么需要转置（我们希望头作为维度 1，以便进行批量矩阵乘法）。
- 对于 GQA，唯一的区别是 K 和 V 有更少的头。`_repeat_kv` 方法弥补了这个差距。
- 因果掩码只是一个上三角布尔矩阵。`torch.triu` 创建它。

---

## 核心要点

1. **注意力基于相关性计算加权和。** 每个 token 查看所有其他 token，并决定对每个 token 关注多少。

2. **除以 sqrt(d_k) 可以稳定训练。** 不这样做的话，大的点积会导致 softmax 饱和，造成梯度消失。

3. **多个头捕获不同的关系。** 每个头可以学习关注输入的不同方面（语法、语义、位置）。

4. **因果掩码实现自回归生成。** 通过阻断未来位置，模型在做预测时只能使用过去的上下文。

5. **GQA 以最小的质量损失减少 KV 缓存大小。** 通过在查询头组之间共享 KV 头，我们获得了显著的推理加速，质量损失很小。

---

## 延伸阅读

- [Attention Is All You Need (Vaswani 等人, 2017)](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [GQA 论文 (Ainslie 等人, 2023)](https://arxiv.org/abs/2305.13245) — 分组查询注意力
- [LLaMA 2 论文 (Touvron 等人, 2023)](https://arxiv.org/abs/2307.09288) — 在生产模型中使用 GQA
- [多查询注意力 (Shazeer, 2019)](https://arxiv.org/abs/1911.02150) — MQA 的原始提案
- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — 注意力的可视化解释
