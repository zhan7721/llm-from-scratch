# 混合专家模型 (Mixture of Experts, MoE)

> **模块 01 — 基础，第 06 章**

标准 Transformer 通过加宽和加深每一层来扩展模型，但这意味着每个 token 都要经过所有参数。混合专家模型 (MoE) 打破了这一假设：不再使用一个大型前馈网络，而是使用多个较小的"专家"网络，由一个学习到的路由器将每个 token 只发送给其中少数几个专家。这样做的结果是模型总参数量更大，但每个 token 的计算量与较小的稠密模型相同。

这就是 Mixtral 8x7B 能够在保持推理速度的同时，达到与更大稠密模型相当的质量的原因。

---

## 前置知识

- Transformer 块结构（第 04 章）
- 前馈网络和 SwiGLU 激活函数
- Softmax 和 top-K 选择

## 文件说明

| 文件 | 用途 |
|------|------|
| `moe.py` | 核心实现：TopKRouter、Expert、MoELayer |
| `exercise.py` | 填空练习，用于巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 什么是 MoE 以及为什么重要

### 扩展的困境

要让 Transformer 更强大，通常需要增加参数量。70 亿参数的模型优于 10 亿参数的模型，700 亿参数的模型优于 70 亿参数的模型。但更多参数意味着每个 token 需要更多计算：每个 token 都流经每一层的每一个权重矩阵。

MoE 打破了这一权衡。通过将 Transformer 块中的前馈网络 (FFN) 替换为多个较小的专家网络和一个路由器，我们可以：

- **增加总参数量**（更多知识容量）
- **保持每个 token 的计算量不变**（每个 token 只使用少数几个专家）
- **扩展模型容量而不成比例地增加成本**

### 核心思想

不再是每层一个 FFN：

```
x -> FFN -> 输出
```

而是有 N 个专家 FFN，路由器为每个 token 选择 top-K 个：

```
x -> 路由器 -> [Expert_0, Expert_1, ..., Expert_N-1]
                 选择 top-K 个，加权平均 -> 输出
```

使用 8 个专家和 top-2 路由时，每个 token 只激活 8 个专家中的 2 个。每个 token 的浮点运算量大致等于单个 FFN，但模型的 FFN 层参数量增加了 8 倍。

---

## 架构

### TopKRouter（Top-K 路由器）

路由器决定哪些专家处理每个 token。它是一个简单的线性层：

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x_flat)              # (B*T, n_experts)
        weights, indices = torch.topk(logits, self.top_k)  # top-K
        weights = F.softmax(weights, dim=-1)    # 归一化
        return indices, weights
```

**工作原理：**
1. 将每个 token 向量投影到每个专家的分数
2. 按分数选择 top-K 个专家
3. 对选中的分数进行 softmax 得到路由权重（总和为 1）

路由器与模型的其余部分一起进行端到端训练。它学习哪些类型的 token 最适合哪些专家。

### Expert（专家）

每个专家是一个独立的 SwiGLU 前馈网络：

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))
```

这与 LLaMA 和其他现代大语言模型使用的 FFN 设计相同。SiLU 门控激活比 ReLU 提供更好的梯度流。

### MoELayer（MoE 层）

完整的层结合了路由和专家计算：

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        self.router = TopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        indices, weights = self.router(x)

        for k in range(self.top_k):
            for e_idx in range(n_experts):
                mask = (indices[:, k] == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](x_flat[mask])
                    output[mask] += weights[mask, k] * expert_out

        return output
```

遍历专家的循环看起来效率不高（对于生产环境确实如此），但它清晰地展示了算法：对于每个路由槽位，找到哪些 token 去哪个专家，通过专家运行这些 token，并累加加权输出。

---

## 负载均衡

### 问题

如果不进行任何正则化，路由器可能会坍缩为总是选择相同的 1-2 个专家。这是一种"强者愈强"的现象：一旦某个专家开始获得更多 token，它就会获得更多训练信号，变得更好，从而吸引更多 token。其他专家则被饿死。

这被称为**专家坍缩**或**路由不均衡**。它浪费了模型容量——你有 8 个专家但只有 2 个在工作。

### 解决方案

**1. 辅助负载均衡损失**

添加一个惩罚项来鼓励均匀路由。Switch Transformer 论文提出了这种方法：

```python
# 路由到每个专家的 token 比例
f_i = 路由到专家 i 的 token 比例

# 每个专家的平均路由权重
p_i = 专家 i 的平均路由权重

# 辅助损失（当路由均匀时最小化）
aux_loss = n_experts * sum(f_i * p_i)
```

该损失以较小的系数（如 0.01）添加到主训练损失中。

**2. 容量因子**

限制每个专家可以处理的最大 token 数量。如果某个专家的容量已满，溢出的 token 将被路由到次优专家或被丢弃。这防止单个专家过载。

**3. 带噪声的随机路由**

在 top-K 选择之前向路由器 logits 添加噪声。这鼓励探索并防止过早坍缩：

```python
logits = self.gate(x) + torch.randn_like(logits) * noise_std
```

**4. 专家选择路由**

不是每个 token 选择专家，而是每个专家选择其 top token。这从结构上保证了均衡的负载，但改变了路由语义。

---

## MoE 的实际应用

### Mixtral 8x7B（Mistral AI，2024）

- 每层 8 个专家，top-2 路由
- 总参数 470 亿，每个 token 活跃参数约 130 亿
- 匹配或超越 LLaMA 2 70B 的质量
- 使用 SwiGLU 专家、滑动窗口注意力

### Switch Transformer（Google，2022）

- Top-1 路由（每个 token 只去一个专家）
- 最多 1.6 万亿参数
- 引入了辅助负载均衡损失
- 证明了 MoE 在相同计算预算下比稠密模型扩展性更好

### DeepSeek-V2 / DeepSeek-MoE（DeepSeek，2024）

- 细粒度专家：许多小专家而非少数大专家
- 共享专家始终激活（捕获通用知识）
- 带辅助损失的 top-K 路由
- 证明了更多、更小的专家可以更高效

### GShard（Google，2020）

- 带容量因子的 top-2 路由
- 首批大规模 MoE 模型之一（6000 亿参数）
- 引入了带噪声的随机路由技巧

---

## 代码详解

### TopKRouter

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)           # (B*T, D)
        logits = self.gate(x_flat)        # (B*T, n_experts)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return indices, weights
```

关键细节：
- 门控是一个没有偏置的单线性层——它只是将 token 表示投影到专家分数
- `torch.topk` 返回 top-K 条目的值和索引
- Softmax 仅应用于 top-K 的值，而非所有专家——选中专家的权重总和为 1

### Expert（SwiGLU）

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))
```

SwiGLU 的工作方式：
1. 通过两个并行路径投影 x：`w1(x)` 和 `w_gate(x)`
2. 对门控路径应用 SiLU 激活：`SiLU(w_gate(x))`
3. 逐元素相乘：`SiLU(w_gate(x)) * w1(x)` — 这就是"门控"
4. 投影回 d_model：`w2(...)`

门控机制让网络学习哪些特征应该通过、哪些应该被抑制，比简单的 ReLU FFN 提供更好的表达能力。

### MoELayer

```python
class MoELayer(nn.Module):
    def forward(self, x):
        B, T, D = x.shape
        indices, weights = self.router(x)   # 路由每个 token
        x_flat = x.view(-1, D)
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):         # 对于每个路由槽位
            for e_idx in range(n_experts):   # 对于每个专家
                mask = (indices[:, k] == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask, k] * expert_output

        return output.view(B, T, D)
```

这是一个清晰但朴素的实现。双重循环使其易于理解但速度较慢。生产实现使用分组矩阵乘法（grouped GEMM）或专门的 CUDA 内核来并行处理所有专家。

---

## 权衡

### 更多专家 = 更多容量

每个专家可以专注于不同类型的输入。使用 8 个专家，模型可以学习 8 种不同的"视角"来处理 token。这就像有 8 个专家而非 1 个通才。

### 稀疏激活 = 恒定计算

即使有 8 个专家，每个 token 也只使用 2 个。每个 token 的浮点运算量大约为：
```
MoE 每个 token 的浮点运算量 = 2 * (3 * d_model * d_ff)   # 与单个 FFN 相同
```
而具有 8 个专家的稠密模型：
```
稠密模型每个 token 的浮点运算量 = 8 * (3 * d_model * d_ff)  # 多 4 倍
```

### 内存是瓶颈

虽然计算是稀疏的，但内存不是。所有专家参数必须存储在 GPU 内存中，即使大多数专家对于任何给定的 token 都处于空闲状态。这是部署 MoE 模型的主要挑战：
- Mixtral 8x7B 有 470 亿参数（fp16 需要约 94GB）
- 但每个 token 只有约 130 亿参数活跃（计算量与 130 亿参数的稠密模型相同）

### 负载均衡至关重要

没有适当的正则化，专家坍缩会浪费容量。辅助损失增加了少量训练成本，但对于模型实际使用所有专家是必不可少的。

### 分布式环境中的通信

在多 GPU 设置中，不同的专家可能位于不同的设备上。将 token 路由到其他 GPU 上的专家需要设备间通信，这可能成为瓶颈。专家并行（将专家分布在多个 GPU 上）是大规模 MoE 模型的关键工程挑战。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/06_moe/tests.py -v
```

### 运行练习

打开 `exercise.py`，填写 `TODO` 部分。然后验证：

```bash
pytest 01-foundations/06_moe/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习自己实现 MoE。

### 练习顺序

1. **`ExpertExercise.__init__`** — 创建 w1、w_gate、w2 线性层
2. **`ExpertExercise.forward`** — 实现 SwiGLU 前向传播
3. **`TopKRouterExercise.__init__`** — 创建门控线性层
4. **`TopKRouterExercise.forward`** — 实现带 top-K 选择的路由
5. **`MoELayerExercise.__init__`** — 创建路由器和专家列表
6. **`MoELayerExercise.forward`** — 实现完整的 MoE 前向传播

### 提示

- 从 Expert 开始。SwiGLU 就是：`w2(SiLU(w_gate(x)) * w1(x))`。三个线性层和一个激活函数。
- 路由器比看起来简单：一个线性层，然后 `torch.topk`，然后 softmax。
- MoE 前向传播是一个循环：对于每个路由槽位，对于每个专家，找到匹配的 token，运行它们，累加。效率不高但正确且清晰。
- `torch.topk` 返回 `(values, indices)` — values 是分数，indices 是选中的专家。

---

## 关键要点

1. **MoE 扩展模型容量而不扩展每个 token 的计算量。** 通过将每个 token 只路由到少数几个专家，我们获得了大模型的知识容量和小模型的计算成本。

2. **路由器是学习到的门控网络。** 它将每个 token 投影到专家分数并选择 top-K。路由权重是对选中专家进行 softmax 归一化后的结果。

3. **专家是独立的 FFN 网络。** 每个专家是一个 SwiGLU 前馈网络，可以专注于不同类型的输入。

4. **负载均衡防止专家坍缩。** 没有正则化时，路由器可能将所有 token 路由到相同的专家，浪费容量。需要辅助损失或容量因子。

5. **MoE 在实际中被广泛使用。** Mixtral、Switch Transformer、DeepSeek 和 GShard 都使用 MoE 来实现比稠密模型更好的扩展性。

---

## 延伸阅读

- [Switch Transformer (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961) — Top-1 路由，MoE 的扩展规律
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) — Mixtral 8x7B 架构
- [GShard (Lepikhin et al., 2021)](https://arxiv.org/abs/2006.16668) — 大规模 MoE 训练
- [DeepSeek-MoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) — 带共享专家的细粒度 MoE
- [ST-MoE (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906) — MoE 的稳定性和设计选择
