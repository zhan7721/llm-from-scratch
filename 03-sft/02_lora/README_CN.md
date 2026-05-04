# 低秩适应 (LoRA)

## 动机

对大型语言模型进行全量微调需要更新网络中的每一个参数。对于一个 70 亿参数的模型，这意味着需要存储和计算 70 亿个浮点数的梯度——需要数十 GB 的 GPU 显存和大量算力。

但关键洞察是：**对于任何特定任务，大部分参数是冗余的。** Aghajanyan 等人（2020）的研究表明，预训练模型具有很低的"内在维度"——解决下游任务所需的真正独立方向数量远少于参数数量。

LoRA 利用这一点，通过学习权重矩阵的**低秩更新**而非完整更新。

## LoRA 的核心思想

不直接更新形状为 (d_out, d_in) 的权重矩阵 W，而是将更新分解为两个小矩阵：

```
Delta_W = B @ A
```

其中：
- A 的形状为 (rank, d_in)
- B 的形状为 (d_out, rank)
- rank << min(d_out, d_in)

前向传播变为：

```
y = x @ W^T + (alpha / rank) * x @ A^T @ B^T
```

原始权重 W 被**冻结**——只有 A 和 B 在训练过程中接收梯度并更新。

## 数学原理

给定预训练权重矩阵 W_0，适应后的权重为：

```
W' = W_0 + (alpha / rank) * B @ A
```

初始化时：
- A 使用 Kaiming 均匀分布初始化（随机）
- B 使用**零**初始化

这意味着训练开始时 Delta_W = 0，因此模型输出与预训练模型完全相同。训练随后逐步学习适应。

### 缩放因子

缩放因子 `alpha / rank` 控制 LoRA 更新相对于原始权重的大小。常见设置是 alpha = 2 * rank，缩放因子为 2.0。

- **alpha 越大** = LoRA 更新越大，表达能力更强但稳定性降低
- **alpha 越小** = LoRA 更新越小，更加保守

## 为什么有效

### 内在维度

预训练模型已经学习了对许多任务接近最优的表示。权重空间中从预训练模型到好的微调模型的"距离"往往位于一个低维子空间中。

LoRA 通过构造限制了权重更新在这个低维子空间中。rank=8 时，每个权重更新只能在 8 个独立方向上移动——但这通常已经足够。

### 参数效率

对于大小为 (d_out, d_in) 的权重矩阵：
- 全量微调：d_out * d_in 个参数
- rank 为 r 的 LoRA：r * (d_in + d_out) 个参数

对于 4096x4096 的权重矩阵，rank=8：
- 全量：1670 万参数
- LoRA：6.55 万参数（减少 256 倍）

## 超参数

### 秩 (r)

低秩分解的秩。常用值：4、8、16、32、64。

- 秩越低 = 参数越少，训练越快，表达能力越弱
- 秩越高 = 参数越多，训练越慢，表达能力越强
- 对于大多数任务，rank 8-16 已经足够

### Alpha

控制 LoRA 更新的幅度。有效更新按 `alpha / rank` 缩放。

- 常见设置：alpha = 2 * rank（缩放因子 = 2.0）
- 一些实践者使用 alpha = rank（缩放因子 = 1.0）

### 目标模块

应用 LoRA 的层。常见选择：

- **仅注意力**：W_q、W_k、W_v、W_o（文献中最常见）
- **注意力 + FFN**：还包括前馈网络中的 w1、w2、w_gate
- **所有线性层**：最全面

原始 LoRA 论文发现同时适应 W_q 和 W_v 效果良好，但现代实践通常针对所有注意力投影。

### Dropout

应用于 LoRA 路径输入的 LoRA 特定 Dropout。有助于防止过拟合，特别是对小数据集。典型值：0.0 到 0.1。

## 合并与部署

LoRA 最大的优势之一：训练后，你可以将 LoRA 权重**合并**回原始模型：

```
W_final = W_0 + (alpha / rank) * B @ A
```

合并后：
- 模型与原始模型具有相同的架构
- 推理时没有额外参数或计算开销
- 合并后的模型可以作为即插即用的替代品

这意味着**零推理开销**——适应后的模型运行速度与基础模型完全相同。

你还可以为不同任务保留多个 LoRA 适配器并在运行时切换，或合并不同的 LoRA 来组合能力。

## 代码解析

### LoRALinear

核心构建块。用可训练的 A 和 B 矩阵包装一个冻结的 `nn.Linear`：

```python
class LoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0, dropout=0.0, bias=True):
        # 冻结的原始层
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear.weight.requires_grad = False

        # 可训练的 LoRA 矩阵
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # B 初始化为零，因此初始输出与冻结模型相同
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
```

前向传播添加 LoRA 贡献：

```python
def forward(self, x):
    orig_output = self.linear(x)
    lora_output = x @ self.lora_A.T @ self.lora_B.T * self.scaling
    return orig_output + lora_output
```

### apply_lora

将目标 `nn.Linear` 模块替换为 `LoRALinear`：

```python
def apply_lora(model, rank=8, alpha=16.0, target_modules=None, dropout=0.0):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(t in name for t in target_modules):
            # 用 LoRA 版本替换，保留原始权重
            ...
```

### 合并/取消合并

合并将 LoRA 折叠到基础权重中以实现零开销推理：

```python
def merge(self):
    self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling

def unmerge(self):
    self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
```

## 对比：LoRA vs 全量微调 vs 前缀调优

| 方面 | 全量微调 | LoRA | 前缀调优 |
|------|---------|------|---------|
| 可训练参数 | 100% | 0.1-1% | 0.1-1% |
| GPU 显存 | 高 | 中等 | 低 |
| 训练速度 | 慢 | 快 | 快 |
| 推理开销 | 无 | 无（合并后） | 略有 |
| 表达能力 | 完整 | 良好 | 有限 |
| 多任务 | 需要多个副本 | 切换适配器 | 切换前缀 |
| 实现复杂度 | 简单 | 中等 | 中等 |

LoRA 找到了一个甜蜜点：比全量微调内存效率高得多，同时保留了大部分表达能力。与前缀调优不同，它直接修改模型权重，可以合并实现零开销推理。

## 参考文献

- Hu et al. (2021). "LoRA: Low-Rank Adaptation of Large Language Models." arXiv:2106.09685
- Aghajanyan et al. (2020). "Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning." arXiv:2012.13255
