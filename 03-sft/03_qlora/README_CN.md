# QLoRA：量化低秩适应

## 动机

虽然LoRA显著减少了可训练参数的数量，但仍需要以FP16精度加载完整模型。对于65B参数的模型，仅权重就需要约130GB的GPU内存，使得在消费级硬件上进行微调变得不可能。

**QLoRA**（Dettmers等人，2023）通过结合以下技术解决了这个问题：
1. **4位NormalFloat（NF4）量化**用于基础权重
2. **LoRA适配器**以全精度进行训练
3. **双重量化**进一步压缩量化常数
4. **分页优化器**处理内存峰值

这使得在单个48GB GPU上微调65B模型成为可能，同时保持完整的16位微调性能。

## NF4量化

### 为什么使用NF4？

标准量化方法（INT4、INT8）假设值均匀分布。然而，神经网络权重近似**正态分布**。NF4对正态分布是信息论最优的。

### 量化级别

16个NF4级别是预先计算的，以最小化标准正态数据的期望量化误差：

```
NF4_LEVELS = [-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
              0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0]
```

这些级别的间距使得在标准正态分布下每个量化箱包含大致相同数量的值。

### 分块量化

为了处理异常值和张量中不同尺度的变化，我们使用**分块量化**：

1. 将张量分成大小为B的块（默认：64）
2. 对每个块，计算绝对最大值作为缩放因子
3. 将块归一化到[-1, 1]
4. 将每个归一化值量化到最近的NF4级别
5. 存储4位索引和每块的缩放因子

```
原始值：[0.5, -0.3, 0.8, -0.1, ...] (float32)
缩放因子：0.8（块的绝对最大值）
归一化：[0.625, -0.375, 1.0, -0.125, ...]
NF4索引：[12, 3, 15, 5, ...]（每个4位）
```

### 内存节省

- **FP32**：每个权重32位
- **FP16**：每个权重16位
- **NF4**：每个权重4位 + 约0.5位用于缩放因子（block_size=64时）

对于7B参数模型：
- FP32：28 GB
- FP16：14 GB
- NF4：约3.5 GB + 0.05 GB缩放因子 ≈ 3.6 GB

## QLoRA架构

QLoRA结合了NF4量化和LoRA：

```
                    输入 x
                       │
           ┌───────────┴───────────┐
           │                       │
           ▼                       ▼
    ┌─────────────┐         ┌─────────────┐
    │  NF4权重    │         │  LoRA路径   │
    │  （冻结）   │         │ （可训练）  │
    └─────────────┘         └─────────────┘
           │                       │
           ▼                       ▼
    反量化到FP16        x → Dropout → A → B
           │                       │
           ▼                       ▼
      W_q @ x                B @ A @ x × (α/r)
           │                       │
           └───────────┬───────────┘
                       │
                       ▼
                  输出 + LoRA
```

### 关键特性

1. **基础权重**：以NF4存储，训练期间冻结
2. **LoRA矩阵**：以FP16/BF16存储，可训练
3. **前向传播**：即时反量化基础权重，添加LoRA贡献
4. **反向传播**：梯度仅通过LoRA矩阵流动

### 内存分解

对于LoRA rank为64的7B模型：
- 基础权重（NF4）：约3.6 GB
- LoRA矩阵（FP16）：约0.1 GB
- 优化器状态（Adam）：约0.2 GB
- 激活值：约2-4 GB
- **总计**：约6-8 GB（适合单GPU）

## 双重量化

QLoRA引入了**双重量化**来进一步压缩量化常数：

1. **第一次量化**：权重 → NF4，block_size=64
2. **第二次量化**：将缩放因子本身量化为FP8，block_size=256

这将缩放因子的内存开销从0.5位/权重降低到约0.127位/权重。

```
缩放因子（FP32）：每个权重0.5位
缩放因子（FP8）：每个权重0.127位
节省：约0.37位/权重
```

## 分页优化器

在训练期间，优化器状态可能导致内存峰值。QLoRA使用**分页优化器**，当GPU内存不足时自动将优化器状态移至CPU内存，并在需要时将其移回。

这是使用NVIDIA的统一内存功能实现的，该功能提供跨越CPU和GPU内存的单一地址空间。

## 代码走读

### NF4Quantizer

```python
class NF4Quantizer:
    NF4_LEVELS = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0,
    ])

    def quantize(self, tensor):
        # 1. 重塑为块
        # 2. 计算每块的绝对最大值缩放因子
        # 3. 归一化到[-1, 1]
        # 4. 找到最近的NF4级别
        # 5. 将两个4位索引打包成一个int8
        return packed, scales

    def dequantize(self, packed, scales, original_shape):
        # 1. 解包4位索引
        # 2. 查找NF4级别
        # 3. 乘以缩放因子
        # 4. 重塑为原始形状
        return dequantized_tensor
```

### QLoRALinear

```python
class QLoRALinear(nn.Module):
    def __init__(self, in_features, out_features, rank=8, alpha=16.0):
        self.quant_linear = QuantizedLinear(in_features, out_features)
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        self.scaling = alpha / rank

    def forward(self, x):
        base_output = self.quant_linear(x)  # 即时反量化
        lora_output = (x @ self.lora_A.T @ self.lora_B.T) * self.scaling
        return base_output + lora_output
```

### apply_qlora

```python
def apply_qlora(model, rank=8, alpha=16.0, target_modules=None):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and target in name:
            # 1. 创建QLoRALinear
            # 2. 复制权重
            # 3. 量化基础权重
            # 4. 替换模块
```

## 内存节省对比

| 模型 | FP32 | FP16 | NF4 | NF4 + LoRA (r=64) |
|------|------|------|-----|-------------------|
| 7B | 28 GB | 14 GB | 3.6 GB | ~6 GB |
| 13B | 52 GB | 26 GB | 6.5 GB | ~10 GB |
| 33B | 132 GB | 66 GB | 16.5 GB | ~22 GB |
| 65B | 260 GB | 130 GB | 32.5 GB | ~42 GB |

## 参考文献

- Dettmers, T., Pagnoni, A., Holtzman, A., & Zettlemoyer, L. (2023). QLoRA: Efficient Finetuning of Quantized Large Language Models. arXiv:2305.14314.
- Hu, E. J., et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.
