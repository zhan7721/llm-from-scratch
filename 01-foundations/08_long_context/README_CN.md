# 长上下文：RoPE 缩放与 YaRN

## 问题：固定上下文长度

大多数大语言模型使用**固定的上下文长度**进行训练——例如，LLaMA 2 在 4096 个 token 上训练，GPT-4 在 8192 或 32K 个 token 上训练。这意味着：

- 模型从未"见过"超过训练长度的位置
- 未见位置的注意力模式变得不可预测
- 盲目扩展会导致质量严重下降

但如果我们需要处理 50 页文档，或者维持长对话怎么办？我们需要**扩展上下文窗口**的技术，而无需从头重新训练模型。

## 背景：旋转位置编码（RoPE）

在理解上下文扩展之前，让我们回顾一下 RoPE 的工作原理（在第 02 章——嵌入中介绍）。

RoPE 通过在注意力中**旋转**查询和键向量来编码位置信息。对于位置 `m` 和维度对 `i`，旋转角度为：

```
theta_i = m * base^(-2i/d_model)
```

其中 `base` 通常为 10000。旋转应用如下：

```
[q_2i, q_2i+1] -> [q_2i * cos(m*theta_i) - q_2i+1 * sin(m*theta_i),
                     q_2i * sin(m*theta_i) + q_2i+1 * cos(m*theta_i)]
```

关键洞察：**每个维度对以不同的频率旋转**。低维度旋转快（编码局部上下文），而高维度旋转慢（编码全局位置）。

## 方法 1：位置插值（线性缩放）

扩展上下文最简单的方法：**压缩位置**使其适应原始训练范围。

### 工作原理

如果模型在位置 `[0, 4096)` 上训练，而我们想处理最大 8192 的位置，我们将所有位置除以 2：

```
position_scaled = position / scale_factor
```

对于 scale_factor = 2.0：
- 原始位置 0 -> 缩放后 0.0
- 原始位置 4096 -> 缩放后 2048.0
- 原始位置 8192 -> 缩放后 4096.0

现在位置 8192 映射到训练时位置 4096 的编码。

### 实现

```python
class ScaledRoPE(nn.Module):
    def __init__(self, d_model, base=10000.0, scale_factor=1.0):
        super().__init__()
        self.scale_factor = scale_factor
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x):
        seq_len = x.shape[1]
        # 缩放位置
        t = torch.arange(seq_len, device=x.device) / self.scale_factor
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        return x * cos + self._rotate_half(x) * sin
```

### 权衡

**优点：**
- 实现简单
- 适用于适度扩展（2x-4x）
- 在 LLaMA 2 Long 中使用效果良好

**缺点：**
- 均等压缩所有频率，包括编码局部上下文的高频分量
- 可能失去相邻 token 的细粒度位置区分
- 需要一些微调来恢复质量

## 方法 2：NTK 感知缩放

NTK 感知缩放不是缩放位置，而是**调整基频率**。这保留了高频分量，同时扩展了低频范围。

### 洞察

回顾每个维度对有不同的频率：

```
theta_i = base^(-2i/d_model)
```

- **低维度**（小 i）：高频，编码局部上下文
- **高维度**（大 i）：低频，编码全局位置

线性缩放压缩了所有内容。但我们只需要扩展**低频**维度（全局位置）。高频维度（局部上下文）应该保持不变！

### 公式

NTK 感知缩放修改基频率：

```
base_scaled = base * (scale_factor ^ (d_model / (d_model - 2)))
```

这产生的效果是：
- 高频维度基本不变
- 低频维度按比例因子扩展

### 为什么叫 "NTK"？

这个名字来自神经切线核理论。关键洞察是位置编码的不同频率分量具有不同的"可学习性"特征。高频分量需要保留以理解局部上下文。

## 方法 3：YaRN（Yet another RoPE extensioN）

YaRN 结合了 NTK 感知缩放和**注意力温度缩放**，效果更好。

### 组成部分

1. **NTK 感知缩放**（如上所述）：调整基频率
2. **注意力温度**：缩放注意力 logits 以补偿频率分布的变化

```python
# 注意力温度因子
attn_factor = 1 / sqrt(scale_factor)
```

### 为什么需要温度缩放？

在 NTK 感知缩放之后，频率分布发生变化。注意力分数（Q 和 K 的点积）可能变得过大，导致：
- 过度自信的注意力模式
- 注意力权重多样性降低
- 潜在的数值不稳定

温度因子 `1/sqrt(scale_factor)` 将注意力分数恢复到合理范围。

### 实现

```python
class YaRNRope(nn.Module):
    def __init__(self, d_model, base=10000.0, scale_factor=4.0):
        super().__init__()
        # NTK 感知基频率缩放
        base_scaled = base * (scale_factor ** (d_model / (d_model - 2)))
        inv_freq = 1.0 / (base_scaled ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer("inv_freq", inv_freq)

        # 注意力温度
        self.attn_factor = 1.0 / math.sqrt(scale_factor)

    def forward(self, x):
        seq_len = x.shape[1]
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        freqs = torch.outer(t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos, sin = emb.cos().unsqueeze(0), emb.sin().unsqueeze(0)
        # 应用旋转和温度缩放
        return (x * cos + self._rotate_half(x) * sin) * self.attn_factor
```

## 线性缩放 vs YaRN：何时使用哪种

| 特性 | 线性缩放 | YaRN |
|------|---------|------|
| **复杂度** | 简单 | 中等 |
| **扩展倍数** | 2x-4x | 4x-16x+ |
| **局部上下文保留** | 差 | 好 |
| **需要微调** | 是（更多） | 是（更少） |
| **实现** | 缩放位置 | 缩放基频率 + 温度 |

**使用线性缩放的场景：**
- 需要快速扩展（2x）
- 计划进行大量微调
- 偏好简单性

**使用 YaRN 的场景：**
- 需要更大扩展（4x+）
- 想要保留局部上下文质量
- 微调预算有限

## 代码详解

### 步骤 1：理解频率谱

```python
d_model = 64
base = 10000.0
inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
print(f"频率范围: {inv_freq[-1]:.6f} 到 {inv_freq[0]:.6f}")
# 输出: 频率范围: 0.000100 到 1.000000
# 低索引 = 高频（快速旋转）
# 高索引 = 低频（慢速旋转）
```

### 步骤 2：应用线性缩放

```python
rope = ScaledRoPE(d_model=64, scale_factor=2.0)
x = torch.randn(1, 100, 64)  # 100 个 token
out = rope(x)  # 位置被减半: 0, 0.5, 1.0, ..., 49.5
```

### 步骤 3：应用 YaRN

```python
yarn = YaRNRope(d_model=64, scale_factor=4.0)
x = torch.randn(1, 400, 64)  # 400 个 token
out = yarn(x)  # NTK 感知缩放 + 温度
```

### 步骤 4：比较输出

```python
x = torch.randn(1, 50, 64)
rope_std = ScaledRoPE(d_model=64, scale_factor=1.0)  # 标准 RoPE
rope_scaled = ScaledRoPE(d_model=64, scale_factor=2.0)
yarn = YaRNRope(d_model=64, scale_factor=4.0)

out_std = rope_std(x)
out_scaled = rope_scaled(x)
out_yarn = yarn(x)

# 不同方法产生不同的编码
print(f"标准 vs 缩放: {(out_std - out_scaled).abs().mean():.4f}")
print(f"标准 vs YaRN: {(out_std - out_yarn).abs().mean():.4f}")
```

## 实用技巧

### 1. 缩放后微调

上下文扩展技术配合微调效果最佳：

```python
# 典型微调设置
model = load_pretrained_model("llama-2-7b")
model.rope = YaRNRope(d_model=4096, scale_factor=4.0)

# 在长序列上微调
for batch in long_sequence_dataloader:
    loss = model(batch)
    loss.backward()
    optimizer.step()
```

### 2. 渐进式扩展

对于非常大的扩展（例如 4K -> 128K），考虑渐进步骤：

```
4K -> 16K (4x) -> 64K (4x) -> 128K (2x)
```

每一步需要的微调比单次大跳跃少。

### 3. 评估

始终评估：
- 各种序列长度的**困惑度**
- **检索任务**：模型能否找到位置 50K 的信息？
- **生成质量**：长上下文生成是否保持连贯？

### 4. 内存考虑

更长的上下文意味着 KV 缓存需要更多内存：

```
内存 = 2 * n_layers * n_heads * seq_len * d_k * sizeof(float16)
```

对于 70B 模型在 128K 上下文下，这可能需要 40+ GB。

## 常见陷阱

1. **过度缩放**：不微调就从 4K 跳到 256K 会产生垃圾输出
2. **忽略注意力温度**：没有温度缩放的 YaRN 可能产生过度自信的注意力
3. **评估不当**：困惑度不能单独反映检索质量
4. **忘记 KV 缓存内存**：长上下文需要成比例的内存

## 核心要点

1. **上下文扩展对实际 LLM 应用至关重要**
2. **线性缩放**简单但会损失局部上下文质量
3. **NTK 感知缩放**通过调整基频率保留高频分量
4. **YaRN** 结合 NTK 缩放和注意力温度，效果最佳
5. **微调**在上下文扩展后几乎总是需要的
6. **渐进式扩展**比大跳跃更好

## 下一步

- **RoPE 变体**：ALiBi、Kerple、FIRE
- **高效注意力**：Flash Attention、Ring Attention 用于超长序列
- **稀疏注意力**：Longformer、BigBird 用于混合局部/全局模式
- **检索增强**：RETRO、Infini-attention 用于真正的无限上下文

## 参考文献

- [Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — 位置插值
- [YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — YaRN
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — 原始 RoPE
- [LLaMA 2 Long](https://arxiv.org/abs/2307.09288) — 实践中的缩放 RoPE
