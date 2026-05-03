# 训练循环

> **模块 02 -- 预训练，第 03 章**

训练循环是预训练的引擎。它反复将批次数据输入模型，计算损失，更新参数，并跟踪指标。一个设计良好的循环包括：先预热后衰减的学习率调度、防止不稳定的梯度裁剪、模拟大批量的梯度累积，以及最大化吞吐量的混合精度训练。本章从零开始构建每个组件。

---

## 前置知识

- 基础 PyTorch（`nn.Module`、`torch.optim`、自动求导）
- 理解反向传播和梯度下降
- 熟悉数据管道（第 01 章）

## 文件说明

| 文件 | 用途 |
|------|------|
| `training_loop.py` | 核心实现：调度器、裁剪器、指标、训练步骤 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | 正确性 pytest 测试 |

---

## 训练循环组件

一个训练步骤有四个阶段：

```
1. 前向传播：   logits = model(input_ids)
2. 计算损失：   loss = cross_entropy(logits, labels)
3. 反向传播：   loss.backward()
4. 参数更新：
   a. 裁剪梯度
   b. optimizer.step()
   c. scheduler.step()  -- 调整学习率
   d. optimizer.zero_grad()
```

每个阶段在大规模训练中都有其微妙之处。

---

## 余弦学习率与预热

学习率可能是最重要的超参数。太高会导致训练发散；太低则训练停滞。LLM 预训练的标准调度策略是**线性预热后余弦衰减**。

### 为什么需要预热？

在训练开始时，模型的参数是随机的。早期步骤中的大梯度可能将模型推入损失景观中的不良区域，使其无法逃脱。预热以极小的学习率开始，在前 N 步中线性增加，让优化器在进行大幅更新之前找到稳定的方向。

```
学习率
 ^
 |        /\
 |       /  \
 |      /    \___
 |     /         \___
 |    /              \___
 |   /                   \___
 +--+-----+-----+-----+-----> 步数
   0    预热   50%    100%
```

### 余弦衰减 vs. 线性衰减

预热后，学习率开始衰减。两种常见策略：

- **线性衰减**：学习率从峰值线性下降到零。简单但可能在早期衰减过快。
- **余弦衰减**：学习率遵循余弦曲线，初期衰减缓慢，后期加速。这通常能获得更好的最终性能，因为模型在中等学习率下花费更多时间来精炼其表征。

公式：

```
scale = min_lr_ratio + 0.5 * (1 - min_lr_ratio) * (1 + cos(pi * progress))
```

其中 `progress` 从 0（衰减开始）到 1（训练结束）。`min_lr_ratio`（通常为 0.1）防止学习率降为零，这有助于后续的微调。

```python
class CosineLRScheduler:
    def get_lr_scale(self, step):
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))
```

### 实际取值

- 预热步数：通常为总步数的 1-5%
- 峰值学习率：3e-4 是 100M-1B 参数模型的常见默认值
- min_lr_ratio：0.1（学习率衰减到峰值的 10%）

---

## 梯度裁剪

在训练过程中，梯度偶尔会因异常批次或数值不稳定性而出现尖峰。如果梯度范数变得非常大，参数更新可能是灾难性的——模型会跳到损失景观的完全不同的区域，训练会崩溃。

### 工作原理

梯度裁剪在全局范数超过阈值时重新缩放梯度：

```
如果 ||g|| > max_norm：
    g = g * (max_norm / ||g||)
```

这保留了梯度的方向但限制了其大小。`max_norm` 通常为 1.0。

### 范数裁剪 vs. 值裁剪

- **范数裁剪**（我们使用的）：基于所有梯度的 L2 范数进行裁剪。保留不同参数梯度的相对大小。
- **值裁剪**：独立裁剪每个梯度元素。可能会扭曲梯度方向。在现代训练中很少使用。

```python
class GradientClipper:
    def clip(self, model):
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, norm_type=self.norm_type
        )
        return total_norm.item()
```

返回的范数可用于监控。如果它持续接近 `max_norm`，说明模型正在经历不稳定，你可能需要降低学习率。

---

## 梯度累积

大批量可以稳定训练并提高吞吐量，但需要更多 GPU 内存。梯度累积通过在多次前向/反向传播中累积梯度，然后才更新参数，来模拟更大的批量。

```
有效批量大小 = 微批量大小 * 梯度累积步数

示例：
  微批量大小 = 4
  梯度累积步数 = 8
  有效批量大小 = 32
```

每个微批量进行一次前向和反向传播，累加到累积梯度中。只有每 N 个微批量后才执行优化器步骤。损失必须除以累积步数，以使累积梯度具有正确的大小。

```python
loss = loss / gradient_accumulation_steps
loss.backward()

if (step + 1) % gradient_accumulation_steps == 0:
    clipper.clip(model)
    optimizer.step()
    optimizer.zero_grad()
```

---

## 混合精度训练

现代 GPU 可以同时以 FP32（32 位浮点）和 FP16/BF16（16 位浮点）进行计算。混合精度在前向和反向传播中使用 16 位（更快、更省内存），同时保留 FP32 的参数副本用于更新（数值更稳定）。

### FP16 vs. BF16

- **FP16**：范围较小，梯度可能下溢为零。需要损失缩放来防止这种情况。
- **BF16**：与 FP32 具有相同的范围但精度较低。不需要损失缩放。在现代硬件（A100、H100）上是首选。

### GradScaler

`GradScaler` 是 FP16 训练所必需的。它在反向传播前将损失乘以一个大因子（缩放梯度以防止下溢），然后在优化器步骤前将梯度缩放回来。

```python
scaler = torch.amp.GradScaler()

with torch.amp.autocast(device_type="cuda"):
    logits = model(input_ids)
    loss = cross_entropy(logits, labels)

scaler.scale(loss).backward()
scaler.unscale_(optimizer)
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
scaler.step(optimizer)
scaler.update()
```

`unscale_` 调用必须在梯度裁剪之前进行，因为裁剪操作的是未缩放的梯度。

---

## 权重衰减

权重衰减通过在每步中从梯度中减去权重的一部分来惩罚大权重。这起到了正则化的作用，防止模型过度依赖任何单一特征。

### 解耦 vs. L2 正则化

- **L2 正则化**：在损失中添加 `0.5 * lambda * ||w||^2`。该项的梯度为 `lambda * w`，加到损失的梯度上。这与学习率相互作用——改变学习率也会改变有效正则化强度。
- **解耦权重衰减**（AdamW）：在优化器步骤后直接从参数中减去 `lambda * w`。这将正则化与学习率解耦，使超参数调优更容易。

AdamW 是 LLM 预训练的标准优化器。

### 哪些参数需要权重衰减？

常见做法：对权重矩阵（维度 >= 2）应用权重衰减，但对偏置和归一化参数（维度 < 2）不应用。这是因为偏置和归一化参数在网络中的作用已经受到约束，衰减它们可能会损害性能。

```python
decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

optimizer = AdamW([
    {"params": decay_params, "weight_decay": 0.1},
    {"params": no_decay_params, "weight_decay": 0.0},
], lr=3e-4, betas=(0.9, 0.95))
```

---

## 代码演练

### 第 1 步：CosineLRScheduler

```python
scheduler = CosineLRScheduler(optimizer, warmup_steps=100, total_steps=10000)
```

创建一个调度器，在 100 步内线性预热，然后在剩余的 9900 步中以余弦曲线衰减。`get_lr_scale` 方法返回一个 0 到 1 之间的乘数，应用于基础学习率。

### 第 2 步：GradientClipper

```python
clipper = GradientClipper(max_norm=1.0)
```

将 `torch.nn.utils.clip_grad_norm_` 封装在一个简单的接口中。`clip` 方法返回裁剪前的梯度范数，可用于日志记录。

### 第 3 步：TrainingMetrics

```python
metrics = TrainingMetrics()
metrics.update(loss=2.5, grad_norm=1.0, lr=1e-3)
print(metrics.summary())
```

随时间跟踪损失、梯度范数和学习率。`summary` 方法返回包含运行平均值和经过时间的字典。

### 第 4 步：create_training_components

```python
components = create_training_components(model, learning_rate=3e-4, total_steps=10000)
```

一次调用创建所有训练组件。返回包含 `optimizer`、`scheduler`、`clipper` 和 `metrics` 的字典。自动处理权重衰减参数分组。

### 第 5 步：training_step

```python
result = training_step(
    model, batch, optimizer, scheduler, clipper, step=0,
    gradient_accumulation_steps=1,
)
```

执行单个训练步骤：前向传播、损失计算、反向传播、梯度裁剪、优化器步骤和学习率更新。支持梯度累积和通过 `GradScaler` 的可选混合精度。

---

## 整合在一起

典型的训练循环：

```python
components = create_training_components(model, total_steps=len(dataloader) * num_epochs)
optimizer = components["optimizer"]
scheduler = components["scheduler"]
clipper = components["clipper"]
metrics = components["metrics"]

for epoch in range(num_epochs):
    for step, batch in enumerate(dataloader):
        result = training_step(model, batch, optimizer, scheduler, clipper, step)
        metrics.update(result["loss"], result["grad_norm"], result["lr"])

        if step % 100 == 0:
            print(metrics.summary())
            metrics.reset()
```

---

## 练习

打开 `exercise.py` 并实现：

1. **`CosineLRScheduler.get_lr_scale`**：实现预热 + 余弦衰减公式。
2. **`GradientClipper.clip`**：使用 `clip_grad_norm_` 实现梯度裁剪。
3. **`training_step`**：实现完整的前向/反向/更新逻辑。

---

## 运行测试

```bash
pytest tests.py -v
```

---

## 总结

| 概念 | 解决的问题 |
|------|-----------|
| 余弦学习率 + 预热 | 稳定启动，渐进衰减以获得更好的收敛 |
| 梯度裁剪 | 防止梯度尖峰导致的灾难性更新 |
| 梯度累积 | 用有限的 GPU 内存模拟大批量 |
| 混合精度 | 更快的训练，更少的内存，相同的质量 |
| AdamW 权重衰减 | 正则化与学习率解耦 |
| TrainingMetrics | 监控损失、梯度范数、学习率以便调试 |

---

## 下一步

- **第 04 章（分布式训练）**：扩展到多 GPU 和多机训练。
- **第 05 章（缩放定律）**：模型大小、数据和计算之间的关系。
