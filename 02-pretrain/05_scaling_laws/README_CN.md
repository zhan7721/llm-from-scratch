# Scaling Laws

Scaling laws (缩放定律) 描述了语言模型性能（以测试损失衡量）如何随着模型规模、
训练数据或计算量的增加而可预测地改善。它们使你能够在投入大量计算资源之前
**预测**昂贵训练运行的结果。

## 为什么缩放定律很重要

在缩放定律被发现之前，选择模型大小和训练时长主要依靠经验猜测。Kaplan 等人
(2020) 发现，损失在多个数量级上遵循平滑的幂律关系。这意味着：

- 你可以训练小模型，拟合曲线，然后预测大模型的性能。
- 你可以在固定计算预算下决定如何分配参数和数据。
- 你可以在训练开始之前设定合理的预期。

## 幂律形式

Kaplan 等人观察到交叉熵损失遵循以下形式：

```
L(x) = a * x^(-alpha) + L_inf
```

其中：
- `x` 是缩放变量（参数量 N 或 token 数 D）。
- `a` 是取决于任务的常数。
- `alpha` 是缩放指数（损失随规模下降的速度）。
- `L_inf` 是不可约损失下界（自然语言约 1.5 nats）。

关键洞察：将 `log(L - L_inf)` 对 `log(x)` 作图会得到一条直线。这使得拟合
和外推变得简单直接。

### 实际拟合方法

```python
import numpy as np

# 观测数据：(参数量, 损失)
x = np.array([1e6, 1e7, 1e8, 1e9])
loss = np.array([3.5, 3.0, 2.7, 2.5])
l_inf = 1.5

adjusted = loss - l_inf
log_x = np.log(x)
log_y = np.log(adjusted)

coeffs = np.polyfit(log_x, log_y, 1)
alpha = -coeffs[0]
a = np.exp(coeffs[1])
```

## 计算量估算

训练计算量近似为：

```
C ≈ 6 * N * D
```

其中：
- N = 参数数量
- D = 训练 token 数量
- 系数 6 考虑了前向传播（2）和反向传播（4）。

这是一个简化。实际 FLOPs 取决于架构细节，但这个近似被广泛用于规划。

## Kaplan 等人 (2020)

这篇原始的 OpenAI 论文发现了三条独立的幂律：

1. **L(N)**：损失 vs. 参数量（数据按比例缩放）。
2. **L(D)**：损失 vs. 数据集大小（模型按比例缩放）。
3. **L(C)**：损失 vs. 计算预算（最优分配）。

主要发现：
- 性能随着每个因素的增加而平滑提升。
- **N 比 D 更重要**：Kaplan 建议将更多计算分配给更大的模型，而不是更多
  训练数据。
- 最优缩放：N_opt ∝ C^0.73，D_opt ∝ C^0.27。

这导致了用相对有限的数据集训练超大模型的策略（GPT-3 的方法）。

## Chinchilla (Hoffmann 等人 2022)

DeepMind 的 Chinchilla 论文用更仔细的实验重新审视了缩放定律，得出了不同
的结论：

- **数据比 Kaplan 认为的更重要。**
- 最优比率大约是 **D ≈ 20 * N**（每个参数 20 个 token）。
- 在固定计算预算下，较小的模型用更多数据训练会优于较大的模型用较少
  token 训练。

### Chinchilla 分配

给定计算预算 C，最优分配为：

```
C = 6 * N * D     (计算约束)
D = 20 * N        (Chinchilla 比率)

求解：N = sqrt(C / (6 * 20))
      D = 20 * N
```

示例：使用 1e24 FLOPs：
- N_opt ≈ 9.1B 参数
- D_opt ≈ 183B token

### 影响

Chinchilla 改变了行业实践。LLaMA 等模型在训练时考虑了 Chinchilla 比率，
使用了远多于 GPT-3 相对于其大小的 token 数。结果是：更小的模型匹配或超越
了更大的前辈。

## 实际应用

### 规划训练运行

1. **估算你的计算预算**（GPU 数量 * GPU FLOPS * 时间）。
2. **使用 Chinchilla 分配**找到最优的 N 和 D。
3. **从小规模试验运行拟合缩放定律**来预测最终损失。
4. **决定是否值得**：将预测损失与目标进行比较。

### 示例：规划 7B 模型

```
目标：7B 参数
Chinchilla 比率：D = 20 * 7B = 140B token
计算量：C = 6 * 7e9 * 140e9 = 5.88e21 FLOPs
```

这大约是 5.88 zettaFLOPs，或在 A100 上约 800 GPU 天。

### 设定预期

如果你已经在 100M、300M 和 1B 参数的模型上训练过，你可以拟合缩放定律
并预测 7B 模型将实现什么。这帮助你：

- 决定是否要扩大规模。
- 选择是在更大的模型还是更多的数据上花费计算。
- 在训练之前向利益相关者报告预期性能。

## 局限性

### 涌现能力

某些能力会在特定规模突然出现（上下文学习、链式思维推理）。缩放定律预测
的是**平均损失**，而非特定能力。一个模型可以有更低的损失，但仍然在需要
最低规模的任务上失败。

### 收益递减

幂律指数 alpha 对于损失 vs. 计算通常是 0.05--0.1。这意味着你需要大约
**10 倍的计算量**来将损失降低 5--10%。超过某个点后，成本可能无法证明
改进的合理性。

### 架构依赖

缩放定律是针对特定架构测量的。Transformer 的缩放定律不直接适用于 Mamba、
混合专家或其他架构。每个模型家族需要自己的测量。

### 训练细节很重要

超参数调优、数据质量和训练稳定性都会影响实际损失。缩放定律给你一个
**上限**（使用最优设置可达到的），而不是保证。

## 代码详解

### `ScalingLaw` 类

通过 log 线性回归将幂律模型 `L(x) = a * x^(-alpha) + L_inf` 拟合到
数据点。拟合后，`predict(x)` 可以外推到新的规模。

### `estimate_compute`

简单公式：`C = 6 * N * D`。适用于快速规划。

### `optimal_allocation_chinchilla`

求解方程组 `C = 6ND` 和 `D = 20N`，找到给定预算下的 Chinchilla 最优
参数量和 token 数。

### `optimal_allocation_kaplan`

实现 Kaplan 的分配方案，其中 N 按 C^0.73 缩放，D 按 C^0.27 缩放。与
Chinchilla 相比，这倾向于更大的模型和更少的数据。

### `compare_allocations`

在同一计算预算下并排比较两种策略。

## 运行测试

```bash
cd 02-pretrain/05_scaling_laws
pytest tests.py -v
```

## 练习

打开 `exercise.py` 并实现 TODO 项目：

1. `ScalingLaw.fit` —— 线性化并拟合幂律。
2. `estimate_compute` —— 实现 FLOPs 公式。
3. `optimal_allocation_chinchilla` —— 求解最优 N 和 D。

使用 `tests.py` 检查你的实现。

## 参考文献

- Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models."
  arXiv:2001.08361.
- Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language
  Models" (Chinchilla). arXiv:2203.15556.
- Touvron, H. et al. (2023). "LLaMA: Open and Efficient Foundation
  Language Models." arXiv:2302.13971.
