# 近端策略优化 (PPO)

> **模块 04 -- 强化学习，第 02 章**

PPO 是 RLHF（基于人类反馈的强化学习）的核心算法。在奖励模型评估文本质量（第 01 章）之后，PPO 微调语言模型以最大化这些奖励，同时保持接近原始模型。它通过一个巧妙的裁剪目标函数来防止灾难性的策略更新。

本章实现 PPO 的核心组件：裁剪代理损失、广义优势估计 (GAE)、PPO 训练器，以及用于生成响应的 rollout 函数。

---

## 前置知识

- 奖励模型基础（模块 04，第 01 章）
- PyTorch nn.Module、autograd
- 策略梯度和优势函数的基本理解

## 文件说明

| 文件 | 用途 |
|------|------|
| `ppo.py` | 核心实现：PPOClipLoss、GAE、PPOTrainer、rollout |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是 PPO？

### 普通策略梯度的问题

在策略梯度方法中，我们更新策略以增加好动作的概率：

```
gradient = E[advantage * grad(log pi(a|s))]
```

但大幅更新可能破坏策略。如果我们过度增加某个动作的概率，策略可能会过拟合到最近的经验，忘记其他所有内容。

### PPO 的解决方案：裁剪代理

PPO 通过裁剪概率比率来防止大幅更新：

```
ratio = pi_new(a|s) / pi_old(a|s)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
loss = -min(surr1, surr2)
```

当 `ratio > 1+eps`（策略在正方向变化过大）：
- `surr2 = (1+eps) * advantages` 限制了目标
- 梯度无法进一步推动策略

当 `ratio < 1-eps`（策略在负方向变化过大）：
- `surr2 = (1-eps) * advantages` 设定了目标下限
- 梯度无法进一步拉回策略

这创造了一个"信任区域"，策略可以在其中安全地改进。

### KL 惩罚

在 RLHF 中，我们还添加了来自参考模型（通常是预训练模型）的 KL 散度惩罚：

```
loss = policy_loss + kl_weight * KL(pi || pi_ref)
```

这防止策略偏离预训练模型太远，否则输出会变得不连贯。

---

## 架构

### PPOClipLoss

```python
class PPOClipLoss(nn.Module):
    def __init__(self, clip_eps, kl_weight):
        ...

    def forward(self, log_probs, old_log_probs, advantages, log_probs_ref):
        kl = approx_kl_divergence(log_probs, log_probs_ref)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-eps, 1+eps) * advantages
        loss = -min(surr1, surr2) + kl_weight * kl
        return loss, kl
```

### GAE（广义优势估计）

GAE 计算平衡偏差和方差的优势估计：

```python
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)    # TD 误差
A_t = delta_t + (gamma * lambda) * A_{t+1}       # 累积优势
```

- `lambda = 0`：单步 TD（高偏差，低方差）
- `lambda = 1`：蒙特卡洛（低偏差，高方差）
- 典型值：`lambda = 0.95`，`gamma = 0.99`

### PPOTrainer

协调完整的 PPO 更新：

```python
trainer = PPOTrainer(model, ref_model, clip_eps=0.2, kl_weight=0.1)
metrics = trainer.step(sequences, old_log_probs, rewards, values, prompt_len)
```

### rollout

从策略生成响应并收集对数概率：

```python
result = rollout(model, prompt, max_new_tokens=50, vocab_size=32000)
# result["sequences"], result["log_probs"], result["values"]
```

---

## 代码详解

### 步骤 1：PPO 裁剪损失

裁剪损失是 PPO 的核心。它计算两个代理目标并取最小值（悲观界）：

```python
ratio = (log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages
loss = -torch.min(surr1, surr2) + kl_weight * kl
```

### 步骤 2：GAE

GAE 通过反向累积 TD 误差来计算优势：

```python
for t in reversed(range(seq_len)):
    advantage = delta_t + gamma * lambda * advantage
    advantages[:, t] = advantage
```

### 步骤 3：Rollout

自回归生成 token，收集对数概率和价值估计：

```python
for _ in range(max_new_tokens):
    logits, values = model(sequences)
    next_token = sample(logits[:, -1, :] / temperature)
    sequences = cat([sequences, next_token])
```

### 步骤 4：PPO 更新

训练器执行多个 epoch 的 PPO 更新：

```python
for epoch in range(ppo_epochs):
    new_log_probs = compute_log_probs(model, sequences)
    ref_log_probs = compute_log_probs(ref_model, sequences)
    loss, kl = ppo_loss(new_log_probs, old_log_probs, advantages, ref_log_probs)
    loss.backward()
    optimizer.step()
```

---

## 训练技巧

### 超参数

| 参数 | 典型范围 | 描述 |
|------|----------|------|
| `clip_eps` | 0.1 - 0.3 | 裁剪参数（0.2 常用） |
| `kl_weight` | 0.01 - 0.5 | KL 惩罚权重 |
| `gamma` | 0.99 | 折扣因子 |
| `lam` | 0.95 | GAE lambda |
| `lr` | 1e-6 - 1e-5 | 学习率 |
| `ppo_epochs` | 2 - 4 | 每批数据的 PPO 更新 epoch 数 |

### 常见陷阱

1. **奖励黑客**：策略找到获得高分的方式，但实际上并没有更好。使用 KL 惩罚和多样化的奖励信号。

2. **KL 散度爆炸**：如果 `kl_weight` 太低，策略可能偏离参考模型，产生不连贯的文本。监控 KL 并在需要时增加 `kl_weight`。

3. **价值函数准确性**：如果价值函数不准确，优势估计会有噪声，导致训练不稳定。考虑预训练价值函数。

4. **学习率敏感性**：PPO 对学习率敏感。太高导致不稳定；太低导致收敛缓慢。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/02_ppo/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/02_ppo/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现 PPO 组件。

### 练习顺序

1. **`PPOClipLossExercise.forward`** -- 实现裁剪代理目标
2. **`GAEExercise.forward`** -- 实现广义优势估计
3. **`rollout_exercise`** -- 实现自回归生成和对数概率收集

### 提示

- 从 `PPOClipLoss` 开始。关键公式是：`-min(ratio * A, clip(ratio) * A)`。
- 对于 `GAE`，从最后一个时间步反向工作。递推公式是：`A_t = delta_t + gamma * lambda * A_{t+1}`。
- 对于 `rollout`，使用 `torch.multinomial` 进行采样，使用 `log_softmax` + `gather` 计算对数概率。
- 生成时使用 `model.eval()` 来禁用 dropout。

---

## 核心要点

1. **PPO 裁剪策略更新。** 通过取 `min(surr1, surr2)`，PPO 确保策略在单次更新中不会变化太大。这是 PPO 稳定性的关键洞察。

2. **GAE 平衡偏差和方差。** Lambda 参数控制单步 TD（有偏差但低方差）和蒙特卡洛（无偏差但高方差）之间的权衡。Lambda=0.95 是一个好的默认值。

3. **KL 惩罚防止发散。** 在 RLHF 中，我们添加来自参考模型的 KL 惩罚，以保持策略接近预训练模型。这防止了奖励黑客并维持文本质量。

4. **PPO 是迭代算法。** 每批经验用于多个 PPO 更新 epoch。这种数据效率很重要，因为生成 rollout 对语言模型来说是昂贵的。

5. **比率是关键量。** `ratio = pi_new / pi_old` 衡量策略变化了多少。PPO 的裁剪确保这个比率保持接近 1。

---

## 扩展阅读

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) -- PPO 原始论文
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT，推广了 PPO 在 RLHF 中的应用
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) -- GAE 论文
- [Some Things You Should Know About Proximal Policy Optimization (Huang et al., 2022)](https://arxiv.org/abs/2209.00796) -- PPO 实用技巧
