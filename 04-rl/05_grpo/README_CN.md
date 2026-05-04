# 群体相对策略优化 (GRPO)

> **模块 04 -- 强化学习，第 05 章**

GRPO 是 RLHF 中 PPO 的一种更简单的替代方案。GRPO 不需要训练价值网络来估计优势，而是为每个提示生成多个响应，并在每个组内对奖励进行归一化。这完全消除了对价值函数的需求，使算法更简单且通常更稳定。

本章实现 GRPO 的核心组件：群体相对优势计算、裁剪代理损失、GRPO 训练器，以及用于单次训练步骤的便捷函数。

---

## 前置知识

- 奖励模型基础（模块 04，第 01 章）
- PPO 概念（模块 04，第 02 章）
- PyTorch nn.Module、autograd

## 文件说明

| 文件 | 用途 |
|------|------|
| `grpo.py` | 核心实现：compute_group_advantages、GRPOLoss、GRPOTrainer、grpo_step |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是 GRPO？

### PPO 用于 RLHF 的问题

PPO 需要一个价值网络（评论家）来估计优势。这增加了复杂性：
- 需要同时训练策略网络和价值网络
- 价值网络可能不准确，导致优势估计有噪声
- 更多参数、更多计算、更多需要调优的超参数

### GRPO 的解决方案：群体相对优势

GRPO 不学习价值函数，而是为每个提示生成 G 个响应，并使用组内的经验奖励分布：

```
对于每个提示：
    1. 生成 G 个响应：y_1, y_2, ..., y_G
    2. 对每个响应评分：r_1, r_2, ..., r_G
    3. 计算群体优势：
       advantage_i = (r_i - mean(r)) / (std(r) + eps)
```

关键特性：
- 优势在每组内求和为零（构造保证）
- 较高奖励的响应获得正优势（被强化）
- 较低奖励的响应获得负优势（被抑制）
- 不需要价值网络！

---

## 架构

### compute_group_advantages

```python
def compute_group_advantages(rewards, eps=1e-8):
    """(num_prompts, G) -> (num_prompts, G)"""
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True)
    return (rewards - mean) / (std + eps)
```

### GRPOLoss

```python
class GRPOLoss(nn.Module):
    def forward(self, log_probs, old_log_probs, advantages, log_probs_ref):
        kl = approx_kl_divergence(log_probs, log_probs_ref)
        ratio = (log_probs - old_log_probs).exp()
        surr1 = ratio * advantages
        surr2 = clip(ratio, 1-eps, 1+eps) * advantages
        loss = -min(surr1, surr2) + kl_weight * kl
        return loss, kl
```

### GRPOTrainer

```python
trainer = GRPOTrainer(model, ref_model, reward_fn, clip_eps=0.2, kl_weight=0.1)
metrics = trainer.step(prompts, max_new_tokens=50, vocab_size=32000)
```

---

## 代码详解

### 步骤 1：群体优势

GRPO 的核心创新。对于每个有 G 个响应的提示：

```python
rewards = tensor([[r1, r2, r3, r4],   # 提示 1
                   [r1, r2, r3, r4]])  # 提示 2
advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + eps)
# 每行求和为零
```

### 步骤 2：GRPO 损失

与 PPO 相同的裁剪代理，但使用群体相对优势：

```python
ratio = (log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
loss = -min(surr1, surr2) + kl_weight * KL
```

### 步骤 3：GRPO 训练循环

```python
对于每个训练步骤：
    对于每个提示：
        生成 G 个响应
        用奖励函数评分
    计算群体优势
    用 GRPO 损失更新策略
```

---

## GRPO vs PPO

| 方面 | PPO | GRPO |
|------|-----|------|
| 价值网络 | 需要 | 不需要 |
| 优势估计 | 使用学习的 V(s) 的 GAE | 组内奖励归一化 |
| 复杂度 | 更高（策略 + 价值） | 更低（仅策略） |
| 稳定性 | 如果 V 不准确可能不稳定 | 更稳定（无 V 估计误差） |
| 样本效率 | 更高（用 GAE 重复使用数据） | 更低（每步需要新样本） |

---

## 训练技巧

### 超参数

| 参数 | 典型范围 | 描述 |
|------|----------|------|
| `clip_eps` | 0.1 - 0.3 | 裁剪参数（0.2 常用） |
| `kl_weight` | 0.01 - 0.5 | KL 惩罚权重 |
| `lr` | 1e-6 - 1e-5 | 学习率 |
| `num_responses_per_prompt` | 4 - 16 | 组大小 G |

### 常见陷阱

1. **小组大小**：G=2 时，优势信号非常粗糙。使用 G >= 4 以获得更好的信号。

2. **奖励尺度**：如果不同提示的奖励尺度差异很大，优势可能有噪声。考虑在批次间归一化奖励。

3. **KL 散度爆炸**：与 PPO 相同——监控 KL 并在需要时调整 `kl_weight`。

4. **生成成本**：为每个提示生成 G 个响应很昂贵。在 G 和批次大小之间取得平衡。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/05_grpo/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/05_grpo/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现 GRPO 组件。

### 练习顺序

1. **`compute_group_advantages`** -- 在组内归一化奖励
2. **`GRPOLossExercise.forward`** -- 实现裁剪代理目标
3. **`grpo_step_exercise`** -- 实现单次 GRPO 训练步骤

### 提示

- 从 `compute_group_advantages` 开始。关键公式是：`(r - mean) / (std + eps)`。
- 对于 `GRPOLoss`，它与 PPO 的裁剪损失相同——只是使用群体优势。
- 对于 `grpo_step`，记得将 1D 优势扩展到匹配序列维度。
- 计算新的对数概率前使用 `model.train()`，生成时使用 `model.eval()`。

---

## 核心要点

1. **GRPO 消除了价值网络。** 通过生成多个响应并在组内归一化奖励，GRPO 无需学习的价值函数即可获得优势估计。这更简单，避免了价值估计误差。

2. **群体优势求和为零。** 这是构造保证的：在组内归一化奖励确保优势以零为中心。相对排名决定训练信号。

3. **GRPO 使用与 PPO 相同的裁剪损失。** 唯一的区别是优势的来源。裁剪机制和 KL 惩罚的工作方式相同。

4. **更大的组提供更好的信号。** 每个提示有更多响应时，优势估计更稳定。但代价是更多的生成时间。

5. **GRPO 特别适合数学/代码任务。** 在奖励定义明确（正确/错误）的情况下，群体相对优势提供了干净的训练信号，没有学习价值函数的噪声。

---

## 扩展阅读

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) -- 提出 GRPO 的论文
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) -- PPO 论文（GRPO 基于 PPO 的裁剪目标）
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT，推广了 PPO 在 RLHF 中的应用
