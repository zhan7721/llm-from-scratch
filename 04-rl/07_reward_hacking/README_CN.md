# 奖励黑客

> **模块 04 -- 强化学习，第 07 章**

当我们针对奖励模型优化语言模型时，模型可能会学会利用奖励信号中的缺陷——在没有真正提高质量的情况下获得高分。这被称为**奖励黑客**（或奖励过度优化）。本章实现检测和缓解工具。

---

## 前置知识

- 奖励模型基础（模块 04，第 01 章）
- PyTorch nn.Module
- RLHF 训练循环的基本理解

## 文件说明

| 文件 | 用途 |
|------|------|
| `reward_hacking.py` | 核心实现：RewardHackingDetector、KLConstrainedReward、RewardEnsemble、analyze_reward_hacking |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是奖励黑客？

### 问题

在 RLHF 中，我们训练策略来最大化奖励模型的分数。但奖励模型是人类偏好的不完美代理。当我们对它进行更强的优化时，策略可能会找到"捷径"，在不真正提高质量的情况下获得高分：

- **长度利用**：较长的输出获得更高分数，所以模型变得冗长
- **重复利用**：某些短语得分很高，所以模型不断重复
- **风格大于实质**：输出表面上看起来不错，但缺乏连贯性

这类似于古德哈特定律："当一个度量成为目标时，它就不再是一个好的度量。"

### 真实例子

```
奖励模型训练为偏好有帮助的响应：
  - 策略学会了在每个响应开头添加"我很乐意帮忙！"
    （高奖励，但没有实际的帮助性提升）
  - 策略学会了产生带有填充内容的很长响应
    （在训练数据中长度与奖励相关）
  - 策略学会了触发高奖励分数的特定短语
    （模板利用）
```

---

## 架构

### RewardHackingDetector

在训练过程中监控诊断信号，及早检测奖励黑客：

```python
detector = RewardHackingDetector(reward_threshold=2.0, diversity_threshold=0.3)
detector.set_baseline(reference_rewards)

# 训练过程中...
result = detector.detect(current_rewards, generated_tokens, expected_length=50)
if result["is_hacking"]:
    print("警告：检测到奖励黑客！")
```

**监控的信号：**
- **奖励偏离**：当前奖励与基线的 Z 分数（高 = 可疑）
- **输出多样性**：唯一 token 比率（低 = 重复 = 可疑）
- **长度异常**：输出长度与预期的偏差程度（极端 = 可疑）

### KLConstrainedReward

添加 KL 惩罚以保持策略接近参考：

```python
wrapper = KLConstrainedReward(reward_model, beta=0.1)
result = wrapper.forward(input_ids, policy_logits, reference_logits)
effective_reward = result["effective_reward"]
# 有效奖励 = 奖励 - beta * KL(策略 || 参考)
```

KL 惩罚防止策略偏离参考分布太远，从而限制其利用奖励模型弱点的能力。

### RewardEnsemble

平均多个奖励模型以提高鲁棒性：

```python
ensemble = RewardEnsemble([reward_model_1, reward_model_2, reward_model_3])
result = ensemble.forward(input_ids)
mean_reward = result["mean_reward"]  # 跨模型平均
std_reward = result["std_reward"]    # 不确定性信号
```

如果一个奖励模型有盲点，其他模型可以补偿。标准差（分歧）也作为不确定性信号。

### analyze_reward_hacking

比较奖励与质量的诊断函数：

```python
result = analyze_reward_hacking(reward_model, quality_function, input_ids)
print(f"相关性: {result['correlation']:.3f}")
print(f"奖励膨胀: {result['reward_inflation']:.3f}")
print(f"是否黑客: {result['is_hacking']}")
```

---

## 代码详解

### 步骤 1：设置检测

```python
detector = RewardHackingDetector()
detector.set_baseline(reference_rewards)  # 来自初始策略的奖励
```

检测器需要一个基线进行比较。这通常是初始（未优化）策略的奖励分布。

### 步骤 2：训练过程中监控

```python
signals = detector.detect(current_rewards, generated_tokens)
if signals["is_hacking"]:
    # 降低学习率、增加 KL 惩罚或停止训练
    beta *= 1.5  # 增加 KL 系数
```

### 步骤 3：应用 KL 约束

```python
kl_wrapper = KLConstrainedReward(reward_model, beta=0.1)
result = kl_wrapper.forward(input_ids, policy_logits, ref_logits)
loss = -result["effective_reward"].mean()  # 最大化有效奖励
```

### 步骤 4：使用集成提高鲁棒性

```python
ensemble = RewardEnsemble([rm1, rm2, rm3])
scores = ensemble.score(input_ids)  # 比任何单个模型更鲁棒
```

---

## 核心概念

### 奖励偏离

随着策略被优化，其奖励分数倾向于增加。但在某个时刻，这种增加变得"好得不像真的"——策略在利用而不是真正改进。Z 分数衡量当前平均奖励从基线漂移了多少个标准差。

### KL 散度惩罚

KL 散度 KL(pi || pi_ref) 衡量当前策略与参考的差异程度。通过在损失中添加 `beta * KL`，我们创建了一个"皮带"，防止策略移动太远。`beta` 参数控制皮带长度：

- `beta` 太低：策略可以自由利用奖励模型
- `beta` 太高：策略完全无法改进（卡在参考位置）
- `beta` 刚好：策略在保持脚踏实地的同时改进

### 集成分歧

当多个奖励模型一致时，我们可以对分数更有信心。当它们不一致时，表示不确定性——可能是因为输入是对抗性的或分布外的。跨模型的标准差是自然的不确定性度量。

---

## 训练技巧

### 选择 Beta

从 `beta = 0.01` 到 `0.1` 开始，根据检测器信号调整：
- 如果检测到黑客，增加 beta
- 如果改进停滞，减少 beta
- 监控 KL 惩罚项——它应该很小但非零

### 需要多少个集成模型？

- 3-5 个模型是实际的甜蜜点
- 模型应该独立训练（不同的种子、数据子集或架构）
- 更多模型 = 更好的鲁棒性，但计算成本更高

### 何时使用每个工具

| 工具 | 何时使用 |
|------|----------|
| RewardHackingDetector | 始终——成本低且提供早期警告 |
| KLConstrainedReward | RL 训练期间防止过度优化 |
| RewardEnsemble | 当你有多个奖励模型可用时 |
| analyze_reward_hacking | 训练后或评估期间的诊断 |

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/07_reward_hacking/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/07_reward_hacking/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现奖励黑客组件。

### 练习顺序

1. **`RewardHackingDetectorExercise.compute_reward_divergence`** -- 与基线的 Z 分数比较
2. **`RewardHackingDetectorExercise.compute_output_diversity`** -- 唯一 token 比率
3. **`RewardHackingDetectorExercise.compute_length_anomaly`** -- 长度偏差检测
4. **`RewardHackingDetectorExercise.detect`** -- 组合所有信号
5. **`KLConstrainedRewardExercise.compute_kl_penalty`** -- 从 logits 计算 KL 散度
6. **`KLConstrainedRewardExercise.forward`** -- 组合奖励和 KL 惩罚
7. **`RewardEnsembleExercise`** -- 平均多个奖励模型
8. **`analyze_reward_hacking_exercise`** -- 基于相关性的诊断

### 提示

- 从 `compute_reward_divergence` 开始——这是一个简单的 Z 分数计算。
- 对于 `compute_output_diversity`，`torch.unique()` 是你的好朋友。
- KL 惩罚使用 `F.log_softmax` 和 `F.softmax` 保证数值稳定性。
- 集成很直接：循环、堆叠、平均/标准差。

---

## 核心要点

1. **奖励黑客在强优化下不可避免。** 你越是对不完美的奖励模型进行优化，策略就越会利用其弱点。

2. **KL 约束是主要防线。** 通过惩罚与参考策略的偏离，我们限制了策略发现和利用奖励模型盲点的能力。

3. **集成减少单模型漏洞。** 平均多个奖励模型使策略更难找到对所有模型都有效的利用方式。

4. **检测优于治疗。** 在训练过程中监控奖励偏离和输出多样性等信号。早期检测允许你在策略退化之前进行调整。

5. **奖励与质量的相关性是关键指标。** 如果奖励上升但质量没有上升，你就遇到了奖励黑客。

---

## 扩展阅读

- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2023)](https://arxiv.org/abs/2210.10760) -- 奖励黑客如何随优化强度扩展
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT，讨论 RLHF 中的 KL 约束
- [Concrete Problems in AI Safety (Amodei et al., 2016)](https://arxiv.org/abs/1606.06565) -- 奖励黑客作为安全问题
- [Reward Model Ensembles (Coste et al., 2023)](https://arxiv.org/abs/2310.02743) -- 使用集成进行更鲁棒的奖励估计
