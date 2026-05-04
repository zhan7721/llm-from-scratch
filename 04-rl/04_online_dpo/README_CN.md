# 在线 DPO (Online DPO)

> **模块 04 -- 强化学习，第 04 章**

在线 DPO 从当前策略动态生成偏好对，而非使用静态数据集。这解决了离线 DPO 的关键限制：数据生成策略与正在训练的策略之间的分布不匹配。

---

## 前置知识

- DPO 基础（模块 04，第 03 章）
- 奖励模型基础（模块 04，第 01 章）
- PyTorch nn.Module、autograd、温度采样

## 文件说明

| 文件 | 用途 |
|------|------|
| `online_dpo.py` | 核心实现：generate_and_score、OnlineDPODataset、OnlineDPOTrainer |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是在线 DPO？

### 离线 DPO 的问题

在标准（离线）DPO 中，偏好对从固定策略收集一次并在整个训练过程中重复使用。这会导致**分布偏移**：数据由旧版本的策略生成，但我们正在训练当前版本。随着策略的改进，旧的偏好数据变得越来越不相关。

### 在线 DPO 的解决方案：每个 epoch 生成新数据

在线 DPO 通过每个 epoch 从当前策略生成新的偏好数据来解决这个问题：

1. **生成**：从当前策略生成多个候选响应
2. **评分**：使用奖励模型对每个候选进行评分
3. **选择**：选择最好的（chosen）和最差的（rejected）作为偏好对
4. **训练**：在新的偏好对上使用标准 DPO 损失进行训练
5. **重复**：每个 epoch 使用更新后的策略生成新数据

权衡是计算成本：生成响应很昂贵。但训练信号的质量要好得多，因为偏好对反映了当前策略的行为。

---

## 架构

### generate_and_score

从当前策略创建偏好对的核心函数：

```python
def generate_and_score(policy, reward_model, prompt, max_new_tokens, num_candidates, temperature):
    # 1. 使用温度采样生成 num_candidates 个响应
    candidates = []
    for _ in range(num_candidates):
        input_ids = prompt.clone()
        for _ in range(max_new_tokens):
            logits = policy(input_ids)
            next_token = sample(softmax(logits[-1] / temperature))
            input_ids = cat([input_ids, next_token])
        candidates.append(input_ids)

    # 2. 使用奖励模型评分
    rewards = reward_model(stack(candidates))

    # 3. 选择最好/最差
    chosen = candidates[rewards.argmax()]
    rejected = candidates[rewards.argmin()]

    # 4. 计算参考对数概率
    chosen_ref_lp = compute_log_probs(policy, chosen)
    rejected_ref_lp = compute_log_probs(policy, rejected)

    return {chosen, rejected, rewards, ref_log_probs}
```

### OnlineDPODataset

从在线生成动态创建偏好对：

```python
dataset = OnlineDPODataset(policy, reward_model, prompts)
# 从当前策略生成新的偏好对
# 可以调用 dataset.refresh() 在策略更新后重新生成
```

### OnlineDPOTrainer

将生成和训练结合在一个循环中：

```python
trainer = OnlineDPOTrainer(model, ref_model, reward_model, prompts)
for epoch in range(num_epochs):
    dataset = trainer.generate()  # 新的偏好对
    for batch in dataset:
        metrics = trainer.train_step(batch)  # DPO 损失
```

---

## 代码详解

### 步骤 1：generate_and_score

生成多个响应并评分：

```python
# 温度采样生成多样化候选
logits = policy(input_ids)[:, -1, :] / temperature
probs = F.softmax(logits, dim=-1)
next_token = torch.multinomial(probs, num_samples=1)

# 使用奖励模型对所有候选评分
rewards = reward_model(candidate_batch)

# 选择最好（chosen）和最差（rejected）
chosen_idx = rewards.argmax()
rejected_idx = rewards.argmin()
```

### 步骤 2：OnlineDPODataset

在生成时预计算参考对数概率：

```python
# 参考对数概率在生成时计算（冻结快照）
chosen_ref_lp = compute_log_probs(policy, chosen_ids, prompt_len)
rejected_ref_lp = compute_log_probs(policy, rejected_ids, prompt_len)
```

### 步骤 3：OnlineDPOTrainer

在新数据上进行标准 DPO 训练：

```python
# 计算当前策略的对数概率
policy_chosen = compute_log_probs(model, chosen_ids, start_idx)
policy_rejected = compute_log_probs(model, rejected_ids, start_idx)

# 使用预计算的参考对数概率计算 DPO 损失
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
loss.backward()
optimizer.step()
```

---

## 训练技巧

### 超参数

| 参数 | 典型范围 | 描述 |
|------|----------|------|
| `beta` | 0.1 - 0.5 | DPO 损失的温度参数 |
| `lr` | 1e-7 - 5e-6 | 学习率 |
| `num_candidates` | 4 - 16 | 每个提示的候选数（越多信号越好） |
| `temperature` | 0.7 - 1.2 | 生成的采样温度 |
| `max_new_tokens` | 32 - 256 | 响应长度 |

### 常见陷阱

1. **候选太少**：只有 2 个候选时，偏好信号很嘈杂。至少使用 4 个。

2. **温度太低**：使所有候选相似，降低偏好对的质量。

3. **温度太高**：产生不连贯的响应，无法提供有用的训练信号。

4. **需要相同长度的提示**：为了批处理，提示应该有相同的长度（或使用填充）。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/04_online_dpo/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/04_online_dpo/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现在线 DPO 组件。

### 练习顺序

1. **`generate_and_score`** -- 生成响应并使用奖励模型评分
2. **`OnlineDPODataset.__init__`** -- 设置动态数据集
3. **`OnlineDPODataset.refresh`** -- 重新生成偏好对
4. **`OnlineDPOTrainer.__init__`** -- 设置训练器
5. **`OnlineDPOTrainer.train_step`** -- 实现 DPO 训练步骤
6. **`OnlineDPOTrainer.train_epoch`** -- 结合生成和训练

### 提示

- 从 `generate_and_score` 开始。关键是使用 `torch.multinomial` 进行温度采样。
- 在 `OnlineDPODataset` 中，冻结奖励模型并在 `__init__` 中调用 `refresh()`。
- 在 `OnlineDPOTrainer.train_epoch` 中，生成新数据然后分批训练。
- 参考对数概率应在生成时计算（冻结快照）。

---

## 核心要点

1. **在线生成解决分布偏移。** 通过每个 epoch 从当前策略生成偏好对，训练数据始终反映策略的当前行为。

2. **奖励模型提供信号。** 与离线 DPO 中预先收集偏好不同，在线 DPO 使用奖励模型对生成的响应进行评分并创建偏好对。

3. **更多候选 = 更好的信号。** 为每个提示生成更多候选，给奖励模型更多选项来区分好的和坏的响应。

4. **权衡：计算 vs 质量。** 在线 DPO 比离线 DPO 更昂贵，因为它每个 epoch 都生成响应。但训练信号要好得多。

5. **参考对数概率是冻结的。** 参考对数概率在生成时计算并存储，而不是在训练期间重新计算。这对于稳定的 DPO 训练很重要。

---

## 扩展阅读

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) -- DPO 原始论文
- [Self-Play Fine-Tuning Converts Weak Language Models to Strong Language Models (Chen et al., 2024)](https://arxiv.org/abs/2401.01335) -- SPIN，相关的在线 DPO 方法
- [Online DPO vs Offline DPO: A Comparison (various)](https://arxiv.org/abs/2402.04792) -- 在线与离线偏好学习的分析
