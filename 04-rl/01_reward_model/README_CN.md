# 奖励模型

> **模块 04 -- 强化学习，第 01 章**

基于下一个 token 预测训练的语言模型并不天生知道什么样的输出从人类角度来看是"好"的。奖励模型弥合了这一差距：它通过在人类偏好数据（选择/拒绝的响应对）上训练来学习评估文本质量。这个分数随后驱动强化学习，引导语言模型产生人类偏好的输出。

本章实现核心组件：包装 transformer 并输出标量分数的奖励模型、Bradley-Terry 成对排序损失函数，以及偏好对数据集。

---

## 前置知识

- Transformer 语言模型基础（模块 01）
- PyTorch nn.Module、Dataset、DataLoader
- RLHF 的基本概念

## 文件说明

| 文件 | 用途 |
|------|------|
| `reward_model.py` | 核心实现：RewardModel、BradleyTerryLoss、RewardDataset、train_reward_model |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是奖励模型？

### 从偏好到分数

在 RLHF（基于人类反馈的强化学习）中，奖励模型在人类偏好上训练。人类比较同一提示的两个响应，指出他们更喜欢哪一个。奖励模型学习为更受欢迎的响应分配更高的分数。

```
提示："解释量子计算"
响应 A："量子计算使用可以同时处于 0 和 1 状态的量子比特..."
响应 B："不知道，哈哈"

人类偏好：A 更好
奖励模型学习：score(A) > score(B)
```

### Bradley-Terry 模型

Bradley-Terry 模型是从成对偏好中学习的数学基础。它将响应 A 优于响应 B 的概率定义为：

```
P(A > B) = sigmoid(r_A - r_B)
```

其中 `r_A` 和 `r_B` 是模型分配的奖励分数。训练损失是负对数似然：

```
loss = -log(sigmoid(r_chosen - r_rejected))
```

这个损失函数有很好的性质：
- 当 `r_chosen >> r_rejected` 时：损失趋近于 0（模型正确排序）
- 当 `r_chosen << r_rejected` 时：损失非常高（模型排序错误）
- 当 `r_chosen == r_rejected` 时：损失 = log(2)（最大不确定性）

---

## 架构

### RewardModel

包装任意 transformer 模型并添加标量奖励头：

```python
class RewardModel(nn.Module):
    def __init__(self, transformer):
        self.transformer = transformer
        self.scalar_head = nn.Linear(d_model, 1)

    def forward(self, input_ids):
        hidden_states = self.transformer(input_ids)  # (batch, seq, d_model)
        last_hidden = hidden_states[:, -1, :]          # (batch, d_model)
        reward = self.scalar_head(last_hidden)          # (batch, 1)
        return reward.squeeze(-1)                       # (batch,)
```

关键设计选择：我们使用**最后一个 token 的隐藏状态**作为序列表示。在自回归（GPT 风格）模型中，最后一个 token 已经关注了所有之前的 token，因此它包含整个序列最完整的表示。

### BradleyTerryLoss

一个简单但优雅的损失函数：

```python
def forward(self, chosen_rewards, rejected_rewards):
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return loss.mean()
```

我们使用 `F.logsigmoid` 而不是 `torch.log(torch.sigmoid(...))` 来保证数值稳定性。`logsigmoid` 函数避免了在计算大正值或大负值的 `log(sigmoid(x))` 时可能出现的溢出/下溢问题。

### RewardDataset

一个加载偏好对的 PyTorch Dataset：

```python
pairs = [
    {"chosen": [token_ids...], "rejected": [token_ids...]},
    {"chosen": [token_ids...], "rejected": [token_ids...]},
    ...
]
```

支持可选的填充到固定 `max_length`，以便进行批量训练。

---

## 代码详解

### 步骤 1：包装 Transformer

```python
model = RewardModel(transformer)
```

RewardModel 接受任何返回形状为 `(batch, seq, d_model)` 隐藏状态的 transformer。它会自动从 transformer 的属性推断 `d_model`。

### 步骤 2：计算奖励

```python
hidden_states = self.transformer(input_ids)  # (batch, seq, d_model)
last_hidden = hidden_states[:, -1, :]          # (batch, d_model)
reward = self.scalar_head(last_hidden).squeeze(-1)  # (batch,)
```

最后一个 token 的隐藏状态被投影到单个标量。这个标量就是奖励分数。

### 步骤 3：计算 Bradley-Terry 损失

```python
loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

损失鼓励 `r_chosen > r_rejected`。损失相对于模型参数的梯度推动选择响应的奖励更高，拒绝响应的奖励更低。

### 步骤 4：训练循环

```python
chosen_rewards = model(batch["chosen_input_ids"])
rejected_rewards = model(batch["rejected_input_ids"])
loss = loss_fn(chosen_rewards, rejected_rewards)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

标准 PyTorch 训练循环。关键洞察是我们在同一次前向传播中计算选择和拒绝响应的奖励（它们是独立的），然后用成对损失组合它们。

---

## 训练技巧

### 学习率

奖励模型训练通常使用：
- 全参数微调：1e-5 到 5e-5
- 使用预训练 transformer：从更低的学习率开始（1e-5）以避免灾难性遗忘

### 数据需求

奖励模型是数据密集型的。典型数量：
- 最小可行：1K-5K 偏好对
- 生产质量：10K-100K+ 对
- 更多样的提示 = 更好的泛化

### 常见陷阱

1. **奖励黑客**：模型找到获得高分的方式，但实际上并没有更好（例如，更长的响应获得更高分数）。通过多样化的训练数据和正则化来缓解。

2. **过拟合**：奖励模型可以记忆训练对。使用 dropout、权重衰减和早停。

3. **分布偏移**：奖励模型在固定数据集上训练，但在 RL 训练期间，策略生成的响应可能与训练数据看起来不同。这是 RLHF 中的根本挑战。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/01_reward_model/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/01_reward_model/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现奖励模型组件。

### 练习顺序

1. **`BradleyTerryLossExercise.forward`** -- 使用 logsigmoid 实现成对排序损失
2. **`RewardModelExercise.forward`** -- 从 transformer 隐藏状态计算奖励分数
3. **`RewardDatasetExercise.__getitem__`** -- 加载和填充偏好对
4. **`train_reward_model_exercise`** -- 编写训练循环

### 提示

- 从 `BradleyTerryLoss` 开始。这是一个使用 `F.logsigmoid` 的单行实现。
- 对于 `RewardModel.forward`，关键洞察是：取最后一个 token 的隐藏状态并将其投影到标量。
- 对于 `RewardDataset.__getitem__`，处理填充和非填充两种情况。
- 对于 `train_reward_model`，遵循标准 PyTorch 训练循环模式。

---

## 核心要点

1. **奖励模型从偏好中学习，而非标签。** 它在（选择、拒绝）响应对上训练，学习为选择的响应打更高分。

2. **Bradley-Terry 是标准的偏好模型。** P(A > B) = sigmoid(r_A - r_B)。损失是 -log(sigmoid(r_chosen - r_rejected))。

3. **使用最后一个 token 的隐藏状态。** 在自回归模型中，最后一个 token 拥有最多的上下文信息，是序列表示的自然选择。

4. **logsigmoid 保证数值稳定性。** 永远不要直接计算 log(sigmoid(x))，使用 F.logsigmoid 来避免溢出/下溢。

5. **奖励模型是 RLHF 的基础。** 一旦训练完成，它的分数驱动 RL 训练循环（PPO、DPO 等）来对齐语言模型与人类偏好。

---

## 扩展阅读

- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- 推广 RLHF 的 InstructGPT 论文
- [Learning to Summarize from Human Feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325) -- 摘要领域奖励模型的早期工作
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) -- 使用 AI 反馈代替人类反馈
- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2023)](https://arxiv.org/abs/2210.10760) -- 奖励黑客如何随模型规模扩展
