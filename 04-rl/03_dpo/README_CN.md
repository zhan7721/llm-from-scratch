# 直接偏好优化 (DPO)

> **模块 04 -- 强化学习，第 03 章**

DPO 是基于 PPO 的 RLHF 的更简单替代方案。DPO 无需训练单独的奖励模型然后使用 RL 优化策略，而是通过巧妙的重参数化直接在偏好数据上优化策略，消除了对显式奖励模型的需求。

关键洞察：你的语言模型本身就是奖励模型。通过在 KL 约束下重参数化最优策略，我们可以用策略自身的对数概率来表示奖励，然后直接优化策略。

---

## 前置知识

- 奖励模型基础（模块 04，第 01 章）
- PyTorch nn.Module、autograd
- 对数概率和 Bradley-Terry 偏好模型的理解

## 文件说明

| 文件 | 用途 |
|------|------|
| `dpo.py` | 核心实现：compute_log_probs、DPOLoss、DPODataset、DPOTrainer |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是 DPO？

### 基于 PPO 的 RLHF 的问题

基于 PPO 的 RLHF 需要三个独立的模型：
1. 奖励模型（在偏好数据上训练）
2. 参考模型（冻结的，用于 KL 惩罚）
3. 策略模型（使用 PPO 训练）

这很复杂、不稳定且昂贵。奖励模型可能被利用，PPO 需要仔细调参，训练期间生成 rollout 也很慢。

### DPO 的解决方案：你的语言模型就是奖励模型

DPO 从一个关键观察出发：KL 约束下的最优策略有封闭形式解：

```
pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)
```

重新整理以求解奖励：

```
r(x,y) = beta * (log pi*(y|x) - log pi_ref(y|x)) + beta * log Z(x)
```

当我们将其代入 Bradley-Terry 偏好模型时，配分函数 Z(x) 会消去，得到一个直接依赖于策略的损失：

```
L_DPO = -log sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x) - log pi(y_l|x) + log pi_ref(y_l|x)))
```

不需要奖励模型。不需要 RL。只需在偏好对上进行监督学习。

### 为什么有效

隐式奖励 `r(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))` 衡量策略相对于参考模型对响应 y 的偏好程度。通过训练策略最大化选中响应的隐式奖励并最小化被拒绝响应的隐式奖励，我们直接在人类偏好上优化策略。

---

## 架构

### compute_log_probs

```python
def compute_log_probs(model, input_ids, response_start_idx):
    logits = model(input_ids)                            # (batch, seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)            # (batch, seq, vocab)
    response_tokens = input_ids[:, response_start_idx:]   # (batch, response_len)
    # 在位置 t-1 处收集位置 t 的 token 的对数概率
    token_log_probs = gather(log_probs[:, start-1:end-1], response_tokens)
    return token_log_probs.sum(dim=-1)                    # (batch,)
```

### DPOLoss

```python
class DPOLoss(nn.Module):
    def __init__(self, beta=0.1):
        ...

    def forward(self, policy_chosen, policy_rejected, ref_chosen, ref_rejected):
        log_ratio_chosen = policy_chosen - ref_chosen
        log_ratio_rejected = policy_rejected - ref_rejected
        loss = -logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        return loss.mean()
```

### DPODataset

在初始化期间预计算参考对数概率：

```python
dataset = DPODataset(pairs, ref_model)
# pairs = [{"prompt": [...], "chosen": [...], "rejected": [...]}, ...]
# 内部计算 ref_chosen_log_probs 和 ref_rejected_log_probs
```

### DPOTrainer

简单的训练循环（无 RL 复杂性）：

```python
trainer = DPOTrainer(model, ref_model, beta=0.1)
metrics = trainer.step(batch)
# 只需：计算策略对数概率 -> 计算损失 -> 反向传播 -> 更新
```

---

## 代码详解

### 步骤 1：compute_log_probs

将语言模型连接到 DPO 损失的核心函数。对于自回归模型，位置 t 的 logits 预测位置 t+1 的 token：

```python
# logits[:, t-1, :] 预测位置 t 的 token
# 所以位置 t 的响应 token 的对数概率使用位置 t-1 的 logits
log_probs_all = F.log_softmax(logits, dim=-1)
response_log_probs = log_probs_all[:, start-1:end-1, :].gather(-1, tokens)
return response_log_probs.sum(dim=-1)
```

### 步骤 2：DPOLoss

消除奖励模型的优雅 DPO 损失：

```python
# 隐式奖励：r(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))
# DPO 损失：-log sigmoid(r(x, y_w) - r(x, y_l))
log_ratio_chosen = policy_chosen - ref_chosen
log_ratio_rejected = policy_rejected - ref_rejected
loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
```

### 步骤 3：DPODataset

预计算参考对数概率以避免训练期间的冗余计算：

```python
with torch.no_grad():
    chosen_lp = compute_log_probs(ref_model, chosen_ids, prompt_len)
    rejected_lp = compute_log_probs(ref_model, rejected_ids, prompt_len)
```

### 步骤 4：DPOTrainer

简单的监督训练循环（无 RL 复杂性）：

```python
policy_chosen = compute_log_probs(model, chosen_ids, start_idx)
policy_rejected = compute_log_probs(model, rejected_ids, start_idx)
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
loss.backward()
optimizer.step()
```

---

## 训练技巧

### 超参数

| 参数 | 典型范围 | 描述 |
|------|----------|------|
| `beta` | 0.1 - 0.5 | 隐式奖励温度（越高越保守） |
| `lr` | 1e-7 - 5e-6 | 学习率（低于典型微调） |
| `max_grad_norm` | 0.5 - 1.0 | 梯度裁剪 |

### 常见陷阱

1. **Beta 太高**：使损失对小差异非常敏感，可能导致训练不稳定。

2. **Beta 太低**：策略学不到太多，因为隐式奖励差异被压缩了。

3. **学习率太高**：DPO 对学习率敏感。从低值（5e-7）开始，谨慎增加。

4. **提示/响应对齐**：确保 `response_start_idx` 正确地将提示与响应分开。不匹配会导致无意义的对数概率。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/03_dpo/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/03_dpo/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现 DPO 组件。

### 练习顺序

1. **`compute_log_probs`** -- 计算响应 token 的对数概率
2. **`DPOLoss.forward`** -- 实现 DPO 损失函数
3. **`DPODataset.__init__`** -- 预计算参考对数概率
4. **`DPOTrainer.step`** -- 实现训练步骤

### 提示

- 从 `compute_log_probs` 开始。记住：位置 t 的 logits 预测位置 t+1 的 token。
- 对于 `DPOLoss`，使用 `F.logsigmoid` 确保数值稳定性。
- 在 `DPODataset` 中，冻结参考模型并在预计算期间使用 `torch.no_grad()`。
- 在 `DPOTrainer.step` 中，流程是：计算策略对数概率 -> 计算损失 -> 反向传播 -> 更新。

---

## 核心要点

1. **你的语言模型就是奖励模型。** DPO 表明 KL 约束下的最优策略隐式定义了奖励。不需要单独的奖励模型。

2. **DPO 是监督学习。** 与需要 RL 的 PPO 不同，DPO 只是在偏好对上最小化损失。这使它更简单、更稳定。

3. **参考模型提供基线。** 冻结的参考模型充当 KL 锚点，防止策略偏离预训练模型太远。

4. **Beta 控制权衡。** 更高的 beta 使隐式奖励对对数概率差异更敏感，导致更激进的优化。更低的 beta 更保守。

5. **预计算节省时间。** 由于参考模型是冻结的，我们可以在数据集初始化期间预计算其对数概率，避免训练期间的冗余计算。

---

## 扩展阅读

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) -- DPO 原始论文
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT，DPO 简化的 RLHF 基线
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) -- AI 反馈对齐的相关工作
