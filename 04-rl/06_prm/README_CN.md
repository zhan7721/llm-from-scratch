# 过程奖励模型 (PRM)

> **模块 04 -- 强化学习，第 06 章**

过程奖励模型（PRM）独立评估每个推理步骤，而不是只评估最终答案。这为训练推理模型提供了更细粒度的反馈——你可以精确定位是哪一步出了问题，而不仅仅是知道最终答案是否正确。

本章实现 PRM 的核心组件：步骤级评分模型、带有步骤标签的数据集、步骤级损失函数，以及 best-of-n 选择过程。

---

## 前置知识

- 奖励模型基础（模块 04，第 01 章）
- Transformer 架构（模块 01）
- PyTorch nn.Module、autograd

## 文件说明

| 文件 | 用途 |
|------|------|
| `process_reward_model.py` | 核心实现：ProcessRewardModel、StepwiseRewardDataset、StepwiseRewardLoss、best_of_n_prm |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 正确性测试 |

---

## 什么是过程奖励模型？

### 结果奖励 vs 过程奖励

传统的奖励模型（结果奖励模型，ORM）只评估最终答案：

```
问题：2 + 3 * 4 等于多少？
回答："2 + 3 = 5，5 * 4 = 20"
ORM 评分：0.1（答案错误，低分）
```

过程奖励模型评估每个步骤：

```
问题：2 + 3 * 4 等于多少？
步骤 1："3 * 4 = 12"       -> PRM 评分：0.9（正确）
步骤 2："2 + 12 = 14"      -> PRM 评分：0.9（正确）
步骤 3："答案是 14"         -> PRM 评分：0.9（正确）
```

对比：

```
步骤 1："2 + 3 = 5"        -> PRM 评分：0.2（运算顺序错误）
步骤 2："5 * 4 = 20"       -> PRM 评分：0.8（计算正确，但输入错误）
步骤 3："答案是 20"         -> PRM 评分：0.3（最终答案错误）
```

PRM 精确定位推理出错的位置。

### 为什么 PRM 很重要

1. **更好的反馈**：模型不是得到单一的"对/错"信号，而是对每一步都得到反馈。
2. **更好的搜索**：在生成多条推理路径时，PRM 帮助选择每一步都正确的路径，而不仅仅是最终答案正确的路径。
3. **更难被欺骗**：ORM 可能被看似合理但实际错误的推理过程欺骗（恰好得到正确答案）。PRM 能够发现这些问题。

---

## 架构

### ProcessRewardModel

```python
class ProcessRewardModel(nn.Module):
    def __init__(self, transformer):
        self.transformer = transformer
        self.step_head = nn.Linear(d_model, 1)

    def forward(self, input_ids, step_boundaries):
        hidden_states = self.transformer(input_ids)  # (batch, seq, d_model)
        step_hidden = hidden_states[batch_indices, step_boundaries]  # (batch, num_steps, d_model)
        step_logits = self.step_head(step_hidden).squeeze(-1)  # (batch, num_steps)
        return step_logits
```

模型使用每个步骤边界位置的隐藏状态。在自回归模型中，步骤的最后一个 token 已经关注了该步骤中的所有 token，提供了完整的表示。

### 步骤边界

步骤边界是 token 序列中每个步骤结束位置的索引：

```
Token:     [a, b, c, d, e, f, g, h, i, j, k]
步骤 1:    [a, b, c]           -> 边界 = 2
步骤 2:    [d, e, f, g]        -> 边界 = 6
步骤 3:    [h, i, j, k]        -> 边界 = 10
step_boundaries = [2, 6, 10]
```

### StepwiseRewardLoss

对每个步骤独立应用标准二元交叉熵：

```python
loss = BCE_with_logits(step_scores, step_labels)  # 每个步骤
loss = loss.mean()  # 所有步骤取平均
```

### best_of_n_prm

给定 N 个候选响应，聚合步骤分数并选择最佳：

```python
# 四种聚合方法：
'min'     -> 最差步骤分数（保守策略）
'sum'     -> 所有步骤分数之和
'mean'    -> 平均步骤分数（长度归一化）
'product' -> 所有步骤都正确的概率（假设独立性）
```

---

## 代码详解

### 步骤 1：ProcessRewardModel

PRM 包装一个 transformer 并添加步骤级线性头：

```python
# 步骤边界的隐藏状态捕获完整上下文
step_hidden = hidden_states[batch_indices, step_boundaries]
# 线性头将每个步骤映射为标量 logit
step_logits = self.step_head(step_hidden).squeeze(-1)
```

### 步骤 2：StepwiseRewardDataset

简单的数据集，存储带有步骤标签的预分词数据：

```python
{
    "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
    "step_boundaries": [3, 7],      # 2 个步骤
    "step_labels": [1, 0],          # 步骤 1 正确，步骤 2 错误
}
```

### 步骤 3：StepwiseRewardLoss

使用 logits 的 BCE，支持可变长度步骤数的掩码：

```python
per_step_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
if mask is not None:
    loss = (per_step_loss * mask).sum() / mask.sum()
else:
    loss = per_step_loss.mean()
```

### 步骤 4：best_of_n_prm

聚合步骤分数以选择最佳候选：

```python
# 保守策略：选择最差步骤分数最高的响应
agg_scores = step_scores.min(dim=1).values
best_idx = agg_scores.argmax()
```

---

## PRM vs ORM 对比

| 方面 | ORM（结果） | PRM（过程） |
|------|------------|------------|
| 粒度 | 每个响应一个分数 | 每个步骤一个分数 |
| 反馈 | "最终答案对/错" | "步骤 2 错了" |
| 奖励欺骗 | 更容易被欺骗 | 更难被欺骗 |
| 标注成本 | 低（检查最终答案） | 高（标注每个步骤） |
| 搜索质量 | 可能选择碰巧正确的错误路径 | 选择持续正确的路径 |
| 训练数据 | 容易收集 | 需要步骤级标注 |

---

## 训练技巧

### 超参数

| 参数 | 典型范围 | 描述 |
|------|----------|------|
| `lr` | 1e-6 - 1e-5 | 学习率 |
| `batch_size` | 16 - 64 | 批次大小 |
| `aggregation` | 'min' | best-of-n 聚合方法 |

### 常见陷阱

1. **步骤边界错误**：如果边界与实际步骤转换不对齐，PRM 会学到无用信息。确保边界在步骤末尾，而不是中间。

2. **类别不平衡**：在许多数据集中，大多数步骤是正确的。考虑对错误步骤进行过采样或使用加权损失。

3. **使用 logits 进行乘积聚合**：乘积聚合假设分数是概率。如果你的 PRM 输出 logits，先应用 sigmoid。

4. **可变步骤数**：不同的候选可能有不同数量的步骤。在损失和 best_of_n 函数中使用 step_mask。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/06_prm/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 04-rl/06_prm/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习实现 PRM 组件。

### 练习顺序

1. **`ProcessRewardModelExercise.forward`** -- 在步骤边界提取隐藏状态并评估每个步骤
2. **`StepwiseRewardDatasetExercise`** -- 存储和检索步骤级训练数据
3. **`StepwiseRewardLossExercise.forward`** -- 计算步骤分数的 BCE 损失
4. **`best_of_n_prm_exercise`** -- 聚合步骤分数以选择最佳候选

### 提示

- 对于 `ProcessRewardModel.forward`，使用高级索引：`hidden_states[batch_indices, step_boundaries]`。
- 对于损失，使用 `F.binary_cross_entropy_with_logits` 并设置 `reduction='none'` 以获得每步损失。
- 对于 `best_of_n_prm`，小心处理掩码：对于 'min'，用 +inf 替换掩码位置；对于 'product'，用 1.0 替换。

---

## 核心要点

1. **PRM 评估每个步骤，而不仅仅是最终答案。** 这为推理模型提供了更丰富的训练信号。不是知道"答案错了"，而是知道"步骤 3 出了问题"。

2. **步骤边界是关键设计选择。** 每个步骤末尾的隐藏状态捕获了到该点为止的完整推理上下文。正确设置边界至关重要。

3. **BCE 损失独立处理每个步骤。** 每个步骤是一个二元分类：正确或错误。损失在所有步骤上取平均。

4. **聚合方法对 best-of-n 很重要。** 'min' 是保守的（所有步骤都必须好），'product' 假设独立性（所有步骤都正确的概率），'sum' 和 'mean' 更宽松。

5. **PRM 训练成本高但很有价值。** 标注每个步骤需要专家标注。但生成的模型为训练推理模型提供了比 ORM 好得多的信号。

---

## 扩展阅读

- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) -- 提出 PRM 用于数学推理的论文
- [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168) -- 引入结果奖励模型和 best-of-n 验证
- [Solving Math Word Problems with Process- and Outcome-Based Feedback (Uesato et al., 2022)](https://arxiv.org/abs/2211.14275) -- 过程奖励 vs 结果奖励模型的对比
