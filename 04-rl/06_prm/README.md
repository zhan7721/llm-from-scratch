# Process Reward Model (PRM)

> **Module 04 -- Reinforcement Learning, Chapter 06**

Process Reward Models (PRMs) score each reasoning step independently, rather than only scoring the final answer. This gives more granular feedback for training reasoning models -- you can identify exactly which step went wrong, not just whether the final answer is correct.

This chapter implements the core PRM components: the step-level scoring model, a dataset with per-step labels, a stepwise loss function, and a best-of-n selection procedure.

---

## Prerequisites

- Reward model basics (Module 04, Chapter 01)
- Transformer architecture (Module 01)
- PyTorch nn.Module, autograd

## Files

| File | Purpose |
|------|---------|
| `process_reward_model.py` | Core implementation: ProcessRewardModel, StepwiseRewardDataset, StepwiseRewardLoss, best_of_n_prm |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is a Process Reward Model?

### Outcome vs Process Reward

Traditional reward models (Outcome Reward Models, ORMs) score only the final answer:

```
Question: What is 2 + 3 * 4?
Response: "2 + 3 = 5, 5 * 4 = 20"
ORM score: 0.1 (wrong answer, low score)
```

A Process Reward Model scores each step:

```
Question: What is 2 + 3 * 4?
Step 1: "3 * 4 = 12"       -> PRM score: 0.9 (correct)
Step 2: "2 + 12 = 14"      -> PRM score: 0.9 (correct)
Step 3: "Answer is 14"      -> PRM score: 0.9 (correct)
```

vs.

```
Step 1: "2 + 3 = 5"        -> PRM score: 0.2 (incorrect order of operations)
Step 2: "5 * 4 = 20"       -> PRM score: 0.8 (arithmetic correct, but wrong inputs)
Step 3: "Answer is 20"      -> PRM score: 0.3 (wrong final answer)
```

The PRM pinpoints exactly where the reasoning went wrong.

### Why PRMs Matter

1. **Better feedback**: Instead of a single "right/wrong" signal, the model gets feedback on each step.
2. **Better search**: When generating multiple reasoning paths, PRMs help select paths where ALL steps are correct, not just the final answer.
3. **Less reward hacking**: ORMs can be fooled by correct-looking but wrong reasoning that happens to reach the right answer. PRMs catch this.

---

## Architecture

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

The model uses the hidden state at each step's boundary position. In autoregressive models, the last token of a step has attended to all tokens in that step, providing a complete representation.

### Step Boundaries

Step boundaries are indices into the token sequence indicating where each step ends:

```
Tokens:    [a, b, c, d, e, f, g, h, i, j, k]
Step 1:    [a, b, c]           -> boundary = 2
Step 2:    [d, e, f, g]        -> boundary = 6
Step 3:    [h, i, j, k]        -> boundary = 10
step_boundaries = [2, 6, 10]
```

### StepwiseRewardLoss

Standard binary cross-entropy applied to each step independently:

```python
loss = BCE_with_logits(step_scores, step_labels)  # per step
loss = loss.mean()  # average over all steps
```

### best_of_n_prm

Given N candidate responses, aggregate step scores and pick the best:

```python
# Four aggregation methods:
'min'     -> worst-case step score (conservative)
'sum'     -> total score across steps
'mean'    -> average step score (length-normalized)
'product' -> probability all steps are correct (assumes independence)
```

---

## Code Walkthrough

### Step 1: ProcessRewardModel

The PRM wraps a transformer and adds a per-step linear head:

```python
# Hidden states at step boundaries capture full context
step_hidden = hidden_states[batch_indices, step_boundaries]
# Linear head maps each step to a scalar logit
step_logits = self.step_head(step_hidden).squeeze(-1)
```

### Step 2: StepwiseRewardDataset

Simple dataset that stores pre-tokenized data with per-step labels:

```python
{
    "input_ids": [1, 2, 3, 4, 5, 6, 7, 8],
    "step_boundaries": [3, 7],      # 2 steps
    "step_labels": [1, 0],          # step 1 correct, step 2 wrong
}
```

### Step 3: StepwiseRewardLoss

BCE with logits, with optional masking for variable-length step counts:

```python
per_step_loss = F.binary_cross_entropy_with_logits(scores, labels, reduction='none')
if mask is not None:
    loss = (per_step_loss * mask).sum() / mask.sum()
else:
    loss = per_step_loss.mean()
```

### Step 4: best_of_n_prm

Aggregate step scores to select the best candidate:

```python
# Conservative: pick the response where the worst step is best
agg_scores = step_scores.min(dim=1).values
best_idx = agg_scores.argmax()
```

---

## PRM vs ORM Comparison

| Aspect | ORM (Outcome) | PRM (Process) |
|--------|---------------|---------------|
| Granularity | One score per response | One score per step |
| Feedback | "Final answer right/wrong" | "Step 2 was wrong" |
| Reward hacking | Easier to fool | Harder to fool |
| Labeling cost | Low (check final answer) | High (label each step) |
| Search quality | May select lucky wrong paths | Selects consistently correct paths |
| Training data | Easy to collect | Requires step-level annotations |

---

## Training Tips

### Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `lr` | 1e-6 - 1e-5 | Learning rate |
| `batch_size` | 16 - 64 | Batch size |
| `aggregation` | 'min' | Best-of-n aggregation method |

### Common Pitfalls

1. **Step boundary errors**: If boundaries don't align with actual step transitions, the PRM learns garbage. Ensure boundaries are at step ends, not middles.

2. **Class imbalance**: In many datasets, most steps are correct. Consider oversampling incorrect steps or using weighted loss.

3. **Product aggregation with logits**: The product aggregation assumes scores are probabilities. Apply sigmoid first if your PRM outputs logits.

4. **Variable step counts**: Different candidates may have different numbers of steps. Use step_mask in the loss and best_of_n functions.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/06_prm/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 04-rl/06_prm/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing PRM components yourself.

### Exercise Order

1. **`ProcessRewardModelExercise.forward`** -- Extract hidden states at step boundaries and score each step
2. **`StepwiseRewardDatasetExercise`** -- Store and retrieve step-level training data
3. **`StepwiseRewardLossExercise.forward`** -- Compute BCE loss over step scores
4. **`best_of_n_prm_exercise`** -- Aggregate step scores to select the best candidate

### Tips

- For `ProcessRewardModel.forward`, use advanced indexing: `hidden_states[batch_indices, step_boundaries]`.
- For the loss, use `F.binary_cross_entropy_with_logits` with `reduction='none'` to get per-step losses.
- For `best_of_n_prm`, handle the mask carefully: for 'min', replace masked positions with +inf; for 'product', replace with 1.0.

---

## Key Takeaways

1. **PRMs score each step, not just the final answer.** This provides much richer training signal for reasoning models. Instead of knowing "the answer is wrong," you know "step 3 is where it went wrong."

2. **Step boundaries are the key design choice.** The hidden state at each step's end position captures the full reasoning context up to that point. Getting boundaries right is critical.

3. **BCE loss treats each step independently.** Each step is a binary classification: correct or incorrect. The loss averages over all steps.

4. **Aggregation method matters for best-of-n.** 'min' is conservative (all steps must be good), 'product' assumes independence (probability all steps are correct), 'sum' and 'mean' are more lenient.

5. **PRMs are expensive to train but valuable.** Labeling each step requires expert annotation. But the resulting model provides much better signal for training reasoning models than ORMs.

---

## Further Reading

- [Let's Verify Step by Step (Lightman et al., 2023)](https://arxiv.org/abs/2305.20050) -- The paper that introduced PRMs for mathematical reasoning
- [Training Verifiers to Solve Math Word Problems (Cobbe et al., 2021)](https://arxiv.org/abs/2110.14168) -- Introduced outcome reward models and best-of-n verification
- [Solving Math Word Problems with Process- and Outcome-Based Feedback (Uesato et al., 2022)](https://arxiv.org/abs/2211.14275) -- Comparison of process vs outcome reward models
