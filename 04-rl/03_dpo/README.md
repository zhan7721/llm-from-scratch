# Direct Preference Optimization (DPO)

> **Module 04 -- Reinforcement Learning, Chapter 03**

DPO is a simpler alternative to PPO-based RLHF. Instead of training a separate reward model and then using RL to optimize the policy, DPO directly optimizes the policy on preference data using a clever reparameterization that eliminates the need for an explicit reward model.

The key insight: your language model is secretly a reward model. By reparameterizing the optimal policy under a KL constraint, we can express the reward in terms of the policy's own log probabilities, then optimize the policy directly.

---

## Prerequisites

- Reward model basics (Module 04, Chapter 01)
- PyTorch nn.Module, autograd
- Understanding of log probabilities and the Bradley-Terry preference model

## Files

| File | Purpose |
|------|---------|
| `dpo.py` | Core implementation: compute_log_probs, DPOLoss, DPODataset, DPOTrainer |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is DPO?

### The Problem with PPO-based RLHF

PPO-based RLHF requires three separate models:
1. A reward model (trained on preference data)
2. A reference model (frozen, for KL penalty)
3. The policy model (being trained with PPO)

This is complex, unstable, and expensive. The reward model can be gamed, PPO requires careful tuning, and generating rollouts during training is slow.

### DPO's Solution: Your LM is a Reward Model

DPO starts from a key observation: the optimal policy under a KL constraint has a closed-form solution:

```
pi*(y|x) = pi_ref(y|x) * exp(r(x,y) / beta) / Z(x)
```

Rearranging to solve for the reward:

```
r(x,y) = beta * (log pi*(y|x) - log pi_ref(y|x)) + beta * log Z(x)
```

The partition function Z(x) cancels out when we plug this into the Bradley-Terry preference model, giving us a loss that directly depends on the policy:

```
L_DPO = -log sigmoid(beta * (log pi(y_w|x) - log pi_ref(y_w|x) - log pi(y_l|x) + log pi_ref(y_l|x)))
```

No reward model needed. No RL needed. Just supervised learning on preference pairs.

### Why This Works

The implicit reward `r(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))` measures how much the policy prefers response y compared to the reference model. By training the policy to maximize this implicit reward for chosen responses and minimize it for rejected responses, we directly optimize the policy on human preferences.

---

## Architecture

### compute_log_probs

```python
def compute_log_probs(model, input_ids, response_start_idx):
    logits = model(input_ids)                            # (batch, seq, vocab)
    log_probs = F.log_softmax(logits, dim=-1)            # (batch, seq, vocab)
    response_tokens = input_ids[:, response_start_idx:]   # (batch, response_len)
    # Gather log probs at position t-1 for token at position t
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

Pre-computes reference log probs during initialization:

```python
dataset = DPODataset(pairs, ref_model)
# pairs = [{"prompt": [...], "chosen": [...], "rejected": [...]}, ...]
# Internally computes ref_chosen_log_probs and ref_rejected_log_probs
```

### DPOTrainer

Simple training loop (no RL complexity):

```python
trainer = DPOTrainer(model, ref_model, beta=0.1)
metrics = trainer.step(batch)
# Just: compute policy log probs, compute loss, backward, step
```

---

## Code Walkthrough

### Step 1: compute_log_probs

The core function that connects a language model to the DPO loss. For autoregressive models, logits at position t predict the token at position t+1:

```python
# logits[:, t-1, :] predicts token at position t
# So log prob of response token at position t uses logits at position t-1
log_probs_all = F.log_softmax(logits, dim=-1)
response_log_probs = log_probs_all[:, start-1:end-1, :].gather(-1, tokens)
return response_log_probs.sum(dim=-1)
```

### Step 2: DPOLoss

The elegant DPO loss that eliminates the reward model:

```python
# Implicit reward: r(x,y) = beta * (log pi(y|x) - log pi_ref(y|x))
# DPO loss: -log sigmoid(r(x, y_w) - r(x, y_l))
log_ratio_chosen = policy_chosen - ref_chosen
log_ratio_rejected = policy_rejected - ref_rejected
loss = -F.logsigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
```

### Step 3: DPODataset

Pre-computes reference log probs to avoid redundant computation during training:

```python
with torch.no_grad():
    chosen_lp = compute_log_probs(ref_model, chosen_ids, prompt_len)
    rejected_lp = compute_log_probs(ref_model, rejected_ids, prompt_len)
```

### Step 4: DPOTrainer

A simple supervised training loop (no RL complexity):

```python
policy_chosen = compute_log_probs(model, chosen_ids, start_idx)
policy_rejected = compute_log_probs(model, rejected_ids, start_idx)
loss = dpo_loss(policy_chosen, policy_rejected, ref_chosen, ref_rejected)
loss.backward()
optimizer.step()
```

---

## Training Tips

### Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `beta` | 0.1 - 0.5 | Temperature for implicit reward (higher = more conservative) |
| `lr` | 1e-7 - 5e-6 | Learning rate (lower than typical supervised fine-tuning) |
| `max_grad_norm` | 0.5 - 1.0 | Gradient clipping |

### Common Pitfalls

1. **Beta too high**: Makes the loss very sensitive to small differences, can cause training instability.

2. **Beta too low**: The policy won't learn much because the implicit reward differences are compressed.

3. **Learning rate too high**: DPO is sensitive to learning rates. Start low (5e-7) and increase carefully.

4. **Prompt/response alignment**: Make sure `response_start_idx` correctly splits prompt from response. Mismatches lead to nonsensical log probs.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/03_dpo/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 04-rl/03_dpo/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing DPO components yourself.

### Exercise Order

1. **`compute_log_probs`** -- Compute log probabilities of response tokens
2. **`DPOLoss.forward`** -- Implement the DPO loss function
3. **`DPODataset.__init__`** -- Pre-compute reference log probs
4. **`DPOTrainer.step`** -- Implement the training step

### Tips

- Start with `compute_log_probs`. Remember: logits at position t predict token at position t+1.
- For `DPOLoss`, use `F.logsigmoid` for numerical stability.
- In `DPODataset`, freeze the reference model and use `torch.no_grad()` during pre-computation.
- In `DPOTrainer.step`, the flow is: compute policy log probs -> compute loss -> backward -> step.

---

## Key Takeaways

1. **Your LM is a reward model.** DPO shows that the optimal policy under a KL constraint implicitly defines a reward. No separate reward model is needed.

2. **DPO is supervised learning.** Unlike PPO which requires RL, DPO is just minimizing a loss on preference pairs. This makes it simpler and more stable.

3. **The reference model provides the baseline.** The frozen reference model acts as the KL anchor, preventing the policy from diverging too far from the pretrained model.

4. **Beta controls the tradeoff.** Higher beta makes the implicit reward more sensitive to log prob differences, leading to more aggressive optimization. Lower beta is more conservative.

5. **Pre-computing saves time.** Since the reference model is frozen, we can pre-compute its log probs once during dataset initialization, avoiding redundant computation during training.

---

## Further Reading

- [Direct Preference Optimization: Your Language Model is Secretly a Reward Model (Rafailov et al., 2023)](https://arxiv.org/abs/2305.18290) -- The original DPO paper
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT, the RLHF baseline that DPO simplifies
- [Constitutional AI: Harmlessness from AI Feedback (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) -- Related work on AI feedback for alignment
