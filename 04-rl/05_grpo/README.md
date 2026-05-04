# Group Relative Policy Optimization (GRPO)

> **Module 04 -- Reinforcement Learning, Chapter 05**

GRPO is a simpler alternative to PPO for RLHF. Instead of training a value network to estimate advantages, GRPO generates multiple responses per prompt and normalizes rewards within each group. This eliminates the need for a value function entirely, making the algorithm simpler and often more stable.

This chapter implements the core GRPO components: group-relative advantage computation, the clipped surrogate loss, a GRPO trainer, and a convenience function for single training steps.

---

## Prerequisites

- Reward model basics (Module 04, Chapter 01)
- PPO concepts (Module 04, Chapter 02)
- PyTorch nn.Module, autograd

## Files

| File | Purpose |
|------|---------|
| `grpo.py` | Core implementation: compute_group_advantages, GRPOLoss, GRPOTrainer, grpo_step |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is GRPO?

### The Problem with PPO for RLHF

PPO requires a value network (critic) to estimate advantages. This adds complexity:
- You need to train both a policy and a value network
- The value network can be inaccurate, leading to noisy advantage estimates
- More parameters, more computation, more hyperparameters to tune

### GRPO's Solution: Group-Relative Advantages

Instead of learning a value function, GRPO generates G responses for each prompt and uses the empirical reward distribution within the group:

```
For each prompt:
    1. Generate G responses: y_1, y_2, ..., y_G
    2. Score each response: r_1, r_2, ..., r_G
    3. Compute group advantages:
       advantage_i = (r_i - mean(r)) / (std(r) + eps)
```

Key properties:
- Advantages sum to zero within each group (by construction)
- Higher-reward responses get positive advantages (reinforced)
- Lower-reward responses get negative advantages (discouraged)
- No value network needed!

---

## Architecture

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

## Code Walkthrough

### Step 1: Group Advantages

The core innovation of GRPO. For each prompt with G responses:

```python
rewards = tensor([[r1, r2, r3, r4],   # prompt 1
                   [r1, r2, r3, r4]])  # prompt 2
advantages = (rewards - rewards.mean(dim=1, keepdim=True)) / (rewards.std(dim=1, keepdim=True) + eps)
# Each row sums to zero
```

### Step 2: GRPO Loss

Same clipped surrogate as PPO, but with group-relative advantages:

```python
ratio = (log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
loss = -min(surr1, surr2) + kl_weight * KL
```

### Step 3: GRPO Training Loop

```python
for each training step:
    for each prompt:
        generate G responses
        score with reward function
    compute group advantages
    update policy with GRPO loss
```

---

## GRPO vs PPO

| Aspect | PPO | GRPO |
|--------|-----|------|
| Value network | Required | Not needed |
| Advantage estimation | GAE with learned V(s) | Reward normalization within group |
| Complexity | Higher (policy + value) | Lower (policy only) |
| Stability | Can be unstable if V is inaccurate | More stable (no V estimation error) |
| Sample efficiency | Higher (reuses data with GAE) | Lower (needs fresh samples each step) |

---

## Training Tips

### Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `clip_eps` | 0.1 - 0.3 | Clipping parameter (0.2 is common) |
| `kl_weight` | 0.01 - 0.5 | KL penalty weight |
| `lr` | 1e-6 - 1e-5 | Learning rate |
| `num_responses_per_prompt` | 4 - 16 | Group size G |

### Common Pitfalls

1. **Small groups**: With G=2, the advantage signal is very coarse. Use G >= 4 for better signal.

2. **Reward scale**: If rewards have very different scales across prompts, the advantages can be noisy. Consider normalizing rewards across the batch.

3. **KL divergence explosion**: Same as PPO -- monitor KL and adjust `kl_weight` if needed.

4. **Generation cost**: Generating G responses per prompt is expensive. Balance G against batch size.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/05_grpo/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 04-rl/05_grpo/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing GRPO components yourself.

### Exercise Order

1. **`compute_group_advantages`** -- Normalize rewards within a group
2. **`GRPOLossExercise.forward`** -- Implement the clipped surrogate objective
3. **`grpo_step_exercise`** -- Implement a single GRPO training step

### Tips

- Start with `compute_group_advantages`. The key formula is: `(r - mean) / (std + eps)`.
- For `GRPOLoss`, it's identical to PPO's clip loss -- just with group advantages.
- For `grpo_step`, remember to expand 1D advantages to match the sequence dimension.
- Use `model.train()` before computing new log probs, and `model.eval()` for generation.

---

## Key Takeaways

1. **GRPO eliminates the value network.** By generating multiple responses and normalizing rewards within the group, GRPO gets advantage estimates without a learned value function. This is simpler and avoids value estimation errors.

2. **Group advantages sum to zero.** This is by construction: normalizing rewards within a group ensures the advantages are zero-centered. The relative ranking determines the training signal.

3. **GRPO uses the same clipped loss as PPO.** The only difference is where the advantages come from. The clipping mechanism and KL penalty work the same way.

4. **Larger groups give better signal.** With more responses per prompt, the advantage estimates are more stable. But this comes at the cost of more generation time.

5. **GRPO is especially good for math/code tasks.** Where rewards are well-defined (correct/incorrect), group-relative advantages provide a clean training signal without the noise of a learned value function.

---

## Further Reading

- [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (Shao et al., 2024)](https://arxiv.org/abs/2402.03300) -- The paper that introduced GRPO
- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) -- The PPO paper (GRPO builds on PPO's clipped objective)
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT, which popularized PPO for RLHF
