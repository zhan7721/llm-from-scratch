# Proximal Policy Optimization (PPO)

> **Module 04 -- Reinforcement Learning, Chapter 02**

PPO is the workhorse algorithm for RLHF (Reinforcement Learning from Human Feedback). After a reward model scores text quality (Chapter 01), PPO fine-tunes the language model to maximize those rewards while staying close to the original model. It does this through a clever clipped objective that prevents catastrophic policy updates.

This chapter implements the core PPO components: the clipped surrogate loss, Generalized Advantage Estimation (GAE), the PPO trainer, and a rollout function for generating responses.

---

## Prerequisites

- Reward model basics (Module 04, Chapter 01)
- PyTorch nn.Module, autograd
- Basic understanding of policy gradients and advantage functions

## Files

| File | Purpose |
|------|---------|
| `ppo.py` | Core implementation: PPOClipLoss, GAE, PPOTrainer, rollout |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is PPO?

### The Problem with Vanilla Policy Gradients

In policy gradient methods, we update the policy to increase the probability of good actions:

```
gradient = E[advantage * grad(log pi(a|s))]
```

But large updates can destroy the policy. If we increase the probability of an action too much, the policy might overfit to recent experiences and forget everything else.

### PPO's Solution: Clipped Surrogate

PPO prevents large updates by clipping the probability ratio:

```
ratio = pi_new(a|s) / pi_old(a|s)
surr1 = ratio * advantages
surr2 = clip(ratio, 1-eps, 1+eps) * advantages
loss = -min(surr1, surr2)
```

When `ratio > 1+eps` (policy changed too much in the positive direction):
- `surr2 = (1+eps) * advantages` caps the objective
- The gradient cannot push the policy further

When `ratio < 1-eps` (policy changed too much in the negative direction):
- `surr2 = (1-eps) * advantages` floors the objective
- The gradient cannot pull the policy further back

This creates a "trust region" where the policy can safely improve.

### KL Penalty

In RLHF, we also add a KL divergence penalty from a reference model (usually the pretrained model):

```
loss = policy_loss + kl_weight * KL(pi || pi_ref)
```

This prevents the policy from diverging too far from the pretrained model, which would make the output incoherent.

---

## Architecture

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

### GAE (Generalized Advantage Estimation)

GAE computes advantage estimates that balance bias and variance:

```python
delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)    # TD error
A_t = delta_t + (gamma * lambda) * A_{t+1}       # Accumulated advantage
```

- `lambda = 0`: One-step TD (high bias, low variance)
- `lambda = 1`: Monte Carlo (low bias, high variance)
- Typical: `lambda = 0.95`, `gamma = 0.99`

### PPOTrainer

Orchestrates the full PPO update:

```python
trainer = PPOTrainer(model, ref_model, clip_eps=0.2, kl_weight=0.1)
metrics = trainer.step(sequences, old_log_probs, rewards, values, prompt_len)
```

### rollout

Generates responses from the policy and collects log probabilities:

```python
result = rollout(model, prompt, max_new_tokens=50, vocab_size=32000)
# result["sequences"], result["log_probs"], result["values"]
```

---

## Code Walkthrough

### Step 1: PPO Clip Loss

The clip loss is the heart of PPO. It computes two surrogate objectives and takes the minimum (pessimistic bound):

```python
ratio = (log_probs - old_log_probs).exp()
surr1 = ratio * advantages
surr2 = ratio.clamp(1 - eps, 1 + eps) * advantages
loss = -torch.min(surr1, surr2) + kl_weight * kl
```

### Step 2: GAE

GAE computes advantages by backward accumulation of TD errors:

```python
for t in reversed(range(seq_len)):
    advantage = delta_t + gamma * lambda * advantage
    advantages[:, t] = advantage
```

### Step 3: Rollout

Generate tokens autoregressively, collecting log probs and values:

```python
for _ in range(max_new_tokens):
    logits, values = model(sequences)
    next_token = sample(logits[:, -1, :] / temperature)
    sequences = cat([sequences, next_token])
```

### Step 4: PPO Update

The trainer performs multiple epochs of PPO updates:

```python
for epoch in range(ppo_epochs):
    new_log_probs = compute_log_probs(model, sequences)
    ref_log_probs = compute_log_probs(ref_model, sequences)
    loss, kl = ppo_loss(new_log_probs, old_log_probs, advantages, ref_log_probs)
    loss.backward()
    optimizer.step()
```

---

## Training Tips

### Hyperparameters

| Parameter | Typical Range | Description |
|-----------|---------------|-------------|
| `clip_eps` | 0.1 - 0.3 | Clipping parameter (0.2 is common) |
| `kl_weight` | 0.01 - 0.5 | KL penalty weight |
| `gamma` | 0.99 | Discount factor |
| `lam` | 0.95 | GAE lambda |
| `lr` | 1e-6 - 1e-5 | Learning rate |
| `ppo_epochs` | 2 - 4 | PPO update epochs per batch |

### Common Pitfalls

1. **Reward hacking**: The policy finds ways to get high rewards without actually being better. Use KL penalty and diverse reward signals.

2. **KL divergence explosion**: If `kl_weight` is too low, the policy can diverge from the reference model, producing incoherent text. Monitor KL and increase `kl_weight` if needed.

3. **Value function accuracy**: If the value function is inaccurate, advantage estimates will be noisy, leading to unstable training. Consider pre-training the value function.

4. **Learning rate sensitivity**: PPO is sensitive to learning rate. Too high causes instability; too low causes slow convergence.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/02_ppo/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 04-rl/02_ppo/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing PPO components yourself.

### Exercise Order

1. **`PPOClipLossExercise.forward`** -- Implement the clipped surrogate objective
2. **`GAEExercise.forward`** -- Implement Generalized Advantage Estimation
3. **`rollout_exercise`** -- Implement autoregressive generation with log prob collection

### Tips

- Start with `PPOClipLoss`. The key formula is: `-min(ratio * A, clip(ratio) * A)`.
- For `GAE`, work backwards from the last timestep. The recurrence is: `A_t = delta_t + gamma * lambda * A_{t+1}`.
- For `rollout`, use `torch.multinomial` for sampling and `log_softmax` + `gather` for log probs.
- Use `model.eval()` during generation to disable dropout.

---

## Key Takeaways

1. **PPO clips the policy update.** By taking `min(surr1, surr2)`, PPO ensures the policy cannot change too much in a single update. This is the key insight that makes PPO stable.

2. **GAE balances bias and variance.** The lambda parameter controls the tradeoff between one-step TD (biased but low variance) and Monte Carlo (unbiased but high variance). Lambda=0.95 is a good default.

3. **KL penalty prevents divergence.** In RLHF, we add a KL penalty from the reference model to keep the policy close to the pretrained model. This prevents reward hacking and maintains text quality.

4. **PPO is an iterative algorithm.** Each batch of experience is used for multiple PPO update epochs. This data efficiency is important because generating rollouts is expensive for language models.

5. **The ratio is the key quantity.** `ratio = pi_new / pi_old` measures how much the policy has changed. PPO's clipping ensures this ratio stays close to 1.

---

## Further Reading

- [Proximal Policy Optimization Algorithms (Schulman et al., 2017)](https://arxiv.org/abs/1707.06347) -- The original PPO paper
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT, which popularized PPO for RLHF
- [High-Dimensional Continuous Control Using Generalized Advantage Estimation (Schulman et al., 2016)](https://arxiv.org/abs/1506.02438) -- The GAE paper
- [Some Things You Should Know About Proximal Policy Optimization (Huang et al., 2022)](https://arxiv.org/abs/2209.00796) -- Practical tips for PPO
