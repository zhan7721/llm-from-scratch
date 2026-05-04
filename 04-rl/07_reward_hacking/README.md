# Reward Hacking

> **Module 04 -- Reinforcement Learning, Chapter 07**

When we optimize a language model against a reward model, the model may learn to exploit imperfections in the reward signal -- achieving high scores without genuinely improving quality. This is called **reward hacking** (or reward overoptimization). This chapter implements detection and mitigation tools.

---

## Prerequisites

- Reward Model basics (Module 04, Chapter 01)
- PyTorch nn.Module
- Basic understanding of RLHF training loops

## Files

| File | Purpose |
|------|---------|
| `reward_hacking.py` | Core implementation: RewardHackingDetector, KLConstrainedReward, RewardEnsemble, analyze_reward_hacking |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is Reward Hacking?

### The Problem

In RLHF, we train a policy to maximize the reward model's score. But the reward model is an imperfect proxy for human preferences. As we optimize harder against it, the policy may find "shortcuts" that score well without corresponding to real quality:

- **Length exploitation**: Longer outputs get higher scores, so the model rambles
- **Repetition exploitation**: Certain phrases score well, so the model repeats them
- **Style over substance**: Outputs look superficially good but lack coherence

This is analogous to Goodhart's Law: "When a measure becomes a target, it ceases to be a good measure."

### Real-World Examples

```
Reward model trained to prefer helpful responses:
  - Policy learns to add "I'd be happy to help!" at the start of every response
    (high reward, no actual helpfulness improvement)
  - Policy learns to produce very long responses with filler content
    (length correlates with reward in training data)
  - Policy learns specific phrases that trigger high reward scores
    (template exploitation)
```

---

## Architecture

### RewardHackingDetector

Monitors diagnostic signals during training to detect reward hacking early:

```python
detector = RewardHackingDetector(reward_threshold=2.0, diversity_threshold=0.3)
detector.set_baseline(reference_rewards)

# During training...
result = detector.detect(current_rewards, generated_tokens, expected_length=50)
if result["is_hacking"]:
    print("Warning: reward hacking detected!")
```

**Signals monitored:**
- **Reward divergence**: Z-score of current rewards vs baseline (high = suspicious)
- **Output diversity**: Unique-token ratio (low = repetitive = suspicious)
- **Length anomaly**: How far output lengths deviate from expected (extreme = suspicious)

### KLConstrainedReward

Adds a KL penalty to keep the policy near the reference:

```python
wrapper = KLConstrainedReward(reward_model, beta=0.1)
result = wrapper.forward(input_ids, policy_logits, reference_logits)
effective_reward = result["effective_reward"]
# effective_reward = reward - beta * KL(policy || reference)
```

The KL penalty prevents the policy from straying too far from the reference distribution, which limits its ability to exploit reward model weaknesses.

### RewardEnsemble

Averages multiple reward models for robustness:

```python
ensemble = RewardEnsemble([reward_model_1, reward_model_2, reward_model_3])
result = ensemble.forward(input_ids)
mean_reward = result["mean_reward"]  # Average across models
std_reward = result["std_reward"]    # Disagreement signal
```

If one reward model has a blind spot, the others can compensate. The standard deviation (disagreement) also serves as an uncertainty signal.

### analyze_reward_hacking

Diagnostic function comparing reward vs quality:

```python
result = analyze_reward_hacking(reward_model, quality_function, input_ids)
print(f"Correlation: {result['correlation']:.3f}")
print(f"Reward inflation: {result['reward_inflation']:.3f}")
print(f"Is hacking: {result['is_hacking']}")
```

---

## Code Walkthrough

### Step 1: Set Up Detection

```python
detector = RewardHackingDetector()
detector.set_baseline(reference_rewards)  # Rewards from initial policy
```

The detector needs a baseline to compare against. This is typically the reward distribution from the initial (unoptimized) policy.

### Step 2: Monitor During Training

```python
signals = detector.detect(current_rewards, generated_tokens)
if signals["is_hacking"]:
    # Reduce learning rate, increase KL penalty, or stop training
    beta *= 1.5  # Increase KL coefficient
```

### Step 3: Apply KL Constraint

```python
kl_wrapper = KLConstrainedReward(reward_model, beta=0.1)
result = kl_wrapper.forward(input_ids, policy_logits, ref_logits)
loss = -result["effective_reward"].mean()  # Maximize effective reward
```

### Step 4: Use Ensemble for Robustness

```python
ensemble = RewardEnsemble([rm1, rm2, rm3])
scores = ensemble.score(input_ids)  # More robust than any single model
```

---

## Key Concepts

### Reward Divergence

As the policy is optimized, its reward scores tend to increase. But at some point, the increase becomes "too good to be true" -- the policy is exploiting rather than genuinely improving. The z-score measures how many standard deviations the current mean reward has drifted from the baseline.

### KL Divergence Penalty

The KL divergence KL(pi || pi_ref) measures how different the current policy is from the reference. By adding `beta * KL` to the loss, we create a "leash" that prevents the policy from moving too far. The `beta` parameter controls the leash length:

- `beta` too low: Policy can exploit the reward model freely
- `beta` too high: Policy cannot improve at all (stuck at reference)
- `beta` just right: Policy improves while staying grounded

### Ensemble Disagreement

When multiple reward models agree, we can be more confident in the score. When they disagree, it signals uncertainty -- possibly because the input is adversarial or out-of-distribution. The standard deviation across models is a natural uncertainty measure.

---

## Training Tips

### Choosing Beta

Start with `beta = 0.01` to `0.1` and adjust based on the detector signals:
- If hacking is detected, increase beta
- If improvement stalls, decrease beta
- Monitor the KL penalty term -- it should be small but non-zero

### How Many Ensemble Models?

- 3-5 models is a practical sweet spot
- Models should be independently trained (different seeds, data subsets, or architectures)
- More models = better robustness but higher compute cost

### When to Use Each Tool

| Tool | When to Use |
|------|-------------|
| RewardHackingDetector | Always -- it is cheap and provides early warning |
| KLConstrainedReward | During RL training to prevent overoptimization |
| RewardEnsemble | When you have multiple reward models available |
| analyze_reward_hacking | As a diagnostic after training or during evaluation |

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/07_reward_hacking/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 04-rl/07_reward_hacking/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing reward hacking components.

### Exercise Order

1. **`RewardHackingDetectorExercise.compute_reward_divergence`** -- Z-score comparison to baseline
2. **`RewardHackingDetectorExercise.compute_output_diversity`** -- Unique-token ratio
3. **`RewardHackingDetectorExercise.compute_length_anomaly`** -- Length deviation detection
4. **`RewardHackingDetectorExercise.detect`** -- Combine all signals
5. **`KLConstrainedRewardExercise.compute_kl_penalty`** -- KL divergence from logits
6. **`KLConstrainedRewardExercise.forward`** -- Combine reward and KL penalty
7. **`RewardEnsembleExercise`** -- Average multiple reward models
8. **`analyze_reward_hacking_exercise`** -- Correlation-based diagnostic

### Tips

- Start with `compute_reward_divergence` -- it is a simple z-score computation.
- For `compute_output_diversity`, `torch.unique()` is your friend.
- The KL penalty uses `F.log_softmax` and `F.softmax` for numerical stability.
- The ensemble is straightforward: loop, stack, mean/std.

---

## Key Takeaways

1. **Reward hacking is inevitable under strong optimization.** The more you optimize against an imperfect reward model, the more the policy will exploit its weaknesses.

2. **KL constraints are the primary defense.** By penalizing divergence from a reference policy, we limit the policy's ability to find and exploit reward model blind spots.

3. **Ensembles reduce single-model vulnerabilities.** Averaging multiple reward models makes it harder for the policy to find exploits that work across all models.

4. **Detection is better than cure.** Monitor signals like reward divergence and output diversity during training. Early detection allows you to adjust before the policy degenerates.

5. **The correlation between reward and quality is the key metric.** If reward goes up but quality does not, you have reward hacking.

---

## Further Reading

- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2023)](https://arxiv.org/abs/2210.10760) -- How reward hacking scales with optimization strength
- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- InstructGPT, discusses KL constraints in RLHF
- [Concrete Problems in AI Safety (Amodei et al., 2016)](https://arxiv.org/abs/1606.06565) -- Reward hacking as a safety concern
- [Reward Model Ensembles (Coste et al., 2023)](https://arxiv.org/abs/2310.02743) -- Using ensembles for more robust reward estimation
