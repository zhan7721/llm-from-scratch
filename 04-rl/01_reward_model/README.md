# Reward Model

> **Module 04 -- Reinforcement Learning, Chapter 01**

A language model trained on next-token prediction does not inherently know what "good" output looks like from a human perspective. The reward model bridges this gap: it learns to score text quality by training on human preference data -- pairs of (chosen, rejected) responses. This score then drives reinforcement learning to steer the language model toward outputs humans prefer.

This chapter implements the core machinery: a reward model that wraps a transformer and outputs scalar scores, the Bradley-Terry pairwise ranking loss, and a dataset for preference pairs.

---

## Prerequisites

- Transformer language model basics (Module 01)
- PyTorch nn.Module, Dataset, DataLoader
- Basic understanding of RLHF concepts

## Files

| File | Purpose |
|------|---------|
| `reward_model.py` | Core implementation: RewardModel, BradleyTerryLoss, RewardDataset, train_reward_model |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is a Reward Model?

### From Preferences to Scores

In RLHF (Reinforcement Learning from Human Feedback), the reward model is trained on human preferences. Humans compare two responses to the same prompt and indicate which one they prefer. The reward model learns to assign higher scores to preferred responses.

```
Prompt: "Explain quantum computing"
Response A: "Quantum computing uses qubits that can be 0 and 1 simultaneously..."
Response B: "idk lol"

Human preference: A is better
Reward model learns: score(A) > score(B)
```

### The Bradley-Terry Model

The Bradley-Terry model is the mathematical foundation for learning from pairwise preferences. It defines the probability that response A is preferred over response B as:

```
P(A > B) = sigmoid(r_A - r_B)
```

Where `r_A` and `r_B` are the reward scores assigned by the model. The training loss is the negative log-likelihood:

```
loss = -log(sigmoid(r_chosen - r_rejected))
```

This has nice properties:
- When `r_chosen >> r_rejected`: loss approaches 0 (model correctly ranks them)
- When `r_chosen << r_rejected`: loss is very high (model is wrong)
- When `r_chosen == r_rejected`: loss = log(2) (maximum uncertainty)

---

## Architecture

### RewardModel

Wraps any transformer model and adds a scalar reward head:

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

Key design choice: we use the **last token's hidden state** as the sequence representation. In autoregressive (GPT-style) models, the last token has attended to all previous tokens, so it contains the most complete representation of the entire sequence.

### BradleyTerryLoss

A simple but elegant loss function:

```python
def forward(self, chosen_rewards, rejected_rewards):
    loss = -F.logsigmoid(chosen_rewards - rejected_rewards)
    return loss.mean()
```

We use `F.logsigmoid` instead of `torch.log(torch.sigmoid(...))` for numerical stability. The `logsigmoid` function avoids the overflow/underflow issues that can occur when computing `log(sigmoid(x))` for large positive or negative values.

### RewardDataset

A PyTorch Dataset that loads preference pairs:

```python
pairs = [
    {"chosen": [token_ids...], "rejected": [token_ids...]},
    {"chosen": [token_ids...], "rejected": [token_ids...]},
    ...
]
```

Supports optional padding to a fixed `max_length` for batched training.

---

## Code Walkthrough

### Step 1: Wrap the Transformer

```python
model = RewardModel(transformer)
```

The RewardModel takes any transformer that returns hidden states of shape `(batch, seq, d_model)`. It automatically infers `d_model` from the transformer's attributes.

### Step 2: Compute Rewards

```python
hidden_states = self.transformer(input_ids)  # (batch, seq, d_model)
last_hidden = hidden_states[:, -1, :]          # (batch, d_model)
reward = self.scalar_head(last_hidden).squeeze(-1)  # (batch,)
```

The last token's hidden state is projected to a single scalar. This scalar is the reward score.

### Step 3: Compute Bradley-Terry Loss

```python
loss = -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

The loss encourages `r_chosen > r_rejected`. The gradient of this loss with respect to the model parameters pushes the reward higher for chosen responses and lower for rejected responses.

### Step 4: Training Loop

```python
chosen_rewards = model(batch["chosen_input_ids"])
rejected_rewards = model(batch["rejected_input_ids"])
loss = loss_fn(chosen_rewards, rejected_rewards)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

Standard PyTorch training loop. The key insight is that we compute rewards for both chosen and rejected responses in the same forward pass (they are independent), then combine them with the pairwise loss.

---

## Training Tips

### Learning Rate

Reward model training typically uses:
- Full fine-tuning: 1e-5 to 5e-5
- With a pre-trained transformer: start lower (1e-5) to avoid catastrophic forgetting

### Data Requirements

Reward models are data-hungry. Typical numbers:
- Minimum viable: 1K-5K preference pairs
- Production quality: 10K-100K+ pairs
- More diverse prompts = better generalization

### Common Pitfalls

1. **Reward hacking**: The model finds ways to get high scores without actually being better (e.g., longer responses get higher scores). Mitigate with diverse training data and regularization.

2. **Overfitting**: Reward models can memorize training pairs. Use dropout, weight decay, and early stopping.

3. **Distribution shift**: The reward model is trained on a fixed dataset, but during RL training, the policy generates responses that may look different from the training data. This is a fundamental challenge in RLHF.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 04-rl/01_reward_model/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 04-rl/01_reward_model/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing reward model components yourself.

### Exercise Order

1. **`BradleyTerryLossExercise.forward`** -- Implement the pairwise ranking loss using logsigmoid
2. **`RewardModelExercise.forward`** -- Compute reward scores from transformer hidden states
3. **`RewardDatasetExercise.__getitem__`** -- Load and pad preference pairs
4. **`train_reward_model_exercise`** -- Write the training loop

### Tips

- Start with `BradleyTerryLoss`. It is a one-line implementation using `F.logsigmoid`.
- For `RewardModel.forward`, the key insight is: take the last token's hidden state and project it to a scalar.
- For `RewardDataset.__getitem__`, handle both the padding and non-padding cases.
- For `train_reward_model`, follow the standard PyTorch training loop pattern.

---

## Key Takeaways

1. **The reward model learns from preferences, not labels.** It trains on pairs of (chosen, rejected) responses, learning to score chosen responses higher.

2. **Bradley-Terry is the standard preference model.** P(A > B) = sigmoid(r_A - r_B). The loss is -log(sigmoid(r_chosen - r_rejected)).

3. **Use the last token's hidden state.** In autoregressive models, the last token has the most context and is the natural choice for sequence-level representation.

4. **logsigmoid for numerical stability.** Never compute log(sigmoid(x)) directly -- use F.logsigmoid to avoid overflow/underflow.

5. **The reward model is the foundation for RLHF.** Once trained, its scores drive the RL training loop (PPO, DPO, etc.) to align the language model with human preferences.

---

## Further Reading

- [Training Language Models to Follow Instructions with Human Feedback (Ouyang et al., 2022)](https://arxiv.org/abs/2203.02155) -- The InstructGPT paper that popularized RLHF
- [Learning to Summarize from Human Feedback (Stiennon et al., 2020)](https://arxiv.org/abs/2009.01325) -- Early work on reward models for summarization
- [Constitutional AI (Bai et al., 2022)](https://arxiv.org/abs/2212.08073) -- Using AI feedback instead of human feedback
- [Scaling Laws for Reward Model Overoptimization (Gao et al., 2023)](https://arxiv.org/abs/2210.10760) -- How reward hacking scales with model size
