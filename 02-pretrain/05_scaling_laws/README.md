# Scaling Laws

Scaling laws describe how language model performance (measured as test loss)
improves predictably as you increase model size, training data, or compute.
They let you **forecast** the result of an expensive training run before
spending the money.

## Why Scaling Laws Matter

Before scaling laws, choosing model size and training duration was largely
guesswork. Kaplan et al. (2020) showed that loss follows smooth power laws
across several orders of magnitude. This means:

- You can train small models, fit a curve, and predict large-model performance.
- You can decide how to split a fixed compute budget between parameters and data.
- You can set realistic expectations for a training run before it starts.

## The Power-Law Form

Kaplan et al. observed that cross-entropy loss follows:

```
L(x) = a * x^(-alpha) + L_inf
```

Where:
- `x` is the scaling variable (parameter count N or token count D).
- `a` is a constant that depends on the task.
- `alpha` is the scaling exponent (how fast loss drops with scale).
- `L_inf` is the irreducible loss floor (~1.5 nats for natural language).

The key insight: plotting `log(L - L_inf)` against `log(x)` gives a straight
line. This makes fitting and extrapolation straightforward.

### Fitting in Practice

```python
import numpy as np

# Observed data: (params, loss)
x = np.array([1e6, 1e7, 1e8, 1e9])
loss = np.array([3.5, 3.0, 2.7, 2.5])
l_inf = 1.5

adjusted = loss - l_inf
log_x = np.log(x)
log_y = np.log(adjusted)

coeffs = np.polyfit(log_x, log_y, 1)
alpha = -coeffs[0]
a = np.exp(coeffs[1])
```

## Compute Estimation

Training compute is approximately:

```
C ≈ 6 * N * D
```

Where:
- N = number of parameters
- D = number of training tokens
- The factor of 6 accounts for forward pass (2), backward pass (4).

This is a simplification. Actual FLOPs depend on architecture details, but
this approximation is widely used for planning.

## Kaplan et al. (2020)

The original OpenAI paper found three separate power laws:

1. **L(N)**: Loss vs. parameter count (with data scaled proportionally).
2. **L(D)**: Loss vs. dataset size (with model scaled proportionally).
3. **L(C)**: Loss vs. compute budget (with optimal allocation).

Key findings:
- Performance improves smoothly with each factor.
- **N matters more than D**: Kaplan recommended allocating more compute to
  larger models rather than more training data.
- Optimal scaling: N_opt ∝ C^0.73, D_opt ∝ C^0.27.

This led to the strategy of training very large models on relatively modest
datasets (the GPT-3 approach).

## Chinchilla (Hoffmann et al. 2022)

DeepMind's Chinchilla paper revisited the scaling laws with more careful
experiments and reached a different conclusion:

- **Data matters more than Kaplan thought.**
- The optimal ratio is roughly **D ≈ 20 * N** (20 tokens per parameter).
- For a fixed compute budget, a smaller model trained on more data
  outperforms a larger model trained on fewer tokens.

### Chinchilla Allocation

Given a compute budget C, the optimal allocation is:

```
C = 6 * N * D     (compute constraint)
D = 20 * N        (Chinchilla ratio)

Solving: N = sqrt(C / (6 * 20))
         D = 20 * N
```

Example: with 1e24 FLOPs:
- N_opt ≈ 9.1B parameters
- D_opt ≈ 183B tokens

### Impact

Chinchilla changed industry practice. Models like LLaMA were trained with
the Chinchilla ratio in mind, using far more tokens than GPT-3 for their
size. The result: smaller models that match or beat larger predecessors.

## Practical Use

### Planning a Training Run

1. **Estimate your compute budget** in FLOPs (GPU count * GPU FLOPS * time).
2. **Use Chinchilla allocation** to find optimal N and D.
3. **Fit scaling law** from small pilot runs to predict final loss.
4. **Decide if it's worth it**: compare predicted loss to your target.

### Example: Plan a 7B Model

```
Target: 7B parameters
Chinchilla ratio: D = 20 * 7B = 140B tokens
Compute: C = 6 * 7e9 * 140e9 = 5.88e21 FLOPs
```

That's roughly 5.88 zettaFLOPs, or about 800 GPU-days on A100s.

### Setting Expectations

If you've trained models at 100M, 300M, and 1B parameters, you can fit a
scaling law and predict what a 7B model will achieve. This helps you:

- Decide whether to scale up at all.
- Choose between spending compute on a bigger model vs. more data.
- Report expected performance to stakeholders before training.

## Limitations

### Emergent Abilities

Some capabilities appear suddenly at certain scales (in-context learning,
chain-of-thought reasoning). Scaling laws predict **average loss**, not
specific capabilities. A model can have lower loss but still fail at tasks
that require a minimum scale.

### Diminishing Returns

The power-law exponent alpha is typically 0.05--0.1 for loss vs. compute.
This means you need roughly **10x more compute** to reduce loss by 5--10%.
Beyond a certain point, the cost may not justify the improvement.

### Architecture Dependence

Scaling laws are measured for specific architectures. A transformer scaling
law does not directly apply to Mamba, mixture-of-experts, or other
architectures. Each family needs its own measurements.

### Training Details Matter

Hyperparameter tuning, data quality, and training stability all affect the
actual loss. Scaling laws give you a **ceiling** (what's achievable with
optimal settings), not a guarantee.

## Code Walkthrough

### `ScalingLaw` Class

Fits the power-law model `L(x) = a * x^(-alpha) + L_inf` to data points
via log-linear regression. After fitting, `predict(x)` extrapolates to
new scales.

### `estimate_compute`

Simple formula: `C = 6 * N * D`. Useful for quick planning.

### `optimal_allocation_chinchilla`

Solves the system of equations `C = 6ND` and `D = 20N` to find the
Chinchilla-optimal parameter count and token count for a given budget.

### `optimal_allocation_kaplan`

Implements Kaplan's allocation where N scales as C^0.73 and D as C^0.27.
This favors larger models with less data compared to Chinchilla.

### `compare_allocations`

Side-by-side comparison of both strategies for the same compute budget.

## Running Tests

```bash
cd 02-pretrain/05_scaling_laws
pytest tests.py -v
```

## Exercises

Open `exercise.py` and implement the TODO items:

1. `ScalingLaw.fit` -- linearize and fit the power law.
2. `estimate_compute` -- implement the FLOPs formula.
3. `optimal_allocation_chinchilla` -- solve for optimal N and D.

Check your work with `tests.py`.

## References

- Kaplan, J. et al. (2020). "Scaling Laws for Neural Language Models."
  arXiv:2001.08361.
- Hoffmann, J. et al. (2022). "Training Compute-Optimal Large Language
  Models" (Chinchilla). arXiv:2203.15556.
- Touvron, H. et al. (2023). "LLaMA: Open and Efficient Foundation
  Language Models." arXiv:2302.13971.
