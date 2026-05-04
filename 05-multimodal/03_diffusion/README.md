# Denoising Diffusion Probabilistic Models (DDPM)

> **Module 05 -- Multimodal, Chapter 03**

Denoising Diffusion Probabilistic Models (Ho et al., 2020) generate images by learning to reverse a gradual noising process. The forward process progressively adds Gaussian noise to clean images over T timesteps, and the reverse process learns to denoise step-by-step, recovering clean images from pure noise.

---

## Prerequisites

- Basic understanding of neural networks and backpropagation
- PyTorch basics: `nn.Module`, `nn.Conv2d`, tensor operations
- Understanding of Gaussian distributions and noise

## Files

| File | Purpose |
|------|---------|
| `diffusion.py` | Core DDPM implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

DDPM generates images through a two-phase process:

```
Training Phase:
    Clean Image x_0
        |
        v
    Forward Diffusion (add noise gradually)
        |
        v
    Noisy Image x_t at random timestep t
        |
        v
    UNet predicts noise epsilon_theta(x_t, t)
        |
        v
    Loss = MSE(predicted_noise, actual_noise)

Generation Phase:
    Pure Noise x_T ~ N(0, I)
        |
        v
    Iterative Denoising (T steps)
        |
        v
    Clean Image x_0
```

### Key Insight: Learn to Denoise

The core idea is simple: instead of learning to generate images directly, learn to remove noise. If you can remove a small amount of noise at each step, you can iteratively denoise pure noise into a clean image.

---

## NoiseScheduler: The Diffusion Schedule

The noise scheduler controls how much noise is added at each timestep.

### Forward Process (Adding Noise)

```python
q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

This means:
- `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`
- As `t` increases, `alpha_bar_t` decreases, so more noise is added
- At `t=0`: almost clean (`alpha_bar_0 ~ 1`)
- At `t=T`: almost pure noise (`alpha_bar_T ~ 0`)

### Linear Beta Schedule

```python
beta_t = beta_start + (beta_end - beta_start) * t / T
```

- `beta_t` controls the noise variance at step `t`
- Linear schedule: `beta_t` increases linearly from `beta_start` to `beta_end`
- Typical values: `beta_start = 1e-4`, `beta_end = 0.02`

### Key Quantities

| Symbol | Formula | Meaning |
|--------|---------|---------|
| `beta_t` | Linear schedule | Noise variance at step t |
| `alpha_t` | `1 - beta_t` | Signal preservation at step t |
| `alpha_bar_t` | `prod(alpha_1...alpha_t)` | Cumulative signal preservation |
| `sqrt(alpha_bar_t)` | `sqrt(prod(alpha_1...alpha_t))` | Signal coefficient |
| `sqrt(1 - alpha_bar_t)` | `sqrt(1 - prod(alpha_1...alpha_t))` | Noise coefficient |

### Reverse Process (Denoising)

```python
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z
```

Where:
- `eps_theta(x_t, t)` is the predicted noise from the UNet
- `sigma_t` is the posterior standard deviation
- `z ~ N(0, I)` is random noise (except at t=0)

---

## UNet: Noise Prediction Network

The UNet predicts the noise added to an image at a given timestep.

### Architecture

```
Input: noisy image x_t (B, C, H, W) + timestep t
    |
    v
Time Embedding: sinusoidal embedding -> MLP
    |
    v
Initial Conv: Conv2d(C, base_channels, 3)
    |
    v
Encoder (Downsampling):
    Level 1: ResBlock x N -> Skip connection
        |
        v (stride-2 conv)
    Level 2: ResBlock x N -> Skip connection
        |
        v (stride-2 conv)
    Level 3: ResBlock x N -> Skip connection
    |
    v
Bottleneck:
    ResBlock -> ResBlock
    |
    v
Decoder (Upsampling):
    Level 3: Concat(skip) -> ResBlock x N
        |
        v (transpose conv)
    Level 2: Concat(skip) -> ResBlock x N
        |
        v (transpose conv)
    Level 1: Concat(skip) -> ResBlock x N
    |
    v
Final: GroupNorm -> SiLU -> Conv2d(base_channels, C, 3)
    |
    v
Output: predicted noise epsilon (B, C, H, W)
```

### Time Embedding

Sinusoidal embeddings (like Transformer positional encodings) encode the timestep into a dense vector:

```python
emb = [sin(t * freq_1), cos(t * freq_1), sin(t * freq_2), cos(t * freq_2), ...]
```

This allows the model to know which timestep it's denoising.

### ResBlock

Each residual block:
1. Apply GroupNorm + SiLU + Conv2d
2. Add time embedding (broadcast across spatial dimensions)
3. Apply GroupNorm + SiLU + Conv2d
4. Add skip connection

### Skip Connections

The UNet's key feature: encoder features are concatenated with decoder features at each resolution level. This preserves spatial information lost during downsampling.

---

## DDPMTrainer: Training Loop

The training objective is simple:

```python
# 1. Sample clean image
x_0 = sample_from_dataset()

# 2. Sample random timestep
t = randint(0, T)

# 3. Sample noise
epsilon = randn_like(x_0)

# 4. Create noisy image
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

# 5. Predict noise
predicted_epsilon = model(x_t, t)

# 6. Compute loss
loss = MSE(predicted_epsilon, epsilon)
```

The model learns to predict the noise that was added. This is equivalent to learning the score function (gradient of the log probability density).

---

## ddpm_sample: Generation

Generation iteratively denoises from pure noise:

```python
# Start from pure noise
x_T = randn(shape)

# Denoise from T-1 to 0
for t in reversed(range(T)):
    predicted_noise = model(x_t, t)
    x_{t-1} = reverse_step(predicted_noise, t, x_t)

# Final output
return x_0
```

Each step removes a small amount of noise. After T steps, we have a clean image.

---

## Shape Trace

For a batch of 2 images (3x16x16) with base_channels=16, channel_mults=(1,2):

```
Input:              (2, 3, 16, 16)    -- noisy images
Time Embedding:     (2, 32)           -- timestep embeddings
Init Conv:          (2, 16, 16, 16)   -- initial features

Encoder Level 1:    (2, 16, 16, 16)   -- ResBlock
  Skip 1:           (2, 16, 16, 16)
  Downsample:       (2, 16, 8, 8)

Encoder Level 2:    (2, 32, 8, 8)     -- ResBlock
  Skip 2:           (2, 32, 8, 8)

Bottleneck:         (2, 32, 8, 8)     -- 2x ResBlock

Decoder Level 2:    (2, 32+32, 8, 8)  -- concat with skip 2
  ResBlock:         (2, 32, 8, 8)
  Upsample:         (2, 32, 16, 16)

Decoder Level 1:    (2, 32+16, 16, 16) -- concat with skip 1
  ResBlock:         (2, 16, 16, 16)

Final Conv:         (2, 3, 16, 16)    -- predicted noise
```

---

## DDPM vs Other Generative Models

| Aspect | GAN | VAE | DDPM |
|--------|-----|-----|------|
| Training | Adversarial (min-max) | ELBO optimization | Simple MSE loss |
| Stability | Mode collapse, training instabilities | Posterior collapse | Stable, reproducible |
| Quality | Sharp images | Blurry images | High quality |
| Speed | Single forward pass | Single forward pass | Many steps (slow) |
| Diversity | Limited by mode collapse | Good | Excellent |
| Likelihood | No | Lower bound | Lower bound |

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/03_diffusion/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 05-multimodal/03_diffusion/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing DDPM yourself.

### Exercise Order

1. **NoiseScheduler**: Implement beta schedule, alpha_bar computation, add_noise, and reverse step
2. **SinusoidalTimeEmbedding**: Implement sinusoidal positional encoding for timesteps
3. **ResBlock**: Implement residual block with time embedding injection
4. **DDPMTrainer**: Implement the training loop with noise prediction loss

### Tips

- The forward process is just a weighted sum: `x_t = signal * x_0 + noise * epsilon`
- The reverse process predicts and removes noise at each step
- Time embeddings let the model know which timestep it's working with
- Skip connections in the UNet preserve spatial information
- The training loss is simply MSE between predicted and actual noise

---

## Key Takeaways

1. **DDPM learns to reverse noise.** Instead of generating images directly, learn to denoise step-by-step.

2. **The forward process is tractable.** We can compute `x_t` for any `t` in closed form using `alpha_bar_t`.

3. **The reverse process is learned.** A UNet predicts the noise at each step, which is used to denoise.

4. **Training is simple.** Just sample a timestep, add noise, predict the noise, and compute MSE loss.

5. **Generation is iterative.** Start from pure noise and denoise for T steps. This is slow but produces high-quality samples.

6. **Time embeddings are crucial.** The model needs to know which timestep it's denoising to predict the correct amount of noise.

---

## Further Reading

- [DDPM Paper (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) -- Original DDPM paper
- [Improved DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) -- Improved noise schedule and sampling
- [Denoising Diffusion-based Generative Modeling: Foundations and Applications (2023)](https://arxiv.org/abs/2303.14643) -- Comprehensive tutorial
- [Score-Based Generative Modeling (Song & Ermon, 2019)](https://arxiv.org/abs/1907.05600) -- Connection to score matching
- [What are Diffusion Models? (Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) -- Excellent blog post explanation
