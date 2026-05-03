# Mixture of Experts (MoE)

> **Module 01 — Foundations, Chapter 06**

Standard transformers scale by making every layer wider and deeper, but this means every token passes through every parameter. Mixture of Experts (MoE) breaks that assumption: instead of one large feed-forward network, we have many smaller "expert" networks, and a learned router sends each token to only a few of them. The result is a model with many more total parameters but the same per-token compute as a smaller dense model.

This is how Mixtral 8x7B achieves quality comparable to much larger dense models while keeping inference fast.

---

## Prerequisites

- Transformer block structure (Chapter 04)
- Feed-forward networks and SwiGLU activation
- Softmax and top-K selection

## Files

| File | Purpose |
|------|---------|
| `moe.py` | Core implementation: TopKRouter, Expert, MoELayer |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is MoE and Why It Matters

### The Scaling Problem

To make a transformer more capable, you typically increase its parameter count. A 7B model is better than a 1B model, and a 70B model is better than a 7B model. But more parameters means more compute per token: every token flows through every weight matrix in every layer.

MoE breaks this tradeoff. By replacing the feed-forward network (FFN) in a transformer block with multiple smaller expert networks and a router, we can:

- **Increase total parameters** (more knowledge capacity)
- **Keep per-token compute constant** (each token only uses a few experts)
- **Scale model capacity without proportional cost**

### The Core Idea

Instead of one FFN per layer:

```
x -> FFN -> output
```

We have N expert FFNs, and a router picks the top-K for each token:

```
x -> Router -> [Expert_0, Expert_1, ..., Expert_N-1]
                  top-K selected, weighted average -> output
```

With 8 experts and top-2 routing, each token activates only 2 out of 8 experts. The per-token FLOPs are roughly the same as a single FFN, but the model has 8x the parameters in its FFN layers.

---

## Architecture

### TopKRouter

The router decides which experts handle each token. It is a simple linear layer:

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        logits = self.gate(x_flat)              # (B*T, n_experts)
        weights, indices = torch.topk(logits, self.top_k)  # top-K
        weights = F.softmax(weights, dim=-1)    # normalize
        return indices, weights
```

**How it works:**
1. Project each token vector to a score for each expert
2. Select the top-K experts by score
3. Softmax the selected scores to get routing weights (sum to 1)

The router is trained end-to-end with the rest of the model. It learns which experts are best for which types of tokens.

### Expert

Each expert is an independent SwiGLU feed-forward network:

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))
```

This is the same FFN design used in LLaMA and other modern LLMs. The SiLU-gated activation provides better gradient flow than ReLU.

### MoELayer

The full layer combines routing and expert computation:

```python
class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, top_k=2):
        self.router = TopKRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        indices, weights = self.router(x)

        for k in range(self.top_k):
            for e_idx in range(n_experts):
                mask = (indices[:, k] == e_idx)
                if mask.any():
                    expert_out = self.experts[e_idx](x_flat[mask])
                    output[mask] += weights[mask, k] * expert_out

        return output
```

The loop over experts looks inefficient (and it is for production), but it clearly shows the algorithm: for each routing slot, find which tokens go to each expert, run those tokens through the expert, and accumulate the weighted output.

---

## Load Balancing

### The Problem

Without any regularization, the router can collapse to always picking the same 1-2 experts. This is a form of "rich get richer": once an expert starts getting more tokens, it gets more training signal, gets better, and attracts even more tokens. The other experts starve.

This is called **expert collapse** or **routing imbalance**. It wastes capacity — you have 8 experts but only 2 are doing anything.

### Solutions

**1. Auxiliary Load Balancing Loss**

Add a penalty term that encourages uniform routing. The Switch Transformer paper introduced this:

```python
# fraction of tokens routed to each expert
f_i = fraction of tokens routed to expert i

# average routing weight for each expert
p_i = mean routing weight for expert i

# auxiliary loss (minimized when routing is uniform)
aux_loss = n_experts * sum(f_i * p_i)
```

This loss is added to the main training loss with a small coefficient (e.g., 0.01).

**2. Capacity Factor**

Limit the maximum number of tokens each expert can process. If an expert's capacity is full, overflow tokens are routed to the next-best expert or dropped. This prevents any single expert from being overwhelmed.

**3. Random Routing with Noise**

Add noise to the router logits before top-K selection. This encourages exploration and prevents premature collapse:

```python
logits = self.gate(x) + torch.randn_like(logits) * noise_std
```

**4. Expert Choice Routing**

Instead of each token choosing experts, each expert chooses its top tokens. This guarantees balanced load by construction but changes the routing semantics.

---

## MoE in Practice

### Mixtral 8x7B (Mistral AI, 2024)

- 8 experts, top-2 routing per layer
- 47B total parameters, ~13B active per token
- Matches or exceeds LLaMA 2 70B quality
- Uses SwiGLU experts, sliding window attention

### Switch Transformer (Google, 2022)

- Top-1 routing (each token goes to exactly one expert)
- Up to 1.6 trillion parameters
- Introduced the auxiliary load balancing loss
- Showed that MoE scales better than dense models at the same compute budget

### DeepSeek-V2 / DeepSeek-MoE (DeepSeek, 2024)

- Fine-grained experts: many small experts instead of few large ones
- Shared experts that always activate (capture common knowledge)
- Top-K routing with auxiliary loss
- Demonstrated that more, smaller experts can be more efficient

### GShard (Google, 2020)

- Top-2 routing with capacity factors
- One of the first large-scale MoE models (600B parameters)
- Introduced the random routing with noise trick

---

## Code Walkthrough

### TopKRouter

```python
class TopKRouter(nn.Module):
    def __init__(self, d_model, n_experts, top_k=2):
        self.gate = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        B, T, D = x.shape
        x_flat = x.view(-1, D)           # (B*T, D)
        logits = self.gate(x_flat)        # (B*T, n_experts)
        weights, indices = torch.topk(logits, self.top_k, dim=-1)
        weights = F.softmax(weights, dim=-1)
        return indices, weights
```

Key details:
- The gate is a single linear layer with no bias — it just projects token representations to expert scores
- `torch.topk` returns both the values and the indices of the top-K entries
- Softmax is applied only over the top-K values, not all experts — the weights for the selected experts sum to 1

### Expert (SwiGLU)

```python
class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w_gate(x)) * self.w1(x))
```

SwiGLU works by:
1. Projecting x through two parallel paths: `w1(x)` and `w_gate(x)`
2. Applying SiLU activation to the gate path: `SiLU(w_gate(x))`
3. Element-wise multiplying: `SiLU(w_gate(x)) * w1(x)` — this is the "gating"
4. Projecting back to d_model: `w2(...)`

The gating mechanism lets the network learn which features to pass through and which to suppress, providing better expressiveness than a simple ReLU FFN.

### MoELayer

```python
class MoELayer(nn.Module):
    def forward(self, x):
        B, T, D = x.shape
        indices, weights = self.router(x)   # route each token
        x_flat = x.view(-1, D)
        output = torch.zeros_like(x_flat)

        for k in range(self.top_k):         # for each routing slot
            for e_idx in range(n_experts):   # for each expert
                mask = (indices[:, k] == e_idx)
                if mask.any():
                    expert_input = x_flat[mask]
                    expert_output = self.experts[e_idx](expert_input)
                    output[mask] += weights[mask, k] * expert_output

        return output.view(B, T, D)
```

This is a clear but naive implementation. The double loop makes it easy to understand but slow. Production implementations use grouped matrix multiplications (grouped GEMM) or specialized CUDA kernels to process all experts in parallel.

---

## Trade-offs

### More Experts = More Capacity

Each expert can specialize in different types of input. With 8 experts, the model can learn 8 different "perspectives" on how to process tokens. This is like having 8 specialists instead of 1 generalist.

### Sparse Activation = Constant Compute

Even with 8 experts, each token only uses 2. The per-token FLOPs are roughly:
```
MoE FLOPs per token = 2 * (3 * d_model * d_ff)   # same as a single FFN
```
compared to a dense model with 8 experts:
```
Dense FLOPs per token = 8 * (3 * d_model * d_ff)  # 4x more
```

### Memory is the Bottleneck

While compute is sparse, memory is not. All expert parameters must be stored in GPU memory, even if most experts are idle for any given token. This is the main challenge for serving MoE models:
- Mixtral 8x7B has 47B parameters (needs ~94GB in fp16)
- But only ~13B are active per token (same compute as a 13B dense model)

### Load Balancing is Critical

Without proper regularization, expert collapse wastes capacity. The auxiliary loss adds a small training cost but is essential for the model to actually use all its experts.

### Communication in Distributed Settings

In multi-GPU setups, different experts may live on different devices. Routing tokens to experts on other GPUs requires inter-device communication, which can become a bottleneck. Expert parallelism (placing experts across GPUs) is a key engineering challenge for large MoE models.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 01-foundations/06_moe/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 01-foundations/06_moe/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing MoE yourself.

### Exercise Order

1. **`ExpertExercise.__init__`** — Create w1, w_gate, w2 linear layers
2. **`ExpertExercise.forward`** — Implement the SwiGLU forward pass
3. **`TopKRouterExercise.__init__`** — Create the gate linear layer
4. **`TopKRouterExercise.forward`** — Implement routing with top-K selection
5. **`MoELayerExercise.__init__`** — Create the router and expert list
6. **`MoELayerExercise.forward`** — Implement the full MoE forward pass

### Tips

- Start with the Expert. SwiGLU is just: `w2(SiLU(w_gate(x)) * w1(x))`. Three linear layers and one activation.
- The Router is simpler than it looks: one linear layer, then `torch.topk`, then softmax.
- The MoE forward pass is a loop: for each routing slot, for each expert, find matching tokens, run them through, accumulate. It is not efficient but it is correct and clear.
- `torch.topk` returns `(values, indices)` — the values are the scores, the indices are which experts were selected.

---

## Key Takeaways

1. **MoE scales model capacity without scaling per-token compute.** By routing each token to only a few experts, we get the knowledge capacity of a large model with the compute cost of a small one.

2. **The router is a learned gating network.** It projects each token to expert scores and selects the top-K. The routing weights are softmax-normalized over the selected experts.

3. **Experts are independent FFN networks.** Each expert is a SwiGLU feed-forward network that can specialize in different types of input.

4. **Load balancing prevents expert collapse.** Without regularization, the router can route all tokens to the same experts, wasting capacity. Auxiliary losses or capacity factors are needed.

5. **MoE is widely used in practice.** Mixtral, Switch Transformer, DeepSeek, and GShard all use MoE to achieve better scaling than dense models.

---

## Further Reading

- [Switch Transformer (Fedus et al., 2022)](https://arxiv.org/abs/2101.03961) — Top-1 routing, scaling laws for MoE
- [Mixtral of Experts (Jiang et al., 2024)](https://arxiv.org/abs/2401.04088) — Mixtral 8x7B architecture
- [GShard (Lepikhin et al., 2021)](https://arxiv.org/abs/2006.16668) — Large-scale MoE training
- [DeepSeek-MoE (Dai et al., 2024)](https://arxiv.org/abs/2401.06066) — Fine-grained MoE with shared experts
- [ST-MoE (Zoph et al., 2022)](https://arxiv.org/abs/2202.08906) — Stability and design choices for MoE
