# Visual Language Model (VLM)

> **Module 05 -- Multimodal, Chapter 02**

We have built a Vision Transformer (ViT) in Chapter 01. Now we combine it with a language model to create a Visual Language Model that can understand both images and text. This chapter implements the core VLM architecture used in models like LLaVA and Flamingo.

---

## Prerequisites

- Vision Transformer (ViT) from Chapter 01
- Transformer architecture from Module 01
- PyTorch basics: `nn.Module`, `nn.Linear`, `dataclasses`

## Files

| File | Purpose |
|------|---------|
| `vlm.py` | Core VLM implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A Visual Language Model combines vision and language understanding. The key insight is to project vision features into the same embedding space as text tokens, then process them together with a language model:

```
Image (B, C, H, W)
    |
    v
Vision Encoder (ViT)
    |
    v
Vision Features (B, num_patches, vision_dim)
    |
    v
Vision Projector (MLP)
    |
    v
Vision Embeddings (B, num_patches, lm_dim)
    |
    +--- Concatenate with Text Embeddings ---+
    |                                         |
    v                                         v
[vision_tokens | text_tokens] (B, num_vision + seq_len, lm_dim)
    |
    v
Language Model (Transformer)
    |
    v
Logits (B, num_vision + seq_len, vocab_size)
```

---

## VLMConfig: Model Hyperparameters

```python
@dataclass
class VLMConfig:
    vision_dim: int = 768          # vision encoder output dimension
    lm_dim: int = 512              # language model embedding dimension
    lm_vocab_size: int = 32000     # language model vocabulary size
    lm_n_heads: int = 8            # language model attention heads
    lm_n_layers: int = 6           # language model transformer layers
    lm_max_seq_len: int = 1024     # max text sequence length
    vision_image_size: int = 224   # input image size
    vision_patch_size: int = 16    # vision encoder patch size
    vision_n_heads: int = 12       # vision encoder attention heads
    vision_n_layers: int = 12      # vision encoder transformer layers
```

---

## Architecture Components

### 1. Vision Encoder (ViT)

The vision encoder extracts features from images using the Vision Transformer from Chapter 01. It converts an image into a sequence of patch embeddings:

```python
# For a 224x224 image with 16x16 patches:
# num_patches = (224/16)^2 = 196 patches
# Plus 1 CLS token = 197 total vision tokens
# Output: (B, 197, vision_dim)
```

### 2. Vision Projector (MLP)

The projector maps vision features from `vision_dim` to `lm_dim` using a simple MLP:

```python
Linear(vision_dim, lm_dim) -> GELU -> Linear(lm_dim, lm_dim)
```

This is the standard approach used in LLaVA. The GELU activation provides non-linearity, and the two-layer MLP allows for more complex transformations than a single linear layer.

### 3. Language Model

The language model is a standard GPT-style transformer with:
- Token embedding
- Positional embedding (learned)
- N transformer blocks with causal attention
- Final layer norm
- Language model head

### 4. Concatenation Strategy

Vision tokens are **prepended** to text tokens. This means the sequence looks like:

```
[vision_token_1, vision_token_2, ..., vision_token_N, text_token_1, text_token_2, ...]
```

This allows the language model to attend to vision tokens when generating text, enabling visual understanding.

---

## Forward Pass Walkthrough

```python
def forward(self, images, input_ids):
    # 1. Extract vision features
    vision_features = self.vision_encoder(images)
    # (B, 3, 224, 224) -> (B, 197, vision_dim)

    # 2. Project to LM embedding space
    vision_embeddings = self.projector(vision_features)
    # (B, 197, vision_dim) -> (B, 197, lm_dim)

    # 3. Get text embeddings
    text_embeddings = self.token_emb(input_ids)
    # (B, seq_len) -> (B, seq_len, lm_dim)

    # 4. Concatenate (vision first)
    combined = torch.cat([vision_embeddings, text_embeddings], dim=1)
    # (B, 197 + seq_len, lm_dim)

    # 5. Add positional embeddings
    positions = torch.arange(combined.shape[1], device=combined.device)
    combined = combined + self.pos_emb(positions)

    # 6. Process through language model
    h = combined
    for layer in self.layers:
        h = layer(h)

    # 7. Final norm and project to vocabulary
    h = self.norm(h)
    logits = self.lm_head(h)
    # (B, 197 + seq_len, vocab_size)

    return logits
```

---

## Key Design Decisions

### Why MLP Projector?

The MLP projector (Linear + GELU + Linear) is used instead of a single linear layer because:
1. It provides non-linearity for better feature transformation
2. It's the standard approach in LLaVA and similar models
3. It's simple and effective

### Why Prepend Vision Tokens?

Vision tokens are prepended (not appended) because:
1. The causal attention mask allows text tokens to attend to all vision tokens
2. This is the standard approach in LLaVA and Flamingo
3. It enables the model to "see" the image before generating text

### Why Learned Positional Embeddings?

We use learned positional embeddings instead of RoPE because:
1. The sequence contains both vision and text tokens with different semantics
2. Learned embeddings can capture position-specific patterns
3. It's simpler to implement for educational purposes

---

## Parameter Counting

For a small VLM with `vision_dim=64`, `lm_dim=64`, 2 transformer layers:

**Vision Encoder (ViT):**
- Patch embedding: 3 * 16 * 16 * 64 = 49,152
- Transformer blocks: ~200K
- Total: ~250K

**Vision Projector:**
- MLP: 64 * 64 + 64 + 64 * 64 + 64 = 8,256

**Language Model:**
- Token embedding: 256 * 64 = 16,384
- Positional embedding: 133 * 64 = 8,512
- Transformer blocks: ~200K
- Total: ~225K

**Total:** ~500K parameters

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/02_vlm/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 05-multimodal/02_vlm/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the VLM yourself.

### Exercise Order

1. **VisionProjector**: Create the MLP projector
2. **VLM.__init__**: Create all components (vision encoder, projector, language model)
3. **VLM.forward**: Wire up the pipeline (vision -> project -> concat -> LM)

### Tips

- Start with the VisionProjector. It's a simple MLP with two linear layers.
- For the VLM, remember to calculate the number of vision tokens correctly (patches + CLS token).
- The positional embedding must be large enough for vision + text tokens.
- Don't forget weight tying between token_emb and lm_head.

---

## Key Takeaways

1. **VLMs combine vision and language.** The key is projecting vision features into the language model's embedding space.

2. **MLP projectors are simple and effective.** A two-layer MLP with GELU activation works well for vision-to-language projection.

3. **Vision tokens are prepended.** This allows text tokens to attend to vision tokens via causal attention.

4. **Positional embeddings handle mixed sequences.** Learned positional embeddings work well for sequences containing both vision and text tokens.

5. **Weight tying reduces parameters.** Sharing weights between token embedding and LM head is a common optimization.

---

## Further Reading

- [LLaVA Paper (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) -- Visual Instruction Tuning
- [Flamingo Paper (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198) -- Visual Language Model for Few-Shot Learning
- [CLIP Paper (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) -- Learning Transferable Visual Models
- [ViT Paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) -- Vision Transformer
