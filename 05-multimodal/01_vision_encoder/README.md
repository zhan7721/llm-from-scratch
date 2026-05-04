# Vision Transformer (ViT)

> **Module 05 -- Multimodal, Chapter 01**

The Vision Transformer (Dosovitskiy et al., 2020) applies the standard Transformer encoder to sequences of image patches. Instead of processing raw pixels with convolutions, ViT splits an image into fixed-size patches, embeds each patch into a vector, and processes the sequence with a Transformer -- the same architecture used for language models.

---

## Prerequisites

- Basic understanding of Transformers and self-attention (Module 01, Chapters 03-04)
- PyTorch basics: `nn.Module`, `nn.Linear`, `nn.Conv2d`, `dataclasses`

## Files

| File | Purpose |
|------|---------|
| `vision_encoder.py` | Core ViT implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A Vision Transformer converts an image into a sequence of patch embeddings, then processes them exactly like a language model processes token embeddings:

```
Input: image (B, C, H, W)
    |
    v
PatchEmbedding          -- split into patches, project to D dimensions
    |
    v
Prepend CLS token       -- learnable [CLS] token for classification
    |
    v
Add Position Embeddings -- learnable per-position vectors
    |
    v
TransformerBlock x N    -- self-attention + FFN (bidirectional, no causal mask)
    |
    v
LayerNorm               -- final normalization
    |
    v
CLS token -> Linear     -- classification head
    |
    v
Output: logits (B, num_classes)
```

### Key Insight: Images as Sequences

The core idea is simple: treat image patches the same way NLP treats tokens. A 224x224 image split into 16x16 patches yields (224/16)^2 = 196 patches. Each patch is a small 16x16x3 = 768-dimensional vector that gets projected to the embedding dimension, just like a word embedding.

---

## ViTConfig: Model Hyperparameters

```python
@dataclass
class ViTConfig:
    image_size: int = 224       # input image size (square)
    patch_size: int = 16        # patch size (square)
    num_channels: int = 3       # RGB channels
    embedding_dim: int = 768    # embedding / hidden dimension
    num_heads: int = 12         # attention heads
    num_layers: int = 12        # transformer blocks
    mlp_ratio: float = 4.0      # FFN hidden dim = mlp_ratio * embedding_dim
    dropout: float = 0.0        # dropout probability
```

### ViT Model Sizes

| Model | embedding_dim | num_heads | num_layers | mlp_ratio | Parameters |
|-------|---------------|-----------|------------|-----------|------------|
| ViT-Ti (Tiny) | 192 | 3 | 12 | 4.0 | 5.7M |
| ViT-S (Small) | 384 | 6 | 12 | 4.0 | 22M |
| ViT-B (Base) | 768 | 12 | 12 | 4.0 | 86M |
| ViT-L (Large) | 1024 | 16 | 24 | 4.0 | 307M |
| ViT-H (Huge) | 1280 | 16 | 32 | 4.0 | 632M |

---

## Architecture Details

### PatchEmbedding

```python
self.projection = nn.Conv2d(
    in_channels=num_channels,
    out_channels=embedding_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

A Conv2d with `kernel_size = stride = patch_size` is equivalent to:
1. Reshape (B, C, H, W) into non-overlapping patches
2. Flatten each patch to a vector
3. Linearly project to `embedding_dim`

This is more efficient than doing these steps manually.

**Number of patches**: `N = (H / P) * (W / P)`

For a 224x224 image with 16x16 patches: `N = 14 * 14 = 196`

### CLS Token

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
```

A learnable vector prepended to the patch sequence. After processing through the transformer, the CLS token's output serves as the aggregate image representation. This is the same approach used in BERT for sentence classification.

### Position Embedding

```python
self.position_embedding = nn.Parameter(
    torch.zeros(1, num_patches + 1, embedding_dim)
)
```

Learnable position embeddings for each patch plus the CLS token. Unlike RoPE (used in the language model), ViT uses absolute learned position embeddings. The position information is added to the patch embeddings before the transformer blocks.

### Transformer Blocks

Each block applies:
1. **Pre-Norm Self-Attention**: `x = x + Attention(LayerNorm(x))`
2. **Pre-Norm FFN**: `x = x + FFN(LayerNorm(x))`

The attention is **bidirectional** (no causal mask) -- every patch can attend to every other patch. This is different from the language model where causal masking prevents future token access.

### FFN (Feed-Forward Network)

```python
FFN(x) = Linear(GELU(Linear(x)))
```

Standard ViT uses GELU activation (not SwiGLU as in LLaMA). The hidden dimension is `mlp_ratio * embedding_dim` (default 4x).

---

## Forward Pass Walkthrough

```python
def forward(self, x):
    B = x.shape[0]

    # 1. Patch embedding: (B, C, H, W) -> (B, N, D)
    x = self.patch_embedding(x)

    # 2. Prepend CLS token: (B, N, D) -> (B, N+1, D)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)

    # 3. Add positional embedding: (B, N+1, D) + (1, N+1, D)
    x = x + self.position_embedding

    # 4. Transformer blocks
    for block in self.blocks:
        x = block(x)

    # 5. Final norm
    x = self.norm(x)

    # 6. Take CLS token and classify
    cls_output = x[:, 0]
    return self.head(cls_output)
```

### Shape Trace

For a batch of 2 images (32x32 RGB) with patch_size=4, embedding_dim=64:

```
Input:          (2, 3, 32, 32)   -- RGB images
PatchEmbed:     (2, 64, 64)      -- 64 patches, each 64-dim
CLS prepend:    (2, 65, 64)      -- CLS + 64 patches
+ Pos Embed:    (2, 65, 64)      -- position info added
Block 1:        (2, 65, 64)      -- attention + FFN
Block 2:        (2, 65, 64)      -- attention + FFN
LayerNorm:      (2, 65, 64)      -- normalized
CLS output:     (2, 64)          -- take first position
Classification: (2, 10)          -- project to 10 classes
```

---

## ViT vs CNN

| Aspect | CNN | ViT |
|--------|-----|-----|
| Inductive bias | Strong (locality, translation equivariance) | Weak (only position embedding) |
| Data efficiency | Good with small datasets | Needs large datasets or pre-training |
| Scaling | Moderate | Excellent (more data + params = better) |
| Global context | Requires deep stacking | Every layer has global attention |
| Interpretability | Feature maps | Attention maps |

ViT's weakness is its lack of spatial inductive bias. Without the locality prior of convolutions, ViT needs more data to learn that nearby pixels are related. This is why ViT is often pre-trained on large datasets (ImageNet-21k, JFT-300M) and then fine-tuned.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/01_vision_encoder/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 05-multimodal/01_vision_encoder/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the Vision Transformer yourself.

### Exercise Order

1. **PatchEmbedding**: Create Conv2d projection and implement forward (flatten + transpose)
2. **TransformerBlock**: Create LayerNorm, MultiheadAttention, FFN, and implement forward
3. **VisionTransformer**: Create CLS token, position embeddings, blocks, and implement forward

### Tips

- The Conv2d trick for PatchEmbedding is the key insight. `kernel_size=stride=patch_size` means each convolution window exactly covers one patch with no overlap.
- `nn.MultiheadAttention` with `batch_first=True` expects input as `(B, N, D)`.
- The CLS token uses `expand(B, -1, -1)` to repeat across the batch without copying memory.
- Position embeddings are simply added to patch embeddings (broadcasting handles the batch dimension).

---

## Key Takeaways

1. **ViT treats images as sequences.** By splitting images into patches and embedding them, we can reuse the exact same Transformer architecture used for language.

2. **PatchEmbedding via Conv2d is elegant.** A single Conv2d layer with the right kernel/stride simultaneously splits the image and projects each patch.

3. **CLS token aggregates information.** The learnable CLS token collects global image features through self-attention, serving as the image representation for classification.

4. **Position embeddings encode spatial structure.** Since the Transformer has no inherent notion of spatial arrangement, learnable position embeddings are essential.

5. **ViT needs more data than CNNs.** Without the inductive biases of convolutions (locality, translation equivariance), ViT requires larger datasets to learn these patterns from scratch.

---

## Further Reading

- [ViT Paper (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) -- Original Vision Transformer
- [An Image is Worth 16x16 Words (blog)](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) -- Google AI blog post
- [DeiT (Touvron et al., 2021)](https://arxiv.org/abs/2012.12877) -- Data-efficient Image Transformers
- [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030) -- Hierarchical Vision Transformer
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377) -- Masked Autoencoder for self-supervised ViT pre-training
