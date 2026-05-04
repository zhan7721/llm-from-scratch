"""Visual Language Model (VLM) implementation.

This module implements a Visual Language Model that combines:
- VisionProjector: Projects vision features to language model embedding space
- VLM: Combines vision encoder + projector + language model
- VLMConfig: Configuration dataclass

The VLM architecture follows the pattern used in models like LLaVA and Flamingo:
1. A vision encoder (ViT) extracts features from images
2. A projector maps vision features to the language model's embedding space
3. Vision embeddings are prepended to text embeddings
4. The combined sequence is processed by the language model

Architecture overview:
    image (B, C, H, W)
        -> VisionEncoder (ViT)         (B, num_patches, vision_dim)
        -> VisionProjector (MLP)       (B, num_patches, lm_dim)

    text tokens (B, seq_len)
        -> TokenEmbedding              (B, seq_len, lm_dim)

    Concatenate: [vision_embeddings | text_embeddings]
        -> (B, num_patches + seq_len, lm_dim)
        -> Language Model (Transformer)
        -> (B, num_patches + seq_len, vocab_size)

Design notes:
- Uses a simple MLP projector (Linear + GELU + Linear) following LLaVA
- Vision tokens are prepended to text tokens (common in VLMs)
- The language model is a standard GPT-style transformer
- Self-contained implementation with no external dependencies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import sys
import os

# Import vision encoder from previous chapter
sys.path.append(os.path.join(os.path.dirname(__file__), "../01_vision_encoder"))
from vision_encoder import ViTConfig, VisionTransformer


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class VLMConfig:
    """Configuration for the Visual Language Model.

    Args:
        vision_dim: Dimensionality of vision encoder output features.
        lm_dim: Dimensionality of the language model embeddings.
        lm_vocab_size: Size of the language model vocabulary.
        lm_n_heads: Number of attention heads in the language model.
        lm_n_layers: Number of transformer layers in the language model.
        lm_max_seq_len: Maximum sequence length for the language model (text only).
        vision_image_size: Input image size for the vision encoder.
        vision_patch_size: Patch size for the vision encoder.
        vision_n_heads: Number of attention heads in the vision encoder.
        vision_n_layers: Number of transformer layers in the vision encoder.
    """

    vision_dim: int = 768
    lm_dim: int = 512
    lm_vocab_size: int = 32000
    lm_n_heads: int = 8
    lm_n_layers: int = 6
    lm_max_seq_len: int = 1024
    vision_image_size: int = 224
    vision_patch_size: int = 16
    vision_n_heads: int = 12
    vision_n_layers: int = 12


# ============================================================================
# Vision Projector
# ============================================================================


class VisionProjector(nn.Module):
    """Projects vision features to language model embedding space.

    This is a simple MLP projector that maps vision encoder output from
    vision_dim to lm_dim. The architecture follows LLaVA's approach:

        Linear(vision_dim, lm_dim) -> GELU -> Linear(lm_dim, lm_dim)

    This allows the vision features to be concatenated with text embeddings
    and processed by the language model.

    Args:
        vision_dim: Dimensionality of input vision features.
        lm_dim: Dimensionality of output (language model embedding dim).
    """

    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.vision_dim = vision_dim
        self.lm_dim = lm_dim

        # Simple MLP projector: two linear layers with GELU activation
        # This is the standard approach used in LLaVA and similar models
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features to language model embedding space.

        Args:
            x: Vision features of shape (batch_size, num_patches, vision_dim).

        Returns:
            Projected features of shape (batch_size, num_patches, lm_dim).
        """
        return self.mlp(x)


# ============================================================================
# Simple Language Model (GPT-style)
# ============================================================================


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block for the language model.

    Uses standard Pre-Norm architecture with LayerNorm and SwiGLU FFN.

    Args:
        d_model: Dimensionality of the model.
        n_heads: Number of attention heads.
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # Pre-norm for attention
        self.attn_norm = nn.LayerNorm(d_model)

        # Multi-head attention (causal for language modeling)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        # Pre-norm for FFN
        self.ffn_norm = nn.LayerNorm(d_model)

        # SwiGLU FFN
        d_ff = int(2 * (4 * d_model) / 3)
        d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer block.

        Args:
            x: Input of shape (batch_size, seq_len, d_model).

        Returns:
            Output of shape (batch_size, seq_len, d_model).
        """
        B, T, C = x.shape

        # Pre-norm attention with causal mask
        normed = self.attn_norm(x)

        # Compute Q, K, V
        Q = self.W_q(normed).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(normed).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(normed).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention with causal mask
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.W_o(attn_output)

        # Pre-norm FFN (SwiGLU)
        normed = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w_gate(normed)) * self.w1(normed))

        return x


class SimpleLanguageModel(nn.Module):
    """A simple GPT-style language model.

    This is a minimal language model implementation that can be used
    as the language component of a VLM. It includes:
    - Token embedding
    - Positional embedding (learned)
    - N transformer blocks with causal attention
    - Final layer norm
    - Language model head

    Args:
        vocab_size: Size of the vocabulary.
        d_model: Dimensionality of the model.
        n_heads: Number of attention heads.
        n_layers: Number of transformer layers.
        max_seq_len: Maximum sequence length (including vision tokens).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        max_seq_len: int,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Learnable positional embedding
        # Note: max_seq_len should be large enough for vision + text tokens
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        # Transformer blocks
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(d_model, n_heads)
            for _ in range(n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(d_model)

        # Language model head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the language model.

        Args:
            x: Token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        B, T = x.shape

        # Token embedding
        h = self.token_emb(x)

        # Add positional embedding
        positions = torch.arange(T, device=x.device).unsqueeze(0)
        h = h + self.pos_emb(positions)

        # Process through transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Final norm and project to vocabulary
        h = self.norm(h)
        return self.lm_head(h)


# ============================================================================
# Visual Language Model
# ============================================================================


class VLM(nn.Module):
    """Visual Language Model combining vision encoder, projector, and language model.

    The VLM processes images and text together:
    1. Vision encoder extracts features from images
    2. Projector maps vision features to language model embedding space
    3. Vision embeddings are prepended to text embeddings
    4. The combined sequence is processed by the language model

    This architecture is similar to LLaVA and Flamingo models.

    Args:
        config: A VLMConfig dataclass with model hyperparameters.
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # Vision encoder (ViT from previous chapter)
        vit_config = ViTConfig(
            image_size=config.vision_image_size,
            patch_size=config.vision_patch_size,
            embedding_dim=config.vision_dim,
            num_heads=config.vision_n_heads,
            num_layers=config.vision_n_layers,
        )
        # Set num_classes=None to get full sequence output
        self.vision_encoder = VisionTransformer(vit_config, num_classes=None)

        # Vision projector: maps vision features to LM embedding space
        self.projector = VisionProjector(config.vision_dim, config.lm_dim)

        # Calculate number of vision tokens (including CLS token)
        num_patches = (config.vision_image_size // config.vision_patch_size) ** 2
        num_vision_tokens = num_patches + 1  # +1 for CLS token

        # Language model with max_seq_len that accounts for vision tokens
        # The max_seq_len in config is for text only, we need to add vision tokens
        total_max_seq_len = config.lm_max_seq_len + num_vision_tokens
        self.language_model = SimpleLanguageModel(
            vocab_size=config.lm_vocab_size,
            d_model=config.lm_dim,
            n_heads=config.lm_n_heads,
            n_layers=config.lm_n_layers,
            max_seq_len=total_max_seq_len,
        )

        # Store number of vision tokens for use in forward pass
        self.num_vision_tokens = num_vision_tokens

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the VLM.

        Args:
            images: Input images of shape (B, C, H, W).
            input_ids: Text token IDs of shape (B, seq_len).

        Returns:
            Logits of shape (B, num_vision_tokens + seq_len, vocab_size).
        """
        B = images.shape[0]

        # Step 1: Extract vision features
        # Vision encoder returns (B, num_patches + 1, vision_dim) including CLS token
        vision_features = self.vision_encoder(images)

        # Step 2: Project vision features to LM embedding space
        # (B, num_patches + 1, vision_dim) -> (B, num_patches + 1, lm_dim)
        vision_embeddings = self.projector(vision_features)

        # Step 3: Get text embeddings
        # (B, seq_len) -> (B, seq_len, lm_dim)
        text_embeddings = self.language_model.token_emb(input_ids)

        # Step 4: Concatenate vision and text embeddings
        # Vision tokens are prepended to text tokens
        # (B, num_vision_tokens + seq_len, lm_dim)
        combined = torch.cat([vision_embeddings, text_embeddings], dim=1)

        # Step 5: Add positional embeddings
        # We need to handle positions for the combined sequence
        total_seq_len = combined.shape[1]
        positions = torch.arange(total_seq_len, device=combined.device).unsqueeze(0)
        combined = combined + self.language_model.pos_emb(positions)

        # Step 6: Process through language model layers
        h = combined
        for layer in self.language_model.layers:
            h = layer(h)

        # Step 7: Final norm and project to vocabulary
        h = self.language_model.norm(h)
        logits = self.language_model.lm_head(h)

        return logits
