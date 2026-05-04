"""Reference solution for the Visual Language Model (VLM) exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../01_vision_encoder"))
from vision_encoder import ViTConfig, VisionTransformer


@dataclass
class VLMConfig:
    """Configuration for the Visual Language Model."""

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


class VisionProjectorSolution(nn.Module):
    """Projects vision features to language model embedding space."""

    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.vision_dim = vision_dim
        self.lm_dim = lm_dim

        # Simple MLP projector: two linear layers with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(vision_dim, lm_dim),
            nn.GELU(),
            nn.Linear(lm_dim, lm_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features to language model embedding space."""
        return self.mlp(x)


class VLMSolution(nn.Module):
    """Visual Language Model combining vision encoder, projector, and language model."""

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
        self.projector = VisionProjectorSolution(config.vision_dim, config.lm_dim)

        # Calculate number of vision tokens (including CLS token)
        num_patches = (config.vision_image_size // config.vision_patch_size) ** 2
        self.num_vision_tokens = num_patches + 1  # +1 for CLS token

        # Token embedding
        self.token_emb = nn.Embedding(config.lm_vocab_size, config.lm_dim)

        # Positional embedding (must be large enough for vision + text)
        total_max_seq_len = config.lm_max_seq_len + self.num_vision_tokens
        self.pos_emb = nn.Embedding(total_max_seq_len, config.lm_dim)

        # Transformer blocks
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(config.lm_dim, config.lm_n_heads)
            for _ in range(config.lm_n_layers)
        ])

        # Final layer norm
        self.norm = nn.LayerNorm(config.lm_dim)

        # Language model head
        self.lm_head = nn.Linear(config.lm_dim, config.lm_vocab_size, bias=False)

        # Weight tying
        self.lm_head.weight = self.token_emb.weight

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
        text_embeddings = self.token_emb(input_ids)

        # Step 4: Concatenate vision and text embeddings
        # Vision tokens are prepended to text tokens
        # (B, num_vision_tokens + seq_len, lm_dim)
        combined = torch.cat([vision_embeddings, text_embeddings], dim=1)

        # Step 5: Add positional embeddings
        total_seq_len = combined.shape[1]
        positions = torch.arange(total_seq_len, device=combined.device).unsqueeze(0)
        combined = combined + self.pos_emb(positions)

        # Step 6: Process through language model layers
        h = combined
        for layer in self.layers:
            h = layer(h)

        # Step 7: Final norm and project to vocabulary
        h = self.norm(h)
        logits = self.lm_head(h)

        return logits


class SimpleTransformerBlock(nn.Module):
    """A simple transformer block for the language model."""

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.attn_norm = nn.LayerNorm(d_model)
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

        self.ffn_norm = nn.LayerNorm(d_model)
        d_ff = int(2 * (4 * d_model) / 3)
        d_ff = ((d_ff + 255) // 256) * 256
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        normed = self.attn_norm(x)
        Q = self.W_q(normed).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(normed).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(normed).view(B, T, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.W_o(attn_output)

        normed = self.ffn_norm(x)
        x = x + self.w2(F.silu(self.w_gate(normed)) * self.w1(normed))

        return x
