"""Reference solution for the Vision Transformer exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class ViTConfig:
    """Configuration for the Vision Transformer."""

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class PatchEmbeddingSolution(nn.Module):
    """Split image into patches and project to embedding dimension."""

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        embedding_dim: int,
    ):
        super().__init__()
        assert image_size % patch_size == 0
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d with kernel_size=stride=patch_size splits and projects
        self.projection = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings."""
        # (B, C, H, W) -> (B, D, H/P, W/P) -> (B, N, D)
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlockSolution(nn.Module):
    """Pre-Norm Transformer encoder block (ViT-style)."""

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embedding_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=embedding_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embedding_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embedding_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer encoder block."""
        # Pre-Norm Self-Attention with residual
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-Norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class VisionTransformerSolution(nn.Module):
    """Vision Transformer (ViT) for image classification."""

    def __init__(self, config: ViTConfig, num_classes: int = None):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embedding = PatchEmbeddingSolution(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embedding_dim=config.embedding_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))

        # Position embeddings
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embedding_dim)
        )
        self.pos_dropout = nn.Dropout(config.dropout)

        # Transformer blocks
        mlp_dim = int(config.embedding_dim * config.mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlockSolution(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                mlp_dim=mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # Final norm
        self.norm = nn.LayerNorm(config.embedding_dim)

        # Classification head
        self.head = (
            nn.Linear(config.embedding_dim, num_classes)
            if num_classes is not None
            else None
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Vision Transformer."""
        B = x.shape[0]

        # Patch embedding: (B, C, H, W) -> (B, N, D)
        x = self.patch_embedding(x)

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.position_embedding
        x = self.pos_dropout(x)

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Output
        if self.head is not None:
            cls_output = x[:, 0]
            return self.head(cls_output)
        else:
            return x
