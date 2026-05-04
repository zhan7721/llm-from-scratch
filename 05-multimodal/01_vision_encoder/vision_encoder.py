"""Vision Transformer (ViT) implementation.

This module implements the core components of a Vision Transformer:
- PatchEmbedding: Splits an image into fixed-size patches and projects each
  patch into an embedding vector via a Conv2d layer.
- VisionTransformer: Full ViT with CLS token, learnable positional embeddings,
  transformer encoder blocks, and layer normalization.

The Vision Transformer (Dosovitskiy et al., 2020) applies the standard
Transformer encoder to sequences of image patches, treating each patch as a
"token" analogous to word tokens in NLP.

Architecture overview:
    image (B, C, H, W)
        -> PatchEmbedding       (B, N, D)  where N = (H/P)*(W/P)
        -> Prepend CLS token    (B, N+1, D)
        -> Add position embeddings
        -> N x TransformerBlock (self-attention + FFN)
        -> LayerNorm
        -> Take CLS token       (B, D)     for classification

Design notes:
- This module is self-contained (no imports from other chapters).
- Uses standard ViT conventions: LayerNorm, GELU activation, non-causal attention.
- The FFN hidden dimension defaults to 4 * embedding_dim (standard ViT ratio).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import math


@dataclass
class ViTConfig:
    """Configuration for the Vision Transformer.

    Args:
        image_size: Height and width of the input image (assumed square).
        patch_size: Height and width of each patch (assumed square).
        num_channels: Number of input channels (e.g., 3 for RGB).
        embedding_dim: Dimensionality of the patch embeddings (d_model).
        num_heads: Number of attention heads in each transformer block.
        num_layers: Number of transformer encoder blocks.
        mlp_ratio: Multiplier for the FFN hidden dimension relative to
            embedding_dim. Default is 4.0 (standard ViT).
        dropout: Dropout probability for embeddings and attention.
    """

    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    embedding_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0


class PatchEmbedding(nn.Module):
    """Split image into patches and project to embedding dimension.

    This uses a Conv2d layer with kernel_size = stride = patch_size, which
    is equivalent to:
      1. Reshape (B, C, H, W) into (B, C, N_h, P_h, N_w, P_w)
      2. Rearrange to (B, N, P_h * P_w * C)
      3. Linear project to (B, N, embedding_dim)

    Using Conv2d is more efficient and idiomatic in PyTorch.

    Args:
        image_size: Spatial size of the input image (height = width).
        patch_size: Spatial size of each patch (height = width).
        num_channels: Number of input channels.
        embedding_dim: Output embedding dimension.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_channels: int,
        embedding_dim: int,
    ):
        super().__init__()
        assert image_size % patch_size == 0, (
            f"image_size ({image_size}) must be divisible by "
            f"patch_size ({patch_size})"
        )

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2

        # Conv2d with kernel_size=stride=patch_size is equivalent to
        # splitting into non-overlapping patches and projecting each one.
        self.projection = nn.Conv2d(
            in_channels=num_channels,
            out_channels=embedding_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Input image tensor of shape (batch_size, num_channels,
               image_size, image_size).

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embedding_dim).
        """
        # (B, C, H, W) -> (B, D, H/P, W/P) via Conv2d
        x = self.projection(x)

        # (B, D, H/P, W/P) -> (B, D, N) -> (B, N, D)
        # Flatten spatial dims and transpose so patches are the sequence dim.
        x = x.flatten(2).transpose(1, 2)

        return x


class TransformerBlock(nn.Module):
    """Pre-Norm Transformer encoder block (ViT-style).

    Architecture:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))

    Uses LayerNorm and GELU activation following the original ViT paper.
    Attention is non-causal (bidirectional) since all patches can attend
    to each other.

    Args:
        embedding_dim: Dimensionality of the model.
        num_heads: Number of attention heads.
        mlp_dim: Hidden dimension for the FFN.
        dropout: Dropout probability.
    """

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
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embedding_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer encoder block.

        Args:
            x: Input of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim).
        """
        # Pre-Norm Self-Attention with residual connection
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed)
        x = x + attn_out

        # Pre-Norm FFN with residual connection
        x = x + self.ffn(self.norm2(x))

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT) for image classification.

    The model converts an image into a sequence of patch embeddings,
    prepends a learnable CLS token, adds positional embeddings, and
    processes the sequence through transformer encoder blocks. The final
    CLS token representation is used for classification.

    Architecture:
        image (B, C, H, W)
            -> PatchEmbedding          (B, N, D)
            -> Prepend CLS token       (B, N+1, D)
            -> Add position embeddings (B, N+1, D)
            -> N x TransformerBlock    (B, N+1, D)
            -> LayerNorm               (B, N+1, D)
            -> CLS token + Linear head (B, num_classes)

    Args:
        config: A ViTConfig dataclass with model hyperparameters.
        num_classes: Number of output classes for classification head.
            If None, no classification head is added and the model returns
            the full sequence of embeddings.
    """

    def __init__(self, config: ViTConfig, num_classes: int = None):
        super().__init__()
        self.config = config

        # --- Patch embedding ---
        self.patch_embedding = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            num_channels=config.num_channels,
            embedding_dim=config.embedding_dim,
        )
        num_patches = self.patch_embedding.num_patches

        # --- CLS token ---
        # A learnable vector prepended to the patch sequence. The
        # transformer's output at this position serves as the aggregate
        # image representation (analogous to BERT's [CLS] token).
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embedding_dim))

        # --- Positional embedding ---
        # Learnable position embeddings for each patch + the CLS token.
        # Total sequence length = num_patches + 1 (for CLS).
        self.position_embedding = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.embedding_dim)
        )
        self.pos_dropout = nn.Dropout(config.dropout)

        # --- Transformer encoder blocks ---
        mlp_dim = int(config.embedding_dim * config.mlp_ratio)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embedding_dim=config.embedding_dim,
                num_heads=config.num_heads,
                mlp_dim=mlp_dim,
                dropout=config.dropout,
            )
            for _ in range(config.num_layers)
        ])

        # --- Final layer norm ---
        self.norm = nn.LayerNorm(config.embedding_dim)

        # --- Classification head (optional) ---
        self.head = (
            nn.Linear(config.embedding_dim, num_classes)
            if num_classes is not None
            else None
        )

        self._init_weights()

    def _init_weights(self):
        """Initialize weights following standard ViT practice.

        - Linear layers: Xavier uniform for weights, zeros for biases.
        - CLS token and position embeddings: initialized to zeros (they
          will be learned during training).
        """
        # Initialize CLS token and position embeddings to zeros
        nn.init.zeros_(self.cls_token)
        nn.init.zeros_(self.position_embedding)

        # Initialize linear and conv layers
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight.view(
                    module.weight.size(0), -1
                ))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the Vision Transformer.

        Args:
            x: Input image tensor of shape
               (batch_size, num_channels, image_size, image_size).

        Returns:
            If num_classes is set:
                Classification logits of shape (batch_size, num_classes).
            Otherwise:
                Full sequence embeddings of shape
                (batch_size, num_patches + 1, embedding_dim).
        """
        B = x.shape[0]

        # Step 1: Patch embedding
        # (B, C, H, W) -> (B, N, D)
        x = self.patch_embedding(x)

        # Step 2: Prepend CLS token
        # Expand CLS token to batch size and prepend: (B, 1, D)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        # (B, N, D) -> (B, N+1, D)
        x = torch.cat([cls_tokens, x], dim=1)

        # Step 3: Add positional embedding
        # position_embedding shape: (1, N+1, D), broadcasts over batch dim
        x = x + self.position_embedding
        x = self.pos_dropout(x)

        # Step 4: Process through transformer blocks
        for block in self.blocks:
            x = block(x)

        # Step 5: Final layer normalization
        x = self.norm(x)

        # Step 6: Output
        if self.head is not None:
            # Take CLS token output (first position) and project to classes
            cls_output = x[:, 0]
            return self.head(cls_output)
        else:
            # Return full sequence (useful for downstream tasks like VLMs)
            return x
