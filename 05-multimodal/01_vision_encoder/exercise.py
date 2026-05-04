"""Exercise: Implement the Vision Transformer (ViT).

Complete the TODOs below to build a Vision Transformer for image classification.
Run `pytest tests.py` to verify your implementation.
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


class PatchEmbeddingExercise(nn.Module):
    """Split image into patches and project to embedding dimension.

    TODO: Implement the __init__ and forward methods.

    Hint: Use nn.Conv2d with kernel_size=stride=patch_size to simultaneously
    split the image into patches AND project each patch to embedding_dim.
    This is equivalent to manually reshaping and applying a linear layer.
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

        # TODO 1: Create the Conv2d projection layer
        # Hint: kernel_size and stride should both be patch_size.
        # This makes each kernel exactly cover one patch with no overlap.
        # in_channels = num_channels, out_channels = embedding_dim
        self.projection = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert image to patch embeddings.

        Args:
            x: Input image tensor of shape (batch_size, num_channels,
               image_size, image_size).

        Returns:
            Patch embeddings of shape (batch_size, num_patches, embedding_dim).
        """
        # TODO 2: Apply the Conv2d projection
        # x = self.projection(x)  -- shape becomes (B, D, H/P, W/P)

        # TODO 3: Flatten the spatial dimensions and transpose
        # x = x.flatten(2)           -- (B, D, N) where N = num_patches
        # x = x.transpose(1, 2)      -- (B, N, D)

        return None  # YOUR CODE HERE


class TransformerBlockExercise(nn.Module):
    """Pre-Norm Transformer encoder block (ViT-style).

    TODO: Implement the __init__ and forward methods.

    Architecture:
        x = x + MultiHeadAttention(LayerNorm(x))
        x = x + FFN(LayerNorm(x))
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        # TODO 4: Create LayerNorm for the attention sub-layer
        self.norm1 = None  # YOUR CODE HERE

        # TODO 5: Create MultiheadAttention layer
        # Hint: nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.attn = None  # YOUR CODE HERE

        # TODO 6: Create LayerNorm for the FFN sub-layer
        self.norm2 = None  # YOUR CODE HERE

        # TODO 7: Create the FFN (two linear layers with GELU activation)
        # Hint: nn.Sequential(nn.Linear(embedding_dim, mlp_dim), nn.GELU(),
        #                     nn.Linear(mlp_dim, embedding_dim))
        self.ffn = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the transformer encoder block.

        Args:
            x: Input of shape (batch_size, seq_len, embedding_dim).

        Returns:
            Output of shape (batch_size, seq_len, embedding_dim).
        """
        # TODO 8: Pre-Norm Self-Attention with residual
        # normed = self.norm1(x)
        # attn_out, _ = self.attn(normed, normed, normed)
        # x = x + attn_out

        # TODO 9: Pre-Norm FFN with residual
        # x = x + self.ffn(self.norm2(x))

        return None  # YOUR CODE HERE


class VisionTransformerExercise(nn.Module):
    """Vision Transformer (ViT) for image classification.

    TODO: Implement the __init__ and forward methods.

    Architecture:
        image (B, C, H, W)
            -> PatchEmbedding          (B, N, D)
            -> Prepend CLS token       (B, N+1, D)
            -> Add position embeddings (B, N+1, D)
            -> N x TransformerBlock    (B, N+1, D)
            -> LayerNorm               (B, N+1, D)
            -> CLS token + Linear head (B, num_classes)
    """

    def __init__(self, config: ViTConfig, num_classes: int = None):
        super().__init__()
        self.config = config

        # TODO 10: Create the PatchEmbedding layer
        self.patch_embedding = None  # YOUR CODE HERE

        num_patches = (config.image_size // config.patch_size) ** 2

        # TODO 11: Create the learnable CLS token
        # Hint: nn.Parameter(torch.zeros(1, 1, config.embedding_dim))
        self.cls_token = None  # YOUR CODE HERE

        # TODO 12: Create learnable position embeddings
        # Hint: Shape should be (1, num_patches + 1, embedding_dim)
        # The +1 is for the CLS token.
        self.position_embedding = None  # YOUR CODE HERE
        self.pos_dropout = nn.Dropout(config.dropout)

        # TODO 13: Create the list of TransformerBlocks
        # Hint: mlp_dim = int(config.embedding_dim * config.mlp_ratio)
        # Use nn.ModuleList([...])
        self.blocks = None  # YOUR CODE HERE

        # TODO 14: Create the final LayerNorm
        self.norm = None  # YOUR CODE HERE

        # TODO 15: Create the classification head (optional)
        # If num_classes is not None, create nn.Linear(embedding_dim, num_classes)
        # Otherwise, set self.head = None
        self.head = None  # YOUR CODE HERE

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

        # TODO 16: Apply patch embedding
        # x = self.patch_embedding(x)  -- (B, N, D)

        # TODO 17: Prepend CLS token
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # x = torch.cat([cls_tokens, x], dim=1)  -- (B, N+1, D)

        # TODO 18: Add positional embedding
        # x = x + self.position_embedding

        # TODO 19: Apply dropout
        # x = self.pos_dropout(x)

        # TODO 20: Process through transformer blocks
        # for block in self.blocks:
        #     x = block(x)

        # TODO 21: Apply final layer normalization
        # x = self.norm(x)

        # TODO 22: Return output
        # If self.head is not None:
        #     cls_output = x[:, 0]       -- take CLS token
        #     return self.head(cls_output)
        # Else:
        #     return x                    -- return full sequence

        return None  # YOUR CODE HERE
