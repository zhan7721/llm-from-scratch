"""Exercise: Implement the Visual Language Model (VLM).

Complete the TODOs below to build a VLM that combines vision and language.
Run `pytest tests.py` to verify your implementation.
"""

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


class VisionProjectorExercise(nn.Module):
    """Projects vision features to language model embedding space.

    This is a simple MLP projector that maps vision encoder output from
    vision_dim to lm_dim. The architecture follows LLaVA's approach:

        Linear(vision_dim, lm_dim) -> GELU -> Linear(lm_dim, lm_dim)

    TODO: Implement the __init__ and forward methods.

    Args:
        vision_dim: Dimensionality of input vision features.
        lm_dim: Dimensionality of output (language model embedding dim).
    """

    def __init__(self, vision_dim: int, lm_dim: int):
        super().__init__()
        self.vision_dim = vision_dim
        self.lm_dim = lm_dim

        # TODO 1: Create the MLP projector
        # Hint: Use nn.Sequential with:
        #   - nn.Linear(vision_dim, lm_dim)
        #   - nn.GELU()
        #   - nn.Linear(lm_dim, lm_dim)
        self.mlp = None  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Project vision features to language model embedding space.

        Args:
            x: Vision features of shape (batch_size, num_patches, vision_dim).

        Returns:
            Projected features of shape (batch_size, num_patches, lm_dim).

        TODO: Implement the forward pass.
        """
        # TODO 2: Apply the MLP to the input
        return None  # YOUR CODE HERE


class VLMExercise(nn.Module):
    """Visual Language Model combining vision encoder, projector, and language model.

    The VLM processes images and text together:
    1. Vision encoder extracts features from images
    2. Projector maps vision features to language model embedding space
    3. Vision embeddings are prepended to text embeddings
    4. The combined sequence is processed by the language model

    TODO: Implement the __init__ and forward methods.

    Args:
        config: A VLMConfig dataclass with model hyperparameters.
    """

    def __init__(self, config: VLMConfig):
        super().__init__()
        self.config = config

        # TODO 3: Create the vision encoder
        # Hint: Use VisionTransformer from the previous chapter
        # Create a ViTConfig with the vision parameters from config
        # Set num_classes=None to get full sequence output (not just CLS)
        self.vision_encoder = None  # YOUR CODE HERE

        # TODO 4: Create the vision projector
        # Hint: Use VisionProjectorExercise with vision_dim and lm_dim from config
        self.projector = None  # YOUR CODE HERE

        # TODO 5: Create the language model components
        # You need to create:
        #   - Token embedding: nn.Embedding(vocab_size, lm_dim)
        #   - Positional embedding: nn.Embedding(max_seq_len, lm_dim)
        #   - Transformer blocks: nn.ModuleList of SimpleTransformerBlock
        #   - Final norm: nn.LayerNorm(lm_dim)
        #   - LM head: nn.Linear(lm_dim, vocab_size, bias=False)

        # Calculate number of vision tokens (including CLS token)
        num_patches = (config.vision_image_size // config.vision_patch_size) ** 2
        self.num_vision_tokens = num_patches + 1  # +1 for CLS token

        # Token embedding
        self.token_emb = None  # YOUR CODE HERE

        # Positional embedding (must be large enough for vision + text)
        total_max_seq_len = config.lm_max_seq_len + self.num_vision_tokens
        self.pos_emb = None  # YOUR CODE HERE

        # Transformer blocks
        self.layers = None  # YOUR CODE HERE

        # Final layer norm
        self.norm = None  # YOUR CODE HERE

        # Language model head
        self.lm_head = None  # YOUR CODE HERE

        # TODO 6: Implement weight tying
        # Hint: Make lm_head.weight point to token_emb.weight
        pass  # YOUR CODE HERE

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

        TODO: Implement the forward pass.
          1. Extract vision features using vision_encoder
          2. Project vision features using projector
          3. Get text embeddings using token_emb
          4. Concatenate vision and text embeddings (vision first)
          5. Add positional embeddings
          6. Process through transformer blocks
          7. Apply final norm
          8. Project to vocabulary with lm_head
        """
        B = images.shape[0]

        # TODO 7: Extract vision features
        # vision_features = ...  # (B, num_vision_tokens, vision_dim)
        vision_features = None  # YOUR CODE HERE

        # TODO 8: Project vision features to LM embedding space
        # vision_embeddings = ...  # (B, num_vision_tokens, lm_dim)
        vision_embeddings = None  # YOUR CODE HERE

        # TODO 9: Get text embeddings
        # text_embeddings = ...  # (B, seq_len, lm_dim)
        text_embeddings = None  # YOUR CODE HERE

        # TODO 10: Concatenate vision and text embeddings
        # Vision tokens are prepended to text tokens
        # combined = ...  # (B, num_vision_tokens + seq_len, lm_dim)
        combined = None  # YOUR CODE HERE

        # TODO 11: Add positional embeddings
        # total_seq_len = combined.shape[1]
        # positions = torch.arange(total_seq_len, device=combined.device).unsqueeze(0)
        # combined = combined + self.pos_emb(positions)
        pass  # YOUR CODE HERE

        # TODO 12: Process through transformer blocks
        # h = combined
        # for layer in self.layers:
        #     h = layer(h)
        h = None  # YOUR CODE HERE

        # TODO 13: Apply final norm and project to vocabulary
        # h = self.norm(h)
        # logits = self.lm_head(h)
        return None  # YOUR CODE HERE


# ============================================================================
# Helper classes (provided for you)
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
