"""NEFTune: Noisy Embedding Fine-Tuning -- Exercise.

Fill in the TODO methods to implement NEFTune noise injection.
Start with `NEFTuneEmbedding.forward`, then `NEFTuneConfig.get_alpha`,
and finally `apply_neftune`.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class NEFTuneEmbeddingExercise(nn.Module):
    """Embedding layer with NEFTune noise injection.

    During training, adds uniform noise scaled by alpha/sqrt(L*d)
    where L is sequence length and d is embedding dimension.

    During eval, behaves like a normal embedding layer.
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        alpha: float = 5.0,
        padding_idx: Optional[int] = None,
    ):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
        self.alpha = alpha
        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with optional noise injection.

        Args:
            x: Token IDs of shape (batch, seq_len).

        Returns:
            Embeddings of shape (batch, seq_len, embedding_dim).

        Hints:
            1. Get the base embeddings: `embeddings = self.embedding(x)`.
            2. If `self.training` and `self.alpha > 0`, add noise.
            3. Compute the noise scale: `alpha / sqrt(seq_len * embedding_dim)`.
               Use `x.shape[1]` for seq_len and `self.embedding_dim` for d.
            4. Generate uniform noise in [-0.5, 0.5]:
               `noise = torch.zeros_like(embeddings).uniform_(-0.5, 0.5)`
            5. Scale the noise: `noise = noise * noise_scale`
            6. Add noise to embeddings: `embeddings = embeddings + noise`
            7. Return the embeddings.
        """
        raise NotImplementedError("TODO: implement forward")


class NEFTuneConfigExercise:
    """Configuration for NEFTune noise injection."""

    def __init__(
        self,
        alpha: float = 5.0,
        enabled: bool = True,
        schedule: str = "constant",  # "constant", "linear_decay", "warmup"
        warmup_steps: int = 0,
    ):
        self.alpha = alpha
        self.enabled = enabled
        self.schedule = schedule
        self.warmup_steps = warmup_steps

    def get_alpha(self, step: int) -> float:
        """Get alpha for current training step.

        Args:
            step: Current training step number.

        Returns:
            The alpha value to use at this step.

        Hints:
            1. If not enabled, return 0.0.
            2. For "constant" schedule, always return self.alpha.
            3. For "warmup" schedule:
               - If step < warmup_steps, return alpha * step / warmup_steps.
               - Otherwise, return alpha.
            4. For "linear_decay" schedule:
               - If step < warmup_steps, return alpha * (1 - step / warmup_steps).
               - Otherwise, return 0.0.
        """
        raise NotImplementedError("TODO: implement get_alpha")


def apply_neftune_exercise(
    model: nn.Module,
    alpha: float = 5.0,
) -> nn.Module:
    """Apply NEFTune to a model's embedding layer.

    Args:
        model: Model with an embedding layer.
        alpha: Noise scale factor.

    Returns:
        Model with NEFTune applied (embedding replaced).

    Hints:
        1. Iterate over `model.named_modules()` to find an nn.Embedding.
        2. Prefer modules whose name contains "token" (case-insensitive).
        3. Create a NEFTuneEmbeddingExercise with the same parameters.
        4. Copy the weight data: `neftune_emb.embedding.weight.data = module.weight.data.clone()`
        5. Navigate to the parent module and replace the child using setattr.
           - Split the name by "." to get parent path and child name.
           - Walk the parent path with getattr.
           - Use setattr to replace the child.
        6. If no "token" embedding is found, fall back to any nn.Embedding.
        7. Return the model.
    """
    raise NotImplementedError("TODO: implement apply_neftune")
