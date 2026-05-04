"""NEFTune: Noisy Embedding Fine-Tuning -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class NEFTuneEmbeddingSolution(nn.Module):
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
        """Forward pass with optional noise injection."""
        embeddings = self.embedding(x)

        if self.training and self.alpha > 0:
            seq_len = x.shape[1]
            noise_scale = self.alpha / math.sqrt(seq_len * self.embedding_dim)
            noise = torch.zeros_like(embeddings).uniform_(-0.5, 0.5) * noise_scale
            embeddings = embeddings + noise

        return embeddings


class NEFTuneConfigSolution:
    """Configuration for NEFTune noise injection."""

    def __init__(
        self,
        alpha: float = 5.0,
        enabled: bool = True,
        schedule: str = "constant",
        warmup_steps: int = 0,
    ):
        self.alpha = alpha
        self.enabled = enabled
        self.schedule = schedule
        self.warmup_steps = warmup_steps

    def get_alpha(self, step: int) -> float:
        """Get alpha for current training step."""
        if not self.enabled:
            return 0.0

        if self.schedule == "constant":
            return self.alpha
        elif self.schedule == "linear_decay":
            if step < self.warmup_steps:
                return self.alpha * (1 - step / max(self.warmup_steps, 1))
            return 0.0
        elif self.schedule == "warmup":
            if step < self.warmup_steps:
                return self.alpha * step / max(self.warmup_steps, 1)
            return self.alpha
        return self.alpha


def apply_neftune_solution(
    model: nn.Module,
    alpha: float = 5.0,
) -> nn.Module:
    """Apply NEFTune to a model's embedding layer."""
    # Find and replace the token embedding
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding) and "token" in name.lower():
            neftune_emb = NEFTuneEmbeddingSolution(
                module.num_embeddings,
                module.embedding_dim,
                alpha=alpha,
                padding_idx=module.padding_idx,
            )
            neftune_emb.embedding.weight.data = module.weight.data.clone()

            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = model
            for part in parent_name.split("."):
                if part:
                    parent = getattr(parent, part)
            setattr(parent, child_name, neftune_emb)
            break
    else:
        # Fallback: look for any embedding
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding):
                neftune_emb = NEFTuneEmbeddingSolution(
                    module.num_embeddings,
                    module.embedding_dim,
                    alpha=alpha,
                    padding_idx=module.padding_idx,
                )
                neftune_emb.embedding.weight.data = module.weight.data.clone()

                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model
                for part in parent_name.split("."):
                    if part:
                        parent = getattr(parent, part)
                setattr(parent, child_name, neftune_emb)
                break

    return model
