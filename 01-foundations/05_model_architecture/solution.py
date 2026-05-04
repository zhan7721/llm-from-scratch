"""Reference solution for the GPT model exercise."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../02_embedding"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../04_transformer_block"))
from embedding import TokenEmbedding, RotaryPositionalEmbedding
from transformer_block import TransformerBlock, RMSNorm


@dataclass
class GPTConfig:
    """Configuration for the GPT model."""

    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = None
    max_seq_len: int = 1024
    dropout: float = 0.0


class GPTSolution(nn.Module):
    """GPT-style language model with LLaMA architecture choices."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Token embedding: maps token IDs to dense vectors, scaled by sqrt(d_model)
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)

        # Rotary positional embedding: encodes position information
        self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)

        # Stack of N transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])

        # Final RMSNorm (LLaMA applies norm after the last block)
        self.norm = RMSNorm(config.d_model)

        # Language model head: projects to vocabulary logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share parameters between embedding and output projection
        self.lm_head.weight = self.token_emb.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire model.

        Args:
            x: Token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # Step 1: Token embedding (with sqrt(d_model) scaling)
        h = self.token_emb(x)

        # Step 2: Rotary positional embedding
        h = self.rope(h)

        # Step 3: Process through N transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Step 4: Final RMSNorm
        h = self.norm(h)

        # Step 5: Project to vocabulary size
        return self.lm_head(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation.

        Args:
            prompt: Starting token IDs of shape (batch_size, seq_len).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Extended sequence of shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len to handle very long sequences
            idx_cond = prompt[:, -self.config.max_seq_len:]

            # Forward pass to get logits
            logits = self(idx_cond)

            # Focus on the last position's logits and apply temperature
            logits = logits[:, -1, :] / temperature

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt
