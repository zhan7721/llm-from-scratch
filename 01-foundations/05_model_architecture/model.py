"""Complete GPT model architecture (LLaMA-style).

This module assembles all previous components into a full language model:
- TokenEmbedding + RoPE for input representation
- N x TransformerBlock (Pre-Norm, SwiGLU, MHA) for deep processing
- RMSNorm for final normalization
- Linear lm_head for next-token prediction

Design choices follow LLaMA: Pre-Norm, RMSNorm, SwiGLU, RoPE, weight tying.
"""

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
    """Configuration for the GPT model.

    Args:
        vocab_size: Number of unique tokens in the vocabulary.
        d_model: Dimensionality of the model (hidden size).
        n_heads: Number of attention heads.
        n_layers: Number of transformer blocks to stack.
        d_ff: Hidden dimension for the FFN. If None, uses the default
              SwiGLU formula: (2/3) * 4 * d_model rounded to nearest 256.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        dropout: Dropout probability (unused in this implementation but
                 kept for compatibility).
    """

    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = None
    max_seq_len: int = 1024
    dropout: float = 0.0


class GPT(nn.Module):
    """GPT-style language model with LLaMA architecture choices.

    Architecture:
        input tokens
            -> TokenEmbedding (scale by sqrt(d_model))
            -> RotaryPositionalEmbedding (RoPE)
            -> N x TransformerBlock (Pre-Norm + SwiGLU + MHA)
            -> RMSNorm (final normalization)
            -> Linear lm_head (project to vocab_size)

    Weight tying: lm_head.weight is shared with token_emb.embedding.weight.
    This reduces parameters and often improves performance.

    Args:
        config: A GPTConfig dataclass with model hyperparameters.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # Input embeddings
        self.token_emb = TokenEmbedding(config.vocab_size, config.d_model)
        self.rope = RotaryPositionalEmbedding(config.d_model, config.max_seq_len)

        # Stack of transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_model, config.n_heads, config.d_ff)
            for _ in range(config.n_layers)
        ])

        # Final normalization
        self.norm = RMSNorm(config.d_model)

        # Language model head: projects hidden states to vocabulary logits
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Weight tying: share parameters between embedding and output projection
        # This is a common trick that reduces parameters and often improves quality.
        # The intuition: tokens that are semantically similar should have similar
        # embeddings AND produce similar logits.
        self.lm_head.weight = self.token_emb.embedding.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire model.

        Args:
            x: Token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).
        """
        # Step 1: Convert token IDs to embeddings, scaled by sqrt(d_model)
        h = self.token_emb(x)

        # Step 2: Apply rotary positional embedding (encodes position info)
        h = self.rope(h)

        # Step 3: Process through N transformer blocks
        for layer in self.layers:
            h = layer(h)

        # Step 4: Final RMSNorm (some architectures apply this, some don't;
        # LLaMA applies it)
        h = self.norm(h)

        # Step 5: Project to vocabulary size for next-token prediction
        return self.lm_head(h)

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation: given a prompt, generate new tokens one at a time.

        Args:
            prompt: Starting token IDs of shape (batch_size, seq_len).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature. Higher = more random, lower = more
                         deterministic. Temperature=0 would be greedy (but we use
                         temperature=1e-8 to avoid division by zero).

        Returns:
            Extended sequence of shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # Crop to max_seq_len to handle very long sequences
            idx_cond = prompt[:, -self.config.max_seq_len:]

            # Forward pass to get logits for the next token
            logits = self(idx_cond)

            # Focus on the last position's logits (next token prediction)
            logits = logits[:, -1, :] / temperature

            # Convert to probabilities and sample
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # Append the new token to the sequence
            prompt = torch.cat([prompt, next_token], dim=1)

        return prompt
