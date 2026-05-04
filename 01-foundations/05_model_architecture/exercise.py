"""Exercise: Implement the complete GPT model.

Complete the TODOs below to build the full GPT language model.
Run `pytest tests.py` to verify your implementation.
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
        d_ff: Hidden dimension for the FFN. If None, uses SwiGLU default.
        max_seq_len: Maximum sequence length for RoPE precomputation.
        dropout: Dropout probability (unused but kept for compatibility).
    """

    vocab_size: int = 32000
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 6
    d_ff: int = None
    max_seq_len: int = 1024
    dropout: float = 0.0


class GPTExercise(nn.Module):
    """GPT-style language model with LLaMA architecture choices.

    Architecture:
        input tokens
            -> TokenEmbedding (scale by sqrt(d_model))
            -> RotaryPositionalEmbedding (RoPE)
            -> N x TransformerBlock (Pre-Norm + SwiGLU + MHA)
            -> RMSNorm (final normalization)
            -> Linear lm_head (project to vocab_size)

    TODO: Implement the __init__ and forward methods.
    """

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        # TODO 1: Create the token embedding layer
        # Hint: TokenEmbedding(vocab_size, d_model)
        self.token_emb = None  # YOUR CODE HERE

        # TODO 2: Create the rotary positional embedding layer
        # Hint: RotaryPositionalEmbedding(d_model, max_seq_len)
        self.rope = None  # YOUR CODE HERE

        # TODO 3: Create a list of N transformer blocks
        # Hint: Use nn.ModuleList and a list comprehension
        # Each block: TransformerBlock(d_model, n_heads, d_ff)
        self.layers = None  # YOUR CODE HERE

        # TODO 4: Create the final RMSNorm layer
        self.norm = None  # YOUR CODE HERE

        # TODO 5: Create the language model head (linear projection to vocab_size)
        # Hint: nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head = None  # YOUR CODE HERE

        # TODO 6: Implement weight tying
        # Hint: Make lm_head.weight point to token_emb.embedding.weight
        # This shares the parameters between embedding and output projection
        pass  # YOUR CODE HERE

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the entire model.

        Args:
            x: Token IDs of shape (batch_size, seq_len).

        Returns:
            Logits of shape (batch_size, seq_len, vocab_size).

        TODO: Implement the forward pass.
          1. Apply token embedding
          2. Apply RoPE
          3. Pass through each transformer block
          4. Apply final RMSNorm
          5. Project to vocabulary with lm_head
        """
        # TODO 7: Convert token IDs to embeddings
        h = None  # YOUR CODE HERE

        # TODO 8: Apply rotary positional embedding
        # h = ...  # YOUR CODE HERE

        # TODO 9: Pass through each transformer block
        # Hint: iterate over self.layers
        # for layer in ...:
        #     h = ...

        # TODO 10: Apply final RMSNorm
        # h = ...  # YOUR CODE HERE

        # TODO 11: Project to vocabulary size
        return None  # YOUR CODE HERE

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        max_new_tokens: int = 100,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Autoregressive generation.

        TODO: Implement the generation loop.
          1. Loop for max_new_tokens iterations
          2. Crop input to max_seq_len
          3. Forward pass to get logits
          4. Get last position's logits, divide by temperature
          5. Convert to probabilities with softmax
          6. Sample next token with torch.multinomial
          7. Append to prompt

        Args:
            prompt: Starting token IDs of shape (batch_size, seq_len).
            max_new_tokens: Number of new tokens to generate.
            temperature: Sampling temperature.

        Returns:
            Extended sequence of shape (batch_size, seq_len + max_new_tokens).
        """
        for _ in range(max_new_tokens):
            # TODO 12: Crop to max_seq_len
            idx_cond = None  # YOUR CODE HERE

            # TODO 13: Get logits from forward pass
            logits = None  # YOUR CODE HERE

            # TODO 14: Get last position logits and apply temperature
            # logits = logits[:, -1, :] / temperature
            pass  # YOUR CODE HERE

            # TODO 15: Convert to probabilities and sample
            # probs = F.softmax(logits, dim=-1)
            # next_token = torch.multinomial(probs, num_samples=1)
            next_token = None  # YOUR CODE HERE

            # TODO 16: Append new token to prompt
            # prompt = torch.cat([prompt, next_token], dim=1)
            pass  # YOUR CODE HERE

        return prompt
