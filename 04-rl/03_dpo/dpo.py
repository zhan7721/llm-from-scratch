"""Direct Preference Optimization (DPO) for language model alignment.

This module implements the core DPO components:
- compute_log_probs: compute log probabilities for sequence tokens
- DPOLoss: direct preference optimization loss (no reward model needed)
- DPODataset: preference pairs with pre-computed reference log probs
- DPOTrainer: training loop with reference model

DPO is a simpler alternative to PPO-based RLHF. Instead of training a
separate reward model and then using RL to optimize the policy, DPO
directly optimizes the policy on preference data using a clever
reparameterization that eliminates the need for an explicit reward model.

The key insight: the optimal policy under a KL constraint can be expressed
in terms of the reference policy and the reward. By substituting this into
the Bradley-Terry preference model, we get a loss that directly optimizes
the policy without ever computing rewards explicitly.

Reference: "Direct Preference Optimization: Your Language Model is Secretly
a Reward Model" (Rafailov et al., 2023).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
from torch.utils.data import Dataset


def compute_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_start_idx: int,
) -> torch.Tensor:
    """Compute log probabilities of response tokens given a language model.

    For an autoregressive language model, the logits at position t predict
    the token at position t+1. So to compute the log probability of the
    response tokens (starting at response_start_idx), we:

    1. Run the model to get logits for all positions.
    2. For each response token at position t (where t >= response_start_idx),
       compute log softmax of logits at position t-1.
    3. Gather the log prob of the actual token at position t.
    4. Sum all response token log probs to get a single score per sequence.

    Args:
        model: A language model that takes input_ids and returns logits
               of shape (batch, seq, vocab_size).
        input_ids: Token IDs of shape (batch, seq_len). Contains
                   [prompt_tokens, response_tokens].
        response_start_idx: Index where response tokens begin.
                           Tokens at positions [response_start_idx, seq_len)
                           are the response whose log probs we compute.

    Returns:
        Sum of log probabilities over response tokens, shape (batch,).
        Each element is a scalar representing the total log probability
        of the response given the prompt.
    """
    # Step 1: Get logits from the model
    # Shape: (batch, seq_len, vocab_size)
    logits = model(input_ids)

    # Step 2: Compute log softmax over the vocabulary dimension
    # Shape: (batch, seq_len, vocab_size)
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Step 3: Gather log probs of the actual response tokens
    # For token at position t, we use logits at position t-1
    # Response tokens are at positions [response_start_idx, seq_len)
    seq_len = input_ids.shape[1]
    response_len = seq_len - response_start_idx

    # Response tokens: input_ids[:, response_start_idx:]
    # Their log probs come from: log_probs_all[:, response_start_idx-1:seq_len-1, :]
    response_tokens = input_ids[:, response_start_idx:]  # (batch, response_len)
    response_logits = log_probs_all[:, response_start_idx - 1 : seq_len - 1, :]  # (batch, response_len, vocab)

    # Gather the log prob of each actual token
    # response_tokens.unsqueeze(-1) -> (batch, response_len, 1) for gather
    token_log_probs = response_logits.gather(
        dim=-1,
        index=response_tokens.unsqueeze(-1),
    ).squeeze(-1)  # (batch, response_len)

    # Step 4: Sum log probs over response tokens
    # Shape: (batch,)
    return token_log_probs.sum(dim=-1)


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss.

    DPO reparameterizes the RLHF objective to eliminate the reward model.
    The loss is derived from the Bradley-Terry preference model combined
    with the optimal policy under a KL constraint:

        r(x, y) = beta * (log pi(y|x) - log pi_ref(y|x))

    The implicit reward is the difference in log probs between the policy
    and reference model, scaled by beta. The loss then becomes:

        loss = -log sigmoid(
            beta * (log pi(y_w|x) - log pi_ref(y_w|x))
            - beta * (log pi(y_l|x) - log pi_ref(y_l|x))
        )

    Where y_w = chosen (winning) response, y_l = rejected (losing) response.

    When the policy assigns higher implicit reward to chosen than rejected,
    the loss is low. When the policy prefers rejected responses, the loss is high.

    The beta parameter controls how much the implicit reward is scaled.
    Higher beta makes the loss more sensitive to differences.

    Args:
        beta: Temperature parameter for the implicit reward (typically 0.1-0.5).
    """

    def __init__(self, beta: float = 0.1) -> None:
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_log_probs: torch.Tensor,
        policy_rejected_log_probs: torch.Tensor,
        ref_chosen_log_probs: torch.Tensor,
        ref_rejected_log_probs: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the DPO loss.

        Args:
            policy_chosen_log_probs: Log pi(y_w|x) from the policy, shape (batch,).
            policy_rejected_log_probs: Log pi(y_l|x) from the policy, shape (batch,).
            ref_chosen_log_probs: Log pi_ref(y_w|x) from the reference, shape (batch,).
            ref_rejected_log_probs: Log pi_ref(y_l|x) from the reference, shape (batch,).

        Returns:
            Scalar DPO loss (mean over the batch).
        """
        # Compute implicit reward differences
        # log_ratio_chosen = log pi(y_w|x) - log pi_ref(y_w|x)
        # log_ratio_rejected = log pi(y_l|x) - log pi_ref(y_l|x)
        log_ratio_chosen = policy_chosen_log_probs - ref_chosen_log_probs
        log_ratio_rejected = policy_rejected_log_probs - ref_rejected_log_probs

        # DPO loss: -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
        # Use logsigmoid for numerical stability
        loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected))

        return loss.mean()


class DPODataset(Dataset):
    """Dataset for DPO preference pairs with pre-computed reference log probs.

    Each example contains:
        - prompt: the input/prompt token IDs
        - chosen: the preferred response token IDs
        - rejected: the rejected response token IDs

    During initialization, the reference model is used to pre-compute
    log pi_ref(y|x) for both chosen and rejected responses. This avoids
    redundant computation during training since the reference model is frozen.

    The dataset stores:
        - chosen_input_ids: [prompt, chosen] concatenated
        - rejected_input_ids: [prompt, rejected] concatenated
        - chosen_ref_log_probs: sum of log probs for chosen response tokens
        - rejected_ref_log_probs: sum of log probs for rejected response tokens
        - response_start_idx: where the response begins (length of prompt)

    Args:
        pairs: List of dicts, each with "prompt", "chosen", "rejected" (token ID lists).
        ref_model: The reference (frozen) language model.
    """

    def __init__(
        self,
        pairs: List[Dict[str, List[int]]],
        ref_model: nn.Module,
    ) -> None:
        self.pairs = pairs

        # Freeze the reference model
        ref_model.training = False
        for param in ref_model.parameters():
            param.requires_grad = False

        # Pre-compute reference log probs
        self._chosen_input_ids = []
        self._rejected_input_ids = []
        self._chosen_ref_log_probs = []
        self._rejected_ref_log_probs = []
        self._response_start_indices = []

        with torch.no_grad():
            for pair in pairs:
                prompt = pair["prompt"]
                chosen = pair["chosen"]
                rejected = pair["rejected"]
                prompt_len = len(prompt)

                # Concatenate prompt + response
                chosen_ids = torch.tensor(prompt + chosen, dtype=torch.long).unsqueeze(0)
                rejected_ids = torch.tensor(prompt + rejected, dtype=torch.long).unsqueeze(0)

                # Compute reference log probs
                chosen_lp = compute_log_probs(ref_model, chosen_ids, prompt_len).squeeze(0)
                rejected_lp = compute_log_probs(ref_model, rejected_ids, prompt_len).squeeze(0)

                self._chosen_input_ids.append(chosen_ids.squeeze(0))
                self._rejected_input_ids.append(rejected_ids.squeeze(0))
                self._chosen_ref_log_probs.append(chosen_lp)
                self._rejected_ref_log_probs.append(rejected_lp)
                self._response_start_indices.append(prompt_len)

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair with pre-computed reference log probs.

        Args:
            idx: Index into the pairs list.

        Returns:
            Dict with:
                - "chosen_input_ids": tensor of [prompt, chosen] token IDs
                - "rejected_input_ids": tensor of [prompt, rejected] token IDs
                - "chosen_ref_log_probs": reference log prob for chosen response
                - "rejected_ref_log_probs": reference log prob for rejected response
                - "response_start_idx": int, where response tokens begin
        """
        return {
            "chosen_input_ids": self._chosen_input_ids[idx],
            "rejected_input_ids": self._rejected_input_ids[idx],
            "chosen_ref_log_probs": self._chosen_ref_log_probs[idx],
            "rejected_ref_log_probs": self._rejected_ref_log_probs[idx],
            "response_start_idx": self._response_start_indices[idx],
        }


class DPOTrainer:
    """DPO Trainer for language model alignment.

    Orchestrates the DPO training loop:
    1. Compute policy log probs for chosen and rejected responses
    2. Use pre-computed reference log probs from the dataset
    3. Compute DPO loss and update the policy

    Unlike PPO, DPO does not need:
    - A separate reward model
    - Rollout generation during training
    - Value function estimation
    - Multiple update epochs per batch

    This makes DPO simpler and more stable than PPO for many use cases.

    Args:
        model: The policy model (being trained).
        ref_model: The reference model (frozen, for KL baseline).
        beta: Temperature parameter for DPO loss (typically 0.1-0.5).
        lr: Learning rate for the optimizer.
        max_grad_norm: Max gradient norm for gradient clipping (0 = no clipping).
    """

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        beta: float = 0.1,
        lr: float = 5e-7,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.max_grad_norm = max_grad_norm

        self.dpo_loss = DPOLoss(beta=beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Freeze reference model
        self.ref_model.training = False
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one DPO training step.

        Args:
            batch: Dict with:
                - "chosen_input_ids": (batch, seq_len) token IDs for chosen
                - "rejected_input_ids": (batch, seq_len) token IDs for rejected
                - "chosen_ref_log_probs": (batch,) reference log probs for chosen
                - "rejected_ref_log_probs": (batch,) reference log probs for rejected
                - "response_start_idx": (batch,) or scalar, where response begins

        Returns:
            Dict with training metrics ("loss").
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Extract batch components
        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        ref_chosen_log_probs = batch["chosen_ref_log_probs"]
        ref_rejected_log_probs = batch["rejected_ref_log_probs"]
        response_start_idx = batch["response_start_idx"]

        # Handle scalar vs per-sample response_start_idx
        if response_start_idx.dim() == 0:
            response_start_idx = response_start_idx.item()
        else:
            # If per-sample, use the first one (assume all same prompt length)
            response_start_idx = response_start_idx[0].item()

        # Compute policy log probs
        policy_chosen_log_probs = compute_log_probs(
            self.model, chosen_input_ids, response_start_idx
        )
        policy_rejected_log_probs = compute_log_probs(
            self.model, rejected_input_ids, response_start_idx
        )

        # Compute DPO loss
        loss = self.dpo_loss(
            policy_chosen_log_probs,
            policy_rejected_log_probs,
            ref_chosen_log_probs,
            ref_rejected_log_probs,
        )

        # Backward and optimize
        loss.backward()

        if self.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.max_grad_norm
            )

        self.optimizer.step()

        return {"loss": loss.item()}
