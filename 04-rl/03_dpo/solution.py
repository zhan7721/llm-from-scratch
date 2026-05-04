"""Direct Preference Optimization (DPO) -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
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
    """Compute log probabilities of response tokens given a language model."""
    # Get logits from the model: (batch, seq_len, vocab_size)
    logits = model(input_ids)

    # Log softmax over vocabulary
    log_probs_all = F.log_softmax(logits, dim=-1)

    # Response tokens and their log probs
    seq_len = input_ids.shape[1]
    response_tokens = input_ids[:, response_start_idx:]
    response_logits = log_probs_all[:, response_start_idx - 1 : seq_len - 1, :]

    # Gather log probs of actual tokens
    token_log_probs = response_logits.gather(
        dim=-1,
        index=response_tokens.unsqueeze(-1),
    ).squeeze(-1)

    # Sum over response tokens
    return token_log_probs.sum(dim=-1)


class DPOLossSolution(nn.Module):
    """Direct Preference Optimization loss.

    loss = -log sigmoid(beta * (log_ratio_chosen - log_ratio_rejected))
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
        """Compute the DPO loss."""
        # Implicit reward differences
        log_ratio_chosen = policy_chosen_log_probs - ref_chosen_log_probs
        log_ratio_rejected = policy_rejected_log_probs - ref_rejected_log_probs

        # DPO loss with logsigmoid for numerical stability
        loss = -F.logsigmoid(self.beta * (log_ratio_chosen - log_ratio_rejected))

        return loss.mean()


class DPODatasetSolution(Dataset):
    """Dataset for DPO preference pairs with pre-computed reference log probs."""

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

                chosen_ids = torch.tensor(prompt + chosen, dtype=torch.long).unsqueeze(0)
                rejected_ids = torch.tensor(prompt + rejected, dtype=torch.long).unsqueeze(0)

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
        """Get a preference pair with pre-computed reference log probs."""
        return {
            "chosen_input_ids": self._chosen_input_ids[idx],
            "rejected_input_ids": self._rejected_input_ids[idx],
            "chosen_ref_log_probs": self._chosen_ref_log_probs[idx],
            "rejected_ref_log_probs": self._rejected_ref_log_probs[idx],
            "response_start_idx": self._response_start_indices[idx],
        }


class DPOTrainerSolution:
    """DPO Trainer for language model alignment."""

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

        self.dpo_loss = DPOLossSolution(beta=beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Freeze reference model
        self.ref_model.training = False
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one DPO training step."""
        self.model.train()
        self.optimizer.zero_grad()

        chosen_input_ids = batch["chosen_input_ids"]
        rejected_input_ids = batch["rejected_input_ids"]
        ref_chosen_log_probs = batch["chosen_ref_log_probs"]
        ref_rejected_log_probs = batch["rejected_ref_log_probs"]
        response_start_idx = batch["response_start_idx"]

        if response_start_idx.dim() == 0:
            response_start_idx = response_start_idx.item()
        else:
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
