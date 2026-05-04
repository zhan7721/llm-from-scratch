"""Online DPO -- Reference Solution.

This is the complete implementation. Try the exercise version first
and only check this if you are stuck.
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from torch.utils.data import Dataset

# Import DPO components from the previous chapter
sys.path.append(os.path.join(os.path.dirname(__file__), "../03_dpo"))
from dpo import compute_log_probs, DPOLoss


def generate_and_score(
    policy: nn.Module,
    reward_model: nn.Module,
    prompt: List[int],
    max_new_tokens: int = 32,
    num_candidates: int = 4,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """Generate multiple responses from the policy, score with reward model,
    and return the best (chosen) and worst (rejected) as a preference pair."""
    # Set models to inference mode
    policy.eval()
    reward_model.eval()

    prompt_tensor = torch.tensor(prompt, dtype=torch.long)
    prompt_len = len(prompt)

    # Step 1: Generate multiple candidate responses
    candidates = []
    with torch.no_grad():
        for _ in range(num_candidates):
            input_ids = prompt_tensor.clone().unsqueeze(0)

            for _ in range(max_new_tokens):
                logits = policy(input_ids)
                next_token_logits = logits[:, -1, :] / temperature
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                input_ids = torch.cat([input_ids, next_token], dim=1)

            candidates.append(input_ids.squeeze(0))

    # Step 2: Score each candidate with the reward model
    candidate_batch = torch.stack(candidates)
    with torch.no_grad():
        rewards = reward_model(candidate_batch)

    # Step 3: Select chosen (highest) and rejected (lowest)
    chosen_idx = rewards.argmax().item()
    rejected_idx = rewards.argmin().item()

    chosen_ids = candidates[chosen_idx]
    rejected_ids = candidates[rejected_idx]

    # Step 4: Compute reference log probs
    with torch.no_grad():
        chosen_ref_lp = compute_log_probs(
            policy, chosen_ids.unsqueeze(0), prompt_len
        ).squeeze(0)
        rejected_ref_lp = compute_log_probs(
            policy, rejected_ids.unsqueeze(0), prompt_len
        ).squeeze(0)

    return {
        "chosen_input_ids": chosen_ids,
        "rejected_input_ids": rejected_ids,
        "chosen_reward": rewards[chosen_idx],
        "rejected_reward": rewards[rejected_idx],
        "chosen_ref_log_probs": chosen_ref_lp,
        "rejected_ref_log_probs": rejected_ref_lp,
        "response_start_idx": prompt_len,
    }


class OnlineDPODataset(Dataset):
    """Dataset that generates preference pairs on-the-fly from the current policy."""

    def __init__(
        self,
        policy: nn.Module,
        reward_model: nn.Module,
        prompts: List[List[int]],
        max_new_tokens: int = 32,
        num_candidates: int = 4,
        temperature: float = 1.0,
    ) -> None:
        self.policy = policy
        self.reward_model = reward_model
        self.prompts = prompts
        self.max_new_tokens = max_new_tokens
        self.num_candidates = num_candidates
        self.temperature = temperature

        # Freeze reward model
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

        # Generate initial preference pairs
        self._pairs: List[Dict[str, torch.Tensor]] = []
        self.refresh()

    def refresh(self) -> None:
        """Regenerate all preference pairs from the current policy."""
        self._pairs = []
        for prompt in self.prompts:
            pair = generate_and_score(
                policy=self.policy,
                reward_model=self.reward_model,
                prompt=prompt,
                max_new_tokens=self.max_new_tokens,
                num_candidates=self.num_candidates,
                temperature=self.temperature,
            )
            self._pairs.append(pair)

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a preference pair with pre-computed reference log probs."""
        pair = self._pairs[idx]
        return {
            "chosen_input_ids": pair["chosen_input_ids"],
            "rejected_input_ids": pair["rejected_input_ids"],
            "chosen_ref_log_probs": pair["chosen_ref_log_probs"],
            "rejected_ref_log_probs": pair["rejected_ref_log_probs"],
            "response_start_idx": pair["response_start_idx"],
        }


class OnlineDPOTrainer:
    """Online DPO Trainer that generates preference data and trains with DPO."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        reward_model: nn.Module,
        prompts: List[List[int]],
        beta: float = 0.1,
        lr: float = 5e-7,
        max_grad_norm: float = 1.0,
        max_new_tokens: int = 32,
        num_candidates: int = 4,
        temperature: float = 1.0,
    ) -> None:
        self.model = model
        self.ref_model = ref_model
        self.reward_model = reward_model
        self.prompts = prompts
        self.max_grad_norm = max_grad_norm
        self.max_new_tokens = max_new_tokens
        self.num_candidates = num_candidates
        self.temperature = temperature

        self.dpo_loss = DPOLoss(beta=beta)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        # Freeze reference model
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False

        # Freeze reward model
        self.reward_model.eval()
        for param in self.reward_model.parameters():
            param.requires_grad = False

    def generate(self) -> OnlineDPODataset:
        """Generate fresh preference pairs from the current policy."""
        return OnlineDPODataset(
            policy=self.model,
            reward_model=self.reward_model,
            prompts=self.prompts,
            max_new_tokens=self.max_new_tokens,
            num_candidates=self.num_candidates,
            temperature=self.temperature,
        )

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
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

    def train_epoch(self, batch_size: int = 4) -> Dict[str, float]:
        """Run one full epoch of online DPO training."""
        dataset = self.generate()

        total_loss = 0.0
        num_batches = 0

        for i in range(0, len(dataset), batch_size):
            indices = list(range(i, min(i + batch_size, len(dataset))))
            items = [dataset[j] for j in indices]
            batch = {
                "chosen_input_ids": torch.stack([it["chosen_input_ids"] for it in items]),
                "rejected_input_ids": torch.stack([it["rejected_input_ids"] for it in items]),
                "chosen_ref_log_probs": torch.stack([it["chosen_ref_log_probs"] for it in items]),
                "rejected_ref_log_probs": torch.stack([it["rejected_ref_log_probs"] for it in items]),
                "response_start_idx": torch.tensor([it["response_start_idx"] for it in items]),
            }

            metrics = self.train_step(batch)
            total_loss += metrics["loss"]
            num_batches += 1

        return {"loss": total_loss / max(num_batches, 1)}
