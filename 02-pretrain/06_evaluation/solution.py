"""Evaluation solution -- complete reference implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../01-foundations/05_model_architecture"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../01-foundations/01_tokenizer"))


def perplexity(model: nn.Module, data_loader, device: torch.device = torch.device("cpu")) -> float:
    """Compute perplexity of a model on a dataset.

    Perplexity = exp(average cross-entropy loss)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            # Compute cross-entropy loss
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
                reduction="sum",
            )

            # Count non-ignored tokens
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens

    avg_loss = total_loss / max(total_tokens, 1)
    return math.exp(avg_loss)


def compute_token_accuracy(model: nn.Module, data_loader, device: torch.device = torch.device("cpu")) -> float:
    """Compute next-token prediction accuracy."""
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            # Get predictions
            preds = logits.argmax(dim=-1)

            # Mask ignored positions
            mask = labels != -100
            correct += ((preds == labels) & mask).sum().item()
            total += mask.sum().item()

    return correct / max(total, 1)


class Evaluator:
    """Orchestrates evaluation with multiple metrics."""

    def __init__(self, model: nn.Module, device: torch.device = torch.device("cpu")):
        self.model = model
        self.device = device

    def evaluate(
        self,
        data_loader,
        tokenizer=None,
        prompts: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Run full evaluation."""
        results = {}

        # Perplexity
        ppl = perplexity(self.model, data_loader, self.device)
        results["perplexity"] = ppl

        # Token accuracy
        acc = compute_token_accuracy(self.model, data_loader, self.device)
        results["token_accuracy"] = acc

        # Generation samples
        if tokenizer is not None and prompts is not None:
            from evaluation import generate_samples
            samples = generate_samples(
                self.model, tokenizer, prompts, device=self.device
            )
            results["generation_samples"] = samples

        return results
