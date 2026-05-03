"""Evaluation exercises -- fill in the TODO sections."""

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

    Steps:
    1. Set model to eval mode.
    2. Iterate over batches, move input_ids and labels to device.
    3. Forward pass to get logits.
    4. Compute cross-entropy loss (sum, ignore_index=-100).
    5. Count non-ignored tokens.
    6. Return exp(total_loss / total_tokens).

    Args:
        model: Language model that returns logits.
        data_loader: DataLoader yielding dicts with "input_ids" and "labels".
        device: Device to run on.

    Returns:
        Perplexity value (lower is better).
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

            # TODO: Compute cross-entropy loss with reduction="sum" and ignore_index=-100
            loss = ...  # F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ...)

            # TODO: Count non-ignored tokens
            n_tokens = ...  # (labels != -100).sum().item()

            total_loss += loss.item()
            total_tokens += n_tokens

    # TODO: Return exp(average loss)
    return ...  # math.exp(total_loss / max(total_tokens, 1))


def compute_token_accuracy(model: nn.Module, data_loader, device: torch.device = torch.device("cpu")) -> float:
    """Compute next-token prediction accuracy.

    Steps:
    1. Set model to eval mode.
    2. Iterate over batches.
    3. Get predictions via argmax on logits.
    4. Mask positions where labels == -100.
    5. Count correct predictions among unmasked positions.
    6. Return correct / total.

    Args:
        model: Language model.
        data_loader: DataLoader with input_ids and labels.
        device: Device.

    Returns:
        Accuracy as a float in [0, 1].
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids)
            logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

            # TODO: Get predictions via argmax
            preds = ...  # logits.argmax(dim=-1)

            # TODO: Create mask for non-ignored positions
            mask = ...  # labels != -100

            # TODO: Count correct predictions where mask is True
            correct += ...  # ((preds == labels) & mask).sum().item()
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
        """Run full evaluation.

        Steps:
        1. Compute perplexity on data_loader.
        2. Compute token accuracy on data_loader.
        3. If tokenizer and prompts are provided, generate samples.
        4. Return dict of all metrics.

        Args:
            data_loader: DataLoader for loss/accuracy computation.
            tokenizer: Tokenizer for generation (optional).
            prompts: Prompts for generation samples (optional).

        Returns:
            Dict of metric name -> value.
        """
        results = {}

        # TODO: Compute perplexity
        # ppl = perplexity(self.model, data_loader, self.device)
        # results["perplexity"] = ppl

        # TODO: Compute token accuracy
        # acc = compute_token_accuracy(self.model, data_loader, self.device)
        # results["token_accuracy"] = acc

        # TODO: If tokenizer and prompts are given, generate samples
        # if tokenizer is not None and prompts is not None:
        #     from evaluation import generate_samples
        #     samples = generate_samples(self.model, tokenizer, prompts, device=self.device)
        #     results["generation_samples"] = samples

        return results
