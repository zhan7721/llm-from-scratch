"""Evaluation utilities for pretrained language models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import math
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "../../01-foundations/05_model_architecture"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../01-foundations/01_tokenizer"))


def perplexity(model: nn.Module, data_loader, device: torch.device = torch.device("cpu")) -> float:
    """Compute perplexity of a model on a dataset.

    Perplexity = exp(average cross-entropy loss)

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


@torch.no_grad()
def generate_samples(
    model: nn.Module,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int = 50,
    temperature: float = 0.8,
    top_k: int = 50,
    device: torch.device = torch.device("cpu"),
) -> List[str]:
    """Generate text samples from a list of prompts.

    Args:
        model: Language model with generate() method.
        tokenizer: Tokenizer with encode() and decode() methods.
        prompts: List of prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature.
        top_k: Top-K sampling.
        device: Device to run on.

    Returns:
        List of generated text strings.
    """
    model.eval()
    results = []

    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

        generated = model.generate(
            input_tensor,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

        output_text = tokenizer.decode(generated[0].tolist())
        results.append(output_text)

    return results


def compute_token_accuracy(model: nn.Module, data_loader, device: torch.device = torch.device("cpu")) -> float:
    """Compute next-token prediction accuracy.

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
        """Run full evaluation.

        Args:
            data_loader: DataLoader for loss/accuracy computation.
            tokenizer: Tokenizer for generation (optional).
            prompts: Prompts for generation samples (optional).

        Returns:
            Dict of metric name -> value.
        """
        results = {}

        # Perplexity
        ppl = perplexity(self.model, data_loader, self.device)
        results["perplexity"] = ppl

        # Token accuracy
        acc = compute_token_accuracy(self.model, data_loader, self.device)
        results["token_accuracy"] = acc

        # Generation samples
        if tokenizer is not None and prompts is not None:
            samples = generate_samples(
                self.model, tokenizer, prompts, device=self.device
            )
            results["generation_samples"] = samples

        return results
