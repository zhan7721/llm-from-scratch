"""Training Loop for LLM Pretraining -- Exercise.

Fill in the TODO methods to implement the training loop components.
Start with `CosineLRScheduler.get_lr_scale`, then `GradientClipper.clip`,
and finally `training_step`.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, List
import math
import time


class CosineLRSchedulerExercise:
    """Cosine annealing with linear warmup.

    LR schedule: warmup -> cosine decay -> min_lr
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.1,
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [group["lr"] for group in optimizer.param_groups]

    def get_lr_scale(self, step: int) -> float:
        """Get LR scale factor for given step.

        Args:
            step: The current training step number.

        Returns:
            A float between 0 and 1 that multiplies the base LR.

        Hints:
            - During warmup (step < warmup_steps), LR scales linearly
              from 0 to 1. The formula is: step / warmup_steps.
            - After warmup, use cosine decay. Compute the progress through
              the decay phase as a fraction from 0 to 1.
            - The cosine formula: min_lr_ratio + 0.5 * (1 - min_lr_ratio) *
              (1 + cos(pi * progress))
            - Use math.cos and math.pi.
            - Be careful with division by zero when warmup_steps=0 or when
              total_steps == warmup_steps. Use max(1, ...) as a guard.
        """
        raise NotImplementedError("TODO: implement get_lr_scale")

    def step(self, step: int):
        """Update learning rate for given step."""
        scale = self.get_lr_scale(step)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class GradientClipperExercise:
    """Gradient clipping by global norm."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, model: nn.Module) -> float:
        """Clip gradients and return the total norm before clipping.

        Args:
            model: The model whose gradients to clip.

        Returns:
            The total gradient norm before clipping (a float).

        Hints:
            - Use `torch.nn.utils.clip_grad_norm_()`.
            - It takes: model.parameters(), max_norm, norm_type.
            - It returns the total norm before clipping (as a tensor).
            - Call .item() on the result to get a Python float.
        """
        raise NotImplementedError("TODO: implement clip")


class TrainingMetricsExercise:
    """Track and compute running averages for training metrics."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.losses = []
        self.grad_norms = []
        self.learning_rates = []
        self.start_time = time.time()

    def update(self, loss: float, grad_norm: float, lr: float):
        self.losses.append(loss)
        self.grad_norms.append(grad_norm)
        self.learning_rates.append(lr)

    @property
    def avg_loss(self) -> float:
        return sum(self.losses) / max(len(self.losses), 1)

    @property
    def avg_grad_norm(self) -> float:
        return sum(self.grad_norms) / max(len(self.grad_norms), 1)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    def summary(self) -> Dict:
        return {
            "avg_loss": self.avg_loss,
            "avg_grad_norm": self.avg_grad_norm,
            "latest_lr": self.learning_rates[-1] if self.learning_rates else 0,
            "elapsed_seconds": self.elapsed_time,
            "steps": len(self.losses),
        }


def training_step_exercise(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRSchedulerExercise,
    clipper: GradientClipperExercise,
    step: int,
    gradient_accumulation_steps: int = 1,
    scaler=None,
) -> Dict[str, float]:
    """Execute a single training step.

    Args:
        model: The model to train.
        batch: Dict with "input_ids", "labels", optionally "attention_mask".
        optimizer: Optimizer.
        scheduler: LR scheduler.
        clipper: Gradient clipper.
        step: Current step number.
        gradient_accumulation_steps: Accumulate gradients over N steps.
        scaler: Optional GradScaler for mixed precision.

    Returns:
        Dict with loss, grad_norm, lr.

    Hints:
        1. Set model to train mode with model.train().
        2. Move input_ids and labels to the model's device.
        3. Forward pass: call model(input_ids). The output may be a tensor
           or an object with a .logits attribute.
        4. Compute cross-entropy loss with ignore_index=-100.
        5. Divide loss by gradient_accumulation_steps before backward.
        6. Call loss.backward().
        7. If this is an accumulation boundary ((step + 1) % accum == 0):
           - Clip gradients with clipper.clip(model).
           - Call optimizer.step() and scheduler.step(step).
           - Call optimizer.zero_grad().
        8. Return dict with loss, grad_norm, lr.
    """
    raise NotImplementedError("TODO: implement training_step")
