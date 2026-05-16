"""Training loop components: LR scheduler, gradient clipping, full training loop."""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from typing import Optional, Dict, List
import math
import time


class CosineLRScheduler:
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
        """Get LR scale factor for given step."""
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    def step(self, step: int):
        """Update learning rate for given step."""
        scale = self.get_lr_scale(step)
        for base_lr, group in zip(self.base_lrs, self.optimizer.param_groups):
            group["lr"] = base_lr * scale

    def get_last_lr(self) -> List[float]:
        return [group["lr"] for group in self.optimizer.param_groups]


class GradientClipper:
    """Gradient clipping by global norm."""

    def __init__(self, max_norm: float = 1.0, norm_type: float = 2.0):
        self.max_norm = max_norm
        self.norm_type = norm_type

    def clip(self, model: nn.Module) -> float:
        """Clip gradients and return the total norm before clipping."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), self.max_norm, norm_type=self.norm_type
        )
        return total_norm.item()


class TrainingMetrics:
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


def create_training_components(
    model: nn.Module,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.1,
    warmup_steps: int = 100,
    total_steps: int = 10000,
    max_grad_norm: float = 1.0,
    betas: tuple = (0.9, 0.95),
) -> Dict:
    """Create all training components.

    Returns dict with optimizer, scheduler, clipper, metrics.
    """
    # Separate weight decay params
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]

    optimizer = AdamW(
        [{"params": decay_params, "weight_decay": weight_decay},
         {"params": no_decay_params, "weight_decay": 0.0}],
        lr=learning_rate,
        betas=betas,
    )

    scheduler = CosineLRScheduler(optimizer, warmup_steps, total_steps)
    clipper = GradientClipper(max_grad_norm)
    metrics = TrainingMetrics()

    return {
        "optimizer": optimizer,
        "scheduler": scheduler,
        "clipper": clipper,
        "metrics": metrics,
    }


def training_step(
    model: nn.Module,
    batch: Dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    scheduler: CosineLRScheduler,
    clipper: GradientClipper,
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
    """
    model.train()

    # Forward pass
    device = next(model.parameters()).device
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    if scaler is not None:
        with torch.amp.autocast(device_type=device.type if device.type != "mps" else "cpu"):
            outputs = model(input_ids)
            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:
                logits = outputs
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=-100,
            )
            loss = loss / gradient_accumulation_steps
        scaler.scale(loss).backward()
    else:
        outputs = model(input_ids)
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        loss = nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100,
        )
        loss = loss / gradient_accumulation_steps
        loss.backward()

    # Track unscaled loss for logging
    log_loss = loss.item() * gradient_accumulation_steps

    # Step optimizer if accumulation is complete
    if (step + 1) % gradient_accumulation_steps == 0:
        if scaler is not None:
            scaler.unscale_(optimizer)

        grad_norm = clipper.clip(model)

        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        scheduler.step(step)
        optimizer.zero_grad()
    else:
        grad_norm = 0.0

    current_lr = scheduler.get_last_lr()[0]

    return {
        "loss": log_loss,
        "grad_norm": grad_norm,
        "lr": current_lr,
    }
