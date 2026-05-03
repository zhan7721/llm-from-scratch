"""Distributed training solution — complete reference implementation.

This file contains the full solutions for all exercises in exercise.py.
"""

import os
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from dataclasses import dataclass


class DDPWrapper:
    """Simplified DDP wrapper that handles model wrapping and gradient averaging.

    Wraps a model for distributed data parallel training.
    Falls back to single-GPU mode when distributed is not available.
    """

    def __init__(self, model: nn.Module, device: torch.device, find_unused_parameters: bool = False):
        self.model = model
        self.device = device
        self.is_distributed = (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        )

        if self.is_distributed:
            self.model = nn.parallel.DistributedDataParallel(
                model.to(device),
                device_ids=[device.index] if device.type == "cuda" else None,
                find_unused_parameters=find_unused_parameters,
            )
        else:
            self.model = model.to(device)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def parameters(self):
        return self.model.parameters()

    def state_dict(self):
        # Always get the underlying model's state dict (without DDP prefix)
        if self.is_distributed:
            return self.model.module.state_dict()
        return self.model.state_dict()

    def load_state_dict(self, state_dict):
        if self.is_distributed:
            self.model.module.load_state_dict(state_dict)
        else:
            self.model.load_state_dict(state_dict)


@dataclass
class FSDPConfig:
    """Configuration for Fully Sharded Data Parallel (FSDP).

    Note: Actual FSDP requires PyTorch 2.0+ and specific setup.
    This config stores the parameters; actual FSDP wrapping uses
    torch.distributed.fsdp.FullyShardedDataParallel.
    """
    sharding_strategy: str = "FULL_SHARD"  # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD
    mixed_precision: bool = True
    activation_checkpointing: bool = True
    min_num_params: int = 1_000_000  # Minimum params to wrap a layer
    backward_prefetch: str = "BACKWARD_PRE"  # BACKWARD_PRE, BACKWARD_POST
    cpu_offload: bool = False


def get_gradient_accumulation_steps(
    global_batch_size: int,
    micro_batch_size: int,
    world_size: int,
) -> int:
    """Calculate gradient accumulation steps for target global batch size.

    global_batch_size = micro_batch_size * world_size * gradient_accumulation_steps
    """
    return max(1, global_batch_size // (micro_batch_size * world_size))


class DistributedSampler:
    """Simple distributed sampler that splits data across ranks."""

    def __init__(self, dataset_len: int, num_replicas: int = 1, rank: int = 0, shuffle: bool = True):
        self.dataset_len = dataset_len
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        self.epoch = 0

    def __len__(self) -> int:
        return (self.dataset_len + self.num_replicas - 1) // self.num_replicas

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        indices = list(range(self.dataset_len))
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(indices), generator=g).tolist()

        # Split into num_replicas chunks
        per_replica = (len(indices) + self.num_replicas - 1) // self.num_replicas
        start = self.rank * per_replica
        end = min(start + per_replica, len(indices))
        return iter(indices[start:end])
