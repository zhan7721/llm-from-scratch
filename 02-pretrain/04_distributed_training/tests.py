import torch
import torch.nn as nn
import pytest
from distributed import (
    DDPWrapper, FSDPConfig, DistributedSampler,
    get_gradient_accumulation_steps, setup_distributed, cleanup_distributed,
)


class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


def test_ddp_wrapper_single_gpu():
    model = DummyModel()
    device = torch.device("cpu")
    wrapper = DDPWrapper(model, device)
    x = torch.randn(2, 10)
    out = wrapper(x)
    assert out.shape == (2, 10)


def test_ddp_wrapper_parameters():
    model = DummyModel()
    wrapper = DDPWrapper(model, torch.device("cpu"))
    params = list(wrapper.parameters())
    assert len(params) > 0


def test_ddp_wrapper_state_dict():
    model = DummyModel()
    wrapper = DDPWrapper(model, torch.device("cpu"))
    state = wrapper.state_dict()
    assert "linear.weight" in state


def test_fsdp_config():
    config = FSDPConfig()
    assert config.sharding_strategy == "FULL_SHARD"
    assert config.mixed_precision is True


def test_fsdp_config_custom():
    config = FSDPConfig(
        sharding_strategy="SHARD_GRAD_OP",
        mixed_precision=False,
        activation_checkpointing=False,
    )
    assert config.sharding_strategy == "SHARD_GRAD_OP"
    assert config.mixed_precision is False


def test_gradient_accumulation_steps():
    # global_batch = micro_batch * world_size * grad_accum
    assert get_gradient_accumulation_steps(32, 4, 1) == 8
    assert get_gradient_accumulation_steps(32, 4, 2) == 4
    assert get_gradient_accumulation_steps(32, 4, 4) == 2
    assert get_gradient_accumulation_steps(32, 8, 4) == 1


def test_distributed_sampler_length():
    sampler = DistributedSampler(dataset_len=100, num_replicas=4, rank=0)
    assert len(sampler) == 25


def test_distributed_sampler_split():
    """Each rank should get different indices."""
    all_indices = set()
    for rank in range(4):
        sampler = DistributedSampler(dataset_len=100, num_replicas=4, rank=rank, shuffle=False)
        indices = list(sampler)
        all_indices.update(indices)
        assert len(indices) == 25

    # All 100 items should be covered
    assert len(all_indices) == 100


def test_distributed_sampler_shuffle():
    sampler = DistributedSampler(dataset_len=100, num_replicas=1, rank=0, shuffle=True)
    sampler.set_epoch(0)
    indices_0 = list(sampler)
    sampler.set_epoch(1)
    indices_1 = list(sampler)
    # Different epochs should produce different order
    assert indices_0 != indices_1
