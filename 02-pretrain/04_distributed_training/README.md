# Distributed Training for LLM Pretraining

## Overview

As language models grow larger, training on a single GPU becomes impractical. A model with 7 billion parameters requires ~14 GB just to store the weights in fp16, and training requires additional memory for gradients, optimizer states, and activations. Distributed training spreads this workload across multiple GPUs and machines.

This chapter covers the core distributed training concepts and utilities you need to pretrain an LLM.

## Why Distributed Training?

Modern LLMs require:

- **More memory**: A 7B model needs ~14 GB for parameters alone. With Adam optimizer states (2x parameters) and gradients, you need ~56 GB — far beyond a single GPU.
- **More compute**: Training on trillions of tokens would take years on one GPU.
- **Larger batch sizes**: Larger batches stabilize training and improve throughput.

Distributed training solves these problems by parallelizing across multiple GPUs.

## Data Parallel vs Distributed Data Parallel

### Data Parallel (DP)

The older `torch.nn.DataParallel` approach:

- Uses a single process with multiple GPUs
- The master GPU collects gradients from all GPUs
- Creates a **bottleneck** on the master GPU
- Does not scale well beyond a single machine

### Distributed Data Parallel (DDP)

The recommended approach with `torch.nn.parallel.DistributedDataParallel`:

- Each GPU runs its own process
- Gradients are averaged using **AllReduce** (no single bottleneck)
- Each process has its own optimizer
- Scales efficiently across multiple machines
- Uses the **NCCL** backend for GPU communication

```
DDP Training Loop (per GPU):

1. Each GPU loads a different mini-batch (via DistributedSampler)
2. Forward pass — compute loss
3. Backward pass — compute gradients
4. AllReduce — average gradients across all GPUs
5. Each GPU updates its optimizer independently
6. All GPUs now have identical weights
```

## Key Environment Variables

When using `torchrun`, the launcher sets these environment variables:

| Variable       | Description                          | Example |
|----------------|--------------------------------------|---------|
| `RANK`         | Global rank of this process          | 0-7     |
| `WORLD_SIZE`   | Total number of processes            | 8       |
| `LOCAL_RANK`   | Rank on this machine                 | 0-3     |
| `MASTER_ADDR`  | Address of the rank 0 process        | "localhost" |
| `MASTER_PORT`  | Port for communication               | 29500   |

- **RANK**: Unique ID across all machines (0 to WORLD_SIZE - 1)
- **WORLD_SIZE**: Total number of processes participating in training
- **LOCAL_RANK**: GPU index on the current machine (0 to num_gpus_per_machine - 1)

## Launching Distributed Training

### Using `torchrun` (recommended)

```bash
# Single machine, 4 GPUs
torchrun --standalone --nproc_per_node=4 train.py

# Multi-machine (2 machines, 4 GPUs each)
# On machine 0:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 train.py

# On machine 1:
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr="192.168.1.1" --master_port=29500 train.py
```

### In Your Training Script

```python
from distributed import setup_distributed, cleanup_distributed, DDPWrapper

def main():
    rank, world_size, local_rank, device = setup_distributed()

    model = build_model()
    wrapper = DDPWrapper(model, device)

    train(wrapper, device)

    cleanup_distributed()
```

## Gradient Accumulation

When your desired global batch size exceeds what fits in GPU memory:

```
global_batch_size = micro_batch_size * world_size * gradient_accumulation_steps
```

Example: You want global batch size = 512, but each GPU can only handle 8 samples, and you have 4 GPUs:

```
512 = 8 * 4 * gradient_accumulation_steps
gradient_accumulation_steps = 16
```

Each GPU processes 16 micro-batches before the AllReduce step.

```python
optimizer.zero_grad()
for step in range(gradient_accumulation_steps):
    batch = get_batch()
    loss = model(batch) / gradient_accumulation_steps  # Scale loss
    loss.backward()  # Gradients accumulate

optimizer.step()  # Single update after all accumulation steps
```

## FSDP: Fully Sharded Data Parallel

FSDP goes beyond DDP by **sharding** model parameters, gradients, and optimizer states across GPUs:

| Aspect          | DDP                        | FSDP                          |
|-----------------|----------------------------|-------------------------------|
| Parameters      | Replicated on every GPU    | Sharded across GPUs           |
| Gradients       | Replicated after AllReduce | Sharded                       |
| Optimizer states| Replicated                 | Sharded                       |
| Memory usage    | O(model_size) per GPU      | O(model_size / num_gpus)      |
| Communication   | AllReduce                  | AllGather + ReduceScatter     |

### FSDP Sharding Strategies

- **FULL_SHARD**: Maximum memory savings. Shards parameters, gradients, and optimizer states.
- **SHARD_GRAD_OP**: Shards gradients and optimizer states only. Good middle ground.
- **NO_SHARD**: Equivalent to DDP. Useful for debugging.

### When to Use FSDP

Use FSDP when:
- Your model doesn't fit on a single GPU even with DDP
- You want to use larger batch sizes
- You're training models > 1B parameters

## Code Walkthrough

### `setup_distributed()`

Initializes the distributed process group:

```python
def setup_distributed(backend="nccl"):
    # Read rank info from environment (set by torchrun)
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
```

### `DDPWrapper`

A convenience wrapper that:
- Wraps with DDP when distributed is available
- Falls back to plain model otherwise
- Handles state_dict correctly (strips DDP's `module.` prefix)

### `DistributedSampler`

Ensures each GPU sees different data:
- Partitions indices into `world_size` chunks
- Each rank gets its chunk
- Shuffling is seeded by epoch for reproducibility

### `FSDPConfig`

Stores FSDP configuration parameters for use with PyTorch's FSDP implementation. Not a full FSDP wrapper — use `torch.distributed.fsdp.FullyShardedDataParallel` directly for actual sharding.

## Best Practices

1. **Always use DDP over DP** — it's faster and scales better
2. **Set the correct device** — use `torch.cuda.set_device(local_rank)` to avoid GPU conflicts
3. **Use gradient accumulation** — to simulate larger batch sizes
4. **Save checkpoints from rank 0 only** — all ranks have identical weights
5. **Use mixed precision** — reduces memory and speeds up training
6. **Profile before scaling** — identify bottlenecks with `torch.profiler`

## Common Pitfalls

- **Forgetting `set_epoch()` on sampler** — leads to identical data ordering across epochs
- **Not calling `cleanup_distributed()`** — can cause hangs on exit
- **Saving from all ranks** — wastes I/O; save only from rank 0
- **Non-deterministic operations** — can cause ranks to diverge silently

## References

- [PyTorch DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP Tutorial](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [torchrun Documentation](https://pytorch.org/docs/stable/elastic/run.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) — reference for large-scale training
