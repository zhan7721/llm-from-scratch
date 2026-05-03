# LLM 预训练的分布式训练

## 概述

随着语言模型规模不断增长，在单个 GPU 上进行训练变得不切实际。一个 70 亿参数的模型仅存储 fp16 权重就需要约 14 GB，而训练还需要额外的内存来存储梯度、优化器状态和激活值。分布式训练将这些工作负载分散到多个 GPU 和多台机器上。

本章涵盖预训练 LLM 所需的核心分布式训练概念和工具。

## 为什么需要分布式训练？

现代 LLM 需要：

- **更多内存**：7B 模型仅参数就需要约 14 GB。加上 Adam 优化器状态（2 倍参数量）和梯度，总共需要约 56 GB —— 远超单个 GPU 的容量。
- **更多计算**：在单个 GPU 上处理数万亿个 token 需要数年时间。
- **更大的批大小**：更大的批大小能稳定训练并提高吞吐量。

分布式训练通过在多个 GPU 间并行化来解决这些问题。

## 数据并行 vs 分布式数据并行

### 数据并行 (DP)

旧版 `torch.nn.DataParallel` 方式：

- 使用单进程多 GPU
- 主 GPU 从所有 GPU 收集梯度
- 在主 GPU 上产生**瓶颈**
- 扩展性不佳

### 分布式数据并行 (DDP)

推荐使用 `torch.nn.parallel.DistributedDataParallel`：

- 每个 GPU 运行独立进程
- 通过 **AllReduce** 平均梯度（无单点瓶颈）
- 每个进程有自己的优化器
- 可高效扩展到多台机器
- 使用 **NCCL** 后端进行 GPU 通信

```
DDP 训练循环（每个 GPU）：

1. 每个 GPU 加载不同的小批次（通过 DistributedSampler）
2. 前向传播 —— 计算损失
3. 反向传播 —— 计算梯度
4. AllReduce —— 在所有 GPU 间平均梯度
5. 每个 GPU 独立更新优化器
6. 所有 GPU 现在拥有相同的权重
```

## 关键环境变量

使用 `torchrun` 时，启动器会设置以下环境变量：

| 变量           | 描述                                 | 示例     |
|----------------|--------------------------------------|----------|
| `RANK`         | 当前进程的全局排名                    | 0-7      |
| `WORLD_SIZE`   | 进程总数                              | 8        |
| `LOCAL_RANK`   | 当前机器上的排名                      | 0-3      |
| `MASTER_ADDR`  | rank 0 进程的地址                     | "localhost" |
| `MASTER_PORT`  | 通信端口                              | 29500    |

- **RANK**：跨所有机器的唯一 ID（0 到 WORLD_SIZE - 1）
- **WORLD_SIZE**：参与训练的进程总数
- **LOCAL_RANK**：当前机器上的 GPU 索引（0 到每台机器的 GPU 数 - 1）

## 启动分布式训练

### 使用 `torchrun`（推荐）

```bash
# 单机 4 GPU
torchrun --standalone --nproc_per_node=4 train.py

# 多机（2 台机器，每台 4 GPU）
# 在机器 0 上：
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
    --master_addr="192.168.1.1" --master_port=29500 train.py

# 在机器 1 上：
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
    --master_addr="192.168.1.1" --master_port=29500 train.py
```

### 在训练脚本中

```python
from distributed import setup_distributed, cleanup_distributed, DDPWrapper

def main():
    rank, world_size, local_rank, device = setup_distributed()

    model = build_model()
    wrapper = DDPWrapper(model, device)

    train(wrapper, device)

    cleanup_distributed()
```

## 梯度累积

当目标全局批大小超出 GPU 内存容量时：

```
全局批大小 = 微批大小 * GPU 数量 * 梯度累积步数
```

示例：目标全局批大小 = 512，但每个 GPU 只能处理 8 个样本，有 4 个 GPU：

```
512 = 8 * 4 * 梯度累积步数
梯度累积步数 = 16
```

每个 GPU 在 AllReduce 步骤前处理 16 个微批次。

```python
optimizer.zero_grad()
for step in range(gradient_accumulation_steps):
    batch = get_batch()
    loss = model(batch) / gradient_accumulation_steps  # 缩放损失
    loss.backward()  # 梯度累积

optimizer.step()  # 所有累积步数后单次更新
```

## FSDP：全分片数据并行

FSDP 超越 DDP，通过**分片**模型参数、梯度和优化器状态来节省内存：

| 方面             | DDP                    | FSDP                      |
|------------------|------------------------|---------------------------|
| 参数             | 每个 GPU 上复制        | 跨 GPU 分片               |
| 梯度             | AllReduce 后复制       | 分片                      |
| 优化器状态       | 复制                   | 分片                      |
| 内存使用         | 每 GPU O(model_size)   | 每 GPU O(model_size/n)    |
| 通信             | AllReduce              | AllGather + ReduceScatter |

### FSDP 分片策略

- **FULL_SHARD**：最大内存节省。分片参数、梯度和优化器状态。
- **SHARD_GRAD_OP**：仅分片梯度和优化器状态。良好的折中方案。
- **NO_SHARD**：等同于 DDP。用于调试。

### 何时使用 FSDP

在以下情况下使用 FSDP：
- 即使用 DDP，模型也无法放入单个 GPU
- 你想使用更大的批大小
- 你在训练超过 10 亿参数的模型

## 代码讲解

### `setup_distributed()`

初始化分布式进程组：

```python
def setup_distributed(backend="nccl"):
    # 从环境变量读取排名信息（由 torchrun 设置）
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if world_size > 1:
        torch.distributed.init_process_group(backend=backend)
        torch.cuda.set_device(local_rank)
```

### `DDPWrapper`

一个便捷封装器：
- 在分布式可用时使用 DDP 封装
- 否则回退到普通模型
- 正确处理 state_dict（去除 DDP 的 `module.` 前缀）

### `DistributedSampler`

确保每个 GPU 看到不同的数据：
- 将索引分成 `world_size` 个块
- 每个 rank 获得自己的块
- 洗牌以 epoch 为种子，保证可复现性

### `FSDPConfig`

存储 FSDP 配置参数，用于 PyTorch 的 FSDP 实现。不是完整的 FSDP 封装 —— 实际分片请直接使用 `torch.distributed.fsdp.FullyShardedDataParallel`。

## 最佳实践

1. **始终使用 DDP 而非 DP** —— 更快且扩展性更好
2. **设置正确的设备** —— 使用 `torch.cuda.set_device(local_rank)` 避免 GPU 冲突
3. **使用梯度累积** —— 模拟更大的批大小
4. **仅从 rank 0 保存检查点** —— 所有 rank 拥有相同权重
5. **使用混合精度** —— 减少内存并加速训练
6. **先分析再扩展** —— 使用 `torch.profiler` 识别瓶颈

## 常见陷阱

- **忘记对采样器调用 `set_epoch()`** —— 导致跨 epoch 数据顺序相同
- **未调用 `cleanup_distributed()`** —— 可能导致退出时挂起
- **从所有 rank 保存** —— 浪费 I/O；仅从 rank 0 保存
- **非确定性操作** —— 可能导致 rank 静默发散

## 参考资料

- [PyTorch DDP 教程](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [PyTorch FSDP 教程](https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [torchrun 文档](https://pytorch.org/docs/stable/elastic/run.html)
- [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) —— 大规模训练参考
