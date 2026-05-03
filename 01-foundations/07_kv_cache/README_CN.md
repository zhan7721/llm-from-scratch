# KV 缓存：高效自回归生成

## 问题：朴素生成效率低下

使用 Transformer 模型生成文本时，我们通常采用**自回归**方式——每次生成一个 token。朴素方法如下：

```
步骤 1: 从输入 [prompt] 生成 token 1
步骤 2: 从输入 [prompt, token1] 生成 token 2
步骤 3: 从输入 [prompt, token1, token2] 生成 token 3
...
```

在每一步，我们对**整个序列**运行**完整的注意力计算**。这意味着：
- 步骤 1: O(1) 注意力
- 步骤 2: O(2) 注意力
- 步骤 3: O(3) 注意力
- ...
- 步骤 n: O(n) 注意力

**总成本: O(n²)**——而大部分工作都是**重复的**！在步骤 3，我们已经在步骤 2 计算过位置 1 和 2 的 Key 和 Value 向量了。为什么要重新计算？

## 解决方案：KV 缓存

**KV 缓存**存储之前步骤的 Key 和 Value 张量，因此我们只需要计算**新 token** 的 K 和 V。

### 工作原理

1. **预分配**足够容纳最大序列长度的缓存缓冲区
2. 在每个生成步骤：
   - 只为**当前 token** 计算 Q, K, V（seq_len = 1）
   - **更新**缓存，在当前位置写入新的 K, V
   - **检索**缓存中的完整 K, V 历史
   - 计算注意力：新的 Q 对所有缓存的 K, V

```python
# 无缓存（朴素）：
for i in range(seq_len):
    # 从头重新计算所有 K, V
    q, k, v = attention(full_sequence[:i+1])

# 有缓存（高效）：
cache = KVCache()
for i in range(seq_len):
    # 只计算新的 K, V
    q, k_new, v_new = attention(current_token)
    cache.update(i, k_new, v_new)
    k_all, v_all = cache.get(i + 1)
    output = attend(q, k_all, v_all)
```

### 复杂度对比

| 方法 | 每步成本 | 总成本（n 个 token） | 内存 |
|------|---------|---------------------|------|
| 朴素 | O(i) | O(n²) | O(1) |
| KV 缓存 | O(1) | O(n) | O(n) |

KV 缓存用**内存**换**计算**——我们存储 O(n) 的缓存条目，但避免了 O(n²) 的重复计算。

## 实现细节

### KVCache 类

```python
class KVCache:
    def __init__(self, max_batch_size, max_seq_len, n_heads, d_k):
        # 预分配缓冲区
        self.k_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
        self.v_cache = torch.zeros(max_batch_size, n_heads, max_seq_len, d_k)
```

关键设计决策：
- **预分配**：预先分配完整缓冲区，避免重复内存分配
- **原地更新**：直接写入预分配的缓冲区
- **批处理感知**：高效处理多个序列

### CachedAttention 类

注意力模块需要处理两种模式：
1. **预填充模式**：处理初始提示（seq_len > 1），不需要缓存
2. **生成模式**：一次生成一个 token（seq_len = 1），使用缓存

```python
def forward(self, x, kv_cache=None, start_pos=0):
    B, T, _ = x.shape

    # 始终为当前输入计算 Q, K, V
    q = self.W_q(x).view(B, T, n_heads, d_k).transpose(1, 2)
    k = self.W_k(x).view(B, T, n_heads, d_k).transpose(1, 2)
    v = self.W_v(x).view(B, T, n_heads, d_k).transpose(1, 2)

    if kv_cache is not None:
        # 存储新的 K, V 并检索完整历史
        kv_cache.update(B, start_pos, k, v)
        k, v = kv_cache.get(B, start_pos + T)

    # 标准注意力计算
    scores = q @ k.T / sqrt(d_k)
    # 如果需要则应用因果掩码
    output = softmax(scores) @ v
    return self.W_o(output)
```

## 多批次处理

在批次中处理多个序列时：
- 所有序列共享同一个缓存缓冲区
- 每个序列使用位置 `[0:actual_length]`
- 将序列填充到相同长度会浪费缓存空间
- 高级方案：使用变长打包提高效率

```python
# 批次处理
cache = KVCache(max_batch_size=4, max_seq_len=100, n_heads=8, d_k=64)

# 同时处理 4 个序列
for pos in range(max_len):
    token = get_next_tokens(batch)  # (4, 1, d_model)
    output = attention(token, cache, start_pos=pos)
```

## 内存分析

对于典型的 Transformer 层：
- **每层缓存大小**: 2 × batch_size × n_heads × seq_len × d_k
- **总缓存**: num_layers × 每层缓存

示例（GPT-2 small）：
- 12 层，12 个头，d_k = 64，seq_len = 1024，batch = 1
- 每层缓存: 2 × 1 × 12 × 1024 × 64 = 1.5 MB
- 总缓存: 12 × 1.5 MB = **18 MB**

对于更长的序列或更大的模型，这可能变得非常显著！

## 性能对比

有缓存 vs 无缓存的基准测试：

```python
# 无缓存: ~O(n²) 时间
start = time.time()
for i in range(1000):
    output = model(full_sequence[:i+1])
naive_time = time.time() - start

# 有缓存: ~O(n) 时间
start = time.time()
cache = KVCache()
for i in range(1000):
    output = model(current_token, cache, start_pos=i)
cached_time = time.time() - start

print(f"加速比: {naive_time / cached_time:.1f}x")
# 典型加速: 长序列 10-100 倍
```

## 代码详解

### 步骤 1：初始化缓存
```python
cache = KVCache(max_batch_size=1, max_seq_len=2048, n_heads=12, d_k=64)
```

### 步骤 2：预填充（处理提示）
```python
prompt = tokenize("Once upon a time")
output = model(prompt, cache, start_pos=0)
# 缓存现在包含位置 0..3 的 K, V
```

### 步骤 3：生成 Token
```python
for i in range(100):
    next_token = sample(output)
    output = model(next_token, cache, start_pos=len(prompt) + i)
    # 缓存每步增长 1 个位置
```

## 核心要点

1. **KV 缓存通过存储之前的 K, V 消除重复计算**
2. **用内存换速度**——O(n) 缓存 vs O(n²) 重复计算
3. **预分配缓冲区**以避免生成期间的分配开销
4. **在注意力中分别处理预填充和生成模式**
5. **对生产推理至关重要**——所有现代 LLM 服务系统都使用 KV 缓存

## 下一步

- **PagedAttention**：更高效的缓存管理（用于 vLLM）
- **多查询注意力 (MQA)**：在头之间共享 K, V 以减少缓存大小
- **分组查询注意力 (GQA)**：MHA 和 MQA 之间的折中方案
- **量化 KV 缓存**：以较低精度存储缓存（int8, int4）

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer 论文
- [Efficiently Scaling Transformer Inference](https://arxiv.org/abs/2211.05102) — KV 缓存权衡分析
- [vLLM](https://github.com/vllm-project/vllm) — PagedAttention 实现
