# 视觉语言模型 (VLM)

> **模块 05 -- 多模态，第 02 章**

我们在第 01 章中构建了视觉 Transformer (ViT)。现在我们将它与语言模型结合，创建一个能够同时理解图像和文本的视觉语言模型。本章实现 LLaVA 和 Flamingo 等模型使用的核心 VLM 架构。

---

## 前置知识

- 第 01 章的视觉 Transformer (ViT)
- 模块 01 的 Transformer 架构
- PyTorch 基础：`nn.Module`、`nn.Linear`、`dataclasses`

## 文件说明

| 文件 | 用途 |
|------|------|
| `vlm.py` | 核心 VLM 实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

视觉语言模型结合了视觉和语言理解。关键洞察是将视觉特征投影到与文本 token 相同的嵌入空间，然后用语言模型一起处理它们：

```
图像 (B, C, H, W)
    |
    v
视觉编码器 (ViT)
    |
    v
视觉特征 (B, num_patches, vision_dim)
    |
    v
视觉投影器 (MLP)
    |
    v
视觉嵌入 (B, num_patches, lm_dim)
    |
    +--- 与文本嵌入拼接 ---+
    |                       |
    v                       v
[视觉 token | 文本 token] (B, num_vision + seq_len, lm_dim)
    |
    v
语言模型 (Transformer)
    |
    v
Logits (B, num_vision + seq_len, vocab_size)
```

---

## VLMConfig：模型超参数

```python
@dataclass
class VLMConfig:
    vision_dim: int = 768          # 视觉编码器输出维度
    lm_dim: int = 512              # 语言模型嵌入维度
    lm_vocab_size: int = 32000     # 语言模型词汇表大小
    lm_n_heads: int = 8            # 语言模型注意力头数
    lm_n_layers: int = 6           # 语言模型 Transformer 层数
    lm_max_seq_len: int = 1024     # 最大文本序列长度
    vision_image_size: int = 224   # 输入图像大小
    vision_patch_size: int = 16    # 视觉编码器 patch 大小
    vision_n_heads: int = 12       # 视觉编码器注意力头数
    vision_n_layers: int = 12      # 视觉编码器 Transformer 层数
```

---

## 架构组件

### 1. 视觉编码器 (ViT)

视觉编码器使用第 01 章的视觉 Transformer 从图像中提取特征。它将图像转换为 patch 嵌入序列：

```python
# 对于 224x224 的图像，使用 16x16 的 patch：
# num_patches = (224/16)^2 = 196 个 patch
# 加上 1 个 CLS token = 197 个视觉 token
# 输出：(B, 197, vision_dim)
```

### 2. 视觉投影器 (MLP)

投影器使用简单的 MLP 将视觉特征从 `vision_dim` 映射到 `lm_dim`：

```python
Linear(vision_dim, lm_dim) -> GELU -> Linear(lm_dim, lm_dim)
```

这是 LLaVA 使用的标准方法。GELU 激活函数提供非线性，两层 MLP 比单层线性层允许更复杂的变换。

### 3. 语言模型

语言模型是标准的 GPT 风格 Transformer，包含：
- Token 嵌入
- 位置嵌入（可学习）
- N 个带因果注意力的 Transformer 块
- 最终层归一化
- 语言模型头

### 4. 拼接策略

视觉 token **前置**于文本 token。这意味着序列如下：

```
[视觉_token_1, 视觉_token_2, ..., 视觉_token_N, 文本_token_1, 文本_token_2, ...]
```

这允许语言模型在生成文本时关注视觉 token，从而实现视觉理解。

---

## 前向传播详解

```python
def forward(self, images, input_ids):
    # 1. 提取视觉特征
    vision_features = self.vision_encoder(images)
    # (B, 3, 224, 224) -> (B, 197, vision_dim)

    # 2. 投影到语言模型嵌入空间
    vision_embeddings = self.projector(vision_features)
    # (B, 197, vision_dim) -> (B, 197, lm_dim)

    # 3. 获取文本嵌入
    text_embeddings = self.token_emb(input_ids)
    # (B, seq_len) -> (B, seq_len, lm_dim)

    # 4. 拼接（视觉在前）
    combined = torch.cat([vision_embeddings, text_embeddings], dim=1)
    # (B, 197 + seq_len, lm_dim)

    # 5. 添加位置嵌入
    positions = torch.arange(combined.shape[1], device=combined.device)
    combined = combined + self.pos_emb(positions)

    # 6. 通过语言模型处理
    h = combined
    for layer in self.layers:
        h = layer(h)

    # 7. 最终归一化并投影到词汇表
    h = self.norm(h)
    logits = self.lm_head(h)
    # (B, 197 + seq_len, vocab_size)

    return logits
```

---

## 关键设计决策

### 为什么使用 MLP 投影器？

使用 MLP 投影器（Linear + GELU + Linear）而不是单层线性层，因为：
1. 它提供非线性以实现更好的特征变换
2. 它是 LLaVA 和类似模型的标准方法
3. 它简单且有效

### 为什么前置视觉 token？

视觉 token 被前置（而非追加），因为：
1. 因果注意力掩码允许文本 token 关注所有视觉 token
2. 这是 LLaVA 和 Flamingo 的标准方法
3. 它使模型在生成文本之前能够"看到"图像

### 为什么使用可学习位置嵌入？

我们使用可学习位置嵌入而不是 RoPE，因为：
1. 序列包含语义不同的视觉和文本 token
2. 可学习嵌入可以捕获位置特定的模式
3. 对于教育目的来说，它更易于实现

---

## 参数计数

对于 `vision_dim=64`、`lm_dim=64`、2 层 Transformer 的小型 VLM：

**视觉编码器 (ViT)：**
- Patch 嵌入：3 * 16 * 16 * 64 = 49,152
- Transformer 块：~200K
- 总计：~250K

**视觉投影器：**
- MLP：64 * 64 + 64 + 64 * 64 + 64 = 8,256

**语言模型：**
- Token 嵌入：256 * 64 = 16,384
- 位置嵌入：133 * 64 = 8,512
- Transformer 块：~200K
- 总计：~225K

**总计：** ~500K 参数

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/02_vlm/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 05-multimodal/02_vlm/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现 VLM。

### 练习顺序

1. **VisionProjector**：创建 MLP 投影器
2. **VLM.__init__**：创建所有组件（视觉编码器、投影器、语言模型）
3. **VLM.forward**：连接管道（视觉 -> 投影 -> 拼接 -> 语言模型）

### 提示

- 从 VisionProjector 开始。它是带两个线性层的简单 MLP。
- 对于 VLM，记住正确计算视觉 token 数量（patch + CLS token）。
- 位置嵌入必须足够大以容纳视觉 + 文本 token。
- 不要忘记 token_emb 和 lm_head 之间的权重绑定。

---

## 关键要点

1. **VLM 结合视觉和语言。** 关键是将视觉特征投影到语言模型的嵌入空间。

2. **MLP 投影器简单有效。** 带 GELU 激活的两层 MLP 在视觉到语言投影中效果很好。

3. **视觉 token 被前置。** 这允许文本 token 通过因果注意力关注视觉 token。

4. **位置嵌入处理混合序列。** 可学习位置嵌入对包含视觉和文本 token 的序列效果很好。

5. **权重绑定减少参数。** token 嵌入和语言模型头之间的权重共享是一种常见的优化。

---

## 延伸阅读

- [LLaVA 论文 (Liu et al., 2023)](https://arxiv.org/abs/2304.08485) -- 视觉指令微调
- [Flamingo 论文 (Alayrac et al., 2022)](https://arxiv.org/abs/2204.14198) -- 少样本学习的视觉语言模型
- [CLIP 论文 (Radford et al., 2021)](https://arxiv.org/abs/2103.00020) -- 学习可迁移的视觉模型
- [ViT 论文 (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) -- 视觉 Transformer
