# Vision Transformer (ViT)

> **模块 05 -- 多模态，第 01 章**

Vision Transformer（Dosovitskiy 等，2020）将标准 Transformer 编码器应用于图像 patch 序列。ViT 不使用卷积处理原始像素，而是将图像分割成固定大小的 patch，将每个 patch 嵌入为向量，然后用 Transformer 处理该序列——与语言模型使用的架构完全相同。

---

## 前置知识

- 对 Transformer 和自注意力的基本理解（模块 01，第 03-04 章）
- PyTorch 基础：`nn.Module`、`nn.Linear`、`nn.Conv2d`、`dataclasses`

## 文件说明

| 文件 | 用途 |
|------|------|
| `vision_encoder.py` | 核心 ViT 实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

Vision Transformer 将图像转换为 patch 嵌入序列，然后像语言模型处理 token 嵌入一样处理它们：

```
输入：image (B, C, H, W)
    |
    v
PatchEmbedding          -- 分割为 patch，投影到 D 维
    |
    v
前置 CLS token          -- 用于分类的可学习 [CLS] token
    |
    v
添加位置嵌入            -- 每个位置的可学习向量
    |
    v
TransformerBlock x N    -- 自注意力 + FFN（双向，无因果掩码）
    |
    v
LayerNorm               -- 最终归一化
    |
    v
CLS token -> Linear     -- 分类头
    |
    v
输出：logits (B, num_classes)
```

### 核心思想：将图像视为序列

核心思想很简单：将图像 patch 视为 NLP 中的 token。224x224 的图像分割成 16x16 的 patch 得到 (224/16)^2 = 196 个 patch。每个 patch 是一个 16x16x3 = 768 维的小向量，被投影到嵌入维度，就像词嵌入一样。

---

## ViTConfig：模型超参数

```python
@dataclass
class ViTConfig:
    image_size: int = 224       # 输入图像大小（正方形）
    patch_size: int = 16        # patch 大小（正方形）
    num_channels: int = 3       # RGB 通道
    embedding_dim: int = 768    # 嵌入 / 隐藏维度
    num_heads: int = 12         # 注意力头数
    num_layers: int = 12        # Transformer 块数
    mlp_ratio: float = 4.0      # FFN 隐藏维度 = mlp_ratio * embedding_dim
    dropout: float = 0.0        # dropout 概率
```

### ViT 模型尺寸

| 模型 | embedding_dim | num_heads | num_layers | mlp_ratio | 参数量 |
|------|---------------|-----------|------------|-----------|--------|
| ViT-Ti (Tiny) | 192 | 3 | 12 | 4.0 | 5.7M |
| ViT-S (Small) | 384 | 6 | 12 | 4.0 | 22M |
| ViT-B (Base) | 768 | 12 | 12 | 4.0 | 86M |
| ViT-L (Large) | 1024 | 16 | 24 | 4.0 | 307M |
| ViT-H (Huge) | 1280 | 16 | 32 | 4.0 | 632M |

---

## 架构详解

### PatchEmbedding

```python
self.projection = nn.Conv2d(
    in_channels=num_channels,
    out_channels=embedding_dim,
    kernel_size=patch_size,
    stride=patch_size,
)
```

`kernel_size = stride = patch_size` 的 Conv2d 等价于：
1. 将 (B, C, H, W) 重塑为不重叠的 patch
2. 将每个 patch 展平为向量
3. 线性投影到 `embedding_dim`

这比手动执行这些步骤更高效。

**Patch 数量**：`N = (H / P) * (W / P)`

对于 224x224 图像，16x16 patch：`N = 14 * 14 = 196`

### CLS Token

```python
self.cls_token = nn.Parameter(torch.zeros(1, 1, embedding_dim))
```

一个可学习的向量，前置到 patch 序列中。经过 Transformer 处理后，CLS token 的输出作为聚合的图像表示。这与 BERT 中用于句子分类的方法相同。

### 位置嵌入

```python
self.position_embedding = nn.Parameter(
    torch.zeros(1, num_patches + 1, embedding_dim)
)
```

每个 patch 加上 CLS token 的可学习位置嵌入。与语言模型中使用的 RoPE 不同，ViT 使用绝对学习位置嵌入。位置信息在 Transformer 块之前添加到 patch 嵌入中。

### Transformer 块

每个块应用：
1. **Pre-Norm 自注意力**：`x = x + Attention(LayerNorm(x))`
2. **Pre-Norm FFN**：`x = x + FFN(LayerNorm(x))`

注意力是**双向的**（无因果掩码）——每个 patch 可以关注其他所有 patch。这与语言模型不同，因果掩码阻止了对未来 token 的访问。

### FFN（前馈网络）

```python
FFN(x) = Linear(GELU(Linear(x)))
```

标准 ViT 使用 GELU 激活函数（不是 LLaMA 中的 SwiGLU）。隐藏维度为 `mlp_ratio * embedding_dim`（默认 4 倍）。

---

## 前向传播详解

```python
def forward(self, x):
    B = x.shape[0]

    # 1. Patch 嵌入：(B, C, H, W) -> (B, N, D)
    x = self.patch_embedding(x)

    # 2. 前置 CLS token：(B, N, D) -> (B, N+1, D)
    cls_tokens = self.cls_token.expand(B, -1, -1)
    x = torch.cat([cls_tokens, x], dim=1)

    # 3. 添加位置嵌入：(B, N+1, D) + (1, N+1, D)
    x = x + self.position_embedding

    # 4. Transformer 块
    for block in self.blocks:
        x = block(x)

    # 5. 最终归一化
    x = self.norm(x)

    # 6. 取 CLS token 并分类
    cls_output = x[:, 0]
    return self.head(cls_output)
```

### 形状追踪

对于 batch 为 2 的图像（32x32 RGB），patch_size=4，embedding_dim=64：

```
输入：          (2, 3, 32, 32)   -- RGB 图像
PatchEmbed：   (2, 64, 64)      -- 64 个 patch，每个 64 维
CLS 前置：     (2, 65, 64)      -- CLS + 64 个 patch
+ 位置嵌入：   (2, 65, 64)      -- 添加位置信息
Block 1：      (2, 65, 64)      -- 注意力 + FFN
Block 2：      (2, 65, 64)      -- 注意力 + FFN
LayerNorm：    (2, 65, 64)      -- 归一化
CLS 输出：     (2, 64)          -- 取第一个位置
分类：         (2, 10)          -- 投影到 10 个类别
```

---

## ViT vs CNN

| 方面 | CNN | ViT |
|------|-----|-----|
| 归纳偏置 | 强（局部性、平移等变性） | 弱（仅位置嵌入） |
| 数据效率 | 小数据集表现好 | 需要大数据集或预训练 |
| 扩展性 | 中等 | 优秀（更多数据 + 参数 = 更好） |
| 全局上下文 | 需要深层堆叠 | 每层都有全局注意力 |
| 可解释性 | 特征图 | 注意力图 |

ViT 的弱点是缺乏空间归纳偏置。没有卷积的局部性先验，ViT 需要更多数据来学习相邻像素是相关的。这就是为什么 ViT 通常在大数据集（ImageNet-21k、JFT-300M）上预训练，然后微调。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/01_vision_encoder/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 05-multimodal/01_vision_encoder/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现 Vision Transformer。

### 练习顺序

1. **PatchEmbedding**：创建 Conv2d 投影并实现 forward（flatten + transpose）
2. **TransformerBlock**：创建 LayerNorm、MultiheadAttention、FFN 并实现 forward
3. **VisionTransformer**：创建 CLS token、位置嵌入、blocks 并实现 forward

### 提示

- PatchEmbedding 的 Conv2d 技巧是关键洞察。`kernel_size=stride=patch_size` 意味着每个卷积窗口恰好覆盖一个 patch，没有重叠。
- `nn.MultiheadAttention` 使用 `batch_first=True` 时，输入期望为 `(B, N, D)`。
- CLS token 使用 `expand(B, -1, -1)` 在 batch 维度上重复而不复制内存。
- 位置嵌入直接加到 patch 嵌入上（广播处理 batch 维度）。

---

## 关键要点

1. **ViT 将图像视为序列。** 通过将图像分割为 patch 并嵌入它们，我们可以复用与语言模型完全相同的 Transformer 架构。

2. **通过 Conv2d 实现 PatchEmbedding 很优雅。** 单个 Conv2d 层配合正确的 kernel/stride 同时完成图像分割和 patch 投影。

3. **CLS token 聚合信息。** 可学习的 CLS token 通过自注意力收集全局图像特征，作为分类的图像表示。

4. **位置嵌入编码空间结构。** 由于 Transformer 没有固有的空间排列概念，可学习的位置嵌入至关重要。

5. **ViT 比 CNN 需要更多数据。** 没有卷积的归纳偏置（局部性、平移等变性），ViT 需要更大的数据集来从头学习这些模式。

---

## 延伸阅读

- [ViT 论文 (Dosovitskiy et al., 2020)](https://arxiv.org/abs/2010.11929) -- 原始 Vision Transformer
- [An Image is Worth 16x16 Words (博客)](https://ai.googleblog.com/2020/12/transformers-for-image-recognition-at.html) -- Google AI 博客文章
- [DeiT (Touvron et al., 2021)](https://arxiv.org/abs/2012.12877) -- 高效数据图像 Transformer
- [Swin Transformer (Liu et al., 2021)](https://arxiv.org/abs/2103.14030) -- 层级 Vision Transformer
- [MAE (He et al., 2022)](https://arxiv.org/abs/2111.06377) -- 用于自监督 ViT 预训练的掩码自编码器
