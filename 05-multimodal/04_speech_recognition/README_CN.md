# Whisper 风格语音识别

> **模块 05 -- 多模态，第 04 章**

Whisper 模型（Radford 等，2022）是一个用于语音识别的编码器-解码器 Transformer。它通过以下步骤将原始音频波形转换为文本转录：(1) 从音频中提取频谱图特征，(2) 用 Transformer 编码器编码这些特征，(3) 通过交叉注意力机制自回归地解码文本 token。

---

## 前置知识

- 对 Transformer、自注意力和交叉注意力的基本理解（模块 01，第 03-04 章）
- PyTorch 基础：`nn.Module`、`nn.Linear`、`nn.Embedding`、`dataclasses`
- 音频处理概念基础（采样率、FFT）

## 文件说明

| 文件 | 用途 |
|------|------|
| `speech_recognition.py` | 核心 Whisper 风格实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

Whisper 风格模型通过三个阶段将语音转换为文本：

```
输入：raw audio (B, num_samples)
    |
    v
AudioFeatureExtractor      -- 加窗 DFT -> 对数幅度 -> 投影
    |
    v
WhisperEncoder              -- 位置编码 + N 个编码器块
    |
    v
WhisperDecoder              -- 因果自注意力 + 对编码器的交叉注意力 + FFN
    |
    v
输出：logits (B, out_len, vocab_size)
```

### 核心思想：将音频视为序列

核心思想与 ViT 相同：将音频帧视为 token。原始音频被分割为重叠的帧，每帧通过 DFT 转换为频率表示，这些特征成为 Transformer 编码器的"token 嵌入"。解码器然后逐个生成文本 token，同时关注编码器的输出。

---

## WhisperConfig：模型超参数

```python
@dataclass
class WhisperConfig:
    sample_rate: int = 16000       # 音频采样率（Hz）
    n_fft: int = 128               # DFT 频率 bin 数
    frame_length: int = 256        # 每帧的音频样本数
    hop_length: int = 128          # 帧之间的跳跃长度
    d_model: int = 64              # 模型维度
    num_encoder_layers: int = 2    # 编码器 Transformer 块数
    num_decoder_layers: int = 2    # 解码器 Transformer 块数
    num_heads: int = 4             # 注意力头数
    dim_feedforward: int = 256     # FFN 隐藏维度
    max_source_positions: int = 1024   # 最大音频帧数
    max_target_positions: int = 512    # 最大解码器 token 数
    vocab_size: int = 256          # 输出词汇表大小
    dropout: float = 0.0           # dropout 概率
```

### Whisper 模型尺寸

| 模型 | d_model | num_heads | num_layers | 参数量 |
|------|---------|-----------|------------|--------|
| Whisper-Tiny | 384 | 6 | 4 | 39M |
| Whisper-Base | 512 | 8 | 6 | 74M |
| Whisper-Small | 768 | 12 | 12 | 244M |
| Whisper-Medium | 1024 | 16 | 24 | 769M |
| Whisper-Large | 1280 | 20 | 32 | 1550M |

我们的实现使用较小的默认值（d_model=64）以加快测试速度。

---

## 架构详解

### AudioFeatureExtractor

我们不使用 librosa 提取梅尔频谱图，而是实现了一个简化版本：

1. **分帧**：将音频分割为重叠的窗口，每窗口 `frame_length` 个样本，步长为 `hop_length`。
2. **汉宁窗**：将每帧乘以汉宁窗以减少频谱泄漏。
3. **DFT**：计算实数 FFT 获取频率频谱。
4. **对数缩放**：应用 `log(1 + magnitude)` 以提高数值稳定性。
5. **线性投影**：从 `n_fft // 2 + 1` 个频率 bin 映射到 `d_model`。

```python
features = torch.fft.rfft(frames, n=self.n_fft)
magnitude = torch.abs(features)
features = torch.log1p(magnitude)
features = self.projection(features)  # (B, num_frames, d_model)
```

### WhisperEncoder

编码器通过以下方式处理音频特征：

1. **位置编码**：可学习的嵌入加到特征上（不是拼接）。
2. **Transformer 块**：使用双向自注意力的 Pre-Norm 块。
3. **最终 LayerNorm**：所有块之后的归一化。

```
audio_features (B, num_frames, d_model)
    -> + positional_encoding[:num_frames]
    -> N x EncoderBlock（自注意力 + FFN）
    -> LayerNorm
    -> 编码输出 (B, num_frames, d_model)
```

### WhisperDecoder

解码器是自回归的，每个块有三个子层：

1. **因果自注意力**：仅关注之前的 token（未来被掩码）。
2. **交叉注意力**：关注编码器输出（Q 来自解码器，K/V 来自编码器）。
3. **FFN**：标准前馈网络。

```
token_ids (B, out_len)
    -> Token 嵌入 + 位置编码
    -> N x DecoderBlock
        -> CausalSelfAttention(LayerNorm(x))
        -> CrossAttention(LayerNorm(x), encoder_output)
        -> FFN(LayerNorm(x))
    -> LayerNorm
    -> Linear -> logits (B, out_len, vocab_size)
```

### 交叉注意力

与仅编码器模型的关键架构差异：

```python
# Q 来自解码器，K 和 V 来自编码器
cross_out, _ = self.cross_attn(
    query=decoder_hidden,    # (B, out_len, d_model)
    key=encoder_output,      # (B, num_frames, d_model)
    value=encoder_output,    # (B, num_frames, d_model)
)
```

这允许解码器的每个位置关注编码器的所有位置，让模型在生成文本时"聆听"音频。

---

## 前向传播详解

```python
def forward(self, audio, decoder_input_ids):
    # 1. 提取音频特征
    audio_features = self.feature_extractor(audio)

    # 2. 编码音频
    encoder_output = self.encoder(audio_features)

    # 3. 解码为文本
    logits = self.decoder(decoder_input_ids, encoder_output)
    return logits
```

### 形状追踪

对于 batch 为 2 的音频片段（num_samples=1408），d_model=32：

```
输入：              (2, 1408)          -- 原始音频样本
FeatureExtractor：  (2, 11, 32)        -- 11 帧，每帧 32 维
Encoder：           (2, 11, 32)        -- 编码的音频特征
Decoder (len=5)：   (2, 5, 64)         -- 5 个 token，64 个词汇 logits
```

---

## 贪心解码

`transcribe` 方法逐个生成 token：

```python
generated = [0]  # 起始 token
for _ in range(max_len):
    logits = model(audio, generated)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated.append(next_token)
```

这是最简单的解码策略。在生产系统中，使用束搜索或带温度的采样以获得更好的质量。

---

## Pre-Norm vs Post-Norm

我们的实现使用 **Pre-Norm**（在注意力/FFN 之前进行 LayerNorm）：

```
Pre-Norm:  x = x + Attention(LayerNorm(x))
Post-Norm: x = LayerNorm(x + Attention(x))
```

Pre-Norm 在训练期间更稳定，因为残差路径是干净的（残差连接上没有应用归一化）。大多数现代 Transformer（GPT-2+、Whisper、LLaMA）使用 Pre-Norm。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/04_speech_recognition/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 05-multimodal/04_speech_recognition/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现 Whisper 风格语音识别模型。

### 练习顺序

1. **AudioFeatureExtractor**：创建线性投影、LayerNorm，并实现 DFT 流水线
2. **EncoderBlock**：创建 LayerNorm、MultiheadAttention、FFN，并实现 Pre-Norm 前向传播
3. **DecoderBlock**：创建因果自注意力、交叉注意力和 FFN 子层
4. **WhisperEncoder**：创建位置编码、层堆栈，并实现编码器前向传播
5. **WhisperDecoder**：创建嵌入、层堆栈，并实现解码器前向传播
6. **WhisperModel**：将所有组件组合成端到端模型

### 提示

- `torch.fft.rfft` 计算实数 FFT，返回复数。使用 `torch.abs` 获取幅度。
- `torch.triu(..., diagonal=1)` 创建用于因果注意力的上三角布尔掩码。
- 交叉注意力与自注意力使用相同的 `nn.MultiheadAttention`，但 Q、K、V 来源不同。
- `batch_first=True` 标志使 `nn.MultiheadAttention` 期望 `(B, N, D)` 而不是 `(N, B, D)`。

---

## 关键要点

1. **Whisper 将音频视为序列。** 通过将音频转换为频谱图特征，我们可以复用 Transformer 架构进行语音识别。

2. **AudioFeatureExtractor 替代了 PatchEmbedding。** 不再使用 Conv2d 处理图像 patch，而是使用加窗 DFT + 线性投影处理音频帧。

3. **交叉注意力连接编码器和解码器。** 解码器对编码器输出的交叉注意力让它在生成文本时能够"聆听"音频。

4. **因果掩码实现自回归生成。** 解码器只能关注之前生成的 token，确保从左到右生成。

5. **Pre-Norm 训练更稳定。** 在注意力/FFN 之前（而不是之后）应用 LayerNorm 保持残差路径干净。

---

## 延伸阅读

- [Whisper 论文 (Radford 等，2022)](https://arxiv.org/abs/2212.04356) -- 通过大规模弱监督实现鲁棒语音识别
- [OpenAI Whisper 博客](https://openai.com/research/whisper) -- 公告和概述
- [Attention Is All You Need (Vaswani 等，2017)](https://arxiv.org/abs/1706.03762) -- 原始 Transformer 论文
- [wav2vec 2.0 (Baevski 等，2020)](https://arxiv.org/abs/2006.11477) -- 自监督语音表示学习
- [HuBERT (Hsu 等，2021)](https://arxiv.org/abs/2106.07447) -- 自监督语音表示学习
