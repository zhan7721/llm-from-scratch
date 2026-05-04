# 文本转语音 (TTS) 生成

> **模块 05 -- 多模态，第 05 章**

文本转语音 (TTS) 将书面文本转换为语音。本模块实现了一个简化的 TTS 流水线，灵感来自 Tacotron 和 WaveNet 等架构。该模型使用 Transformer 编码器编码文本 token，预测梅尔帧持续时间，生成梅尔频谱图，并使用带有空洞卷积的 WaveNet 风格声码器将其转换为波形。

---

## 前置知识

- 对 Transformer 和自注意力的基本理解（模块 01，第 03-04 章）
- PyTorch 基础：`nn.Module`、`nn.Linear`、`nn.Embedding`、`nn.Conv1d`、`dataclasses`
- 音频概念基础（采样率、梅尔频谱图）

## 文件说明

| 文件 | 用途 |
|------|------|
| `speech_generation.py` | 核心 TTS 实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试用例 |

---

## 全局视图

TTS 模型通过多个阶段将文本转换为音频：

```
输入：text_ids (B, seq_len)
    |
    v
TextEncoder               -- token 嵌入 + 位置编码 + N 个编码器块
    |
    v
DurationPredictor          -- 线性投影 -> 每个 token 的帧数
    |
    v
Expand / Interpolate       -- 重复编码器输出以匹配梅尔帧数
    |
    v
Linear Projection          -- d_model -> n_mels（预测的梅尔频谱图）
    |
    v
Vocoder                    -- 空洞卷积 -> 波形样本
    |
    v
输出：waveform (B, num_samples)
```

### 核心思想：基于持续时间的扩展

TTS 的一个关键挑战是文本和音频的长度差异很大。一个音素可能对应多个音频帧。持续时间预测器学习每个文本 token 应该产生多少梅尔帧。训练期间，我们插值编码器输出以匹配目标梅尔长度。推理期间，我们使用预测的持续时间。

---

## TTSConfig：模型超参数

```python
@dataclass
class TTSConfig:
    vocab_size: int = 256           # 文本词汇表大小
    d_model: int = 64               # 模型维度
    num_heads: int = 4              # TextEncoder 中的注意力头数
    num_layers: int = 2             # Transformer 编码器块数
    dim_feedforward: int = 256      # FFN 隐藏维度
    max_text_positions: int = 512   # 最大文本序列长度
    n_mels: int = 80                # 梅尔频谱图频率 bin 数
    vocoder_channels: int = 64      # 声码器隐藏通道数
    vocoder_num_layers: int = 4     # 声码器中的空洞卷积层数
    hop_length: int = 256           # 每个梅尔帧的音频样本数
    dropout: float = 0.0            # dropout 概率
```

我们的实现使用较小的默认值（d_model=64）以加快测试速度。

---

## 架构详解

### TextEncoder

文本编码器将 token ID 转换为连续隐藏表示：

1. **Token 嵌入**：将 token ID 映射到 d_model 维向量。
2. **位置编码**：可学习的嵌入，加到 token 嵌入上。
3. **Transformer 块**：使用双向自注意力的 Pre-Norm 编码器块。
4. **最终 LayerNorm**：所有块之后的归一化。

```
text_ids (B, seq_len)
    -> Token 嵌入 + 位置编码
    -> N 个 EncoderBlock（自注意力 + FFN）
    -> LayerNorm
    -> 隐藏状态 (B, seq_len, d_model)
```

### DurationPredictor

一个简单的线性层，预测每个文本 token 应该产生多少梅尔帧：

```python
durations = F.relu(self.proj(encoder_output).squeeze(-1))
```

ReLU 确保持续时间非负。

### Vocoder

简化的 WaveNet 风格声码器，将梅尔频谱图转换为波形：

1. **上采样**：将每个梅尔帧重复 `hop_length` 次以达到样本级分辨率。
2. **输入投影**：将 n_mels 映射到 vocoder_channels。
3. **空洞卷积**：具有指数递增空洞率（1, 2, 4, 8, ...）的层堆栈。
4. **门控激活**：tanh(branch) * sigmoid(branch) 提供表达性非线性。
5. **跳跃连接**：在最终投影之前求和所有层的输出。
6. **输出投影**：映射到单个波形样本值。

```
mel (B, n_mels, num_frames)
    -> 上采样 (repeat_interleave)
    -> 输入投影 (n_mels -> channels)
    -> N 个 DilatedConvLayer
    -> 求和跳跃连接
    -> 输出投影 (channels -> 1)
    -> waveform (B, num_samples)
```

### 空洞卷积

空洞卷积允许声码器在不需要很多层的情况下拥有大的感受野：

| 层 | 空洞率 | 核大小 | 感受野 |
|----|--------|--------|--------|
| 0  | 1      | 3      | 3      |
| 1  | 2      | 3      | 7      |
| 2  | 4      | 3      | 15     |
| 3  | 8      | 3      | 31     |

每层使用因果填充（左侧额外填充）并修剪输出以保持正确长度。

### 门控激活

WaveNet 使用受 LSTM 启发的门控机制：

```python
gate = conv(x)
tanh_out, sigmoid_out = gate.chunk(2, dim=1)
gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
```

这允许网络在每一层选择性地传递或阻断信息。

---

## 前向传播详解

### 训练

```python
def forward(self, text_ids, target_mel=None):
    # 1. 编码文本
    encoder_out = self.encoder(text_ids)  # (B, seq_len, d_model)

    # 2. 预测持续时间
    durations = self.duration_predictor(encoder_out)  # (B, seq_len)

    # 3. 确定输出长度
    if target_mel is not None:
        num_frames = target_mel.shape[2]
    else:
        num_frames = int(durations.sum(dim=1).mean().item())

    # 4. 扩展编码器输出以匹配梅尔帧数
    expanded = F.interpolate(encoder_out.transpose(1, 2), size=num_frames)
    expanded = expanded.transpose(1, 2)

    # 5. 投影到梅尔频谱图
    predicted_mel = self.mel_proj(expanded).transpose(1, 2)
    return predicted_mel
```

### 形状追踪

对于 batch 为 2 的文本序列（seq_len=10），d_model=32，n_mels=40：

```
输入：              (2, 10)              -- 文本 token ID
TextEncoder：       (2, 10, 32)          -- 隐藏表示
DurationPredictor： (2, 10)              -- 每个 token 的帧数
Interpolate：       (2, 30, 32)          -- 扩展到 30 帧
Mel Projection：    (2, 40, 30)          -- 预测的梅尔频谱图
Vocoder：           (2, 3840)            -- 波形（30 * 128 样本）
```

### 推理（合成）

```python
@torch.no_grad()
def synthesize(self, text_ids):
    self.eval()
    predicted_mel = self.forward(text_ids)  # 无 target_mel
    waveform = self.vocoder(predicted_mel)
    return waveform
```

推理期间，预测的持续时间决定输出长度。

---

## Pre-Norm Transformer 块

我们的实现使用 **Pre-Norm**（在注意力/FFN 之前进行 LayerNorm）：

```
Pre-Norm:  x = x + Attention(LayerNorm(x))
Post-Norm: x = LayerNorm(x + Attention(x))
```

Pre-Norm 在训练期间更稳定，因为残差路径是干净的。

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/05_speech_generation/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 05-multimodal/05_speech_generation/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己练习实现 TTS 模型。

### 练习顺序

1. **EncoderBlock**：创建 LayerNorm、MultiheadAttention、FFN，并实现 Pre-Norm 前向传播
2. **TextEncoder**：创建嵌入、位置编码、层堆栈，并实现编码器前向传播
3. **DurationPredictor**：创建线性投影并实现前向传播
4. **DilatedConvLayer**：创建空洞卷积、残差/跳跃卷积和门控激活
5. **Vocoder**：创建输入/输出投影、层堆栈和声码器前向传播
6. **SimpleTTS**：将所有组件组合成端到端模型

### 提示

- `nn.MultiheadAttention` 使用 `batch_first=True` 时期望 `(B, N, D)` 输入。
- `F.interpolate` 使用 `mode="linear"` 可以对 3D 张量调整序列维度大小。
- `repeat_interleave` 是通过重复每个元素来上采样的简单方法。
- 空洞卷积使用 `padding=dilation` 来保持长度，然后修剪右侧以实现因果行为。
- 门控激活：`torch.tanh(x) * torch.sigmoid(y)` 允许选择性信息流。

---

## 关键要点

1. **TTS 是多阶段流水线。** 文本编码、持续时间预测、梅尔生成和声码是协同工作的独立组件。

2. **持续时间预测桥接文本和音频长度。** 一个简单的线性层可以学习每个文本 token 应该产生多少音频帧。

3. **空洞卷积实现大感受野。** 通过指数递增空洞率，每一层可以看到更远而不需要添加很多参数。

4. **门控激活很强大。** tanh * sigmoid 门控机制（来自 WaveNet/LSTM）允许网络控制信息流。

5. **跳跃连接聚合多尺度特征。** 求和所有层的跳跃连接将局部和全局信息组合用于最终输出。

---

## 延伸阅读

- [WaveNet (van den Oord 等，2016)](https://arxiv.org/abs/1609.03499) -- 原始音频的生成模型
- [Tacotron (Wang 等，2017)](https://arxiv.org/abs/1703.10135) -- 迈向端到端语音合成
- [Tacotron 2 (Shen 等，2018)](https://arxiv.org/abs/1712.05884) -- 通过梅尔频谱图条件化 WaveNet 实现自然 TTS 合成
- [Attention Is All You Need (Vaswani 等，2017)](https://arxiv.org/abs/1706.03762) -- 原始 Transformer 论文
- [FastSpeech 2 (Ren 等，2020)](https://arxiv.org/abs/2006.04558) -- 快速高质量端到端文本转语音
