# Text-to-Speech (TTS) Generation

> **Module 05 -- Multimodal, Chapter 05**

Text-to-Speech (TTS) converts written text into spoken audio. This module implements a simplified TTS pipeline inspired by architectures like Tacotron and WaveNet. The model encodes text tokens with a Transformer encoder, predicts mel frame durations, generates a mel spectrogram, and converts it to a waveform using a WaveNet-style vocoder with dilated convolutions.

---

## Prerequisites

- Basic understanding of Transformers and self-attention (Module 01, Chapters 03-04)
- PyTorch basics: `nn.Module`, `nn.Linear`, `nn.Embedding`, `nn.Conv1d`, `dataclasses`
- Familiarity with audio concepts (sampling rate, mel spectrograms)

## Files

| File | Purpose |
|------|---------|
| `speech_generation.py` | Core TTS implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A TTS model converts text to audio through several stages:

```
Input: text_ids (B, seq_len)
    |
    v
TextEncoder               -- token embedding + positional encoding + N x encoder blocks
    |
    v
DurationPredictor          -- linear projection -> frame count per token
    |
    v
Expand / Interpolate       -- repeat encoder output to match mel frame count
    |
    v
Linear Projection          -- d_model -> n_mels (predicted mel spectrogram)
    |
    v
Vocoder                    -- dilated convolutions -> waveform samples
    |
    v
Output: waveform (B, num_samples)
```

### Key Insight: Duration-Based Expansion

A key challenge in TTS is that text and audio have very different lengths. A single phoneme might correspond to many audio frames. The duration predictor learns how many mel frames each text token should produce. During training, we interpolate the encoder output to match the target mel length. During inference, we use the predicted durations.

---

## TTSConfig: Model Hyperparameters

```python
@dataclass
class TTSConfig:
    vocab_size: int = 256           # text vocabulary size
    d_model: int = 64               # model dimension
    num_heads: int = 4              # attention heads in TextEncoder
    num_layers: int = 2             # transformer encoder blocks
    dim_feedforward: int = 256      # FFN hidden dimension
    max_text_positions: int = 512   # max text sequence length
    n_mels: int = 80                # mel spectrogram frequency bins
    vocoder_channels: int = 64      # vocoder hidden channels
    vocoder_num_layers: int = 4     # dilated conv layers in vocoder
    hop_length: int = 256           # audio samples per mel frame
    dropout: float = 0.0            # dropout probability
```

Our implementation uses small defaults (d_model=64) for fast testing.

---

## Architecture Details

### TextEncoder

The text encoder converts token IDs into continuous hidden representations:

1. **Token Embedding**: Maps token IDs to d_model-dimensional vectors.
2. **Positional Encoding**: Learnable embeddings added to token embeddings.
3. **Transformer Blocks**: Pre-Norm encoder blocks with bidirectional self-attention.
4. **Final LayerNorm**: Normalization after all blocks.

```
text_ids (B, seq_len)
    -> Token embedding + positional encoding
    -> N x EncoderBlock (self-attention + FFN)
    -> LayerNorm
    -> hidden states (B, seq_len, d_model)
```

### DurationPredictor

A simple linear layer that predicts how many mel frames each text token should produce:

```python
durations = F.relu(self.proj(encoder_output).squeeze(-1))
```

The ReLU ensures non-negative durations.

### Vocoder

A simplified WaveNet-style vocoder that converts mel spectrograms to waveforms:

1. **Upsampling**: Repeat each mel frame `hop_length` times to reach sample-level resolution.
2. **Input Projection**: Map n_mels to vocoder_channels.
3. **Dilated Convolutions**: Stack of layers with exponentially increasing dilation (1, 2, 4, 8, ...).
4. **Gated Activation**: tanh(branch) * sigmoid(branch) for expressive nonlinearity.
5. **Skip Connections**: Sum outputs from all layers before final projection.
6. **Output Projection**: Map to a single waveform sample value.

```
mel (B, n_mels, num_frames)
    -> Upsample (repeat_interleave)
    -> Input projection (n_mels -> channels)
    -> N x DilatedConvLayer
    -> Sum skip connections
    -> Output projection (channels -> 1)
    -> waveform (B, num_samples)
```

### Dilated Convolutions

Dilated convolutions allow the vocoder to have a large receptive field without many layers:

| Layer | Dilation | Kernel | Receptive Field |
|-------|----------|--------|-----------------|
| 0     | 1        | 3      | 3               |
| 1     | 2        | 3      | 7               |
| 2     | 4        | 3      | 15              |
| 3     | 8        | 3      | 31              |

Each layer uses causal padding (extra padding on the left) and trims the output to maintain the correct length.

### Gated Activation

WaveNet uses a gating mechanism inspired by LSTM:

```python
gate = conv(x)
tanh_out, sigmoid_out = gate.chunk(2, dim=1)
gated = torch.tanh(tanh_out) * torch.sigmoid(sigmoid_out)
```

This allows the network to selectively pass or block information at each layer.

---

## Forward Pass Walkthrough

### Training

```python
def forward(self, text_ids, target_mel=None):
    # 1. Encode text
    encoder_out = self.encoder(text_ids)  # (B, seq_len, d_model)

    # 2. Predict durations
    durations = self.duration_predictor(encoder_out)  # (B, seq_len)

    # 3. Determine output length
    if target_mel is not None:
        num_frames = target_mel.shape[2]
    else:
        num_frames = int(durations.sum(dim=1).mean().item())

    # 4. Expand encoder output to match mel frames
    expanded = F.interpolate(encoder_out.transpose(1, 2), size=num_frames)
    expanded = expanded.transpose(1, 2)

    # 5. Project to mel spectrogram
    predicted_mel = self.mel_proj(expanded).transpose(1, 2)
    return predicted_mel
```

### Shape Trace

For a batch of 2 text sequences (seq_len=10) with d_model=32, n_mels=40:

```
Input:              (2, 10)              -- text token IDs
TextEncoder:        (2, 10, 32)          -- hidden representations
DurationPredictor:  (2, 10)              -- frame counts per token
Interpolate:        (2, 30, 32)          -- expanded to 30 frames
Mel Projection:     (2, 40, 30)          -- predicted mel spectrogram
Vocoder:            (2, 3840)            -- waveform (30 * 128 samples)
```

### Inference (Synthesize)

```python
@torch.no_grad()
def synthesize(self, text_ids):
    self.eval()
    predicted_mel = self.forward(text_ids)  # no target_mel
    waveform = self.vocoder(predicted_mel)
    return waveform
```

During inference, the predicted durations determine the output length.

---

## Pre-Norm Transformer Blocks

Our implementation uses **Pre-Norm** (LayerNorm before attention/FFN):

```
Pre-Norm:  x = x + Attention(LayerNorm(x))
Post-Norm: x = LayerNorm(x + Attention(x))
```

Pre-Norm is more stable during training because the residual path is clean.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/05_speech_generation/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 05-multimodal/05_speech_generation/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the TTS model yourself.

### Exercise Order

1. **EncoderBlock**: Create LayerNorm, MultiheadAttention, FFN, and implement Pre-Norm forward
2. **TextEncoder**: Create embeddings, positional encoding, layer stack, and encoder forward
3. **DurationPredictor**: Create the linear projection and implement the forward pass
4. **DilatedConvLayer**: Create the dilated conv, residual/skip convolutions, and gated activation
5. **Vocoder**: Create input/output projections, layer stack, and vocoder forward
6. **SimpleTTS**: Combine all components into the end-to-end model

### Tips

- `nn.MultiheadAttention` with `batch_first=True` expects `(B, N, D)` input.
- `F.interpolate` with `mode="linear"` works for 3D tensors to resize the sequence dimension.
- `repeat_interleave` is a simple way to upsample by repeating each element.
- Dilated convolutions use `padding=dilation` to maintain length, then trim the right side for causal behavior.
- Gated activation: `torch.tanh(x) * torch.sigmoid(y)` allows selective information flow.

---

## Key Takeaways

1. **TTS is a multi-stage pipeline.** Text encoding, duration prediction, mel generation, and vocoding are separate components that work together.

2. **Duration prediction bridges text and audio lengths.** A simple linear layer can learn how many audio frames each text token should produce.

3. **Dilated convolutions enable large receptive fields.** By exponentially increasing the dilation factor, each layer can see further without adding many parameters.

4. **Gated activations are powerful.** The tanh * sigmoid gating mechanism (from WaveNet/LSTM) allows the network to control information flow.

5. **Skip connections aggregate multi-scale features.** Summing skip connections from all layers combines local and global information for the final output.

---

## Further Reading

- [WaveNet (van den Oord et al., 2016)](https://arxiv.org/abs/1609.03499) -- A Generative Model for Raw Audio
- [Tacotron (Wang et al., 2017)](https://arxiv.org/abs/1703.10135) -- Towards End-to-End Speech Synthesis
- [Tacotron 2 (Shen et al., 2018)](https://arxiv.org/abs/1712.05884) -- Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) -- Original Transformer paper
- [FastSpeech 2 (Ren et al., 2020)](https://arxiv.org/abs/2006.04558) -- Fast and High-Quality End-to-End Text to Speech
