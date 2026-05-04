# Whisper-style Speech Recognition

> **Module 05 -- Multimodal, Chapter 04**

The Whisper model (Radford et al., 2022) is an encoder-decoder Transformer for speech recognition. It converts raw audio waveforms into text transcriptions by: (1) extracting spectrogram features from audio, (2) encoding those features with a Transformer encoder, and (3) autoregressively decoding text tokens with cross-attention to the encoder output.

---

## Prerequisites

- Basic understanding of Transformers, self-attention, and cross-attention (Module 01, Chapters 03-04)
- PyTorch basics: `nn.Module`, `nn.Linear`, `nn.Embedding`, `dataclasses`
- Familiarity with audio processing concepts (sampling rate, FFT)

## Files

| File | Purpose |
|------|---------|
| `speech_recognition.py` | Core Whisper-style implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A Whisper-style model converts speech to text through three stages:

```
Input: raw audio (B, num_samples)
    |
    v
AudioFeatureExtractor      -- windowed DFT -> log magnitude -> project
    |
    v
WhisperEncoder              -- positional encoding + N x encoder blocks
    |
    v
WhisperDecoder              -- causal self-attn + cross-attn to encoder + FFN
    |
    v
Output: logits (B, out_len, vocab_size)
```

### Key Insight: Audio as a Sequence

The core idea is the same as ViT: treat audio frames as tokens. Raw audio is split into overlapping frames, each frame is converted to a frequency representation via DFT, and these features become the "token embeddings" for a Transformer encoder. The decoder then generates text tokens one at a time, attending to the encoder's output.

---

## WhisperConfig: Model Hyperparameters

```python
@dataclass
class WhisperConfig:
    sample_rate: int = 16000       # audio sample rate in Hz
    n_fft: int = 128               # number of DFT frequency bins
    frame_length: int = 256        # audio samples per frame
    hop_length: int = 128          # hop between frames
    d_model: int = 64              # model dimension
    num_encoder_layers: int = 2    # encoder transformer blocks
    num_decoder_layers: int = 2    # decoder transformer blocks
    num_heads: int = 4             # attention heads
    dim_feedforward: int = 256     # FFN hidden dimension
    max_source_positions: int = 1024   # max audio frames
    max_target_positions: int = 512    # max decoder tokens
    vocab_size: int = 256          # output vocabulary size
    dropout: float = 0.0           # dropout probability
```

### Whisper Model Sizes

| Model | d_model | num_heads | num_layers | Parameters |
|-------|---------|-----------|------------|------------|
| Whisper-Tiny | 384 | 6 | 4 | 39M |
| Whisper-Base | 512 | 8 | 6 | 74M |
| Whisper-Small | 768 | 12 | 12 | 244M |
| Whisper-Medium | 1024 | 16 | 24 | 769M |
| Whisper-Large | 1280 | 20 | 32 | 1550M |

Our implementation uses small defaults (d_model=64) for fast testing.

---

## Architecture Details

### AudioFeatureExtractor

Instead of using librosa for mel spectrograms, we implement a simplified version:

1. **Framing**: Split audio into overlapping windows of `frame_length` samples with `hop_length` stride.
2. **Hann Window**: Multiply each frame by a Hann window to reduce spectral leakage.
3. **DFT**: Compute the real FFT to get the frequency spectrum.
4. **Log Scaling**: Apply `log(1 + magnitude)` for numerical stability.
5. **Linear Projection**: Map from `n_fft // 2 + 1` frequency bins to `d_model`.

```python
features = torch.fft.rfft(frames, n=self.n_fft)
magnitude = torch.abs(features)
features = torch.log1p(magnitude)
features = self.projection(features)  # (B, num_frames, d_model)
```

### WhisperEncoder

The encoder processes audio features with:

1. **Positional Encoding**: Learnable embeddings added to features (not concatenated).
2. **Transformer Blocks**: Pre-Norm blocks with bidirectional self-attention.
3. **Final LayerNorm**: Normalization after all blocks.

```
audio_features (B, num_frames, d_model)
    -> + positional_encoding[:num_frames]
    -> N x EncoderBlock (self-attention + FFN)
    -> LayerNorm
    -> encoded output (B, num_frames, d_model)
```

### WhisperDecoder

The decoder is autoregressive with three sub-layers per block:

1. **Causal Self-Attention**: Attends only to previous tokens (future is masked).
2. **Cross-Attention**: Attends to encoder output (Q from decoder, K/V from encoder).
3. **FFN**: Standard feed-forward network.

```
token_ids (B, out_len)
    -> Token embedding + positional encoding
    -> N x DecoderBlock
        -> CausalSelfAttention(LayerNorm(x))
        -> CrossAttention(LayerNorm(x), encoder_output)
        -> FFN(LayerNorm(x))
    -> LayerNorm
    -> Linear -> logits (B, out_len, vocab_size)
```

### Cross-Attention

The key architectural difference from encoder-only models:

```python
# Q comes from the decoder, K and V come from the encoder
cross_out, _ = self.cross_attn(
    query=decoder_hidden,    # (B, out_len, d_model)
    key=encoder_output,      # (B, num_frames, d_model)
    value=encoder_output,    # (B, num_frames, d_model)
)
```

This allows each decoder position to attend to all encoder positions, letting the model "listen" to the audio while generating text.

---

## Forward Pass Walkthrough

```python
def forward(self, audio, decoder_input_ids):
    # 1. Extract audio features
    audio_features = self.feature_extractor(audio)

    # 2. Encode audio
    encoder_output = self.encoder(audio_features)

    # 3. Decode to text
    logits = self.decoder(decoder_input_ids, encoder_output)
    return logits
```

### Shape Trace

For a batch of 2 audio clips (num_samples=1408) with d_model=32:

```
Input:              (2, 1408)          -- raw audio samples
FeatureExtractor:   (2, 11, 32)        -- 11 frames, each 32-dim
Encoder:            (2, 11, 32)        -- encoded audio features
Decoder (len=5):    (2, 5, 64)         -- 5 tokens, 64 vocab logits
```

---

## Greedy Decoding

The `transcribe` method generates tokens one at a time:

```python
generated = [0]  # start token
for _ in range(max_len):
    logits = model(audio, generated)
    next_token = logits[:, -1, :].argmax(dim=-1)
    generated.append(next_token)
```

This is the simplest decoding strategy. In production systems, beam search or sampling with temperature is used for better quality.

---

## Pre-Norm vs Post-Norm

Our implementation uses **Pre-Norm** (LayerNorm before attention/FFN):

```
Pre-Norm:  x = x + Attention(LayerNorm(x))
Post-Norm: x = LayerNorm(x + Attention(x))
```

Pre-Norm is more stable during training because the residual path is clean (no normalization applied to the residual connection). Most modern Transformers (GPT-2+, Whisper, LLaMA) use Pre-Norm.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/04_speech_recognition/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 05-multimodal/04_speech_recognition/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the Whisper-style speech recognition model yourself.

### Exercise Order

1. **AudioFeatureExtractor**: Create the linear projection, LayerNorm, and implement the DFT pipeline
2. **EncoderBlock**: Create LayerNorm, MultiheadAttention, FFN, and implement Pre-Norm forward
3. **DecoderBlock**: Create causal self-attention, cross-attention, and FFN sub-layers
4. **WhisperEncoder**: Create positional encoding, layer stack, and implement the encoder forward
5. **WhisperDecoder**: Create embeddings, layer stack, and implement the decoder forward
6. **WhisperModel**: Combine all components into the end-to-end model

### Tips

- `torch.fft.rfft` computes the real FFT, returning complex numbers. Use `torch.abs` to get magnitudes.
- `torch.triu(..., diagonal=1)` creates an upper-triangular boolean mask for causal attention.
- Cross-attention uses the same `nn.MultiheadAttention` as self-attention, but with different Q, K, V sources.
- The `batch_first=True` flag makes `nn.MultiheadAttention` expect `(B, N, D)` instead of `(N, B, D)`.

---

## Key Takeaways

1. **Whisper treats audio as a sequence.** By converting audio to spectrogram features, we can reuse the Transformer architecture for speech recognition.

2. **AudioFeatureExtractor replaces PatchEmbedding.** Instead of Conv2d for image patches, we use windowed DFT + linear projection for audio frames.

3. **Cross-attention bridges encoder and decoder.** The decoder's cross-attention to encoder output lets it "listen" to the audio while generating text.

4. **Causal masking enables autoregressive generation.** The decoder can only attend to previously generated tokens, ensuring left-to-right generation.

5. **Pre-Norm is more training-stable.** Applying LayerNorm before (not after) attention/FFN keeps the residual path clean.

---

## Further Reading

- [Whisper Paper (Radford et al., 2022)](https://arxiv.org/abs/2212.04356) -- Robust Speech Recognition via Large-Scale Weak Supervision
- [OpenAI Whisper Blog](https://openai.com/research/whisper) -- Announcement and overview
- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) -- Original Transformer paper
- [wav2vec 2.0 (Baevski et al., 2020)](https://arxiv.org/abs/2006.11477) -- Self-supervised speech representation learning
- [HuBERT (Hsu et al., 2021)](https://arxiv.org/abs/2106.07447) -- Self-supervised speech representation learning
