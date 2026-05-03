# LLM From Scratch

Build a Large Language Model from scratch in pure PyTorch.

## Overview

This project teaches you how to build an LLM by implementing every component from scratch — no high-level frameworks, just PyTorch. Each chapter includes:
- A clear explanation (English + Chinese)
- Working implementation code
- Hands-on exercises with TODOs to fill in
- Reference solutions
- Tests to verify your work

## Learning Path

### 01-foundations — Core Components

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01_tokenizer | BPE Tokenization | Byte-Pair Encoding, subword tokenization |
| 02_embedding | Embeddings | Token embedding, RoPE positional encoding |
| 03_attention | Attention | Multi-Head Attention, Grouped Query Attention |
| 04_transformer_block | Transformer Block | Pre-Norm, SwiGLU FFN, RMSNorm, Residual |
| 05_model_architecture | GPT Model | Full model assembly, weight tying, generation |
| 06_moe | Mixture of Experts | Top-K routing, expert networks, load balancing |
| 07_kv_cache | KV Cache | Efficient autoregressive inference |
| 08_long_context | Long Context | RoPE scaling, YaRN, position interpolation |

### 02-pretrain — Pre-training (Coming Soon)

Data pipeline, data engineering, training loop, distributed training, scaling laws, evaluation.

### 03-sft — Supervised Fine-Tuning (Coming Soon)

Instruction tuning, LoRA, QLoRA, NEFTune, chat templates, long context SFT.

### 04-rl — Reinforcement Learning (Coming Soon)

Reward model, PPO, DPO, Online DPO, GRPO, Process Reward Model, reward hacking defense.

### 05-multimodal — Multimodal (Coming Soon)

Vision encoder (ViT), Vision-Language Model, Diffusion, speech recognition, speech generation.

### 06-agent — Agents (Coming Soon)

Tool calling, reasoning (CoT), MCP, multi-agent systems, code interpreter.

### 07-inference — Inference Optimization (Coming Soon)

Quantization (GPTQ/AWQ), speculative decoding, continuous batching, serving.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
pytest 01-foundations/ -v

# Run a specific chapter
pytest 01-foundations/03_attention/tests.py -v

# Run the tokenizer demo
python 01-foundations/01_tokenizer/train_tokenizer.py
```

## Each Chapter Contains

```
chapter_name/
├── README.md / README_CN.md   # Tutorial (English + Chinese)
├── implementation.py           # Core implementation
├── exercise.py                 # Fill in the TODOs
├── solution.py                 # Reference solution
└── tests.py                    # Verify your work
```

## Hardware Requirements

| Requirement | Minimum | Recommended |
|------------|---------|-------------|
| GPU | Mac M1/M2 (MPS) or RTX 3060 (12GB) | RTX 3090 (24GB) |
| RAM | 8GB | 16GB |
| Python | 3.10+ | 3.11+ |

## Dependencies

- PyTorch 2.0+
- pytest (testing)
- tiktoken (BPE comparison)
- transformers (weight loading for comparison)
- datasets (sample data download)
- wandb (optional, training monitoring)

## Project Structure

```
llm-from-scratch/
├── 01-foundations/     # Core LLM components
├── 02-pretrain/        # Pre-training
├── 03-sft/             # Supervised fine-tuning
├── 04-rl/              # Reinforcement learning
├── 05-multimodal/      # Vision + Speech
├── 06-agent/           # Agents + Tools
├── 07-inference/       # Inference optimization
├── configs/            # Training configs
├── data/               # Sample datasets
├── scripts/            # Utility scripts
└── tests/              # Integration tests
```

## References

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer
- [LLaMA: Open and Efficient Foundation LLMs](https://arxiv.org/abs/2302.13971) — LLaMA architecture
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) — YaRN
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — MoE
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO

## License

MIT
