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

### 02-pretrain — Pre-training

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01_data_pipeline | Data Pipeline | Dynamic padding, sequence packing, DataLoader |
| 02_data_engineering | Data Engineering | MinHash deduplication, quality filtering, data mixing |
| 03_training_loop | Training Loop | AdamW, cosine LR schedule, gradient clipping, gradient accumulation |
| 04_distributed | Distributed Training | DDP, FSDP, multi-GPU training |
| 05_scaling_laws | Scaling Laws | Chinchilla optimal, compute-optimal training |
| 06_evaluation | Evaluation | Perplexity, benchmarks, emergent abilities |

### 03-sft — Supervised Fine-Tuning

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01_instruction_tuning | Instruction Tuning | Alpaca format, label masking, response-only loss |
| 02_lora | LoRA | Low-rank adaptation, parameter-efficient fine-tuning |
| 03_qlora | QLoRA | NF4 quantization + LoRA, 4-bit training |
| 04_neftune | NEFTune | Noise embedding for better generalization |
| 06_long_context_sft | Long Context SFT | Position interpolation, NTK-aware scaling |

### 04-rl — Reinforcement Learning (238 tests)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01_reward_model | Reward Model | Bradley-Terry model, scalar head, preference ranking |
| 02_ppo | PPO | Clipped surrogate, GAE, KL penalty, rollout |
| 03_dpo | DPO | Direct preference optimization, reference-free |
| 04_online_dpo | Online DPO | Dynamic preference pairs, on-policy generation |
| 05_grpo | GRPO | Group relative advantages, no value network |
| 06_prm | Process Reward Model | Step-level scoring, best-of-N selection |
| 07_reward_hacking | Reward Hacking | Detection, KL constraints, reward ensembles |

### 05-multimodal — Multimodal (137 tests)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01_vision_encoder | Vision Transformer | Patch embedding, CLS token, positional encoding |
| 02_vlm | Vision-Language Model | Vision projector, LLaVA-style, cross-modal fusion |
| 03_diffusion | Diffusion (DDPM) | Noise scheduler, U-Net, sinusoidal time embedding |
| 04_speech_recognition | Speech Recognition | Whisper-style encoder-decoder, cross-attention |
| 05_speech_generation | Speech Generation | TTS, WaveNet-style vocoder, duration prediction |

### 06-agent — Agents (216 tests)

| Chapter | Topic | Key Concepts |
|---------|-------|--------------|
| 01_tool_calling | Tool Calling | Tool registry, parser, executor, agent loop |
| 02_reasoning | Reasoning | Chain-of-Thought, self-consistency, step extraction |
| 03_mcp | Model Context Protocol | MCP messages, client-server, standardized protocol |
| 04_multi_agent | Multi-Agent | Orchestrator, shared memory, debate mode |
| 05_code_interpreter | Code Interpreter | Sandbox execution, code parsing, artifact storage |

### 07-inference — Inference Optimization (Coming Soon)

Quantization (GPTQ/AWQ), speculative decoding, continuous batching, serving.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (591+ tests across 6 modules)
pytest 04-rl/ 05-multimodal/ 06-agent/ -v

# Run a specific module
pytest 04-rl/ -v

# Run a specific chapter
pytest 04-rl/03_dpo/tests.py -v
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

### Architecture
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — Original Transformer
- [LLaMA: Open and Efficient Foundation LLMs](https://arxiv.org/abs/2302.13971) — LLaMA architecture
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) — YaRN
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — MoE

### Training & Alignment
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — Scaling Laws
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — QLoRA
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO
- [DeepSeekMath: Pushing the Limits of Math Reasoning](https://arxiv.org/abs/2402.03300) — GRPO
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — Process Reward Model

### Multimodal
- [An Image is Worth 16x16 Words: ViT](https://arxiv.org/abs/2010.11929) — Vision Transformer
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — VLM
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) — Speech Recognition

### Agents
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — Tool Calling
- [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — CoT
- [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171) — Self-Consistency

## License

MIT
