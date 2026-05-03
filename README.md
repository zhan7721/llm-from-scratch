# LLM From Scratch — Build a Large Language Model from Scratch in Pure PyTorch

A hands-on learning project that guides you through building a complete Large Language Model using only PyTorch. From tokenization to serving, every component is implemented from scratch with clear explanations and practical exercises. This project covers the full LLM lifecycle: foundations, pretraining, fine-tuning, reinforcement learning, multimodal capabilities, agents, and inference optimization.

## Learning Path

This project is organized into 7 modules, each covering a key area of LLM development:

| # | Module | Topics |
|---|--------|--------|
| 01 | **Foundations** | Tokenizer, Embedding, Attention, Transformer Block, Model Architecture, MoE, KV Cache, Long Context |
| 02 | **Pretrain** | Data Pipeline, Data Engineering, Training Loop, Distributed Training, Scaling Laws, Evaluation |
| 03 | **SFT** | Instruction Tuning, LoRA, QLoRA, NEFTune, Chat Template, Long Context SFT |
| 04 | **RL** | Reward Model, PPO, DPO, Online DPO, GRPO, PRM, Reward Hacking |
| 05 | **Multimodal** | Vision Encoder, VLM, Diffusion, Speech Recognition, Speech Generation |
| 06 | **Agent** | Tool Calling, Reasoning, MCP, Multi-Agent, Code Interpreter |
| 07 | **Inference** | Quantization, Speculative Decoding, Continuous Batching, Serving |

## Prerequisites

- Python 3.10+
- PyTorch 2.0+

## How to Use

Each chapter follows a consistent structure:

- **README.md** — Concepts and theory explained
- **\*.py** — Core implementation building the component from scratch
- **exercise.py** — Practice problems to reinforce understanding
- **solution.py** — Reference solutions for the exercises
- **tests.py** — pytest-based tests to verify correctness

## Hardware Requirements

| Tier | Hardware | Notes |
|------|----------|-------|
| Minimum | Mac M1/M2 or NVIDIA RTX 3060 | Sufficient for small models and learning exercises |
| Recommended | NVIDIA RTX 3090 | Enables training larger models and distributed experiments |

## Quick Start

```bash
pip install -r requirements.txt
pytest 01-foundations/ -v
```
