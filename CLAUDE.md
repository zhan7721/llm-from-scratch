# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A PyTorch educational project for building Large Language Models from scratch. Pure PyTorch, no high-level frameworks. 591+ tests across 6 modules.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests (04-rl, 05-multimodal, 06-agent have tests)
pytest 04-rl/ 05-multimodal/ 06-agent/ -v

# Run a specific module
pytest 04-rl/ -v

# Run a single chapter's tests
pytest 04-rl/03_dpo/tests.py -v

# Run a specific test
pytest 04-rl/03_dpo/tests.py::test_dpo_loss -v
```

## Architecture

### Module Structure

The codebase is organized into 7 progressive modules:

- `01-foundations/` — Core LLM components (tokenizer, embeddings, attention, transformer blocks, model architecture, MoE, KV cache, long context)
- `02-pretrain/` — Pre-training pipeline (data pipeline, data engineering, training loop, distributed training, scaling laws, evaluation)
- `03-sft/` — Supervised fine-tuning (instruction tuning, LoRA, QLoRA, NEFTune, long context SFT)
- `04-rl/` — Reinforcement learning (reward model, PPO, DPO, online DPO, GRPO, process reward model, reward hacking)
- `05-multimodal/` — Vision and speech (ViT, VLM/LLaVA, DDPM diffusion, Whisper speech recognition, TTS)
- `06-agent/` — Agents and tools (tool calling, reasoning/CoT, MCP, multi-agent, code interpreter)
- `07-inference/` — Inference optimization (coming soon: quantization, speculative decoding, serving)

### Chapter Structure

Each chapter follows a consistent pattern:

```
NN_topic_name/
├── README.md / README_CN.md   # Tutorial in English and Chinese
├── implementation.py           # Core reference implementation (e.g., attention.py, dpo.py)
├── exercise.py                 # Fill-in-the-blank exercises with TODO markers
├── solution.py                 # Reference solutions for exercises
├── tests.py                    # pytest tests to verify correctness
└── conftest.py                 # Adds chapter dir to sys.path for imports
```

The main implementation file name varies by chapter (e.g., `attention.py`, `dpo.py`, `tool_calling.py`).

### Testing Conventions

- pytest discovers both `tests.py` and `test_*.py` files (configured in `pytest.ini`)
- Uses `--import-mode=importlib` to avoid module name collisions across chapter directories
- Each chapter has a `conftest.py` that inserts its directory into `sys.path`
- Root `conftest.py` ignores `test_exercise.py` from collection
- Tests import directly from the chapter's implementation module (e.g., `from attention import MultiHeadAttention`)
- Tests use pure PyTorch assertions (`torch.allclose`, shape checks) and pytest parametrize where appropriate

### Dependencies

- PyTorch 2.0+ (core framework)
- pytest (testing)
- tiktoken (BPE tokenizer comparison)
- transformers (weight loading comparison)
- datasets (sample data)
- wandb (optional, training monitoring)
- sentencepiece (tokenizer support)

### Hardware

- Minimum: Mac M1/M2 (MPS) or RTX 3060 (12GB)
- Recommended: RTX 3090 (24GB)
- Python 3.10+ required, 3.11+ recommended
