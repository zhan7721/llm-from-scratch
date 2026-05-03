# LLM From Scratch

从零开始，使用纯 PyTorch 构建大语言模型。

## 项目概述

本项目教你如何从零实现 LLM 的每个组件 —— 不使用高级框架，仅用 PyTorch。每个章节包含：
- 清晰的讲解（英文 + 中文）
- 可运行的实现代码
- 带 TODO 的动手练习
- 参考答案
- 验证你工作的测试

## 学习路径

### 01-foundations — 核心组件

| 章节 | 主题 | 核心概念 |
|------|------|----------|
| 01_tokenizer | BPE 分词 | Byte-Pair Encoding，子词分词 |
| 02_embedding | 嵌入 | 词嵌入，RoPE 位置编码 |
| 03_attention | 注意力机制 | 多头注意力，分组查询注意力 |
| 04_transformer_block | Transformer Block | Pre-Norm，SwiGLU FFN，RMSNorm，残差连接 |
| 05_model_architecture | GPT 模型 | 完整模型组装，权重共享，生成 |
| 06_moe | 混合专家 | Top-K 路由，专家网络，负载均衡 |
| 07_kv_cache | KV Cache | 高效自回归推理 |
| 08_long_context | 长上下文 | RoPE 缩放，YaRN，位置插值 |

### 02-pretrain — 预训练（即将推出）

数据管道，数据工程，训练循环，分布式训练，缩放定律，评估。

### 03-sft — 监督微调（即将推出）

指令微调，LoRA，QLoRA，NEFTune，聊天模板，长上下文 SFT。

### 04-rl — 强化学习（即将推出）

奖励模型，PPO，DPO，Online DPO，GRPO，过程奖励模型，奖励黑客防御。

### 05-multimodal — 多模态（即将推出）

视觉编码器 (ViT)，视觉语言模型，扩散模型，语音识别，语音生成。

### 06-agent — 智能体（即将推出）

工具调用，推理 (CoT)，MCP，多智能体系统，代码解释器。

### 07-inference — 推理优化（即将推出）

量化 (GPTQ/AWQ)，投机解码，连续批处理，模型服务。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行所有测试
pytest 01-foundations/ -v

# 运行特定章节
pytest 01-foundations/03_attention/tests.py -v

# 运行分词器演示
python 01-foundations/01_tokenizer/train_tokenizer.py
```

## 每个章节包含

```
chapter_name/
├── README.md / README_CN.md   # 教程（英文 + 中文）
├── implementation.py           # 核心实现
├── exercise.py                 # 填写 TODO
├── solution.py                 # 参考答案
└── tests.py                    # 验证你的工作
```

## 硬件要求

| 要求 | 最低配置 | 推荐配置 |
|------|----------|----------|
| GPU | Mac M1/M2 (MPS) 或 RTX 3060 (12GB) | RTX 3090 (24GB) |
| 内存 | 8GB | 16GB |
| Python | 3.10+ | 3.11+ |

## 依赖

- PyTorch 2.0+
- pytest（测试）
- tiktoken（BPE 对比）
- transformers（权重加载对比）
- datasets（示例数据下载）
- wandb（可选，训练监控）

## 项目结构

```
llm-from-scratch/
├── 01-foundations/     # 核心 LLM 组件
├── 02-pretrain/        # 预训练
├── 03-sft/             # 监督微调
├── 04-rl/              # 强化学习
├── 05-multimodal/      # 视觉 + 语音
├── 06-agent/           # 智能体 + 工具
├── 07-inference/       # 推理优化
├── configs/            # 训练配置
├── data/               # 示例数据集
├── scripts/            # 工具脚本
└── tests/              # 集成测试
```

## 参考文献

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer
- [LLaMA: Open and Efficient Foundation LLMs](https://arxiv.org/abs/2302.13971) — LLaMA 架构
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) — YaRN
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — MoE
- [DPO: Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO

## 许可证

MIT
