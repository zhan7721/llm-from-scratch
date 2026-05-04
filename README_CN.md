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

### 02-pretrain — 预训练

| 章节 | 主题 | 核心概念 |
|------|------|----------|
| 01_data_pipeline | 数据管道 | 动态填充，序列打包，DataLoader |
| 02_data_engineering | 数据工程 | MinHash 去重，质量过滤，数据混合 |
| 03_training_loop | 训练循环 | AdamW，余弦学习率，梯度裁剪，梯度累积 |
| 04_distributed | 分布式训练 | DDP，FSDP，多 GPU 训练 |
| 05_scaling_laws | 缩放定律 | Chinchilla 最优，计算最优训练 |
| 06_evaluation | 评估 | 困惑度，基准测试，涌现能力 |

### 03-sft — 监督微调

| 章节 | 主题 | 核心概念 |
|------|------|----------|
| 01_instruction_tuning | 指令微调 | Alpaca 格式，标签掩码，仅响应损失 |
| 02_lora | LoRA | 低秩适配，参数高效微调 |
| 03_qlora | QLoRA | NF4 量化 + LoRA，4-bit 训练 |
| 04_neftune | NEFTune | 嵌入噪声提升泛化 |
| 06_long_context_sft | 长上下文 SFT | 位置插值，NTK 感知缩放 |

### 04-rl — 强化学习（238 个测试）

| 章节 | 主题 | 核心概念 |
|------|------|----------|
| 01_reward_model | 奖励模型 | Bradley-Terry 模型，标量头，偏好排序 |
| 02_ppo | PPO | 裁剪代理，GAE，KL 惩罚，rollout |
| 03_dpo | DPO | 直接偏好优化，无参考模型 |
| 04_online_dpo | Online DPO | 动态偏好对，在线生成 |
| 05_grpo | GRPO | 组相对优势，无价值网络 |
| 06_prm | 过程奖励模型 | 步骤级评分，best-of-N 选择 |
| 07_reward_hacking | 奖励黑客 | 检测，KL 约束，奖励集成 |

### 05-multimodal — 多模态（137 个测试）

| 章节 | 主题 | 核心概念 |
|------|------|----------|
| 01_vision_encoder | 视觉 Transformer | Patch 嵌入，CLS token，位置编码 |
| 02_vlm | 视觉语言模型 | 视觉投影器，LLaVA 架构，跨模态融合 |
| 03_diffusion | 扩散模型 (DDPM) | 噪声调度器，U-Net，正弦时间嵌入 |
| 04_speech_recognition | 语音识别 | Whisper 编码器-解码器，交叉注意力 |
| 05_speech_generation | 语音生成 | TTS，WaveNet 声码器，时长预测 |

### 06-agent — 智能体（216 个测试）

| 章节 | 主题 | 核心概念 |
|------|------|----------|
| 01_tool_calling | 工具调用 | 工具注册，解析器，执行器，智能体循环 |
| 02_reasoning | 推理 | 思维链，自一致性，步骤提取 |
| 03_mcp | 模型上下文协议 | MCP 消息，客户端-服务器，标准化协议 |
| 04_multi_agent | 多智能体 | 编排器，共享内存，辩论模式 |
| 05_code_interpreter | 代码解释器 | 沙盒执行，代码解析，工件存储 |

### 07-inference — 推理优化（即将推出）

量化 (GPTQ/AWQ)，投机解码，连续批处理，模型服务。

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 运行所有测试（6 个模块，591+ 个测试）
pytest 04-rl/ 05-multimodal/ 06-agent/ -v

# 运行特定模块
pytest 04-rl/ -v

# 运行特定章节
pytest 04-rl/03_dpo/tests.py -v
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

### 架构
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 原始 Transformer
- [LLaMA: Open and Efficient Foundation LLMs](https://arxiv.org/abs/2302.13971) — LLaMA 架构
- [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE
- [YaRN: Efficient Context Window Extension of LLMs](https://arxiv.org/abs/2309.00071) — YaRN
- [Switch Transformers: Scaling to Trillion Parameter Models](https://arxiv.org/abs/2101.03961) — MoE

### 训练与对齐
- [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) — 缩放定律
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA
- [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) — QLoRA
- [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO
- [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — PPO
- [DeepSeekMath: Pushing the Limits of Math Reasoning](https://arxiv.org/abs/2402.03300) — GRPO
- [Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — 过程奖励模型

### 多模态
- [An Image is Worth 16x16 Words: ViT](https://arxiv.org/abs/2010.11929) — 视觉 Transformer
- [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) — 视觉语言模型
- [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM
- [Robust Speech Recognition via Large-Scale Weak Supervision (Whisper)](https://arxiv.org/abs/2212.04356) — 语音识别

### 智能体
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761) — 工具调用
- [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — 思维链
- [Self-Consistency Improves CoT Reasoning](https://arxiv.org/abs/2203.11171) — 自一致性

## 许可证

MIT
