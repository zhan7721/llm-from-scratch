# 去噪扩散概率模型 (DDPM)

> **模块 05 -- 多模态，第 03 章**

去噪扩散概率模型 (Ho et al., 2020) 通过学习反转逐渐添加噪声的过程来生成图像。前向过程在 T 个时间步内逐渐向干净图像添加高斯噪声，反向过程学习逐步去噪，从纯噪声中恢复干净图像。

---

## 前置要求

- 对神经网络和反向传播有基本理解
- PyTorch 基础：`nn.Module`、`nn.Conv2d`、张量操作
- 理解高斯分布和噪声

## 文件

| 文件 | 用途 |
|------|------|
| `diffusion.py` | DDPM 核心实现 |
| `exercise.py` | 填空练习，巩固理解 |
| `solution.py` | 练习参考答案 |
| `tests.py` | pytest 测试 |

---

## 整体概览

DDPM 通过两阶段过程生成图像：

```
训练阶段：
    干净图像 x_0
        |
        v
    前向扩散（逐渐添加噪声）
        |
        v
    在随机时间步 t 的噪声图像 x_t
        |
        v
    UNet 预测噪声 epsilon_theta(x_t, t)
        |
        v
    损失 = MSE(预测噪声, 实际噪声)

生成阶段：
    纯噪声 x_T ~ N(0, I)
        |
        v
    迭代去噪（T 步）
        |
        v
    干净图像 x_0
```

### 核心思想：学习去噪

核心思想很简单：不是直接学习生成图像，而是学习去除噪声。如果你能在每一步去除少量噪声，你就可以通过迭代去噪将纯噪声变成干净图像。

---

## NoiseScheduler：扩散调度器

噪声调度器控制在每个时间步添加多少噪声。

### 前向过程（添加噪声）

```python
q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1 - alpha_bar_t) * I)
```

这意味着：
- `x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon`
- 随着 `t` 增加，`alpha_bar_t` 减小，因此添加更多噪声
- 在 `t=0` 时：几乎干净（`alpha_bar_0 ~ 1`）
- 在 `t=T` 时：几乎纯噪声（`alpha_bar_T ~ 0`）

### 线性 Beta 调度

```python
beta_t = beta_start + (beta_end - beta_start) * t / T
```

- `beta_t` 控制步骤 `t` 的噪声方差
- 线性调度：`beta_t` 从 `beta_start` 线性增加到 `beta_end`
- 典型值：`beta_start = 1e-4`，`beta_end = 0.02`

### 关键量

| 符号 | 公式 | 含义 |
|------|------|------|
| `beta_t` | 线性调度 | 步骤 t 的噪声方差 |
| `alpha_t` | `1 - beta_t` | 步骤 t 的信号保留 |
| `alpha_bar_t` | `prod(alpha_1...alpha_t)` | 累积信号保留 |
| `sqrt(alpha_bar_t)` | `sqrt(prod(alpha_1...alpha_t))` | 信号系数 |
| `sqrt(1 - alpha_bar_t)` | `sqrt(1 - prod(alpha_1...alpha_t))` | 噪声系数 |

### 反向过程（去噪）

```python
x_{t-1} = (1/sqrt(alpha_t)) * (x_t - beta_t/sqrt(1-alpha_bar_t) * eps_theta(x_t, t)) + sigma_t * z
```

其中：
- `eps_theta(x_t, t)` 是 UNet 预测的噪声
- `sigma_t` 是后验标准差
- `z ~ N(0, I)` 是随机噪声（t=0 时除外）

---

## UNet：噪声预测网络

UNet 预测在给定时间步添加到图像中的噪声。

### 架构

```
输入：噪声图像 x_t (B, C, H, W) + 时间步 t
    |
    v
时间嵌入：正弦嵌入 -> MLP
    |
    v
初始卷积：Conv2d(C, base_channels, 3)
    |
    v
编码器（下采样）：
    级别 1：ResBlock x N -> 跳跃连接
        |
        v (步长 2 卷积)
    级别 2：ResBlock x N -> 跳跃连接
        |
        v (步长 2 卷积)
    级别 3：ResBlock x N -> 跳跃连接
    |
    v
瓶颈层：
    ResBlock -> ResBlock
    |
    v
解码器（上采样）：
    级别 3：Concat(skip) -> ResBlock x N
        |
        v (转置卷积)
    级别 2：Concat(skip) -> ResBlock x N
        |
        v (转置卷积)
    级别 1：Concat(skip) -> ResBlock x N
    |
    v
最终：GroupNorm -> SiLU -> Conv2d(base_channels, C, 3)
    |
    v
输出：预测噪声 epsilon (B, C, H, W)
```

### 时间嵌入

正弦嵌入（类似 Transformer 位置编码）将时间步编码为密集向量：

```python
emb = [sin(t * freq_1), cos(t * freq_1), sin(t * freq_2), cos(t * freq_2), ...]
```

这允许模型知道它正在去噪哪个时间步。

### ResBlock

每个残差块：
1. 应用 GroupNorm + SiLU + Conv2d
2. 添加时间嵌入（在空间维度上广播）
3. 应用 GroupNorm + SiLU + Conv2d
4. 添加跳跃连接

### 跳跃连接

UNet 的关键特征：编码器特征在每个分辨率级别与解码器特征连接。这保留了下采样过程中丢失的空间信息。

---

## DDPMTrainer：训练循环

训练目标很简单：

```python
# 1. 采样干净图像
x_0 = 从数据集采样()

# 2. 采样随机时间步
t = 随机整数(0, T)

# 3. 采样噪声
epsilon = 与 x_0 形状相同的随机噪声()

# 4. 创建噪声图像
x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

# 5. 预测噪声
predicted_epsilon = model(x_t, t)

# 6. 计算损失
loss = MSE(predicted_epsilon, epsilon)
```

模型学习预测添加的噪声。这等价于学习分数函数（对数概率密度的梯度）。

---

## ddpm_sample：生成

生成通过迭代去噪从纯噪声开始：

```python
# 从纯噪声开始
x_T = randn(shape)

# 从 T-1 到 0 去噪
for t in reversed(range(T)):
    predicted_noise = model(x_t, t)
    x_{t-1} = 反向步骤(predicted_noise, t, x_t)

# 最终输出
return x_0
```

每一步去除少量噪声。经过 T 步后，我们得到干净图像。

---

## DDPM vs 其他生成模型

| 方面 | GAN | VAE | DDPM |
|------|-----|-----|------|
| 训练 | 对抗性（最小-最大） | ELBO 优化 | 简单 MSE 损失 |
| 稳定性 | 模式崩溃、训练不稳定 | 后验崩溃 | 稳定、可重现 |
| 质量 | 清晰图像 | 模糊图像 | 高质量 |
| 速度 | 单次前向传播 | 单次前向传播 | 多步（慢） |
| 多样性 | 受模式崩溃限制 | 好 | 优秀 |
| 似然 | 无 | 下界 | 下界 |

---

## 如何运行

### 运行测试

```bash
cd /path/to/llm-from-scratch
pytest 05-multimodal/03_diffusion/tests.py -v
```

### 运行练习

打开 `exercise.py` 并填写 `TODO` 部分。然后验证：

```bash
pytest 05-multimodal/03_diffusion/tests.py -v
```

---

## 练习

打开 `exercise.py` 来练习自己实现 DDPM。

### 练习顺序

1. **NoiseScheduler**：实现 beta 调度、alpha_bar 计算、add_noise 和反向步骤
2. **SinusoidalTimeEmbedding**：实现时间步的正弦位置编码
3. **ResBlock**：实现带时间嵌入注入的残差块
4. **DDPMTrainer**：实现带噪声预测损失的训练循环

### 提示

- 前向过程只是一个加权和：`x_t = signal * x_0 + noise * epsilon`
- 反向过程在每一步预测并去除噪声
- 时间嵌入让模型知道它在处理哪个时间步
- UNet 中的跳跃连接保留空间信息
- 训练损失就是预测噪声和实际噪声之间的 MSE

---

## 关键要点

1. **DDPM 学习反转噪声。** 不是直接生成图像，而是学习逐步去噪。

2. **前向过程是可处理的。** 我们可以使用 `alpha_bar_t` 闭式计算任意 `t` 的 `x_t`。

3. **反向过程是学习得到的。** UNet 在每一步预测噪声，用于去噪。

4. **训练很简单。** 只需采样时间步、添加噪声、预测噪声并计算 MSE 损失。

5. **生成是迭代的。** 从纯噪声开始，进行 T 步去噪。这很慢但产生高质量样本。

6. **时间嵌入至关重要。** 模型需要知道它在去噪哪个时间步以预测正确的噪声量。

---

## 延伸阅读

- [DDPM 论文 (Ho et al., 2020)](https://arxiv.org/abs/2006.11239) -- 原始 DDPM 论文
- [改进的 DDPM (Nichol & Dhariwal, 2021)](https://arxiv.org/abs/2102.09672) -- 改进的噪声调度和采样
- [基于去噪扩散的生成建模：基础与应用 (2023)](https://arxiv.org/abs/2303.14643) -- 综合教程
- [基于分数的生成建模 (Song & Ermon, 2019)](https://arxiv.org/abs/1907.05600) -- 与分数匹配的联系
- [什么是扩散模型？(Lil'Log)](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) -- 优秀的博客文章解释
