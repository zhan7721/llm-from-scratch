# 评估 (Evaluation)

训练语言模型后，你需要衡量它的效果。评估告诉你模型是否学到了你想要的东西、
与其他模型相比如何，以及是否可以用于下游任务。

## 为什么评估很重要

训练损失告诉你模型对训练数据的拟合程度，但它不能告诉你：

- 模型是否能泛化到未见过的文本。
- 生成的输出是否连贯或有用。
- 模型与其他模型相比处于什么水平。

良好的评估结合了**定量指标**（可以随时间追踪的数字）和**定性检查**
（阅读实际输出）。

## 困惑度 (Perplexity)

困惑度是语言模型的标准指标。它衡量模型对测试数据的"惊讶"程度。

### 定义

困惑度是平均交叉熵损失的指数：

```
PPL = exp(L)
L = -(1/T) * sum_t log P(x_t | x_{<t})
```

其中 T 是总 token 数，P(x_t | x_{<t}) 是模型在给定前文的情况下对 token
x_t 的预测概率。

### 解读

- **越低越好。** 困惑度为 1 表示模型对每个 token 的预测都是确定的。
- **随机基线。** 对于大小为 V 的词表，均匀随机模型的困惑度为 V。在 50,000
  词表上 PPL=100 的模型远优于随机模型。
- **粗略直觉。** PPL=10 表示模型在每个位置大约在 10 个可能的下一个 token
  之间选择。

### 与损失的关系

困惑度和交叉熵损失包含相同的信息。困惑度只是损失的指数，所以：

- 损失 4.6 -> PPL = exp(4.6) ~ 100
- 损失 3.0 -> PPL = exp(3.0) ~ 20
- 损失 2.3 -> PPL = exp(2.3) ~ 10

人们通常报告困惑度，因为它比原始的 nats 损失更直观。

### 实际注意事项

- **忽略填充。** 在交叉熵中使用 `ignore_index=-100` 来跳过填充 token。
- **Token 级别 vs. 词级别。** 困惑度取决于分词器。使用 50K 词表 BPE 的模型
  和使用 100K 词表词级别分词的模型的困惑度不可直接比较。
- **上下文长度。** 使用完整上下文（如 2048 tokens）计算的困惑度会低于
  短上下文的困惑度。

## Token 准确率

Token 准确率衡量模型的 top-1 预测与真实下一个 token 匹配的频率。

```
准确率 = 正确预测数 / 总 token 数
```

### 适用场景

- **训练诊断。** 如果准确率持平而损失下降，说明模型变得更自信但没有改变
  它的首选猜测。
- **模型比较。** 损失相似的两个模型可能有不同的准确率分布。

### 局限性

- 准确率只检查 top-1 预测。将正确 token 排在第二位的模型与排在最后的模型
  得分相同。
- 对于大词表，即使是好模型也可能有较低的准确率（50K+ 词表模型 10-30%
  是正常的）。

## 生成质量

困惑度和准确率衡量下一个 token 的预测，但用户交互的是生成的文本。生成
质量更难衡量。

### 自动指标

- **BLEU / ROUGE。** 通过 n-gram 重叠将生成文本与参考文本比较。常见于
  翻译和摘要任务，对开放式生成不太有用。
- **Distinct-n。** 通过计算唯一 n-gram 的数量来衡量多样性。低 distinct-n
  表示生成重复。
- **生成文本的困惑度。** 用一个单独的"评判"模型对输出进行评分。来自
  评判模型的较低困惑度表示更流畅的文本。

### 人工评估

金标准。让人来评分：

- **流畅性。** 文本是否合乎语法、自然？
- **连贯性。** 跨句子的文本是否合理？
- **事实性。** 声明是否准确？
- **有用性。** 文本是否回答了提示？

人工评估昂贵且缓慢，但它能发现自动指标遗漏的问题。

### 实际方法

对于预训练评估，常见的工作流程是：

1. 在训练期间跟踪在保留验证集上的困惑度。
2. 定期生成样本并手动阅读。
3. 如果模型通过了"嗅觉测试"（连贯、切题），它很可能学得不错。

## 基准数据集

标准数据集允许跨论文比较模型。

### WikiText-103

- 来自维基百科的 100M token。
- 原始为词级别分词，常适配为 BPE。
- 长距离依赖基准（完整文章，非随机打乱的句子）。

### C4 (Colossal Clean Crawled Corpus)

- 约 750GB 的清洗后网页文本。
- 被 T5 及许多后续模型使用。
- 通用语言建模的良好代理。

### The Pile

- 来自 22 个子语料库的 800GB 多样化文本。
- 包括书籍、代码、学术论文、StackExchange 等。
- 测试广泛知识，而非仅仅是维基百科风格的文本。

### LAMBADA

- 通过预测段落最后一个词来测试长距离依赖。
- 衡量模型是否能利用远距离上下文。

## 训练期间的评估

### 验证损失

最常见的方法：保留一部分数据，每 N 步计算一次损失。

```python
# 训练循环中
if step % eval_interval == 0:
    val_loss = evaluate(model, val_loader)
    print(f"Step {step}: val_loss={val_loss:.4f}, ppl={math.exp(val_loss):.1f}")
```

### 早停 (Early Stopping)

如果验证损失停止改善，停止训练以节省计算并防止过拟合。

```python
best_val_loss = float("inf")
patience_counter = 0

for step in range(max_steps):
    train_step(model, batch)
    if step % eval_interval == 0:
        val_loss = evaluate(model, val_loader)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            save_checkpoint(model)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping")
                break
```

### 检查点选择

定期保存检查点，并在基准测试套件上评估每个检查点。选择验证指标最好的
检查点，而不是最后一个。

## 代码详解

### `perplexity(model, data_loader, device)`

遍历数据加载器，累积交叉熵损失（求和归约）和 token 数量，然后返回
`exp(total_loss / total_tokens)`。使用 `ignore_index=-100` 跳过填充位置。

### `compute_token_accuracy(model, data_loader, device)`

从 logits 中获取 argmax 预测，与标签比较，并屏蔽 `labels == -100` 的位置。
返回正确预测的比例。

### `generate_samples(model, tokenizer, prompts, ...)`

编码每个提示，使用温度和 top-k 采样运行 `model.generate()`，解码输出，
返回生成的字符串列表。

### `Evaluator`

一个便利类，将困惑度、准确率和可选的生成整合到单个 `evaluate()` 调用中。
返回指标字典。

## 运行测试

```bash
cd 02-pretrain/06_evaluation
pytest tests.py -v
```

## 练习

打开 `exercise.py` 并实现 TODO 项目：

1. `perplexity` —— 计算损失并转换为困惑度。
2. `compute_token_accuracy` —— 计算正确预测数。
3. `Evaluator.evaluate` —— 编排指标计算。

使用 `tests.py` 检查你的实现。

## 参考文献

- Chen, M. et al. (2021). "Evaluating Large Language Models Trained on
  Code" (Codex/HumanEval). arXiv:2107.03374.
- Merity, S. et al. (2016). "Pointer Sentinel Mixture Models"
  (WikiText-103). arXiv:1609.07843.
- Gao, L. et al. (2020). "The Pile: An 800GB Dataset of Diverse Text for
  Language Modeling." arXiv:2101.00027.
- Raffel, C. et al. (2019). "Exploring the Limits of Transfer Learning
  with a Unified Text-to-Text Transformer" (T5/C4). arXiv:1910.10683.
