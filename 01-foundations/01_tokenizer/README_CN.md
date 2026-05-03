# BPE 分词器

> **模块 01 — 基础，第 01 章**

大语言模型不能直接阅读文本，它们处理的是数字。**分词器（Tokenizer）**是连接人类语言和模型数值世界的桥梁。在本章中，我们将从零开始实现一个[字节对编码（BPE）](https://en.wikipedia.org/wiki/Byte_pair_encoding)分词器——这与 GPT-2、LLaMA 以及几乎所有现代大语言模型使用的算法相同。

---

## 前置要求

- 基础 Python 知识（字典、元组、循环）
- 本章不需要 ML 或 PyTorch 知识

## 文件说明

| 文件 | 用途 |
|------|------|
| `tokenizer.py` | BPE 核心实现（训练、编码、解码） |
| `train_tokenizer.py` | 演示脚本——在示例文本上训练并展示结果 |
| `exercise.py` | 填空练习，加深理解 |
| `solution.py` | 练习的参考答案 |
| `tests.py` | pytest 正确性测试 |
| `test_exercise.py` | 练习版本的 pytest 测试 |

---

## 为什么需要分词？

神经网络是一个数学函数，输入数字，输出数字。原始文本如 `"Hello world"` 不是数字——它是一串 Unicode 字符。我们需要一个确定性的、可逆的映射，将文本转换为整数。

最简单的方法：给每个字符分配一个 ID。

```
H → 72, e → 101, l → 108, l → 108, o → 111, ...
```

这种方式可以工作，但存在问题：

1. **词汇表爆炸。** Unicode 有超过 149,000 个字符，字符级分词器需要巨大的词汇表，而且罕见字符会带来问题。
2. **没有语义分组。** 单词 `"playing"` 被拆分成 7 个独立的 token。模型必须从零学习 `p-l-a-y-i-n-g` 是一个整体概念。
3. **序列过长。** 每个词的 token 越多，输入序列越长，计算开销越大。

**BPE** 通过学习**子词单元**来解决这些问题——这些片段比单个字符大，但比整个词小。常见模式如 `"the"`、`"ing"` 或 `"tion"` 会成为单个 token，而罕见词会被拆分为已知的子词。

---

## BPE 算法

BPE 全称是**字节对编码（Byte Pair Encoding）**，最初是一种数据压缩算法（[Gage, 1994](https://www.derczynski.com/papers/archive/bpe_gage.pdf)）。核心思想非常简单：

> 重复找到最频繁的相邻 token 对，将它们合并为一个新的 token。

### 分步示例

让我们在一个小语料库上演示 BPE：

```
corpus = "aa ab aa ab aa ab"
```

**第 0 步：初始化**

将语料库拆分成词并统计频率：

```
"aa" → 3 次
"ab" → 3 次
```

将每个词表示为字节序列：

```
"a", "a" → 3
"a", "b" → 3
```

初始词汇表是 256 个可能的字节值（0-255）。

**第 1 步：统计配对并合并**

统计所有词中所有相邻对的出现次数：

```
(a, a) → 3 次  （来自 "aa"）
(a, b) → 3 次  （来自 "ab"）
```

两个对的频率相同（3）。平局时可以任意选择，假设我们选 `(a, a)`。

将 `(a, a)` 合并为新 token `aa`，分配 ID 256。

```
word_freqs 变为：
  (aa,) → 3      # 原来是 (a, a)
  (a, b) → 3     # 未变
```

**第 2 步：再次统计配对并合并**

```
(a, b) → 3 次
```

将 `(a, b)` 合并为 `ab`，分配 ID 257。

```
word_freqs 变为：
  (aa,) → 3
  (ab,) → 3
```

**第 3 步：没有更多配对**

每个词现在都是单个 token，没有相邻对可以合并。算法停止。

**结果：**

- 词汇表：256 个基础字节 + `aa`（ID 256）+ `ab`（ID 257）= 258 个 token
- 学到的合并规则：`['a' + 'a' → 'aa']`，`['a' + 'b' → 'ab']`

输入文本 `"aa ab aa ab"` 现在可以编码为 `[256, 257, 256, 257]`——只需 4 个 token，而不是 11 个字节。

### 为什么用字节？

现代 BPE 实现（包括我们的）在**字节**而非字符上操作。这有一个主要优势：任何可能的字符串都能被表示，即使它包含不常见的 Unicode 字符、表情符号或二进制数据。基础词汇表始终是 256 个 token——每个可能的字节值对应一个。

---

## 代码详解

以下是 `tokenizer.py` 中核心 `BPETokenizer` 类的工作原理。

### 训练：`train(corpus)`

```python
def train(self, corpus: str):
    self._build_base_vocab()           # 1. 初始化 256 字节词汇表
    words = corpus.split()             # 2. 拆分成词
    word_freqs = ...                   # 3. 统计词频
    # 每个词是一个单字节 token 的元组

    for _ in range(num_merges):        # 4. 重复 (vocab_size - 256) 次
        pair_counts = self._get_pair_counts(word_freqs)  # 统计配对
        best_pair = max(pair_counts, key=...)             # 找到最频繁的
        merged = best_pair[0] + best_pair[1]              # 拼接字节
        self.vocab[new_id] = merged    # 添加到词汇表
        self.merges.append(best_pair)  # 记录合并规则
        word_freqs = self._merge_pair(word_freqs, best_pair)  # 应用合并
```

训练循环是 BPE 的核心。每次迭代找到最常见的配对并合并。`word_freqs` 字典会被原地更新，因此下一次迭代会看到合并后的 token。

### 编码：`encode(text)`

```python
def encode(self, text: str) -> List[int]:
    tokens = [bytes([b]) for b in text.encode("utf-8")]  # 从字节开始
    for pair in self.merges:              # 按顺序应用每条合并规则
        tokens = merge(tokens, pair)
    return [self.inverse_vocab[t] for t in tokens]  # 转换为 ID
```

编码按照**训练时发现的相同顺序**应用学习到的合并规则。这种从左到右的贪婪应用使得编码快速且确定。

### 解码：`decode(ids)`

```python
def decode(self, ids: List[int]) -> str:
    token_bytes = b"".join(self.vocab[id] for id in ids)
    return token_bytes.decode("utf-8", errors="replace")
```

解码非常简单：查找每个 ID 对应的字节，拼接起来，然后解码为字符串。`errors="replace"` 标志确保了对无效字节序列的鲁棒性。

### 关键数据结构

- **`vocab`**：将 token ID（int）映射到字节。大小 = 256 + 合并次数。
- **`inverse_vocab`**：反向查找——字节到 ID。编码时使用。
- **`merges`**：训练时学到的有序 `(bytes, bytes)` 对列表。这就是"模型"——它定义了如何压缩文本。

---

## 如何运行

### 训练分词器

```bash
cd /path/to/llm-from-scratch
python 01-foundations/01_tokenizer/train_tokenizer.py
```

预期输出：

```
Vocabulary size: 300
Number of merges: 44

Original: The quick brown fox
Token IDs: [84, 104, 101, ...]
Decoded: The quick brown fox
Roundtrip OK: True

First 10 merges:
  1. b' ' + b't' -> b' t'
  2. b' ' + b'a' -> b' a'
  ...
```

### 运行测试

```bash
pytest 01-foundations/01_tokenizer/tests.py -v
```

---

## 练习

打开 `exercise.py` 来自己动手实现 BPE。文件中有一个 `BPETokenizerExercise` 类，每个关键方法都有 `TODO` 占位符。

### 练习顺序

1. **`_get_pair_counts`** — 统计所有词中的相邻配对
2. **`_merge_pair`** — 将配对的所有出现替换为合并后的 token
3. **`train`** — 将所有部分组装成训练循环
4. **`encode`** — 应用合并规则来编码新文本
5. **`decode`** — 将 token ID 转换回文本

### 建议

- 从 `_get_pair_counts` 开始。它是最简单的方法，能帮助你建立 BPE 如何统计频率的直觉。
- `_merge_pair` 是最棘手的部分。从左到右遍历词的元组，构建一个新列表。
- 一旦你有了 `_get_pair_counts` 和 `_merge_pair`，`train` 方法主要是将它们串联起来。
- `encode` 复用了与 `_merge_pair` 相同的合并逻辑，但作用于列表而非字典。

### 验证你的答案

```bash
pytest 01-foundations/01_tokenizer/test_exercise.py -v
```

---

## 核心要点

1. **分词是一个压缩问题。** BPE 在训练语料库中找到重复的字节模式，并将它们合并为单个 token。更频繁的模式获得更短的表示。

2. **BPE 在字节而非字符上操作。** 这意味着它可以处理任何文本——英文、中文、表情符号或二进制数据——无需特殊规则。基础词汇表始终是 256 个 token。

3. **合并顺序很重要。** 编码按照学习到的精确顺序应用合并规则。不同的训练语料库产生不同的合并规则，从而产生不同的分词结果。

4. **子词分词平衡了词汇表大小和序列长度。** 常见词成为单个 token；罕见词被拆分为已知子词。这使词汇表保持在可管理的范围内，同时能表示任何文本。

---

## 延伸阅读

- [原始 BPE 论文 (Sennrich et al., 2016)](https://arxiv.org/abs/1608.00221) — BPE 在机器翻译中的应用
- [Hugging Face Tokenizers](https://huggingface.co/docs/tokenizers/) — 生产级分词库
- [Let's build the GPT Tokenizer (Karpathy)](https://www.youtube.com/watch?v=zduSFxRajkE) — BPE 视频讲解
- [minbpe](https://github.com/karpathy/minbpe) — Andrej Karpathy 的 BPE 最小实现
