# LLM预训练数据工程

数据质量可以说是训练强大语言模型最重要的因素。正如那句老话：垃圾进，垃圾出。本章介绍现代LLM预训练管道中使用的三种核心技术数据工程技术。

## 为什么数据质量重要

用于预训练LLM的数据塑造了模型学习的一切。用噪声、重复或低质量文本训练的模型会产生噪声大、质量低的输出。近期研究的主要发现：

- **GPT-3**（2020）：对Common Crawl数据进行仔细过滤至关重要。他们训练了一个分类器来区分"高质量"网页文本和低质量页面。
- **The Pile**（2020）：EleutherAI证明，精心策划的22个多样化数据集的混合比原始网络爬取产生的模型更好。
- **Chinchilla**（2022）：DeepMind证明数据质量比纯粹的数量更重要——用更好数据训练的小模型可以超越用更差数据训练的大模型。
- **FineWeb**（2024）：HuggingFace证明对Common Crawl进行积极的质量过滤可以产生超越所有先前开放网络数据集的数据集。

### 数据管道

典型的预训练数据管道遵循以下阶段：

```
原始爬取数据
    |
    v
URL过滤           -- 移除已知的不良域名
    |
    v
文本提取           -- HTML转干净文本
    |
    v
语言检测           -- 保留目标语言
    |
    v
去重               -- 移除完全重复和近似重复
    |
    v
质量过滤           -- 移除低质量文档
    |
    v
数据混合           -- 按所需比例混合数据源
    |
    v
分词               -- 将文本转换为token序列
```

本章实现了去重、质量过滤和数据混合阶段。

## MinHash + LSH：近似重复检测

### 问题

完全重复检测很容易——只需对每个文档进行哈希。但近似重复很常见且有害：

- 从不同镜像爬取的同一维基百科文章，只有轻微的格式差异
- 跨不同媒体转发的新闻文章，只有轻微编辑
- 跨数千页面重复的样板文本（隐私政策、服务条款）

在近似重复上训练会浪费计算资源并使模型偏向重复模式。

### Shingling

为了比较文档，我们首先将文档转换为**shingles**集合——重叠的字符n-gram。例如，n=5时：

```
"hello world" -> {"hello", "ello ", "llo w", "lo wo", "o wor", " worl", "world"}
```

两个文档shingle集合之间的Jaccard相似度衡量它们的重叠程度：

```
J(A, B) = |A ∩ B| / |A ∪ B|
```

Jaccard相似度为1.0表示shingle集合相同；0.0表示没有重叠。

### MinHash签名

直接计算Jaccard相似度对大的shingle集合来说计算成本很高。MinHash提供了一种高效的近似方法：

1. 定义k个随机哈希函数 h_1, h_2, ..., h_k
2. 对于每个哈希函数，计算文档中所有shingle的最小哈希值
3. 得到的k个值构成文档的**MinHash签名**

关键洞察：对于两个集合A和B，P(h_min(A) == h_min(B)) = J(A, B)。因此两个签名中匹配位置的比例估计了它们的Jaccard相似度。

### 局部敏感哈希（LSH）

即使使用MinHash，所有签名的成对比较仍然是O(n^2)。LSH通过以下方式加速：

1. 将每个签名分成b个band，每个band有r行
2. 将每个band哈希到一个桶中
3. 只比较共享至少一个桶的文档

这将问题从O(n^2)降低到大约O(n)，并且可以通过band/row比例调整敏感度。

### 实现详解

```python
class MinHashDeduplicator:
    def __init__(self, num_hashes=128, ngram_size=5, threshold=0.8):
        ...

    def _get_shingles(self, text):
        # 提取字符n-gram
        text = text.lower().strip()
        return {text[i:i+self.ngram_size] for i in range(len(text) - self.ngram_size + 1)}

    def _minhash(self, shingles):
        # 对于每个哈希函数，找到所有shingle中的最小哈希值
        signature = []
        for a, b in self.hash_params:
            min_hash = float('inf')
            for shingle in shingles:
                h = (a * hash(shingle) + b) & 0xFFFFFFFF
                min_hash = min(min_hash, h)
            signature.append(min_hash)
        return signature

    def _estimate_similarity(self, sig1, sig2):
        # 匹配位置的比例估计Jaccard相似度
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        return matches / len(sig1)

    def deduplicate(self, documents):
        # 计算签名，然后成对比较
        signatures = [self._minhash(self._get_shingles(doc)) for doc in documents]
        keep = [True] * len(documents)
        for i in range(len(documents)):
            for j in range(i+1, len(documents)):
                if self._estimate_similarity(signatures[i], signatures[j]) >= self.threshold:
                    keep[j] = False
        return [doc for doc, k in zip(documents, keep) if k]
```

## 质量过滤

并非所有网页文本都适合训练。质量过滤器会移除可能降低模型性能的文档。

### 常见质量信号

| 信号 | 重要性 |
|------|--------|
| 词数 | 太少 = 信息不足；太多 = 可能是数据转储 |
| 字母比例 | 低比例意味着大量数字、代码或乱码 |
| 数字比例 | 高比例意味着数值数据、表格或垃圾邮件 |
| 平均词长 | 非常短 = 可能不是真实文本；非常长 = 可能是URL或哈希值 |

### 高级过滤器（此处未实现）

生产管道使用额外的过滤器：

- **困惑度过滤**：使用小型LM对文本评分；拒绝困惑度非常低（重复）或非常高（乱码）的文档
- **基于分类器的过滤**：训练分类器区分维基百科类文本和随机网页文本（GPT-3方法）
- **PII检测**：移除包含电子邮件、电话号码、社会安全号码的文档
- **毒性过滤**：移除仇恨、冒犯或有害内容
- **启发式规则**：过度大写、过多特殊字符、重复n-gram

### 实现详解

```python
class QualityFilter:
    def _is_quality(self, text):
        words = text.split()
        n_words = len(words)

        # 检查词数边界
        if n_words < self.min_words or n_words > self.max_words:
            return False

        # 检查足够的字符是字母
        alpha_count = sum(1 for c in text if c.isalpha())
        if alpha_count / max(len(text), 1) < self.min_alpha_ratio:
            return False

        # 检查数字字符不过多
        digit_count = sum(1 for c in text if c.isdigit())
        if digit_count / max(len(text), 1) > self.max_digit_ratio:
            return False

        # 检查平均词长
        avg_word_len = sum(len(w) for w in words) / max(n_words, 1)
        if not (self.min_avg_word_len <= avg_word_len <= self.max_avg_word_len):
            return False

        return True
```

## 数据混合

### 为什么要混合数据源？

LLM从多样化的训练数据中受益。只在维基百科上训练的模型知识丰富但僵化；只在网络文本上训练的模型流畅但不可靠。关键是找到合适的混合比例。

### 领域比例

常见的混合策略：

- **按比例**：按每个数据源的大小按比例采样（大型网络语料库占主导）
- **上采样**：相对于自然频率过度代表高质量数据源（维基百科、书籍）
- **基于温度**：使用温度缩放来平滑或锐化分布

GPT-3的混合（简化版）：
- 60% 过滤后的Common Crawl
- 22% WebText2
- 16% 书籍（Books1 + Books2）
- 3% 维基百科

### 课程学习

一些研究建议按质量或难度排序数据：

1. 从更干净、更简单的文本开始
2. 逐渐引入更复杂或有噪声的数据
3. 以最高质量的数据结束

这可以提高收敛速度和最终性能，尽管证据尚不明确。

### 实现详解

```python
class DataMixer:
    def __init__(self, ratios):
        # 将比例归一化为总和为1.0
        total = sum(ratios.values())
        self.ratios = {k: v / total for k, v in ratios.items()}

    def mix(self, data, total_tokens=None, tokens_per_doc=512):
        if total_tokens is None:
            # 按可用数据的比例混合
            min_docs = min(len(docs) for docs in data.values())
            result = []
            for source, ratio in self.ratios.items():
                n = max(1, int(min_docs * ratio))
                result.extend(data[source][:n])
            return result

        # 基于token预算的混合
        result = []
        for source, ratio in self.ratios.items():
            n_docs = max(1, int((total_tokens * ratio) / tokens_per_doc))
            result.extend(data[source][:n_docs])
        return result
```

## 真实世界的数据管道

### The Pile（EleutherAI，2020）

一个为LLM训练设计的825 GiB英语文本语料库，由22个不同的子语料库组成：

- Pile-CC（过滤后的Common Crawl）
- PubMed、ArXiv、GitHub、StackExchange
- 维基百科、BookCorpus2、Project Gutenberg
- 以及更多

关键创新：通过有意的来源混合实现多样性，而不是依赖单一的大规模爬取。

### RedPajama（Together AI，2023）

LLaMA训练数据配方的开放复制：

- 1.2万亿token
- 来源：Common Crawl、C4、GitHub、维基百科、书籍、ArXiv、StackExchange
- 记录的过滤和去重管道

### FineWeb（HuggingFace，2024）

一个15万亿token的数据集，来自Common Crawl，经过广泛过滤：

- URL过滤（移除已知低质量域名）
- 使用trafilatura进行文本提取
- 使用fastText进行语言识别
- MinHash去重
- 使用小型LM进行困惑度过滤
- 重复和质量启发式

FineWeb-Edu进一步使用分类器过滤"教育性"内容，该分类器在教育网页上训练。

## 运行代码

```bash
# 运行测试
pytest tests.py -v

# 尝试练习
# 编辑exercise.py实现TODO部分，然后运行测试
pytest tests.py -v
```

## 关键要点

1. **去重至关重要**：近似重复浪费计算资源并使训练产生偏差。MinHash为大规模去重提供了高效的近似方法。

2. **质量过滤改善结果**：简单的启发式方法（词数、字符比例）可以显著提高数据集质量。生产管道使用基于分类器的过滤。

3. **混合比例很重要**：不同数据源的比例显著影响模型能力。高质量数据源应该相对于其自然频率进行上采样。

4. **管道是迭代的**：数据工程不是一次性过程。期望根据下游模型性能迭代过滤器、阈值和比例。
