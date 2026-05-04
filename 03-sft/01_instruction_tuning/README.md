# Instruction Tuning

> **Module 03 — Supervised Fine-Tuning, Chapter 01**

A pre-trained language model knows a lot about language, but it does not know how to follow instructions. It completes text; it does not answer questions. Instruction tuning bridges that gap: by fine-tuning on (instruction, response) pairs, we teach the model to treat the instruction as context and generate a helpful response.

This chapter implements the core machinery: Alpaca-style prompt formatting, label masking so the model only learns from the response, and a loss function that respects the mask.

---

## Prerequisites

- Transformer language model basics (Module 01)
- Pre-training loop and loss computation (Module 02)
- PyTorch Dataset and DataLoader

## Files

| File | Purpose |
|------|---------|
| `instruction_tuning.py` | Core implementation: InstructionDataset, format_instruction, compute_instruction_loss |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## What is Instruction Tuning

### From Text Completion to Task Completion

A pre-trained LLM is trained on next-token prediction. Given "The capital of France is", it will produce "Paris" -- not because it understands the question, but because that is the most likely continuation in its training data.

Instruction tuning changes the objective. Instead of completing arbitrary text, the model learns to respond to structured instructions:

```
Instruction: Translate the following sentence to French.
Input: Hello, how are you?
Output: Bonjour, comment allez-vous ?
```

After instruction tuning, the model has learned a general pattern: given an instruction and optional input, produce an appropriate output. This generalizes to unseen instructions at inference time.

### Why Not Just Prompt Engineering?

Prompt engineering can coax pre-trained models into following instructions, but it is fragile. Small changes in phrasing can produce wildly different outputs. Instruction tuning makes the model robust: it reliably follows instructions across many phrasings because it has been trained to do so.

### The Alpaca Format

The Stanford Alpaca project (2023) popularized a simple three-part template:

```
### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}
```

The `### Input:` section is omitted when there is no input context. This format has three advantages:

1. **Clear delimiters** -- the `###` markers make it easy to find each section programmatically
2. **Human-readable** -- you can inspect training examples directly
3. **Widely adopted** -- many open datasets and models use this exact format

---

## Label Masking: The Key Technique

### The Problem

If we train on the full text (instruction + response), the model wastes capacity learning to predict the instruction tokens. The instruction is given at inference time -- we do not need the model to generate it. We only want the model to learn to produce the response.

### The Solution

We create two token sequences:

1. **Full sequence**: instruction + response (used as `input_ids`)
2. **Prompt only**: instruction without response (used to compute the mask length)

Then we set `labels[:prompt_length] = -100`. PyTorch's `cross_entropy` with `ignore_index=-100` skips these positions entirely. The model receives the full sequence as input (so it sees the instruction through its attention mechanism) but only computes loss on the response tokens.

```
input_ids:  [tok_1, tok_2, ..., tok_p, tok_p+1, ..., tok_n]
labels:     [-100,  -100,  ..., -100,  tok_p+1, ..., tok_n]
              ^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^
              masked (instruction)      trained (response)
```

### Why This Matters

Without label masking, instruction tuning degrades into continued pre-training. The model spends most of its gradient signal predicting instruction tokens it already knows. With label masking, 100% of the training signal goes to learning the response behavior -- this is why instruction tuning is data-efficient.

---

## Architecture

### InstructionDataset

A PyTorch Dataset that:

1. Stores a list of examples, each with `instruction`, optional `input`, and `output`
2. Formats each example into the Alpaca template
3. Tokenizes the full text and the prompt-only text
4. Creates masked labels for response-only training

```python
class InstructionDataset(Dataset):
    def __init__(self, examples, tokenizer=None, max_length=512, template="alpaca"):
        self.examples = examples
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.template = template
```

The `tokenizer` parameter is optional. If `None`, a simple character-level fallback is used (useful for testing without a real tokenizer).

### format_instruction

A standalone function that formats an instruction example into a prompt string. Supports two templates:

- **alpaca**: Uses `### Instruction:`, `### Input:`, `### Response:` with double newlines
- **simple**: Uses `Instruction:`, `Input:`, `Output:` with single newlines

```python
def format_instruction(instruction, input_text="", output="", template="alpaca"):
    if template == "alpaca":
        if input_text:
            prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
        else:
            prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
        return prompt + output
```

### compute_instruction_loss

Computes cross-entropy loss with `ignore_index=-100`, so masked positions (the instruction) do not contribute to the gradient:

```python
def compute_instruction_loss(model, batch):
    logits = model(batch["input_ids"])
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)),
        batch["labels"].view(-1),
        ignore_index=-100,
    )
    return loss
```

---

## Code Walkthrough

### Step 1: Format the Example

```python
full_text = self._format_alpaca(example)
# "### Instruction:\nSummarize\n\n### Response:\nThe summary."
```

The full text contains both the instruction and the response. This is what the model sees as input.

### Step 2: Format the Prompt Only

```python
prompt_text = self._format_prompt_only(example)
# "### Instruction:\nSummarize\n\n### Response:\n"
```

The prompt text ends at the response marker. Its length tells us where the response starts.

### Step 3: Tokenize

```python
full_ids = self.tokenizer.encode(full_text)[:self.max_length]
prompt_ids = self.tokenizer.encode(prompt_text)[:self.max_length]
```

Both are truncated to `max_length`. The prompt IDs are always a prefix of the full IDs because the prompt is a prefix of the full text.

### Step 4: Create Masked Labels

```python
input_ids = torch.tensor(full_ids, dtype=torch.long)
labels = input_ids.clone()
labels[:len(prompt_ids)] = -100
```

The labels are a copy of the input IDs, but the instruction portion is replaced with -100. The model sees the full sequence but only learns from the response.

### Step 5: Return the Batch Item

```python
return {
    "input_ids": input_ids,
    "labels": labels,
    "attention_mask": torch.ones(len(full_ids), dtype=torch.long),
}
```

The attention mask is all ones (every token is attended to). In practice, you might use padding and set mask to 0 for padded positions.

---

## Training Tips

### Learning Rate

Instruction tuning uses a lower learning rate than pre-training. Typical values:

- Full fine-tuning: 1e-5 to 5e-5
- LoRA: 1e-4 to 3e-4 (higher because fewer parameters are updated)

Too high a learning rate causes catastrophic forgetting -- the model loses its pre-trained knowledge.

### Data Quality Over Quantity

The Alpaca project showed that 52K high-quality examples can produce a strong instruction-following model. More data helps, but quality matters more. Clean, diverse instructions with accurate responses are the priority.

### Epochs

1-3 epochs is usually sufficient. More epochs risk overfitting, especially on small datasets. Monitor validation loss and stop early if it plateaus.

### Gradient Accumulation

With large models, you may need gradient accumulation to achieve an effective batch size larger than what fits in GPU memory:

```python
for i, batch in enumerate(dataloader):
    loss = compute_instruction_loss(model, batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()
```

### Mixed Precision

Use `torch.amp` for mixed-precision training to reduce memory usage and speed up computation:

```python
with torch.amp.autocast("cuda"):
    loss = compute_instruction_loss(model, batch)
```

---

## Real-World Instruction Tuning Datasets

| Dataset | Size | Source |
|---------|------|--------|
| Stanford Alpaca | 52K | GPT-4 generated |
| ShareGPT | ~90K | User-shared ChatGPT conversations |
| OpenAssistant | ~160K | Human-written conversations |
| Dolly 2.0 | 15K | Databricks employees |
| LIMA | 1K | Carefully curated (research) |

The LIMA paper ("Less Is More for Alignment", 2023) showed that just 1,000 carefully curated examples can produce a competitive model. Data quality beats data quantity.

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 03-sft/01_instruction_tuning/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 03-sft/01_instruction_tuning/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing instruction tuning yourself.

### Exercise Order

1. **`format_instruction_exercise`** -- Format an instruction example into the Alpaca template
2. **`InstructionDatasetExercise.__getitem__`** -- Tokenize, create masked labels, return the batch item
3. **`compute_instruction_loss_exercise`** -- Compute cross-entropy loss with label masking

### Tips

- Start with `format_instruction`. It is a string formatting exercise -- no tensors involved.
- For `__getitem__`, the key insight is: tokenize the full text and the prompt-only text separately, then mask the labels for the prompt portion.
- For `compute_instruction_loss`, remember that `ignore_index=-100` is the magic that skips the instruction tokens in the loss.

---

## Key Takeaways

1. **Instruction tuning teaches a model to follow instructions.** By fine-tuning on (instruction, response) pairs, the model learns to treat instructions as context and generate helpful responses.

2. **Label masking is the core technique.** Setting `labels[:prompt_length] = -100` ensures the model only learns from the response, not the instruction. This makes training efficient and focused.

3. **The Alpaca format is simple and effective.** Three sections (Instruction, Input, Response) with clear delimiters. The Input section is optional.

4. **Data quality matters more than quantity.** 1K-52K high-quality examples can produce strong instruction-following models. Focus on diverse, accurate examples.

5. **The loss function is standard cross-entropy with masking.** `F.cross_entropy(logits, labels, ignore_index=-100)` does all the work. The masking happens in the labels, not the loss function.

---

## Further Reading

- [Stanford Alpaca (Taori et al., 2023)](https://github.com/tatsu-lab/stanford_alpaca) -- The dataset and training recipe
- [LIMA: Less Is More for Alignment (Zhou et al., 2023)](https://arxiv.org/abs/2305.11206) -- 1,000 examples can be enough
- [Self-Instruct (Wang et al., 2023)](https://arxiv.org/abs/2212.10560) -- Generating instruction data from the model itself
- [Scaling Data-Constrained Language Models (Muennighoff et al., 2023)](https://arxiv.org/abs/2305.16264) -- How much data is enough
- [The Flan Collection (Longpre et al., 2023)](https://arxiv.org/abs/2301.13688) -- Large-scale instruction tuning datasets
