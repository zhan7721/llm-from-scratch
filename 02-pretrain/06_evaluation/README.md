# Evaluation

After training a language model, you need to measure how well it works.
Evaluation tells you whether your model learned what you intended, how it
compares to baselines, and whether it is ready for downstream use.

## Why Evaluation Matters

Training loss tells you how well the model fits the training data, but it
does not tell you:

- Whether the model generalizes to unseen text.
- Whether the generated output is coherent or useful.
- How the model compares to other models.

Good evaluation combines **quantitative metrics** (numbers you can track
over time) with **qualitative inspection** (reading the actual output).

## Perplexity

Perplexity is the standard metric for language models. It measures how
"surprised" the model is by the test data.

### Definition

Perplexity is the exponential of the average cross-entropy loss:

```
PPL = exp(L)
L = -(1/T) * sum_t log P(x_t | x_{<t})
```

Where T is the total number of tokens and P(x_t | x_{<t}) is the model's
predicted probability for token x_t given the preceding context.

### Interpretation

- **Lower is better.** A perplexity of 1 means the model predicts every
  token with certainty.
- **Random baseline.** For a vocabulary of size V, a uniform random model
  has perplexity V. A model with PPL=100 on a vocab of 50,000 is much
  better than random.
- **Rough intuition.** PPL=10 means the model is effectively choosing
  among 10 plausible next tokens at each position.

### Relationship to Loss

Perplexity and cross-entropy loss carry the same information. Perplexity
is just the exponential of loss, so:

- Loss 4.6 -> PPL = exp(4.6) ~ 100
- Loss 3.0 -> PPL = exp(3.0) ~ 20
- Loss 2.3 -> PPL = exp(2.3) ~ 10

People often report perplexity because it is more intuitive than raw
nats of loss.

### Practical Considerations

- **Ignore padding.** Use `ignore_index=-100` in cross-entropy to skip
  padding tokens.
- **Token-level vs. word-level.** Perplexity depends on the tokenizer.
  A model using BPE with 50K vocab and a model using word-level with
  100K vocab have incomparable perplexities.
- **Context length.** Perplexity computed with full context (e.g., 2048
  tokens) will be lower than with short context.

## Token Accuracy

Token accuracy measures how often the model's top-1 prediction matches
the true next token.

```
accuracy = correct_predictions / total_tokens
```

### When It Is Useful

- **Training diagnostics.** If accuracy is flat while loss decreases,
  the model is becoming more confident but not changing its top guess.
- **Comparing models.** Two models with similar loss can have different
  accuracy distributions.

### Limitations

- Accuracy only checks the top-1 prediction. A model that ranks the
  correct token second is scored the same as one that ranks it last.
- For large vocabularies, even good models may have low accuracy (10-30%
  is typical for 50K+ vocab models).

## Generation Quality

Perplexity and accuracy measure next-token prediction, but users interact
with generated text. Generation quality is harder to measure.

### Automatic Metrics

- **BLEU / ROUGE.** Compare generated text to reference text via n-gram
  overlap. Common in translation and summarization, less useful for open-ended
  generation.
- **Distinct-n.** Measures diversity by counting unique n-grams. Low
  distinct-n indicates repetitive generation.
- **Perplexity of generated text.** Run a separate "judge" model over the
  output. Lower perplexity from the judge suggests more fluent text.

### Human Evaluation

The gold standard. Ask humans to rate:

- **Fluency.** Is the text grammatical and natural?
- **Coherence.** Does the text make sense across sentences?
- **Factuality.** Are the claims accurate?
- **Helpfulness.** Does the text answer the prompt?

Human evaluation is expensive and slow, but it catches issues that
automatic metrics miss.

### Practical Approach

For pretraining evaluation, a common workflow is:

1. Track perplexity on a held-out validation set during training.
2. Periodically generate samples and read them manually.
3. If the model passes the "smell test" (coherent, on-topic), it is
   likely learning well.

## Benchmark Datasets

Standard datasets allow comparing models across papers.

### WikiText-103

- 100M tokens from Wikipedia.
- Word-level tokenization (originally), often adapted for BPE.
- Long-range dependency benchmark (full articles, not shuffled sentences).

### C4 (Colossal Clean Crawled Corpus)

- ~750GB of cleaned web text.
- Used by T5 and many subsequent models.
- Good proxy for general-purpose language modeling.

### The Pile

- 800GB of diverse text from 22 sub-corpora.
- Includes books, code, academic papers, StackExchange, etc.
- Tests broad knowledge, not just Wikipedia-style text.

### LAMBADA

- Tests long-range dependency by predicting the last word of a paragraph.
- Measures whether the model can use distant context.

## Evaluation During Training

### Validation Loss

The most common approach: hold out a portion of data and compute loss on
it every N steps.

```python
# During training loop
if step % eval_interval == 0:
    val_loss = evaluate(model, val_loader)
    print(f"Step {step}: val_loss={val_loss:.4f}, ppl={math.exp(val_loss):.1f}")
```

### Early Stopping

If validation loss stops improving, stop training to save compute and
prevent overfitting.

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

### Checkpoint Selection

Save checkpoints periodically and evaluate each on your benchmark suite.
Select the checkpoint with the best validation metrics, not the last one.

## Code Walkthrough

### `perplexity(model, data_loader, device)`

Iterates over a data loader, accumulates cross-entropy loss (sum reduction)
and token count, then returns `exp(total_loss / total_tokens)`. Uses
`ignore_index=-100` to skip padding positions.

### `compute_token_accuracy(model, data_loader, device)`

Gets the argmax prediction from logits, compares to labels, and masks
out positions where `labels == -100`. Returns the fraction of correct
predictions.

### `generate_samples(model, tokenizer, prompts, ...)`

Encodes each prompt, runs `model.generate()` with temperature and top-k
sampling, decodes the output, and returns a list of generated strings.

### `Evaluator`

A convenience class that bundles perplexity, accuracy, and optional
generation into a single `evaluate()` call. Returns a dict of metrics.

## Running Tests

```bash
cd 02-pretrain/06_evaluation
pytest tests.py -v
```

## Exercises

Open `exercise.py` and implement the TODO items:

1. `perplexity` -- compute loss and convert to perplexity.
2. `compute_token_accuracy` -- count correct predictions.
3. `Evaluator.evaluate` -- orchestrate metrics.

Check your work with `tests.py`.

## References

- Chen, M. et al. (2021). "Evaluating Large Language Models Trained on
  Code" (Codex/HumanEval). arXiv:2107.03374.
- Merity, S. et al. (2016). "Pointer Sentinel Mixture Models"
  (WikiText-103). arXiv:1609.07843.
- Gao, L. et al. (2020). "The Pile: An 800GB Dataset of Diverse Text for
  Language Modeling." arXiv:2101.00027.
- Raffel, C. et al. (2019). "Exploring the Limits of Transfer Learning
  with a Unified Text-to-Text Transformer" (T5/C4). arXiv:1910.10683.
