# Chain-of-Thought Reasoning

> **Module 06 -- Agents, Chapter 02**

Chain-of-thought (CoT) prompting encourages LLMs to reason step by step before producing a final answer. Instead of jumping straight to a conclusion, the model shows its work, which dramatically improves accuracy on multi-step problems.

---

## Prerequisites

- Python basics: functions, strings, lists, dictionaries
- Understanding of how LLMs generate text (Module 01)
- Basic familiarity with prompting techniques

## Files

| File | Purpose |
|------|---------|
| `reasoning.py` | Core CoT reasoning implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A chain-of-thought reasoning system works as follows:

```
User question
    |
    v
ChainOfThoughtPrompter adds "think step by step" instructions
    |
    v
LLM generates a reasoning trace
    |
    v
StepExtractor splits trace into discrete steps
    |
    v
ReasoningVerifier checks the chain
    |
    v
Final answer extracted
```

For robustness, SelfConsistency repeats this multiple times:

```
Question
    |
    +---> Path 1 --> Answer A
    |
    +---> Path 2 --> Answer A
    |
    +---> Path 3 --> Answer B
    |
    +---> Path 4 --> Answer A
    |
    +---> Path 5 --> Answer B
    |
    v
Majority vote --> Answer A
```

### Key Insight: Showing Work Improves Accuracy

When an LLM reasons step by step, each intermediate step is generated conditioned on all previous steps. This autoregressive chain breaks complex problems into simpler sub-problems, reducing errors at each stage.

---

## Architecture Details

### ChainOfThoughtPrompter

Formats questions with CoT instructions. Supports three styles:

- **step_by_step**: "Let's think step by step." (classic Kojima et al., 2022)
- **think_aloud**: "Think out loud, showing your reasoning process."
- **structured**: "Break this down into numbered steps and show your work."

Also supports few-shot CoT with worked examples.

```python
prompter = ChainOfThoughtPrompter()
prompt = prompter.format_prompt("What is 15 * 23?")
# -> "What is 15 * 23?\n\nLet's think step by step."
```

### StepExtractor

Parses LLM reasoning output into individual steps. Handles three formats:

1. **Numbered**: "1. First step  2. Second step"
2. **Bullet points**: "- First step  - Second step"
3. **Paragraph breaks**: Steps separated by blank lines

Tries numbered first, then bullets, then paragraphs.

```python
extractor = StepExtractor()
steps = extractor.extract_steps("1. Multiply 15*20=300\n2. Multiply 15*3=45\n3. Add 300+45=345")
# -> ["Multiply 15*20=300", "Multiply 15*3=45", "Add 300+45=345"]
```

### SelfConsistency

Implements the self-consistency technique (Wang et al., 2022):

1. Generates N reasoning paths (with temperature > 0 for diversity)
2. Extracts the final answer from each path
3. Returns the most common answer via majority voting

```python
def my_llm(prompt: str) -> str:
    return "Step 1: ... The answer is 42."

sc = SelfConsistency(my_llm, num_paths=5)
answer = sc.solve("What is 6 * 7?")
```

### ReasoningVerifier

Checks reasoning chains for issues using heuristic rules:

- Empty or missing steps
- Steps that are too short to be meaningful
- Contradictory steps (steps that negate earlier ones)
- Answer not supported by the reasoning

```python
verifier = ReasoningVerifier()
result = verifier.verify_chain(["Step 1...", "Step 2...", "Step 3..."])
# -> {"is_valid": True, "issues": []}
```

---

## Forward Pass Walkthrough

### Simple CoT

```
Input:   "What is 15 * 23?"
Prompt:  "What is 15 * 23?\n\nLet's think step by step."
LLM:     "1. 15 * 20 = 300\n2. 15 * 3 = 45\n3. 300 + 45 = 345\nThe answer is 345."
Steps:   ["15 * 20 = 300", "15 * 3 = 45", "300 + 45 = 345"]
Answer:  "345"
```

### Self-Consistency

```
Path 1:  "15 * 20 = 300, 15 * 3 = 45, 300 + 45 = 345. Answer: 345."
Path 2:  "15 * 23 = 345. Answer: 345."
Path 3:  "15 * 20 = 300... 300 + 45 = 345. Answer: 345."
Path 4:  "Let me compute 15 * 23... = 345. Answer: 345."
Path 5:  "15 * 23 = 345. Answer: 345."

Majority: "345" (5/5 paths agree)
```

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/02_reasoning/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 06-agent/02_reasoning/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing chain-of-thought reasoning yourself.

### Exercise Order

1. **ChainOfThoughtPrompter**: Format prompts with CoT instructions, implement few-shot examples
2. **StepExtractor**: Parse numbered steps, bullets, and paragraphs
3. **SelfConsistency**: Implement answer extraction and majority voting
4. **ReasoningVerifier**: Check for empty steps, contradictions, and answer support

### Tips

- For answer extraction, search from the end of the text to get the last occurrence.
- Use `text.lower().rfind(pattern)` for case-insensitive searching from the end.
- For majority voting, normalize (lowercase, strip) before counting, but return the original casing.
- The verifier should use simple heuristics, not call another LLM.

---

## Key Takeaways

1. **CoT prompting is a simple but powerful technique.** Adding "Let's think step by step" to a prompt can dramatically improve reasoning accuracy with zero training.

2. **Self-consistency improves reliability.** By sampling multiple reasoning paths and voting, we reduce the impact of any single reasoning error.

3. **Step extraction enables verification.** Breaking a reasoning trace into discrete steps allows us to check each step independently.

4. **Heuristic verification catches obvious errors.** While not perfect, checking for empty steps, contradictions, and unsupported answers catches many common failure modes.

5. **The answer extraction pattern matters.** LLMs often signal their final answer with phrases like "The answer is" or "Therefore". Recognizing these patterns is key to reliable self-consistency.

---

## Further Reading

- [Chain-of-Thought Prompting (Wei et al., 2022)](https://arxiv.org/abs/2201.11903) -- The original CoT paper
- [Self-Consistency (Wang et al., 2022)](https://arxiv.org/abs/2203.11171) -- Sampling multiple reasoning paths
- [Large Language Models are Zero-Shot Reasoners (Kojima et al., 2022)](https://arxiv.org/abs/2205.11916) -- "Let's think step by step"
- [Tree of Thoughts (Yao et al., 2023)](https://arxiv.org/abs/2305.10601) -- Extending CoT with tree search
