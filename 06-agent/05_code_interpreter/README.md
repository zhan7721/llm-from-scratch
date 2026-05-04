# Code Interpreter

> **Module 06 -- Agents, Chapter 05**

A code interpreter allows an LLM to generate Python code, execute it in a sandboxed environment, and feed the results back for iterative refinement. This is the pattern behind tools like ChatGPT's Code Interpreter and Claude's analysis tool -- the LLM writes code, a runtime executes it, and the output informs the next step.

---

## Prerequisites

- Python basics: functions, classes, exec(), threading
- Understanding of LLM agents (Module 06, Chapters 01-03)
- Familiarity with message-based LLM interaction (OpenAI format)

## Files

| File | Purpose |
|------|---------|
| `code_interpreter.py` | Core code interpreter implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A code interpreter creates a generate-execute loop between an LLM and a Python runtime:

```
User task
    |
    v
CodeInterpreterAgent
    |
    +---> LLM generates response with ```python code blocks
    |
    +---> CodeParser extracts code blocks
    |
    +---> SandboxExecutor runs code (isolated namespace, timeout)
    |
    +---> Results fed back to LLM
    |
    +---> Repeat until no code or max_turns reached
    |
    v
Final answer
```

### Core Components

```
SandboxExecutor
    +-- execute(code, timeout)   (run code, return output/error/success)
    +-- namespace                (persistent across calls)

CodeParser
    +-- parse(text)              (extract ```python blocks)

CodeInterpreterAgent
    +-- run(task)                (generate-execute loop)
    +-- sandbox: SandboxExecutor
    +-- parser: CodeParser

ArtifactStore
    +-- store(name, data, type)  (save artifact)
    +-- retrieve(name)           (load artifact)
    +-- list_artifacts()         (see all names)
    +-- clear()                  (reset)
```

### Key Insight: The Generate-Execute Loop

The power of a code interpreter comes from iteration. The LLM does not need to get the code right on the first try -- it can see execution results, debug errors, and refine its approach. This mirrors how humans write code: try something, see the output, adjust.

---

## Architecture Details

### SandboxExecutor

Executes Python code in a persistent, isolated namespace. Uses `exec()` with a restricted globals dict (builtins only). Captures stdout and stderr separately. Enforces a timeout using `threading.Thread` with `join(timeout=...)`.

```python
executor = SandboxExecutor()
result = executor.execute("x = 42\nprint(x)")
# result == {"output": "42\n", "error": "", "success": True}

result = executor.execute("print(x + 1)")  # x persists!
# result == {"output": "43\n", "error": "", "success": True}
```

### CodeParser

Extracts Python code from LLM markdown output using a regex pattern. Supports `python`, `py`, and `python3` language tags (case-insensitive). Returns a list of code strings.

```python
parser = CodeParser()
blocks = parser.parse("```python\nprint('hi')\n```")
# blocks == ["print('hi')"]
```

### CodeInterpreterAgent

Ties everything together. Takes an LLM function and a max_turns limit. Runs the generate-execute loop: call the LLM, parse code blocks, execute them, feed results back, repeat.

```python
agent = CodeInterpreterAgent(llm_fn=my_llm, max_turns=5)
result = agent.run("Calculate the first 10 Fibonacci numbers")
```

### ArtifactStore

A simple key-value store for artifacts produced during code interpretation. Artifacts can be any Python object -- text, data, images, models.

```python
store = ArtifactStore()
store.store("result", [1, 2, 3], artifact_type="data")
store.retrieve("result")  # -> [1, 2, 3]
```

---

## Forward Pass Walkthrough

### Single Turn

```
Task: "What is 2 + 2?"

LLM: "Let me calculate:
      ```python
      print(2 + 2)
      ```"

Parser: extracts "print(2 + 2)"

Executor: runs code -> output: "4\n"

LLM: "The result is 4."

No code blocks -> return "The result is 4."
```

### Multi-Turn with Debugging

```
Task: "Calculate factorial of 10"

Turn 1:
    LLM: "```python
          def fact(n):
              return n * fact(n-1) if n > 1 else 1
          print(fact(10))
          ```"
    Executor: RecursionError!
    Results fed back: "Error: RecursionError: ..."

Turn 2:
    LLM: "```python
          def fact(n):
              r = 1
              for i in range(2, n+1):
                  r *= i
              return r
          print(fact(10))
          ```"
    Executor: output: "3628800\n"

Turn 3:
    LLM: "The factorial of 10 is 3,628,800."
    No code blocks -> return answer
```

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/05_code_interpreter/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 06-agent/05_code_interpreter/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the code interpreter yourself.

### Exercise Order

1. **SandboxExecutor**: Implement namespace initialization and code execution with timeout
2. **CodeParser**: Implement regex pattern and code block extraction
3. **CodeInterpreterAgent**: Implement the generate-execute loop
4. **ArtifactStore**: Implement store, retrieve, list, and clear

### Tips

- For SandboxExecutor, use a daemon thread so it does not block the main process on timeout.
- For CodeParser, use `re.DOTALL` so the `.` in the regex matches newlines.
- For CodeInterpreterAgent, build the messages list incrementally -- append the assistant response and the execution results as separate messages.
- The LLM function signature is `fn(messages: list[dict]) -> str`, where messages follow OpenAI format.

---

## Key Takeaways

1. **Code execution extends LLM capabilities.** By generating and running code, LLMs can perform precise calculations, data analysis, and visualization that pure text generation cannot match.

2. **The generate-execute loop is iterative.** The LLM does not need to be perfect on the first try. Seeing execution results (including errors) allows it to self-correct, much like a human programmer.

3. **Sandboxing matters.** Restricting the execution namespace and enforcing timeouts prevents runaway code and limits the blast radius of errors. Production systems add additional layers (containers, resource limits).

4. **Persistent state enables multi-step workflows.** Variables that survive across executions let the LLM build up complex analyses incrementally -- define helpers in one turn, use them in the next.

5. **Parsing is the bridge.** The code parser converts unstructured LLM text into executable code. Supporting multiple language tags and handling edge cases (no blocks, multiple blocks) makes the system robust.

---

## Further Reading

- [OpenAI Code Interpreter](https://openai.com/index/chatgpt-can-now-see-hear-and-speak/) -- OpenAI's code execution capability
- [Open Interpreter](https://github.com/OpenInterpreter/open-interpreter) -- Open-source code interpreter for LLMs
- [E2B: Code Interpreting for AI](https://e2b.dev/) -- Sandboxed cloud environments for AI code execution
- [Exec in Python](https://docs.python.org/3/library/functions.html#exec) -- Python documentation for exec()
