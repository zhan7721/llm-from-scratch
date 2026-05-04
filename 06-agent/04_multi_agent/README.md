# Multi-Agent Systems

> **Module 06 -- Agents, Chapter 04**

Multi-agent systems allow multiple LLM-backed agents to collaborate on tasks that are too complex or multifaceted for a single agent. Instead of one agent doing everything, specialized agents each handle a part of the problem, share their results, and build on each other's work. This module implements three coordination patterns: sequential pipelines, parallel execution, and multi-round debates.

---

## Prerequisites

- Python basics: functions, classes, dictionaries, type hints
- Understanding of LLM agents (Module 06, Chapters 01-03)
- Familiarity with message-based LLM interaction (OpenAI format)

## Files

| File | Purpose |
|------|---------|
| `multi_agent.py` | Core multi-agent implementation |
| `exercise.py` | Fill-in-the-blank exercises to reinforce understanding |
| `solution.py` | Reference solutions for the exercises |
| `tests.py` | pytest tests for correctness |

---

## The Big Picture

A multi-agent system coordinates several specialized agents to solve a task together:

```
User task
    |
    v
MultiAgentSystem
    |
    +---> Orchestrator (coordination strategy)
    |         |
    |         +---> Sequential: Agent A -> Agent B -> Agent C
    |         +---> Parallel:   Agent A, Agent B, Agent C (all at once)
    |         +---> Debate:     Agent A <-> Agent B <-> Agent C (iterative)
    |
    +---> SharedMemory (inter-agent context)
    |
    v
Final result
```

### Core Components

```
Agent
    +-- name: str           (unique identifier)
    +-- role: str           (e.g. "researcher", "critic")
    +-- goal: str           (e.g. "find factual information")
    +-- llm_fn: Callable    (fn(messages) -> str)

SharedMemory
    +-- store(key, value)   (save data)
    +-- retrieve(key)       (load data)
    +-- get_all()           (see everything)
    +-- clear()             (reset)

Orchestrator
    +-- run_sequential(task)  (chain agents, pass context)
    +-- run_parallel(task)    (independent agents, collect results)
    +-- run_debate(task)      (iterative refinement over rounds)

MultiAgentSystem
    +-- mode: str           ("sequential", "parallel", "debate")
    +-- run(task)           (execute with chosen mode)
```

### Key Insight: Coordination Over Complexity

Each agent is simple -- it just calls an LLM with a prompt. The intelligence comes from how agents are coordinated: what context they see, when they run, and how their outputs are combined.

---

## Architecture Details

### Agent

An agent is a thin wrapper around an LLM function. It has an identity (name, role, goal) that gets injected into the system prompt, and a `step()` method that calls the LLM with the task and optional context.

```python
agent = Agent(
    name="researcher",
    role="research analyst",
    goal="find accurate information",
    llm_fn=my_llm_function,
)
result = agent.step("What is quantum computing?")
```

### SharedMemory

A simple dict-based key-value store. Agents write their outputs here so other agents (or the orchestrator) can access them later. This enables loose coupling between agents.

```python
memory = SharedMemory()
memory.store("research", "Quantum computing uses qubits.")
memory.retrieve("research")  # -> "Quantum computing uses qubits."
```

### Orchestrator

The orchestrator implements three coordination strategies:

**Sequential** -- agents run one after another. Each agent receives the original task plus all previous agents' outputs as context. This is useful for pipelines where later stages build on earlier ones (e.g., research -> analysis -> summary).

**Parallel** -- all agents run independently on the same task. No agent sees the others' outputs. Results are collected into a list. This is useful for getting diverse perspectives (e.g., multiple independent analyses).

**Debate** -- agents take turns refining a shared answer over multiple rounds. Each agent sees the current best answer and the full debate history. The last agent's response becomes the new best answer. This is useful for iterative refinement and error correction.

```python
orch = Orchestrator(agents, memory)
result = orch.run_sequential("task")  # or run_parallel, run_debate
```

### MultiAgentSystem

The top-level class that ties everything together. It creates the shared memory and orchestrator internally, and dispatches to the correct coordination method based on the configured mode.

```python
system = MultiAgentSystem([agent_a, agent_b], mode="debate")
result = system.run("Should we use microservices?")
```

---

## Forward Pass Walkthrough

### Sequential Mode

```
Task: "Analyze the impact of AI on healthcare"

Agent 1 (researcher):
    Input:  task (no context)
    Output: "AI improves diagnostics, enables personalized medicine..."

Agent 2 (analyst):
    Input:  task + "[researcher]: AI improves diagnostics..."
    Output: "Key benefits: faster diagnosis, reduced costs..."

Agent 3 (critic):
    Input:  task + "[researcher]: ... \n[analyst]: ..."
    Output: "Concerns: data privacy, bias in training data..."

Final output: Agent 3's response
```

### Parallel Mode

```
Task: "Evaluate this business proposal"

Agent 1 (optimist):  "Strong market opportunity..."
Agent 2 (pessimist): "High risk factors..."
Agent 3 (analyst):    "Neutral assessment..."

Output: ["Strong market...", "High risk...", "Neutral..."]
```

### Debate Mode

```
Task: "Is remote work better than office work?"

Round 1:
    Agent A: "Remote work is better because..."
    Agent B: "Office work is better because..."
    Current answer: Agent B's response

Round 2:
    Agent A sees current answer + history, refines argument
    Agent B sees updated answer + history, refines argument
    Current answer: Agent B's updated response

Round 3:
    Same pattern, further refinement

Final output: The refined answer after all rounds
```

---

## How to Run

### Run Tests

```bash
cd /path/to/llm-from-scratch
pytest 06-agent/04_multi_agent/tests.py -v
```

### Run Exercises

Open `exercise.py` and fill in the `TODO` sections. Then verify:

```bash
pytest 06-agent/04_multi_agent/tests.py -v
```

---

## Exercises

Open `exercise.py` to practice implementing the multi-agent system yourself.

### Exercise Order

1. **Agent**: Implement creation and the `step()` method
2. **SharedMemory**: Implement store, retrieve, get_all, clear
3. **Orchestrator**: Implement sequential, parallel, and debate modes
4. **MultiAgentSystem**: Implement mode validation and dispatch

### Tips

- For Agent, always build a system prompt with role and goal before the user message.
- For SharedMemory, `retrieve()` should raise `KeyError` for missing keys, and `get_all()` should return a copy.
- In sequential mode, accumulate context as a string with each agent's output labeled.
- In debate mode, track both the current best answer and the full debate history.
- The LLM function signature is `fn(messages: list[dict]) -> str`, where messages follow OpenAI format.

---

## Key Takeaways

1. **Agents are wrappers, not monoliths.** An agent is just an LLM function plus an identity. The real power comes from coordination.

2. **Coordination patterns matter.** Sequential is for pipelines, parallel is for diversity, and debate is for refinement. Choosing the right pattern depends on the task.

3. **Shared memory enables loose coupling.** Agents don't talk to each other directly -- they read and write to shared memory. This makes the system easy to extend.

4. **Context is everything.** What an agent sees determines what it produces. Sequential mode gives rich context; parallel mode gives none; debate mode gives iterative context.

5. **Debate improves quality.** By having agents challenge and refine each other's answers, debate mode can catch errors and produce more balanced results than any single agent.

---

## Further Reading

- [AutoGen: Multi-Agent Conversation Framework](https://github.com/microsoft/autogen) -- Microsoft's multi-agent framework
- [CrewAI: Framework for Orchestrating Role-Playing Agents](https://github.com/crewAIInc/crewAI) -- Role-based multi-agent coordination
- [Multi-Agent Debate Improves LLM Reasoning (Liang et al., 2023)](https://arxiv.org/abs/2305.14325) -- Research on debate-based coordination
- [ReAct: Reasoning and Acting (Yao et al., 2022)](https://arxiv.org/abs/2210.03629) -- Foundation for agent reasoning loops
