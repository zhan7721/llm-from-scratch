"""Multi-Agent Systems for LLM coordination.

This module implements a multi-agent framework where multiple LLM-backed
agents collaborate to solve tasks. It contains:
- Agent: Base agent with a role, goal, and LLM backend.
- SharedMemory: Dict-based key-value store for sharing context between agents.
- Orchestrator: Coordinates agents using sequential, parallel, or debate modes.
- MultiAgentSystem: End-to-end pipeline that ties everything together.

Architecture overview:
    User task
        -> MultiAgentSystem selects coordination mode
        -> Orchestrator manages agent execution
        -> Agents call their LLM functions with context from SharedMemory
        -> Results are collected and optionally stored in SharedMemory
        -> Final answer is returned

Design notes:
- Agents are simple wrappers around LLM functions.
- SharedMemory is a plain dict-based key-value store.
- The Orchestrator manages three coordination patterns:
    * sequential: each agent sees the previous agent's output.
    * parallel: all agents run independently on the same task.
    * debate: agents take turns refining a shared answer.
- The LLM function signature is fn(messages: list[dict]) -> str.
- This module is self-contained (no external dependencies beyond the
  standard library).
"""

from typing import Any, Callable


__all__ = [
    "Agent",
    "SharedMemory",
    "Orchestrator",
    "MultiAgentSystem",
]


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class Agent:
    """An LLM-backed agent with a name, role, and goal.

    Each agent wraps an LLM function and maintains a conversation history.
    When given a task, the agent constructs a prompt that includes its role
    and goal, then calls the LLM to produce a response.

    Attributes:
        name: Unique identifier for this agent.
        role: The agent's role description (e.g. "researcher", "critic").
        goal: The agent's objective (e.g. "find factual information").
        llm_fn: A callable that takes a list of message dicts and returns
            a string response.

    Example:
        agent = Agent(
            name="researcher",
            role="research analyst",
            goal="find accurate information",
            llm_fn=my_llm_function,
        )
        result = agent.step("What is quantum computing?")
    """

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        llm_fn: Callable[[list[dict[str, str]]], str],
    ):
        """Initialize an Agent.

        Args:
            name: Unique identifier for this agent.
            role: A short description of the agent's role.
            goal: The agent's objective or purpose.
            llm_fn: A callable that takes a list of message dicts (OpenAI
                format) and returns a string (the LLM's response).
        """
        self.name = name
        self.role = role
        self.goal = goal
        self.llm_fn = llm_fn

    def step(self, task: str, context: str = "") -> str:
        """Execute one step: send the task to the LLM and return the response.

        Constructs a system prompt from the agent's role and goal, then
        builds a message list with the task and optional context.

        Args:
            task: The task or question for this agent to work on.
            context: Optional additional context (e.g. previous agent output).

        Returns:
            The LLM's response as a string.
        """
        system_content = (
            f"You are a {self.role}. "
            f"Your goal is: {self.goal}."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_content},
        ]

        user_content = task
        if context:
            user_content = f"{task}\n\nContext:\n{context}"
        messages.append({"role": "user", "content": user_content})

        return self.llm_fn(messages)


# ---------------------------------------------------------------------------
# SharedMemory
# ---------------------------------------------------------------------------


class SharedMemory:
    """A dict-based key-value store for sharing context between agents.

    Agents can store their outputs and retrieve values stored by other
    agents, enabling coordination without direct agent-to-agent messaging.

    Example:
        memory = SharedMemory()
        memory.store("research", "Quantum computing uses qubits.")
        memory.retrieve("research")  # -> "Quantum computing uses qubits."
    """

    def __init__(self) -> None:
        """Initialize an empty shared memory store."""
        self._data: dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Store a value under the given key.

        Args:
            key: The key to store the value under.
            value: The value to store (any type).
        """
        self._data[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve a value by key.

        Args:
            key: The key to look up.

        Returns:
            The stored value.

        Raises:
            KeyError: If the key does not exist.
        """
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in shared memory.")
        return self._data[key]

    def get_all(self) -> dict[str, Any]:
        """Return a copy of all stored key-value pairs.

        Returns:
            A dict containing all stored data.
        """
        return dict(self._data)

    def clear(self) -> None:
        """Remove all stored values."""
        self._data.clear()


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """Coordinates multiple agents using different execution strategies.

    The orchestrator supports three coordination patterns:
    - Sequential: agents run one after another; each sees previous results.
    - Parallel: all agents run independently on the same task.
    - Debate: agents take turns refining a shared answer over multiple rounds.

    Attributes:
        agents: The list of agents to coordinate.
        memory: Shared memory for inter-agent context.

    Example:
        agents = [agent_a, agent_b]
        memory = SharedMemory()
        orch = Orchestrator(agents, memory)
        result = orch.run_sequential("Analyze this data...")
    """

    def __init__(self, agents: list[Agent], memory: SharedMemory) -> None:
        """Initialize the orchestrator.

        Args:
            agents: List of Agent instances to coordinate.
            memory: SharedMemory instance for storing intermediate results.
        """
        self.agents = agents
        self.memory = memory

    def run_sequential(self, task: str) -> str:
        """Run agents one after another, each seeing previous results.

        Each agent receives the original task plus the output of all
        preceding agents as context. The final agent's output is returned.

        Args:
            task: The task for the agents to work on.

        Returns:
            The output of the last agent as a string.
        """
        context = ""
        result = ""
        for agent in self.agents:
            result = agent.step(task, context=context)
            self.memory.store(f"sequential_{agent.name}", result)
            if context:
                context = f"{context}\n\n[{agent.name}]: {result}"
            else:
                context = f"[{agent.name}]: {result}"
        return result

    def run_parallel(self, task: str) -> list[str]:
        """Run all agents independently on the same task.

        Each agent receives only the original task with no context from
        other agents. All results are collected and returned.

        Args:
            task: The task for the agents to work on.

        Returns:
            A list of response strings, one per agent.
        """
        results: list[str] = []
        for agent in self.agents:
            result = agent.step(task)
            self.memory.store(f"parallel_{agent.name}", result)
            results.append(result)
        return results

    def run_debate(self, task: str, rounds: int = 3) -> str:
        """Run a multi-round debate where agents refine a shared answer.

        In each round, every agent sees the current best answer and the
        full debate history, then produces a refined response. The last
        agent's response in the final round becomes the new best answer.

        Args:
            task: The topic or question for the debate.
            rounds: Number of debate rounds (default 3).

        Returns:
            The refined answer after all debate rounds.
        """
        current_answer = ""
        debate_history: list[str] = []

        for round_num in range(rounds):
            for agent in self.agents:
                context_parts: list[str] = []
                if current_answer:
                    context_parts.append(
                        f"Current best answer:\n{current_answer}"
                    )
                if debate_history:
                    context_parts.append(
                        "Debate history:\n" + "\n".join(debate_history)
                    )
                context = "\n\n".join(context_parts)

                result = agent.step(task, context=context)
                debate_history.append(
                    f"[Round {round_num + 1}, {agent.name}]: {result}"
                )
                current_answer = result

        self.memory.store("debate_final", current_answer)
        self.memory.store("debate_history", debate_history)
        return current_answer


# ---------------------------------------------------------------------------
# MultiAgentSystem
# ---------------------------------------------------------------------------


class MultiAgentSystem:
    """End-to-end multi-agent pipeline.

    Combines agents, shared memory, and an orchestrator into a single
    system that can be run with a chosen coordination mode.

    Supported modes:
        - "sequential": agents run one after another with shared context.
        - "parallel": all agents run independently on the same task.
        - "debate": agents refine an answer over multiple rounds.

    Example:
        system = MultiAgentSystem([agent_a, agent_b], mode="sequential")
        result = system.run("Analyze the pros and cons of X")
    """

    VALID_MODES = ("sequential", "parallel", "debate")

    def __init__(
        self,
        agents: list[Agent],
        mode: str = "sequential",
    ) -> None:
        """Initialize the multi-agent system.

        Args:
            agents: List of Agent instances.
            mode: Coordination mode -- "sequential", "parallel", or "debate".

        Raises:
            ValueError: If mode is not one of the valid modes.
        """
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}"
            )
        self.agents = agents
        self.mode = mode
        self.memory = SharedMemory()
        self.orchestrator = Orchestrator(agents, self.memory)

    def run(self, task: str) -> str:
        """Run the multi-agent system on the given task.

        Dispatches to the appropriate orchestrator method based on the
        configured mode.

        Args:
            task: The task for the agents to work on.

        Returns:
            The final result as a string. For parallel mode, returns the
            results joined by newlines.
        """
        if self.mode == "sequential":
            return self.orchestrator.run_sequential(task)
        elif self.mode == "parallel":
            results = self.orchestrator.run_parallel(task)
            return "\n".join(results)
        else:  # debate
            return self.orchestrator.run_debate(task)
