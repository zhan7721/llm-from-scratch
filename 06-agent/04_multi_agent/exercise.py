"""Exercise: Implement Multi-Agent Systems for LLM coordination.

Complete the TODOs below to build a multi-agent framework where multiple
LLM-backed agents collaborate to solve tasks using different coordination
strategies.
Run `pytest tests.py` to verify your implementation.
"""

from typing import Any, Callable


class AgentExercise:
    """An LLM-backed agent with a name, role, and goal.

    TODO: Implement all methods.

    Each agent wraps an LLM function and constructs prompts that include
    its role and goal before calling the LLM.

    Example:
        agent = AgentExercise(
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
                format) and returns a string.
        """
        # TODO 1: Store name, role, goal, and llm_fn
        pass  # YOUR CODE HERE

    def step(self, task: str, context: str = "") -> str:
        """Execute one step: send the task to the LLM and return the response.

        Args:
            task: The task or question for this agent to work on.
            context: Optional additional context (e.g. previous agent output).

        Returns:
            The LLM's response as a string.
        """
        # TODO 2: Build a system prompt from role and goal
        # Hint: "You are a {role}. Your goal is: {goal}."
        # TODO 3: Build a user message with the task and optional context
        # Hint: if context is non-empty, append it after the task
        # TODO 4: Call self.llm_fn with the messages list and return the result
        return ""  # YOUR CODE HERE


class SharedMemoryExercise:
    """A dict-based key-value store for sharing context between agents.

    TODO: Implement all methods.

    Example:
        memory = SharedMemoryExercise()
        memory.store("research", "Quantum computing uses qubits.")
        memory.retrieve("research")  # -> "Quantum computing uses qubits."
    """

    def __init__(self) -> None:
        """Initialize an empty shared memory store."""
        # TODO 5: Initialize internal storage
        pass  # YOUR CODE HERE

    def store(self, key: str, value: Any) -> None:
        """Store a value under the given key.

        Args:
            key: The key to store the value under.
            value: The value to store (any type).
        """
        # TODO 6: Store value in internal dict
        pass  # YOUR CODE HERE

    def retrieve(self, key: str) -> Any:
        """Retrieve a value by key.

        Args:
            key: The key to look up.

        Returns:
            The stored value.

        Raises:
            KeyError: If the key does not exist.
        """
        # TODO 7: Look up key, raise KeyError if not found
        return None  # YOUR CODE HERE

    def get_all(self) -> dict[str, Any]:
        """Return a copy of all stored key-value pairs.

        Returns:
            A dict containing all stored data.
        """
        # TODO 8: Return a copy of the internal dict
        return {}  # YOUR CODE HERE

    def clear(self) -> None:
        """Remove all stored values."""
        # TODO 9: Clear the internal dict
        pass  # YOUR CODE HERE


class OrchestratorExercise:
    """Coordinates multiple agents using different execution strategies.

    TODO: Implement all methods.

    Supports three coordination patterns:
    - Sequential: agents run one after another with shared context.
    - Parallel: all agents run independently on the same task.
    - Debate: agents refine an answer over multiple rounds.

    Example:
        orch = OrchestratorExercise(agents, memory)
        result = orch.run_sequential("Analyze this data...")
    """

    def __init__(
        self,
        agents: list["AgentExercise"],
        memory: "SharedMemoryExercise",
    ) -> None:
        """Initialize the orchestrator.

        Args:
            agents: List of AgentExercise instances to coordinate.
            memory: SharedMemoryExercise instance for intermediate results.
        """
        # TODO 10: Store agents and memory
        pass  # YOUR CODE HERE

    def run_sequential(self, task: str) -> str:
        """Run agents one after another, each seeing previous results.

        Args:
            task: The task for the agents to work on.

        Returns:
            The output of the last agent.
        """
        # TODO 11: Loop through agents, passing accumulated context
        # TODO 12: Store each result in memory with key "sequential_{name}"
        return ""  # YOUR CODE HERE

    def run_parallel(self, task: str) -> list[str]:
        """Run all agents independently on the same task.

        Args:
            task: The task for the agents to work on.

        Returns:
            A list of response strings, one per agent.
        """
        # TODO 13: Loop through agents, each gets only the task (no context)
        # TODO 14: Store each result in memory with key "parallel_{name}"
        return []  # YOUR CODE HERE

    def run_debate(self, task: str, rounds: int = 3) -> str:
        """Run a multi-round debate where agents refine a shared answer.

        Args:
            task: The topic or question for the debate.
            rounds: Number of debate rounds (default 3).

        Returns:
            The refined answer after all debate rounds.
        """
        # TODO 15: For each round, loop through agents
        # TODO 16: Build context from current best answer and debate history
        # TODO 17: Store final answer and history in memory
        return ""  # YOUR CODE HERE


class MultiAgentSystemExercise:
    """End-to-end multi-agent pipeline.

    TODO: Implement all methods.

    Combines agents, shared memory, and an orchestrator into a single
    system that can be run with a chosen coordination mode.

    Example:
        system = MultiAgentSystemExercise([agent_a, agent_b], mode="sequential")
        result = system.run("Analyze the pros and cons of X")
    """

    VALID_MODES = ("sequential", "parallel", "debate")

    def __init__(
        self,
        agents: list["AgentExercise"],
        mode: str = "sequential",
    ) -> None:
        """Initialize the multi-agent system.

        Args:
            agents: List of AgentExercise instances.
            mode: "sequential", "parallel", or "debate".

        Raises:
            ValueError: If mode is not one of the valid modes.
        """
        # TODO 18: Validate mode
        # TODO 19: Create SharedMemoryExercise and OrchestratorExercise
        pass  # YOUR CODE HERE

    def run(self, task: str) -> str:
        """Run the multi-agent system on the given task.

        Args:
            task: The task for the agents to work on.

        Returns:
            The final result as a string.
        """
        # TODO 20: Dispatch to the correct orchestrator method based on mode
        # Hint: for parallel mode, join results with newlines
        return ""  # YOUR CODE HERE
