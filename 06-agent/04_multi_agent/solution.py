"""Reference solution for the Multi-Agent Systems exercise."""

from typing import Any, Callable


class AgentSolution:
    """An LLM-backed agent with a name, role, and goal."""

    def __init__(
        self,
        name: str,
        role: str,
        goal: str,
        llm_fn: Callable[[list[dict[str, str]]], str],
    ):
        """Initialize an Agent."""
        self.name = name
        self.role = role
        self.goal = goal
        self.llm_fn = llm_fn

    def step(self, task: str, context: str = "") -> str:
        """Execute one step: send the task to the LLM and return the response."""
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


class SharedMemorySolution:
    """A dict-based key-value store for sharing context between agents."""

    def __init__(self) -> None:
        """Initialize an empty shared memory store."""
        self._data: dict[str, Any] = {}

    def store(self, key: str, value: Any) -> None:
        """Store a value under the given key."""
        self._data[key] = value

    def retrieve(self, key: str) -> Any:
        """Retrieve a value by key."""
        if key not in self._data:
            raise KeyError(f"Key '{key}' not found in shared memory.")
        return self._data[key]

    def get_all(self) -> dict[str, Any]:
        """Return a copy of all stored key-value pairs."""
        return dict(self._data)

    def clear(self) -> None:
        """Remove all stored values."""
        self._data.clear()


class OrchestratorSolution:
    """Coordinates multiple agents using different execution strategies."""

    def __init__(
        self,
        agents: list["AgentSolution"],
        memory: "SharedMemorySolution",
    ) -> None:
        """Initialize the orchestrator."""
        self.agents = agents
        self.memory = memory

    def run_sequential(self, task: str) -> str:
        """Run agents one after another, each seeing previous results."""
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
        """Run all agents independently on the same task."""
        results: list[str] = []
        for agent in self.agents:
            result = agent.step(task)
            self.memory.store(f"parallel_{agent.name}", result)
            results.append(result)
        return results

    def run_debate(self, task: str, rounds: int = 3) -> str:
        """Run a multi-round debate where agents refine a shared answer."""
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


class MultiAgentSystemSolution:
    """End-to-end multi-agent pipeline."""

    VALID_MODES = ("sequential", "parallel", "debate")

    def __init__(
        self,
        agents: list["AgentSolution"],
        mode: str = "sequential",
    ) -> None:
        """Initialize the multi-agent system."""
        if mode not in self.VALID_MODES:
            raise ValueError(
                f"Invalid mode '{mode}'. Must be one of: {self.VALID_MODES}"
            )
        self.agents = agents
        self.mode = mode
        self.memory = SharedMemorySolution()
        self.orchestrator = OrchestratorSolution(agents, self.memory)

    def run(self, task: str) -> str:
        """Run the multi-agent system on the given task."""
        if self.mode == "sequential":
            return self.orchestrator.run_sequential(task)
        elif self.mode == "parallel":
            results = self.orchestrator.run_parallel(task)
            return "\n".join(results)
        else:  # debate
            return self.orchestrator.run_debate(task)
