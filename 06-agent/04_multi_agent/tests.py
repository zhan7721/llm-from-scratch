"""Tests for the Multi-Agent Systems implementation."""

from multi_agent import Agent, SharedMemory, Orchestrator, MultiAgentSystem


# ---------------------------------------------------------------------------
# Helper LLM functions
# ---------------------------------------------------------------------------


def make_echo_llm(prefix: str = ""):
    """Create an LLM function that echoes back the user message with a prefix."""
    def llm(messages: list[dict[str, str]]) -> str:
        for msg in reversed(messages):
            if msg["role"] == "user":
                return f"{prefix}{msg['content']}" if prefix else msg["content"]
        return prefix
    return llm


def make_constant_llm(response: str):
    """Create an LLM function that always returns the same response."""
    def llm(messages: list[dict[str, str]]) -> str:
        return response
    return llm


def make_sequential_llm(responses: list[str]):
    """Create an LLM function that returns responses in order."""
    index = [0]
    def llm(messages: list[dict[str, str]]) -> str:
        idx = index[0]
        index[0] += 1
        return responses[idx % len(responses)]
    return llm


def make_context_aware_llm():
    """Create an LLM function that includes context information in its response."""
    def llm(messages: list[dict[str, str]]) -> str:
        for msg in messages:
            if msg["role"] == "user" and "Context:" in msg.get("content", ""):
                return f"Saw context in: {msg['content'][:50]}"
        return "No context"
    return llm


# ===========================================================================
# Agent tests
# ===========================================================================


class TestAgent:
    """Tests for the Agent class."""

    def test_creation(self):
        """Should create an agent with correct attributes."""
        agent = Agent(
            name="test",
            role="assistant",
            goal="be helpful",
            llm_fn=make_constant_llm("ok"),
        )
        assert agent.name == "test"
        assert agent.role == "assistant"
        assert agent.goal == "be helpful"

    def test_step_basic(self):
        """Should call the LLM and return its response."""
        agent = Agent(
            name="worker",
            role="analyst",
            goal="analyze data",
            llm_fn=make_constant_llm("analysis complete"),
        )
        result = agent.step("analyze this")
        assert result == "analysis complete"

    def test_step_with_context(self):
        """Should include context in the user message."""
        seen_messages: list[list[dict[str, str]]] = []

        def llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            return "ok"

        agent = Agent(name="a", role="r", goal="g", llm_fn=llm)
        agent.step("task", context="some context")

        user_msg = seen_messages[0][-1]["content"]
        assert "task" in user_msg
        assert "some context" in user_msg

    def test_step_without_context(self):
        """Should not add context section when context is empty."""
        seen_messages: list[list[dict[str, str]]] = []

        def llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            return "ok"

        agent = Agent(name="a", role="r", goal="g", llm_fn=llm)
        agent.step("task")

        user_msg = seen_messages[0][-1]["content"]
        assert "task" in user_msg
        assert "Context:" not in user_msg

    def test_step_system_prompt_contains_role_and_goal(self):
        """System prompt should include the agent's role and goal."""
        seen_messages: list[list[dict[str, str]]] = []

        def llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            return "ok"

        agent = Agent(
            name="researcher",
            role="research analyst",
            goal="find facts",
            llm_fn=llm,
        )
        agent.step("test")

        system_msg = seen_messages[0][0]
        assert system_msg["role"] == "system"
        assert "research analyst" in system_msg["content"]
        assert "find facts" in system_msg["content"]

    def test_step_message_format(self):
        """Messages should follow OpenAI format with role and content."""
        seen_messages: list[list[dict[str, str]]] = []

        def llm(messages: list[dict[str, str]]) -> str:
            seen_messages.append(messages)
            return "ok"

        agent = Agent(name="a", role="r", goal="g", llm_fn=llm)
        agent.step("hello")

        msgs = seen_messages[0]
        assert len(msgs) == 2  # system + user
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        for msg in msgs:
            assert "content" in msg


# ===========================================================================
# SharedMemory tests
# ===========================================================================


class TestSharedMemory:
    """Tests for the SharedMemory class."""

    def test_store_and_retrieve(self):
        """Should store and retrieve values by key."""
        mem = SharedMemory()
        mem.store("key1", "value1")
        assert mem.retrieve("key1") == "value1"

    def test_store_overwrites(self):
        """Should overwrite existing values."""
        mem = SharedMemory()
        mem.store("key", "old")
        mem.store("key", "new")
        assert mem.retrieve("key") == "new"

    def test_retrieve_missing_key_raises(self):
        """Should raise KeyError for missing keys."""
        mem = SharedMemory()
        try:
            mem.retrieve("nonexistent")
            assert False, "Expected KeyError"
        except KeyError as e:
            assert "nonexistent" in str(e)

    def test_get_all_empty(self):
        """Should return empty dict when nothing is stored."""
        mem = SharedMemory()
        assert mem.get_all() == {}

    def test_get_all_with_data(self):
        """Should return all stored key-value pairs."""
        mem = SharedMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        mem.store("c", 3)
        result = mem.get_all()
        assert result == {"a": 1, "b": 2, "c": 3}

    def test_get_all_returns_copy(self):
        """Should return a copy, not a reference to internal data."""
        mem = SharedMemory()
        mem.store("x", 10)
        data = mem.get_all()
        data["y"] = 20
        assert "y" not in mem.get_all()

    def test_clear(self):
        """Should remove all stored values."""
        mem = SharedMemory()
        mem.store("a", 1)
        mem.store("b", 2)
        mem.clear()
        assert mem.get_all() == {}

    def test_store_various_types(self):
        """Should handle different value types."""
        mem = SharedMemory()
        mem.store("str", "hello")
        mem.store("int", 42)
        mem.store("list", [1, 2, 3])
        mem.store("dict", {"nested": True})
        mem.store("none", None)

        assert mem.retrieve("str") == "hello"
        assert mem.retrieve("int") == 42
        assert mem.retrieve("list") == [1, 2, 3]
        assert mem.retrieve("dict") == {"nested": True}
        assert mem.retrieve("none") is None


# ===========================================================================
# Orchestrator tests
# ===========================================================================


class TestOrchestrator:
    """Tests for the Orchestrator class."""

    def test_run_sequential_basic(self):
        """Should run agents in order and return last result."""
        agent_a = Agent("a", "A", "g", make_constant_llm("result_a"))
        agent_b = Agent("b", "B", "g", make_constant_llm("result_b"))
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        result = orch.run_sequential("task")
        assert result == "result_b"

    def test_run_sequential_passes_context(self):
        """Each agent should see previous agents' outputs as context."""
        llm_a = make_constant_llm("from_a")
        llm_b = make_context_aware_llm()

        agent_a = Agent("a", "A", "g", llm_a)
        agent_b = Agent("b", "B", "g", llm_b)
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        result = orch.run_sequential("my task")
        assert "from_a" in result

    def test_run_sequential_stores_results(self):
        """Should store each agent's result in shared memory."""
        agent_a = Agent("a", "A", "g", make_constant_llm("out_a"))
        agent_b = Agent("b", "B", "g", make_constant_llm("out_b"))
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        orch.run_sequential("task")
        assert memory.retrieve("sequential_a") == "out_a"
        assert memory.retrieve("sequential_b") == "out_b"

    def test_run_sequential_single_agent(self):
        """Should work with a single agent."""
        agent = Agent("solo", "S", "g", make_constant_llm("solo_result"))
        memory = SharedMemory()
        orch = Orchestrator([agent], memory)

        result = orch.run_sequential("task")
        assert result == "solo_result"

    def test_run_parallel_basic(self):
        """Should run all agents and collect results."""
        agent_a = Agent("a", "A", "g", make_constant_llm("pa"))
        agent_b = Agent("b", "B", "g", make_constant_llm("pb"))
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        results = orch.run_parallel("task")
        assert results == ["pa", "pb"]

    def test_run_parallel_agents_independent(self):
        """Parallel agents should not see each other's results."""
        seen_contents: list[str] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            for msg in messages:
                if msg["role"] == "user":
                    seen_contents.append(msg["content"])
                    return "response"
            return "response"

        agent_a = Agent("a", "A", "g", tracking_llm)
        agent_b = Agent("b", "B", "g", tracking_llm)
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        orch.run_parallel("my task")
        # Neither should have the other's context
        for content in seen_contents:
            assert "Context:" not in content

    def test_run_parallel_stores_results(self):
        """Should store each agent's result in shared memory."""
        agent_a = Agent("a", "A", "g", make_constant_llm("pa"))
        agent_b = Agent("b", "B", "g", make_constant_llm("pb"))
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        orch.run_parallel("task")
        assert memory.retrieve("parallel_a") == "pa"
        assert memory.retrieve("parallel_b") == "pb"

    def test_run_debate_basic(self):
        """Should run debate and return final answer."""
        agent_a = Agent("a", "A", "g", make_constant_llm("answer_a"))
        agent_b = Agent("b", "B", "g", make_constant_llm("answer_b"))
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        result = orch.run_debate("topic", rounds=2)
        # Last agent's response in last round wins
        assert result == "answer_b"

    def test_run_debate_stores_final(self):
        """Should store the debate result and history in memory."""
        agent = Agent("solo", "S", "g", make_constant_llm("final"))
        memory = SharedMemory()
        orch = Orchestrator([agent], memory)

        orch.run_debate("topic", rounds=1)
        assert memory.retrieve("debate_final") == "final"
        assert isinstance(memory.retrieve("debate_history"), list)

    def test_run_debate_agents_see_previous_answers(self):
        """Debate agents should see the current best answer."""
        seen_contexts: list[str] = []

        def tracking_llm(messages: list[dict[str, str]]) -> str:
            for msg in messages:
                if msg["role"] == "user" and "Context:" in msg.get("content", ""):
                    seen_contexts.append(msg["content"])
            return "response"

        agent_a = Agent("a", "A", "g", tracking_llm)
        agent_b = Agent("b", "B", "g", tracking_llm)
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        orch.run_debate("topic", rounds=2)
        # After round 1, agents should see context with current best answer
        assert len(seen_contexts) > 0

    def test_run_debate_multiple_rounds(self):
        """Should execute the correct number of rounds."""
        call_count = [0]

        def counting_llm(messages: list[dict[str, str]]) -> str:
            call_count[0] += 1
            return f"response_{call_count[0]}"

        agent = Agent("counter", "C", "g", counting_llm)
        memory = SharedMemory()
        orch = Orchestrator([agent], memory)

        orch.run_debate("topic", rounds=3)
        assert call_count[0] == 3  # 1 agent * 3 rounds

    def test_run_debate_history_grows(self):
        """Debate history should accumulate across rounds."""
        agent_a = Agent("a", "A", "g", make_constant_llm("a"))
        agent_b = Agent("b", "B", "g", make_constant_llm("b"))
        memory = SharedMemory()
        orch = Orchestrator([agent_a, agent_b], memory)

        orch.run_debate("topic", rounds=3)
        history = memory.retrieve("debate_history")
        # 2 agents * 3 rounds = 6 entries
        assert len(history) == 6


# ===========================================================================
# MultiAgentSystem tests
# ===========================================================================


class TestMultiAgentSystem:
    """Tests for the MultiAgentSystem class."""

    def test_creation_sequential(self):
        """Should create a system with sequential mode."""
        agent = Agent("a", "r", "g", make_constant_llm("ok"))
        system = MultiAgentSystem([agent], mode="sequential")
        assert system.mode == "sequential"

    def test_creation_parallel(self):
        """Should create a system with parallel mode."""
        agent = Agent("a", "r", "g", make_constant_llm("ok"))
        system = MultiAgentSystem([agent], mode="parallel")
        assert system.mode == "parallel"

    def test_creation_debate(self):
        """Should create a system with debate mode."""
        agent = Agent("a", "r", "g", make_constant_llm("ok"))
        system = MultiAgentSystem([agent], mode="debate")
        assert system.mode == "debate"

    def test_invalid_mode_raises(self):
        """Should raise ValueError for invalid mode."""
        agent = Agent("a", "r", "g", make_constant_llm("ok"))
        try:
            MultiAgentSystem([agent], mode="unknown")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "unknown" in str(e)
            assert "sequential" in str(e)

    def test_run_sequential(self):
        """Should run in sequential mode and return result."""
        agent_a = Agent("a", "A", "g", make_constant_llm("first"))
        agent_b = Agent("b", "B", "g", make_constant_llm("second"))
        system = MultiAgentSystem([agent_a, agent_b], mode="sequential")

        result = system.run("task")
        assert result == "second"

    def test_run_parallel(self):
        """Should run in parallel mode and return joined results."""
        agent_a = Agent("a", "A", "g", make_constant_llm("p1"))
        agent_b = Agent("b", "B", "g", make_constant_llm("p2"))
        system = MultiAgentSystem([agent_a, agent_b], mode="parallel")

        result = system.run("task")
        assert "p1" in result
        assert "p2" in result

    def test_run_debate(self):
        """Should run in debate mode and return refined answer."""
        agent_a = Agent("a", "A", "g", make_constant_llm("debated_a"))
        agent_b = Agent("b", "B", "g", make_constant_llm("debated_b"))
        system = MultiAgentSystem([agent_a, agent_b], mode="debate")

        result = system.run("topic")
        assert result == "debated_b"

    def test_end_to_end_sequential_flow(self):
        """Sequential flow should chain agent outputs through context."""
        results = ["step1_done", "step2_done"]

        def step_llm(messages: list[dict[str, str]]) -> str:
            for msg in messages:
                if msg["role"] == "user":
                    if "step1" not in msg["content"] and "Context:" in msg["content"]:
                        return results[1]
                    return results[0]
            return results[0]

        agent_a = Agent("step1", "first", "do step 1", step_llm)
        agent_b = Agent("step2", "second", "do step 2", step_llm)
        system = MultiAgentSystem([agent_a, agent_b], mode="sequential")

        result = system.run("do the work")
        # The second agent should see the first agent's output
        assert result in ["step1_done", "step2_done"]

    def test_system_creates_shared_memory(self):
        """System should create its own SharedMemory instance."""
        agent = Agent("a", "r", "g", make_constant_llm("ok"))
        system = MultiAgentSystem([agent])
        assert isinstance(system.memory, SharedMemory)

    def test_system_creates_orchestrator(self):
        """System should create an Orchestrator with its agents and memory."""
        agent = Agent("a", "r", "g", make_constant_llm("ok"))
        system = MultiAgentSystem([agent])
        assert isinstance(system.orchestrator, Orchestrator)
        assert system.orchestrator.agents == [agent]
        assert system.orchestrator.memory is system.memory

    def test_parallel_results_joined_by_newlines(self):
        """Parallel mode should join results with newlines."""
        responses = ["alpha", "beta", "gamma"]
        agents = [
            Agent(f"a{i}", "r", "g", make_constant_llm(r))
            for i, r in enumerate(responses)
        ]
        system = MultiAgentSystem(agents, mode="parallel")

        result = system.run("task")
        lines = result.split("\n")
        assert lines == ["alpha", "beta", "gamma"]
