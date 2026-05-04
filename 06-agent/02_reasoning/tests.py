"""Tests for the Chain-of-Thought Reasoning implementation."""

from reasoning import (
    ChainOfThoughtPrompter,
    StepExtractor,
    SelfConsistency,
    ReasoningVerifier,
)


# ===========================================================================
# ChainOfThoughtPrompter tests
# ===========================================================================


class TestChainOfThoughtPrompter:
    """Tests for the ChainOfThoughtPrompter class."""

    def test_format_prompt_default_style(self):
        """Default style should append 'Let's think step by step.'."""
        prompter = ChainOfThoughtPrompter()
        result = prompter.format_prompt("What is 2 + 2?")

        assert "What is 2 + 2?" in result
        assert "Let's think step by step." in result

    def test_format_prompt_step_by_step(self):
        """step_by_step style should use the classic CoT prompt."""
        prompter = ChainOfThoughtPrompter()
        result = prompter.format_prompt("What is 5 * 6?", style="step_by_step")

        assert "What is 5 * 6?" in result
        assert "Let's think step by step." in result

    def test_format_prompt_think_aloud(self):
        """think_aloud style should use think-aloud instructions."""
        prompter = ChainOfThoughtPrompter()
        result = prompter.format_prompt("What is 5 * 6?", style="think_aloud")

        assert "What is 5 * 6?" in result
        assert "Think out loud" in result

    def test_format_prompt_structured(self):
        """structured style should ask for numbered steps."""
        prompter = ChainOfThoughtPrompter()
        result = prompter.format_prompt("What is 5 * 6?", style="structured")

        assert "What is 5 * 6?" in result
        assert "numbered steps" in result

    def test_format_prompt_invalid_style(self):
        """Unknown style should raise ValueError."""
        prompter = ChainOfThoughtPrompter()

        try:
            prompter.format_prompt("test", style="nonexistent")
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "nonexistent" in str(e)

    def test_format_prompt_preserves_question(self):
        """The original question should appear unchanged in the output."""
        prompter = ChainOfThoughtPrompter()
        question = "If a train travels 60 mph for 2.5 hours, how far does it go?"
        result = prompter.format_prompt(question)

        assert question in result

    def test_format_prompt_with_examples_single(self):
        """Should format a single example correctly."""
        prompter = ChainOfThoughtPrompter()
        examples = [
            {"question": "What is 2 + 2?", "answer": "2 + 2 = 4. The answer is 4."},
        ]
        result = prompter.format_prompt_with_examples(
            "What is 3 + 3?", examples=examples
        )

        assert "Example 1" in result
        assert "What is 2 + 2?" in result
        assert "2 + 2 = 4" in result
        assert "What is 3 + 3?" in result
        assert "Let's think step by step." in result

    def test_format_prompt_with_examples_multiple(self):
        """Should format multiple examples correctly."""
        prompter = ChainOfThoughtPrompter()
        examples = [
            {"question": "What is 1 + 1?", "answer": "1 + 1 = 2."},
            {"question": "What is 2 + 2?", "answer": "2 + 2 = 4."},
        ]
        result = prompter.format_prompt_with_examples(
            "What is 3 + 3?", examples=examples
        )

        assert "Example 1" in result
        assert "Example 2" in result
        assert "What is 1 + 1?" in result
        assert "What is 2 + 2?" in result
        assert "Now solve this:" in result

    def test_format_prompt_with_examples_and_style(self):
        """Should respect the style parameter for the final instruction."""
        prompter = ChainOfThoughtPrompter()
        examples = [{"question": "Q1", "answer": "A1"}]
        result = prompter.format_prompt_with_examples(
            "Q2", examples=examples, style="structured"
        )

        assert "numbered steps" in result

    def test_format_prompt_with_empty_examples(self):
        """Should work with no examples (just question + instruction)."""
        prompter = ChainOfThoughtPrompter()
        result = prompter.format_prompt_with_examples("What is 1+1?", examples=[])

        assert "What is 1+1?" in result
        assert "Let's think step by step." in result
        assert "Example" not in result

    def test_format_prompt_question_comes_after_examples(self):
        """The actual question should appear after the examples."""
        prompter = ChainOfThoughtPrompter()
        examples = [{"question": "Q1", "answer": "A1"}]
        result = prompter.format_prompt_with_examples("MyQuestion", examples=examples)

        example_pos = result.find("Example 1")
        question_pos = result.find("MyQuestion")
        assert example_pos < question_pos


# ===========================================================================
# StepExtractor tests
# ===========================================================================


class TestStepExtractor:
    """Tests for the StepExtractor class."""

    def test_extract_numbered_steps(self):
        """Should extract steps from numbered list format."""
        text = (
            "1. First, we calculate 15 * 20 = 300\n"
            "2. Then, we calculate 15 * 3 = 45\n"
            "3. Finally, we add 300 + 45 = 345"
        )
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 3
        assert "15 * 20 = 300" in steps[0]
        assert "15 * 3 = 45" in steps[1]
        assert "300 + 45 = 345" in steps[2]

    def test_extract_numbered_steps_parentheses(self):
        """Should extract steps with '1) 2) 3)' format."""
        text = "1) Step one\n2) Step two\n3) Step three"
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 3
        assert steps[0] == "Step one"
        assert steps[1] == "Step two"
        assert steps[2] == "Step three"

    def test_extract_bullet_dashes(self):
        """Should extract steps from dash bullet points."""
        text = "- First point\n- Second point\n- Third point"
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 3
        assert steps[0] == "First point"
        assert steps[1] == "Second point"
        assert steps[2] == "Third point"

    def test_extract_bullet_asterisks(self):
        """Should extract steps from asterisk bullet points."""
        text = "* First point\n* Second point"
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 2
        assert steps[0] == "First point"
        assert steps[1] == "Second point"

    def test_extract_paragraph_breaks(self):
        """Should extract steps from paragraph-separated text."""
        text = "First step is here.\n\nSecond step is here.\n\nThird step."
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 3
        assert steps[0] == "First step is here."
        assert steps[1] == "Second step is here."
        assert steps[2] == "Third step."

    def test_extract_empty_text(self):
        """Empty text should return empty list."""
        assert StepExtractor.extract_steps("") == []
        assert StepExtractor.extract_steps("   ") == []

    def test_extract_none_like(self):
        """Empty/whitespace text should return empty list."""
        assert StepExtractor.extract_steps("\n\n\n") == []

    def test_extract_single_step(self):
        """Single step should return a list with one element."""
        steps = StepExtractor.extract_steps("Just one step.")
        assert len(steps) == 1
        assert steps[0] == "Just one step."

    def test_extract_numbered_skips_non_numbered_lines(self):
        """Numbered extraction should skip lines that don't match the pattern."""
        text = "Here is my reasoning:\n1. Step one\nSome note\n2. Step two"
        steps = StepExtractor.extract_steps(text)

        # Should extract only the numbered steps
        assert len(steps) == 2
        assert steps[0] == "Step one"
        assert steps[1] == "Step two"

    def test_extract_multiline_numbered_steps(self):
        """Each numbered line should be one step, ignoring continuation lines."""
        text = "1. Start here\n2. Continue here\n3. End here"
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 3

    def test_extract_prefers_numbered_over_bullets(self):
        """When both numbered and bullet patterns exist, numbered should win."""
        text = "1. First numbered\n2. Second numbered\n- A bullet\n- Another bullet"
        steps = StepExtractor.extract_steps(text)

        # Numbered should be extracted since it's tried first
        assert len(steps) == 2
        assert "numbered" in steps[0]

    def test_extract_unicode_bullets(self):
        """Should handle unicode bullet characters."""
        text = "- Step one\n- Step two"
        steps = StepExtractor.extract_steps(text)

        assert len(steps) == 2

    def test_extract_preserves_content(self):
        """Should preserve the actual content of each step."""
        text = "1. Calculate 15 * 23 = 345\n2. Verify: 345 / 15 = 23"
        steps = StepExtractor.extract_steps(text)

        assert "345" in steps[0]
        assert "345 / 15 = 23" in steps[1]


# ===========================================================================
# SelfConsistency tests
# ===========================================================================


class TestSelfConsistency:
    """Tests for the SelfConsistency class."""

    def test_majority_vote_clear_winner(self):
        """Should return the answer that appears most often."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._majority_vote(["42", "42", "42", "99", "100"])
        assert result == "42"

    def test_majority_vote_case_insensitive(self):
        """Voting should be case-insensitive but return original casing."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._majority_vote(["Yes", "yes", "YES", "No"])
        # The first occurrence of the winner should be returned
        assert result == "Yes"

    def test_majority_vote_tie(self):
        """In a tie, the first occurring answer should win."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._majority_vote(["A", "B", "A", "B"])
        assert result == "A"

    def test_majority_vote_single_answer(self):
        """Single answer should be returned as-is."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._majority_vote(["42"])
        assert result == "42"

    def test_majority_vote_empty(self):
        """Empty list should return empty string."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._majority_vote([])
        assert result == ""

    def test_majority_vote_whitespace_handling(self):
        """Should strip whitespace before voting."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._majority_vote([" 42 ", "42", " 42"])
        assert result == "42"

    def test_solve_consistent_answers(self):
        """When LLM always returns same answer, should return that answer."""
        def llm_fn(prompt: str) -> str:
            return "Step 1: Think about it. The answer is 42."

        sc = SelfConsistency(llm_fn, num_paths=3)
        answer = sc.solve("What is 6 * 7?")
        assert answer == "42"

    def test_solve_majority_wins(self):
        """The most common answer should win even if not unanimous."""
        call_count = [0]
        responses = [
            "The answer is 42.",
            "The answer is 42.",
            "The answer is 42.",
            "The answer is 99.",
            "The answer is 99.",
        ]

        def llm_fn(prompt: str) -> str:
            idx = call_count[0] % len(responses)
            call_count[0] += 1
            return responses[idx]

        sc = SelfConsistency(llm_fn, num_paths=5)
        answer = sc.solve("What is 6 * 7?")
        assert answer == "42"

    def test_solve_uses_cot_prompter(self):
        """solve() should use CoT prompting."""
        seen_prompts: list[str] = []

        def llm_fn(prompt: str) -> str:
            seen_prompts.append(prompt)
            return "The answer is 5."

        sc = SelfConsistency(llm_fn, num_paths=2)
        sc.solve("What is 2 + 3?")

        # The prompt should contain the CoT instruction
        assert len(seen_prompts) == 2
        assert "Let's think step by step." in seen_prompts[0]

    def test_extract_answer_the_answer_is(self):
        """Should extract from 'The answer is X' pattern."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._extract_answer("Some reasoning. The answer is 42. Done.")
        assert result == "42"

    def test_extract_answer_final_answer(self):
        """Should extract from 'Final answer: X' pattern."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._extract_answer("Working through it... Final answer: Paris")
        assert result == "Paris"

    def test_extract_answer_therefore(self):
        """Should extract from 'Therefore, X' pattern."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._extract_answer("Step 1... Step 2... Therefore, 345.")
        assert result == "345"

    def test_extract_answer_fallback_last_line(self):
        """Should fall back to last line when no pattern matches."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._extract_answer("Thinking...\nMore thinking...\n42")
        assert result == "42"

    def test_extract_answer_empty_text(self):
        """Empty text should return empty string."""
        sc = SelfConsistency(lambda p: "", num_paths=1)
        result = sc._extract_answer("")
        assert result == ""

    def test_num_paths_called_correctly(self):
        """Should call LLM exactly num_paths times."""
        call_count = [0]

        def llm_fn(prompt: str) -> str:
            call_count[0] += 1
            return "The answer is 1."

        sc = SelfConsistency(llm_fn, num_paths=7)
        sc.solve("test")

        assert call_count[0] == 7


# ===========================================================================
# ReasoningVerifier tests
# ===========================================================================


class TestReasoningVerifier:
    """Tests for the ReasoningVerifier class."""

    def test_verify_valid_chain(self):
        """A valid chain should pass verification."""
        steps = [
            "First, 15 * 20 = 300",
            "Then, 15 * 3 = 45",
            "Finally, 300 + 45 = 345",
        ]
        result = ReasoningVerifier.verify_chain(steps)

        assert result["is_valid"] is True
        assert result["issues"] == []

    def test_verify_empty_chain(self):
        """An empty chain should be invalid."""
        result = ReasoningVerifier.verify_chain([])

        assert result["is_valid"] is False
        assert len(result["issues"]) > 0
        assert "empty" in result["issues"][0].lower()

    def test_verify_chain_with_empty_step(self):
        """A chain with an empty step should report an issue."""
        steps = ["First step", "", "Third step"]
        result = ReasoningVerifier.verify_chain(steps)

        assert result["is_valid"] is False
        assert any("short" in issue.lower() or "empty" in issue.lower() for issue in result["issues"])

    def test_verify_chain_with_short_step(self):
        """A chain with a very short step should report an issue."""
        steps = ["First step is detailed enough", "ok", "Third step is also fine"]
        result = ReasoningVerifier.verify_chain(steps)

        assert result["is_valid"] is False
        assert any("short" in issue.lower() or "empty" in issue.lower() for issue in result["issues"])

    def test_verify_valid_answer(self):
        """Answer appearing in steps should be valid."""
        steps = [
            "First, 15 * 20 = 300",
            "Then, 15 * 3 = 45",
            "Finally, 300 + 45 = 345",
        ]
        result = ReasoningVerifier.verify_answer(steps, "345")

        assert result["is_valid"] is True
        assert result["issues"] == []

    def test_verify_answer_not_in_steps(self):
        """Answer not appearing in steps should flag an issue."""
        steps = [
            "First, 15 * 20 = 300",
            "Then, 15 * 3 = 45",
            "Finally, 300 + 45 = 345",
        ]
        result = ReasoningVerifier.verify_answer(steps, "999")

        assert result["is_valid"] is False
        assert any("does not appear" in issue.lower() for issue in result["issues"])

    def test_verify_empty_answer(self):
        """Empty answer should be invalid."""
        steps = ["Step one", "Step two"]
        result = ReasoningVerifier.verify_answer(steps, "")

        assert result["is_valid"] is False
        assert any("empty" in issue.lower() for issue in result["issues"])

    def test_verify_answer_with_empty_chain(self):
        """Empty chain should make answer verification fail too."""
        result = ReasoningVerifier.verify_answer([], "42")

        assert result["is_valid"] is False
        assert any("empty" in issue.lower() for issue in result["issues"])

    def test_verify_chain_returns_dict_structure(self):
        """verify_chain should return dict with is_valid and issues keys."""
        result = ReasoningVerifier.verify_chain(["step 1"])

        assert "is_valid" in result
        assert "issues" in result
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["issues"], list)

    def test_verify_answer_returns_dict_structure(self):
        """verify_answer should return dict with is_valid and issues keys."""
        result = ReasoningVerifier.verify_answer(["step 1"], "step 1")

        assert "is_valid" in result
        assert "issues" in result
        assert isinstance(result["is_valid"], bool)
        assert isinstance(result["issues"], list)

    def test_verify_contradiction_detection(self):
        """Should detect possible contradictions."""
        steps = [
            "The temperature is positive",
            "The temperature is not positive",
        ]
        result = ReasoningVerifier.verify_chain(steps)

        # Should detect a contradiction
        assert result["is_valid"] is False
        assert any("contradiction" in issue.lower() for issue in result["issues"])

    def test_verify_no_contradiction(self):
        """Consistent steps should not report contradictions."""
        steps = [
            "The temperature is 20 degrees",
            "This is a warm day",
            "We should wear light clothing",
        ]
        result = ReasoningVerifier.verify_chain(steps)

        assert result["is_valid"] is True

    def test_verify_answer_in_last_step(self):
        """Answer in the final step should be valid."""
        steps = [
            "We need to calculate 7 * 6",
            "7 * 6 = 42",
        ]
        result = ReasoningVerifier.verify_answer(steps, "42")

        assert result["is_valid"] is True
