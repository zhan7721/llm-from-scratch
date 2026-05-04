"""Chain-of-Thought (CoT) Reasoning implementation for LLMs.

This module implements techniques that encourage LLMs to show their reasoning
process step by step. It contains:
- ChainOfThoughtPrompter: Wraps user questions with CoT prompting instructions.
- StepExtractor: Parses model output into individual reasoning steps.
- SelfConsistency: Generates multiple reasoning paths and votes on the answer.
- ReasoningVerifier: Checks reasoning chains for logical consistency.

Architecture overview:
    User question
        -> ChainOfThoughtPrompter adds "think step by step" instructions
        -> LLM generates a reasoning trace
        -> StepExtractor splits trace into discrete steps
        -> ReasoningVerifier checks the chain
        -> Answer extracted

    For robustness (SelfConsistency):
        -> Repeat N times with temperature > 0
        -> Extract answer from each path
        -> Majority vote on final answer

Design notes:
- The LLM function signature is fn(prompt: str) -> str (simple single-turn).
- Parsing is text-based, using simple string operations rather than regex.
- Self-consistency uses majority voting on extracted final answers.
- The verifier uses heuristic checks (no second LLM call).
- Educational focus: clear, readable code with thorough docstrings.
"""

from collections import Counter
from typing import Any, Callable


__all__ = [
    "ChainOfThoughtPrompter",
    "StepExtractor",
    "SelfConsistency",
    "ReasoningVerifier",
]


# ---------------------------------------------------------------------------
# ChainOfThoughtPrompter
# ---------------------------------------------------------------------------


class ChainOfThoughtPrompter:
    """Wrap user questions with chain-of-thought prompting instructions.

    Chain-of-thought (CoT) prompting encourages LLMs to reason step by step
    before giving a final answer. This class provides methods to format
    questions with different CoT styles.

    Supported styles:
        - "step_by_step": "Let's think step by step."
        - "think_aloud": "Think out loud, showing your reasoning."
        - "structured": "Break this into numbered steps."

    Example:
        prompter = ChainOfThoughtPrompter()
        prompt = prompter.format_prompt("What is 15 * 23?")
        # -> "What is 15 * 23?\n\nLet's think step by step."
    """

    STYLES: dict[str, str] = {
        "step_by_step": "Let's think step by step.",
        "think_aloud": "Think out loud, showing your reasoning process.",
        "structured": "Break this down into numbered steps and show your work.",
    }

    def format_prompt(self, question: str, style: str = "step_by_step") -> str:
        """Format a question with chain-of-thought instructions.

        Args:
            question: The user's question.
            style: The CoT style to use. Must be one of "step_by_step",
                "think_aloud", or "structured".

        Returns:
            The question followed by the CoT instruction.

        Raises:
            ValueError: If the style is not recognized.
        """
        if style not in self.STYLES:
            raise ValueError(
                f"Unknown style '{style}'. Choose from: {list(self.STYLES.keys())}"
            )
        instruction = self.STYLES[style]
        return f"{question}\n\n{instruction}"

    def format_prompt_with_examples(
        self,
        question: str,
        examples: list[dict[str, str]],
        style: str = "step_by_step",
    ) -> str:
        """Format a question with few-shot CoT examples.

        Each example should have "question" and "answer" keys, where "answer"
        contains a step-by-step reasoning trace. The examples teach the model
        the expected reasoning format before the actual question is posed.

        Args:
            question: The user's question.
            examples: List of example dicts, each with "question" and "answer"
                keys (both strings).
            style: The CoT style for the final instruction.

        Returns:
            A prompt with few-shot examples followed by the question and
            CoT instruction.
        """
        parts: list[str] = []

        for i, example in enumerate(examples, 1):
            q = example.get("question", "")
            a = example.get("answer", "")
            parts.append(f"Example {i}:\nQ: {q}\nA: {a}")

        parts.append(f"Now solve this:\nQ: {question}")

        if style not in self.STYLES:
            raise ValueError(
                f"Unknown style '{style}'. Choose from: {list(self.STYLES.keys())}"
            )
        instruction = self.STYLES[style]
        parts.append(instruction)

        return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# StepExtractor
# ---------------------------------------------------------------------------


class StepExtractor:
    """Extract individual reasoning steps from model output.

    Parses a reasoning trace produced by an LLM into discrete steps. Handles
    three common formats:
        1. Numbered steps: "1. First step  2. Second step"
        2. Bullet points: "- First step  - Second step" or "* First step"
        3. Paragraph breaks: Steps separated by blank lines

    The extractor tries numbered steps first, then bullets, then falls back
    to paragraph splitting.

    Example:
        extractor = StepExtractor()
        steps = extractor.extract_steps(
            "1. Multiply 15 * 20 = 300\\n"
            "2. Multiply 15 * 3 = 45\\n"
            "3. Add 300 + 45 = 345"
        )
        # -> ["Multiply 15 * 20 = 300", "Multiply 15 * 3 = 45", "Add 300 + 45 = 345"]
    """

    @staticmethod
    def extract_steps(text: str) -> list[str]:
        """Extract reasoning steps from text.

        Attempts to parse the text as numbered steps first, then bullet
        points, then falls back to paragraph-based splitting. Empty strings
        and whitespace-only inputs return an empty list.

        Args:
            text: The model's reasoning output text.

        Returns:
            A list of strings, one per reasoning step. Leading/trailing
            whitespace is stripped from each step.
        """
        if not text or not text.strip():
            return []

        # Try numbered steps: "1. text" or "1) text"
        steps = StepExtractor._extract_numbered(text)
        if steps:
            return steps

        # Try bullet points: "- text" or "* text"
        steps = StepExtractor._extract_bullets(text)
        if steps:
            return steps

        # Fall back to paragraph splitting
        return StepExtractor._extract_paragraphs(text)

    @staticmethod
    def _extract_numbered(text: str) -> list[str]:
        """Extract steps from numbered list format.

        Looks for lines starting with a number followed by '.' or ')'.
        """
        steps: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            # Check for "1. " or "1) " pattern at start of line
            dot_pos = stripped.find(". ")
            paren_pos = stripped.find(") ")
            # Find whichever pattern appears first
            sep_pos = -1
            if dot_pos >= 0 and paren_pos >= 0:
                sep_pos = min(dot_pos, paren_pos)
            elif dot_pos >= 0:
                sep_pos = dot_pos
            elif paren_pos >= 0:
                sep_pos = paren_pos

            if sep_pos > 0:
                prefix = stripped[:sep_pos]
                if prefix.isdigit():
                    steps.append(stripped[sep_pos + 2 :].strip())

        return steps

    @staticmethod
    def _extract_bullets(text: str) -> list[str]:
        """Extract steps from bullet point format.

        Recognizes lines starting with '-', '*', or unicode bullets.
        """
        bullet_chars = ("-", "•", "‣", "●", "*")
        steps: list[str] = []

        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            for bullet in bullet_chars:
                if stripped.startswith(bullet + " ") or stripped.startswith(bullet + "\t"):
                    steps.append(stripped[len(bullet) + 1 :].strip())
                    break

        return steps

    @staticmethod
    def _extract_paragraphs(text: str) -> list[str]:
        """Extract steps by splitting on blank lines.

        Groups of non-blank lines are treated as single steps.
        """
        steps: list[str] = []
        current: list[str] = []

        for line in text.split("\n"):
            if line.strip():
                current.append(line.strip())
            else:
                if current:
                    steps.append(" ".join(current))
                    current = []

        # Don't forget the last paragraph
        if current:
            steps.append(" ".join(current))

        return steps


# ---------------------------------------------------------------------------
# SelfConsistency
# ---------------------------------------------------------------------------


class SelfConsistency:
    """Generate multiple reasoning paths and vote on the answer.

    Self-consistency (Wang et al., 2022) improves reasoning reliability by
    sampling multiple diverse reasoning paths and selecting the most common
    final answer via majority voting.

    The algorithm:
        1. Generate N reasoning paths (with temperature > 0 for diversity).
        2. Extract the final answer from each path.
        3. Return the most frequently occurring answer.

    Args:
        llm_fn: A callable that takes a prompt string and returns the LLM's
            response string.
        num_paths: Number of reasoning paths to generate (default: 5).

    Example:
        def my_llm(prompt: str) -> str:
            return "Step 1: ... Step 2: ... The answer is 42."

        sc = SelfConsistency(my_llm, num_paths=5)
        answer = sc.solve("What is 6 * 7?")
    """

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        num_paths: int = 5,
    ):
        """Initialize the self-consistency solver.

        Args:
            llm_fn: Callable that takes a prompt string and returns the LLM's
                response string.
            num_paths: Number of reasoning paths to generate.
        """
        self.llm_fn = llm_fn
        self.num_paths = num_paths
        self.prompter = ChainOfThoughtPrompter()
        self.extractor = StepExtractor()

    def solve(self, question: str) -> str:
        """Solve a question using self-consistency.

        Generates multiple reasoning paths, extracts answers from each, and
        returns the most common answer via majority voting.

        Args:
            question: The question to solve.

        Returns:
            The most frequently occurring answer string. If there is a tie,
            the answer from the first path among the tied answers is returned.
        """
        prompt = self.prompter.format_prompt(question)
        answers: list[str] = []

        for _ in range(self.num_paths):
            response = self.llm_fn(prompt)
            answer = self._extract_answer(response)
            answers.append(answer)

        return self._majority_vote(answers)

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from a reasoning trace.

        Looks for common answer patterns:
            - "The answer is X"
            - "Therefore, X"
            - "So, X"
            - "Final answer: X"

        Falls back to the last sentence if no pattern matches.

        Args:
            text: The full reasoning trace from the LLM.

        Returns:
            The extracted answer string.
        """
        text_lower = text.lower()

        # Try common answer patterns (order matters - check most specific first)
        patterns = [
            "the answer is ",
            "final answer: ",
            "answer: ",
            "therefore, ",
            "thus, ",
            "so, ",
        ]

        for pattern in patterns:
            # Search from the end to get the last occurrence
            idx = text_lower.rfind(pattern)
            if idx != -1:
                answer_start = idx + len(pattern)
                answer = text[answer_start:].strip()
                # Take only until the next sentence boundary
                for end_char in (".", "\n", "!"):
                    end_idx = answer.find(end_char)
                    if end_idx > 0:
                        answer = answer[:end_idx].strip()
                        break
                if answer:
                    return answer

        # Fallback: return the last non-empty line
        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if lines:
            return lines[-1]
        return text.strip()

    @staticmethod
    def _majority_vote(answers: list[str]) -> str:
        """Return the most common answer.

        In case of a tie, returns the answer that appeared first among
        the tied candidates.

        Args:
            answers: List of answer strings.

        Returns:
            The most common answer.
        """
        if not answers:
            return ""

        # Normalize for voting (case-insensitive, stripped)
        normalized: list[str] = []
        for a in answers:
            normalized.append(a.strip().lower())

        counts = Counter(normalized)
        # Get the most common normalized answer
        most_common_normalized = counts.most_common(1)[0][0]

        # Return the original (first occurrence) casing
        for original, norm in zip(answers, normalized):
            if norm == most_common_normalized:
                return original.strip()

        return answers[0].strip()


# ---------------------------------------------------------------------------
# ReasoningVerifier
# ---------------------------------------------------------------------------


class ReasoningVerifier:
    """Check reasoning chains for logical consistency.

    Uses heuristic checks (no second LLM call) to identify common issues
    in reasoning chains:
        - Empty or missing steps
        - Steps that are too short to be meaningful
        - Contradictory steps (steps that negate earlier ones)
        - Answer not supported by the reasoning steps

    Example:
        verifier = ReasoningVerifier()
        result = verifier.verify_chain([
            "First, 15 * 20 = 300",
            "Then, 15 * 3 = 45",
            "Finally, 300 + 45 = 345",
        ])
        # -> {"is_valid": True, "issues": []}
    """

    @staticmethod
    def verify_chain(steps: list[str]) -> dict[str, Any]:
        """Verify that a reasoning chain is logically consistent.

        Performs heuristic checks on the chain structure and content:
            1. Chain must not be empty.
            2. Each step must have meaningful content (not too short).
            3. Steps must not directly contradict each other.

        Args:
            steps: List of reasoning step strings.

        Returns:
            A dict with keys:
                - "is_valid" (bool): True if no issues found.
                - "issues" (list[str]): Description of each issue found.
        """
        issues: list[str] = []

        if not steps:
            issues.append("Reasoning chain is empty.")
            return {"is_valid": False, "issues": issues}

        # Check for empty or near-empty steps
        for i, step in enumerate(steps):
            if not step or len(step.strip()) < 3:
                issues.append(f"Step {i + 1} is empty or too short.")

        # Check for contradictions
        contradiction_issues = ReasoningVerifier._check_contradictions(steps)
        issues.extend(contradiction_issues)

        return {"is_valid": len(issues) == 0, "issues": issues}

    @staticmethod
    def verify_answer(steps: list[str], answer: str) -> dict[str, Any]:
        """Verify that an answer is supported by the reasoning steps.

        Checks whether the answer (or something resembling it) appears in
        the reasoning chain, and whether the chain itself is valid.

        Args:
            steps: List of reasoning step strings.
            answer: The final answer to verify.

        Returns:
            A dict with keys:
                - "is_valid" (bool): True if no issues found.
                - "issues" (list[str]): Description of each issue found.
        """
        # First verify the chain itself
        result = ReasoningVerifier.verify_chain(steps)
        issues = list(result["issues"])

        if not answer or not answer.strip():
            issues.append("Answer is empty.")
            return {"is_valid": False, "issues": issues}

        # Check if the answer or a close variant appears in the steps
        answer_lower = answer.strip().lower()
        chain_text = " ".join(steps).lower()

        if answer_lower not in chain_text:
            issues.append(
                "Answer does not appear in the reasoning steps. "
                "The conclusion may not follow from the reasoning."
            )

        return {"is_valid": len(issues) == 0, "issues": issues}

    @staticmethod
    def _check_contradictions(steps: list[str]) -> list[str]:
        """Check for contradictory statements in reasoning steps.

        Looks for patterns where a later step negates an earlier one,
        e.g., "X is positive" followed by "X is not positive".

        This is a simple heuristic check, not a full logical analysis.

        Args:
            steps: List of reasoning step strings.

        Returns:
            List of contradiction issue descriptions.
        """
        issues: list[str] = []
        negation_words = ("not", "no", "never", "isn't", "doesn't", "cannot", "can't")

        for i, step in enumerate(steps):
            step_lower = step.lower()
            # Check if this step contains a negation
            has_negation = any(
                f" {w} " in f" {step_lower} " or step_lower.startswith(f"{w} ")
                for w in negation_words
            )
            if not has_negation:
                continue

            # Compare with earlier steps to find potential contradictions
            for j in range(i):
                earlier_lower = steps[j].lower()

                # Simple heuristic: check if removing negation words creates overlap
                for neg_word in negation_words:
                    # If removing the negation from this step makes it similar
                    # to an earlier step, it might be a contradiction
                    cleaned = step_lower.replace(f" {neg_word} ", " ").replace(
                        f"{neg_word} ", ""
                    )
                    # Check for significant overlap (shared meaningful words)
                    current_words = set(cleaned.split())
                    earlier_words = set(earlier_lower.split())

                    # Remove common stop words for better matching
                    stop_words = {
                        "the", "a", "an", "is", "are", "was", "were",
                        "and", "or", "but", "in", "on", "at", "to",
                        "of", "for", "it", "this", "that", "with",
                    }
                    current_meaningful = current_words - stop_words
                    earlier_meaningful = earlier_words - stop_words

                    if len(current_meaningful) >= 2 and len(earlier_meaningful) >= 2:
                        overlap = current_meaningful & earlier_meaningful
                        # If most meaningful words overlap, it may be a contradiction
                        if len(overlap) >= min(
                            len(current_meaningful), len(earlier_meaningful)
                        ) * 0.6:
                            issues.append(
                                f"Possible contradiction between step {j + 1} "
                                f"and step {i + 1}."
                            )
                            break

        return issues
