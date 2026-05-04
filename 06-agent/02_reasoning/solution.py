"""Reference solution for the Chain-of-Thought Reasoning exercise."""

from collections import Counter
from typing import Any, Callable


class ChainOfThoughtPrompterSolution:
    """Wrap user questions with chain-of-thought prompting instructions."""

    STYLES: dict[str, str] = {
        "step_by_step": "Let's think step by step.",
        "think_aloud": "Think out loud, showing your reasoning process.",
        "structured": "Break this down into numbered steps and show your work.",
    }

    def format_prompt(self, question: str, style: str = "step_by_step") -> str:
        """Format a question with chain-of-thought instructions."""
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
        """Format a question with few-shot CoT examples."""
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


class StepExtractorSolution:
    """Extract individual reasoning steps from model output."""

    @staticmethod
    def extract_steps(text: str) -> list[str]:
        """Extract reasoning steps from text."""
        if not text or not text.strip():
            return []

        # Try numbered steps
        steps = StepExtractorSolution._extract_numbered(text)
        if steps:
            return steps

        # Try bullet points
        steps = StepExtractorSolution._extract_bullets(text)
        if steps:
            return steps

        # Fall back to paragraph splitting
        return StepExtractorSolution._extract_paragraphs(text)

    @staticmethod
    def _extract_numbered(text: str) -> list[str]:
        """Extract steps from numbered list format."""
        steps: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if not stripped:
                continue
            dot_pos = stripped.find(". ")
            paren_pos = stripped.find(") ")
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
        """Extract steps from bullet point format."""
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
        """Extract steps by splitting on blank lines."""
        steps: list[str] = []
        current: list[str] = []

        for line in text.split("\n"):
            if line.strip():
                current.append(line.strip())
            else:
                if current:
                    steps.append(" ".join(current))
                    current = []

        if current:
            steps.append(" ".join(current))

        return steps


class SelfConsistencySolution:
    """Generate multiple reasoning paths and vote on the answer."""

    def __init__(
        self,
        llm_fn: Callable[[str], str],
        num_paths: int = 5,
    ):
        """Initialize the self-consistency solver."""
        self.llm_fn = llm_fn
        self.num_paths = num_paths
        self.prompter = ChainOfThoughtPrompterSolution()
        self.extractor = StepExtractorSolution()

    def solve(self, question: str) -> str:
        """Solve a question using self-consistency."""
        prompt = self.prompter.format_prompt(question)
        answers: list[str] = []

        for _ in range(self.num_paths):
            response = self.llm_fn(prompt)
            answer = self._extract_answer(response)
            answers.append(answer)

        return self._majority_vote(answers)

    def _extract_answer(self, text: str) -> str:
        """Extract the final answer from a reasoning trace."""
        text_lower = text.lower()

        patterns = [
            "the answer is ",
            "final answer: ",
            "answer: ",
            "therefore, ",
            "thus, ",
            "so, ",
        ]

        for pattern in patterns:
            idx = text_lower.rfind(pattern)
            if idx != -1:
                answer_start = idx + len(pattern)
                answer = text[answer_start:].strip()
                for end_char in (".", "\n", "!"):
                    end_idx = answer.find(end_char)
                    if end_idx > 0:
                        answer = answer[:end_idx].strip()
                        break
                if answer:
                    return answer

        lines = [line.strip() for line in text.strip().split("\n") if line.strip()]
        if lines:
            return lines[-1]
        return text.strip()

    @staticmethod
    def _majority_vote(answers: list[str]) -> str:
        """Return the most common answer."""
        if not answers:
            return ""

        normalized: list[str] = []
        for a in answers:
            normalized.append(a.strip().lower())

        counts = Counter(normalized)
        most_common_normalized = counts.most_common(1)[0][0]

        for original, norm in zip(answers, normalized):
            if norm == most_common_normalized:
                return original.strip()

        return answers[0].strip()


class ReasoningVerifierSolution:
    """Check reasoning chains for logical consistency."""

    @staticmethod
    def verify_chain(steps: list[str]) -> dict[str, Any]:
        """Verify that a reasoning chain is logically consistent."""
        issues: list[str] = []

        if not steps:
            issues.append("Reasoning chain is empty.")
            return {"is_valid": False, "issues": issues}

        for i, step in enumerate(steps):
            if not step or len(step.strip()) < 3:
                issues.append(f"Step {i + 1} is empty or too short.")

        contradiction_issues = ReasoningVerifierSolution._check_contradictions(steps)
        issues.extend(contradiction_issues)

        return {"is_valid": len(issues) == 0, "issues": issues}

    @staticmethod
    def verify_answer(steps: list[str], answer: str) -> dict[str, Any]:
        """Verify that an answer is supported by the reasoning steps."""
        result = ReasoningVerifierSolution.verify_chain(steps)
        issues = list(result["issues"])

        if not answer or not answer.strip():
            issues.append("Answer is empty.")
            return {"is_valid": False, "issues": issues}

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
        """Check for contradictory statements in reasoning steps."""
        issues: list[str] = []
        negation_words = ("not", "no", "never", "isn't", "doesn't", "cannot", "can't")

        for i, step in enumerate(steps):
            step_lower = step.lower()
            has_negation = any(
                f" {w} " in f" {step_lower} " or step_lower.startswith(f"{w} ")
                for w in negation_words
            )
            if not has_negation:
                continue

            for j in range(i):
                earlier_lower = steps[j].lower()

                for neg_word in negation_words:
                    cleaned = step_lower.replace(f" {neg_word} ", " ").replace(
                        f"{neg_word} ", ""
                    )
                    current_words = set(cleaned.split())
                    earlier_words = set(earlier_lower.split())

                    stop_words = {
                        "the", "a", "an", "is", "are", "was", "were",
                        "and", "or", "but", "in", "on", "at", "to",
                        "of", "for", "it", "this", "that", "with",
                    }
                    current_meaningful = current_words - stop_words
                    earlier_meaningful = earlier_words - stop_words

                    if len(current_meaningful) >= 2 and len(earlier_meaningful) >= 2:
                        overlap = current_meaningful & earlier_meaningful
                        if len(overlap) >= min(
                            len(current_meaningful), len(earlier_meaningful)
                        ) * 0.6:
                            issues.append(
                                f"Possible contradiction between step {j + 1} "
                                f"and step {i + 1}."
                            )
                            break

        return issues
