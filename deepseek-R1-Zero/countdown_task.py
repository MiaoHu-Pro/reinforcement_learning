"""Countdown arithmetic data and fully verifiable rule-based rewards.

No chain-of-thought answer is provided by the dataset. The model receives only
the problem and output-format instructions, which is the important "Zero"
property: reasoning traces are explored through RL rather than imitated from an
SFT reasoning dataset.
"""

import ast
from collections import Counter
from fractions import Fraction
from pathlib import Path
import re
from typing import Any

from datasets import load_dataset
from torch.utils.data import Dataset

from data_types import Problem


PROMPT_TEMPLATE = """You are a reasoning assistant.

Use each of these numbers exactly once: {numbers}
Construct an arithmetic expression equal to {target}.
You may use only +, -, *, /, and parentheses.

Think through the problem, then return exactly this structure:
<think>
your reasoning
</think>
<answer>
your final arithmetic expression
</answer>

Begin now.
<think>
"""


class CountdownDataset(Dataset):
    """Read the local Countdown parquet file and build plain-text prompts."""

    def __init__(
        self,
        data_file: Path,
        tokenizer: Any,
        split: str,
        test_size: int = 128,
    ) -> None:
        if split not in {"train", "test"}:
            raise ValueError("split must be 'train' or 'test'")
        if not data_file.is_file():
            raise FileNotFoundError(f"Countdown parquet file not found: {data_file}")

        dataset = load_dataset(
            "parquet",
            data_files=str(data_file),
            split="train",
        )
        if not 0 < test_size < len(dataset):
            raise ValueError("test_size must be between 1 and dataset_size - 1")

        self.dataset = (
            dataset.select(range(0, len(dataset) - test_size))
            if split == "train"
            else dataset.select(range(len(dataset) - test_size, len(dataset)))
        )
        self.tokenizer = tokenizer
        self.index_offset = 0 if split == "train" else len(dataset) - test_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Problem:
        row = self.dataset[index]
        numbers = [int(number) for number in row["nums"]]
        target = int(row["target"])
        prompt = PROMPT_TEMPLATE.format(numbers=numbers, target=target)
        prompt_token_ids = self.tokenizer.encode(
            prompt,
            add_special_tokens=False,
        )
        return Problem(
            problem_id=self.index_offset + index,
            numbers=numbers,
            target=target,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
        )

    @staticmethod
    def collate_fn(problems: list[Problem]) -> list[Problem]:
        # Prompts have different lengths, so rollout tokenizes/pads them later.
        return problems


def _evaluate_expression(node: ast.AST, used_numbers: list[int]) -> Fraction:
    """Evaluate only the arithmetic grammar allowed by the task."""
    if isinstance(node, ast.Expression):
        return _evaluate_expression(node.body, used_numbers)

    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, int):
            raise ValueError("Only integer constants are allowed")
        used_numbers.append(int(node.value))
        return Fraction(int(node.value), 1)

    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = _evaluate_expression(node.operand, used_numbers)
        return value if isinstance(node.op, ast.UAdd) else -value

    if isinstance(node, ast.BinOp):
        left = _evaluate_expression(node.left, used_numbers)
        right = _evaluate_expression(node.right, used_numbers)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise ZeroDivisionError
            return left / right

    raise ValueError("Expression contains a forbidden operation")


def answer_reward(response: str, numbers: list[int], target: int) -> float:
    """Return 1 only for a correct expression using every supplied number once."""
    matches = re.findall(
        r"<answer>\s*(.*?)\s*</answer>",
        response,
        flags=re.DOTALL,
    )
    if len(matches) != 1:
        return 0.0

    expression = matches[0].strip()
    if not expression or not re.fullmatch(r"[0-9+\-*/()\s]+", expression):
        return 0.0

    try:
        syntax_tree = ast.parse(expression, mode="eval")
        used_numbers: list[int] = []
        result = _evaluate_expression(syntax_tree, used_numbers)
    except (SyntaxError, ValueError, ZeroDivisionError, OverflowError):
        return 0.0

    if Counter(used_numbers) != Counter(numbers):
        return 0.0
    return float(result == Fraction(target, 1))


def format_reward(response: str) -> float:
    """Reward the requested reasoning/answer tags without judging reasoning text."""
    full_response = "<think>\n" + response.strip()
    exact_pattern = (
        r"^<think>\s*.+?\s*</think>\s*"
        r"<answer>\s*.+?\s*</answer>$"
    )
    if re.fullmatch(exact_pattern, full_response, flags=re.DOTALL):
        return 1.0

    # Partial credit makes the very sparse early exploration signal gentler.
    partial = 0.0
    if re.search(r"</think>", full_response):
        partial += 0.25
    if re.search(r"<answer>.*?</answer>", full_response, flags=re.DOTALL):
        partial += 0.50
    return partial


def reward_function(
    response: str,
    numbers: list[int],
    target: int,
    finished: bool,
) -> tuple[float, dict[str, float]]:
    """Combine verifiable accuracy and formatting rewards.

    Correctness dominates. Format reward has weight 0.1, matching the role of
    a small structural reward rather than a learned preference model.
    """
    accuracy = answer_reward(response, numbers, target)
    formatting = format_reward(response)
    total = accuracy + 0.1 * formatting
    return total, {
        "answer_reward": accuracy,
        "format_reward": formatting,
        "finished": float(finished),
    }
