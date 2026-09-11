"""Small data containers shared by rollout, reward, and GRPO code."""

from dataclasses import dataclass, field


@dataclass
class Problem:
    """One verifiable arithmetic problem and its already-tokenized prompt."""

    problem_id: int
    numbers: list[int]
    target: int
    prompt: str
    prompt_token_ids: list[int]


@dataclass
class Episode:
    """One trajectory: one problem followed by one sampled response."""

    problem_id: int
    numbers: list[int]
    target: int
    prompt: str
    prompt_token_ids: list[int]
    response_text: str
    response_token_ids: list[int]
    finished: bool
    reward: float
    reward_info: dict[str, float]

    # Filled after rollout, before the first GRPO update. There is one old
    # log-probability for each response token/action.
    old_log_probs: list[float] = field(default_factory=list)

    # Filled by group-relative reward normalization.
    advantage: float = 0.0
