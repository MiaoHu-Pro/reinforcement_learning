"""Memory-conscious rollout and clipped GRPO for R1-Zero-style training."""

from collections import defaultdict
from dataclasses import dataclass
import random
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from countdown_task import reward_function
from data_types import Episode, Problem


@dataclass
class TensorBatch:
    """Padded tensors aligned for causal next-token prediction."""

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    response_mask: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor


def _trim_generated_tokens(
    token_ids: list[int],
    eos_token_id: int | None,
    pad_token_id: int,
) -> tuple[list[int], bool]:
    """Keep one generated response through its first EOS token."""
    if eos_token_id is not None and eos_token_id in token_ids:
        eos_position = token_ids.index(eos_token_id)
        return token_ids[:eos_position + 1], True

    # When PAD differs from EOS, generate() can append PAD after completion.
    while token_ids and token_ids[-1] == pad_token_id:
        token_ids.pop()
    return token_ids, False


@torch.no_grad()
def generate_episodes(
    model: Any,
    tokenizer: Any,
    problems: list[Problem],
    group_size: int,
    generation_batch_size: int,
    max_new_tokens: int,
    device: torch.device,
) -> list[Episode]:
    """Sample `group_size` independent responses for every problem.

    Sampling uses temperature=1 and no top-k/top-p truncation. Therefore the
    behavior distribution is the policy itself, which makes the subsequently
    recomputed old-policy log-probabilities valid for GRPO importance ratios.
    """
    model.eval()
    model.config.use_cache = True

    repeated_problems = [
        problem
        for problem in problems
        for _ in range(group_size)
    ]
    episodes: list[Episode] = []

    for start in range(0, len(repeated_problems), generation_batch_size):
        generation_problems = repeated_problems[
            start:start + generation_batch_size
        ]
        prompts = [problem.prompt for problem in generation_problems]

        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            padding=True,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }
        padded_prompt_length = encoded["input_ids"].shape[1]

        generated = model.generate(
            **encoded,
            do_sample=True,
            temperature=1.0,
            top_p=1.0,
            top_k=0,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True,
        ).cpu()

        for row, problem in zip(generated, generation_problems):
            raw_response_ids = row[padded_prompt_length:].tolist()
            response_ids, finished = _trim_generated_tokens(
                raw_response_ids,
                tokenizer.eos_token_id,
                tokenizer.pad_token_id,
            )
            # An immediate EOS is still one valid action. This fallback is for
            # unusual tokenizers/generation backends that return no token.
            if not response_ids:
                response_ids = [
                    tokenizer.eos_token_id
                    if tokenizer.eos_token_id is not None
                    else tokenizer.pad_token_id
                ]
                finished = True

            response_text = tokenizer.decode(
                response_ids,
                skip_special_tokens=True,
            )
            reward, reward_info = reward_function(
                response=response_text,
                numbers=problem.numbers,
                target=problem.target,
                finished=finished,
            )
            episodes.append(Episode(
                problem_id=problem.problem_id,
                numbers=problem.numbers,
                target=problem.target,
                prompt=problem.prompt,
                prompt_token_ids=problem.prompt_token_ids,
                response_text=response_text,
                response_token_ids=response_ids,
                finished=finished,
                reward=reward,
                reward_info=reward_info,
            ))

        del generated, encoded

    return episodes


def assign_group_advantages(
    episodes: list[Episode],
    epsilon: float = 1e-4,
) -> dict[str, float]:
    """Compute reward-relative-to-siblings advantages without a critic.

    All responses sampled for one problem form a group. If every response has
    the same reward, that group has zero advantage and therefore supplies no
    policy-gradient signal. This is expected GRPO behavior.
    """
    groups: dict[int, list[Episode]] = defaultdict(list)
    for episode in episodes:
        groups[episode.problem_id].append(episode)

    zero_variance_groups = 0
    for group in groups.values():
        rewards = np.asarray([episode.reward for episode in group], dtype=np.float32)
        reward_mean = float(rewards.mean())
        reward_std = float(rewards.std())
        if reward_std < epsilon:
            zero_variance_groups += 1
            for episode in group:
                episode.advantage = 0.0
            continue

        for episode in group:
            episode.advantage = (
                episode.reward - reward_mean
            ) / (reward_std + epsilon)

    return {
        "num_groups": float(len(groups)),
        "zero_variance_groups": float(zero_variance_groups),
    }


def build_tensor_batch(
    episodes: list[Episode],
    pad_token_id: int,
    device: torch.device,
    require_old_log_probs: bool,
) -> TensorBatch:
    """Right-pad trajectories and align response actions with shifted logits."""
    sequence_lengths = [
        len(episode.prompt_token_ids) + len(episode.response_token_ids)
        for episode in episodes
    ]
    max_length = max(sequence_lengths)
    batch_size = len(episodes)

    input_ids = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=device,
    )
    # labels[t] is the token predicted by logits[t]. Prompt/padding positions
    # use -100 so cross_entropy ignores them.
    labels = torch.full(
        (batch_size, max_length - 1),
        -100,
        dtype=torch.long,
        device=device,
    )
    response_mask = torch.zeros(
        (batch_size, max_length - 1),
        dtype=torch.bool,
        device=device,
    )
    old_log_probs = torch.zeros(
        (batch_size, max_length - 1),
        dtype=torch.float32,
        device=device,
    )

    for row, episode in enumerate(episodes):
        full_ids = episode.prompt_token_ids + episode.response_token_ids
        input_ids[row, :len(full_ids)] = torch.tensor(
            full_ids,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :len(full_ids)] = 1

        # The first response token is predicted at prompt_length - 1.
        response_start = len(episode.prompt_token_ids) - 1
        response_end = response_start + len(episode.response_token_ids)
        labels[row, response_start:response_end] = torch.tensor(
            episode.response_token_ids,
            dtype=torch.long,
            device=device,
        )
        response_mask[row, response_start:response_end] = True

        if require_old_log_probs:
            if len(episode.old_log_probs) != len(episode.response_token_ids):
                raise RuntimeError(
                    "old_log_probs must contain one value per response token"
                )
            old_log_probs[row, response_start:response_end] = torch.tensor(
                episode.old_log_probs,
                dtype=torch.float32,
                device=device,
            )

    advantages = torch.tensor(
        [episode.advantage for episode in episodes],
        dtype=torch.float32,
        device=device,
    )
    return TensorBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        response_mask=response_mask,
        old_log_probs=old_log_probs,
        advantages=advantages,
    )


def selected_response_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Select log p(actual response token) without materializing FP32 logits.

    The DAPO demo called `.float()` on `[B,T,vocab]`, which caused allocations
    near 1 GB for only two long Qwen trajectories. Cross entropy can consume
    BF16 logits directly and performs its numerically sensitive reduction with
    an appropriate internal accumulator.
    """
    shifted_logits = logits[:, :-1, :]
    # CUDA autocast normally promotes cross_entropy to FP32. Disable autocast
    # specifically here so it consumes the BF16 logits directly instead of
    # allocating an additional FP32 `[B,T,vocab]` tensor. The fused loss still
    # uses a stable log-sum-exp calculation internally.
    with torch.autocast(device_type=logits.device.type, enabled=False):
        token_nll = F.cross_entropy(
            shifted_logits.reshape(-1, shifted_logits.shape[-1]),
            labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        )
    return -token_nll.reshape_as(labels).float()


@torch.no_grad()
def store_old_policy_log_probs(
    model: Any,
    episodes: list[Episode],
    micro_batch_size: int,
    pad_token_id: int,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    """Snapshot the rollout policy probabilities before any optimizer step."""
    model.eval()
    model.config.use_cache = False

    for start in range(0, len(episodes), micro_batch_size):
        micro_episodes = episodes[start:start + micro_batch_size]
        batch = build_tensor_batch(
            micro_episodes,
            pad_token_id,
            device,
            require_old_log_probs=False,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=dtype,
            enabled=device.type == "cuda",
        ):
            logits = model(
                input_ids=batch.input_ids,
                attention_mask=batch.attention_mask,
                use_cache=False,
            ).logits
            selected = selected_response_log_probs(logits, batch.labels)

        for row, episode in enumerate(micro_episodes):
            episode.old_log_probs = (
                selected[row][batch.response_mask[row]]
                .detach()
                .cpu()
                .tolist()
            )
        del logits, selected, batch


def update_policy_grpo(
    policy: Any,
    reference_policy: Any,
    optimizer: torch.optim.Optimizer,
    episodes: list[Episode],
    pad_token_id: int,
    micro_batch_size: int,
    grpo_epochs: int,
    clip_epsilon: float,
    kl_beta: float,
    max_grad_norm: float,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    """Apply clipped GRPO updates to one fixed on-policy rollout batch.

    For each response token:

        ratio = pi_theta(a|s) / pi_old(a|s)
        objective = min(ratio*A, clip(ratio, 1-eps, 1+eps)*A)

    A frozen base-model reference supplies the non-negative sampled KL
    estimator `exp(log_ref-log_policy) - (log_ref-log_policy) - 1`.
    There is no critic/value model: group-relative rewards are the baseline.
    """
    if not episodes:
        raise ValueError("Cannot update from an empty rollout")

    reference_policy.eval()
    policy.config.use_cache = False
    total_episodes = len(episodes)
    final_metrics: dict[str, float] = {}

    for grpo_epoch in range(grpo_epochs):
        policy.train()
        optimizer.zero_grad(set_to_none=True)
        shuffled = list(episodes)
        random.shuffle(shuffled)

        metric_policy_loss = 0.0
        metric_reference_kl = 0.0
        metric_old_approx_kl = 0.0
        metric_clip_fraction = 0.0
        metric_sampled_entropy = 0.0
        metric_tokens = 0

        for start in range(0, total_episodes, micro_batch_size):
            micro_episodes = shuffled[start:start + micro_batch_size]
            batch = build_tensor_batch(
                micro_episodes,
                pad_token_id,
                device,
                require_old_log_probs=True,
            )

            # Reference logits are consumed first and deleted before the policy
            # forward pass, avoiding two simultaneous `[B,T,vocab]` tensors.
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                reference_logits = reference_policy(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                ).logits
                reference_log_probs = selected_response_log_probs(
                    reference_logits,
                    batch.labels,
                )
            del reference_logits

            with torch.autocast(
                device_type=device.type,
                dtype=dtype,
                enabled=device.type == "cuda",
            ):
                policy_logits = policy(
                    input_ids=batch.input_ids,
                    attention_mask=batch.attention_mask,
                    use_cache=False,
                ).logits
                policy_log_probs = selected_response_log_probs(
                    policy_logits,
                    batch.labels,
                )

                log_ratio = policy_log_probs - batch.old_log_probs
                ratio = torch.exp(torch.clamp(log_ratio, min=-20.0, max=20.0))
                unclipped_objective = ratio * batch.advantages[:, None]
                clipped_ratio = torch.clamp(
                    ratio,
                    min=1.0 - clip_epsilon,
                    max=1.0 + clip_epsilon,
                )
                clipped_objective = clipped_ratio * batch.advantages[:, None]
                policy_loss_tokens = -torch.minimum(
                    unclipped_objective,
                    clipped_objective,
                )

                log_reference_ratio = reference_log_probs - policy_log_probs
                reference_kl_tokens = (
                    torch.exp(torch.clamp(log_reference_ratio, min=-20.0, max=20.0))
                    - log_reference_ratio
                    - 1.0
                )

                token_loss = policy_loss_tokens + kl_beta * reference_kl_tokens
                valid_counts = batch.response_mask.sum(dim=1).clamp_min(1)
                # Original GRPO gives each sampled response equal weight,
                # regardless of response length.
                sample_losses = (
                    (token_loss * batch.response_mask).sum(dim=1)
                    / valid_counts
                )
                # Dividing by the complete rollout size makes gradient
                # accumulation across microbatches exactly one batch mean.
                loss = sample_losses.sum() / total_episodes

            loss.backward()

            with torch.no_grad():
                valid = batch.response_mask
                token_count = int(valid.sum().item())
                metric_tokens += token_count
                metric_policy_loss += float((policy_loss_tokens * valid).sum())
                metric_reference_kl += float((reference_kl_tokens * valid).sum())
                metric_old_approx_kl += float(((-log_ratio) * valid).sum())
                metric_clip_fraction += float(
                    (((ratio - 1.0).abs() > clip_epsilon) * valid).sum()
                )
                # -log pi(a|s) over sampled actions is a cheap sampled
                # surprisal metric; unlike exact entropy it creates no second
                # full-vocabulary probability tensor.
                metric_sampled_entropy += float(((-policy_log_probs) * valid).sum())

            del (
                policy_logits,
                policy_log_probs,
                reference_log_probs,
                policy_loss_tokens,
                reference_kl_tokens,
                loss,
                batch,
            )

        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(),
            max_norm=max_grad_norm,
        )
        optimizer.step()

        denominator = max(metric_tokens, 1)
        final_metrics = {
            "policy_loss": metric_policy_loss / denominator,
            "reference_kl": metric_reference_kl / denominator,
            "old_approx_kl": metric_old_approx_kl / denominator,
            "clip_fraction": metric_clip_fraction / denominator,
            "sampled_entropy": metric_sampled_entropy / denominator,
            "grad_norm": float(grad_norm),
            "tokens": float(metric_tokens),
            "grpo_epoch": float(grpo_epoch + 1),
        }

    return final_metrics
