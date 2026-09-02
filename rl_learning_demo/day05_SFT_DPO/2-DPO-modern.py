"""Direct Preference Optimization for the modern Qwen3 SFT checkpoint.

This is the current-PyTorch/Transformers rewrite of the legacy ``2-DPO.py``.
The legacy file is left untouched.

DPO requires preference pairs for the same prompt:

    (x, y_chosen, y_rejected)

The local UltraChat 200k directory used by ``1-SFT-modern.py`` is not a
preference dataset: it has conversations but no chosen/rejected columns. This
script therefore uses the separate UltraFeedback Binarized preference dataset.
The input source can be selected explicitly:

* ``--data-source local`` reads the already-downloaded repository at
  ``data/ultrafeedback_binarized`` (the default);
* ``--data-source huggingface`` streams ``HuggingFaceH4/ultrafeedback_binarized``;
* ``--data-source path --data-path ...`` reads custom Parquet/JSON/JSONL data.

For one pair, define the policy/reference log-ratios:

    log_ratio_chosen   = log pi_theta(y_chosen|x)
                         - log pi_ref(y_chosen|x)

    log_ratio_rejected = log pi_theta(y_rejected|x)
                         - log pi_ref(y_rejected|x)

The DPO margin and loss are:

    margin = beta * (log_ratio_chosen - log_ratio_rejected)

    L_DPO = -log sigmoid(margin).

Only final assistant-completion tokens contribute to sequence log probability;
system text, prompt/history tokens, and padding are excluded.

Validate a local preference file without loading model weights:

    uv run python rl_learning_demo/day05_SFT_DPO/2-DPO-modern.py \
        --preprocess-only --data-source path \
        --data-path /path/to/preferences.jsonl \
        --tokenizer-path /home/mh1f25/scratch/llms_model/Qwen3-0.6B-Base

Train after running ``1-SFT-modern.py``:

    uv run python rl_learning_demo/day05_SFT_DPO/2-DPO-modern.py --device cuda
"""

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT_MODEL_PATH = (
    PROJECT_ROOT / "outputs" / "Qwen3-0.6B-Base-UltraChat-SFT"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "Qwen3-0.6B-Base-UltraChat-SFT-DPO"
)
DEFAULT_DATASET_NAME = "HuggingFaceH4/ultrafeedback_binarized"
DEFAULT_LOCAL_DATA_DIR = PROJECT_ROOT / "data" / "ultrafeedback_binarized"
IGNORE_INDEX = -100


@dataclass
class PreferenceExample:
    """One tokenized prompt with its chosen and rejected completions."""

    chosen_input_ids: list[int]
    rejected_input_ids: list[int]
    chosen_response_mask: list[int]
    rejected_response_mask: list[int]
    prompt_token_count: int
    chosen_token_count: int
    rejected_token_count: int


class PreferenceDataset(Dataset[PreferenceExample]):
    """In-memory preference pairs prepared before model loading."""

    def __init__(self, examples: list[PreferenceExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PreferenceExample:
        return self.examples[index]


class PreferenceCollator:
    """Concatenate chosen/rejected sequences and dynamically right-pad.

    The returned order is:

        [chosen_0, ..., chosen_(B-1), rejected_0, ..., rejected_(B-1)].

    Running each model once on this 2B batch is simpler and normally faster
    than four separate chosen/rejected forward passes.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: list[PreferenceExample],
    ) -> dict[str, torch.Tensor]:
        all_input_ids = [
            example.chosen_input_ids for example in examples
        ] + [
            example.rejected_input_ids for example in examples
        ]
        all_response_masks = [
            example.chosen_response_mask for example in examples
        ] + [
            example.rejected_response_mask for example in examples
        ]

        input_tensors = [
            torch.tensor(ids, dtype=torch.long) for ids in all_input_ids
        ]
        response_mask_tensors = [
            torch.tensor(mask, dtype=torch.bool)
            for mask in all_response_masks
        ]
        attention_tensors = [
            torch.ones(len(ids), dtype=torch.long) for ids in all_input_ids
        ]

        input_ids = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        response_mask = pad_sequence(
            response_mask_tensors,
            batch_first=True,
            padding_value=False,
        )
        attention_mask = pad_sequence(
            attention_tensors,
            batch_first=True,
            padding_value=0,
        )

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "response_mask": response_mask,
            "pair_count": torch.tensor(len(examples), dtype=torch.long),
        }


def normalize_messages(
    raw_messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Copy supported roles and add a system message when absent."""
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or content is None:
            continue
        messages.append({"role": role, "content": str(content)})

    if not messages or messages[0]["role"] != "system":
        messages.insert(
            0,
            {"role": "system", "content": system_prompt},
        )
    return messages


def common_prompt_and_completions(
    raw_chosen: list[dict[str, Any]],
    raw_rejected: list[dict[str, Any]],
    system_prompt: str,
) -> tuple[list[dict[str, str]], list[dict[str, str]], list[dict[str, str]]] | None:
    """Validate that a pair differs only in its final assistant completion."""
    chosen = normalize_messages(raw_chosen, system_prompt)
    rejected = normalize_messages(raw_rejected, system_prompt)

    if not chosen or not rejected:
        return None
    if chosen[-1]["role"] != "assistant" or rejected[-1]["role"] != "assistant":
        return None

    chosen_prompt = chosen[:-1]
    rejected_prompt = rejected[:-1]
    if chosen_prompt != rejected_prompt:
        # DPO compares two answers for the same context. Silently accepting
        # different prompts would make the preference likelihood ratio invalid.
        return None

    return chosen_prompt, chosen, rejected


def apply_chat_template_ids(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
) -> list[int]:
    """Return a flat token list with explicit Transformers 5 behavior."""
    result = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        return_dict=False,
    )
    return list(result)


def split_prompt_and_completion(
    tokenizer: Any,
    prompt_messages: list[dict[str, str]],
    complete_messages: list[dict[str, str]],
) -> tuple[list[int], list[int]] | None:
    """Render a prompt and isolate tokens added by its final assistant reply.

    Rendering the prompt with ``add_generation_prompt=True`` includes the
    assistant role header. Consequently, completion tokens begin after that
    header and include response content plus the message-ending token.
    """
    prompt_ids = apply_chat_template_ids(
        tokenizer,
        prompt_messages,
        add_generation_prompt=True,
    )
    full_ids = apply_chat_template_ids(
        tokenizer,
        complete_messages,
        add_generation_prompt=False,
    )

    if full_ids[: len(prompt_ids)] != prompt_ids:
        # Chat templates can theoretically render completed and generation
        # prompts differently. Do not guess token alignment if this invariant
        # fails for a different model/template.
        return None

    completion_ids = full_ids[len(prompt_ids) :]
    if not completion_ids:
        return None
    return prompt_ids, completion_ids


def tokenize_preference_pair(
    raw_chosen: list[dict[str, Any]],
    raw_rejected: list[dict[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    max_length: int,
    max_prompt_length: int,
) -> PreferenceExample | None:
    """Tokenize one valid pair while preserving completion space.

    Long prompts are left-truncated to keep the most recent conversation. Each
    response is then right-truncated to fit ``max_length``. This is safer for
    DPO than right-truncating the entire conversation, which can remove both
    chosen and rejected answers and leave no preference signal.
    """
    conversation_parts = common_prompt_and_completions(
        raw_chosen,
        raw_rejected,
        system_prompt,
    )
    if conversation_parts is None:
        return None
    prompt_messages, chosen_messages, rejected_messages = conversation_parts

    chosen_parts = split_prompt_and_completion(
        tokenizer,
        prompt_messages,
        chosen_messages,
    )
    rejected_parts = split_prompt_and_completion(
        tokenizer,
        prompt_messages,
        rejected_messages,
    )
    if chosen_parts is None or rejected_parts is None:
        return None

    chosen_prompt_ids, chosen_completion_ids = chosen_parts
    rejected_prompt_ids, rejected_completion_ids = rejected_parts
    if chosen_prompt_ids != rejected_prompt_ids:
        return None

    # Keep recent prompt tokens and reserve at least one completion position.
    prompt_limit = min(max_prompt_length, max_length - 1)
    prompt_ids = chosen_prompt_ids[-prompt_limit:]
    available_completion_length = max_length - len(prompt_ids)
    chosen_completion_ids = chosen_completion_ids[:available_completion_length]
    rejected_completion_ids = rejected_completion_ids[:available_completion_length]

    if not chosen_completion_ids or not rejected_completion_ids:
        return None
    if chosen_completion_ids == rejected_completion_ids:
        return None

    chosen_input_ids = prompt_ids + chosen_completion_ids
    rejected_input_ids = prompt_ids + rejected_completion_ids
    chosen_response_mask = (
        [0] * len(prompt_ids) + [1] * len(chosen_completion_ids)
    )
    rejected_response_mask = (
        [0] * len(prompt_ids) + [1] * len(rejected_completion_ids)
    )

    return PreferenceExample(
        chosen_input_ids=chosen_input_ids,
        rejected_input_ids=rejected_input_ids,
        chosen_response_mask=chosen_response_mask,
        rejected_response_mask=rejected_response_mask,
        prompt_token_count=len(prompt_ids),
        chosen_token_count=len(chosen_completion_ids),
        rejected_token_count=len(rejected_completion_ids),
    )


def discover_local_data_files(
    data_path: Path,
    dataset_split: str | None = None,
) -> tuple[str, list[Path]]:
    """Resolve a local preference file or directory to one dataset format."""
    if data_path.is_file():
        files = [data_path]
    elif data_path.is_dir():
        # A downloaded Hugging Face dataset repository can contain train_sft,
        # train_gen, test, and preference files together.  When a split is
        # supplied, load only files whose basename starts with that exact split.
        parquet_pattern = (
            f"{dataset_split}-*.parquet" if dataset_split else "*.parquet"
        )
        files = sorted(data_path.rglob(parquet_pattern))
        if not files:
            jsonl_pattern = f"{dataset_split}-*.jsonl" if dataset_split else "*.jsonl"
            files = sorted(data_path.rglob(jsonl_pattern))
        if not files:
            json_pattern = f"{dataset_split}-*.json" if dataset_split else "*.json"
            files = sorted(data_path.rglob(json_pattern))
    else:
        raise FileNotFoundError(f"Preference data path does not exist: {data_path}")

    if not files:
        raise FileNotFoundError(f"No Parquet/JSON preference files under {data_path}")

    suffixes = {file.suffix.lower() for file in files}
    if suffixes == {".parquet"}:
        return "parquet", files
    if suffixes <= {".json", ".jsonl"}:
        return "json", files
    raise ValueError("Local preference files must use one consistent format")


def load_data(
    data_source: str,
    local_data_dir: Path,
    data_path: Path | None,
    dataset_name: str,
    dataset_split: str,
    cache_dir: Path,
) -> Any:
    """Load the selected preference-data source as a streaming dataset.

    ``local`` is tailored to a downloaded Hugging Face dataset repository and
    therefore selects only ``<dataset_split>-*`` files.  ``path`` accepts one
    custom file or all files of one format beneath a custom directory.
    ``huggingface`` lets the datasets library fetch/stream the named dataset.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)

    if data_source in {"local", "path"}:
        selected_path = local_data_dir if data_source == "local" else data_path
        if selected_path is None:  # Protected by validate_args; keeps API safe.
            raise ValueError("--data-path is required when --data-source path")
        split_filter = dataset_split if data_source == "local" else None
        dataset_format, files = discover_local_data_files(
            selected_path,
            dataset_split=split_filter,
        )
        print(f"Preference data source: {data_source}")
        print("Preference files:")
        for file in files:
            print(f"  - {file}")
        return load_dataset(
            dataset_format,
            data_files=[str(file) for file in files],
            split="train",
            streaming=True,
            cache_dir=str(cache_dir),
        )

    print("Preference data source: huggingface")
    print(f"Preference dataset: {dataset_name}, split: {dataset_split}")
    print(
        "This dataset is separate from local UltraChat 200k and may require "
        "network access on its first use."
    )
    return load_dataset(
        dataset_name,
        split=dataset_split,
        streaming=True,
        cache_dir=str(cache_dir),
    )


def prepare_examples(
    tokenizer: Any,
    rows: Any,
    chosen_column: str,
    rejected_column: str,
    system_prompt: str,
    max_samples: int,
    max_length: int,
    max_prompt_length: int,
) -> tuple[list[PreferenceExample], int]:
    """Validate and tokenize up to ``max_samples`` chosen/rejected pairs."""
    examples: list[PreferenceExample] = []
    skipped_examples = 0

    for row in rows:
        if chosen_column not in row or rejected_column not in row:
            raise KeyError(
                f"Preference row must contain {chosen_column!r} and "
                f"{rejected_column!r}; available columns: {list(row)}"
            )

        example = tokenize_preference_pair(
            raw_chosen=row[chosen_column],
            raw_rejected=row[rejected_column],
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        if example is None:
            skipped_examples += 1
            continue

        examples.append(example)
        if len(examples) == 1 or len(examples) % 1_000 == 0:
            print(
                f"Prepared {len(examples):,}/{max_samples:,} preference "
                f"pairs (skipped {skipped_examples:,})."
            )
        if len(examples) >= max_samples:
            break

    if not examples:
        raise RuntimeError("No valid chosen/rejected preference pairs were produced")
    return examples, skipped_examples


def print_example_summary(example: PreferenceExample, tokenizer: Any) -> None:
    """Print prompt and completion prefixes to verify DPO alignment."""
    prompt_ids = example.chosen_input_ids[: example.prompt_token_count]
    chosen_ids = example.chosen_input_ids[example.prompt_token_count :]
    rejected_ids = example.rejected_input_ids[example.prompt_token_count :]

    print("\nFirst prepared preference pair")
    print("-" * 72)
    print(f"Prompt tokens: {example.prompt_token_count:,}")
    print(f"Chosen completion tokens: {example.chosen_token_count:,}")
    print(f"Rejected completion tokens: {example.rejected_token_count:,}")
    print("Prompt suffix:")
    print(tokenizer.decode(prompt_ids)[-1_000:])
    print("\nChosen completion prefix:")
    print(tokenizer.decode(chosen_ids)[:1_000])
    print("\nRejected completion prefix:")
    print(tokenizer.decode(rejected_ids)[:1_000])
    print("-" * 72)


def sequence_log_probabilities(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    reduction: str,
) -> torch.Tensor:
    """Calculate each response's conditional log probability.

    Causal logits at position t predict token t+1, so labels and masks are
    shifted left relative to logits. ``logsumexp`` plus selected logits avoids
    materializing a second full-vocabulary log-softmax tensor.

    The mathematically standard sequence log probability uses ``sum``:

        log pi(y|x) = sum_t log pi(y_t | x, y_<t).

    ``mean`` reproduces the legacy script's length-normalized variant.
    """
    shifted_logits = logits[:, :-1, :]
    shifted_labels = input_ids[:, 1:]
    shifted_mask = response_mask[:, 1:].to(dtype=shifted_logits.dtype)

    selected_logits = torch.gather(
        shifted_logits,
        dim=-1,
        index=shifted_labels.unsqueeze(-1),
    ).squeeze(-1)
    token_log_probabilities = (
        selected_logits - torch.logsumexp(shifted_logits, dim=-1)
    )
    masked_log_probabilities = token_log_probabilities * shifted_mask
    sequence_sums = masked_log_probabilities.sum(dim=-1)

    token_counts = shifted_mask.sum(dim=-1)
    if torch.any(token_counts == 0):
        raise RuntimeError("A DPO sequence has no scored completion tokens")
    if reduction == "mean":
        return sequence_sums / token_counts
    return sequence_sums


def resolve_device(requested_device: str) -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(requested_dtype: str, device: torch.device) -> torch.dtype:
    if requested_dtype == "float32":
        return torch.float32
    if requested_dtype == "bfloat16":
        if device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("This CUDA device does not support bfloat16")
        return torch.bfloat16
    if requested_dtype == "float16":
        if device.type != "cuda":
            raise RuntimeError("float16 training is supported only on CUDA here")
        return torch.float16
    if device.type == "cuda" and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.type == "cuda":
        return torch.float16
    return torch.float32


def create_lr_scheduler(
    optimizer: optim.Optimizer,
    total_optimizer_steps: int,
    warmup_ratio: float,
    minimum_lr_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by cosine decay."""
    warmup_steps = min(
        total_optimizer_steps,
        int(total_optimizer_steps * warmup_ratio),
    )

    def lr_multiplier(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return max((step + 1) / warmup_steps, 1e-8)
        decay_steps = max(total_optimizer_steps - warmup_steps, 1)
        progress = min(
            max((step - warmup_steps) / decay_steps, 0.0),
            1.0,
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_lr_ratio + (1.0 - minimum_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(
    policy_model: Any,
    reference_model: Any,
    data_loader: DataLoader,
    device: torch.device,
    model_dtype: torch.dtype,
    beta: float,
    log_probability_reduction: str,
    epochs: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    minimum_learning_rate: float,
    warmup_ratio: float,
    max_gradient_norm: float,
    log_every: int,
    log_path: Path,
) -> None:
    """Optimize the policy while keeping the reference model frozen."""
    batches_per_epoch = len(data_loader)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        (parameter for parameter in policy_model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = create_lr_scheduler(
        optimizer,
        total_optimizer_steps,
        warmup_ratio,
        minimum_learning_rate / learning_rate,
    )

    use_fp16_scaler = device.type == "cuda" and model_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    use_autocast = device.type == "cuda" and model_dtype in {
        torch.float16,
        torch.bfloat16,
    }

    policy_model.train()
    reference_model.eval()
    reference_model.requires_grad_(False)
    optimizer.zero_grad(set_to_none=True)

    optimizer_step = 0
    recent_metrics: list[dict[str, float]] = []
    training_started = time.time()

    for epoch in range(epochs):
        for batch_index, batch in enumerate(data_loader):
            index_in_window = batch_index % gradient_accumulation_steps
            if index_in_window == 0:
                remaining_batches = batches_per_epoch - batch_index
                accumulation_window_size = min(
                    gradient_accumulation_steps,
                    remaining_batches,
                )

            pair_count = int(batch.pop("pair_count").item())
            batch = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }

            # Calculate frozen reference log probabilities first and release its
            # large vocabulary logits before building the policy autograd graph.
            with torch.no_grad(), torch.autocast(
                device_type=device.type,
                dtype=model_dtype,
                enabled=use_autocast,
            ):
                reference_logits = reference_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                reference_log_probabilities = sequence_log_probabilities(
                    reference_logits,
                    batch["input_ids"],
                    batch["response_mask"],
                    log_probability_reduction,
                )
                del reference_logits

            with torch.autocast(
                device_type=device.type,
                dtype=model_dtype,
                enabled=use_autocast,
            ):
                policy_logits = policy_model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    use_cache=False,
                ).logits
                policy_log_probabilities = sequence_log_probabilities(
                    policy_logits,
                    batch["input_ids"],
                    batch["response_mask"],
                    log_probability_reduction,
                )
                del policy_logits

                policy_chosen, policy_rejected = torch.split(
                    policy_log_probabilities,
                    pair_count,
                )
                reference_chosen, reference_rejected = torch.split(
                    reference_log_probabilities,
                    pair_count,
                )

                # DPO compares how much the policy moved from the reference on
                # chosen versus rejected completions.
                chosen_log_ratios = policy_chosen - reference_chosen
                rejected_log_ratios = policy_rejected - reference_rejected
                margins = beta * (chosen_log_ratios - rejected_log_ratios)

                # -log(sigmoid(margin)) is implemented as -logsigmoid for
                # numerical stability when margins have large magnitude.
                sample_losses = -F.logsigmoid(margins)
                unscaled_loss = sample_losses.mean()
                loss = unscaled_loss / accumulation_window_size

            scaler.scale(loss).backward()

            with torch.no_grad():
                chosen_rewards = beta * chosen_log_ratios
                rejected_rewards = beta * rejected_log_ratios
                reward_margins = chosen_rewards - rejected_rewards
                preference_accuracy = (
                    chosen_rewards > rejected_rewards
                ).float().mean()
                recent_metrics.append(
                    {
                        "loss": float(unscaled_loss.item()),
                        "chosen_log_probability": float(policy_chosen.mean().item()),
                        "rejected_log_probability": float(policy_rejected.mean().item()),
                        "chosen_reward": float(chosen_rewards.mean().item()),
                        "rejected_reward": float(rejected_rewards.mean().item()),
                        "reward_margin": float(reward_margins.mean().item()),
                        "preference_accuracy": float(preference_accuracy.item()),
                    }
                )

            is_window_end = index_in_window + 1 == accumulation_window_size
            if not is_window_end:
                continue

            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy_model.parameters(),
                max_norm=max_gradient_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % log_every == 0 or optimizer_step == total_optimizer_steps:
                averaged = {
                    key: float(np.mean([item[key] for item in recent_metrics]))
                    for key in recent_metrics[0]
                }
                record = {
                    "event": "training_step",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "optimizer_step": optimizer_step,
                    "total_optimizer_steps": total_optimizer_steps,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": time.time() - training_started,
                    **averaged,
                }
                append_jsonl(log_path, record)
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"step {optimizer_step}/{total_optimizer_steps} | "
                    f"loss {averaged['loss']:.4f} | "
                    f"margin {averaged['reward_margin']:.4f} | "
                    f"accuracy {averaged['preference_accuracy']:.1%} | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"grad norm {float(gradient_norm):.3f}"
                )
                recent_metrics.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPO on chosen/rejected conversational preference pairs."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_SFT_MODEL_PATH)
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=None,
        help="Defaults to --model-path; useful for preprocessing validation.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--data-source",
        choices=["local", "huggingface", "path"],
        default="local",
        help=(
            "local: downloaded UltraFeedback directory; huggingface: Hub dataset; "
            "path: custom --data-path."
        ),
    )
    parser.add_argument(
        "--local-data-dir",
        type=Path,
        default=DEFAULT_LOCAL_DATA_DIR,
        help="Downloaded ultrafeedback_binarized repository used by local mode.",
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=None,
        help="Custom preference file/directory used by --data-source path.",
    )
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET_NAME)
    parser.add_argument("--dataset-split", default="train_prefs")
    parser.add_argument("--chosen-column", default="chosen")
    parser.add_argument("--rejected-column", default="rejected")
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rl-huggingface-cache"),
    )
    parser.add_argument("--max-samples", type=int, default=30_000)
    parser.add_argument("--max-length", type=int, default=1_700)
    parser.add_argument("--max-prompt-length", type=int, default=1_024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument(
        "--log-probability-reduction",
        choices=["sum", "mean"],
        default="sum",
        help="sum is standard sequence log-probability; mean matches legacy code.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--minimum-learning-rate", type=float, default=1e-7)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--system-prompt", default="You are a helpful assistant.")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate preference pairs without loading either model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> Path:
    tokenizer_path = args.tokenizer_path or args.model_path
    if not tokenizer_path.is_dir():
        raise FileNotFoundError(f"Tokenizer directory does not exist: {tokenizer_path}")
    if not args.preprocess_only and not args.model_path.is_dir():
        raise FileNotFoundError(
            f"SFT model directory does not exist: {args.model_path}. "
            "Run 1-SFT-modern.py first or pass --model-path."
        )
    if args.data_source == "local" and not args.local_data_dir.exists():
        raise FileNotFoundError(
            f"Downloaded preference-data directory does not exist: "
            f"{args.local_data_dir}"
        )
    if args.data_source == "path" and args.data_path is None:
        raise ValueError("--data-path is required when --data-source path")
    if args.data_source == "path" and not args.data_path.exists():
        raise FileNotFoundError(f"Preference data path does not exist: {args.data_path}")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.max_length < 16:
        raise ValueError("--max-length must be at least 16")
    if not 1 <= args.max_prompt_length < args.max_length:
        raise ValueError("--max-prompt-length must be >=1 and < --max-length")
    if args.batch_size < 1 or args.gradient_accumulation_steps < 1:
        raise ValueError("Batch size and accumulation steps must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.beta <= 0.0:
        raise ValueError("--beta must be positive")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError("Minimum learning rate must be positive and <= maximum")
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.max_gradient_norm <= 0.0 or args.log_every < 1:
        raise ValueError("Gradient norm and log frequency must be positive")
    return tokenizer_path


def main() -> None:
    args = parse_args()
    tokenizer_path = validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad nor EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    rows = load_data(
        data_source=args.data_source,
        local_data_dir=args.local_data_dir,
        data_path=args.data_path,
        dataset_name=args.dataset_name,
        dataset_split=args.dataset_split,
        cache_dir=args.cache_dir,
    )
    examples, skipped_examples = prepare_examples(
        tokenizer=tokenizer,
        rows=rows,
        chosen_column=args.chosen_column,
        rejected_column=args.rejected_column,
        system_prompt=args.system_prompt,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_prompt_length=args.max_prompt_length,
    )
    print_example_summary(examples[0], tokenizer)
    print(
        f"Prepared {len(examples):,} valid pairs; "
        f"skipped {skipped_examples:,}."
    )

    if args.preprocess_only:
        print("Preference preprocessing completed; model weights were not loaded.")
        return

    device = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    print(f"Training device: {device}")
    print(f"Model dtype: {model_dtype}")
    if device.type == "cpu":
        print(
            "WARNING: DPO loads two 0.6B models and is impractically slow on "
            "CPU. Use a CUDA training node when available."
        )

    # Both models begin from the same SFT checkpoint. Only policy_model is
    # optimized; reference_model remains frozen for the complete DPO run.
    policy_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        local_files_only=True,
    )
    reference_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        local_files_only=True,
    )
    for model in (policy_model, reference_model):
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False
        model.to(device)

    reference_model.requires_grad_(False)
    reference_model.eval()

    if args.gradient_checkpointing:
        policy_model.gradient_checkpointing_enable()
        if hasattr(policy_model, "enable_input_require_grads"):
            policy_model.enable_input_require_grads()

    dataset = PreferenceDataset(examples)
    generator = torch.Generator().manual_seed(args.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=PreferenceCollator(tokenizer.pad_token_id),
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / "training-log.jsonl"
    append_jsonl(
        log_path,
        {
            "event": "configuration",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "arguments": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "prepared_examples": len(examples),
            "skipped_examples": skipped_examples,
        },
    )

    train(
        policy_model=policy_model,
        reference_model=reference_model,
        data_loader=data_loader,
        device=device,
        model_dtype=model_dtype,
        beta=args.beta,
        log_probability_reduction=args.log_probability_reduction,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_gradient_norm=args.max_gradient_norm,
        log_every=args.log_every,
        log_path=log_path,
    )

    policy_model.config.use_cache = True
    policy_model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved DPO policy and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()

#  - local — your downloaded ./data/ultrafeedback_binarized dataset; now the default.
#   - huggingface — streams/downloads from Hugging Face.
#   - path — loads a custom Parquet, JSON, or JSONL file.
#
#   Run with your local dataset:
#
#   uv run python rl_learning_demo/day05_SFT_DPO/2-DPO-modern.py \
#       --data-source local \
#       --device cuda
#
#   Use Hugging Face instead:
#
#   uv run python rl_learning_demo/day05_SFT_DPO/2-DPO-modern.py \
#       --data-source huggingface \
#       --device cuda
#
#   Use a custom file:
#
#   uv run python rl_learning_demo/day05_SFT_DPO/2-DPO-modern.py \
#       --data-source path \
#       --data-path /path/to/preferences.jsonl \
#       --device cuda
#
#   The local loader correctly selects only:
#
#   data/ultrafeedback_binarized/data/train_prefs-00000-of-00001.parquet
#
#   It will not accidentally combine train_sft, train_gen, or test splits. I validated
#   preprocessing against your actual downloaded UltraFeedback data and the Qwen
#   tokenizer. The legacy rl_learning_demo/day05_SFT_DPO/2-DPO.py remains untouched.