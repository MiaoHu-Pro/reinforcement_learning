"""Educational, fixed version of ``code-2.py`` for Qwen3 SFT.

This script keeps the original program's step-by-step training style while
fixing its path, tokenization, masking, device, padding, numerical-stability,
gradient-accumulation, scheduling, and data-order issues.

The training objective is assistant-only causal language modelling:

    L(theta) = -mean(log p_theta(x_t | x_0, ..., x_(t-1)))

Only assistant response tokens (including ``<|im_end|>``) participate in the
mean. System text, user text, assistant headers, and padding do not.

Useful commands:

    # Check local data, tokenization, and masks without loading model weights.
    uv run python code-2-fixed.py --preprocess-only --max-samples 8

    # Run the default 5,000-example training job.
    uv run python code-2-fixed.py --device cuda
"""

import argparse
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import datasets
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


# ``code-2-fixed.py`` is under:
# reinforcement_learning/rl_learning_demo/day05_SFT_DPO/
# Therefore parents[2] is the reinforcement_learning project root. Deriving
# paths from __file__ makes them independent of the terminal's current folder.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass
class SFTConfig:
    """All important experiment settings in one visible place."""

    model_path: Path = Path(
        "~/scratch/llms_model/Qwen3-0.6B"
    ).expanduser()
    data_dir: Path = PROJECT_ROOT / "data" / "ultrachat_200k" / "data"
    output_dir: Path = (
        Path("~/scratch/llms_model/post_trained_models/")
        .expanduser()
        / "Qwen3-0.6B-SFT-fixed"
    )
    cache_dir: Path = Path("/tmp/rl-huggingface-cache")
    log_path: Path = PROJECT_ROOT / "logs" / "Qwen3-0.6B-SFT-fixed.log"

    max_length: int = 2500
    max_samples: int = 200_000
    batch_size: int = 2
    gradient_accumulation_steps: int = 8
    epochs: int = 1

    log_iter: int = 100
    max_lr: float = 2e-5
    min_lr: float = 2e-6
    warmup_ratio: float = 0.03
    max_gradient_norm: float = 1.0
    seed: int = 42
    system_prompt: str = "You are a helpful assistant"
    gradient_checkpointing: bool = True


def parse_args() -> argparse.Namespace:
    """Keep common experiments configurable without editing source code."""
    parser = argparse.ArgumentParser(
        description="Assistant-only SFT for local UltraChat 200k data."
    )
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--device",
        choices=["auto", "cuda", "cpu"],
        default="auto",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate data/tokenization/masks without loading the model.",
    )
    return parser.parse_args()


def apply_command_line_overrides(
    config: SFTConfig,
    args: argparse.Namespace,
) -> None:
    """Apply only options explicitly supplied by the user."""
    if args.model_path is not None:
        config.model_path = args.model_path.expanduser()
    if args.data_dir is not None:
        config.data_dir = args.data_dir.expanduser()
    if args.output_dir is not None:
        config.output_dir = args.output_dir.expanduser()
    if args.max_samples is not None:
        config.max_samples = args.max_samples
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.epochs is not None:
        config.epochs = args.epochs


def validate_config(config: SFTConfig, preprocess_only: bool) -> None:
    """Fail early with understandable path and numeric-setting errors."""
    if not config.model_path.is_dir():
        raise FileNotFoundError(f"Model/tokenizer directory missing: {config.model_path}")
    if not config.data_dir.is_dir():
        raise FileNotFoundError(f"UltraChat data directory missing: {config.data_dir}")
    if not list(config.data_dir.glob("train_sft-*.parquet")):
        raise FileNotFoundError(
            f"No train_sft Parquet shards found in {config.data_dir}"
        )
    if config.max_samples < 1 or config.batch_size < 1 or config.epochs < 1:
        raise ValueError("max_samples, batch_size, and epochs must be positive")
    if config.max_length < 16:
        raise ValueError("max_length must be at least 16")
    if config.gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    if not 0.0 <= config.warmup_ratio <= 1.0:
        raise ValueError("warmup_ratio must be between 0 and 1")
    if not 0.0 < config.min_lr <= config.max_lr:
        raise ValueError("Learning rates must satisfy 0 < min_lr <= max_lr")

    # During preprocessing the tokenizer is needed, but model weights are not.
    if not preprocess_only and not config.output_dir.parent.exists():
        config.output_dir.parent.mkdir(parents=True, exist_ok=True)


def choose_device(requested_device: str) -> torch.device:
    """Select a real device instead of unconditionally assuming CUDA."""
    if requested_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but CUDA is unavailable. "
                "Use a GPU session or pass --device cpu for a slow test."
            )
        return torch.device("cuda")
    if requested_device == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_subsequence(
    sequence: list[int],
    pattern: list[int],
    start: int,
) -> int | None:
    """Find the first occurrence of a multi-token delimiter."""
    if not pattern:
        raise ValueError("The token pattern cannot be empty")
    final_start = len(sequence) - len(pattern)
    for position in range(start, final_start + 1):
        if sequence[position : position + len(pattern)] == pattern:
            return position
    return None


def create_answer_mask_for_sequence(
    input_ids: list[int],
    assistant_header_ids: list[int],
    message_end_ids: list[int],
) -> list[int]:
    """Mark assistant responses using tokenizer-derived delimiter sequences.

    Approximate Qwen ChatML layout:

        <|im_start|>assistant\n RESPONSE <|im_end|>\n

    Unlike the original ``user_end + 3`` logic, this function does not assume
    fixed header lengths or that ``<|im_end|>`` is exactly one token. The
    closing message-end token is deliberately supervised: learning when to
    stop is part of learning an assistant response.
    """
    answer_mask = [0] * len(input_ids)
    search_position = 0

    while True:
        header_start = find_subsequence(
            input_ids,
            assistant_header_ids,
            search_position,
        )
        if header_start is None:
            break

        answer_start = header_start + len(assistant_header_ids)
        message_end_start = find_subsequence(
            input_ids,
            message_end_ids,
            answer_start,
        )

        if message_end_start is None:
            # Right truncation cut the response. Train on available response
            # tokens, then stop because no later complete message can exist.
            answer_end = len(input_ids)
        else:
            answer_end = message_end_start + len(message_end_ids)

        for position in range(answer_start, answer_end):
            answer_mask[position] = 1

        if message_end_start is None:
            break
        search_position = answer_end

    return answer_mask


def normalize_messages(
    raw_messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Copy messages and add one system prompt without mutating the dataset."""
    messages: list[dict[str, str]] = []
    for message in raw_messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role in {"system", "user", "assistant"} and content is not None:
            messages.append({"role": role, "content": str(content)})

    if not messages or messages[0]["role"] != "system":
        messages.insert(
            0,
            {"content": system_prompt, "role": "system"},
        )
    return messages


def tokenize_and_format(
    raw_messages: list[dict[str, Any]],
    tokenizer: Any,
    config: SFTConfig,
) -> list[int]:
    """Apply Qwen's chat template and return a plain Python token-ID list."""
    messages = normalize_messages(raw_messages, config.system_prompt)

    # Transformers 5 can otherwise return tokenizers.Encoding. Requesting a
    # mapping explicitly and extracting input_ids prevents torch.tensor(...)
    # from failing later with "Could not infer dtype of tokenizers.Encoding".
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=config.max_length,
        return_dict=True,
    )
    return list(encoded["input_ids"])


def load_and_tokenize_data(
    tokenizer: Any,
    config: SFTConfig,
    assistant_header_ids: list[int],
    message_end_ids: list[int],
) -> tuple[list[list[int]], int]:
    """Stream, validate, tokenize, and reproducibly shuffle local SFT rows."""
    train_shards = sorted(config.data_dir.glob("train_sft-*.parquet"))
    config.cache_dir.mkdir(parents=True, exist_ok=True)

    # Load only train_sft. The local repository also contains train_gen and
    # test files, which must not accidentally be mixed into SFT training.
    training_rows = datasets.load_dataset(
        "parquet",
        data_files=[str(path) for path in train_shards],
        split="train",
        streaming=True,
        cache_dir=str(config.cache_dir),
    )

    # Streaming shuffle avoids always selecting the dataset's first 5,000
    # records. Its finite buffer is deterministic for this seed.
    training_rows = training_rows.shuffle(
        seed=config.seed,
        buffer_size=max(10_000, config.max_samples * 2),
    )

    chosen_input_ids_list: list[list[int]] = []
    skipped_examples = 0

    for row in training_rows:
        input_ids = tokenize_and_format(row["messages"], tokenizer, config)
        answer_mask = create_answer_mask_for_sequence(
            input_ids,
            assistant_header_ids,
            message_end_ids,
        )

        # Discard a conversation when truncation removed every assistant token.
        # Otherwise its masked mean loss would divide by zero.
        if not any(answer_mask):
            skipped_examples += 1
            continue

        chosen_input_ids_list.append(input_ids)

        if len(chosen_input_ids_list) == 1:
            print("First formatted training conversation:")
            print("-" * 70)
            print(tokenizer.decode(input_ids))
            print("-" * 70)

        if len(chosen_input_ids_list) % 1000 == 0:
            print(
                f"已处理 {len(chosen_input_ids_list):,} 条有效数据，"
                f"跳过 {skipped_examples:,} 条"
            )

        if len(chosen_input_ids_list) >= config.max_samples:
            break

    if not chosen_input_ids_list:
        raise RuntimeError("No valid assistant-containing training examples found")

    # Shuffle the prepared list too, so batch composition is reproducible and
    # is not tied to the stream's final order.
    random.Random(config.seed).shuffle(chosen_input_ids_list)
    return chosen_input_ids_list, skipped_examples


def collate_batch(
    sequences: list[list[int]],
    pad_token_id: int,
    assistant_header_ids: list[int],
    message_end_ids: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Right-pad one batch and construct attention and assistant masks."""
    max_sequence_length = max(len(sequence) for sequence in sequences)
    batch_size = len(sequences)

    batch_input_tensor = torch.full(
        (batch_size, max_sequence_length),
        fill_value=pad_token_id,
        dtype=torch.long,
    )
    attention_mask = torch.zeros(
        (batch_size, max_sequence_length),
        dtype=torch.long,
    )
    assistant_answer_mask = torch.zeros(
        (batch_size, max_sequence_length),
        dtype=torch.bool,
    )

    for sample_index, sequence in enumerate(sequences):
        sequence_length = len(sequence)
        batch_input_tensor[sample_index, :sequence_length] = torch.tensor(
            sequence,
            dtype=torch.long,
        )
        attention_mask[sample_index, :sequence_length] = 1

        sequence_answer_mask = create_answer_mask_for_sequence(
            sequence,
            assistant_header_ids,
            message_end_ids,
        )
        assistant_answer_mask[sample_index, :sequence_length] = torch.tensor(
            sequence_answer_mask,
            dtype=torch.bool,
        )

    return batch_input_tensor, attention_mask, assistant_answer_mask


def learning_rate_for_step(
    optimizer_step: int,
    warmup_steps: int,
    total_optimizer_steps: int,
    max_lr: float,
    min_lr: float,
) -> float:
    """Linear warmup followed by cosine decay, measured in weight updates."""
    if warmup_steps > 0 and optimizer_step < warmup_steps:
        return max_lr * (optimizer_step + 1) / warmup_steps

    decay_steps = total_optimizer_steps - warmup_steps
    if decay_steps <= 1:
        return min_lr

    decay_position = optimizer_step - warmup_steps
    progress = min(max(decay_position / (decay_steps - 1), 0.0), 1.0)
    decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * decay


def log_call(
    log_path: Path,
    optimizer_step: int,
    average_loss: float,
    learning_rate: float,
) -> None:
    """Append one readable training-progress record."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log_file:
        log_file.write(
            f"time:{time.strftime('%Y-%m-%d %H:%M:%S')}, "
            f"optimizer_step:{optimizer_step}, "
            f"average_loss:{average_loss:.6f}, "
            f"learning_rate:{learning_rate:.8e}\n"
        )


def train(
    model: Any,
    tokenizer: Any,
    chosen_input_ids_list: list[list[int]],
    assistant_header_ids: list[int],
    message_end_ids: list[int],
    config: SFTConfig,
    device: torch.device,
    model_dtype: torch.dtype,
) -> list[float]:
    """Run the explicit assistant-only causal-language-model training loop."""
    batch_size = config.batch_size
    accumulation_steps = config.gradient_accumulation_steps
    batches_per_epoch = math.ceil(len(chosen_input_ids_list) / batch_size)
    updates_per_epoch = math.ceil(batches_per_epoch / accumulation_steps)
    total_optimizer_steps = updates_per_epoch * config.epochs
    warmup_steps = int(total_optimizer_steps * config.warmup_ratio)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.max_lr)

    # Gradient scaling is needed for float16. Bfloat16 has a wider exponent
    # range and normally does not require a scaler.
    use_gradient_scaler = device.type == "cuda" and model_dtype == torch.float16
    gradient_scaler = torch.amp.GradScaler(
        device="cuda",
        enabled=use_gradient_scaler,
    )
    use_autocast = device.type == "cuda"

    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)

    training_losses: list[float] = []
    optimizer_step = 0

    print("=" * 70)
    print(f"训练设备: {device}")
    print(f"模型 dtype: {model_dtype}")
    print(f"每轮批次数: {batches_per_epoch}")
    print(f"总权重更新次数: {total_optimizer_steps}")
    print(f"warmup 权重更新次数: {warmup_steps}")
    print("=" * 70)

    for epoch in range(config.epochs):
        # Use a different but reproducible order each epoch.
        random.Random(config.seed + epoch).shuffle(chosen_input_ids_list)

        for batch_idx in range(batches_per_epoch):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(chosen_input_ids_list))
            current_batch_sequences = chosen_input_ids_list[batch_start:batch_end]

            # ==================== 数据准备阶段 ====================
            (
                batch_input_tensor,
                full_attention_mask,
                full_assistant_mask,
            ) = collate_batch(
                current_batch_sequences,
                tokenizer.pad_token_id,
                assistant_header_ids,
                message_end_ids,
            )

            # ==================== 构建因果输入输出对 ====================
            # Original tokens: [x0, x1, x2, ..., xT]
            # Model inputs:    [x0, x1, x2, ..., x(T-1)]
            # Target labels:   [x1, x2, x3, ..., xT]
            model_inputs = batch_input_tensor[:, :-1].to(device)
            target_labels = batch_input_tensor[:, 1:].to(device)
            model_attention_mask = full_attention_mask[:, :-1].to(device)

            # Shift the mask exactly like the labels. A mask position now
            # describes the target token predicted at that position, removing
            # the original code's implicit and confusing off-by-one behavior.
            final_loss_mask = (
                full_assistant_mask[:, 1:]
                & full_attention_mask[:, 1:].bool()
            ).to(device)

            tokens_per_sample = final_loss_mask.sum(dim=-1)
            if torch.any(tokens_per_sample == 0):
                # Preprocessing filters these, so reaching this branch signals
                # a real mask bug rather than silently discarding gradients.
                raise RuntimeError("A prepared sample has no assistant target tokens")

            # ==================== 模型前向传播 ====================
            with torch.autocast(
                device_type=device.type,
                dtype=model_dtype if use_autocast else None,
                enabled=use_autocast,
            ):
                model_logits = model(
                    input_ids=model_inputs,
                    attention_mask=model_attention_mask,
                ).logits

                # ==================== 稳定的掩码损失 ====================
                # F.cross_entropy internally uses log_softmax and avoids the
                # unstable log(softmax(logits)) calculation from code-2.py.
                token_losses = F.cross_entropy(
                    model_logits.reshape(-1, model_logits.size(-1)),
                    target_labels.reshape(-1),
                    reduction="none",
                ).view_as(target_labels)

                masked_token_losses = token_losses * final_loss_mask
                sample_losses = (
                    masked_token_losses.sum(dim=-1)
                    / final_loss_mask.sum(dim=-1)
                )
                batch_average_loss = sample_losses.mean()

                # The last accumulation group can contain fewer microbatches,
                # and its last microbatch can contain fewer samples. Weight by
                # the real number of examples so every conversation contributes
                # equally to the accumulated gradient.
                group_start = (
                    batch_idx // accumulation_steps
                ) * accumulation_steps
                actual_group_size = min(
                    accumulation_steps,
                    batches_per_epoch - group_start,
                )
                group_example_start = group_start * batch_size
                group_example_end = min(
                    (group_start + actual_group_size) * batch_size,
                    len(chosen_input_ids_list),
                )
                examples_in_group = group_example_end - group_example_start
                scaled_loss = (
                    batch_average_loss
                    * len(current_batch_sequences)
                    / examples_in_group
                )

            # ==================== 反向传播 ====================
            gradient_scaler.scale(scaled_loss).backward()
            training_losses.append(float(batch_average_loss.detach().cpu()))

            is_accumulation_boundary = (
                (batch_idx + 1) % accumulation_steps == 0
                or (batch_idx + 1) == batches_per_epoch
            )

            if is_accumulation_boundary:
                # The scheduler counts actual optimizer updates, not batches.
                current_learning_rate = learning_rate_for_step(
                    optimizer_step,
                    warmup_steps,
                    total_optimizer_steps,
                    config.max_lr,
                    config.min_lr,
                )
                for parameter_group in optimizer.param_groups:
                    parameter_group["lr"] = current_learning_rate

                gradient_scaler.unscale_(optimizer)
                gradient_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    config.max_gradient_norm,
                )
                gradient_scaler.step(optimizer)
                gradient_scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1

                if (
                    optimizer_step % config.log_iter == 0
                    or optimizer_step == total_optimizer_steps
                ):
                    # Show losses from roughly the latest log interval.
                    recent_count = config.log_iter * accumulation_steps
                    recent_average_loss = float(
                        np.mean(training_losses[-recent_count:])
                    )
                    print(
                        f"⏰ {time.strftime('%Y-%m-%d %H:%M:%S')} | "
                        f"轮次 {epoch + 1}/{config.epochs} | "
                        f"权重更新 {optimizer_step}/{total_optimizer_steps} | "
                        f"平均损失 {recent_average_loss:.4f} | "
                        f"学习率 {current_learning_rate:.2e} | "
                        f"梯度范数 {float(gradient_norm):.3f}"
                    )
                    log_call(
                        config.log_path,
                        optimizer_step,
                        recent_average_loss,
                        current_learning_rate,
                    )

    return training_losses


def main() -> None:
    args = parse_args()
    config = SFTConfig()
    apply_command_line_overrides(config, args)
    validate_config(config, args.preprocess_only)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    # Validate an explicitly requested CUDA device before spending time
    # tokenizing thousands of examples. Preprocessing-only mode needs no model
    # device and therefore intentionally skips this check.
    device = None if args.preprocess_only else choose_device(args.device)

    # ==================== 加载分词器 ====================
    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path),
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither pad_token_id nor eos_token_id")
        tokenizer.pad_token = tokenizer.eos_token

    # Derive ChatML delimiters from the tokenizer instead of hardcoding token
    # IDs such as 151645 or assuming an assistant header has three tokens.
    assistant_header_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    message_end_ids = tokenizer.encode(
        "<|im_end|>",
        add_special_tokens=False,
    )

    # ==================== 准备训练数据 ====================
    chosen_input_ids_list, skipped_examples = load_and_tokenize_data(
        tokenizer,
        config,
        assistant_header_ids,
        message_end_ids,
    )
    print(
        f"数据准备完成: {len(chosen_input_ids_list):,} 条有效数据，"
        f"跳过 {skipped_examples:,} 条"
    )

    if args.preprocess_only:
        # Verify one mask visually without allocating the 0.6B model.
        validation_ids = chosen_input_ids_list[0]
        validation_mask = create_answer_mask_for_sequence(
            validation_ids,
            assistant_header_ids,
            message_end_ids,
        )
        selected_ids = [
            token_id
            for token_id, selected in zip(validation_ids, validation_mask)
            if selected
        ]
        print("第一条样本中参与 SFT loss 的 assistant tokens:")
        print("-" * 70)
        print(tokenizer.decode(selected_ids))
        print("-" * 70)
        print("预处理验证成功；未加载模型权重。")
        return

    # ==================== 选择设备并加载模型 ====================
    assert device is not None
    if device.type == "cuda":
        model_dtype = (
            torch.bfloat16
            if torch.cuda.is_bf16_supported()
            else torch.float16
        )
    else:
        model_dtype = torch.float32

    # A single explicit device avoids conflicts between device_map="auto" and
    # the original hardcoded model_inputs.to("cuda").
    model = AutoModelForCausalLM.from_pretrained(
        str(config.model_path),
        dtype=model_dtype,
        local_files_only=True,
    ).to(device)

    if config.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    # These settings affect later generation, not SFT loss. They are kept here
    # only so the saved checkpoint has convenient inference defaults.
    model.generation_config.do_sample = True
    model.generation_config.temperature = 0.7
    model.generation_config.top_p = 0.8
    model.generation_config.top_k = 20
    model.generation_config.repetition_penalty = 1.05
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # ==================== 开始训练 ====================
    training_losses = train(
        model,
        tokenizer,
        chosen_input_ids_list,
        assistant_header_ids,
        message_end_ids,
        config,
        device,
        model_dtype,
    )

    print("🎉 训练完成!")
    print(f"有效 microbatch 数: {len(training_losses)}")
    print(f"最后最多 100 个 microbatch 的平均损失: "
          f"{np.mean(training_losses[-100:]):.4f}")

    # ==================== 保存模型与分词器 ====================
    config.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"模型已保存到: {config.output_dir}")


if __name__ == "__main__":
    main()


# It retains the original educational, step-by-step style while fixing:
#
#   - Working-directory-independent model and dataset paths
#   - Transformers 5 Encoding compatibility
#   - Tokenizer-derived assistant boundaries instead of fixed offsets
#   - Explicit causal shifting of labels and assistant masks
#   - Dynamic padding and proper attention masks
#   - Stable F.cross_entropy() loss
#   - Automatic CPU/CUDA device selection
#   - Float16 gradient scaling and bfloat16 support
#   - Gradient clipping and checkpointing
#   - Correct partial gradient accumulation
#   - Learning-rate scheduling based on weight updates
#   - Deterministic dataset shuffling
#   - Configurable sample count, batch size and epochs
#   - Safe output and logging paths
#
#   Validate preprocessing and masking:
#
#   cd rl_learning_demo/day05_SFT_DPO
#
#   uv run python code-2-fixed.py \
#       --preprocess-only \
#       --max-samples 8
#
#   Start GPU training:
#
#   uv run python code-2-fixed.py --device cuda
