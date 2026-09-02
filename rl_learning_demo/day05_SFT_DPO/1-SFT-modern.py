"""Supervised fine-tune Qwen3-0.6B-Base on the local UltraChat 200k data.

This is the modern, local-data rewrite of ``1-SFT.py``. The legacy file is
left untouched.

The training objective is assistant-only causal language modelling. Given a
tokenized conversation x_0, ..., x_(T-1), the model predicts each next token,
but loss is retained only for tokens belonging to assistant messages:

    L_SFT(theta) = -(1/N) * sum_t mask_t
                   * log p_theta(x_t | x_0, ..., x_(t-1)).

``mask_t`` is 1 for assistant response tokens and 0 for system text, user text,
and padding. In PyTorch/Transformers, ignored labels are represented by -100.

Important design choices
------------------------
* Read the three local ``train_sft`` Parquet shards; do not download data.
* Use the tokenizer's chat template and tokenizer-derived ChatML delimiters;
  do not hardcode model-specific token IDs or fixed ``+3`` offsets.
* Use ``AutoModelForCausalLM(...).to(device)`` because ``accelerate`` is not a
  project dependency and is required by ``device_map="auto"``.
* Dynamically pad each batch and ignore padding in both attention and loss.
* Support gradient accumulation, gradient clipping, warmup, cosine decay,
  gradient checkpointing, reproducible shuffling, JSONL logging, and a
  preprocessing-only validation mode.

Example commands
----------------
Validate local data and assistant masks without loading model weights:

    uv run python rl_learning_demo/day05_SFT_DPO/1-SFT-modern.py \
        --preprocess-only --max-samples 8

Run the default 5,000-sample training job:

    uv run python rl_learning_demo/day05_SFT_DPO/1-SFT-modern.py
"""

import argparse
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.optim as optim
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "ultrachat_200k" / "data"
DEFAULT_MODEL_PATH = Path(
    "/home/mh1f25/scratch/llms_model/Qwen3-0.6B-Base"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "outputs" / "Qwen3-0.6B-Base-UltraChat-SFT"
)
IGNORE_INDEX = -100


@dataclass
class TrainingExample:
    """One tokenized conversation and its assistant-only labels."""

    input_ids: list[int]
    labels: list[int]
    assistant_token_count: int


class TokenizedConversationDataset(Dataset[TrainingExample]):
    """A small in-memory collection of already-tokenized conversations."""

    def __init__(self, examples: list[TrainingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> TrainingExample:
        return self.examples[index]


class AssistantOnlyCollator:
    """Dynamically right-pad examples and preserve ignored label positions.

    Padding to the longest sequence in each batch is cheaper than padding every
    sample to ``max_length`` during preprocessing.
    """

    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(
        self,
        examples: list[TrainingExample],
    ) -> dict[str, torch.Tensor]:
        input_tensors = [
            torch.tensor(example.input_ids, dtype=torch.long)
            for example in examples
        ]
        label_tensors = [
            torch.tensor(example.labels, dtype=torch.long)
            for example in examples
        ]
        attention_tensors = [
            torch.ones(len(example.input_ids), dtype=torch.long)
            for example in examples
        ]

        input_ids = pad_sequence(
            input_tensors,
            batch_first=True,
            padding_value=self.pad_token_id,
        )
        labels = pad_sequence(
            label_tensors,
            batch_first=True,
            padding_value=IGNORE_INDEX,
        )
        attention_mask = pad_sequence(
            attention_tensors,
            batch_first=True,
            padding_value=0,
        )

        # Build attention from sequence lengths instead of ``input_ids !=
        # pad_token_id``. Some tokenizers use the EOS token as padding; a real
        # occurrence of that ID must not accidentally be treated as padding.
        # Padding labels are -100, so padding also contributes no loss.

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


def find_subsequence(
    sequence: list[int],
    pattern: list[int],
    start: int,
) -> int | None:
    """Return the first pattern position at or after ``start``."""
    if not pattern:
        raise ValueError("The token pattern must not be empty")

    final_start = len(sequence) - len(pattern)
    for index in range(start, final_start + 1):
        if sequence[index : index + len(pattern)] == pattern:
            return index
    return None


def build_assistant_labels(
    input_ids: list[int],
    assistant_header_ids: list[int],
    message_end_ids: list[int],
) -> tuple[list[int], int]:
    """Create labels for assistant content and message-end tokens only.

    Qwen's chat template renders an assistant message as approximately:

        <|im_start|>assistant\n RESPONSE <|im_end|>\n

    The delimiter strings are tokenized at runtime, so this function does not
    assume that a delimiter is one token or that an assistant header is always
    three tokens. Tokens in the header are ignored; response tokens and the
    closing ``<|im_end|>`` sequence are supervised.

    If truncation cuts off the final ``<|im_end|>``, available response tokens
    from the final assistant header to the sequence end remain supervised.
    """
    labels = [IGNORE_INDEX] * len(input_ids)
    assistant_token_count = 0
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
        end_start = find_subsequence(
            input_ids,
            message_end_ids,
            answer_start,
        )

        if end_start is None:
            # The response was truncated. Supervise every available response
            # token, then stop because there can be no later complete message.
            answer_end = len(input_ids)
            search_position = len(input_ids)
        else:
            # Include the message-end token sequence. Learning when to stop is
            # part of learning a useful assistant response.
            answer_end = end_start + len(message_end_ids)
            search_position = answer_end

        for position in range(answer_start, answer_end):
            labels[position] = input_ids[position]
            assistant_token_count += 1

        if end_start is None:
            break

    return labels, assistant_token_count


def normalize_messages(
    raw_messages: list[dict[str, Any]],
    system_prompt: str,
) -> list[dict[str, str]]:
    """Copy valid messages and prepend a system prompt when one is absent."""
    messages: list[dict[str, str]] = []

    for message in raw_messages:
        role = str(message.get("role", ""))
        content = message.get("content")
        if role not in {"system", "user", "assistant"}:
            continue
        if content is None:
            continue
        messages.append({"role": role, "content": str(content)})

    if not messages or messages[0]["role"] != "system":
        messages.insert(
            0,
            {"role": "system", "content": system_prompt},
        )

    return messages


def tokenize_conversation(
    raw_messages: list[dict[str, Any]],
    tokenizer: Any,
    max_length: int,
    system_prompt: str,
    assistant_header_ids: list[int],
    message_end_ids: list[int],
) -> TrainingExample | None:
    """Apply the chat template and build aligned assistant-only labels."""
    messages = normalize_messages(raw_messages, system_prompt)
    if not any(message["role"] == "assistant" for message in messages):
        return None

    # Transformers 5 returns BatchEncoding by default. Requesting return_dict
    # explicitly makes the API expectation clear and avoids treating that
    # object as the flat list returned by older Transformers versions.
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        truncation=True,
        max_length=max_length,
        return_dict=True,
    )
    input_ids = list(encoded["input_ids"])

    labels, assistant_token_count = build_assistant_labels(
        input_ids,
        assistant_header_ids,
        message_end_ids,
    )
    if assistant_token_count == 0:
        # This usually means right truncation removed every assistant response.
        # Keeping the example would produce a NaN loss because every label is
        # ignored, so filter it before batching.
        return None

    return TrainingExample(
        input_ids=input_ids,
        labels=labels,
        assistant_token_count=assistant_token_count,
    )


def discover_training_shards(data_dir: Path) -> list[Path]:
    """Find only local UltraChat ``train_sft`` Parquet shards."""
    shards = sorted(data_dir.glob("train_sft-*.parquet"))
    if not shards:
        raise FileNotFoundError(
            f"No train_sft Parquet shards found under {data_dir}"
        )
    return shards


def prepare_examples(
    tokenizer: Any,
    data_dir: Path,
    cache_dir: Path,
    max_samples: int,
    max_length: int,
    system_prompt: str,
) -> tuple[list[TrainingExample], int]:
    """Stream local Parquet rows and tokenize up to ``max_samples`` examples."""
    shards = discover_training_shards(data_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    # ``streaming=True`` reads rows incrementally instead of materializing the
    # complete 200k-example dataset as another Arrow copy on disk or in RAM.
    rows = load_dataset(
        "parquet",
        data_files=[str(path) for path in shards],
        split="train",
        streaming=True,
        cache_dir=str(cache_dir),
    )

    # Derive delimiter IDs from the tokenizer. This replaces the legacy code's
    # hardcoded EOS ID and ``user_end + 3`` assistant-header assumption.
    assistant_header_ids = tokenizer.encode(
        "<|im_start|>assistant\n",
        add_special_tokens=False,
    )
    message_end_ids = tokenizer.encode(
        "<|im_end|>",
        add_special_tokens=False,
    )

    examples: list[TrainingExample] = []
    skipped_examples = 0

    for row in rows:
        example = tokenize_conversation(
            raw_messages=row["messages"],
            tokenizer=tokenizer,
            max_length=max_length,
            system_prompt=system_prompt,
            assistant_header_ids=assistant_header_ids,
            message_end_ids=message_end_ids,
        )

        if example is None:
            skipped_examples += 1
            continue

        examples.append(example)
        if len(examples) == 1 or len(examples) % 1_000 == 0:
            print(
                f"Prepared {len(examples):,}/{max_samples:,} examples "
                f"(skipped {skipped_examples:,})."
            )

        if len(examples) >= max_samples:
            break

    if not examples:
        raise RuntimeError("No examples with assistant tokens were produced")

    return examples, skipped_examples


def print_example_summary(
    example: TrainingExample,
    tokenizer: Any,
) -> None:
    """Show enough of one sample to verify masking without dumping all text."""
    supervised_ids = [
        token_id
        for token_id, label in zip(
            example.input_ids,
            example.labels,
            strict=True,
        )
        if label != IGNORE_INDEX
    ]

    print("\nFirst prepared example")
    print("-" * 72)
    print(f"Total tokens: {len(example.input_ids):,}")
    print(f"Supervised assistant tokens: {example.assistant_token_count:,}")
    print("Conversation prefix:")
    print(tokenizer.decode(example.input_ids)[:1_200])
    print("\nAssistant-only supervised text prefix:")
    print(tokenizer.decode(supervised_ids)[:1_200])
    print("-" * 72)


def resolve_device(requested_device: str) -> torch.device:
    """Resolve ``auto`` and fail clearly when unavailable CUDA was requested."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but torch.cuda.is_available() is False")
    return device


def resolve_dtype(
    requested_dtype: str,
    device: torch.device,
) -> torch.dtype:
    """Select a numerically appropriate model/autocast data type."""
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

    # Automatic choice: bf16 on supporting CUDA hardware and fp32 otherwise.
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
    """Create linear warmup followed by cosine learning-rate decay."""
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
        cosine_multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (
            minimum_lr_ratio
            + (1.0 - minimum_lr_ratio) * cosine_multiplier
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_multiplier)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    """Append one structured training record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def train(
    model: Any,
    data_loader: DataLoader,
    device: torch.device,
    model_dtype: torch.dtype,
    epochs: int,
    gradient_accumulation_steps: int,
    learning_rate: float,
    minimum_learning_rate: float,
    warmup_ratio: float,
    max_gradient_norm: float,
    log_every: int,
    log_path: Path,
) -> None:
    """Run assistant-only causal-LM SFT with gradient accumulation."""
    batches_per_epoch = len(data_loader)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / gradient_accumulation_steps
    )
    total_optimizer_steps = optimizer_steps_per_epoch * epochs

    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
        betas=(0.9, 0.95),
        weight_decay=0.1,
    )
    scheduler = create_lr_scheduler(
        optimizer,
        total_optimizer_steps=total_optimizer_steps,
        warmup_ratio=warmup_ratio,
        minimum_lr_ratio=minimum_learning_rate / learning_rate,
    )

    # fp16 needs dynamic gradient scaling. bf16 has a wider exponent range and
    # normally does not. CPU training uses ordinary fp32 operations.
    use_fp16_scaler = device.type == "cuda" and model_dtype == torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16_scaler)
    use_autocast = device.type == "cuda" and model_dtype in {
        torch.float16,
        torch.bfloat16,
    }

    model.train()
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    recent_losses: list[float] = []
    training_started = time.time()

    append_jsonl(
        log_path,
        {
            "event": "training_start",
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs": epochs,
            "batches_per_epoch": batches_per_epoch,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "total_optimizer_steps": total_optimizer_steps,
            "device": str(device),
            "dtype": str(model_dtype),
        },
    )

    for epoch in range(epochs):
        for batch_index, batch in enumerate(data_loader):
            # Use the real size of the final partial accumulation window. This
            # prevents its gradient from being incorrectly divided by a larger
            # configured accumulation count.
            index_in_window = batch_index % gradient_accumulation_steps
            if index_in_window == 0:
                remaining_batches = batches_per_epoch - batch_index
                accumulation_window_size = min(
                    gradient_accumulation_steps,
                    remaining_batches,
                )

            batch = {
                name: tensor.to(device, non_blocking=True)
                for name, tensor in batch.items()
            }

            # AutoModelForCausalLM shifts logits and labels internally:
            # logits at position t predict labels at position t+1. Labels equal
            # to -100 are excluded from the cross-entropy mean.
            with torch.autocast(
                device_type=device.type,
                dtype=model_dtype,
                enabled=use_autocast,
            ):
                outputs = model(**batch, use_cache=False)
                unscaled_loss = outputs.loss
                loss = unscaled_loss / accumulation_window_size

            scaler.scale(loss).backward()
            recent_losses.append(float(unscaled_loss.detach().item()))

            is_window_end = (
                index_in_window + 1 == accumulation_window_size
            )
            if not is_window_end:
                continue

            # Unscale before clipping so max_gradient_norm applies to the real
            # gradients rather than fp16-scaled gradients.
            scaler.unscale_(optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=max_gradient_norm,
            )
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_step += 1

            if optimizer_step % log_every == 0 or optimizer_step == total_optimizer_steps:
                average_loss = float(np.mean(recent_losses))
                current_lr = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - training_started
                record = {
                    "event": "training_step",
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch + 1,
                    "optimizer_step": optimizer_step,
                    "total_optimizer_steps": total_optimizer_steps,
                    "average_loss": average_loss,
                    "learning_rate": current_lr,
                    "gradient_norm": float(gradient_norm),
                    "elapsed_seconds": elapsed,
                }
                append_jsonl(log_path, record)
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"step {optimizer_step}/{total_optimizer_steps} | "
                    f"loss {average_loss:.4f} | "
                    f"lr {current_lr:.2e} | "
                    f"grad norm {float(gradient_norm):.3f} | "
                    f"elapsed {elapsed / 60:.1f} min"
                )
                recent_losses.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assistant-only SFT on local UltraChat 200k Parquet data."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("/tmp/rl-huggingface-cache"),
    )
    parser.add_argument("--max-samples", type=int, default=5_000)
    parser.add_argument("--max-length", type=int, default=2_500)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--minimum-learning-rate", type=float, default=2e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-gradient-norm", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, cuda, or a device such as cuda:1",
    )
    parser.add_argument(
        "--dtype",
        choices=["auto", "float32", "bfloat16", "float16"],
        default="auto",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Trade additional compute for lower activation memory.",
    )
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Validate local data and masks without loading/training the model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    """Fail before expensive loading when a configuration is invalid."""
    if not args.model_path.is_dir():
        raise FileNotFoundError(f"Model directory does not exist: {args.model_path}")
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.data_dir}")
    if args.max_samples < 1:
        raise ValueError("--max-samples must be at least 1")
    if args.max_length < 16:
        raise ValueError("--max-length must be at least 16")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be at least 1")
    if args.gradient_accumulation_steps < 1:
        raise ValueError("--gradient-accumulation-steps must be at least 1")
    if args.epochs < 1:
        raise ValueError("--epochs must be at least 1")
    if args.learning_rate <= 0.0:
        raise ValueError("--learning-rate must be positive")
    if not 0.0 < args.minimum_learning_rate <= args.learning_rate:
        raise ValueError(
            "--minimum-learning-rate must be positive and <= --learning-rate"
        )
    if not 0.0 <= args.warmup_ratio <= 1.0:
        raise ValueError("--warmup-ratio must be between 0 and 1")
    if args.max_gradient_norm <= 0.0:
        raise ValueError("--max-gradient-norm must be positive")
    if args.log_every < 1:
        raise ValueError("--log-every must be at least 1")


def main() -> None:
    args = parse_args()
    validate_args(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    shards = discover_training_shards(args.data_dir)
    print("Local UltraChat train_sft shards:")
    for shard in shards:
        print(f"  - {shard}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise RuntimeError("Tokenizer has neither a pad token nor an EOS token")
        tokenizer.pad_token = tokenizer.eos_token

    examples, skipped_examples = prepare_examples(
        tokenizer=tokenizer,
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        max_samples=args.max_samples,
        max_length=args.max_length,
        system_prompt=args.system_prompt,
    )
    print_example_summary(examples[0], tokenizer)
    print(
        f"Prepared {len(examples):,} usable examples; "
        f"skipped {skipped_examples:,}."
    )

    if args.preprocess_only:
        print("Preprocessing validation completed; model weights were not loaded.")
        return

    device = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)
    print(f"Training device: {device}")
    print(f"Model dtype: {model_dtype}")
    if device.type == "cpu":
        print(
            "WARNING: full fine-tuning a 0.6B model on CPU will be very slow. "
            "Use a CUDA training node when available."
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=model_dtype,
        local_files_only=True,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.use_cache = False

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        # Required by some model classes when checkpointing all transformer
        # blocks during full-parameter training.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    model.to(device)

    dataset = TokenizedConversationDataset(examples)
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=AssistantOnlyCollator(tokenizer.pad_token_id),
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
            "first_example": asdict(examples[0]),
        },
    )

    train(
        model=model,
        data_loader=data_loader,
        device=device,
        model_dtype=model_dtype,
        epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        minimum_learning_rate=args.minimum_learning_rate,
        warmup_ratio=args.warmup_ratio,
        max_gradient_norm=args.max_gradient_norm,
        log_every=args.log_every,
        log_path=log_path,
    )

    # Save one explicit output only. The legacy script wrote two differently
    # named copies, one of which referenced the wrong base-model name.
    model.config.use_cache = True
    model.save_pretrained(args.output_dir, safe_serialization=True)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved SFT model and tokenizer to: {args.output_dir}")


if __name__ == "__main__":
    main()

# The original rl_learning_demo/day05_SFT_DPO/1-SFT.py was left untouched.
#
#   Key improvements:
#
#   - Reads all three local train_sft Parquet shards
#   - Does not download UltraChat through ModelScope
#   - Supports Transformers 5’s current chat-template API
#   - Constructs assistant-only labels without hardcoded token IDs or offsets
#   - Uses -100 labels for system, user, and padding tokens
#   - Handles truncated assistant responses
#   - Dynamically pads each batch
#   - Supports gradient accumulation and correct final partial accumulation
#   - Uses warmup plus cosine learning-rate decay
#   - Supports bf16/fp16 autocast and fp16 gradient scaling
#   - Includes gradient clipping and checkpointing
#   - Writes structured JSONL training logs
#   - Saves one clearly named model output
#   - Contains detailed theory and implementation comments
#
#   First validate the data without loading the model:
#
#   uv run python rl_learning_demo/day05_SFT_DPO/1-SFT-modern.py \
#       --preprocess-only \
#       --max-samples 8
#
#   Then train on CUDA:
#
#   uv run python rl_learning_demo/day05_SFT_DPO/1-SFT-modern.py \
#       --device cuda
#
#   The current session reports torch.cuda.is_available() == False. Running full 0.6B fine-tuning here
#   would fall back to CPU and be extremely slow, so I validated preprocessing rather than launching
#   training.