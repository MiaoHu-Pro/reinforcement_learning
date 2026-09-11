"""Train Qwen2.5-3B Base with an educational DeepSeek-R1-Zero-style recipe.

This is a faithful small-scale implementation of the central recipe, not a
claim to reproduce DeepSeek's model, data, compute scale, or final capability:

1. start from a BASE pretrained model (no reasoning SFT warm-up);
2. sample a group of responses for each verifiable problem;
3. score responses with rules, not a learned reward model;
4. use the group reward mean/std as the baseline (no critic);
5. snapshot old-policy token log-probabilities;
6. optimize a clipped GRPO importance-ratio objective;
7. constrain drift with a frozen base reference policy.
"""

import argparse
from dataclasses import asdict, dataclass
import gc
import json
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from countdown_task import CountdownDataset, reward_function
from grpo import (
    assign_group_advantages,
    generate_episodes,
    store_old_policy_log_probs,
    update_policy_grpo,
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent


@dataclass
class TrainingConfig:
    """All important experiment choices in one visible, serializable place."""

    # Strict R1-Zero-style initialization uses the Base checkpoint, not the
    # already instruction-tuned Qwen2.5-3B-Instruct checkpoint.
    model_path: Path = Path(
        "~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B"
    ).expanduser()
    data_file: Path = (
        PROJECT_ROOT
        / "rl_learning_demo"
        / "day07_GRPO"
        / "dapo"
        / "Countdown-Tasks-3to4"
        / "data"
        / "train-00000-of-00001.parquet"
    )
    output_dir: Path = Path(
        "~/scratch/llms_model/GRPO-Zero/Qwen2.5-3B-R1-Zero"
    ).expanduser()

    seed: int = 1337
    test_size: int = 128
    max_steps: int = 1000

    # Four questions × eight responses = 32 rollout trajectories per step.
    # Group size eight gives the within-question relative baseline enough
    # diversity while remaining practical on one A100 80 GB.
    question_batch_size: int = 4
    group_size: int = 8
    generation_batch_size: int = 32
    max_new_tokens: int = 512

    # Update one long trajectory at a time. Gradients accumulate across all 32
    # trajectories before optimizer.step(), so this does not reduce the
    # mathematical rollout batch size.
    micro_batch_size: int = 1
    grpo_epochs: int = 2
    learning_rate: float = 1e-6
    clip_epsilon: float = 0.2
    kl_beta: float = 0.001
    max_grad_norm: float = 1.0

    eval_every: int = 50
    save_every: int = 50
    eval_questions: int = 64
    eval_batch_size: int = 16
    gradient_checkpointing: bool = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="R1-Zero-style clipped GRPO training for Qwen2.5-3B Base."
    )
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--data-file", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--question-batch-size", type=int)
    parser.add_argument("--group-size", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--grpo-epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--clip-epsilon", type=float)
    parser.add_argument("--kl-beta", type=float)
    parser.add_argument("--eval-every", type=int)
    parser.add_argument("--save-every", type=int)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument(
        "--preprocess-only",
        action="store_true",
        help="Inspect one problem/reward without loading model weights.",
    )
    return parser.parse_args()


def apply_overrides(config: TrainingConfig, args: argparse.Namespace) -> None:
    """Apply only command-line settings explicitly supplied by the user."""
    for field_name in (
        "model_path",
        "data_file",
        "output_dir",
        "max_steps",
        "question_batch_size",
        "group_size",
        "generation_batch_size",
        "micro_batch_size",
        "max_new_tokens",
        "grpo_epochs",
        "learning_rate",
        "clip_epsilon",
        "kl_beta",
        "eval_every",
        "save_every",
    ):
        value = getattr(args, field_name)
        if value is not None:
            if isinstance(value, Path):
                value = value.expanduser()
            setattr(config, field_name, value)


def validate_config(config: TrainingConfig) -> None:
    if not config.model_path.is_dir():
        raise FileNotFoundError(f"Qwen base model directory missing: {config.model_path}")
    if "instruct" in config.model_path.name.lower():
        raise ValueError(
            "R1-Zero-style training must start from Qwen2.5-3B Base, not an "
            f"Instruct checkpoint: {config.model_path}"
        )
    if not config.data_file.is_file():
        raise FileNotFoundError(f"Countdown data file missing: {config.data_file}")

    positive_integer_fields = (
        "test_size",
        "max_steps",
        "question_batch_size",
        "group_size",
        "generation_batch_size",
        "max_new_tokens",
        "micro_batch_size",
        "grpo_epochs",
        "eval_every",
        "save_every",
        "eval_questions",
        "eval_batch_size",
    )
    for field_name in positive_integer_fields:
        if getattr(config, field_name) < 1:
            raise ValueError(f"{field_name} must be positive")
    if config.group_size < 2:
        raise ValueError("group_size must be at least 2 for a relative baseline")
    if config.learning_rate <= 0 or config.kl_beta < 0:
        raise ValueError("learning_rate must be positive and kl_beta non-negative")
    if not 0 < config.clip_epsilon < 1:
        raise ValueError("clip_epsilon must be between 0 and 1")


def choose_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        raise RuntimeError("This 3B full-parameter training script requires a CUDA GPU")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    # The update intentionally avoids a GradScaler because the requested A100
    # supports BF16. Failing clearly is safer than silently training FP16 with
    # underflow-prone unscaled gradients on a different GPU.
    raise RuntimeError("This implementation requires a BF16-capable GPU")


def load_policy_and_reference(
    config: TrainingConfig,
    resume_from: Path | None,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Any, Any]:
    """Load trainable policy and a permanently frozen base reference."""
    policy_source = resume_from.expanduser() if resume_from else config.model_path
    if not policy_source.is_dir():
        raise FileNotFoundError(f"Policy checkpoint missing: {policy_source}")

    common_load_args = {
        "dtype": dtype,
        "local_files_only": True,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
    }
    policy = AutoModelForCausalLM.from_pretrained(
        str(policy_source),
        **common_load_args,
    ).to(device)
    reference = AutoModelForCausalLM.from_pretrained(
        str(config.model_path),
        **common_load_args,
    ).to(device)
    reference.eval()
    reference.requires_grad_(False)

    if config.gradient_checkpointing:
        try:
            policy.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        except TypeError:
            policy.gradient_checkpointing_enable()
    return policy, reference


def make_optimizer(policy: Any, learning_rate: float) -> torch.optim.Optimizer:
    """Use fused AdamW when supported by the installed CUDA/PyTorch build."""
    try:
        return torch.optim.AdamW(
            policy.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
            fused=True,
        )
    except (RuntimeError, TypeError):
        return torch.optim.AdamW(
            policy.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )


def save_checkpoint(
    policy: Any,
    tokenizer: Any,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    step: int,
    final: bool,
) -> Path:
    checkpoint_dir = (
        config.output_dir
        if final
        else config.output_dir / f"checkpoint-{step:06d}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    policy.config.use_cache = True
    policy.save_pretrained(checkpoint_dir, safe_serialization=True)
    tokenizer.save_pretrained(checkpoint_dir)
    torch.save(
        {
            "step": step,
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_dir / "trainer_state.pt",
    )
    return checkpoint_dir


@torch.no_grad()
def evaluate(
    policy: Any,
    tokenizer: Any,
    dataloader: DataLoader,
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    """Generate one answer per held-out problem and report rule accuracy."""
    rewards: list[float] = []
    accuracies: list[float] = []
    formats: list[float] = []
    lengths: list[int] = []

    for problems in dataloader:
        remaining = config.eval_questions - len(rewards)
        if remaining <= 0:
            break
        problems = problems[:remaining]
        episodes = generate_episodes(
            model=policy,
            tokenizer=tokenizer,
            problems=problems,
            group_size=1,
            generation_batch_size=config.eval_batch_size,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
        for episode in episodes:
            rewards.append(episode.reward)
            accuracies.append(episode.reward_info["answer_reward"])
            formats.append(episode.reward_info["format_reward"])
            lengths.append(len(episode.response_token_ids))

    return {
        "eval_reward": float(np.mean(rewards)),
        "eval_accuracy": float(np.mean(accuracies)),
        "eval_format": float(np.mean(formats)),
        "eval_response_length": float(np.mean(lengths)),
    }


def append_json_log(log_file: Path, values: dict[str, Any]) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a", encoding="utf-8") as output:
        output.write(json.dumps(values, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    config = TrainingConfig()
    apply_overrides(config, args)
    validate_config(config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        str(config.model_path),
        local_files_only=True,
    )
    if tokenizer.eos_token_id is None:
        raise RuntimeError("Qwen tokenizer must define eos_token_id")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Decoder-only batched generation must left-pad prompts so logits[:, -1]
    # always corresponds to the end of the real prompt.
    tokenizer.padding_side = "left"

    train_dataset = CountdownDataset(
        config.data_file,
        tokenizer,
        split="train",
        test_size=config.test_size,
    )
    eval_dataset = CountdownDataset(
        config.data_file,
        tokenizer,
        split="test",
        test_size=config.test_size,
    )

    if args.preprocess_only:
        example = train_dataset[0]
        example_response = (
            "Try adding the numbers.</think>\n"
            f"<answer>{' + '.join(map(str, example.numbers))}</answer>"
        )
        total, details = reward_function(
            example_response,
            example.numbers,
            example.target,
            finished=True,
        )
        print(example.prompt)
        print("Prompt token count:", len(example.prompt_token_ids))
        print("Example response:", example_response)
        print("Example reward:", total, details)
        print("Preprocessing check complete; model weights were not loaded.")
        return

    device = torch.device("cuda")
    dtype = choose_dtype()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    resume_from = args.resume_from.expanduser() if args.resume_from else None
    policy, reference_policy = load_policy_and_reference(
        config,
        resume_from,
        device,
        dtype,
    )
    optimizer = make_optimizer(policy, config.learning_rate)
    starting_step = 0
    if resume_from is not None:
        trainer_state_path = resume_from / "trainer_state.pt"
        if not trainer_state_path.is_file():
            raise FileNotFoundError(f"Resume state missing: {trainer_state_path}")
        trainer_state = torch.load(trainer_state_path, map_location="cpu")
        optimizer.load_state_dict(trainer_state["optimizer_state_dict"])
        starting_step = int(trainer_state["step"])

    generator = torch.Generator()
    generator.manual_seed(config.seed)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.question_batch_size,
        shuffle=True,
        collate_fn=CountdownDataset.collate_fn,
        generator=generator,
        num_workers=0,
        drop_last=True,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=config.eval_batch_size,
        shuffle=False,
        collate_fn=CountdownDataset.collate_fn,
        num_workers=0,
    )

    config.output_dir.mkdir(parents=True, exist_ok=True)
    log_file = config.output_dir / "training_log.jsonl"
    serializable_config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in asdict(config).items()
    }
    with (config.output_dir / "training_config.json").open(
        "w", encoding="utf-8"
    ) as output:
        json.dump(serializable_config, output, ensure_ascii=False, indent=2)

    print("=" * 72)
    print("R1-Zero-style Qwen2.5-3B Base training")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Dtype: {dtype}")
    print(f"Questions per step: {config.question_batch_size}")
    print(f"Responses per question: {config.group_size}")
    print(
        "Trajectories per step: "
        f"{config.question_batch_size * config.group_size}"
    )
    print(f"Update microbatch: {config.micro_batch_size}")
    print(f"Maximum generated tokens: {config.max_new_tokens}")
    print("=" * 72)

    completed_step = starting_step
    for data_iteration, problems in enumerate(train_dataloader, start=1):
        step = starting_step + data_iteration
        if step > config.max_steps:
            break
        step_started = time.time()

        # ------------------------- COLLECT -------------------------
        # The policy is fixed while all trajectories and old log-probabilities
        # for this GRPO batch are collected.
        episodes = generate_episodes(
            model=policy,
            tokenizer=tokenizer,
            problems=problems,
            group_size=config.group_size,
            generation_batch_size=config.generation_batch_size,
            max_new_tokens=config.max_new_tokens,
            device=device,
        )
        group_metrics = assign_group_advantages(episodes)
        store_old_policy_log_probs(
            model=policy,
            episodes=episodes,
            micro_batch_size=config.micro_batch_size,
            pad_token_id=tokenizer.pad_token_id,
            device=device,
            dtype=dtype,
        )

        # Release generation/cache blocks before allocating backward tensors.
        gc.collect()
        torch.cuda.empty_cache()

        # -------------------------- UPDATE -------------------------
        update_metrics = update_policy_grpo(
            policy=policy,
            reference_policy=reference_policy,
            optimizer=optimizer,
            episodes=episodes,
            pad_token_id=tokenizer.pad_token_id,
            micro_batch_size=config.micro_batch_size,
            grpo_epochs=config.grpo_epochs,
            clip_epsilon=config.clip_epsilon,
            kl_beta=config.kl_beta,
            max_grad_norm=config.max_grad_norm,
            device=device,
            dtype=dtype,
        )
        completed_step = step

        rewards = [episode.reward for episode in episodes]
        accuracies = [
            episode.reward_info["answer_reward"] for episode in episodes
        ]
        formats = [episode.reward_info["format_reward"] for episode in episodes]
        lengths = [len(episode.response_token_ids) for episode in episodes]
        finished = [float(episode.finished) for episode in episodes]

        log_values: dict[str, Any] = {
            "step": step,
            "seconds": time.time() - step_started,
            "reward": float(np.mean(rewards)),
            "reward_std": float(np.std(rewards)),
            "accuracy": float(np.mean(accuracies)),
            "format": float(np.mean(formats)),
            "finished_fraction": float(np.mean(finished)),
            "response_length": float(np.mean(lengths)),
            **group_metrics,
            **update_metrics,
        }

        print(
            f"step {step:04d} | reward {log_values['reward']:.3f} | "
            f"accuracy {log_values['accuracy']:.3f} | "
            f"format {log_values['format']:.3f} | "
            f"length {log_values['response_length']:.1f} | "
            f"ref_kl {log_values['reference_kl']:.5f} | "
            f"clip {log_values['clip_fraction']:.3f} | "
            f"grad {log_values['grad_norm']:.3f} | "
            f"{log_values['seconds']:.1f}s"
        )

        # Print one sampled trajectory occasionally for interpretability.
        if step == 1 or step % 25 == 0:
            sample = episodes[0]
            print("-" * 72)
            print(f"numbers={sample.numbers}, target={sample.target}")
            print(sample.response_text)
            print(
                f"reward={sample.reward:.3f}, advantage={sample.advantage:.3f}"
            )
            print("-" * 72)

        if step % config.eval_every == 0:
            evaluation = evaluate(
                policy,
                tokenizer,
                eval_dataloader,
                config,
                device,
            )
            log_values.update(evaluation)
            print(
                f"evaluation | accuracy {evaluation['eval_accuracy']:.3f} | "
                f"format {evaluation['eval_format']:.3f} | "
                f"reward {evaluation['eval_reward']:.3f}"
            )

        append_json_log(log_file, log_values)

        if step % config.save_every == 0:
            checkpoint = save_checkpoint(
                policy,
                tokenizer,
                optimizer,
                config,
                step,
                final=False,
            )
            print(f"Saved checkpoint: {checkpoint}")

    final_directory = save_checkpoint(
        policy,
        tokenizer,
        optimizer,
        config,
        completed_step,
        final=True,
    )
    print(f"Training complete. Final policy saved to: {final_directory}")


if __name__ == "__main__":
    main()
