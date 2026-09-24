"""Central configuration and command-line interface for the DALL-E 2 demo.

Configuration is applied in this order:

1. dataclass defaults;
2. dataset/pretrained-CLIP/large-U-Net presets;
3. explicit command-line overrides;
4. validation and experiment-directory resolution.

Every training run writes ``effective_config.json`` beside its checkpoints.
Inference can therefore load a complete experiment using only ``--run-name``.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field, fields
import json
from pathlib import Path
import re
from typing import Any

import torch


DALLE2_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DALLE2_DIR.parent.parent
DEFAULT_CHECKPOINT_ROOT = DALLE2_DIR / "trained_models"
CONFIG_FILENAME = "effective_config.json"


@dataclass
class CLIPConfig:
    patch_size: tuple[int, int] = (4, 4)
    vit_width: int = 256
    vit_layers: int = 6
    vit_heads: int = 8
    text_width: int = 256
    text_layers: int = 6
    text_heads: int = 8
    dropout: float = 0.2
    r_mlp: int = 4
    bias: bool = False
    augment_data: bool = True
    validate: bool = True
    num_workers: int = 0
    batch_size: int = 128
    lr: float = 5e-4
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 200
    warmup_epochs: int = 5
    grad_max_norm: float = 1.0
    get_val_accuracy: bool = False
    model_location: str = ""


@dataclass
class PriorConfig:
    max_time: int = 1000
    schedule: str = "cosine"
    schedule_offset: float = 0.008
    width: int = 256
    n_layers: int = 6
    n_heads: int = 8
    dropout: float = 0.2
    r_mlp: int = 4
    bias: bool = False
    augment_data: bool = False
    validate: bool = True
    num_workers: int = 0
    batch_size: int = 128
    lr: float = 5e-4
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 150
    warmup_epochs: int = 5
    grad_max_norm: float = 1.0
    model_location: str = ""


@dataclass
class DecoderConfig:
    max_time: int = 1000
    schedule: str = "cosine"
    n_groups: int = 8
    kernel_size: tuple[int, int] = (3, 3)
    model_channels: int = 32
    cond_channels: int = 128
    channel_ratios: list[int] = field(default_factory=lambda: [1, 2, 4, 8])
    n_layer_blocks: int = 2
    dropout: float = 0.1
    use_scale_shift: bool = True
    n_heads: int = 8
    stride: int = 2
    down_pool: bool = False
    r_mlp: int = 4
    bias: bool = False
    text_layers: int = 4
    n_img_tokens: int = 4
    augment_data: bool = False
    validate: bool = True
    num_workers: int = 0
    batch_size: int = 32
    lr: float = 5e-4
    lr_min: float = 1e-5
    weight_decay: float = 1e-4
    epochs: int = 100
    warmup_epochs: int = 5
    grad_max_norm: float = 1.0
    sample_after_epoch: bool = False
    model_location: str = ""


@dataclass
class Dalle2Config:
    latent_dim: int = 256
    using_pretrained_clip: bool = False
    pretrained_clip_path: str = str(
        Path("~/scratch/llms_model/clip-vit-base-patch32").expanduser()
    )
    large_unet: bool = False
    prior_num_candidates: int = 2

    dataset: str = "flickr30k"
    data_location: str = str(PROJECT_ROOT / "datasets" / "flickr30k" / "data")
    fashion_mnist_data_location: str = str(DALLE2_DIR / "datasets")
    flickr8k_data_location: str = str(
        PROJECT_ROOT / "datasets" / "flickr8k" / "data"
    )
    flickr30k_data_location: str = str(
        PROJECT_ROOT / "datasets" / "flickr30k" / "data"
    )
    img_size: tuple[int, int] = (64, 64)
    img_channels: int = 3
    vocab_size: int = 256
    text_seq_length: int = 64
    prob_hflip: float = 0.5
    crop_padding: int = 4
    train_mean: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    train_std: list[float] = field(default_factory=lambda: [0.5, 0.5, 0.5])
    train_val_split: tuple[int, int] = (50000, 10000)
    device: torch.device = field(default_factory=lambda: torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    ))

    run_name: str | None = None
    checkpoint_root: str = str(DEFAULT_CHECKPOINT_ROOT)
    run_dir: str = ""
    resume: bool = False
    overwrite: bool = False

    clip: CLIPConfig = field(default_factory=CLIPConfig)
    prior: PriorConfig = field(default_factory=PriorConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)


# Old modules and notebooks imported this name. Keep it as a real alias.
FMNISTConfig = Dalle2Config


SUPPORTED_DATASETS = ("fashion_mnist", "flickr8k", "flickr30k", "all")


def normalize_dataset_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_")
    return {
        "fashionmnist": "fashion_mnist",
        "flick8k": "flickr8k",
    }.get(normalized, normalized)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed < 1.0:
        raise argparse.ArgumentTypeError("value must be in [0, 1)")
    return parsed


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    """Add every shared dataset, architecture, and training CLI option."""
    parser.add_argument(
        "--dataset", "--data", dest="dataset",
        type=normalize_dataset_name, choices=SUPPORTED_DATASETS, default=None,
        help=(
            "Dataset selection. 'all' combines Flickr8k and Flickr30k; "
            "default: flickr30k."
        ),
    )
    parser.add_argument(
        "--data-dir", type=Path, default=None,
        help=(
            "Dataset directory override. For --dataset all, supply a root "
            "containing flickr8k/data and flickr30k/data."
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--using-pre-CLIP", "--using-pre-clip",
        dest="using_pretrained_clip", action="store_true", default=None,
    )
    parser.add_argument(
        "--pretrained-clip-path", type=Path, default=None,
    )
    parser.add_argument(
        "--large-UNet", "--large-unet",
        dest="large_unet", action="store_true", default=None,
    )
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--checkpoint-root", type=Path, default=None)
    run_mode = parser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "--resume", action="store_true",
        help="Skip completed stages and continue with the first missing stage.",
    )
    run_mode.add_argument(
        "--overwrite", action="store_true",
        help="Explicitly allow existing stage checkpoints to be replaced.",
    )

    # Custom CLIP training.
    parser.add_argument("--clip-lr", type=_positive_float, default=None)
    parser.add_argument("--clip-lr-min", type=_positive_float, default=None)
    parser.add_argument("--clip-batch-size", type=_positive_int, default=None)
    parser.add_argument("--clip-epochs", type=_positive_int, default=None)
    parser.add_argument("--clip-warmup-epochs", type=_non_negative_int, default=None)
    parser.add_argument("--clip-weight-decay", type=float, default=None)
    parser.add_argument("--clip-dropout", type=_probability, default=None)
    parser.add_argument("--clip-num-workers", type=_non_negative_int, default=None)
    parser.add_argument(
        "--clip-validate", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--clip-augment-data", action=argparse.BooleanOptionalAction, default=None,
    )

    # Diffusion prior.
    parser.add_argument("--prior-lr", type=_positive_float, default=None)
    parser.add_argument("--prior-lr-min", type=_positive_float, default=None)
    parser.add_argument("--prior-batch-size", type=_positive_int, default=None)
    parser.add_argument("--prior-epochs", type=_positive_int, default=None)
    parser.add_argument("--prior-warmup-epochs", type=_non_negative_int, default=None)
    parser.add_argument("--prior-weight-decay", type=float, default=None)
    parser.add_argument("--prior-dropout", type=_probability, default=None)
    parser.add_argument("--prior-num-workers", type=_non_negative_int, default=None)
    parser.add_argument("--prior-layers", type=_positive_int, default=None)
    parser.add_argument("--prior-heads", type=_positive_int, default=None)
    parser.add_argument("--prior-max-time", type=_positive_int, default=None)
    parser.add_argument("--prior-candidates", type=_positive_int, default=None)
    parser.add_argument(
        "--prior-validate", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--prior-augment-data", action=argparse.BooleanOptionalAction, default=None,
    )

    # Pixel diffusion decoder.
    parser.add_argument("--decoder-lr", type=_positive_float, default=None)
    parser.add_argument("--decoder-lr-min", type=_positive_float, default=None)
    parser.add_argument("--decoder-batch-size", type=_positive_int, default=None)
    parser.add_argument("--decoder-epochs", type=_positive_int, default=None)
    parser.add_argument("--decoder-warmup-epochs", type=_non_negative_int, default=None)
    parser.add_argument("--decoder-weight-decay", type=float, default=None)
    parser.add_argument("--decoder-dropout", type=_probability, default=None)
    parser.add_argument("--decoder-num-workers", type=_non_negative_int, default=None)
    parser.add_argument("--decoder-model-channels", type=_positive_int, default=None)
    parser.add_argument("--decoder-cond-channels", type=_positive_int, default=None)
    parser.add_argument("--decoder-layer-blocks", type=_positive_int, default=None)
    parser.add_argument("--decoder-heads", type=_positive_int, default=None)
    parser.add_argument("--decoder-text-layers", type=_positive_int, default=None)
    parser.add_argument("--decoder-image-tokens", type=_positive_int, default=None)
    parser.add_argument("--decoder-max-time", type=_positive_int, default=None)
    parser.add_argument(
        "--decoder-validate", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--decoder-augment-data", action=argparse.BooleanOptionalAction, default=None,
    )
    parser.add_argument(
        "--decoder-sample-after-epoch",
        action=argparse.BooleanOptionalAction,
        default=None,
    )


# Backward-compatible function name used by earlier scripts.
add_dataset_arguments = add_config_arguments


def configure_dataset(
    config: Dalle2Config,
    dataset: str,
    data_location: str | Path | None = None,
) -> Dalle2Config:
    """Apply the shape and path preset for one dataset selection."""
    dataset = normalize_dataset_name(dataset)
    if dataset == "fashion_mnist":
        config.dataset = dataset
        config.data_location = str(
            Path(data_location).expanduser()
            if data_location is not None else DALLE2_DIR / "datasets"
        )
        config.fashion_mnist_data_location = config.data_location
        config.img_size = (32, 32)
        config.img_channels = 1
        config.train_mean = [0.2855552]
        config.train_std = [0.33848408]
        config.train_val_split = (50000, 10000)
    elif dataset in {"flickr8k", "flickr30k"}:
        config.dataset = dataset
        config.data_location = str(
            Path(data_location).expanduser()
            if data_location is not None
            else PROJECT_ROOT / "datasets" / dataset / "data"
        )
        setattr(config, f"{dataset}_data_location", config.data_location)
        config.img_size = (64, 64)
        config.img_channels = 3
        config.train_mean = [0.5, 0.5, 0.5]
        config.train_std = [0.5, 0.5, 0.5]
        config.train_val_split = (6000, 1000)
    elif dataset == "all":
        config.dataset = dataset
        if data_location is not None:
            datasets_root = Path(data_location).expanduser()
            config.data_location = str(datasets_root)
            config.flickr8k_data_location = str(datasets_root / "flickr8k" / "data")
            config.flickr30k_data_location = str(datasets_root / "flickr30k" / "data")
        else:
            config.data_location = str(PROJECT_ROOT / "datasets")
        config.img_size = (64, 64)
        config.img_channels = 3
        config.train_mean = [0.5, 0.5, 0.5]
        config.train_std = [0.5, 0.5, 0.5]
    else:
        raise ValueError(f"Unsupported dataset: {dataset!r}")

    if config.using_pretrained_clip:
        config.latent_dim = 512
        config.text_seq_length = 77
        config.prior.batch_size = min(config.prior.batch_size, 32)
        config.decoder.batch_size = min(config.decoder.batch_size, 16)

    if config.large_unet:
        config.decoder.model_channels = 64
        config.decoder.cond_channels = 256
        config.decoder.n_img_tokens = 8
        config.decoder.batch_size = min(config.decoder.batch_size, 16)
        config.decoder.num_workers = max(config.decoder.num_workers, 4)
    return config


def _apply_if_set(target: Any, args: argparse.Namespace, mapping: dict[str, str]) -> None:
    for argument_name, attribute_name in mapping.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            setattr(target, attribute_name, value)


def _format_float_for_name(value: float) -> str:
    value_string = f"{value:.0e}"
    return value_string.replace("e-0", "e-").replace("e+0", "e").replace("e+", "e")


def automatic_run_name(config: Dalle2Config) -> str:
    clip_name = "preclip" if config.using_pretrained_clip else "customclip"
    unet_name = "large" if config.large_unet else "base"
    learning_rate = _format_float_for_name(config.decoder.lr)
    return f"{config.dataset}-{clip_name}-{unet_name}-lr{learning_rate}"


def _validate_run_name(run_name: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_name):
        raise ValueError(
            "--run-name may contain only letters, numbers, '.', '_', and '-'"
        )


def finalize_experiment(config: Dalle2Config) -> Dalle2Config:
    if config.run_name is None:
        config.run_name = automatic_run_name(config)
    _validate_run_name(config.run_name)
    run_dir = Path(config.checkpoint_root).expanduser().resolve() / config.run_name
    config.run_dir = str(run_dir)
    config.clip.model_location = str(run_dir / "clip.pt")
    config.prior.model_location = str(run_dir / "prior.pt")
    config.decoder.model_location = str(run_dir / "decoder.pt")
    return config


def validate_config(config: Dalle2Config) -> None:
    if config.resume and config.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a GPU")
    if config.using_pretrained_clip and not Path(config.pretrained_clip_path).is_dir():
        raise FileNotFoundError(
            f"Pretrained CLIP directory does not exist: {config.pretrained_clip_path}"
        )
    for stage_name, stage in (
        ("clip", config.clip), ("prior", config.prior), ("decoder", config.decoder)
    ):
        if stage.warmup_epochs >= stage.epochs:
            raise ValueError(
                f"{stage_name} warmup epochs must be smaller than total epochs"
            )
        if stage.lr_min > stage.lr:
            raise ValueError(f"{stage_name} lr_min must not exceed lr")
        if stage.weight_decay < 0:
            raise ValueError(f"{stage_name} weight decay must be non-negative")
    channels = [
        config.decoder.model_channels * ratio
        for ratio in config.decoder.channel_ratios
    ]
    if any(channel % config.decoder.n_groups != 0 for channel in channels):
        raise ValueError("Every decoder channel width must be divisible by n_groups")
    if any(channel % config.decoder.n_heads != 0 for channel in channels):
        raise ValueError(
            "Every attended decoder channel width must be divisible by n_heads"
        )
    if config.latent_dim % config.prior.n_heads != 0:
        raise ValueError("latent_dim must be divisible by prior n_heads")
    if config.clip.vit_width % config.clip.vit_heads != 0:
        raise ValueError("CLIP vit_width must be divisible by vit_heads")
    if config.clip.text_width % config.clip.text_heads != 0:
        raise ValueError("CLIP text_width must be divisible by text_heads")
    downsample_factor = config.decoder.stride ** (
        len(config.decoder.channel_ratios) - 1
    )
    if any(size % downsample_factor != 0 for size in config.img_size):
        raise ValueError(
            f"Image dimensions must be divisible by {downsample_factor} for "
            "the configured U-Net"
        )


def _dataclass_from_dict(cls, values: dict[str, Any]):
    valid_names = {item.name for item in fields(cls)}
    return cls(**{key: value for key, value in values.items() if key in valid_names})


def effective_config_dict(config: Dalle2Config) -> dict[str, Any]:
    payload = asdict(config)
    payload["device"] = str(config.device)
    # These control filesystem behavior, not model reproducibility.
    payload.pop("resume", None)
    payload.pop("overwrite", None)
    # JSON represents tuples as lists. Canonicalize before comparing a new
    # configuration with an existing manifest, otherwise an identical second
    # stage would look different merely because one side still has tuples.
    return json.loads(json.dumps(payload))


def load_effective_config(run_name: str, checkpoint_root: str | Path) -> Dalle2Config:
    run_dir = Path(checkpoint_root).expanduser().resolve() / run_name
    config_path = run_dir / CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(f"Experiment configuration is missing: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    clip_values = payload.pop("clip")
    prior_values = payload.pop("prior")
    decoder_values = payload.pop("decoder")
    payload["device"] = torch.device(payload.get("device", "cpu"))
    config = _dataclass_from_dict(Dalle2Config, payload)
    config.clip = _dataclass_from_dict(CLIPConfig, clip_values)
    config.prior = _dataclass_from_dict(PriorConfig, prior_values)
    config.decoder = _dataclass_from_dict(DecoderConfig, decoder_values)
    config.checkpoint_root = str(Path(checkpoint_root).expanduser().resolve())
    config.run_name = run_name
    return finalize_experiment(config)


def save_effective_config(config: Dalle2Config) -> Path:
    """Create/verify a run manifest without silently changing an experiment."""
    run_dir = Path(config.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / CONFIG_FILENAME
    new_payload = effective_config_dict(config)
    if config_path.is_file() and not config.overwrite:
        old_payload = json.loads(config_path.read_text(encoding="utf-8"))
        if old_payload != new_payload:
            raise RuntimeError(
                f"Experiment {config.run_name!r} already has a different "
                f"configuration. Choose another --run-name or use --overwrite."
            )
    else:
        config_path.write_text(
            json.dumps(new_payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return config_path


def prepare_training_stage(config: Dalle2Config, stage: str, checkpoint: str) -> bool:
    """Return True to train, or False when --resume skips a complete stage."""
    checkpoint_path = Path(checkpoint)
    completion_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".complete")
    config_path = Path(config.run_dir) / CONFIG_FILENAME
    if checkpoint_path.is_file() and not config.resume and not config.overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing {stage} checkpoint: "
            f"{checkpoint_path}. Use --resume to skip it, --overwrite to "
            "replace it, or choose a different --run-name."
        )
    if config.resume and checkpoint_path.is_file() and not config_path.is_file():
        raise FileNotFoundError(
            f"Cannot safely resume {stage}: {checkpoint_path} exists but its "
            f"configuration manifest is missing: {config_path}"
        )
    config_path = save_effective_config(config)
    print(f"Experiment: {config.run_name}", flush=True)
    print(f"Run directory: {config.run_dir}", flush=True)
    print(f"Effective configuration: {config_path}", flush=True)
    if config.resume:
        if checkpoint_path.is_file() and completion_path.is_file():
            print(
                f"[Resume] {stage} completed previously; skipping: "
                f"{checkpoint_path}",
                flush=True,
            )
            return False
        if completion_path.is_file() and not checkpoint_path.is_file():
            raise FileNotFoundError(
                f"{stage} completion marker exists but checkpoint is missing: "
                f"{checkpoint_path}"
            )
        if checkpoint_path.is_file():
            print(
                f"[Resume] {stage} has an incomplete checkpoint; restarting "
                "this stage from epoch 0.",
                flush=True,
            )
    if checkpoint_path.is_file() and config.overwrite:
        print(f"[Overwrite] Replacing {stage} checkpoint: {checkpoint_path}", flush=True)
    # Retraining an upstream stage invalidates downstream completion markers.
    # Keep the old weights recoverable, but never advertise them as compatible
    # with newly trained upstream weights if this job is interrupted.
    normalized_stage = stage.lower()
    invalidated = {
        "clip": (config.clip.model_location, config.prior.model_location,
                 config.decoder.model_location),
        "prior": (config.prior.model_location, config.decoder.model_location),
        "decoder": (config.decoder.model_location,),
    }.get(normalized_stage, (checkpoint,))
    for invalidated_checkpoint in invalidated:
        path = Path(invalidated_checkpoint)
        marker = path.with_suffix(path.suffix + ".complete")
        if marker.is_file():
            marker.unlink()
    return True


def training_stage_is_complete(checkpoint: str | Path) -> bool:
    """A stage is reusable only when both its weights and marker exist."""
    checkpoint_path = Path(checkpoint)
    completion_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".complete")
    return (
        checkpoint_path.is_file()
        and checkpoint_path.stat().st_size > 0
        and completion_path.is_file()
    )


def mark_training_stage_complete(
    config: Dalle2Config,
    stage: str,
    checkpoint: str,
) -> Path:
    """Mark a stage complete only after its entire epoch loop has finished."""
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_file() or checkpoint_path.stat().st_size == 0:
        raise FileNotFoundError(
            f"Cannot mark {stage} complete; checkpoint is missing or empty: "
            f"{checkpoint_path}"
        )
    completion_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".complete")
    temporary_path = completion_path.with_suffix(completion_path.suffix + ".tmp")
    temporary_path.write_text(
        f"stage={stage}\nrun_name={config.run_name}\n",
        encoding="utf-8",
    )
    temporary_path.replace(completion_path)
    print(f"[Complete] Wrote stage marker: {completion_path}", flush=True)
    return completion_path


def _validate_loaded_selectors(config: Dalle2Config, args: argparse.Namespace) -> None:
    requested_dataset = getattr(args, "dataset", None)
    if requested_dataset is not None and requested_dataset != config.dataset:
        raise ValueError(
            f"--run-name {config.run_name!r} uses dataset={config.dataset!r}, "
            f"not {requested_dataset!r}"
        )
    requested_preclip = getattr(args, "using_pretrained_clip", None)
    if requested_preclip is not None and requested_preclip != config.using_pretrained_clip:
        raise ValueError("--using-pre-CLIP does not match the saved experiment")
    requested_large = getattr(args, "large_unet", None)
    if requested_large is not None and requested_large != config.large_unet:
        raise ValueError("--large-UNet does not match the saved experiment")


def config_from_args(
    args: argparse.Namespace,
    *,
    load_existing_run: bool = False,
) -> Dalle2Config:
    checkpoint_root = str(
        (args.checkpoint_root or DEFAULT_CHECKPOINT_ROOT).expanduser().resolve()
        if isinstance(args.checkpoint_root, Path)
        else Path(args.checkpoint_root or DEFAULT_CHECKPOINT_ROOT).expanduser().resolve()
    )

    if load_existing_run and args.run_name is not None:
        config = load_effective_config(args.run_name, checkpoint_root)
        _validate_loaded_selectors(config, args)
        if args.device is not None:
            config.device = torch.device(args.device)
        if args.pretrained_clip_path is not None:
            # The model identity/architecture is unchanged; this only permits
            # a saved experiment to be moved to a machine with another local
            # model-cache path.
            config.pretrained_clip_path = str(
                args.pretrained_clip_path.expanduser()
            )
        if args.prior_candidates is not None:
            config.prior_num_candidates = args.prior_candidates
        validate_config(config)
        return config

    config = Dalle2Config()
    config.checkpoint_root = checkpoint_root
    if args.using_pretrained_clip is not None:
        config.using_pretrained_clip = args.using_pretrained_clip
    if args.pretrained_clip_path is not None:
        config.pretrained_clip_path = str(args.pretrained_clip_path.expanduser())
    if args.large_unet is not None:
        config.large_unet = args.large_unet
    config.run_name = args.run_name
    config.resume = args.resume
    config.overwrite = args.overwrite

    dataset = args.dataset or config.dataset
    config = configure_dataset(config, dataset, args.data_dir)

    _apply_if_set(config.clip, args, {
        "clip_lr": "lr", "clip_lr_min": "lr_min",
        "clip_batch_size": "batch_size", "clip_epochs": "epochs",
        "clip_warmup_epochs": "warmup_epochs",
        "clip_weight_decay": "weight_decay", "clip_dropout": "dropout",
        "clip_num_workers": "num_workers", "clip_validate": "validate",
        "clip_augment_data": "augment_data",
    })
    _apply_if_set(config.prior, args, {
        "prior_lr": "lr", "prior_lr_min": "lr_min",
        "prior_batch_size": "batch_size", "prior_epochs": "epochs",
        "prior_warmup_epochs": "warmup_epochs",
        "prior_weight_decay": "weight_decay", "prior_dropout": "dropout",
        "prior_num_workers": "num_workers", "prior_layers": "n_layers",
        "prior_heads": "n_heads", "prior_max_time": "max_time",
        "prior_validate": "validate", "prior_augment_data": "augment_data",
    })
    _apply_if_set(config.decoder, args, {
        "decoder_lr": "lr", "decoder_lr_min": "lr_min",
        "decoder_batch_size": "batch_size", "decoder_epochs": "epochs",
        "decoder_warmup_epochs": "warmup_epochs",
        "decoder_weight_decay": "weight_decay",
        "decoder_dropout": "dropout", "decoder_num_workers": "num_workers",
        "decoder_model_channels": "model_channels",
        "decoder_cond_channels": "cond_channels",
        "decoder_layer_blocks": "n_layer_blocks",
        "decoder_heads": "n_heads", "decoder_text_layers": "text_layers",
        "decoder_image_tokens": "n_img_tokens",
        "decoder_max_time": "max_time", "decoder_validate": "validate",
        "decoder_augment_data": "augment_data",
        "decoder_sample_after_epoch": "sample_after_epoch",
    })
    if args.prior_candidates is not None:
        config.prior_num_candidates = args.prior_candidates
    if args.device is not None:
        config.device = torch.device(args.device)

    finalize_experiment(config)
    validate_config(config)
    return config


def _main() -> None:
    parser = argparse.ArgumentParser(description="Resolve a DALL-E 2 experiment configuration")
    add_config_arguments(parser)
    parser.add_argument("--print-run-dir", action="store_true")
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args()
    config = config_from_args(args)
    if args.print_run_dir:
        print(config.run_dir)
    else:
        print(json.dumps(effective_config_dict(config), indent=2, sort_keys=True))


if __name__ == "__main__":
    _main()
