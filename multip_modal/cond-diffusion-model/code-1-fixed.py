"""Conditional DDPM on MNIST with separate training and test workflows.

Train and save a checkpoint:

    python code-1-fixed.py --mode train

Load that checkpoint and generate images conditioned on digit 7:

    python code-1-fixed.py --mode test --digit 7 --num-samples 16

The test command starts from Gaussian noise and supplies label 7 to the U-Net
at every reverse-diffusion step. It therefore generates new digit-7 images; it
does not retrieve examples from the MNIST test set.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
PROJECT_DATA_DIR = PROJECT_ROOT / "datasets"
LEGACY_DATA_DIR = SCRIPT_DIR.parent / "datasets"
# Prefer the server layout requested by the project. Keep compatibility with
# the original script's ../datasets location when running the local copy.
DEFAULT_DATA_DIR = (
    PROJECT_DATA_DIR
    if (PROJECT_DATA_DIR / "MNIST").is_dir()
    else LEGACY_DATA_DIR
)
DEFAULT_CHECKPOINT = SCRIPT_DIR / "checkpoints" / "mnist-cond-diffusion.pt"


@dataclass(frozen=True)
class ModelConfig:
    image_channels: int = 1
    image_size: int = 28
    time_embedding_dim: int = 100
    num_labels: int = 10


@dataclass(frozen=True)
class DiffusionConfig:
    num_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train or test a label-conditioned MNIST DDPM."
    )
    parser.add_argument("--mode", choices=("train", "test"), default="train")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--time-embedding-dim", type=int, default=100)
    parser.add_argument(
        "--digit",
        type=int,
        default=7,
        help="Digit condition used in test mode; must be between 0 and 9.",
    )
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Test PNG path; defaults beside the checkpoint.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BF16/FP16 mixed precision while training on CUDA.",
    )
    args = parser.parse_args()
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    for name in (
        "epochs",
        "batch_size",
        "learning_rate",
        "max_grad_norm",
        "num_timesteps",
        "time_embedding_dim",
        "num_samples",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if not 0 <= args.digit <= 9:
        raise ValueError("--digit must be between 0 and 9")
    if not 0 < args.beta_start < args.beta_end < 1:
        raise ValueError("require 0 < beta-start < beta-end < 1")


def select_device(requested: str | torch.device) -> torch.device:
    if isinstance(requested, torch.device):
        return requested
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    output_dim: int,
) -> torch.Tensor:
    """Vectorized sinusoidal encoding for integer diffusion timesteps."""
    half_dim = output_dim // 2
    denominator = max(half_dim - 1, 1)
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half_dim, device=timesteps.device)
        / denominator
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if output_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ConvBlock(nn.Module):
    """Time/label-conditioned convolution block.

    GroupNorm is used instead of BatchNorm because the distribution of x_t
    changes strongly with timestep and sampling uses no batch statistics.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        groups = next(
            group_count
            for group_count in range(min(8, output_channels), 0, -1)
            if output_channels % group_count == 0
        )
        self.condition_projection = nn.Sequential(
            nn.SiLU(),
            nn.Linear(condition_dim, output_channels),
        )
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(groups, output_channels)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(groups, output_channels)

    def forward(
        self,
        images: torch.Tensor,
        condition: torch.Tensor,
    ) -> torch.Tensor:
        hidden = F.silu(self.norm1(self.conv1(images)))
        hidden = hidden + self.condition_projection(condition)[:, :, None, None]
        return F.silu(self.norm2(self.conv2(hidden)))


class UNetCond(nn.Module):
    """Small U-Net conditioned jointly on timestep and MNIST class label."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.label_embedding = nn.Embedding(
            config.num_labels,
            config.time_embedding_dim,
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim),
        )

        self.down1 = ConvBlock(
            config.image_channels,
            64,
            config.time_embedding_dim,
        )
        self.down2 = ConvBlock(64, 128, config.time_embedding_dim)
        self.middle = ConvBlock(128, 256, config.time_embedding_dim)
        self.up2 = ConvBlock(128 + 256, 128, config.time_embedding_dim)
        self.up1 = ConvBlock(64 + 128, 64, config.time_embedding_dim)
        self.output = nn.Conv2d(64, config.image_channels, 1)

        self.max_pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=False,
        )
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = noisy_images.shape[0]
        if timesteps.shape != (batch_size,) or labels.shape != (batch_size,):
            raise ValueError(
                "timesteps and labels must each have shape [image batch size]"
            )
        if labels.min() < 0 or labels.max() >= self.config.num_labels:
            raise ValueError("labels are outside the configured class range")

        time_embedding = sinusoidal_time_embedding(
            timesteps,
            self.config.time_embedding_dim,
        )
        condition = self.time_mlp(time_embedding) + self.label_embedding(labels)

        skip1 = self.down1(noisy_images, condition)       # 28 x 28
        skip2 = self.down2(self.max_pool(skip1), condition)  # 14 x 14
        hidden = self.middle(self.max_pool(skip2), condition)  # 7 x 7
        hidden = self.upsample(hidden)                    # 14 x 14
        hidden = self.up2(torch.cat((hidden, skip2), dim=1), condition)
        hidden = self.upsample(hidden)                    # 28 x 28
        hidden = self.up1(torch.cat((hidden, skip1), dim=1), condition)
        return self.output(hidden)


class Diffuser:
    """DDPM forward process and class-conditioned ancestral sampler."""

    def __init__(
        self,
        config: DiffusionConfig,
        device: torch.device,
    ) -> None:
        self.config = config
        self.device = device
        self.betas = torch.linspace(
            config.beta_start,
            config.beta_end,
            config.num_timesteps,
            device=device,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # For t=0 the previous cumulative alpha is alpha_bar_{-1}=1. The old
        # code accidentally indexed alpha_bars[-1], i.e. the final timestep.
        self.alpha_bars_previous = F.pad(
            self.alpha_bars[:-1],
            (1, 0),
            value=1.0,
        )
        self.posterior_variance = (
            self.betas
            * (1.0 - self.alpha_bars_previous)
            / (1.0 - self.alpha_bars)
        ).clamp(min=1e-20)

    @staticmethod
    def _extract(
        values: torch.Tensor,
        timesteps: torch.Tensor,
        image_shape: torch.Size,
    ) -> torch.Tensor:
        values = values.gather(0, timesteps)
        return values.view(timesteps.shape[0], 1, 1, 1)

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        noise = torch.randn_like(clean_images)
        alpha_bar = self._extract(
            self.alpha_bars,
            timesteps,
            clean_images.shape,
        )
        noisy_images = (
            alpha_bar.sqrt() * clean_images
            + (1.0 - alpha_bar).sqrt() * noise
        )
        return noisy_images, noise

    @torch.inference_mode()
    def reverse_step(
        self,
        model: UNetCond,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        predicted_noise = model(noisy_images, timesteps, labels)
        alpha = self._extract(self.alphas, timesteps, noisy_images.shape)
        alpha_bar = self._extract(
            self.alpha_bars,
            timesteps,
            noisy_images.shape,
        )
        beta = self._extract(self.betas, timesteps, noisy_images.shape)
        variance = self._extract(
            self.posterior_variance,
            timesteps,
            noisy_images.shape,
        )
        mean = (
            noisy_images
            - beta * predicted_noise / (1.0 - alpha_bar).sqrt()
        ) / alpha.sqrt()
        random_noise = torch.randn_like(noisy_images)
        nonzero_mask = (timesteps > 0).float()[:, None, None, None]
        return mean + nonzero_mask * variance.sqrt() * random_noise

    @torch.inference_mode()
    def sample(
        self,
        model: UNetCond,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Generate exactly one image for every supplied label.

        Deriving the batch size from ``labels`` fixes the original mismatch:
        requesting labels ``[1, 8]`` now creates two images rather than keeping
        the old default batch of twenty images.
        """
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("labels must be a non-empty one-dimensional tensor")
        labels = labels.to(self.device, dtype=torch.long)
        images = torch.randn(
            len(labels),
            model.config.image_channels,
            model.config.image_size,
            model.config.image_size,
            device=self.device,
        )
        was_training = model.training
        model.eval()
        for timestep in tqdm(
            range(self.config.num_timesteps - 1, -1, -1),
            desc="conditional sampling",
        ):
            timesteps = torch.full(
                (len(labels),),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            images = self.reverse_step(model, images, timesteps, labels)
        model.train(was_training)
        return images.clamp(-1.0, 1.0)


def amp_settings(
    device: torch.device,
    requested: bool,
) -> tuple[bool, torch.dtype]:
    enabled = requested and device.type == "cuda"
    if not enabled:
        return False, torch.float32
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def build_training_loader(
    data_dir: Path,
    batch_size: int,
    num_workers: int,
    download: bool,
    seed: int,
    device: torch.device,
) -> DataLoader:
    # Diffusion is centered around zero, so map MNIST pixels [0,1] -> [-1,1].
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]
    )
    dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=download,
    )
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
        "generator": torch.Generator().manual_seed(seed),
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(dataset, **options)


def save_grid(images: torch.Tensor, path: Path, row_size: int) -> None:
    """Save normalized diffusion images without requiring a GUI display."""
    path.parent.mkdir(parents=True, exist_ok=True)
    display_images = (images.float().cpu() + 1.0) / 2.0
    save_image(display_images, path, nrow=row_size, padding=2)


def save_training_preview(
    model: UNetCond,
    diffuser: Diffuser,
    device: torch.device,
    path: Path,
    seed: int,
) -> None:
    """Generate two examples per digit and save one comparable preview grid.

    Every epoch uses the same sampling seed, hence the same initial Gaussian
    noise and reverse-process noise. ``fork_rng`` restores the training random
    state afterwards, so preview generation cannot change the next epoch's
    data order, sampled timesteps, or training noise.
    """
    cuda_devices: list[int] = []
    if device.type == "cuda":
        cuda_devices = [
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        ]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(seed)
        preview_labels = torch.arange(
            model.config.num_labels,
            device=device,
        ).repeat_interleave(2)
        preview_images = diffuser.sample(model, preview_labels)
    save_grid(preview_images, path, row_size=4)
    print(f"Saved training preview: {path}")


def save_checkpoint(
    path: Path,
    model: UNetCond,
    optimizer: Adam,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    epoch: int,
    losses: list[float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "model_config": asdict(model_config),
            "diffusion_config": asdict(diffusion_config),
            "epoch": epoch,
            "losses": losses,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def train(args: argparse.Namespace) -> Path:
    """Train the conditional DDPM, save it as .pt, and return its path."""
    seed_everything(args.seed)
    device = select_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    model_config = ModelConfig(
        time_embedding_dim=args.time_embedding_dim,
    )
    diffusion_config = DiffusionConfig(
        num_timesteps=args.num_timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
    )
    loader = build_training_loader(
        args.data_dir,
        args.batch_size,
        args.num_workers,
        args.download,
        args.seed,
        device,
    )
    model = UNetCond(model_config).to(device)
    diffuser = Diffuser(diffusion_config, device)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    amp_enabled, amp_dtype = amp_settings(device, args.amp)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    print("=" * 80)
    print("Training class-conditioned MNIST DDPM")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Training images: {len(loader.dataset):,}")
    print(f"Epochs: {args.epochs}")
    print(f"Diffusion timesteps: {diffusion_config.num_timesteps}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 80)

    losses: list[float] = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        loss_sum = 0.0
        sample_count = 0
        progress = tqdm(loader, desc=f"epoch {epoch}/{args.epochs}")
        for clean_images, labels in progress:
            clean_images = clean_images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            timesteps = torch.randint(
                0,
                diffusion_config.num_timesteps,
                (clean_images.shape[0],),
                device=device,
            )
            # 添加noise for clean-image
            noisy_images, target_noise = diffuser.add_noise(
                clean_images,
                timesteps,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=amp_dtype,
                enabled=amp_enabled,
            ):
                predicted_noise = model(noisy_images, timesteps, labels)
                loss = F.mse_loss(
                    predicted_noise.float(),
                    target_noise.float(),
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            gradient_norm = nn.utils.clip_grad_norm_(
                model.parameters(),
                args.max_grad_norm,
            )
            scaler.step(optimizer)
            scaler.update()

            loss_sum += loss.item() * clean_images.shape[0]
            sample_count += clean_images.shape[0]
            progress.set_postfix(
                loss=f"{loss.item():.4f}",
                grad=f"{float(gradient_norm):.2f}",
            )
        average_loss = loss_sum / sample_count
        losses.append(average_loss)
        print(f"epoch {epoch}/{args.epochs} | average loss {average_loss:.6f}")

        # Save the latest usable checkpoint before the expensive 1000-step
        # reverse preview. If sampling or the job is interrupted, training up
        # to this epoch is still recoverable from the .pt file.
        save_checkpoint(
            args.checkpoint,
            model,
            optimizer,
            model_config,
            diffusion_config,
            epoch,
            losses,
        )
        epoch_preview_path = args.checkpoint.with_name(
            f"training-epoch-{epoch}-preview.png"
        )
        save_training_preview(
            model,
            diffuser,
            device,
            epoch_preview_path,
            args.seed,
        )

    # Write the final checkpoint once more to make the completed state explicit.
    save_checkpoint(
        args.checkpoint,
        model,
        optimizer,
        model_config,
        diffusion_config,
        args.epochs,
        losses,
    )
    history_path = args.checkpoint.with_suffix(".json")
    with history_path.open("w", encoding="utf-8") as output_file:
        json.dump(
            {"epoch": args.epochs, "losses": losses},
            output_file,
            indent=2,
        )
    print(f"Saved checkpoint: {args.checkpoint}")
    print(f"Saved loss history: {history_path}")

    # Generate a clearly named final preview after all optimization is done.
    final_preview_path = args.checkpoint.with_name(
        "training-final-preview.png"
    )
    save_training_preview(
        model,
        diffuser,
        device,
        final_preview_path,
        args.seed,
    )
    return args.checkpoint


def load_trained_model(
    checkpoint_path: str | Path,
    device: str | torch.device = "auto",
) -> tuple[UNetCond, Diffuser, torch.device]:
    """Reconstruct the exact model and diffusion schedule from a checkpoint."""
    checkpoint_path = Path(checkpoint_path).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    resolved_device = select_device(device)
    checkpoint = torch.load(
        checkpoint_path,
        map_location=resolved_device,
        weights_only=True,
    )
    model_config = ModelConfig(**checkpoint["model_config"])
    diffusion_config = DiffusionConfig(**checkpoint["diffusion_config"])
    model = UNetCond(model_config).to(resolved_device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    diffuser = Diffuser(diffusion_config, resolved_device)
    return model, diffuser, resolved_device


def test(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    digit: int = 7,
    num_samples: int = 16,
    output_path: str | Path | None = None,
    device: str | torch.device = "auto",
    seed: int = 42,
) -> tuple[torch.Tensor, Path]:
    """Load a .pt checkpoint and generate ``num_samples`` of one digit.

    Example from another Python file:

        from importlib import import_module
        # Because this filename contains a hyphen, CLI use is usually easier.
        # images, path = test(digit=7, num_samples=16)
    """
    if not 0 <= digit <= 9:
        raise ValueError("digit must be between 0 and 9")
    if num_samples < 1:
        raise ValueError("num_samples must be positive")
    seed_everything(seed)
    checkpoint_path = Path(checkpoint_path).expanduser()
    model, diffuser, resolved_device = load_trained_model(
        checkpoint_path,
        device,
    )
    labels = torch.full(
        (num_samples,),
        digit,
        device=resolved_device,
        dtype=torch.long,
    )
    images = diffuser.sample(model, labels)
    if output_path is None:
        output_path = checkpoint_path.with_name(f"test-digit-{digit}.png")
    output_path = Path(output_path).expanduser()
    row_size = max(1, math.ceil(math.sqrt(num_samples)))
    save_grid(images, output_path, row_size)
    print(f"Generated {num_samples} image(s) conditioned on digit {digit}")
    print(f"Saved test image grid: {output_path}")
    return images.cpu(), output_path


def main() -> None:
    args = parse_args()
    if args.mode == "train":
        train(args)
        return
    test(
        checkpoint_path=args.checkpoint,
        digit=args.digit,
        num_samples=args.num_samples,
        output_path=args.output,
        device=args.device,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

#    ### 1. Train and save the model
#
#   cd ~/scratch/dips_project/reinforcement_learning/multip_modal/cond-
#   diffusion-model
#
  # python code-1-fixed.py \
  #     --mode train \
  #     --device cuda \
  #     --epochs 10
#
#   After training, it saves:
#
#   checkpoints/mnist-cond-diffusion.pt
#   checkpoints/mnist-cond-diffusion.json
#   checkpoints/training-preview.png
#
#   The .pt checkpoint contains:
#
#   - U-Net weights
#   - Optimizer state
#   - Model configuration
#   - Diffusion configuration
#   - Training epoch and loss history
#
#   The checkpoint is saved before preview generation, so sampling
#   interruption will not lose the trained model.
#
#   ### 2. Generate pictures of digit 7
#
#   python code-1-fixed.py \
#       --mode test \
#       --device cuda \
#       --digit 7 \
#       --num-samples 16
#
#   This loads the saved .pt model and generates 16 new images conditioned on
#   label 7.
#
#   Output:
#
#   checkpoints/test-digit-7.png
#
#   You can generate another digit similarly:
#
#   python code-1-fixed.py \
#       --mode test \
#       --digit 3 \
#       --num-samples 25
#
#   The callable Python function is available at test() (multip_modal/cond-
#   diffusion-model/code-1-fixed.py:624).
#
#   I also corrected the reverse-diffusion final-step indexing, label/batch-
#   size mismatch, image normalization, server-incompatible plotting, model-
#   mode switching, and checkpoint reconstruction. The checkpoint directory is
#   now ignored by Git.
