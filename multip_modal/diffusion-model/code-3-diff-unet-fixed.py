"""Train an educational DDPM with a time-conditioned U-Net.

This is a corrected, server-friendly version of ``code-3.py``. It supports:

* ``--dataset demo`` (default): repeatedly trains on trump.jpeg and biden.jpeg;
* ``--dataset flickr8k``: trains an *unconditional* image generator on the
  6,000 Flickr8k training images and evaluates denoising loss on its official
  validation/test splits.

Important Flickr8k distinction
===============================
The model in this file is not text-conditioned. Flickr8k captions are ignored;
the model learns p(image), not p(image | caption). To generate an image from a
caption, the U-Net would need text embeddings, cross-attention or another
conditioning mechanism, classifier-free guidance training, and a conditional
sampling path. Flickr8k is still valid as a small natural-image dataset for
learning and testing unconditional diffusion.

DDPM objective
==============
For a clean image x_0, timestep t, and Gaussian noise epsilon:

    x_t = sqrt(alpha_bar_t) x_0 + sqrt(1 - alpha_bar_t) epsilon

The U-Net receives (x_t, t) and predicts epsilon. Training minimizes:

    L = E[ ||epsilon - epsilon_theta(x_t, t)||^2 ]

Sampling starts with x_T ~ N(0, I) and repeatedly applies the learned reverse
transition until x_0 is reached.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

import numpy as np
from PIL import Image
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.utils import save_image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_FLICKR8K_DIR = PROJECT_ROOT / "datasets" / "flickr8k" / "data"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "code-3-diff-unet-fixed"


@dataclass(frozen=True)
class UNetConfig:
    image_channels: int = 3
    base_channels: int = 64
    time_embedding_dim: int = 256
    dropout: float = 0.1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a DDPM U-Net on the two-image demo or Flickr8k."
    )
    parser.add_argument(
        "--dataset",
        choices=("demo", "flickr8k"),
        default="demo",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_FLICKR8K_DIR,
        help="Directory containing Flickr8k train/validation/test parquet files.",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Defaults to 1000 for demo and 100 for Flickr8k.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--time-embedding-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument(
        "--sample-every",
        type=int,
        default=None,
        help="Defaults to 100 epochs for demo and 10 for Flickr8k.",
    )
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument(
        "--demo-repeats",
        type=int,
        default=64,
        help="Repeat each of the two demo images this many times per epoch.",
    )
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Smoke-test limit; 0 uses the complete training split.",
    )
    parser.add_argument(
        "--max-eval-images",
        type=int,
        default=0,
        help="Smoke-test limit per validation/test split; 0 uses all images.",
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
        help="Use BF16/FP16 mixed precision on CUDA.",
    )
    args = parser.parse_args()
    if args.epochs is None:
        args.epochs = 1000 if args.dataset == "demo" else 100
    if args.sample_every is None:
        args.sample_every = 100 if args.dataset == "demo" else 10
    validate_args(args)
    return args


def validate_args(args: argparse.Namespace) -> None:
    positive = (
        "epochs",
        "batch_size",
        "image_size",
        "learning_rate",
        "max_grad_norm",
        "timesteps",
        "base_channels",
        "time_embedding_dim",
        "sample_every",
        "eval_every",
        "num_samples",
        "demo_repeats",
    )
    for name in positive:
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("--num-workers cannot be negative")
    if args.max_train_images < 0 or args.max_eval_images < 0:
        raise ValueError("dataset limits cannot be negative")
    if args.image_size % 4:
        raise ValueError("--image-size must be divisible by 4")
    if args.base_channels % 8:
        raise ValueError("--base-channels must be divisible by 8")
    if args.time_embedding_dim < 4:
        raise ValueError("--time-embedding-dim must be at least 4")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1)")
    if not 0.0 < args.ema_decay < 1.0:
        raise ValueError("--ema-decay must be in (0, 1)")
    if not 0.0 < args.beta_start < args.beta_end < 1.0:
        raise ValueError("require 0 < beta-start < beta-end < 1")
    if args.weight_decay < 0:
        raise ValueError("--weight-decay cannot be negative")


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested, but CUDA is unavailable")
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


class DemoImageDataset(Dataset):
    """The original two images, repeated to form one real mini-batch."""

    def __init__(self, transform: Any, repeats: int) -> None:
        self.paths = (SCRIPT_DIR / "trump.jpeg", SCRIPT_DIR / "biden.jpeg")
        missing = [str(path) for path in self.paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(f"Missing demo images: {missing}")
        self.transform = transform
        self.repeats = repeats

    def __len__(self) -> int:
        return len(self.paths) * self.repeats

    def __getitem__(self, index: int) -> torch.Tensor:
        path = self.paths[index % len(self.paths)]
        with Image.open(path) as image:
            return self.transform(image.convert("RGB"))


class Flickr8kImageDataset(Dataset):
    """Expose only images; captions are intentionally unused by this DDPM."""

    def __init__(self, dataset: Any, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = self.dataset[index]["image"].convert("RGB")
        return self.transform(image)


def create_transforms(image_size: int) -> tuple[Any, Any]:
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.8, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size + image_size // 8,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform, evaluation_transform


def find_parquet_files(data_dir: Path, split: str) -> list[str]:
    files = sorted(data_dir.glob(f"{split}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {split}-*.parquet files found in {data_dir}"
        )
    return [str(path) for path in files]


def limit_dataset(dataset: Any, maximum: int) -> Any:
    if maximum == 0 or maximum >= len(dataset):
        return dataset
    return dataset.select(range(maximum))


def load_flickr8k(data_dir: Path) -> Any:
    """Load local parquet only; this never downloads from the Hub."""
    from datasets import load_dataset

    if not data_dir.is_dir():
        raise FileNotFoundError(f"Flickr8k directory missing: {data_dir}")
    dataset = load_dataset(
        "parquet",
        data_files={
            split: find_parquet_files(data_dir, split)
            for split in ("train", "validation", "test")
        },
    )
    for split in ("train", "validation", "test"):
        if "image" not in dataset[split].column_names:
            raise ValueError(f"Flickr8k {split} split has no image column")
    return dataset


def loader_options(
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": num_workers > 0,
    }
    if num_workers > 0:
        options["prefetch_factor"] = 2
    return options


def create_data_loaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader | None, DataLoader | None, dict[str, int]]:
    train_transform, evaluation_transform = create_transforms(args.image_size)
    options = loader_options(args.batch_size, args.num_workers, device)
    generator = torch.Generator().manual_seed(args.seed)

    if args.dataset == "demo":
        training_dataset = DemoImageDataset(
            train_transform,
            args.demo_repeats,
        )
        training_loader = DataLoader(
            training_dataset,
            shuffle=True,
            generator=generator,
            **options,
        )
        return (
            training_loader,
            None,
            None,
            {"train": len(training_dataset), "validation": 0, "test": 0},
        )

    dataset = load_flickr8k(args.data_dir)
    dataset["train"] = limit_dataset(
        dataset["train"],
        args.max_train_images,
    )
    dataset["validation"] = limit_dataset(
        dataset["validation"],
        args.max_eval_images,
    )
    dataset["test"] = limit_dataset(
        dataset["test"],
        args.max_eval_images,
    )
    training_dataset = Flickr8kImageDataset(
        dataset["train"],
        train_transform,
    )
    validation_dataset = Flickr8kImageDataset(
        dataset["validation"],
        evaluation_transform,
    )
    test_dataset = Flickr8kImageDataset(
        dataset["test"],
        evaluation_transform,
    )
    counts = {
        "train": len(training_dataset),
        "validation": len(validation_dataset),
        "test": len(test_dataset),
    }
    return (
        DataLoader(
            training_dataset,
            shuffle=True,
            generator=generator,
            **options,
        ),
        DataLoader(validation_dataset, shuffle=False, **options),
        DataLoader(test_dataset, shuffle=False, **options),
        counts,
    )


def sinusoidal_time_embedding(
    timesteps: torch.Tensor,
    embedding_dim: int,
) -> torch.Tensor:
    """Vectorized Transformer-style sinusoidal timestep representation."""
    half_dim = embedding_dim // 2
    denominator = max(half_dim - 1, 1)
    frequencies = torch.exp(
        -math.log(10_000)
        * torch.arange(half_dim, device=timesteps.device)
        / denominator
    )
    angles = timesteps.float()[:, None] * frequencies[None, :]
    embedding = torch.cat((angles.sin(), angles.cos()), dim=-1)
    if embedding_dim % 2:
        embedding = F.pad(embedding, (0, 1))
    return embedding


class ResidualTimeBlock(nn.Module):
    """Residual convolution block modulated by the diffusion timestep."""

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        time_embedding_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        # The first block has three RGB channels, so a fixed eight groups would
        # be invalid. Choose the largest group count up to eight that divides
        # the channel count exactly.
        input_groups = next(
            groups
            for groups in range(min(8, input_channels), 0, -1)
            if input_channels % groups == 0
        )
        output_groups = next(
            groups
            for groups in range(min(8, output_channels), 0, -1)
            if output_channels % groups == 0
        )
        self.norm1 = nn.GroupNorm(input_groups, input_channels)
        self.conv1 = nn.Conv2d(input_channels, output_channels, 3, padding=1)
        self.time_projection = nn.Linear(
            time_embedding_dim,
            output_channels,
        )
        self.norm2 = nn.GroupNorm(output_groups, output_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(output_channels, output_channels, 3, padding=1)
        self.residual = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1)
        )

    def forward(
        self,
        images: torch.Tensor,
        time_embedding: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(images)))
        time_features = self.time_projection(F.silu(time_embedding))
        hidden = hidden + time_features[:, :, None, None]
        hidden = self.conv2(self.dropout(F.silu(self.norm2(hidden))))
        return hidden + self.residual(images)


class DiffusionUNet(nn.Module):
    """Two-level U-Net with skip connections and timestep conditioning."""

    def __init__(self, config: UNetConfig) -> None:
        super().__init__()
        channels = config.base_channels
        self.time_embedding_dim = config.time_embedding_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(config.time_embedding_dim, config.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(config.time_embedding_dim * 4, config.time_embedding_dim),
        )

        self.down_block1 = ResidualTimeBlock(
            config.image_channels,
            channels,
            config.time_embedding_dim,
            config.dropout,
        )
        self.downsample1 = nn.Conv2d(channels, channels, 4, stride=2, padding=1)
        self.down_block2 = ResidualTimeBlock(
            channels,
            channels * 2,
            config.time_embedding_dim,
            config.dropout,
        )
        self.downsample2 = nn.Conv2d(
            channels * 2,
            channels * 2,
            4,
            stride=2,
            padding=1,
        )

        self.middle1 = ResidualTimeBlock(
            channels * 2,
            channels * 4,
            config.time_embedding_dim,
            config.dropout,
        )
        self.middle2 = ResidualTimeBlock(
            channels * 4,
            channels * 4,
            config.time_embedding_dim,
            config.dropout,
        )

        self.upsample2 = nn.ConvTranspose2d(
            channels * 4,
            channels * 2,
            4,
            stride=2,
            padding=1,
        )
        self.up_block2 = ResidualTimeBlock(
            channels * 4,
            channels * 2,
            config.time_embedding_dim,
            config.dropout,
        )
        self.upsample1 = nn.ConvTranspose2d(
            channels * 2,
            channels,
            4,
            stride=2,
            padding=1,
        )
        self.up_block1 = ResidualTimeBlock(
            channels * 2,
            channels,
            config.time_embedding_dim,
            config.dropout,
        )
        self.output_norm = nn.GroupNorm(8, channels)
        self.output = nn.Conv2d(channels, config.image_channels, 3, padding=1)
        # Starting near a zero noise prediction stabilizes early optimization.
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        time_embedding = sinusoidal_time_embedding(
            timesteps,
            self.time_embedding_dim,
        )
        time_embedding = self.time_mlp(time_embedding)

        skip1 = self.down_block1(noisy_images, time_embedding)
        hidden = self.downsample1(skip1)
        skip2 = self.down_block2(hidden, time_embedding)
        hidden = self.downsample2(skip2)

        hidden = self.middle1(hidden, time_embedding)
        hidden = self.middle2(hidden, time_embedding)

        hidden = self.upsample2(hidden)
        hidden = self.up_block2(
            torch.cat((hidden, skip2), dim=1),
            time_embedding,
        )
        hidden = self.upsample1(hidden)
        hidden = self.up_block1(
            torch.cat((hidden, skip1), dim=1),
            time_embedding,
        )
        return self.output(F.silu(self.output_norm(hidden)))


class DiffusionSchedule:
    """Forward noising coefficients and ancestral DDPM reverse sampling."""

    def __init__(
        self,
        timesteps: int,
        beta_start: float,
        beta_end: float,
        device: torch.device,
    ) -> None:
        self.timesteps = timesteps
        self.device = device
        self.betas = torch.linspace(
            beta_start,
            beta_end,
            timesteps,
            device=device,
            dtype=torch.float32,
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        # alpha_bar_previous[0] must be 1, not alpha_bar[-1]. This fixes the
        # t=0 indexing bug in the original implementation.
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
        extracted = values.gather(0, timesteps)
        return extracted.view(timesteps.shape[0], *([1] * (len(image_shape) - 1)))

    def add_noise(
        self,
        clean_images: torch.Tensor,
        timesteps: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
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
        model: nn.Module,
        noisy_images: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        predicted_noise = model(noisy_images, timesteps)
        alpha = self._extract(self.alphas, timesteps, noisy_images.shape)
        alpha_bar = self._extract(
            self.alpha_bars,
            timesteps,
            noisy_images.shape,
        )
        beta = self._extract(self.betas, timesteps, noisy_images.shape)
        posterior_variance = self._extract(
            self.posterior_variance,
            timesteps,
            noisy_images.shape,
        )
        posterior_mean = (
            noisy_images
            - beta * predicted_noise / (1.0 - alpha_bar).sqrt()
        ) / alpha.sqrt()

        noise = torch.randn_like(noisy_images)
        nonzero_mask = (timesteps > 0).float().view(
            timesteps.shape[0],
            1,
            1,
            1,
        )
        return (
            posterior_mean
            + nonzero_mask * posterior_variance.sqrt() * noise
        )

    @torch.inference_mode()
    def sample(
        self,
        model: nn.Module,
        sample_count: int,
        image_size: int,
    ) -> torch.Tensor:
        was_training = model.training
        model.eval()
        images = torch.randn(
            sample_count,
            3,
            image_size,
            image_size,
            device=self.device,
        )
        for timestep in tqdm(
            range(self.timesteps - 1, -1, -1),
            desc="DDPM sampling",
            leave=False,
        ):
            timesteps = torch.full(
                (sample_count,),
                timestep,
                device=self.device,
                dtype=torch.long,
            )
            images = self.reverse_step(model, images, timesteps)
        model.train(was_training)
        return images.clamp(-1.0, 1.0)


@torch.no_grad()
def update_ema(
    ema_model: nn.Module,
    model: nn.Module,
    decay: float,
) -> None:
    for ema_parameter, parameter in zip(
        ema_model.parameters(),
        model.parameters(),
        strict=True,
    ):
        ema_parameter.lerp_(parameter, 1.0 - decay)
    for ema_buffer, buffer in zip(
        ema_model.buffers(),
        model.buffers(),
        strict=True,
    ):
        ema_buffer.copy_(buffer)


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


def train_one_epoch(
    model: nn.Module,
    ema_model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    schedule: DiffusionSchedule,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
    ema_decay: float,
    epoch: int,
    total_epochs: int,
) -> float:
    model.train()
    loss_sum = 0.0
    example_count = 0
    progress = tqdm(loader, desc=f"epoch {epoch}/{total_epochs}")
    for clean_images in progress:
        clean_images = clean_images.to(device, non_blocking=True)
        timesteps = torch.randint(
            0,
            schedule.timesteps,
            (clean_images.shape[0],),
            device=device,
        )
        noisy_images, target_noise = schedule.add_noise(
            clean_images,
            timesteps,
        )
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            predicted_noise = model(noisy_images, timesteps)
            loss = F.mse_loss(predicted_noise.float(), target_noise.float())

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        gradient_norm = nn.utils.clip_grad_norm_(
            model.parameters(),
            max_grad_norm,
        )
        scaler.step(optimizer)
        scaler.update()
        update_ema(ema_model, model, ema_decay)

        batch_size = clean_images.shape[0]
        loss_sum += loss.item() * batch_size
        example_count += batch_size
        progress.set_postfix(
            loss=f"{loss.item():.4f}",
            grad=f"{float(gradient_norm):.2f}",
        )
    return loss_sum / max(example_count, 1)


@torch.inference_mode()
def evaluate_denoising_loss(
    model: nn.Module,
    loader: DataLoader,
    schedule: DiffusionSchedule,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> float:
    model.eval()
    loss_sum = 0.0
    example_count = 0
    for clean_images in loader:
        clean_images = clean_images.to(device, non_blocking=True)
        timesteps = torch.randint(
            0,
            schedule.timesteps,
            (clean_images.shape[0],),
            device=device,
        )
        noisy_images, target_noise = schedule.add_noise(
            clean_images,
            timesteps,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            predicted_noise = model(noisy_images, timesteps)
        loss = F.mse_loss(
            predicted_noise.float(),
            target_noise.float(),
            reduction="sum",
        )
        loss_sum += loss.item()
        example_count += clean_images.numel()
    return loss_sum / max(example_count, 1)


def save_sample_grid(
    images: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    images = (images.float().cpu() + 1.0) / 2.0
    columns = max(1, int(math.sqrt(len(images))))
    save_image(images, path, nrow=columns, padding=2)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    ema_model: nn.Module,
    optimizer: AdamW,
    config: UNetConfig,
    args: argparse.Namespace,
    epoch: int,
    validation_loss: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "ema_model_state_dict": ema_model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": asdict(config),
        "training_args": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "epoch": epoch,
        "validation_loss": validation_loss,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    device = select_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    training_loader, validation_loader, test_loader, split_sizes = (
        create_data_loaders(args, device)
    )
    config = UNetConfig(
        base_channels=args.base_channels,
        time_embedding_dim=args.time_embedding_dim,
        dropout=args.dropout,
    )
    model = DiffusionUNet(config).to(device)
    ema_model = deepcopy(model).to(device).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    schedule = DiffusionSchedule(
        args.timesteps,
        args.beta_start,
        args.beta_end,
        device,
    )
    amp_enabled, amp_dtype = amp_settings(device, args.amp)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )
    parameter_count = sum(p.numel() for p in model.parameters())

    print("=" * 80)
    print("DDPM time-conditioned U-Net")
    print(f"Dataset: {args.dataset}")
    print(f"Split sizes: {split_sizes}")
    if args.dataset == "flickr8k":
        print("Conditioning: unconditional; Flickr8k captions are not used")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {amp_enabled} ({amp_dtype if amp_enabled else 'disabled'})")
    print(f"Parameters: {parameter_count:,}")
    print(f"Image size: {args.image_size}x{args.image_size}")
    print(f"Diffusion timesteps: {args.timesteps}")
    print(f"Epochs: {args.epochs}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)

    checkpoint_path = args.output_dir / "best.pt"
    history: list[dict[str, float | int]] = []
    best_validation_loss = float("inf")
    started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = train_one_epoch(
            model,
            ema_model,
            training_loader,
            optimizer,
            scaler,
            schedule,
            device,
            amp_enabled,
            amp_dtype,
            args.max_grad_norm,
            args.ema_decay,
            epoch,
            args.epochs,
        )

        validation_loss = train_loss
        should_evaluate = (
            validation_loader is not None
            and (epoch % args.eval_every == 0 or epoch == args.epochs)
        )
        if should_evaluate:
            validation_loss = evaluate_denoising_loss(
                ema_model,
                validation_loader,
                schedule,
                device,
                amp_enabled,
                amp_dtype,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation_loss,
            }
        )
        print(
            f"epoch {epoch:04d}/{args.epochs} | train {train_loss:.6f} | "
            f"validation {validation_loss:.6f} | "
            f"{time.time() - epoch_started:.1f}s"
        )

        # For Flickr8k, compare checkpoints only on actual evaluation epochs.
        # For the demo, training loss is the only available diagnostic.
        eligible_for_best = args.dataset == "demo" or should_evaluate
        if eligible_for_best and validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            save_checkpoint(
                checkpoint_path,
                model,
                ema_model,
                optimizer,
                config,
                args,
                epoch,
                validation_loss,
            )
            print(f"Saved best checkpoint: {checkpoint_path}")

        if epoch % args.sample_every == 0 or epoch == args.epochs:
            samples = schedule.sample(
                ema_model,
                args.num_samples,
                args.image_size,
            )
            sample_path = args.output_dir / "samples" / f"epoch-{epoch:04d}.png"
            save_sample_grid(samples, sample_path)
            print(f"Saved samples: {sample_path}")

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    ema_model.load_state_dict(checkpoint["ema_model_state_dict"])
    test_loss: float | None = None
    if test_loader is not None:
        test_loss = evaluate_denoising_loss(
            ema_model,
            test_loader,
            schedule,
            device,
            amp_enabled,
            amp_dtype,
        )
        print(f"Test denoising MSE per pixel/channel: {test_loss:.6f}")

    final_samples = schedule.sample(
        ema_model,
        args.num_samples,
        args.image_size,
    )
    save_sample_grid(final_samples, args.output_dir / "final-samples.png")
    summary = {
        "dataset": args.dataset,
        "unconditional": True,
        "captions_used": False,
        "split_sizes": split_sizes,
        "best_epoch": checkpoint["epoch"],
        "best_validation_loss": checkpoint["validation_loss"],
        "test_loss": test_loss,
        "parameters": parameter_count,
        "training_seconds": time.time() - started,
        "history": history,
    }
    with (args.output_dir / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as output_file:
        json.dump(summary, output_file, indent=2)

    print("=" * 80)
    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Final samples: {args.output_dir / 'final-samples.png'}")
    print(f"Metrics: {args.output_dir / 'metrics.json'}")
    print("=" * 80)


if __name__ == "__main__":
    main()
