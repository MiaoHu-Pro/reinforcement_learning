"""Train a small Vision Transformer (ViT) on MNIST.

This script is an educational, corrected version of ``code-1-vit.py``.  It
implements the important ViT operations explicitly instead of hiding them in
``nn.TransformerEncoder``.

Vision Transformer overview
---------------------------
An image has shape ``[B, C, H, W]``. ViT treats the image as a token sequence:

1. Split the image into non-overlapping ``P x P`` patches.
2. Project every patch to a ``d_model``-dimensional token embedding.
3. Prepend a learned ``[CLS]`` token used to summarize the whole image.
4. Add positional information because self-attention alone does not know the
   spatial order of patches.
5. Repeatedly apply multi-head self-attention and an MLP, both with residual
   connections and LayerNorm.
6. Feed the final ``[CLS]`` representation to a linear classification head.

With the defaults, each MNIST image is ``28 x 28`` and each patch is ``7 x 7``.
The image therefore produces ``(28 / 7)^2 = 16`` patch tokens. After adding
``[CLS]``, the Transformer processes a sequence of 17 tokens.

Important corrections compared with the old script
--------------------------------------------------
* The classifier returns raw logits. ``CrossEntropyLoss`` already applies
  ``log_softmax`` internally, so applying ``Softmax`` in the model was wrong.
* A validation subset selects the best checkpoint; the test set is evaluated
  only once after training.
* ``model.train()`` and ``model.eval()`` are used explicitly.
* Device selection, random seeds, data paths, AMP, and checkpoint paths are
  configurable.
* The patch count is calculated per spatial dimension and input shapes are
  validated with clear errors.
* AdamW, dropout, input normalization, gradient clipping, and cosine learning-
  rate decay make optimization more stable.

The script works on CPU, but the accompanying Slurm script requests one GPU.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import MNIST


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ViTConfig:
    """Architecture settings shared by construction and checkpoint metadata."""

    image_size: tuple[int, int] = (28, 28)
    patch_size: tuple[int, int] = (7, 7)
    channels: int = 1
    num_classes: int = 10
    d_model: int = 96
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1


def pair(value: int) -> tuple[int, int]:
    """Convert a scalar image/patch size to ``(height, width)``."""
    return value, value


class PatchEmbedding(nn.Module):
    """Turn non-overlapping image patches into a sequence of token vectors.

    A convolution with ``kernel_size == stride == patch_size`` performs both
    patch extraction and linear projection. For a 28x28 input and 7x7 patches:

        [B, 1, 28, 28] -> [B, d_model, 4, 4] -> [B, 16, d_model]

    This is mathematically equivalent to flattening every patch and applying
    the same learned linear layer to all patches, but Conv2d is more efficient.
    """

    def __init__(
        self,
        image_size: tuple[int, int],
        patch_size: tuple[int, int],
        channels: int,
        d_model: int,
    ) -> None:
        super().__init__()
        image_height, image_width = image_size
        patch_height, patch_width = patch_size

        if image_height % patch_height != 0:
            raise ValueError("image height must be divisible by patch height")
        if image_width % patch_width != 0:
            raise ValueError("image width must be divisible by patch width")

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (
            (image_height // patch_height)
            * (image_width // patch_width)
        )
        self.projection = nn.Conv2d(
            in_channels=channels,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4:
            raise ValueError(
                "images must have shape [batch, channels, height, width]"
            )
        actual_size = tuple(images.shape[-2:])
        if actual_size != self.image_size:
            raise ValueError(
                f"expected image size {self.image_size}, got {actual_size}"
            )

        patch_grid = self.projection(images)
        # Flatten only the two spatial patch-grid dimensions, then transpose so
        # sequence length comes before embedding dimension.
        patch_tokens = patch_grid.flatten(start_dim=2).transpose(1, 2)
        return patch_tokens


class ClassTokenAndPositionEncoding(nn.Module):
    """Prepend a learned class token and add fixed sinusoidal positions.

    Self-attention is permutation-equivariant: without positions it cannot
    distinguish the top-left patch from the bottom-right patch. Sinusoidal
    encodings inject a different deterministic vector at every sequence
    position. The original ViT paper commonly uses learned position embeddings;
    fixed encodings are used here to make the mechanism easy to inspect.
    """

    def __init__(
        self,
        d_model: int,
        sequence_length: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.dropout = nn.Dropout(dropout)

        # Vectorized implementation of:
        # PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
        # PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
        position = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / d_model)
        )
        encoding = torch.zeros(1, sequence_length, d_model)
        encoding[0, :, 0::2] = torch.sin(position * frequencies)
        # The slice is needed when d_model is odd. Our default is even, but the
        # implementation remains correct for either case.
        odd_dimensions = encoding[0, :, 1::2].shape[1]
        encoding[0, :, 1::2] = torch.cos(
            position * frequencies[:odd_dimensions]
        )
        self.register_buffer(
            "position_encoding",
            encoding,
            persistent=True,
        )
        nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        batch_size = patch_tokens.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat((cls_tokens, patch_tokens), dim=1)

        if tokens.shape[1] > self.position_encoding.shape[1]:
            raise ValueError("token sequence is longer than the position encoding")
        tokens = tokens + self.position_encoding[:, :tokens.shape[1]]
        return self.dropout(tokens)


class MultiHeadSelfAttention(nn.Module):
    """Let every image token collect information from every other token.

    One linear layer creates queries, keys, and values for all heads. Each head
    works in a smaller subspace of size ``head_dim = d_model / num_heads``:

        Attention(Q, K, V) = softmax(Q K^T / sqrt(head_dim)) V

    Different heads can learn different relationships, such as stroke
    continuity, distant patch interactions, or the link between ``[CLS]`` and
    informative digit patches.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, d_model)
        self.output_dropout = nn.Dropout(dropout)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, d_model = tokens.shape

        # [B, N, 3*d_model] -> [3, B, heads, N, head_dim]
        qkv = self.qkv(tokens)
        qkv = qkv.reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)

        attention_scores = (queries @ keys.transpose(-2, -1)) * self.scale
        attention_weights = attention_scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended_values = attention_weights @ values

        # Put all heads back beside one another:
        # [B, heads, N, head_dim] -> [B, N, d_model].
        attended_values = attended_values.transpose(1, 2).contiguous()
        attended_values = attended_values.reshape(
            batch_size,
            sequence_length,
            d_model,
        )
        output = self.output_projection(attended_values)
        return self.output_dropout(output)


class TransformerEncoderBlock(nn.Module):
    """One pre-normalized Transformer encoder block.

    Both sublayers use a residual connection. The first sublayer mixes
    information across tokens with attention. The second independently
    transforms every token with an MLP:

        x = x + MSA(LayerNorm(x))
        x = x + MLP(LayerNorm(x))
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        hidden_dim = d_model * mlp_ratio
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens + self.attention(self.norm1(tokens))
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class VisionTransformer(nn.Module):
    """A compact ViT classifier returning raw class logits."""

    def __init__(self, config: ViTConfig) -> None:
        super().__init__()
        if config.num_layers < 1:
            raise ValueError("num_layers must be positive")
        if config.mlp_ratio < 1:
            raise ValueError("mlp_ratio must be positive")
        if not 0.0 <= config.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.config = config
        self.patch_embedding = PatchEmbedding(
            image_size=config.image_size,
            patch_size=config.patch_size,
            channels=config.channels,
            d_model=config.d_model,
        )
        sequence_length = self.patch_embedding.num_patches + 1
        self.token_and_position = ClassTokenAndPositionEncoding(
            d_model=config.d_model,
            sequence_length=sequence_length,
            dropout=config.dropout,
        )
        self.encoder = nn.Sequential(
            *[
                TransformerEncoderBlock(
                    d_model=config.d_model,
                    num_heads=config.num_heads,
                    mlp_ratio=config.mlp_ratio,
                    dropout=config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out")
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        patch_tokens = self.patch_embedding(images)
        tokens = self.token_and_position(patch_tokens)
        tokens = self.encoder(tokens)
        tokens = self.final_norm(tokens)

        # Only the [CLS] token (position zero) goes to the classifier. Return
        # raw logits: CrossEntropyLoss performs log-softmax itself.
        cls_representation = tokens[:, 0]
        logits = self.classifier(cls_representation)
        return logits


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an educational Vision Transformer on MNIST."
    )
    parser.add_argument("--data-dir", type=Path, default=SCRIPT_DIR / "datasets")
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "code-1-vit-fixed-best.pt",
    )
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--d-model", type=int, default=96)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="'auto' uses CUDA when available, otherwise CPU.",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use mixed precision on CUDA; enabled by default.",
    )
    parser.add_argument(
        "--download",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Allow torchvision to download MNIST when it is not already local.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "epochs",
        "batch_size",
        "validation_size",
        "image_size",
        "patch_size",
        "d_model",
        "num_heads",
        "num_layers",
        "mlp_ratio",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive")
    if args.weight_decay < 0:
        raise ValueError("weight-decay cannot be negative")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.image_size % args.patch_size != 0:
        raise ValueError("image-size must be divisible by patch-size")
    if args.d_model % args.num_heads != 0:
        raise ValueError("d-model must be divisible by num-heads")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")
    if args.validation_size >= 60_000:
        raise ValueError("validation-size must be smaller than MNIST train size")


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda was requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    """Seed model initialization, data splitting, and loader shuffling."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_dataloaders(
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create a train/validation/test split without test-set model selection."""
    # These are the standard MNIST training-set statistics. Normalization gives
    # the optimizer inputs with a more convenient scale than raw [0, 1] pixels.
    transform = transforms.Compose(
        [
            transforms.Resize(pair(args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
        ]
    )
    full_training_set = MNIST(
        root=args.data_dir,
        train=True,
        download=args.download,
        transform=transform,
    )
    test_set = MNIST(
        root=args.data_dir,
        train=False,
        download=args.download,
        transform=transform,
    )

    training_size = len(full_training_set) - args.validation_size
    split_generator = torch.Generator().manual_seed(args.seed)
    training_set, validation_set = random_split(
        full_training_set,
        lengths=(training_size, args.validation_size),
        generator=split_generator,
    )

    pin_memory = device.type == "cuda"
    common_loader_args = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": args.num_workers > 0,
    }
    shuffle_generator = torch.Generator().manual_seed(args.seed)
    training_loader = DataLoader(
        training_set,
        shuffle=True,
        generator=shuffle_generator,
        **common_loader_args,
    )
    validation_loader = DataLoader(
        validation_set,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = DataLoader(
        test_set,
        shuffle=False,
        **common_loader_args,
    )
    return training_loader, validation_loader, test_loader


def amp_settings(
    device: torch.device,
    requested: bool,
) -> tuple[bool, torch.dtype]:
    """Prefer BF16 on modern GPUs; otherwise use FP16 CUDA autocast."""
    enabled = requested and device.type == "cuda"
    if not enabled:
        return False, torch.float32
    if torch.cuda.is_bf16_supported():
        return True, torch.bfloat16
    return True, torch.float16


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: AdamW,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        # GradScaler is enabled only for FP16. BF16 has enough exponent range
        # that loss scaling is normally unnecessary.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        examples += batch_size

    return loss_sum / examples, correct / examples


@torch.inference_mode()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits = model(images)
            loss = criterion(logits, labels)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == labels).sum().item()
        examples += batch_size

    return loss_sum / examples, correct / examples


def save_checkpoint(
    path: Path,
    model: VisionTransformer,
    config: ViTConfig,
    epoch: int,
    validation_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "epoch": epoch,
        "validation_accuracy": validation_accuracy,
    }
    # Write to a neighboring temporary file and then rename it, reducing the
    # chance of leaving a partial checkpoint if a job is interrupted mid-save.
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)
    device = select_device(args.device)

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    config = ViTConfig(
        image_size=pair(args.image_size),
        patch_size=pair(args.patch_size),
        d_model=args.d_model,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    training_loader, validation_loader, test_loader = create_dataloaders(
        args,
        device,
    )
    model = VisionTransformer(config).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled, amp_dtype = amp_settings(device, args.amp)
    fp16_scaling = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=fp16_scaling)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    patch_count = model.patch_embedding.num_patches
    print("=" * 72)
    print("MNIST Vision Transformer training")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {amp_enabled} ({amp_dtype if amp_enabled else 'disabled'})")
    print(f"Parameters: {parameter_count:,}")
    print(f"Patch tokens: {patch_count}; total tokens with [CLS]: {patch_count + 1}")
    print(f"Training/validation/test: {len(training_loader.dataset):,} / "
          f"{len(validation_loader.dataset):,} / {len(test_loader.dataset):,}")
    print(f"Checkpoint: {args.output}")
    print("=" * 72)

    best_validation_accuracy = -1.0
    training_started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            criterion=criterion,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_grad_norm=args.max_grad_norm,
        )
        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=validation_loader,
            criterion=criterion,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        scheduler.step()

        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(
                path=args.output,
                model=model,
                config=config,
                epoch=epoch,
                validation_accuracy=validation_accuracy,
            )

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | train acc {train_accuracy:.2%} | "
            f"val loss {validation_loss:.4f} | val acc {validation_accuracy:.2%} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{time.time() - epoch_started:.1f}s"
            + (" | saved" if improved else "")
        )

    # Evaluate the untouched test set with the validation-selected checkpoint.
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )

    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_accuracy": float(checkpoint["validation_accuracy"]),
        "test_loss": test_loss,
        "test_accuracy": test_accuracy,
        "training_seconds": time.time() - training_started,
        "checkpoint": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("=" * 72)
    print(f"Best validation epoch: {summary['best_epoch']}")
    print(f"Best validation accuracy: {summary['best_validation_accuracy']:.2%}")
    print(f"Final test loss: {test_loss:.4f}")
    print(f"Final test accuracy: {test_accuracy:.2%}")
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved summary: {summary_path}")
    print("=" * 72)


if __name__ == "__main__":
    main()
