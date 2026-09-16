"""Train a small educational CLIP model on paired MNIST images and captions.

CLIP theory
===========
CLIP (Contrastive Language-Image Pre-training) learns two encoders:

* an image encoder maps an image ``x_i`` to a vector ``v_i``;
* a text encoder maps a caption ``t_i`` to a vector ``u_i``.

Both encoders project into the same embedding space. Their outputs are L2
normalized, so their dot product is cosine similarity:

    v_i = normalize(image_encoder(x_i))
    u_j = normalize(text_encoder(t_j))
    similarity(i, j) = v_i dot u_j

For a batch, all image/text similarities form a ``[B, B]`` matrix. A learned
positive scale ``exp(logit_scale)`` sharpens or softens that matrix:

    logits(i, j) = exp(logit_scale) * similarity(i, j)

Standard CLIP assumes one matching caption per image inside a batch. It applies
cross-entropy in both directions:

* image-to-text: each row should select its paired caption;
* text-to-image: each column should select its paired image.

The two losses are averaged. This creates *in-batch negatives*: all non-paired
items in the batch act as negative examples without a separate negative-data
pipeline.

Why MNIST needs a multi-positive loss
-------------------------------------
This dataset has only ten captions, so a batch normally contains many images
with the same digit and identical captions. A diagonal-only CLIP loss would
incorrectly say that one image of digit 5 matches only its own caption while
other identical "digit 5" captions are negatives. The implementation below
uses the digit label to mark every same-class image/caption pair as positive.
It averages log-probability across all valid positives in both directions. If
every batch item had a unique label, this reduces to the ordinary symmetric
CLIP objective.

Zero-shot-style classification
------------------------------
After training, the ten class prompts are encoded once. Each test image is
compared with those ten text embeddings, and the most similar prompt determines
the prediction. No additional linear classifier is trained.

This MNIST exercise demonstrates CLIP mechanics, but it is not a reproduction
of large-scale CLIP. There are only ten concepts and one short prompt per
concept, so the result is closer to a contrastively trained multimodal digit
classifier than a general open-vocabulary vision-language model.

Important corrections compared with ``code-2-clip.py``
------------------------------------------------------
* Same-digit pairs are positives instead of false in-batch negatives.
* Tokenization safely truncates UTF-8 bytes and gives padding/BOS/EOS distinct
  IDs; the original tokenizer could exceed its configured sequence length.
* The text mask is ``[B, L]`` and is transformed into a causal key mask inside
  attention; the dataset no longer stores a wasteful ``[L, L]`` mask per item.
* Targets and tensors use the actual logits device instead of hard-coded CUDA.
* The learned value is correctly named ``logit_scale`` and capped at 100 to
  avoid unstable exponential growth.
* Encoders include final LayerNorm, stable initialization, dropout, and safe
  L2 normalization.
* Validation selects the best checkpoint; the test split is used only once.
* Evaluation compares predicted digit IDs directly, not whole token arrays.
* Paths, device, seeds, mixed precision, workers, and output are configurable.


  Important corrections include:

  - Uses a multi-positive CLIP loss for repeated MNIST captions
  - Robust UTF-8 byte tokenizer with distinct PAD/BOS/EOS tokens
  - Correct [batch, sequence] padding masks and causal text attention
  - Image and text features are safely normalized
  - Learned temperature is capped to avoid numerical instability
  - Temperature, LayerNorm, and bias parameters are excluded from weight decay
  - Uses validation data to select the best checkpoint
  - Leaves the test set untouched until final evaluation
  - Evaluates by comparing images against ten class prompts
  - Supports CUDA, mixed precision, gradient clipping, and cosine scheduling
  - Loads the local parquet files without internet access
  - Contains a detailed CLIP theory explanation in its module documentation and comments

  All targeted tests passed, including tokenization, duplicate-positive loss, standard CLIP-loss equivalence, local data loading, forward/backward propagation, training/evaluation, and checkpoint serialization.

"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import time
from typing import Any

from datasets import load_dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = SCRIPT_DIR / "clip-mnist" / "mnist"

DIGIT_NAMES = (
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
)
# desc of the image
CLASS_PROMPTS = tuple(f"an image of digit {name}" for name in DIGIT_NAMES)


@dataclass(frozen=True)
class CLIPConfig:
    """Architecture settings stored alongside the trained weights."""

    image_size: tuple[int, int] = (28, 28)
    patch_size: tuple[int, int] = (7, 7)
    channels: int = 1
    image_width: int = 96
    image_layers: int = 4
    image_heads: int = 4
    text_width: int = 96
    text_layers: int = 2
    text_heads: int = 4
    max_text_length: int = 32
    embedding_dim: int = 64
    mlp_ratio: int = 4
    dropout: float = 0.1


def pair(value: int) -> tuple[int, int]:
    return value, value


class ByteTokenizer:
    """A dependency-free UTF-8 byte tokenizer suitable for this small demo.

    Token IDs are deliberately non-overlapping:

    * 0: padding;
    * 1..256: raw byte value plus one;
    * 257: beginning of text;
    * 258: end of text.

    A real CLIP system normally uses a learned subword tokenizer. Byte-level
    encoding is sufficient for ten short prompts and handles non-ASCII text
    without producing token IDs outside the vocabulary.
    """

    PAD_ID = 0
    BOS_ID = 257
    EOS_ID = 258
    VOCAB_SIZE = 259

    def __init__(self, max_length: int) -> None:
        if max_length < 3:
            raise ValueError("max_length must leave room for BOS, content, and EOS")
        self.max_length = max_length

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        payload = list(text.encode("utf-8"))[: self.max_length - 2]
        token_ids = [self.BOS_ID]
        token_ids.extend(byte + 1 for byte in payload)
        token_ids.append(self.EOS_ID)

        valid_length = len(token_ids)
        token_ids.extend([self.PAD_ID] * (self.max_length - valid_length))
        attention_mask = [1] * valid_length + [0] * (
            self.max_length - valid_length
        )
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        byte_values: list[int] = []
        for token_id in token_ids.tolist():
            if token_id == self.EOS_ID:
                break
            if 1 <= token_id <= 256:
                byte_values.append(token_id - 1)
        return bytes(byte_values).decode("utf-8", errors="replace")


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention supporting text padding and causal masks."""

    def __init__(self, width: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if width % num_heads != 0:
            raise ValueError("encoder width must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(width, 3 * width)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(width, width)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch_size, sequence_length, width = tokens.shape
        qkv = self.qkv(tokens).reshape(
            batch_size,
            sequence_length,
            3,
            self.num_heads,
            self.head_dim,
        ).permute(2, 0, 3, 1, 4)
        queries, keys, values = qkv.unbind(dim=0)
        scores = (queries @ keys.transpose(-2, -1)) * self.scale

        allowed = torch.ones(
            (batch_size, 1, sequence_length, sequence_length),
            dtype=torch.bool,
            device=tokens.device,
        )
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError(
                    "attention_mask must have shape [batch, sequence_length]"
                )
            # Mask keys (the last score dimension). Padded query outputs are
            # harmless because only the valid EOS representation is selected.
            allowed = allowed & attention_mask[:, None, None, :]
        if causal:
            causal_mask = torch.ones(
                (sequence_length, sequence_length),
                dtype=torch.bool,
                device=tokens.device,
            ).tril()
            allowed = allowed & causal_mask[None, None, :, :]

        # Every valid sequence begins with BOS, so every query has at least one
        # allowed key and softmax cannot become all -inf/NaN.
        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        attention_weights = scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended = attention_weights @ values
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            width,
        )
        return self.output_dropout(self.output_projection(attended))


class TransformerEncoderBlock(nn.Module):
    """Pre-LayerNorm attention and MLP sublayers with residual connections."""

    def __init__(
        self,
        width: int,
        num_heads: int,
        mlp_ratio: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(width)
        self.attention = MultiHeadSelfAttention(width, num_heads, dropout)
        self.norm2 = nn.LayerNorm(width)
        self.mlp = nn.Sequential(
            nn.Linear(width, width * mlp_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width * mlp_ratio, width),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        tokens: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal: bool = False,
    ) -> torch.Tensor:
        tokens = tokens + self.attention(
            self.norm1(tokens),
            attention_mask=attention_mask,
            causal=causal,
        )
        tokens = tokens + self.mlp(self.norm2(tokens))
        return tokens


class ImageEncoder(nn.Module):
    """Vision Transformer that maps an image to the shared CLIP space."""

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        image_height, image_width = config.image_size
        patch_height, patch_width = config.patch_size
        if image_height % patch_height or image_width % patch_width:
            raise ValueError("image dimensions must be divisible by patch dimensions")

        self.image_size = config.image_size
        self.num_patches = (
            image_height // patch_height
        ) * (
            image_width // patch_width
        )
        self.patch_projection = nn.Conv2d(
            config.channels,
            config.image_width,
            kernel_size=config.patch_size,
            stride=config.patch_size,
        )
        self.class_token = nn.Parameter(
            torch.zeros(1, 1, config.image_width)
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, config.image_width)
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.image_width,
                    config.image_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.image_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.image_width)
        self.projection = nn.Linear(
            config.image_width,
            config.embedding_dim,
            bias=False,
        )

        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.01)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if images.ndim != 4 or tuple(images.shape[-2:]) != self.image_size:
            raise ValueError(
                f"images must have shape [B, C, {self.image_size[0]}, "
                f"{self.image_size[1]}]"
            )
        patch_tokens = self.patch_projection(images)
        patch_tokens = patch_tokens.flatten(start_dim=2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, patch_tokens), dim=1)
        tokens = self.input_dropout(tokens + self.position_embedding)

        # Image patches use bidirectional attention: every patch can see every
        # other patch because the complete image is available simultaneously.
        for block in self.blocks:
            tokens = block(tokens, causal=False)
        class_representation = self.final_norm(tokens)[:, 0]
        image_features = self.projection(class_representation)
        return F.normalize(image_features, dim=-1, eps=1e-6)


class TextEncoder(nn.Module):
    """Causal Transformer that maps caption EOS states to the shared space."""

    def __init__(self, config: CLIPConfig, vocab_size: int) -> None:
        super().__init__()
        self.max_text_length = config.max_text_length
        self.token_embedding = nn.Embedding(
            vocab_size,
            config.text_width,
            padding_idx=ByteTokenizer.PAD_ID,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.text_width)
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderBlock(
                    config.text_width,
                    config.text_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.text_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.text_width)
        self.projection = nn.Linear(
            config.text_width,
            config.embedding_dim,
            bias=False,
        )
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        with torch.no_grad():
            self.token_embedding.weight[ByteTokenizer.PAD_ID].zero_()
        nn.init.trunc_normal_(self.position_embedding, std=0.01)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.ndim != 2:
            raise ValueError("token_ids must have shape [batch, sequence_length]")
        if token_ids.shape != attention_mask.shape:
            raise ValueError("token_ids and attention_mask must have equal shape")
        if token_ids.shape[1] > self.max_text_length:
            raise ValueError("text sequence exceeds configured maximum length")
        if not torch.all(attention_mask.sum(dim=1) >= 2):
            raise ValueError("each text must contain at least BOS and EOS")

        sequence_length = token_ids.shape[1]
        tokens = self.token_embedding(token_ids)
        tokens = tokens + self.position_embedding[:, :sequence_length]
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            tokens = block(
                tokens,
                attention_mask=attention_mask,
                causal=True,
            )
        tokens = self.final_norm(tokens)

        # Since valid tokens are [BOS, payload, EOS], mask.sum() - 1 is the EOS
        # position even when different examples contain different text lengths.
        eos_positions = attention_mask.sum(dim=1, dtype=torch.long) - 1
        batch_indices = torch.arange(token_ids.shape[0], device=token_ids.device)
        eos_states = tokens[batch_indices, eos_positions]
        text_features = self.projection(eos_states)
        return F.normalize(text_features, dim=-1, eps=1e-6)


class CLIPModel(nn.Module):
    """Dual encoder producing symmetric image/text similarity logits."""

    MAX_LOGIT_SCALE = 100.0

    def __init__(self, config: CLIPConfig, vocab_size: int) -> None:
        super().__init__()
        self.config = config
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config, vocab_size)
        # CLIP parameterizes the positive scale in log space. Initial scale:
        # exp(log(1/0.07)) ~= 14.29.
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32)
        )
        self.apply(self._initialize_module)

        # `apply` initializes Linear/Conv/LayerNorm only, so the specialized
        # class, position, token, and logit-scale initialization above remains.

    @staticmethod
    def _initialize_module(module: nn.Module) -> None:
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

    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        return self.image_encoder(images)

    def encode_text(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.text_encoder(token_ids, attention_mask)

    def similarity_logits(
        self,
        image_features: torch.Tensor,
        text_features: torch.Tensor,
    ) -> torch.Tensor:
        scale = self.logit_scale.exp().clamp(max=self.MAX_LOGIT_SCALE)
        return scale * image_features @ text_features.transpose(0, 1)

    def forward(
        self,
        images: torch.Tensor,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image_features = self.encode_image(images)
        text_features = self.encode_text(token_ids, attention_mask)
        logits_per_image = self.similarity_logits(
            image_features,
            text_features,
        )
        return logits_per_image, logits_per_image.transpose(0, 1)


def multi_positive_contrastive_loss(
    logits_per_image: torch.Tensor,
    labels: torch.Tensor,
) -> torch.Tensor:
    """Symmetric CLIP loss where all examples of the same digit are positive.

    For image ``i``, ``positive[i, j]`` is true whenever image ``i`` and caption
    ``j`` describe the same digit. We average log-probability over all positive
    captions for the image, then repeat in the text-to-image direction.
    """
    if logits_per_image.ndim != 2:
        raise ValueError("logits_per_image must be a rank-2 matrix")
    if logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ValueError("training requires equal image and text batch sizes")
    if labels.shape != (logits_per_image.shape[0],):
        raise ValueError("labels must contain one class ID per batch item")

    positive_mask = labels[:, None].eq(labels[None, :])
    positive_weights = positive_mask.float()
    positive_counts = positive_weights.sum(dim=1).clamp_min(1.0)

    image_log_probabilities = F.log_softmax(logits_per_image, dim=1)
    image_to_text_loss = -(
        (image_log_probabilities * positive_weights).sum(dim=1)
        / positive_counts
    ).mean()

    text_log_probabilities = F.log_softmax(
        logits_per_image.transpose(0, 1),
        dim=1,
    )
    text_to_image_loss = -(
        (text_log_probabilities * positive_weights.transpose(0, 1)).sum(dim=1)
        / positive_counts
    ).mean()
    return 0.5 * (image_to_text_loss + text_to_image_loss)


class MNISTCaptionDataset(Dataset):
    """Adapt one local Hugging Face MNIST split to CLIP training records."""

    def __init__(
        self,
        dataset: Any,
        tokenizer: ByteTokenizer,
        image_size: tuple[int, int],
    ) -> None:
        self.dataset = dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
            ]
        )
        self.encoded_prompts = [tokenizer.encode(text) for text in CLASS_PROMPTS]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataset[index]
        label = int(row["label"])
        token_ids, attention_mask = self.encoded_prompts[label]
        return {
            "image": self.transform(row["image"]),
            "token_ids": token_ids.clone(),
            "attention_mask": attention_mask.clone(),
            "label": torch.tensor(label, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train an educational multi-positive CLIP model on MNIST."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=SCRIPT_DIR / "checkpoints" / "code-2-clip-fixed-best.pt",
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-size", type=int, default=5_000)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image-size", type=int, default=28)
    parser.add_argument("--patch-size", type=int, default=7)
    parser.add_argument("--image-width", type=int, default=96)
    parser.add_argument("--image-layers", type=int, default=4)
    parser.add_argument("--image-heads", type=int, default=4)
    parser.add_argument("--text-width", type=int, default=96)
    parser.add_argument("--text-layers", type=int, default=2)
    parser.add_argument("--text-heads", type=int, default=4)
    parser.add_argument("--max-text-length", type=int, default=32)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use BF16/FP16 mixed precision on CUDA; enabled by default.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_integer_names = (
        "epochs",
        "batch_size",
        "validation_size",
        "image_size",
        "patch_size",
        "image_width",
        "image_layers",
        "image_heads",
        "text_width",
        "text_layers",
        "text_heads",
        "max_text_length",
        "embedding_dim",
        "mlp_ratio",
    )
    for name in positive_integer_names:
        if getattr(args, name) < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    if args.validation_size >= 60_000:
        raise ValueError("validation-size must be smaller than 60,000")
    if args.num_workers < 0:
        raise ValueError("num-workers cannot be negative")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive; weight-decay non-negative")
    if args.max_grad_norm <= 0:
        raise ValueError("max-grad-norm must be positive")
    if args.image_size % args.patch_size:
        raise ValueError("image-size must be divisible by patch-size")
    if args.image_width % args.image_heads:
        raise ValueError("image-width must be divisible by image-heads")
    if args.text_width % args.text_heads:
        raise ValueError("text-width must be divisible by text-heads")
    if args.max_text_length < 3:
        raise ValueError("max-text-length must be at least 3")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("dropout must be in [0, 1)")

    train_file = args.data_dir / "train-00000-of-00001.parquet"
    test_file = args.data_dir / "test-00000-of-00001.parquet"
    if not train_file.is_file() or not test_file.is_file():
        raise FileNotFoundError(
            "Expected local MNIST parquet files at "
            f"{train_file} and {test_file}"
        )


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


def load_local_mnist(data_dir: Path) -> Any:
    """Load parquet files directly, avoiding network or dataset-script code."""
    return load_dataset(
        "parquet",
        data_files={
            "train": str(data_dir / "train-00000-of-00001.parquet"),
            "test": str(data_dir / "test-00000-of-00001.parquet"),
        },
    )


def create_dataloaders(
    args: argparse.Namespace,
    config: CLIPConfig,
    tokenizer: ByteTokenizer,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset = load_local_mnist(args.data_dir)
    complete_training_set = MNISTCaptionDataset(
        dataset["train"],
        tokenizer,
        config.image_size,
    )
    test_set = MNISTCaptionDataset(
        dataset["test"],
        tokenizer,
        config.image_size,
    )
    training_size = len(complete_training_set) - args.validation_size
    training_set, validation_set = random_split(
        complete_training_set,
        lengths=(training_size, args.validation_size),
        generator=torch.Generator().manual_seed(args.seed),
    )

    common = {
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "persistent_workers": args.num_workers > 0,
    }
    training_loader = DataLoader(
        training_set,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
        **common,
    )
    validation_loader = DataLoader(validation_set, shuffle=False, **common)
    test_loader = DataLoader(test_set, shuffle=False, **common)
    return training_loader, validation_loader, test_loader


def encode_class_prompts(
    tokenizer: ByteTokenizer,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = [tokenizer.encode(prompt) for prompt in CLASS_PROMPTS]
    token_ids = torch.stack([item[0] for item in encoded]).to(device)
    attention_masks = torch.stack([item[1] for item in encoded]).to(device)
    return token_ids, attention_masks


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


def create_optimizer(
    model: CLIPModel,
    learning_rate: float,
    weight_decay: float,
) -> AdamW:
    """Apply weight decay to matrix weights, not scales, norms, or biases.

    In particular, decaying ``logit_scale`` would directly interfere with the
    learned contrastive temperature. LayerNorm scales and bias vectors are also
    conventionally excluded from AdamW decay.
    """
    decay_parameters: list[nn.Parameter] = []
    no_decay_parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name:
            no_decay_parameters.append(parameter)
        else:
            decay_parameters.append(parameter)

    return AdamW(
        [
            {"params": decay_parameters, "weight_decay": weight_decay},
            {"params": no_decay_parameters, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def batch_retrieval_accuracy(
    logits_per_image: torch.Tensor,
    labels: torch.Tensor,
) -> float:
    """Measure whether each image retrieves text belonging to its digit."""
    retrieved_text_indices = logits_per_image.argmax(dim=1)
    predicted_labels = labels[retrieved_text_indices]
    return float((predicted_labels == labels).float().mean())


def train_one_epoch(
    model: CLIPModel,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    accuracy_sum = 0.0
    examples = 0

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits_per_image, _ = model(images, token_ids, attention_mask)
            loss = multi_positive_contrastive_loss(logits_per_image, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        # CLIP implementations conventionally prevent the scale from exceeding
        # 100. Clamping the parameter avoids exp(logit_scale) overflow.
        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(model.MAX_LOGIT_SCALE))

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        accuracy_sum += batch_retrieval_accuracy(
            logits_per_image.detach(),
            labels,
        ) * batch_size
        examples += batch_size

    return loss_sum / examples, accuracy_sum / examples


@torch.inference_mode()
def evaluate(
    model: CLIPModel,
    loader: DataLoader,
    class_token_ids: torch.Tensor,
    class_attention_masks: torch.Tensor,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, float]:
    """Return contrastive loss and ten-prompt classification accuracy."""
    model.eval()
    loss_sum = 0.0
    correct = 0
    examples = 0

    with torch.autocast(
        device_type=device.type,
        dtype=amp_dtype,
        enabled=amp_enabled,
    ):
        class_text_features = model.encode_text(
            class_token_ids,
            class_attention_masks,
        )

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            image_features = model.encode_image(images)
            paired_text_features = model.encode_text(token_ids, attention_mask)
            paired_logits = model.similarity_logits(
                image_features,
                paired_text_features,
            )
            loss = multi_positive_contrastive_loss(paired_logits, labels)
            class_logits = model.similarity_logits(
                image_features,
                class_text_features,
            )

        predictions = class_logits.argmax(dim=1)
        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (predictions == labels).sum().item()
        examples += batch_size

    return loss_sum / examples, correct / examples


def save_checkpoint(
    path: Path,
    model: CLIPModel,
    config: CLIPConfig,
    epoch: int,
    validation_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "tokenizer": {
            "type": "utf8-byte",
            "vocab_size": ByteTokenizer.VOCAB_SIZE,
            "pad_id": ByteTokenizer.PAD_ID,
            "bos_id": ByteTokenizer.BOS_ID,
            "eos_id": ByteTokenizer.EOS_ID,
            "max_length": config.max_text_length,
        },
        "class_prompts": list(CLASS_PROMPTS),
        "epoch": epoch,
        "validation_accuracy": validation_accuracy,
    }
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

    config = CLIPConfig(
        image_size=pair(args.image_size),
        patch_size=pair(args.patch_size),
        image_width=args.image_width,
        image_layers=args.image_layers,
        image_heads=args.image_heads,
        text_width=args.text_width,
        text_layers=args.text_layers,
        text_heads=args.text_heads,
        max_text_length=args.max_text_length,
        embedding_dim=args.embedding_dim,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
    )
    tokenizer = ByteTokenizer(config.max_text_length)
    training_loader, validation_loader, test_loader = create_dataloaders(
        args,
        config,
        tokenizer,
        device,
    )
    class_token_ids, class_attention_masks = encode_class_prompts(
        tokenizer,
        device,
    )

    model = CLIPModel(config, ByteTokenizer.VOCAB_SIZE).to(device)
    optimizer = create_optimizer(
        model=model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled, amp_dtype = amp_settings(device, args.amp)
    fp16_scaling = amp_enabled and amp_dtype == torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=fp16_scaling)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("=" * 76)
    print("MNIST multi-positive CLIP training")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {amp_enabled} ({amp_dtype if amp_enabled else 'disabled'})")
    print(f"Parameters: {parameter_count:,}")
    print(f"Image patch tokens: {model.image_encoder.num_patches}")
    print(f"Shared embedding dimension: {config.embedding_dim}")
    print(f"Class prompts: {len(CLASS_PROMPTS)}")
    print(
        f"Training/validation/test: {len(training_loader.dataset):,} / "
        f"{len(validation_loader.dataset):,} / {len(test_loader.dataset):,}"
    )
    print(f"Checkpoint: {args.output}")
    print("=" * 76)

    best_validation_accuracy = -1.0
    training_started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss, train_retrieval_accuracy = train_one_epoch(
            model=model,
            loader=training_loader,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
            max_grad_norm=args.max_grad_norm,
        )
        validation_loss, validation_accuracy = evaluate(
            model=model,
            loader=validation_loader,
            class_token_ids=class_token_ids,
            class_attention_masks=class_attention_masks,
            device=device,
            amp_enabled=amp_enabled,
            amp_dtype=amp_dtype,
        )
        scheduler.step()

        improved = validation_accuracy > best_validation_accuracy
        if improved:
            best_validation_accuracy = validation_accuracy
            save_checkpoint(
                args.output,
                model,
                config,
                epoch,
                validation_accuracy,
            )

        print(
            f"epoch {epoch:02d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"batch retrieval {train_retrieval_accuracy:.2%} | "
            f"val loss {validation_loss:.4f} | "
            f"val class acc {validation_accuracy:.2%} | "
            f"scale {model.logit_scale.exp().item():.2f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{time.time() - epoch_started:.1f}s"
            + (" | saved" if improved else "")
        )

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_loss, test_accuracy = evaluate(
        model=model,
        loader=test_loader,
        class_token_ids=class_token_ids,
        class_attention_masks=class_attention_masks,
        device=device,
        amp_enabled=amp_enabled,
        amp_dtype=amp_dtype,
    )
    summary = {
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_accuracy": float(checkpoint["validation_accuracy"]),
        "test_contrastive_loss": test_loss,
        "test_classification_accuracy": test_accuracy,
        "logit_scale": model.logit_scale.detach().exp().cpu().item(),
        "training_seconds": time.time() - training_started,
        "checkpoint": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("=" * 76)
    print(f"Best validation epoch: {summary['best_epoch']}")
    print(f"Best validation accuracy: {summary['best_validation_accuracy']:.2%}")
    print(f"Test contrastive loss: {test_loss:.4f}")
    print(f"Test class-prompt accuracy: {test_accuracy:.2%}")
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved summary: {summary_path}")
    print("=" * 76)


if __name__ == "__main__":
    main()
