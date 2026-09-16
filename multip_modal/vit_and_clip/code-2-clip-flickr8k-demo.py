"""Train a compact CLIP model from scratch on the local Flickr8k parquet data.

Dataset expected by this script
===============================
The server dataset is the parquet release of ``jxie/flickr8k``:

    datasets/flickr8k/data/
        train-00000-of-00002-....parquet
        train-00001-of-00002-....parquet
        validation-00000-of-00001-....parquet
        test-00000-of-00001-....parquet

Each row represents one image and contains these columns:

    image, caption_0, caption_1, caption_2, caption_3, caption_4

The predefined splits contain 6,000 training images, 1,000 validation images,
and 1,000 test images. The script loads only local parquet files; it does not
download the dataset from the Hugging Face Hub.

CLIP theory
===========
CLIP is a dual-encoder model. The image encoder and text encoder do not attend
to one another. Instead, both map their inputs into the same embedding space:

    image embedding: v_i = normalize(f_image(image_i))
    text embedding:  u_j = normalize(f_text(caption_j))

Because the embeddings have unit length, ``v_i @ u_j`` is cosine similarity.
For a batch containing B paired images and captions, CLIP constructs a B by B
similarity matrix:

    logits[i, j] = exp(logit_scale) * (v_i @ u_j)

The diagonal contains the observed image-caption pairs. All other batch items
are in-batch negatives. CLIP applies cross-entropy in both directions:

    image-to-text loss: image i must retrieve caption i
    text-to-image loss: caption i must retrieve image i
    CLIP loss:          (image-to-text + text-to-image) / 2

The learnable ``logit_scale`` is the inverse temperature in log space. A large
scale makes the probability distribution sharper. It is capped at 100 to avoid
unstable exponential growth.

How the five captions are handled
=================================
Every training image appears once per epoch and one of its five captions is
sampled. This guarantees that an image is not duplicated within an epoch's
sampling stream, so the standard diagonal CLIP objective is valid. Across
epochs the model sees different descriptions of the same image.

During validation and testing, all five captions are retained. Retrieval is
correct when:

* image-to-text: any of an image's five captions occurs in the top K;
* text-to-image: the caption's source image occurs in the top K.

The script reports Recall@1, Recall@5, and Recall@10 in both directions. It
selects the best checkpoint using the mean of these six validation recalls and
evaluates the test split only after training is complete.

Architecture used in this demo
==============================
* Image encoder: a small Vision Transformer over 16x16 image patches.
* Text encoder: a causal Transformer over a vocabulary learned only from the
  training captions.
* Projection heads: linear maps into one shared embedding dimension.
* Training: AdamW, symmetric CLIP loss, cosine learning-rate decay, gradient
  clipping, and BF16 mixed precision on an A100.

This is an educational from-scratch experiment. Flickr8k is far too small to
reproduce the broad open-vocabulary ability of CLIP models trained on hundreds
of millions of pairs. It is, however, large enough to demonstrate genuine
image-caption contrastive learning and standard retrieval evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable

from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_dataset
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "datasets" / "flickr8k" / "data"
CAPTION_COLUMNS = tuple(f"caption_{index}" for index in range(5))

# CLIP's commonly used RGB normalization statistics.
CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)


@dataclass(frozen=True)
class CLIPConfig:
    """Model hyperparameters saved in the checkpoint."""

    image_size: int = 224
    patch_size: int = 16
    image_width: int = 256
    image_layers: int = 6
    image_heads: int = 8
    text_width: int = 256
    text_layers: int = 4
    text_heads: int = 8
    max_text_length: int = 40
    embedding_dim: int = 256
    mlp_ratio: int = 4
    dropout: float = 0.1


class WordTokenizer:
    """Small offline word/punctuation tokenizer built from training captions.

    Unlike the byte tokenizer in the MNIST demo, a
    word vocabulary represents
    natural Flickr8k captions compactly. Only training captions build the
    vocabulary, preventing validation/test vocabulary leakage.
    """

    PAD_TOKEN = "<pad>"
    BOS_TOKEN = "<bos>"
    EOS_TOKEN = "<eos>"
    UNK_TOKEN = "<unk>"
    SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN)

    def __init__(self, vocabulary: list[str], max_length: int) -> None:
        if vocabulary[:4] != list(self.SPECIAL_TOKENS):
            raise ValueError("vocabulary must begin with PAD/BOS/EOS/UNK")
        if max_length < 3:
            raise ValueError("max_length must be at least 3")
        self.id_to_token = vocabulary
        self.token_to_id = {
            token: index for index, token in enumerate(vocabulary)
        }
        self.max_length = max_length
        self.pad_id = self.token_to_id[self.PAD_TOKEN]
        self.bos_id = self.token_to_id[self.BOS_TOKEN]
        self.eos_id = self.token_to_id[self.EOS_TOKEN]
        self.unk_id = self.token_to_id[self.UNK_TOKEN]

    @staticmethod
    def split_text(text: str) -> list[str]:
        # Keep contractions together and keep punctuation as separate tokens.
        return re.findall(
            r"[a-z0-9]+(?:'[a-z0-9]+)?|[^\w\s]",
            text.lower(),
            flags=re.UNICODE,
        )

    @classmethod
    def build(
        cls,
        captions: Iterable[str],
        max_length: int,
        max_vocabulary_size: int,
        minimum_frequency: int,
    ) -> "WordTokenizer":
        counts: Counter[str] = Counter()
        for caption in captions:
            counts.update(cls.split_text(caption))

        # Frequency first and alphabetical tie-breaking make construction fully
        # deterministic across machines.
        candidates = [
            (token, frequency)
            for token, frequency in counts.items()
            if frequency >= minimum_frequency
        ]
        candidates.sort(key=lambda item: (-item[1], item[0]))
        available_slots = max_vocabulary_size - len(cls.SPECIAL_TOKENS)
        if available_slots < 1:
            raise ValueError("max_vocabulary_size is too small")
        vocabulary = list(cls.SPECIAL_TOKENS)
        vocabulary.extend(token for token, _ in candidates[:available_slots])
        return cls(vocabulary, max_length)

    def encode(self, text: str) -> tuple[torch.Tensor, torch.Tensor]:
        content_ids = [
            self.token_to_id.get(token, self.unk_id)
            for token in self.split_text(text)
        ][: self.max_length - 2]
        token_ids = [self.bos_id, *content_ids, self.eos_id]
        valid_length = len(token_ids)
        token_ids.extend([self.pad_id] * (self.max_length - valid_length))
        attention_mask = [1] * valid_length + [0] * (
            self.max_length - valid_length
        )
        return (
            torch.tensor(token_ids, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.bool),
        )

    @property
    def vocabulary_size(self) -> int:
        return len(self.id_to_token)


class MultiHeadSelfAttention(nn.Module):
    """Self-attention shared by the vision and text Transformer blocks."""

    def __init__(self, width: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if width % num_heads:
            raise ValueError("Transformer width must be divisible by heads")
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

        if attention_mask is not None or causal:
            allowed = torch.ones(
                (batch_size, 1, sequence_length, sequence_length),
                dtype=torch.bool,
                device=tokens.device,
            )
            if attention_mask is not None:
                if attention_mask.shape != (batch_size, sequence_length):
                    raise ValueError("invalid attention_mask shape")
                # Only valid text tokens may act as keys/values.
                allowed = allowed & attention_mask[:, None, None, :]
            if causal:
                causal_mask = torch.ones(
                    (sequence_length, sequence_length),
                    dtype=torch.bool,
                    device=tokens.device,
                ).tril()
                allowed = allowed & causal_mask[None, None, :, :]
            scores = scores.masked_fill(
                ~allowed,
                torch.finfo(scores.dtype).min,
            )

        attention_weights = scores.softmax(dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        attended = attention_weights @ values
        attended = attended.transpose(1, 2).contiguous().reshape(
            batch_size,
            sequence_length,
            width,
        )
        return self.output_dropout(self.output_projection(attended))


class TransformerBlock(nn.Module):
    """Pre-normalized Transformer block with two residual connections."""

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
            attention_mask,
            causal,
        )
        return tokens + self.mlp(self.norm2(tokens))


class ImageEncoder(nn.Module):
    """Vision Transformer that returns a normalized shared-space vector."""

    def __init__(self, config: CLIPConfig) -> None:
        super().__init__()
        if config.image_size % config.patch_size:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = config.image_size
        patches_per_side = config.image_size // config.patch_size
        self.num_patches = patches_per_side**2
        self.patch_projection = nn.Conv2d(
            3,
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
                TransformerBlock(
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
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [batch, 3, height, width]")
        if tuple(images.shape[-2:]) != (self.image_size, self.image_size):
            raise ValueError("images do not match configured image_size")
        tokens = self.patch_projection(images)
        tokens = tokens.flatten(start_dim=2).transpose(1, 2)
        class_tokens = self.class_token.expand(images.shape[0], -1, -1)
        tokens = torch.cat((class_tokens, tokens), dim=1)
        tokens = self.input_dropout(tokens + self.position_embedding)
        for block in self.blocks:
            # Image attention is bidirectional because all patches are known.
            tokens = block(tokens)
        class_state = self.final_norm(tokens)[:, 0]
        return F.normalize(self.projection(class_state), dim=-1, eps=1e-6)


class TextEncoder(nn.Module):
    """Causal text Transformer using the EOS state as caption representation."""

    def __init__(
        self,
        config: CLIPConfig,
        vocabulary_size: int,
        padding_id: int,
    ) -> None:
        super().__init__()
        self.max_text_length = config.max_text_length
        self.token_embedding = nn.Embedding(
            vocabulary_size,
            config.text_width,
            padding_idx=padding_id,
        )
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.max_text_length, config.text_width)
        )
        self.input_dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
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
            self.token_embedding.weight[padding_id].zero_()
        nn.init.trunc_normal_(self.position_embedding, std=0.01)

    def forward(
        self,
        token_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        if token_ids.shape != attention_mask.shape or token_ids.ndim != 2:
            raise ValueError("token IDs and mask must have equal [B, L] shape")
        sequence_length = token_ids.shape[1]
        if sequence_length > self.max_text_length:
            raise ValueError("caption exceeds max_text_length")

        tokens = self.token_embedding(token_ids)
        tokens = tokens + self.position_embedding[:, :sequence_length]
        tokens = self.input_dropout(tokens)
        for block in self.blocks:
            tokens = block(tokens, attention_mask, causal=True)
        tokens = self.final_norm(tokens)

        # Valid sequence = [BOS, caption tokens, EOS], followed by PAD.
        eos_positions = attention_mask.sum(dim=1, dtype=torch.long) - 1
        batch_indices = torch.arange(token_ids.shape[0], device=token_ids.device)
        eos_states = tokens[batch_indices, eos_positions]
        return F.normalize(self.projection(eos_states), dim=-1, eps=1e-6)


class CLIPModel(nn.Module):
    """Image/text dual encoder and learned contrastive temperature."""

    MAX_LOGIT_SCALE = 100.0

    def __init__(
        self,
        config: CLIPConfig,
        vocabulary_size: int,
        padding_id: int,
    ) -> None:
        super().__init__()
        self.image_encoder = ImageEncoder(config)
        self.text_encoder = TextEncoder(config, vocabulary_size, padding_id)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / 0.07), dtype=torch.float32)
        )
        self.apply(self._initialize_module)

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


def symmetric_clip_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
) -> torch.Tensor:
    """Standard diagonal CLIP loss for one unique image per batch item."""
    if logits_per_image.shape != logits_per_text.transpose(0, 1).shape:
        raise ValueError("image and text logits are not transposes")
    if logits_per_image.shape[0] != logits_per_image.shape[1]:
        raise ValueError("CLIP training requires equal image/text batch sizes")
    targets = torch.arange(
        logits_per_image.shape[0],
        device=logits_per_image.device,
    )
    image_to_text = F.cross_entropy(logits_per_image, targets)
    text_to_image = F.cross_entropy(logits_per_text, targets)
    return 0.5 * (image_to_text + text_to_image)


def extract_captions(row: dict[str, Any]) -> tuple[str, ...]:
    captions = tuple(
        str(row[column]).strip()
        for column in CAPTION_COLUMNS
        if row[column] is not None and str(row[column]).strip()
    )
    if not captions:
        raise ValueError("dataset row has no non-empty captions")
    return captions


class Flickr8kTrainingDataset(Dataset):
    """Return each image once per epoch with one randomly selected caption."""

    def __init__(
        self,
        dataset: HFDataset,
        tokenizer: WordTokenizer,
        transform: Any,
    ) -> None:
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.dataset[index]
        captions = extract_captions(row)
        # DataLoader gives every worker its own seeded torch RNG. Sampling with
        # torch therefore changes captions across epochs while remaining
        # reproducible when the experiment seed is fixed.
        caption_index = int(torch.randint(len(captions), (1,)).item())
        token_ids, attention_mask = self.tokenizer.encode(
            captions[caption_index]
        )
        image = self.transform(row["image"].convert("RGB"))
        return {
            "image": image,
            "token_ids": token_ids,
            "attention_mask": attention_mask,
        }


class Flickr8kImageDataset(Dataset):
    """One transformed image per evaluation row, in stable row order."""

    def __init__(self, dataset: HFDataset, transform: Any) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image = self.dataset[index]["image"].convert("RGB")
        return {
            "image": self.transform(image),
            "image_index": torch.tensor(index, dtype=torch.long),
        }


class Flickr8kCaptionDataset(Dataset):
    """Flatten five captions per image and retain each source image index."""

    def __init__(self, dataset: HFDataset, tokenizer: WordTokenizer) -> None:
        self.tokenizer = tokenizer
        self.items: list[tuple[int, str]] = []
        # Access caption columns directly so building this index never decodes
        # the parquet image column.
        caption_columns = {
            column: dataset[column] for column in CAPTION_COLUMNS
        }
        for image_index in range(len(dataset)):
            for column in CAPTION_COLUMNS:
                caption = caption_columns[column][image_index]
                if caption is not None and str(caption).strip():
                    self.items.append((image_index, str(caption).strip()))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image_index, caption = self.items[index]
        token_ids, attention_mask = self.tokenizer.encode(caption)
        return {
            "token_ids": token_ids,
            "attention_mask": attention_mask,
            "image_index": torch.tensor(image_index, dtype=torch.long),
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a compact CLIP model from scratch on Flickr8k."
    )
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            SCRIPT_DIR
            / "checkpoints"
            / "code-2-clip-flickr8k-demo-best.pt"
        ),
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eval-every", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--patch-size", type=int, default=16)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-layers", type=int, default=6)
    parser.add_argument("--image-heads", type=int, default=8)
    parser.add_argument("--text-width", type=int, default=256)
    parser.add_argument("--text-layers", type=int, default=4)
    parser.add_argument("--text-heads", type=int, default=8)
    parser.add_argument("--max-text-length", type=int, default=40)
    parser.add_argument("--embedding-dim", type=int, default=256)
    parser.add_argument("--mlp-ratio", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--max-vocabulary-size", type=int, default=12_000)
    parser.add_argument("--minimum-token-frequency", type=int, default=1)
    parser.add_argument(
        "--max-train-images",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 uses all training images.",
    )
    parser.add_argument(
        "--max-validation-images",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 uses all validation images.",
    )
    parser.add_argument(
        "--max-test-images",
        type=int,
        default=0,
        help="Optional smoke-test limit; 0 uses all test images.",
    )
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
    parser.add_argument(
        "--inspect-only",
        action="store_true",
        help="Validate data/tokenizer and print one record without loading a model.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_names = (
        "epochs",
        "batch_size",
        "eval_batch_size",
        "num_workers",
        "eval_every",
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
        "max_vocabulary_size",
        "minimum_token_frequency",
    )
    for name in positive_names:
        value = getattr(args, name)
        if name == "num_workers":
            if value < 0:
                raise ValueError("num-workers cannot be negative")
        elif value < 1:
            raise ValueError(f"{name.replace('_', '-')} must be positive")
    for name in (
        "max_train_images",
        "max_validation_images",
        "max_test_images",
    ):
        if getattr(args, name) < 0:
            raise ValueError(f"{name.replace('_', '-')} cannot be negative")
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
    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Flickr8k data directory missing: {args.data_dir}")


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


def find_parquet_files(data_dir: Path, prefix: str) -> list[str]:
    files = sorted(data_dir.glob(f"{prefix}-*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No {prefix}-*.parquet files found in {data_dir}"
        )
    return [str(path) for path in files]


def load_local_flickr8k(data_dir: Path) -> DatasetDict:
    """Load the exact server parquet layout with no Hub/network request."""
    dataset = load_dataset(
        "parquet",
        data_files={
            "train": find_parquet_files(data_dir, "train"),
            "validation": find_parquet_files(data_dir, "validation"),
            "test": find_parquet_files(data_dir, "test"),
        },
    )
    required_columns = {"image", *CAPTION_COLUMNS}
    for split_name in ("train", "validation", "test"):
        missing = required_columns.difference(dataset[split_name].column_names)
        if missing:
            raise ValueError(
                f"{split_name} split is missing columns: {sorted(missing)}"
            )
    return dataset


def training_caption_iterator(dataset: HFDataset) -> Iterable[str]:
    """Yield text columns without decoding the large image column."""
    for column in CAPTION_COLUMNS:
        for caption in dataset[column]:
            if caption is not None and str(caption).strip():
                yield str(caption).strip()


def maybe_limit(dataset: HFDataset, maximum: int) -> HFDataset:
    if maximum == 0 or maximum >= len(dataset):
        return dataset
    return dataset.select(range(maximum))


def create_transforms(image_size: int) -> tuple[Any, Any]:
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                image_size,
                scale=(0.7, 1.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ]
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(
                image_size + 32,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(CLIP_IMAGE_MEAN, CLIP_IMAGE_STD),
        ]
    )
    return train_transform, eval_transform


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


def create_training_loader(
    dataset: HFDataset,
    tokenizer: WordTokenizer,
    transform: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> DataLoader:
    wrapped = Flickr8kTrainingDataset(dataset, tokenizer, transform)
    return DataLoader(
        wrapped,
        shuffle=True,
        # CLIP only requires aligned pairs inside each batch, not a fixed
        # batch size. Keeping the final batch also makes small smoke tests
        # perform a real update when their subset is smaller than batch_size.
        drop_last=False,
        generator=torch.Generator().manual_seed(args.seed),
        **loader_options(args.batch_size, args.num_workers, device),
    )


def create_evaluation_loaders(
    dataset: HFDataset,
    tokenizer: WordTokenizer,
    transform: Any,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[DataLoader, DataLoader]:
    image_dataset = Flickr8kImageDataset(dataset, transform)
    caption_dataset = Flickr8kCaptionDataset(dataset, tokenizer)
    options = loader_options(args.eval_batch_size, args.num_workers, device)
    return (
        DataLoader(image_dataset, shuffle=False, **options),
        DataLoader(caption_dataset, shuffle=False, **options),
    )


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
    """Do not decay biases, normalization scales, or temperature."""
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if parameter.ndim < 2 or name.endswith(".bias") or "norm" in name:
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=learning_rate,
    )


def train_one_epoch(
    model: CLIPModel,
    loader: DataLoader,
    optimizer: AdamW,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
    max_grad_norm: float,
) -> float:
    model.train()
    loss_sum = 0.0
    examples = 0
    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            logits_per_image, logits_per_text = model(
                images,
                token_ids,
                attention_mask,
            )
            loss = symmetric_clip_loss(logits_per_image, logits_per_text)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        with torch.no_grad():
            model.logit_scale.clamp_(max=math.log(model.MAX_LOGIT_SCALE))

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        examples += batch_size
    return loss_sum / max(examples, 1)


@torch.inference_mode()
def encode_retrieval_features(
    model: CLIPModel,
    image_loader: DataLoader,
    caption_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Encode every image and caption, preserving their evaluation IDs."""
    model.eval()
    image_features: list[torch.Tensor] = []
    image_indices: list[torch.Tensor] = []
    for batch in image_loader:
        images = batch["image"].to(device, non_blocking=True)
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            features = model.encode_image(images)
        image_features.append(features.float().cpu())
        image_indices.append(batch["image_index"])

    text_features: list[torch.Tensor] = []
    text_image_indices: list[torch.Tensor] = []
    for batch in caption_loader:
        token_ids = batch["token_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(
            device,
            non_blocking=True,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=amp_dtype,
            enabled=amp_enabled,
        ):
            features = model.encode_text(token_ids, attention_mask)
        text_features.append(features.float().cpu())
        text_image_indices.append(batch["image_index"])

    images = torch.cat(image_features)
    ordered_image_indices = torch.cat(image_indices)
    expected = torch.arange(len(images), dtype=torch.long)
    if not torch.equal(ordered_image_indices, expected):
        raise RuntimeError("evaluation images are not in stable row order")
    return images, torch.cat(text_features), torch.cat(text_image_indices)


@torch.inference_mode()
def retrieval_metrics(
    model: CLIPModel,
    image_loader: DataLoader,
    caption_loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    amp_dtype: torch.dtype,
) -> dict[str, float]:
    image_features, text_features, caption_image_ids = (
        encode_retrieval_features(
            model,
            image_loader,
            caption_loader,
            device,
            amp_enabled,
            amp_dtype,
        )
    )
    # Flickr8k evaluation is small: 1,000 x 5,000 = five million similarities,
    # about 20 MB in FP32, so exact retrieval easily fits on an A100 80 GB.
    similarities = image_features.to(device) @ text_features.to(device).T
    caption_image_ids = caption_image_ids.to(device)
    image_ids = torch.arange(len(image_features), device=device)

    metrics: dict[str, float] = {}
    recall_values: list[float] = []
    for requested_k in (1, 5, 10):
        text_k = min(requested_k, similarities.shape[1])
        top_text_indices = similarities.topk(text_k, dim=1).indices
        top_text_image_ids = caption_image_ids[top_text_indices]
        image_to_text = (
            top_text_image_ids.eq(image_ids[:, None]).any(dim=1).float().mean()
        )

        image_k = min(requested_k, similarities.shape[0])
        top_image_indices = similarities.T.topk(image_k, dim=1).indices
        text_to_image = (
            top_image_indices.eq(caption_image_ids[:, None]).any(dim=1).float().mean()
        )
        metrics[f"image_to_text_R@{requested_k}"] = image_to_text.item()
        metrics[f"text_to_image_R@{requested_k}"] = text_to_image.item()
        recall_values.extend([image_to_text.item(), text_to_image.item()])

    metrics["mean_recall"] = float(np.mean(recall_values))
    return metrics


def save_checkpoint(
    path: Path,
    model: CLIPModel,
    config: CLIPConfig,
    tokenizer: WordTokenizer,
    epoch: int,
    validation_metrics: dict[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": asdict(config),
        "tokenizer": {
            "type": "lowercase-word-punctuation",
            "vocabulary": tokenizer.id_to_token,
            "max_length": tokenizer.max_length,
        },
        "epoch": epoch,
        "validation_metrics": validation_metrics,
    }
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(checkpoint, temporary_path)
    temporary_path.replace(path)


def print_metrics(prefix: str, metrics: dict[str, float]) -> None:
    print(
        f"{prefix} | "
        f"I2T R@1 {metrics['image_to_text_R@1']:.2%} "
        f"R@5 {metrics['image_to_text_R@5']:.2%} "
        f"R@10 {metrics['image_to_text_R@10']:.2%} | "
        f"T2I R@1 {metrics['text_to_image_R@1']:.2%} "
        f"R@5 {metrics['text_to_image_R@5']:.2%} "
        f"R@10 {metrics['text_to_image_R@10']:.2%} | "
        f"mean {metrics['mean_recall']:.2%}"
    )


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_everything(args.seed)

    dataset = load_local_flickr8k(args.data_dir)
    full_counts = {name: len(dataset[name]) for name in dataset}
    tokenizer = WordTokenizer.build(
        captions=training_caption_iterator(dataset["train"]),
        max_length=args.max_text_length,
        max_vocabulary_size=args.max_vocabulary_size,
        minimum_frequency=args.minimum_token_frequency,
    )

    print("Dataset columns:", dataset["train"].column_names)
    print("Full split sizes:", full_counts)
    print("Training-only vocabulary size:", tokenizer.vocabulary_size)
    sample_captions = extract_captions(dataset["train"][0])
    print("Example caption:", sample_captions[0])
    if args.inspect_only:
        ids, mask = tokenizer.encode(sample_captions[0])
        print("Example token IDs:", ids[mask].tolist())
        print("Inspection complete; model was not loaded.")
        return

    dataset["train"] = maybe_limit(dataset["train"], args.max_train_images)
    dataset["validation"] = maybe_limit(
        dataset["validation"],
        args.max_validation_images,
    )
    dataset["test"] = maybe_limit(dataset["test"], args.max_test_images)
    if any(
        value > 0
        for value in (
            args.max_train_images,
            args.max_validation_images,
            args.max_test_images,
        )
    ):
        print("WARNING: dataset limits are active; metrics are smoke-test only.")

    device = select_device(args.device)
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True

    config = CLIPConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
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
    train_transform, eval_transform = create_transforms(config.image_size)
    training_loader = create_training_loader(
        dataset["train"],
        tokenizer,
        train_transform,
        args,
        device,
    )
    validation_image_loader, validation_caption_loader = (
        create_evaluation_loaders(
            dataset["validation"],
            tokenizer,
            eval_transform,
            args,
            device,
        )
    )
    test_image_loader, test_caption_loader = create_evaluation_loaders(
        dataset["test"],
        tokenizer,
        eval_transform,
        args,
        device,
    )

    model = CLIPModel(
        config,
        vocabulary_size=tokenizer.vocabulary_size,
        padding_id=tokenizer.pad_id,
    ).to(device)
    optimizer = create_optimizer(
        model,
        args.learning_rate,
        args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled, amp_dtype = amp_settings(device, args.amp)
    scaler = torch.amp.GradScaler(
        device.type,
        enabled=amp_enabled and amp_dtype == torch.float16,
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("=" * 80)
    print("Flickr8k CLIP-from-scratch demo")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"AMP: {amp_enabled} ({amp_dtype if amp_enabled else 'disabled'})")
    print(f"Parameters: {parameter_count:,}")
    print(f"Patch tokens: {model.image_encoder.num_patches}")
    print(
        "Active split sizes: "
        f"{len(dataset['train']):,} / {len(dataset['validation']):,} / "
        f"{len(dataset['test']):,}"
    )
    print(f"Training pairs per epoch: {len(training_loader.dataset):,}")
    print(f"Validation captions: {len(validation_caption_loader.dataset):,}")
    print(f"Checkpoint: {args.output}")
    print("=" * 80)

    best_score = -1.0
    best_epoch = 0
    training_started = time.time()
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.time()
        train_loss = train_one_epoch(
            model,
            training_loader,
            optimizer,
            scaler,
            device,
            amp_enabled,
            amp_dtype,
            args.max_grad_norm,
        )
        scheduler.step()
        print(
            f"epoch {epoch:03d}/{args.epochs} | train loss {train_loss:.4f} | "
            f"scale {model.logit_scale.detach().exp().item():.2f} | "
            f"lr {scheduler.get_last_lr()[0]:.2e} | "
            f"{time.time() - epoch_started:.1f}s"
        )

        should_evaluate = epoch % args.eval_every == 0 or epoch == args.epochs
        if should_evaluate:
            validation_metrics = retrieval_metrics(
                model,
                validation_image_loader,
                validation_caption_loader,
                device,
                amp_enabled,
                amp_dtype,
            )
            print_metrics("validation", validation_metrics)
            if validation_metrics["mean_recall"] > best_score:
                best_score = validation_metrics["mean_recall"]
                best_epoch = epoch
                save_checkpoint(
                    args.output,
                    model,
                    config,
                    tokenizer,
                    epoch,
                    validation_metrics,
                )
                print(f"Saved new best checkpoint: {args.output}")

    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = retrieval_metrics(
        model,
        test_image_loader,
        test_caption_loader,
        device,
        amp_enabled,
        amp_dtype,
    )
    print_metrics("test", test_metrics)

    summary = {
        "best_epoch": best_epoch,
        "best_validation_metrics": checkpoint["validation_metrics"],
        "test_metrics": test_metrics,
        "full_split_sizes": full_counts,
        "active_split_sizes": {
            split_name: len(dataset[split_name])
            for split_name in ("train", "validation", "test")
        },
        "vocabulary_size": tokenizer.vocabulary_size,
        "parameter_count": parameter_count,
        "training_seconds": time.time() - training_started,
        "checkpoint": str(args.output.resolve()),
    }
    summary_path = args.output.with_suffix(".json")
    with summary_path.open("w", encoding="utf-8") as output_file:
        json.dump(summary, output_file, indent=2)

    print("=" * 80)
    print(f"Best validation epoch: {best_epoch}")
    print(f"Saved checkpoint: {args.output}")
    print(f"Saved summary: {summary_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()



"""
The implementation matches your local dataset:

  - Loads both training Parquet shards.
  - Uses image and caption_0 through caption_4.
  - Samples one of the five captions per image during training.
  - Uses all five captions for validation and testing.
  - Implements CLIP from scratch with a Vision Transformer and text Transformer.
  - Uses symmetric image-to-text/text-to-image contrastive loss.
  - Reports Recall@1, Recall@5, and Recall@10.
  - Uses BF16 automatically on your A100.
  - Saves the best checkpoint and a JSON result summary.

  The documented split is 6,000/1,000/1,000 images, consistent with the Flickr8k dataset card (https://huggingface.co/datasets/jxie/flickr8k).

  First run a small GPU smoke test:

  cd ~/scratch/dips_project/reinforcement_learning/multip_modal/vit_and_clip

  sbatch submit-code-2-clip-flickr8k-demo.sh \
      --epochs 1 \
      --batch-size 128 \
      --max-train-images 512 \
      --max-validation-images 100 \
      --max-test-images 100

  If successful, submit the complete training:

  sbatch submit-code-2-clip-flickr8k-demo.sh

  Monitor it with:

  squeue -u "$USER"
  tail -f result_out/flickr8k-clip-<JOB_ID>.out

  The best model will be saved at:

  multip_modal/vit_and_clip/checkpoints/code-2-clip-flickr8k-demo-best.pt
  
"""