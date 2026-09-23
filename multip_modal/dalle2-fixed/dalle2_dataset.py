"""Dataset adapters for the educational DALL-E 2 pipeline.

The default FashionMNIST mode preserves the original demonstration.  Passing
``--dataset flickr8k`` to a training script reads only local parquet files from
``datasets/flickr8k/data``; nothing is downloaded from the Hugging Face Hub.

Flickr8k stores one image and five captions in each row.  During training we
choose one of those captions at random, so an image occurs only once per epoch
and CLIP's diagonal in-batch contrastive targets remain valid.  Validation and
test use the first non-empty caption deterministically.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST
from torchvision.transforms import InterpolationMode

from data.FMNISTConfig import FMNISTConfig, configure_dataset
from data.data_utils import tokenizer


CAPTION_COLUMNS = tuple(f"caption_{index}" for index in range(5))
SUPPORTED_DATASETS = ("fashion_mnist", "flickr8k")


def add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--dataset",
        choices=SUPPORTED_DATASETS,
        default="fashion_mnist",
        help="Dataset to train on (default: fashion_mnist).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the selected dataset directory.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device such as cuda, cuda:0, or cpu.",
    )
    parser.add_argument(
        "--using-pre-CLIP",
        "--using-pre-clip",
        dest="using_pretrained_clip",
        action="store_true",
        help=(
            "Use a frozen local pretrained CLIP and skip CLIP training. "
            "This creates separate *_preclip.pt checkpoints."
        ),
    )
    parser.add_argument(
        "--pretrained-clip-path",
        type=Path,
        default=Path("~/scratch/llms_model/clip-vit-base-patch32"),
        help="Local Hugging Face CLIP directory (no network download).",
    )
    parser.add_argument(
        "--large-UNet",
        "--large-unet",
        dest="large_unet",
        action="store_true",
        help=(
            "Use the 64/128/256/512-channel decoder and a separate "
            "*_largeunet.pt checkpoint."
        ),
    )


def config_from_args(args: argparse.Namespace) -> FMNISTConfig:
    config = FMNISTConfig()
    config.using_pretrained_clip = args.using_pretrained_clip
    config.pretrained_clip_path = str(args.pretrained_clip_path.expanduser())
    config.large_unet = args.large_unet
    config = configure_dataset(config, args.dataset, args.data_dir)
    if args.device is not None:
        config.device = torch.device(args.device)
    if config.device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but PyTorch cannot see a GPU")
    if config.using_pretrained_clip:
        model_path = Path(config.pretrained_clip_path)
        if not model_path.is_dir():
            raise FileNotFoundError(
                f"Pretrained CLIP directory does not exist: {model_path}"
            )
    return config


def _image_transform(config, train: bool, augment_data: bool) -> T.Compose:
    operations: list[Any] = []

    if config.dataset == "flickr8k" and train and augment_data:
        # Flickr8k source images are much larger than 64x64. RandomCrop(64)
        # before resizing would retain only a tiny, frequently irrelevant
        # fragment and destroy the image-caption correspondence. Instead,
        # sample a large portion (75%-100%) of the source image and resize it.
        operations.append(
            T.RandomResizedCrop(
                config.img_size,
                scale=(0.75, 1.0),
                ratio=(0.75, 4.0 / 3.0),
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        if config.prob_hflip > 0:
            operations.append(T.RandomHorizontalFlip(config.prob_hflip))
    else:
        # Validation is deterministic. FashionMNIST is first resized to 32x32
        # and only then receives its conventional padded random crop.
        operations.append(
            T.Resize(
                config.img_size,
                interpolation=InterpolationMode.BICUBIC,
                antialias=True,
            )
        )
        if train and augment_data:
            if config.prob_hflip > 0:
                operations.append(T.RandomHorizontalFlip(config.prob_hflip))
            if config.crop_padding > 0:
                operations.append(
                    T.RandomCrop(config.img_size, padding=config.crop_padding)
                )

    operations.extend(
        [
            T.ToTensor(),
            T.Normalize(config.train_mean, config.train_std),
        ]
    )
    return T.Compose(operations)


class FashionMNISTPairs(Dataset):
    """FashionMNIST image/class-caption pairs used by the original demo."""

    captions = {
        0: "An image of a t-shirt/top",
        1: "An image of trousers",
        2: "An image of a pullover",
        3: "An image of a dress",
        4: "An image of a coat",
        5: "An image of a sandal",
        6: "An image of a shirt",
        7: "An image of a sneaker",
        8: "An image of a bag",
        9: "An image of an ankle boot",
    }

    def __init__(self, config, train: bool, augment_data: bool) -> None:
        self.text_seq_length = config.text_seq_length
        self.transform = _image_transform(config, train, augment_data)
        self.dataset = FashionMNIST(
            root=config.data_location,
            train=train,
            download=train,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[index]
        caption, mask = tokenizer(
            self.captions[int(label)],
            text_seq_length=self.text_seq_length,
        )
        return {
            "image": self.transform(image),
            "caption": caption,
            "mask": mask,
        }

    def sample_texts(self) -> list[str]:
        return list(self.captions.values())


class Flickr8kPairs(Dataset):
    """Local Flickr8k parquet image-caption pairs.

    One row is one unique image. Training randomly selects one of its five
    captions on every access; evaluation always selects the first valid one.
    """

    def __init__(
        self,
        config,
        split: str,
        augment_data: bool = False,
    ) -> None:
        from datasets import load_dataset

        data_dir = Path(config.data_location).expanduser().resolve()
        files = sorted(data_dir.glob(f"{split}-*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No Flickr8k {split!r} parquet files found in {data_dir}"
            )
        self.rows = load_dataset(
            "parquet",
            data_files={split: [str(path) for path in files]},
            split=split,
        )
        required = {"image", *CAPTION_COLUMNS}
        missing = required.difference(self.rows.column_names)
        if missing:
            raise ValueError(f"Flickr8k columns are missing: {sorted(missing)}")

        self.training = split == "train"
        self.text_seq_length = config.text_seq_length
        self.transform = _image_transform(
            config,
            train=self.training,
            augment_data=augment_data,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _valid_captions(row: dict[str, Any]) -> list[str]:
        captions = [
            str(row[column]).strip()
            for column in CAPTION_COLUMNS
            if row[column] is not None and str(row[column]).strip()
        ]
        if not captions:
            raise ValueError("A Flickr8k row has no non-empty caption")
        return captions

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.rows[int(index)]
        captions = self._valid_captions(row)
        text = random.choice(captions) if self.training else captions[0]
        caption, mask = tokenizer(text, text_seq_length=self.text_seq_length)
        return {
            "image": self.transform(row["image"].convert("RGB")),
            "caption": caption,
            "mask": mask,
        }

    def sample_texts(self, count: int = 10) -> list[str]:
        return [
            self._valid_captions(self.rows[index])[0]
            for index in range(min(count, len(self.rows)))
        ]


def get_train_set(config, augment_data: bool = False):
    if config.dataset == "fashion_mnist":
        dataset = FashionMNISTPairs(config, train=True, augment_data=augment_data)
    elif config.dataset == "flickr8k":
        dataset = Flickr8kPairs(
            config,
            split="train",
            augment_data=augment_data,
        )
    else:
        raise ValueError(f"Unsupported dataset: {config.dataset}")
    return dataset, config.train_mean, config.train_std


def get_test_set(config, mean=None, std=None):
    # mean/std remain accepted for compatibility with the original call sites;
    # normalization is now a stable part of the dataset configuration.
    del mean, std
    if config.dataset == "fashion_mnist":
        return FashionMNISTPairs(config, train=False, augment_data=False)
    if config.dataset == "flickr8k":
        # The validation split is used while fitting model stages. The held-out
        # test split remains untouched for final evaluation experiments.
        return Flickr8kPairs(config, split="validation", augment_data=False)
    raise ValueError(f"Unsupported dataset: {config.dataset}")
