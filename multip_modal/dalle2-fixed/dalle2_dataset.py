"""Dataset adapters for the educational DALL-E 2 pipeline.

Flickr30k is the default. ``--dataset flickr8k`` and ``--data fashionMNIST``
select one source, while ``--dataset all`` concatenates Flickr8k and Flickr30k.
The natural-image datasets share the same RGB 64x64 representation.

The Flickr parquet loader supports both numbered columns (``caption_0`` ...)
and the common list-valued ``caption``/``captions`` representation. During
training it randomly chooses one valid caption for each image; validation uses
the first caption deterministically.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
from typing import Any

import torch
from torch.utils.data import ConcatDataset, Dataset
import torchvision.transforms as T
from torchvision.datasets import FashionMNIST
from torchvision.transforms import InterpolationMode

from config import (
    Dalle2Config,
    FMNISTConfig,
    add_config_arguments,
    add_dataset_arguments,
    config_from_args,
    configure_dataset,
)
from data.data_utils import tokenizer


CAPTION_COLUMNS = tuple(f"caption_{index}" for index in range(5))
CAPTION_CONTAINER_COLUMNS = ("caption", "captions", "sentences")
MERGED_DATASET_VALIDATION_FRACTION = 0.05
MERGED_DATASET_SPLIT_SEED = 42
def _image_transform(
    config,
    train: bool,
    augment_data: bool,
    natural_image: bool,
) -> T.Compose:
    operations: list[Any] = []

    if natural_image and train and augment_data:
        # Flickr source images are much larger than 64x64. RandomCrop(64)
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
        self.img_channels = config.img_channels
        self.transform = _image_transform(
            config,
            train,
            augment_data,
            natural_image=False,
        )
        self.dataset = FashionMNIST(
            root=config.fashion_mnist_data_location,
            train=train,
            download=train,
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        image, label = self.dataset[index]
        if self.img_channels == 3:
            image = image.convert("RGB")
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


class FlickrPairs(Dataset):
    """Local Flickr8k/Flickr30k parquet image-caption pairs.

    One row is one unique image. Training randomly selects one of its five
    captions on every access; evaluation always selects the first valid one.
    """

    def __init__(
        self,
        config,
        dataset_name: str,
        data_location: str | Path,
        split: str,
        augment_data: bool = False,
    ) -> None:
        from datasets import load_dataset

        self.dataset_name = dataset_name
        data_dir = Path(data_location).expanduser().resolve()
        split_aliases = {
            "train": ("train",),
            "validation": ("validation", "val", "dev"),
            "test": ("test",),
        }[split]
        files = sorted(
            {
                path
                for alias in split_aliases
                for path in data_dir.glob(f"{alias}-*.parquet")
            }
        )
        must_filter_internal_split = not files
        if must_filter_internal_split:
            # Some Flickr30k parquet exports put every original split in files
            # named test-*.parquet and retain the real split in a row column.
            files = sorted(data_dir.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(
                f"No parquet files found for {dataset_name} in {data_dir}"
            )
        print(
            f"[Data:{dataset_name}] requested split={split!r} | "
            f"directory={data_dir} | parquet shards={len(files)}",
            flush=True,
        )
        print(
            f"[Data:{dataset_name}] first shard={files[0].name} | "
            f"last shard={files[-1].name}",
            flush=True,
        )
        self.rows = load_dataset(
            "parquet",
            data_files={"records": [str(path) for path in files]},
            split="records",
        )
        loaded_rows = len(self.rows)
        if "split" in self.rows.column_names:
            split_counts = dict(
                sorted(Counter(map(str, self.rows["split"])).items())
            )
            print(
                f"[Data:{dataset_name}] internal split counts="
                f"{split_counts}",
                flush=True,
            )
            accepted = {value.lower() for value in split_aliases}
            self.rows = self.rows.filter(
                lambda row: str(row["split"]).lower() in accepted
            )
            if len(self.rows) == 0:
                raise ValueError(
                    f"The internal split column contains no {split!r} rows "
                    f"for {dataset_name}"
                )
        elif must_filter_internal_split:
            if split not in {"train", "validation"}:
                raise FileNotFoundError(
                    f"No {split!r} parquet files and no internal 'split' "
                    f"column were found for {dataset_name} in {data_dir}"
                )

            # lmms-lab-encoder/flickr30k publishes all 31,783 images in nine
            # test-*.parquet files and does not preserve the original split.
            # Create the same non-overlapping holdout on every invocation so
            # training and validation remain reproducible across all stages.
            partitions = self.rows.train_test_split(
                test_size=MERGED_DATASET_VALIDATION_FRACTION,
                seed=MERGED_DATASET_SPLIT_SEED,
                shuffle=True,
            )
            partition_name = "train" if split == "train" else "test"
            self.rows = partitions[partition_name]
            print(
                f"[Data:{dataset_name}] WARNING: source has one merged "
                f"physical split and no internal split column; using a "
                f"deterministic "
                f"{100 * (1 - MERGED_DATASET_VALIDATION_FRACTION):.0f}%/"
                f"{100 * MERGED_DATASET_VALIDATION_FRACTION:.0f}% "
                f"train/validation partition (seed="
                f"{MERGED_DATASET_SPLIT_SEED}).",
                flush=True,
            )

        if "image" not in self.rows.column_names:
            raise ValueError(
                f"{dataset_name} parquet data has no 'image' column"
            )
        numbered_columns = tuple(
            column for column in CAPTION_COLUMNS
            if column in self.rows.column_names
        )
        container_columns = tuple(
            column for column in CAPTION_CONTAINER_COLUMNS
            if column in self.rows.column_names
        )
        self.caption_columns = numbered_columns or container_columns
        if not self.caption_columns:
            raise ValueError(
                f"{dataset_name} has no supported caption columns; found "
                f"{self.rows.column_names}"
            )
        print(
            f"[Data:{dataset_name}] loaded rows={loaded_rows:,} | "
            f"selected {split} rows={len(self.rows):,} | "
            f"caption columns={list(self.caption_columns)}",
            flush=True,
        )

        self.training = split == "train"
        self.text_seq_length = config.text_seq_length
        self.transform = _image_transform(
            config,
            train=self.training,
            augment_data=augment_data,
            natural_image=True,
        )

    def __len__(self) -> int:
        return len(self.rows)

    @staticmethod
    def _caption_strings(value: Any) -> list[str]:
        """Flatten strings from list/dict caption representations."""
        if value is None:
            return []
        if isinstance(value, str):
            value = value.strip()
            return [value] if value else []
        if isinstance(value, dict):
            preferred = ("raw", "sentence", "text", "caption")
            keys = [key for key in preferred if key in value]
            if not keys:
                keys = list(value)
            return [
                text
                for key in keys
                for text in FlickrPairs._caption_strings(value[key])
            ]
        if isinstance(value, (list, tuple)):
            return [
                text
                for item in value
                for text in FlickrPairs._caption_strings(item)
            ]
        text = str(value).strip()
        return [text] if text else []

    def _valid_captions(self, row: dict[str, Any]) -> list[str]:
        captions = [
            text
            for column in self.caption_columns
            for text in self._caption_strings(row[column])
        ]
        if not captions:
            raise ValueError(
                f"A {self.dataset_name} row has no non-empty caption"
            )
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


class Flickr8kPairs(FlickrPairs):
    """Backward-compatible Flickr8k-only constructor."""

    def __init__(self, config, split: str, augment_data: bool = False) -> None:
        super().__init__(
            config,
            dataset_name="flickr8k",
            data_location=config.flickr8k_data_location,
            split=split,
            augment_data=augment_data,
        )


class CombinedPairs(ConcatDataset):
    """Concatenate compatible datasets while preserving helper methods."""

    def sample_texts(self, count: int = 10) -> list[str]:
        texts: list[str] = []
        for dataset in self.datasets:
            if hasattr(dataset, "sample_texts"):
                texts.extend(dataset.sample_texts(count))
            if len(texts) >= count:
                break
        return texts[:count]

    @property
    def captions(self) -> dict[int, str]:
        return {
            index: text for index, text in enumerate(self.sample_texts(100))
        }


def describe_dataset(dataset: Dataset, label: str) -> None:
    """Print a compact dataset summary suitable for a Slurm output log."""
    print(f"[Data summary] {label} total samples: {len(dataset):,}", flush=True)
    if isinstance(dataset, CombinedPairs):
        for component in dataset.datasets:
            name = getattr(component, "dataset_name", type(component).__name__)
            print(
                f"[Data summary]   {name}: {len(component):,} samples",
                flush=True,
            )


def _flickr_pairs(config, dataset_name, split, augment_data=False):
    location = getattr(config, f"{dataset_name}_data_location")
    return FlickrPairs(
        config,
        dataset_name=dataset_name,
        data_location=location,
        split=split,
        augment_data=augment_data,
    )


def get_train_set(config, augment_data: bool = False):
    if config.dataset == "fashion_mnist":
        dataset = FashionMNISTPairs(config, train=True, augment_data=augment_data)
    elif config.dataset in {"flickr8k", "flickr30k"}:
        dataset = _flickr_pairs(
            config, config.dataset, "train", augment_data
        )
    elif config.dataset == "all":
        dataset = CombinedPairs(
            [
                _flickr_pairs(config, "flickr8k", "train", augment_data),
                _flickr_pairs(config, "flickr30k", "train", augment_data),
            ]
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
    if config.dataset in {"flickr8k", "flickr30k"}:
        # The validation split is used while fitting model stages. The held-out
        # test split remains untouched for final evaluation experiments.
        return _flickr_pairs(config, config.dataset, "validation")
    if config.dataset == "all":
        return CombinedPairs(
            [
                _flickr_pairs(config, "flickr8k", "validation"),
                _flickr_pairs(config, "flickr30k", "validation"),
            ]
        )
    raise ValueError(f"Unsupported dataset: {config.dataset}")
