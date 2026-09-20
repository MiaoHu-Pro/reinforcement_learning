"""Backward-compatible dataset imports.

The implementations now live in :mod:`dalle2_dataset`, which supports both
FashionMNIST and the local Flickr8k parquet dataset.
"""

from dalle2_dataset import (  # noqa: F401
    FashionMNISTPairs as FMNIST,
    Flickr8kPairs,
    get_test_set,
    get_train_set,
)


def get_train_val_split(config, augment_data=False):
    """Return independent train and validation datasets."""
    train_set, mean, std = get_train_set(config, augment_data=augment_data)
    val_set = get_test_set(config, mean=mean, std=std)
    return train_set, val_set, mean, std
