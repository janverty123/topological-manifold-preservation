"""
data.py
--------
Implements Methodology Section I: "Gathering of Data".

Loads the standard MNIST dataset and partitions it into a sequential
two-task continual-learning stream (Split-MNIST):
    Task 1 -> digits 0-4
    Task 2 -> digits 5-9

Both tasks are exposed with the FULL 10-dimensional label space so a
single-head classifier can be used; a class mask is provided so that
task-specific losses / accuracies can be computed correctly (standard
practice in class-incremental continual-learning benchmarks).
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np


MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


class _FlattenTransform:
    """
    Picklable replacement for `transforms.Lambda(lambda x: x.view(-1))`.
    Windows' multiprocessing uses `spawn` (not `fork`), which requires
    pickling the Dataset (including its transform pipeline) to hand off
    to DataLoader worker processes. Plain lambdas cannot be pickled, so
    a named callable class is used instead — this is required for
    `num_workers > 0` to work on Windows.
    """

    def __call__(self, x):
        return x.view(-1)


def _mnist_transform():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
            _FlattenTransform(),  # flatten 28x28 -> 784
        ]
    )


def load_raw_mnist(data_dir: str):
    """Downloads (if needed) and returns the raw torchvision MNIST train/test sets."""
    transform = _mnist_transform()
    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set


def _indices_for_classes(dataset, classes):
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in classes:
        mask |= (targets == c)
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def build_split_mnist(data_dir: str, task1_classes, task2_classes):
    """
    Returns a dict with train/test Subsets for Task 1 and Task 2.

    {
        "task1": {"train": Subset, "test": Subset},
        "task2": {"train": Subset, "test": Subset},
    }
    """
    train_set, test_set = load_raw_mnist(data_dir)

    task1_train_idx = _indices_for_classes(train_set, task1_classes)
    task1_test_idx = _indices_for_classes(test_set, task1_classes)
    task2_train_idx = _indices_for_classes(train_set, task2_classes)
    task2_test_idx = _indices_for_classes(test_set, task2_classes)

    return {
        "task1": {
            "train": Subset(train_set, task1_train_idx),
            "test": Subset(test_set, task1_test_idx),
        },
        "task2": {
            "train": Subset(train_set, task2_train_idx),
            "test": Subset(test_set, task2_test_idx),
        },
    }


def make_loader(subset, batch_size, shuffle):
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def get_dataloaders(cfg):
    """
    High level convenience wrapper used by the training scripts.
    `cfg` is the parsed YAML config (dict-like).
    """
    splits = build_split_mnist(cfg["data_dir"], cfg["task1_classes"], cfg["task2_classes"])

    loaders = {
        "task1_train": make_loader(splits["task1"]["train"], cfg["batch_size"], shuffle=True),
        "task1_test": make_loader(splits["task1"]["test"], cfg["eval_batch_size"], shuffle=False),
        "task2_train": make_loader(splits["task2"]["train"], cfg["batch_size"], shuffle=True),
        "task2_test": make_loader(splits["task2"]["test"], cfg["eval_batch_size"], shuffle=False),
    }
    return loaders
