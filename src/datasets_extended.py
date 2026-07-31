"""
datasets_extended.py
---------------------
Generalizes the original 2-task Split-MNIST (src/data.py) to an
arbitrary number of sequential class-incremental tasks, used to test
whether TMP's protection holds up over a longer sequence of smaller
tasks within the same dataset your research plan specifies.

Returns a list of per-task dicts:
    {"train": <Dataset>, "test": <Dataset>, "classes": [int, ...]}
which src/train_general.py consumes to run N-task sequential continual
learning under finetune / EWC / TMP.
"""

import torch
from torch.utils.data import Subset
from torchvision import datasets, transforms

from src.data import _FlattenTransform

MNIST_MEAN, MNIST_STD = 0.1307, 0.3081


def _indices_for_classes(dataset, classes):
    targets = dataset.targets
    if not torch.is_tensor(targets):
        targets = torch.tensor(targets)
    mask = torch.zeros_like(targets, dtype=torch.bool)
    for c in classes:
        mask |= (targets == c)
    return torch.nonzero(mask, as_tuple=True)[0].tolist()


def build_split_mnist_tasks(data_dir: str, num_tasks: int = 5, seed: int = 42):
    """
    Class-incremental Split-MNIST generalized to N sequential tasks.
    The original 2-task version (src/data.py: digits 0-4 then 5-9) is
    the N=2 special case of this. Splits the 10 MNIST digit classes
    into `num_tasks` disjoint, sequential groups of 10/num_tasks
    classes each in numeric order -- e.g. num_tasks=5 gives
    [0,1] -> [2,3] -> [4,5] -> [6,7] -> [8,9].

    Class order here is fixed (not seed-shuffled) to match the same
    digit ordering convention as the original 2-task Split-MNIST setup,
    for direct comparability. `seed` is accepted for interface
    consistency but unused.
    """
    if 10 % num_tasks != 0:
        raise ValueError(f"num_tasks ({num_tasks}) must evenly divide 10 MNIST classes.")
    classes_per_task = 10 // num_tasks

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
        _FlattenTransform(),
    ])
    train_set = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)

    tasks = []
    for t in range(num_tasks):
        task_classes = list(range(t * classes_per_task, (t + 1) * classes_per_task))
        train_idx = _indices_for_classes(train_set, task_classes)
        test_idx = _indices_for_classes(test_set, task_classes)
        tasks.append({
            "train": Subset(train_set, train_idx),
            "test": Subset(test_set, test_idx),
            "classes": task_classes,
        })
    return tasks


DATASET_BUILDERS = {
    "split_mnist": build_split_mnist_tasks,
}


def build_tasks(dataset_name: str, data_dir: str, num_tasks: int, seed: int = 42):
    if dataset_name not in DATASET_BUILDERS:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Options: {list(DATASET_BUILDERS.keys())}")
    return DATASET_BUILDERS[dataset_name](data_dir, num_tasks=num_tasks, seed=seed)
