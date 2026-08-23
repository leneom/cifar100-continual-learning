from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class TaskSpec:
    task_id: int
    class_ids: tuple[int, ...]


@dataclass
class TaskStream:
    specs: list[TaskSpec]
    train_datasets: list[Dataset]
    test_datasets: list[Dataset]
    class_order: list[int]
    num_classes: int


def make_class_order(num_classes: int, seed: int) -> list[int]:
    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    generator = torch.Generator().manual_seed(seed)
    return torch.randperm(num_classes, generator=generator).tolist()


def make_task_specs(
    num_classes: int,
    classes_per_task: int,
    class_order: Sequence[int] | None = None,
) -> list[TaskSpec]:
    if classes_per_task <= 0:
        raise ValueError("classes_per_task must be positive")
    if num_classes % classes_per_task != 0:
        raise ValueError("num_classes must be divisible by classes_per_task")

    order = list(range(num_classes)) if class_order is None else list(class_order)
    if sorted(order) != list(range(num_classes)):
        raise ValueError("class_order must contain every class exactly once")

    return [
        TaskSpec(task_id=task_id, class_ids=tuple(order[start : start + classes_per_task]))
        for task_id, start in enumerate(range(0, num_classes, classes_per_task))
    ]


class ClassSubset(Dataset):
    """A dataset view containing only selected labels, without remapping them."""

    def __init__(self, dataset: Dataset, class_ids: Sequence[int]):
        targets = getattr(dataset, "targets", None)
        if targets is None:
            raise ValueError("dataset must expose a targets attribute")
        selected = set(int(class_id) for class_id in class_ids)
        self.dataset = dataset
        self.indices = [index for index, target in enumerate(targets) if int(target) in selected]
        self.targets = [int(targets[index]) for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.dataset[self.indices[index]]


class SyntheticImageDataset(Dataset):
    """Deterministic CIFAR-shaped images for fast end-to-end checks."""

    def __init__(
        self,
        num_classes: int,
        samples_per_class: int,
        seed: int,
        noise: float = 0.08,
    ):
        if num_classes <= 0 or samples_per_class <= 0:
            raise ValueError("num_classes and samples_per_class must be positive")
        generator = torch.Generator().manual_seed(seed)
        images: list[torch.Tensor] = []
        targets: list[int] = []
        for class_id in range(num_classes):
            prototype = torch.zeros(3, 32, 32)
            channel = class_id % 3
            row = 2 + (class_id * 5) % 24
            column = 2 + (class_id * 7) % 24
            prototype[channel, row : row + 7, :] = 0.8
            prototype[(channel + 1) % 3, :, column : column + 7] = 0.5
            prototype += class_id / max(1, num_classes - 1) * 0.15
            for _ in range(samples_per_class):
                sample = prototype + noise * torch.randn(3, 32, 32, generator=generator)
                images.append(sample.clamp(0.0, 1.0))
                targets.append(class_id)
        self.images = torch.stack(images)
        self.targets = targets

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        return self.images[index], self.targets[index]


def build_synthetic_stream(
    num_classes: int,
    classes_per_task: int,
    train_samples_per_class: int,
    test_samples_per_class: int,
    seed: int,
) -> TaskStream:
    order = make_class_order(num_classes, seed)
    specs = make_task_specs(num_classes, classes_per_task, order)
    train = SyntheticImageDataset(num_classes, train_samples_per_class, seed=seed + 10)
    test = SyntheticImageDataset(num_classes, test_samples_per_class, seed=seed + 20)
    return TaskStream(
        specs=specs,
        train_datasets=[ClassSubset(train, spec.class_ids) for spec in specs],
        test_datasets=[ClassSubset(test, spec.class_ids) for spec in specs],
        class_order=order,
        num_classes=num_classes,
    )


def build_cifar100_stream(
    root: str | Path,
    classes_per_task: int,
    seed: int,
    download: bool,
) -> TaskStream:
    from torchvision import datasets, transforms

    root = Path(root)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ]
    )
    train = datasets.CIFAR100(root=root, train=True, transform=train_transform, download=download)
    test = datasets.CIFAR100(root=root, train=False, transform=test_transform, download=download)
    order = make_class_order(100, seed)
    specs = make_task_specs(100, classes_per_task, order)
    return TaskStream(
        specs=specs,
        train_datasets=[ClassSubset(train, spec.class_ids) for spec in specs],
        test_datasets=[ClassSubset(test, spec.class_ids) for spec in specs],
        class_order=order,
        num_classes=100,
    )

