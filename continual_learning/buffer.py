from __future__ import annotations

import random

import torch


class ReplayBuffer:
    """Fixed-size replay memory maintained with reservoir sampling."""

    def __init__(self, capacity: int, seed: int):
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = capacity
        self._rng = random.Random(seed)
        self._examples: list[torch.Tensor] = []
        self._labels: list[int] = []
        self.seen = 0

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(self._labels)

    def add_batch(self, examples: torch.Tensor, labels: torch.Tensor) -> None:
        for example, label in zip(examples.detach().cpu(), labels.detach().cpu()):
            self.seen += 1
            if self.capacity == 0:
                continue
            if len(self._labels) < self.capacity:
                self._examples.append(example.clone())
                self._labels.append(int(label))
                continue
            candidate = self._rng.randrange(self.seen)
            if candidate < self.capacity:
                self._examples[candidate] = example.clone()
                self._labels[candidate] = int(label)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._labels:
            raise RuntimeError("cannot sample an empty replay buffer")
        count = min(batch_size, len(self._labels))
        indices = self._rng.sample(range(len(self._labels)), count)
        examples = torch.stack([self._examples[index] for index in indices]).to(device)
        labels = torch.tensor([self._labels[index] for index in indices], dtype=torch.long, device=device)
        return examples, labels

