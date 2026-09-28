from __future__ import annotations

import random

import torch

SAMPLING_POLICIES = ("uniform", "loss")


class ReplayBuffer:
    """Fixed-size replay memory maintained with reservoir sampling.

    Storage is always reservoir sampling. ``sampling`` only changes which stored
    examples are drawn for replay:

    - ``uniform``: every stored example is equally likely (the original policy);
    - ``loss``: prioritized replay in the style of Schaul et al. (2016). Each slot
      keeps priority ``(last replay loss + epsilon) ** alpha`` and is drawn without
      replacement with probability proportional to it. New examples receive the
      current maximum priority so that each is replayed at least once soon.
    """

    def __init__(
        self,
        capacity: int,
        seed: int,
        sampling: str = "uniform",
        priority_alpha: float = 0.6,
        priority_epsilon: float = 1e-3,
    ):
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        if sampling not in SAMPLING_POLICIES:
            raise ValueError(f"unknown replay sampling policy: {sampling}")
        if priority_alpha < 0 or priority_epsilon <= 0:
            raise ValueError("priority_alpha must be non-negative and priority_epsilon positive")
        self.capacity = capacity
        self.sampling = sampling
        self.priority_alpha = priority_alpha
        self.priority_epsilon = priority_epsilon
        self._rng = random.Random(seed)
        self._examples: list[torch.Tensor] = []
        self._labels: list[int] = []
        self._priorities: list[float] = []
        self._max_priority = 1.0
        self.seen = 0

    def __len__(self) -> int:
        return len(self._labels)

    @property
    def labels(self) -> tuple[int, ...]:
        return tuple(self._labels)

    @property
    def priorities(self) -> tuple[float, ...]:
        return tuple(self._priorities)

    def add_batch(self, examples: torch.Tensor, labels: torch.Tensor) -> None:
        for example, label in zip(examples.detach().cpu(), labels.detach().cpu()):
            self.seen += 1
            if self.capacity == 0:
                continue
            if len(self._labels) < self.capacity:
                self._examples.append(example.clone())
                self._labels.append(int(label))
                self._priorities.append(self._max_priority)
                continue
            candidate = self._rng.randrange(self.seen)
            if candidate < self.capacity:
                self._examples[candidate] = example.clone()
                self._labels[candidate] = int(label)
                self._priorities[candidate] = self._max_priority

    def _draw_indices(self, count: int) -> list[int]:
        if self.sampling == "uniform":
            return self._rng.sample(range(len(self._labels)), count)
        # Weighted sampling without replacement (Efraimidis & Spirakis, 2006):
        # keep the ``count`` largest keys u ** (1 / w).
        keys = [self._rng.random() ** (1.0 / priority) for priority in self._priorities]
        return sorted(range(len(keys)), key=keys.__getitem__, reverse=True)[:count]

    def sample_indexed(
        self,
        batch_size: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if not self._labels:
            raise RuntimeError("cannot sample an empty replay buffer")
        count = min(batch_size, len(self._labels))
        indices = self._draw_indices(count)
        examples = torch.stack([self._examples[index] for index in indices]).to(device)
        labels = torch.tensor([self._labels[index] for index in indices], dtype=torch.long, device=device)
        return examples, labels, indices

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        examples, labels, _ = self.sample_indexed(batch_size, device)
        return examples, labels

    def update_priorities(self, indices: list[int], losses: torch.Tensor) -> None:
        """Set each slot's priority from its latest per-example replay loss."""
        if self.sampling != "loss":
            return
        values = losses.detach().float().cpu().tolist()
        if len(values) != len(indices):
            raise ValueError("indices and losses must have the same length")
        for index, loss in zip(indices, values):
            priority = (max(loss, 0.0) + self.priority_epsilon) ** self.priority_alpha
            self._priorities[index] = priority
            self._max_priority = max(self._max_priority, priority)
