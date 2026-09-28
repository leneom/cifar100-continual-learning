import random
import unittest
from collections import Counter

import torch

from continual_learning.buffer import ReplayBuffer


def legacy_reservoir_draws(capacity: int, seed: int, labels: list[int], draws: int, batch: int) -> list[list[int]]:
    """Index draws of the original uniform ReplayBuffer, reimplemented verbatim."""
    rng = random.Random(seed)
    stored: list[int] = []
    for seen, label in enumerate(labels, start=1):
        if len(stored) < capacity:
            stored.append(label)
            continue
        candidate = rng.randrange(seen)
        if candidate < capacity:
            stored[candidate] = label
    return [[stored[i] for i in rng.sample(range(len(stored)), batch)] for _ in range(draws)]


class ReplaySamplingTests(unittest.TestCase):
    def _filled(self, sampling: str, capacity: int = 20, count: int = 50) -> ReplayBuffer:
        buffer = ReplayBuffer(capacity, seed=5, sampling=sampling)
        buffer.add_batch(torch.zeros(count, 1), torch.arange(count))
        return buffer

    def test_uniform_matches_original_implementation(self):
        buffer = self._filled("uniform")
        expected = legacy_reservoir_draws(20, 5, list(range(50)), draws=3, batch=8)
        observed = [buffer.sample(8, torch.device("cpu"))[1].tolist() for _ in range(3)]
        self.assertEqual(observed, expected)

    def test_uniform_ignores_priority_updates(self):
        buffer = self._filled("uniform")
        buffer.update_priorities([0, 1], torch.tensor([5.0, 5.0]))
        self.assertEqual(set(buffer.priorities), {1.0})

    def test_loss_sampling_prefers_high_loss_slots_without_replacement(self):
        buffer = self._filled("loss", capacity=10, count=10)
        losses = torch.tensor([10.0] * 2 + [0.0] * 8)
        buffer.update_priorities(list(range(10)), losses)
        counts: Counter[int] = Counter()
        for _ in range(500):
            _, labels, indices = buffer.sample_indexed(4, torch.device("cpu"))
            self.assertEqual(len(set(indices)), 4)
            counts.update(labels.tolist())
        self.assertGreater(min(counts[0], counts[1]), 450)
        self.assertLess(max(counts[label] for label in range(2, 10)), 300)

    def test_new_examples_receive_maximum_priority(self):
        buffer = ReplayBuffer(4, seed=1, sampling="loss")
        buffer.add_batch(torch.zeros(2, 1), torch.tensor([0, 1]))
        buffer.update_priorities([0, 1], torch.tensor([3.0, 0.0]))
        buffer.add_batch(torch.zeros(1, 1), torch.tensor([2]))
        self.assertAlmostEqual(buffer.priorities[2], max(buffer.priorities))
        self.assertAlmostEqual(buffer.priorities[2], (3.0 + 1e-3) ** 0.6)


if __name__ == "__main__":
    unittest.main()
