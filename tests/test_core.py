import unittest

import torch
from torch import nn

from continual_learning.buffer import ReplayBuffer
from continual_learning.data import ClassSubset, SyntheticImageDataset, make_class_order, make_task_specs
from continual_learning.ewc import OnlineEWC
from continual_learning.metrics import summarize_accuracy_matrix


class TaskSplitTests(unittest.TestCase):
    def test_task_specs_cover_every_class_once(self):
        order = make_class_order(12, seed=7)
        specs = make_task_specs(12, 3, order)
        flattened = [class_id for spec in specs for class_id in spec.class_ids]
        self.assertEqual(flattened, order)
        self.assertEqual(sorted(flattened), list(range(12)))

    def test_class_subset_keeps_original_labels(self):
        dataset = SyntheticImageDataset(4, samples_per_class=3, seed=1)
        subset = ClassSubset(dataset, [1, 3])
        self.assertEqual(len(subset), 6)
        self.assertEqual(set(subset.targets), {1, 3})


class ReplayBufferTests(unittest.TestCase):
    def test_reservoir_never_exceeds_capacity(self):
        buffer = ReplayBuffer(capacity=5, seed=2)
        examples = torch.randn(20, 3, 4, 4)
        labels = torch.arange(20)
        buffer.add_batch(examples, labels)
        self.assertEqual(len(buffer), 5)
        self.assertEqual(buffer.seen, 20)
        sampled_examples, sampled_labels = buffer.sample(4, torch.device("cpu"))
        self.assertEqual(sampled_examples.shape, (4, 3, 4, 4))
        self.assertEqual(sampled_labels.shape, (4,))


class MetricTests(unittest.TestCase):
    def test_accuracy_and_forgetting(self):
        matrix = [
            [0.8, None, None],
            [0.5, 0.7, None],
            [0.4, 0.6, 0.9],
        ]
        summary = summarize_accuracy_matrix(matrix)
        self.assertAlmostEqual(summary["final_average_accuracy"], 1.9 / 3)
        self.assertAlmostEqual(summary["final_average_forgetting"], 0.25)


class EWCTests(unittest.TestCase):
    def test_penalty_is_zero_at_mean_and_positive_after_change(self):
        model = nn.Linear(2, 2, bias=False)
        ewc = OnlineEWC()
        ewc.means = {"weight": model.weight.detach().clone()}
        ewc.fisher = {"weight": torch.ones_like(model.weight)}
        self.assertAlmostEqual(ewc.penalty(model).item(), 0.0)
        with torch.no_grad():
            model.weight.add_(1.0)
        self.assertGreater(ewc.penalty(model).item(), 0.0)


if __name__ == "__main__":
    unittest.main()
