import random
import unittest

from continual_learning.reproductions.tee_zhang_2023 import (
    ClassBalancedReplayBuffer,
    build_interleave_plan,
    paper_class_order,
    paper_metrics,
)


class PaperProtocolTests(unittest.TestCase):
    def test_class_order_matches_released_code(self):
        self.assertEqual(
            paper_class_order(seed=100)[:10],
            [57, 76, 27, 11, 17, 34, 83, 89, 19, 31],
        )

    def test_equal_budget_uses_every_example_once(self):
        plan = build_interleave_plan(
            current_indices=list(range(15)),
            replay_indices=list(range(100, 112)),
            division=4,
            rng=random.Random(3),
            implementation="equal-budget",
        )
        self.assertEqual(plan.presented_current_count, 15)
        self.assertEqual(plan.presented_replay_count, 12)
        self.assertEqual(len(plan.indices), 27)

    def test_released_code_duplicates_current_remainder(self):
        plan = build_interleave_plan(
            current_indices=list(range(15)),
            replay_indices=list(range(100, 112)),
            division=4,
            rng=random.Random(3),
            implementation="released-code",
        )
        self.assertEqual(plan.presented_current_count, 18)
        self.assertEqual(plan.presented_replay_count, 12)
        self.assertEqual(len(plan.indices), 30)

    def test_replay_buffer_is_class_balanced(self):
        buffer = ClassBalancedReplayBuffer(capacity=12, rng=random.Random(5))
        buffer.update({label: list(range(label * 20, label * 20 + 20)) for label in range(5)}, 5)
        self.assertEqual(sorted(buffer.class_counts().values()), [2, 2, 2, 3, 3])
        self.assertEqual(len(buffer), 12)
        buffer.update(
            {label: list(range(label * 20, label * 20 + 20)) for label in range(5, 10)},
            10,
        )
        self.assertEqual(sorted(buffer.class_counts().values()), [1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
        self.assertEqual(len(buffer), 12)

    def test_paper_metrics_average_over_task_trajectory(self):
        metrics = paper_metrics(
            seen_accuracy=[0.8, 0.7, 0.6],
            task1_accuracy=[0.8, 0.6, 0.5],
        )
        self.assertAlmostEqual(metrics["continual_average_accuracy"], 0.7)
        self.assertAlmostEqual(metrics["forgetfulness"], (0.25 + 0.375) / 2)


if __name__ == "__main__":
    unittest.main()
