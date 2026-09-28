import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))

from analyze_tee_zhang_equal_budget import paired_stats  # noqa: E402
from continual_learning.reproductions.tee_zhang_2023 import build_interleave_plan  # noqa: E402


class EqualBudgetAnalysisTests(unittest.TestCase):
    def test_division_one_sequence_is_identical_across_implementations(self):
        current, replay = list(range(2250)), list(range(10_000, 11_200))
        released = build_interleave_plan(current, replay, 1, random.Random(3), "released-code")
        equal = build_interleave_plan(current, replay, 1, random.Random(3), "equal-budget")
        self.assertEqual(released.indices, equal.indices)

    def test_paired_stats_sign_flip_and_interval(self):
        stats = paired_stats([0.01, 0.02, 0.03, 0.04])
        self.assertAlmostEqual(stats["mean_diff"], 0.025)
        self.assertEqual(stats["sign_flip_p"], 0.125)
        self.assertLess(stats["ci95_low"], 0.025)
        self.assertGreater(stats["ci95_high"], 0.025)
        self.assertEqual(paired_stats([0.01, -0.01, 0.01, -0.01])["sign_flip_p"], 1.0)


if __name__ == "__main__":
    unittest.main()
