import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

from run_replay_ablation import matches, summarize_records, write_summary_svg


def result(buffer_size, seed, accuracy, forgetting):
    return {
        "config": {
            "dataset": "cifar100",
            "method": "replay",
            "model": "resnet18",
            "num_classes": 100,
            "classes_per_task": 10,
            "epochs_per_task": 10,
            "batch_size": 128,
            "learning_rate": 0.1,
            "momentum": 0.9,
            "weight_decay": 5e-4,
            "replay_batch_size": 64,
            "seed": seed,
            "buffer_size": buffer_size,
            "deterministic": True,
            "max_train_samples_per_task": None,
        },
        "metrics": {
            "final_average_accuracy": accuracy,
            "final_average_forgetting": forgetting,
        },
    }


class AblationTests(unittest.TestCase):
    def test_matches_all_comparison_fields(self):
        item = result(500, 42, 0.2, 0.4)
        self.assertTrue(matches(item, item["config"]))
        different = dict(item["config"])
        different["seed"] = 1
        self.assertFalse(matches(item, different))

    def test_summary_groups_seeds_and_computes_sample_std(self):
        summary = summarize_records(
            [
                result(200, 0, 0.1, 0.5),
                result(200, 1, 0.3, 0.3),
                result(500, 0, 0.4, 0.2),
            ]
        )
        self.assertEqual([row["buffer_size"] for row in summary], [200, 500])
        self.assertEqual(summary[0]["seeds"], [0, 1])
        self.assertAlmostEqual(summary[0]["accuracy_mean"], 0.2)
        self.assertAlmostEqual(summary[0]["accuracy_std"], 2**0.5 / 10)
        self.assertEqual(summary[1]["accuracy_std"], 0.0)

    def test_svg_contains_parseable_standard_deviation_error_bars(self):
        summary = summarize_records(
            [
                result(200, 0, 0.1, 0.5),
                result(200, 1, 0.3, 0.3),
                result(500, 0, 0.4, 0.2),
            ]
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "summary.svg"
            write_summary_svg(summary, path)
            root = ET.parse(path).getroot()
            error_bars = root.findall(".//*[@class='error-bar']")
            self.assertEqual(len(error_bars), 6)
            self.assertIn("sample standard deviation", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
