"""Paired-seed comparison of released-code vs equal-budget Tee & Zhang runs.

Both arms must come from the same platform: Windows and WSL runs differ in the
CPU bicubic resize at the float-rounding level, so cross-platform pairs are not
bit-comparable. Division 1 is identical under both sequence implementations, so
its runs serve as the shared no-interleaving reference.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import statistics
from pathlib import Path

# Two-sided 95% Student-t critical values by degrees of freedom.
T_CRIT_95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
METRICS = ("continual_average_accuracy", "forgetfulness")


def load_arm(root: Path, implementation: str, division: int, seeds: list[int]) -> dict[int, dict]:
    runs = {}
    for seed in seeds:
        path = root / implementation / f"division-{division:03d}" / f"seed-{seed}" / "results.json"
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        config = result["config"]
        if (config["sequence_implementation"], config["division"], config["seed"]) != (implementation, division, seed):
            raise ValueError(f"config mismatch in {path}")
        runs[seed] = result
    return runs


def paired_stats(differences: list[float]) -> dict[str, float | int]:
    n = len(differences)
    mean = statistics.mean(differences)
    stats: dict[str, float | int] = {"n": n, "mean_diff": mean}
    if n > 1:
        sd = statistics.stdev(differences)
        half_width = T_CRIT_95[n - 1] * sd / math.sqrt(n)
        stats.update(sd_diff=sd, ci95_low=mean - half_width, ci95_high=mean + half_width)
        # Exact two-sided sign-flip permutation test; the minimum attainable p is 2 / 2**n.
        observed = abs(sum(differences))
        flips = list(itertools.product((1, -1), repeat=n))
        extreme = sum(1 for signs in flips if abs(sum(s * d for s, d in zip(signs, differences))) >= observed - 1e-12)
        stats["sign_flip_p"] = extreme / len(flips)
    stats["positive_pairs"] = sum(d > 0 for d in differences)
    return stats


def compare(
    label: str,
    arm_a: dict[int, dict],
    arm_b: dict[int, dict],
) -> tuple[dict, list[dict]]:
    """Differences are arm_b - arm_a for each seed present in both arms."""
    seeds = sorted(set(arm_a) & set(arm_b))
    rows = []
    summary: dict[str, object] = {"comparison": label, "seeds": seeds}
    for metric in METRICS:
        diffs = []
        for seed in seeds:
            a = float(arm_a[seed]["metrics"][metric])
            b = float(arm_b[seed]["metrics"][metric])
            diffs.append(b - a)
            rows.append({"comparison": label, "metric": metric, "seed": seed, "a": a, "b": b, "diff": b - a})
        if diffs:
            summary[metric] = paired_stats(diffs)
    return summary, rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", default="runs/tee-zhang-2023/wsl-platform-check",
                        help="root holding released-code/ runs from the same platform as the equal-budget arm")
    parser.add_argument("--equal-budget-root", default="runs/tee-zhang-2023")
    parser.add_argument("--division1-root", default="runs/tee-zhang-2023",
                        help="root holding released-code/division-001 (identical sequence under both implementations)")
    parser.add_argument("--divisions", type=int, nargs="+", default=[120, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--output-dir", default="runs/tee-zhang-2023/equal-budget-comparison")
    args = parser.parse_args()

    baseline_root = Path(args.baseline_root)
    equal_root = Path(args.equal_budget_root)
    division1 = load_arm(Path(args.division1_root), "released-code", 1, args.seeds)

    summaries, rows = [], []
    for division in args.divisions:
        released = load_arm(baseline_root, "released-code", division, args.seeds)
        equal = load_arm(equal_root, "equal-budget", division, args.seeds)
        for label, a, b in (
            (f"div{division}: equal-budget - released-code", released, equal),
            (f"div{division} equal-budget - div1", division1, equal),
            (f"div{division} released-code - div1", division1, released),
        ):
            summary, comparison_rows = compare(label, a, b)
            summaries.append(summary)
            rows.extend(comparison_rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open("w", encoding="utf-8") as handle:
        json.dump({"sources": vars(args), "summaries": summaries}, handle, indent=2)
    with (output_dir / "paired_rows.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["comparison", "metric", "seed", "a", "b", "diff"])
        writer.writeheader()
        writer.writerows(rows)

    for summary in summaries:
        print(summary["comparison"], "seeds", summary["seeds"])
        for metric in METRICS:
            stats = summary.get(metric)
            if not stats:
                continue
            text = f"  {metric}: mean diff {100 * stats['mean_diff']:+.2f} pp"
            if "ci95_low" in stats:
                text += (f", 95% CI [{100 * stats['ci95_low']:+.2f}, {100 * stats['ci95_high']:+.2f}]"
                         f", sign-flip p={stats['sign_flip_p']:.3f}")
            print(text + f", positive {stats['positive_pairs']}/{stats['n']}")


if __name__ == "__main__":
    main()
