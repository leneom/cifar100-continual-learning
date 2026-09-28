"""Same-platform Table 1 curve for the released-code Tee & Zhang runs.

Summarizes every division under one output root (WSL by default) against the
paper and reports paired-seed differences versus division 1 and between
adjacent divisions.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from analyze_tee_zhang_equal_budget import METRICS, compare, load_arm  # noqa: E402

PAPER_TABLE = {
    1: {"forgetfulness": 0.639, "continual_average_accuracy": 0.399},
    8: {"forgetfulness": 0.626, "continual_average_accuracy": 0.407},
    60: {"forgetfulness": 0.574, "continual_average_accuracy": 0.446},
    120: {"forgetfulness": 0.551, "continual_average_accuracy": 0.466},
    300: {"forgetfulness": 0.561, "continual_average_accuracy": 0.466},
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="runs/tee-zhang-2023/wsl-platform-check")
    parser.add_argument("--divisions", type=int, nargs="+", default=sorted(PAPER_TABLE))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(8)))
    parser.add_argument("--output-dir", default="runs/tee-zhang-2023/wsl-curve")
    args = parser.parse_args()

    arms = {division: load_arm(Path(args.root), "released-code", division, args.seeds) for division in args.divisions}
    rows = []
    for division, arm in arms.items():
        if not arm:
            continue
        row: dict[str, object] = {"division": division, "n": len(arm), "seeds": " ".join(map(str, sorted(arm)))}
        for metric in METRICS:
            values = [float(result["metrics"][metric]) for result in arm.values()]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
            row[f"paper_{metric}"] = PAPER_TABLE[division][metric]
        rows.append(row)

    divisions = [division for division in args.divisions if arms[division]]
    pairs = [(1, division) for division in divisions if division != 1]
    pairs += [(a, b) for a, b in zip(divisions, divisions[1:]) if a != 1]
    summaries = []
    for a, b in pairs:
        summary, _ = compare(f"div{b} - div{a}", arms[a], arms[b])
        summaries.append(summary)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "curve.json").open("w", encoding="utf-8") as handle:
        json.dump({"sources": vars(args), "divisions": rows, "paired": summaries}, handle, indent=2)
    with (output_dir / "curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        acc, forget = "continual_average_accuracy", "forgetfulness"
        print(
            f"division {row['division']:>3} n={row['n']}: "
            f"Avg Acc {100 * row[f'{acc}_mean']:.2f} +/- {100 * row[f'{acc}_std']:.2f} "
            f"(paper {100 * row[f'paper_{acc}']:.1f}), "
            f"F {100 * row[f'{forget}_mean']:.2f} +/- {100 * row[f'{forget}_std']:.2f} "
            f"(paper {100 * row[f'paper_{forget}']:.1f})"
        )
    for summary in summaries:
        parts = []
        for metric in METRICS:
            stats = summary[metric]
            parts.append(
                f"{metric.split('_')[-1]} {100 * stats['mean_diff']:+.2f} "
                f"[{100 * stats['ci95_low']:+.2f}, {100 * stats['ci95_high']:+.2f}] p={stats['sign_flip_p']:.3f}"
            )
        print(f"{summary['comparison']} (n={len(summary['seeds'])}): " + "; ".join(parts))


if __name__ == "__main__":
    main()
