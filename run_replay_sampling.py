"""Paired-seed comparison of uniform vs loss-prioritized replay sampling.

Both arms use the same CIFAR-100 class-incremental protocol, reservoir storage,
buffer size and replay batch size; only the draw policy differs. The protocol
is pre-registered in docs/REPLAY_SAMPLING_PLAN.md.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "tools"))

from analyze_tee_zhang_equal_budget import paired_stats  # noqa: E402

POLICIES = ("uniform", "loss")
METRICS = ("final_average_accuracy", "final_average_forgetting")


def run_dir(root: Path, policy: str, seed: int) -> Path:
    return root / policy / f"seed-{seed}"


def expected_config(args: argparse.Namespace, policy: str, seed: int) -> dict[str, object]:
    return {
        "dataset": "cifar100",
        "method": "replay",
        "model": "resnet18",
        "epochs_per_task": args.epochs_per_task,
        "buffer_size": args.buffer_size,
        "replay_batch_size": args.replay_batch_size,
        "replay_sampling": policy,
        "priority_alpha": args.priority_alpha,
        "seed": seed,
        "deterministic": True,
        "max_train_samples_per_task": None,
    }


def load_completed(path: Path, expected: dict[str, object]) -> dict | None:
    result_path = path / "results.json"
    if not result_path.exists():
        return None
    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    mismatches = {k: (result["config"].get(k), v) for k, v in expected.items() if result["config"].get(k) != v}
    if mismatches:
        raise ValueError(f"refusing to reuse mismatched run {result_path}: {mismatches}")
    return result


def summarize(root: Path, args: argparse.Namespace) -> dict[str, object]:
    arms = {
        policy: {
            seed: result
            for seed in args.seeds
            if (result := load_completed(run_dir(root, policy, seed), expected_config(args, policy, seed))) is not None
        }
        for policy in POLICIES
    }
    rows = []
    for policy, runs in arms.items():
        if not runs:
            continue
        row: dict[str, object] = {"policy": policy, "n": len(runs), "seeds": " ".join(map(str, sorted(runs)))}
        for metric in METRICS:
            values = [float(result["metrics"][metric]) for result in runs.values()]
            row[f"{metric}_mean"] = statistics.mean(values)
            row[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else 0.0
        row["elapsed_minutes_mean"] = statistics.mean(float(r["elapsed_seconds"]) / 60 for r in runs.values())
        rows.append(row)

    paired_seeds = sorted(set(arms["uniform"]) & set(arms["loss"]))
    paired: dict[str, object] = {"comparison": "loss - uniform", "seeds": paired_seeds}
    for metric in METRICS:
        diffs = [
            float(arms["loss"][seed]["metrics"][metric]) - float(arms["uniform"][seed]["metrics"][metric])
            for seed in paired_seeds
        ]
        if diffs:
            paired[metric] = paired_stats(diffs)

    summary = {"arms": rows, "paired": paired}
    root.mkdir(parents=True, exist_ok=True)
    with (root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    if rows:
        with (root / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/replay-sampling"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--priority-alpha", type=float, default=0.6)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    # Interleave policies within each seed so a partial run stays paired.
    for seed in args.seeds:
        for policy in args.policies:
            path = run_dir(args.output_root, policy, seed)
            if load_completed(path, expected_config(args, policy, seed)) is not None:
                print(f"skip completed policy={policy} seed={seed}", flush=True)
                continue
            command = [
                sys.executable, "train.py",
                "--dataset", "cifar100", "--method", "replay", "--model", "resnet18",
                "--epochs-per-task", str(args.epochs_per_task),
                "--buffer-size", str(args.buffer_size),
                "--replay-batch-size", str(args.replay_batch_size),
                "--replay-sampling", policy,
                "--priority-alpha", str(args.priority_alpha),
                "--seed", str(seed),
                "--output-dir", str(path),
            ]
            print("run " + " ".join(command), flush=True)
            if not args.dry_run:
                subprocess.run(command, check=True)
                summarize(args.output_root, args)

    summary = summarize(args.output_root, args)
    for row in summary["arms"]:
        print(
            f"{row['policy']:>7} n={row['n']}: final acc {100 * row['final_average_accuracy_mean']:.2f} "
            f"+/- {100 * row['final_average_accuracy_std']:.2f}, forgetting "
            f"{100 * row['final_average_forgetting_mean']:.2f} +/- {100 * row['final_average_forgetting_std']:.2f}"
        )
    paired = summary["paired"]
    for metric in METRICS:
        stats = paired.get(metric)
        if stats and "ci95_low" in stats:
            print(
                f"loss - uniform {metric}: {100 * stats['mean_diff']:+.2f} pp "
                f"[{100 * stats['ci95_low']:+.2f}, {100 * stats['ci95_high']:+.2f}], "
                f"sign-flip p={stats['sign_flip_p']:.3f}, positive {stats['positive_pairs']}/{stats['n']}"
            )


if __name__ == "__main__":
    main()
