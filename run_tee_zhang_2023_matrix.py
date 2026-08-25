from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


PAPER_TABLE = {
    1: {"forgetfulness": 0.639, "continual_average_accuracy": 0.399},
    8: {"forgetfulness": 0.626, "continual_average_accuracy": 0.407},
    60: {"forgetfulness": 0.574, "continual_average_accuracy": 0.446},
    120: {"forgetfulness": 0.551, "continual_average_accuracy": 0.466},
    300: {"forgetfulness": 0.561, "continual_average_accuracy": 0.466},
}


def _run_dir(root: Path, implementation: str, division: int, seed: int) -> Path:
    return root / implementation / f"division-{division:03d}" / f"seed-{seed}"


def _load_completed(path: Path, implementation: str, division: int, seed: int) -> dict | None:
    result_path = path / "results.json"
    if not result_path.exists():
        return None
    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    config = result["config"]
    expected = {
        "sequence_implementation": implementation,
        "division": division,
        "seed": seed,
        "max_tasks": 20,
        "max_train_samples_per_class": None,
    }
    mismatches = {key: (config.get(key), value) for key, value in expected.items() if config.get(key) != value}
    if mismatches:
        raise ValueError(f"refusing to resume mismatched run {result_path}: {mismatches}")
    return result


def _aggregate(
    root: Path,
    implementation: str,
    divisions: list[int],
    seeds: list[int],
) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    for division in divisions:
        results = []
        completed_seeds = []
        for seed in seeds:
            result = _load_completed(_run_dir(root, implementation, division, seed), implementation, division, seed)
            if result is not None:
                results.append(result)
                completed_seeds.append(seed)
        if not results:
            continue
        accuracies = [float(result["metrics"]["continual_average_accuracy"]) for result in results]
        forgetfulness = [float(result["metrics"]["forgetfulness"]) for result in results]
        elapsed = [float(result["elapsed_seconds"]) for result in results]
        paper = PAPER_TABLE.get(division, {})
        rows.append(
            {
                "implementation": implementation,
                "division": division,
                "completed_seeds": " ".join(str(seed) for seed in completed_seeds),
                "n": len(results),
                "average_accuracy_mean": statistics.mean(accuracies),
                "average_accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "paper_average_accuracy": paper.get("continual_average_accuracy", float("nan")),
                "forgetfulness_mean": statistics.mean(forgetfulness),
                "forgetfulness_std": statistics.stdev(forgetfulness) if len(forgetfulness) > 1 else 0.0,
                "paper_forgetfulness": paper.get("forgetfulness", float("nan")),
                "elapsed_seconds_mean": statistics.mean(elapsed),
            }
        )

    summary_dir = root / implementation
    summary_dir.mkdir(parents=True, exist_ok=True)
    with (summary_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2, ensure_ascii=False)
    if rows:
        with (summary_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and resume Tee & Zhang reproduction matrix")
    parser.add_argument("--output-root", default="runs/tee-zhang-2023")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--sequence-implementation", choices=("released-code", "equal-budget"), default="released-code")
    parser.add_argument("--divisions", type=int, nargs="+", default=[1, 120, 300])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--image-size", type=int, default=74)
    parser.add_argument("--max-epochs-per-task", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--force", action="store_true", help="rerun completed matching configurations")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    output_root = (project_root / args.output_root).resolve()
    data_root = (project_root / args.data_root).resolve()
    env = os.environ.copy()
    env.setdefault("TORCH_HOME", str(project_root / ".torch-cache"))

    for division in args.divisions:
        for seed in args.seeds:
            run_dir = _run_dir(output_root, args.sequence_implementation, division, seed)
            completed = _load_completed(
                run_dir,
                args.sequence_implementation,
                division,
                seed,
            )
            if completed is not None and not args.force:
                print(f"skip completed division={division} seed={seed}", flush=True)
                continue
            command = [
                sys.executable,
                str(project_root / "reproduce_tee_zhang_2023.py"),
                "--data-root",
                str(data_root),
                "--output-dir",
                str(run_dir),
                "--seed",
                str(seed),
                "--division",
                str(division),
                "--sequence-implementation",
                args.sequence_implementation,
                "--image-size",
                str(args.image_size),
                "--max-epochs-per-task",
                str(args.max_epochs_per_task),
                "--max-tasks",
                "20",
                "--num-workers",
                str(args.num_workers),
                "--device",
                args.device,
            ]
            print("run " + " ".join(command), flush=True)
            if args.dry_run:
                continue
            subprocess.run(command, cwd=project_root, env=env, check=True)
            rows = _aggregate(
                output_root,
                args.sequence_implementation,
                args.divisions,
                args.seeds,
            )
            print(f"updated summary with {len(rows)} division rows", flush=True)

    rows = _aggregate(
        output_root,
        args.sequence_implementation,
        args.divisions,
        args.seeds,
    )
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
