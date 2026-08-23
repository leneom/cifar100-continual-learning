from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
import subprocess
import sys
from pathlib import Path


COMPARISON_FIELDS = (
    "dataset",
    "method",
    "model",
    "num_classes",
    "classes_per_task",
    "epochs_per_task",
    "batch_size",
    "learning_rate",
    "momentum",
    "weight_decay",
    "replay_batch_size",
    "seed",
    "buffer_size",
    "deterministic",
    "max_train_samples_per_task",
)


def read_result(path: Path) -> dict[str, object] | None:
    try:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None
    if "config" not in result or "metrics" not in result:
        return None
    result["_path"] = str(path.resolve())
    return result


def discover_results(root: Path) -> list[dict[str, object]]:
    results = []
    if not root.exists():
        return results
    for path in root.rglob("results.json"):
        result = read_result(path)
        if result is not None:
            results.append(result)
    return results


def matches(result: dict[str, object], expected: dict[str, object]) -> bool:
    config = result["config"]
    if not isinstance(config, dict):
        return False
    return all(config.get(field) == expected[field] for field in COMPARISON_FIELDS)


def summarize_records(results: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[int, list[dict[str, object]]] = {}
    for result in results:
        config = result["config"]
        if not isinstance(config, dict):
            continue
        grouped.setdefault(int(config["buffer_size"]), []).append(result)

    summary = []
    for buffer_size, group in sorted(grouped.items()):
        accuracies = [float(item["metrics"]["final_average_accuracy"]) for item in group]
        forgetting = [float(item["metrics"]["final_average_forgetting"]) for item in group]
        seeds = sorted(int(item["config"]["seed"]) for item in group)
        summary.append(
            {
                "buffer_size": buffer_size,
                "seeds": seeds,
                "runs": len(group),
                "accuracy_mean": statistics.mean(accuracies),
                "accuracy_std": statistics.stdev(accuracies) if len(accuracies) > 1 else 0.0,
                "forgetting_mean": statistics.mean(forgetting),
                "forgetting_std": statistics.stdev(forgetting) if len(forgetting) > 1 else 0.0,
            }
        )
    return summary


def write_summary_csv(summary: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary[0].keys())
        writer.writeheader()
        writer.writerows(summary)


def _chart_points(
    summary: list[dict[str, object]],
    key: str,
    x: int,
    y: int,
    width: int,
    height: int,
    y_max: float,
) -> list[tuple[float, float]]:
    logs = [math.log10(int(row["buffer_size"])) for row in summary]
    minimum, maximum = min(logs), max(logs)
    span = maximum - minimum or 1.0
    return [
        (
            x + (log_value - minimum) / span * width,
            y + height - float(row[key]) / y_max * height,
        )
        for row, log_value in zip(summary, logs)
    ]


def _svg_chart(
    summary: list[dict[str, object]],
    key: str,
    std_key: str,
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
    y_max: float,
) -> list[str]:
    points = _chart_points(summary, key, x, y, width, height, y_max)
    elements = [
        f'<text x="{x}" y="{y - 22}" font-size="19" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#475569"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#475569"/>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        tick_y = y + height - height * tick / 5
        elements.append(
            f'<line x1="{x}" y1="{tick_y:.1f}" x2="{x + width}" y2="{tick_y:.1f}" stroke="#e2e8f0"/>'
        )
        elements.append(
            f'<text x="{x - 9}" y="{tick_y + 4:.1f}" text-anchor="end" font-size="11">{value:.2f}</text>'
        )
    elements.append(
        '<polyline points="'
        + " ".join(f"{point_x:.1f},{point_y:.1f}" for point_x, point_y in points)
        + '" fill="none" stroke="#2563eb" stroke-width="3"/>'
    )
    for row, (point_x, point_y) in zip(summary, points):
        error = float(row[std_key])
        if error > 0:
            mean = float(row[key])
            top_y = y + height - min(y_max, mean + error) / y_max * height
            bottom_y = y + height - max(0.0, mean - error) / y_max * height
            elements.extend(
                (
                    f'<line class="error-bar" x1="{point_x:.1f}" y1="{top_y:.1f}" '
                    f'x2="{point_x:.1f}" y2="{bottom_y:.1f}" stroke="#1d4ed8" stroke-width="1.5"/>',
                    f'<line class="error-bar" x1="{point_x - 6:.1f}" y1="{top_y:.1f}" '
                    f'x2="{point_x + 6:.1f}" y2="{top_y:.1f}" stroke="#1d4ed8" stroke-width="1.5"/>',
                    f'<line class="error-bar" x1="{point_x - 6:.1f}" y1="{bottom_y:.1f}" '
                    f'x2="{point_x + 6:.1f}" y2="{bottom_y:.1f}" stroke="#1d4ed8" stroke-width="1.5"/>',
                )
            )
        elements.append(f'<circle cx="{point_x:.1f}" cy="{point_y:.1f}" r="5" fill="#2563eb"/>')
        elements.append(
            f'<text x="{point_x:.1f}" y="{y + height + 22}" text-anchor="middle" font-size="11">'
            f'{int(row["buffer_size"])}</text>'
        )
        elements.append(
            f'<text x="{point_x:.1f}" y="{point_y - 10:.1f}" text-anchor="middle" font-size="11" '
            f'font-weight="600">{float(row[key]) * 100:.1f}% ± {error * 100:.1f}</text>'
        )
    elements.append(
        f'<text x="{x + width / 2:.1f}" y="{y + height + 48}" text-anchor="middle" font-size="12">Replay buffer size (log scale)</text>'
    )
    return elements


def write_summary_svg(summary: list[dict[str, object]], path: Path) -> None:
    width, height = 1140, 500
    forgetting_max = max(float(row["forgetting_mean"]) for row in summary)
    forgetting_axis = max(0.2, min(1.0, math.ceil((forgetting_max + 0.05) * 10) / 10))
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#0f172a">',
        '<text x="570" y="30" text-anchor="middle" font-size="22" font-weight="700">CIFAR-100 Replay Buffer Ablation</text>',
    ]
    elements.append(
        '<text x="570" y="55" text-anchor="middle" font-size="12" fill="#475569">'
        'Points show means; error bars show sample standard deviation</text>'
    )
    elements.extend(
        _svg_chart(
            summary,
            "accuracy_mean",
            "accuracy_std",
            "Final average accuracy",
            75,
            95,
            430,
            315,
            1.0,
        )
    )
    elements.extend(
        _svg_chart(
            summary,
            "forgetting_mean",
            "forgetting_std",
            "Final average forgetting",
            650,
            95,
            430,
            315,
            forgetting_axis,
        )
    )
    elements.extend(("</g>", "</svg>"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(elements), encoding="utf-8")


def expected_config(args: argparse.Namespace, buffer_size: int, seed: int) -> dict[str, object]:
    return {
        "dataset": "cifar100",
        "method": "replay",
        "model": "resnet18",
        "num_classes": 100,
        "classes_per_task": 10,
        "epochs_per_task": args.epochs_per_task,
        "batch_size": args.batch_size,
        "learning_rate": 0.1,
        "momentum": 0.9,
        "weight_decay": 5e-4,
        "replay_batch_size": args.replay_batch_size,
        "seed": seed,
        "buffer_size": buffer_size,
        "deterministic": True,
        "max_train_samples_per_task": None,
    }


def run_one(args: argparse.Namespace, buffer_size: int, seed: int, output_dir: Path) -> None:
    command = [
        sys.executable,
        "train.py",
        "--dataset",
        "cifar100",
        "--method",
        "replay",
        "--epochs-per-task",
        str(args.epochs_per_task),
        "--batch-size",
        str(args.batch_size),
        "--buffer-size",
        str(buffer_size),
        "--replay-batch-size",
        str(args.replay_batch_size),
        "--seed",
        str(seed),
        "--model",
        "resnet18",
        "--output-dir",
        str(output_dir),
    ]
    print("running:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run and summarize the Replay buffer-size ablation")
    parser.add_argument("--buffer-sizes", type=int, nargs="+", default=[200, 500, 1000, 2000, 5000])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42])
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--runs-root", type=Path, default=Path("runs"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/replay-buffer-ablation"))
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    if any(size <= 0 for size in args.buffer_sizes):
        raise ValueError("buffer sizes must be positive")
    selected: list[dict[str, object]] = []

    for seed in args.seeds:
        for buffer_size in args.buffer_sizes:
            expected = expected_config(args, buffer_size, seed)
            existing = next((item for item in discover_results(args.runs_root) if matches(item, expected)), None)
            if existing is not None:
                print(f"reuse buffer={buffer_size} seed={seed}: {existing['_path']}", flush=True)
                selected.append(existing)
                continue
            output_dir = args.output_dir / f"buffer-{buffer_size}-seed-{seed}"
            if args.dry_run:
                print(f"would run buffer={buffer_size} seed={seed} -> {output_dir}")
                continue
            run_one(args, buffer_size, seed, output_dir)
            result = read_result(output_dir / "results.json")
            if result is None or not matches(result, expected):
                raise RuntimeError(f"completed run did not produce the expected result: {output_dir}")
            selected.append(result)

    if args.dry_run:
        return
    summary = summarize_records(selected)
    if not summary:
        raise RuntimeError("no matching results were collected")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_summary_csv(summary, args.output_dir / "summary.csv")
    write_summary_svg(summary, args.output_dir / "buffer-size.svg")
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump({"summary": summary, "results": selected}, handle, indent=2, ensure_ascii=False)
    print(args.output_dir / "summary.csv")
    print(args.output_dir / "buffer-size.svg")


if __name__ == "__main__":
    main()
