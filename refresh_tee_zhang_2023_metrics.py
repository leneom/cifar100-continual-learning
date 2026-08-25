from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from continual_learning.reproductions.tee_zhang_2023 import paper_metrics


def refresh_result(result_path: Path) -> float:
    with result_path.open(encoding="utf-8") as handle:
        result = json.load(handle)
    result["metrics"].update(
        paper_metrics(
            result["seen_accuracy_trajectory"],
            result["task1_accuracy_trajectory"],
        )
    )

    json_temp = result_path.with_name(result_path.name + ".tmp")
    json_temp.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    checkpoint_path = result_path.parent / "checkpoint.pt"
    checkpoint_temp = checkpoint_path.with_name(checkpoint_path.name + ".tmp")
    if checkpoint_path.exists():
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        payload["result"] = result
        torch.save(payload, checkpoint_temp)

    if checkpoint_path.exists():
        os.replace(checkpoint_temp, checkpoint_path)
    os.replace(json_temp, result_path)
    return float(result["metrics"]["forgetfulness"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute saved Tee & Zhang paper metrics")
    parser.add_argument("--root", default="runs/tee-zhang-2023")
    args = parser.parse_args()
    root = Path(args.root)
    result_paths = sorted(root.rglob("results.json"))
    if not result_paths:
        raise FileNotFoundError(f"no results.json files found under {root}")
    for result_path in result_paths:
        forgetfulness = refresh_result(result_path)
        print(f"refreshed {result_path} F={forgetfulness:.8f}", flush=True)


if __name__ == "__main__":
    main()
