from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


COLORS = ("#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2")


def load_series(paths: list[Path]) -> list[dict[str, object]]:
    series = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            result = json.load(handle)
        method = result["config"]["method"]
        if method == "replay":
            method = f"replay-{result['config']['buffer_size']}"
        series.append(
            {
                "label": method,
                "accuracy": result["metrics"]["average_accuracy"],
                "forgetting": result["metrics"]["average_forgetting"],
            }
        )
    return series


def points(values: list[float], x: int, y: int, width: int, height: int, y_max: float) -> str:
    x_step = width / max(1, len(values) - 1)
    return " ".join(
        f"{x + index * x_step:.1f},{y + height - value / y_max * height:.1f}"
        for index, value in enumerate(values)
    )


def chart(
    series: list[dict[str, object]],
    key: str,
    title: str,
    x: int,
    y: int,
    width: int,
    height: int,
    y_max: float,
) -> list[str]:
    elements = [
        f'<text x="{x}" y="{y - 20}" font-size="18" font-weight="600">{html.escape(title)}</text>',
        f'<line x1="{x}" y1="{y + height}" x2="{x + width}" y2="{y + height}" stroke="#475569"/>',
        f'<line x1="{x}" y1="{y}" x2="{x}" y2="{y + height}" stroke="#475569"/>',
    ]
    for tick in range(6):
        value = y_max * tick / 5
        tick_y = y + height - height * tick / 5
        elements.append(f'<line x1="{x}" y1="{tick_y:.1f}" x2="{x + width}" y2="{tick_y:.1f}" stroke="#e2e8f0"/>')
        elements.append(f'<text x="{x - 8}" y="{tick_y + 4:.1f}" text-anchor="end" font-size="11">{value:.1f}</text>')
    task_count = max(len(item[key]) for item in series)
    for index in range(task_count):
        tick_x = x + index * width / max(1, task_count - 1)
        elements.append(f'<text x="{tick_x:.1f}" y="{y + height + 20}" text-anchor="middle" font-size="11">{index + 1}</text>')
    for index, item in enumerate(series):
        color = COLORS[index % len(COLORS)]
        value_list = [float(value) for value in item[key]]
        elements.append(
            f'<polyline points="{points(value_list, x, y, width, height, y_max)}" fill="none" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{x + 12 + (index % 3) * 145}" y="{y + 18 + (index // 3) * 20}" fill="{color}" font-size="12">'
            f'{html.escape(str(item["label"]))}</text>'
        )
    elements.append(f'<text x="{x + width / 2:.1f}" y="{y + height + 42}" text-anchor="middle" font-size="12">Tasks learned</text>')
    return elements


def write_svg(series: list[dict[str, object]], output: Path) -> None:
    if not series:
        raise ValueError("at least one result file is required")
    width, height = 1120, 470
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<g font-family="Arial, sans-serif" fill="#0f172a">',
    ]
    elements.extend(chart(series, "accuracy", "Average accuracy", 70, 70, 430, 310, 1.0))
    max_forgetting = max(max(float(v) for v in item["forgetting"]) for item in series)
    forgetting_axis = max(0.2, min(1.0, (int(max_forgetting * 10) + 2) / 10))
    elements.extend(chart(series, "forgetting", "Average forgetting", 630, 70, 430, 310, forgetting_axis))
    elements.extend(("</g>", "</svg>"))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(elements), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot continual-learning result files to SVG")
    parser.add_argument("results", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("runs/comparison.svg"))
    args = parser.parse_args()
    write_svg(load_series(args.results), args.output)
    print(args.output)


if __name__ == "__main__":
    main()

