from __future__ import annotations

from math import fsum


def _mean(values: list[float]) -> float:
    return fsum(values) / len(values) if values else 0.0


def summarize_accuracy_matrix(matrix: list[list[float | None]]) -> dict[str, object]:
    if not matrix:
        raise ValueError("accuracy matrix cannot be empty")
    task_count = len(matrix)
    if any(len(row) != task_count for row in matrix):
        raise ValueError("accuracy matrix must be square")

    average_accuracy: list[float] = []
    average_forgetting: list[float] = []
    for step, row in enumerate(matrix):
        learned = [float(row[task]) for task in range(step + 1) if row[task] is not None]
        if len(learned) != step + 1:
            raise ValueError("learned-task entries cannot be missing")
        average_accuracy.append(_mean(learned))

        forgetting: list[float] = []
        for task in range(step):
            previous = [
                float(matrix[past_step][task])
                for past_step in range(task, step)
                if matrix[past_step][task] is not None
            ]
            forgetting.append(max(previous) - float(row[task]))
        average_forgetting.append(_mean(forgetting))

    return {
        "average_accuracy": average_accuracy,
        "average_forgetting": average_forgetting,
        "final_average_accuracy": average_accuracy[-1],
        "final_average_forgetting": average_forgetting[-1],
    }

