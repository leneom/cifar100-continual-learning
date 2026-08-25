from __future__ import annotations

import argparse
import copy
import csv
import json
import platform
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Sequence

import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from continual_learning.cifair import CiFAIR100
from continual_learning.experiment import resolve_device, set_seed
from continual_learning.metrics import summarize_accuracy_matrix


SequenceImplementation = Literal["released-code", "equal-budget"]


@dataclass
class TeeZhangConfig:
    data_root: str
    output_dir: str
    seed: int
    division: int
    sequence_implementation: SequenceImplementation
    class_order_seed: int
    buffer_size: int
    image_size: int
    batch_size: int
    learning_rate: float
    momentum: float
    patience: int
    max_epochs_per_task: int
    max_tasks: int
    max_train_samples_per_class: int | None
    num_workers: int
    device: str
    deterministic: bool
    pretrained: bool
    amp: bool
    download: bool


@dataclass(frozen=True)
class InterleavePlan:
    indices: list[int]
    source_is_replay: list[bool]
    original_current_count: int
    original_replay_count: int
    presented_current_count: int
    presented_replay_count: int


@dataclass
class CiFAIR100ProtocolData:
    train_dataset: Dataset
    test_dataset: Dataset
    class_order: list[int]
    label_map: dict[int, int]
    train_indices_by_label: dict[int, list[int]]
    val_indices_by_label: dict[int, list[int]]
    test_indices_by_label: dict[int, list[int]]


class RemappedIndexDataset(Dataset):
    """Dataset view over base indices with labels remapped to stream order."""

    def __init__(self, dataset: Dataset, indices: Sequence[int], label_map: dict[int, int]):
        self.dataset = dataset
        self.indices = list(indices)
        self.label_map = label_map

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        image, original_label = self.dataset[self.indices[index]]
        return image, self.label_map[int(original_label)]


class ExpandableMobileNetV3Small(nn.Module):
    """MobileNetV3-Small whose classifier grows by five classes per task."""

    def __init__(self, initial_classes: int = 5, pretrained: bool = True):
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        model = mobilenet_v3_small(weights=weights)
        self.features = model.features
        self.avgpool = model.avgpool
        self.classifier = model.classifier
        final = self.classifier[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("unexpected MobileNetV3 classifier layout")
        self.classifier[-1] = nn.Linear(final.in_features, initial_classes)

    @property
    def num_classes(self) -> int:
        final = self.classifier[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("unexpected MobileNetV3 classifier layout")
        return final.out_features

    def expand(self, new_classes: int = 5) -> None:
        if new_classes <= 0:
            raise ValueError("new_classes must be positive")
        old = self.classifier[-1]
        if not isinstance(old, nn.Linear):
            raise TypeError("unexpected MobileNetV3 classifier layout")

        # This deliberately mirrors the released code: construct a full larger
        # layer, preserve all old rows, and take the first new_classes randomly
        # initialized rows for the newly introduced labels.
        expanded = nn.Linear(old.in_features, old.out_features + new_classes).to(
            device=old.weight.device,
            dtype=old.weight.dtype,
        )
        with torch.no_grad():
            new_weight = expanded.weight[:new_classes].clone()
            new_bias = expanded.bias[:new_classes].clone()
            expanded.weight[: old.out_features].copy_(old.weight)
            expanded.bias[: old.out_features].copy_(old.bias)
            expanded.weight[old.out_features :].copy_(new_weight)
            expanded.bias[old.out_features :].copy_(new_bias)
        self.classifier[-1] = expanded

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.features(inputs)
        features = self.avgpool(features)
        return self.classifier(torch.flatten(features, 1))


class ClassBalancedReplayBuffer:
    """Released-paper buffer policy: equal quota for every seen class."""

    def __init__(self, capacity: int, rng: random.Random):
        if capacity < 0:
            raise ValueError("capacity cannot be negative")
        self.capacity = capacity
        self.rng = rng
        self.indices_by_label: dict[int, list[int]] = {}

    def __len__(self) -> int:
        return sum(len(indices) for indices in self.indices_by_label.values())

    def flattened_indices(self) -> list[int]:
        return [index for indices in self.indices_by_label.values() for index in indices]

    def class_counts(self) -> dict[int, int]:
        return {label: len(indices) for label, indices in self.indices_by_label.items()}

    def update(self, new_indices_by_label: dict[int, list[int]], seen_class_count: int) -> None:
        if seen_class_count <= 0:
            raise ValueError("seen_class_count must be positive")
        for label, indices in new_indices_by_label.items():
            if label in self.indices_by_label:
                self.indices_by_label[label].extend(indices)
            else:
                self.indices_by_label[label] = list(indices)

        quota, remainder = divmod(self.capacity, seen_class_count)
        for position, (label, indices) in enumerate(self.indices_by_label.items()):
            target = quota + (1 if position < remainder else 0)
            target = min(target, len(indices))
            self.indices_by_label[label] = self.rng.sample(indices, target)


def paper_class_order(num_classes: int = 100, seed: int = 100) -> list[int]:
    order = list(range(num_classes))
    random.Random(seed).shuffle(order)
    return order


def _balanced_chunks(items: Sequence[int], divisions: int) -> list[list[int]]:
    if divisions <= 0:
        raise ValueError("division must be positive")
    quotient, remainder = divmod(len(items), divisions)
    chunks: list[list[int]] = []
    start = 0
    for chunk_index in range(divisions):
        size = quotient + (1 if chunk_index < remainder else 0)
        chunks.append(list(items[start : start + size]))
        start += size
    if start != len(items):
        raise AssertionError("balanced chunking did not consume every item")
    return chunks


def build_interleave_plan(
    current_indices: Sequence[int],
    replay_indices: Sequence[int],
    division: int,
    rng: random.Random,
    implementation: SequenceImplementation,
) -> InterleavePlan:
    if division <= 0:
        raise ValueError("division must be positive")
    if implementation not in ("released-code", "equal-budget"):
        raise ValueError(f"unknown sequence implementation: {implementation}")

    current = rng.sample(list(current_indices), len(current_indices))
    replay = rng.sample(list(replay_indices), len(replay_indices))
    combined: list[int] = []
    sources: list[bool] = []

    if implementation == "equal-budget":
        current_chunks = _balanced_chunks(current, division)
        replay_chunks = _balanced_chunks(replay, division)
        for current_chunk, replay_chunk in zip(current_chunks, replay_chunks):
            combined.extend(current_chunk)
            sources.extend([False] * len(current_chunk))
            combined.extend(replay_chunk)
            sources.extend([True] * len(replay_chunk))
    else:
        # Exact structure of ciFAIR-100/Code/VaryDiv.py lines 115-134.
        # The final current-tail append duplicates the already distributed
        # remainder; retaining it makes the released implementation auditable.
        current_per_division, current_remainder = divmod(len(current), division)
        replay_per_division = len(replay) // division
        current_end = 0
        for chunk_index in range(division):
            current_size = current_per_division + (1 if chunk_index < current_remainder else 0)
            current_chunk = current[current_end : current_end + current_size]
            current_end += current_size
            replay_chunk = replay[
                chunk_index * replay_per_division : (chunk_index + 1) * replay_per_division
            ]
            combined.extend(current_chunk)
            sources.extend([False] * len(current_chunk))
            combined.extend(replay_chunk)
            sources.extend([True] * len(replay_chunk))

        duplicate_current_tail = current[division * current_per_division :]
        replay_tail = replay[division * replay_per_division :]
        combined.extend(duplicate_current_tail)
        sources.extend([False] * len(duplicate_current_tail))
        combined.extend(replay_tail)
        sources.extend([True] * len(replay_tail))

    presented_current = sum(not source for source in sources)
    presented_replay = sum(sources)
    return InterleavePlan(
        indices=combined,
        source_is_replay=sources,
        original_current_count=len(current_indices),
        original_replay_count=len(replay_indices),
        presented_current_count=presented_current,
        presented_replay_count=presented_replay,
    )


def paper_metrics(seen_accuracy: Sequence[float], task1_accuracy: Sequence[float]) -> dict[str, float]:
    if not seen_accuracy or len(seen_accuracy) != len(task1_accuracy):
        raise ValueError("accuracy trajectories must be non-empty and equally sized")
    task1_reference = float(task1_accuracy[0])
    if task1_reference <= 0:
        raise ValueError("task-1 reference accuracy must be positive")
    # The authors' released varydiv.ipynb computes a relative decrease and
    # averages rows 2..20, explicitly excluding the zero-valued Task-1 row.
    relative_forgetfulness = [
        (task1_reference - float(accuracy)) / task1_reference
        for accuracy in task1_accuracy[1:]
    ]
    return {
        "continual_average_accuracy": sum(float(value) for value in seen_accuracy) / len(seen_accuracy),
        "forgetfulness": (
            sum(relative_forgetfulness) / len(relative_forgetfulness)
            if relative_forgetfulness
            else 0.0
        ),
        "task1_reference_accuracy": task1_reference,
    }


def _indices_by_original_label(dataset: Dataset, num_classes: int = 100) -> dict[int, list[int]]:
    targets = getattr(dataset, "targets", None)
    if targets is None:
        raise ValueError("ciFAIR dataset must expose targets")
    result = {label: [] for label in range(num_classes)}
    for index, label in enumerate(targets):
        result[int(label)].append(index)
    expected = {500} if len(dataset) == 50_000 else {100}
    observed = {len(indices) for indices in result.values()}
    if observed != expected:
        raise ValueError(f"unexpected per-class counts: {sorted(observed)}")
    return result


def build_protocol_data(config: TeeZhangConfig) -> CiFAIR100ProtocolData:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            # torchvision 0.13 did not antialias tensor resize by default.
            transforms.Resize(
                (config.image_size, config.image_size),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=False,
            ),
        ]
    )
    train = CiFAIR100(config.data_root, train=True, download=config.download, transform=transform)
    test = CiFAIR100(config.data_root, train=False, download=config.download, transform=transform)
    class_order = paper_class_order(seed=config.class_order_seed)
    label_map = {original_label: stream_label for stream_label, original_label in enumerate(class_order)}
    train_by_original = _indices_by_original_label(train)
    test_by_original = _indices_by_original_label(test)

    train_indices_by_label: dict[int, list[int]] = {}
    val_indices_by_label: dict[int, list[int]] = {}
    test_indices_by_label: dict[int, list[int]] = {}
    for stream_label, original_label in enumerate(class_order):
        all_train_indices = train_by_original[original_label]
        task_train_indices = all_train_indices[:-50]
        if config.max_train_samples_per_class is not None:
            if config.max_train_samples_per_class <= 0:
                raise ValueError("max_train_samples_per_class must be positive")
            task_train_indices = task_train_indices[: config.max_train_samples_per_class]
        train_indices_by_label[stream_label] = task_train_indices
        val_indices_by_label[stream_label] = all_train_indices[-50:]
        test_indices_by_label[stream_label] = test_by_original[original_label]

    return CiFAIR100ProtocolData(
        train_dataset=train,
        test_dataset=test,
        class_order=class_order,
        label_map=label_map,
        train_indices_by_label=train_indices_by_label,
        val_indices_by_label=val_indices_by_label,
        test_indices_by_label=test_indices_by_label,
    )


def _flatten_label_range(indices_by_label: dict[int, list[int]], start: int, end: int) -> list[int]:
    return [index for label in range(start, end) for index in indices_by_label[label]]


def _make_loader(
    dataset: Dataset,
    batch_size: int,
    num_workers: int,
    device: torch.device,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )


def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("training loss became non-finite")
        loss.backward()
        optimizer.step()
        loss_sum += loss.detach().item() * labels.size(0)
        correct += (logits.detach().argmax(1) == labels).sum().item()
        count += labels.size(0)
    return loss_sum / count, correct / count


@torch.inference_mode()
def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    amp: bool,
) -> tuple[float, float]:
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(inputs)
            loss = nn.functional.cross_entropy(logits, labels)
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    if count == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return loss_sum / count, correct / count


def _save_result(output_dir: Path, result: dict[str, object], model: nn.Module) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    task_logs = result["task_logs"]
    with (output_dir / "task_trajectory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "task",
                "seen_accuracy",
                "task1_accuracy",
                "epochs_run",
                "best_validation_loss",
                "buffer_before",
                "buffer_after",
                "current_original",
                "current_presented",
                "replay_original",
                "replay_presented",
            ],
        )
        writer.writeheader()
        for task_log in task_logs:
            writer.writerow({key: task_log[key] for key in writer.fieldnames})

    matrix = result["accuracy_matrix"]
    with (output_dir / "accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task", *[f"task_{index + 1}" for index in range(len(matrix))]])
        for task_index, row in enumerate(matrix):
            writer.writerow(
                [task_index + 1, *["" if value is None else f"{value:.6f}" for value in row]]
            )

    with (output_dir / "epoch_log.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["task", "epoch", "train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        )
        writer.writeheader()
        writer.writerows(result["epoch_logs"])

    torch.save({"model_state_dict": model.state_dict(), "result": result}, output_dir / "checkpoint.pt")


def _validate_config(config: TeeZhangConfig) -> None:
    if config.division <= 0:
        raise ValueError("division must be positive")
    if config.buffer_size < 0:
        raise ValueError("buffer_size cannot be negative")
    if not 1 <= config.max_tasks <= 20:
        raise ValueError("max_tasks must be between 1 and 20")
    if config.max_epochs_per_task <= 0:
        raise ValueError("max_epochs_per_task must be positive")
    if config.patience <= 0:
        raise ValueError("patience must be positive")


def run_reproduction(config: TeeZhangConfig) -> dict[str, object]:
    _validate_config(config)
    set_seed(config.seed, config.deterministic)
    device = resolve_device(config.device)
    data = build_protocol_data(config)
    rng = random.Random(config.seed)
    model = ExpandableMobileNetV3Small(initial_classes=5, pretrained=config.pretrained).to(device)
    buffer = ClassBalancedReplayBuffer(config.buffer_size, rng)
    task_count = config.max_tasks
    accuracy_matrix: list[list[float | None]] = [[None] * task_count for _ in range(task_count)]
    seen_accuracy_trajectory: list[float] = []
    task1_accuracy_trajectory: list[float] = []
    task_logs: list[dict[str, object]] = []
    epoch_logs: list[dict[str, float | int]] = []
    started = time.perf_counter()

    print(
        f"device={device} division={config.division} implementation={config.sequence_implementation} "
        f"tasks={task_count} image_size={config.image_size}",
        flush=True,
    )

    for task_index in range(task_count):
        first_label = task_index * 5
        next_label = first_label + 5
        current_indices = _flatten_label_range(data.train_indices_by_label, first_label, next_label)
        buffer_before = len(buffer)
        plan = build_interleave_plan(
            current_indices=current_indices,
            replay_indices=buffer.flattened_indices(),
            division=config.division,
            rng=rng,
            implementation=config.sequence_implementation,
        )
        train_dataset = RemappedIndexDataset(data.train_dataset, plan.indices, data.label_map)
        val_indices = _flatten_label_range(data.val_indices_by_label, first_label, next_label)
        val_dataset = RemappedIndexDataset(data.train_dataset, val_indices, data.label_map)
        train_loader = _make_loader(
            train_dataset, config.batch_size, config.num_workers, device
        )
        val_loader = _make_loader(val_dataset, config.batch_size, config.num_workers, device)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
        )

        best_validation_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None
        epochs_without_improvement = 0
        epochs_run = 0
        for epoch_index in range(config.max_epochs_per_task):
            train_loss, train_accuracy = _train_epoch(
                model, train_loader, optimizer, device, config.amp
            )
            val_loss, val_accuracy = _evaluate(model, val_loader, device, config.amp)
            epochs_run = epoch_index + 1
            epoch_logs.append(
                {
                    "task": task_index + 1,
                    "epoch": epochs_run,
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy,
                }
            )
            print(
                f"task={task_index + 1}/{task_count} epoch={epochs_run} "
                f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}",
                flush=True,
            )

            if val_loss < best_validation_loss:
                best_validation_loss = val_loss
                best_state = copy.deepcopy(model.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= config.patience:
                    break

        if best_state is None:
            raise RuntimeError("no validation checkpoint was created")
        model.load_state_dict(best_state)

        seen_test_indices = _flatten_label_range(data.test_indices_by_label, 0, next_label)
        seen_test_dataset = RemappedIndexDataset(data.test_dataset, seen_test_indices, data.label_map)
        _, seen_accuracy = _evaluate(
            model,
            _make_loader(seen_test_dataset, config.batch_size, config.num_workers, device),
            device,
            config.amp,
        )
        task1_test_indices = _flatten_label_range(data.test_indices_by_label, 0, 5)
        task1_test_dataset = RemappedIndexDataset(
            data.test_dataset, task1_test_indices, data.label_map
        )
        _, task1_accuracy = _evaluate(
            model,
            _make_loader(task1_test_dataset, config.batch_size, config.num_workers, device),
            device,
            config.amp,
        )

        for eval_task_index in range(task_index + 1):
            eval_start = eval_task_index * 5
            eval_indices = _flatten_label_range(
                data.test_indices_by_label, eval_start, eval_start + 5
            )
            eval_dataset = RemappedIndexDataset(data.test_dataset, eval_indices, data.label_map)
            _, task_accuracy = _evaluate(
                model,
                _make_loader(eval_dataset, config.batch_size, config.num_workers, device),
                device,
                config.amp,
            )
            accuracy_matrix[task_index][eval_task_index] = task_accuracy

        new_indices_by_label = {
            label: list(data.train_indices_by_label[label]) for label in range(first_label, next_label)
        }
        buffer.update(new_indices_by_label, seen_class_count=next_label)
        buffer_after = len(buffer)
        seen_accuracy_trajectory.append(seen_accuracy)
        task1_accuracy_trajectory.append(task1_accuracy)
        task_logs.append(
            {
                "task": task_index + 1,
                "stream_labels": list(range(first_label, next_label)),
                "original_class_ids": data.class_order[first_label:next_label],
                "seen_accuracy": seen_accuracy,
                "task1_accuracy": task1_accuracy,
                "epochs_run": epochs_run,
                "best_validation_loss": best_validation_loss,
                "stopped_by_patience": epochs_without_improvement >= config.patience,
                "buffer_before": buffer_before,
                "buffer_after": buffer_after,
                "buffer_class_counts": buffer.class_counts(),
                "current_original": plan.original_current_count,
                "current_presented": plan.presented_current_count,
                "replay_original": plan.original_replay_count,
                "replay_presented": plan.presented_replay_count,
            }
        )
        print(
            f"eval task={task_index + 1} seen_acc={seen_accuracy:.4f} "
            f"task1_acc={task1_accuracy:.4f} buffer={buffer_after} "
            f"presented_current={plan.presented_current_count}/{plan.original_current_count} "
            f"presented_replay={plan.presented_replay_count}/{plan.original_replay_count}",
            flush=True,
        )

        if task_index + 1 < task_count:
            model.expand(5)

    metrics = paper_metrics(seen_accuracy_trajectory, task1_accuracy_trajectory)
    metrics.update(summarize_accuracy_matrix(accuracy_matrix))
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "paper": {
            "title": "Integrating Curricula with Replays: Its Effects on Continual Learning",
            "authors": ["Ren Jie Tee", "Mengmi Zhang"],
            "target": "Table 1 ciFAIR-100 interleave divisions",
        },
        "config": asdict(config),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "device": str(device),
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "class_order": data.class_order,
        "seen_accuracy_trajectory": seen_accuracy_trajectory,
        "task1_accuracy_trajectory": task1_accuracy_trajectory,
        "accuracy_matrix": accuracy_matrix,
        "metrics": metrics,
        "task_logs": task_logs,
        "epoch_logs": epoch_logs,
        "elapsed_seconds": elapsed,
    }
    _save_result(Path(config.output_dir), result, model)
    print(
        f"done paper_avg_acc={metrics['continual_average_accuracy']:.4f} "
        f"paper_F={metrics['forgetfulness']:.4f} elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Tee & Zhang (2023) ciFAIR-100 interleave divisions"
    )
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs/tee-zhang-2023/smoke")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--division", type=int, default=1)
    parser.add_argument(
        "--sequence-implementation",
        choices=("released-code", "equal-budget"),
        default="released-code",
    )
    parser.add_argument("--class-order-seed", type=int, default=100)
    parser.add_argument("--buffer-size", type=int, default=1200)
    parser.add_argument("--image-size", type=int, default=74)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max-epochs-per-task", type=int, default=100)
    parser.add_argument("--max-tasks", type=int, default=20)
    parser.add_argument("--max-train-samples-per-class", type=int)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--download", action="store_true")
    return parser


def config_from_args(args: argparse.Namespace) -> TeeZhangConfig:
    return TeeZhangConfig(**vars(args))


def main() -> None:
    args = make_parser().parse_args()
    run_reproduction(config_from_args(args))


if __name__ == "__main__":
    main()
