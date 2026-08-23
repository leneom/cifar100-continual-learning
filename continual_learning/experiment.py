from __future__ import annotations

import argparse
import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from .buffer import ReplayBuffer
from .data import TaskStream, build_cifar100_stream, build_synthetic_stream
from .ewc import OnlineEWC
from .metrics import summarize_accuracy_matrix
from .model import build_model


@dataclass
class ExperimentConfig:
    dataset: str
    data_root: str
    output_dir: str
    method: str
    model: str
    num_classes: int
    classes_per_task: int
    epochs_per_task: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    seed: int
    deterministic: bool
    num_workers: int
    device: str
    download: bool
    buffer_size: int
    replay_batch_size: int
    ewc_lambda: float
    ewc_decay: float
    fisher_samples: int
    train_samples_per_class: int
    test_samples_per_class: int
    max_train_samples_per_task: int | None


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.use_deterministic_algorithms(True)


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    seed: int,
    device: torch.device,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def limit_dataset(dataset: Dataset, limit: int | None, seed: int) -> Dataset:
    if limit is None or limit >= len(dataset):
        return dataset
    if limit <= 0:
        raise ValueError("max_train_samples_per_task must be positive")
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed))[:limit].tolist()
    return Subset(dataset, indices)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    method: str,
    replay_buffer: ReplayBuffer | None,
    replay_batch_size: int,
    ewc: OnlineEWC | None,
    ewc_lambda: float,
) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if method == "replay" and replay_buffer is not None and len(replay_buffer) > 0:
            old_inputs, old_labels = replay_buffer.sample(replay_batch_size, device)
            inputs = torch.cat((inputs, old_inputs), dim=0)
            labels = torch.cat((labels, old_labels), dim=0)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = nn.functional.cross_entropy(logits, labels)
        if method == "ewc" and ewc is not None and ewc.ready:
            loss = loss + ewc_lambda * ewc.penalty(model)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                "loss became non-finite; lower the learning rate or regularization strength "
                f"(method={method}, ewc_lambda={ewc_lambda})"
            )
        loss.backward()
        optimizer.step()

        total_loss += loss.detach().item() * labels.size(0)
        total_correct += (logits.detach().argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)
    return total_loss / total_samples, total_correct / total_samples


@torch.inference_mode()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    correct = 0
    total = 0
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        predictions = model(inputs).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
    if total == 0:
        raise ValueError("cannot evaluate an empty dataset")
    return correct / total


def add_dataset_to_buffer(buffer: ReplayBuffer, loader: DataLoader) -> None:
    for inputs, labels in loader:
        buffer.add_batch(inputs, labels)


def build_stream(config: ExperimentConfig) -> TaskStream:
    if config.dataset == "synthetic":
        return build_synthetic_stream(
            num_classes=config.num_classes,
            classes_per_task=config.classes_per_task,
            train_samples_per_class=config.train_samples_per_class,
            test_samples_per_class=config.test_samples_per_class,
            seed=config.seed,
        )
    if config.dataset == "cifar100":
        if config.num_classes != 100:
            raise ValueError("CIFAR-100 requires --num-classes 100")
        return build_cifar100_stream(
            root=config.data_root,
            classes_per_task=config.classes_per_task,
            seed=config.seed,
            download=config.download,
        )
    raise ValueError(f"unknown dataset: {config.dataset}")


def save_results(
    output_dir: Path,
    result: dict[str, object],
    model: nn.Module,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    matrix = result["accuracy_matrix"]
    with (output_dir / "accuracy_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["after_task", *[f"task_{index + 1}" for index in range(len(matrix))]])
        for index, row in enumerate(matrix):
            writer.writerow([index + 1, *["" if value is None else f"{value:.6f}" for value in row]])

    torch.save({"model_state_dict": model.state_dict(), "result": result}, output_dir / "checkpoint.pt")


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    set_seed(config.seed, config.deterministic)
    device = resolve_device(config.device)
    stream = build_stream(config)
    model_name = config.model
    if model_name == "auto":
        model_name = "resnet18" if config.dataset == "cifar100" else "tiny"
    model = build_model(model_name, stream.num_classes).to(device)

    replay_buffer = ReplayBuffer(config.buffer_size, config.seed + 100) if config.method == "replay" else None
    ewc = OnlineEWC(config.ewc_decay) if config.method == "ewc" else None
    task_count = len(stream.specs)
    accuracy_matrix: list[list[float | None]] = [[None] * task_count for _ in range(task_count)]
    task_logs: list[dict[str, object]] = []
    started = time.perf_counter()

    print(f"device={device} method={config.method} model={model_name} tasks={task_count}", flush=True)
    for task_index, spec in enumerate(stream.specs):
        task_dataset = limit_dataset(
            stream.train_datasets[task_index],
            config.max_train_samples_per_task,
            config.seed + task_index,
        )
        train_loader = make_loader(
            task_dataset,
            config.batch_size,
            shuffle=True,
            num_workers=config.num_workers,
            seed=config.seed + task_index,
            device=device,
        )
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
        )

        epoch_logs: list[dict[str, float | int]] = []
        for epoch in range(config.epochs_per_task):
            loss, train_accuracy = train_epoch(
                model=model,
                loader=train_loader,
                optimizer=optimizer,
                device=device,
                method=config.method,
                replay_buffer=replay_buffer,
                replay_batch_size=config.replay_batch_size,
                ewc=ewc,
                ewc_lambda=config.ewc_lambda,
            )
            epoch_logs.append({"epoch": epoch + 1, "loss": loss, "train_accuracy": train_accuracy})
            print(
                f"task={task_index + 1}/{task_count} epoch={epoch + 1}/{config.epochs_per_task} "
                f"loss={loss:.4f} train_acc={train_accuracy:.4f}",
                flush=True,
            )

        for eval_index in range(task_index + 1):
            eval_loader = make_loader(
                stream.test_datasets[eval_index],
                config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                seed=config.seed,
                device=device,
            )
            accuracy_matrix[task_index][eval_index] = evaluate(model, eval_loader, device)

        fisher_count = 0
        if ewc is not None:
            fisher_loader = make_loader(
                task_dataset,
                config.batch_size,
                shuffle=True,
                num_workers=config.num_workers,
                seed=config.seed + 1000 + task_index,
                device=device,
            )
            fisher_count = ewc.consolidate(model, fisher_loader, device, config.fisher_samples)

        if replay_buffer is not None:
            buffer_loader = make_loader(
                task_dataset,
                config.batch_size,
                shuffle=False,
                num_workers=config.num_workers,
                seed=config.seed,
                device=device,
            )
            add_dataset_to_buffer(replay_buffer, buffer_loader)

        learned_accuracies = [accuracy_matrix[task_index][index] for index in range(task_index + 1)]
        print(
            "eval=" + ", ".join(f"T{index + 1}:{accuracy:.4f}" for index, accuracy in enumerate(learned_accuracies)),
            flush=True,
        )
        task_logs.append(
            {
                "task": task_index + 1,
                "class_ids": list(spec.class_ids),
                "epochs": epoch_logs,
                "replay_buffer_size": len(replay_buffer) if replay_buffer is not None else 0,
                "fisher_samples": fisher_count,
            }
        )

    metrics = summarize_accuracy_matrix(accuracy_matrix)
    elapsed = time.perf_counter() - started
    result: dict[str, object] = {
        "config": asdict(config),
        "resolved_device": str(device),
        "resolved_model": model_name,
        "class_order": stream.class_order,
        "tasks": [list(spec.class_ids) for spec in stream.specs],
        "accuracy_matrix": accuracy_matrix,
        "metrics": metrics,
        "task_logs": task_logs,
        "elapsed_seconds": elapsed,
    }
    save_results(Path(config.output_dir), result, model)
    print(
        f"done final_average_accuracy={metrics['final_average_accuracy']:.4f} "
        f"final_average_forgetting={metrics['final_average_forgetting']:.4f} "
        f"elapsed_seconds={elapsed:.1f}",
        flush=True,
    )
    return result


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Class-incremental learning on CIFAR-100")
    parser.add_argument("--dataset", choices=("cifar100", "synthetic"), default="synthetic")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--output-dir", default="runs/experiment")
    parser.add_argument("--method", choices=("naive", "replay", "ewc"), default="naive")
    parser.add_argument("--model", choices=("auto", "resnet18", "tiny"), default="auto")
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--classes-per-task", type=int, default=10)
    parser.add_argument("--epochs-per-task", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use deterministic CUDA algorithms for fair comparisons (default: true)",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--buffer-size", type=int, default=2000)
    parser.add_argument("--replay-batch-size", type=int, default=64)
    parser.add_argument("--ewc-lambda", type=float, default=10.0)
    parser.add_argument("--ewc-decay", type=float, default=0.9)
    parser.add_argument("--fisher-samples", type=int, default=1024)
    parser.add_argument("--train-samples-per-class", type=int, default=32)
    parser.add_argument("--test-samples-per-class", type=int, default=16)
    parser.add_argument("--max-train-samples-per-task", type=int)
    return parser


def config_from_args(args: argparse.Namespace) -> ExperimentConfig:
    return ExperimentConfig(**vars(args))


def main() -> None:
    parser = make_parser()
    config = config_from_args(parser.parse_args())
    run_experiment(config)


if __name__ == "__main__":
    main()
