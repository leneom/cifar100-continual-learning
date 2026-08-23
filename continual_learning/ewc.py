from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


class OnlineEWC:
    """Online EWC with a diagonal empirical-Fisher approximation."""

    def __init__(self, decay: float = 1.0):
        if not 0.0 <= decay <= 1.0:
            raise ValueError("decay must be between 0 and 1")
        self.decay = decay
        self.means: dict[str, torch.Tensor] = {}
        self.fisher: dict[str, torch.Tensor] = {}

    @property
    def ready(self) -> bool:
        return bool(self.fisher)

    def penalty(self, model: nn.Module) -> torch.Tensor:
        first_parameter = next(model.parameters())
        if not self.ready:
            return first_parameter.new_zeros(())
        total = first_parameter.new_zeros(())
        for name, parameter in model.named_parameters():
            if name in self.fisher:
                total = total + (self.fisher[name] * (parameter - self.means[name]).pow(2)).sum()
        return 0.5 * total

    def consolidate(
        self,
        model: nn.Module,
        loader: Iterable,
        device: torch.device,
        max_samples: int,
    ) -> int:
        if max_samples <= 0:
            raise ValueError("max_samples must be positive")
        was_training = model.training
        model.eval()
        estimated = {
            name: torch.zeros_like(parameter, device=device)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        total_samples = 0
        for inputs, labels in loader:
            remaining = max_samples - total_samples
            if remaining <= 0:
                break
            inputs = inputs[:remaining].to(device)
            labels = labels[:remaining].to(device)
            model.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(inputs), labels)
            loss.backward()
            batch_size = labels.size(0)
            for name, parameter in model.named_parameters():
                if parameter.grad is not None and name in estimated:
                    estimated[name] += parameter.grad.detach().pow(2) * batch_size
            total_samples += batch_size

        if total_samples == 0:
            raise ValueError("cannot estimate Fisher information from an empty loader")

        for name in estimated:
            estimated[name] /= total_samples
            if name in self.fisher:
                estimated[name] += self.decay * self.fisher[name]
        self.fisher = {name: value.detach().clone() for name, value in estimated.items()}
        self.means = {
            name: parameter.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        model.train(was_training)
        return total_samples

