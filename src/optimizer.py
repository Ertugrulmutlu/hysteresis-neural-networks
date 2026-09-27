"""Shared optimizer construction and relaxation-state policy."""
from typing import Any, Iterable

import torch
from torch import nn


def build_optimizer(parameters: Iterable[nn.Parameter], train_config: dict[str, Any]) -> torch.optim.Optimizer:
    name = str(train_config["optimizer"]).lower()
    kwargs = {"lr": float(train_config["lr"]), "weight_decay": float(train_config["weight_decay"])}
    if name == "sgd":
        return torch.optim.SGD(parameters, momentum=float(train_config["momentum"]), **kwargs)
    if name == "adam":
        return torch.optim.Adam(parameters, **kwargs)
    raise ValueError(f"Unsupported optimizer: {name}")


def apply_relaxation_optimizer_policy(
    optimizer: torch.optim.Optimizer, model: nn.Module, train_config: dict[str, Any]
) -> torch.optim.Optimizer:
    """Preserve optimizer state or replace it with an identically configured empty optimizer."""
    policy = str(train_config.get("relaxation_optimizer_policy", "preserve")).lower()
    if policy == "preserve":
        return optimizer
    if policy == "reset":
        return build_optimizer(model.parameters(), train_config)
    raise ValueError("relaxation_optimizer_policy must be 'preserve' or 'reset'")
