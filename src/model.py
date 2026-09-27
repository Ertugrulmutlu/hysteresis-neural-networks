"""Model definitions used by training and analysis."""
from typing import Dict, Tuple, Union

import torch
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Apply LayerNorm over channels independently at every spatial location."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"LayerNorm2d expects NCHW input, got shape {tuple(x.shape)}")
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()


def norm_layer(norm_type: str, channels: int, groups: int = 8) -> nn.Module:
    norm_type = norm_type.lower()
    if norm_type == "none":
        return nn.Identity()
    if norm_type == "group":
        if channels % groups:
            raise ValueError(f"channels ({channels}) must be divisible by groups ({groups})")
        return nn.GroupNorm(groups, channels)
    if norm_type == "layer":
        return LayerNorm2d(channels)
    raise ValueError(f"Unsupported normalization {norm_type!r}; choose none, group, or layer")


class SimpleCNN(nn.Module):
    """Part 1 CNN, with optional access to distinct post-ReLU activations."""

    def __init__(self, norm: str = "none", gn_groups: int = 8, activation: str = "relu",
                 leaky_relu_negative_slope: float = 0.01) -> None:
        super().__init__()
        activation = activation.lower()
        if leaky_relu_negative_slope < 0:
            raise ValueError("leaky_relu_negative_slope must be non-negative")
        if activation == "relu":
            activation_factory = nn.ReLU
        elif activation == "leaky_relu":
            activation_factory = lambda: nn.LeakyReLU(leaky_relu_negative_slope)
        else:
            raise ValueError(f"Unsupported activation {activation!r}; choose relu or leaky_relu")
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.norm1 = norm_layer(norm, 32, gn_groups)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.norm2 = norm_layer(norm, 64, gn_groups)
        self.relu1, self.relu2, self.relu3 = activation_factory(), activation_factory(), activation_factory()
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(
        self, x: Tensor, return_activations: bool = False
    ) -> Union[Tensor, Tuple[Tensor, Dict[str, Tensor]]]:
        conv1 = self.relu1(self.norm1(self.conv1(x)))
        conv2 = self.relu2(self.norm2(self.conv2(self.pool(conv1))))
        fc1 = self.relu3(self.fc1(self.pool(conv2).flatten(1)))
        logits = self.fc2(fc1)
        if return_activations:
            return logits, {"conv1": conv1, "conv2": conv2, "fc1": fc1, "logits": logits}
        return logits
