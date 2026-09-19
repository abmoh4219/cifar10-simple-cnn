"""Model definition: a simple CNN with at most two convolutional layers.

This constraint comes from the assessment spec and must not be exceeded.
"""

import torch
from torch import nn

from src.config import CFG


class SimpleCNN(nn.Module):
    """Two-conv-layer CNN for 32x32 RGB images with 10 output classes."""

    def __init__(self, num_classes: int = 10):
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


def build_model(cfg=CFG) -> nn.Module:
    """Construct the model from ``cfg`` and return it (on CPU)."""
    raise NotImplementedError
