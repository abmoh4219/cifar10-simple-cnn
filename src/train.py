"""Training loop: fit the model on the train split, select on the val split.

Saves the best checkpoint (by validation accuracy) to
``CFG.artifacts_dir / CFG.checkpoint_name`` and writes loss / accuracy curves
alongside it. The test set is never touched here.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CFG


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and torch (CPU + CUDA) for reproducibility."""
    raise NotImplementedError


def get_device() -> torch.device:
    """Return cuda if available, else mps if available, else cpu."""
    raise NotImplementedError


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch of training. Return ``(mean_loss, accuracy)``."""
    raise NotImplementedError


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a loader without gradients. Return ``(mean_loss, accuracy)``."""
    raise NotImplementedError


def plot_curves(history: dict, out_dir: str) -> None:
    """Save loss and accuracy curves as PNG files under ``out_dir``."""
    raise NotImplementedError


def main(cfg=CFG) -> None:
    """Entry point: seed, build data + model, train, checkpoint, plot."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
