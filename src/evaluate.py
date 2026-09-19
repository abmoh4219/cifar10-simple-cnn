"""Final evaluation: load the best checkpoint and report test-set metrics.

This is the ONLY module that reads the CIFAR-10 test set. It runs exactly once
after training is complete and produces overall accuracy, per-class accuracy,
and a confusion matrix saved to ``CFG.artifacts_dir``.
"""

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CFG


def load_checkpoint(path: str, device: torch.device) -> nn.Module:
    """Build the model and load weights from ``path``."""
    raise NotImplementedError


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict:
    """Return a dict with overall accuracy, per-class accuracy, and confusion matrix."""
    raise NotImplementedError


def plot_confusion_matrix(cm, class_names: tuple[str, ...], out_path: str) -> None:
    """Render and save the confusion matrix as a PNG."""
    raise NotImplementedError


def main(cfg=CFG) -> None:
    """Entry point: load checkpoint, evaluate on test set, save report."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
