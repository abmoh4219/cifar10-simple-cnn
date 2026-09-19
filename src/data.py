"""Data pipeline: download CIFAR-10, apply transforms, and build DataLoaders.

Responsible for the train / validation / test split. The validation split is
carved from the official training set using ``CFG.val_split`` and ``CFG.seed``;
the official test set is never used for model selection.
"""

from torch.utils.data import DataLoader

from src.config import CFG


def get_transforms(train: bool):
    """Return the torchvision transform pipeline for train or eval mode."""
    raise NotImplementedError


def get_dataloaders(cfg=CFG) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)``."""
    raise NotImplementedError
