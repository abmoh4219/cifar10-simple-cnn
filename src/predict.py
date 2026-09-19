"""Inference on arbitrary image files using the saved checkpoint.

Runs as a standalone process: it cold-loads the checkpoint from disk, applies
the eval transform to each image in ``CFG.samples_dir`` (or paths given on the
command line), and prints the predicted class with its confidence.
"""

import torch
from torch import nn

from src.config import CFG


def load_image(path: str) -> torch.Tensor:
    """Open an image file and return a normalised ``(1, 3, 32, 32)`` tensor."""
    raise NotImplementedError


@torch.no_grad()
def predict(model: nn.Module, x: torch.Tensor, device: torch.device) -> tuple[str, float]:
    """Return ``(class_name, confidence)`` for a single preprocessed image."""
    raise NotImplementedError


def main(argv: list[str] | None = None, cfg=CFG) -> None:
    """Entry point: parse image paths, load checkpoint, print predictions."""
    raise NotImplementedError


if __name__ == "__main__":
    main()
