"""Small helpers shared by every stage: seeding and device selection.

Kept separate from ``data.py`` because ``train.py``, ``evaluate.py`` and
``predict.py`` all need these without pulling in the dataset machinery.
"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Seed every RNG the pipeline can touch so runs are bit-for-bit repeatable.

    Python's ``random``, NumPy, and torch each keep separate global generators,
    and torchvision augmentations draw from torch's. cuDNN is additionally
    forced to deterministic kernels: its autotuner ("benchmark") may pick
    different convolution algorithms run-to-run, which changes float rounding
    and therefore the trained weights.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Pick the fastest available backend: CUDA, then Apple MPS, then CPU.

    Centralised here so every script agrees on the device and the checkpoint
    saved by ``train.py`` on one machine can be loaded by ``predict.py`` on
    another via ``map_location``.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
