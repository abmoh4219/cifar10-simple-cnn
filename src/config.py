"""Single source of truth for hyperparameters, paths, and dataset constants.

Every other module imports ``CFG`` from here. Nothing is hardcoded elsewhere.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # Reproducibility
    seed: int = 42

    # Optimisation
    batch_size: int = 128
    epochs: int = 20
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # Data
    val_split: int = 5000  # images held out from the 50k train set for validation
    num_workers: int = 2

    # Paths
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"
    samples_dir: str = "samples"
    checkpoint_name: str = "best_model.pt"

    # CIFAR-10 per-channel statistics (RGB), computed on the training set
    mean: tuple[float, float, float] = (0.4914, 0.4822, 0.4465)
    std: tuple[float, float, float] = (0.2470, 0.2435, 0.2616)

    # Class names in label-index order (0..9)
    classes: tuple[str, ...] = (
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )


CFG = Config()
