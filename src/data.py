"""Data pipeline: download CIFAR-10, apply transforms, and build DataLoaders.

Owns the train / validation / test split. The validation split is carved out
of the official 50k training set with a seeded permutation, so it is identical
across runs and disjoint from the training images. The official 10k test set
is loaded here for completeness but is only *consumed* by ``evaluate.py``.
"""

import time
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose

from src.config import CFG
from src.utils import set_seed

# CIFAR-10 images are 32x32; the padding for RandomCrop is an augmentation
# choice, not a dataset constant, so it lives here rather than in Config.
IMAGE_SIZE = 32
CROP_PADDING = 4


def get_transforms() -> tuple[Compose, Compose]:
    """Return ``(train_tf, eval_tf)``.

    Augmentation (random crop with padding + horizontal flip) is deliberately
    confined to ``train_tf``. Validation and test images must look exactly
    like what the model will see at inference, otherwise measured accuracy is
    a property of the augmentation lottery rather than of the model.
    """
    normalize = transforms.Normalize(CFG.mean, CFG.std)
    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(IMAGE_SIZE, padding=CROP_PADDING),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    eval_tf = transforms.Compose([transforms.ToTensor(), normalize])
    return train_tf, eval_tf


def _split_indices(n_total: int, val_size: int, seed: int) -> tuple[list[int], list[int]]:
    """Return disjoint ``(train_idx, val_idx)`` from one seeded permutation.

    A dedicated ``torch.Generator`` is used instead of the global RNG so the
    split does not depend on how many random numbers were drawn before this
    call — the same seed always yields the same split, wherever it is invoked.
    """
    gen = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=gen).tolist()
    return perm[: n_total - val_size], perm[n_total - val_size :]


def get_datasets() -> tuple[Dataset, Dataset, Dataset]:
    """Return ``(train_ds, val_ds, test_ds)``.

    The training data is instantiated *twice* from the same files: one copy
    with ``train_tf`` and one with ``eval_tf``. A single seeded permutation is
    then sliced so the first ``50000 - CFG.val_split`` indices select from the
    augmented copy and the remaining ``CFG.val_split`` select from the clean
    copy.

    Why not ``random_split`` on one dataset? Both halves would share the parent
    object's transform, so validation images would be randomly cropped and
    flipped — quietly inflating variance in the very metric used for model
    selection. Two dataset objects with two transforms is the only way to get
    disjoint indices *and* a clean validation set.
    """
    train_tf, eval_tf = get_transforms()

    train_full_aug = CIFAR10(CFG.data_dir, train=True, download=True, transform=train_tf)
    train_full_clean = CIFAR10(CFG.data_dir, train=True, download=False, transform=eval_tf)
    test_ds = CIFAR10(CFG.data_dir, train=False, download=True, transform=eval_tf)

    train_idx, val_idx = _split_indices(len(train_full_aug), CFG.val_split, CFG.seed)
    train_ds = Subset(train_full_aug, train_idx)
    val_ds = Subset(train_full_clean, val_idx)
    return train_ds, val_ds, test_ds


def get_dataloaders(
    batch_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Return ``(train_loader, val_loader, test_loader)``.

    Only the training loader shuffles, and it does so through its own seeded
    generator so the batch order is reproducible independent of the global
    RNG state. ``pin_memory`` is a CUDA-only optimisation (page-locked host
    memory for faster H2D copies); enabling it on CPU/MPS just emits warnings.
    """
    batch_size = CFG.batch_size if batch_size is None else batch_size
    train_ds, val_ds, test_ds = get_datasets()

    pin_memory = torch.cuda.is_available()
    shuffle_gen = torch.Generator().manual_seed(CFG.seed)
    common = dict(num_workers=CFG.num_workers, pin_memory=pin_memory)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, generator=shuffle_gen, **common
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, **common)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, **common)
    return train_loader, val_loader, test_loader


def export_sample_images() -> list[Path]:
    """Save one test image per class to ``CFG.samples_dir`` as PNGs.

    These feed the later ``predict.py`` demo, which must accept ordinary image
    files rather than pre-processed tensors. The dataset is therefore loaded
    with ``transform=None`` so we get raw PIL images — writing back a
    normalised tensor would produce a garbled picture and would also leak the
    preprocessing step out of the inference script that is meant to own it.

    The first occurrence of each label is taken, in label order, so the demo
    covers every class exactly once instead of whatever the first few test
    images happen to be. The true label is baked into the filename.
    """
    out_dir = Path(CFG.samples_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_test = CIFAR10(CFG.data_dir, train=False, download=True, transform=None)
    # targets is a plain list of ints, so .index() gives the first occurrence
    # without decoding any image.
    written: list[Path] = []
    for label, name in enumerate(CFG.classes):
        img, _ = raw_test[raw_test.targets.index(label)]
        assert isinstance(img, Image.Image)
        path = out_dir / f"sample_{label}_{name}.png"
        img.save(path)
        written.append(path)
    return written


if __name__ == "__main__":
    set_seed(CFG.seed)

    train_ds, val_ds, test_ds = get_datasets()
    print(f"train: {len(train_ds):>6}  val: {len(val_ds):>6}  test: {len(test_ds):>6}")

    train_loader, val_loader, _ = get_dataloaders()
    x, y = next(iter(train_loader))
    print(f"batch x: {tuple(x.shape)} {x.dtype}   y: {tuple(y.shape)} {y.dtype}")

    # After Normalize, per-channel stats should sit near mean≈0 / std≈1.
    # The val batch hits this; the train batch is pulled negative because
    # RandomCrop's zero-padding normalises to ≈-2 and covers ~13% of pixels.
    def _stats(t: torch.Tensor) -> str:
        m = [round(v, 3) for v in t.mean(dim=(0, 2, 3)).tolist()]
        s = [round(v, 3) for v in t.std(dim=(0, 2, 3)).tolist()]
        return f"mean={m} std={s}"

    xv, _ = next(iter(val_loader))
    print(f"train batch (augmented) : {_stats(x)}")
    print(f"val batch   (clean)     : {_stats(xv)}")

    val_targets = [val_ds.dataset.targets[i] for i in val_ds.indices]
    counts = Counter(val_targets)
    print("val class distribution:")
    for label in range(len(CFG.classes)):
        print(f"  {CFG.classes[label]:<10} {counts[label]:>4}")

    overlap = set(train_ds.indices) & set(val_ds.indices)
    print(f"train/val index overlap: {len(overlap)}  (disjoint={not overlap})")

    n_batches = 20
    t0 = time.perf_counter()
    for i, _ in enumerate(train_loader, start=1):
        if i == n_batches:
            break
    elapsed = time.perf_counter() - t0
    print(f"{n_batches} train batches in {elapsed:.2f}s ({elapsed / n_batches * 1000:.0f} ms/batch)")

    paths = export_sample_images()
    print(f"wrote {len(paths)} sample images to {Path(CFG.samples_dir)}/")
    for p in paths:
        print(f"  {p}")
