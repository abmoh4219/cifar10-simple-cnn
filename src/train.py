"""Training loop: fit the model on the train split, select on the val split.

Writes three artifacts: the best checkpoint (by validation accuracy), a
per-epoch ``history.csv``, and the training curves PNG.

The CIFAR-10 **test set is never touched here**. Every decision in this file —
which epoch's weights to keep, when to stop — is made from the validation
split alone, so the test score reported by ``evaluate.py`` stays an honest
estimate of generalisation.
"""

import csv
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed on a server or in CI

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import CFG
from src.data import get_dataloaders
from src.model import build_model, count_parameters
from src.utils import get_device, set_seed

HISTORY_NAME = "history.csv"
CURVES_NAME = "training_curves.png"
HISTORY_FIELDS = (
    "epoch",
    "train_loss",
    "train_acc",
    "val_loss",
    "val_acc",
    "lr",
    "epoch_seconds",
)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, float]:
    """Run one epoch of training. Return ``(mean_loss, accuracy)``.

    Loss is accumulated weighted by batch size rather than averaged over
    batches: the final batch is usually smaller (45000 % 128 = 8 images), so a
    plain mean-of-means would over-weight it and give a subtly wrong figure.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    seen = 0

    progress = tqdm(loader, desc="train", leave=False)
    for inputs, targets in progress:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        running_loss += loss.item() * batch_size
        correct += (logits.argmax(dim=1) == targets).sum().item()
        seen += batch_size
        progress.set_description(f"train loss={running_loss / seen:.4f}")

    return running_loss / seen, correct / seen


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate on a loader without training. Return ``(mean_loss, accuracy)``.

    Both ``model.eval()`` and ``torch.no_grad()`` are required, for different
    reasons. Without ``eval()`` dropout stays active and BatchNorm updates its
    running statistics from validation data, which corrupts both the metric
    and the model itself. ``no_grad()`` only disables the autograd graph —
    it saves memory and time but does *not* switch layer behaviour.
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    seen = 0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="val", leave=False):
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model(inputs)
            loss = criterion(logits, targets)

            batch_size = targets.size(0)
            running_loss += loss.item() * batch_size
            correct += (logits.argmax(dim=1) == targets).sum().item()
            seen += batch_size

    return running_loss / seen, correct / seen


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_acc: float,
    path: Path | str,
) -> None:
    """Write model + optimizer state, provenance, and the full config to disk.

    The config travels *with* the weights so ``predict.py`` can check that the
    checkpoint it cold-loads was trained under the same normalisation
    constants and class order it is about to apply. A silent mismatch there
    produces confident, wrong predictions and is hard to debug otherwise.
    The optimizer state is included so training could be resumed.
    """
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_acc": val_acc,
            "config": asdict(CFG),
        },
        path,
    )


def plot_curves(history_csv_path: Path | str, out_path: Path | str) -> None:
    """Render loss and accuracy curves side by side from ``history.csv``.

    Reading back the CSV rather than an in-memory list means the figure can be
    regenerated after the fact from a finished — or interrupted — run without
    retraining.
    """
    history = pd.read_csv(history_csv_path)
    best_row = history.loc[history["val_acc"].idxmax()]
    best_epoch = int(best_row["epoch"])
    best_acc = float(best_row["val_acc"])

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(history["epoch"], history["train_loss"], marker="o", ms=3, label="train")
    ax_loss.plot(history["epoch"], history["val_loss"], marker="o", ms=3, label="validation")
    ax_loss.set_title("Cross-entropy loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")

    ax_acc.plot(history["epoch"], history["train_acc"], marker="o", ms=3, label="train")
    ax_acc.plot(history["epoch"], history["val_acc"], marker="o", ms=3, label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy")

    # The dashed line marks the checkpoint that evaluate.py will load — the
    # epoch chosen by validation accuracy, not the last epoch trained.
    for ax in (ax_loss, ax_acc):
        ax.axvline(best_epoch, color="grey", linestyle="--", linewidth=1)
        ax.grid(alpha=0.3)
        ax.legend()
        # Epochs are whole numbers; the default locator would show 1.5, 2.5, ...
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # The best epoch is usually near the end of the run, where a right-hand
    # label would spill outside the axes — so flip it to the left there.
    on_right = best_epoch > history["epoch"].median()
    ax_acc.annotate(
        f"best: epoch {best_epoch}\nval acc {best_acc:.4f}",
        xy=(best_epoch, best_acc),
        xytext=(-10 if on_right else 10, -30),
        textcoords="offset points",
        ha="right" if on_right else "left",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="grey", alpha=0.85),
    )

    fig.suptitle("SimpleCNN on CIFAR-10 — training history")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Train for ``CFG.epochs``, checkpointing whenever validation improves."""
    set_seed(CFG.seed)
    device = get_device()

    artifacts_dir = Path(CFG.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    history_path = artifacts_dir / HISTORY_NAME
    checkpoint_path = artifacts_dir / CFG.checkpoint_name

    # The third loader is the test set. It is deliberately discarded: nothing
    # in this file may read it, or the final test score stops being honest.
    train_loader, val_loader, _ = get_dataloaders()

    model = build_model().to(device)
    total, trainable = count_parameters(model)
    print(f"device     : {device}")
    print(f"parameters : {total:,} total / {trainable:,} trainable")
    print(f"data       : {len(train_loader.dataset):,} train / {len(val_loader.dataset):,} val")
    print(f"schedule   : {CFG.epochs} epochs, Adam lr={CFG.lr}, wd={CFG.weight_decay}, cosine decay\n")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.epochs)

    best_val_acc = 0.0
    best_epoch = 0
    start = time.perf_counter()

    # Written and flushed row by row so an interrupted run still leaves a
    # usable history that plot_curves can render.
    with open(history_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_FIELDS)
        writer.writeheader()

        for epoch in range(1, CFG.epochs + 1):
            epoch_start = time.perf_counter()
            lr = optimizer.param_groups[0]["lr"]

            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            val_loss, val_acc = validate(model, val_loader, criterion, device)
            scheduler.step()

            epoch_seconds = time.perf_counter() - epoch_start
            writer.writerow(
                {
                    "epoch": epoch,
                    "train_loss": round(train_loss, 6),
                    "train_acc": round(train_acc, 6),
                    "val_loss": round(val_loss, 6),
                    "val_acc": round(val_acc, 6),
                    "lr": lr,
                    "epoch_seconds": round(epoch_seconds, 2),
                }
            )
            fh.flush()

            line = (
                f"epoch {epoch:>2}/{CFG.epochs}  "
                f"train loss {train_loss:.4f} acc {train_acc:.4f}  |  "
                f"val loss {val_loss:.4f} acc {val_acc:.4f}  |  "
                f"lr {lr:.2e}  {epoch_seconds:.1f}s"
            )
            if val_acc > best_val_acc:
                best_val_acc, best_epoch = val_acc, epoch
                save_checkpoint(model, optimizer, epoch, val_acc, checkpoint_path)
                line += f"  <- saved ({checkpoint_path})"
            print(line)

    elapsed = time.perf_counter() - start
    print(f"\ntotal time   : {elapsed / 60:.1f} min ({elapsed:.0f}s)")
    print(f"best val acc : {best_val_acc:.4f} at epoch {best_epoch}")
    print(f"checkpoint   : {checkpoint_path}")

    curves_path = artifacts_dir / CURVES_NAME
    plot_curves(history_path, curves_path)
    print(f"history      : {history_path}")
    print(f"curves       : {curves_path}")


if __name__ == "__main__":
    main()
