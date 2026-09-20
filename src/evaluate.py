"""Final evaluation: load the best checkpoint and report test-set metrics.

**This is the only module in the project that touches the CIFAR-10 test set,
and it touches it exactly once.** Every earlier decision — which epoch to keep,
which hyperparameters to use — was made from the validation split alone, so
the number produced here is an unbiased estimate of generalisation. Running
this file repeatedly to tune anything would destroy that property.

Outputs: ``test_metrics.json``, ``confusion_matrix.png``, ``misclassified.png``.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed on a server or in CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import CFG
from src.data import get_dataloaders
from src.model import build_model
from src.utils import get_device, set_seed

METRICS_NAME = "test_metrics.json"
CONFUSION_NAME = "confusion_matrix.png"
MISCLASSIFIED_NAME = "misclassified.png"

# Config fields that change what the pixels look like or how the split was
# drawn. A checkpoint trained under different values cannot be compared to a
# model evaluated under the current ones.
CRITICAL_FIELDS = ("seed", "batch_size", "mean", "std")


def load_checkpoint(path: Path | str, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    """Load weights from ``path`` into a fresh model, ready for inference.

    Returns ``(model, metadata)`` where metadata is everything in the
    checkpoint except the two state dicts — the provenance (epoch, val_acc,
    config) without the megabytes of tensors.

    ``map_location`` is passed so a checkpoint trained on CUDA can be loaded
    on a CPU-only machine; without it, torch tries to restore tensors onto a
    device that may not exist here.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # eval() switches BatchNorm to its running statistics and disables dropout.
    # Without it the reported test accuracy would depend on batch composition.
    model.eval()

    metadata = {k: v for k, v in checkpoint.items() if not k.endswith("state_dict")}
    print(f"checkpoint : {path}")
    print(f"  trained to epoch {metadata['epoch']}, val_acc {metadata['val_acc']:.4f}")

    _warn_on_config_mismatch(metadata.get("config", {}))
    return model, metadata


def _warn_on_config_mismatch(saved_config: dict[str, Any]) -> None:
    """Warn if the checkpoint was trained under different preprocessing.

    A mismatch in ``mean``/``std`` silently shifts every input pixel, and the
    model still returns confident predictions — just wrong ones. That failure
    is invisible without this check, so it is worth the ten lines.
    """

    def agrees(saved: Any, current: Any) -> bool:
        # torch.save round-trips tuples faithfully, but a checkpoint written by
        # another tool may hold lists — compare those by value, not by type.
        if isinstance(current, tuple):
            return isinstance(saved, (list, tuple)) and tuple(saved) == current
        return saved == current

    mismatches = [
        (field, saved_config.get(field, "<missing>"), getattr(CFG, field))
        for field in CRITICAL_FIELDS
        if not agrees(saved_config.get(field, "<missing>"), getattr(CFG, field))
    ]
    if mismatches:
        print("  WARNING: checkpoint config disagrees with current CFG:")
        for field, saved, now in mismatches:
            print(f"    {field}: checkpoint={saved!r}  current={now!r}")
        print("  Predictions may be invalid — retrain or restore the original config.")
    else:
        print(f"  config matches current CFG on {', '.join(CRITICAL_FIELDS)}")


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    """Run the test set once and collect everything downstream reporting needs.

    Predictions, labels and confidences are gathered in a single pass rather
    than recomputed per metric — the test set must be read exactly once, and
    every figure below is derived from these same arrays.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    total_loss = 0.0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_conf: list[np.ndarray] = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        total_loss += criterion(logits, targets).item()

        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)
        all_true.append(targets.cpu().numpy())
        all_pred.append(pred.cpu().numpy())
        all_conf.append(conf.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_conf = np.concatenate(all_conf)
    return {
        "loss": total_loss / len(y_true),
        "accuracy": float((y_true == y_pred).mean()),
        "y_true": y_true,
        "y_pred": y_pred,
        "y_conf": y_conf,
    }


def per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """Per-class support, correct, accuracy, precision, recall and F1.

    Computed with numpy rather than pulling in scikit-learn: six formulas do
    not justify a dependency, and writing them out makes the definitions
    visible instead of hidden behind an import.

    Note that per-class accuracy and recall are the same quantity here
    (TP / support) — both are kept because reviewers look for each by name.
    Rows are sorted by accuracy ascending so the weakest classes are first,
    which is where the interesting failure analysis lives.
    """
    rows = []
    for label, name in enumerate(CFG.classes):
        is_true = y_true == label
        is_pred = y_pred == label
        support = int(is_true.sum())
        true_positive = int((is_true & is_pred).sum())
        predicted = int(is_pred.sum())

        # Guard every denominator: a class the model never predicts would
        # otherwise produce a divide-by-zero rather than a precision of 0.
        recall = true_positive / support if support else 0.0
        precision = true_positive / predicted if predicted else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append(
            {
                "class": name,
                "support": support,
                "correct": true_positive,
                "accuracy": recall,
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    table = pd.DataFrame(rows).sort_values("accuracy", ascending=True).reset_index(drop=True)
    # Macro average weights every class equally, which is the right summary for
    # a balanced test set and exposes a model that carries one weak class.
    macro = {
        "class": "macro avg",
        "support": int(table["support"].sum()),
        "correct": int(table["correct"].sum()),
        "accuracy": float(table["accuracy"].mean()),
        "precision": float(table["precision"].mean()),
        "recall": float(table["recall"].mean()),
        "f1": float(table["f1"].mean()),
    }
    return pd.concat([table, pd.DataFrame([macro])], ignore_index=True)


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Counts matrix with rows = true class, cols = predicted class.

    ``np.add.at`` accumulates into the flattened matrix in one vectorised
    pass, which keeps this readable without a Python loop over 10,000 samples.
    """
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def plot_confusion_matrix(cm: np.ndarray, out_path: Path | str) -> None:
    """Render a row-normalised confusion matrix.

    Rows are normalised to sum to 1 so each cell reads as "of the true cats,
    this fraction went here". Raw counts would only be comparable because
    CIFAR-10's test set happens to be balanced; normalising makes the figure
    correct regardless and lets colour encode a rate rather than a count.
    """
    row_totals = cm.sum(axis=1, keepdims=True)
    normalised = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals != 0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    image = ax.imshow(normalised, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(image, ax=ax, label="fraction of true class")

    ticks = np.arange(len(CFG.classes))
    ax.set_xticks(ticks, CFG.classes, rotation=45, ha="right")
    ax.set_yticks(ticks, CFG.classes)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("CIFAR-10 test set — row-normalised confusion matrix")

    # Flip the text to white on dark cells so every annotation stays legible.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            value = normalised[i, j]
            ax.text(
                j,
                i,
                f"{value * 100:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value > 0.5 else "black",
            )

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def top_confusions(cm: np.ndarray, k: int = 5) -> list[tuple[str, str, int]]:
    """The ``k`` largest off-diagonal cells as ``(true, predicted, count)``.

    The diagonal is masked out because correct predictions are not confusions;
    what matters for error analysis is which wrong class the model reaches for.
    """
    errors = cm.copy()
    np.fill_diagonal(errors, 0)
    flat_order = np.argsort(errors, axis=None)[::-1][:k]
    pairs = np.unravel_index(flat_order, errors.shape)
    return [
        (CFG.classes[int(t)], CFG.classes[int(p)], int(errors[t, p]))
        for t, p in zip(*pairs)
        if errors[t, p] > 0
    ]


def _denormalise(image: torch.Tensor) -> np.ndarray:
    """Undo Normalize so a tensor can be shown as a picture.

    Display needs the original pixel range; the normalised tensor contains
    negative values that imshow would clip into a false-coloured mess.
    """
    mean = torch.tensor(CFG.mean).view(3, 1, 1)
    std = torch.tensor(CFG.std).view(3, 1, 1)
    return (image.cpu() * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()


@torch.no_grad()
def plot_misclassified(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_path: Path | str,
    n: int = 16,
) -> None:
    """Grid of the first ``n`` misclassified test images, for qualitative review.

    Aggregate metrics say how often the model is wrong; only looking at the
    failures says whether they are reasonable (a cat called a dog) or alarming
    (a truck called a bird). That distinction is the point of this figure.
    """
    model.eval()
    images: list[np.ndarray] = []
    captions: list[str] = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        probs = torch.softmax(model(inputs), dim=1)
        conf, pred = probs.max(dim=1)

        for idx in (pred != targets).nonzero(as_tuple=True)[0]:
            images.append(_denormalise(inputs[idx]))
            captions.append(
                f"{CFG.classes[targets[idx]]} -> {CFG.classes[pred[idx]]} ({conf[idx]:.2f})"
            )
            if len(images) == n:
                break
        if len(images) == n:
            break

    side = int(np.ceil(np.sqrt(n)))
    fig, axes = plt.subplots(side, side, figsize=(2 * side, 2.2 * side))
    for ax, image, caption in zip(axes.flat, images, captions):
        ax.imshow(image)
        ax.set_title(caption, fontsize=8)
    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle("Misclassified test images — true -> predicted (confidence)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    """Load the best checkpoint, score the test set once, and write the report."""
    set_seed(CFG.seed)
    device = get_device()
    artifacts_dir = Path(CFG.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    print(f"device     : {device}")
    model, metadata = load_checkpoint(artifacts_dir / CFG.checkpoint_name, device)

    # Only the test loader is used here; train and val are discarded.
    *_, test_loader = get_dataloaders()
    results = evaluate(model, test_loader, device)

    print(f"\ntest accuracy : {results['accuracy']:.4f}")
    print(f"test loss     : {results['loss']:.4f}")

    table = per_class_metrics(results["y_true"], results["y_pred"])
    print("\nper-class metrics (weakest first):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    cm = confusion_matrix(results["y_true"], results["y_pred"], len(CFG.classes))
    print("\ntop confusions:")
    for true_name, pred_name, count in top_confusions(cm, k=5):
        share = count / cm[CFG.classes.index(true_name)].sum() * 100
        print(f"  {count} {true_name}s were called {pred_name} ({share:.1f}% of all {true_name}s)")

    correct_mask = results["y_true"] == results["y_pred"]
    conf_correct = float(results["y_conf"][correct_mask].mean())
    conf_wrong = float(results["y_conf"][~correct_mask].mean())
    print(f"\nmean confidence when correct   : {conf_correct:.4f}")
    print(f"mean confidence when incorrect : {conf_wrong:.4f}")

    metrics_path = artifacts_dir / METRICS_NAME
    with open(metrics_path, "w") as fh:
        json.dump(
            {
                "checkpoint_epoch": metadata["epoch"],
                "checkpoint_val_acc": metadata["val_acc"],
                "test_accuracy": results["accuracy"],
                "test_loss": results["loss"],
                "mean_confidence_correct": conf_correct,
                "mean_confidence_incorrect": conf_wrong,
                "per_class": table.to_dict(orient="records"),
                "confusion_matrix": cm.tolist(),
                "class_names": list(CFG.classes),
            },
            fh,
            indent=2,
        )

    confusion_path = artifacts_dir / CONFUSION_NAME
    misclassified_path = artifacts_dir / MISCLASSIFIED_NAME
    plot_confusion_matrix(cm, confusion_path)
    plot_misclassified(model, test_loader, device, misclassified_path)

    print(f"\nmetrics    : {metrics_path}")
    print(f"confusion  : {confusion_path}")
    print(f"misclassed : {misclassified_path}")


if __name__ == "__main__":
    main()
