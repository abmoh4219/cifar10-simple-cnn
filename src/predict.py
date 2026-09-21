"""Standalone inference on image files using the saved checkpoint.

**This module runs as a separate process from training and carries no state
across from it.** It cold-loads the checkpoint from disk, rebuilds the model
from ``src.model``, and reapplies the evaluation transform from ``src.data``.
Nothing is handed to it in memory. That is deliberate: if this script works
from a clean interpreter, the saved artifact is genuinely self-sufficient and
the model can be deployed anywhere the checkpoint file can be copied.

    python -m src.predict --images samples/ --topk 3
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: no display needed on a server or in CI

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch import nn

from src.config import CFG
from src.data import get_transforms
from src.model import build_model
from src.utils import get_device

IMAGE_SIZE = 32
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
# Matches the files written by data.export_sample_images(), e.g.
# "sample_3_cat.png" -> "cat".
SAMPLE_NAME_PATTERN = re.compile(r"^sample_\d+_(?P<label>[a-z]+)$")


def load_checkpoint(path: Path | str, device: torch.device) -> tuple[nn.Module, dict]:
    """Rebuild the model and load weights from disk, ready for inference.

    ``map_location`` lets a checkpoint trained on a GPU machine load on a
    CPU-only one, which is the whole point of shipping a checkpoint file.
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model = build_model()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    # eval() disables dropout and switches BatchNorm to its stored running
    # statistics. Skipping it makes single-image predictions depend on noise.
    model.eval()
    metadata = {k: v for k, v in checkpoint.items() if not k.endswith("state_dict")}
    return model, metadata


def load_image(path: Path | str) -> torch.Tensor:
    """Open an image file and return a normalised ``(1, 3, 32, 32)`` tensor.

    ``convert("RGB")`` collapses greyscale and alpha-channel images to the
    three channels the model expects; without it a PNG with transparency
    arrives as four channels and the first conv layer fails.

    A resize is announced rather than done silently: a photo downscaled from
    4000px to 32px loses almost everything, and a bad prediction on it is a
    property of the input, not a model failure. The viewer should know which.
    """
    image = Image.open(path).convert("RGB")
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        print(f"  note: resized from {image.size[0]}x{image.size[1]} to {IMAGE_SIZE}x{IMAGE_SIZE}")
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BILINEAR)

    # The eval transform is imported, never redefined: a second copy of the
    # normalisation constants is exactly how preprocessing drifts out of sync
    # with the weights it was trained against.
    _, eval_tf = get_transforms()
    return eval_tf(image).unsqueeze(0)


@torch.no_grad()
def predict(
    model: nn.Module,
    tensor: torch.Tensor,
    device: torch.device,
    topk: int = 3,
) -> list[tuple[str, float]]:
    """Return the top-``k`` ``(class_name, probability)`` pairs, descending.

    Softmax is applied for reporting only — it does not change the ranking,
    but a raw logit of 4.2 means nothing to a viewer whereas "83%" does.
    """
    model.eval()
    logits = model(tensor.to(device))
    probs = torch.softmax(logits, dim=1).squeeze(0)
    values, indices = probs.topk(min(topk, len(CFG.classes)))
    return [(CFG.classes[int(i)], float(v)) for v, i in zip(values, indices)]


def parse_true_label(path: Path | str) -> str | None:
    """Recover the true class from a ``sample_<idx>_<classname>.png`` filename.

    Used purely to display a correct/incorrect mark — it is read after the
    prediction is made and never reaches the model. Any file not matching the
    pattern, or naming an unknown class, returns None and is simply reported
    without a verdict.
    """
    match = SAMPLE_NAME_PATTERN.match(Path(path).stem)
    if match is None:
        return None
    label = match.group("label")
    return label if label in CFG.classes else None


def collect_image_paths(inputs: list[str]) -> list[Path]:
    """Expand a mix of files and directories into a sorted list of image paths.

    Directories are expanded so ``--images samples/`` works as naturally as
    passing ten filenames, which is what anyone demoing this will reach for.
    """
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(
                sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
            )
        else:
            paths.append(path)
    return paths


def plot_predictions(
    results: list[tuple[Path, list[tuple[str, float]], str | None]],
    out_path: Path | str,
) -> None:
    """Grid of inputs titled with the prediction, colour-coded by correctness.

    Green / red / grey encodes correct / wrong / unknown so the outcome reads
    at a glance on a video frame, without the viewer parsing ten text blocks.
    """
    count = len(results)
    columns = min(4, count)
    rows = int((count + columns - 1) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(2.4 * columns, 2.8 * rows), squeeze=False)

    for ax, (path, predictions, true_label) in zip(axes.flat, results):
        name, prob = predictions[0]
        if true_label is None:
            colour = "grey"
        else:
            colour = "green" if name == true_label else "red"
        ax.imshow(Image.open(path).convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE)))
        ax.set_title(f"{name} ({prob * 100:.0f}%)", fontsize=9, color=colour)
    for ax in axes.flat:
        ax.axis("off")

    fig.suptitle("SimpleCNN predictions — green correct, red wrong, grey unknown")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Command-line interface for the inference demo."""
    parser = argparse.ArgumentParser(
        prog="python -m src.predict",
        description="Classify images with the trained SimpleCNN checkpoint.",
    )
    parser.add_argument(
        "--images",
        nargs="+",
        default=[CFG.samples_dir],
        help=f"image files or a directory (default: {CFG.samples_dir}/)",
    )
    parser.add_argument(
        "--checkpoint",
        default=str(Path(CFG.artifacts_dir) / CFG.checkpoint_name),
        help="path to the trained checkpoint",
    )
    parser.add_argument("--topk", type=int, default=3, help="how many classes to show")
    parser.add_argument("--save-figure", default=None, help="optional path for a grid PNG")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Cold-load the checkpoint and classify every requested image."""
    args = parse_args(argv)

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: no checkpoint at {checkpoint_path}", file=sys.stderr)
        print("Train one first:  python -m src.train", file=sys.stderr)
        return 1

    image_paths = collect_image_paths(args.images)
    missing = [p for p in image_paths if not p.is_file()]
    if missing:
        print(f"ERROR: no such image file(s): {', '.join(str(p) for p in missing)}", file=sys.stderr)
        return 1
    if not image_paths:
        print(f"ERROR: no images found in {', '.join(args.images)}", file=sys.stderr)
        print("Generate the samples first:  python -m src.data", file=sys.stderr)
        return 1

    device = get_device()
    model, metadata = load_checkpoint(checkpoint_path, device)
    print(f"device     : {device}")
    print(f"checkpoint : {checkpoint_path}")
    print(f"  trained to epoch {metadata['epoch']}, val_acc {metadata['val_acc']:.4f}")
    print(f"images     : {len(image_paths)}\n")

    results: list[tuple[Path, list[tuple[str, float]], str | None]] = []
    known = 0
    correct = 0

    for path in image_paths:
        print(f"{path}")
        true_label = parse_true_label(path)
        predictions = load_and_predict(model, path, device, args.topk)
        results.append((path, predictions, true_label))

        top_name = predictions[0][0]
        if true_label is not None:
            known += 1
            is_right = top_name == true_label
            correct += int(is_right)
            mark = "[CORRECT]" if is_right else "[WRONG]  "
            print(f"  true: {true_label:<10} {mark}")
        for rank, (name, prob) in enumerate(predictions, start=1):
            print(f"    {rank}. {name:<10} {prob * 100:6.2f}%")
        print()

    if known:
        print(f"accuracy on {known} labelled image(s): {correct}/{known} = {correct / known:.4f}")
    else:
        print("no true labels recoverable from filenames — predictions shown without a verdict")

    if args.save_figure:
        plot_predictions(results, args.save_figure)
        print(f"figure     : {args.save_figure}")
    return 0


def load_and_predict(
    model: nn.Module,
    path: Path,
    device: torch.device,
    topk: int,
) -> list[tuple[str, float]]:
    """Load one image and classify it, keeping ``main`` readable."""
    tensor = load_image(path)
    return predict(model, tensor, device, topk)


if __name__ == "__main__":
    sys.exit(main())
