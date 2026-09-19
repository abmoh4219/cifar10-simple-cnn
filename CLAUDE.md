# CLAUDE.md — project context

## What this is

A take-home assessment for an ML engineering role. The deliverable will be
walked through on a recorded screen share, so **structure and clarity matter
as much as correctness**. Keep code readable, modules single-purpose, and
avoid cleverness.

## Hard constraints from the spec

- **Model:** a SIMPLE CNN with **1–2 convolutional layers**. This limit must
  not be exceeded regardless of any accuracy benefit. Do not add a third conv
  layer, residual blocks, or a pretrained backbone.
- **Framework:** PyTorch (+ torchvision for the dataset and transforms).

## Engineering rules

- **Reproducibility:** every random source (Python `random`, NumPy, torch CPU,
  torch CUDA, DataLoader workers, the train/val split) is seeded from
  `CFG.seed`. No unseeded randomness anywhere.
- **Test-set discipline:** the CIFAR-10 test set is touched **exactly once**,
  in `src/evaluate.py`. Model selection (best checkpoint, early stopping,
  hyperparameter choices) uses the **validation split only**, which is carved
  out of the official training set.
- **Cold-start inference:** `src/predict.py` must load the checkpoint from
  disk in a **separate process** — it never receives an in-memory model from
  training. This proves the saved artifact is self-sufficient.
- **Configuration:** all hyperparameters and paths live in `src/config.py`
  (`CFG`). They are never hardcoded in any other module.

## Layout

| Path | Responsibility |
|---|---|
| `src/config.py` | Frozen `Config` dataclass, single instance `CFG` |
| `src/data.py` | Download, transforms, train/val/test loaders |
| `src/model.py` | `SimpleCNN` (≤ 2 conv layers) |
| `src/train.py` | Training loop, val-based checkpointing, curves |
| `src/evaluate.py` | The only test-set consumer; metrics + confusion matrix |
| `src/predict.py` | Standalone inference from checkpoint |
| `artifacts/` | Committed: checkpoint, plots, metrics |
| `samples/` | Example images for `predict.py` |
| `verify_env.py` | Environment check (versions, device, conv forward pass) |

## Workflow

Run modules from the repo root as packages so imports resolve:
`python -m src.train`, `python -m src.evaluate`, `python -m src.predict`.
