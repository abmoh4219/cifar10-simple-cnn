# CIFAR-10 Simple CNN

A small, fully reproducible PyTorch pipeline that trains a 2-conv-layer CNN on CIFAR-10, evaluates it once on the held-out test set, and runs inference on new images from a saved checkpoint.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python verify_env.py
```

## Project Structure

```
cifar10-simple-cnn/
├── CLAUDE.md            # Project constraints and context for AI-assisted sessions
├── README.md            # This file
├── requirements.txt     # Runtime dependencies (minimum-version pins)
├── .gitignore
├── verify_env.py        # Prints versions + device, runs a one-off conv forward pass
├── src/
│   ├── __init__.py
│   ├── config.py        # All hyperparameters, paths, and dataset constants (CFG)
│   ├── data.py          # Download, transforms, train/val/test DataLoaders
│   ├── model.py         # SimpleCNN definition (≤ 2 conv layers)
│   ├── train.py         # Training loop, val-based checkpointing, learning curves
│   ├── evaluate.py      # One-time test-set evaluation + confusion matrix
│   └── predict.py       # Standalone inference on image files from the checkpoint
├── artifacts/           # Committed outputs: best_model.pt, plots, metrics
└── samples/             # Example images for predict.py
```

## Results

(filled in after training)
