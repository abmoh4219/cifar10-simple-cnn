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

## Inference

`predict.py` runs as its own process and cold-loads the checkpoint from disk —
nothing is carried over from training. It reuses the evaluation transform from
`src/data.py` rather than redefining the normalisation constants.

```bash
python -m src.predict                                    # all images in samples/
python -m src.predict --images path/to/cat.png --topk 5  # specific files
python -m src.predict --save-figure artifacts/predictions.png
```

```
device     : cpu
checkpoint : artifacts/best_model.pt
  trained to epoch 19, val_acc 0.7480
images     : 10

samples/sample_3_cat.png
  true: cat        [CORRECT]
    1. cat         85.04%
    2. dog          7.89%
    3. frog         4.11%

samples/sample_4_deer.png
  true: deer       [WRONG]
    1. bird        29.18%
    2. deer        28.00%
    3. ship        20.30%

accuracy on 10 labelled image(s): 8/10 = 0.8000
```

Images that are not 32x32 are resized, with a printed notice. Greyscale and
transparent images are converted to RGB. The true label shown is parsed from
the `sample_<idx>_<classname>.png` filename for display only — it never
reaches the model.

## Results

(filled in after training)
