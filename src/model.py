"""Model definition: a simple CNN with EXACTLY two convolutional layers.

The two-conv limit is a hard constraint from the assessment brief. It is not
exceeded here regardless of the accuracy a deeper network would give.
"""

import time

import torch
from torch import nn

from src.config import CFG

# Spatial geometry implied by the architecture: 32x32 input, two 2x2 max-pools.
INPUT_SIZE = 32
IN_CHANNELS = 3
CONV1_CHANNELS = 32
CONV2_CHANNELS = 64
FEATURE_SIZE = INPUT_SIZE // 2 // 2  # 8
DROPOUT = 0.25


class SimpleCNN(nn.Module):
    """Two conv blocks followed by a single linear classifier.

    ``features`` and ``classifier`` are kept as separate ``nn.Sequential``
    modules so the 64x8x8 feature map can be inspected on its own and the
    head can be swapped or probed without touching the convolutions.
    """

    def __init__(self, num_classes: int = len(CFG.classes)) -> None:
        super().__init__()
        # BatchNorm sits *before* ReLU so it normalises the full pre-activation
        # distribution (both signs). Placing it after ReLU would normalise a
        # half-rectified, non-negative signal, which weakens its effect.
        self.features = nn.Sequential(
            # Block 1: 3x32x32 -> 32x16x16
            nn.Conv2d(IN_CHANNELS, CONV1_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(CONV1_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            # Block 2: 32x16x16 -> 64x8x8
            nn.Conv2d(CONV1_CHANNELS, CONV2_CHANNELS, kernel_size=3, padding=1),
            nn.BatchNorm2d(CONV2_CHANNELS),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Dropout on the flattened 4096-d map: the linear layer is where
            # almost all the parameters live, so it is where overfitting starts.
            nn.Dropout(DROPOUT),
            nn.Linear(CONV2_CHANNELS * FEATURE_SIZE * FEATURE_SIZE, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits of shape ``(N, num_classes)``.

        The shape check is explicit because a silently-broadcast or resized
        input would still run through the convs and only fail at the Linear
        layer with an unhelpful "mat1 and mat2 shapes cannot be multiplied".
        """
        expected = (IN_CHANNELS, INPUT_SIZE, INPUT_SIZE)
        if x.ndim != 4 or tuple(x.shape[1:]) != expected:
            raise ValueError(
                f"SimpleCNN expects input of shape (N, {IN_CHANNELS}, {INPUT_SIZE}, "
                f"{INPUT_SIZE}), got {tuple(x.shape)}"
            )
        return self.classifier(self.features(x))


def count_parameters(model: nn.Module) -> tuple[int, int]:
    """Return ``(total, trainable)`` parameter counts.

    Reported separately because a frozen backbone or a BatchNorm running
    statistic would make the two differ; for this model they should match.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def build_model() -> nn.Module:
    """Construct the model on CPU; callers move it to their device."""
    return SimpleCNN(num_classes=len(CFG.classes))


if __name__ == "__main__":
    from src.data import get_dataloaders
    from src.utils import get_device, set_seed

    set_seed(CFG.seed)
    device = get_device()
    model = build_model().to(device)

    print(f"device: {device}")
    print(model)

    total, trainable = count_parameters(model)
    print(f"\nparameters: total={total:,}  trainable={trainable:,}")

    # Forward pass on a small random batch — checks the wiring end to end.
    x = torch.randn(4, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)
    model.eval()
    with torch.no_grad():
        out = model(x)
        feats = model.features(x)
    assert out.shape == (4, len(CFG.classes)), out.shape
    print(f"\nlogits shape   : {tuple(out.shape)}")
    print(f"features shape : {tuple(feats.shape)}  -> Linear in_features = {feats[0].numel()}")

    # Shape guard: a wrong input must fail loudly with a readable message.
    try:
        model(torch.randn(4, IN_CHANNELS, 28, 28, device=device))
        raise AssertionError("shape guard did not fire")
    except ValueError as exc:
        print(f"\nshape guard OK : {exc}")

    # Single-batch overfit test. If the model, loss, and optimiser are wired
    # correctly, it must be able to memorise one batch completely. Failing
    # this means a bug, not a capacity problem — catch it before training.
    print("\n--- single-batch overfit test ---")
    set_seed(CFG.seed)
    train_loader, _, _ = get_dataloaders()
    xb, yb = next(iter(train_loader))
    xb, yb = xb.to(device), yb.to(device)

    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)
    criterion = nn.CrossEntropyLoss()
    n_steps = 100
    model.train()
    acc = 0.0
    for step in range(1, n_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        acc = (logits.argmax(dim=1) == yb).float().mean().item()
        if step % 20 == 0 or step == 1:
            print(f"step {step:>3}  loss={loss.item():.4f}  batch_acc={acc:.3f}")
    verdict = "PASS" if acc == 1.0 else "FAIL"
    print(f"overfit test: {verdict} (final batch accuracy {acc:.3f})")

    # Throughput: one forward+backward on a full batch, to estimate epoch time.
    model.train()
    xb_full = torch.randn(CFG.batch_size, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE, device=device)
    yb_full = torch.randint(0, len(CFG.classes), (CFG.batch_size,), device=device)
    for _ in range(3):  # warm-up so lazy init / allocator don't skew the timing
        criterion(model(xb_full), yb_full).backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    n_timed = 10
    t0 = time.perf_counter()
    for _ in range(n_timed):
        optimizer.zero_grad(set_to_none=True)
        criterion(model(xb_full), yb_full).backward()
        optimizer.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_timed * 1000
    steps_per_epoch = len(train_loader)
    print(
        f"\ntrain step (batch={CFG.batch_size}): {ms:.1f} ms  "
        f"-> ~{ms * steps_per_epoch / 1000:.0f}s per epoch of {steps_per_epoch} steps"
    )
