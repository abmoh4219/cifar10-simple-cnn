"""Environment sanity check. Run this before anything else.

Prints interpreter / library versions and the selected compute device, then
performs one forward pass of a throwaway 3x3 convolution on that device.
Exit code 0 on success, 1 with a clear message on failure.
"""

import sys


def select_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> int:
    print(f"Python      : {sys.version.split()[0]}")

    try:
        import torch
        import torchvision
    except ImportError as exc:
        print(f"FAIL: missing dependency — {exc}. Run: pip install -r requirements.txt")
        return 1

    print(f"torch       : {torch.__version__}")
    print(f"torchvision : {torchvision.__version__}")

    device = select_device()
    print(f"device      : {device}")

    try:
        conv = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, padding=1).to(device)
        x = torch.randn(1, 3, 32, 32, device=device)
        with torch.no_grad():
            y = conv(x)
        print(f"conv output : {tuple(y.shape)}")
    except Exception as exc:  # noqa: BLE001 — report anything that breaks the forward pass
        print(f"FAIL: forward pass on {device} raised {type(exc).__name__}: {exc}")
        return 1

    print("OK: environment is ready.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
