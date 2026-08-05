"""Torch device selection -- ONE implementation, shared by every trainer and by `play`.

There were three near-copies of this (train_rl, train_bc, play), all of them cuda-or-cpu, so a Mac ran
the whole pipeline on CPU with no indication that an Apple-silicon GPU was sitting idle. This is the
single place that decides, and it understands four settings:

    auto    prefer CUDA, then MPS (Apple silicon), then CPU        <- the default
    cuda    NVIDIA; falls back to CPU with an explanatory message if unusable
    mps     Apple silicon Metal; falls back to CPU the same way
    cpu     forced

EVERY BACKEND IS PROBED, NOT JUST QUERIED. `torch.cuda.is_available()` returning True does not mean the
build can actually run a kernel on this GPU -- an RTX 50-series card with a pre-Blackwell wheel reports
available and then throws on the first op. MPS has the same failure mode with an old torch. So each
candidate has an actual tensor op run on it before it is accepted, and a failure prints the fix rather
than silently degrading to CPU.

MPS CAVEAT worth knowing before you trust a run: Metal does not implement every op, and PyTorch raises
rather than falling back by default. If a training run dies with "not currently implemented for MPS",
set `PYTORCH_ENABLE_MPS_FALLBACK=1` (the offending op then runs on CPU, which is slower but correct).
"""
from __future__ import annotations

from typing import Optional

_VALID = ("auto", "cuda", "mps", "cpu")


def _probe(device: str) -> Optional[str]:
    """Run a real op on `device`; return the error string on failure, None on success."""
    import torch
    try:
        _ = (torch.zeros(1, device=device) + 1).item()
        return None
    except Exception as exc:  # noqa: BLE001
        return str(exc)


def _cuda_ready() -> bool:
    import torch
    return bool(torch.cuda.is_available())


def _mps_ready() -> bool:
    import torch
    return bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())


def describe(device: str) -> str:
    """A short human label for the chosen device (GPU name where we can get one)."""
    import torch
    if device == "cuda":
        try:
            return f" ({torch.cuda.get_device_name(0)})"
        except Exception:  # noqa: BLE001
            return " (cuda)"
    if device == "mps":
        return " (Apple silicon / Metal)"
    return ""


def pick_device(cfg, label: str = "train") -> str:
    """Resolve `train.device` to a usable torch device string.

    `label` is only used to tag the log lines with the caller (train-rl / train-bc / play).
    """
    import torch
    _ = torch.__version__      # fail here, not deep inside a probe, if torch is missing entirely

    want = str(cfg.get("train", "device", default="auto") or "auto").strip().lower()
    if want not in _VALID:
        print(f"[{label}] train.device={want!r} is not one of {_VALID}; using 'auto'.")
        want = "auto"

    if want == "cpu":
        return "cpu"

    if want == "auto":
        # Preference order is a throughput ordering, not a nicety: a discrete NVIDIA card beats
        # unified-memory Metal, and both beat CPU by enough that silently landing on CPU (which is
        # what the old cuda-or-nothing code did on a Mac) is a real waste rather than a small one.
        for cand in ("cuda", "mps"):
            ready = _cuda_ready() if cand == "cuda" else _mps_ready()
            if not ready:
                continue
            err = _probe(cand)
            if err is None:
                return cand
            print(f"[{label}] {cand} is reported available but cannot run an op ({err.splitlines()[0]}); "
                  f"trying the next device.")
        return "cpu"

    # an EXPLICIT request: say clearly why it was not honoured rather than quietly downgrading
    ready = _cuda_ready() if want == "cuda" else _mps_ready()
    if not ready:
        if want == "cuda":
            print(f"[{label}] train.device: cuda, but CUDA is not available -- using CPU. Install the "
                  f"CUDA build of torch (https://pytorch.org/get-started/locally/), or set "
                  f"train.device: auto.")
        else:
            print(f"[{label}] train.device: mps, but Metal is not available -- using CPU. MPS needs "
                  f"Apple silicon and torch >= 2.0, or set train.device: auto.")
        return "cpu"
    err = _probe(want)
    if err is None:
        return want
    if want == "cuda":
        print(f"[{label}] GPU detected but this torch build can't run kernels on it:\n    {err}\n"
              "  Newer GPUs need a matching build -- RTX 50-series (Blackwell) = CUDA 12.8:\n"
              "    pip install torch --index-url https://download.pytorch.org/whl/cu128\n"
              "  Falling back to CPU.")
    else:
        print(f"[{label}] Metal detected but this torch build can't run on it:\n    {err}\n"
              "  If this is a missing-op error, PYTORCH_ENABLE_MPS_FALLBACK=1 routes that op to CPU.\n"
              "  Falling back to CPU.")
    return "cpu"
