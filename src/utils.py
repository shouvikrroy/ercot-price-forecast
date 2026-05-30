"""Shared utilities for the model layer."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ─────────────────────────── Hardware detection ──────────────────────────────


def get_torch_device() -> "torch.device":
    """
    Return the best available torch.device: CUDA > MPS > CPU.
    Logs which device was selected and, for CUDA, the GPU name.
    Used by LSTM, PatchTST, and Chronos.
    """
    import torch

    if torch.cuda.is_available():
        device = torch.device("cuda")
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        logger.info("CUDA available — using GPU: %s (%.1f GB VRAM)", name, vram)
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Apple MPS available — using MPS device")
    else:
        device = torch.device("cpu")
        logger.info("No GPU found — using CPU")

    return device


def get_lgbm_device() -> str:
    """
    Return 'cuda' if LightGBM was compiled with CUDA support, else 'cpu'.
    LightGBM GPU support is not included in the standard pip wheel — this
    probes a tiny fit so the fallback is silent rather than a crash.
    """
    try:
        import lightgbm as lgb
        import numpy as np

        X = np.random.default_rng(0).random((100, 4))
        y = np.random.default_rng(0).random(100)
        lgb.LGBMRegressor(n_estimators=1, device="cuda", verbose=-1).fit(X, y)
        logger.info("LightGBM CUDA available — using GPU")
        return "cuda"
    except Exception:
        logger.info("LightGBM CUDA not available — using CPU")
        return "cpu"


def log_transform(x: np.ndarray | pd.Series) -> np.ndarray:
    """
    Signed log1p transform: sign(x) * log(1 + |x|).

    Works for all real values including ERCOT negative prices.
    Compresses spike magnitude while preserving sign and ordering.
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.log1p(np.abs(x))


def inv_log_transform(x: np.ndarray | pd.Series) -> np.ndarray:
    """Inverse of log_transform: sign(x) * (exp(|x|) - 1)."""
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.expm1(np.abs(x))
