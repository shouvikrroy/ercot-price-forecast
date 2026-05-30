"""PatchTST model for HB_NORTH day-ahead prices."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import PatchTSTConfig, PatchTSTForPrediction

from src.utils import get_torch_device, inv_log_transform, log_transform

logger = logging.getLogger(__name__)

TARGET = "price_HB_NORTH"
CONTEXT_LEN = 512   # ~3 weeks of hourly history
PRED_LEN = 1        # one-step-ahead, consistent with all other models
PATCH_LEN = 16      # one patch = 16 hours
STRIDE = 8          # overlapping patches → 63 patches per window


# ─────────────────────────── Internal dataset ────────────────────────────────


class _WindowDataset(Dataset):
    """
    Sliding-window dataset over a 1-D standardised price series.
    Each sample: context window of length context_len → scalar target pred_len steps ahead.
    Channel dimension is added so shapes match HuggingFace PatchTST expectations:
        past_values:   (context_len, 1)
        future_values: (pred_len,    1)
    """

    def __init__(
        self, y: np.ndarray, context_len: int, pred_len: int
    ) -> None:
        self.y = torch.from_numpy(y).float()
        self.context_len = context_len
        self.pred_len = pred_len

    def __len__(self) -> int:
        return len(self.y) - self.context_len - self.pred_len + 1

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.y[idx : idx + self.context_len].unsqueeze(-1)
        target = self.y[idx + self.context_len : idx + self.context_len + self.pred_len].unsqueeze(-1)
        return x, target


# ─────────────────────────── Public model ────────────────────────────────────


class PatchTSTModel:
    """
    PatchTST for 1-step-ahead HB_NORTH price forecasting.

    The input is a single channel: log-transformed, mean/std standardised
    HB_NORTH price.  The 512-hour context window is divided into 63 overlapping
    patches of length 16 (stride 8), which a transformer encoder processes.

    Huber loss is computed outside the HuggingFace model to match the LSTM
    treatment.  The last context_len training values are stored to provide
    a full context window for the first test prediction.
    """

    def __init__(
        self,
        context_len: int = CONTEXT_LEN,
        pred_len: int = PRED_LEN,
        patch_len: int = PATCH_LEN,
        stride: int = STRIDE,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        ffn_dim: int = 256,
        dropout: float = 0.2,
        batch_size: int = 128,
        epochs: int = 30,
        lr: float = 1e-4,
        patience: int = 5,
        val_frac: float = 0.1,
    ) -> None:
        self.context_len = context_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self._device = get_torch_device()
        self._model: PatchTSTForPrediction | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._tail: np.ndarray | None = None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _normalise(self, y_log: np.ndarray) -> np.ndarray:
        return ((y_log - self._mean) / self._std).astype(np.float32)

    def _denormalise(self, y_norm: np.ndarray) -> np.ndarray:
        return (y_norm * self._std + self._mean).astype(np.float32)

    # ── Interface ─────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "PatchTSTModel":
        y_log = log_transform(train_df[TARGET].values).astype(np.float32)

        # Fit normalisation on training log-prices
        self._mean = float(y_log.mean())
        self._std = float(y_log.std()) + 1e-8
        y = self._normalise(y_log)

        # Save tail for seeding test predictions
        self._tail = y[-self.context_len :].copy()

        # Chronological train / val split
        n_val = max(self.context_len + self.pred_len, int(len(y) * self.val_frac))
        n_tr = len(y) - n_val
        y_tr = y[:n_tr]
        # Val context prefix: context_len rows from training — no lookahead
        y_val_ctx = y[n_tr - self.context_len :]

        tr_ds = _WindowDataset(y_tr, self.context_len, self.pred_len)
        val_ds = _WindowDataset(y_val_ctx, self.context_len, self.pred_len)
        tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, pin_memory=True)

        config = PatchTSTConfig(
            num_input_channels=1,
            context_length=self.context_len,
            prediction_length=self.pred_len,
            patch_length=self.patch_len,
            stride=self.stride,
            d_model=self.d_model,
            num_attention_heads=self.n_heads,
            num_hidden_layers=self.n_layers,
            ffn_dim=self.ffn_dim,
            dropout=self.dropout,
            head_dropout=self.dropout,
            positional_encoding_type="sincos",
            pooling_type="mean",
        )
        self._model = PatchTSTForPrediction(config).to(self._device)
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr)

        best_val_loss = float("inf")
        patience_count = 0
        best_state: dict | None = None

        for epoch in range(1, self.epochs + 1):
            # Training
            self._model.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                preds = self._model(past_values=xb).prediction_outputs
                F.huber_loss(preds, yb, delta=1.0).backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation
            self._model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    preds = self._model(past_values=xb).prediction_outputs
                    val_loss += F.huber_loss(preds, yb, delta=1.0).item() * len(yb)
            val_loss /= len(val_ds)

            logger.info(
                "PatchTST epoch %d/%d — val_loss=%.5f", epoch, self.epochs, val_loss
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in self._model.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        y_log = log_transform(test_df[TARGET].values).astype(np.float32)
        y_test = self._normalise(y_log)

        # Prepend training tail so every test row has a full context window
        y_full = np.concatenate([self._tail, y_test])
        n_test = len(y_test)

        # Build (n_test, context_len, 1) tensor
        windows = np.stack(
            [y_full[i : i + self.context_len] for i in range(n_test)]
        )[:, :, np.newaxis]  # (n_test, context_len, 1)

        self._model.eval()
        preds_norm: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n_test, self.batch_size):
                batch = torch.from_numpy(windows[i : i + self.batch_size]).to(self._device)
                out = self._model(past_values=batch).prediction_outputs
                # out: (batch, pred_len=1, channels=1) → extract scalar per sample
                preds_norm.append(out[:, 0, 0].cpu().numpy())

        preds_log = self._denormalise(np.concatenate(preds_norm))
        preds = inv_log_transform(preds_log)
        return pd.Series(preds, index=test_df.index, name=TARGET)
