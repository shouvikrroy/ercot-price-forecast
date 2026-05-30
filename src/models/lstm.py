"""LSTM model for HB_NORTH day-ahead prices."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from src.utils import get_torch_device, inv_log_transform, log_transform

logger = logging.getLogger(__name__)

TARGET = "price_HB_NORTH"
PRICE_COLS = ["price_HB_HOUSTON", "price_HB_NORTH", "price_HB_SOUTH", "price_HB_WEST"]
CALENDAR_COLS = ["hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"]
EXOG_COLS = ["net_load", "wind_gen", "solar_gen", "gas_price"]
FEATURE_COLS = PRICE_COLS + CALENDAR_COLS + EXOG_COLS  # 13 input features per timestep

SEQ_LEN = 168  # one-week lookback


# ─────────────────────────── Internal dataset ────────────────────────────────


class _SequenceDataset(Dataset):
    """Sliding-window dataset: input window of length seq_len → scalar target."""

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int) -> None:
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.y) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx : idx + self.seq_len], self.y[idx + self.seq_len]


# ─────────────────────────── Network ─────────────────────────────────────────


class _LSTMNet(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, num_layers: int, dropout: float
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)


# ─────────────────────────── Public model ────────────────────────────────────


class LSTMModel:
    """
    Two-layer LSTM for 1-step-ahead HB_NORTH price forecasting.

    Input per timestep: log-transformed prices for all four hubs plus five
    calendar features (9 total).  A StandardScaler is fit on training data
    and applied at both fit and predict time.

    The last seq_len rows of training data are stored so that the first test
    predictions have a full week of context without lookahead.
    """

    def __init__(
        self,
        seq_len: int = SEQ_LEN,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        batch_size: int = 512,
        epochs: int = 50,
        lr: float = 1e-3,
        patience: int = 5,
        val_frac: float = 0.1,
    ) -> None:
        self.seq_len = seq_len
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.patience = patience
        self.val_frac = val_frac
        self._device = get_torch_device()
        self._net: _LSTMNet | None = None
        self._scaler: StandardScaler | None = None
        self._tail: np.ndarray | None = None
        self._active_cols: list[str] = []

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_features(self, df: pd.DataFrame) -> np.ndarray:
        """Return float32 array with price columns log-transformed. Exog cols used as-is (StandardScaler normalises them)."""
        active_cols = [c for c in FEATURE_COLS if c in df.columns]
        X = df[active_cols].values.copy().astype(np.float64)
        for i, col in enumerate(active_cols):
            if col in PRICE_COLS:
                X[:, i] = log_transform(X[:, i])
        return X.astype(np.float32)

    # ── Interface ─────────────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "LSTMModel":
        self._active_cols = [c for c in FEATURE_COLS if c in train_df.columns]
        X_raw = self._build_features(train_df)
        y = log_transform(train_df[TARGET].values).astype(np.float32)

        self._scaler = StandardScaler().fit(X_raw)
        X = self._scaler.transform(X_raw).astype(np.float32)

        # Save tail for seeding test predictions
        self._tail = X[-self.seq_len :].copy()

        # Chronological train / val split
        n_val = max(self.seq_len + 1, int(len(X) * self.val_frac))
        n_tr = len(X) - n_val
        X_tr, y_tr = X[:n_tr], y[:n_tr]
        # Val context includes seq_len training rows as prefix — no lookahead
        X_val_ctx = X[n_tr - self.seq_len :]
        y_val_ctx = y[n_tr - self.seq_len :]

        tr_ds = _SequenceDataset(X_tr, y_tr, self.seq_len)
        val_ds = _SequenceDataset(X_val_ctx, y_val_ctx, self.seq_len)
        tr_dl = DataLoader(tr_ds, batch_size=self.batch_size, shuffle=True, pin_memory=True)
        val_dl = DataLoader(val_ds, batch_size=self.batch_size, pin_memory=True)

        self._net = _LSTMNet(
            len(self._active_cols), self.hidden_size, self.num_layers, self.dropout
        ).to(self._device)
        optimizer = torch.optim.Adam(self._net.parameters(), lr=self.lr)
        criterion = nn.HuberLoss(delta=1.0)

        best_val_loss = float("inf")
        patience_count = 0
        best_state: dict | None = None

        for epoch in range(1, self.epochs + 1):
            # Training
            self._net.train()
            for xb, yb in tr_dl:
                xb, yb = xb.to(self._device), yb.to(self._device)
                optimizer.zero_grad()
                loss = criterion(self._net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), max_norm=1.0)
                optimizer.step()

            # Validation
            self._net.eval()
            val_loss = 0.0
            with torch.no_grad():
                for xb, yb in val_dl:
                    xb, yb = xb.to(self._device), yb.to(self._device)
                    val_loss += criterion(self._net(xb), yb).item() * len(yb)
            val_loss /= len(val_ds)

            logger.info("LSTM epoch %d/%d — val_loss=%.5f", epoch, self.epochs, val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_count = 0
                best_state = {k: v.cpu().clone() for k, v in self._net.state_dict().items()}
            else:
                patience_count += 1
                if patience_count >= self.patience:
                    logger.info("Early stopping at epoch %d", epoch)
                    break

        if best_state is not None:
            self._net.load_state_dict(best_state)
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        X_raw = self._build_features(test_df)
        X_test = self._scaler.transform(X_raw).astype(np.float32)

        # Prepend training tail so every test row has a full seq_len context
        X_full = np.vstack([self._tail, X_test])
        n_test = len(X_test)
        seqs = np.stack([X_full[i : i + self.seq_len] for i in range(n_test)])

        self._net.eval()
        preds_log: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n_test, self.batch_size):
                batch = torch.from_numpy(seqs[i : i + self.batch_size]).to(self._device)
                preds_log.append(self._net(batch).cpu().numpy())

        preds = inv_log_transform(np.concatenate(preds_log))
        return pd.Series(preds, index=test_df.index, name=TARGET)
