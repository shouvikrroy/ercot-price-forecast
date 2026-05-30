"""LightGBM model for HB_NORTH day-ahead prices."""

from __future__ import annotations

import lightgbm as lgb
import pandas as pd

from src.utils import get_lgbm_device, inv_log_transform, log_transform

TARGET = "price_HB_NORTH"


def _feature_cols(df: pd.DataFrame) -> list[str]:
    """All lag, rolling, and calendar columns — excludes raw price columns."""
    return [c for c in df.columns if not c.startswith("price_")]


class LGBMModel:
    """
    LightGBM regressor on log-transformed HB_NORTH prices.

    Features are the lag (1h, 24h, 168h), rolling mean (24h, 168h), and
    calendar columns already present in the processed DataFrame.  Price
    columns for other hubs are excluded to keep the target well-defined;
    their information is already captured in the shared lag features.

    The log transform is applied to the target only — LightGBM's tree splits
    are rank-based so feature spikes don't distort splits.
    """

    def __init__(self) -> None:
        self._model: lgb.LGBMRegressor | None = None
        self._feature_cols: list[str] | None = None
        self._device: str = get_lgbm_device()

    def fit(self, train_df: pd.DataFrame) -> "LGBMModel":
        self._feature_cols = _feature_cols(train_df)
        X = train_df[self._feature_cols]
        y = log_transform(train_df[TARGET].values)

        self._model = lgb.LGBMRegressor(
            n_estimators=1_000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            device=self._device,
        )
        self._model.fit(X, y)
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        X = test_df[self._feature_cols]
        preds_log = self._model.predict(X)
        preds = inv_log_transform(preds_log)
        return pd.Series(preds, index=test_df.index, name=TARGET)
