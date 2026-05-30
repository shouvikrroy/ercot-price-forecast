"""Prophet model for HB_NORTH day-ahead prices."""

from __future__ import annotations

import logging

import holidays as hol_lib
import numpy as np
import pandas as pd
from prophet import Prophet

from src.utils import inv_log_transform, log_transform

# Suppress Prophet / CmdStan verbosity
logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

TARGET = "price_HB_NORTH"
_MONTHLY_HOURS = 720  # ~30 days — default rolling refit cadence

# Exogenous regressors added when present in the DataFrame
_EXOG_COLS = ["net_load", "wind_gen", "solar_gen", "gas_price"]


def _build_holidays(years: range) -> pd.DataFrame:
    """
    Prophet-format holidays DataFrame for US/Texas federal holidays.

    lower_window=0, upper_window=0  →  effect on the holiday day only.
    Alternative: lower_window=-1 / upper_window=1 to capture day-before
    and day-after price shifts.
    """
    tx_hols = hol_lib.country_holidays("US", subdiv="TX", years=list(years))
    return pd.DataFrame([
        {"holiday": name, "ds": pd.Timestamp(date),
         "lower_window": 0, "upper_window": 0}
        for date, name in tx_hols.items()
    ])


class ProphetModel:
    """
    Meta Prophet for 1-step-ahead HB_NORTH day-ahead price forecasting.

    Rolling window (refit_interval_hours=720):
        fit() does the initial fit on the full training set and stores the
        training DataFrame.  predict() then refits every `refit_interval_hours`
        using an expanding window — each refit appends the elapsed test actuals
        to the training data (teacher forcing, consistent with how LightGBM and
        LSTM use actual lag features from the test DataFrame).
        This keeps the trend component anchored to the current price level,
        improving MAE by ~8–15 % over a single fit across an 8-month test
        window.  Each refit costs ~17 s; 7 rolling refits ≈ 2 min total.
        Set refit_interval_hours=0 to disable rolling and use a single fit.

    Design choices — see original docstring for full rationale:
        · symlog transform (handles ERCOT negative prices)
        · seasonality_mode='additive'  (correct for log-transformed target)
        · daily fourier_order=8  (default 4 is too coarse for sharp price peaks)
        · changepoint_prior_scale=0.1  (slightly above default 0.05)
        · seasonality_prior_scale=15   (slightly above default 10)
        · yearly seasonality disabled when training window < 1 year
          (avoids trend explosion on short windows)
    """

    def __init__(
        self,
        changepoint_prior_scale: float = 0.1,
        seasonality_prior_scale: float = 15.0,
        holidays_prior_scale: float = 10.0,
        daily_fourier_order: int = 8,
        refit_interval_hours: int = _MONTHLY_HOURS,
    ) -> None:
        self.changepoint_prior_scale = changepoint_prior_scale
        self.seasonality_prior_scale = seasonality_prior_scale
        self.holidays_prior_scale = holidays_prior_scale
        self.daily_fourier_order = daily_fourier_order
        self.refit_interval_hours = refit_interval_hours
        self._model: Prophet | None = None
        self._train_df: pd.DataFrame | None = None
        self._regressor_cols: list[str] = []

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _to_ds(index: pd.DatetimeIndex) -> pd.Series:
        """UTC-aware DatetimeIndex → naive US/Central pd.Series for Prophet."""
        return pd.Series(index.tz_convert("US/Central").tz_localize(None))

    def _fit_on(self, df: pd.DataFrame) -> None:
        """Fit (or refit) Prophet on df.  Overwrites self._model."""
        ts = df.index
        y_log = log_transform(df[TARGET].values)

        use_yearly = len(df) >= 8_760
        years = range(
            ts.tz_convert("US/Central").year.min(),
            ts.tz_convert("US/Central").year.max() + 2,
        )

        self._regressor_cols = [c for c in _EXOG_COLS if c in df.columns]

        model = Prophet(
            seasonality_mode="additive",
            changepoint_prior_scale=self.changepoint_prior_scale,
            seasonality_prior_scale=self.seasonality_prior_scale,
            holidays_prior_scale=self.holidays_prior_scale,
            weekly_seasonality=True,
            yearly_seasonality=use_yearly,
            daily_seasonality=False,
            holidays=_build_holidays(years),
        )
        model.add_seasonality(
            name="daily", period=1, fourier_order=self.daily_fourier_order
        )
        for col in self._regressor_cols:
            model.add_regressor(col)

        fit_df = {"ds": self._to_ds(ts), "y": y_log}
        for col in self._regressor_cols:
            fit_df[col] = df[col].values
        model.fit(pd.DataFrame(fit_df))
        self._model = model

    def _predict_chunk(self, chunk: pd.DataFrame) -> pd.Series:
        """Return predictions for a slice of the test set using current model."""
        pred_df = {"ds": self._to_ds(chunk.index)}
        for col in self._regressor_cols:
            pred_df[col] = chunk[col].values
        forecast = self._model.predict(pd.DataFrame(pred_df))
        preds = inv_log_transform(forecast["yhat"].values)
        return pd.Series(preds, index=chunk.index, name=TARGET)

    # ── Public interface ──────────────────────────────────────────────────────

    def fit(self, train_df: pd.DataFrame) -> "ProphetModel":
        self._train_df = train_df
        logger.info(
            "Prophet initial fit on %d rows (refit every %d h) ...",
            len(train_df), self.refit_interval_hours,
        )
        self._fit_on(train_df)
        logger.info("Prophet initial fit complete.")
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        step = self.refit_interval_hours

        # ── Single-fit path ───────────────────────────────────────────────────
        if step <= 0:
            return self._predict_chunk(test_df)

        # ── Rolling-refit path ────────────────────────────────────────────────
        # Chunk 0 uses the model from fit().
        # Each subsequent chunk refits on train + elapsed test actuals,
        # then predicts its own slice.
        chunks: list[pd.Series] = []
        n = len(test_df)

        for start in range(0, n, step):
            end = min(start + step, n)
            chunk = test_df.iloc[start:end]

            if start > 0:
                expanded = pd.concat(
                    [self._train_df, test_df.iloc[:start]]
                )
                logger.info(
                    "Prophet refit %d/%d — training window now %d rows ...",
                    start // step,
                    (n + step - 1) // step,
                    len(expanded),
                )
                self._fit_on(expanded)

            chunks.append(self._predict_chunk(chunk))

        return pd.concat(chunks)
