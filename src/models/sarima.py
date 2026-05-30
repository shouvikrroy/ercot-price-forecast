"""SARIMA model for HB_NORTH day-ahead prices, tuned with pmdarima.auto_arima."""

from __future__ import annotations

import logging
import warnings

import numpy as np
import pandas as pd
import pmdarima as pm
import statsmodels.api as sm

from src.utils import inv_log_transform, log_transform

logger = logging.getLogger(__name__)

TARGET = "price_HB_NORTH"

_MAX_TRAIN_HOURS = 8_760   # auto_arima training window (1 year)
_REFIT_WINDOW    = 336     # rolling refit window: 2 weeks of hourly data
_STEP            = 24      # refit once per day, forecast 24 h ahead each time


class SARIMAModel:
    """
    Rolling-window SARIMA for 1-step-ahead HB_NORTH price forecasting.

    fit():
        Runs auto_arima once on the most recent year of training data
        (log-transformed, m=24) to select the best (p,d,q)(P,D,Q) order.
        The fitted order is stored; parameters are NOT carried forward.

    predict():
        Slides a 2-week window over the test set, refitting a SARIMAX model
        with the fixed order every 24 hours and forecasting the next 24 hours.
        Actual observed prices are used to advance the window (teacher forcing),
        consistent with how LightGBM uses precomputed lag features from the
        test DataFrame.
    """

    def __init__(
        self,
        max_train_hours: int = _MAX_TRAIN_HOURS,
        refit_window: int = _REFIT_WINDOW,
        step: int = _STEP,
    ) -> None:
        self._max_train_hours = max_train_hours
        self._refit_window = refit_window
        self._step = step
        self._order: tuple | None = None
        self._seasonal_order: tuple | None = None
        self._tail_log: np.ndarray | None = None  # last refit_window rows of training

    def fit(self, train_df: pd.DataFrame) -> "SARIMAModel":
        series = train_df[TARGET].iloc[-self._max_train_hours :]
        y = log_transform(series.values)

        logger.info("Running auto_arima (m=24, stepwise) on %d pts ...", len(y))
        arima = pm.auto_arima(
            y,
            seasonal=True,
            m=24,
            max_p=3,
            max_q=3,
            max_P=1,
            max_Q=1,
            d=None,
            D=None,
            information_criterion="aic",
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
        )
        self._order = arima.order
        self._seasonal_order = arima.seasonal_order
        self._tail_log = y[-self._refit_window :].copy()

        logger.info(
            "Selected order: ARIMA%s x %s[24]", self._order, self._seasonal_order[:3]
        )
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        y_test_log = log_transform(test_df[TARGET].values).astype(np.float64)
        n_test = len(test_df)

        # Full series available at prediction time: training tail + test actuals
        y_all = np.concatenate([self._tail_log, y_test_log])
        tail_len = len(self._tail_log)

        preds_log = np.empty(n_test)

        for start in range(0, n_test, self._step):
            end = min(start + self._step, n_test)
            h = end - start

            win_end = tail_len + start
            win_start = max(0, win_end - self._refit_window)
            y_window = y_all[win_start:win_end]

            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    result = sm.tsa.SARIMAX(
                        y_window,
                        order=self._order,
                        seasonal_order=self._seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    ).fit(disp=False, method="lbfgs", maxiter=100)
                preds_log[start:end] = result.forecast(h)
            except Exception as exc:
                logger.warning("SARIMAX refit failed at step %d: %s", start, exc)
                preds_log[start:end] = y_all[win_end - 1]  # fallback: last observed

            if start % 240 == 0:
                logger.info(
                    "SARIMA rolling predict: %d / %d steps done", start, n_test
                )

        preds = inv_log_transform(preds_log)
        return pd.Series(preds, index=test_df.index, name=TARGET)
