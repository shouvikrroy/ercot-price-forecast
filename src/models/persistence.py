"""Persistence (naive) baseline: forecast = last observed training value."""

from __future__ import annotations

import pandas as pd

TARGET = "price_HB_NORTH"


class PersistenceModel:
    """
    Naive baseline that repeats the final training observation for every
    test hour.  No transform needed — operates directly in $/MWh.
    Establishes the floor any real model should comfortably beat.
    """

    def __init__(self) -> None:
        self._last_value: float | None = None

    def fit(self, train_df: pd.DataFrame) -> "PersistenceModel":
        self._last_value = float(train_df[TARGET].iloc[-1])
        return self

    def predict(self, test_df: pd.DataFrame) -> pd.Series:
        return pd.Series(self._last_value, index=test_df.index, name=TARGET)
