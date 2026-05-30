"""Tests for src/pipeline.py — all fixtures use real ERCOT data (Q1 2023)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.pipeline import HUBS, make_splits, run_pipeline


# ─────────────────────────── clean_data ──────────────────────────────────────


class TestCleanData:
    def test_price_columns_present(self, prices_df: pd.DataFrame) -> None:
        for hub in HUBS:
            assert f"price_{hub}" in prices_df.columns

    def test_continuous_hourly_index(self, prices_df: pd.DataFrame) -> None:
        expected = pd.date_range(
            prices_df.index.min(),
            prices_df.index.max(),
            freq="h",
            tz="UTC",
            name="timestamp",
        )
        pd.testing.assert_index_equal(prices_df.index, expected)

    def test_no_missing_prices(self, prices_df: pd.DataFrame) -> None:
        assert not prices_df.isna().any().any()

    def test_index_is_utc(self, prices_df: pd.DataFrame) -> None:
        assert prices_df.index.tz is not None
        assert str(prices_df.index.tz) == "UTC"

    def test_rejects_missing_columns(self, raw_df: pd.DataFrame) -> None:
        from src.pipeline import clean_data
        bad = raw_df.drop(columns=["SPP"])
        with pytest.raises(ValueError, match="missing required columns"):
            clean_data(bad)

    def test_rejects_unknown_hubs(self, raw_df: pd.DataFrame) -> None:
        from src.pipeline import clean_data
        bad = raw_df.copy()
        bad["Location"] = "HB_FAKE"
        with pytest.raises(ValueError, match="No rows found for hubs"):
            clean_data(bad)


# ─────────────────────────── engineer_features ───────────────────────────────


class TestEngineerFeatures:
    def test_no_missing_values(self, features_df: pd.DataFrame) -> None:
        nan_cols = features_df.columns[features_df.isna().any()].tolist()
        assert not nan_cols, f"NaN found in: {nan_cols}"

    def test_price_columns_present(self, features_df: pd.DataFrame) -> None:
        for hub in HUBS:
            assert f"price_{hub}" in features_df.columns

    def test_temporal_columns_present(self, features_df: pd.DataFrame) -> None:
        for col in ("hour_of_day", "day_of_week", "month", "is_weekend", "is_holiday"):
            assert col in features_df.columns, f"Missing: {col}"

    def test_lag_columns_present(self, features_df: pd.DataFrame) -> None:
        for hub in HUBS:
            for lag in (1, 24, 168):
                assert f"lag_{lag}h_{hub}" in features_df.columns

    def test_rolling_columns_present(self, features_df: pd.DataFrame) -> None:
        for hub in HUBS:
            for win in (24, 168):
                assert f"rolling_mean_{win}h_{hub}" in features_df.columns

    def test_hour_of_day_range(self, features_df: pd.DataFrame) -> None:
        assert features_df["hour_of_day"].between(0, 23).all()

    def test_is_weekend_binary(self, features_df: pd.DataFrame) -> None:
        assert set(features_df["is_weekend"].unique()).issubset({0, 1})

    def test_is_holiday_binary(self, features_df: pd.DataFrame) -> None:
        assert set(features_df["is_holiday"].unique()).issubset({0, 1})

    def test_lag_1h_matches_shifted_price(self, features_df: pd.DataFrame) -> None:
        hub = HUBS[0]
        price = features_df[f"price_{hub}"].values
        lag1 = features_df[f"lag_1h_{hub}"].values
        # lag_1h at row i must equal price at row i-1
        np.testing.assert_allclose(lag1[1:], price[:-1])

    def test_mlk_day_is_holiday(self, features_df: pd.DataFrame) -> None:
        # MLK Day 2023 = January 16 — well after the 7-day lag warmup period
        mlk = features_df[
            features_df.index.tz_convert("US/Central").normalize()
            == pd.Timestamp("2023-01-16", tz="US/Central")
        ]
        assert not mlk.empty, "MLK Day 2023-01-16 not found in features_df"
        assert (mlk["is_holiday"] == 1).all()


# ─────────────────────────── make_splits ─────────────────────────────────────


class TestMakeSplits:
    def test_non_overlapping(self, features_df: pd.DataFrame) -> None:
        train, val, test = make_splits(features_df)
        assert len(set(train.index) & set(val.index)) == 0
        assert len(set(train.index) & set(test.index)) == 0
        assert len(set(val.index) & set(test.index)) == 0

    def test_chronological_order(self, features_df: pd.DataFrame) -> None:
        train, val, test = make_splits(features_df)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_covers_all_rows(self, features_df: pd.DataFrame) -> None:
        train, val, test = make_splits(features_df)
        assert len(train) + len(val) + len(test) == len(features_df)

    def test_approximate_fractions(self, features_df: pd.DataFrame) -> None:
        train, val, test = make_splits(features_df)
        n = len(features_df)
        assert abs(len(train) / n - 0.70) < 0.02
        assert abs(len(val) / n - 0.15) < 0.02

    def test_custom_fractions(self, features_df: pd.DataFrame) -> None:
        train, val, test = make_splits(features_df, train_frac=0.60, val_frac=0.20)
        assert train.index.max() < val.index.min()
        assert val.index.max() < test.index.min()

    def test_splits_are_sorted(self, features_df: pd.DataFrame) -> None:
        train, val, test = make_splits(features_df)
        assert train.index.is_monotonic_increasing
        assert val.index.is_monotonic_increasing
        assert test.index.is_monotonic_increasing


# ─────────────────────────── run_pipeline (end-to-end) ───────────────────────


class TestRunPipeline:
    def test_end_to_end(self, raw_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
        import src.pipeline as pipe

        monkeypatch.setattr(pipe, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(pipe, "PROCESSED_PATH", tmp_path / "ercot_processed.parquet")

        result = pipe.run_pipeline(raw_df=raw_df)

        assert isinstance(result, pd.DataFrame)
        assert not result.isna().any().any(), "Pipeline output contains NaN"
        assert (tmp_path / "ercot_processed.parquet").exists()

    def test_output_columns(self, raw_df: pd.DataFrame, tmp_path, monkeypatch) -> None:
        import src.pipeline as pipe

        monkeypatch.setattr(pipe, "PROCESSED_DIR", tmp_path)
        monkeypatch.setattr(pipe, "PROCESSED_PATH", tmp_path / "ercot_processed.parquet")

        result = pipe.run_pipeline(raw_df=raw_df)

        for hub in HUBS:
            assert f"price_{hub}" in result.columns
            assert f"lag_168h_{hub}" in result.columns
            assert f"rolling_mean_168h_{hub}" in result.columns
        for col in ("hour_of_day", "is_weekend", "is_holiday"):
            assert col in result.columns
