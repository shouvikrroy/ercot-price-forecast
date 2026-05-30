"""Tests for src/evaluate.py."""

from __future__ import annotations

import numpy as np
import pytest

from src.evaluate import comparison_table, mae, mape, rmse, score


class TestMAE:
    def test_perfect_prediction(self):
        assert mae([1, 2, 3], [1, 2, 3]) == 0.0

    def test_constant_error(self):
        assert mae([0, 0, 0], [1, 1, 1]) == pytest.approx(1.0)

    def test_symmetric(self):
        assert mae([0, 1], [1, 0]) == mae([1, 0], [0, 1])


class TestRMSE:
    def test_perfect_prediction(self):
        assert rmse([1, 2, 3], [1, 2, 3]) == 0.0

    def test_penalises_large_errors(self):
        # RMSE should be higher than MAE when errors vary
        y_true = [10, 10, 10, 10]
        y_pred = [10, 10, 10, 50]  # one large spike
        assert rmse(y_true, y_pred) > mae(y_true, y_pred)

    def test_known_value(self):
        assert rmse([0, 0], [3, 4]) == pytest.approx(np.sqrt((9 + 16) / 2))


class TestMAPE:
    def test_perfect_prediction(self):
        assert mape([10, 20, 30], [10, 20, 30]) == pytest.approx(0.0)

    def test_excludes_near_zero(self):
        # Row with |y_true| < 1 should be ignored
        result_with = mape([0.5, 10, 20], [100, 10, 20])
        result_without = mape([10, 20], [10, 20])
        assert result_with == pytest.approx(result_without)

    def test_known_value(self):
        # 10% error on each: MAPE = 10%
        assert mape([100, 200], [110, 220]) == pytest.approx(10.0)


class TestScore:
    def test_returns_all_keys(self):
        result = score([10, 20], [10, 20], "TestModel")
        assert set(result.keys()) == {"model", "MAE", "RMSE", "MAPE (%)"}

    def test_model_name_stored(self):
        assert score([1], [1], "MyModel")["model"] == "MyModel"


class TestComparisonTable:
    def test_sorted_by_rmse(self):
        results = [
            {"model": "A", "MAE": 1.0, "RMSE": 5.0, "MAPE (%)": 10.0},
            {"model": "B", "MAE": 2.0, "RMSE": 3.0, "MAPE (%)": 20.0},
        ]
        table = comparison_table(results)
        assert table.index[0] == "B"  # lower RMSE first
