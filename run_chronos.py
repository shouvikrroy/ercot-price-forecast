"""Run Chronos (zero-shot) only and compare against saved benchmark results."""

from __future__ import annotations

import logging
import time

import pandas as pd

from src.evaluate import score
from src.models.chronos_eval import ChronosModel
from src.pipeline import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/ercot_processed.parquet"

# Benchmark numbers from prior full evaluation run (test set: ~5,750 hours)
BENCHMARK = [
    {"model": "LightGBM",         "MAE": 6.7875,  "RMSE": 36.9458,  "MAPE (%)": 17.8428},
    {"model": "LSTM (avg 5)",     "MAE": 7.4810,  "RMSE": 41.8275,  "MAPE (%)": 17.9223},
    {"model": "PatchTST (avg 3)", "MAE": 7.9469,  "RMSE": 56.0911,  "MAPE (%)": 17.7859},
    {"model": "Prophet",          "MAE": 14.5267, "RMSE": 67.1809,  "MAPE (%)": 45.4625},
    {"model": "SARIMA",           "MAE": 22.4073, "RMSE": 148.0301, "MAPE (%)": 49.9868},
    {"model": "Persistence",      "MAE": 22.8301, "RMSE": 82.5640,  "MAPE (%)": 67.1366},
]


def main() -> None:
    logger.info("Loading %s", PROCESSED_PATH)
    df = pd.read_parquet(PROCESSED_PATH)
    train, val, test = make_splits(df)
    fit_df = pd.concat([train, val]).sort_index()
    logger.info("fit=%d rows  test=%d rows", len(fit_df), len(test))

    y_true = test["price_HB_NORTH"]

    t0 = time.time()
    model = ChronosModel()
    model.fit(fit_df)
    preds = model.predict(test)
    elapsed = time.time() - t0

    result = score(y_true, preds, "Chronos (zero-shot)")
    result["time_s"] = round(elapsed, 1)
    logger.info(
        "%-22s  MAE=%7.4f  RMSE=%8.4f  MAPE=%7.4f%%  [%.0fs]",
        result["model"], result["MAE"], result["RMSE"], result["MAPE (%)"], elapsed,
    )

    # ── Comparison table ──────────────────────────────────────────────────────
    all_results = BENCHMARK + [result]
    summary = (
        pd.DataFrame(all_results)
        .set_index("model")
        .sort_values("MAPE (%)")
    )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON — test set ({:,} hours)".format(len(test)))
    print("  (* = prior benchmark run;  Chronos = this run)")
    print("=" * 70)
    print(summary.to_string())

    from datetime import datetime
    out_path = f"data/chronos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary.to_csv(out_path)
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
