"""Full model evaluation: all models on the test set, LSTM averaged over 5 trials."""

from __future__ import annotations

import logging
import random
import time

import numpy as np
import pandas as pd
import torch

from src.evaluate import score
from src.models.chronos_eval import ChronosModel
from src.models.lgbm import LGBMModel
from src.models.lstm import LSTMModel
from src.models.patchtst import PatchTSTModel
from src.models.persistence import PersistenceModel
from src.models.prophet_model import ProphetModel
from src.models.sarima import SARIMAModel
from src.pipeline import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/ercot_processed.parquet"
LSTM_SEEDS = [42, 7, 13, 99, 2025]
PATCHTST_SEEDS = [42, 7, 13]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_model(name: str, model, fit_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    y_true = test_df["price_HB_NORTH"]
    t0 = time.time()
    model.fit(fit_df)
    preds = model.predict(test_df)
    elapsed = time.time() - t0
    result = score(y_true, preds, name)
    result["time_s"] = round(elapsed, 1)
    logger.info(
        "%-20s  MAE=%7.4f  RMSE=%8.4f  MAPE=%7.4f%%  [%.0fs]",
        name, result["MAE"], result["RMSE"], result["MAPE (%)"], elapsed,
    )
    return result


def run_lstm_trials(fit_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    y_true = test_df["price_HB_NORTH"]
    trial_results = []
    t0 = time.time()
    for i, seed in enumerate(LSTM_SEEDS, 1):
        logger.info("LSTM trial %d/5 (seed=%d)...", i, seed)
        set_seed(seed)
        m = LSTMModel()
        m.fit(fit_df)
        preds = m.predict(test_df)
        trial_results.append(score(y_true, preds, f"LSTM trial {i}"))
    elapsed = time.time() - t0

    avg = {
        "model": "LSTM (avg 5)",
        "MAE":      round(float(np.mean([r["MAE"]      for r in trial_results])), 4),
        "RMSE":     round(float(np.mean([r["RMSE"]     for r in trial_results])), 4),
        "MAPE (%)": round(float(np.mean([r["MAPE (%)"] for r in trial_results])), 4),
        "time_s":   round(elapsed, 1),
    }
    std_mape = round(float(np.std([r["MAPE (%)"] for r in trial_results])), 4)
    logger.info(
        "%-20s  MAE=%7.4f  RMSE=%8.4f  MAPE=%7.4f%% ±%.4f  [%.0fs]",
        avg["model"], avg["MAE"], avg["RMSE"], avg["MAPE (%)"], std_mape, elapsed,
    )
    return avg, trial_results


def run_patchtst_trials(fit_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    y_true = test_df["price_HB_NORTH"]
    trial_results = []
    t0 = time.time()
    for i, seed in enumerate(PATCHTST_SEEDS, 1):
        logger.info("PatchTST trial %d/3 (seed=%d)...", i, seed)
        set_seed(seed)
        m = PatchTSTModel()
        m.fit(fit_df)
        preds = m.predict(test_df)
        trial_results.append(score(y_true, preds, f"PatchTST trial {i}"))
    elapsed = time.time() - t0

    avg = {
        "model": "PatchTST (avg 3)",
        "MAE":      round(float(np.mean([r["MAE"]      for r in trial_results])), 4),
        "RMSE":     round(float(np.mean([r["RMSE"]     for r in trial_results])), 4),
        "MAPE (%)": round(float(np.mean([r["MAPE (%)"] for r in trial_results])), 4),
        "time_s":   round(elapsed, 1),
    }
    std_mape = round(float(np.std([r["MAPE (%)"] for r in trial_results])), 4)
    logger.info(
        "%-20s  MAE=%7.4f  RMSE=%8.4f  MAPE=%7.4f%% ±%.4f  [%.0fs]",
        avg["model"], avg["MAE"], avg["RMSE"], avg["MAPE (%)"], std_mape, elapsed,
    )
    return avg, trial_results


def main() -> None:
    logger.info("Loading %s", PROCESSED_PATH)
    df = pd.read_parquet(PROCESSED_PATH)
    train, val, test = make_splits(df)
    fit_df = pd.concat([train, val]).sort_index()

    # Forward-fill publication-lag NaNs in exogenous columns (EIA/gridstatus lag)
    exog_cols = [c for c in test.columns if not c.startswith("price_")]
    test = test.copy()
    test[exog_cols] = test[exog_cols].ffill()

    logger.info("fit=%d rows  test=%d rows", len(fit_df), len(test))

    results = []

    # ── Persistence ───────────────────────────────────────────────────────────
    results.append(run_model("Persistence", PersistenceModel(), fit_df, test))

    # ── LightGBM ──────────────────────────────────────────────────────────────
    results.append(run_model("LightGBM", LGBMModel(), fit_df, test))

    # ── SARIMA ────────────────────────────────────────────────────────────────
    results.append(run_model("SARIMA", SARIMAModel(), fit_df, test))

    # ── Prophet ───────────────────────────────────────────────────────────────
    results.append(run_model("Prophet", ProphetModel(), fit_df, test))

    # ── LSTM (5 trials) ───────────────────────────────────────────────────────
    lstm_avg, lstm_trials = run_lstm_trials(fit_df, test)
    results.append(lstm_avg)

    # ── PatchTST (3 trials) ───────────────────────────────────────────────────
    ptst_avg, ptst_trials = run_patchtst_trials(fit_df, test)
    results.append(ptst_avg)

    # ── Chronos (zero-shot) ───────────────────────────────────────────────────
    results.append(run_model("Chronos (zero-shot)", ChronosModel(), fit_df, test))

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FULL MODEL COMPARISON — test set ({:,} hours)".format(len(test)))
    print("=" * 70)
    summary = pd.DataFrame(results).set_index("model").sort_values("MAPE (%)")
    print(summary.to_string())

    from datetime import datetime
    out_path = f"data/results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    summary.to_csv(out_path)
    logger.info("Results saved to %s", out_path)

    print("\n--- LSTM per-trial breakdown ---")
    print(pd.DataFrame(lstm_trials).set_index("model").to_string())

    print("\n--- PatchTST per-trial breakdown ---")
    print(pd.DataFrame(ptst_trials).set_index("model").to_string())

    # ── Per-regime MAPE breakdown (reuse the LightGBM already fitted above) ───
    print("\n" + "=" * 70)
    print("LightGBM regime breakdown (best deterministic model)")
    print("=" * 70)
    y_true = test["price_HB_NORTH"]
    lgbm = LGBMModel()
    lgbm.fit(fit_df)
    y_pred = lgbm.predict(test)


    for label, mask in [
        ("Normal  ($1–$100)",  (y_true >= 1) & (y_true <= 100)),
        ("Elevated($100–$200)",(y_true > 100) & (y_true <= 200)),
        ("Spike   (>$200)",    y_true > 200),
        ("Near-zero (<$1)",    y_true < 1),
    ]:
        if mask.sum() == 0:
            continue
        mae_r  = float(np.mean(np.abs(y_true[mask] - y_pred[mask])))
        rmse_r = float(np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2)))
        safe   = np.abs(y_true[mask]) >= 1
        mape_r = float(np.mean(np.abs((y_true[mask][safe] - y_pred[mask][safe]) / y_true[mask][safe])) * 100) if safe.any() else float("nan")
        print(f"  {label}: n={mask.sum():5d}  MAE={mae_r:8.2f}  RMSE={rmse_r:8.2f}  MAPE={mape_r:6.2f}%")


if __name__ == "__main__":
    main()
