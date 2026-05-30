"""Run LSTM 5 times with different seeds and report averaged metrics."""

from __future__ import annotations

import logging
import random

import numpy as np
import pandas as pd
import torch

from src.evaluate import score, comparison_table
from src.models.lstm import LSTMModel
from src.pipeline import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/ercot_processed.parquet"
N_TRIALS = 5
SEEDS = [42, 7, 13, 99, 2025]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    logger.info("Loading processed data from %s", PROCESSED_PATH)
    df = pd.read_parquet(PROCESSED_PATH)

    train_df, val_df, test_df = make_splits(df)
    # Combine train+val for fitting; evaluate on test
    fit_df = pd.concat([train_df, val_df]).sort_index()
    y_true = test_df["price_HB_NORTH"]

    logger.info(
        "Split sizes — fit: %d, test: %d rows", len(fit_df), len(test_df)
    )

    trial_results: list[dict] = []

    for trial, seed in enumerate(SEEDS, start=1):
        logger.info("=" * 60)
        logger.info("Trial %d / %d  (seed=%d)", trial, N_TRIALS, seed)
        logger.info("=" * 60)

        set_seed(seed)
        model = LSTMModel()
        model.fit(fit_df)
        y_pred = model.predict(test_df)

        metrics = score(y_true, y_pred, model_name=f"LSTM trial {trial}")
        trial_results.append(metrics)
        logger.info(
            "Trial %d — MAE=%.4f  RMSE=%.4f  MAPE=%.4f%%",
            trial,
            metrics["MAE"],
            metrics["RMSE"],
            metrics["MAPE (%)"],
        )

    # Per-trial table
    print("\n" + "=" * 60)
    print("Per-trial results")
    print("=" * 60)
    trial_df = pd.DataFrame(trial_results).set_index("model")
    print(trial_df.to_string())

    # Averaged results
    avg = {
        "MAE": round(float(np.mean([r["MAE"] for r in trial_results])), 4),
        "RMSE": round(float(np.mean([r["RMSE"] for r in trial_results])), 4),
        "MAPE (%)": round(float(np.mean([r["MAPE (%)"] for r in trial_results])), 4),
    }
    std = {
        "MAE": round(float(np.std([r["MAE"] for r in trial_results])), 4),
        "RMSE": round(float(np.std([r["RMSE"] for r in trial_results])), 4),
        "MAPE (%)": round(float(np.std([r["MAPE (%)"] for r in trial_results])), 4),
    }

    print("\n" + "=" * 60)
    print(f"LSTM averaged over {N_TRIALS} trials")
    print("=" * 60)
    for metric in ["MAE", "RMSE", "MAPE (%)"]:
        print(f"  {metric:10s}: {avg[metric]:.4f}  ± {std[metric]:.4f}")
    print()


if __name__ == "__main__":
    main()
