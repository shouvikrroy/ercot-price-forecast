"""Deep analysis of Chronos vs LightGBM: regime, temporal, bias, and directional accuracy."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.evaluate import mae, rmse, mape
from src.models.chronos_eval import ChronosModel
from src.models.lgbm import LGBMModel
from src.pipeline import make_splits

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_PATH = "data/processed/ercot_processed.parquet"
TARGET = "price_HB_NORTH"


def regime_breakdown(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> None:
    print(f"\n  {model_name}")
    print(f"  {'Regime':<24} {'n':>6}  {'MAE':>8}  {'RMSE':>9}  {'MAPE':>8}  {'Bias':>8}")
    print("  " + "-" * 70)
    for label, mask in [
        ("Normal  ($1–$100)",   (y_true >= 1)   & (y_true <= 100)),
        ("Elevated($100–$200)", (y_true > 100)  & (y_true <= 200)),
        ("Spike   (>$200)",      y_true > 200),
        ("Near-zero (<$1)",      y_true < 1),
    ]:
        if mask.sum() == 0:
            continue
        yt, yp = y_true[mask], y_pred[mask]
        safe = np.abs(yt) >= 1
        mape_r = float(np.mean(np.abs((yt[safe] - yp[safe]) / yt[safe])) * 100) if safe.any() else float("nan")
        bias_r = float((yp - yt).mean())
        print(
            f"  {label:<24} {mask.sum():>6}  {mae(yt,yp):>8.2f}  {rmse(yt,yp):>9.2f}"
            f"  {mape_r:>7.2f}%  {bias_r:>+8.2f}"
        )


def hour_breakdown(y_true: pd.Series, y_pred: pd.Series, model_name: str) -> None:
    hours = y_true.index.hour
    off_peak = (hours < 7) | (hours >= 22)
    shoulder = (hours >= 7) & (hours < 14)
    peak = (hours >= 14) & (hours < 22)

    print(f"\n  {model_name}")
    print(f"  {'Period':<20} {'n':>6}  {'MAE':>8}  {'RMSE':>9}  {'MAPE':>8}  {'Bias':>8}")
    print("  " + "-" * 62)
    for label, mask in [
        ("Off-peak (22–06)",  off_peak),
        ("Shoulder (07–13)",  shoulder),
        ("Peak     (14–21)",  peak),
    ]:
        yt, yp = y_true[mask], y_pred[mask]
        safe = np.abs(yt) >= 1
        mape_r = float(np.mean(np.abs((yt[safe] - yp[safe]) / yt[safe])) * 100) if safe.any() else float("nan")
        bias_r = float((yp - yt).mean())
        print(
            f"  {label:<20} {mask.sum():>6}  {mae(yt,yp):>8.2f}  {rmse(yt,yp):>9.2f}"
            f"  {mape_r:>7.2f}%  {bias_r:>+8.2f}"
        )


def directional_accuracy(y_true: pd.Series, y_pred: pd.Series) -> float:
    """Fraction of hours where model correctly predicts direction of price change."""
    actual_dir = np.sign(y_true.diff().dropna())
    pred_dir = np.sign(y_pred.diff().dropna())
    aligned = actual_dir.index.intersection(pred_dir.index)
    return float((actual_dir[aligned] == pred_dir[aligned]).mean() * 100)


def spike_detection(y_true: pd.Series, y_pred: pd.Series, threshold: float = 200.0) -> dict:
    """Treat spike (>threshold) as a binary classification problem."""
    actual_spike = y_true > threshold
    pred_spike = y_pred > threshold
    tp = (actual_spike & pred_spike).sum()
    fp = (~actual_spike & pred_spike).sum()
    fn = (actual_spike & ~pred_spike).sum()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    return {"spikes": int(actual_spike.sum()), "tp": int(tp), "fp": int(fp), "fn": int(fn),
            "precision": precision, "recall": recall}


def main() -> None:
    logger.info("Loading data")
    df = pd.read_parquet(PROCESSED_PATH)
    train, val, test = make_splits(df)
    fit_df = pd.concat([train, val]).sort_index()

    exog_cols = [c for c in test.columns if not c.startswith("price_")]
    test = test.copy()
    test[exog_cols] = test[exog_cols].ffill()

    y_true = test[TARGET]

    # ── Fit & predict ─────────────────────────────────────────────────────────
    logger.info("Fitting LightGBM...")
    lgbm = LGBMModel()
    lgbm.fit(fit_df)
    y_lgbm = lgbm.predict(test)

    logger.info("Fitting Chronos...")
    chronos = ChronosModel()
    chronos.fit(fit_df)
    y_chronos = chronos.predict(test)

    # ── 1. Price-regime breakdown ─────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("1. PRICE-REGIME BREAKDOWN")
    print("=" * 74)
    regime_breakdown(y_true, y_chronos, "Chronos (zero-shot)")
    regime_breakdown(y_true, y_lgbm,    "LightGBM")

    # ── 2. Hour-of-day breakdown ──────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("2. HOUR-OF-DAY BREAKDOWN")
    print("=" * 74)
    hour_breakdown(y_true, y_chronos, "Chronos (zero-shot)")
    hour_breakdown(y_true, y_lgbm,    "LightGBM")

    # ── 3. Spike detection (binary classification at $200) ────────────────────
    print("\n" + "=" * 74)
    print("3. SPIKE DETECTION  (threshold = $200)")
    print("=" * 74)
    for name, y_pred in [("Chronos (zero-shot)", y_chronos), ("LightGBM", y_lgbm)]:
        s = spike_detection(y_true, y_pred, threshold=200)
        print(
            f"  {name:<24}  spikes={s['spikes']}  TP={s['tp']}  FP={s['fp']}  FN={s['fn']}"
            f"  precision={s['precision']:.2f}  recall={s['recall']:.2f}"
        )

    # ── 4. Directional accuracy ───────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("4. DIRECTIONAL ACCURACY  (correct up/down from prior hour)")
    print("=" * 74)
    for name, y_pred in [("Chronos (zero-shot)", y_chronos), ("LightGBM", y_lgbm)]:
        da = directional_accuracy(y_true, y_pred)
        print(f"  {name:<24}  {da:.2f}%")

    # ── 5. Mean bias error ────────────────────────────────────────────────────
    print("\n" + "=" * 74)
    print("5. MEAN BIAS ERROR  (positive = over-predicts)")
    print("=" * 74)
    for name, y_pred in [("Chronos (zero-shot)", y_chronos), ("LightGBM", y_lgbm)]:
        bias = float((y_pred - y_true).mean())
        print(f"  {name:<24}  {bias:+.4f}")

    # ── Save predictions for further analysis ─────────────────────────────────
    from datetime import datetime
    out = pd.DataFrame({"y_true": y_true, "chronos": y_chronos, "lgbm": y_lgbm})
    out_path = f"data/predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
    out.to_parquet(out_path)
    logger.info("Predictions saved to %s", out_path)


if __name__ == "__main__":
    main()
