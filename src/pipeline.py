"""ERCOT DAM Settlement Point Price data pipeline."""

from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Tuple

import holidays as hol_lib
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from gridstatus import Ercot

load_dotenv()  # loads EIA_API_KEY (and any other secrets) from .env

logger = logging.getLogger(__name__)

HUBS: list[str] = ["HB_NORTH", "HB_SOUTH", "HB_HOUSTON", "HB_WEST"]

_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = _ROOT / "data" / "raw"
PROCESSED_DIR = _ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "ercot_processed.parquet"

_EIA_FUEL_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"
_EIA_GAS_URL  = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"


# ─────────────────────────── Price download ──────────────────────────────────


def download_ercot_data(
    start_year: int = 2022,
    end_year: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download ERCOT DAM SPP for HUBS for start_year through end_year.

    Uses gridstatus.Ercot.get_dam_spp(year) which returns the full calendar
    year of hourly hub and load-zone prices by combining ERCOT MIS report
    sources. Results are cached in data/raw/ as a parquet file.

    Returns a long-format DataFrame with gridstatus column schema.
    """
    if end_year is None:
        end_year = datetime.now().year

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"ercot_raw_{start_year}_{end_year}.parquet"

    if use_cache and cache_path.exists():
        logger.info("Raw cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    iso = Ercot()
    frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching DAM SPP for %d ...", year)
        frames.append(iso.get_dam_spp(year))

    raw = pd.concat(frames, ignore_index=True)
    raw.to_parquet(cache_path)
    logger.info("Cached raw data: %s  (%d rows)", cache_path, len(raw))
    return raw


# ─────────────────────────── Exogenous download ──────────────────────────────


def _fetch_eia_hourly(
    fuel_type: str,
    start_year: int,
    end_year: int,
    api_key: str,
) -> pd.Series:
    """
    Fetch hourly fuel-type generation from the EIA API for ERCOT.

    fuel_type: "WND" (wind) or "SUN" (solar).
    Paginates automatically (EIA max 5 000 records per request).
    Returns a UTC-indexed Series.
    """
    col_name = {"WND": "wind_gen", "SUN": "solar_gen"}[fuel_type]
    all_records: list[dict] = []

    for year in range(start_year, end_year + 1):
        offset = 0
        while True:
            params = {
                "api_key": api_key,
                "frequency": "hourly",
                "data[0]": "value",
                "facets[respondent][]": "ERCO",
                "facets[fueltype][]": fuel_type,
                "start": f"{year}-01-01T00",
                "end": f"{year}-12-31T23",
                "sort[0][column]": "period",
                "sort[0][direction]": "asc",
                "offset": offset,
                "length": 5000,
            }
            resp = requests.get(_EIA_FUEL_URL, params=params, timeout=60)
            resp.raise_for_status()
            payload = resp.json()["response"]
            rows = payload.get("data", [])
            if not rows:
                break
            all_records.extend(rows)
            total = int(payload.get("total", 0))
            if offset + len(rows) >= total:
                break
            offset += len(rows)

    df = pd.DataFrame(all_records)
    df["timestamp"] = pd.to_datetime(df["period"], format="%Y-%m-%dT%H", utc=True)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = (
        df.sort_values("timestamp")
        .drop_duplicates("timestamp")
        .set_index("timestamp")["value"]
        .rename(col_name)
    )
    return series


def _fetch_gas_price_daily(
    start_year: int,
    end_year: int,
    api_key: str,
) -> pd.Series:
    """
    Fetch Henry Hub natural gas spot price (daily, $/MMBtu) from EIA API v2.

    Series RNGWHHD is published on trading days only; weekend/holiday gaps are
    forward-filled when the result is reindexed to an hourly UTC index in
    download_exogenous_data().  Returns a date-indexed Series named 'gas_price'.
    """
    all_records: list[dict] = []
    offset = 0
    while True:
        params = {
            "api_key": api_key,
            "frequency": "daily",
            "data[0]": "value",
            "facets[series][]": "RNGWHHD",
            "start": f"{start_year}-01-01",
            "end": f"{end_year}-12-31",
            "sort[0][column]": "period",
            "sort[0][direction]": "asc",
            "offset": offset,
            "length": 5000,
        }
        resp = requests.get(_EIA_GAS_URL, params=params, timeout=60)
        resp.raise_for_status()
        payload = resp.json()["response"]
        rows = payload.get("data", [])
        if not rows:
            break
        all_records.extend(rows)
        total = int(payload.get("total", 0))
        if offset + len(rows) >= total:
            break
        offset += len(rows)

    df = pd.DataFrame(all_records)
    df["date"] = pd.to_datetime(df["period"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    series = (
        df.sort_values("date")
        .drop_duplicates("date")
        .set_index("date")["value"]
        .rename("gas_price")
    )
    return series


def download_exogenous_data(
    start_year: int = 2022,
    end_year: int | None = None,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Download hourly ERCOT load (gridstatus), wind + solar generation (EIA),
    and Henry Hub natural gas spot price (EIA, daily → forward-filled hourly).

    Returns a UTC-indexed DataFrame with columns:
        ercot_load  – system-wide demand (MWh)
        wind_gen    – wind generation (MWh)
        solar_gen   – solar generation (MWh)
        net_load    – ercot_load minus renewables (MWh)
        gas_price   – Henry Hub spot price ($/MMBtu), ffilled from daily

    Load is sourced from gridstatus post-settlement actuals (multi-year archive).
    Wind and solar are sourced from the EIA Hourly Electric Grid Monitor API.
    Gas price is sourced from EIA series RNGWHHD (Henry Hub daily spot).
    All require EIA_API_KEY to be set in your .env file.

    Using actuals (not forecasts) is appropriate for backtesting — it
    establishes the performance ceiling assuming perfect exogenous knowledge.
    """
    if end_year is None:
        end_year = datetime.now().year

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = RAW_DIR / f"ercot_exog_{start_year}_{end_year}.parquet"

    if use_cache and cache_path.exists():
        logger.info("Exogenous cache hit: %s", cache_path)
        return pd.read_parquet(cache_path)

    # ── Load (gridstatus) ─────────────────────────────────────────────────────
    iso = Ercot()
    load_frames: list[pd.DataFrame] = []
    for year in range(start_year, end_year + 1):
        logger.info("Fetching load for %d ...", year)
        load_frames.append(iso.get_hourly_load_post_settlements(str(year)))

    load_df = pd.concat(load_frames, ignore_index=True)
    load_series = (
        load_df.set_index("Interval Start")["ERCOT"]
        .rename("ercot_load")
    )
    load_series.index = load_series.index.tz_convert("UTC")

    # ── Wind + solar (EIA) ────────────────────────────────────────────────────
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "EIA_API_KEY not set. Add EIA_API_KEY=<your_key> to your .env file."
        )

    logger.info("Fetching wind generation from EIA (%d–%d)...", start_year, end_year)
    wind = _fetch_eia_hourly("WND", start_year, end_year, api_key)

    logger.info("Fetching solar generation from EIA (%d–%d)...", start_year, end_year)
    solar = _fetch_eia_hourly("SUN", start_year, end_year, api_key)

    logger.info("Fetching Henry Hub gas price from EIA (%d–%d)...", start_year, end_year)
    gas_daily = _fetch_gas_price_daily(start_year, end_year, api_key)

    # ── Combine, align to continuous hourly UTC index ─────────────────────────
    exog = pd.DataFrame(
        {"ercot_load": load_series, "wind_gen": wind, "solar_gen": solar}
    ).sort_index()

    full_idx = pd.date_range(exog.index.min(), exog.index.max(), freq="h", tz="UTC")
    exog = exog.reindex(full_idx)
    exog.index.name = "timestamp"

    missing_pct = exog["ercot_load"].isna().mean() * 100
    if missing_pct > 5:
        logger.warning(
            "%.1f%% of exog rows are NaN after join — possible timezone mismatch", missing_pct
        )

    exog = exog.ffill(limit=3).bfill(limit=3)
    exog["net_load"] = (exog["ercot_load"] - exog["wind_gen"] - exog["solar_gen"]).clip(lower=0)

    # Gas price: daily → hourly by forward-filling across the UTC index
    gas_hourly = (
        gas_daily
        .reindex(exog.index.normalize().tz_localize(None))
        .values
    )
    exog["gas_price"] = gas_hourly
    exog["gas_price"] = exog["gas_price"].ffill().bfill()
    logger.info(
        "Gas price: %d daily obs, %.1f%% hourly coverage after ffill",
        len(gas_daily),
        exog["gas_price"].notna().mean() * 100,
    )

    exog.to_parquet(cache_path)
    logger.info("Cached exogenous data: %s (%d rows)", cache_path, len(exog))
    return exog


# ─────────────────────────── Clean ───────────────────────────────────────────


def clean_data(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Transform gridstatus long-format DataFrame into a wide hourly price table.

    Steps:
    - Validate expected gridstatus columns.
    - Filter to HUBS of interest.
    - Convert timestamps to UTC (unambiguous across DST transitions).
    - Deduplicate, pivot to wide format with columns 'price_<HUB>'.
    - Enforce a continuous hourly UTC index; fill short gaps (≤3 h) via ffill/bfill.

    Returns a DataFrame with a UTC-aware DatetimeIndex.
    """
    df = raw.copy()

    required = {"Interval Start", "Location", "SPP"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Raw data is missing required columns: {missing}")

    df = df[df["Location"].isin(HUBS)].copy()
    if df.empty:
        raise ValueError(f"No rows found for hubs {HUBS}.")

    df["timestamp"] = df["Interval Start"].dt.tz_convert("UTC")
    df = df.sort_values("timestamp")
    df = df.drop_duplicates(subset=["timestamp", "Location"], keep="first")

    wide = df.pivot(index="timestamp", columns="Location", values="SPP")
    wide.columns = [f"price_{c}" for c in wide.columns]
    wide = wide.sort_index()
    wide.index.name = "timestamp"

    full_idx = pd.date_range(wide.index.min(), wide.index.max(), freq="h", tz="UTC")
    wide = wide.reindex(full_idx)
    wide.index.name = "timestamp"

    n_missing = int(wide.isna().sum().sum())
    if n_missing:
        logger.info("Filling %d missing values (ffill/bfill, limit=3 h)", n_missing)
        wide = wide.ffill(limit=3).bfill(limit=3)

    return wide


# ─────────────────────────── Feature engineering ─────────────────────────────


def engineer_features(
    prices: pd.DataFrame,
    exog: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add temporal, lag/rolling, and (optionally) exogenous features.

    Temporal features are derived in US/Central time.
    Per-hub: lag_{1,24,168}h and rolling_mean_{24,168}h.
    Exogenous (when exog is provided):
        raw:    ercot_load, wind_gen, solar_gen, net_load
        ratios: wind_penetration, solar_penetration, renewable_penetration
        lags:   ercot_load_lag_{24,168}h  and  net_load_lag_{24,168}h

    Drops leading NaN rows produced by the 168-hour lag.
    """
    df = prices.copy()
    ts_utc = df.index
    ts_local = ts_utc.tz_convert("US/Central")

    # ── Temporal ──────────────────────────────────────────────────────────────
    df["hour_of_day"] = ts_local.hour
    df["day_of_week"] = ts_local.dayofweek
    df["month"] = ts_local.month
    df["is_weekend"] = (ts_local.dayofweek >= 5).astype(np.int8)

    years = range(ts_local.year.min(), ts_local.year.max() + 1)
    tx_hols = hol_lib.country_holidays("US", subdiv="TX", years=list(years))
    df["is_holiday"] = pd.array(
        [int(t.date() in tx_hols) for t in ts_local], dtype="int8"
    )

    # ── Per-hub lag / rolling ─────────────────────────────────────────────────
    for col in [c for c in df.columns if c.startswith("price_")]:
        hub = col.removeprefix("price_")
        series = df[col]
        df[f"lag_1h_{hub}"] = series.shift(1)
        df[f"lag_24h_{hub}"] = series.shift(24)
        df[f"lag_168h_{hub}"] = series.shift(168)
        shifted = series.shift(1)
        df[f"rolling_mean_24h_{hub}"] = shifted.rolling(24, min_periods=12).mean()
        df[f"rolling_mean_168h_{hub}"] = shifted.rolling(168, min_periods=84).mean()

    # ── Exogenous features ────────────────────────────────────────────────────
    if exog is not None:
        ex = exog.reindex(df.index)

        for col in ["ercot_load", "wind_gen", "solar_gen", "net_load", "gas_price"]:
            if col in ex.columns:
                df[col] = ex[col]

        load_safe = ex["ercot_load"].clip(lower=1)
        df["wind_penetration"] = ex["wind_gen"] / load_safe
        df["solar_penetration"] = ex["solar_gen"] / load_safe
        df["renewable_penetration"] = (ex["wind_gen"] + ex["solar_gen"]) / load_safe

        for col in ["ercot_load", "net_load", "gas_price"]:
            if col in df.columns:
                df[f"{col}_lag_24h"] = df[col].shift(24)
                df[f"{col}_lag_168h"] = df[col].shift(168)

    df = df.dropna(subset=[f"lag_168h_{HUBS[0]}"]).copy()
    return df


# ─────────────────────────── Splits ──────────────────────────────────────────


def make_splits(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Chronological 70/15/15 split with no lookahead leakage.

    Splits by row position in time-sorted order. Returns (train, val, test).
    """
    df = df.sort_index()
    n = len(df)
    i_val = int(n * train_frac)
    i_test = int(n * (train_frac + val_frac))
    return df.iloc[:i_val].copy(), df.iloc[i_val:i_test].copy(), df.iloc[i_test:].copy()


# ─────────────────────────── End-to-end entry point ──────────────────────────


def run_pipeline(
    start_year: int = 2022,
    end_year: int | None = None,
    raw_df: pd.DataFrame | None = None,
    use_exogenous: bool = True,
) -> pd.DataFrame:
    """
    Full pipeline: download → clean → feature-engineer → save parquet.

    Pass raw_df to skip the price download (offline / test mode).
    Set use_exogenous=False to produce the baseline feature set only.
    Saves to data/processed/ercot_processed.parquet.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if raw_df is None:
        raw_df = download_ercot_data(start_year=start_year, end_year=end_year)

    prices = clean_data(raw_df)

    exog = None
    if use_exogenous:
        try:
            exog = download_exogenous_data(start_year=start_year, end_year=end_year)
        except RuntimeError as exc:
            logger.warning("Exogenous features skipped: %s", exc)

    features = engineer_features(prices, exog=exog)
    features.to_parquet(PROCESSED_PATH)
    logger.info(
        "Saved → %s  (%d rows, %d columns)", PROCESSED_PATH, len(features), features.shape[1]
    )
    return features


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    run_pipeline()
