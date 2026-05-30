"""Session-scoped fixtures backed by real ERCOT data.

On first run the fixture downloads Q1 2023 DAM SPP from ERCOT via gridstatus
and caches it to tests/fixtures/sample_raw.parquet.  Subsequent runs load
from the cache and require no network access.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from gridstatus import Ercot

from src.pipeline import clean_data, engineer_features

FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_RAW_PATH = FIXTURES_DIR / "sample_raw.parquet"


def _download_sample() -> pd.DataFrame:
    FIXTURES_DIR.mkdir(exist_ok=True)
    iso = Ercot()
    df = iso.get_dam_spp(2023)
    # Keep Q1 only — real data, small enough for fast tests
    df = df[df["Interval Start"].dt.month <= 3].copy()
    df.to_parquet(SAMPLE_RAW_PATH)
    return df


@pytest.fixture(scope="session")
def raw_df() -> pd.DataFrame:
    """Real ERCOT DAM SPP long-format data for Q1 2023."""
    if SAMPLE_RAW_PATH.exists():
        return pd.read_parquet(SAMPLE_RAW_PATH)
    return _download_sample()


@pytest.fixture(scope="session")
def prices_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    return clean_data(raw_df)


@pytest.fixture(scope="session")
def features_df(prices_df: pd.DataFrame) -> pd.DataFrame:
    return engineer_features(prices_df)
