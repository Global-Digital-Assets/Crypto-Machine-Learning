"""Unit tests for feature engineering functions."""
from __future__ import annotations

import polars as pl
import pytest

import feature_engineering as fe


def sample_df() -> pl.DataFrame:
    """Create a small deterministic OHLCV dataframe covering ~24h of 5-min bars."""
    n = 288  # 24 h of 5-min bars
    return pl.DataFrame(
        {
            "datetime": pl.datetime_range("2025-01-01", periods=n, interval="5m", eager=True),
            "symbol": ["BTCUSDT"] * n,
            "open": [1.0] * n,
            "high": [1.01] * n,
            "low": [0.99] * n,
            "close": [1.0] * n,
            "volume": [100.0] * n,
        }
    )


def test_build_features():
    df = sample_df()
    enriched = fe.build_features(df, bar_sec=300)

    # 1. Column coverage
    missing = [col for col in fe.FEATURE_NAMES if col not in enriched.columns]
    assert not missing, f"Missing features: {missing}"

    # 2. No null values in any feature column
    null_counts = (
        enriched.select([pl.col(c).is_null().sum().alias(c) for c in fe.FEATURE_NAMES])
        .row(0)
    )
    assert sum(null_counts) == 0, "Null values present in engineered features"
