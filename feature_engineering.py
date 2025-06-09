"""Feature engineering module — builds the full 11-column feature set expected by production LightGBM models.

All transformations are vectorised using Polars for performance.

Public API
----------
    build_features(df: pl.DataFrame, bar_sec: int = 300) -> pl.DataFrame
        Returns *df* with new feature columns added. Input df must contain the
        raw OHLCV columns ['datetime','symbol','open','high','low','close','volume'].
"""
from __future__ import annotations

from typing import List

import polars as pl

# Ordered list must match training metadata exactly
FEATURE_NAMES: List[str] = [
    "ret_5m",
    "ret_1h",
    "ret_4h",
    "ret_24h",
    "vol_spike",
    "volatility_1h",
    "volatility_4h",
    "hl_spread",
    "bar_return",
    "sma_distance",
    "rsi14",
]


def build_features(df: pl.DataFrame, bar_sec: int = 300) -> pl.DataFrame:  # noqa: N802
    """Add production features to a `polars.DataFrame`.

    Parameters
    ----------
    df : pl.DataFrame
        Must contain columns: symbol, datetime, open, high, low, close, volume
    bar_sec : int, default 300
        Bar duration in **seconds** (300 = 5-minute bar). Use 900 for 15-minute bars.
    """
    # Horizon conversion: how many bars for each look-back window
    bars_per_5m = max(int(300 // bar_sec), 1)
    bars_per_1h = max(int(3600 // bar_sec), 1)
    bars_per_4h = max(int(14400 // bar_sec), 1)
    bars_per_24h = max(int(86400 // bar_sec), 1)

    df = df.sort(["symbol", "datetime"])

    # ---------- Momentum / returns ----------
    df = (
        df.with_columns(
            (
                (pl.col("close") / pl.col("close").shift(bars_per_5m).over("symbol") - 1)
            ).alias("ret_5m")
        )
        .with_columns([
            (pl.col("close") / pl.col("close").shift(bars_per_1h).over("symbol") - 1).alias("ret_1h"),
            (pl.col("close") / pl.col("close").shift(bars_per_4h).over("symbol") - 1).alias("ret_4h"),
            (pl.col("close") / pl.col("close").shift(bars_per_24h).over("symbol") - 1).alias("ret_24h"),
        ])
    )

    # ---------- Volume & volatility ----------
    df = df.with_columns(
        pl.col("volume").rolling_mean(bars_per_1h).over("symbol").alias("vol_ma_1h")
    ).with_columns(
        (pl.col("volume") / pl.col("vol_ma_1h")).alias("vol_spike")
    ).with_columns([
        pl.col("ret_5m").rolling_std(bars_per_1h).over("symbol").alias("volatility_1h"),
        pl.col("ret_5m").rolling_std(bars_per_4h).over("symbol").alias("volatility_4h"),
    ])

    # ---------- Price structure ----------
    df = df.with_columns(
        ((pl.col("high") + pl.col("low")) / 2 / pl.col("close") - 1).alias("hl_spread")
    ).with_columns(
        (pl.col("close") / pl.col("open") - 1).alias("bar_return")
    )

    # SMA distance (14-period)
    df = df.with_columns(pl.col("close").rolling_mean(14).over("symbol").alias("sma_14"))
    df = df.with_columns((pl.col("close") / pl.col("sma_14") - 1).alias("sma_distance"))

    # ---------- RSI-14 ----------
    df = df.with_columns(
        (
            pl.when(pl.col("ret_5m") > 0)
            .then(pl.col("ret_5m"))
            .otherwise(0)
            .rolling_mean(14)
            .over("symbol")
            .alias("avg_gains")
        )
    ).with_columns(
        (
            pl.when(pl.col("ret_5m") < 0)
            .then(-pl.col("ret_5m"))
            .otherwise(0)
            .rolling_mean(14)
            .over("symbol")
            .alias("avg_losses")
        )
    )

    df = df.with_columns(
        (
            100 - 100 / (1 + pl.col("avg_gains") / (pl.col("avg_losses") + 1e-10))
        ).alias("rsi14")
    )

    # Drop helper columns
    df = df.drop(["vol_ma_1h", "sma_14", "avg_gains", "avg_losses"])

    return df
