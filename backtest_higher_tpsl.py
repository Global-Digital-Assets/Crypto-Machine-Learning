#!/usr/bin/env python3
"""Back-tester that uses the full *11-feature* production pipeline.

It reproduces the live feature engineering step (`feature_engineering.build_features`) so
that evaluation is apples-to-apples with the signal generator & production model.

The trade simulation / percentile sweep logic is identical to
`backtest_swing_bucket.py` but relies on the feature names contained
inside the LightGBM model so it automatically adapts if the feature set changes.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import polars as pl
from tqdm import tqdm

# ------------------------------------------------------------
# Import the production feature builder
# ------------------------------------------------------------
ROOT_ML_ENGINE = os.getenv("ML_ENGINE_ROOT", "/root/ml-engine")
sys.path.append(ROOT_ML_ENGINE)  # noqa: E402 – after sys.path tweak
from feature_engineering import build_features  # type: ignore  # noqa: E402

# ------------------------------------------------------------
# Bucket configuration (same TP / SL table)
# ------------------------------------------------------------
BUCKET_TP_SL: Dict[str, Tuple[float, float]] = {
    "stable": (1.0, 0.3),
    "low": (2.0, 0.6),
    "mid": (3.5, 1.1),
    "high": (7.0, 1.8),
    "ultra": (11.0, 2.5),
}

TIME_STOP_BARS = 24  # 6-hour time-stop on 15-min bars
FEE_PCT = 0.04       # round-trip fee/slippage in percent
BARS_PER_DAY = 96

# ------------------------------------------------------------
# Trade simulation helper
# ------------------------------------------------------------

def simulate_trades(
    df: pl.DataFrame,
    entries_mask: np.ndarray,
    tp_pct: float,
    sl_pct: float,
) -> Tuple[int, float, float]:
    """Simulate trades on a *single* symbol DataFrame."""
    closes = df["close"].to_numpy()
    highs = df["high"].to_numpy()
    lows = df["low"].to_numpy()

    n = len(df)
    i = 0
    trades = wins = 0
    total_ret = 0.0

    while i < n:
        if entries_mask[i]:
            entry_price = closes[i]
            tp_price = entry_price * (1 + tp_pct / 100)
            sl_price = entry_price * (1 - sl_pct / 100)
            j = 1
            exit_price = closes[i]
            while True:
                if i + j >= n:
                    exit_price = closes[-1]
                    break
                hi = highs[i + j]
                lo = lows[i + j]
                if hi >= tp_price:
                    exit_price = tp_price
                    break
                if lo <= sl_price:
                    exit_price = sl_price
                    break
                if j >= TIME_STOP_BARS:
                    exit_price = closes[i + j]
                    break
                j += 1
            pct = (exit_price / entry_price - 1) * 100 - FEE_PCT
            if pct > 0:
                wins += 1
            total_ret += pct
            trades += 1
            i += j  # jump to exit bar
        else:
            i += 1

    avg_pct = total_ret / trades if trades else 0.0
    win_rate = wins / trades if trades else 0.0
    return trades, avg_pct, win_rate

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet-dir", required=True,
                        help="Directory containing 15-min parquet files one per symbol (SYMBOL.parquet)")
    parser.add_argument("--model", required=True, help="Path to LightGBM model file")
    parser.add_argument("--bucket-map", required=True, help="CSV mapping symbol,bucket")
    parser.add_argument("--percentiles", default="97,95,90", help="Comma-sep percentile list e.g. '97,95'")
    parser.add_argument("--buckets", default="high,ultra",
                        help="Comma list of buckets to evaluate (default high,ultra)")
    parser.add_argument("--out", default="backtest_results.json", help="Output JSON file")
    parser.add_argument("--min-proba", type=float, default=0.25,
                        help="Minimum model probability filter")
    args = parser.parse_args()

    percentiles = [float(p) for p in args.percentiles.split(",") if p]
    chosen_buckets = args.buckets.split(",") if args.buckets != "all" else None

    # ---------- bucket mapping ----------
    bucket_df = pl.read_csv(args.bucket_map)
    symbol_to_bucket = dict(zip(bucket_df["symbol"], bucket_df["bucket"]))

    # ---------- load model ----------
    print(f"[backtest] loading model → {args.model}")
    model = lgb.Booster(model_file=args.model)
    model_features = model.feature_name()

    # ---------- compute probabilities per symbol ----------
    parquet_files = glob.glob(os.path.join(args.parquet_dir, "*.parquet"))
    all_rows: List[pl.DataFrame] = []
    for pq in tqdm(parquet_files, desc="Predicting", ncols=80):
        symbol = os.path.splitext(os.path.basename(pq))[0]
        bucket = symbol_to_bucket.get(symbol)
        if bucket is None:
            continue  # unknown symbol
        if chosen_buckets and bucket not in chosen_buckets:
            continue

        df_raw = pl.read_parquet(pq)
        # ensure required 'symbol' column exists for feature_engineering
        if "symbol" not in df_raw.columns:
            df_raw = df_raw.with_columns(pl.lit(symbol).alias("symbol"))
        df_feat = build_features(df_raw, bar_sec=900)  # 15-min bars
        df_feat = df_feat.drop_nulls(model_features)
        if df_feat.is_empty():
            continue
        X = df_feat.select(model_features).to_numpy()
        proba = model.predict(X, num_iteration=model.best_iteration)

        sub = pl.DataFrame(
            {
                "datetime": df_feat["datetime"],
                "date": df_feat["datetime"].dt.date(),
                "symbol": symbol,
                "bucket": bucket,
                "proba": proba,
                "high": df_feat["high"],
                "low": df_feat["low"],
                "close": df_feat["close"],
            }
        )
        all_rows.append(sub)

    if not all_rows:
        print("No data rows after feature engineering – check parquet dir & buckets")
        sys.exit(1)

    big = pl.concat(all_rows)
    print(f"[backtest] dataframe rows: {big.height:,}")

    results = []
    for perc in percentiles:
        print(f"→ Percentile {perc}")
        keep_frac = (100 - perc) / 100  # e.g. 97 → 0.03 (top-3 %)

        df_ranked = (
            big.with_columns([
                pl.col("proba").rank("ordinal", descending=True).over("date").alias("rank"),
                pl.count().over("date").alias("n_rows"),
            ])
            .with_columns(((pl.col("n_rows") * keep_frac) + 0.999).floor().cast(pl.Int64).alias("keep_n"))
            .with_columns((pl.col("rank") <= pl.col("keep_n")).alias("selected"))
        )

        df_thr = df_ranked.filter((pl.col("selected")) & (pl.col("proba") >= args.min_proba))

        # --- per-symbol simulation ---
        bucket_stats = defaultdict(lambda: [0, 0.0, 0.0])  # trades, pct_sum, win_sum
        for symbol in df_thr["symbol"].unique():
            sub = df_thr.filter(pl.col("symbol") == symbol).sort("datetime")
            bucket = sub["bucket"][0]
            tp, sl = BUCKET_TP_SL[bucket]
            entries_mask = sub["selected"].to_numpy()
            trades, avg_pct, win_rate = simulate_trades(sub, entries_mask, tp, sl)
            bucket_stats[bucket][0] += trades
            bucket_stats[bucket][1] += avg_pct * trades
            bucket_stats[bucket][2] += win_rate * trades

        overall_trades = sum(v[0] for v in bucket_stats.values())
        overall_pct_sum = sum(v[1] for v in bucket_stats.values())
        overall_win_sum = sum(v[2] for v in bucket_stats.values())
        overall_avg = overall_pct_sum / overall_trades if overall_trades else 0.0
        overall_win = overall_win_sum / overall_trades if overall_trades else 0.0

        res = {
            "percentile": perc,
            "trades": overall_trades,
            "avg_pct": round(overall_avg, 4),
            "win_rate": round(overall_win, 4),
            "bucket_breakdown": {},
        }
        for b, (t, pct_sum, win_sum) in bucket_stats.items():
            res["bucket_breakdown"][b] = {
                "trades": t,
                "avg_pct": round(pct_sum / t if t else 0.0, 4),
                "win_rate": round(win_sum / t if t else 0.0, 4),
            }
        results.append(res)
        print(f"   trades {overall_trades:,} | avg {overall_avg:.3f}% | win {overall_win:.2%}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("[backtest] saved →", args.out)


if __name__ == "__main__":
    main()
