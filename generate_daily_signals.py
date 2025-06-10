#!/usr/bin/env python3
"""Daily signal generator for live trading.

Uses the *winning* configuration discovered in back-tests:
    • Buckets: high, ultra
    • Percentile: top-1 % per calendar day (rank across bucket symbols)
    • Min probability ≥ 0.25
    • Bucket-specific TP/SL (6 %/1.5 % for *high*, 10 %/2 % for *ultra*)

Outputs a JSON file under ``signals/YYYYMMDD.json`` with one entry per
selected trade signal.  This file can be consumed by the execution layer
(live or paper).

Run daily **after** the 15-minute parquet feed has been updated.

Example:
    python generate_daily_signals.py \
        --parquet-dir data/swing/15m \
        --model models/lgbm_swing_20250608_054520.txt
"""
from __future__ import annotations

import argparse
import glob
import json
import feature_engineering as fe
FEATURE_COLS = fe.FEATURE_NAMES
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Tuple
import time

import lightgbm as lgb

# Optimize for 8-core usage
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

import numpy as np
import polars as pl
import urllib.request
import json as _json  # for API parsing
import json

# ---------------------------------------------------------------------------
# Config defaults (can be overridden via CLI)
# ---------------------------------------------------------------------------
DEFAULT_BUCKETS = ["ultra", "high", "mid", "low", "stable"]
DEFAULT_PERCENTILE = 99.0  # top-1 %
MIN_PROBA = 0.25

BUCKET_TP_SL: Dict[str, Tuple[float, float]] = {
    "ultra": (10.0, 2.0),
    "high": (6.0, 1.5),
    "mid": (3.5, 2.3),
    "low": (2.5, 1.7),
    "stable": (1.5, 1.0),
}
BARS_PER_DAY = 96  # 15-min bars

# ---------------------------------------------------------------------------
# Feature engineering (same logic as training/back-test)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Helpers for optional Data-API mode
# ---------------------------------------------------------------------------

def _fetch_df_from_api(symbol: str, api_base: str, limit: int = BARS_PER_DAY + 60) -> pl.DataFrame:
    """Fetch latest candles for *symbol* from the data-service HTTP API.

    The API is expected to expose /candles/{symbol}/15m returning JSON with
    keys: symbol, tf, count, candles[list]. Each candle has ts (ms), open, high,
    low, close, vol.
    Returns an empty DataFrame on error or missing data so the caller can skip
    gracefully.
    """
    for attempt in range(3):
        try:
            url = f"{api_base.rstrip('/')}/candles/{symbol}/15m?limit={limit}"
            with urllib.request.urlopen(url, timeout=10) as resp:
                payload = _json.loads(resp.read().decode())
            break  # success
        except Exception as api_exc:
            if attempt == 2:
                print("[api] fetch failed:", symbol, api_exc)
                return pl.DataFrame()
            time.sleep(1 + attempt)
    if payload.get("count", 0) == 0:
        return pl.DataFrame()

    rows = [
        {
            "datetime": datetime.fromtimestamp(
                c["ts"] / 1000 if c["ts"] > 1e12 else c["ts"],
                tz=timezone.utc,
            ),
            "open": c["open"],
            "high": c["high"],
            "low": c["low"],
            "close": c["close"],
            "volume": c["vol"],
        }
        for c in payload["candles"]
    ]
    return pl.from_dicts(rows).with_columns(pl.lit(symbol).alias("symbol")).with_columns(pl.lit(symbol).alias(symbol))


def generate_signals(
    parquet_dir: str,
    model_path: str,
    bucket_map: str,
    buckets: List[str] = DEFAULT_BUCKETS,
    percentile: float = DEFAULT_PERCENTILE,
    min_proba: float = MIN_PROBA,
    data_api_url: str | None = None,
) -> List[dict]:
    bucket_df = pl.read_csv(bucket_map)
    bucket_df = bucket_df.filter(pl.col("bucket").is_in(buckets))
    symbols = bucket_df["symbol"].to_list()
    if not symbols:
        raise SystemExit("No symbols match selected buckets")

    booster = lgb.Booster(model_file=model_path)

    signal_rows: List[dict] = []
    all_probs: dict[str, float] = {}
    stale_count = 0

    for sym in symbols:
        # ------------------------- Data loading -------------------------
        if data_api_url:
            df = _fetch_df_from_api(sym, data_api_url)
            if df.is_empty() or df.height < BARS_PER_DAY + 5:
                # fallback to local parquet if API failed or insufficient rows
                pq_path = Path(parquet_dir) / f"{sym}.parquet"
                if pq_path.exists():
                    df = pl.read_parquet(pq_path)
                else:
                    stale_count += 1
                    continue
            # age via timestamp column
            latest_ts = df.sort("datetime").tail(1)["datetime"][0]
            age_hours = (datetime.now(timezone.utc) - latest_ts).total_seconds() / 3600
            if age_hours > 3:
                stale_count += 1
                continue
        else:
            pq_path = Path(parquet_dir) / f"{sym}.parquet"
            if not pq_path.exists():
                stale_count += 1
                continue
            df = pl.read_parquet(pq_path)
            if df.height < BARS_PER_DAY + 5:
                continue

            # Freshness check using parquet file's modification time (robust & cheap)
            import time
            age_hours = (time.time() - os.path.getmtime(pq_path)) / 3600
            if age_hours > 3:
                stale_count += 1
                continue

        df = df.with_columns(pl.lit(sym).alias("symbol"))
        df = fe.build_features(df, bar_sec=900).drop_nulls(FEATURE_COLS)
        if df.is_empty():
            continue
        latest = df.tail(1)
        X = latest.select(FEATURE_COLS).to_numpy()
        proba = float(booster.predict(X, num_iteration=booster.best_iteration)[0])
        all_probs[sym] = proba
        if proba < min_proba:
            continue
        bucket = bucket_df.filter(pl.col("symbol") == sym)["bucket"][0]
        tp_pct, sl_pct = BUCKET_TP_SL[bucket]
        close = float(latest["close"][0])
        signal_rows.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "symbol": sym,
                "probability": proba,
                "bucket": bucket,
                "tp_pct": tp_pct,
                "sl_pct": sl_pct,
                "entry_price": close,
                "tp_price": close * (1 + tp_pct / 100),
                "sl_price": close * (1 - sl_pct / 100),
            }
        )

    # Rank & take top-percentile
    signal_rows.sort(key=lambda x: x["probability"], reverse=True)
    keep_n = max(1, int(len(signal_rows) * (100 - percentile) / 100))
    return signal_rows[:keep_n], all_probs, stale_count


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate daily trading signals")
    ap.add_argument("--parquet-dir", default="data/swing/15m")
    ap.add_argument("--model", required=True)
    ap.add_argument("--bucket-map", default="bucket_mapping.csv")
    ap.add_argument("--percentile", type=float, default=DEFAULT_PERCENTILE)
    ap.add_argument("--min-proba", type=float, default=MIN_PROBA)
    ap.add_argument("--out-dir", default="signals")
    ap.add_argument("--buckets", default="{}".format(",".join(DEFAULT_BUCKETS)), help="Comma-separated bucket list (e.g. 'ultra,high')")
    ap.add_argument("--data-api-url", help="Optional HTTP base URL for candle API. When provided parquet-dir is ignored.")
    args = ap.parse_args()

    buckets_list = [b.strip() for b in args.buckets.split(",") if b.strip()]

    signals, all_probs, stale = generate_signals(
        parquet_dir=args.parquet_dir,
        model_path=args.model,
        bucket_map=args.bucket_map,
        buckets=buckets_list,
        percentile=args.percentile,
        min_proba=args.min_proba,
        data_api_url=args.data_api_url,
    )

    # ---- persist per-symbol probabilities for monitoring ----
    try:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(exist_ok=True, parents=True)
        ts_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        probs_path = out_dir / f"probs_{ts_str}.json"
        with probs_path.open("w") as fp:
            json.dump(all_probs, fp, default=str)
        # symlink latest
        latest_link = out_dir / "latest_probs.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(probs_path.name)
    except Exception as exc:
        print("[persist] failed to write probabilities:", exc)

    date_str = datetime.utcnow().strftime("%Y%m%d")
    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    file_path = out_path / f"signals_{date_str}.json"

    result = {
        "generated_at": datetime.utcnow().isoformat(),
        "config": {
            "buckets": DEFAULT_BUCKETS,
            "percentile": args.percentile,
            "min_proba": args.min_proba,
        },
        "signals": signals,
    }

    with open(file_path, "w") as f:
        json.dump(result, f, indent=2)

    # --- legacy API compatibility: also publish to multi_token_analysis.json ---
    try:
        import shutil, pathlib
        legacy_path = pathlib.Path(__file__).resolve().parent / "multi_token_analysis.json"
        shutil.copy(file_path, legacy_path)
    except Exception as copy_exc:
        print("[signals] failed to write multi_token_analysis.json:", copy_exc)


    # create/update symlink for latest_signals.json (backward-compat)
    try:
        latest_link = out_path / "latest_signals.json"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        latest_link.symlink_to(file_path.name)
    except Exception as link_exc:
        print("[signals] failed to create latest_signals.json symlink:", link_exc)
    print(f"[signals] saved → {file_path}  (count={len(signals)})")

    # -------- logging diagnostics --------
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "total_symbols_checked": len(all_probs),
        "stale_symbols": stale,
        "max_probability": max(all_probs.values()) if all_probs else None,
        "above_threshold": sum(1 for p in all_probs.values() if p >= args.min_proba),
        "signals_generated": len(signals),
    }
    log_path = Path(args.out_dir) / "generation_log.jsonl"
    with log_path.open("a") as lf:
        lf.write(json.dumps(log_entry) + "\n")
    print("[signals] diagnostics →", log_path)

    if len(signals) == 0:
        print("[signals] No signals today. Reasons could be: low volatility / strict threshold / stale data.")

    # Telegram notifications removed – handled by external service.


if __name__ == "__main__":
    main()