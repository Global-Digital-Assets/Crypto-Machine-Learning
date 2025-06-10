#!/usr/bin/env bash
set -euo pipefail
# Usage: post_retrain_check.sh <NEW_MODEL_PATH>
NEW="${1:-}"
if [[ -z "$NEW" ]]; then
  NEW=$(ls -t /root/ml-engine/models/lgbm_production_*.txt 2>/dev/null | head -n1 || true)
fi
if [[ -z "$NEW" ]]; then
  logger -t post_retrain_gate "no new model provided"
  exit 1
fi
BASE="/root/ml-engine/models/lgbm_monthly_20250607_131503.txt"
PARQUET_DIR="/root/analytics-tool-v2/tmp_parquet_ultra"
BACKTEST="/root/analytics-tool-v2/backtest_production_features.py"
BUCKET_MAP="/root/ml-engine/bucket_mapping.csv"
PY="/root/ml-engine/venv/bin/python"
TMP_NEW=$(mktemp)
TMP_BASE=$(mktemp)
$PY $BACKTEST --model "$NEW" --parquet-dir "$PARQUET_DIR" --bucket-map "$BUCKET_MAP" --buckets high,ultra --percentiles 97 --out "$TMP_NEW"
$PY $BACKTEST --model "$BASE" --parquet-dir "$PARQUET_DIR" --bucket-map "$BUCKET_MAP" --buckets high,ultra --percentiles 97 --out "$TMP_BASE"
NEW_WIN=$(jq -r '.[0].win_rate' "$TMP_NEW")
BASE_WIN=$(jq -r '.[0].win_rate' "$TMP_BASE")
DIFF=$(awk -v n="$NEW_WIN" -v b="$BASE_WIN" 'BEGIN{print b-n}')
logger -t post_retrain_gate "base=$BASE_WIN new=$NEW_WIN diff=$DIFF"
# accept if win-rate not worse by >2 percentage points
awk -v d="$DIFF" 'BEGIN{exit (d>0.02)}'
