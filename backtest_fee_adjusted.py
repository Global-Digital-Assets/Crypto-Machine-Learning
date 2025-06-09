#!/usr/bin/env python3
"""
Fee-adjusted backtest: Subtract realistic trading costs from returns
- Exchange fees: 0.04% (taker fees)
- Slippage buffer: 0.02% (ultra bucket can be volatile)
- Total cost: 0.06% per trade
"""
import os
import sys
import json
import polars as pl
import numpy as np
from datetime import datetime
import argparse

# Add paths for imports
sys.path.append('/root/analytics-tool-v2')
sys.path.append('/root/ml-engine')

def load_bucket_mapping(bucket_map_path):
    """Load bucket mapping CSV"""
    df = pl.read_csv(bucket_map_path)
    return df.select(["symbol", "bucket"]).to_dict(as_series=False)

def adjust_returns_for_fees(returns_list, fee_pct=0.06):
    """
    Adjust returns by subtracting trading fees
    fee_pct: Total cost per trade (exchange fees + slippage)
    """
    adjusted_returns = []
    for ret in returns_list:
        # Subtract fee percentage from each trade
        adjusted_ret = ret - fee_pct
        adjusted_returns.append(adjusted_ret)
    
    return adjusted_returns

def run_fee_adjusted_backtest(model_path, parquet_dir, bucket_map_path, 
                             buckets="ultra", percentile=99, min_proba=0.20,
                             fee_pct=0.06):
    """
    Run backtest with fee adjustments
    """
    print(f"🔧 Fee-Adjusted Backtest")
    print(f"   Model: {os.path.basename(model_path)}")
    print(f"   Fee rate: {fee_pct:.2%} per trade")
    print(f"   Target: ≥0.55% after fees")
    
    # Import backtest logic (simplified version)
    from backtest_production_features import ProductionFeatureBacktester
    
    # Run standard backtest first
    backtester = ProductionFeatureBacktester(
        model_path=model_path,
        parquet_dir=parquet_dir,
        bucket_map_path=bucket_map_path
    )
    
    # Get raw results
    raw_results = backtester.run_backtest(
        buckets=[buckets] if isinstance(buckets, str) else buckets,
        percentiles=[percentile],
        min_proba=min_proba
    )
    
    if not raw_results:
        print("❌ No backtest results generated")
        return None
    
    # Process first result (99th percentile)
    result = raw_results[0]
    raw_avg_return = result.get('avg_pct', 0)
    raw_trades = result.get('trades', 0)
    raw_win_rate = result.get('win_rate', 0)
    
    print(f"\n📊 RAW Results (before fees):")
    print(f"   Trades: {raw_trades}")
    print(f"   Avg Return: {raw_avg_return:.3f}%")
    print(f"   Win Rate: {raw_win_rate:.1%}")
    
    # Apply fee adjustment
    fee_adjusted_return = raw_avg_return - fee_pct
    
    print(f"\n💰 FEE-ADJUSTED Results:")
    print(f"   Trades: {raw_trades}")
    print(f"   Avg Return: {fee_adjusted_return:.3f}%")
    print(f"   Win Rate: {raw_win_rate:.1%}")
    print(f"   Fee Impact: -{fee_pct:.2f}%")
    
    # Validation
    target_return = 0.55  # 0.35% + safety buffer
    passed = fee_adjusted_return >= target_return and raw_trades >= 10
    
    print(f"\n✅ Validation:")
    print(f"   Target: ≥{target_return}%")
    print(f"   Actual: {fee_adjusted_return:.3f}%")
    print(f"   Status: {'✅ PASS' if passed else '❌ FAIL'}")
    
    if passed:
        print(f"   💡 Net edge after costs: {fee_adjusted_return:.3f}%")
        if fee_adjusted_return >= 0.6:
            print(f"   🚀 Excellent edge - ready for 3x leverage!")
    else:
        print(f"   ⚠️  Edge too thin after fees - consider optimization")
    
    return {
        'passed': passed,
        'raw_avg_return': raw_avg_return,
        'fee_adjusted_return': fee_adjusted_return,
        'trades': raw_trades,
        'win_rate': raw_win_rate,
        'fee_pct': fee_pct,
        'target_return': target_return
    }

def main():
    parser = argparse.ArgumentParser(description="Fee-adjusted backtest")
    parser.add_argument("--model", required=True, help="Model file path")
    parser.add_argument("--parquet-dir", required=True, help="Parquet directory")
    parser.add_argument("--bucket-map", required=True, help="Bucket mapping CSV")
    parser.add_argument("--buckets", default="ultra", help="Bucket filter")
    parser.add_argument("--percentile", type=int, default=99, help="Percentile threshold")
    parser.add_argument("--min-proba", type=float, default=0.20, help="Min probability")
    parser.add_argument("--fee-pct", type=float, default=0.06, help="Fee percentage per trade")
    parser.add_argument("--out", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Run fee-adjusted backtest
    results = run_fee_adjusted_backtest(
        model_path=args.model,
        parquet_dir=args.parquet_dir,
        bucket_map_path=args.bucket_map,
        buckets=args.buckets,
        percentile=args.percentile,
        min_proba=args.min_proba,
        fee_pct=args.fee_pct
    )
    
    if results and args.out:
        with open(args.out, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\n💾 Results saved: {args.out}")
    
    return 0 if results and results['passed'] else 1

if __name__ == "__main__":
    sys.exit(main())
