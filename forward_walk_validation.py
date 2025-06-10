#!/usr/bin/env python3
"""
Forward-walk validation: Train on data up to T-14 days, test on last 14 days
This prevents data leakage and validates model performance on truly unseen data
"""
import os
import sys
import json
import sqlite3
import polars as pl
from datetime import datetime, timedelta
import subprocess
import tempfile
import shutil

# Add the current directory to Python path for imports
sys.path.append('/root/analytics-tool-v2')
sys.path.append('/root/ml-engine')

from production_ml_pipeline import ProductionMLPipeline

class ForwardWalkValidator:
    def __init__(self, db_path="/root/analytics-tool-v2/market_data.db", 
                 parquet_dir="/root/analytics-tool-v2/tmp_parquet_ultra",
                 bucket_map="/root/ml-engine/bucket_mapping.csv"):
        self.db_path = db_path
        self.parquet_dir = parquet_dir
        self.bucket_map = bucket_map
        self.results = {}
        
    def run_forward_walk(self, split_days=14, target_return=0.5):
        """
        Run forward-walk validation:
        - Train on all data except last split_days
        - Test backtest on last split_days only
        - Validate EV >= target_return %
        """
        print(f"🔄 Forward-walk validation: {split_days}d holdout")
        
        # Calculate split date
        end_date = datetime.now()
        split_date = end_date - timedelta(days=split_days)
        split_date_str = split_date.strftime("%Y-%m-%d")
        
        print(f"📅 Split date: {split_date_str}")
        print(f"   Training: up to {split_date_str}")
        print(f"   Testing: {split_date_str} to present")
        
        # Step 1: Train model on pre-split data
        print("\n🔧 Training model on pre-split data...")
        train_model_path = self._train_forward_walk_model(split_date_str)
        
        # Step 2: Create holdout parquet data 
        print("\n📊 Creating holdout test dataset...")
        holdout_parquet_dir = self._create_holdout_parquet(split_date_str)
        
        # Step 3: Backtest on holdout data only
        print("\n🧪 Backtesting on unseen holdout data...")
        backtest_results = self._backtest_holdout(train_model_path, holdout_parquet_dir)
        
        # Step 4: Validate results
        print("\n✅ Forward-walk validation results:")
        avg_return = backtest_results.get('avg_pct', 0)
        trades = backtest_results.get('trades', 0)
        win_rate = backtest_results.get('win_rate', 0)
        
        print(f"   Trades: {trades}")
        print(f"   Avg Return: {avg_return:.3f}%")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Target: ≥{target_return}%")
        
        passed = avg_return >= target_return and trades >= 10
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"   Status: {status}")
        
        # Cleanup
        self._cleanup_temp_files(train_model_path, holdout_parquet_dir)
        
        return {
            'passed': passed,
            'avg_return': avg_return,
            'trades': trades,
            'win_rate': win_rate,
            'target_return': target_return
        }
    
    def _train_forward_walk_model(self, split_date_str):
        """Train model only on data before split_date"""
        # Create temporary model path
        temp_model_path = f"/tmp/forward_walk_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        
        # Load and filter training data
        pipeline = ProductionMLPipeline()
        
        # Load data with date filter
        print(f"   Loading training data (< {split_date_str})...")
        df = pipeline.load_and_prepare_data(days_back=30)
        
        # Filter to before split date
        df_train = df.filter(pl.col("timestamp") < split_date_str)
        
        print(f"   Training samples: {len(df_train):,}")
        
        # Train model
        model, metrics = pipeline.train_model(df_train)
        
        # Save model
        model.save_model(temp_model_path)
        print(f"   Model saved: {temp_model_path}")
        print(f"   Training metrics: AUC={metrics.get('auc', 0):.3f}")
        
        return temp_model_path
    
    def _create_holdout_parquet(self, split_date_str):
        """Create parquet files for holdout period only"""
        temp_parquet_dir = f"/tmp/holdout_parquet_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(temp_parquet_dir, exist_ok=True)
        
        # Copy structure from original parquet dir
        if os.path.exists(self.parquet_dir):
            for file in os.listdir(self.parquet_dir):
                if file.endswith('.parquet'):
                    src_path = os.path.join(self.parquet_dir, file)
                    dst_path = os.path.join(temp_parquet_dir, file)
                    
                    # Load, filter by date, save
                    df = pl.read_parquet(src_path)
                    df_holdout = df.filter(pl.col("timestamp") >= split_date_str)
                    
                    if len(df_holdout) > 0:
                        df_holdout.write_parquet(dst_path)
                        print(f"   Created holdout: {file} ({len(df_holdout)} rows)")
        
        return temp_parquet_dir
    
    def _backtest_holdout(self, model_path, holdout_parquet_dir):
        """Run backtest on holdout data only"""
        output_file = f"/tmp/forward_walk_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Run backtest command
        cmd = [
            "/root/ml-engine/venv/bin/python",
            "/root/analytics-tool-v2/backtest_production_features.py",
            "--model", model_path,
            "--parquet-dir", holdout_parquet_dir,
            "--bucket-map", self.bucket_map,
            "--buckets", "ultra",
            "--percentiles", "99",
            "--min-proba", "0.20",
            "--out", output_file
        ]
        
        result = subprocess.run(cmd, cwd="/root/analytics-tool-v2", 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Backtest failed: {result.stderr}")
            return {}
        
        # Load results
        if os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            
            if results and len(results) > 0:
                return results[0]  # First percentile result
        
        return {}
    
    def _cleanup_temp_files(self, model_path, parquet_dir):
        """Clean up temporary files"""
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
            if os.path.exists(parquet_dir):
                shutil.rmtree(parquet_dir)
        except Exception as e:
            print(f"⚠️  Cleanup warning: {e}")

def main():
    validator = ForwardWalkValidator()
    
    print("🚀 Starting Forward-Walk Validation")
    print("=" * 50)
    
    # Run validation with 14-day holdout, 0.5% target
    results = validator.run_forward_walk(split_days=14, target_return=0.5)
    
    print("\n" + "=" * 50)
    print("📊 FINAL RESULTS:")
    print(f"Forward-walk validation: {'PASSED' if results['passed'] else 'FAILED'}")
    print(f"Unseen data performance: {results['avg_return']:.3f}% avg return")
    print(f"Trade count: {results['trades']}")
    
    if results['passed']:
        print("✅ Model generalizes well - safe for deployment!")
    else:
        print("❌ Model may be overfitted - consider retraining")
    
    return 0 if results['passed'] else 1

if __name__ == "__main__":
    sys.exit(main())
