#!/usr/bin/env python3
"""
🚨 EMERGENCY 4-CORE OPTIMIZED RETRAIN
Uses all 4 CPU cores with memory-efficient processing
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/ml-engine/emergency_retrain_4core.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_command(cmd: str, timeout: int = None) -> tuple:
    """Execute shell command with proper error handling"""
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {cmd}")
        return 1, "", "Command timed out"
    except Exception as e:
        logger.error(f"Command failed: {cmd}, Error: {e}")
        return 1, "", str(e)

def stop_trading():
    """Stop trading services"""
    logger.info("🔒 Stopping trading...")
    services = ['execution-engine', 'ml-generator']
    for service in services:
        code, out, err = run_command(f"systemctl stop {service}")
        if code == 0 or "not loaded" in err:
            logger.info(f"✅ {service} stopped")

def emergency_retrain_4core():
    """4-core optimized emergency retrain"""
    logger.info("🛠 4-CORE EMERGENCY RETRAIN STARTING...")
    
    os.chdir("/root/ml-engine")
    
    # Create optimized training script
    train_script = '''
import os
import sys
import gc
sys.path.append('/root/ml-engine')
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

from datetime import datetime, timezone
from production_ml_pipeline import ProductionMLPipeline
import polars as pl
import lightgbm as lgb
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    logger.info("🚀 4-Core optimized training starting...")
    
    pipe = ProductionMLPipeline(
        db_path="/root/analytics-tool-v2/market_data.db",
        models_dir="models"
    )
    
    # Load 14 days instead of 30 to reduce memory pressure
    logger.info("Loading 14-day dataset for speed...")
    X, y, meta = pipe.load_and_prepare_data(days_back=14)
    
    # Filter to high/ultra symbols only
    try:
        bucket_df = pl.read_csv("/root/ml-engine/bucket_mapping.csv")
        target_symbols = bucket_df.filter(
            pl.col("bucket").is_in(["ultra", "high"])
        )["symbol"].to_list()
        logger.info(f"Filtering to {len(target_symbols)} high/ultra symbols")
        
        # Get symbol column from metadata or create from data
        if hasattr(meta, 'get') and meta.get("symbols"):
            symbols = meta["symbols"]
        else:
            # Reconstruct from data shape - this is approximate
            symbols_per_period = len(pipe.symbols)
            total_periods = len(X) // symbols_per_period
            symbols = (pipe.symbols * (total_periods + 1))[:len(X)]
        
        # Create mask for target symbols
        symbol_mask = pl.Series(symbols).is_in(target_symbols)
        X_filtered = X[symbol_mask]
        y_filtered = y[symbol_mask]
        
        logger.info(f"Filtered: {len(X):,} → {len(X_filtered):,} samples")
        X, y = X_filtered, y_filtered
        
    except Exception as e:
        logger.warning(f"Symbol filtering failed: {e}, using all data")
    
    # Force garbage collection
    gc.collect()
    
    logger.info(f"Training on {len(X):,} samples with 4 cores...")
    
    # Split data maintaining time order
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 4-core optimized LightGBM params
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.1,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'num_threads': 4,  # Use all 4 cores
        'force_row_wise': True,  # Better for multi-core
        'seed': 42
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=pipe.feature_names)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # Train with 4 cores
    logger.info("Training LightGBM with 4 cores...")
    model = lgb.train(
        params, 
        train_data, 
        num_boost_round=300,  # Reduced for speed
        valid_sets=[test_data],
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(50)]
    )
    
    # Quick CV (reduced splits for speed)
    logger.info("Quick 2-fold CV...")
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, roc_auc_score
    
    tscv = TimeSeriesSplit(n_splits=2)
    cv_scores = []
    
    for train_idx, val_idx in tscv.split(X):
        X_cv_train, X_cv_val = X[train_idx], X[val_idx]
        y_cv_train, y_cv_val = y[train_idx], y[val_idx]
        
        cv_data = lgb.Dataset(X_cv_train, label=y_cv_train)
        cv_model = lgb.train(params, cv_data, num_boost_round=200, callbacks=[lgb.log_evaluation(0)])
        
        y_pred = cv_model.predict(X_cv_val)
        auc = roc_auc_score(y_cv_val, y_pred)
        cv_scores.append(auc)
    
    cv_mean = sum(cv_scores) / len(cv_scores)
    logger.info(f"CV AUC: {cv_mean:.3f}")
    
    # Final evaluation
    y_pred_test = model.predict(X_test)
    test_auc = roc_auc_score(y_test, y_pred_test)
    test_acc = accuracy_score(y_test, (y_pred_test > 0.5).astype(int))
    
    # Save model
    tag = f"emergency_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    model_filename = f"lgbm_{tag}.txt"
    model_path = f"models/{model_filename}"
    
    model.save_model(model_path)
    
    # Create metadata
    metadata = {
        "model_info": {
            "filename": model_filename,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "training_samples": len(X_train),
            "test_auc": test_auc,
            "test_accuracy": test_acc,
            "cv_auc_mean": cv_mean,
            "cores_used": 4,
            "days_back": 14
        }
    }
    
    # Save metadata
    with open(f"models/{tag}_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    # Write model path for backtesting
    with open("/root/ml-engine/latest_emergency_model.txt", "w") as f:
        f.write(model_path)
    
    logger.info(f"✅ 4-core training complete!")
    logger.info(f"Model: {model_path}")
    logger.info(f"Test AUC: {test_auc:.3f}, Accuracy: {test_acc:.3f}")
    logger.info(f"CV AUC: {cv_mean:.3f}")
    
    print(f"SUCCESS:{model_path}")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    import traceback
    traceback.print_exc()
    print(f"ERROR:{e}")
    sys.exit(1)
'''
    
    # Write and execute training script
    with open("/root/ml-engine/train_4core.py", "w") as f:
        f.write(train_script)
    
    logger.info("Executing 4-core training...")
    code, out, err = run_command(
        "cd /root/ml-engine && /bin/bash -c 'source venv/bin/activate && python train_4core.py'",
        timeout=1200  # 20 minutes
    )
    
    if code != 0:
        logger.error(f"4-core training failed: {err}")
        return None
        
    if "SUCCESS:" in out:
        model_path = out.split("SUCCESS:")[1].strip()
        logger.info(f"✅ 4-core training completed: {model_path}")
        return model_path
    else:
        logger.error("Training completed but no model path found")
        return None

def parallel_backtest(model_path: str) -> dict:
    """Run backtests in parallel using 4 cores"""
    logger.info(f"📊 4-CORE PARALLEL BACKTESTING: {model_path}")
    
    run_command("mkdir -p /root/ml-backtests")
    
    # Create backtest scripts for each percentile
    percentiles = [97, 95, 90]
    scripts = []
    
    for p in percentiles:
        script_content = f'''#!/bin/bash
cd /root/ml-engine
source venv/bin/activate
python /root/analytics-tool-v2/backtest_production_features.py \\
    --parquet-dir /root/analytics-tool-v2/tmp_parquet_ultra \\
    --model {model_path} \\
    --bucket-map /root/ml-engine/bucket_mapping.csv \\
    --buckets ultra,high \\
    --percentiles {p} \\
    --min-proba 0.20 \\
    --out /root/ml-backtests/emergency_p{p}.json
'''
        script_path = f"/root/ml-engine/backtest_p{p}.sh"
        with open(script_path, "w") as f:
            f.write(script_content)
        run_command(f"chmod +x {script_path}")
        scripts.append(script_path)
    
    # Run all backtests in parallel
    logger.info("Starting parallel backtests...")
    processes = []
    
    for script in scripts:
        cmd = f"nohup {script} > {script}.log 2>&1 &"
        run_command(cmd)
    
    # Wait for completion (check every 10 seconds)
    max_wait = 300  # 5 minutes
    waited = 0
    
    while waited < max_wait:
        time.sleep(10)
        waited += 10
        
        # Check if all output files exist
        all_done = True
        for p in percentiles:
            if not os.path.exists(f"/root/ml-backtests/emergency_p{p}.json"):
                all_done = False
                break
        
        if all_done:
            logger.info(f"✅ All backtests completed in {waited}s")
            break
    
    # Parse results
    results = {}
    for p in percentiles:
        try:
            with open(f"/root/ml-backtests/emergency_p{p}.json", 'r') as f:
                data = json.load(f)
                if data and len(data) > 0:
                    result = data[0]
                    results[p] = {
                        'trades': result.get('trades', 0),
                        'win_rate': result.get('win_rate', 0),
                        'avg_pct': result.get('avg_pct', 0)
                    }
                    logger.info(f"p{p}: {result.get('trades')} trades, "
                               f"{result.get('avg_pct')*100:.3f}% avg return")
        except Exception as e:
            logger.error(f"Failed to parse p{p} results: {e}")
    
    return results

def deploy_best_model(model_path: str, results: dict) -> bool:
    """Deploy if profitable"""
    logger.info("🎯 Checking profitability...")
    
    best_return = 0
    best_config = None
    
    for p, metrics in results.items():
        avg_pct = metrics.get('avg_pct', 0)
        if avg_pct > best_return:
            best_return = avg_pct
            best_config = p
    
    if best_return >= 0.0035:  # 0.35%
        logger.info(f"✅ PROFITABLE: p{best_config} = {best_return*100:.3f}%")
        
        # Deploy
        run_command(f"ln -sf {model_path} /root/ml-engine/models/latest_model.txt")
        run_command("systemctl start ml-generator.service")
        
        logger.info("✅ MODEL DEPLOYED & GENERATOR STARTED")
        return True
    else:
        logger.warning(f"❌ UNPROFITABLE: Best = {best_return*100:.3f}% < 0.35%")
        return False

def main():
    """Main 4-core execution"""
    logger.info("🚨 4-CORE EMERGENCY RETRAIN STARTING")
    
    try:
        # Stop trading
        stop_trading()
        
        # 4-core retrain
        model_path = emergency_retrain_4core()
        if not model_path:
            return 1
        
        # Parallel backtest
        results = parallel_backtest(model_path)
        if not results:
            return 1
        
        # Deploy if profitable
        deployed = deploy_best_model(model_path, results)
        
        # Summary
        logger.info("🏁 4-CORE RETRAIN COMPLETE")
        logger.info(f"Model: {model_path}")
        for p, metrics in results.items():
            logger.info(f"p{p}: {metrics['avg_pct']*100:.3f}% return")
        
        return 0 if deployed else 1
        
    except Exception as e:
        logger.error(f"❌ 4-core retrain failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
