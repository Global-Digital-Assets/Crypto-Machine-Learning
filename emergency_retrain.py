#!/usr/bin/env python3
"""
🚨 EMERGENCY MODEL RETRAIN & BACKTEST
AAA-grade implementation with robust error handling and logging
"""

import os
import sys
import json
import time
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/ml-engine/emergency_retrain.log'),
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

def stop_trading_services():
    """Stop all trading services safely"""
    logger.info("🔒 Stopping trading services...")
    services = ['execution-engine', 'ml-generator']
    
    for service in services:
        code, out, err = run_command(f"systemctl stop {service}")
        if code == 0 or "not loaded" in err:
            logger.info(f"✅ {service} stopped/inactive")
        else:
            logger.warning(f"⚠️ {service} stop returned: {err}")

def emergency_retrain():
    """Perform emergency retrain with bucket filtering"""
    logger.info("🛠 Starting emergency 30-day retrain...")
    
    # Change to ml-engine directory
    os.chdir("/root/ml-engine")
    
    # Create Python script for training
    train_script = """
import sys
sys.path.append('/root/ml-engine')
from datetime import datetime, timezone
from production_ml_pipeline import ProductionMLPipeline
import polars as pl
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    # Initialize pipeline with correct DB path
    pipe = ProductionMLPipeline(
        db_path="/root/analytics-tool-v2/market_data.db",
        models_dir="models"
    )
    
    logger.info("Loading 30-day dataset...")
    X, y, meta = pipe.load_and_prepare_data(days_back=30)
    
    # Filter to high/ultra bucket symbols only to speed up training
    bucket_map_path = "/root/ml-engine/bucket_mapping.csv"
    if os.path.exists(bucket_map_path):
        bucket_df = pl.read_csv(bucket_map_path)
        ultra_high_symbols = bucket_df.filter(
            pl.col("bucket").is_in(["ultra", "high"])
        )["symbol"].to_list()
        
        # Filter training data to these symbols
        symbol_mask = pl.Series(meta.get("symbols", [])).is_in(ultra_high_symbols)
        if symbol_mask.sum() > 0:
            X, y = X[symbol_mask], y[symbol_mask]
            logger.info(f"Filtered to {len(ultra_high_symbols)} high/ultra symbols, {len(X)} samples")
    
    logger.info(f"Training on {len(X):,} samples...")
    model, results = pipe.train_production_model(X, y)
    
    logger.info("Running cross-validation...")
    cv_results = pipe.walk_forward_validation(X, y, n_splits=3)
    
    # Save model with emergency tag
    tag = f"emergency_{datetime.now(timezone.utc):%Y%m%d_%H%M%S}"
    model_path = pipe.save_versioned_model(model, meta, results, cv_results, tag=tag)
    
    logger.info(f"✅ Model saved: {model_path}")
    logger.info(f"📊 Training Results: Accuracy={results['accuracy']:.3f}, AUC={results['auc']:.3f}")
    
    # Write model path to file for backtesting
    with open("/root/ml-engine/latest_emergency_model.txt", "w") as f:
        f.write(model_path)
    
    print(f"SUCCESS: {model_path}")
    
except Exception as e:
    logger.error(f"Training failed: {e}")
    print(f"ERROR: {e}")
    sys.exit(1)
"""
    
    # Write training script to file
    script_path = "/root/ml-engine/train_emergency.py"
    with open(script_path, "w") as f:
        f.write(train_script)
    
    # Execute training with venv
    logger.info("Executing training script...")
    code, out, err = run_command(
        "cd /root/ml-engine && /bin/bash -c 'source venv/bin/activate && python train_emergency.py'",
        timeout=1800  # 30 minute timeout
    )
    
    if code != 0:
        logger.error(f"Training failed: {err}")
        return None
        
    if "SUCCESS:" in out:
        model_path = out.split("SUCCESS: ")[1].strip()
        logger.info(f"✅ Training completed: {model_path}")
        return model_path
    else:
        logger.error("Training completed but no model path found")
        return None

def backtest_model(model_path: str) -> dict:
    """Backtest the new model across multiple percentiles"""
    logger.info(f"📊 Backtesting model: {model_path}")
    
    results = {}
    
    # Ensure directories exist
    run_command("mkdir -p /root/ml-backtests")
    
    parquet_dir = "/root/analytics-tool-v2/tmp_parquet_ultra"
    bucket_map = "/root/ml-engine/bucket_mapping.csv"
    
    for percentile in [97, 95, 90]:
        logger.info(f"Testing percentile {percentile}...")
        
        output_file = f"/root/ml-backtests/emergency_p{percentile}.json"
        
        cmd = f"""
        cd /root/ml-engine && /bin/bash -c 'source venv/bin/activate && \
        python /root/analytics-tool-v2/backtest_production_features.py \
            --parquet-dir {parquet_dir} \
            --model {model_path} \
            --bucket-map {bucket_map} \
            --buckets ultra,high \
            --percentiles {percentile} \
            --min-proba 0.20 \
            --out {output_file}'
        """
        
        code, out, err = run_command(cmd, timeout=600)  # 10 min timeout
        
        if code != 0:
            logger.error(f"Backtest failed for p{percentile}: {err}")
            continue
            
        # Parse results
        try:
            with open(output_file, 'r') as f:
                backtest_data = json.load(f)
                
            if backtest_data and len(backtest_data) > 0:
                result = backtest_data[0]
                results[percentile] = {
                    'trades': result.get('trades', 0),
                    'win_rate': result.get('win_rate', 0),
                    'avg_pct': result.get('avg_pct', 0)
                }
                logger.info(f"p{percentile}: {result.get('trades')} trades, "
                           f"{result.get('win_rate')*100:.1f}% win rate, "
                           f"{result.get('avg_pct')*100:.3f}% avg return")
        except Exception as e:
            logger.error(f"Failed to parse backtest results for p{percentile}: {e}")
    
    return results

def deploy_if_profitable(model_path: str, results: dict) -> bool:
    """Deploy model if any percentile shows >=0.35% average return"""
    logger.info("🎯 Checking profitability threshold...")
    
    profitable_configs = []
    for percentile, metrics in results.items():
        avg_pct = metrics.get('avg_pct', 0)
        if avg_pct >= 0.0035:  # 0.35%
            profitable_configs.append((percentile, avg_pct))
            
    if profitable_configs:
        best_config = max(profitable_configs, key=lambda x: x[1])
        logger.info(f"✅ PROFITABLE CONFIG FOUND: p{best_config[0]} with {best_config[1]*100:.3f}% avg return")
        
        # Deploy model
        latest_model_link = "/root/ml-engine/models/latest_model.txt"
        code, out, err = run_command(f"ln -sf {model_path} {latest_model_link}")
        
        if code == 0:
            logger.info(f"✅ Model deployed: {latest_model_link} -> {model_path}")
            
            # Start ML generator
            code, out, err = run_command("systemctl start ml-generator.service")
            if code == 0:
                logger.info("✅ ML generator started")
            else:
                logger.warning(f"⚠️ ML generator start issue: {err}")
            
            return True
        else:
            logger.error(f"❌ Failed to deploy model: {err}")
            return False
    else:
        max_return = max([r.get('avg_pct', 0) for r in results.values()]) if results else 0
        logger.warning(f"❌ NO PROFITABLE CONFIG: Best return {max_return*100:.3f}% < 0.35% threshold")
        logger.warning("Model needs further optimization before deployment")
        return False

def main():
    """Main execution flow"""
    logger.info("🚨 EMERGENCY RETRAIN STARTING")
    
    try:
        # Step 1: Stop trading
        stop_trading_services()
        
        # Step 2: Emergency retrain
        model_path = emergency_retrain()
        if not model_path:
            logger.error("❌ Emergency retrain failed")
            return 1
            
        # Step 3: Backtest
        results = backtest_model(model_path)
        if not results:
            logger.error("❌ Backtesting failed")
            return 1
            
        # Step 4: Deploy if profitable
        deployed = deploy_if_profitable(model_path, results)
        
        # Final summary
        logger.info("🏁 EMERGENCY RETRAIN COMPLETED")
        logger.info(f"Model: {model_path}")
        logger.info("Results:")
        for p, metrics in results.items():
            logger.info(f"  p{p}: {metrics['trades']} trades, "
                       f"{metrics['avg_pct']*100:.3f}% avg return")
        
        if deployed:
            logger.info("✅ MODEL DEPLOYED - Trading can resume")
            return 0
        else:
            logger.info("❌ MODEL NOT DEPLOYED - Needs further work")
            return 1
            
    except Exception as e:
        logger.error(f"❌ Emergency retrain failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
