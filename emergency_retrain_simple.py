#!/usr/bin/env python3
"""
🚨 SIMPLE EMERGENCY RETRAIN
Uses exact same approach as continuous_learner.py (which works!)
"""

import os
import sys
import json
import subprocess
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/ml-engine/emergency_simple.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_cmd(cmd: str, timeout: int = None) -> tuple:
    """Run command with timeout"""
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return 1, "", "Timeout"
    except Exception as e:
        return 1, "", str(e)

def main():
    logger.info("🚨 SIMPLE EMERGENCY RETRAIN")
    
    # Change to ML engine directory
    os.chdir("/root/ml-engine")
    
    # Stop trading services
    logger.info("🔒 Stopping services...")
    run_cmd("systemctl stop execution-engine ml-generator")
    
    # Create emergency training script (exact copy of continuous_learner approach)
    train_script = '''
import sys
sys.path.append('/root/ml-engine')

import logging
from datetime import datetime
from pathlib import Path
from production_ml_pipeline import ProductionMLPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("emergency")

def emergency_train():
    # Use EXACT same setup as continuous_learner.py
    MODELS_DIR = Path("models")
    DB_PATH = "/root/analytics-tool-v2/market_data.db"  # Correct path
    
    pipe = ProductionMLPipeline(db_path=DB_PATH, models_dir=str(MODELS_DIR))
    
    # Use 30-day window instead of 180 for speed
    logger.info("Loading 30-day dataset...")
    X, y, metadata = pipe.load_and_prepare_data(days_back=30)
    
    logger.info("Running walk-forward validation...")
    cv_results = pipe.walk_forward_validation(X, y, n_splits=3)  # Reduced splits for speed
    
    logger.info("Training production model...")
    model, results = pipe.train_production_model(X, y)
    
    # Save with emergency tag
    tag = "emergency"
    model_path = pipe.save_versioned_model(model, metadata, results, cv_results, tag=tag)
    
    # Write model path for backtesting
    with open("/root/ml-engine/latest_emergency_model.txt", "w") as f:
        f.write(str(model_path))
    
    logger.info(f"✅ Emergency training complete: {model_path}")
    logger.info(f"AUC: {results.get('auc', 0):.3f}, Accuracy: {results.get('accuracy', 0):.3f}")
    
    print(f"SUCCESS:{model_path}")

if __name__ == "__main__":
    try:
        emergency_train()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"ERROR:{e}")
        sys.exit(1)
'''
    
    # Write training script
    with open("/root/ml-engine/emergency_train_simple.py", "w") as f:
        f.write(train_script)
    
    # Execute training
    logger.info("🛠 Starting emergency training...")
    code, out, err = run_cmd(
        "cd /root/ml-engine && /root/ml-engine/venv/bin/python emergency_train_simple.py",
        timeout=900  # 15 minutes
    )
    
    if code != 0:
        logger.error(f"Training failed: {err}")
        return 1
    
    if "SUCCESS:" not in out:
        logger.error("Training completed but no success message")
        return 1
    
    # Extract model path
    model_path = out.split("SUCCESS:")[1].strip().split('\n')[0]
    logger.info(f"✅ Training completed: {model_path}")
    
    # Quick backtest at p97
    logger.info("📊 Running quick backtest at p97...")
    backtest_cmd = f"""
    cd /root/ml-engine && 
    /root/ml-engine/venv/bin/python /root/analytics-tool-v2/backtest_production_features.py \\
        --parquet-dir /root/analytics-tool-v2/tmp_parquet_ultra \\
        --model {model_path} \\
        --bucket-map /root/ml-engine/bucket_mapping.csv \\
        --buckets ultra,high \\
        --percentiles 97 \\
        --min-proba 0.20 \\
        --out /root/ml-backtests/emergency_quick.json
    """
    
    os.makedirs("/root/ml-backtests", exist_ok=True)
    code, out, err = run_cmd(backtest_cmd, timeout=300)
    
    if code != 0:
        logger.error(f"Backtest failed: {err}")
        return 1
    
    # Parse results
    try:
        with open("/root/ml-backtests/emergency_quick.json", 'r') as f:
            results = json.load(f)
        
        if results and len(results) > 0:
            result = results[0]
            trades = result.get('trades', 0)
            avg_pct = result.get('avg_pct', 0)
            win_rate = result.get('win_rate', 0)
            
            logger.info(f"📈 Backtest: {trades} trades, {avg_pct*100:.3f}% avg return, {win_rate*100:.1f}% win rate")
            
            # Deploy if profitable
            if avg_pct >= 0.0035:  # 0.35%
                logger.info("✅ PROFITABLE - Deploying model...")
                run_cmd(f"ln -sf {model_path} /root/ml-engine/models/latest_model.txt")
                run_cmd("systemctl start ml-generator")
                logger.info("🚀 Model deployed and generator started")
                return 0
            else:
                logger.warning(f"❌ UNPROFITABLE: {avg_pct*100:.3f}% < 0.35%")
                return 1
                
    except Exception as e:
        logger.error(f"Failed to parse backtest: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
