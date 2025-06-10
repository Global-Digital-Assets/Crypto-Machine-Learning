#!/usr/bin/env python3
"""
Direct emergency training - no timeout, simpler approach
Uses exact same pattern as continuous_learner.py
"""

import sys
import os
import json
sys.path.append('/root/ml-engine')
os.chdir('/root/ml-engine')

import logging
from datetime import datetime
from pathlib import Path
from production_ml_pipeline import ProductionMLPipeline

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/ml-engine/train_direct.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("direct_emergency")

def main():
    logger.info("🚨 DIRECT EMERGENCY TRAINING - NO TIMEOUT")
    
    # Stop services
    os.system("systemctl stop execution-engine ml-generator")
    
    # Use EXACT same setup as continuous_learner.py
    MODELS_DIR = Path("models")
    DB_PATH = "/root/analytics-tool-v2/market_data.db"
    
    pipe = ProductionMLPipeline(db_path=DB_PATH, models_dir=str(MODELS_DIR))
    
    # Use 21-day window for faster training
    logger.info("Loading 21-day dataset...")
    X, y, metadata = pipe.load_and_prepare_data(days_back=21)
    
    logger.info("Running quick 2-fold validation...")
    cv_results = pipe.walk_forward_validation(X, y, n_splits=2)
    
    logger.info("Training production model...")
    model, results = pipe.train_production_model(X, y)
    
    # Save with emergency tag
    tag = "emergency"
    model_path = pipe.save_versioned_model(model, metadata, results, cv_results, tag=tag)
    
    logger.info(f"✅ Emergency training complete: {model_path}")
    logger.info(f"AUC: {results.get('auc', 0):.3f}, Accuracy: {results.get('accuracy', 0):.3f}")
    
    # Quick backtest
    logger.info("📊 Quick backtest...")
    os.makedirs("/root/ml-backtests", exist_ok=True)
    
    backtest_cmd = f"""
    /root/ml-engine/venv/bin/python /root/analytics-tool-v2/backtest_production_features.py \
        --parquet-dir /root/analytics-tool-v2/tmp_parquet_ultra \
        --model {model_path} \
        --bucket-map /root/ml-engine/bucket_mapping.csv \
        --buckets ultra,high \
        --percentiles 97 \
        --min-proba 0.20 \
        --out /root/ml-backtests/emergency_final.json
    """
    
    logger.info("Running backtest...")
    result = os.system(backtest_cmd)
    
    if result == 0:
        try:
            with open("/root/ml-backtests/emergency_final.json", 'r') as f:
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
                    os.system(f"ln -sf {model_path} /root/ml-engine/models/latest_model.txt")
                    os.system("systemctl start ml-generator")
                    logger.info("🚀 Model deployed and generator started")
                    print("SUCCESS: Model deployed")
                else:
                    logger.warning(f"❌ UNPROFITABLE: {avg_pct*100:.3f}% < 0.35%")
                    print(f"UNPROFITABLE: {avg_pct*100:.3f}%")
        except Exception as e:
            logger.error(f"Failed to parse backtest: {e}")
            print(f"ERROR: {e}")
    else:
        logger.error("Backtest failed")
        print("ERROR: Backtest failed")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")
        sys.exit(1)
