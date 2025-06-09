
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
