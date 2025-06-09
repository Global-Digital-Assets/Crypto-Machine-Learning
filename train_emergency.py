
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
