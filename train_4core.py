
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
