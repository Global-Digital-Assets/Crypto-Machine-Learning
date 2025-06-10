#!/usr/bin/env python3
"""
🏭 PRODUCTION ML PIPELINE
Complete institutional-grade ML system with versioning, monitoring, and auto-retrain
"""

import json
import logging
import sqlite3
import time
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import polars as pl
import lightgbm as lgb

# Optimize for 8-core usage
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')

import numpy as np
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import joblib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_ml.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ProductionMLPipeline:
    """Production-grade ML pipeline with versioning, monitoring, and auto-retrain"""
    
    def __init__(self, db_path: str = "market_data.db", models_dir: str = "models"):
        self.db_path = db_path
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        self.symbols = [
            'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
            'ADAUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT', 'ATOMUSDT',
            'MATICUSDT', 'ALGOUSDT', 'LTCUSDT', 'BCHUSDT', 'FILUSDT',
            'TRXUSDT', 'VETUSDT', 'XLMUSDT', 'ICPUSDT', 'THETAUSDT',
            'EOSUSDT', 'AXSUSDT', 'SANDUSDT', 'MANAUSDT', 'GALAUSDT',
            'CHZUSDT', 'ENJUSDT', 'ZILUSDT', 'BATUSDT', 'ZECUSDT',
            'BONKUSDT', 'FLOKIUSDT', 'LUNCUSDT'
        ]
        
        self.feature_names = [
            'ret_5m', 'ret_1h', 'ret_4h', 'ret_24h',
            'vol_spike', 'volatility_1h', 'volatility_4h', 
            'hl_spread', 'bar_return', 'sma_distance', 'rsi14'
        ]
    
    def load_and_prepare_data(self, days_back: int = 180, bar_sec: int = 300) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """Load and prepare training data with metadata tracking"""
        logger.info(f"📊 Loading data for last {days_back} days...")
        
        conn = sqlite3.connect(self.db_path)
        cutoff_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        query = """
        SELECT symbol, timestamp, open, high, low, close, volume
        FROM candles 
        WHERE symbol IN ({}) AND timestamp >= {}
        ORDER BY symbol, timestamp
        """.format(','.join([f"'{s}'" for s in self.symbols]), cutoff_timestamp)
        
        df = pl.read_database(query, conn)
        conn.close()
        
        # Create bars and engineer features
        df_bars = self._create_bars(df, bar_sec)
        df_features = self._engineer_features(df_bars, bar_sec)
        
        # Prepare final dataset
        clean_df = df_features.drop_nulls()
        X = clean_df.select(self.feature_names).to_numpy()
        y = clean_df.select("target").to_numpy().flatten()
        
        # Metadata for versioning
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "days_back": days_back,
            "total_samples": len(X),
            "positive_rate": float(y.mean()),
            "feature_names": self.feature_names,
            "symbols_count": len(self.symbols),
            "data_range": {
                "start": cutoff_timestamp,
                "end": int(datetime.now().timestamp())
            }
        }
        
        logger.info(f"✅ Prepared {len(X):,} samples with {len(self.feature_names)} features")
        return X, y, metadata
    
    def walk_forward_validation(self, X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> Dict:
        """Walk-forward time series validation to detect overfitting"""
        logger.info("🔍 Running walk-forward validation...")
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        scores = []
        auc_scores = []
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
            X_train_fold, X_test_fold = X[train_idx], X[test_idx]
            y_train_fold, y_test_fold = y[train_idx], y[test_idx]
            
            # Train fold model
            train_data = lgb.Dataset(X_train_fold, label=y_train_fold)
            params = self._get_lgb_params()
            
            model = lgb.train(
                params, train_data, num_boost_round=200, 
                callbacks=[lgb.log_evaluation(0)]
            )
            
            # Evaluate
            y_pred = model.predict(X_test_fold)
            y_pred_binary = (y_pred > 0.5).astype(int)
            
            acc = accuracy_score(y_test_fold, y_pred_binary)
            auc = roc_auc_score(y_test_fold, y_pred)
            
            scores.append(acc)
            auc_scores.append(auc)
            
            logger.info(f"Fold {fold+1}: Accuracy={acc:.3f}, AUC={auc:.3f}")
        
        cv_results = {
            "mean_accuracy": np.mean(scores),
            "std_accuracy": np.std(scores),
            "mean_auc": np.mean(auc_scores),
            "std_auc": np.std(auc_scores),
            "all_scores": scores,
            "all_auc": auc_scores
        }
        
        logger.info(f"📈 CV Results: Acc={cv_results['mean_accuracy']:.3f}±{cv_results['std_accuracy']:.3f}, AUC={cv_results['mean_auc']:.3f}±{cv_results['std_auc']:.3f}")
        return cv_results
    
    def train_production_model(self, X: np.ndarray, y: np.ndarray) -> Tuple[lgb.Booster, Dict]:
        """Train final production model with feature importance analysis"""
        logger.info("🚀 Training production model...")
        
        # Split data (keeping time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        params = self._get_lgb_params()
        model = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[test_data], 
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Feature importance analysis
        importance = model.feature_importance(importance_type='gain')
        feature_importance = dict(zip(self.feature_names, importance.tolist()))
        
        # Final evaluation
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred_binary),
            "auc": roc_auc_score(y_test, y_pred),
            "feature_importance": feature_importance,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_params": params
        }
        
        logger.info(f"📊 Final Model: Accuracy={results['accuracy']:.3f}, AUC={results['auc']:.3f}")
        return model, results
    
    def save_versioned_model(self, model: lgb.Booster, metadata: Dict, results: Dict, cv_results: Dict, tag: str = "production"):
        """Save model with complete versioning and metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"lgbm_{tag}_{timestamp}.txt"
        model_path = self.models_dir / model_filename
        
        # Save LightGBM model
        model.save_model(str(model_path))
        
        # Create comprehensive metadata
        full_metadata = {
            "model_info": {
                "filename": model_filename,
                "timestamp": timestamp,
                "model_type": "LightGBM",
                "version": "production_v1.0",
                "tag": tag
            },
            "data_info": metadata,
            "performance": results,
            "cross_validation": cv_results,
            "feature_names": self.feature_names,
            "symbols": self.symbols
        }
        
        # Save metadata
        metadata_path = self.models_dir / f"metadata_{tag}_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(full_metadata, f, indent=2)
        
        # Update latest model symlink
        latest_model_path = self.models_dir / f"latest_model_{tag}.txt"
        latest_metadata_path = self.models_dir / f"latest_metadata_{tag}.json"
        
        if latest_model_path.exists():
            latest_model_path.unlink()
        if latest_metadata_path.exists():
            latest_metadata_path.unlink()
            
        latest_model_path.symlink_to(model_filename)
        latest_metadata_path.symlink_to(f"metadata_{tag}_{timestamp}.json")
        
        logger.info(f"💾 Model saved: {model_path}")
        logger.info(f"📋 Metadata saved: {metadata_path}")
        
        # Clean up old models (keep last 7)
        self._cleanup_old_models(tag)
        
        return model_path, metadata_path
    
    def _create_bars(self, df: pl.DataFrame, bar_sec: int) -> pl.DataFrame:
        """Create OHLCV bars with given interval in seconds"""
        return (
            df
            .with_columns([(pl.col("timestamp") // bar_sec * bar_sec).alias("ts")])
            .group_by(["symbol", "ts"])
            .agg([
                pl.col("open").first(),
                pl.col("high").max(),
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum()
            ])
            .sort(["symbol", "ts"])
        )
    
    def _engineer_features(self, df: pl.DataFrame, bar_sec: int) -> pl.DataFrame:
        """Engineer ML features based on dynamic bar size"""
        bars_per_1h = int(3600 / bar_sec)
        bars_per_4h = int(4 * 3600 / bar_sec)
        bars_per_24h = int(24 * 3600 / bar_sec)
        return (
            df
            .sort(["symbol", "ts"])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret_5m"),
            ])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(bars_per_1h).over("symbol") - 1).alias("ret_1h"),
                (pl.col("close") / pl.col("close").shift(bars_per_4h).over("symbol") - 1).alias("ret_4h"),
                (pl.col("close") / pl.col("close").shift(bars_per_24h).over("symbol") - 1).alias("ret_24h"),
                pl.col("volume").rolling_mean(bars_per_1h).over("symbol").alias("vol_ma_1h"),
                (pl.col("volume") / pl.col("volume").rolling_mean(bars_per_1h).over("symbol")).alias("vol_spike"),
                pl.col("ret_5m").rolling_std(bars_per_1h).over("symbol").alias("volatility_1h"),
                pl.col("ret_5m").rolling_std(bars_per_4h).over("symbol").alias("volatility_4h"),
                ((pl.col("high") + pl.col("low")) / 2 / pl.col("close") - 1).alias("hl_spread"),
                (pl.col("close") / pl.col("open") - 1).alias("bar_return"),
                pl.col("close").rolling_mean(14).over("symbol").alias("sma_14"),
                (pl.col("close") / pl.col("close").rolling_mean(14).over("symbol") - 1).alias("sma_distance"),
                pl.when(pl.col("ret_5m") > 0).then(pl.col("ret_5m")).otherwise(0).rolling_mean(14).over("symbol").alias("avg_gains"),
                pl.when(pl.col("ret_5m") < 0).then(-pl.col("ret_5m")).otherwise(0).rolling_mean(14).over("symbol").alias("avg_losses")
            ])
            .with_columns([
                (100 - 100 / (1 + pl.col("avg_gains") / (pl.col("avg_losses") + 1e-10))).alias("rsi14"),
                (pl.col("ret_1h").shift(-bars_per_1h).over("symbol") > 0.005).alias("target")
            ])
            .drop(["avg_gains", "avg_losses", "vol_ma_1h", "sma_14"])
        )
    
    def _get_lgb_params(self) -> Dict:
        """Get optimized LightGBM parameters"""
        return {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 63,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 100,
            'num_threads': 8,
            'verbosity': -1,
            'random_state': 42
        }
    
    def _cleanup_old_models(self, tag: str = "production"):
        """Keep only the last 7 model versions"""
        model_files = sorted([f for f in self.models_dir.glob(f"lgbm_{tag}_*.txt")], reverse=True)
        metadata_files = sorted([f for f in self.models_dir.glob(f"metadata_{tag}_*.json")], reverse=True)
        
        for old_model in model_files[7:]:
            old_model.unlink()
            logger.info(f"🗑️ Cleaned up old model: {old_model.name}")
            
        for old_metadata in metadata_files[7:]:
            old_metadata.unlink()
    
    def run_full_pipeline(self, days_back: int = 180, tag: str = "production", bar_sec: int = 300):
        """Run complete production training pipeline"""
        start_time = time.time()
        logger.info("🏭 Starting PRODUCTION ML Pipeline")
        
        try:
            # 1. Load and prepare data
            X, y, metadata = self.load_and_prepare_data(days_back, bar_sec)
            
            # 2. Walk-forward validation
            cv_results = self.walk_forward_validation(X, y)
            
            # 3. Train production model
            model, results = self.train_production_model(X, y)
            
            # 4. Save versioned model
            model_path, metadata_path = self.save_versioned_model(model, metadata, results, cv_results, tag)
            
            total_time = time.time() - start_time
            logger.info(f"✅ PRODUCTION PIPELINE COMPLETE in {total_time:.1f} seconds")
            
            return {
                "model_path": model_path,
                "metadata_path": metadata_path,
                "performance": results,
                "cv_results": cv_results
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def run_warm_start_pipeline(self, days_back: int = 180, tag: str = "production", bar_sec: int = 300):
        """Run warm-start pipeline"""
        start_time = time.time()
        logger.info("🏭 Starting WARM-START ML Pipeline")
        
        try:
            # 1. Load and prepare data
            X, y, metadata = self.load_and_prepare_data(days_back, bar_sec)
            
            # 2. Load existing model
            latest_model_path = self.models_dir / f"latest_model_{tag}.txt"
            model = lgb.Booster(model_file=str(latest_model_path))
            
            # 3. Train warm-start model
            model, results = self.train_warm_start_model(model, X, y)
            
            # 4. Save versioned model
            model_path, metadata_path = self.save_versioned_model(model, metadata, results, {}, tag)
            
            total_time = time.time() - start_time
            logger.info(f"✅ WARM-START PIPELINE COMPLETE in {total_time:.1f} seconds")
            
            return {
                "model_path": model_path,
                "metadata_path": metadata_path,
                "performance": results
            }
            
        except Exception as e:
            logger.error(f"❌ Pipeline failed: {e}")
            raise
    
    def train_warm_start_model(self, model: lgb.Booster, X: np.ndarray, y: np.ndarray) -> Tuple[lgb.Booster, Dict]:
        """Train warm-start model"""
        logger.info("🚀 Training warm-start model...")
        
        # Split data (keeping time order)
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_names)
        test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
        
        # Train model
        params = self._get_lgb_params()
        model = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[test_data], 
            init_model=model,
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
        )
        
        # Final evaluation
        y_pred = model.predict(X_test)
        y_pred_binary = (y_pred > 0.5).astype(int)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred_binary),
            "auc": roc_auc_score(y_test, y_pred),
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "model_params": params
        }
        
        logger.info(f"📊 Final Model: Accuracy={results['accuracy']:.3f}, AUC={results['auc']:.3f}")
        return model, results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Production ML Pipeline')
    parser.add_argument('--days', type=int, default=180, help='Days of historical data')
    parser.add_argument('--symbols', type=str, default='all', help='Symbol set (all/top10)')
    parser.add_argument('--warm', action='store_true', help='Warm-start from existing model (additive learning)')
    parser.add_argument('--tag', type=str, default='production', help='Model version tag')
    parser.add_argument('--bar', type=str, default='5m', help='Bar interval (e.g., 5m,3m)')
    
    args = parser.parse_args()
    
    pipeline = ProductionMLPipeline()
    
    if args.symbols == 'top10':
        pipeline.symbols = pipeline.symbols[:10]  # Top 10 symbols only
    
    bar_map = {'5m':300,'3m':180,'1m':60}
    bar_sec = bar_map.get(args.bar, 300)
    
    # Run pipeline with warm-start if requested
    if args.warm:
        pipeline.run_warm_start_pipeline(days_back=args.days, tag=args.tag, bar_sec=bar_sec)
    else:
        pipeline.run_full_pipeline(days_back=args.days, tag=args.tag, bar_sec=bar_sec)
