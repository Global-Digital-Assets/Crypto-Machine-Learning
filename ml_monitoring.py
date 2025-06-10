#!/usr/bin/env python3
"""
 ML MONITORING & AUTO-RETRAIN
Drift detection and automated model retraining system
"""

import json
import logging
import sqlite3
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import polars as pl
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import lightgbm as lgb
from prometheus_client import Gauge, start_http_server


# ---- Prometheus metrics ----
start_http_server(9101)
ML_ACCURACY = Gauge('ml_accuracy', 'Current model accuracy')
ML_AUC = Gauge('ml_auc', 'Current model AUC')
ML_DRIFT = Gauge('ml_drift', 'AUC drift vs baseline')
logger = logging.getLogger(__name__)

class MLMonitoringSystem:
    """Monitor model performance and trigger retraining"""
    
    def __init__(self, models_dir: str = "models", db_path: str = "market_data.db"):
        self.models_dir = Path(models_dir)
        self.db_path = db_path
        self.monitoring_db = "ml_monitoring.db"
        
        # Performance thresholds
        self.accuracy_threshold = 0.85  # Retrain if below 85%
        self.auc_threshold = 0.75      # Retrain if below 0.75
        self.drift_threshold = 0.05    # 5% performance drop triggers retrain
        
        self._init_monitoring_db()
    
    def _init_monitoring_db(self):
        """Initialize monitoring database"""
        conn = sqlite3.connect(self.monitoring_db, timeout=60, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_log (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                model_version TEXT,
                test_accuracy REAL,
                test_auc REAL,
                samples_evaluated INTEGER,
                feature_drift_score REAL,
                retrain_triggered BOOLEAN,
                notes TEXT
            )
        """)
        
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_log (
                id INTEGER PRIMARY KEY,
                timestamp DATETIME,
                symbol TEXT,
                prediction_proba REAL,
                actual_outcome INTEGER,
                model_version TEXT
            )
        """)
        
        conn.commit()
        conn.close()
        
        logger.info("✅ Monitoring database initialized")
    
    def evaluate_current_model(self, days_back: int = 7) -> Dict:
        """Evaluate current model on recent data"""
        try:
            # Load current model metadata
            metadata_path = self.models_dir / "latest_metadata.json"
            if not metadata_path.exists():
                logger.error("❌ No current model found")
                return {}
            
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            # Load recent data for evaluation
            conn = sqlite3.connect(self.db_path, timeout=60, check_same_thread=False)
            cutoff_timestamp = int((datetime.now() - timedelta(days=days_back)).timestamp())
            
            symbols = metadata['symbols']
            query = """
            SELECT symbol, timestamp, open, high, low, close, volume
            FROM candles 
            WHERE symbol IN ({}) AND timestamp >= {}
            ORDER BY symbol, timestamp
            """.format(','.join([f"'{s}'" for s in symbols]), cutoff_timestamp)
            
            df = pl.read_database(query, conn)
            conn.close()
            
            if len(df) == 0:
                logger.warning("⚠️ No recent data available for evaluation")
                return {}
            
            # Create features and labels
            X_eval, y_eval = self._prepare_evaluation_data(df, metadata['feature_names'])
            
            if len(X_eval) == 0:
                logger.warning("⚠️ No evaluation samples available")
                return {}
            
            # Load model and predict
            model_path = self.models_dir / "latest_model.txt"
            model = lgb.Booster(model_file=str(model_path))
            
            y_pred_proba = model.predict(X_eval)
            y_pred_binary = (y_pred_proba > 0.5).astype(int)
            
            # Calculate metrics
            current_accuracy = accuracy_score(y_eval, y_pred_binary)
            current_auc = roc_auc_score(y_eval, y_pred_proba)
            
            # Extract CV baseline from metadata for proper drift detection
            cv_baseline_auc = metadata.get('cross_validation', {}).get('mean_auc', 0.785)
            
            evaluation_results = {
                "timestamp": datetime.now().isoformat(),
                "model_version": metadata['model_info']['timestamp'],
                "current_performance": {
                    "accuracy": current_accuracy,
                    "auc": current_auc,
                    "samples": len(X_eval)
                },
                "training_performance": {
                    "accuracy": metadata['performance']['accuracy'],
                    "auc": metadata['performance']['auc']
                },
                "drift": {
                    "accuracy_drift": abs(current_accuracy - 0.90),  # Conservative baseline
                    "auc_drift": abs(current_auc - cv_baseline_auc),
                    "evaluation_period_days": days_back
                },
                "baseline_auc": cv_baseline_auc,  # Use CV mean as realistic baseline
                "alerts": self._generate_alerts(current_accuracy, current_auc, abs(current_accuracy - 0.90), abs(current_auc - cv_baseline_auc))
            }
            
            # Log to monitoring database
            self._log_performance(evaluation_results)
            
            logger.info(f"📊 Model Evaluation: Acc={current_accuracy:.3f} (drift: {abs(current_accuracy - 0.90):+.3f}), AUC={current_auc:.3f} (drift: {abs(current_auc - cv_baseline_auc):+.3f})")
            
            
            # Update Prometheus metrics
            ML_ACCURACY.set(current_accuracy)
            ML_AUC.set(current_auc)
            ML_DRIFT.set(abs(current_auc - cv_baseline_auc))

            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Model evaluation failed: {e}")
            return {}
    
    def check_retrain_conditions(self, evaluation_results: Dict) -> Dict:
        """Check if model should be retrained with institutional-grade logic"""
        if not evaluation_results:
            return {"should_retrain": False, "reason": "No evaluation data"}
        
        current_perf = evaluation_results['current_performance']
        drift = evaluation_results['drift']
        
        # 🎯 INSTITUTIONAL DRIFT DETECTION - Fixed per user specifications
        # Load CV baseline from metadata (not final model AUC)
        try:
            metadata_path = self.models_dir / "latest_metadata.json"
            with open(metadata_path, 'r') as f:
                meta = json.load(f)
            BASELINE_AUC = meta["cross_validation"]["mean_auc"]  # Correct baseline
        except:
            BASELINE_AUC = 0.785  # Fallback
        
        # Configurable threshold - 5pp for crypto volatility
        DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.05"))  # 5 percentage points
        
        # Calculate actual degradation
        auc_degradation = BASELINE_AUC - current_perf['auc']
        
        # Check absolute performance thresholds (emergency)
        critical_accuracy_drop = current_perf['accuracy'] < 0.85  # Below 85%
        critical_auc_drop = current_perf['auc'] < 0.75  # Below 75%
        
        # Check institutional drift threshold
        significant_drift = auc_degradation > DRIFT_THRESHOLD
        
        # Determine retrain type and urgency
        retrain_decision = {
            "should_retrain": False,
            "retrain_type": "none",
            "urgency": "normal",
            "reason": "",
            "metrics": {
                "current_auc": current_perf['auc'],
                "baseline_auc": BASELINE_AUC,
                "auc_degradation": auc_degradation,
                "drift_threshold": DRIFT_THRESHOLD
            }
        }
        
        if critical_accuracy_drop or critical_auc_drop:
            retrain_decision.update({
                "should_retrain": True,
                "retrain_type": "emergency",
                "urgency": "critical",
                "reason": f"Critical performance drop - Acc: {current_perf['accuracy']:.3f}, AUC: {current_perf['auc']:.3f}"
            })
        elif significant_drift:
            retrain_decision.update({
                "should_retrain": True,
                "retrain_type": "drift_hotfix",
                "urgency": "high",
                "reason": f"Drift detected - AUC degraded by {auc_degradation:.3f} (>{DRIFT_THRESHOLD:.3f})"
            })
        
        if retrain_decision["should_retrain"]:
            logger.warning(f"🚨 RETRAIN TRIGGERED [{retrain_decision['retrain_type'].upper()}]: {retrain_decision['reason']}")
        else:
            logger.info(f"✅ Model stable - AUC: {current_perf['auc']:.3f} vs baseline: {BASELINE_AUC:.3f} (degradation: {auc_degradation:.3f}, threshold: {DRIFT_THRESHOLD:.3f})")
        
        return retrain_decision
    
    def auto_retrain_pipeline(self) -> bool:
        """Automatically retrain model if conditions are met"""
        try:
            logger.info("🔍 Running automated model monitoring...")
            
            # Evaluate current model
            evaluation_results = self.evaluate_current_model()
            
            if not evaluation_results:
                logger.error("❌ Could not evaluate current model")
                return False
            
            # Check if retrain is needed
            retrain_decision = self.check_retrain_conditions(evaluation_results)
            
            if not retrain_decision["should_retrain"]:
                logger.info("✅ Model performance acceptable, no retrain needed")
                return True
            
            # Trigger retrain based on type
            logger.info(f"🔄 Starting automated retrain: {retrain_decision['retrain_type']}")
            
            # Use enhanced retrain manager
            from enhanced_retrain import EnhancedRetrainManager
            
            retrain_manager = EnhancedRetrainManager()
            
            if retrain_decision['retrain_type'] == 'emergency':
                # Emergency: full retrain on 90 days 
                success = retrain_manager.full_retrain(days=90, tag="emergency")
            elif retrain_decision['retrain_type'] == 'drift_hotfix':
                # Drift: warm-start retrain on 180 days (preserves knowledge)
                success = retrain_manager.drift_hotfix_retrain(days=180)
            else:
                # Default full retrain
                success = retrain_manager.full_retrain(days=180, tag="auto")
            
            if success:
                logger.info(f"✅ Automated {retrain_decision['retrain_type']} retrain completed successfully")
                return True
            else:
                logger.error(f"❌ Automated {retrain_decision['retrain_type']} retrain failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Auto-retrain pipeline failed: {e}")
            return False
    
    def get_performance_history(self, days: int = 30) -> List[Dict]:
        """Get model performance history"""
        conn = sqlite3.connect(self.monitoring_db, timeout=60, check_same_thread=False)
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        
        cursor.execute("""
            SELECT timestamp, model_version, test_accuracy, test_auc, 
                   samples_evaluated, retrain_triggered, notes
            FROM performance_log 
            WHERE timestamp >= ?
            ORDER BY timestamp DESC
        """, (cutoff,))
        
        history = []
        for row in cursor.fetchall():
            history.append({
                "timestamp": row[0],
                "model_version": row[1],
                "accuracy": row[2],
                "auc": row[3],
                "samples": row[4],
                "retrain_triggered": bool(row[5]),
                "notes": row[6]
            })
        
        conn.close()
        return history
    
    def _prepare_evaluation_data(self, df: pl.DataFrame, feature_names: List[str]):
        """Prepare evaluation dataset"""
        # Create 5-minute bars
        df_5m = (
            df
            .with_columns([(pl.col("timestamp") // 300 * 300).alias("ts5m")])
            .group_by(["symbol", "ts5m"])
            .agg([
                pl.col("open").first(),
                pl.col("high").max(), 
                pl.col("low").min(),
                pl.col("close").last(),
                pl.col("volume").sum()
            ])
            .sort(["symbol", "ts5m"])
        )
        
        # Engineer features (same as training)
        df_features = (
            df_5m
            .sort(["symbol", "ts5m"])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(1).over("symbol") - 1).alias("ret_5m"),
            ])
            .with_columns([
                (pl.col("close") / pl.col("close").shift(12).over("symbol") - 1).alias("ret_1h"), 
                (pl.col("close") / pl.col("close").shift(48).over("symbol") - 1).alias("ret_4h"),
                (pl.col("close") / pl.col("close").shift(288).over("symbol") - 1).alias("ret_24h"),
                pl.col("volume").rolling_mean(12).over("symbol").alias("vol_ma_1h"),
                (pl.col("volume") / pl.col("volume").rolling_mean(12).over("symbol")).alias("vol_spike"),
                pl.col("ret_5m").rolling_std(12).over("symbol").alias("volatility_1h"),
                pl.col("ret_5m").rolling_std(48).over("symbol").alias("volatility_4h"),
                ((pl.col("high") + pl.col("low")) / 2 / pl.col("close") - 1).alias("hl_spread"),
                (pl.col("close") / pl.col("open") - 1).alias("bar_return"),
                pl.col("close").rolling_mean(14).over("symbol").alias("sma_14"),
                (pl.col("close") / pl.col("close").rolling_mean(14).over("symbol") - 1).alias("sma_distance"),
                pl.when(pl.col("ret_5m") > 0).then(pl.col("ret_5m")).otherwise(0).rolling_mean(14).over("symbol").alias("avg_gains"),
                pl.when(pl.col("ret_5m") < 0).then(-pl.col("ret_5m")).otherwise(0).rolling_mean(14).over("symbol").alias("avg_losses")
            ])
            .with_columns([
                (100 - 100 / (1 + pl.col("avg_gains") / (pl.col("avg_losses") + 1e-10))).alias("rsi14"),
                (pl.col("ret_1h").shift(-12).over("symbol") > 0.005).alias("target")
            ])
            .drop(["avg_gains", "avg_losses", "vol_ma_1h", "sma_14"])
        )
        
        # Get clean data
        clean_df = df_features.drop_nulls()
        
        if len(clean_df) == 0:
            return np.array([]), np.array([])
        
        X = clean_df.select(feature_names).to_numpy()
        y = clean_df.select("target").to_numpy().flatten()
        
        return X, y
    
    def _generate_alerts(self, accuracy: float, auc: float, acc_drift: float, auc_drift: float) -> List[str]:
        """Generate performance alerts"""
        alerts = []
        
        if accuracy < self.accuracy_threshold:
            alerts.append(f"LOW_ACCURACY: {accuracy:.3f} < {self.accuracy_threshold}")
        
        if auc < self.auc_threshold:
            alerts.append(f"LOW_AUC: {auc:.3f} < {self.auc_threshold}")
        
        if acc_drift > self.drift_threshold:
            alerts.append(f"ACCURACY_DRIFT: {acc_drift:.3f} > {self.drift_threshold}")
        
        if auc_drift > self.drift_threshold:
            alerts.append(f"AUC_DRIFT: {auc_drift:.3f} > {self.drift_threshold}")
        
        return alerts
    
    def _log_performance(self, evaluation_results: Dict):
        """Log performance to monitoring database"""
        conn = sqlite3.connect(self.monitoring_db, timeout=60, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_log 
            (timestamp, model_version, test_accuracy, test_auc, samples_evaluated, 
             feature_drift_score, retrain_triggered, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            evaluation_results['timestamp'],
            evaluation_results['model_version'],
            evaluation_results['current_performance']['accuracy'],
            evaluation_results['current_performance']['auc'],
            evaluation_results['current_performance']['samples'],
            max(evaluation_results['drift']['accuracy_drift'], evaluation_results['drift']['auc_drift']),
            False,
            json.dumps(evaluation_results['alerts'])
        ))
        
        conn.commit()
        conn.close()
    
    def _log_retrain_event(self, evaluation_results: Dict, retrain_results: Dict):
        """Log retrain event"""
        conn = sqlite3.connect(self.monitoring_db, timeout=60, check_same_thread=False)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO performance_log 
            (timestamp, model_version, test_accuracy, test_auc, samples_evaluated, 
             feature_drift_score, retrain_triggered, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            retrain_results['performance']['model_params']['random_state'],  # Simple version tracking
            retrain_results['performance']['accuracy'],
            retrain_results['performance']['auc'],
            retrain_results['performance']['test_samples'],
            0.0,  # Reset after retrain
            True,
            "Automated retrain completed"
        ))
        
        conn.commit()
        conn.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ML Monitoring System')
    parser.add_argument('--action', choices=['evaluate', 'monitor', 'history'], 
                       default='monitor', help='Action to perform')
    parser.add_argument('--days', type=int, default=7, help='Days to look back')
    
    args = parser.parse_args()
    
    monitor = MLMonitoringSystem()
    
    if args.action == 'evaluate':
        results = monitor.evaluate_current_model(days_back=args.days)
        print(json.dumps(results, indent=2))
    
    elif args.action == 'monitor':
        success = monitor.auto_retrain_pipeline()
        print(f"Monitoring completed: {'SUCCESS' if success else 'FAILED'}")
    
    elif args.action == 'history':
        history = monitor.get_performance_history(days=args.days)
        print(json.dumps(history, indent=2))
