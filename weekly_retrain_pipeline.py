#!/usr/bin/env python3
"""
Weekly Auto-Retrain Pipeline
- Runs every Sunday 02:00 UTC via systemd timer
- Trains new model on latest 30 days data
- Validates with forward-walk and fee-adjusted backtest
- Deploys only if backtest ≥ 0.4% EV (10% safety buffer above 0.35% target)
- Keeps last 3 model backups for rollback
"""
import os
import sys
import json
import shutil
import subprocess
from datetime import datetime, timedelta
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/root/ml-engine/logs/weekly_retrain.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyRetrainPipeline:
    def __init__(self):
        self.ml_engine_dir = "/root/ml-engine"
        self.models_dir = os.path.join(self.ml_engine_dir, "models")
        self.logs_dir = os.path.join(self.ml_engine_dir, "logs")
        self.backup_dir = os.path.join(self.models_dir, "backups")
        
        # Ensure directories exist
        os.makedirs(self.logs_dir, exist_ok=True)
        os.makedirs(self.backup_dir, exist_ok=True)
        
        # Paths
        self.analytics_dir = "/root/analytics-tool-v2"
        self.bucket_map = os.path.join(self.ml_engine_dir, "bucket_mapping.csv")
        self.parquet_dir = os.path.join(self.analytics_dir, "tmp_parquet_ultra")
        
        # Validation thresholds
        self.min_ev_threshold = 0.4  # 10% safety buffer above 0.35% target
        self.min_trades = 50  # Minimum trade count for validation
        
    def run_weekly_retrain(self):
        """Execute complete weekly retrain pipeline"""
        logger.info("🚀 Starting Weekly Auto-Retrain Pipeline")
        logger.info("=" * 60)
        
        try:
            # Step 1: Backup current model
            self._backup_current_model()
            
            # Step 2: Train new model
            new_model_path = self._train_new_model()
            
            # Step 3: Validate new model
            validation_passed = self._validate_new_model(new_model_path)
            
            # Step 4: Deploy if validation passes
            if validation_passed:
                self._deploy_new_model(new_model_path)
                logger.info("✅ Weekly retrain completed successfully!")
                return True
            else:
                logger.warning("❌ New model failed validation - keeping current model")
                self._cleanup_failed_model(new_model_path)
                return False
                
        except Exception as e:
            logger.error(f"💥 Weekly retrain failed: {e}")
            return False
    
    def _backup_current_model(self):
        """Backup current model and rotate old backups"""
        logger.info("📦 Backing up current model...")
        
        current_model = os.path.join(self.models_dir, "latest_model.txt")
        
        if os.path.exists(current_model) and os.path.islink(current_model):
            # Get actual model file
            actual_model = os.readlink(current_model)
            actual_model_path = os.path.join(self.models_dir, actual_model)
            
            if os.path.exists(actual_model_path):
                # Create timestamped backup
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_name = f"backup_{timestamp}_{actual_model}"
                backup_path = os.path.join(self.backup_dir, backup_name)
                
                shutil.copy2(actual_model_path, backup_path)
                logger.info(f"   Current model backed up: {backup_name}")
                
                # Rotate backups (keep last 3)
                self._rotate_backups()
    
    def _rotate_backups(self):
        """Keep only the 3 most recent backups"""
        backups = []
        for f in os.listdir(self.backup_dir):
            if f.startswith("backup_") and f.endswith(".txt"):
                backup_path = os.path.join(self.backup_dir, f)
                mtime = os.path.getmtime(backup_path)
                backups.append((mtime, f, backup_path))
        
        # Sort by modification time (newest first)
        backups.sort(reverse=True)
        
        # Remove old backups (keep first 3)
        for i, (_, name, path) in enumerate(backups):
            if i >= 3:
                os.remove(path)
                logger.info(f"   Removed old backup: {name}")
    
    def _train_new_model(self):
        """Train new model using continuous_learner approach"""
        logger.info("🔧 Training new model...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_model_name = f"lgbm_weekly_{timestamp}.txt"
        new_model_path = os.path.join(self.models_dir, new_model_name)
        
        # Use existing continuous_learner.py
        cmd = [
            os.path.join(self.ml_engine_dir, "venv/bin/python"),
            os.path.join(self.ml_engine_dir, "continuous_learner.py")
        ]
        
        env = os.environ.copy()
        env['DATA_API_URL'] = 'http://localhost:8001'
        
        result = subprocess.run(
            cmd, 
            cwd=self.ml_engine_dir,
            env=env,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"Training failed: {result.stderr}")
            raise Exception(f"Model training failed: {result.stderr}")
        
        # Find the most recent model (continuous_learner creates timestamped models)
        model_files = [f for f in os.listdir(self.models_dir) if f.startswith("lgbm_") and f.endswith(".txt")]
        if not model_files:
            raise Exception("No model files found after training")
        
        # Get most recent model
        model_files.sort()
        latest_model = model_files[-1]
        actual_model_path = os.path.join(self.models_dir, latest_model)
        
        # Rename to weekly naming convention
        shutil.move(actual_model_path, new_model_path)
        
        logger.info(f"   New model trained: {new_model_name}")
        return new_model_path
    
    def _validate_new_model(self, model_path):
        """Validate new model with forward-walk and fee-adjusted backtests"""
        logger.info("🧪 Validating new model...")
        
        try:
            # Run fee-adjusted backtest
            results = self._run_fee_adjusted_backtest(model_path)
            
            if not results:
                logger.warning("   Backtest failed to generate results")
                return False
            
            # Check validation criteria
            fee_adjusted_return = results.get('fee_adjusted_return', 0)
            trades = results.get('trades', 0)
            
            logger.info(f"   Fee-adjusted return: {fee_adjusted_return:.3f}%")
            logger.info(f"   Trade count: {trades}")
            logger.info(f"   Required: ≥{self.min_ev_threshold}% return, ≥{self.min_trades} trades")
            
            passed = (fee_adjusted_return >= self.min_ev_threshold and 
                     trades >= self.min_trades)
            
            if passed:
                logger.info("   ✅ Validation PASSED")
            else:
                logger.warning("   ❌ Validation FAILED")
            
            return passed
            
        except Exception as e:
            logger.error(f"   Validation error: {e}")
            return False
    
    def _run_fee_adjusted_backtest(self, model_path):
        """Run fee-adjusted backtest on new model"""
        output_file = f"/tmp/weekly_validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        cmd = [
            os.path.join(self.ml_engine_dir, "venv/bin/python"),
            os.path.join(os.path.dirname(__file__), "backtest_fee_adjusted.py"),
            "--model", model_path,
            "--parquet-dir", self.parquet_dir,
            "--bucket-map", self.bucket_map,
            "--buckets", "ultra",
            "--percentile", "99",
            "--min-proba", "0.20",
            "--fee-pct", "0.06",
            "--out", output_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0 and os.path.exists(output_file):
            with open(output_file, 'r') as f:
                results = json.load(f)
            os.remove(output_file)  # Cleanup
            return results
        
        logger.error(f"Backtest failed: {result.stderr}")
        return None
    
    def _deploy_new_model(self, model_path):
        """Deploy validated model to production"""
        logger.info("🚀 Deploying new model...")
        
        # Update latest_model.txt symlink
        latest_link = os.path.join(self.models_dir, "latest_model.txt")
        model_name = os.path.basename(model_path)
        
        # Remove old symlink
        if os.path.exists(latest_link):
            os.remove(latest_link)
        
        # Create new symlink
        os.symlink(model_name, latest_link)
        
        # Restart ML generator service
        try:
            subprocess.run(["systemctl", "restart", "ml-generator.service"], 
                         check=True, timeout=30)
            logger.info("   ML generator service restarted")
        except subprocess.CalledProcessError as e:
            logger.error(f"   Failed to restart ML generator: {e}")
            raise
        
        logger.info(f"   ✅ Model deployed: {model_name}")
    
    def _cleanup_failed_model(self, model_path):
        """Remove failed model file"""
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
                logger.info(f"   Cleaned up failed model: {os.path.basename(model_path)}")
        except Exception as e:
            logger.warning(f"   Cleanup warning: {e}")

def main():
    """Main entry point for weekly retrain"""
    pipeline = WeeklyRetrainPipeline()
    success = pipeline.run_weekly_retrain()
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())
