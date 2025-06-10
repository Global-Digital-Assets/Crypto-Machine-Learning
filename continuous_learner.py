#!/usr/bin/env python3
"""Nightly Continuous Learner

Retrains (or fine-tunes) the LightGBM model on the latest data every night.
Saves a versioned model and logs high-level metrics into `model_performance`.

The script is designed to be idempotent and safe to run unattended via
systemd-timer.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path

import prediction_db
from production_ml_pipeline import ProductionMLPipeline
from s3_uploader import upload_model_to_s3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s – %(levelname)s – %(message)s",
)
logger = logging.getLogger("continuous_learner")


MODELS_DIR = Path("models")
DB_PATH = "market_data.db"


def _train_and_log():
    pipe = ProductionMLPipeline(db_path=DB_PATH, models_dir=str(MODELS_DIR))

    # Use 180-day window for training; adjust as needed.
    X, y, metadata = pipe.load_and_prepare_data(days_back=180)
    cv_results = pipe.walk_forward_validation(X, y, n_splits=5)
    model, results = pipe.train_production_model(X, y, early_stopping_rounds=50)

    # Save with nightly tag
    tag = "nightly"
    pipe.save_versioned_model(model, metadata, results, cv_results, tag=tag)

    # S3 backup if credentials & bucket configured
    s3_bucket = os.getenv("ML_MODEL_S3_BUCKET")
    if s3_bucket:
        model_file_path = MODELS_DIR / f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        if model_file_path.exists():
            upload_model_to_s3(str(model_file_path), s3_bucket, model_file_path.name)

    model_version = f"{tag}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Log basic metrics into SQLite for dashboarding
    perf_row = {
        "model_version": model_version,
        "timestamp": datetime.utcnow().timestamp(),
        "window": "full",
        "auc": results.get("auc"),
        "precision": results.get("accuracy"),
        "recall": None,
        "sharpe": None,
        "sample_count": metadata.get("total_samples"),
    }
    prediction_db.log_model_performance(perf_row)

    logger.info("✅ Continuous learner completed – new model %s saved", model_version)


def main():
    try:
        _train_and_log()
    except Exception as exc:
        logger.exception("Continuous learner failed: %s", exc)
        raise


if __name__ == "__main__":
    main()
