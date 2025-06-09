#!/usr/bin/env python3
"""Nightly Continuous Learner

Retrains (or fine-tunes) the LightGBM model on the latest data every night.
Saves a versioned model and logs high-level metrics into `model_performance`.

The script is designed to be idempotent and safe to run unattended via
systemd-timer.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import prediction_db
from production_ml_pipeline import ProductionMLPipeline

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
    model, results = pipe.train_production_model(X, y)

    # Save with nightly tag
    tag = "nightly"
    pipe.save_versioned_model(model, metadata, results, cv_results, tag=tag)

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
