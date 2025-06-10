"""Simple S3 uploader helper used by continuous_learner.py.

Relies on standard AWS environment variables:
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_DEFAULT_REGION

Assumes the target bucket already exists and caller has put-object permission.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Final

import boto3
from botocore.exceptions import BotoCoreError, ClientError

logger: Final = logging.getLogger("s3_uploader")


def upload_model_to_s3(local_path: str, bucket: str, key: str) -> None:  # noqa: D401
    """Upload *local_path* → s3://bucket/key.

    Any error is logged but not raised so training jobs never fail due to
    transient S3 issues. Caller decides whether to retry or ignore.
    """
    client = boto3.client("s3")
    file_path = Path(local_path)
    if not file_path.exists():
        logger.error("S3 upload aborted – file not found: %s", local_path)
        return

    try:
        client.upload_file(str(file_path), bucket, key)
        logger.info("📤 Uploaded %s → s3://%s/%s", file_path.name, bucket, key)
    except (BotoCoreError, ClientError) as exc:
        logger.exception("S3 upload failed: %s", exc)
