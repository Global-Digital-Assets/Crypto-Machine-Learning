#!/usr/bin/env bash
set -e
BUCKET=s3://gda-ml-backup
STAMP=$(date +%F)
aws s3 sync /root/ml-engine/models        $BUCKET/models/$STAMP/   --exclude "latest_*" --storage-class STANDARD_IA
aws s3 sync /root/ml-engine/signals       $BUCKET/signals/$STAMP/  --storage-class STANDARD_IA
aws s3 cp  /root/analytics-tool-v2/market_data.db $BUCKET/db/$STAMP/market_data.db
logger -t ml_backup "backup completed to $BUCKET/$STAMP"
