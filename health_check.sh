#!/usr/bin/env bash
URL="http://localhost:8080/api/signals"
if ! curl -fs --max-time 6 "$URL" >/dev/null; then
  logger -t ml_health "signals API down – restarting ml-generator"
  systemctl restart ml-generator.service
fi
