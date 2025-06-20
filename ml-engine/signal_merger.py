#!/usr/bin/env python3
"""Signal Merger v2 – robust daily snapshot + fresh symlinks

Collects per-token *_long.json / *_short.json files created by crypto-models
and produces:
  • signals_YYYYMMDD.json         – actionable buy/sell only (prob≥0.65)
  • probs_YYYYMMDDTHHMMSSZ.json   – raw probabilities for all tokens
  • latest_signals.json / latest_probs.json symlinks updated every run
  • analyzer_debug.json for Analytics backward-compat.

Idempotent: safe to run concurrently (atomic write + rename).
"""
from __future__ import annotations
import json, os, tempfile
from pathlib import Path
from datetime import datetime, timezone

SIG_DIR = Path('/root/crypto-models/signals')
OUT_DIR = Path('/root/ml-engine/signals')
OUT_DIR.mkdir(parents=True, exist_ok=True)

now = datetime.now(timezone.utc)
today = now.strftime('%Y%m%d')
iso_ts = now.strftime('%Y%m%dT%H%M%SZ')

signals_file = OUT_DIR / f'signals_{today}.json'
probs_file   = OUT_DIR / f'probs_{iso_ts}.json'

probs: dict[str, float] = {}
signals: list[dict] = []

for path in sorted(SIG_DIR.glob('*_long.json')) + sorted(SIG_DIR.glob('*_short.json')):
    try:
        data = json.loads(path.read_text())
        token = path.stem  # e.g. BTCUSDT_long
        proba = data.get('probability')
        if proba is None:
            continue
        probs[token] = proba
        if proba >= 0.65 and data.get('signal') in {'buy', 'sell'}:
            signals.append(data)
    except Exception as exc:
        print('skip', path, exc)

# atomic writes
for target_path, payload in ((probs_file, probs), (signals_file, signals)):
    with tempfile.NamedTemporaryFile('w', delete=False, dir=str(OUT_DIR)) as tmp:
        json.dump(payload, tmp)
        tmp.flush(); os.fsync(tmp.fileno())
        tmp_name = tmp.name
    os.replace(tmp_name, target_path)

# refresh symlinks
for link_name, target in ((OUT_DIR / 'latest_probs.json', probs_file.name),
                          (OUT_DIR / 'latest_signals.json', signals_file.name)):
    try:
        if link_name.exists() or link_name.is_symlink():
            link_name.unlink()
        link_name.symlink_to(target)
        os.utime(link_name, None, follow_symlinks=False)
    except Exception as exc:
        print('symlink update failed', link_name, exc)

analysis = {
    'timestamp': now.isoformat(),
    'metadata': {'tokens_analyzed': len(probs), 'version': 'signal_merger_v2'},
    'probabilities': probs,
    'opportunities': signals,
}
Path('/root/analytics-tool-v2/analyzer_debug.json').write_text(json.dumps(analysis) + '\n')

print(f'Merged {len(probs)} probabilities; {len(signals)} actionable signals')
