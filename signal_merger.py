#!/usr/bin/env python3
import json
from pathlib import Path
from datetime import datetime, timezone

SIG_DIR = Path('/root/crypto-models/signals')
OUT_DIR = Path('/root/ml-engine/signals')
OUT_DIR.mkdir(parents=True, exist_ok=True)

probs: dict[str, float] = {}
signals: list[dict] = []

for path in list(SIG_DIR.glob('*_short.json')) + list(SIG_DIR.glob('*_long.json')):
    try:
        data = json.loads(path.read_text())
        token = path.stem.rsplit('_', 1)[0]  # e.g. JUPUSDT_short -> JUPUSDT
        if token in probs:  # skip duplicate (prefer first)
            continue
        proba = data.get('probability')
        if proba is None:
            continue
        probs[token] = proba
        # keep only actionable signals (prob≥0.60)
        if proba >= 0.60 and data.get('signal') in {'buy', 'sell'}:
            signals.append(data)
    except Exception as exc:
        print('skip', path, exc)

# Write merged outputs
(OUT_DIR / 'latest_probs.json').write_text(json.dumps(probs))
(OUT_DIR / 'latest_signals.json').write_text(json.dumps(signals))

# Also update analyzer_debug.json for Analytics API compatibility
analysis = {
    'timestamp': datetime.now(timezone.utc).isoformat(),
    'metadata': {
        'tokens_analyzed': len(probs),
        'version': 'signal_merger_v1'
    },
    'probabilities': probs,
    'opportunities': signals,
}
ANALYSIS_PATH = Path('/root/analytics-tool-v2/analyzer_debug.json')
ANALYSIS_PATH.write_text(json.dumps(analysis) + '\n')
print(f'Merged {len(probs)} probabilities; {len(signals)} actionable signals')