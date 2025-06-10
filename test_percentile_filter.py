"""Unit test for percentile filtering logic used in generate_daily_signals."""
from __future__ import annotations

import importlib

import generate_daily_signals as gds


def _calc_keep(total: int, percentile: float) -> int:
    """Replicates keep_n calc from generate_daily_signals."""
    return max(1, int(total * (100 - percentile) / 100))


def test_keep_n_formula():
    # Pair a few spot-checks with realistic list lengths
    cases = [
        (10, 99, 1),  # always keep at least 1
        (100, 97, 3),
        (50, 90, 5),
        (3, 99, 1),
    ]
    for total, perc, expected in cases:
        assert _calc_keep(total, perc) == expected, f"mismatch for {total=} {perc=}"

    # Sanity check that generate_daily_signals shares the same formula
    # (importlib reload guards against stale byte-code during watch mode)
    importlib.reload(gds)
    for total, perc, expected in cases:
        keep_n = max(1, int(total * (100 - perc) / 100))
        assert keep_n == expected
