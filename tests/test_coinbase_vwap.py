from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from tradebot.tools.coinbase_vwap import RollingVWAP


def test_rolling_vwap_60s_excludes_old_trade() -> None:
    now_ms = 1_700_000_000_000

    v = RollingVWAP()
    v.add_trade(now_ms - 70_000, price=100.0, size=1.0)  # excluded
    v.add_trade(now_ms - 50_000, price=110.0, size=2.0)
    v.add_trade(now_ms - 10_000, price=90.0, size=1.0)

    got = v.vwap(now_ms, window_seconds=60)
    assert got is not None

    # Expected: (110*2 + 90*1) / (2+1) = 310/3
    assert abs(float(got) - (310.0 / 3.0)) < 1e-12

def test_latest_ts_ms_tracks_newest_trade_in_buffer() -> None:
    vwap = RollingVWAP()
    vwap.add_trade(1000, 100.0, 1.0)
    vwap.add_trade(2000, 101.0, 1.0)

    # Out-of-order insert should not break latest tracking.
    vwap.add_trade(1500, 99.0, 1.0)

    # Calling vwap() prunes but does not remove these.
    _ = vwap.vwap(3000, window_seconds=60)
    assert vwap.latest_ts_ms() == 2000

    # Even if we ingest nothing later, "latest" should reflect buffer state.
    _ = vwap.vwap(3500, window_seconds=60)
    assert vwap.latest_ts_ms() == 2000


def test_latest_ts_ms_resets_when_pruned_empty() -> None:
    vwap = RollingVWAP()
    vwap.add_trade(1000, 100.0, 1.0)
    assert vwap.latest_ts_ms() == 1000

    # Prune everything by moving now far ahead with a short window.
    got = vwap.vwap(10_000, window_seconds=1)
    assert got is None
    assert vwap.latest_ts_ms() is None
