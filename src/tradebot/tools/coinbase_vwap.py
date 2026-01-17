from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque


@dataclass(frozen=True)
class _Trade:
    ts_ms: int
    price: float
    size: float


class RollingVWAP:
    """Rolling VWAP calculator for Coinbase-style trades.

    Stores recent trades and computes VWAP over a rolling time window.
    """

    def __init__(self) -> None:
        self._trades: Deque[_Trade] = deque()
        self._latest_ts_ms: int | None = None

    def add_trade(self, ts_ms: int, price: float, size: float) -> None:
        """Add a trade to the window.

        Trades with non-positive size are ignored.
        """

        try:
            size_f = float(size)
        except Exception:
            return

        if size_f <= 0:
            return

        try:
            ts_i = int(ts_ms)
            self._trades.append(_Trade(ts_ms=ts_i, price=float(price), size=size_f))
        except Exception:
            return

        if self._latest_ts_ms is None or ts_i > self._latest_ts_ms:
            self._latest_ts_ms = ts_i

    def count(self) -> int:
        return len(self._trades)

    def latest_ts_ms(self) -> int | None:
        """Return the newest trade timestamp currently in the buffer."""

        return self._latest_ts_ms

    def vwap(self, now_ms: int, window_seconds: int = 60) -> float | None:
        """Compute VWAP for trades in the last `window_seconds`.

        Drops trades older than (now_ms - window_seconds*1000).
        Returns None if there are no trades in-window or total size is 0.
        """

        cutoff_ms = int(now_ms) - int(window_seconds) * 1000

        # Prune old trades. Maintain _latest_ts_ms even if ingestion was out-of-order.
        while self._trades and int(self._trades[0].ts_ms) < cutoff_ms:
            removed = self._trades.popleft()
            if self._latest_ts_ms is not None and int(removed.ts_ms) == int(self._latest_ts_ms):
                # The newest trade was pruned (possible with out-of-order inserts).
                if not self._trades:
                    self._latest_ts_ms = None
                else:
                    self._latest_ts_ms = max(int(t.ts_ms) for t in self._trades)

        if not self._trades:
            self._latest_ts_ms = None

        num = 0.0
        denom = 0.0
        for t in self._trades:
            try:
                num += float(t.price) * float(t.size)
                denom += float(t.size)
            except Exception:
                continue

        if denom <= 0:
            return None

        return float(num) / float(denom)
