"""Inventory / exposure checks.

This module summarizes current Kalshi positions for the tickers we are polling,
so we can enforce max exposure / max contracts before placing new trades.

Kalshi `get_positions()` returns (per market):
- position: signed integer contracts (positive = long, negative = short)
- market_exposure: exposure in cents (per Kalshi API)

We treat `abs(market_exposure)` as the risk/exposure measure (in cents) and
convert to USD.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from tradebot.kalshi.client import KalshiClient


@dataclass(frozen=True)
class TickerInventory:
    ticker: str
    position: int
    abs_contracts: int
    exposure_usd: float


@dataclass(frozen=True)
class InventorySummary:
    tickers: list[str]
    per_ticker: dict[str, TickerInventory]

    total_abs_contracts: int
    total_exposure_usd: float

    max_abs_contracts: int
    max_exposure_usd: float


def _to_int(v: Any, *, default: int = 0) -> int:
    try:
        return int(v)
    except Exception:
        return int(default)


def _to_float(v: Any, *, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def summarize_positions_for_tickers(*, positions_payload: dict[str, Any], tickers: list[str]) -> InventorySummary:
    """Pure helper: summarize a `get_positions()` payload for a set of tickers."""

    tickers_norm = [str(t).strip() for t in tickers if str(t).strip()]
    ticker_set = set(tickers_norm)

    per: dict[str, TickerInventory] = {}

    market_positions = list(positions_payload.get("market_positions") or [])
    for mp in market_positions:
        if not isinstance(mp, dict):
            continue
        ticker = str(mp.get("ticker") or "").strip()
        if not ticker or ticker not in ticker_set:
            continue

        pos = _to_int(mp.get("position"), default=0)
        exposure_cents = _to_float(mp.get("market_exposure"), default=0.0)
        exposure_usd = abs(exposure_cents) / 100.0

        per[ticker] = TickerInventory(
            ticker=ticker,
            position=int(pos),
            abs_contracts=int(abs(pos)),
            exposure_usd=float(exposure_usd),
        )

    # Ensure all tickers are present (default 0)
    for t in tickers_norm:
        if t not in per:
            per[t] = TickerInventory(ticker=t, position=0, abs_contracts=0, exposure_usd=0.0)

    total_abs_contracts = int(sum(v.abs_contracts for v in per.values()))
    total_exposure_usd = float(sum(v.exposure_usd for v in per.values()))

    max_abs_contracts = int(max((v.abs_contracts for v in per.values()), default=0))
    max_exposure_usd = float(max((v.exposure_usd for v in per.values()), default=0.0))

    return InventorySummary(
        tickers=tickers_norm,
        per_ticker=per,
        total_abs_contracts=total_abs_contracts,
        total_exposure_usd=total_exposure_usd,
        max_abs_contracts=max_abs_contracts,
        max_exposure_usd=max_exposure_usd,
    )


async def fetch_inventory_summary(*, client: KalshiClient, tickers: list[str]) -> InventorySummary:
    """Fetch current positions from Kalshi and summarize for the given tickers."""

    # Pull only non-zero positions if possible.
    payload = await client.get_positions(limit=1000, count_filter="position")
    return summarize_positions_for_tickers(positions_payload=payload, tickers=tickers)
