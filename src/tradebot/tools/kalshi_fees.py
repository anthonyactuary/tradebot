"""Kalshi fee helpers.

Implements the fee formulas (USD) and rounding rules used by Kalshi.

Given (from user):
- General/taker:
    fees = round_up_to_cent(0.07 * C * P * (1-P))
- Maker:
    fees = round_up_to_cent(0.0175 * C * P * (1-P))

Where:
- P is the contract price in dollars, in [0, 1]
- C is the number of contracts
- round_up_to_cent rounds to the next cent (ceil).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from tradebot.tools.kalshi_market_poll import MarketSnapshot


def kalshi_expected_fee_usd(*, P: float, C: int, maker: bool) -> float:
    """Compute Kalshi fee in USD, rounded up to the next cent."""
    C = int(C)
    if C <= 0:
        raise ValueError("C must be >= 1")

    P = float(P)
    if not (0.0 <= P <= 1.0) or not math.isfinite(P):
        raise ValueError("P must be within [0, 1]")

    rate = 0.0175 if maker else 0.07
    fee = rate * float(C) * P * (1.0 - P)

    # Round up to the next cent.
    # Epsilon avoids floating-point artifacts where fee*100 is already integral.
    eps = 1e-12
    return math.ceil((fee - eps) * 100.0) / 100.0


def price_dollars_from_cents(v: int | None) -> float | None:
    if v is None:
        return None
    try:
        cents = int(v)
    except Exception:
        return None
    if cents < 0:
        return None
    return float(cents) / 100.0


def fees_from_snapshot(*, snap: "MarketSnapshot", contracts: int) -> dict[str, float | None]:
    """Compute maker/taker fees for buying YES/NO at current asks."""
    P_yes = price_dollars_from_cents(snap.best_yes_ask)
    P_no = price_dollars_from_cents(snap.best_no_ask)

    def f(P: float | None, maker: bool) -> float | None:
        if P is None:
            return None
        try:
            return kalshi_expected_fee_usd(P=P, C=int(contracts), maker=bool(maker))
        except Exception:
            return None

    return {
        "taker_fee_yes_usd": f(P_yes, maker=False),
        "maker_fee_yes_usd": f(P_yes, maker=True),
        "taker_fee_no_usd": f(P_no, maker=False),
        "maker_fee_no_usd": f(P_no, maker=True),
    }
