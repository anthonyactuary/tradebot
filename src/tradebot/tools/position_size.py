"""Position sizing helpers (Kelly).

For a binary Kalshi contract (buying YES or NO):
- You pay price m (in dollars) to buy 1 contract.
- If correct, payout is $1.
- If wrong, you lose your cost.

Net odds b = (1-m)/m.
Kelly fraction (full Kelly):
  f* = (p(b+1) - 1) / b

For this binary payout, that simplifies to:
  f* = (p - m) / (1 - m)

Where:
- p is your model probability of being correct for the side you buy.
- m is the market price you pay for that side, in dollars.

This module also supports a fractional Kelly multiplier and optional per-contract fees.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class PositionSize:
    side: str
    p_win: float
    price: float
    fee_per_contract_usd: float

    kelly_full_fraction: float
    kelly_fraction: float

    stake_usd: float | None
    contracts: int | None


def kelly_fraction_binary(*, p: float, m: float) -> float:
    """Full Kelly fraction for a binary contract.

    Returns f* (can be negative). Caller typically clamps to >= 0.
    """
    p = float(p)
    m = float(m)

    if not (0.0 <= p <= 1.0) or not math.isfinite(p):
        raise ValueError("p must be within [0, 1]")
    if not (0.0 < m < 1.0) or not math.isfinite(m):
        raise ValueError("m must be within (0, 1)")

    # f* = (p - m) / (1 - m)
    denom = 1.0 - m
    if denom <= 0:
        return -1.0
    return (p - m) / denom


def calc_position_size(
    *,
    side: str,
    p_win: float,
    price: float,
    bankroll_usd: float | None = None,
    kelly_multiplier: float = 1.0,
    fee_per_contract_usd: float = 0.0,
    max_fraction: float = 1.0,
    min_contracts_if_positive_edge: int = 1,
) -> PositionSize:
    """Calculate position size using (fractional) Kelly.

    Args:
        side: Label for the side being bought (e.g. "YES" or "NO").
        p_win: Model probability of being correct for that side.
        price: Market price paid for that side (dollars in (0,1)).
        bankroll_usd: If provided, returns a contract count sized to bankroll.
        kelly_multiplier: Fractional Kelly scaling (1.0 = full Kelly).
        fee_per_contract_usd: Approx per-contract fee in USD (added to effective cost).
        max_fraction: Cap the Kelly fraction to this maximum.

    Returns:
        PositionSize including full Kelly fraction and scaled/clamped fraction.
    """

    k_full = float(kelly_fraction_binary(p=float(p_win), m=float(price)))

    km = float(kelly_multiplier)
    if not math.isfinite(km) or km < 0:
        raise ValueError("kelly_multiplier must be >= 0")

    k = k_full * km

    # Only bet positive edge, and cap to [0, max_fraction].
    cap = float(max_fraction)
    if not math.isfinite(cap) or cap <= 0:
        raise ValueError("max_fraction must be > 0")

    k = max(0.0, min(k, cap))

    stake_usd: float | None = None
    contracts: int | None = None

    if bankroll_usd is not None:
        br = float(bankroll_usd)
        if not math.isfinite(br) or br <= 0:
            raise ValueError("bankroll_usd must be > 0")

        stake_usd = br * k

        eff_cost = float(price) + float(fee_per_contract_usd)
        if eff_cost <= 0 or not math.isfinite(eff_cost):
            contracts = 0
        else:
            # Round down to avoid exceeding bankroll fraction.
            contracts = int(math.floor(stake_usd / eff_cost))
            contracts = max(0, contracts)

            min_c = int(min_contracts_if_positive_edge)
            if contracts == 0 and min_c > 0:
                # Only allow a min-lot when full Kelly indicates true positive edge.
                # Do not override when kelly_full_fraction <= 0.
                if float(k_full) > 0:
                    # Ensure we can afford the minimum lot.
                    if br >= float(eff_cost) * float(min_c):
                        contracts = min_c

    return PositionSize(
        side=str(side),
        p_win=float(p_win),
        price=float(price),
        fee_per_contract_usd=float(fee_per_contract_usd),
        kelly_full_fraction=float(k_full),
        kelly_fraction=float(k),
        stake_usd=stake_usd,
        contracts=contracts,
    )
