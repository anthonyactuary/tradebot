"""Trade edge + decision helpers.

This module is intentionally pure/side-effect free so it can be reused by:
- live inference printing
- quoting / execution logic

Definitions (per contract, in dollars):
- market_p_yes/no: Kalshi ask-implied probabilities (e.g. 83c => 0.83)
- p_yes/no: your model probability

Expected value (ignoring fees):
- EV_yes = p_yes * 1 - market_p_yes
- EV_no  = p_no  * 1 - market_p_no

Decision rule (user-specified):
- candidate buy YES if EV_yes > fees_yes
- candidate buy NO  if EV_no  > fees_no

Optionally, you can pass expected fees per contract (USD) and we will compute
fee-adjusted EV before applying the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


Side = Literal["YES", "NO"]


@dataclass(frozen=True)
class EdgeResult:
    p_yes: float
    p_no: float
    market_p_yes: float
    market_p_no: float
    ev_yes: float
    ev_no: float
    ev_yes_after_fees: float
    ev_no_after_fees: float


@dataclass(frozen=True)
class TradeDecision:
    side: Side | None
    reason: str
    edge: EdgeResult


def trade_edge_calculation(
    *,
    p_yes: float,
    market_p_yes: float,
    market_p_no: float | None = None,
    p_no: float | None = None,
    fee_yes_usd_per_contract: float = 0.0,
    fee_no_usd_per_contract: float = 0.0,
) -> EdgeResult:
    """Compute YES/NO EV (optionally fee-adjusted).

    Args:
        p_yes: Model probability for YES in [0, 1].
        market_p_yes: Market-implied probability/price for YES in [0, 1].
        market_p_no: Market-implied probability/price for NO in [0, 1]. If omitted, uses 1 - market_p_yes.
        p_no: Model probability for NO in [0, 1]. If omitted, uses 1 - p_yes.
        fee_yes_usd_per_contract: Expected fees for buying YES (USD per contract).
        fee_no_usd_per_contract: Expected fees for buying NO (USD per contract).

    Returns:
        EdgeResult with raw EV and EV after fees.
    """

    p_yes = float(p_yes)
    market_p_yes = float(market_p_yes)

    if p_no is None:
        p_no = 1.0 - p_yes
    if market_p_no is None:
        market_p_no = 1.0 - market_p_yes

    p_no = float(p_no)
    market_p_no = float(market_p_no)

    ev_yes = p_yes - market_p_yes
    ev_no = p_no - market_p_no

    fee_yes = float(fee_yes_usd_per_contract)
    fee_no = float(fee_no_usd_per_contract)

    ev_yes_after_fees = ev_yes - fee_yes
    ev_no_after_fees = ev_no - fee_no

    return EdgeResult(
        p_yes=p_yes,
        p_no=p_no,
        market_p_yes=market_p_yes,
        market_p_no=market_p_no,
        ev_yes=ev_yes,
        ev_no=ev_no,
        ev_yes_after_fees=ev_yes_after_fees,
        ev_no_after_fees=ev_no_after_fees,
    )


def trade_decision(
    *,
    edge: EdgeResult,
    threshold: float = 0.0,
) -> TradeDecision:
    """Convert an edge calculation into a single trade candidate.

    Decision uses fee-adjusted EV fields. With correctly computed fees,
    the natural threshold is 0.0 (positive EV after fees):
      - BUY YES if ev_yes_after_fees > threshold
      - BUY NO  if ev_no_after_fees  > threshold

    If both qualify, picks the larger fee-adjusted EV.
    """

    threshold = float(threshold)

    yes_ok = edge.ev_yes_after_fees > threshold
    no_ok = edge.ev_no_after_fees > threshold

    if yes_ok and no_ok:
        if edge.ev_yes_after_fees >= edge.ev_no_after_fees:
            return TradeDecision(side="YES", reason="both_positive_pick_higher", edge=edge)
        return TradeDecision(side="NO", reason="both_positive_pick_higher", edge=edge)

    if yes_ok:
        return TradeDecision(side="YES", reason="ev_yes_above_threshold", edge=edge)
    if no_ok:
        return TradeDecision(side="NO", reason="ev_no_above_threshold", edge=edge)

    return TradeDecision(side=None, reason="no_edge_above_threshold", edge=edge)
