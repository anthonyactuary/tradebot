from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class BestPrices:
    best_yes_bid: int | None
    best_yes_ask: int | None
    best_no_bid: int | None
    best_no_ask: int | None


def _best_bid(levels: list[list[int]] | list[tuple[int, int]] | None) -> int | None:
    if not levels:
        return None
    # Arrays are sorted ascending; best bid is last element.
    return int(levels[-1][0])


def compute_best_prices(orderbook_json: dict[str, Any]) -> BestPrices:
    ob = orderbook_json.get("orderbook") or {}
    yes_levels = ob.get("yes") or []
    no_levels = ob.get("no") or []

    best_yes_bid = _best_bid(yes_levels)
    best_no_bid = _best_bid(no_levels)

    # Kalshi returns bids only; implied asks via reciprocal relationship.
    best_yes_ask = (100 - best_no_bid) if best_no_bid is not None else None
    best_no_ask = (100 - best_yes_bid) if best_yes_bid is not None else None

    return BestPrices(
        best_yes_bid=best_yes_bid,
        best_yes_ask=best_yes_ask,
        best_no_bid=best_no_bid,
        best_no_ask=best_no_ask,
    )


def compute_spread(orderbook_json: dict[str, Any]) -> int | None:
    prices = compute_best_prices(orderbook_json)
    if prices.best_yes_bid is None or prices.best_yes_ask is None:
        return None
    return prices.best_yes_ask - prices.best_yes_bid


def iter_levels(orderbook_json: dict[str, Any], *, side: str) -> Iterable[tuple[int, int]]:
    ob = orderbook_json.get("orderbook") or {}
    levels = ob.get(side) or []
    for price, qty in levels:
        yield int(price), int(qty)
