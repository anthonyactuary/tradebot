"""Order execution helpers.

This module is responsible for translating a trade decision into a Kalshi order.
It does NOT run the polling/inference loop; it only places (or dry-runs) orders.

When an order is placed, we log:
- ticker
- side (YES/NO)
- contracts (position size)
- contract price (ask)
- estimated total cost
- estimated fees
- EV after fees (per contract) and estimated $ expectation

Note: this implementation places a limit order at the current best ask for the
chosen side and uses IoC by default (to avoid leaving resting orders).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from tradebot.kalshi.client import KalshiClient
from tradebot.tools.kalshi_fees import kalshi_expected_fee_usd
from tradebot.tools.kalshi_market_poll import MarketSnapshot
from tradebot.tools.inventory_check import InventorySummary, fetch_inventory_summary
from tradebot.tools.position_size import PositionSize
from tradebot.tools.trade_edge_calculation import EdgeResult, TradeDecision


log = logging.getLogger(__name__)

FeeMode = Literal["maker", "taker"]
Side = Literal["YES", "NO"]
EntryMode = Literal["taker_ioc", "maker_only"]


def _fmt_usd(v: float | None) -> str:
    if v is None:
        return "-"
    try:
        return f"${float(v):.2f}"
    except Exception:
        return "-"


@dataclass(frozen=True)
class _LastEntry:
    ts: float
    side: Side


_LAST_ENTRY_BY_TICKER: dict[str, _LastEntry] = {}

# If we flatten a market (ticker), do not allow re-entry in that same market.
# We keep a per-ticker "no re-entry until" timestamp (epoch seconds).
_NO_REENTRY_UNTIL_BY_TICKER: dict[str, float] = {}

# If we re-enter immediately after a flip-flatten (decision change), we may want
# to hold that new position until expiry and never flatten again during the
# same market.
_HOLD_TO_EXPIRY_UNTIL_BY_TICKER: dict[str, float] = {}

# Rate-limit repeated entry order attempts per ticker (primarily for maker-only).
_LAST_ENTRY_ORDER_TS_BY_TICKER: dict[str, float] = {}

# Track a single live maker-only entry order per ticker so we can cancel/replace
# instead of stacking multiple resting GTC orders.
_OPEN_ENTRY_ORDER_ID_BY_TICKER: dict[str, str] = {}


@dataclass(frozen=True)
class _WorkingEntryOrder:
    order_id: str
    ts: float
    side: Side


_WORKING_ENTRY_ORDER_BY_TICKER: dict[str, _WorkingEntryOrder] = {}

# Track last-seen positions so we can detect a flatten that happened outside of
# our flip path (e.g., manual sell, other strategy/module). This is used to
# enforce the "no re-entry after flatten" policy.
_LAST_SEEN_POSITION_BY_TICKER: dict[str, int] = {}


@dataclass(frozen=True)
class _PendingFlipReentry:
    ts: float
    from_side: Side
    to_side: Side
    until: float


_PENDING_FLIP_REENTRY_BY_TICKER: dict[str, _PendingFlipReentry] = {}


# Track partial (tiered) flatten state per ticker.
# When a flip is first detected, we flatten half. After flatten_tier_seconds,
# if flip signal persists, we flatten the remainder.
@dataclass(frozen=True)
class _PartialFlatten:
    ts: float  # when first tier was executed
    from_side: Side  # side we're flattening from
    to_side: Side  # side we want to flip to
    original_qty: int  # original position size
    flattened_qty: int  # how many contracts flattened so far
    remaining_qty: int  # contracts still to flatten


_PARTIAL_FLATTEN_BY_TICKER: dict[str, _PartialFlatten] = {}


# If Kalshi reports trading is paused, back off briefly to avoid spamming orders.
_TRADING_PAUSED_UNTIL_TS: float | None = None
_TRADING_PAUSED_BACKOFF_SECONDS = 60.0


def _is_trading_paused_http_error(e: Exception) -> bool:
    if not isinstance(e, httpx.HTTPStatusError):
        return False
    try:
        if int(e.response.status_code) != 409:
            return False
    except Exception:
        return False

    # Best-effort parse Kalshi error payload:
    # {"error":{"code":"trading_is_paused", ...}}
    try:
        payload = e.response.json()
        code = ((payload or {}).get("error") or {}).get("code")
        return str(code) == "trading_is_paused"
    except Exception:
        # Fall back to substring match.
        try:
            return "trading_is_paused" in (e.response.text or "")
        except Exception:
            return False


def _set_trading_paused_backoff(*, ticker: str, reason: str) -> None:
    global _TRADING_PAUSED_UNTIL_TS
    now = float(time.time())
    _TRADING_PAUSED_UNTIL_TS = float(now + float(_TRADING_PAUSED_BACKOFF_SECONDS))
    log.warning(
        "TRADING_PAUSED_BACKOFF %s reason=%s until_epoch=%.3f backoff_s=%.0f",
        str(ticker),
        str(reason),
        float(_TRADING_PAUSED_UNTIL_TS),
        float(_TRADING_PAUSED_BACKOFF_SECONDS),
    )


def _trading_paused_backoff_active() -> bool:
    until = _TRADING_PAUSED_UNTIL_TS
    if until is None:
        return False
    return float(time.time()) < float(until)


def _sync_hold_timer_from_inventory(*, snap: MarketSnapshot, position: int) -> None:
    """If we're holding a position but have no hold-timer state, start it.

    This is especially important for maker-only entries where the fill can happen
    asynchronously long after the original order ack.
    """

    held = _held_side_from_position(int(position))
    if held is None:
        return
    last = _LAST_ENTRY_BY_TICKER.get(snap.ticker)
    if last is None or last.side != held:
        _LAST_ENTRY_BY_TICKER[snap.ticker] = _LastEntry(ts=time.time(), side=held)
        log.info("HOLD_TIMER_SYNC %s held=%s reason=inventory_nonzero", snap.ticker, held)


def re_entry(*, snap: MarketSnapshot) -> bool:
    """Whether we are allowed to enter *any* new position for this ticker.

    Policy: if we flattened during this market, we do not re-enter until the
    market expires (i.e., until after its cutoff / seconds_to_expiry passes).
    """

    now = time.time()
    until = _NO_REENTRY_UNTIL_BY_TICKER.get(snap.ticker)
    if until is None:
        return True
    if now >= float(until):
        _NO_REENTRY_UNTIL_BY_TICKER.pop(snap.ticker, None)
        return True
    return False


def _mark_no_reentry_until_expiry(*, snap: MarketSnapshot) -> float:
    now = time.time()
    # Prefer seconds_to_expiry from the snapshot; fall back to a conservative 1h.
    tte = None
    if snap.seconds_to_expiry is not None:
        try:
            tte = max(0, int(snap.seconds_to_expiry))
        except Exception:
            tte = None

    # Safety: if seconds_to_expiry is 0/negative (clock skew, stale snapshot, or
    # very late in the market), we still want to block re-entry long enough that
    # we don't flatten then immediately re-buy in the same ticker.
    min_block_seconds = 60
    block_for = int(tte) if tte is not None else 3600
    block_for = max(int(min_block_seconds), int(block_for))

    until = now + float(block_for)
    _NO_REENTRY_UNTIL_BY_TICKER[snap.ticker] = float(until)
    log.info(
        "NO_REENTRY_SET %s until_epoch=%.3f block_for=%ss tte=%ss",
        snap.ticker,
        float(until),
        str(int(block_for)),
        "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
    )
    return float(until)


def _hold_to_expiry_active(*, snap: MarketSnapshot) -> bool:
    now = time.time()
    until = _HOLD_TO_EXPIRY_UNTIL_BY_TICKER.get(snap.ticker)
    if until is None:
        return False
    if now >= float(until):
        _HOLD_TO_EXPIRY_UNTIL_BY_TICKER.pop(snap.ticker, None)
        return False
    return True


def _mark_hold_to_expiry(*, snap: MarketSnapshot) -> float:
    """Prevent future flattening for this ticker until expiry."""

    now = time.time()
    tte: int | None = None
    if snap.seconds_to_expiry is not None:
        try:
            tte = max(0, int(snap.seconds_to_expiry))
        except Exception:
            tte = None

    # Safety: keep at least a short lock even if tte is missing/0.
    min_lock_seconds = 60
    lock_for = int(tte) if tte is not None else 3600
    lock_for = max(int(min_lock_seconds), int(lock_for))

    until = now + float(lock_for)
    _HOLD_TO_EXPIRY_UNTIL_BY_TICKER[snap.ticker] = float(until)
    log.info(
        "HOLD_TO_EXPIRY_SET %s until_epoch=%.3f lock_for=%ss tte=%ss",
        snap.ticker,
        float(until),
        str(int(lock_for)),
        "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
    )
    return float(until)


def _get_pending_flip_reentry(*, snap: MarketSnapshot) -> _PendingFlipReentry | None:
    now = time.time()
    pending = _PENDING_FLIP_REENTRY_BY_TICKER.get(snap.ticker)
    if pending is None:
        return None
    if now >= float(pending.until):
        _PENDING_FLIP_REENTRY_BY_TICKER.pop(snap.ticker, None)
        return None
    return pending


def _arm_pending_flip_reentry(*, snap: MarketSnapshot, from_side: Side, to_side: Side) -> _PendingFlipReentry:
    now = time.time()

    tte: int | None = None
    if snap.seconds_to_expiry is not None:
        try:
            tte = max(0, int(snap.seconds_to_expiry))
        except Exception:
            tte = None

    # Keep at least a short window even if tte is missing/0.
    min_seconds = 60
    block_for = int(tte) if tte is not None else 3600
    block_for = max(int(min_seconds), int(block_for))

    pending = _PendingFlipReentry(
        ts=float(now),
        from_side=str(from_side),
        to_side=str(to_side),
        until=float(now + float(block_for)),
    )
    _PENDING_FLIP_REENTRY_BY_TICKER[snap.ticker] = pending
    log.info(
        "FLIP_REENTRY_ARMED %s from=%s to=%s until_epoch=%.3f tte=%ss",
        snap.ticker,
        str(from_side),
        str(to_side),
        float(pending.until),
        "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
    )
    return pending


async def _order_has_any_fill(*, client: KalshiClient, order_id: str) -> bool:
    """Best-effort check whether an order has at least one fill.

    Kalshi can be slightly delayed in surfacing fills right after an order ack.
    We retry briefly to reduce false negatives, which would otherwise break
    min-hold-time gating.
    """

    # Keep this short; we're in the trading loop.
    for attempt in range(6):
        try:
            # Give Kalshi a brief moment to record fills.
            await asyncio.sleep(0.25 if attempt == 0 else 0.20)
            fills_payload = await client.get_fills(order_id=str(order_id), limit=5)
            fills = list(fills_payload.get("fills") or [])
            if fills:
                return True
        except Exception:
            # We'll retry a couple times; caller will log final failure.
            pass
    return False


@dataclass(frozen=True)
class RiskLimits:
    """Simple pre-trade risk limits based on current Kalshi positions.

    Limits apply to the set of `risk_tickers` passed to `place_order()`.
    All fields are optional; only provided limits are enforced.
    """

    max_total_abs_contracts: int | None = None
    max_total_exposure_usd: float | None = None

    max_ticker_abs_contracts: int | None = None
    max_ticker_exposure_usd: float | None = None


@dataclass(frozen=True)
class PlacedOrder:
    ticker: str
    side: Literal["YES", "NO"]
    contracts: int
    price_cents: int
    price_dollars: float

    ev_after_fees_per_contract: float
    expected_profit_usd: float

    expected_fee_usd: float
    expected_cost_usd: float

    dry_run: bool
    response: dict[str, Any] | None


def _extract_order_id(resp: dict[str, Any] | None) -> str | None:
    if not resp:
        return None
    # Kalshi responses vary slightly in older code.
    order = resp.get("order") if isinstance(resp.get("order"), dict) else None
    if order and isinstance(order.get("order_id"), str):
        return order.get("order_id")
    v = resp.get("order_id")
    if isinstance(v, str):
        return v
    v2 = resp.get("id")
    if isinstance(v2, str):
        return v2
    return None


def _side_prices_from_snapshot(
    snap: MarketSnapshot, *, side: Side
) -> tuple[int | None, float | None]:
    """Return (best_ask_cents, ask_price_dollars) for the requested side."""
    if side == "YES":
        ask_c = snap.best_yes_ask
    else:
        ask_c = snap.best_no_ask

    if ask_c is None:
        return (None, None)

    try:
        cents = int(ask_c)
    except Exception:
        return (None, None)

    if cents < 0:
        return (None, None)

    return (cents, float(cents) / 100.0)


def _ev_after_fees_per_contract(edge: EdgeResult, *, side: Literal["YES", "NO"]) -> float:
    return float(edge.ev_yes_after_fees if side == "YES" else edge.ev_no_after_fees)


def _best_bid_cents_from_snapshot(snap: MarketSnapshot, *, side: Side) -> int | None:
    v = snap.best_yes_bid if side == "YES" else snap.best_no_bid
    if v is None:
        return None
    try:
        cents = int(v)
    except Exception:
        return None
    if cents < 0:
        return None
    return cents


def _held_side_from_position(pos: int) -> Side | None:
    # Position convention: positive=long YES, negative=long NO.
    if pos > 0:
        return "YES"
    if pos < 0:
        return "NO"
    return None


def _p_for_side(edge: EdgeResult, *, side: Side) -> float:
    return float(edge.p_yes if side == "YES" else edge.p_no)


def _spot_strike_diff_fraction(*, snap: MarketSnapshot) -> float | None:
    """Return |spot-strike|/spot if both are available."""

    spot = snap.btc_spot_usd
    strike = snap.price_to_beat
    if spot is None or strike is None:
        return None
    try:
        spot_f = float(spot)
        strike_f = float(strike)
    except Exception:
        return None
    if spot_f <= 0 or strike_f <= 0:
        return None
    return abs(spot_f - strike_f) / spot_f


async def _buy_opposite_to_flat(
    *,
    client: KalshiClient,
    snap: MarketSnapshot,
    held_side: Side,
    close_qty: int,
    time_in_force: Literal["fill_or_kill", "good_till_canceled", "immediate_or_cancel"] | None,
    dry_run: bool,
) -> dict[str, Any] | None:
    """Flatten position by buying the opposite side.

    Instead of selling YES at YES_bid, we buy NO at NO_ask.
    This is economically equivalent but incurs lower fees because:
    - Selling YES @ 80c -> fee calculated on 80c
    - Buying NO @ 20c -> fee calculated on 20c (much lower!)

    The position automatically offsets on Kalshi.
    """
    # Determine the opposite side to buy
    opposite_side: Side = "NO" if held_side == "YES" else "YES"

    # Get best ask for the opposite side
    ask_cents, ask_dollars = _side_prices_from_snapshot(snap, side=opposite_side)
    if ask_cents is None or ask_dollars is None:
        log.warning(
            "FLATTEN_BLOCK %s held=%s qty=%d reason=missing_opposite_ask opposite=%s best_yes_ask=%s best_no_ask=%s",
            snap.ticker,
            held_side,
            int(close_qty),
            opposite_side,
            str(snap.best_yes_ask),
            str(snap.best_no_ask),
        )
        return None

    # Calculate fee savings for logging
    old_bid_cents = _best_bid_cents_from_snapshot(snap, side=held_side)
    fee_saved_pct = 0.0
    if old_bid_cents is not None and int(old_bid_cents) > 0:
        # Old method: sell at bid_cents, new method: buy at ask_cents
        fee_saved_pct = (1.0 - float(ask_cents) / float(old_bid_cents)) * 100.0

    log.info(
        "FLATTEN %s buy_%s_to_close_%s qty=%d @ %dc (vs sell_%s @ %dc) fee_saved~%.1f%% spot=%s strike=%s tte=%ss",
        snap.ticker,
        opposite_side,
        held_side,
        int(close_qty),
        int(ask_cents),
        held_side,
        int(old_bid_cents or 0),
        float(fee_saved_pct),
        _fmt_usd(snap.btc_spot_usd),
        _fmt_usd(snap.price_to_beat),
        "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
    )

    if dry_run:
        return {
            "dry_run": True,
            "ticker": snap.ticker,
            "action": "buy",
            "side": opposite_side,
            "count": int(close_qty),
            "price": int(ask_cents),
            "flatten_method": "buy_opposite",
        }

    client_order_id = str(uuid.uuid4())
    if opposite_side == "YES":
        try:
            return await client.create_order(
                ticker=snap.ticker,
                side="yes",
                action="buy",
                count=int(close_qty),
                order_type="limit",
                yes_price=int(ask_cents),
                client_order_id=client_order_id,
                time_in_force=time_in_force,
            )
        except Exception as e:
            if _is_trading_paused_http_error(e):
                _set_trading_paused_backoff(ticker=snap.ticker, reason="flatten_trading_is_paused")
                return None
            raise
    try:
        return await client.create_order(
            ticker=snap.ticker,
            side="no",
            action="buy",
            count=int(close_qty),
            order_type="limit",
            no_price=int(ask_cents),
            client_order_id=client_order_id,
            time_in_force=time_in_force,
        )
    except Exception as e:
        if _is_trading_paused_http_error(e):
            _set_trading_paused_backoff(ticker=snap.ticker, reason="flatten_trading_is_paused")
            return None
        raise


def _get_partial_flatten(*, snap: MarketSnapshot) -> _PartialFlatten | None:
    """Get any active partial flatten state for this ticker."""
    return _PARTIAL_FLATTEN_BY_TICKER.get(snap.ticker)


def _clear_partial_flatten(*, snap: MarketSnapshot) -> None:
    """Clear partial flatten state for this ticker."""
    _PARTIAL_FLATTEN_BY_TICKER.pop(snap.ticker, None)


def _risk_delta_contracts(*, side: Literal["YES", "NO"], qty: int) -> int:
    # Kalshi position convention in archived code: positive=long YES, negative=short YES.
    # Buying YES increases position; buying NO decreases position.
    return int(qty) if side == "YES" else -int(qty)


def _check_risk_limits(
    *,
    summary: InventorySummary,
    ticker: str,
    current_position: int,
    projected_position: int,
    projected_total_abs_contracts: int,
    projected_total_exposure_usd: float,
    projected_ticker_abs_contracts: int,
    projected_ticker_exposure_usd: float,
    limits: RiskLimits,
) -> str | None:
    if limits.max_total_abs_contracts is not None and projected_total_abs_contracts > int(limits.max_total_abs_contracts):
        return (
            f"risk_block total_abs_contracts={projected_total_abs_contracts}"
            f" > max_total_abs_contracts={int(limits.max_total_abs_contracts)}"
        )

    if limits.max_total_exposure_usd is not None and projected_total_exposure_usd > float(limits.max_total_exposure_usd):
        return (
            f"risk_block total_exposure=${projected_total_exposure_usd:.2f}"
            f" > max_total_exposure=${float(limits.max_total_exposure_usd):.2f}"
        )

    if limits.max_ticker_abs_contracts is not None and projected_ticker_abs_contracts > int(limits.max_ticker_abs_contracts):
        return (
            f"risk_block {ticker} abs_contracts={projected_ticker_abs_contracts}"
            f" > max_ticker_abs_contracts={int(limits.max_ticker_abs_contracts)}"
            f" (pos {current_position} -> {projected_position})"
        )

    if limits.max_ticker_exposure_usd is not None and projected_ticker_exposure_usd > float(limits.max_ticker_exposure_usd):
        return (
            f"risk_block {ticker} exposure=${projected_ticker_exposure_usd:.2f}"
            f" > max_ticker_exposure=${float(limits.max_ticker_exposure_usd):.2f}"
            f" (pos {current_position} -> {projected_position})"
        )

    return None


async def place_order(
    *,
    client: KalshiClient,
    snap: MarketSnapshot,
    decision: TradeDecision,
    edge: EdgeResult,
    position: PositionSize | None,
    run_id: str | None = None,
    poll_id: int | None = None,
    max_seconds_to_expiry: int | None = None,
    enable_flip: bool = True,
    min_hold_seconds: int = 120,
    entry_mode: EntryMode = "taker_ioc",
    max_entry_spread_cents: int | None = 5,
    maker_improve_cents: int = 0,
    min_seconds_between_entry_orders: int = 20,
    min_entry_edge: float = 0.025,
    min_entry_edge_late: float = 0.05,
    entry_edge_tte_threshold: int = 300,
    dead_zone: float = 0.05,
    exit_delta: float = 0.09,
    catastrophic_exit_delta: float = 0.20,
    monotonicity_guard_enabled: bool = True,
    monotonicity_tte_threshold: int = 360,
    monotonicity_market_confidence: float = 0.80,
    monotonicity_model_diff: float = 0.05,
    allow_reentry_after_flatten: bool = False,
    flatten_tier_enabled: bool = True,
    flatten_tier_seconds: int = 30,
    confirm_entry_fill: bool = True,
    risk_limits: RiskLimits | None = None,
    risk_tickers: list[str] | None = None,
    inventory: InventorySummary | None = None,
    fee_mode: FeeMode = "taker",
    dry_run: bool = True,
    spot_strike_sanity_enabled: bool = True,
    max_spot_strike_deviation_fraction: float = 0.02,
    time_in_force: Literal["fill_or_kill", "good_till_canceled", "immediate_or_cancel"] | None = "immediate_or_cancel",
) -> PlacedOrder | None:
    """Place a Kalshi order for the trade decision.

    Returns:
        PlacedOrder or None if there is no actionable decision/size.
    """

    if decision.side is None:
        return None

    if _trading_paused_backoff_active():
        log.info("TRADING_PAUSED_SKIP %s reason=backoff_active", snap.ticker)
        log.info(
            "ORDER_BLOCK %s side=%s reason_block=trading_paused_backoff",
            snap.ticker,
            str(decision.side),
            extra={
                "csv_fields": {
                    "run_id": run_id or "",
                    "poll_id": poll_id if poll_id is not None else "",
                    "ticker": snap.ticker,
                    "side": str(decision.side),
                    "reason": "trading_paused_backoff",
                }
            },
        )
        return None

    side: Side = "YES" if decision.side == "YES" else "NO"

    # Determine current position from the caller-provided inventory snapshot if available.
    # We use this to:
    # - Only apply the re-entry block when flat (so exits/flattening are never blocked)
    # - Detect a flatten that happened outside this module (pos!=0 -> 0)
    current_pos_snapshot: int | None = None
    if inventory is not None:
        ti0 = inventory.per_ticker.get(snap.ticker)
        if ti0 is not None:
            try:
                current_pos_snapshot = int(ti0.position)
            except Exception:
                current_pos_snapshot = None

    if current_pos_snapshot is not None:
        last_pos = _LAST_SEEN_POSITION_BY_TICKER.get(snap.ticker)
        # If we were previously holding and are now flat, enforce no-reentry.
        if last_pos is not None and int(last_pos) != 0 and int(current_pos_snapshot) == 0:
            pending = _get_pending_flip_reentry(snap=snap) if bool(allow_reentry_after_flatten) else None
            if bool(allow_reentry_after_flatten) and pending is not None:
                # A flip-flatten may complete asynchronously; don't set the global
                # no-reentry block in that case.
                _NO_REENTRY_UNTIL_BY_TICKER.pop(snap.ticker, None)
                log.info(
                    "FLATTEN_DETECTED %s last_pos=%d now_pos=%d reason=inventory_transition pending_flip_reentry=true from=%s to=%s",
                    snap.ticker,
                    int(last_pos),
                    int(current_pos_snapshot),
                    str(pending.from_side),
                    str(pending.to_side),
                )
            else:
                _mark_no_reentry_until_expiry(snap=snap)
                log.info(
                    "FLATTEN_DETECTED %s last_pos=%d now_pos=%d reason=inventory_transition",
                    snap.ticker,
                    int(last_pos),
                    int(current_pos_snapshot),
                )
        _LAST_SEEN_POSITION_BY_TICKER[snap.ticker] = int(current_pos_snapshot)

    pending = _get_pending_flip_reentry(snap=snap) if bool(allow_reentry_after_flatten) else None
    flat_now = (current_pos_snapshot is None or int(current_pos_snapshot) == 0)
    pending_target_ok = bool(pending is not None and flat_now and str(pending.to_side) == str(side))

    # If we previously flattened this ticker/market, do not re-enter until expiry.
    # IMPORTANT: only apply this when we're flat; we never want to block exits/flattening
    # due to stale/noisy state.
    if flat_now and not re_entry(snap=snap) and not bool(pending_target_ok):
        until = _NO_REENTRY_UNTIL_BY_TICKER.get(snap.ticker)
        log.info(
            "REENTRY_BLOCK %s reason=flattened_prior until_epoch=%.3f tte=%ss",
            snap.ticker,
            float(until or 0.0),
            "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
        )
        log.info(
            "ORDER_BLOCK %s side=%s reason_block=reentry_after_flatten",
            snap.ticker,
            str(side),
            extra={
                "csv_fields": {
                    "run_id": run_id or "",
                    "poll_id": poll_id if poll_id is not None else "",
                    "ticker": snap.ticker,
                    "side": str(side),
                    "reason": "reentry_after_flatten",
                    "tte_s": snap.seconds_to_expiry,
                }
            },
        )
        return None

    # If a pending flip re-entry exists, only allow entry for the intended side.
    if pending is not None and flat_now and not bool(pending_target_ok):
        log.info(
            "PENDING_FLIP_REENTRY_BLOCK %s want=%s pending_to=%s reason=side_mismatch",
            snap.ticker,
            str(side),
            str(pending.to_side),
        )
        log.info(
            "ORDER_BLOCK %s side=%s reason_block=pending_flip_side_mismatch",
            snap.ticker,
            str(side),
            extra={
                "csv_fields": {
                    "run_id": run_id or "",
                    "poll_id": poll_id if poll_id is not None else "",
                    "ticker": snap.ticker,
                    "side": str(side),
                    "reason": "pending_flip_side_mismatch",
                }
            },
        )
        return None

    ev_new_side_after_fees = _ev_after_fees_per_contract(edge, side=side)

    qty: int | None = None
    if position is not None and position.contracts is not None:
        qty = int(position.contracts)

    if qty is None:
        # Fallback: allow a 1-lot if caller didn't size.
        qty = 1

    if qty <= 0:
        return None

    # Rate-limit entry orders to prevent spam (applies to all entry modes).
    # Only applies to fresh entries; flatten/exit are not rate-limited.
    if min_seconds_between_entry_orders > 0:
        now_ts = time.time()
        last_ts = _LAST_ENTRY_ORDER_TS_BY_TICKER.get(snap.ticker)
        if last_ts is not None and (now_ts - float(last_ts)) < float(min_seconds_between_entry_orders):
            log.info(
                "ENTRY_RATE_BLOCK %s mode=%s since_last=%.1fs < min_seconds_between_entry_orders=%ss",
                snap.ticker,
                str(entry_mode),
                float(now_ts - float(last_ts)),
                str(int(min_seconds_between_entry_orders)),
            )
            log.info(
                "ORDER_BLOCK %s side=%s reason_block=entry_rate_limit",
                snap.ticker,
                str(side),
                extra={
                    "csv_fields": {
                        "run_id": run_id or "",
                        "poll_id": poll_id if poll_id is not None else "",
                        "ticker": snap.ticker,
                        "side": str(side),
                        "reason": "entry_rate_limit",
                        "mode": str(entry_mode),
                    }
                },
            )
            return None

    # Maker-only entry guardrails (only for entry buys; flatten/exit remains unchanged).
    # NOTE: This function only places BUY entry orders and SELL reduce-only flatten orders.
    if str(entry_mode) == "maker_only":
        # "Single position only": do not place a second entry if we already have any position.
        tickers_for_inv = risk_tickers if risk_tickers is not None else [snap.ticker]
        inv_for_entry = inventory if inventory is not None else await fetch_inventory_summary(client=client, tickers=tickers_for_inv)
        ti_entry = inv_for_entry.per_ticker.get(snap.ticker)
        cur_pos_entry = int(ti_entry.position) if ti_entry is not None else 0

        # If a maker order is working and we now have a position, treat that as a fill/position-change signal:
        # start hold timer, and cancel any remaining resting order to avoid adding.
        if cur_pos_entry != 0:
            _sync_hold_timer_from_inventory(snap=snap, position=int(cur_pos_entry))
            open_id = _OPEN_ENTRY_ORDER_ID_BY_TICKER.pop(snap.ticker, None)
            _WORKING_ENTRY_ORDER_BY_TICKER.pop(snap.ticker, None)
            if open_id:
                try:
                    await client.cancel_order(order_id=str(open_id))
                    log.info("ENTRY_MAKER_CANCEL_ON_POSITION %s order_id=%s", snap.ticker, str(open_id))
                except Exception as e:
                    log.warning(
                        "ENTRY_MAKER_CANCEL_ON_POSITION_ERROR %s order_id=%s error=%s",
                        snap.ticker,
                        str(open_id),
                        e,
                    )
            log.info("ENTRY_POSITION_BLOCK %s mode=maker_only cur_pos=%d", snap.ticker, int(cur_pos_entry))
            log.info(
                "ORDER_BLOCK %s side=%s reason_block=maker_position_exists",
                snap.ticker,
                str(side),
                extra={
                    "csv_fields": {
                        "run_id": run_id or "",
                        "poll_id": poll_id if poll_id is not None else "",
                        "ticker": snap.ticker,
                        "side": str(side),
                        "reason": "maker_position_exists",
                        "mode": str(entry_mode),
                    }
                },
            )
            return None

    # Flip handling (do not open opposite-side while still holding current).
    # If holding YES and we want NO: buy NO to flatten (cheaper fees than selling YES).
    # If holding NO and we want YES: buy YES to flatten (cheaper fees than selling NO).
    #
    # Tiered flatten: First flatten HALF, wait flatten_tier_seconds, then flatten remainder
    # if flip signal persists. This protects against whipsaw noise.
    did_flatten = False
    did_partial_flatten = False
    flip_reentry = False
    flip_from_side: Side | None = None
    hold_to_expiry_after_entry = False
    base_total_abs_contracts: int | None = None
    base_total_exposure_usd: float | None = None
    current_pos_override: int | None = None
    current_exposure_override: float | None = None

    if enable_flip:
        # We need inventory to know current position.
        tickers_for_inv = risk_tickers if risk_tickers is not None else [snap.ticker]
        inv_for_flip = inventory if inventory is not None else await fetch_inventory_summary(client=client, tickers=tickers_for_inv)
        ti = inv_for_flip.per_ticker.get(snap.ticker)
        cur_pos = int(ti.position) if ti is not None else 0
        held = _held_side_from_position(cur_pos)

        # Check for existing partial flatten state
        partial = _get_partial_flatten(snap=snap)

        if held is not None and held != side:
            flip_from_side = held
            if _hold_to_expiry_active(snap=snap):
                until = _HOLD_TO_EXPIRY_UNTIL_BY_TICKER.get(snap.ticker)
                log.info(
                    "FLIP_BLOCK %s held=%s want=%s reason=hold_to_expiry until_epoch=%.3f tte=%ss",
                    snap.ticker,
                    held,
                    side,
                    float(until or 0.0),
                    "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
                )
                return None

            # If we have a live position but no hold timer, start it from inventory.
            # This matters most with maker-only entries where fill can occur later.
            _sync_hold_timer_from_inventory(snap=snap, position=int(cur_pos))

            # Exit gate: avoid instant-close churn.
            # Only flip if the probability of the currently-held side has moved
            # meaningfully past 0.5 by exit_delta (or catastrophic threshold).
            p_held = _p_for_side(edge, side=held)
            if p_held >= 0.5 - float(exit_delta):
                # Signal not strong enough - clear any partial flatten state
                if partial is not None:
                    log.info(
                        "PARTIAL_FLATTEN_CANCEL %s held=%s want=%s reason=exit_delta_not_met p_held=%.3f",
                        snap.ticker,
                        held,
                        side,
                        float(p_held),
                    )
                    _clear_partial_flatten(snap=snap)
                log.info(
                    "FLIP_BLOCK %s held=%s want=%s p_held=%.3f >= %.3f (0.5-exit_delta)",
                    snap.ticker,
                    held,
                    side,
                    float(p_held),
                    0.5 - float(exit_delta),
                )
                return None

            # Minimum hold time unless catastrophic.
            now_ts = time.time()
            last = _LAST_ENTRY_BY_TICKER.get(snap.ticker)
            if last is not None and last.side == held:
                held_for = float(now_ts - float(last.ts))
                catastrophic = p_held < (0.5 - float(catastrophic_exit_delta))
                if held_for < float(min_hold_seconds) and not catastrophic:
                    log.info(
                        "HOLD_BLOCK %s held=%s held_for=%.1fs < min_hold_seconds=%ss p_held=%.3f",
                        snap.ticker,
                        held,
                        float(held_for),
                        str(int(min_hold_seconds)),
                        float(p_held),
                    )
                    return None

            # Only attempt flip if the new side still has positive EV after fees.
            if float(ev_new_side_after_fees) > 0:
                close_qty = abs(cur_pos)
                if close_qty > 0:
                    # Determine how much to flatten this tier
                    flatten_qty: int = 0
                    is_tier_two: bool = False

                    if bool(flatten_tier_enabled) and partial is None:
                        # TIER 1: First time seeing flip signal - flatten HALF
                        flatten_qty = max(1, close_qty // 2)
                        log.info(
                            "FLATTEN_TIER1 %s held=%s want=%s full_qty=%d flatten_qty=%d (half)",
                            snap.ticker,
                            held,
                            side,
                            int(close_qty),
                            int(flatten_qty),
                        )
                    elif bool(flatten_tier_enabled) and partial is not None:
                        # Check if enough time has passed for tier 2
                        elapsed = float(now_ts - float(partial.ts))
                        if elapsed < float(flatten_tier_seconds):
                            log.info(
                                "FLATTEN_TIER2_WAIT %s held=%s want=%s elapsed=%.1fs < tier_seconds=%ds remaining=%d",
                                snap.ticker,
                                held,
                                side,
                                float(elapsed),
                                int(flatten_tier_seconds),
                                int(partial.remaining_qty),
                            )
                            return None
                        # TIER 2: Time elapsed and signal persists - flatten remainder
                        flatten_qty = int(partial.remaining_qty)
                        is_tier_two = True
                        log.info(
                            "FLATTEN_TIER2 %s held=%s want=%s elapsed=%.1fs remaining_qty=%d (completing flatten)",
                            snap.ticker,
                            held,
                            side,
                            float(elapsed),
                            int(flatten_qty),
                        )
                    else:
                        # Tiered flatten disabled - flatten full position
                        flatten_qty = close_qty

                    if flatten_qty <= 0:
                        log.warning("FLATTEN_SKIP %s flatten_qty=%d", snap.ticker, int(flatten_qty))
                        return None

                    # Execute flatten by buying opposite side (lower fees!)
                    resp = await _buy_opposite_to_flat(
                        client=client,
                        snap=snap,
                        held_side=held,
                        close_qty=int(flatten_qty),
                        time_in_force=time_in_force,
                        dry_run=bool(dry_run),
                    )
                    if resp is None:
                        return None

                    # Update partial flatten state
                    remaining_after = close_qty - flatten_qty
                    if remaining_after > 0 and bool(flatten_tier_enabled) and not is_tier_two:
                        # Store partial flatten state for tier 2
                        _PARTIAL_FLATTEN_BY_TICKER[snap.ticker] = _PartialFlatten(
                            ts=float(now_ts),
                            from_side=held,
                            to_side=side,
                            original_qty=int(close_qty),
                            flattened_qty=int(flatten_qty),
                            remaining_qty=int(remaining_after),
                        )
                        did_partial_flatten = True
                        log.info(
                            "PARTIAL_FLATTEN_ARMED %s from=%s to=%s flattened=%d remaining=%d wait_seconds=%d",
                            snap.ticker,
                            held,
                            side,
                            int(flatten_qty),
                            int(remaining_after),
                            int(flatten_tier_seconds),
                        )
                        # Don't proceed to re-entry yet - wait for tier 2
                        return None
                    else:
                        # Full flatten complete (either tier 2 or tiering disabled)
                        _clear_partial_flatten(snap=snap)

                    if bool(allow_reentry_after_flatten):
                        _arm_pending_flip_reentry(snap=snap, from_side=held, to_side=side)

                    if dry_run:
                        # Simulate flatten for subsequent checks.
                        did_flatten = True
                        current_pos_override = 0
                        current_exposure_override = 0.0
                        # If we flattened (even simulated), we are no longer holding the prior side.
                        # Clear any existing hold-timer state for this ticker.
                        _LAST_ENTRY_BY_TICKER.pop(snap.ticker, None)
                        # Default policy blocks re-entry; optional config allows re-entry after a flip.
                        if not bool(allow_reentry_after_flatten):
                            _mark_no_reentry_until_expiry(snap=snap)
                        base_total_abs_contracts = int(
                            inv_for_flip.total_abs_contracts
                            - (ti.abs_contracts if ti is not None else abs(cur_pos))
                        )
                        base_total_exposure_usd = float(
                            inv_for_flip.total_exposure_usd - (ti.exposure_usd if ti is not None else 0.0)
                        )
                    else:
                        # Re-fetch to confirm we're flat before flipping.
                        inv2 = await fetch_inventory_summary(client=client, tickers=tickers_for_inv)
                        ti2 = inv2.per_ticker.get(snap.ticker)
                        pos2 = int(ti2.position) if ti2 is not None else 0
                        if pos2 != 0:
                            log.warning("FLATTEN_INCOMPLETE %s pos=%d; skipping flip", snap.ticker, pos2)
                            return None
                        did_flatten = True
                        current_pos_override = 0
                        current_exposure_override = 0.0
                        # We are flat now; clear any prior hold-timer state for this ticker.
                        _LAST_ENTRY_BY_TICKER.pop(snap.ticker, None)
                        # Default policy blocks re-entry; optional config allows re-entry after a flip.
                        if not bool(allow_reentry_after_flatten):
                            _mark_no_reentry_until_expiry(snap=snap)
                        base_total_abs_contracts = int(inv2.total_abs_contracts)
                        base_total_exposure_usd = float(inv2.total_exposure_usd)

                    # Policy: default is flatten-only (no re-entry). If enabled, allow re-entry
                    # after flip-flatten and then hold that new position until expiry.
                    if did_flatten and not bool(allow_reentry_after_flatten):
                        log.info(
                            "FLATTEN_ONLY %s want=%s reason=no_reentry_after_flatten ev_after_fees=%.4f",
                            snap.ticker,
                            side,
                            float(ev_new_side_after_fees),
                        )
                        return None

                    if did_flatten and bool(allow_reentry_after_flatten):
                        # Ensure any stale no-reentry state does not block this re-entry.
                        _NO_REENTRY_UNTIL_BY_TICKER.pop(snap.ticker, None)
                        hold_to_expiry_after_entry = True
                        flip_reentry = True

        elif held is None or held == side:
            # No flip needed, but clear any stale partial flatten state if signal changed
            if partial is not None and (held is None or str(partial.to_side) != str(side)):
                log.info(
                    "PARTIAL_FLATTEN_CLEAR %s reason=no_longer_flipping partial_to=%s current_side=%s",
                    snap.ticker,
                    str(partial.to_side),
                    str(side) if held else "flat",
                )
                _clear_partial_flatten(snap=snap)

    allow_entry_post_flatten = bool(did_flatten and allow_reentry_after_flatten and hold_to_expiry_after_entry)

    # If flatten completed in a prior poll and we're now flat, honor pending flip re-entry.
    if pending is not None and flat_now and bool(pending_target_ok) and not bool(_hold_to_expiry_active(snap=snap)):
        hold_to_expiry_after_entry = True
        flip_reentry = True
        flip_from_side = pending.from_side

    # Sanity gate: if the API is returning an obviously bad strike early in the market,
    # block entry orders to avoid trading on bad data.
    if bool(spot_strike_sanity_enabled) and float(max_spot_strike_deviation_fraction) > 0:
        diff = _spot_strike_diff_fraction(snap=snap)
        if diff is not None and float(diff) > float(max_spot_strike_deviation_fraction):
            log.warning(
                "SPOT_STRIKE_BLOCK %s side=%s spot=%s strike=%s diff_pct=%.2f > max_pct=%.2f strike_src=%s",
                snap.ticker,
                str(side),
                _fmt_usd(snap.btc_spot_usd),
                _fmt_usd(snap.price_to_beat),
                float(diff) * 100.0,
                float(max_spot_strike_deviation_fraction) * 100.0,
                str(snap.price_to_beat_source or "?"),
            )
            log.info(
                "ORDER_BLOCK %s side=%s reason_block=spot_strike_deviation",
                snap.ticker,
                str(side),
                extra={
                    "csv_fields": {
                        "run_id": run_id or "",
                        "poll_id": poll_id if poll_id is not None else "",
                        "ticker": snap.ticker,
                        "side": str(side),
                        "reason": "spot_strike_deviation",
                        "spot_usd": snap.btc_spot_usd,
                        "strike_usd": snap.price_to_beat,
                        "strike_src": str(snap.price_to_beat_source or "?"),
                        "diff_pct": float(diff) * 100.0,
                        "max_pct": float(max_spot_strike_deviation_fraction) * 100.0,
                    }
                },
            )
            return None

    # Dead-zone guardrail: block fresh entries when model is near a coin flip.
    # Applies when we are flat (or don't have an inventory snapshot).
    if (
        (not did_flatten or allow_entry_post_flatten)
        and (current_pos_snapshot is None or int(current_pos_snapshot) == 0)
        and float(dead_zone) > 0.0
    ):
        p_side = _p_for_side(edge, side=side)
        dist = abs(float(p_side) - 0.5)
        if float(dist) < float(dead_zone):
            log.info(
                "DEADZONE_BLOCK %s side=%s p_side=%.4f dist=%.4f < dead_zone=%.4f p_yes=%.4f p_no=%.4f market_yes=%.4f market_no=%.4f ev_after_fees=%.4f",
                snap.ticker,
                side,
                float(p_side),
                float(dist),
                float(dead_zone),
                float(edge.p_yes),
                float(edge.p_no),
                float(edge.market_p_yes),
                float(edge.market_p_no),
                float(ev_new_side_after_fees),
            )
            log.info(
                "ORDER_BLOCK %s side=%s reason_block=deadzone",
                snap.ticker,
                str(side),
                extra={
                    "csv_fields": {
                        "run_id": run_id or "",
                        "poll_id": poll_id if poll_id is not None else "",
                        "ticker": snap.ticker,
                        "side": str(side),
                        "reason": "deadzone",
                        "p_yes": edge.p_yes,
                        "p_no": edge.p_no,
                        "market_p_yes": edge.market_p_yes,
                        "market_p_no": edge.market_p_no,
                    }
                },
            )
            return None

    # Entry gate: don't open fresh positions when model ~= market.
    # Applies to opening/adding positions; does not apply to flattening.
    # Use TTE-based dynamic threshold: require higher edge near expiry.
    tte_for_edge = snap.seconds_to_expiry if snap.seconds_to_expiry is not None else 9999
    effective_min_entry_edge = (
        float(min_entry_edge_late)
        if int(tte_for_edge) < int(entry_edge_tte_threshold)
        else float(min_entry_edge)
    )
    if (not did_flatten or allow_entry_post_flatten) and float(ev_new_side_after_fees) <= effective_min_entry_edge:
        log.info(
            "ENTRY_BLOCK %s side=%s ev_after_fees=%.4f <= min_entry_edge=%.4f (tte=%ds threshold=%ds)",
            snap.ticker,
            side,
            float(ev_new_side_after_fees),
            effective_min_entry_edge,
            int(tte_for_edge),
            int(entry_edge_tte_threshold),
        )
        log.info(
            "ORDER_BLOCK %s side=%s reason_block=min_entry_edge",
            snap.ticker,
            str(side),
            extra={
                "csv_fields": {
                    "run_id": run_id or "",
                    "poll_id": poll_id if poll_id is not None else "",
                    "ticker": snap.ticker,
                    "side": str(side),
                    "reason": "min_entry_edge",
                    "ev_after_fees": float(ev_new_side_after_fees),
                    "effective_min_entry_edge": effective_min_entry_edge,
                    "tte_s": int(tte_for_edge),
                }
            },
        )
        return None

    # Monotonicity guard: Block trades in late markets when market is already very confident
    # and model agrees with market. This prevents microstructure noise from triggering trades
    # when both model and market already indicate a near-certain outcome.
    if bool(monotonicity_guard_enabled) and (not did_flatten or allow_entry_post_flatten):
        tte_mono = snap.seconds_to_expiry if snap.seconds_to_expiry is not None else 9999
        market_p_yes = edge.market_p_yes if edge.market_p_yes is not None else 0.5
        market_p_no = edge.market_p_no if edge.market_p_no is not None else 0.5
        p_model = edge.p_yes if edge.p_yes is not None else 0.5

        # Check if market is confident in either direction
        market_confident_yes = float(market_p_yes) >= float(monotonicity_market_confidence)
        market_confident_no = float(market_p_no) >= float(monotonicity_market_confidence)

        # Model agreement: model is close to market price
        if market_confident_yes:
            model_diff = abs(float(p_model) - float(market_p_yes))
        elif market_confident_no:
            model_diff = abs((1.0 - float(p_model)) - float(market_p_no))
        else:
            model_diff = 1.0  # No confidence threshold met, won't block

        should_block_monotonicity = (
            int(tte_mono) < int(monotonicity_tte_threshold)
            and (market_confident_yes or market_confident_no)
            and float(model_diff) < float(monotonicity_model_diff)
        )

        if should_block_monotonicity:
            confident_side = "YES" if market_confident_yes else "NO"
            confident_p = float(market_p_yes) if market_confident_yes else float(market_p_no)
            log.info(
                "MONOTONICITY_BLOCK %s side=%s tte=%ds market_p_%s=%.2f p_model=%.2f diff=%.4f < threshold=%.4f",
                snap.ticker,
                side,
                int(tte_mono),
                confident_side.lower(),
                confident_p,
                float(p_model),
                float(model_diff),
                float(monotonicity_model_diff),
            )
            log.info(
                "ORDER_BLOCK %s side=%s reason_block=monotonicity_guard",
                snap.ticker,
                str(side),
                extra={
                    "csv_fields": {
                        "run_id": run_id or "",
                        "poll_id": poll_id if poll_id is not None else "",
                        "ticker": snap.ticker,
                        "side": str(side),
                        "reason": "monotonicity_guard",
                        "tte_s": int(tte_mono),
                        "market_p_yes": float(market_p_yes),
                        "market_p_no": float(market_p_no),
                        "p_model": float(p_model),
                        "model_diff": float(model_diff),
                    }
                },
            )
            return None

    # Time gate: optionally block trades that are "too early" in the market window.
    # Applies to opening/adding, not to flattening.
    if max_seconds_to_expiry is not None and snap.seconds_to_expiry is not None:
        try:
            tte = int(snap.seconds_to_expiry)
        except Exception:
            tte = None
        if tte is not None and tte > int(max_seconds_to_expiry):
            log.info(
                "TTE_BLOCK %s tte=%ss > max_tte=%ss",
                snap.ticker,
                str(tte),
                str(int(max_seconds_to_expiry)),
            )
            return None

    # Entry pricing
    ask_cents: int | None
    ask_dollars: float | None
    entry_price_cents: int | None = None
    entry_price_dollars: float | None = None
    time_in_force_entry: Literal["fill_or_kill", "good_till_canceled", "immediate_or_cancel"] | None = time_in_force
    maker_for_fee_assumption = bool(fee_mode == "maker")

    if str(entry_mode) == "maker_only":
        bid_cents = _best_bid_cents_from_snapshot(snap, side=side)
        ask_cents, ask_dollars = _side_prices_from_snapshot(snap, side=side)
        if bid_cents is None or ask_cents is None or ask_dollars is None:
            log.info(
                "ENTRY_MAKER_BLOCK %s side=%s reason=missing_bid_or_ask bid=%s ask=%s",
                snap.ticker,
                side,
                str(bid_cents),
                str(ask_cents),
            )
            return None

        spread = int(ask_cents) - int(bid_cents)
        if max_entry_spread_cents is not None and spread > int(max_entry_spread_cents):
            log.info(
                "ENTRY_MAKER_BLOCK %s side=%s reason=spread_too_wide bid=%dc ask=%dc spread=%dc > max_spread=%dc",
                snap.ticker,
                side,
                int(bid_cents),
                int(ask_cents),
                int(spread),
                int(max_entry_spread_cents),
            )
            return None

        # Choose a maker-friendly limit price.
        entry_price_cents = min(int(bid_cents) + int(maker_improve_cents), int(ask_cents) - 1)
        if entry_price_cents <= 0 or entry_price_cents >= int(ask_cents):
            log.info(
                "ENTRY_MAKER_BLOCK %s side=%s reason=bad_entry_price bid=%dc ask=%dc improve=%dc entry=%dc",
                snap.ticker,
                side,
                int(bid_cents),
                int(ask_cents),
                int(maker_improve_cents),
                int(entry_price_cents),
            )
            return None

        entry_price_dollars = float(entry_price_cents) / 100.0
        time_in_force_entry = "good_till_canceled"
        # Maker-only entries are intended to rest on the book.
        maker_for_fee_assumption = True

        log.info(
            "ENTRY_MAKER %s side=%s bid=%dc ask=%dc spread=%dc entry=%dc tif=%s",
            snap.ticker,
            side,
            int(bid_cents),
            int(ask_cents),
            int(spread),
            int(entry_price_cents),
            str(time_in_force_entry),
        )
    else:
        # Default taker entry: buy at best ask with IoC (or caller-provided TIF).
        ask_cents, ask_dollars = _side_prices_from_snapshot(snap, side=side)
        if ask_cents is None or ask_dollars is None:
            log.warning(
                "ORDER_BLOCK %s side=%s reason=missing_best_ask best_yes_ask=%s best_no_ask=%s best_yes_bid=%s best_no_bid=%s",
                snap.ticker,
                side,
                str(snap.best_yes_ask),
                str(snap.best_no_ask),
                str(snap.best_yes_bid),
                str(snap.best_no_bid),
            )
            return None
        entry_price_cents = int(ask_cents)
        entry_price_dollars = float(ask_dollars)

    # Risk limits (inventory/exposure)
    if risk_limits is not None:
        tickers = risk_tickers if risk_tickers is not None else [snap.ticker]
        inv = inventory if inventory is not None else await fetch_inventory_summary(client=client, tickers=tickers)

        current_ticker = inv.per_ticker.get(snap.ticker)
        current_pos = int(current_ticker.position) if current_ticker is not None else 0
        current_abs_contracts = int(current_ticker.abs_contracts) if current_ticker is not None else abs(current_pos)
        current_exposure_usd = float(current_ticker.exposure_usd) if current_ticker is not None else 0.0

        # If we flattened first (or simulated flatten in dry-run), apply post-flatten baselines.
        if did_flatten and current_pos_override is not None:
            current_pos = int(current_pos_override)
            current_abs_contracts = abs(int(current_pos_override))
            current_exposure_usd = float(current_exposure_override or 0.0)

        delta_pos = _risk_delta_contracts(side=side, qty=int(qty))
        projected_pos = int(current_pos + delta_pos)
        projected_ticker_abs_contracts = int(abs(projected_pos))

        # Approximate incremental exposure by the trade's notional cost (price * qty).
        # For YES: price is YES ask; for NO: price is NO ask.
        incremental_exposure_usd = float(entry_price_dollars) * float(qty)
        projected_ticker_exposure_usd = float(current_exposure_usd) + float(incremental_exposure_usd)

        total_abs_base = int(base_total_abs_contracts) if base_total_abs_contracts is not None else int(inv.total_abs_contracts)
        total_exp_base = float(base_total_exposure_usd) if base_total_exposure_usd is not None else float(inv.total_exposure_usd)

        projected_total_abs_contracts = int(total_abs_base - current_abs_contracts + projected_ticker_abs_contracts)
        projected_total_exposure_usd = float(total_exp_base - current_exposure_usd + projected_ticker_exposure_usd)

        reason = _check_risk_limits(
            summary=inv,
            ticker=snap.ticker,
            current_position=current_pos,
            projected_position=projected_pos,
            projected_total_abs_contracts=projected_total_abs_contracts,
            projected_total_exposure_usd=projected_total_exposure_usd,
            projected_ticker_abs_contracts=projected_ticker_abs_contracts,
            projected_ticker_exposure_usd=projected_ticker_exposure_usd,
            limits=risk_limits,
        )
        if reason is not None:
            log.warning(
                "RISK_BLOCK %s side=%s qty=%d price=%dc (%s)",
                snap.ticker,
                side,
                int(qty),
                int(ask_cents),
                reason,
            )
            log.info(
                "ORDER_BLOCK %s side=%s reason_block=risk_limits",
                snap.ticker,
                str(side),
                extra={
                    "csv_fields": {
                        "run_id": run_id or "",
                        "poll_id": poll_id if poll_id is not None else "",
                        "ticker": snap.ticker,
                        "side": str(side),
                        "qty": int(qty),
                        "price_cents": int(entry_price_cents),
                        "price_dollars": float(entry_price_dollars),
                        "reason": f"risk_limits:{reason}",
                    }
                },
            )
            return None

    fee_total_usd = kalshi_expected_fee_usd(
        P=float(entry_price_dollars),
        C=int(qty),
        maker=bool(maker_for_fee_assumption),
    )
    fee_per_contract_usd = float(fee_total_usd) / float(max(1, int(qty)))

    ev_per_contract = _ev_after_fees_per_contract(edge, side=side)
    expected_profit_usd = float(ev_per_contract) * float(qty)

    expected_cost_usd = float(entry_price_dollars) * float(qty) + float(fee_total_usd)

    if bool(flip_reentry):
        log.info(
            "REENTRY_AFTER_FLATTEN %s from=%s to=%s qty=%d @ %dc EV_after_fees=%.4f hold_to_expiry=true",
            snap.ticker,
            "?" if flip_from_side is None else str(flip_from_side),
            str(side),
            int(qty),
            int(entry_price_cents),
            float(ev_per_contract),
        )

    log.info(
        "ORDER %s %s qty=%d @ %dc ($%.2f) mode=%s fee_assumption=%s spot=%s strike=%s EV_after_fees=%.3f%% exp_profit=$%.2f exp_fee_total=$%.2f exp_fee_per_contract=$%.4f exp_cost=$%.2f tte=%ss",
        snap.ticker,
        side,
        int(qty),
        int(entry_price_cents),
        float(entry_price_dollars),
        str(entry_mode),
        "maker" if bool(maker_for_fee_assumption) else "taker",
        _fmt_usd(snap.btc_spot_usd),
        _fmt_usd(snap.price_to_beat),
        float(ev_per_contract) * 100.0,
        float(expected_profit_usd),
        float(fee_total_usd),
        float(fee_per_contract_usd),
        float(expected_cost_usd),
        "?" if snap.seconds_to_expiry is None else str(int(snap.seconds_to_expiry)),
    )

    if dry_run:
        # Update rate limit timestamp for all entry modes
        _LAST_ENTRY_ORDER_TS_BY_TICKER[snap.ticker] = float(time.time())
        if str(entry_mode) == "maker_only":
            # Simulate cancel/replace semantics: ensure only one open entry order per ticker.
            _OPEN_ENTRY_ORDER_ID_BY_TICKER[snap.ticker] = str(
                f"dry-open-{snap.ticker}-{int(time.time()*1000)}"
            )
        if bool(hold_to_expiry_after_entry):
            _mark_hold_to_expiry(snap=snap)
            _PENDING_FLIP_REENTRY_BY_TICKER.pop(snap.ticker, None)
        return PlacedOrder(
            ticker=snap.ticker,
            side=side,
            contracts=int(qty),
            price_cents=int(entry_price_cents),
            price_dollars=float(entry_price_dollars),
            ev_after_fees_per_contract=float(ev_per_contract),
            expected_profit_usd=float(expected_profit_usd),
            expected_fee_usd=float(fee_total_usd),
            expected_cost_usd=float(expected_cost_usd),
            dry_run=True,
            response={"dry_run": True, "client_order_id": f"dry-{int(time.time()*1000)}"},
        )

    client_order_id = str(uuid.uuid4())

    # Update rate limit timestamp for all entry modes
    _LAST_ENTRY_ORDER_TS_BY_TICKER[snap.ticker] = float(time.time())

    if str(entry_mode) == "maker_only":
        # Cancel/replace: if we already have a tracked open maker entry order,
        # cancel it before placing a new one so we don't stack GTC orders.
        open_id = _OPEN_ENTRY_ORDER_ID_BY_TICKER.get(snap.ticker)
        if open_id:
            try:
                log.info(
                    "ENTRY_CANCEL_REPLACE %s old_order_id=%s new_entry=%dc",
                    snap.ticker,
                    str(open_id),
                    int(entry_price_cents),
                )
                await client.cancel_order(order_id=str(open_id))
                _OPEN_ENTRY_ORDER_ID_BY_TICKER.pop(snap.ticker, None)
            except Exception as e:
                # Safer to do nothing than to risk stacking multiple live orders.
                log.warning(
                    "ENTRY_CANCEL_REPLACE_BLOCK %s old_order_id=%s error=%s",
                    snap.ticker,
                    str(open_id),
                    e,
                )
                log.info(
                    "ORDER_BLOCK %s side=%s reason_block=entry_cancel_replace_failed",
                    snap.ticker,
                    str(side),
                    extra={
                        "csv_fields": {
                            "run_id": run_id or "",
                            "poll_id": poll_id if poll_id is not None else "",
                            "ticker": snap.ticker,
                            "side": str(side),
                            "reason": "entry_cancel_replace_failed",
                            "order_id": str(open_id),
                            "error": str(e),
                            "error_type": type(e).__name__,
                        }
                    },
                )
                return None

    # Structured attempt event right before hitting the Kalshi order endpoint.
    log.info(
        "ORDER_ATTEMPT %s side=%s qty=%d @ %dc mode=%s",
        snap.ticker,
        str(side),
        int(qty),
        int(entry_price_cents),
        str(entry_mode),
        extra={
            "csv_fields": {
                "run_id": run_id or "",
                "poll_id": poll_id if poll_id is not None else "",
                "ticker": snap.ticker,
                "side": str(side),
                "action": "buy",
                "qty": int(qty),
                "price_cents": int(entry_price_cents),
                "price_dollars": float(entry_price_dollars),
                "mode": str(entry_mode),
                "time_in_force": str(time_in_force_entry),
                "fee_assumption": "maker" if bool(maker_for_fee_assumption) else "taker",
                "fee_per_contract": float(fee_per_contract_usd),
                "spot_usd": snap.btc_spot_usd,
                "strike_usd": snap.price_to_beat,
                "strike_src": str(snap.price_to_beat_source or "?"),
                "tte_s": snap.seconds_to_expiry,
                "ev_after_fees": float(ev_per_contract),
            }
        },
    )

    if side == "YES":
        try:
            resp = await client.create_order(
                ticker=snap.ticker,
                side="yes",
                action="buy",
                count=int(qty),
                order_type="limit",
                yes_price=int(entry_price_cents),
                client_order_id=client_order_id,
                time_in_force=time_in_force_entry,
            )
        except Exception as e:
            if _is_trading_paused_http_error(e):
                _set_trading_paused_backoff(ticker=snap.ticker, reason="entry_trading_is_paused")
                return None
            raise
    else:
        try:
            resp = await client.create_order(
                ticker=snap.ticker,
                side="no",
                action="buy",
                count=int(qty),
                order_type="limit",
                no_price=int(entry_price_cents),
                client_order_id=client_order_id,
                time_in_force=time_in_force_entry,
            )
        except Exception as e:
            if _is_trading_paused_http_error(e):
                _set_trading_paused_backoff(ticker=snap.ticker, reason="entry_trading_is_paused")
                return None
            raise

    order_id = _extract_order_id(resp)
    log.info(
        "ORDER_ACK %s side=%s order_id=%s",
        snap.ticker,
        side,
        order_id or "?",
        extra={
            "csv_fields": {
                "run_id": run_id or "",
                "poll_id": poll_id if poll_id is not None else "",
                "ticker": snap.ticker,
                "side": str(side),
                "action": "buy",
                "qty": int(qty),
                "price_cents": int(entry_price_cents),
                "price_dollars": float(entry_price_dollars),
                "mode": str(entry_mode),
                "time_in_force": str(time_in_force_entry),
                "order_id": str(order_id or "?"),
            }
        },
    )

    # If this entry was a re-entry right after a flip-flatten, lock the ticker
    # to hold until expiry so we never flatten this new position.
    if bool(hold_to_expiry_after_entry):
        _mark_hold_to_expiry(snap=snap)
        _PENDING_FLIP_REENTRY_BY_TICKER.pop(snap.ticker, None)

    # Track open maker entry order id so future attempts cancel/replace instead of stacking.
    if str(entry_mode) == "maker_only" and order_id:
        _OPEN_ENTRY_ORDER_ID_BY_TICKER[snap.ticker] = str(order_id)
        _WORKING_ENTRY_ORDER_BY_TICKER[snap.ticker] = _WorkingEntryOrder(
            order_id=str(order_id),
            ts=float(time.time()),
            side=side,
        )
        log.info("ENTRY_WORKING %s mode=maker_only side=%s order_id=%s", snap.ticker, side, str(order_id))

    # Best-effort local state tracking for hold-time gating.
    # IMPORTANT: only update this after we have evidence the entry actually filled.
    if confirm_entry_fill and order_id:
        try:
            if await _order_has_any_fill(client=client, order_id=str(order_id)):
                _LAST_ENTRY_BY_TICKER[snap.ticker] = _LastEntry(ts=time.time(), side=side)

                # If we got any fill on a maker-only GTC entry, cancel the remaining
                # resting quantity to preserve the "single position / no add" intent.
                if str(entry_mode) == "maker_only":
                    _OPEN_ENTRY_ORDER_ID_BY_TICKER.pop(snap.ticker, None)
                    _WORKING_ENTRY_ORDER_BY_TICKER.pop(snap.ticker, None)
                    try:
                        await client.cancel_order(order_id=str(order_id))
                        log.info("ENTRY_MAKER_CANCEL_REMAINDER %s order_id=%s", snap.ticker, str(order_id))
                    except Exception as e:
                        log.warning(
                            "ENTRY_MAKER_CANCEL_REMAINDER_ERROR %s order_id=%s error=%s",
                            snap.ticker,
                            str(order_id),
                            e,
                        )
            else:
                log.info("ENTRY_NOT_FILLED %s side=%s order_id=%s", snap.ticker, side, order_id)
        except Exception as e:
            # If we can't confirm fills, err on the side of NOT setting the hold timer.
            log.warning("ENTRY_FILL_CHECK_ERROR %s side=%s order_id=%s error=%s", snap.ticker, side, order_id, e)
    elif not confirm_entry_fill:
        _LAST_ENTRY_BY_TICKER[snap.ticker] = _LastEntry(ts=time.time(), side=side)

    return PlacedOrder(
        ticker=snap.ticker,
        side=side,
        contracts=int(qty),
        price_cents=int(entry_price_cents),
        price_dollars=float(entry_price_dollars),
        ev_after_fees_per_contract=float(ev_per_contract),
        expected_profit_usd=float(expected_profit_usd),
        expected_fee_usd=float(fee_total_usd),
        expected_cost_usd=float(expected_cost_usd),
        dry_run=False,
        response=resp,
    )
