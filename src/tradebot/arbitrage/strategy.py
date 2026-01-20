"""
Delayed Update Arbitrage Strategy for Kalshi BTC 15m markets.

Core logic:
- Staleness filter: only trade when spot has moved meaningfully
- Edge check: fees + buffers
- Hybrid execution: maker-snipe then optional taker fallback
- Risk controls: tier sizing, caps, cooldown, time-stop + adverse stop
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from tradebot.arbitrage.spot import (
    PFairEstimator,
    SyntheticSpot,
    now_ms,
    clamp,
)
from tradebot.kalshi.client import KalshiClient


log = logging.getLogger(__name__)


# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for the delayed update arbitrage strategy."""
    
    # --- Staleness detection ---
    # Only trade when spot/p-fair has moved enough (indicates stale Kalshi book)
    stale_spot_bps: float = 6.0        # Spot moved threshold (bps) within window
    stale_pf: float = 0.03             # p_fair moved threshold
    stale_window_ms: int = 2500        # Lookback window to measure spot/p move

    # --- Edge thresholds ---
    taker_fee_per_contract: float = 0.02  # ~2 cents per contract in "prob" units
    maker_rebate_per_contract: float = 0.00  # Set if you have a rebate
    buffer_slippage: float = 0.01
    buffer_model: float = 0.015

    # --- Maker snipe ---
    maker_snipe_ms: int = 450           # Post-only order lifetime
    maker_improve_ticks: int = 1        # How aggressively to price inside the book
    tick_size: float = 0.01             # Kalshi pricing granularity in "prob"

    # --- Taker fallback ---
    taker_min_edge_extra: float = 0.02  # Require more edge for taker vs maker
    max_spread_to_taker: float = 0.20   # Don't cross if book is too wide

    # --- Risk controls ---
    max_contracts_per_ticker: int = 12
    max_open_orders_per_ticker: int = 2
    cooldown_after_stopouts: int = 2
    cooldown_ms: int = 45_000

    # --- Stops ---
    time_stop_ms: int = 6500            # Flatten if no improvement after this time
    adverse_stop_extra: float = 0.005   # Stop if edge collapses to within fee+this

    # --- Trading windows ---
    disable_last_seconds: int = 75      # Avoid last N seconds to expiry


# ---------------------------
# Data Types
# ---------------------------

@dataclass(frozen=True)
class BookTop:
    """Top of book for a market."""
    yes_bid: float
    yes_ask: float
    no_bid: float
    no_ask: float

    @property
    def yes_spread(self) -> float:
        return self.yes_ask - self.yes_bid

    @property
    def no_spread(self) -> float:
        return self.no_ask - self.no_bid


@dataclass(frozen=True)
class MarketInfo:
    """Information about a Kalshi market."""
    ticker: str
    strike: float
    expiry_ts_ms: int


@dataclass(frozen=True)
class OrderIntent:
    """Intent to place an order."""
    ticker: str
    side: str        # "buy" or "sell"
    outcome: str     # "yes" or "no"
    price_cents: int
    qty: int
    order_type: str  # "maker" or "taker"
    ttl_ms: int = 0  # Used for maker snipe


@dataclass
class OpenPosition:
    """Signed position in YES contracts: + means long YES, - means long NO."""
    signed_yes: int = 0


@dataclass
class LiveTradeState:
    """State for an active trade being managed."""
    entry_ts_ms: int
    intent: OrderIntent
    order_id: str
    filled_qty: int = 0
    best_edge_seen: float = 0.0


# ---------------------------
# Risk Manager
# ---------------------------

class RiskManager:
    """Manages risk controls: cooldowns, stopouts, position sizing."""
    
    def __init__(self, cfg: StrategyConfig) -> None:
        self.cfg = cfg
        self.cooldown_until_ms: Dict[str, int] = {}
        self.stopouts_recent: Dict[str, int] = {}

    def in_cooldown(self, ticker: str) -> bool:
        return now_ms() < self.cooldown_until_ms.get(ticker, 0)

    def register_stopout(self, ticker: str) -> None:
        n = self.stopouts_recent.get(ticker, 0) + 1
        self.stopouts_recent[ticker] = n
        if n >= self.cfg.cooldown_after_stopouts:
            self.cooldown_until_ms[ticker] = now_ms() + self.cfg.cooldown_ms
            self.stopouts_recent[ticker] = 0  # Reset after cooldown
            log.info("COOLDOWN_START %s until=%d", ticker, self.cooldown_until_ms[ticker])

    def size_from_edge(self, net_edge: float, remaining_depth: int) -> int:
        """
        Edge-tier sizing.
        
        net_edge is already AFTER fees (edge - fee).
        """
        if net_edge < 0.015:
            return 0
        if net_edge < 0.03:
            return 1
        if net_edge < 0.06:
            return min(3, remaining_depth)
        return min(8, remaining_depth)


# ---------------------------
# Execution Helpers
# ---------------------------

def round_to_tick(px: float, tick: float) -> float:
    """Round price to nearest tick, clamped to valid range."""
    return clamp(round(px / tick) * tick, 0.01, 0.99)


def prob_to_cents(prob: float) -> int:
    """Convert probability (0-1) to Kalshi cents (1-99)."""
    return int(clamp(round(prob * 100), 1, 99))


def cents_to_prob(cents: int) -> float:
    """Convert Kalshi cents (1-99) to probability (0-1)."""
    return clamp(cents / 100.0, 0.01, 0.99)


def maker_price_for_buy(best_bid: float, best_ask: float, improve_ticks: int, tick: float) -> float:
    """Calculate maker price for a buy order (improve inside spread)."""
    px = best_bid + improve_ticks * tick
    if px >= best_ask:
        px = best_bid  # Don't cross on maker
    return round_to_tick(px, tick)


# ---------------------------
# Strategy Core
# ---------------------------

class DelayedUpdateArbStrategy:
    """
    Delayed Update Arbitrage Strategy.
    
    Exploits the fact that Kalshi prices can lag behind spot moves.
    When spot moves significantly (staleness detected), check if Kalshi
    prices offer edge vs fair value computed from spot.
    
    If dry_run=True, logs order intents but does NOT call kalshi.create_order().
    If verbose=True, logs decision snapshots when near-trigger conditions are met.
    """
    
    # Spot freshness threshold (ms)
    SPOT_MAX_AGE_MS: int = 1500
    
    def __init__(
        self,
        cfg: StrategyConfig,
        kalshi: KalshiClient,
        p_est: PFairEstimator,
        risk: RiskManager,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> None:
        self.cfg = cfg
        self.kalshi = kalshi
        self.p_est = p_est
        self.risk = risk
        self.dry_run = dry_run
        self.verbose = verbose

        # p-fair history for staleness calculation
        self.p_history: Dict[str, List[Tuple[int, float]]] = {}
        
        # Order tracking
        self.open_orders: Dict[str, List[str]] = {}
        self.live_trade: Dict[str, LiveTradeState] = {}

    def _record_p(self, ticker: str, p: float) -> None:
        """Record p-fair value for staleness tracking."""
        arr = self.p_history.setdefault(ticker, [])
        arr.append((now_ms(), p))
        cutoff = now_ms() - 10_000
        while arr and arr[0][0] < cutoff:
            arr.pop(0)

    def _pf_move_over(self, ticker: str, window_ms: int) -> Optional[float]:
        """Calculate p-fair movement over window."""
        arr = self.p_history.get(ticker, [])
        if not arr:
            return None
        t = now_ms()
        target = t - window_ms
        past = None
        for ts, p in reversed(arr):
            if ts <= target:
                past = p
                break
        if past is None:
            return None
        return abs(arr[-1][1] - past)

    def _effective_fee(self, order_type: str) -> float:
        """Get effective fee for order type."""
        if order_type == "taker":
            return self.cfg.taker_fee_per_contract
        # Maker rebate reduces fee
        return max(0.0, self.cfg.taker_fee_per_contract - self.cfg.maker_rebate_per_contract)

    def _edge_threshold(self, order_type: str) -> float:
        """Get minimum edge threshold for order type."""
        fee = self._effective_fee(order_type)
        base = fee + self.cfg.buffer_slippage + self.cfg.buffer_model
        if order_type == "taker":
            base += self.cfg.taker_min_edge_extra
        return base

    def _is_trade_window(self, m: MarketInfo) -> bool:
        """Check if we're within the trading window."""
        rem_s = (m.expiry_ts_ms - now_ms()) / 1000.0
        return rem_s > self.cfg.disable_last_seconds

    async def _get_book_top(self, ticker: str) -> Optional[BookTop]:
        """Fetch top of book from Kalshi using the market endpoint.
        
        The market endpoint provides pre-computed BBO (best bid/offer):
        - yes_bid, yes_ask: best bid/ask for YES side
        - no_bid, no_ask: best bid/ask for NO side
        
        This is more reliable than computing from orderbook depth.
        """
        try:
            resp = await self.kalshi.get_market(ticker)
            mkt = resp.get("market", {})
            
            # BBO is provided directly in cents
            yes_bid_cents = mkt.get("yes_bid", 1)
            yes_ask_cents = mkt.get("yes_ask", 99)
            no_bid_cents = mkt.get("no_bid", 1)
            no_ask_cents = mkt.get("no_ask", 99)
            
            # Handle None values
            if yes_bid_cents is None:
                yes_bid_cents = 1
            if yes_ask_cents is None:
                yes_ask_cents = 99
            if no_bid_cents is None:
                no_bid_cents = 1
            if no_ask_cents is None:
                no_ask_cents = 99
            
            # Convert to prob (0-1 scale)
            yes_bid = cents_to_prob(max(1, yes_bid_cents))
            yes_ask = cents_to_prob(min(99, yes_ask_cents))
            no_bid = cents_to_prob(max(1, no_bid_cents))
            no_ask = cents_to_prob(min(99, no_ask_cents))
            
            return BookTop(yes_bid=yes_bid, yes_ask=yes_ask, no_bid=no_bid, no_ask=no_ask)
        except Exception as e:
            log.warning("BOOK_FETCH_ERROR %s error=%s", ticker, e)
            return None

    async def _get_position(self, ticker: str) -> OpenPosition:
        """Fetch current position from Kalshi."""
        try:
            resp = await self.kalshi.get_positions(ticker=ticker)
            positions = resp.get("market_positions", [])
            for pos in positions:
                if pos.get("ticker") == ticker:
                    return OpenPosition(signed_yes=int(pos.get("position", 0)))
            return OpenPosition(signed_yes=0)
        except Exception as e:
            log.warning("POSITION_FETCH_ERROR %s error=%s", ticker, e)
            return OpenPosition(signed_yes=0)

    def _log_decision_snapshot(
        self,
        m: MarketInfo,
        book: BookTop,
        p: float,
        yes_edge: float,
        no_edge: float,
        spot_mid: float | None,
        spot_age_ms: int | None,
        spot_move_bps: float | None,
        pf_move: float | None,
        stale: bool,
        decision: str,
    ) -> None:
        """Log a compact, grep-friendly decision snapshot."""
        remaining_s = max(0.0, (m.expiry_ts_ms - now_ms()) / 1000.0)
        maker_thresh = self._edge_threshold("maker")
        taker_thresh = self._edge_threshold("taker")
        
        log.info(
            "DECISION ticker=%s strike=%.0f rem_s=%.1f "
            "spot=%.2f spot_age=%s spot_bps=%s pf_move=%s "
            "p_fair=%.4f "
            "yes_bid=%.2f yes_ask=%.2f no_bid=%.2f no_ask=%.2f "
            "yes_edge=%.4f no_edge=%.4f "
            "maker_th=%.4f taker_th=%.4f stale=%s "
            "decision=%s",
            m.ticker, m.strike, remaining_s,
            spot_mid or 0.0,
            spot_age_ms if spot_age_ms is not None else "-",
            f"{spot_move_bps:.1f}" if spot_move_bps is not None else "-",
            f"{pf_move:.4f}" if pf_move is not None else "-",
            p,
            book.yes_bid, book.yes_ask, book.no_bid, book.no_ask,
            yes_edge, no_edge,
            maker_thresh, taker_thresh, stale,
            decision,
        )

    async def tick_market(self, m: MarketInfo) -> None:
        """Process one tick for a market."""
        if self.risk.in_cooldown(m.ticker):
            return
        if not self._is_trade_window(m):
            return
        if len(self.open_orders.get(m.ticker, [])) >= self.cfg.max_open_orders_per_ticker:
            return
        if m.ticker in self.live_trade:
            await self._manage_open_trade(m)
            return

        # Spot freshness guard
        spot_age_ms = self.p_est.spot.age_ms()
        if spot_age_ms is not None and spot_age_ms > self.SPOT_MAX_AGE_MS:
            log.warning("SPOT_STALE_SKIP ticker=%s age_ms=%d", m.ticker, spot_age_ms)
            return

        # Compute p-fair
        remaining_s = max(0.0, (m.expiry_ts_ms - now_ms()) / 1000.0)
        p = self.p_est.p_fair(strike=m.strike, remaining_s=remaining_s)
        if p is None:
            return
        self._record_p(m.ticker, p)

        # Read Kalshi book
        book = await self._get_book_top(m.ticker)
        if book is None:
            return

        # Edge calc (raw vs ask)
        yes_edge = p - book.yes_ask
        no_edge = (1.0 - p) - book.no_ask

        # Staleness checks: only trade if spot/p-fair has moved
        pf_move = self._pf_move_over(m.ticker, self.cfg.stale_window_ms)
        spot_move_bps = self.p_est.spot.spot_move_bps_over(self.cfg.stale_window_ms)
        stale = (
            (pf_move is not None and pf_move >= self.cfg.stale_pf)
            or (spot_move_bps is not None and spot_move_bps >= self.cfg.stale_spot_bps)
        )
        
        # Compute thresholds and best edge
        maker_thresh = self._edge_threshold("maker")
        best_edge = max(yes_edge, no_edge)
        near_trigger = best_edge >= (maker_thresh - 0.01)
        
        # Determine decision
        decision = "NONE"
        if stale and best_edge >= maker_thresh:
            decision = f"MAKER_{'YES' if yes_edge >= no_edge else 'NO'}"
        
        # Log decision snapshot if verbose and (stale + near trigger) OR always if verbose for debugging
        should_log = self.verbose and (stale and near_trigger)
        # Also log periodically even if no trigger for debugging (every ~10 ticks when verbose)
        if self.verbose and not should_log:
            import random
            should_log = random.random() < 1.0  # 100% sample when not triggered
        
        if should_log:
            spot_mid = self.p_est.spot.mid()
            self._log_decision_snapshot(
                m, book, p, yes_edge, no_edge,
                spot_mid, spot_age_ms, spot_move_bps, pf_move,
                stale, decision,
            )
        
        if not stale:
            return

        # Decide side based on edge
        if yes_edge >= no_edge:
            await self._maybe_enter(m, "yes", yes_edge, book)
        else:
            await self._maybe_enter(m, "no", no_edge, book)

    async def _maybe_enter(
        self, m: MarketInfo, outcome: str, edge: float, book: BookTop
    ) -> None:
        """Attempt to enter a position if edge is sufficient."""
        # Risk: cap per ticker
        pos = await self._get_position(m.ticker)
        remaining = self.cfg.max_contracts_per_ticker - abs(pos.signed_yes)
        if remaining <= 0:
            return

        maker_thresh = self._edge_threshold("maker")
        taker_thresh = self._edge_threshold("taker")

        # Size for maker attempt
        net_edge_for_size = edge - self._effective_fee("maker")
        qty = self.risk.size_from_edge(net_edge_for_size, remaining)
        if qty <= 0:
            return

        # Maker attempt first
        if edge >= maker_thresh:
            intent = self._build_maker_intent(m, outcome, qty, book)
            await self._place_and_track(m, intent)
            return

        # Taker fallback only if truly strong edge
        if edge >= taker_thresh:
            intent = self._build_taker_intent(m, outcome, qty, book)
            if intent.qty > 0:
                await self._place_and_track(m, intent)

    def _build_maker_intent(
        self, m: MarketInfo, outcome: str, qty: int, book: BookTop
    ) -> OrderIntent:
        """Build a maker (post-only) order intent."""
        if outcome == "yes":
            price = maker_price_for_buy(
                book.yes_bid, book.yes_ask,
                self.cfg.maker_improve_ticks, self.cfg.tick_size
            )
        else:
            price = maker_price_for_buy(
                book.no_bid, book.no_ask,
                self.cfg.maker_improve_ticks, self.cfg.tick_size
            )

        return OrderIntent(
            ticker=m.ticker,
            side="buy",
            outcome=outcome,
            price_cents=prob_to_cents(price),
            qty=qty,
            order_type="maker",
            ttl_ms=self.cfg.maker_snipe_ms,
        )

    def _build_taker_intent(
        self, m: MarketInfo, outcome: str, qty: int, book: BookTop
    ) -> OrderIntent:
        """Build a taker (cross the spread) order intent."""
        if outcome == "yes":
            spread = book.yes_spread
            if spread > self.cfg.max_spread_to_taker:
                qty = 0
            price = book.yes_ask
        else:
            spread = book.no_spread
            if spread > self.cfg.max_spread_to_taker:
                qty = 0
            price = book.no_ask

        return OrderIntent(
            ticker=m.ticker,
            side="buy",
            outcome=outcome,
            price_cents=prob_to_cents(price),
            qty=max(0, qty),
            order_type="taker",
            ttl_ms=0,
        )

    async def _place_and_track(self, m: MarketInfo, intent: OrderIntent) -> None:
        """Place order and set up tracking.
        
        If dry_run=True, logs the intent but does NOT call kalshi.create_order().
        """
        if intent.qty <= 0:
            return

        # Dry run mode: log intent only (with all fields for debugging)
        if self.dry_run:
            spot_mid = self.p_est.spot.mid()
            log.info(
                "DRY_RUN_ORDER ticker=%s outcome=%s action=%s qty=%d price_c=%d "
                "type=%s ttl_ms=%d spot=%.2f",
                intent.ticker, intent.outcome, intent.side,
                intent.qty, intent.price_cents, intent.order_type,
                intent.ttl_ms, spot_mid or 0.0,
            )
            return

        try:
            # Determine price parameter
            yes_price = intent.price_cents if intent.outcome == "yes" else None
            no_price = intent.price_cents if intent.outcome == "no" else None
            
            resp = await self.kalshi.create_order(
                ticker=intent.ticker,
                side=intent.outcome,
                action=intent.side,
                count=intent.qty,
                order_type="limit",
                yes_price=yes_price,
                no_price=no_price,
                post_only=(intent.order_type == "maker"),
                time_in_force="immediate_or_cancel" if intent.order_type == "taker" else "good_till_canceled",
            )
            
            order_id = resp.get("order", {}).get("order_id") or resp.get("order_id", "")
            if not order_id:
                log.warning("ORDER_NO_ID %s", intent.ticker)
                return
                
            log.info(
                "ORDER_PLACED %s %s %s qty=%d @ %dc order_id=%s",
                intent.ticker, intent.outcome, intent.side,
                intent.qty, intent.price_cents, order_id
            )
            
            self.open_orders.setdefault(intent.ticker, []).append(order_id)
            self.live_trade[intent.ticker] = LiveTradeState(
                entry_ts_ms=now_ms(),
                intent=intent,
                order_id=order_id,
            )

            # Schedule cancel for maker orders
            if intent.order_type == "maker" and intent.ttl_ms > 0:
                asyncio.create_task(self._cancel_after(intent.ticker, order_id, intent.ttl_ms))
                
        except Exception as e:
            log.warning("ORDER_PLACE_ERROR %s error=%s", intent.ticker, e)

    async def _cancel_after(self, ticker: str, order_id: str, ttl_ms: int) -> None:
        """Cancel order after TTL expires.
        
        Also cleans up tracking state if the order had 0 fills to avoid
        "stuck" state if polling stalls.
        """
        await asyncio.sleep(ttl_ms / 1000.0)
        try:
            await self.kalshi.cancel_order(order_id)
            log.info("ORDER_CANCEL_TTL %s order_id=%s", ticker, order_id)
        except Exception:
            pass  # Order may already be filled/cancelled
        
        # Immediately clean up if still the active trade with 0 fills
        # This prevents stuck state if polling is slow
        st = self.live_trade.get(ticker)
        if st is not None and st.order_id == order_id and st.filled_qty == 0:
            arr = self.open_orders.get(ticker, [])
            self.open_orders[ticker] = [oid for oid in arr if oid != order_id]
            self.live_trade.pop(ticker, None)
            log.debug("CLEANUP_AFTER_TTL %s order_id=%s", ticker, order_id)

    async def _manage_open_trade(self, m: MarketInfo) -> None:
        """Manage an open trade: check fills, stops."""
        st = self.live_trade.get(m.ticker)
        if st is None:
            return

        order_ids = self.open_orders.get(m.ticker, [])
        if not order_ids:
            self.live_trade.pop(m.ticker, None)
            return

        # Check fill status
        try:
            fills_resp = await self.kalshi.get_fills(order_id=st.order_id, limit=10)
            fills = fills_resp.get("fills", [])
            # Explicitly filter by order_id (belt and suspenders) and use abs(count)
            # to handle any signed/unsigned ambiguity in the API response
            filled_qty = sum(
                abs(int(f.get("count", 0)))
                for f in fills
                if f.get("order_id") == st.order_id
            )
            st.filled_qty = filled_qty
        except Exception as e:
            log.warning("FILL_CHECK_ERROR %s error=%s", m.ticker, e)
            return

        # If maker TTL elapsed and not filled, cleanup and maybe taker fallback
        if st.filled_qty == 0 and st.intent.order_type == "maker":
            age_ms = now_ms() - st.entry_ts_ms
            if age_ms >= st.intent.ttl_ms:
                await self._cleanup_trade(m.ticker, st.order_id)
                await self._maybe_taker_fallback(m)
                return

        # If got fill, manage stops
        if st.filled_qty > 0:
            await self._manage_stops_and_exit(m, st)

    async def _maybe_taker_fallback(self, m: MarketInfo) -> None:
        """Attempt taker entry after maker failed."""
        if self.risk.in_cooldown(m.ticker):
            return
        if not self._is_trade_window(m):
            return

        remaining_s = max(0.0, (m.expiry_ts_ms - now_ms()) / 1000.0)
        p = self.p_est.p_fair(m.strike, remaining_s)
        if p is None:
            return

        book = await self._get_book_top(m.ticker)
        if book is None:
            return
            
        yes_edge = p - book.yes_ask
        no_edge = (1 - p) - book.no_ask
        
        if yes_edge >= no_edge and yes_edge >= self._edge_threshold("taker"):
            intent = self._build_taker_intent(m, "yes", qty=1, book=book)
            if intent.qty > 0:
                await self._place_and_track(m, intent)
        elif no_edge > yes_edge and no_edge >= self._edge_threshold("taker"):
            intent = self._build_taker_intent(m, "no", qty=1, book=book)
            if intent.qty > 0:
                await self._place_and_track(m, intent)

    async def _manage_stops_and_exit(self, m: MarketInfo, st: LiveTradeState) -> None:
        """Check stop conditions and exit if triggered."""
        remaining_s = max(0.0, (m.expiry_ts_ms - now_ms()) / 1000.0)
        p = self.p_est.p_fair(m.strike, remaining_s)
        if p is None:
            return
            
        book = await self._get_book_top(m.ticker)
        if book is None:
            return

        # Calculate exit edge (what we'd get if we sold now)
        if st.intent.outcome == "yes":
            exit_edge = p - book.yes_bid
        else:
            exit_edge = (1 - p) - book.no_bid

        st.best_edge_seen = max(st.best_edge_seen, exit_edge)

        # Time stop
        age_ms = now_ms() - st.entry_ts_ms
        if age_ms >= self.cfg.time_stop_ms:
            log.info("TIME_STOP %s age_ms=%d", m.ticker, age_ms)
            await self._flatten_position(m, st)
            return

        # Adverse stop: edge collapsed
        fee = self._effective_fee("taker")
        if exit_edge < fee + self.cfg.adverse_stop_extra:
            log.info("ADVERSE_STOP %s exit_edge=%.4f threshold=%.4f", m.ticker, exit_edge, fee + self.cfg.adverse_stop_extra)
            await self._flatten_position(m, st)
            return

    async def _flatten_position(self, m: MarketInfo, st: LiveTradeState) -> None:
        """Flatten the position by selling."""
        qty = st.filled_qty
        if qty <= 0:
            await self._cleanup_trade(m.ticker, st.order_id)
            return

        book = await self._get_book_top(m.ticker)
        if book is None:
            await self._cleanup_trade(m.ticker, st.order_id)
            return
            
        if st.intent.outcome == "yes":
            price_cents = prob_to_cents(book.yes_bid)
        else:
            price_cents = prob_to_cents(book.no_bid)

        try:
            yes_price = price_cents if st.intent.outcome == "yes" else None
            no_price = price_cents if st.intent.outcome == "no" else None
            
            await self.kalshi.create_order(
                ticker=m.ticker,
                side=st.intent.outcome,
                action="sell",
                count=qty,
                order_type="limit",
                yes_price=yes_price,
                no_price=no_price,
                time_in_force="immediate_or_cancel",
            )
            log.info("FLATTEN %s %s qty=%d @ %dc", m.ticker, st.intent.outcome, qty, price_cents)
        except Exception as e:
            log.warning("FLATTEN_ERROR %s error=%s", m.ticker, e)

        self.risk.register_stopout(m.ticker)
        await self._cleanup_trade(m.ticker, st.order_id)

    async def _cleanup_trade(self, ticker: str, order_id: str) -> None:
        """Clean up trade tracking state."""
        try:
            await self.kalshi.cancel_order(order_id)
        except Exception:
            pass  # May already be filled/cancelled

        arr = self.open_orders.get(ticker, [])
        self.open_orders[ticker] = [oid for oid in arr if oid != order_id]
        self.live_trade.pop(ticker, None)
