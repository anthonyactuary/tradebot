from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import logging
import math
import os
import time
from dataclasses import dataclass, field
from typing import Any

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_best_prices
from tradebot.strategies.crypto_mm_15m import (
    FairProbabilityModel,
    RiskLimits,
    SpreadPolicy,
    VolatilityEstimator,
)


# ----------------------------
# Easy-to-edit defaults
# ----------------------------

# If you want to "just run the file" without passing flags, edit these.
DEFAULT_ASSETS = "BTC"
DEFAULT_HORIZON_MINUTES = 60
DEFAULT_PER_ASSET = 2
DEFAULT_POLL_SECONDS = 3.0
DEFAULT_DURATION_MINUTES = 120.0  # 2 hours
DEFAULT_BANKROLL_DOLLARS = 100.0
DEFAULT_BASE_ORDER_DOLLARS = 1.5
DEFAULT_OUT_DIR = "runs"

# Conservative knobs to reduce inventory drift / adverse selection.
# These are in CONTRACTS (each contract notionally $1 at settlement).
DEFAULT_MAX_ABS_POS_PER_MARKET = 10
DEFAULT_MAX_ABS_POS_TOTAL = 25

# Quote one tick behind top-of-book to reduce getting picked off.
DEFAULT_BEHIND_TOP_CENTS = 1

# At end of run, close any remaining inventory at current bid/ask.
DEFAULT_FLATTEN_AT_END = True

# --- Time-decay / gamma-aware settings ---
# Stop quoting earlier (high gamma risk near expiry)
DEFAULT_STOP_QUOTING_SECONDS = 120  # was 60; now 2 min before expiry

# Spread widens as expiry approaches (time-decay multiplier)
# At 10+ min out: 1.0x spread; at 2 min out: up to 2.5x spread
DEFAULT_TIME_SPREAD_MULTIPLIER_MAX = 2.5

# Position cap shrinks near expiry (don't build inventory late)
# At 10+ min: full cap; at 2 min: only 30% of cap allowed
DEFAULT_TIME_POS_CAP_MIN_PCT = 0.30

# --- Inventory skew (exit positions faster) ---
# Skew per contract of inventory: shift quote by this many cents
DEFAULT_INVENTORY_SKEW_CENTS_PER_CONTRACT = 1

# --- Adverse selection detection ---
# Track recent fills; if one-sided, widen that side
DEFAULT_ADVERSE_SELECTION_LOOKBACK = 10  # last N fills
DEFAULT_ADVERSE_SELECTION_WIDEN_CENTS = 2  # widen by this if detected

# --- Auto-flatten trigger ---
# If unrealized loss on a single ticker exceeds this, flatten immediately
DEFAULT_AUTO_FLATTEN_LOSS_DOLLARS = 3.0

# Kalshi fee schedule (as provided):
# taker_fee = ceil(0.07 * C * P * (1-P)) to the next cent
# maker_fee = ceil(0.0175 * C * P * (1-P)) to the next cent
DEFAULT_TAKER_FEE_RATE = 0.07
DEFAULT_MAKER_FEE_RATE = 0.0175


def _utcnow() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _to_dt(v: Any) -> datetime.datetime | None:
    if not v:
        return None
    if isinstance(v, datetime.datetime):
        return v
    if isinstance(v, str):
        try:
            return datetime.datetime.fromisoformat(v.replace("Z", "+00:00"))
        except Exception:
            return None
    return None


def _discover_series_ticker(asset: str) -> str:
    return f"KX{asset.upper()}15M"


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round_price_cents(p: float) -> int:
    cents = int(round(p * 100.0))
    return max(1, min(99, cents))


def _round_up_to_cent(x: float) -> float:
    if x <= 0:
        return 0.0
    # Avoid float edge cases where x*100 is e.g. 1.00000000002.
    return math.ceil((x * 100.0) - 1e-12) / 100.0


def _fee_dollars(*, price_cents: int, count: int, maker: bool) -> float:
    p = _clamp(price_cents / 100.0, 0.0, 1.0)
    c = max(0, int(count))
    if c == 0 or p <= 0.0 or p >= 1.0:
        return 0.0
    rate = DEFAULT_MAKER_FEE_RATE if maker else DEFAULT_TAKER_FEE_RATE
    fee = rate * c * p * (1.0 - p)
    return float(_round_up_to_cent(fee))


def _max_loss_yes_buy(*, yes_price_cents: int, count: int) -> float:
    return (yes_price_cents / 100.0) * count


def _max_loss_yes_sell(*, yes_price_cents: int, count: int) -> float:
    return ((100 - yes_price_cents) / 100.0) * count


@dataclass
class PaperOrder:
    order_id: str
    ticker: str
    action: str  # buy|sell (YES)
    price_cents: int
    count: int
    placed_ts: datetime.datetime


@dataclass
class PaperPortfolio:
    starting_cash: float
    cash: float

    fees_paid: float = 0.0

    # YES inventory per ticker; negative means short
    yes_pos: dict[str, int] = field(default_factory=dict)

    # last mid price (0..1) per ticker for mark-to-market
    last_mid: dict[str, float] = field(default_factory=dict)

    def mark_mid(self, *, ticker: str, mid: float) -> None:
        self.last_mid[ticker] = float(_clamp(mid, 0.0, 1.0))

    def equity(self) -> float:
        inv_value = 0.0
        for ticker, qty in self.yes_pos.items():
            mid = self.last_mid.get(ticker)
            if mid is None:
                continue
            inv_value += qty * mid
        return float(self.cash + inv_value)

    def pnl(self) -> float:
        return float(self.equity() - self.starting_cash)


@dataclass
class PaperEngine:
    limits: RiskLimits

    # Worst-case risk of *resting* orders (not positions)
    open_order_risk: dict[str, float] = field(default_factory=dict)  # order_id -> risk

    # Active resting orders
    orders_by_ticker: dict[str, list[PaperOrder]] = field(default_factory=dict)

    def _ticker_open_risk(self, ticker: str) -> float:
        orders = self.orders_by_ticker.get(ticker) or []
        return float(sum(self.open_order_risk.get(o.order_id, 0.0) for o in orders))

    def total_open_risk(self) -> float:
        return float(sum(self.open_order_risk.values()))

    def can_place(self, *, ticker: str, add_risk: float) -> bool:
        if add_risk <= 0:
            return True
        if self._ticker_open_risk(ticker) + add_risk > self.limits.max_exposure_per_market_dollars:
            return False
        if self.total_open_risk() + add_risk > self.limits.max_exposure_total_dollars:
            return False
        return True

    def cancel_all(self, *, ticker: str) -> None:
        for o in self.orders_by_ticker.get(ticker) or []:
            self.open_order_risk.pop(o.order_id, None)
        self.orders_by_ticker.pop(ticker, None)

    def place(self, order: PaperOrder, *, risk_dollars: float) -> None:
        self.orders_by_ticker.setdefault(order.ticker, []).append(order)
        self.open_order_risk[order.order_id] = float(max(0.0, risk_dollars))

    def remove(self, order: PaperOrder) -> None:
        if order.ticker in self.orders_by_ticker:
            self.orders_by_ticker[order.ticker] = [o for o in self.orders_by_ticker[order.ticker] if o.order_id != order.order_id]
            if not self.orders_by_ticker[order.ticker]:
                self.orders_by_ticker.pop(order.ticker, None)
        self.open_order_risk.pop(order.order_id, None)


async def pick_active_markets(
    *,
    client: KalshiClient,
    assets: list[str],
    horizon_minutes: int,
    per_asset: int,
) -> dict[str, list[dict[str, Any]]]:
    now = _utcnow()
    horizon = datetime.timedelta(minutes=max(1, horizon_minutes))

    out: dict[str, list[dict[str, Any]]] = {}
    for asset in assets:
        series_ticker = _discover_series_ticker(asset)
        page = await client.get_markets_page(
            limit=1000,
            status="open",
            series_ticker=series_ticker,
            mve_filter="exclude",
        )
        markets = list(page.get("markets") or [])

        keep: list[tuple[datetime.datetime, dict[str, Any]]] = []
        for m in markets:
            exp = _to_dt(m.get("expected_expiration_time"))
            if exp is None:
                continue
            if exp < now:
                continue
            if exp - now > horizon:
                continue
            keep.append((exp, m))

        keep.sort(key=lambda t: t[0])
        out[asset] = [m for _, m in keep[: max(1, per_asset)]]

    return out


def _choose_size(*, limits: RiskLimits, volatility: float, time_factor: float = 1.0) -> int:
    """Choose order size, reduced by volatility and time-to-expiry.
    
    time_factor: 1.0 = far from expiry, 0.0 = at expiry
    """
    max_single = limits.bankroll_dollars * limits.max_single_order_pct_bankroll
    base = min(limits.base_order_dollars, max_single)

    if volatility >= 0.02:
        base *= 0.5
    if volatility >= 0.05:
        base *= 0.5

    # Reduce size near expiry (don't build inventory late)
    base *= max(0.3, time_factor)

    return max(1, int(round(base)))


def _compute_time_factor(*, seconds_to_expiry: int, stop_quoting_seconds: int = 120) -> float:
    """Returns 1.0 far from expiry, tapering to 0.0 at stop_quoting_seconds.
    
    Used to scale spread, position caps, and size.
    """
    if seconds_to_expiry <= stop_quoting_seconds:
        return 0.0
    # Linear taper from 10 min (600s) to stop_quoting_seconds
    full_time = 600  # 10 minutes
    if seconds_to_expiry >= full_time:
        return 1.0
    return (seconds_to_expiry - stop_quoting_seconds) / (full_time - stop_quoting_seconds)


def _time_adjusted_spread(*, base_spread: float, time_factor: float, max_multiplier: float = 2.5) -> float:
    """Widen spread as expiry approaches (gamma risk increases)."""
    # At time_factor=1.0: multiplier=1.0; at time_factor=0.0: multiplier=max_multiplier
    multiplier = 1.0 + (max_multiplier - 1.0) * (1.0 - time_factor)
    return base_spread * multiplier


def _time_adjusted_pos_cap(*, base_cap: int, time_factor: float, min_pct: float = 0.30) -> int:
    """Shrink position cap as expiry approaches."""
    # At time_factor=1.0: full cap; at time_factor=0.0: min_pct of cap
    scale = min_pct + (1.0 - min_pct) * time_factor
    return max(1, int(round(base_cap * scale)))


def _inventory_skew(
    *,
    p_fair: float,
    spread: float,
    inventory: int,
    skew_cents_per_contract: int = 1,
    time_factor: float = 1.0,
) -> tuple[float, float]:
    """Skew bid/ask to exit inventory faster. More aggressive near expiry.
    
    If long (inventory > 0): lower the ask to sell faster, raise the bid less.
    If short (inventory < 0): raise the bid to buy faster, lower the ask less.
    """
    # Urgency increases as time_factor decreases
    urgency = 1.0 + 2.0 * (1.0 - time_factor)  # 1x to 3x
    skew_per_contract = (skew_cents_per_contract / 100.0) * urgency
    
    total_skew = inventory * skew_per_contract
    
    # Positive inventory => lower both bid and ask (want to sell)
    # Negative inventory => raise both bid and ask (want to buy)
    bid = p_fair - (spread / 2.0) - total_skew
    ask = p_fair + (spread / 2.0) - total_skew
    
    return bid, ask


@dataclass
class AdverseSelectionTracker:
    """Track recent fills to detect one-sided adverse selection."""
    lookback: int = 10
    _recent_fills: list[str] = field(default_factory=list)  # 'buy' or 'sell'
    
    def record_fill(self, side: str) -> None:
        self._recent_fills.append(side.lower())
        if len(self._recent_fills) > self.lookback:
            self._recent_fills.pop(0)
    
    def get_skew_recommendation(self) -> tuple[int, int]:
        """Returns (bid_widen_cents, ask_widen_cents) based on fill imbalance."""
        if len(self._recent_fills) < 3:
            return (0, 0)
        
        buys = sum(1 for f in self._recent_fills if f == 'buy')
        sells = len(self._recent_fills) - buys
        
        # If heavily one-sided, widen that side
        ratio = buys / max(1, sells) if sells > 0 else (buys if buys > 0 else 1)
        
        if ratio >= 2.0:  # 2x more buys than sells => we're getting picked off on buys
            return (DEFAULT_ADVERSE_SELECTION_WIDEN_CENTS, 0)
        elif ratio <= 0.5:  # 2x more sells than buys
            return (0, DEFAULT_ADVERSE_SELECTION_WIDEN_CENTS)
        return (0, 0)


def _apply_pressure(*, limits: RiskLimits, p_fair: float, spread: float, open_risk: float) -> tuple[float, float]:
    """Legacy pressure function - kept for compatibility but inventory_skew is preferred."""
    pressure = _clamp(open_risk / max(1e-6, limits.max_exposure_per_market_dollars), 0.0, 1.0)
    shift = 0.01 * pressure
    bid = p_fair - (spread / 2.0) - shift
    ask = p_fair + (spread / 2.0) + shift
    return bid, ask


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _total_abs_pos(yes_pos: dict[str, int]) -> int:
    return int(sum(abs(v) for v in yes_pos.values()))


def _cap_quote_prices_behind_top(*, bid_c: int, ask_c: int, best_bid: int, best_ask: int, behind_top_cents: int) -> tuple[int, int]:
    """Enforce maker-ish quoting and optionally quote behind top-of-book.

    bid <= best_bid - behind
    ask >= best_ask + behind
    """

    behind = max(0, int(behind_top_cents))
    bid_c = min(bid_c, max(1, best_bid - behind))
    ask_c = max(ask_c, min(99, best_ask + behind))
    return bid_c, ask_c


async def _flatten_positions(
    *,
    client: KalshiClient,
    portfolio: "PaperPortfolio",
    trades_w: csv.writer,
    now: datetime.datetime,
) -> int:
    """Close any remaining YES inventory at current bid/ask.

    This turns mark-to-market PnL into a more "realized" proxy.
    """

    flattened = 0
    for ticker, qty in list(portfolio.yes_pos.items()):
        if qty == 0:
            continue

        ob = await client.get_orderbook(ticker)
        prices = compute_best_prices(ob)
        if prices.best_yes_bid is None or prices.best_yes_ask is None:
            continue

        mid_now = (prices.best_yes_bid + prices.best_yes_ask) / 200.0
        portfolio.mark_mid(ticker=ticker, mid=mid_now)

        if qty > 0:
            # Sell long at bid.
            fill_c = int(prices.best_yes_bid)
            fill_credit = (fill_c / 100.0) * qty
            fee = _fee_dollars(price_cents=fill_c, count=qty, maker=False)
            portfolio.fees_paid += fee
            portfolio.cash += (fill_credit - fee)
            portfolio.yes_pos[ticker] = 0
            flattened += 1

            trades_w.writerow([
                now.isoformat(),
                ticker,
                "sell",
                fill_c,
                qty,
                f"{fee:.2f}",
                f"{portfolio.cash:.4f}",
                0,
                f"{portfolio.equity():.4f}",
                "flatten",
            ])
        else:
            # Buy to cover short at ask.
            cover = abs(qty)
            fill_c = int(prices.best_yes_ask)
            fill_cost = (fill_c / 100.0) * cover
            fee = _fee_dollars(price_cents=fill_c, count=cover, maker=False)
            portfolio.fees_paid += fee
            portfolio.cash -= (fill_cost + fee)
            portfolio.yes_pos[ticker] = 0
            flattened += 1

            trades_w.writerow([
                now.isoformat(),
                ticker,
                "buy",
                fill_c,
                cover,
                f"{fee:.2f}",
                f"{portfolio.cash:.4f}",
                0,
                f"{portfolio.equity():.4f}",
                "flatten",
            ])

    return flattened


async def run_paper(
    *,
    client: KalshiClient,
    assets: list[str],
    horizon_minutes: int,
    per_asset: int,
    poll_seconds: float,
    duration_minutes: float,
    bankroll_dollars: float,
    base_order_dollars: float,
    out_dir: str,
    max_abs_pos_per_market: int,
    max_abs_pos_total: int,
    behind_top_cents: int,
    flatten_at_end: bool,
) -> None:
    log = logging.getLogger("tradebot.crypto_mm_paper")

    limits = RiskLimits(bankroll_dollars=bankroll_dollars, base_order_dollars=base_order_dollars)
    portfolio = PaperPortfolio(starting_cash=bankroll_dollars, cash=bankroll_dollars)
    engine = PaperEngine(limits=limits)

    model = FairProbabilityModel()
    vol = VolatilityEstimator()
    spread_policy = SpreadPolicy()

    prev_mid: dict[str, float] = {}
    ema_mid: dict[str, float] = {}
    
    # Adverse selection tracking per ticker
    adverse_tracker: dict[str, AdverseSelectionTracker] = {}
    
    # Track entry prices for auto-flatten
    entry_prices: dict[str, float] = {}  # ticker -> avg entry price (mid when first filled)

    _ensure_dir(out_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join(out_dir, f"paper_trades_{ts}.csv")
    equity_path = os.path.join(out_dir, f"paper_equity_{ts}.csv")
    quotes_path = os.path.join(out_dir, f"paper_quotes_{ts}.csv")

    trades_f = open(trades_path, "w", newline="", encoding="utf-8")
    equity_f = open(equity_path, "w", newline="", encoding="utf-8")
    quotes_f = open(quotes_path, "w", newline="", encoding="utf-8")

    trades_w = csv.writer(trades_f)
    equity_w = csv.writer(equity_f)
    quotes_w = csv.writer(quotes_f)

    trades_w.writerow(["ts", "ticker", "action", "price_cents", "count", "fee_dollars", "cash_after", "pos_after", "equity_after", "note"])
    equity_w.writerow(["ts", "cash", "equity", "pnl", "fees_paid", "open_order_risk"])
    quotes_w.writerow(["ts", "ticker", "best_yes_bid", "best_yes_ask", "mid", "p_fair", "vol", "spread", "our_bid", "our_ask", "size"])

    end_ts = time.monotonic() + (duration_minutes * 60.0)

    log.info(
        "Paper run start assets=%s duration=%.1fmin poll=%.1fs bankroll=$%.2f out=%s",
        ",".join(assets),
        duration_minutes,
        poll_seconds,
        bankroll_dollars,
        out_dir,
    )

    trade_count = 0

    try:
        while time.monotonic() < end_ts:
            loop_started = time.monotonic()

            selection = await pick_active_markets(
                client=client,
                assets=assets,
                horizon_minutes=horizon_minutes,
                per_asset=per_asset,
            )

            # For each selected market: simulate fills on existing orders, then cancel/replace.
            for _, markets in selection.items():
                for m in markets:
                    ticker = str(m.get("ticker") or "")
                    if not ticker:
                        continue

                    exp = _to_dt(m.get("expected_expiration_time"))
                    now = _utcnow()
                    seconds_to_expiry = int((exp - now).total_seconds()) if exp else 9999
                    
                    # Use configurable stop-quoting (default 120s, was 60s)
                    if seconds_to_expiry <= DEFAULT_STOP_QUOTING_SECONDS:
                        engine.cancel_all(ticker=ticker)
                        
                        # AUTO-EXIT before expiry: if we have a position, flatten it now
                        # This prevents holding into settlement (unknown outcome in paper mode)
                        pos_here = int(portfolio.yes_pos.get(ticker, 0))
                        if pos_here != 0:
                            ob = await client.get_orderbook(ticker)
                            prices = compute_best_prices(ob)
                            if prices.best_yes_bid is not None and prices.best_yes_ask is not None:
                                mid_now = (prices.best_yes_bid + prices.best_yes_ask) / 200.0
                                portfolio.mark_mid(ticker=ticker, mid=mid_now)
                                
                                if pos_here > 0:
                                    fill_c = int(prices.best_yes_bid)
                                    fill_credit = (fill_c / 100.0) * pos_here
                                    fee = _fee_dollars(price_cents=fill_c, count=pos_here, maker=False)
                                    portfolio.fees_paid += fee
                                    portfolio.cash += (fill_credit - fee)
                                    trades_w.writerow([
                                        now.isoformat(), ticker, "sell", fill_c, pos_here,
                                        f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                                        f"{portfolio.equity():.4f}", "pre-expiry-exit",
                                    ])
                                else:
                                    cover = abs(pos_here)
                                    fill_c = int(prices.best_yes_ask)
                                    fill_cost = (fill_c / 100.0) * cover
                                    fee = _fee_dollars(price_cents=fill_c, count=cover, maker=False)
                                    portfolio.fees_paid += fee
                                    portfolio.cash -= (fill_cost + fee)
                                    trades_w.writerow([
                                        now.isoformat(), ticker, "buy", fill_c, cover,
                                        f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                                        f"{portfolio.equity():.4f}", "pre-expiry-exit",
                                    ])
                                portfolio.yes_pos[ticker] = 0
                                entry_prices.pop(ticker, None)
                                trade_count += 1
                                log.info("Pre-expiry exit %s (seconds_left=%d)", ticker, seconds_to_expiry)
                        continue
                    
                    # Compute time factor for scaling spread/caps/size
                    time_factor = _compute_time_factor(
                        seconds_to_expiry=seconds_to_expiry,
                        stop_quoting_seconds=DEFAULT_STOP_QUOTING_SECONDS,
                    )

                    ob = await client.get_orderbook(ticker)
                    prices = compute_best_prices(ob)
                    if prices.best_yes_bid is None or prices.best_yes_ask is None:
                        continue

                    mid_now = (prices.best_yes_bid + prices.best_yes_ask) / 200.0
                    portfolio.mark_mid(ticker=ticker, mid=mid_now)

                    # Inventory caps: prevent large position drift.
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    cur_total_abs = _total_abs_pos(portfolio.yes_pos)

                    # 1) Fill simulation for existing resting orders.
                    existing = list(engine.orders_by_ticker.get(ticker) or [])
                    for o in existing:
                        if o.action == "buy":
                            if prices.best_yes_ask is not None and prices.best_yes_ask <= o.price_cents:
                                fill_c = int(prices.best_yes_ask)
                                fill_cost = (fill_c / 100.0) * o.count
                                fee = _fee_dollars(price_cents=fill_c, count=o.count, maker=True)
                                portfolio.fees_paid += fee
                                portfolio.cash -= (fill_cost + fee)
                                portfolio.yes_pos[ticker] = int(portfolio.yes_pos.get(ticker, 0) + o.count)

                                engine.remove(o)
                                trade_count += 1
                                
                                # Track for adverse selection
                                if ticker not in adverse_tracker:
                                    adverse_tracker[ticker] = AdverseSelectionTracker()
                                adverse_tracker[ticker].record_fill('buy')
                                
                                # Track entry price for auto-flatten
                                if ticker not in entry_prices:
                                    entry_prices[ticker] = mid_now

                                trades_w.writerow([
                                    now.isoformat(),
                                    ticker,
                                    "buy",
                                    fill_c,
                                    o.count,
                                    f"{fee:.2f}",
                                    f"{portfolio.cash:.4f}",
                                    portfolio.yes_pos[ticker],
                                    f"{portfolio.equity():.4f}",
                                    "",
                                ])

                        elif o.action == "sell":
                            if prices.best_yes_bid is not None and prices.best_yes_bid >= o.price_cents:
                                fill_c = int(prices.best_yes_bid)
                                fill_credit = (fill_c / 100.0) * o.count
                                fee = _fee_dollars(price_cents=fill_c, count=o.count, maker=True)
                                portfolio.fees_paid += fee
                                portfolio.cash += (fill_credit - fee)
                                portfolio.yes_pos[ticker] = int(portfolio.yes_pos.get(ticker, 0) - o.count)

                                engine.remove(o)
                                trade_count += 1
                                
                                # Track for adverse selection
                                if ticker not in adverse_tracker:
                                    adverse_tracker[ticker] = AdverseSelectionTracker()
                                adverse_tracker[ticker].record_fill('sell')
                                
                                # Track entry price for auto-flatten
                                if ticker not in entry_prices:
                                    entry_prices[ticker] = mid_now

                                trades_w.writerow([
                                    now.isoformat(),
                                    ticker,
                                    "sell",
                                    fill_c,
                                    o.count,
                                    f"{fee:.2f}",
                                    f"{portfolio.cash:.4f}",
                                    portfolio.yes_pos[ticker],
                                    f"{portfolio.equity():.4f}",
                                    "",
                                ])
                    
                    # --- AUTO-FLATTEN CHECK ---
                    # If we have a position and unrealized loss exceeds threshold, flatten now
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    if cur_pos != 0 and ticker in entry_prices:
                        entry_mid = entry_prices[ticker]
                        # Unrealized P/L: (current_mid - entry_mid) * position
                        unrealized = (mid_now - entry_mid) * cur_pos
                        if unrealized < -DEFAULT_AUTO_FLATTEN_LOSS_DOLLARS:
                            # Emergency flatten this position
                            if cur_pos > 0:
                                fill_c = int(prices.best_yes_bid)
                                fill_credit = (fill_c / 100.0) * cur_pos
                                fee = _fee_dollars(price_cents=fill_c, count=cur_pos, maker=False)
                                portfolio.fees_paid += fee
                                portfolio.cash += (fill_credit - fee)
                                trades_w.writerow([
                                    now.isoformat(), ticker, "sell", fill_c, cur_pos,
                                    f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                                    f"{portfolio.equity():.4f}", "auto-flatten-loss",
                                ])
                            else:
                                cover = abs(cur_pos)
                                fill_c = int(prices.best_yes_ask)
                                fill_cost = (fill_c / 100.0) * cover
                                fee = _fee_dollars(price_cents=fill_c, count=cover, maker=False)
                                portfolio.fees_paid += fee
                                portfolio.cash -= (fill_cost + fee)
                                trades_w.writerow([
                                    now.isoformat(), ticker, "buy", fill_c, cover,
                                    f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                                    f"{portfolio.equity():.4f}", "auto-flatten-loss",
                                ])
                            portfolio.yes_pos[ticker] = 0
                            entry_prices.pop(ticker, None)
                            trade_count += 1
                            log.warning("Auto-flattened %s due to loss > $%.2f", ticker, DEFAULT_AUTO_FLATTEN_LOSS_DOLLARS)
                            continue  # Skip quoting this ticker this cycle

                    # 2) Compute next quotes (cancel/replace behaviour).
                    engine.cancel_all(ticker=ticker)

                    prev_ema = ema_mid.get(ticker)
                    ema = mid_now if prev_ema is None else (0.9 * prev_ema + 0.1 * mid_now)
                    ema_mid[ticker] = ema

                    mid_prev = prev_mid.get(ticker)
                    prev_mid[ticker] = mid_now

                    volatility = vol.update(mid_now=mid_now, mid_prev=mid_prev)
                    base_spread = spread_policy.compute(volatility=volatility)
                    
                    # Time-decay spread widening (gamma risk increases near expiry)
                    spread = _time_adjusted_spread(
                        base_spread=base_spread,
                        time_factor=time_factor,
                        max_multiplier=DEFAULT_TIME_SPREAD_MULTIPLIER_MAX,
                    )
                    
                    p_fair = model.compute(mid_now=mid_now, mid_prev=mid_prev, ema_mid=ema)

                    # Time-adjusted size (smaller near expiry)
                    size = _choose_size(limits=limits, volatility=volatility, time_factor=time_factor)

                    # Get current inventory for this ticker
                    cur_inventory = int(portfolio.yes_pos.get(ticker, 0))
                    
                    # Inventory skew: exit positions faster, more aggressive near expiry
                    bid_p, ask_p = _inventory_skew(
                        p_fair=p_fair,
                        spread=spread,
                        inventory=cur_inventory,
                        skew_cents_per_contract=DEFAULT_INVENTORY_SKEW_CENTS_PER_CONTRACT,
                        time_factor=time_factor,
                    )
                    
                    # Adverse selection adjustment: if getting picked off one side, widen it
                    if ticker in adverse_tracker:
                        bid_widen, ask_widen = adverse_tracker[ticker].get_skew_recommendation()
                        bid_p -= bid_widen / 100.0
                        ask_p += ask_widen / 100.0

                    bid_c = _round_price_cents(bid_p)
                    ask_c = _round_price_cents(ask_p)
                    if bid_c >= ask_c:
                        bid_c = max(1, min(bid_c, ask_c - 1))

                    # Maker-only: ensure we don't cross the current book.
                    bid_c, ask_c = _cap_quote_prices_behind_top(
                        bid_c=bid_c,
                        ask_c=ask_c,
                        best_bid=int(prices.best_yes_bid),
                        best_ask=int(prices.best_yes_ask),
                        behind_top_cents=behind_top_cents,
                    )
                    if bid_c >= ask_c:
                        continue

                    # Re-check position caps right before placing.
                    # Caps shrink as expiry approaches (don't build inventory late)
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    cur_total_abs = _total_abs_pos(portfolio.yes_pos)
                    
                    per_cap = _time_adjusted_pos_cap(
                        base_cap=max(0, int(max_abs_pos_per_market)),
                        time_factor=time_factor,
                        min_pct=DEFAULT_TIME_POS_CAP_MIN_PCT,
                    )
                    tot_cap = _time_adjusted_pos_cap(
                        base_cap=max(0, int(max_abs_pos_total)),
                        time_factor=time_factor,
                        min_pct=DEFAULT_TIME_POS_CAP_MIN_PCT,
                    )

                    allow_buy = True
                    allow_sell = True

                    if per_cap and cur_pos >= per_cap:
                        allow_buy = False
                    if per_cap and cur_pos <= -per_cap:
                        allow_sell = False

                    if tot_cap and cur_total_abs >= tot_cap:
                        # Only allow quotes that reduce total abs exposure.
                        if cur_pos > 0:
                            allow_buy = False
                        elif cur_pos < 0:
                            allow_sell = False
                        else:
                            allow_buy = False
                            allow_sell = False

                    # Risk caps (resting-order worst-case loss).
                    buy_risk = _max_loss_yes_buy(yes_price_cents=bid_c, count=size)
                    sell_risk = _max_loss_yes_sell(yes_price_cents=ask_c, count=size)
                    add_risk = (buy_risk if allow_buy else 0.0) + (sell_risk if allow_sell else 0.0)

                    if not engine.can_place(ticker=ticker, add_risk=add_risk):
                        continue

                    # Place new resting orders (paper).
                    ts_now = _utcnow()
                    if allow_buy:
                        buy_order = PaperOrder(
                            order_id=f"PAPER-{ticker}-B-{int(ts_now.timestamp() * 1000)}",
                            ticker=ticker,
                            action="buy",
                            price_cents=bid_c,
                            count=size,
                            placed_ts=ts_now,
                        )
                        engine.place(buy_order, risk_dollars=buy_risk)

                    if allow_sell:
                        sell_order = PaperOrder(
                            order_id=f"PAPER-{ticker}-S-{int(ts_now.timestamp() * 1000)}",
                            ticker=ticker,
                            action="sell",
                            price_cents=ask_c,
                            count=size,
                            placed_ts=ts_now,
                        )
                        engine.place(sell_order, risk_dollars=sell_risk)

                    quotes_w.writerow([
                        ts_now.isoformat(),
                        ticker,
                        prices.best_yes_bid,
                        prices.best_yes_ask,
                        f"{mid_now:.4f}",
                        f"{p_fair:.4f}",
                        f"{volatility:.6f}",
                        f"{spread:.4f}",
                        bid_c,
                        ask_c,
                        size,
                    ])

            # Equity snapshot
            now = _utcnow()
            equity_w.writerow([
                now.isoformat(),
                f"{portfolio.cash:.4f}",
                f"{portfolio.equity():.4f}",
                f"{portfolio.pnl():.4f}",
                f"{portfolio.fees_paid:.4f}",
                f"{engine.total_open_risk():.4f}",
            ])

            # Keep cadence.
            elapsed = time.monotonic() - loop_started
            await asyncio.sleep(max(0.0, poll_seconds - elapsed))

    finally:
        # If the run is interrupted (Ctrl+C), asyncio cancellation may occur. Shield
        # finalization so we still attempt to flatten and flush logs.
        if flatten_at_end:
            try:
                now = _utcnow()
                flattened = await asyncio.shield(
                    _flatten_positions(
                        client=client,
                        portfolio=portfolio,
                        trades_w=trades_w,
                        now=now,
                    )
                )
                # One more equity snapshot after flatten.
                equity_w.writerow([
                    now.isoformat(),
                    f"{portfolio.cash:.4f}",
                    f"{portfolio.equity():.4f}",
                    f"{portfolio.pnl():.4f}",
                    f"{portfolio.fees_paid:.4f}",
                    f"{engine.total_open_risk():.4f}",
                ])
                log.info("Flattened positions in %d tickers", flattened)
            except BaseException:
                # CancelledError inherits BaseException in newer Python versions.
                log.exception("Flatten-at-end failed")

        try:
            trades_f.close()
            equity_f.close()
            quotes_f.close()
        finally:
            # Always emit a final summary if possible.
            log.info(
                "Paper run done trades=%d final_equity=%.4f pnl=%.4f trades_csv=%s equity_csv=%s quotes_csv=%s",
                trade_count,
                portfolio.equity(),
                portfolio.pnl(),
                trades_path,
                equity_path,
                quotes_path,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-run the crypto 15m market maker (NO TRADES SENT)")
    parser.add_argument("--assets", default=DEFAULT_ASSETS, help="Comma-separated assets")
    parser.add_argument("--horizon-minutes", type=int, default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--per-asset", type=int, default=DEFAULT_PER_ASSET)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--duration-minutes", type=float, default=DEFAULT_DURATION_MINUTES)

    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL_DOLLARS)
    parser.add_argument("--base-order", type=float, default=DEFAULT_BASE_ORDER_DOLLARS)

    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="Output folder for CSV logs")

    parser.add_argument("--max-abs-pos-per-market", type=int, default=DEFAULT_MAX_ABS_POS_PER_MARKET)
    parser.add_argument("--max-abs-pos-total", type=int, default=DEFAULT_MAX_ABS_POS_TOTAL)
    parser.add_argument("--behind-top-cents", type=int, default=DEFAULT_BEHIND_TOP_CENTS)
    parser.add_argument(
        "--flatten-at-end",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_FLATTEN_AT_END,
        help="Close any remaining inventory at end (default: on)",
    )

    args = parser.parse_args()

    settings = Settings.load()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    assets = [a.strip().upper() for a in (args.assets or "").split(",") if a.strip()]

    client = KalshiClient.from_settings(settings)

    async def runner() -> None:
        try:
            await run_paper(
                client=client,
                assets=assets,
                horizon_minutes=int(args.horizon_minutes),
                per_asset=int(args.per_asset),
                poll_seconds=float(args.poll_seconds),
                duration_minutes=float(args.duration_minutes),
                bankroll_dollars=float(args.bankroll),
                base_order_dollars=float(args.base_order),
                out_dir=str(args.out_dir),
                max_abs_pos_per_market=int(args.max_abs_pos_per_market),
                max_abs_pos_total=int(args.max_abs_pos_total),
                behind_top_cents=int(args.behind_top_cents),
                flatten_at_end=bool(args.flatten_at_end),
            )
        finally:
            await client.aclose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
