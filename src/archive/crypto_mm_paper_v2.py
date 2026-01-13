"""Enhanced Paper Trader with Statistical Models (V8).

This version integrates:
1. Order Flow Imbalance - detect directional pressure
2. GARCH Volatility - better vol estimates with clustering
3. Kelly Criterion - optimal position sizing
4. Regime Detection - trend vs mean-reversion
5. Avellaneda-Stoikov - optimal spread computation
6. Fill Probability - model P(fill | distance)
7. BTC Price Feed - live BTC price vs strike comparison (V6)
8. P_fair Smoothing - EMA to prevent whiplash entries (V7)
9. Sustained Edge - require edge to persist for N polls (V7)
10. Active Exit Management - take-profit, trailing stop, stop-loss (V8)
11. Live Trading Mode - real orders with --live flag (V8)
12. Dry Run Mode - log what would happen with --dry-run flag (V8)

Key improvements over v1:
- Regime-aware: follows momentum in trends, fades in mean-reversion
- Better spread computation using A-S framework
- OFI-based quote skewing (anticipate price moves)
- GARCH volatility captures clustering
- Kelly-based sizing for edge-weighted positions
- V5: Position-aware adding, OFI confirmation
- V6: BTC price confirmation against strike
- V7: p_fair smoothing + sustained edge requirement
- V8: Active position exit management (no more holding to expiry)

Usage:
  Paper trading (simulation): python crypto_mm_paper_v2.py
  Dry run (log only):         python crypto_mm_paper_v2.py --live --dry-run
  Live trading (real money):  python crypto_mm_paper_v2.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import logging
import math
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_best_prices

# Import the new statistical models
from tradebot.models.order_flow import OrderFlowImbalance, OrderbookSnapshot
from tradebot.models.garch_vol import GARCHVolatility
from tradebot.models.kelly import KellyCriterion
from tradebot.models.regime import RegimeDetector, Regime
from tradebot.models.avellaneda_stoikov import AvellanedaStoikov
from tradebot.models.fill_probability import FillProbabilityModel
from tradebot.models.btc_price_feed import BTCPriceFeed


# ----------------------------
# Easy-to-edit defaults
# ----------------------------

DEFAULT_ASSETS = "BTC"
DEFAULT_HORIZON_MINUTES = 60
DEFAULT_PER_ASSET = 2
DEFAULT_POLL_SECONDS = 1.0
DEFAULT_DURATION_MINUTES = 240.0
DEFAULT_BANKROLL_DOLLARS = 100.0
DEFAULT_BASE_ORDER_DOLLARS = 1.5
DEFAULT_OUT_DIR = "runs"

DEFAULT_MAX_ABS_POS_PER_MARKET = 10
DEFAULT_MAX_ABS_POS_TOTAL = 25
DEFAULT_BEHIND_TOP_CENTS = 0  # Quote AT best price for better fills
DEFAULT_FLATTEN_AT_END = True

# Live trading mode (V8) - conservative defaults for real money
DEFAULT_LIVE_MAX_POS_PER_MARKET = 15  # Max contracts in one market
DEFAULT_LIVE_MAX_POS_TOTAL = 15  # Max contracts total (same since only 1 market)
DEFAULT_LIVE_BASE_ORDER = 1  # $1 orders in live mode
DEFAULT_STOP_QUOTING_SECONDS = 60
DEFAULT_AUTO_FLATTEN_LOSS_DOLLARS = 3.0

# Trend protection thresholds
DEFAULT_TREND_HURST_THRESHOLD = 0.55  # Above this = strong trend
DEFAULT_OFI_BLOCK_THRESHOLD = 0.15    # OFI above this blocks counter-trend trades (loosened from 0.03)
DEFAULT_SAFE_WIN_THRESHOLD = 0.80     # Price >= 80c = safe win for shorts at expiry (was 0.85)
DEFAULT_SAFE_LOSS_THRESHOLD = 0.20    # Price <= 20c = safe win for longs at expiry (was 0.15)
DEFAULT_TREND_SIZE_REDUCTION = 0.5    # Cut size by 50% in trending regimes

# Momentum filter (V8.5 - no longer blocks, just tracked)
DEFAULT_MOMENTUM_BLOCK_THRESHOLD = 0.03  # Legacy - not used for blocking anymore
DEFAULT_MIN_EDGE_CENTS = 2  # Require 2c edge (lower since we follow momentum now)

# Drawdown protection (V8 fix) - don't add to underwater positions
DEFAULT_MAX_UNREALIZED_LOSS_CENTS = 10  # Don't add if already down >10c per contract
DEFAULT_WATERFALL_DROP_PCT = 0.15  # Don't buy if price dropped >15% in tracking window
DEFAULT_WATERFALL_WINDOW = 10  # Track last 10 price observations for waterfall detection

# BTC Price Feed (V6)
DEFAULT_USE_BTC_FEED = True  # Enable live BTC price confirmation
DEFAULT_BTC_EDGE_THRESHOLD = 0.10  # Block if BTC signal says >10c opposite edge
DEFAULT_CMC_KEY_FILE = "cmckey.txt"  # CoinMarketCap API key file
DEFAULT_BTC_WEIGHT_IN_PFAIR = 0.3  # Weight BTC signal in p_fair calculation

# P_fair Smoothing (V7) - prevent whiplash entries
DEFAULT_PFAIR_EMA_ALPHA = 0.3  # EMA smoothing factor (0.3 = moderate smoothing)

# Sustained Edge (V7) - require edge to persist before entering
DEFAULT_SUSTAINED_EDGE_POLLS = 2  # Number of consecutive polls with edge required (was 3)
DEFAULT_SUSTAINED_EDGE_MIN = 0.03  # Minimum edge (3%) to count as "having edge" (was 5%)

# Active Exit Management (V8.5) - QUICK PROFITS, tight risk management
DEFAULT_TAKE_PROFIT_CENTS = 3  # Take profit at 3c (scalping approach)
DEFAULT_TAKE_PROFIT_PCT = 0.25  # Or 25% ROI (whichever hits first)
DEFAULT_TRAILING_STOP_CENTS = 5  # Trailing stop at 5c (tighter to lock in gains)
DEFAULT_STOP_LOSS_CENTS = 8  # Stop loss at 8c (cut losses quickly)
DEFAULT_EXPIRY_PROFIT_TAKE_SECONDS = 180  # Take profits if <3 min to expiry and profitable
DEFAULT_EXPIRY_MIN_PROFIT_PCT = 0.20  # Need at least 20% profit to take near expiry

# V8.5 Momentum Hedges - protect against whipsaws
DEFAULT_STOP_LOSS_COOLDOWN_SECONDS = 30  # After stop-loss, wait 30s before re-entering
DEFAULT_EXTREME_PRICE_BUY_MAX = 0.85  # Don't BUY above 85c (limited upside)
DEFAULT_EXTREME_PRICE_SELL_MIN = 0.15  # Don't SELL below 15c (limited upside)
DEFAULT_MIN_OFI_STRENGTH = 0.05  # Require OFI magnitude > 5% to trade (avoid noise)
DEFAULT_MAX_LOSSES_PER_MARKET = 4  # Stop trading market after 4 consecutive losses

# Avellaneda-Stoikov parameters
DEFAULT_AS_GAMMA = 0.5  # Risk aversion (lower = tighter spreads)
DEFAULT_AS_KAPPA = 1.5  # Order arrival intensity

# Kelly parameters
DEFAULT_KELLY_FRACTION = 0.25  # Use 1/4 Kelly for safety
DEFAULT_MIN_EDGE = 0.03  # Require 3% edge to trade

# Fee rates
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
    action: str
    price_cents: int
    count: int
    placed_ts: datetime.datetime


@dataclass
class PaperPortfolio:
    starting_cash: float
    cash: float
    fees_paid: float = 0.0
    yes_pos: dict[str, int] = field(default_factory=dict)
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
    max_exposure_per_market: float = 10.0
    max_exposure_total: float = 25.0
    
    open_order_risk: dict[str, float] = field(default_factory=dict)
    orders_by_ticker: dict[str, list[PaperOrder]] = field(default_factory=dict)

    def _ticker_open_risk(self, ticker: str) -> float:
        orders = self.orders_by_ticker.get(ticker) or []
        return float(sum(self.open_order_risk.get(o.order_id, 0.0) for o in orders))

    def total_open_risk(self) -> float:
        return float(sum(self.open_order_risk.values()))

    def can_place(self, *, ticker: str, add_risk: float) -> bool:
        if add_risk <= 0:
            return True
        if self._ticker_open_risk(ticker) + add_risk > self.max_exposure_per_market:
            return False
        if self.total_open_risk() + add_risk > self.max_exposure_total:
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
            self.orders_by_ticker[order.ticker] = [
                o for o in self.orders_by_ticker[order.ticker] 
                if o.order_id != order.order_id
            ]
            if not self.orders_by_ticker[order.ticker]:
                self.orders_by_ticker.pop(order.ticker, None)
        self.open_order_risk.pop(order.order_id, None)


@dataclass
class TradeStats:
    """Track trade statistics for win rate calculation."""
    round_trips: list[float] = field(default_factory=list)  # PnL for each completed round-trip
    flattens: list[float] = field(default_factory=list)  # PnL for each flatten
    total_buys: int = 0
    total_sells: int = 0
    
    def record_round_trip(self, pnl: float) -> None:
        """Record a completed round-trip trade."""
        self.round_trips.append(pnl)
    
    def record_flatten(self, pnl: float) -> None:
        """Record a flatten/exit trade."""
        self.flattens.append(pnl)
    
    def win_rate(self) -> float:
        """Calculate win rate from round-trips."""
        all_trades = self.round_trips + self.flattens
        if not all_trades:
            return 0.0
        wins = sum(1 for pnl in all_trades if pnl > 0)
        return wins / len(all_trades)
    
    def profit_factor(self) -> float:
        """Calculate profit factor (gross wins / gross losses)."""
        all_trades = self.round_trips + self.flattens
        gross_wins = sum(pnl for pnl in all_trades if pnl > 0)
        gross_losses = abs(sum(pnl for pnl in all_trades if pnl < 0))
        if gross_losses == 0:
            return float('inf') if gross_wins > 0 else 0.0
        return gross_wins / gross_losses
    
    def avg_win(self) -> float:
        """Average winning trade."""
        all_trades = self.round_trips + self.flattens
        wins = [pnl for pnl in all_trades if pnl > 0]
        return sum(wins) / len(wins) if wins else 0.0
    
    def avg_loss(self) -> float:
        """Average losing trade."""
        all_trades = self.round_trips + self.flattens
        losses = [pnl for pnl in all_trades if pnl < 0]
        return sum(losses) / len(losses) if losses else 0.0
    
    def total_trades(self) -> int:
        """Total completed trades."""
        return len(self.round_trips) + len(self.flattens)
    
    def summary(self) -> str:
        """Generate summary string."""
        all_trades = self.round_trips + self.flattens
        if not all_trades:
            return "No completed trades"
        
        wins = sum(1 for pnl in all_trades if pnl > 0)
        losses = sum(1 for pnl in all_trades if pnl < 0)
        breakeven = sum(1 for pnl in all_trades if pnl == 0)
        
        return (
            f"Trades: {self.total_trades()} (W:{wins} L:{losses} BE:{breakeven}) | "
            f"Win Rate: {self.win_rate()*100:.1f}% | "
            f"Profit Factor: {self.profit_factor():.2f} | "
            f"Avg Win: ${self.avg_win():.3f} | "
            f"Avg Loss: ${self.avg_loss():.3f}"
        )


@dataclass
class PositionInfo:
    """Track position details for exit management (V8)."""
    entry_price_cents: int  # Price we entered at (in cents)
    entry_time: datetime.datetime  # When we entered
    quantity: int  # Position size (positive=long, negative=short)
    high_water_cents: int  # Best price seen (for trailing stop)
    
    def update_high_water(self, current_price_cents: int) -> None:
        """Update high water mark based on position direction."""
        if self.quantity > 0:  # Long - track highest bid
            self.high_water_cents = max(self.high_water_cents, current_price_cents)
        else:  # Short - track lowest ask
            self.high_water_cents = min(self.high_water_cents, current_price_cents)


@dataclass
class TickerModels:
    """Per-ticker statistical models."""
    ofi: OrderFlowImbalance = field(default_factory=OrderFlowImbalance)
    garch: GARCHVolatility = field(default_factory=GARCHVolatility)
    regime: RegimeDetector = field(default_factory=RegimeDetector)
    fill_prob: FillProbabilityModel = field(default_factory=FillProbabilityModel)
    prev_mid: float | None = None
    entry_price: float | None = None
    # Momentum tracking: store last few mids for momentum filter
    recent_mids: list[float] = field(default_factory=list)
    # P_fair smoothing (V7)
    smoothed_p_fair: float | None = None
    # Sustained edge tracking (V7)
    buy_edge_streak: int = 0  # Consecutive polls with buy edge
    sell_edge_streak: int = 0  # Consecutive polls with sell edge
    # Position tracking (V8)
    position_info: PositionInfo | None = None
    # Waterfall detection (V8 fix) - track price for crash detection
    waterfall_prices: list[float] = field(default_factory=list)
    # V8.5 Momentum hedges
    last_stop_loss_time: float = 0.0  # Timestamp of last stop-loss hit
    last_stop_loss_direction: str = ""  # "long" or "short" - which direction got stopped
    consecutive_losses: int = 0  # Count of consecutive losses on this market
    
    def update_momentum(self, mid: float, max_history: int = 3) -> None:
        """Track recent mids for momentum calculation."""
        self.recent_mids.append(mid)
        if len(self.recent_mids) > max_history:
            self.recent_mids.pop(0)
    
    def momentum_change(self) -> float:
        """Return price change from oldest to newest mid (positive = rising)."""
        if len(self.recent_mids) < 2:
            return 0.0
        return self.recent_mids[-1] - self.recent_mids[0]
    
    def update_waterfall(self, mid: float, window: int = DEFAULT_WATERFALL_WINDOW) -> None:
        """Track prices for waterfall (crash) detection."""
        self.waterfall_prices.append(mid)
        if len(self.waterfall_prices) > window:
            self.waterfall_prices.pop(0)
    
    def is_waterfall(self, threshold: float = DEFAULT_WATERFALL_DROP_PCT) -> bool:
        """Check if price has crashed (dropped by threshold %) over tracking window."""
        if len(self.waterfall_prices) < 3:
            return False
        max_price = max(self.waterfall_prices)
        current = self.waterfall_prices[-1]
        if max_price <= 0:
            return False
        drop_pct = (max_price - current) / max_price
        return drop_pct >= threshold
    
    def smooth_p_fair(self, raw_p_fair: float, alpha: float = DEFAULT_PFAIR_EMA_ALPHA) -> float:
        """Apply EMA smoothing to p_fair to prevent whiplash."""
        if self.smoothed_p_fair is None:
            self.smoothed_p_fair = raw_p_fair
        else:
            self.smoothed_p_fair = alpha * raw_p_fair + (1 - alpha) * self.smoothed_p_fair
        return self.smoothed_p_fair
    
    def update_edge_streaks(self, mid: float, p_fair: float, min_edge: float = DEFAULT_SUSTAINED_EDGE_MIN) -> None:
        """Track consecutive polls with buy/sell edge."""
        buy_edge = p_fair - mid  # Positive = p_fair > mid = buy edge
        sell_edge = mid - p_fair  # Positive = mid > p_fair = sell edge
        
        if buy_edge >= min_edge:
            self.buy_edge_streak += 1
            self.sell_edge_streak = 0  # Reset opposite
        elif sell_edge >= min_edge:
            self.sell_edge_streak += 1
            self.buy_edge_streak = 0  # Reset opposite
        else:
            # No clear edge - reset both
            self.buy_edge_streak = 0
            self.sell_edge_streak = 0
    
    def has_sustained_buy_edge(self, required_polls: int = DEFAULT_SUSTAINED_EDGE_POLLS) -> bool:
        """Check if we have sustained buy edge."""
        return self.buy_edge_streak >= required_polls
    
    def has_sustained_sell_edge(self, required_polls: int = DEFAULT_SUSTAINED_EDGE_POLLS) -> bool:
        """Check if we have sustained sell edge."""
        return self.sell_edge_streak >= required_polls


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
            if exp is None or exp < now or exp - now > horizon:
                continue
            keep.append((exp, m))

        keep.sort(key=lambda t: t[0])
        out[asset] = [m for _, m in keep[: max(1, per_asset)]]

    return out


def _compute_time_factor(*, seconds_to_expiry: int, stop_quoting_seconds: int = 120) -> float:
    """Returns 1.0 far from expiry, tapering to 0.0 at stop_quoting_seconds."""
    if seconds_to_expiry <= stop_quoting_seconds:
        return 0.0
    full_time = 600  # 10 minutes
    if seconds_to_expiry >= full_time:
        return 1.0
    return (seconds_to_expiry - stop_quoting_seconds) / (full_time - stop_quoting_seconds)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _total_abs_pos(yes_pos: dict[str, int]) -> int:
    return int(sum(abs(v) for v in yes_pos.values()))


async def _flatten_positions(
    *,
    client: KalshiClient,
    portfolio: PaperPortfolio,
    trades_w: csv.writer,
    now: datetime.datetime,
) -> int:
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
            fill_c = int(prices.best_yes_bid)
            fill_credit = (fill_c / 100.0) * qty
            fee = _fee_dollars(price_cents=fill_c, count=qty, maker=False)
            portfolio.fees_paid += fee
            portfolio.cash += (fill_credit - fee)
            portfolio.yes_pos[ticker] = 0
            flattened += 1
            trades_w.writerow([
                now.isoformat(), ticker, "sell", fill_c, qty,
                f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                f"{portfolio.equity():.4f}", "flatten",
            ])
        else:
            cover = abs(qty)
            fill_c = int(prices.best_yes_ask)
            fill_cost = (fill_c / 100.0) * cover
            fee = _fee_dollars(price_cents=fill_c, count=cover, maker=False)
            portfolio.fees_paid += fee
            portfolio.cash -= (fill_cost + fee)
            portfolio.yes_pos[ticker] = 0
            flattened += 1
            trades_w.writerow([
                now.isoformat(), ticker, "buy", fill_c, cover,
                f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                f"{portfolio.equity():.4f}", "flatten",
            ])

    return flattened


async def _sync_live_positions(
    *,
    client: KalshiClient,
    portfolio: PaperPortfolio,
    ticker_models: dict[str, "TickerModels"],
    log: logging.Logger,
) -> dict[str, int]:
    """Sync positions from Kalshi API to local portfolio tracking.
    
    This is CRITICAL for live mode - without it, the bot is blind to its own positions.
    
    Returns:
        Dict mapping ticker -> position (positive=long YES, negative=short YES)
    """
    try:
        positions_resp = await client.get_positions(count_filter="position")
        market_positions = positions_resp.get("market_positions", [])
        
        synced: dict[str, int] = {}
        
        for pos in market_positions:
            ticker = pos.get("ticker", "")
            position = pos.get("position", 0)  # Positive=long, negative=short
            market_exposure = pos.get("market_exposure", 0)  # In cents
            realized_pnl = pos.get("realized_pnl", 0)  # In cents
            
            if not ticker or position == 0:
                continue
            
            synced[ticker] = position
            
            # Update portfolio tracking
            old_pos = portfolio.yes_pos.get(ticker, 0)
            if old_pos != position:
                log.info("[POSITION SYNC] %s: local=%d -> kalshi=%d (exposure=$%.2f, realized_pnl=$%.2f)",
                         ticker, old_pos, position, market_exposure/100.0, realized_pnl/100.0)
                portfolio.yes_pos[ticker] = position
            
            # Initialize position_info if we have a position but no tracking
            if ticker in ticker_models:
                models = ticker_models[ticker]
                if position != 0 and models.position_info is None:
                    # We have a position but no entry info - estimate from exposure
                    # exposure = price * qty for longs, (100-price) * qty for shorts
                    if position > 0:
                        estimated_entry = int(abs(market_exposure) / abs(position)) if position != 0 else 50
                    else:
                        estimated_entry = 100 - int(abs(market_exposure) / abs(position)) if position != 0 else 50
                    estimated_entry = max(1, min(99, estimated_entry))
                    
                    models.position_info = PositionInfo(
                        entry_price_cents=estimated_entry,
                        entry_time=_utcnow(),
                        quantity=position,
                        high_water_cents=estimated_entry,
                    )
                    log.info("[POSITION SYNC] %s: created position_info (estimated_entry=%dc, qty=%d)",
                             ticker, estimated_entry, position)
                elif position != 0 and models.position_info is not None:
                    # Update quantity if changed
                    if models.position_info.quantity != position:
                        log.info("[POSITION SYNC] %s: updated position_info qty %d -> %d",
                                 ticker, models.position_info.quantity, position)
                        models.position_info.quantity = position
                elif position == 0 and models.position_info is not None:
                    # Position closed
                    log.info("[POSITION SYNC] %s: position closed, clearing position_info", ticker)
                    models.position_info = None
        
        # Also clear positions that Kalshi says we don't have
        for ticker in list(portfolio.yes_pos.keys()):
            if ticker not in synced and portfolio.yes_pos[ticker] != 0:
                log.info("[POSITION SYNC] %s: clearing local position (kalshi has none)", ticker)
                portfolio.yes_pos[ticker] = 0
                if ticker in ticker_models and ticker_models[ticker].position_info is not None:
                    ticker_models[ticker].position_info = None
        
        return synced
        
    except Exception as e:
        log.error("[POSITION SYNC] Failed to sync positions: %s", e)
        return {}


def compute_regime_fair_price(
    *,
    mid_now: float,
    mid_prev: float | None,
    regime: Regime,
    regime_momentum_weight: float,
    ofi_signal: float,
    btc_fair_prob: float | None = None,
    btc_weight: float = DEFAULT_BTC_WEIGHT_IN_PFAIR,
) -> float:
    """V8.5: MOMENTUM-FOLLOWING fair price calculation.
    
    KEY CHANGE: Instead of computing what we think "fair" is and betting against
    the market, we now FOLLOW momentum and OFI signals.
    
    Strategy:
    - If OFI is positive (buying pressure) -> p_fair slightly ABOVE mid -> triggers BUY
    - If OFI is negative (selling pressure) -> p_fair slightly BELOW mid -> triggers SELL
    - We go WITH the flow, not against it
    - Take quick profits along the way
    
    This is fundamentally different from the old approach which tried to
    "correct" the market back to some computed fair value.
    """
    # OFI is our PRIMARY signal for direction
    # Positive OFI = buying pressure = price likely to rise = we want to BUY
    # Negative OFI = selling pressure = price likely to fall = we want to SELL
    
    # Scale OFI contribution - this determines how much we lean into the direction
    # OFI typically ranges from -0.3 to +0.3
    ofi_lean = ofi_signal * 0.15  # If OFI=0.2, we lean p_fair 3c toward that direction
    
    # Momentum confirmation - if price is moving, lean further into that direction
    momentum = 0.0
    if mid_prev is not None and mid_prev > 0:
        price_change = mid_now - mid_prev
        momentum = price_change * 2.0  # Amplify recent price movement direction
    
    # In trending regime, lean harder into the direction
    if regime == Regime.TRENDING:
        direction_lean = ofi_lean * 1.5 + momentum * 1.0
    elif regime == Regime.MEAN_REVERTING:
        # Even in mean reverting, don't fight strong OFI
        direction_lean = ofi_lean * 0.8 + momentum * 0.3
    else:
        # Random regime - moderate lean
        direction_lean = ofi_lean * 1.0 + momentum * 0.5
    
    # P_fair is mid PLUS our directional lean
    # If OFI is positive, p_fair > mid, which triggers BUY signal
    # If OFI is negative, p_fair < mid, which triggers SELL signal
    p_fair = mid_now + direction_lean
    
    # Cap the lean - don't get too extreme
    max_lean = 0.06  # Max 6c lean from mid
    p_fair = max(mid_now - max_lean, min(mid_now + max_lean, p_fair))
    
    # BTC signal as a secondary confirmation (reduced weight)
    # Only use if it agrees with our direction
    if btc_fair_prob is not None:
        btc_direction = btc_fair_prob - mid_now  # Positive = BTC says up
        our_direction = p_fair - mid_now  # Positive = we say up
        
        # Only blend BTC if it agrees with our OFI-based direction
        if (btc_direction > 0 and our_direction > 0) or (btc_direction < 0 and our_direction < 0):
            # Agreement - blend in some BTC signal
            p_fair = p_fair * 0.85 + btc_fair_prob * 0.15
        # If BTC disagrees, ignore it (OFI is more real-time)
    
    return _clamp(p_fair, 0.05, 0.95)


async def run_paper_v2(
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
    live_mode: bool = False,
    dry_run: bool = False,
) -> None:
    log = logging.getLogger("tradebot.crypto_mm_paper_v2")
    
    # ======= LIVE MODE INITIALIZATION =======
    if live_mode:
        mode_str = "DRY-RUN" if dry_run else "LIVE TRADING"
        log.info("="*60)
        log.info("  %s MODE ACTIVE", mode_str)
        log.info("="*60)
        
        # Get and display account balance
        try:
            balance_resp = await client.get_balance()
            balance_cents = balance_resp.get("balance", 0)
            portfolio_value = balance_resp.get("portfolio_value", 0)
            log.info("Account Balance: $%.2f", balance_cents / 100.0)
            log.info("Portfolio Value: $%.2f", portfolio_value / 100.0)
            log.info("Total: $%.2f", (balance_cents + portfolio_value) / 100.0)
        except Exception as e:
            log.error("Failed to get balance: %s", e)
            if not dry_run:
                raise
        
        if not dry_run:
            log.warning("⚠️  REAL MONEY MODE - Orders will be placed!")
            log.warning("⚠️  Max position per market: %d contracts", max_abs_pos_per_market)
            log.warning("⚠️  Max total position: %d contracts", max_abs_pos_total)
            log.info("Starting in 5 seconds... Press Ctrl+C to abort.")
            await asyncio.sleep(5)
        else:
            log.info("📋 DRY-RUN: Will log what WOULD be traded, no real orders.")
        
        log.info("="*60)
    
    # Track live orders for cancellation
    live_orders: dict[str, list[str]] = {}  # ticker -> [order_ids]
    
    # V8.4: Order cooldown to prevent spam while waiting for fills
    last_order_time: dict[str, float] = {}  # ticker -> timestamp of last order placed
    ORDER_COOLDOWN_SECONDS = 15.0  # Don't place new orders for 15 seconds after placing one

    portfolio = PaperPortfolio(starting_cash=bankroll_dollars, cash=bankroll_dollars)
    engine = PaperEngine(
        max_exposure_per_market=max_abs_pos_per_market,
        max_exposure_total=max_abs_pos_total,
    )

    # Global models (shared across tickers for optimal sizing)
    kelly = KellyCriterion(
        kelly_fraction=DEFAULT_KELLY_FRACTION,
        min_edge=DEFAULT_MIN_EDGE,
    )
    avellaneda = AvellanedaStoikov(
        gamma=DEFAULT_AS_GAMMA,
        kappa=DEFAULT_AS_KAPPA,
    )

    # BTC Price Feed (V6) - optional live BTC confirmation
    btc_feed: BTCPriceFeed | None = None
    if DEFAULT_USE_BTC_FEED:
        try:
            btc_feed = BTCPriceFeed.from_file(DEFAULT_CMC_KEY_FILE)
            await btc_feed.calibrate_volatility()
            log.info("BTC feed initialized, 15-min vol: %.4f%%", btc_feed.get_calibrated_vol() * 100)
        except Exception as e:
            log.warning("BTC feed initialization failed, continuing without: %s", e)
            btc_feed = None

    # Per-ticker models
    ticker_models: dict[str, TickerModels] = {}

    _ensure_dir(out_dir)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join(out_dir, f"paper_trades_v2_{ts}.csv")
    equity_path = os.path.join(out_dir, f"paper_equity_v2_{ts}.csv")
    quotes_path = os.path.join(out_dir, f"paper_quotes_v2_{ts}.csv")

    trades_f = open(trades_path, "w", newline="", encoding="utf-8")
    equity_f = open(equity_path, "w", newline="", encoding="utf-8")
    quotes_f = open(quotes_path, "w", newline="", encoding="utf-8")

    trades_w = csv.writer(trades_f)
    equity_w = csv.writer(equity_f)
    quotes_w = csv.writer(quotes_f)

    trades_w.writerow([
        "ts", "ticker", "action", "price_cents", "count", "fee_dollars",
        "cash_after", "pos_after", "equity_after", "note"
    ])
    equity_w.writerow([
        "ts", "cash", "equity", "pnl", "fees_paid", "open_order_risk"
    ])
    quotes_w.writerow([
        "ts", "ticker", "best_yes_bid", "best_yes_ask", "mid", "p_fair",
        "regime", "hurst", "ofi", "vol_garch", "spread", "our_bid", "our_ask", "size"
    ])

    end_ts = time.monotonic() + (duration_minutes * 60.0)

    log.info(
        "Paper V2 run start assets=%s duration=%.1fmin poll=%.1fs bankroll=$%.2f",
        ",".join(assets), duration_minutes, poll_seconds, bankroll_dollars,
    )

    trade_count = 0
    trade_stats = TradeStats()
    
    # Track entry prices for PnL calculation on round-trips
    entry_prices: dict[str, list[tuple[float, int]]] = {}  # ticker -> [(price, qty), ...]

    try:
        while time.monotonic() < end_ts:
            loop_started = time.monotonic()
            
            # ======= LIVE MODE: SYNC POSITIONS FROM KALSHI =======
            # This is CRITICAL - without this, we're blind to our own positions
            if live_mode and not dry_run:
                synced_positions = await _sync_live_positions(
                    client=client,
                    portfolio=portfolio,
                    ticker_models=ticker_models,
                    log=log,
                )
                if synced_positions:
                    total_pos = sum(abs(p) for p in synced_positions.values())
                    log.info("[POSITION SYNC] Active positions: %s (total %d contracts)",
                             synced_positions, total_pos)

            selection = await pick_active_markets(
                client=client,
                assets=assets,
                horizon_minutes=horizon_minutes,
                per_asset=per_asset,
            )

            for _, markets in selection.items():
                for m in markets:
                    ticker = str(m.get("ticker") or "")
                    if not ticker:
                        continue

                    # Initialize per-ticker models if needed
                    if ticker not in ticker_models:
                        ticker_models[ticker] = TickerModels()
                    models = ticker_models[ticker]

                    exp = _to_dt(m.get("expected_expiration_time"))
                    now = _utcnow()
                    seconds_to_expiry = int((exp - now).total_seconds()) if exp else 9999

                    # Stop quoting near expiry
                    if seconds_to_expiry <= DEFAULT_STOP_QUOTING_SECONDS:
                        engine.cancel_all(ticker=ticker)
                        
                        # SMART Pre-expiry exit - only flatten uncertain positions
                        pos_here = int(portfolio.yes_pos.get(ticker, 0))
                        if pos_here != 0:
                            ob = await client.get_orderbook(ticker)
                            prices = compute_best_prices(ob)
                            if prices.best_yes_bid is not None and prices.best_yes_ask is not None:
                                mid_now = (prices.best_yes_bid + prices.best_yes_ask) / 200.0
                                portfolio.mark_mid(ticker=ticker, mid=mid_now)
                                
                                # Determine if we should hold or flatten
                                should_flatten = True
                                hold_reason = ""
                                
                                if pos_here > 0:  # Long YES
                                    # If price is high (>=85c), we're winning - HOLD to expiry
                                    if mid_now >= DEFAULT_SAFE_WIN_THRESHOLD:
                                        should_flatten = False
                                        hold_reason = f"hold-long-winning-{int(mid_now*100)}c"
                                    # If price is very low (<=15c), max loss is small - let it expire
                                    elif mid_now <= DEFAULT_SAFE_LOSS_THRESHOLD:
                                        should_flatten = False
                                        hold_reason = f"hold-long-small-loss-{int(mid_now*100)}c"
                                else:  # Short YES (pos_here < 0)
                                    # If price is low (<=15c), we're winning - HOLD to expiry  
                                    if mid_now <= DEFAULT_SAFE_LOSS_THRESHOLD:
                                        should_flatten = False
                                        hold_reason = f"hold-short-winning-{int(mid_now*100)}c"
                                    # If price is very high (>=85c), max loss is small - let it expire
                                    elif mid_now >= DEFAULT_SAFE_WIN_THRESHOLD:
                                        should_flatten = False
                                        hold_reason = f"hold-short-small-loss-{int(mid_now*100)}c"
                                
                                if should_flatten:
                                    # Only flatten uncertain positions (30-70c range)
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
                                    models.entry_price = None
                                    trade_count += 1
                                else:
                                    log.info("Holding position %s: %s (pos=%d, mid=%.2f)",
                                             ticker, hold_reason, pos_here, mid_now)
                        continue

                    time_factor = _compute_time_factor(
                        seconds_to_expiry=seconds_to_expiry,
                        stop_quoting_seconds=DEFAULT_STOP_QUOTING_SECONDS,
                    )

                    ob = await client.get_orderbook(ticker)
                    prices = compute_best_prices(ob)
                    if prices.best_yes_bid is None or prices.best_yes_ask is None:
                        continue

                    best_bid = int(prices.best_yes_bid)
                    best_ask = int(prices.best_yes_ask)
                    mid_now = (best_bid + best_ask) / 200.0
                    portfolio.mark_mid(ticker=ticker, mid=mid_now)

                    # --- Update all models ---
                    
                    # 1. Order Flow Imbalance
                    # Get orderbook depth for OFI
                    bid_size = sum(
                        level[1] for level in (ob.get("yes") or []) 
                        if level[0] == best_bid
                    ) if ob.get("yes") else 1
                    ask_size = sum(
                        level[1] for level in (ob.get("no") or [])
                        if (100 - level[0]) == best_ask
                    ) if ob.get("no") else 1
                    
                    ofi_snapshot = OrderbookSnapshot(
                        best_bid=best_bid,
                        best_ask=best_ask,
                        bid_size=bid_size,
                        ask_size=ask_size,
                    )
                    ofi_signal = models.ofi.update(ofi_snapshot)
                    
                    # 2. GARCH Volatility
                    vol_garch = models.garch.update(mid_now)
                    
                    # 3. Regime Detection
                    regime_info = models.regime.update(mid_now)
                    regime = regime_info.regime
                    hurst = regime_info.hurst
                    
                    # --- Simulate fills on existing orders ---
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    existing = list(engine.orders_by_ticker.get(ticker) or [])
                    
                    for o in existing:
                        if o.action == "buy":
                            if best_ask <= o.price_cents:
                                # Fill at OUR bid price (maker order filled)
                                fill_c = o.price_cents
                                fill_cost = (fill_c / 100.0) * o.count
                                fee = _fee_dollars(price_cents=fill_c, count=o.count, maker=True)
                                portfolio.fees_paid += fee
                                portfolio.cash -= (fill_cost + fee)
                                
                                old_pos = int(portfolio.yes_pos.get(ticker, 0))
                                portfolio.yes_pos[ticker] = old_pos + o.count
                                engine.remove(o)
                                trade_count += 1
                                trade_stats.total_buys += 1
                                
                                # Track entry for PnL calculation
                                if old_pos <= 0 and portfolio.yes_pos[ticker] > 0:
                                    # Opened or added to long position
                                    entry_prices.setdefault(ticker, []).append((fill_c / 100.0, o.count))
                                elif old_pos < 0:
                                    # Covering short - calculate PnL
                                    cover_qty = min(o.count, abs(old_pos))
                                    if ticker in entry_prices and entry_prices[ticker]:
                                        entry = entry_prices[ticker].pop(0)
                                        pnl = (entry[0] - fill_c / 100.0) * cover_qty - fee
                                        trade_stats.record_round_trip(pnl)
                                
                                if models.entry_price is None:
                                    models.entry_price = mid_now
                                
                                # V8: Track position for exit management
                                new_pos = portfolio.yes_pos[ticker]
                                if new_pos > 0 and (models.position_info is None or models.position_info.quantity <= 0):
                                    # Opened new long position
                                    models.position_info = PositionInfo(
                                        entry_price_cents=fill_c,
                                        entry_time=now,
                                        quantity=new_pos,
                                        high_water_cents=best_bid,
                                    )
                                elif new_pos > 0 and models.position_info is not None:
                                    # Added to long - update quantity
                                    models.position_info.quantity = new_pos
                                elif new_pos == 0 and models.position_info is not None:
                                    # Closed position
                                    models.position_info = None
                                
                                # Record fill for fill probability calibration
                                models.fill_prob.record_fill(
                                    quote_price_cents=o.price_cents,
                                    mid_at_quote_cents=int(mid_now * 100),
                                    side="bid",
                                    time_to_fill_seconds=poll_seconds,
                                    filled=True,
                                )

                                trades_w.writerow([
                                    now.isoformat(), ticker, "buy", fill_c, o.count,
                                    f"{fee:.2f}", f"{portfolio.cash:.4f}",
                                    portfolio.yes_pos[ticker], f"{portfolio.equity():.4f}", "",
                                ])

                        elif o.action == "sell":
                            if best_bid >= o.price_cents:
                                # Fill at OUR ask price (maker order filled)
                                fill_c = o.price_cents
                                fill_credit = (fill_c / 100.0) * o.count
                                fee = _fee_dollars(price_cents=fill_c, count=o.count, maker=True)
                                portfolio.fees_paid += fee
                                portfolio.cash += (fill_credit - fee)
                                
                                old_pos = int(portfolio.yes_pos.get(ticker, 0))
                                portfolio.yes_pos[ticker] = old_pos - o.count
                                engine.remove(o)
                                trade_count += 1
                                trade_stats.total_sells += 1
                                
                                # Track entry for PnL calculation
                                if old_pos >= 0 and portfolio.yes_pos[ticker] < 0:
                                    # Opened or added to short position
                                    entry_prices.setdefault(ticker, []).append((fill_c / 100.0, o.count))
                                elif old_pos > 0:
                                    # Closing long - calculate PnL
                                    close_qty = min(o.count, old_pos)
                                    if ticker in entry_prices and entry_prices[ticker]:
                                        entry = entry_prices[ticker].pop(0)
                                        pnl = (fill_c / 100.0 - entry[0]) * close_qty - fee
                                        trade_stats.record_round_trip(pnl)
                                
                                if models.entry_price is None:
                                    models.entry_price = mid_now
                                
                                # V8: Track position for exit management
                                new_pos = portfolio.yes_pos[ticker]
                                if new_pos < 0 and (models.position_info is None or models.position_info.quantity >= 0):
                                    # Opened new short position
                                    models.position_info = PositionInfo(
                                        entry_price_cents=fill_c,
                                        entry_time=now,
                                        quantity=new_pos,
                                        high_water_cents=best_ask,
                                    )
                                elif new_pos < 0 and models.position_info is not None:
                                    # Added to short - update quantity
                                    models.position_info.quantity = new_pos
                                elif new_pos == 0 and models.position_info is not None:
                                    # Closed position
                                    models.position_info = None
                                
                                models.fill_prob.record_fill(
                                    quote_price_cents=o.price_cents,
                                    mid_at_quote_cents=int(mid_now * 100),
                                    side="ask",
                                    time_to_fill_seconds=poll_seconds,
                                    filled=True,
                                )

                                trades_w.writerow([
                                    now.isoformat(), ticker, "sell", fill_c, o.count,
                                    f"{fee:.2f}", f"{portfolio.cash:.4f}",
                                    portfolio.yes_pos[ticker], f"{portfolio.equity():.4f}", "",
                                ])

                    # --- V8: ACTIVE EXIT MANAGEMENT ---
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    
                    # Log warning if we have position but no position_info (shouldn't happen with sync)
                    if cur_pos != 0 and models.position_info is None:
                        log.warning("[EXIT MGMT] %s: have pos=%d but NO position_info! Position sync may have failed.", 
                                    ticker, cur_pos)
                    
                    if cur_pos != 0 and models.position_info is not None:
                        pos_info = models.position_info
                        exit_reason: str | None = None
                        
                        # Update high water mark
                        if cur_pos > 0:
                            pos_info.update_high_water(best_bid)
                            current_price = best_bid
                        else:
                            pos_info.update_high_water(best_ask)
                            current_price = best_ask
                        
                        # Calculate profit metrics
                        if cur_pos > 0:  # Long position
                            profit_cents = best_bid - pos_info.entry_price_cents
                            # ROI = profit / cost (cost = entry price for long)
                            roi = profit_cents / pos_info.entry_price_cents if pos_info.entry_price_cents > 0 else 0
                        else:  # Short position
                            profit_cents = pos_info.entry_price_cents - best_ask
                            # ROI = profit / risk (risk = 100 - entry price for short)
                            risk_cents = 100 - pos_info.entry_price_cents
                            roi = profit_cents / risk_cents if risk_cents > 0 else 0
                        
                        # 1. TAKE PROFIT CHECK (cents-based)
                        if profit_cents >= DEFAULT_TAKE_PROFIT_CENTS:
                            exit_reason = f"take-profit-{profit_cents}c"
                        
                        # 2. TAKE PROFIT CHECK (percentage-based)
                        if exit_reason is None and roi >= DEFAULT_TAKE_PROFIT_PCT:
                            exit_reason = f"take-profit-{roi*100:.0f}%"
                        
                        # 3. TIME-BASED PROFIT TAKING - near expiry, lock in gains
                        if exit_reason is None and seconds_to_expiry <= DEFAULT_EXPIRY_PROFIT_TAKE_SECONDS:
                            if roi >= DEFAULT_EXPIRY_MIN_PROFIT_PCT:
                                exit_reason = f"expiry-profit-{roi*100:.0f}%-{seconds_to_expiry}s"
                        
                        # 4. TRAILING STOP CHECK
                        if exit_reason is None:
                            if cur_pos > 0:  # Long - check if price dropped from high
                                drop_from_high = pos_info.high_water_cents - best_bid
                                if drop_from_high >= DEFAULT_TRAILING_STOP_CENTS:
                                    exit_reason = f"trailing-stop-{drop_from_high}c"
                            else:  # Short - check if price rose from low
                                rise_from_low = best_ask - pos_info.high_water_cents
                                if rise_from_low >= DEFAULT_TRAILING_STOP_CENTS:
                                    exit_reason = f"trailing-stop-{rise_from_low}c"
                        
                        # 5. STOP LOSS CHECK (tighter than old auto-flatten)
                        if exit_reason is None:
                            if cur_pos > 0:
                                loss_cents = pos_info.entry_price_cents - best_bid
                                if loss_cents >= DEFAULT_STOP_LOSS_CENTS:
                                    exit_reason = f"stop-loss-{loss_cents}c"
                            else:
                                loss_cents = best_ask - pos_info.entry_price_cents
                                if loss_cents >= DEFAULT_STOP_LOSS_CENTS:
                                    exit_reason = f"stop-loss-{loss_cents}c"
                        
                        # Log exit check status (helps debugging)
                        if exit_reason is None:
                            if cur_pos > 0:
                                loss_c = pos_info.entry_price_cents - best_bid
                            else:
                                loss_c = best_ask - pos_info.entry_price_cents
                            log.debug("[EXIT CHECK] %s: pos=%d entry=%dc current=%dc profit=%dc roi=%.1f%% loss=%dc hwm=%dc - NO EXIT",
                                      ticker, cur_pos, pos_info.entry_price_cents, current_price, 
                                      profit_cents, roi*100, loss_c, pos_info.high_water_cents)
                        
                        # EXECUTE EXIT if triggered
                        if exit_reason is not None:
                            log.info("[EXIT TRIGGERED] %s: %s (entry=%dc, current=%dc, roi=%.1f%%)",
                                     ticker, exit_reason, pos_info.entry_price_cents, current_price, roi*100)
                            
                            if live_mode:
                                # ======= LIVE MODE: Place actual closing order =======
                                if cur_pos > 0:
                                    # Close long by selling YES
                                    try:
                                        if not dry_run:
                                            resp = await client.create_order(
                                                ticker=ticker,
                                                side="yes",
                                                action="sell",
                                                count=cur_pos,
                                                order_type="limit",
                                                yes_price=best_bid,  # Sell at bid for immediate fill
                                                client_order_id=str(uuid.uuid4()),
                                            )
                                            order_id = resp.get("order", {}).get("order_id") or resp.get("order_id", "")
                                            log.info("[LIVE EXIT] SELL %d YES @ %dc -> %s (%s)",
                                                     cur_pos, best_bid, order_id, exit_reason)
                                        else:
                                            log.info("[DRY-RUN EXIT] Would SELL %d YES @ %dc (%s)",
                                                     cur_pos, best_bid, exit_reason)
                                    except Exception as e:
                                        log.error("[LIVE EXIT] Failed to close long: %s", e)
                                        continue
                                else:
                                    # Close short by buying YES
                                    cover = abs(cur_pos)
                                    try:
                                        if not dry_run:
                                            resp = await client.create_order(
                                                ticker=ticker,
                                                side="yes",
                                                action="buy",
                                                count=cover,
                                                order_type="limit",
                                                yes_price=best_ask,  # Buy at ask for immediate fill
                                                client_order_id=str(uuid.uuid4()),
                                            )
                                            order_id = resp.get("order", {}).get("order_id") or resp.get("order_id", "")
                                            log.info("[LIVE EXIT] BUY %d YES @ %dc -> %s (%s)",
                                                     cover, best_ask, order_id, exit_reason)
                                        else:
                                            log.info("[DRY-RUN EXIT] Would BUY %d YES @ %dc (%s)",
                                                     cover, best_ask, exit_reason)
                                    except Exception as e:
                                        log.error("[LIVE EXIT] Failed to close short: %s", e)
                                        continue
                                
                                # Clear position tracking (order placed, assume it fills)
                                portfolio.yes_pos[ticker] = 0
                                models.entry_price = None
                                
                                # V8.5: Track stop-loss/trailing-stop for cooldown
                                if "stop-loss" in exit_reason or "trailing-stop" in exit_reason:
                                    models.last_stop_loss_time = time.monotonic()
                                    models.last_stop_loss_direction = "long" if cur_pos > 0 else "short"
                                    models.consecutive_losses += 1
                                    log.warning("[LOSS TRACKER] %s: consecutive_losses=%d, direction=%s, cooldown=%.0fs",
                                               ticker, models.consecutive_losses, models.last_stop_loss_direction,
                                               DEFAULT_STOP_LOSS_COOLDOWN_SECONDS)
                                else:
                                    # Profitable exit - reset loss counter
                                    models.consecutive_losses = 0
                                
                                models.position_info = None
                                trade_count += 1
                                continue
                            else:
                                # ======= PAPER MODE: Simulate the exit =======
                                if cur_pos > 0:
                                    fill_c = best_bid
                                    fill_credit = (fill_c / 100.0) * cur_pos
                                    fee = _fee_dollars(price_cents=fill_c, count=cur_pos, maker=False)
                                    portfolio.fees_paid += fee
                                    portfolio.cash += (fill_credit - fee)
                                    
                                    pnl = (fill_c - pos_info.entry_price_cents) / 100.0 * cur_pos - fee
                                    trade_stats.record_round_trip(pnl)
                                    
                                    trades_w.writerow([
                                        now.isoformat(), ticker, "sell", fill_c, cur_pos,
                                        f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                                        f"{portfolio.equity():.4f}", exit_reason,
                                    ])
                                else:
                                    cover = abs(cur_pos)
                                    fill_c = best_ask
                                    fill_cost = (fill_c / 100.0) * cover
                                    fee = _fee_dollars(price_cents=fill_c, count=cover, maker=False)
                                    portfolio.fees_paid += fee
                                    portfolio.cash -= (fill_cost + fee)
                                    
                                    pnl = (pos_info.entry_price_cents - fill_c) / 100.0 * cover - fee
                                    trade_stats.record_round_trip(pnl)
                                    
                                    trades_w.writerow([
                                        now.isoformat(), ticker, "buy", fill_c, cover,
                                        f"{fee:.2f}", f"{portfolio.cash:.4f}", 0,
                                        f"{portfolio.equity():.4f}", exit_reason,
                                    ])
                                
                                portfolio.yes_pos[ticker] = 0
                                models.entry_price = None
                                models.position_info = None
                                trade_count += 1
                                log.info("V8 Exit: %s %s (entry=%dc, exit=%dc, roi=%.1f%%)", 
                                         ticker, exit_reason, pos_info.entry_price_cents, fill_c, roi*100)
                                continue

                    # --- Auto-flatten on loss (with smart exception for near-wins) ---
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    if cur_pos != 0 and models.entry_price is not None:
                        unrealized = (mid_now - models.entry_price) * cur_pos
                        if unrealized < -DEFAULT_AUTO_FLATTEN_LOSS_DOLLARS:
                            # Check if position is actually in a "safe" zone
                            # Don't panic flatten if we're likely to win at expiry
                            should_panic_flatten = True
                            
                            if cur_pos > 0:  # Long YES
                                # If price is very high, we're actually winning - don't flatten!
                                if mid_now >= DEFAULT_SAFE_WIN_THRESHOLD:
                                    should_panic_flatten = False
                                    log.info("Skipping panic flatten: long @ %.2f is winning", mid_now)
                            else:  # Short YES
                                # If price is very low, we're actually winning - don't flatten!
                                if mid_now <= DEFAULT_SAFE_LOSS_THRESHOLD:
                                    should_panic_flatten = False
                                    log.info("Skipping panic flatten: short @ %.2f is winning", mid_now)
                            
                            if should_panic_flatten:
                                if cur_pos > 0:
                                    fill_c = best_bid
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
                                    fill_c = best_ask
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
                                models.entry_price = None
                                trade_count += 1
                                log.warning("Auto-flattened %s due to loss", ticker)
                                continue

                    # --- Compute quotes using new models ---
                    engine.cancel_all(ticker=ticker)

                    # ======= BTC PRICE SIGNAL (V6/V7) =======
                    # Fetch BTC signal FIRST so we can integrate into p_fair
                    btc_fair_prob: float | None = None
                    btc_signal = None
                    if btc_feed is not None:
                        try:
                            strike = float(m.get("floor_strike", 0))
                            if strike > 0:
                                btc_snapshot = await btc_feed.get_price()
                                btc_signal = btc_feed.compute_strike_signal(
                                    strike=strike,
                                    seconds_to_expiry=seconds_to_expiry,
                                )
                                if btc_signal is not None:
                                    btc_fair_prob = btc_signal.fair_prob
                        except Exception as e:
                            log.debug("BTC feed error (continuing): %s", e)

                    # Regime-aware fair price (now includes BTC signal - V7)
                    raw_p_fair = compute_regime_fair_price(
                        mid_now=mid_now,
                        mid_prev=models.prev_mid,
                        regime=regime,
                        regime_momentum_weight=models.regime.momentum_weight(),
                        ofi_signal=ofi_signal,
                        btc_fair_prob=btc_fair_prob,
                    )
                    models.prev_mid = mid_now
                    
                    # Apply EMA smoothing to p_fair (V7) - prevents whiplash
                    p_fair = models.smooth_p_fair(raw_p_fair)
                    
                    # Track sustained edge (V7)
                    models.update_edge_streaks(mid_now, p_fair)

                    # Avellaneda-Stoikov optimal quotes
                    cur_inventory = int(portfolio.yes_pos.get(ticker, 0))
                    bid_p, ask_p = avellaneda.optimal_quotes(
                        mid_price=p_fair,
                        inventory=cur_inventory,
                        volatility=vol_garch,
                        time_remaining=time_factor,
                    )

                    # Additional OFI-based skew (anticipate direction)
                    ofi_skew = ofi_signal * 0.02  # 2 cents max
                    bid_p += ofi_skew
                    ask_p += ofi_skew

                    bid_c = _round_price_cents(bid_p)
                    ask_c = _round_price_cents(ask_p)

                    # Ensure valid spread
                    if bid_c >= ask_c:
                        bid_c = max(1, ask_c - 1)

                    # Stay behind top of book (maker behavior)
                    behind = max(0, behind_top_cents)
                    bid_c = min(bid_c, max(1, best_bid - behind))
                    ask_c = max(ask_c, min(99, best_ask + behind))

                    if bid_c >= ask_c:
                        continue

                    # Kelly-based sizing
                    buy_size_dollars = kelly.optimal_size_buy_yes(
                        fair_prob=p_fair,
                        bid_price=bid_c / 100.0,
                        bankroll=bankroll_dollars,
                    )
                    sell_size_dollars = kelly.optimal_size_sell_yes(
                        fair_prob=p_fair,
                        ask_price=ask_c / 100.0,
                        bankroll=bankroll_dollars,
                    )

                    # Convert to contracts (each ~$1 notional)
                    # Use minimum of Kelly size and base order, ensure at least 1
                    buy_size = max(1, min(int(buy_size_dollars), int(base_order_dollars * 2)))
                    sell_size = max(1, min(int(sell_size_dollars), int(base_order_dollars * 2)))

                    # Scale down near expiry
                    buy_size = max(1, int(buy_size * max(0.3, time_factor)))
                    sell_size = max(1, int(sell_size * max(0.3, time_factor)))

                    # Position caps
                    cur_pos = int(portfolio.yes_pos.get(ticker, 0))
                    cur_total_abs = _total_abs_pos(portfolio.yes_pos)

                    per_cap = max(1, int(max_abs_pos_per_market * max(0.3, time_factor)))
                    tot_cap = max(1, int(max_abs_pos_total * max(0.3, time_factor)))

                    allow_buy = True
                    allow_sell = True

                    if cur_pos >= per_cap:
                        allow_buy = False
                    if cur_pos <= -per_cap:
                        allow_sell = False
                    if cur_total_abs >= tot_cap:
                        if cur_pos > 0:
                            allow_buy = False
                        elif cur_pos < 0:
                            allow_sell = False
                        else:
                            allow_buy = False
                            allow_sell = False

                    # ======= TREND PROTECTION =======
                    # In trending markets with strong directional OFI, block counter-trend trades
                    is_strong_trend = (regime == Regime.TRENDING and hurst >= DEFAULT_TREND_HURST_THRESHOLD)
                    
                    if is_strong_trend:
                        # OFI > 0 means buying pressure -> price likely to rise -> don't short
                        if ofi_signal > DEFAULT_OFI_BLOCK_THRESHOLD:
                            allow_sell = False
                            log.debug("Trend protection: blocking SELL (OFI=%.3f, hurst=%.3f)", 
                                      ofi_signal, hurst)
                        # OFI < 0 means selling pressure -> price likely to fall -> don't buy
                        elif ofi_signal < -DEFAULT_OFI_BLOCK_THRESHOLD:
                            allow_buy = False
                            log.debug("Trend protection: blocking BUY (OFI=%.3f, hurst=%.3f)",
                                      ofi_signal, hurst)
                    
                    # ======= MOMENTUM TRACKING (V8.5 - NO LONGER BLOCKS) =======
                    # We track momentum but don't block based on it
                    # Instead, p_fair calculation uses momentum to FOLLOW the market
                    models.update_momentum(mid_now)
                    momentum = models.momentum_change()
                    
                    # ======= V8.5 MOMENTUM HEDGES =======
                    
                    # 1. STOP-LOSS COOLDOWN - don't re-enter same direction right after stop
                    time_since_stop = time.monotonic() - models.last_stop_loss_time
                    if time_since_stop < DEFAULT_STOP_LOSS_COOLDOWN_SECONDS:
                        if models.last_stop_loss_direction == "long" and allow_buy:
                            allow_buy = False
                            log.debug("[COOLDOWN] Blocking BUY - stopped out of long %.0fs ago (need %.0fs)",
                                     time_since_stop, DEFAULT_STOP_LOSS_COOLDOWN_SECONDS)
                        if models.last_stop_loss_direction == "short" and allow_sell:
                            allow_sell = False
                            log.debug("[COOLDOWN] Blocking SELL - stopped out of short %.0fs ago (need %.0fs)",
                                     time_since_stop, DEFAULT_STOP_LOSS_COOLDOWN_SECONDS)
                    
                    # 2. MAX LOSSES PER MARKET - stop trading after too many consecutive losses
                    if models.consecutive_losses >= DEFAULT_MAX_LOSSES_PER_MARKET:
                        allow_buy = False
                        allow_sell = False
                        log.warning("[MAX LOSSES] Blocking all trades on %s - %d consecutive losses",
                                   ticker, models.consecutive_losses)
                    
                    # 3. EXTREME PRICE AVOIDANCE - don't chase exhausted moves
                    if mid_now >= DEFAULT_EXTREME_PRICE_BUY_MAX:
                        allow_buy = False
                        log.debug("[EXTREME PRICE] Blocking BUY - price %.0fc too high (max %.0fc)",
                                 mid_now * 100, DEFAULT_EXTREME_PRICE_BUY_MAX * 100)
                    if mid_now <= DEFAULT_EXTREME_PRICE_SELL_MIN:
                        allow_sell = False
                        log.debug("[EXTREME PRICE] Blocking SELL - price %.0fc too low (min %.0fc)",
                                 mid_now * 100, DEFAULT_EXTREME_PRICE_SELL_MIN * 100)
                    
                    # 4. MINIMUM OFI STRENGTH - avoid trading on noise
                    if abs(ofi_signal) < DEFAULT_MIN_OFI_STRENGTH:
                        allow_buy = False
                        allow_sell = False
                        log.debug("[WEAK OFI] Blocking trades - OFI %.3f too weak (need %.3f)",
                                 abs(ofi_signal), DEFAULT_MIN_OFI_STRENGTH)
                    
                    # ======= WATERFALL PROTECTION (V8 fix) =======
                    # Track price for crash detection and block buying during crashes
                    models.update_waterfall(mid_now)
                    if models.is_waterfall():
                        allow_buy = False
                        log.warning("[WATERFALL] Blocking BUY: price crashed %.0f%% in last %d observations",
                                   DEFAULT_WATERFALL_DROP_PCT * 100, DEFAULT_WATERFALL_WINDOW)
                    
                    # ======= POSITION-AWARE ADDING =======
                    # V8.4: STRICT - If we have a position, DON'T add more - just manage it
                    # This prevents the order spam when waiting for fills
                    
                    if cur_pos > 0:  # Already LONG - don't add more, let exit management handle it
                        allow_buy = False
                        log.debug("Position-aware: blocking BUY - already long %d contracts", cur_pos)
                    
                    elif cur_pos < 0:  # Already SHORT - don't add more, let exit management handle it
                        allow_sell = False
                        log.debug("Position-aware: blocking SELL - already short %d contracts", abs(cur_pos))
                    
                    # ======= OFI DIRECTION (V8.5 - FOLLOW, DON'T BLOCK) =======
                    # OFI now determines our DIRECTION via p_fair calculation
                    # We don't block based on OFI - we follow it
                    # Strong positive OFI -> p_fair > mid -> BUY signal
                    # Strong negative OFI -> p_fair < mid -> SELL signal
                    
                    # ======= EDGE REQUIREMENT =======
                    # Only trade if we have sufficient edge (p_fair differs from mid)
                    edge_cents = abs(p_fair - mid_now) * 100
                    if edge_cents < DEFAULT_MIN_EDGE_CENTS:
                        # No clear edge - skip this quote cycle
                        log.debug("Edge filter: skipping (edge=%.1fc < %.1fc required)",
                                  edge_cents, DEFAULT_MIN_EDGE_CENTS)
                        continue
                    
                    # ======= SUSTAINED EDGE REQUIREMENT (V7) =======
                    # For NEW positions, require edge to persist for multiple polls
                    # This prevents entering on spike signals that immediately reverse
                    if cur_pos == 0:
                        has_sustained_buy = models.has_sustained_buy_edge()
                        has_sustained_sell = models.has_sustained_sell_edge()
                        
                        if not has_sustained_buy:
                            allow_buy = False
                            log.debug("Sustained edge: blocking new BUY (streak=%d < %d required)",
                                      models.buy_edge_streak, DEFAULT_SUSTAINED_EDGE_POLLS)
                        
                        if not has_sustained_sell:
                            allow_sell = False
                            log.debug("Sustained edge: blocking new SELL (streak=%d < %d required)",
                                      models.sell_edge_streak, DEFAULT_SUSTAINED_EDGE_POLLS)
                    
                    # ======= BTC PRICE SIGNAL (V8.5 - INFORMATIONAL ONLY) =======
                    # BTC signal is now only used informatively in p_fair calculation
                    # We no longer use it to BLOCK trades - OFI is more real-time
                    # The BTC signal often lags and can disagree with current orderbook flow
                    
                    # ======= SIZE IN TRENDS (V8.5 - NO REDUCTION) =======\n                    # We now WANT to follow trends, so no size reduction

                    # ======= DECISION LOGGING (V8.2 - MOVED AFTER ALL FILTERS) =======
                    # Log FINAL decision after all filters are applied
                    btc_fair_str = f"{btc_fair_prob:.2f}" if btc_fair_prob is not None else "N/A"
                    pos_info_str = "none"
                    if models.position_info is not None:
                        pi = models.position_info
                        pos_info_str = f"entry={pi.entry_price_cents}c,qty={pi.quantity},hwm={pi.high_water_cents}c"
                    
                    # Calculate edge direction for clarity
                    buy_edge = p_fair - mid_now  # Positive = undervalued, good to buy
                    sell_edge = mid_now - p_fair  # Positive = overvalued, good to sell
                    edge_dir = "BUY" if buy_edge > 0.01 else ("SELL" if sell_edge > 0.01 else "FLAT")
                    
                    log.info("[DECISION] mid=%.2f p_fair=%.2f btc_fair=%s edge=%s | pos=%d pos_info=[%s] | regime=%s ofi=%.3f mom=%.3f | buy=%s sell=%s",
                             mid_now, p_fair, btc_fair_str, edge_dir, cur_pos, pos_info_str, regime.value, ofi_signal, momentum,
                             "YES" if allow_buy else "no", "YES" if allow_sell else "no")

                    # Risk caps
                    buy_risk = _max_loss_yes_buy(yes_price_cents=bid_c, count=buy_size)
                    sell_risk = _max_loss_yes_sell(yes_price_cents=ask_c, count=sell_size)
                    add_risk = (buy_risk if allow_buy else 0.0) + (sell_risk if allow_sell else 0.0)

                    if not engine.can_place(ticker=ticker, add_risk=add_risk):
                        continue

                    # Place orders
                    ts_now = _utcnow()
                    
                    if live_mode:
                        # ======= V8.4: ORDER COOLDOWN CHECK =======
                        # Don't spam orders - wait for cooldown after placing
                        time_since_last_order = time.monotonic() - last_order_time.get(ticker, 0)
                        if time_since_last_order < ORDER_COOLDOWN_SECONDS:
                            remaining = ORDER_COOLDOWN_SECONDS - time_since_last_order
                            log.debug("[COOLDOWN] Skipping order on %s (%.1fs remaining)", ticker, remaining)
                            continue  # Skip this ticker until cooldown expires
                        
                        # ======= LIVE/DRY-RUN ORDER PLACEMENT =======
                        # Cancel existing orders for this ticker first
                        if ticker in live_orders:
                            for old_order_id in live_orders[ticker]:
                                try:
                                    if not dry_run:
                                        await client.cancel_order(old_order_id)
                                    log.debug("Cancelled order %s", old_order_id)
                                except Exception:
                                    pass  # Order may already be filled/cancelled
                            live_orders[ticker] = []
                        
                        if allow_buy:
                            if dry_run:
                                log.info("[DRY-RUN] Would BUY %d YES @ %dc on %s (risk=$%.2f)",
                                         buy_size, bid_c, ticker, buy_risk)
                            else:
                                try:
                                    resp = await client.create_order(
                                        ticker=ticker,
                                        side="yes",
                                        action="buy",
                                        count=buy_size,
                                        order_type="limit",
                                        yes_price=bid_c,
                                        client_order_id=str(uuid.uuid4()),
                                        post_only=True,
                                    )
                                    order_id = resp.get("order", {}).get("order_id") or resp.get("order_id", "")
                                    if order_id:
                                        live_orders.setdefault(ticker, []).append(order_id)
                                        last_order_time[ticker] = time.monotonic()  # Start cooldown
                                    log.info("[LIVE] BUY %d YES @ %dc on %s -> order_id=%s (cur_pos=%d)",
                                             buy_size, bid_c, ticker, order_id, cur_pos)
                                except Exception as e:
                                    log.error("[LIVE] BUY order failed: %s", e)
                        
                        if allow_sell:
                            if dry_run:
                                log.info("[DRY-RUN] Would SELL %d YES @ %dc on %s (risk=$%.2f)",
                                         sell_size, ask_c, ticker, sell_risk)
                            else:
                                try:
                                    resp = await client.create_order(
                                        ticker=ticker,
                                        side="yes",
                                        action="sell",
                                        count=sell_size,
                                        order_type="limit",
                                        yes_price=ask_c,
                                        client_order_id=str(uuid.uuid4()),
                                        post_only=True,
                                    )
                                    order_id = resp.get("order", {}).get("order_id") or resp.get("order_id", "")
                                    if order_id:
                                        live_orders.setdefault(ticker, []).append(order_id)
                                        last_order_time[ticker] = time.monotonic()  # Start cooldown
                                    log.info("[LIVE] SELL %d YES @ %dc on %s -> order_id=%s (cur_pos=%d)",
                                             sell_size, ask_c, ticker, order_id, cur_pos)
                                except Exception as e:
                                    log.error("[LIVE] SELL order failed: %s", e)
                    else:
                        # ======= PAPER TRADING (simulation) =======
                        if allow_buy:
                            buy_order = PaperOrder(
                                order_id=f"PAPER-{ticker}-B-{int(ts_now.timestamp() * 1000)}",
                                ticker=ticker,
                                action="buy",
                                price_cents=bid_c,
                                count=buy_size,
                                placed_ts=ts_now,
                            )
                            engine.place(buy_order, risk_dollars=buy_risk)

                        if allow_sell:
                            sell_order = PaperOrder(
                                order_id=f"PAPER-{ticker}-S-{int(ts_now.timestamp() * 1000)}",
                                ticker=ticker,
                                action="sell",
                                price_cents=ask_c,
                                count=sell_size,
                                placed_ts=ts_now,
                            )
                            engine.place(sell_order, risk_dollars=sell_risk)

                    # Use larger of buy/sell size for logging
                    size = max(buy_size if allow_buy else 0, sell_size if allow_sell else 0)
                    spread = (ask_c - bid_c) / 100.0

                    quotes_w.writerow([
                        ts_now.isoformat(), ticker, best_bid, best_ask,
                        f"{mid_now:.4f}", f"{p_fair:.4f}",
                        regime.value, f"{hurst:.3f}", f"{ofi_signal:.4f}",
                        f"{vol_garch:.6f}", f"{spread:.4f}",
                        bid_c, ask_c, size,
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

            elapsed = time.monotonic() - loop_started
            await asyncio.sleep(max(0.0, poll_seconds - elapsed))

    finally:
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
                log.exception("Flatten-at-end failed")

        try:
            trades_f.close()
            equity_f.close()
            quotes_f.close()
        finally:
            # Close BTC feed if initialized
            if btc_feed is not None:
                try:
                    await btc_feed.close()
                except Exception:
                    pass
            
            # Calculate and log final statistics
            duration_hours = duration_minutes / 60.0
            pnl_per_hour = portfolio.pnl() / duration_hours if duration_hours > 0 else 0.0
            
            log.info("="*60)
            log.info("PAPER TRADING RUN COMPLETE")
            log.info("="*60)
            log.info("Duration: %.1f minutes (%.2f hours)", duration_minutes, duration_hours)
            log.info("Final Equity: $%.2f | PnL: $%.2f (%.2f%%)", 
                     portfolio.equity(), portfolio.pnl(), 
                     (portfolio.pnl() / bankroll_dollars) * 100)
            log.info("PnL/Hour: $%.2f", pnl_per_hour)
            log.info("Total Fees Paid: $%.2f", portfolio.fees_paid)
            log.info("-"*60)
            log.info("Trade Statistics:")
            log.info("  %s", trade_stats.summary())
            log.info("  Entry Orders: %d buys, %d sells", 
                     trade_stats.total_buys, trade_stats.total_sells)
            log.info("="*60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-run V2 with statistical models")
    parser.add_argument("--assets", default=DEFAULT_ASSETS)
    parser.add_argument("--horizon-minutes", type=int, default=DEFAULT_HORIZON_MINUTES)
    parser.add_argument("--per-asset", type=int, default=DEFAULT_PER_ASSET)
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS)
    parser.add_argument("--duration-minutes", type=float, default=DEFAULT_DURATION_MINUTES)
    parser.add_argument("--bankroll", type=float, default=DEFAULT_BANKROLL_DOLLARS)
    parser.add_argument("--base-order", type=float, default=DEFAULT_BASE_ORDER_DOLLARS)
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--max-abs-pos-per-market", type=int, default=DEFAULT_MAX_ABS_POS_PER_MARKET)
    parser.add_argument("--max-abs-pos-total", type=int, default=DEFAULT_MAX_ABS_POS_TOTAL)
    parser.add_argument("--behind-top-cents", type=int, default=DEFAULT_BEHIND_TOP_CENTS)
    parser.add_argument("--flatten-at-end", action=argparse.BooleanOptionalAction, default=DEFAULT_FLATTEN_AT_END)
    
    # Live trading mode
    parser.add_argument("--live", action="store_true", 
                        help="Enable live trading mode (real orders)")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --live, log what would be traded without placing orders")

    args = parser.parse_args()
    
    # Apply conservative defaults for live mode
    if args.live and not args.dry_run:
        if args.max_abs_pos_per_market == DEFAULT_MAX_ABS_POS_PER_MARKET:
            args.max_abs_pos_per_market = DEFAULT_LIVE_MAX_POS_PER_MARKET
        if args.max_abs_pos_total == DEFAULT_MAX_ABS_POS_TOTAL:
            args.max_abs_pos_total = DEFAULT_LIVE_MAX_POS_TOTAL
        if args.base_order == DEFAULT_BASE_ORDER_DOLLARS:
            args.base_order = DEFAULT_LIVE_BASE_ORDER

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
            await run_paper_v2(
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
                live_mode=bool(args.live),
                dry_run=bool(args.dry_run),
            )
        finally:
            await client.aclose()

    asyncio.run(runner())


if __name__ == "__main__":
    main()
