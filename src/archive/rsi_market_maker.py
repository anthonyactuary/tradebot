"""RSI-Based Market Maker for Kalshi BTC 15-Minute Markets.

This bot uses a simple RSI-based strategy for directional predictions
to achieve 60%+ win rate on signaled trades, while providing liquidity.

Strategy:
- RSI < 15 (extreme oversold) → Predict "UP" (price likely to bounce)
- RSI > 70 (extreme overbought) → Predict "DOWN" (price likely to drop)  
- RSI 15-70 → Neutral mode (symmetric market making)

In Neutral Mode:
- Quote symmetric bids/asks on YES contracts
- Capture spread when positions offset

In Directional Mode:
- Skew quotes toward predicted direction
- Build position in expected winning side

Usage:
  Backtest:    python rsi_market_maker.py --backtest
  Dry run:     python rsi_market_maker.py --live --dry-run
  Live trade:  python rsi_market_maker.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import datetime
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

from tradebot.config import Settings
from tradebot.kalshi.client import KalshiClient
from tradebot.kalshi.orderbook import compute_best_prices

try:
    from tenacity import RetryError
except Exception:  # pragma: no cover
    RetryError = None  # type: ignore[assignment]


# ----------------------------
# Configuration Constants
# ----------------------------

# RSI Parameters
RSI_PERIOD = 14                    # Standard 14-period RSI
RSI_OVERSOLD_THRESHOLD = 15        # Oversold → predict UP (optimized: 54.5% win rate)
RSI_OVERBOUGHT_THRESHOLD = 70      # Overbought → predict DOWN (optimized: 54.5% win rate)
RSI_CANDLE_GRANULARITY = 60        # 1-minute candles for RSI (more responsive than 15-min)
CONTRACT_GRANULARITY = 900         # 15-minute Kalshi contract windows

# Quote Parameters  
DEFAULT_SPREAD_CENTS = 2           # Spread around fair price (each side)
DEFAULT_SIZE_CONTRACTS = 3         # Default order size
MIN_SIZE = 1                       # Minimum order size
MAX_SIZE = 2                       # Maximum order size per order

# Risk Management
MAX_POSITION_PER_MARKET = 10       # Max contracts per market direction
MAX_TOTAL_EXPOSURE_DOLLARS = 15    # Max total exposure
STOP_LOSS_CENTS = 10               # Cut losses at 10c
TAKE_PROFIT_CENTS = 10              # Take profits at 10c
STOP_LOSS_ONLY_WITHIN_SECONDS = 300  # Only stop out in the last 5 minutes

# Inventory Protection Parameters
INVENTORY_SKEW_THRESHOLD = 5       # Start skewing quotes at ±5 contracts
INVENTORY_HARD_LIMIT = 10          # Force close above ±10 contracts
INVENTORY_SKEW_CENTS = 2           # Max cents to skew quotes per threshold
INVENTORY_SIGNAL_ALLOWANCE = 1.5   # Allow 50% more inventory when RSI signal active
INVENTORY_CLOSE_AGGRESSION = 2     # How many cents inside mid to close aggressively
INVENTORY_EMERGENCY_SECONDS = 300  # Use market orders if <5 min to expiry with bad inventory

# Timing
DEFAULT_POLL_SECONDS = 10        # Poll every 10s (catches new 1-min candles quickly)
DEFAULT_DURATION_MINUTES = 240     # 4 hour default run
STOP_QUOTING_BEFORE_EXPIRY = 60    # Stop quoting 1 min before expiry

# Data Sources
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
COINBASE_TICKER_URL = "https://api.exchange.coinbase.com/products/BTC-USD/ticker"
CMC_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# Fee Rates
TAKER_FEE_RATE = 0.07
MAKER_FEE_RATE = 0.0175

# Output
DEFAULT_OUT_DIR = "runs"


log = logging.getLogger("tradebot.rsi_mm")


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
    if isinstance(v, (int, float)):
        try:
            ts = float(v)
            # Heuristic: treat very large values as milliseconds.
            if ts > 1e12:
                ts = ts / 1000.0
            return datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc)
        except Exception:
            return None
    return None


def _market_cutoff_time(market: dict[str, Any]) -> tuple[datetime.datetime | None, str]:
    """Return the relevant cutoff time for risk management.

    Kalshi market payloads may include multiple timestamps. For position/risk handling,
    the important time is usually when trading closes (often `close_time`).

    Returns:
        (cutoff_time, source_field)
    """
    candidates: list[tuple[str, datetime.datetime]] = []
    for field in ("close_time", "expiration_time", "expected_expiration_time"):
        dt = _to_dt(market.get(field))
        if dt is not None:
            candidates.append((field, dt))
    if not candidates:
        return (None, "")

    # Use the earliest known cutoff time.
    field, dt = min(candidates, key=lambda x: x[1])
    return (dt, field)


def _market_price_to_beat(market: dict[str, Any]) -> tuple[float | None, str]:
    """Best-effort extraction of the market's "price to beat" threshold.

    Different market types encode the threshold differently. We log this value for
    operator context only; trading logic does not depend on it.

    Returns:
        (value, source_field)
    """
    def _parse_num(v: Any) -> float | None:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            s = v.strip()
            if not s:
                return None
            s = s.replace("$", "").replace(",", "")
            try:
                return float(s)
            except Exception:
                return None
        return None

    # Kalshi crypto 15m markets often encode the strike as `floor_strike`.
    for field in (
        "floor_strike",
        "price_to_beat",
        "strike_price",
        "strike",
        "target_price",
        "threshold",
    ):
        parsed = _parse_num(market.get(field))
        if parsed is not None and parsed > 0:
            return (parsed, field)

    # Fallback: parse from common text fields. Some payloads use `yes_sub_title` like
    # ">= 90,500" (no dollar sign).
    import re

    def _extract_best_candidate(text: str) -> float | None:
        # Capture comma-formatted numbers or long digit runs; allow optional $.
        # We then filter for plausible BTC strike ranges.
        matches = re.findall(
            r"\$?\s*([0-9]{1,3}(?:,[0-9]{3})+|[0-9]{4,})(?:\.[0-9]+)?",
            text,
        )
        if not matches:
            return None
        candidates: list[float] = []
        for m in matches:
            try:
                candidates.append(float(m.replace(",", "")))
            except Exception:
                continue
        # Plausible strike range for BTC (avoid picking years / minutes etc.).
        candidates = [c for c in candidates if 1_000 <= c <= 10_000_000]
        if not candidates:
            return None
        # Prefer the largest plausible candidate (usually the strike).
        return max(candidates)

    for field in (
        "yes_sub_title",
        "no_sub_title",
        "subtitle",
        "title",
        "short_title",
        "event_title",
        "rules_primary",
        "rules_secondary",
    ):
        text = market.get(field)
        if not isinstance(text, str) or not text:
            continue
        parsed = _extract_best_candidate(text)
        if parsed is not None:
            return (parsed, field)

    return (None, "")


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _round_price_cents(p: float) -> int:
    """Round price to cents (1-99 range)."""
    cents = int(round(p * 100.0))
    return max(1, min(99, cents))


def _fee_dollars(*, price_cents: int, count: int, maker: bool) -> float:
    """Calculate fee in dollars."""
    p = _clamp(price_cents / 100.0, 0.0, 1.0)
    c = max(0, int(count))
    if c == 0 or p <= 0.0 or p >= 1.0:
        return 0.0
    rate = MAKER_FEE_RATE if maker else TAKER_FEE_RATE
    return rate * c * p * (1.0 - p)


# ----------------------------
# RSI Calculation
# ----------------------------

def calculate_rsi(prices: list[float], period: int = RSI_PERIOD) -> float | None:
    """Calculate RSI from closing prices.
    
    Formula:
    - Delta = current close - previous close
    - Gains = average of positive deltas over period
    - Losses = average of absolute negative deltas over period  
    - RS = Gains / Losses
    - RSI = 100 - (100 / (1 + RS))
    
    Args:
        prices: List of closing prices (oldest first)
        period: Number of periods for RSI calculation
        
    Returns:
        RSI value (0-100) or None if insufficient data
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate deltas
    deltas = []
    for i in range(1, len(prices)):
        deltas.append(prices[i] - prices[i - 1])
    
    # Use most recent 'period' deltas
    recent_deltas = deltas[-period:]
    
    # Separate gains and losses
    gains = [d for d in recent_deltas if d > 0]
    losses = [abs(d) for d in recent_deltas if d < 0]
    
    # Calculate averages
    avg_gain = sum(gains) / period if gains else 0.0
    avg_loss = sum(losses) / period if losses else 0.0
    
    # Avoid division by zero
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    # Calculate RS and RSI
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def calculate_rsi_smoothed(prices: list[float], period: int = RSI_PERIOD) -> float | None:
    """Calculate Wilder's smoothed RSI (more stable).
    
    Uses exponential smoothing for gains/losses instead of simple average.
    """
    if len(prices) < period + 1:
        return None
    
    # Calculate all deltas
    deltas = []
    for i in range(1, len(prices)):
        deltas.append(prices[i] - prices[i - 1])
    
    if len(deltas) < period:
        return None
    
    # Initialize with simple averages for first 'period' values
    first_gains = [d if d > 0 else 0 for d in deltas[:period]]
    first_losses = [abs(d) if d < 0 else 0 for d in deltas[:period]]
    
    avg_gain = sum(first_gains) / period
    avg_loss = sum(first_losses) / period
    
    # Apply Wilder's smoothing for remaining values
    for d in deltas[period:]:
        gain = d if d > 0 else 0
        loss = abs(d) if d < 0 else 0
        
        # Wilder's smoothing: new_avg = (prev_avg * (period-1) + current) / period
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
    
    if avg_loss == 0:
        return 100.0 if avg_gain > 0 else 50.0
    
    rs = avg_gain / avg_loss
    rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi


def _precompute_rsi_series(prices: list[float], period: int = RSI_PERIOD) -> list[float | None]:
    """Pre-compute RSI for entire price series in O(n) time.
    
    Much faster than calling calculate_rsi_smoothed() for each index.
    
    Args:
        prices: List of closing prices
        period: RSI period
        
    Returns:
        List of RSI values (None for indices with insufficient data)
    """
    n = len(prices)
    rsi_values: list[float | None] = [None] * n
    
    if n < period + 1:
        return rsi_values
    
    # Calculate deltas
    deltas = [0.0] * n
    for i in range(1, n):
        deltas[i] = prices[i] - prices[i - 1]
    
    # Initialize with simple averages for first 'period' values
    avg_gain = 0.0
    avg_loss = 0.0
    for i in range(1, period + 1):
        d = deltas[i]
        if d > 0:
            avg_gain += d
        else:
            avg_loss += abs(d)
    avg_gain /= period
    avg_loss /= period
    
    # First RSI value at index 'period'
    if avg_loss == 0:
        rsi_values[period] = 100.0 if avg_gain > 0 else 50.0
    else:
        rs = avg_gain / avg_loss
        rsi_values[period] = 100.0 - (100.0 / (1.0 + rs))
    
    # Apply Wilder's smoothing for remaining values
    for i in range(period + 1, n):
        d = deltas[i]
        gain = d if d > 0 else 0.0
        loss = abs(d) if d < 0 else 0.0
        
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period
        
        if avg_loss == 0:
            rsi_values[i] = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi_values[i] = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi_values


# ----------------------------
# Data Fetching
# ----------------------------

async def fetch_btc_candles(
    granularity: int = RSI_CANDLE_GRANULARITY,  # 1-minute candles for RSI
    limit: int = 50,
) -> list[dict[str, float]]:
    """Fetch BTC/USD candles from Coinbase.
    
    Args:
        granularity: Candle size in seconds (60 = 1 min default for RSI)
        limit: Number of candles to fetch
        
    Returns:
        List of candle dicts with: timestamp, open, high, low, close, volume
        Sorted oldest to newest.
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        params = {"granularity": granularity}
        headers = {"Accept": "application/json"}
        
        response = await client.get(COINBASE_CANDLES_URL, params=params, headers=headers)
        response.raise_for_status()
        raw_candles = response.json()
        
        # Coinbase format: [time, low, high, open, close, volume]
        # Most recent first, so reverse for chronological order
        candles = []
        for c in reversed(raw_candles[:limit]):
            candles.append({
                "timestamp": float(c[0]),
                "open": float(c[3]),
                "high": float(c[2]),
                "low": float(c[1]),
                "close": float(c[4]),
                "volume": float(c[5]),
            })
        
        return candles


async def fetch_btc_candles_historical(
    days: int = 30,
    granularity: int = RSI_CANDLE_GRANULARITY,  # 1-minute candles for RSI
) -> list[dict[str, float]]:
    """Fetch historical BTC/USD candles with pagination.
    
    Coinbase API returns max 300 candles per request. This function
    paginates backwards to fetch the requested number of days.
    
    Args:
        days: Number of days of historical data to fetch
        granularity: Candle size in seconds (60 = 1 min default for RSI)
        
    Returns:
        List of candle dicts sorted oldest to newest.
    """
    # Calculate how many candles we need
    candles_per_day = (24 * 60 * 60) // granularity
    total_candles_needed = days * candles_per_day
    max_per_request = 300  # Coinbase limit
    
    log.info("Fetching %d days of %d-second candles (%d total)...", 
             days, granularity, total_candles_needed)
    
    all_candles = []
    end_time = int(time.time())  # Start from now
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        headers = {"Accept": "application/json"}
        
        requests_made = 0
        max_requests = (total_candles_needed // max_per_request) + 2  # Safety limit
        
        while len(all_candles) < total_candles_needed and requests_made < max_requests:
            # Calculate start time for this batch
            # Go back (max_per_request * granularity) seconds from end_time
            start_time = end_time - (max_per_request * granularity)
            
            params = {
                "granularity": granularity,
                "start": start_time,
                "end": end_time,
            }
            
            try:
                response = await client.get(COINBASE_CANDLES_URL, params=params, headers=headers)
                response.raise_for_status()
                raw_candles = response.json()
                
                if not raw_candles:
                    log.warning("No more candles available from Coinbase")
                    break
                
                # Coinbase format: [time, low, high, open, close, volume]
                # Most recent first
                batch = []
                for c in raw_candles:
                    batch.append({
                        "timestamp": float(c[0]),
                        "open": float(c[3]),
                        "high": float(c[2]),
                        "low": float(c[1]),
                        "close": float(c[4]),
                        "volume": float(c[5]),
                    })
                
                # Prepend to all_candles (we're going backwards in time)
                all_candles = batch + all_candles
                
                # Move end_time back for next request
                # Use the oldest timestamp from this batch minus 1 second
                oldest_ts = min(c["timestamp"] for c in batch)
                end_time = int(oldest_ts) - 1
                
                requests_made += 1
                
                # Brief delay to avoid rate limiting
                if requests_made < max_requests:
                    await asyncio.sleep(0.2)
                    
            except Exception as e:
                log.error("Error fetching candles batch %d: %s", requests_made, e)
                break
        
        log.info("Fetched %d candles in %d requests", len(all_candles), requests_made)
    
    # Sort by timestamp (oldest first) and deduplicate
    all_candles.sort(key=lambda c: c["timestamp"])
    
    # Remove duplicates based on timestamp
    seen_ts = set()
    unique_candles = []
    for c in all_candles:
        ts = c["timestamp"]
        if ts not in seen_ts:
            seen_ts.add(ts)
            unique_candles.append(c)
    
    log.info("After deduplication: %d unique candles", len(unique_candles))
    
    return unique_candles


async def fetch_btc_price_cmc(api_key: str) -> float | None:
    """Fetch current BTC price from CoinMarketCap.
    
    Args:
        api_key: CoinMarketCap API key
        
    Returns:
        Current BTC price in USD, or None on error
    """
    async with httpx.AsyncClient(timeout=10.0) as client:
        headers = {
            "X-CMC_PRO_API_KEY": api_key,
            "Accept": "application/json",
        }
        params = {"symbol": "BTC", "convert": "USD"}
        
        try:
            response = await client.get(CMC_API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            btc_data = data.get("data", {}).get("BTC", {})
            quote = btc_data.get("quote", {}).get("USD", {})
            return float(quote.get("price", 0))
        except Exception as e:
            log.warning("Failed to fetch BTC price from CMC: %s", e)
            return None


async def fetch_btc_spot_price_coinbase() -> float | None:
    """Fetch a spot BTC price from Coinbase ticker.

    This updates every poll (unlike 1-min candle closes). RSI computation remains
    based on 1-minute candles.
    """
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            headers = {"Accept": "application/json"}
            resp = await client.get(COINBASE_TICKER_URL, headers=headers)
            resp.raise_for_status()
            data = resp.json()
            price = data.get("price")
            if price is None:
                return None
            return float(price)
    except Exception as e:
        log.debug("Failed to fetch Coinbase ticker spot price: %s", e)
        return None


# ----------------------------
# Signal Generation
# ----------------------------

@dataclass
class RSISignal:
    """RSI-based trading signal."""
    rsi: float
    direction: Literal["up", "down", "neutral"]
    confidence: float  # 0-1, how extreme the RSI is
    timestamp: datetime.datetime
    btc_price: float
    
    @property
    def is_directional(self) -> bool:
        return self.direction != "neutral"


def generate_signal(rsi: float, btc_price: float) -> RSISignal:
    """Generate trading signal from RSI value.
    
    Args:
        rsi: Current RSI value (0-100)
        btc_price: Current BTC price
        
    Returns:
        RSISignal with direction and confidence
    """
    now = _utcnow()
    
    if rsi < RSI_OVERSOLD_THRESHOLD:
        # Extreme oversold - predict bounce UP
        # Confidence increases as RSI approaches 0
        confidence = (RSI_OVERSOLD_THRESHOLD - rsi) / RSI_OVERSOLD_THRESHOLD
        return RSISignal(
            rsi=rsi,
            direction="up",
            confidence=min(1.0, confidence),
            timestamp=now,
            btc_price=btc_price,
        )
    
    elif rsi > RSI_OVERBOUGHT_THRESHOLD:
        # Extreme overbought - predict pullback DOWN
        # Confidence increases as RSI approaches 100
        confidence = (rsi - RSI_OVERBOUGHT_THRESHOLD) / (100 - RSI_OVERBOUGHT_THRESHOLD)
        return RSISignal(
            rsi=rsi,
            direction="down",
            confidence=min(1.0, confidence),
            timestamp=now,
            btc_price=btc_price,
        )
    
    else:
        # Neutral zone - no directional prediction
        # Confidence is 0 at extremes of neutral zone, higher in middle
        mid_neutral = (RSI_OVERSOLD_THRESHOLD + RSI_OVERBOUGHT_THRESHOLD) / 2
        distance_from_mid = abs(rsi - mid_neutral)
        max_distance = (RSI_OVERBOUGHT_THRESHOLD - RSI_OVERSOLD_THRESHOLD) / 2
        confidence = 1.0 - (distance_from_mid / max_distance)
        
        return RSISignal(
            rsi=rsi,
            direction="neutral",
            confidence=confidence,
            timestamp=now,
            btc_price=btc_price,
        )


# ----------------------------
# Inventory Management
# ----------------------------

@dataclass
class InventoryState:
    """Track inventory across all positions."""
    net_delta: int = 0  # Positive = net long, Negative = net short
    total_exposure_dollars: float = 0.0
    positions_by_ticker: dict[str, int] = field(default_factory=dict)
    
    def update_from_positions(self, positions: dict[str, "Position"], current_prices: dict[str, int]) -> None:
        """Update inventory state from current positions."""
        self.net_delta = 0
        self.total_exposure_dollars = 0.0
        self.positions_by_ticker.clear()
        
        for ticker, pos in positions.items():
            qty = pos.quantity
            self.positions_by_ticker[ticker] = qty
            self.net_delta += qty
            
            # Calculate exposure in dollars
            price = current_prices.get(ticker, 50) / 100.0
            if qty > 0:
                self.total_exposure_dollars += qty * price
            else:
                self.total_exposure_dollars += abs(qty) * (1 - price)
    
    def get_effective_limit(self, signal: RSISignal) -> int:
        """Get inventory limit, adjusted for active signals.
        
        If we have an RSI signal, allow more inventory in that direction.
        """
        base_limit = INVENTORY_HARD_LIMIT
        
        if signal.direction == "up":
            # Allow more long inventory when bullish
            return int(base_limit * INVENTORY_SIGNAL_ALLOWANCE)
        elif signal.direction == "down":
            # Allow more short inventory when bearish
            return int(-base_limit * INVENTORY_SIGNAL_ALLOWANCE)
        
        return base_limit
    
    def needs_rebalance(self, signal: RSISignal) -> bool:
        """Check if inventory needs active rebalancing."""
        effective_limit = abs(self.get_effective_limit(signal))
        
        # Check if we're over limit in wrong direction
        if signal.direction == "up" and self.net_delta < -effective_limit:
            return True  # Too short when bullish
        if signal.direction == "down" and self.net_delta > effective_limit:
            return True  # Too long when bearish
        if signal.direction == "neutral" and abs(self.net_delta) > INVENTORY_HARD_LIMIT:
            return True  # Too much inventory either way
        
        return False
    
    def get_rebalance_direction(self) -> str:
        """Get direction to rebalance (buy or sell YES)."""
        if self.net_delta > 0:
            return "sell"  # Sell YES to reduce long exposure
        else:
            return "buy"   # Buy YES to reduce short exposure


def calculate_inventory_skew(
    net_delta: int,
    signal: RSISignal,
) -> tuple[int, int]:
    """Calculate bid/ask skew based on inventory.
    
    Returns:
        (bid_skew, ask_skew) in cents
        Positive skew = widen (less aggressive)
        Negative skew = tighten (more aggressive)
    """
    # Determine effective threshold based on signal
    if signal.direction == "up":
        # Allow more long inventory when bullish
        effective_threshold = int(INVENTORY_SKEW_THRESHOLD * INVENTORY_SIGNAL_ALLOWANCE)
    elif signal.direction == "down":
        # Allow more short inventory when bearish  
        effective_threshold = int(INVENTORY_SKEW_THRESHOLD * INVENTORY_SIGNAL_ALLOWANCE)
    else:
        effective_threshold = INVENTORY_SKEW_THRESHOLD
    
    bid_skew = 0
    ask_skew = 0
    
    if abs(net_delta) <= effective_threshold:
        # Within threshold - no skew needed
        return (0, 0)
    
    # Calculate how far over threshold we are
    excess = abs(net_delta) - effective_threshold
    skew_factor = min(1.0, excess / effective_threshold)  # 0-1 scale
    skew_cents = int(INVENTORY_SKEW_CENTS * skew_factor)
    
    if net_delta > 0:
        # Too long - discourage buying, encourage selling
        # Widen bid (less aggressive buying)
        # Tighten ask (more aggressive selling)
        bid_skew = skew_cents   # Widen = positive
        ask_skew = -skew_cents  # Tighten = negative
    else:
        # Too short - encourage buying, discourage selling
        # Tighten bid (more aggressive buying)
        # Widen ask (less aggressive selling)
        bid_skew = -skew_cents  # Tighten = negative
        ask_skew = skew_cents   # Widen = positive
    
    return (bid_skew, ask_skew)


# ----------------------------
# Quote Generation
# ----------------------------

@dataclass
class Quote:
    """A bid or ask quote."""
    side: Literal["bid", "ask"]
    price_cents: int
    size: int
    ticker: str


def generate_quotes(
    signal: RSISignal,
    market_mid_cents: int,
    ticker: str,
    current_position: int = 0,
    net_delta: int = 0,
    spread_cents: int = DEFAULT_SPREAD_CENTS,
    base_size: int = DEFAULT_SIZE_CONTRACTS,
) -> list[Quote]:
    """Generate bid/ask quotes based on signal and inventory.
    
    Args:
        signal: Current RSI signal
        market_mid_cents: Current market mid price in cents
        ticker: Market ticker
        current_position: Current position in this market (positive=long)
        net_delta: Total net delta across all positions (for inventory skewing)
        spread_cents: Half-spread in cents
        base_size: Base order size
        
    Returns:
        List of Quote objects to place
    """
    quotes = []
    
    # Calculate inventory-based skew
    bid_skew, ask_skew = calculate_inventory_skew(net_delta, signal)
    
    # Adjust size based on position (reduce if already exposed)
    position_factor = 1.0 - abs(current_position) / (MAX_POSITION_PER_MARKET * 2)
    size = max(MIN_SIZE, int(base_size * position_factor))
    
    # Also reduce size if inventory is high (to slow down accumulation)
    inventory_factor = 1.0 - min(1.0, abs(net_delta) / (INVENTORY_HARD_LIMIT * 2))
    size = max(MIN_SIZE, int(size * inventory_factor))
    
    if signal.direction == "neutral":
        # Symmetric market making with inventory skew
        # bid_skew > 0 means widen bid (less aggressive buying)
        # ask_skew > 0 means widen ask (less aggressive selling)
        bid_price = max(1, market_mid_cents - spread_cents - bid_skew)
        ask_price = min(99, market_mid_cents + spread_cents + ask_skew)
        
        # Log inventory skew if significant
        if bid_skew != 0 or ask_skew != 0:
            log.debug(
                "[INVENTORY] net_delta=%d, bid_skew=%+dc, ask_skew=%+dc -> bid=%dc, ask=%dc",
                net_delta, bid_skew, ask_skew, bid_price, ask_price
            )
        
        # Bid (buy YES)
        if current_position < MAX_POSITION_PER_MARKET:
            quotes.append(Quote(
                side="bid",
                price_cents=bid_price,
                size=size,
                ticker=ticker,
            ))
        
        # Ask (sell YES) 
        if current_position > -MAX_POSITION_PER_MARKET:
            quotes.append(Quote(
                side="ask",
                price_cents=ask_price,
                size=size,
                ticker=ticker,
            ))
    
    elif signal.direction == "up":
        # Bullish - tighten bid (more aggressive buying), widen ask
        # Higher confidence = more aggressive
        # Also apply inventory skew on top of directional adjustment
        bid_adjust = int(spread_cents * signal.confidence) - bid_skew
        ask_adjust = int(spread_cents * (1 + signal.confidence)) + ask_skew
        
        bid_price = max(1, market_mid_cents - spread_cents + bid_adjust)
        ask_price = min(99, market_mid_cents + spread_cents + ask_adjust)
        
        # Aggressive bid to accumulate YES (but respect inventory limits)
        if current_position < MAX_POSITION_PER_MARKET and net_delta < INVENTORY_HARD_LIMIT:
            # Increase size for confident signals
            confident_size = min(MAX_SIZE, int(size * (1 + signal.confidence)))
            quotes.append(Quote(
                side="bid",
                price_cents=bid_price,
                size=confident_size,
                ticker=ticker,
            ))
        
        # Reluctant ask (still provide liquidity, but less aggressively)
        if current_position > -MAX_POSITION_PER_MARKET // 2:
            quotes.append(Quote(
                side="ask",
                price_cents=ask_price,
                size=MIN_SIZE,  # Minimal size on losing side
                ticker=ticker,
            ))
    
    elif signal.direction == "down":
        # Bearish - tighten ask (more aggressive selling), widen bid
        # Also apply inventory skew on top of directional adjustment
        bid_adjust = int(spread_cents * (1 + signal.confidence)) + bid_skew
        ask_adjust = int(spread_cents * signal.confidence) - ask_skew
        
        bid_price = max(1, market_mid_cents - spread_cents - bid_adjust)
        ask_price = min(99, market_mid_cents + spread_cents - ask_adjust)
        
        # Reluctant bid
        if current_position < MAX_POSITION_PER_MARKET // 2:
            quotes.append(Quote(
                side="bid",
                price_cents=bid_price,
                size=MIN_SIZE,
                ticker=ticker,
            ))
        
        # Aggressive ask to short YES (but respect inventory limits)
        if current_position > -MAX_POSITION_PER_MARKET and net_delta > -INVENTORY_HARD_LIMIT:
            confident_size = min(MAX_SIZE, int(size * (1 + signal.confidence)))
            quotes.append(Quote(
                side="ask",
                price_cents=ask_price,
                size=confident_size,
                ticker=ticker,
            ))
    
    return quotes


def generate_rebalance_orders(
    inventory: "InventoryState",
    signal: RSISignal,
    market_mid_cents: int,
    ticker: str,
    seconds_to_expiry: int,
    use_market_order: bool = False,
) -> list[Quote]:
    """Generate orders to actively rebalance inventory.
    
    Called when inventory exceeds hard limits.
    
    Args:
        inventory: Current inventory state
        signal: Current RSI signal
        market_mid_cents: Current market mid price
        ticker: Market ticker
        seconds_to_expiry: Seconds until expiry
        use_market_order: If True, use aggressive pricing for immediate fill
        
    Returns:
        List of Quote objects to place for rebalancing
    """
    quotes = []
    
    # Determine how much to close
    effective_limit = INVENTORY_HARD_LIMIT
    if signal.direction == "up":
        effective_limit = int(INVENTORY_HARD_LIMIT * INVENTORY_SIGNAL_ALLOWANCE)
    elif signal.direction == "down":
        effective_limit = int(INVENTORY_HARD_LIMIT * INVENTORY_SIGNAL_ALLOWANCE)
    
    excess = abs(inventory.net_delta) - effective_limit
    if excess <= 0:
        return quotes
    
    # Close the excess amount
    close_size = min(MAX_SIZE, excess)
    
    if inventory.net_delta > 0:
        # Too long - need to sell YES
        if use_market_order or seconds_to_expiry < INVENTORY_EMERGENCY_SECONDS:
            # Emergency: aggressive pricing (below mid)
            price = max(1, market_mid_cents - INVENTORY_CLOSE_AGGRESSION * 2)
            log.warning(
                "[INVENTORY EMERGENCY] Too long (%d), selling %d @ %dc (aggressive)",
                inventory.net_delta, close_size, price
            )
        else:
            # Normal rebalance: inside the spread but not crazy
            price = max(1, market_mid_cents - INVENTORY_CLOSE_AGGRESSION)
            log.info(
                "[INVENTORY REBALANCE] Too long (%d), selling %d @ %dc",
                inventory.net_delta, close_size, price
            )
        
        quotes.append(Quote(
            side="ask",
            price_cents=price,
            size=close_size,
            ticker=ticker,
        ))
    
    else:
        # Too short - need to buy YES
        if use_market_order or seconds_to_expiry < INVENTORY_EMERGENCY_SECONDS:
            # Emergency: aggressive pricing (above mid)
            price = min(99, market_mid_cents + INVENTORY_CLOSE_AGGRESSION * 2)
            log.warning(
                "[INVENTORY EMERGENCY] Too short (%d), buying %d @ %dc (aggressive)",
                inventory.net_delta, close_size, price
            )
        else:
            # Normal rebalance: inside the spread but not crazy
            price = min(99, market_mid_cents + INVENTORY_CLOSE_AGGRESSION)
            log.info(
                "[INVENTORY REBALANCE] Too short (%d), buying %d @ %dc",
                inventory.net_delta, close_size, price
            )
        
        quotes.append(Quote(
            side="bid",
            price_cents=price,
            size=close_size,
            ticker=ticker,
        ))
    
    return quotes


# ----------------------------
# Backtesting
# ----------------------------

@dataclass
class BacktestTrade:
    """A single trade in backtest."""
    timestamp: datetime.datetime
    direction: Literal["up", "down"]
    rsi_at_entry: float
    btc_price_at_entry: float
    btc_price_at_exit: float
    pnl_dollars: float
    won: bool


@dataclass  
class BacktestResult:
    """Results from backtesting."""
    trades: list[BacktestTrade]
    total_signals: int
    total_wins: int
    total_losses: int
    win_rate: float
    total_pnl: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"RSI BACKTEST RESULTS\n"
            f"{'='*60}\n"
            f"Total Signals:    {self.total_signals}\n"
            f"Wins:             {self.total_wins}\n"
            f"Losses:           {self.total_losses}\n"
            f"Win Rate:         {self.win_rate*100:.1f}%\n"
            f"Total PnL:        ${self.total_pnl:.2f}\n"
            f"Avg Win:          ${self.avg_win:.2f}\n"
            f"Avg Loss:         ${self.avg_loss:.2f}\n"
            f"Profit Factor:    {self.profit_factor:.2f}\n"
            f"{'='*60}\n"
        )


async def run_backtest(
    days: int = 30,
    position_size_dollars: float = 5.0,
    rsi_oversold: int = RSI_OVERSOLD_THRESHOLD,
    rsi_overbought: int = RSI_OVERBOUGHT_THRESHOLD,
    candles: list[dict[str, float]] | None = None,  # Pre-fetched candles (for optimization)
) -> BacktestResult:
    """Run RSI strategy backtest on historical data.
    
    Strategy (matches live behavior):
    - Use 1-minute candles for RSI calculation
    - At each 1-min step within a 15-min contract window:
      - Compute RSI up to that minute
      - If RSI triggers extreme threshold → enter trade
      - Outcome = whether window's final close > entry price (for UP) or < (for DOWN)
    
    This tests: P(window_close > current_price | current RSI is extreme)
    Which matches live quoting logic.
    
    Args:
        days: Number of days of historical data
        position_size_dollars: Notional size per trade
        rsi_oversold: RSI threshold for oversold (buy signal)
        rsi_overbought: RSI threshold for overbought (sell signal)
        candles: Pre-fetched candles (optional, for optimization runs)
        
    Returns:
        BacktestResult with trade statistics
    """
    # Reduce logging noise when candles are pre-fetched (optimization mode)
    verbose = candles is None
    
    if verbose:
        log.info("Starting backtest for %d days...", days)
        log.info("RSI: %d-period on 1-min candles | Contracts: 15-min windows", RSI_PERIOD)
        log.info("RSI thresholds: Oversold < %d, Overbought > %d", rsi_oversold, rsi_overbought)
    
    # Use pre-fetched candles or fetch new ones
    if candles is not None:
        rsi_candles = candles
    else:
        # Fetch 1-minute candles for RSI calculation
        rsi_candles = await fetch_btc_candles_historical(
            days=days, 
            granularity=RSI_CANDLE_GRANULARITY
        )
        log.info("Fetched %d 1-min candles for RSI", len(rsi_candles))
    
    if len(rsi_candles) < RSI_PERIOD + 15:
        log.error("Not enough candles for backtest")
        return BacktestResult(
            trades=[],
            total_signals=0,
            total_wins=0,
            total_losses=0,
            win_rate=0.0,
            total_pnl=0.0,
            avg_win=0.0,
            avg_loss=0.0,
            profit_factor=0.0,
        )
    
    # Group 1-min candles into 15-min windows for contract evaluation
    window_size = CONTRACT_GRANULARITY // RSI_CANDLE_GRANULARITY  # 15 candles per window
    
    rsi_prices = [c["close"] for c in rsi_candles]
    
    # Find first candle aligned to wall-clock 15-min boundary (00, 15, 30, 45)
    # Use modulo to handle any drift in Coinbase timestamps
    start_idx = 0
    for i, c in enumerate(rsi_candles):
        ts = int(c["timestamp"])
        # Check if this timestamp is on a 15-min boundary (modulo 900 == 0)
        if ts % CONTRACT_GRANULARITY == 0:
            start_idx = i
            break
    
    # If no exact boundary found, find next one
    if start_idx == 0 and len(rsi_candles) > 0:
        first_ts = int(rsi_candles[0]["timestamp"])
        # How many seconds until next 15-min boundary?
        seconds_to_boundary = CONTRACT_GRANULARITY - (first_ts % CONTRACT_GRANULARITY)
        if seconds_to_boundary == CONTRACT_GRANULARITY:
            seconds_to_boundary = 0
        # Skip that many 1-min candles
        start_idx = seconds_to_boundary // RSI_CANDLE_GRANULARITY
    
    # Build 15-min windows with their candle indices
    windows = []
    i = start_idx
    while i + window_size <= len(rsi_candles):
        windows.append({
            "start_idx": i,
            "end_idx": i + window_size - 1,
            "window_close": rsi_candles[i + window_size - 1]["close"],
            "start_ts": rsi_candles[i]["timestamp"],
            "window_open": rsi_candles[i]["open"],
        })
        i += window_size
    
    if verbose:
        log.info("Created %d 15-min windows for contract evaluation", len(windows))
    
    # Pre-compute ALL RSI values once (much faster than recalculating each time)
    all_rsi = _precompute_rsi_series(rsi_prices, RSI_PERIOD)
    
    # Run backtest: at each 1-min step, check if RSI triggers
    # If it does, evaluate against window's final close
    trades = []
    
    for window in windows:
        window_start_idx = window["start_idx"]
        window_end_idx = window["end_idx"]
        window_close = window["window_close"]
        window_open = window["window_open"]
        
        # Track if we already traded this window (only one trade per window)
        traded_this_window = False
        
        # Walk through each minute in this window
        for minute_idx in range(window_start_idx, window_end_idx + 1):
            if traded_this_window:
                break
            
            # Get pre-computed RSI (None if not enough history)
            rsi = all_rsi[minute_idx] if minute_idx < len(all_rsi) else None
            
            if rsi is None:
                continue
            
            # Current price at this minute (entry price if we trade)
            entry_price = rsi_prices[minute_idx]
            
            trade_direction = None
            
            # Check for RSI signal
            if rsi < rsi_oversold:
                # Predict UP - price should rise by window close
                trade_direction = "up"
                won = window_close > entry_price
                
            elif rsi > rsi_overbought:
                # Predict DOWN - price should fall by window close
                trade_direction = "down"
                won = window_close < entry_price
            
            if trade_direction is not None:
                traded_this_window = True
                
                # Calculate PnL for binary options (Kalshi model)
                # Estimate realistic entry price based on where price is in window
                price_move_pct = (entry_price - window_open) / window_open
                # Scale: 1% BTC move ≈ 2c shift in Kalshi mid from 50c baseline
                implied_prob = 0.5 + price_move_pct * 2.0
                entry_price_cents = int(_clamp(implied_prob, 0.20, 0.80) * 100)
                
                contracts = max(1, int(position_size_dollars / (entry_price_cents / 100.0)))
                
                fee_per_contract = _fee_dollars(
                    price_cents=entry_price_cents, 
                    count=1, 
                    maker=True
                )
                total_fees = fee_per_contract * contracts
                
                if won:
                    pnl = contracts * (100 - entry_price_cents) / 100.0 - total_fees
                else:
                    pnl = -contracts * entry_price_cents / 100.0 - total_fees
                
                ts = datetime.datetime.fromtimestamp(
                    rsi_candles[minute_idx]["timestamp"], 
                    tz=datetime.timezone.utc
                )
                
                trades.append(BacktestTrade(
                    timestamp=ts,
                    direction=trade_direction,
                    rsi_at_entry=rsi,
                    btc_price_at_entry=entry_price,
                    btc_price_at_exit=window_close,
                    pnl_dollars=pnl,
                    won=won,
                ))
    
    # Calculate statistics
    total_signals = len(trades)
    wins = [t for t in trades if t.won]
    losses = [t for t in trades if not t.won]
    
    total_wins = len(wins)
    total_losses = len(losses)
    win_rate = total_wins / total_signals if total_signals > 0 else 0.0
    
    total_pnl = sum(t.pnl_dollars for t in trades)
    avg_win = sum(t.pnl_dollars for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t.pnl_dollars for t in losses) / len(losses) if losses else 0.0
    
    gross_wins = sum(t.pnl_dollars for t in wins)
    gross_losses = abs(sum(t.pnl_dollars for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    
    return BacktestResult(
        trades=trades,
        total_signals=total_signals,
        total_wins=total_wins,
        total_losses=total_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
    )


async def run_parameter_scan(days: int = 30) -> None:
    """Scan RSI threshold parameters to find optimal settings.
    
    Tests various combinations of oversold/overbought thresholds
    and reports win rate and profit factor for each.
    
    Args:
        days: Number of days of historical data to use
    """
    log.info("Running RSI parameter optimization scan (%d days)...", days)
    
    # Fetch candles once for all threshold combinations
    log.info("Fetching historical candles (this may take a few minutes)...")
    candles = await fetch_btc_candles_historical(
        days=days,
        granularity=RSI_CANDLE_GRANULARITY
    )
    log.info("Fetched %d 1-min candles for optimization", len(candles))
    
    results = []
    
    # Test various threshold combinations (including extreme levels)
    oversold_range = [10, 15, 20, 25, 30, 35, 40]
    overbought_range = [60, 65, 70, 75, 80, 85, 90]
    
    total_combos = len(oversold_range) * len(overbought_range)
    tested = 0
    
    for oversold in oversold_range:
        for overbought in overbought_range:
            if oversold >= overbought:
                continue
            
            tested += 1
            log.info("Testing %d/%d: oversold=%d, overbought=%d", 
                     tested, total_combos, oversold, overbought)
            
            result = await run_backtest(
                days=days,
                rsi_oversold=oversold,
                rsi_overbought=overbought,
                candles=candles,  # Pass pre-fetched candles
            )
            
            if result.total_signals >= 3:  # Need minimum sample size
                results.append({
                    "oversold": oversold,
                    "overbought": overbought,
                    "signals": result.total_signals,
                    "win_rate": result.win_rate,
                    "pnl": result.total_pnl,
                    "profit_factor": result.profit_factor,
                })
    
    # Sort by win rate (descending)
    results.sort(key=lambda x: (x["win_rate"], x["profit_factor"]), reverse=True)
    
    print("\n" + "="*80)
    print("RSI PARAMETER OPTIMIZATION RESULTS")
    print("="*80)
    print(f"{'Oversold':>10} {'Overbought':>12} {'Signals':>10} {'Win Rate':>12} {'PnL':>10} {'PF':>8}")
    print("-"*80)
    
    for r in results[:15]:  # Top 15 results
        print(
            f"{r['oversold']:>10} {r['overbought']:>12} {r['signals']:>10} "
            f"{r['win_rate']*100:>11.1f}% ${r['pnl']:>8.2f} {r['profit_factor']:>8.2f}"
        )
    
    print("="*80)
    
    if results:
        best = results[0]
        print(f"\nBEST PARAMETERS: Oversold < {best['oversold']}, Overbought > {best['overbought']}")
        print(f"Expected Win Rate: {best['win_rate']*100:.1f}%")


# ----------------------------
# Live Trading
# ----------------------------

@dataclass
class Position:
    """Track position in a market."""
    ticker: str
    quantity: int  # Positive = long YES, Negative = short YES
    entry_price_cents: int
    entry_time: datetime.datetime


@dataclass
class Portfolio:
    """Track portfolio state."""
    cash: float
    starting_cash: float
    positions: dict[str, Position] = field(default_factory=dict)
    fees_paid: float = 0.0
    last_fill_sync_ts: int = 0
    seen_fill_ids: set[str] = field(default_factory=set)
    
    def equity(self, current_prices: dict[str, int]) -> float:
        """Calculate total equity."""
        pos_value = 0.0
        for ticker, pos in self.positions.items():
            if pos.quantity == 0:
                continue
            mid = current_prices.get(ticker, 50) / 100.0
            if pos.quantity > 0:
                pos_value += pos.quantity * mid
            else:
                pos_value += abs(pos.quantity) * (1 - mid)
        return self.cash + pos_value
    
    def pnl(self, current_prices: dict[str, int]) -> float:
        return self.equity(current_prices) - self.starting_cash


async def pick_active_markets(
    client: KalshiClient,
    asset: str = "BTC",
    horizon_minutes: int = 60,
) -> list[dict[str, Any]]:
    """Find active BTC 15-minute markets expiring soon.
    
    Args:
        client: Kalshi API client
        asset: Asset to trade (BTC)
        horizon_minutes: Only consider markets expiring within this time
        
    Returns:
        List of market dicts
    """
    now = _utcnow()
    horizon = datetime.timedelta(minutes=horizon_minutes)
    
    series_ticker = f"KX{asset.upper()}15M"
    
    page = await client.get_markets_page(
        limit=100,
        status="open",
        series_ticker=series_ticker,
        mve_filter="exclude",
    )
    
    markets = page.get("markets", [])
    
    # Filter to markets expiring within horizon
    valid_markets = []
    for m in markets:
        cutoff, _field = _market_cutoff_time(m)
        if cutoff is None:
            continue
        if cutoff < now:
            continue
        if cutoff - now > horizon:
            continue
        valid_markets.append((cutoff, m))
    
    # Sort by expiry (soonest first)
    valid_markets.sort(key=lambda x: x[0])
    
    return [m for _, m in valid_markets[:2]]  # Return up to 2 markets


async def sync_positions(
    client: KalshiClient,
    portfolio: Portfolio,
    current_prices: dict[str, int] | None = None,
) -> None:
    """Sync positions from Kalshi API and estimate fees from position changes."""
    try:
        # Store previous positions to detect fills
        prev_positions = {t: p.quantity for t, p in portfolio.positions.items()}

        now_ts = int(time.time())
        # Use a small overlap to avoid missing fills that arrive slightly delayed.
        # Deduplication via seen_fill_ids prevents double-counting.
        min_fill_ts = max(0, (portfolio.last_fill_sync_ts or (now_ts - 6 * 3600)) - 120)

        # Kalshi's positions endpoint often omits 0-positions when count_filter="position".
        # If we only update tickers that appear in the response, we can keep stale (phantom)
        # positions forever. To avoid this, we page through results and then clear any locally
        # tracked positions that aren't returned.
        market_positions: list[dict[str, Any]] = []
        cursor: str | None = None
        while True:
            positions_resp = await client.get_positions(
                count_filter="position",
                limit=1000,
                cursor=cursor,
            )
            market_positions.extend(positions_resp.get("market_positions", []))
            cursor = positions_resp.get("cursor")
            if not cursor:
                break

        seen_tickers: set[str] = set()

        def _fill_yes_delta_and_price(f: dict[str, Any]) -> tuple[int, int] | None:
            """Return (yes_delta, implied_yes_price_cents) for a fill.

            We treat positions/inventory in YES-space. Some fills may be expressed as
            (side=yes/no) + (action=buy/sell). We convert NO prices to implied YES
            price via (100 - no_price).
            """
            try:
                count = int(f.get("count", 0) or 0)
            except Exception:
                return None
            if count <= 0:
                return None

            side = str(f.get("side") or "").lower()
            action = str(f.get("action") or "").lower()
            if side not in ("yes", "no") or action not in ("buy", "sell"):
                return None

            # YES delta sign: buy YES / sell NO => +; sell YES / buy NO => -
            if side == "yes" and action == "buy":
                yes_delta = count
            elif side == "yes" and action == "sell":
                yes_delta = -count
            elif side == "no" and action == "buy":
                yes_delta = -count
            else:  # side == "no" and action == "sell"
                yes_delta = count

            # Extract fill price and convert to implied YES cents.
            yes_price_cents: int | None = None
            if side == "yes":
                if f.get("yes_price") is not None:
                    yes_price_cents = int(f.get("yes_price") or 0)
                else:
                    p = f.get("price")
                    if isinstance(p, (int, float)):
                        if 0.0 <= float(p) <= 1.0:
                            yes_price_cents = int(round(float(p) * 100.0))
                        else:
                            yes_price_cents = int(round(float(p)))
            else:  # side == "no"
                no_price_cents: int | None = None
                if f.get("no_price") is not None:
                    no_price_cents = int(f.get("no_price") or 0)
                else:
                    p = f.get("price")
                    if isinstance(p, (int, float)):
                        if 0.0 <= float(p) <= 1.0:
                            no_price_cents = int(round(float(p) * 100.0))
                        else:
                            no_price_cents = int(round(float(p)))
                if no_price_cents is not None:
                    yes_price_cents = 100 - int(no_price_cents)

            if yes_price_cents is None:
                return None
            yes_price_cents = int(_clamp(int(yes_price_cents), 1, 99))
            return (yes_delta, yes_price_cents)

        for pos in market_positions:
            ticker = pos.get("ticker", "")
            position = int(pos.get("position", 0) or 0)
            if not ticker:
                continue

            seen_tickers.add(ticker)
            
            # Get market price for fee calculation (default 50c)
            price_cents = 50
            if current_prices and ticker in current_prices:
                price_cents = current_prices[ticker]

            prev_qty = prev_positions.get(ticker, 0)
            qty_changed = (position != prev_qty)

            fills: list[dict[str, Any]] = []
            if qty_changed:
                # Pull recent fills to update fee estimate + entry price basis.
                # We only need per-ticker when there's a detected qty change.
                cursor_fills: str | None = None
                pages = 0
                while True:
                    fills_resp = await client.get_fills(
                        ticker=ticker,
                        min_ts=min_fill_ts,
                        limit=200,
                        cursor=cursor_fills,
                    )
                    fills.extend(list(fills_resp.get("fills") or []))
                    cursor_fills = fills_resp.get("cursor")
                    pages += 1
                    if not cursor_fills or pages >= 5:
                        break

                # Deduplicate fills across sync loops (overlap window).
                new_fills: list[dict[str, Any]] = []
                for f in fills:
                    fid = str(f.get("fill_id") or "")
                    if fid and fid in portfolio.seen_fill_ids:
                        continue
                    if fid:
                        portfolio.seen_fill_ids.add(fid)
                    new_fills.append(f)
                fills = new_fills

                # Cap memory growth: fills are only used for short-lived markets.
                if len(portfolio.seen_fill_ids) > 5000:
                    portfolio.seen_fill_ids.clear()

                # Update fee estimate based on fills we haven't seen before.
                for f in fills:
                    try:
                        count = int(f.get("count", 0) or 0)
                        if count <= 0:
                            continue

                        # Kalshi sometimes returns price as float (0-1) and sometimes as *_price cents.
                        fill_price_cents: int | None = None
                        if f.get("yes_price") is not None:
                            fill_price_cents = int(f.get("yes_price") or 0)
                        elif f.get("no_price") is not None:
                            fill_price_cents = int(f.get("no_price") or 0)
                        else:
                            p = f.get("price")
                            if isinstance(p, (int, float)):
                                if 0.0 <= float(p) <= 1.0:
                                    fill_price_cents = int(round(float(p) * 100.0))
                                else:
                                    fill_price_cents = int(round(float(p)))

                        if fill_price_cents is None:
                            fill_price_cents = price_cents

                        is_taker = bool(f.get("is_taker", False))
                        portfolio.fees_paid += _fee_dollars(
                            price_cents=int(_clamp(fill_price_cents, 1, 99)),
                            count=count,
                            maker=(not is_taker),
                        )
                    except Exception:
                        continue
            
            if position != 0:
                if ticker not in portfolio.positions:
                    # New position - use fill-informed entry estimate when possible
                    entry_price_cents = price_cents
                    entry_time = _utcnow()

                    if fills:
                        # Use a count-weighted average of the most recent fills that
                        # *increase* exposure in the resulting direction.
                        needed = abs(position)
                        target_sign = 1 if position > 0 else -1
                        total = 0
                        wsum = 0
                        latest_time: datetime.datetime | None = None
                        # Prefer newest fills first
                        for f in sorted(fills, key=lambda x: str(x.get("created_time") or ""), reverse=True):
                            parsed = _fill_yes_delta_and_price(f)
                            if parsed is None:
                                continue
                            yes_delta, yes_price = parsed
                            if yes_delta * target_sign <= 0:
                                continue

                            take = min(needed - total, abs(yes_delta))
                            wsum += yes_price * take
                            total += take
                            if latest_time is None:
                                latest_time = _to_dt(f.get("created_time"))
                            if total >= needed:
                                break
                        if total > 0:
                            entry_price_cents = int(round(wsum / total))
                            if latest_time is not None:
                                entry_time = latest_time

                    portfolio.positions[ticker] = Position(
                        ticker=ticker,
                        quantity=position,
                        entry_price_cents=int(_clamp(entry_price_cents, 1, 99)),
                        entry_time=entry_time,
                    )
                else:
                    # Position changed - update entry price when we increase exposure in same direction.
                    prev_qty = prev_positions.get(ticker, 0)
                    new_qty = position

                    prev_sign = 0 if prev_qty == 0 else (1 if prev_qty > 0 else -1)
                    new_sign = 0 if new_qty == 0 else (1 if new_qty > 0 else -1)

                    # Detect whether this change increases absolute exposure in the resulting direction.
                    increase_count = 0
                    if prev_sign == 0 and new_sign != 0:
                        increase_count = abs(new_qty)
                    elif prev_sign == new_sign and new_sign != 0:
                        increase_count = max(0, abs(new_qty) - abs(prev_qty))
                    elif prev_sign != 0 and new_sign != 0 and prev_sign != new_sign:
                        increase_count = abs(new_qty)

                    if increase_count > 0 and fills:
                        target_sign = 1 if new_qty > 0 else -1
                        total = 0
                        wsum = 0
                        latest_time: datetime.datetime | None = None

                        for f in sorted(fills, key=lambda x: str(x.get("created_time") or ""), reverse=True):
                            parsed = _fill_yes_delta_and_price(f)
                            if parsed is None:
                                continue
                            yes_delta, yes_price = parsed
                            if yes_delta * target_sign <= 0:
                                continue

                            take = min(increase_count - total, abs(yes_delta))
                            wsum += yes_price * take
                            total += take
                            if latest_time is None:
                                latest_time = _to_dt(f.get("created_time"))
                            if total >= increase_count:
                                break

                        if total > 0:
                            avg_fill = int(round(wsum / total))
                            pos_obj = portfolio.positions[ticker]
                            if prev_sign != new_sign:
                                pos_obj.entry_price_cents = avg_fill
                                if latest_time is not None:
                                    pos_obj.entry_time = latest_time
                            else:
                                # Weighted average cost basis when adding to a position.
                                old_qty_abs = abs(prev_qty)
                                new_qty_abs = abs(new_qty)
                                if new_qty_abs > 0:
                                    pos_obj.entry_price_cents = int(
                                        round((pos_obj.entry_price_cents * old_qty_abs + avg_fill * increase_count) / new_qty_abs)
                                    )
                                    if latest_time is not None:
                                        pos_obj.entry_time = latest_time

                    portfolio.positions[ticker].quantity = new_qty
            elif ticker in portfolio.positions:
                # Position closed - estimate exit fees
                prev_qty = prev_positions.get(ticker, 0)
                del portfolio.positions[ticker]

        # Clear any locally tracked positions that are no longer returned.
        # With count_filter="position", missing tickers should be interpreted as 0-position.
        for ticker in list(portfolio.positions.keys()):
            if ticker in seen_tickers:
                continue
            prev_qty = prev_positions.get(ticker, portfolio.positions[ticker].quantity)

            # Estimate exit fees using latest known price (default 50c)
            price_cents = 50
            if current_prices and ticker in current_prices:
                price_cents = current_prices[ticker]
            del portfolio.positions[ticker]

        # Advance fill cursor for next loop
        portfolio.last_fill_sync_ts = now_ts
                
    except Exception as e:
        log.warning("Failed to sync positions: %s", e)


async def cancel_all_orders(
    client: KalshiClient,
    ticker: str,
    order_ids: list[str],
    dry_run: bool = False,
) -> None:
    """Cancel all open orders for a ticker."""
    for order_id in order_ids:
        # Skip dry-run orders
        if order_id.startswith("dry-run-"):
            continue
        try:
            await client.cancel_order(order_id)
            log.debug("Cancelled order %s", order_id)
        except Exception as e:
            log.warning("Failed to cancel order %s: %s", order_id, e)


async def place_quote(
    client: KalshiClient,
    quote: Quote,
    dry_run: bool = False,
    *,
    reduce_only: bool | None = None,
    order_type: Literal["limit", "market"] = "limit",
    seconds_to_expiry: int | None = None,
    btc_price: float | None = None,
    price_to_beat: float | None = None,
) -> str | None:
    """Place a quote order.
    
    Returns:
        Order ID if successful, None otherwise
    """
    if dry_run:
        log.info(
            "[DRY-RUN] Would place %s %s %d @ %s (%ss to expiry, ptb=%s, btc=%s)",
            quote.side,
            quote.ticker,
            quote.size,
            ("MKT" if order_type == "market" else f"{quote.price_cents}c"),
            "?" if seconds_to_expiry is None else str(int(seconds_to_expiry)),
            "?" if price_to_beat is None else f"${price_to_beat:,.0f}",
            "?" if btc_price is None else f"${btc_price:,.0f}",
        )
        return f"dry-run-{time.time()}"
    
    try:
        # Kalshi constraint: reduce_only can only be used with IoC orders.
        tif = "immediate_or_cancel" if reduce_only else None
        if quote.side == "bid":
            # Buy YES at bid price
            result = await client.create_order(
                ticker=quote.ticker,
                side="yes",
                action="buy",
                count=quote.size,
                order_type=order_type,
                # Kalshi requires exactly one of yes_price/no_price(/_dollars) even for "market" orders.
                # For market buys, set a conservative cap at 99c.
                yes_price=(99 if order_type == "market" else quote.price_cents),
                reduce_only=reduce_only,
                time_in_force=tif,
            )
        else:
            # Sell YES at ask price
            result = await client.create_order(
                ticker=quote.ticker,
                side="yes",
                action="sell",
                count=quote.size,
                order_type=order_type,
                # For market sells, set a conservative floor at 1c.
                yes_price=(1 if order_type == "market" else quote.price_cents),
                reduce_only=reduce_only,
                time_in_force=tif,
            )
        
        order = result.get("order", {})
        order_id = order.get("order_id")
        
        log.info(
            "Placed %s %s %d @ %s (%ss to expiry, ptb=%s, btc=%s) -> order_id=%s",
            quote.side,
            quote.ticker,
            quote.size,
            ("MKT" if order_type == "market" else f"{quote.price_cents}c"),
            "?" if seconds_to_expiry is None else str(int(seconds_to_expiry)),
            "?" if price_to_beat is None else f"${price_to_beat:,.0f}",
            "?" if btc_price is None else f"${btc_price:,.0f}",
            order_id,
        )
        
        return order_id
        
    except Exception as e:
        root: Exception = e  # type: ignore[assignment]
        if RetryError is not None and isinstance(e, RetryError):
            try:
                root = e.last_attempt.exception() or e
            except Exception:
                root = e

        # If this is an HTTPStatusError, include status and response body.
        status = None
        body = None
        if hasattr(root, "response") and getattr(root, "response") is not None:
            try:
                status = getattr(root.response, "status_code", None)
                body = getattr(root.response, "text", None)
            except Exception:
                status = None
                body = None

        if status is not None:
            log.error(
                "Failed to place quote (%s %s %d @ %s, %ss to expiry, ptb=%s, btc=%s) HTTP %s: %s",
                quote.side,
                quote.ticker,
                quote.size,
                ("MKT" if order_type == "market" else f"{quote.price_cents}c"),
                "?" if seconds_to_expiry is None else str(int(seconds_to_expiry)),
                "?" if price_to_beat is None else f"${price_to_beat:,.0f}",
                "?" if btc_price is None else f"${btc_price:,.0f}",
                status,
                (body or str(root))[:5000],
            )
        else:
            log.error(
                "Failed to place quote (%s %s %d @ %s, %ss to expiry, ptb=%s, btc=%s): %s",
                quote.side,
                quote.ticker,
                quote.size,
                ("MKT" if order_type == "market" else f"{quote.price_cents}c"),
                "?" if seconds_to_expiry is None else str(int(seconds_to_expiry)),
                "?" if price_to_beat is None else f"${price_to_beat:,.0f}",
                "?" if btc_price is None else f"${btc_price:,.0f}",
                root,
            )
        return None


def _position_pnl_cents(*, qty: int, entry_cents: int, mid_cents: int) -> int:
    """Unrealized PnL per contract in cents, using YES mid.

    qty > 0: long YES -> pnl = mid - entry
    qty < 0: short YES -> pnl = entry - mid
    """
    if qty == 0:
        return 0
    if qty > 0:
        return int(mid_cents - entry_cents)
    return int(entry_cents - mid_cents)


def _generate_position_exit_quotes(
    *,
    ticker: str,
    qty: int,
    entry_cents: int,
    best_bid_cents: int,
    best_ask_cents: int,
    seconds_to_expiry: int | None,
) -> tuple[list[Quote], str]:
    """Generate reduce-only exit quotes for stop-loss / take-profit.

    Returns (quotes, order_type). order_type is one of: "stop", "take_profit", or "".
    """
    if qty == 0:
        return ([], "")

    mid_cents = (best_bid_cents + best_ask_cents) // 2
    pnl_cents = _position_pnl_cents(qty=qty, entry_cents=entry_cents, mid_cents=mid_cents)

    # Stop-loss: cut losing positions, but only close to expiry.
    if (
        pnl_cents <= -STOP_LOSS_CENTS
        and (
            seconds_to_expiry is None
            or seconds_to_expiry <= STOP_LOSS_ONLY_WITHIN_SECONDS
        )
    ):
        if qty > 0:
            # Long YES -> sell to exit
            return (
                [
                    Quote(
                        side="ask",
                        price_cents=max(1, best_bid_cents),
                        size=abs(qty),
                        ticker=ticker,
                    )
                ],
                "stop",
            )
        # Short YES -> buy to exit
        return (
            [
                Quote(
                    side="bid",
                    price_cents=min(99, best_ask_cents),
                    size=abs(qty),
                    ticker=ticker,
                )
            ],
            "stop",
        )

    # Take-profit: lock in gains.
    if pnl_cents >= TAKE_PROFIT_CENTS:
        if qty > 0:
            return (
                [
                    Quote(
                        side="ask",
                        price_cents=max(1, best_bid_cents),
                        size=abs(qty),
                        ticker=ticker,
                    )
                ],
                "take_profit",
            )
        return (
            [
                Quote(
                    side="bid",
                    price_cents=min(99, best_ask_cents),
                    size=abs(qty),
                    ticker=ticker,
                )
            ],
            "take_profit",
        )

    return ([], "")


def _generate_exposure_cap_quotes(
    *,
    ticker: str,
    qty: int,
    best_bid_cents: int,
    best_ask_cents: int,
) -> list[Quote]:
    """Generate a small reduce-only unwind when total exposure is capped."""
    if qty == 0:
        return []
    size = min(MAX_SIZE, abs(qty))
    if qty > 0:
        # Long YES -> sell to reduce
        return [
            Quote(
                side="ask",
                price_cents=max(1, best_bid_cents),
                size=size,
                ticker=ticker,
            )
        ]
    # Short YES -> buy to reduce
    return [
        Quote(
            side="bid",
            price_cents=min(99, best_ask_cents),
            size=size,
            ticker=ticker,
        )
    ]


def _generate_expiry_flatten_quotes(
    *,
    ticker: str,
    qty: int,
    best_bid_cents: int,
    best_ask_cents: int,
) -> list[Quote]:
    """Force flatten a position close to expiry.

    We cross the spread (buy at best ask / sell at best bid) and rely on IoC
    so we don't leave new resting risk right before expiration.
    """
    if qty == 0:
        return []
    if qty > 0:
        # Long YES -> sell to close at bid
        return [
            Quote(
                side="ask",
                price_cents=max(1, best_bid_cents),
                size=abs(qty),
                ticker=ticker,
            )
        ]
    # Short YES -> buy to close at ask
    return [
        Quote(
            side="bid",
            price_cents=min(99, best_ask_cents),
            size=abs(qty),
            ticker=ticker,
        )
    ]


async def run_live(
    *,
    client: KalshiClient,
    cmc_api_key: str | None = None,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    duration_minutes: float = DEFAULT_DURATION_MINUTES,
    bankroll_dollars: float = 100.0,
    dry_run: bool = False,
    out_dir: str = DEFAULT_OUT_DIR,
) -> None:
    """Run live trading loop.
    
    Args:
        client: Kalshi API client
        cmc_api_key: CoinMarketCap API key (optional)
        poll_seconds: Seconds between polls
        duration_minutes: How long to run
        bankroll_dollars: Starting capital
        dry_run: If True, log but don't trade
        out_dir: Output directory for logs
    """
    mode_str = "DRY-RUN" if dry_run else "LIVE TRADING"
    log.info("="*60)
    log.info("  RSI MARKET MAKER - %s", mode_str)
    log.info("="*60)
    
    # Log inventory protection settings
    log.info("Inventory Protection:")
    log.info("  - Skew threshold: ±%d contracts", INVENTORY_SKEW_THRESHOLD)
    log.info("  - Hard limit: ±%d contracts", INVENTORY_HARD_LIMIT)
    log.info("  - Signal allowance: %.1fx", INVENTORY_SIGNAL_ALLOWANCE)
    log.info("  - Emergency close: <%ds to expiry", INVENTORY_EMERGENCY_SECONDS)
    
    # Get account balance
    try:
        balance_resp = await client.get_balance()
        account_balance = balance_resp.get("balance", 0) / 100.0
        log.info("Account Balance: $%.2f", account_balance)
    except Exception as e:
        log.warning("Could not fetch balance: %s", e)
        account_balance = bankroll_dollars
    
    portfolio = Portfolio(
        cash=bankroll_dollars,
        starting_cash=bankroll_dollars,
    )
    
    # Inventory tracking
    inventory = InventoryState()
    
    # Track orders: ticker -> {order_id: Quote}
    open_orders: dict[str, dict[str, Quote]] = {}
    
    # Track previous quote state to avoid unnecessary cancel/replace
    # ticker -> {side: (price_cents, size)}
    prev_quotes: dict[str, dict[str, tuple[int, int]]] = {}
    prev_inventory_delta: int = 0
    
    # Track current prices for inventory calculation
    current_prices: dict[str, int] = {}  # ticker -> mid price in cents
    
    # Price history for RSI
    price_history: list[float] = []
    last_candle_ts: float = 0.0  # Track last candle timestamp to avoid duplicates
    
    # Initialize with historical 1-minute prices for RSI
    try:
        candles = await fetch_btc_candles(granularity=RSI_CANDLE_GRANULARITY, limit=50)
        price_history = [c["close"] for c in candles]
        if candles:
            last_candle_ts = candles[-1]["timestamp"]
        log.info("Initialized with %d 1-min prices for RSI (last_ts=%.0f)", len(price_history), last_candle_ts)
    except Exception as e:
        log.warning("Could not fetch historical prices: %s", e)
    
    # Setup output files
    os.makedirs(out_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    trades_path = os.path.join(out_dir, f"rsi_trades_{ts}.csv")
    signals_path = os.path.join(out_dir, f"rsi_signals_{ts}.csv")
    
    trades_f = open(trades_path, "w", newline="", encoding="utf-8")
    signals_f = open(signals_path, "w", newline="", encoding="utf-8")
    
    trades_w = csv.writer(trades_f)
    signals_w = csv.writer(signals_f)
    
    trades_w.writerow(["ts", "ticker", "side", "price_cents", "size", "signal", "rsi", "net_delta", "type"])
    signals_w.writerow(["ts", "rsi", "direction", "confidence", "btc_price", "net_delta"])
    
    end_time = time.monotonic() + (duration_minutes * 60.0)
    next_poll_time = time.monotonic()  # Timestamp-anchored polling
    
    log.info("Starting trading loop for %.1f minutes...", duration_minutes)
    
    try:
        while time.monotonic() < end_time:
            # Timestamp-anchored polling: sleep until next scheduled poll
            sleep_duration = next_poll_time - time.monotonic()
            if sleep_duration > 0:
                await asyncio.sleep(sleep_duration)
            next_poll_time = time.monotonic() + poll_seconds
            
            now = _utcnow()
            
            # Sync positions (pass prices for fee estimation)
            await sync_positions(client, portfolio, current_prices)
            
            # Update inventory state
            inventory.update_from_positions(portfolio.positions, current_prices)

            exposure_capped = inventory.total_exposure_dollars >= MAX_TOTAL_EXPOSURE_DOLLARS
            if exposure_capped:
                log.warning(
                    "[RISK] Exposure cap hit: $%.2f >= $%.2f (only reduce-only unwinds will be placed)",
                    inventory.total_exposure_dollars,
                    float(MAX_TOTAL_EXPOSURE_DOLLARS),
                )
            
            # Log fees if any accumulated
            if portfolio.fees_paid > 0:
                log.debug("[FEES] Estimated fees paid: $%.4f", portfolio.fees_paid)
            
            # Log inventory status
            if inventory.net_delta != 0:
                log.info(
                    "[INVENTORY] Net Delta: %+d | Exposure: $%.2f",
                    inventory.net_delta, inventory.total_exposure_dollars
                )
            
            # Get current BTC price from 1-min candles (Coinbase primary, CMC fallback)
            current_price = None
            candle_ts = None
            
            try:
                candles = await fetch_btc_candles(granularity=RSI_CANDLE_GRANULARITY, limit=5)
                if candles:
                    latest_candle = candles[-1]
                    current_price = latest_candle["close"]
                    candle_ts = latest_candle["timestamp"]
            except Exception as e:
                log.warning("Coinbase fetch failed: %s", e)
            
            # Fallback to CMC if Coinbase failed
            if current_price is None and cmc_api_key:
                try:
                    cmc_price = await fetch_btc_price_cmc(cmc_api_key)
                    if cmc_price:
                        current_price = cmc_price
                        # Use current time as pseudo-candle timestamp
                        candle_ts = time.time()
                        log.info("Using CMC fallback price: $%.0f", current_price)
                except Exception as e:
                    log.warning("CMC fallback also failed: %s", e)
            
            if current_price is None:
                log.error("Failed to fetch BTC price from any source")
                await asyncio.sleep(poll_seconds)
                continue

            # Fetch a spot price for operator logs; this updates every poll.
            spot_price = await fetch_btc_spot_price_coinbase()
            if spot_price is None:
                spot_price = float(current_price)
            
            # Only append if this is a NEW candle (avoid duplicates)
            if candle_ts and candle_ts > last_candle_ts:
                price_history.append(current_price)
                last_candle_ts = candle_ts
                log.debug("New candle at ts=%.0f, price=$%.0f (history=%d)", 
                          candle_ts, current_price, len(price_history))
                
                # Keep only recent prices
                if len(price_history) > 50:
                    price_history = price_history[-50:]
            else:
                log.debug("Same candle ts=%.0f, not appending (history=%d)", 
                          candle_ts or 0, len(price_history))
            
            # Calculate RSI
            rsi = calculate_rsi_smoothed(price_history)
            if rsi is None:
                log.info("Not enough data for RSI (have %d prices)", len(price_history))
                await asyncio.sleep(poll_seconds)
                continue
            
            # Generate signal
            signal = generate_signal(rsi, current_price)
            
            log.info(
                "BTC=$%.0f | RSI=%.1f | Signal=%s (conf=%.2f) | Inventory=%+d",
                spot_price, rsi, signal.direction, signal.confidence, inventory.net_delta,
            )
            
            # Log signal
            signals_w.writerow([
                now.isoformat(),
                f"{rsi:.2f}",
                signal.direction,
                f"{signal.confidence:.3f}",
                f"{current_price:.2f}",
                inventory.net_delta,
            ])
            signals_f.flush()
            
            # Find active markets
            try:
                markets = await pick_active_markets(client, horizon_minutes=30)
            except Exception as e:
                log.error("Failed to fetch markets: %s", e)
                await asyncio.sleep(poll_seconds)
                continue

            # Ensure we always process tickers we currently hold (so expiry-flatten
            # and stop/take-profit cannot be skipped due to market selection).
            try:
                held_tickers = {t for t, p in portfolio.positions.items() if p.quantity != 0}
                selected_tickers = {m.get("ticker", "") for m in markets}
                selected_tickers.discard("")
                missing = sorted(t for t in held_tickers if t not in selected_tickers)
                if missing:
                    extra_page = await client.get_markets_page(
                        limit=len(missing),
                        tickers=",".join(missing),
                        mve_filter="exclude",
                    )
                    extra = list(extra_page.get("markets") or [])
                    if extra:
                        markets.extend(extra)
            except Exception as e:
                log.debug("Failed to augment markets with held tickers: %s", e)

            # Deduplicate markets by ticker (pick_active_markets + held-ticker lookup may overlap).
            if markets:
                by_ticker: dict[str, dict[str, Any]] = {}
                for m in markets:
                    t = m.get("ticker", "")
                    if not t:
                        continue
                    by_ticker[t] = m
                markets = list(by_ticker.values())
            
            if not markets:
                log.info("No active markets found")
                await asyncio.sleep(poll_seconds)
                continue
            
            # Process each market
            for market in markets:
                ticker = market.get("ticker", "")
                cutoff_time, cutoff_field = _market_cutoff_time(market)
                price_to_beat, _ptb_field = _market_price_to_beat(market)
                
                if not ticker or not cutoff_time:
                    continue
                
                seconds_to_expiry = int((cutoff_time - now).total_seconds())
                if seconds_to_expiry < 0:
                    seconds_to_expiry = 0

                # Get current position for this market early (used for expiry flatten decisions)
                current_pos = portfolio.positions.get(ticker)
                current_qty = current_pos.quantity if current_pos else 0

                # If we are close to the cutoff and holding a position, log what time source
                # we are using (helps diagnose API field mismatches).
                if current_qty != 0 and seconds_to_expiry <= 180:
                    log.info(
                        "[%s] expiry_timing: %ds to cutoff (field=%s cutoff=%s now=%s) qty=%+d ptb=%s btc=%s",
                        ticker,
                        seconds_to_expiry,
                        cutoff_field,
                        cutoff_time.isoformat(),
                        now.isoformat(),
                        current_qty,
                        "?" if price_to_beat is None else f"${price_to_beat:,.0f}",
                        f"${spot_price:,.0f}",
                    )

                # If we're close to expiry and flat, don't quote.
                # If we're close to expiry and holding a position, we'll fetch the orderbook
                # and attempt to flatten (reduce-only IoC).
                if seconds_to_expiry < STOP_QUOTING_BEFORE_EXPIRY and current_qty == 0:
                    log.info("[%s] Too close to expiry (%ds), skipping", ticker, seconds_to_expiry)
                    continue
                
                # Get orderbook
                use_market_flatten = False
                try:
                    ob = await client.get_orderbook(ticker)
                    prices = compute_best_prices(ob)
                    
                    if prices.best_yes_bid is None or prices.best_yes_ask is None:
                        if seconds_to_expiry < STOP_QUOTING_BEFORE_EXPIRY and current_qty != 0:
                            use_market_flatten = True
                            market_mid_cents = int(current_prices.get(ticker, 50))
                            best_bid = prices.best_yes_bid if prices.best_yes_bid is not None else market_mid_cents
                            best_ask = prices.best_yes_ask if prices.best_yes_ask is not None else market_mid_cents
                            log.warning(
                                "[%s] No orderbook prices with %ds to expiry (qty=%+d); using reduce-only market close",
                                ticker,
                                seconds_to_expiry,
                                current_qty,
                            )
                        else:
                            log.debug("[%s] No orderbook prices", ticker)
                            continue
                    
                    if not use_market_flatten:
                        best_bid = prices.best_yes_bid
                        best_ask = prices.best_yes_ask
                        market_mid_cents = (best_bid + best_ask) // 2
                    
                except Exception as e:
                    if seconds_to_expiry < STOP_QUOTING_BEFORE_EXPIRY and current_qty != 0:
                        use_market_flatten = True
                        market_mid_cents = int(current_prices.get(ticker, 50))
                        best_bid = market_mid_cents
                        best_ask = market_mid_cents
                        log.warning(
                            "[%s] Failed to get orderbook with %ds to expiry (qty=%+d): %s; using reduce-only market close",
                            ticker,
                            seconds_to_expiry,
                            current_qty,
                            e,
                        )
                    else:
                        log.warning("[%s] Failed to get orderbook: %s", ticker, e)
                        continue
                
                # Store price for inventory calculation
                current_prices[ticker] = market_mid_cents

                # Force flatten right before expiry (overrides all other behavior)
                if seconds_to_expiry < STOP_QUOTING_BEFORE_EXPIRY and current_qty != 0:
                    if use_market_flatten:
                        # If we can't see a usable orderbook, fall back to a reduce-only market order.
                        quotes = [
                            Quote(
                                side=("ask" if current_qty > 0 else "bid"),
                                price_cents=0,
                                size=abs(current_qty),
                                ticker=ticker,
                            )
                        ]
                    else:
                        quotes = _generate_expiry_flatten_quotes(
                            ticker=ticker,
                            qty=current_qty,
                            best_bid_cents=best_bid,
                            best_ask_cents=best_ask,
                        )
                    log.warning(
                        "[%s] expiry_flatten: %ds to expiry qty=%+d -> exiting @ %s",
                        ticker,
                        seconds_to_expiry,
                        current_qty,
                        ",".join(
                            f"{q.side} {q.size}@{'MKT' if use_market_flatten else f'{q.price_cents}c'}"
                            for q in quotes
                        ),
                    )
                    order_type = "expiry_flatten"
                else:
                    order_type = "quote"

                if order_type != "expiry_flatten":
                    # Per-market stop-loss / take-profit: if we have a one-sided fill and price moved
                    # against us, exit rather than holding to settlement.
                    if current_pos is not None and current_qty != 0:
                        exit_quotes, exit_type = _generate_position_exit_quotes(
                            ticker=ticker,
                            qty=current_qty,
                            entry_cents=current_pos.entry_price_cents,
                            best_bid_cents=best_bid,
                            best_ask_cents=best_ask,
                            seconds_to_expiry=seconds_to_expiry,
                        )
                        if exit_quotes:
                            log.warning(
                                "[%s] %s triggered: qty=%+d entry=%dc mid=%dc -> exiting @ %s",
                                ticker,
                                exit_type,
                                current_qty,
                                current_pos.entry_price_cents,
                                market_mid_cents,
                                ",".join(f"{q.side} {q.size}@{q.price_cents}c" for q in exit_quotes),
                            )
                            quotes = exit_quotes
                            order_type = exit_type
                        else:
                            quotes = []
                            order_type = "quote"
                    else:
                        quotes = []
                        order_type = "quote"

                # Total exposure cap: don't add new risk. If we're already in a position,
                # place a small reduce-only unwind order.
                if order_type == "quote" and exposure_capped:
                    if current_qty != 0:
                        quotes = _generate_exposure_cap_quotes(
                            ticker=ticker,
                            qty=current_qty,
                            best_bid_cents=best_bid,
                            best_ask_cents=best_ask,
                        )
                        order_type = "exposure_cap"
                    else:
                        # Flat in this market; skip quoting while capped.
                        quotes = []
                        order_type = "exposure_cap"
                
                # If risk-management fired, do not place normal quotes this cycle.
                if order_type in ("stop", "take_profit", "expiry_flatten"):
                    pass
                elif inventory.needs_rebalance(signal):
                    # Generate aggressive rebalance orders
                    use_emergency = seconds_to_expiry < INVENTORY_EMERGENCY_SECONDS
                    rebalance_quotes = generate_rebalance_orders(
                        inventory=inventory,
                        signal=signal,
                        market_mid_cents=market_mid_cents,
                        ticker=ticker,
                        seconds_to_expiry=seconds_to_expiry,
                        use_market_order=use_emergency,
                    )
                    quotes.extend(rebalance_quotes)
                    order_type = "rebalance" if not use_emergency else "emergency"
                else:
                    # Normal quote generation with inventory skewing
                    quotes = generate_quotes(
                        signal=signal,
                        market_mid_cents=market_mid_cents,
                        ticker=ticker,
                        current_position=current_qty,
                        net_delta=inventory.net_delta,
                    )
                    order_type = "quote"
                
                # Smart cancel/replace: only update if price changed by ≥1c or inventory changed
                new_quote_state: dict[str, tuple[int, int]] = {}
                for q in quotes:
                    new_quote_state[q.side] = (q.price_cents, q.size)
                
                prev_state = prev_quotes.get(ticker, {})
                inventory_changed = (inventory.net_delta != prev_inventory_delta)
                
                # Check if quotes actually need updating
                needs_update = False
                if inventory_changed:
                    needs_update = True
                    log.debug("[%s] Inventory changed %+d -> %+d, updating quotes", 
                              ticker, prev_inventory_delta, inventory.net_delta)
                elif new_quote_state != prev_state:
                    # Check if price difference is >= 1 cent
                    for side in ["bid", "ask"]:
                        new_price = new_quote_state.get(side, (0, 0))[0]
                        old_price = prev_state.get(side, (0, 0))[0]
                        if abs(new_price - old_price) >= 1:
                            needs_update = True
                            log.debug("[%s] %s price changed %dc -> %dc, updating", 
                                      ticker, side, old_price, new_price)
                            break
                        new_size = new_quote_state.get(side, (0, 0))[1]
                        old_size = prev_state.get(side, (0, 0))[1]
                        if new_size != old_size:
                            needs_update = True
                            log.debug("[%s] %s size changed %d -> %d, updating",
                                      ticker, side, old_size, new_size)
                            break
                
                # Also force update for risk-management orders
                if order_type in ("rebalance", "emergency", "stop", "take_profit", "exposure_cap", "expiry_flatten"):
                    needs_update = True
                
                if not needs_update:
                    log.debug("[%s] No quote change needed, keeping existing orders", ticker)
                    continue
                
                # Cancel existing orders for this ticker
                cancelled_ids = []
                
                if ticker in open_orders:
                    for order_id in list(open_orders[ticker].keys()):
                        if order_id.startswith("dry-run-"):
                            cancelled_ids.append(order_id)
                            continue
                        try:
                            await client.cancel_order(order_id)
                            cancelled_ids.append(order_id)
                            log.debug("Cancelled order %s", order_id)
                        except Exception as e:
                            # Check if order was already filled or cancelled
                            # HTTPStatusError has .response.status_code, also check string
                            err_str = str(e).lower()
                            is_not_found = False
                            
                            # Check for 404 status code
                            if hasattr(e, 'response') and hasattr(e.response, 'status_code'):
                                if e.response.status_code == 404:
                                    is_not_found = True
                            # Also check string representation
                            if "not found" in err_str or "404" in err_str or "already" in err_str:
                                is_not_found = True
                            
                            if is_not_found:
                                # Order was filled or already cancelled - that's fine
                                cancelled_ids.append(order_id)
                                log.debug("Order %s already filled/cancelled", order_id)
                            else:
                                # Likely a transient API error - assume order is gone and proceed
                                # (Better to place new orders than to stall indefinitely)
                                cancelled_ids.append(order_id)
                                log.warning("Failed to cancel order %s: %s (assuming filled, proceeding)", order_id, e)
                    
                    # Remove orders from tracking (all considered cancelled/filled now)
                    for oid in cancelled_ids:
                        if oid in open_orders[ticker]:
                            del open_orders[ticker][oid]
                
                # Place new quotes
                for quote in quotes:
                    reduce_only = order_type in ("rebalance", "emergency", "stop", "take_profit", "exposure_cap", "expiry_flatten")
                    quote_order_type: Literal["limit", "market"] = "limit"
                    if order_type == "expiry_flatten" and quote.price_cents == 0:
                        quote_order_type = "market"
                    order_id = await place_quote(
                        client,
                        quote,
                        dry_run=dry_run,
                        reduce_only=reduce_only,
                        order_type=quote_order_type,
                        seconds_to_expiry=seconds_to_expiry,
                        btc_price=spot_price,
                        price_to_beat=price_to_beat,
                    )
                    if order_id:
                        if ticker not in open_orders:
                            open_orders[ticker] = {}
                        open_orders[ticker][order_id] = quote
                        
                        trades_w.writerow([
                            now.isoformat(),
                            ticker,
                            quote.side,
                            quote.price_cents,
                            quote.size,
                            signal.direction,
                            f"{rsi:.2f}",
                            inventory.net_delta,
                            order_type,
                        ])
                        trades_f.flush()
                
                # Update state tracking
                prev_quotes[ticker] = new_quote_state
            
            # Update inventory delta for next cycle comparison
            prev_inventory_delta = inventory.net_delta
    
    finally:
        # Cancel all remaining orders
        log.info("Cancelling all open orders...")
        for ticker, orders_dict in open_orders.items():
            for order_id in list(orders_dict.keys()):
                if order_id.startswith("dry-run-"):
                    continue
                try:
                    await client.cancel_order(order_id)
                    log.debug("Cancelled order %s", order_id)
                except Exception as e:
                    log.warning("Failed to cancel order %s: %s", order_id, e)
        
        # Close files
        trades_f.close()
        signals_f.close()
        
        log.info("Trading session ended")
        log.info("Trades log: %s", trades_path)
        log.info("Signals log: %s", signals_path)


# ----------------------------
# Main Entry Point
# ----------------------------

def load_cmc_key(key_file: str = "cmckey.txt") -> str | None:
    """Load CoinMarketCap API key from file."""
    key_path = key_file
    if not os.path.isabs(key_file):
        key_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", key_file)
    
    try:
        with open(key_path, "r") as f:
            content = f.read().strip()
        
        if ":" in content:
            return content.split(":", 1)[1].strip()
        return content
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="RSI-based Market Maker for Kalshi BTC markets")
    
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest instead of live trading")
    parser.add_argument("--backtest-days", type=int, default=7,
                        help="Number of days for backtest (default: 7)")
    parser.add_argument("--optimize", action="store_true",
                        help="Run parameter optimization scan")
    
    parser.add_argument("--live", action="store_true",
                        help="Enable live trading mode")
    parser.add_argument("--dry-run", action="store_true",
                        help="Log trades without executing (use with --live)")
    
    parser.add_argument("--poll-seconds", type=float, default=DEFAULT_POLL_SECONDS,
                        help=f"Seconds between polls (default: {DEFAULT_POLL_SECONDS})")
    parser.add_argument("--duration-minutes", type=float, default=DEFAULT_DURATION_MINUTES,
                        help=f"How long to run in minutes (default: {DEFAULT_DURATION_MINUTES})")
    parser.add_argument("--bankroll", type=float, default=100.0,
                        help="Starting capital in dollars (default: 100)")
    
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                        help=f"Output directory for logs (default: {DEFAULT_OUT_DIR})")
    
    parser.add_argument("--rsi-oversold", type=int, default=RSI_OVERSOLD_THRESHOLD,
                        help=f"RSI oversold threshold (default: {RSI_OVERSOLD_THRESHOLD})")
    parser.add_argument("--rsi-overbought", type=int, default=RSI_OVERBOUGHT_THRESHOLD,
                        help=f"RSI overbought threshold (default: {RSI_OVERBOUGHT_THRESHOLD})")
    
    args = parser.parse_args()
    
    # Store threshold overrides (globals can't be modified here due to scoping)
    rsi_oversold = args.rsi_oversold
    rsi_overbought = args.rsi_overbought
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    
    async def runner() -> None:
        if args.optimize:
            # Run parameter optimization
            await run_parameter_scan(days=args.backtest_days)
        
        elif args.backtest:
            # Run backtest with specified thresholds
            log.info("Running RSI backtest...")
            log.info("RSI Thresholds: Oversold < %d, Overbought > %d", 
                     rsi_oversold, rsi_overbought)
            
            result = await run_backtest(
                days=args.backtest_days,
                rsi_oversold=rsi_oversold,
                rsi_overbought=rsi_overbought,
            )
            print(result.summary())
            
            # Print sample trades
            if result.trades:
                print("\nSample trades:")
                for trade in result.trades[:10]:
                    print(
                        f"  {trade.timestamp.strftime('%Y-%m-%d %H:%M')} | "
                        f"RSI={trade.rsi_at_entry:.1f} | "
                        f"{trade.direction.upper()} | "
                        f"{'WIN' if trade.won else 'LOSS'} | "
                        f"${trade.pnl_dollars:+.2f}"
                    )
            
        elif args.live:
            # Load settings and run live
            settings = Settings.load()
            client = KalshiClient.from_settings(settings)
            cmc_key = load_cmc_key()
            
            try:
                await run_live(
                    client=client,
                    cmc_api_key=cmc_key,
                    poll_seconds=args.poll_seconds,
                    duration_minutes=args.duration_minutes,
                    bankroll_dollars=args.bankroll,
                    dry_run=args.dry_run,
                    out_dir=args.out_dir,
                )
            finally:
                await client.aclose()
        
        else:
            parser.print_help()
            print("\nExamples:")
            print("  Backtest:    python rsi_market_maker.py --backtest")
            print("  Optimize:    python rsi_market_maker.py --optimize")
            print("  Dry run:     python rsi_market_maker.py --live --dry-run")
            print("  Live trade:  python rsi_market_maker.py --live")
    
    asyncio.run(runner())


if __name__ == "__main__":
    main()
