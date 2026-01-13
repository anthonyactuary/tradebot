"""Live BTC Price Feed from CoinMarketCap + Coinbase.

Provides real-time BTC price data to enhance trading decisions
by comparing actual BTC price to Kalshi market strike prices.

Features:
- Live BTC price from CoinMarketCap
- Empirical 15-min volatility from Coinbase candles
- Strike signal computation with calibrated vol

Usage:
    feed = BTCPriceFeed(api_key="your-cmc-key")
    await feed.calibrate_volatility()  # Get empirical 15-min vol
    price = await feed.get_price()
    signal = feed.compute_strike_signal(strike=90000.0, seconds_to_expiry=300)
"""

from __future__ import annotations

import asyncio
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from typing import Any

import httpx


log = logging.getLogger(__name__)


# CoinMarketCap API endpoint for latest quotes
CMC_API_URL = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/quotes/latest"

# Coinbase public API for candles (no auth needed)
COINBASE_CANDLES_URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"

# Default cache duration (seconds) - balance freshness vs API rate limits
DEFAULT_CACHE_SECONDS = 60.0

# Default 15-min volatility (fallback if calibration fails)
DEFAULT_15M_VOL = 0.0020  # 0.20% - based on empirical data

# BTC symbol ID on CoinMarketCap
BTC_SYMBOL = "BTC"


@dataclass
class BTCPriceSnapshot:
    """A snapshot of BTC price data."""
    price: float                    # Current USD price
    timestamp: float                # Unix timestamp when fetched
    percent_change_1h: float = 0.0  # 1-hour change %
    percent_change_24h: float = 0.0 # 24-hour change %
    volume_24h: float = 0.0         # 24h volume
    
    def age_seconds(self) -> float:
        """How old is this snapshot in seconds."""
        return time.time() - self.timestamp
    
    def is_stale(self, max_age: float = 60.0) -> bool:
        """Check if snapshot is too old."""
        return self.age_seconds() > max_age


@dataclass
class StrikeSignal:
    """Signal based on BTC price vs strike comparison."""
    btc_price: float           # Current BTC price
    strike_price: float        # Strike to beat
    distance_dollars: float    # BTC - strike (positive = above strike)
    distance_percent: float    # % distance from strike
    seconds_to_expiry: int     # Time remaining
    vol_15m: float = DEFAULT_15M_VOL  # Calibrated 15-min volatility
    
    # Computed signals
    direction: str = ""        # "above", "below", "at"
    confidence: float = 0.0    # 0-1, how confident YES will win
    fair_prob: float = 0.5     # Estimated fair probability for YES
    recommended_action: str = ""  # "favor_yes", "favor_no", "neutral"
    
    def __post_init__(self) -> None:
        # Compute direction
        if self.distance_dollars > 10:  # $10 buffer
            self.direction = "above"
        elif self.distance_dollars < -10:
            self.direction = "below"
        else:
            self.direction = "at"
        
        # === FAIR PROBABILITY ESTIMATION ===
        # For "BTC up in 15 min" markets:
        # - Strike is the price at market OPEN (15 min ago)
        # - YES wins if BTC at expiry >= strike
        # - We need P(BTC at expiry >= strike | BTC now)
        
        # Use empirical 15-min volatility (calibrated from Coinbase)
        # Scale volatility by sqrt(time remaining / 15 min)
        full_window_seconds = 900  # 15 minutes
        time_fraction = max(0.01, self.seconds_to_expiry / full_window_seconds)
        
        # Stdev in dollars for remaining time
        btc_15m_stdev_dollars = self.strike_price * self.vol_15m
        btc_remaining_stdev = btc_15m_stdev_dollars * (time_fraction ** 0.5)
        
        # How many standard deviations away from strike?
        z_score = self.distance_dollars / btc_remaining_stdev if btc_remaining_stdev > 0 else 0
        
        # Normal CDF approximation using logistic function
        # This gives P(BTC at expiry >= strike)
        def norm_cdf_approx(z: float) -> float:
            """Approximate normal CDF using logistic function."""
            # Logistic approximation: CDF(z) ≈ 1 / (1 + e^(-1.7*z))
            clamped_z = max(-6, min(6, z))  # Prevent overflow
            return 1.0 / (1.0 + 2.71828 ** (-1.7 * clamped_z))
        
        # Fair probability that YES wins
        self.fair_prob = norm_cdf_approx(z_score)
        self.fair_prob = max(0.02, min(0.98, self.fair_prob))
        
        # === CONFIDENCE ===
        # How confident are we in our fair_prob estimate?
        abs_z = abs(z_score)
        
        # Time factor (less time = more confident in our estimate)
        if self.seconds_to_expiry <= 60:
            time_factor = 1.0
        elif self.seconds_to_expiry <= 180:
            time_factor = 0.8
        elif self.seconds_to_expiry <= 300:
            time_factor = 0.6
        elif self.seconds_to_expiry <= 600:
            time_factor = 0.4
        else:
            time_factor = 0.2
        
        # Z-score factor (further from strike = more confident)
        if abs_z >= 2.0:
            z_factor = 1.0
        elif abs_z >= 1.0:
            z_factor = 0.8
        elif abs_z >= 0.5:
            z_factor = 0.6
        elif abs_z >= 0.25:
            z_factor = 0.4
        else:
            z_factor = 0.2
        
        self.confidence = time_factor * z_factor
        
        # === RECOMMENDED ACTION ===
        if self.confidence >= 0.3:
            if self.fair_prob > 0.55:
                self.recommended_action = "favor_yes"
            elif self.fair_prob < 0.45:
                self.recommended_action = "favor_no"
            else:
                self.recommended_action = "neutral"
        else:
            self.recommended_action = "neutral"


@dataclass
class BTCPriceFeed:
    """Async BTC price feed from CoinMarketCap.
    
    Features:
    - Caches prices to respect rate limits
    - Computes signals relative to strike prices
    - Tracks price momentum
    - Calibrates volatility from Coinbase 15-min candles
    """
    
    api_key: str
    cache_seconds: float = DEFAULT_CACHE_SECONDS
    
    # Internal state
    _client: httpx.AsyncClient | None = None
    _cache: BTCPriceSnapshot | None = None
    _price_history: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, price)
    _max_history: int = 100
    _vol_15m: float = DEFAULT_15M_VOL  # Calibrated volatility
    _vol_calibrated_at: float = 0.0    # When we last calibrated
    
    @classmethod
    def from_file(cls, key_file: str = "cmckey.txt", **kwargs) -> "BTCPriceFeed":
        """Create feed from API key file."""
        key_path = key_file
        if not os.path.isabs(key_file):
            # Look in project root
            key_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", key_file)
        
        with open(key_path, "r") as f:
            content = f.read().strip()
        
        # Handle "API Key: xxx" format
        if ":" in content:
            api_key = content.split(":", 1)[1].strip()
        else:
            api_key = content
        
        return cls(api_key=api_key, **kwargs)
    
    async def _ensure_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client."""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=10.0)
        return self._client
    
    async def close(self) -> None:
        """Close HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
    
    async def calibrate_volatility(self, force: bool = False) -> float:
        """Calibrate 15-min volatility from Coinbase candles.
        
        Args:
            force: Force recalibration even if recent
            
        Returns:
            Calibrated 15-min volatility as decimal (e.g., 0.002 = 0.2%)
        """
        # Only recalibrate every 30 minutes unless forced
        if not force and (time.time() - self._vol_calibrated_at) < 1800:
            return self._vol_15m
        
        client = await self._ensure_client()
        
        try:
            params = {"granularity": 900}  # 15 minute candles
            headers = {"Accept": "application/json"}
            
            response = await client.get(COINBASE_CANDLES_URL, params=params, headers=headers)
            response.raise_for_status()
            candles = response.json()
            
            if len(candles) < 10:
                log.warning("Not enough candles for calibration: %d", len(candles))
                return self._vol_15m
            
            # Coinbase format: [time, low, high, open, close, volume]
            # Most recent first, so reverse for chronological order
            closes = [float(c[4]) for c in candles]
            closes.reverse()
            
            # Compute 15-min returns
            returns = []
            for i in range(1, len(closes)):
                ret = (closes[i] - closes[i - 1]) / closes[i - 1]
                returns.append(ret)
            
            if len(returns) < 5:
                return self._vol_15m
            
            # Standard deviation of 15-min returns
            self._vol_15m = statistics.stdev(returns)
            self._vol_calibrated_at = time.time()
            
            log.info(
                "Calibrated 15-min vol: %.4f%% (from %d candles)",
                self._vol_15m * 100,
                len(candles),
            )
            
            return self._vol_15m
            
        except Exception as e:
            log.warning("Volatility calibration failed: %s", e)
            return self._vol_15m
    
    async def fetch_price(self) -> BTCPriceSnapshot:
        """Fetch fresh BTC price from CoinMarketCap.
        
        Returns:
            BTCPriceSnapshot with current price data
        """
        client = await self._ensure_client()
        
        headers = {
            "X-CMC_PRO_API_KEY": self.api_key,
            "Accept": "application/json",
        }
        
        params = {
            "symbol": BTC_SYMBOL,
            "convert": "USD",
        }
        
        try:
            response = await client.get(CMC_API_URL, headers=headers, params=params)
            response.raise_for_status()
            data = response.json()
            
            btc_data = data.get("data", {}).get("BTC", {})
            quote = btc_data.get("quote", {}).get("USD", {})
            
            snapshot = BTCPriceSnapshot(
                price=float(quote.get("price", 0)),
                timestamp=time.time(),
                percent_change_1h=float(quote.get("percent_change_1h", 0)),
                percent_change_24h=float(quote.get("percent_change_24h", 0)),
                volume_24h=float(quote.get("volume_24h", 0)),
            )
            
            # Update cache and history
            self._cache = snapshot
            self._price_history.append((snapshot.timestamp, snapshot.price))
            if len(self._price_history) > self._max_history:
                self._price_history.pop(0)
            
            log.debug("Fetched BTC price: $%.2f", snapshot.price)
            return snapshot
            
        except Exception as e:
            log.error("Failed to fetch BTC price: %s", e)
            # Return cached value if available
            if self._cache is not None:
                log.warning("Using cached price (age: %.1fs)", self._cache.age_seconds())
                return self._cache
            raise
    
    async def get_price(self) -> BTCPriceSnapshot:
        """Get BTC price, using cache if fresh enough.
        
        Returns:
            BTCPriceSnapshot (may be cached)
        """
        if self._cache is not None and self._cache.age_seconds() < self.cache_seconds:
            return self._cache
        
        return await self.fetch_price()
    
    def get_cached_price(self) -> BTCPriceSnapshot | None:
        """Get cached price without fetching (non-async).
        
        Returns:
            Cached snapshot or None if no cache
        """
        return self._cache
    
    def compute_strike_signal(
        self,
        strike: float,
        seconds_to_expiry: int,
        btc_price: float | None = None,
    ) -> StrikeSignal | None:
        """Compute trading signal based on BTC vs strike.
        
        Args:
            strike: The strike price to beat (floor_strike from Kalshi)
            seconds_to_expiry: Time until market expires
            btc_price: Optional BTC price (uses cache if not provided)
            
        Returns:
            StrikeSignal with recommendation, or None if no price available
        """
        if btc_price is None:
            if self._cache is None:
                return None
            btc_price = self._cache.price
        
        if strike <= 0:
            return None
        
        distance_dollars = btc_price - strike
        distance_percent = (distance_dollars / strike) * 100
        
        return StrikeSignal(
            btc_price=btc_price,
            strike_price=strike,
            distance_dollars=distance_dollars,
            distance_percent=distance_percent,
            seconds_to_expiry=seconds_to_expiry,
            vol_15m=self._vol_15m,  # Use calibrated vol
        )
    
    def get_calibrated_vol(self) -> float:
        """Get current calibrated 15-min volatility."""
        return self._vol_15m
    
    def price_momentum(self, lookback_seconds: float = 60.0) -> float:
        """Calculate recent price momentum.
        
        Args:
            lookback_seconds: How far back to look
            
        Returns:
            Price change in dollars over lookback period (positive = rising)
        """
        if len(self._price_history) < 2:
            return 0.0
        
        now = time.time()
        cutoff = now - lookback_seconds
        
        # Find oldest price within lookback
        old_price = None
        for ts, price in self._price_history:
            if ts >= cutoff:
                old_price = price
                break
        
        if old_price is None:
            old_price = self._price_history[0][1]
        
        current_price = self._price_history[-1][1]
        return current_price - old_price
    
    def price_volatility(self, lookback_seconds: float = 300.0) -> float:
        """Calculate recent price volatility (standard deviation).
        
        Args:
            lookback_seconds: How far back to look
            
        Returns:
            Standard deviation of prices in dollars
        """
        if len(self._price_history) < 3:
            return 0.0
        
        now = time.time()
        cutoff = now - lookback_seconds
        
        prices = [p for ts, p in self._price_history if ts >= cutoff]
        if len(prices) < 3:
            prices = [p for _, p in self._price_history[-10:]]
        
        if len(prices) < 2:
            return 0.0
        
        mean = sum(prices) / len(prices)
        variance = sum((p - mean) ** 2 for p in prices) / len(prices)
        return variance ** 0.5


async def test_feed() -> None:
    """Test the BTC price feed."""
    logging.basicConfig(level=logging.INFO)
    
    feed = BTCPriceFeed.from_file("cmckey.txt")
    
    try:
        # Calibrate volatility first
        print("Calibrating 15-min volatility from Coinbase...")
        vol = await feed.calibrate_volatility()
        print(f"Calibrated 15-min vol: {vol*100:.4f}%")
        print()
        
        print("Fetching BTC price from CoinMarketCap...")
        snapshot = await feed.get_price()
        print(f"BTC Price: ${snapshot.price:,.2f}")
        print(f"1h Change: {snapshot.percent_change_1h:+.2f}%")
        print(f"24h Change: {snapshot.percent_change_24h:+.2f}%")
        print()
        
        # Test strike signal
        strike = 90000.0
        signal = feed.compute_strike_signal(strike=strike, seconds_to_expiry=300)
        if signal:
            print(f"Strike Signal (strike=${strike:,.2f}, 5 min left):")
            print(f"  Direction: {signal.direction}")
            print(f"  Distance: ${signal.distance_dollars:+,.2f} ({signal.distance_percent:+.3f}%)")
            print(f"  15-min vol used: {signal.vol_15m*100:.4f}%")
            print(f"  Fair Prob (YES): {signal.fair_prob:.1%}")
            print(f"  Confidence: {signal.confidence:.2f}")
            print(f"  Recommendation: {signal.recommended_action}")
        
    finally:
        await feed.close()


if __name__ == "__main__":
    asyncio.run(test_feed())
