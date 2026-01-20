"""
Spot price aggregation and volatility estimation for arbitrage strategy.

Components:
- SyntheticSpot: Aggregates Coinbase + Kraken mids with outlier rejection
- EwmaVariance: EWMA variance on 1-second log returns for diffusion model
- PFairEstimator: Computes fair probability using spot/strike/volatility
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------
# Utilities
# ---------------------------

def now_ms() -> int:
    return int(time.time() * 1000)


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def norm_cdf(x: float) -> float:
    """Standard normal CDF using erf."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def safe_log(x: float) -> float:
    if x <= 0:
        return float("-inf")
    return math.log(x)


# ---------------------------
# Configuration
# ---------------------------

@dataclass(frozen=True)
class VolConfig:
    """EWMA variance configuration for 1-second log returns."""
    # var_t = lam*var_{t-1} + (1-lam)*r^2
    lam: float = 0.985
    min_var: float = 1e-12


# ---------------------------
# Spot Sample
# ---------------------------

@dataclass
class SpotSample:
    ts_ms: int
    mid: float


# ---------------------------
# Synthetic Spot
# ---------------------------

@dataclass
class SyntheticSpot:
    """
    Keeps a synthetic mid from multiple exchanges with simple outlier rejection.
    
    Updates from Coinbase and Kraken are combined, with outlier detection
    when the two sources diverge significantly.
    """
    max_divergence_bps: float = 12.0

    # Latest mids from each source
    cb_mid: Optional[float] = None
    kr_mid: Optional[float] = None
    cb_ts_ms: Optional[int] = None
    kr_ts_ms: Optional[int] = None

    # History for staleness calculation
    history: List[SpotSample] = field(default_factory=list)

    def update_coinbase_mid(self, mid: float, ts_ms: Optional[int] = None) -> None:
        self.cb_mid = mid
        self.cb_ts_ms = ts_ms if ts_ms is not None else now_ms()
        self._append_history(mid, self.cb_ts_ms)

    def update_kraken_mid(self, mid: float, ts_ms: Optional[int] = None) -> None:
        self.kr_mid = mid
        self.kr_ts_ms = ts_ms if ts_ms is not None else now_ms()
        self._append_history(mid, self.kr_ts_ms)

    def _append_history(self, mid: float, ts_ms: int) -> None:
        self.history.append(SpotSample(ts_ms=ts_ms, mid=mid))
        # Keep only a few seconds of history (cheap ring buffer)
        cutoff = ts_ms - 10_000
        while self.history and self.history[0].ts_ms < cutoff:
            self.history.pop(0)

    def mid(self) -> Optional[float]:
        """Return the synthetic mid, handling missing data and outliers."""
        if self.cb_mid is None and self.kr_mid is None:
            return None
        if self.cb_mid is None:
            return self.kr_mid
        if self.kr_mid is None:
            return self.cb_mid

        # Outlier check: if sources diverge significantly, pick the more reliable one
        bps = abs(safe_log(self.cb_mid / self.kr_mid)) * 10_000
        if bps > self.max_divergence_bps:
            # Downweight outlier: choose the one closer to recent median
            recent = [s.mid for s in self.history[-10:]] or [self.cb_mid, self.kr_mid]
            med = sorted(recent)[len(recent) // 2]
            if abs(self.cb_mid - med) < abs(self.kr_mid - med):
                return self.cb_mid
            return self.kr_mid

        return 0.5 * self.cb_mid + 0.5 * self.kr_mid

    def spot_move_bps_over(self, window_ms: int) -> Optional[float]:
        """Calculate absolute move in bps over the given window."""
        cur = self.mid()
        if cur is None:
            return None
        t = now_ms()
        target = t - window_ms
        past = None
        for s in reversed(self.history):
            if s.ts_ms <= target:
                past = s.mid
                break
        if past is None:
            return None
        return abs(safe_log(cur / past)) * 10_000

    def age_ms(self) -> Optional[int]:
        """Return age of the most recent update in milliseconds."""
        latest_ts = None
        if self.cb_ts_ms is not None:
            latest_ts = self.cb_ts_ms
        if self.kr_ts_ms is not None:
            if latest_ts is None or self.kr_ts_ms > latest_ts:
                latest_ts = self.kr_ts_ms
        if latest_ts is None:
            return None
        return now_ms() - latest_ts


# ---------------------------
# EWMA Variance
# ---------------------------

@dataclass
class EwmaVariance:
    """
    EWMA variance estimator on 1-second log returns.
    
    Used for the diffusion model to estimate remaining volatility to expiry.
    """
    cfg: VolConfig
    last_ts_ms: Optional[int] = None
    last_mid: Optional[float] = None
    var_1s: float = 1e-8  # Start with small non-zero

    def update(self, ts_ms: int, mid: float) -> None:
        if self.last_ts_ms is None or self.last_mid is None:
            self.last_ts_ms, self.last_mid = ts_ms, mid
            return

        dt_ms = ts_ms - self.last_ts_ms
        if dt_ms <= 0:
            return

        # Convert to "per-second" log return by scaling to 1s
        r = safe_log(mid / self.last_mid)
        r_1s = r * (1000.0 / dt_ms)

        lam = self.cfg.lam
        self.var_1s = max(self.cfg.min_var, lam * self.var_1s + (1 - lam) * (r_1s * r_1s))

        self.last_ts_ms, self.last_mid = ts_ms, mid

    def sd_remaining(self, remaining_s: float) -> float:
        """Return standard deviation over remaining horizon in log space."""
        return math.sqrt(self.var_1s * max(0.0, remaining_s))

    def annualized_vol(self) -> float:
        """Return annualized volatility (for diagnostics)."""
        # sqrt(var_1s) is 1-second vol; annualize with sqrt(seconds_per_year)
        seconds_per_year = 365.25 * 24 * 60 * 60
        return math.sqrt(self.var_1s * seconds_per_year)


# ---------------------------
# P-Fair Estimator
# ---------------------------

@dataclass
class PFairEstimator:
    """
    Fair probability estimator using diffusion model.
    
    p_fair = Phi(ln(S/K) / sigma_remaining)
    
    where:
    - S is spot price
    - K is strike price
    - sigma_remaining is the expected standard deviation over remaining time
    """
    spot: SyntheticSpot
    vol: EwmaVariance

    def p_fair(self, strike: float, remaining_s: float) -> Optional[float]:
        """
        Compute fair probability that spot > strike at expiry.
        
        Args:
            strike: The strike/reference price
            remaining_s: Seconds until expiry
            
        Returns:
            Fair probability [0.01, 0.99] or None if data unavailable
        """
        mid = self.spot.mid()
        if mid is None:
            return None
        if remaining_s <= 0:
            # Already expired: deterministic outcome
            return 1.0 if mid > strike else 0.0

        sd = self.vol.sd_remaining(remaining_s)
        if sd <= 0:
            return 1.0 if mid > strike else 0.0

        z = safe_log(mid / strike) / sd
        p = norm_cdf(z)
        return clamp(p, 0.01, 0.99)

    def p_fair_with_details(
        self, strike: float, remaining_s: float
    ) -> tuple[Optional[float], dict]:
        """
        Compute p_fair with diagnostic details.
        
        Returns:
            (p_fair, details_dict)
        """
        mid = self.spot.mid()
        details = {
            "spot_mid": mid,
            "strike": strike,
            "remaining_s": remaining_s,
            "var_1s": self.vol.var_1s,
        }
        
        if mid is None:
            return None, details
            
        if remaining_s <= 0:
            p = 1.0 if mid > strike else 0.0
            details["sd_remaining"] = 0.0
            details["z_score"] = float("inf") if mid > strike else float("-inf")
            return p, details

        sd = self.vol.sd_remaining(remaining_s)
        details["sd_remaining"] = sd
        
        if sd <= 0:
            p = 1.0 if mid > strike else 0.0
            details["z_score"] = float("inf") if mid > strike else float("-inf")
            return p, details

        z = safe_log(mid / strike) / sd
        p = norm_cdf(z)
        p = clamp(p, 0.01, 0.99)
        
        details["z_score"] = z
        details["p_fair"] = p
        
        return p, details
