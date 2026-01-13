"""Fill Probability Model.

Estimates P(fill | price, time, market conditions).

Key insight: Orders closer to mid fill more often, but also have
more adverse selection. Orders far from mid rarely fill but are safer.

This helps optimize quote placement:
- Quote aggressively when fill probability is low (wide market)
- Quote conservatively when fill probability is high (tight market)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple


class FillRecord(NamedTuple):
    """Record of a fill event."""
    distance_from_mid: float  # How far our quote was from mid (cents)
    time_to_fill: float  # Seconds from quote to fill
    side: str  # "bid" or "ask"
    filled: bool  # Whether it filled


@dataclass
class FillProbabilityModel:
    """Model to estimate probability of order fill.
    
    Uses an exponential decay model:
    P(fill) = base_prob × exp(-decay × distance_from_mid)
    
    Where distance_from_mid is how far our quote is from the current mid.
    
    This is calibrated from historical fill data when available,
    otherwise uses reasonable defaults.
    """
    
    # Base fill probability at mid (if we quoted exactly at mid)
    base_prob: float = 0.8
    
    # Decay rate per cent of distance from mid
    # Higher = fill probability drops faster with distance
    decay_per_cent: float = 0.15
    
    # Time decay - longer orders have higher fill probability
    time_boost_per_second: float = 0.01
    max_time_boost: float = 0.3
    
    # Track historical fills for calibration
    _fill_history: deque[FillRecord] = field(default_factory=lambda: deque(maxlen=100))
    _calibrated: bool = False
    
    def estimate(
        self,
        *,
        quote_price_cents: int,
        mid_price_cents: int,
        side: str,  # "bid" or "ask"
        time_remaining_seconds: float = 60.0,
        spread_cents: int | None = None,
    ) -> float:
        """Estimate probability of fill for a quote.
        
        Args:
            quote_price_cents: Our quote price in cents
            mid_price_cents: Current mid price in cents
            side: "bid" or "ask"
            time_remaining_seconds: Time until we'd cancel
            spread_cents: Current bid-ask spread (optional, for context)
            
        Returns:
            Estimated fill probability in [0, 1]
        """
        # Distance from mid
        if side == "bid":
            # For bids, positive distance means we're below mid (less likely to fill)
            distance = mid_price_cents - quote_price_cents
        else:
            # For asks, positive distance means we're above mid
            distance = quote_price_cents - mid_price_cents
        
        # Base probability with exponential decay
        prob = self.base_prob * math.exp(-self.decay_per_cent * max(0, distance))
        
        # Time boost - longer exposure = higher fill chance
        time_boost = min(self.max_time_boost, self.time_boost_per_second * time_remaining_seconds)
        prob = prob + time_boost * (1 - prob)  # Asymptotic boost
        
        # Spread context - tighter spreads mean more competition
        if spread_cents is not None and spread_cents > 0:
            # If spread is tight, fill probability is higher but so is adverse selection
            spread_factor = 1.0 + 0.1 * max(0, 5 - spread_cents)  # Boost for tight spreads
            prob = min(0.95, prob * spread_factor)
        
        return max(0.01, min(0.95, prob))
    
    def record_fill(
        self,
        *,
        quote_price_cents: int,
        mid_at_quote_cents: int,
        side: str,
        time_to_fill_seconds: float,
        filled: bool,
    ) -> None:
        """Record a fill event for model calibration.
        
        Args:
            quote_price_cents: Price of our quote
            mid_at_quote_cents: Mid price when we placed the quote
            side: "bid" or "ask"
            time_to_fill_seconds: Time from quote placement to fill (or cancel)
            filled: Whether it filled
        """
        if side == "bid":
            distance = mid_at_quote_cents - quote_price_cents
        else:
            distance = quote_price_cents - mid_at_quote_cents
        
        record = FillRecord(
            distance_from_mid=distance,
            time_to_fill=time_to_fill_seconds,
            side=side,
            filled=filled,
        )
        self._fill_history.append(record)
        
        # Recalibrate if we have enough data
        if len(self._fill_history) >= 20:
            self._calibrate()
    
    def _calibrate(self) -> None:
        """Calibrate model parameters from historical fills.
        
        Uses simple binning by distance to estimate decay rate.
        """
        if len(self._fill_history) < 20:
            return
        
        # Bin by distance
        bins: dict[int, list[bool]] = {}
        for record in self._fill_history:
            bin_idx = int(record.distance_from_mid)
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(record.filled)
        
        # Need at least 2 bins with data
        valid_bins = [(d, fills) for d, fills in bins.items() if len(fills) >= 3]
        if len(valid_bins) < 2:
            return
        
        # Estimate fill rate at each distance
        fill_rates = []
        for distance, fills in valid_bins:
            rate = sum(fills) / len(fills)
            if rate > 0:
                fill_rates.append((distance, rate))
        
        if len(fill_rates) < 2:
            return
        
        # Fit exponential decay: log(rate) = log(base) - decay × distance
        # Simple linear regression in log space
        log_rates = [(d, math.log(r)) for d, r in fill_rates if r > 0]
        if len(log_rates) < 2:
            return
        
        n = len(log_rates)
        mean_d = sum(d for d, _ in log_rates) / n
        mean_log_r = sum(lr for _, lr in log_rates) / n
        
        numerator = sum((d - mean_d) * (lr - mean_log_r) for d, lr in log_rates)
        denominator = sum((d - mean_d) ** 2 for d, _ in log_rates)
        
        if denominator > 0:
            self.decay_per_cent = -numerator / denominator  # Negative because decay
            self.base_prob = math.exp(mean_log_r + self.decay_per_cent * mean_d)
            
            # Clamp to reasonable values
            self.decay_per_cent = max(0.05, min(0.5, self.decay_per_cent))
            self.base_prob = max(0.3, min(0.95, self.base_prob))
            
            self._calibrated = True
    
    def optimal_distance(
        self,
        *,
        target_fill_prob: float = 0.5,
    ) -> float:
        """Get optimal distance from mid for target fill probability.
        
        Args:
            target_fill_prob: Desired fill probability
            
        Returns:
            Recommended distance from mid in cents
        """
        if target_fill_prob >= self.base_prob:
            return 0.0  # Need to be at mid or better
        
        if target_fill_prob <= 0:
            return 100.0  # Impossible, return large distance
        
        # Solve: target = base × exp(-decay × distance)
        # distance = -ln(target/base) / decay
        distance = -math.log(target_fill_prob / self.base_prob) / self.decay_per_cent
        
        return max(0.0, distance)
    
    def expected_profit(
        self,
        *,
        quote_price_cents: int,
        mid_price_cents: int,
        side: str,
        profit_if_fill_cents: float,
        loss_if_adverse_cents: float = 0.0,
    ) -> float:
        """Compute expected profit from a quote placement.
        
        Args:
            quote_price_cents: Our quote price
            mid_price_cents: Current mid
            side: "bid" or "ask"
            profit_if_fill_cents: Profit if filled and price stays
            loss_if_adverse_cents: Loss if filled and price moves against
            
        Returns:
            Expected profit in cents
        """
        fill_prob = self.estimate(
            quote_price_cents=quote_price_cents,
            mid_price_cents=mid_price_cents,
            side=side,
        )
        
        # Simple expected value
        # Assumes adverse selection probability increases with fill probability
        adverse_prob = fill_prob * 0.3  # 30% of fills are adverse
        
        expected = (
            fill_prob * (1 - adverse_prob) * profit_if_fill_cents
            - fill_prob * adverse_prob * loss_if_adverse_cents
        )
        
        return expected
    
    def is_calibrated(self) -> bool:
        """Check if model has been calibrated from data."""
        return self._calibrated
    
    def reset(self) -> None:
        """Reset model to defaults."""
        self._fill_history.clear()
        self._calibrated = False
        self.base_prob = 0.8
        self.decay_per_cent = 0.15
