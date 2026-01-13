"""Regime Detection Model.

Detects whether the market is in:
1. TRENDING regime (momentum works, mean reversion fails)
2. MEAN_REVERTING regime (mean reversion works, momentum fails)
3. RANDOM regime (neither works reliably)

Uses the Hurst exponent as the primary signal:
- H > 0.5: trending (persistent)
- H < 0.5: mean-reverting (anti-persistent)
- H ≈ 0.5: random walk
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import NamedTuple


class Regime(Enum):
    """Market regime classification."""
    TRENDING = "trending"
    MEAN_REVERTING = "mean_reverting"
    RANDOM = "random"


class RegimeInfo(NamedTuple):
    """Regime detection result."""
    regime: Regime
    hurst: float
    confidence: float  # 0-1, how confident in the classification


@dataclass
class RegimeDetector:
    """Detect market regime using Hurst exponent and other signals.
    
    The Hurst exponent H indicates:
    - H > 0.55: Trending/persistent - price moves tend to continue
    - H < 0.45: Mean-reverting/anti-persistent - moves tend to reverse  
    - 0.45 ≤ H ≤ 0.55: Random walk, no predictable pattern
    
    We use the R/S (rescaled range) method for estimation.
    """
    
    # Window for Hurst estimation
    window: int = 50
    
    # Thresholds for regime classification
    trending_threshold: float = 0.55
    mean_reverting_threshold: float = 0.45
    
    # Minimum observations needed
    min_observations: int = 20
    
    # Internal state
    _prices: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _returns: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _hurst: float = 0.5  # Default to random
    _prev_price: float | None = None
    
    def update(self, price: float) -> RegimeInfo:
        """Update regime detection with a new price.
        
        Args:
            price: Current price (mid probability)
            
        Returns:
            Current regime information
        """
        self._prices.append(price)
        
        if self._prev_price is not None and self._prev_price > 0:
            ret = (price - self._prev_price) / self._prev_price
            self._returns.append(ret)
        
        self._prev_price = price
        
        # Need minimum observations for Hurst estimation
        if len(self._returns) < self.min_observations:
            return RegimeInfo(
                regime=Regime.RANDOM,
                hurst=0.5,
                confidence=0.0,
            )
        
        # Compute Hurst exponent
        self._hurst = self._compute_hurst()
        
        # Classify regime
        if self._hurst > self.trending_threshold:
            regime = Regime.TRENDING
        elif self._hurst < self.mean_reverting_threshold:
            regime = Regime.MEAN_REVERTING
        else:
            regime = Regime.RANDOM
        
        # Confidence based on distance from 0.5
        confidence = min(1.0, abs(self._hurst - 0.5) * 4)
        
        return RegimeInfo(
            regime=regime,
            hurst=self._hurst,
            confidence=confidence,
        )
    
    def _compute_hurst(self) -> float:
        """Compute Hurst exponent using R/S method.
        
        The R/S method:
        1. Compute mean-adjusted cumulative sum
        2. R = max(cumsum) - min(cumsum) (range)
        3. S = std(returns)
        4. R/S scales as n^H
        5. Estimate H from log-log regression
        """
        returns = list(self._returns)[-self.window:]
        n = len(returns)
        
        if n < self.min_observations:
            return 0.5
        
        # We'll compute R/S for different sub-periods and regress
        # Simplified: just compute for full period and a few sub-periods
        
        rs_values = []
        sizes = []
        
        for size in [n // 4, n // 2, n]:
            if size < 5:
                continue
            sub_returns = returns[:size]
            rs = self._compute_rs(sub_returns)
            if rs > 0:
                rs_values.append(math.log(rs))
                sizes.append(math.log(size))
        
        if len(rs_values) < 2:
            return 0.5
        
        # Simple linear regression for Hurst
        # H = slope of log(R/S) vs log(n)
        n_points = len(sizes)
        mean_x = sum(sizes) / n_points
        mean_y = sum(rs_values) / n_points
        
        numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(sizes, rs_values))
        denominator = sum((x - mean_x) ** 2 for x in sizes)
        
        if denominator == 0:
            return 0.5
        
        hurst = numerator / denominator
        
        # Clamp to valid range
        return max(0.0, min(1.0, hurst))
    
    def _compute_rs(self, returns: list[float]) -> float:
        """Compute R/S statistic for a return series."""
        n = len(returns)
        if n < 2:
            return 0.0
        
        mean = sum(returns) / n
        
        # Mean-adjusted cumulative sum
        adjusted = [r - mean for r in returns]
        cumsum = []
        total = 0.0
        for a in adjusted:
            total += a
            cumsum.append(total)
        
        # Range
        R = max(cumsum) - min(cumsum)
        
        # Standard deviation
        variance = sum((r - mean) ** 2 for r in returns) / n
        S = math.sqrt(variance) if variance > 0 else 1e-10
        
        return R / S if S > 0 else 0.0
    
    def regime(self) -> Regime:
        """Get current regime classification."""
        if self._hurst > self.trending_threshold:
            return Regime.TRENDING
        elif self._hurst < self.mean_reverting_threshold:
            return Regime.MEAN_REVERTING
        return Regime.RANDOM
    
    def hurst(self) -> float:
        """Get current Hurst exponent estimate."""
        return self._hurst
    
    def should_follow_momentum(self) -> bool:
        """Should we follow price momentum?"""
        return self._hurst > self.trending_threshold
    
    def should_fade_moves(self) -> bool:
        """Should we fade (bet against) price moves?"""
        return self._hurst < self.mean_reverting_threshold
    
    def momentum_weight(self) -> float:
        """Get weight for momentum signal.
        
        Returns:
            1.0 in trending regime, 0.0 in mean-reverting, 0.5 in random
        """
        if self._hurst >= 0.5:
            # Scale from 0.5 (at H=0.5) to 1.0 (at H=1.0)
            return 0.5 + (self._hurst - 0.5)
        else:
            # Scale from 0.5 (at H=0.5) to 0.0 (at H=0.0)
            return self._hurst
    
    def mean_reversion_weight(self) -> float:
        """Get weight for mean reversion signal.
        
        Returns:
            1.0 in mean-reverting regime, 0.0 in trending, 0.5 in random
        """
        return 1.0 - self.momentum_weight()
    
    def reset(self) -> None:
        """Reset the regime detector state."""
        self._prices.clear()
        self._returns.clear()
        self._hurst = 0.5
        self._prev_price = None
