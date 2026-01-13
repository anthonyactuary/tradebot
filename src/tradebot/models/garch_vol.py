"""GARCH(1,1) Volatility Model.

Captures volatility clustering - periods of high vol tend to persist.
Better than simple RMS volatility for predicting near-term vol.

Model: σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}

Where:
- ω (omega): long-run variance weight  
- α (alpha): weight on recent squared return (shock impact)
- β (beta): weight on previous variance (persistence)
- α + β should be < 1 for stationarity (typically ~0.95)
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field


@dataclass
class GARCHVolatility:
    """GARCH(1,1) volatility estimator.
    
    This is a simplified online implementation that doesn't require
    full MLE estimation. We use reasonable default parameters that
    work well for high-frequency data.
    """
    
    # GARCH parameters
    omega: float = 0.00001   # Long-run variance baseline
    alpha: float = 0.10      # Shock impact (recent return weight)
    beta: float = 0.85       # Persistence (previous variance weight)
    
    # Minimum variance floor (prevents vol from going to zero)
    min_variance: float = 1e-8
    
    # For initialization, we need a few observations
    warmup_window: int = 5
    
    # Internal state
    _prev_price: float | None = None
    _variance: float = 0.0001  # Initial variance estimate
    _returns: deque[float] = field(default_factory=lambda: deque(maxlen=100))
    _initialized: bool = False
    
    def __post_init__(self) -> None:
        # Validate parameters
        if self.alpha + self.beta >= 1.0:
            raise ValueError("alpha + beta must be < 1 for stationarity")
    
    def update(self, price: float) -> float:
        """Update volatility estimate with a new price.
        
        Args:
            price: Current mid price (probability in [0,1])
            
        Returns:
            Current volatility estimate (standard deviation)
        """
        if self._prev_price is None:
            self._prev_price = price
            return self.volatility()
        
        # Compute return (handle edge cases for probability prices)
        if self._prev_price > 0:
            ret = (price - self._prev_price) / self._prev_price
        else:
            ret = 0.0
        
        self._returns.append(ret)
        self._prev_price = price
        
        # Wait for warmup period before using GARCH
        if len(self._returns) < self.warmup_window:
            # Use simple variance during warmup
            if len(self._returns) >= 2:
                mean_ret = sum(self._returns) / len(self._returns)
                self._variance = sum((r - mean_ret) ** 2 for r in self._returns) / len(self._returns)
            return self.volatility()
        
        # Initialize variance from sample if not done
        if not self._initialized:
            mean_ret = sum(self._returns) / len(self._returns)
            self._variance = sum((r - mean_ret) ** 2 for r in self._returns) / len(self._returns)
            self._initialized = True
        
        # GARCH(1,1) update
        # σ²_t = ω + α × ε²_{t-1} + β × σ²_{t-1}
        epsilon_sq = ret * ret  # Squared return (shock)
        self._variance = (
            self.omega 
            + self.alpha * epsilon_sq 
            + self.beta * self._variance
        )
        
        # Apply floor
        self._variance = max(self.min_variance, self._variance)
        
        return self.volatility()
    
    def volatility(self) -> float:
        """Get current volatility estimate (standard deviation).
        
        Returns:
            Volatility as standard deviation of returns
        """
        return math.sqrt(max(self.min_variance, self._variance))
    
    def variance(self) -> float:
        """Get current variance estimate.
        
        Returns:
            Variance of returns
        """
        return max(self.min_variance, self._variance)
    
    def long_run_volatility(self) -> float:
        """Get the unconditional (long-run) volatility.
        
        This is the volatility the process mean-reverts to.
        
        Returns:
            Long-run volatility estimate
        """
        # Unconditional variance = ω / (1 - α - β)
        unconditional_var = self.omega / max(0.01, 1.0 - self.alpha - self.beta)
        return math.sqrt(unconditional_var)
    
    def half_life(self) -> float:
        """Get the half-life of volatility shocks in periods.
        
        How many periods until a shock decays to half its impact.
        
        Returns:
            Half-life in number of updates
        """
        persistence = self.alpha + self.beta
        if persistence >= 1.0 or persistence <= 0.0:
            return float('inf')
        return math.log(0.5) / math.log(persistence)
    
    def is_high_vol(self, threshold_mult: float = 1.5) -> bool:
        """Check if current vol is elevated vs long-run.
        
        Args:
            threshold_mult: Multiplier above long-run to consider "high"
            
        Returns:
            True if current vol > threshold_mult × long_run_vol
        """
        return self.volatility() > threshold_mult * self.long_run_volatility()
    
    def reset(self) -> None:
        """Reset the GARCH estimator state."""
        self._prev_price = None
        self._variance = 0.0001
        self._returns.clear()
        self._initialized = False
