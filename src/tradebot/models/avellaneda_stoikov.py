"""Avellaneda-Stoikov Optimal Market Making Model.

The academic gold standard for market-making spread computation.

Reference:
- Avellaneda & Stoikov (2008): "High-frequency trading in a limit order book"

Key insight: The optimal spread depends on:
1. Volatility (σ): Higher vol → wider spread
2. Time to horizon (T-t): Less time → tighter spread (reduce inventory risk)
3. Risk aversion (γ): More risk-averse → wider spread
4. Inventory (q): More inventory → skew quotes to reduce it

The reservation price (where MM is indifferent to trading):
    r(t) = s(t) - q × γ × σ² × (T - t)

The optimal spread:
    δ = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/k)

Where k is the order arrival intensity parameter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class AvellanedaStoikov:
    """Avellaneda-Stoikov optimal spread calculator.
    
    This implements the continuous-time optimal market-making model
    adapted for discrete 15-minute binary options.
    """
    
    # Risk aversion parameter (higher = more conservative spreads)
    # Typical range: 0.1 to 10.0
    gamma: float = 1.0
    
    # Order arrival intensity (higher = tighter spreads acceptable)
    # This is λ in the paper, representing fill rate
    # Typical range: 0.1 to 5.0
    kappa: float = 1.5
    
    # Minimum spread (floor)
    min_spread: float = 0.02  # 2 cents
    
    # Maximum spread (cap)
    max_spread: float = 0.15  # 15 cents
    
    def reservation_price(
        self,
        *,
        mid_price: float,
        inventory: int,  # Positive = long, negative = short
        volatility: float,
        time_remaining: float,  # In whatever units (e.g., fraction of 15 min)
    ) -> float:
        """Compute the reservation price.
        
        The reservation price is where the market maker is indifferent
        to buying or selling. It's shifted from mid based on inventory.
        
        r = s - q × γ × σ² × (T - t)
        
        If we're long (q > 0), reservation price is BELOW mid (want to sell)
        If we're short (q < 0), reservation price is ABOVE mid (want to buy)
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory (contracts)
            volatility: Estimated volatility
            time_remaining: Time until horizon (0-1, where 1 = full period)
            
        Returns:
            Reservation price
        """
        # Inventory adjustment
        adjustment = inventory * self.gamma * (volatility ** 2) * time_remaining
        reservation = mid_price - adjustment
        
        # Clamp to valid probability range
        return max(0.01, min(0.99, reservation))
    
    def optimal_spread(
        self,
        *,
        volatility: float,
        time_remaining: float,
    ) -> float:
        """Compute the optimal bid-ask spread.
        
        δ = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/k)
        
        Args:
            volatility: Estimated volatility
            time_remaining: Time until horizon (0-1)
            
        Returns:
            Optimal spread (total, divide by 2 for half-spread)
        """
        # Inventory risk component
        inventory_component = self.gamma * (volatility ** 2) * time_remaining
        
        # Adverse selection / fill probability component
        # This accounts for the fact that wider spreads reduce fill probability
        if self.gamma > 0 and self.kappa > 0:
            adverse_component = (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.kappa)
        else:
            adverse_component = 0.0
        
        spread = inventory_component + adverse_component
        
        # Apply bounds
        return max(self.min_spread, min(self.max_spread, spread))
    
    def optimal_quotes(
        self,
        *,
        mid_price: float,
        inventory: int,
        volatility: float,
        time_remaining: float,
    ) -> tuple[float, float]:
        """Compute optimal bid and ask prices.
        
        Combines reservation price (inventory skew) with optimal spread.
        
        Args:
            mid_price: Current mid price
            inventory: Current inventory (positive = long)
            volatility: Estimated volatility
            time_remaining: Time to horizon (0-1)
            
        Returns:
            Tuple of (bid, ask) prices
        """
        # Get reservation price (mid adjusted for inventory)
        reservation = self.reservation_price(
            mid_price=mid_price,
            inventory=inventory,
            volatility=volatility,
            time_remaining=time_remaining,
        )
        
        # Get optimal spread
        spread = self.optimal_spread(
            volatility=volatility,
            time_remaining=time_remaining,
        )
        
        half_spread = spread / 2.0
        
        bid = reservation - half_spread
        ask = reservation + half_spread
        
        # Clamp to valid range
        bid = max(0.01, min(0.98, bid))
        ask = max(0.02, min(0.99, ask))
        
        # Ensure bid < ask
        if bid >= ask:
            mid = (bid + ask) / 2
            bid = mid - 0.01
            ask = mid + 0.01
        
        return bid, ask
    
    def inventory_skew_cents(
        self,
        *,
        inventory: int,
        volatility: float,
        time_remaining: float,
    ) -> float:
        """Get the inventory-induced price skew in cents.
        
        Useful for understanding how much inventory shifts quotes.
        
        Args:
            inventory: Current inventory
            volatility: Estimated volatility
            time_remaining: Time to horizon
            
        Returns:
            Skew in cents (positive = shift down, negative = shift up)
        """
        skew = inventory * self.gamma * (volatility ** 2) * time_remaining
        return skew * 100  # Convert to cents
    
    def urgency_factor(
        self,
        *,
        inventory: int,
        max_inventory: int = 10,
    ) -> float:
        """Get an urgency factor for position reduction.
        
        As inventory approaches limits, urgency increases.
        
        Args:
            inventory: Current inventory (absolute value used)
            max_inventory: Maximum acceptable inventory
            
        Returns:
            Urgency factor from 1.0 (no urgency) to 2.0+ (high urgency)
        """
        abs_inv = abs(inventory)
        if abs_inv <= 0:
            return 1.0
        
        utilization = abs_inv / max(1, max_inventory)
        
        # Exponential urgency as we approach max
        # At 50% utilization: 1.25x
        # At 75% utilization: 1.5x
        # At 100% utilization: 2.0x
        return 1.0 + utilization
