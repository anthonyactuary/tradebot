"""Kelly Criterion Position Sizing.

The Kelly Criterion maximizes long-run growth rate of capital.
For binary outcomes: f* = (p × b - q) / b

Where:
- f* = optimal fraction of bankroll to bet
- p = probability of winning
- q = 1 - p = probability of losing  
- b = odds (payout ratio, e.g., 2:1 means b=2)

We use fractional Kelly (typically 0.25-0.5x) to reduce variance.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass  
class KellyCriterion:
    """Kelly Criterion position sizer for binary options.
    
    For Kalshi markets:
    - Buy YES at price p: win (1-p) if YES, lose p if NO
    - Sell YES at price p: win p if NO, lose (1-p) if YES
    """
    
    # Fraction of Kelly to use (reduces variance at cost of growth)
    kelly_fraction: float = 0.25  # Quarter Kelly is conservative
    
    # Maximum fraction of bankroll per trade (hard cap)
    max_position_pct: float = 0.05  # Never more than 5%
    
    # Minimum edge required to take a position
    min_edge: float = 0.02  # Need at least 2% edge
    
    def optimal_size_buy_yes(
        self, 
        *,
        fair_prob: float,
        bid_price: float,  # Price we'd buy at (probability)
        bankroll: float,
    ) -> float:
        """Compute optimal position size for buying YES.
        
        When buying YES at price p:
        - If YES: we get $1, net profit = (1 - p)
        - If NO: we get $0, net loss = p
        
        Kelly: f* = (fair × odds - (1-fair)) / odds
        Where odds = (1-p)/p for buying YES
        
        Args:
            fair_prob: Our estimate of true probability
            bid_price: Price we're buying at (as probability 0-1)
            bankroll: Current bankroll in dollars
            
        Returns:
            Optimal position size in dollars (0 if no edge)
        """
        if bid_price <= 0 or bid_price >= 1:
            return 0.0
        if fair_prob <= 0 or fair_prob >= 1:
            return 0.0
            
        # Edge = our fair prob - price we pay
        edge = fair_prob - bid_price
        
        if edge < self.min_edge:
            return 0.0  # Not enough edge
        
        # Odds for buying YES: win (1-p), lose p
        # b = (1-p) / p
        odds = (1.0 - bid_price) / bid_price
        
        # Kelly formula: f* = (p × b - q) / b
        # Where p = fair_prob, q = 1 - fair_prob
        q = 1.0 - fair_prob
        kelly = (fair_prob * odds - q) / odds
        
        # Apply fractional Kelly and caps
        kelly = max(0.0, kelly)  # Never negative
        position_pct = kelly * self.kelly_fraction
        position_pct = min(position_pct, self.max_position_pct)
        
        return position_pct * bankroll
    
    def optimal_size_sell_yes(
        self,
        *,
        fair_prob: float,
        ask_price: float,  # Price we'd sell at (probability)
        bankroll: float,
    ) -> float:
        """Compute optimal position size for selling YES.
        
        When selling YES at price p:
        - If NO: we keep $p, net profit = p  
        - If YES: we pay $1, net loss = (1-p)
        
        Args:
            fair_prob: Our estimate of true probability
            ask_price: Price we're selling at (as probability 0-1)
            bankroll: Current bankroll in dollars
            
        Returns:
            Optimal position size in dollars (0 if no edge)
        """
        if ask_price <= 0 or ask_price >= 1:
            return 0.0
        if fair_prob <= 0 or fair_prob >= 1:
            return 0.0
        
        # Edge for selling = price we get - fair prob
        edge = ask_price - fair_prob
        
        if edge < self.min_edge:
            return 0.0  # Not enough edge
        
        # For selling YES, we're effectively betting on NO
        # Odds: win p, lose (1-p)
        # b = p / (1-p)
        odds = ask_price / (1.0 - ask_price)
        
        # Our "win" probability is P(NO) = 1 - fair_prob
        p_win = 1.0 - fair_prob
        q_lose = fair_prob
        
        kelly = (p_win * odds - q_lose) / odds
        
        # Apply fractional Kelly and caps
        kelly = max(0.0, kelly)
        position_pct = kelly * self.kelly_fraction
        position_pct = min(position_pct, self.max_position_pct)
        
        return position_pct * bankroll
    
    def recommended_size(
        self,
        *,
        fair_prob: float,
        bid_price: float,
        ask_price: float,
        bankroll: float,
    ) -> tuple[float, float]:
        """Get recommended sizes for both sides.
        
        Args:
            fair_prob: Our estimate of true probability
            bid_price: Best bid (what we'd buy at)
            ask_price: Best ask (what we'd sell at)  
            bankroll: Current bankroll
            
        Returns:
            Tuple of (buy_size, sell_size) in dollars
        """
        buy_size = self.optimal_size_buy_yes(
            fair_prob=fair_prob,
            bid_price=bid_price,
            bankroll=bankroll,
        )
        sell_size = self.optimal_size_sell_yes(
            fair_prob=fair_prob,
            ask_price=ask_price,
            bankroll=bankroll,
        )
        return buy_size, sell_size
    
    def edge_required_for_size(
        self,
        *,
        target_size_pct: float,
        price: float,
    ) -> float:
        """Compute edge required to justify a given position size.
        
        Useful for understanding what edge our quotes need.
        
        Args:
            target_size_pct: Desired position as pct of bankroll
            price: Expected fill price
            
        Returns:
            Required edge (difference between fair and price)
        """
        # Reverse engineer Kelly
        # position_pct = kelly × kelly_fraction
        # kelly = edge / (1 - price) roughly
        kelly_needed = target_size_pct / self.kelly_fraction
        edge = kelly_needed * (1.0 - price)
        return edge
