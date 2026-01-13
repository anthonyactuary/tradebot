"""Order Flow Imbalance (OFI) Model.

Tracks changes in bid/ask volume to predict short-term price direction.
Strong OFI predicts price movement in that direction.

References:
- Cont, Kukanov, Stoikov (2014): "The Price Impact of Order Book Events"
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import NamedTuple


class OrderbookSnapshot(NamedTuple):
    """A snapshot of the top-of-book."""
    best_bid: int  # cents
    best_ask: int  # cents
    bid_size: int  # total contracts at best bid
    ask_size: int  # total contracts at best ask


@dataclass
class OrderFlowImbalance:
    """Compute Order Flow Imbalance from orderbook changes.
    
    OFI = Σ(ΔBid_volume when bid improves) - Σ(ΔAsk_volume when ask improves)
    
    Positive OFI → buying pressure → price likely to rise
    Negative OFI → selling pressure → price likely to fall
    
    We normalize by recent volume to get a signal in [-1, 1].
    """
    
    # Rolling window for OFI accumulation
    window: int = 20
    
    # Decay factor for exponential smoothing of OFI
    decay: float = 0.9
    
    # Internal state
    _prev_snapshot: OrderbookSnapshot | None = None
    _raw_ofi: float = 0.0
    _smoothed_ofi: float = 0.0
    _ofi_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    _volume_history: deque[float] = field(default_factory=lambda: deque(maxlen=20))
    
    def update(self, snapshot: OrderbookSnapshot) -> float:
        """Update OFI with a new orderbook snapshot.
        
        Args:
            snapshot: Current top-of-book state
            
        Returns:
            Normalized OFI signal in approximately [-1, 1]
        """
        if self._prev_snapshot is None:
            self._prev_snapshot = snapshot
            return 0.0
        
        prev = self._prev_snapshot
        curr = snapshot
        
        # Compute raw OFI contribution from this update
        delta_ofi = 0.0
        
        # Bid side contribution
        if curr.best_bid > prev.best_bid:
            # Bid improved (price went up) - add all current bid size
            delta_ofi += curr.bid_size
        elif curr.best_bid == prev.best_bid:
            # Same price level - track size change
            delta_ofi += (curr.bid_size - prev.bid_size)
        else:
            # Bid dropped - subtract the lost volume
            delta_ofi -= prev.bid_size
            
        # Ask side contribution (opposite sign)
        if curr.best_ask < prev.best_ask:
            # Ask improved (price went down) - subtract current ask size
            delta_ofi -= curr.ask_size
        elif curr.best_ask == prev.best_ask:
            # Same price level - track size change (negative = more supply)
            delta_ofi -= (curr.ask_size - prev.ask_size)
        else:
            # Ask lifted - add the consumed volume
            delta_ofi += prev.ask_size
        
        # Track volume for normalization
        total_volume = curr.bid_size + curr.ask_size
        self._volume_history.append(max(1.0, float(total_volume)))
        
        # Update smoothed OFI
        self._smoothed_ofi = self.decay * self._smoothed_ofi + (1 - self.decay) * delta_ofi
        self._ofi_history.append(self._smoothed_ofi)
        
        # Store for next iteration
        self._prev_snapshot = curr
        
        # Normalize by average volume to get signal in ~[-1, 1]
        avg_volume = sum(self._volume_history) / len(self._volume_history) if self._volume_history else 1.0
        normalized = self._smoothed_ofi / max(1.0, avg_volume)
        
        # Clamp to reasonable bounds
        return max(-1.0, min(1.0, normalized))
    
    def signal(self) -> float:
        """Get current OFI signal without updating.
        
        Returns:
            Normalized OFI signal in approximately [-1, 1]
        """
        if not self._volume_history:
            return 0.0
        avg_volume = sum(self._volume_history) / len(self._volume_history)
        normalized = self._smoothed_ofi / max(1.0, avg_volume)
        return max(-1.0, min(1.0, normalized))
    
    def directional_bias(self) -> float:
        """Get directional bias for quote skewing.
        
        Returns:
            Value in [-1, 1] where:
            - Positive: skew quotes UP (raise bid and ask)
            - Negative: skew quotes DOWN (lower bid and ask)
        """
        return self.signal()
    
    def reset(self) -> None:
        """Reset the OFI tracker state."""
        self._prev_snapshot = None
        self._raw_ofi = 0.0
        self._smoothed_ofi = 0.0
        self._ofi_history.clear()
        self._volume_history.clear()
