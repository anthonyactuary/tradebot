"""
Offline Replay & Simulation Harness for Delayed Update Arbitrage Strategy.

Components:
- ReplaySnapshot / MarketSnapshot: Data classes for recorded state
- FakeKalshiClient: Mock client for simulation
- ReplayEngine: Drives replay from JSONL file
- ReplayReport: Post-replay analytics

Usage:
    python -m tradebot.arbitrage.replay_sim --file data/day1.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from tradebot.arbitrage.spot import (
    SyntheticSpot,
    EwmaVariance,
    VolConfig,
    PFairEstimator,
    clamp,
)
from tradebot.arbitrage.strategy import (
    StrategyConfig,
    MarketInfo,
    BookTop,
    RiskManager,
    DelayedUpdateArbStrategy,
    cents_to_prob,
    prob_to_cents,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------
# Replay Snapshot Dataclasses
# ---------------------------

@dataclass
class MarketSnapshot:
    """Snapshot of a single Kalshi market's orderbook."""
    ticker: str
    strike: float
    expiry_ts_ms: int
    yes_bid_cents: int
    yes_ask_cents: int
    no_bid_cents: int
    no_ask_cents: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ticker": self.ticker,
            "strike": self.strike,
            "expiry_ts_ms": self.expiry_ts_ms,
            "yes_bid_cents": self.yes_bid_cents,
            "yes_ask_cents": self.yes_ask_cents,
            "no_bid_cents": self.no_bid_cents,
            "no_ask_cents": self.no_ask_cents,
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MarketSnapshot":
        return MarketSnapshot(
            ticker=d["ticker"],
            strike=float(d["strike"]),
            expiry_ts_ms=int(d["expiry_ts_ms"]),
            yes_bid_cents=int(d["yes_bid_cents"]),
            yes_ask_cents=int(d["yes_ask_cents"]),
            no_bid_cents=int(d["no_bid_cents"]),
            no_ask_cents=int(d["no_ask_cents"]),
        )


@dataclass
class ReplaySnapshot:
    """Full snapshot of world state at a point in time."""
    timestamp_ms: int
    coinbase_mid: Optional[float]
    kraken_mid: Optional[float]
    markets: List[MarketSnapshot]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp_ms": self.timestamp_ms,
            "coinbase_mid": self.coinbase_mid,
            "kraken_mid": self.kraken_mid,
            "markets": [m.to_dict() for m in self.markets],
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ReplaySnapshot":
        return ReplaySnapshot(
            timestamp_ms=int(d["timestamp_ms"]),
            coinbase_mid=d.get("coinbase_mid"),
            kraken_mid=d.get("kraken_mid"),
            markets=[MarketSnapshot.from_dict(m) for m in d.get("markets", [])],
        )


# ---------------------------
# Simulated Order / Fill
# ---------------------------

@dataclass
class SimulatedOrder:
    """Internal representation of a simulated order."""
    order_id: str
    ticker: str
    side: str          # "yes" or "no"
    action: str        # "buy" or "sell"
    count: int
    price_cents: int
    order_type: str    # "maker" or "taker"
    created_ts_ms: int
    filled_qty: int = 0
    cancelled: bool = False


@dataclass
class SimulatedFill:
    """Record of a simulated fill."""
    fill_id: str
    order_id: str
    ticker: str
    side: str
    action: str
    count: int
    price_cents: int
    ts_ms: int
    is_taker: bool


# ---------------------------
# Fake Kalshi Client
# ---------------------------

class FakeKalshiClient:
    """
    Mock KalshiClient for simulation.
    
    Implements the subset of methods used by DelayedUpdateArbStrategy:
    - get_orderbook
    - get_positions
    - create_order
    - cancel_order
    - get_fills
    """
    
    def __init__(self) -> None:
        self.orders: Dict[str, SimulatedOrder] = {}
        self.fills: List[SimulatedFill] = []
        self.positions: Dict[str, int] = {}  # ticker -> signed YES position
        
        # Current snapshot state (updated by ReplayEngine)
        self.current_books: Dict[str, MarketSnapshot] = {}
        self.current_ts_ms: int = 0
        self.next_books: Optional[Dict[str, MarketSnapshot]] = None  # For maker fill sim
        
        # Statistics
        self.total_orders: int = 0
        self.maker_fills: int = 0
        self.taker_fills: int = 0

    def set_snapshot(
        self,
        ts_ms: int,
        markets: List[MarketSnapshot],
        next_markets: Optional[List[MarketSnapshot]] = None,
    ) -> None:
        """Update the fake client's view of the world."""
        self.current_ts_ms = ts_ms
        self.current_books = {m.ticker: m for m in markets}
        if next_markets is not None:
            self.next_books = {m.ticker: m for m in next_markets}
        else:
            self.next_books = None
        
        # Check if any pending maker orders should fill based on next snapshot
        self._check_maker_fills()

    def _check_maker_fills(self) -> None:
        """Check if pending maker orders would fill against next snapshot."""
        if self.next_books is None:
            return
            
        for order in list(self.orders.values()):
            if order.cancelled or order.filled_qty >= order.count:
                continue
            if order.order_type != "maker":
                continue
                
            next_book = self.next_books.get(order.ticker)
            if next_book is None:
                continue
            
            # Check if maker price crosses or equals opposing best
            if order.action == "buy":
                # Buying YES: need ask to come down to our bid
                if order.side == "yes":
                    if order.price_cents >= next_book.yes_ask_cents:
                        self._fill_order(order, order.count - order.filled_qty)
                # Buying NO: need NO ask to come down to our bid
                else:
                    if order.price_cents >= next_book.no_ask_cents:
                        self._fill_order(order, order.count - order.filled_qty)
            else:
                # Selling YES: need bid to come up to our ask
                if order.side == "yes":
                    if order.price_cents <= next_book.yes_bid_cents:
                        self._fill_order(order, order.count - order.filled_qty)
                else:
                    if order.price_cents <= next_book.no_bid_cents:
                        self._fill_order(order, order.count - order.filled_qty)

    def _fill_order(self, order: SimulatedOrder, qty: int) -> None:
        """Record a fill for an order."""
        if qty <= 0:
            return
            
        fill = SimulatedFill(
            fill_id=str(uuid.uuid4())[:8],
            order_id=order.order_id,
            ticker=order.ticker,
            side=order.side,
            action=order.action,
            count=qty,
            price_cents=order.price_cents,
            ts_ms=self.current_ts_ms,
            is_taker=(order.order_type == "taker"),
        )
        self.fills.append(fill)
        order.filled_qty += qty
        
        if fill.is_taker:
            self.taker_fills += 1
        else:
            self.maker_fills += 1
        
        # Update position
        signed_delta = qty if order.action == "buy" else -qty
        if order.side == "no":
            signed_delta = -signed_delta  # NO position is negative YES
        self.positions[order.ticker] = self.positions.get(order.ticker, 0) + signed_delta

    async def get_orderbook(self, ticker: str) -> Dict[str, Any]:
        """Return orderbook in the format expected by strategy."""
        book = self.current_books.get(ticker)
        if book is None:
            return {"yes": {"bids": [], "asks": []}, "no": {"bids": [], "asks": []}}
        
        return {
            "yes": {
                "bids": [[book.yes_bid_cents, 100]],  # [price_cents, qty]
                "asks": [[book.yes_ask_cents, 100]],
            },
            "no": {
                "bids": [[book.no_bid_cents, 100]],
                "asks": [[book.no_ask_cents, 100]],
            },
        }

    async def get_market(self, ticker: str) -> Dict[str, Any]:
        """Return market data with BBO in the format expected by strategy."""
        book = self.current_books.get(ticker)
        if book is None:
            return {"market": {"yes_bid": 1, "yes_ask": 99, "no_bid": 1, "no_ask": 99}}
        
        return {
            "market": {
                "yes_bid": book.yes_bid_cents,
                "yes_ask": book.yes_ask_cents,
                "no_bid": book.no_bid_cents,
                "no_ask": book.no_ask_cents,
            }
        }

    async def get_positions(self, ticker: str = None) -> Dict[str, Any]:
        """Return positions in the format expected by strategy."""
        positions = []
        if ticker is not None:
            pos = self.positions.get(ticker, 0)
            if pos != 0:
                positions.append({"ticker": ticker, "position": pos})
        else:
            for t, p in self.positions.items():
                if p != 0:
                    positions.append({"ticker": t, "position": p})
        return {"market_positions": positions}

    async def create_order(
        self,
        ticker: str,
        side: str,
        action: str,
        count: int,
        order_type: str = "limit",
        yes_price: Optional[int] = None,
        no_price: Optional[int] = None,
        post_only: bool = False,
        time_in_force: str = "good_till_canceled",
        **kwargs,
    ) -> Dict[str, Any]:
        """Create a simulated order."""
        order_id = f"sim_{uuid.uuid4().hex[:12]}"
        
        price_cents = yes_price if side == "yes" else no_price
        if price_cents is None:
            price_cents = 50  # Fallback
        
        is_taker = (time_in_force == "immediate_or_cancel") or not post_only
        order_type_str = "taker" if is_taker else "maker"
        
        order = SimulatedOrder(
            order_id=order_id,
            ticker=ticker,
            side=side,
            action=action,
            count=count,
            price_cents=price_cents,
            order_type=order_type_str,
            created_ts_ms=self.current_ts_ms,
        )
        
        self.orders[order_id] = order
        self.total_orders += 1
        
        # Taker orders fill immediately at best price
        if is_taker:
            book = self.current_books.get(ticker)
            if book is not None:
                if action == "buy":
                    fill_price = book.yes_ask_cents if side == "yes" else book.no_ask_cents
                else:
                    fill_price = book.yes_bid_cents if side == "yes" else book.no_bid_cents
                order.price_cents = fill_price
                self._fill_order(order, count)
        
        return {"order": {"order_id": order_id}}

    async def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel a simulated order."""
        order = self.orders.get(order_id)
        if order is not None:
            order.cancelled = True
        return {"order_id": order_id}

    async def get_fills(
        self, order_id: Optional[str] = None, limit: int = 100, **kwargs
    ) -> Dict[str, Any]:
        """Return fills for an order."""
        if order_id is not None:
            fills = [f for f in self.fills if f.order_id == order_id]
        else:
            fills = self.fills[-limit:]
        
        return {
            "fills": [
                {
                    "fill_id": f.fill_id,
                    "order_id": f.order_id,
                    "ticker": f.ticker,
                    "side": f.side,
                    "action": f.action,
                    "count": f.count,
                    "price": f.price_cents,
                    "ts": f.ts_ms,
                }
                for f in fills[:limit]
            ]
        }

    async def aclose(self) -> None:
        """No-op for compatibility."""
        pass


# ---------------------------
# Trade Tracking for Report
# ---------------------------

@dataclass
class TradeRecord:
    """Record of a completed trade for analytics."""
    ticker: str
    outcome: str
    entry_ts_ms: int
    exit_ts_ms: int
    entry_price_cents: int
    exit_price_cents: int
    qty: int
    is_maker_entry: bool
    is_stopout: bool


# ---------------------------
# Replay Engine
# ---------------------------

class ReplayEngine:
    """
    Drives replay from a JSONL file of snapshots.
    
    For each snapshot:
    1. Update SyntheticSpot with coinbase_mid / kraken_mid
    2. Update EwmaVariance with synthetic mid
    3. Update FakeKalshiClient's internal orderbook state
    4. Call strategy.tick_market(market) for each market
    5. Track fills, entries, exits, stopouts
    """
    
    def __init__(
        self,
        snapshots: List[ReplaySnapshot],
        cfg: Optional[StrategyConfig] = None,
        vol_cfg: Optional[VolConfig] = None,
    ) -> None:
        self.snapshots = snapshots
        self.cfg = cfg or StrategyConfig()
        self.vol_cfg = vol_cfg or VolConfig()
        
        # Components
        self.spot = SyntheticSpot()
        self.vol = EwmaVariance(cfg=self.vol_cfg)
        self.p_est = PFairEstimator(spot=self.spot, vol=self.vol)
        self.fake_kalshi = FakeKalshiClient()
        self.risk = RiskManager(self.cfg)
        
        # Strategy (will be initialized in run())
        self.strategy: Optional[DelayedUpdateArbStrategy] = None
        
        # Trade records for report
        self.trade_records: List[TradeRecord] = []
        self.entry_info: Dict[str, Dict] = {}  # ticker -> entry info
        
        # Tracking
        self.current_idx: int = 0
        self._virtual_now_ms: int = 0
        self._seen_order_ids: set[str] = set()  # Fix A: track processed orders
        self._max_exposure_over_time: int = 0    # Fix C: track max exposure

    @staticmethod
    def load_from_jsonl(filepath: str) -> List[ReplaySnapshot]:
        """Load snapshots from a JSONL file."""
        snapshots = []
        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                snapshots.append(ReplaySnapshot.from_dict(d))
        return snapshots

    def _patch_now_ms(self) -> None:
        """Monkey-patch now_ms() to use virtual time."""
        import tradebot.arbitrage.spot as spot_module
        import tradebot.arbitrage.strategy as strategy_module
        
        def virtual_now_ms() -> int:
            return self._virtual_now_ms
        
        spot_module.now_ms = virtual_now_ms
        strategy_module.now_ms = virtual_now_ms

    async def run(self) -> "ReplayReport":
        """Run the full replay and return a report."""
        if not self.snapshots:
            return ReplayReport.empty()
        
        # Patch time
        self._patch_now_ms()
        
        # Initialize strategy with fake client and dry_run=True
        self.strategy = DelayedUpdateArbStrategy(
            cfg=self.cfg,
            kalshi=self.fake_kalshi,
            p_est=self.p_est,
            risk=self.risk,
            dry_run=False,  # We want to actually "place" orders with fake client
        )
        
        log.info("Starting replay with %d snapshots", len(self.snapshots))
        
        for i, snap in enumerate(self.snapshots):
            self._virtual_now_ms = snap.timestamp_ms
            
            # Get next snapshot for maker fill simulation
            next_snap = self.snapshots[i + 1] if i + 1 < len(self.snapshots) else None
            next_markets = next_snap.markets if next_snap else None
            
            # Update spot prices
            if snap.coinbase_mid is not None:
                self.spot.update_coinbase_mid(snap.coinbase_mid, snap.timestamp_ms)
            if snap.kraken_mid is not None:
                self.spot.update_kraken_mid(snap.kraken_mid, snap.timestamp_ms)
            
            # Update volatility
            mid = self.spot.mid()
            if mid is not None:
                self.vol.update(snap.timestamp_ms, mid)
            
            # Update fake client's view
            self.fake_kalshi.set_snapshot(
                snap.timestamp_ms,
                snap.markets,
                next_markets,
            )
            
            # Tick strategy for each market
            for m in snap.markets:
                market_info = MarketInfo(
                    ticker=m.ticker,
                    strike=m.strike,
                    expiry_ts_ms=m.expiry_ts_ms,
                )
                try:
                    await self.strategy.tick_market(market_info)
                except Exception as e:
                    log.warning("TICK_ERROR %s error=%s", m.ticker, e)
            
            # Track entries/exits
            self._track_trades(snap)
        
        # Build report
        return self._build_report()

    def _track_trades(self, snap: ReplaySnapshot) -> None:
        """Track trade entries and exits for reporting."""
        # Fix C: track max exposure over time (sum of abs positions)
        current_exposure = sum(abs(p) for p in self.fake_kalshi.positions.values())
        self._max_exposure_over_time = max(self._max_exposure_over_time, current_exposure)
        
        # Check for new entries (Fix A: only process orders we haven't seen)
        for order_id, order in self.fake_kalshi.orders.items():
            # Fix A: skip if already processed
            if order_id in self._seen_order_ids:
                continue
            
            # Fix B: only BUY orders are entries
            if order.action != "buy":
                self._seen_order_ids.add(order_id)
                continue
            
            # Only process if filled
            if order.filled_qty <= 0:
                continue
            
            # Mark as seen
            self._seen_order_ids.add(order_id)
            
            # Record entry if we don't have one for this ticker
            if order.ticker not in self.entry_info:
                self.entry_info[order.ticker] = {
                    "entry_ts_ms": snap.timestamp_ms,
                    "entry_price_cents": order.price_cents,
                    "qty": order.filled_qty,
                    "outcome": order.side,
                    "is_maker_entry": (order.order_type == "maker"),
                    "entry_order_id": order_id,
                }
        
        # Fix B: Check for exits - SELL orders that close a position
        for order_id, order in self.fake_kalshi.orders.items():
            # Only process sell orders we haven't seen
            if order_id in self._seen_order_ids:
                continue
            if order.action != "sell":
                continue
            if order.filled_qty <= 0:
                continue
            
            # Mark as seen
            self._seen_order_ids.add(order_id)
            
            # Match to an entry if we have one
            if order.ticker in self.entry_info:
                info = self.entry_info[order.ticker]
                
                self.trade_records.append(TradeRecord(
                    ticker=order.ticker,
                    outcome=info["outcome"],
                    entry_ts_ms=info["entry_ts_ms"],
                    exit_ts_ms=snap.timestamp_ms,
                    entry_price_cents=info["entry_price_cents"],
                    exit_price_cents=order.price_cents,
                    qty=min(info["qty"], order.filled_qty),
                    is_maker_entry=info["is_maker_entry"],
                    is_stopout=True,  # Mark as stopout (can refine later)
                ))
                del self.entry_info[order.ticker]
        
        # Also check for exits due to position returning to 0 (e.g. settlement simulation)
        for ticker, info in list(self.entry_info.items()):
            pos = self.fake_kalshi.positions.get(ticker, 0)
            if pos == 0:
                # Find exit price from most recent sell fill for this ticker
                exit_fills = [
                    f for f in self.fake_kalshi.fills
                    if f.ticker == ticker and f.action == "sell"
                ]
                exit_price = exit_fills[-1].price_cents if exit_fills else info["entry_price_cents"]
                
                self.trade_records.append(TradeRecord(
                    ticker=ticker,
                    outcome=info["outcome"],
                    entry_ts_ms=info["entry_ts_ms"],
                    exit_ts_ms=snap.timestamp_ms,
                    entry_price_cents=info["entry_price_cents"],
                    exit_price_cents=exit_price,
                    qty=info["qty"],
                    is_maker_entry=info["is_maker_entry"],
                    is_stopout=True,
                ))
                del self.entry_info[ticker]

    def _build_report(self) -> "ReplayReport":
        """Build the final report."""
        total_trades = len(self.trade_records)
        maker_entries = sum(1 for t in self.trade_records if t.is_maker_entry)
        taker_entries = total_trades - maker_entries
        
        # Edge calculations (entry - exit in probability units)
        entry_edges = []
        exit_edges = []
        holding_times = []
        pnl_prob_units = 0.0
        
        for t in self.trade_records:
            entry_prob = t.entry_price_cents / 100.0
            exit_prob = t.exit_price_cents / 100.0
            
            # For buys: profit if exit > entry
            pnl = (exit_prob - entry_prob) * t.qty
            pnl_prob_units += pnl
            
            entry_edges.append(entry_prob)
            exit_edges.append(exit_prob)
            holding_times.append(t.exit_ts_ms - t.entry_ts_ms)
        
        avg_entry_edge = sum(entry_edges) / len(entry_edges) if entry_edges else 0.0
        avg_exit_edge = sum(exit_edges) / len(exit_edges) if exit_edges else 0.0
        avg_holding_ms = sum(holding_times) / len(holding_times) if holding_times else 0.0
        
        # Fix C: Use max exposure tracked over time (not just at end)
        max_exposure = self._max_exposure_over_time
        
        return ReplayReport(
            total_trades=total_trades,
            maker_fills=self.fake_kalshi.maker_fills,
            taker_fills=self.fake_kalshi.taker_fills,
            maker_entries=maker_entries,
            taker_entries=taker_entries,
            avg_entry_price=avg_entry_edge,
            avg_exit_price=avg_exit_edge,
            num_stopouts=total_trades,  # Simplified
            avg_holding_ms=avg_holding_ms,
            max_concurrent_exposure=max_exposure,
            pnl_prob_units=pnl_prob_units,
            total_orders=self.fake_kalshi.total_orders,
        )


# ---------------------------
# Replay Report
# ---------------------------

@dataclass
class ReplayReport:
    """Summary report from a replay simulation."""
    total_trades: int
    maker_fills: int
    taker_fills: int
    maker_entries: int
    taker_entries: int
    avg_entry_price: float
    avg_exit_price: float
    num_stopouts: int
    avg_holding_ms: float
    max_concurrent_exposure: int
    pnl_prob_units: float
    total_orders: int

    @staticmethod
    def empty() -> "ReplayReport":
        return ReplayReport(
            total_trades=0,
            maker_fills=0,
            taker_fills=0,
            maker_entries=0,
            taker_entries=0,
            avg_entry_price=0.0,
            avg_exit_price=0.0,
            num_stopouts=0,
            avg_holding_ms=0.0,
            max_concurrent_exposure=0,
            pnl_prob_units=0.0,
            total_orders=0,
        )

    def print_report(self) -> None:
        """Print formatted report to console."""
        print()
        print("=" * 60)
        print("REPLAY SIMULATION REPORT")
        print("=" * 60)
        print()
        print(f"Total Orders Placed:        {self.total_orders}")
        print(f"Total Trades (Roundtrips):  {self.total_trades}")
        print()
        print("FILL BREAKDOWN:")
        print(f"  Maker Fills:              {self.maker_fills}")
        print(f"  Taker Fills:              {self.taker_fills}")
        print(f"  Maker Entries:            {self.maker_entries}")
        print(f"  Taker Entries:            {self.taker_entries}")
        print()
        print("EDGE METRICS:")
        print(f"  Avg Entry Price:          {self.avg_entry_price:.4f}")
        print(f"  Avg Exit Price:           {self.avg_exit_price:.4f}")
        print()
        print("RISK METRICS:")
        print(f"  Stopouts:                 {self.num_stopouts}")
        print(f"  Avg Holding Time (ms):    {self.avg_holding_ms:.1f}")
        print(f"  Max Concurrent Exposure:  {self.max_concurrent_exposure}")
        print()
        print("PERFORMANCE:")
        print(f"  PnL (prob units):         {self.pnl_prob_units:+.4f}")
        print(f"  PnL (cents approx):       {self.pnl_prob_units * 100:+.2f}c")
        print()
        print("=" * 60)


# ---------------------------
# CLI
# ---------------------------

async def main() -> None:
    parser = argparse.ArgumentParser(description="Replay simulation for arbitrage strategy")
    parser.add_argument("--file", required=True, help="Path to JSONL snapshot file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if not Path(args.file).exists():
        log.error("File not found: %s", args.file)
        return
    
    log.info("Loading snapshots from %s", args.file)
    snapshots = ReplayEngine.load_from_jsonl(args.file)
    log.info("Loaded %d snapshots", len(snapshots))
    
    engine = ReplayEngine(snapshots)
    report = await engine.run()
    report.print_report()


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
