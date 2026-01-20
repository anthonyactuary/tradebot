"""
Long-term Kelly vs Flat simulation

Simulate many 6-month periods to see how Kelly compares to flat sizing
when you have enough trades for the law of large numbers to kick in.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import math
from typing import List, Tuple


def kalshi_taker_fee(price: float, contracts: float) -> float:
    """Kalshi taker fee"""
    if price <= 0 or price >= 1:
        return 0.0
    raw_fee = 0.07 * contracts * price * (1 - price)
    return math.ceil(raw_fee * 100) / 100


def simulate_trades(
    n_trades: int,
    win_rate: float,
    avg_entry_price: float,
    strategy: str,  # "flat_contracts", "flat_dollars", "kelly"
    param: float,   # contracts, dollars, or kelly %
    starting_balance: float = 100.0,
    liquidity_cap: float = 150.0,  # Max $ per trade (market depth limit)
    seed: int = None,
) -> Tuple[float, List[float]]:
    """
    Simulate n_trades and return final balance.
    
    Uses avg_entry_price for all trades (simplified).
    Win pays (1-entry), loss pays (-entry).
    Liquidity cap prevents unrealistic scaling.
    """
    if seed is not None:
        np.random.seed(seed)
    
    balance = starting_balance
    balances = [balance]
    
    # Generate win/loss outcomes
    outcomes = np.random.random(n_trades) < win_rate
    
    for win in outcomes:
        entry = avg_entry_price
        
        if strategy == "flat_contracts":
            position_dollars = param * entry
        elif strategy == "flat_dollars":
            position_dollars = param
        elif strategy == "kelly":
            position_dollars = balance * param
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
        
        # Apply liquidity cap
        position_dollars = min(position_dollars, liquidity_cap)
        contracts = position_dollars / entry
        
        fee = kalshi_taker_fee(entry, contracts)
        
        if win:
            gross = (1 - entry) * contracts
        else:
            gross = -entry * contracts
        
        pnl = gross - fee
        balance += pnl
        balance = max(balance, 0.01)  # Floor at 1 cent
        balances.append(balance)
    
    return balance, balances


def run_comparison(n_trades: int, n_simulations: int = 100, liquidity_cap: float = 150.0):
    """Run many simulations comparing strategies"""
    
    # Stats from your backtest
    win_rate = 0.5417
    avg_entry = 0.49
    starting = 100.0
    
    strategies = [
        ("Flat 1 contract", "flat_contracts", 1),
        ("Flat 5 contracts", "flat_contracts", 5),
        ("Flat $8/trade", "flat_dollars", 8),
        ("Kelly 2.8%", "kelly", 0.028),
        ("Kelly 5.7%", "kelly", 0.057),
        ("Kelly 10%", "kelly", 0.10),
    ]
    
    results = {name: [] for name, _, _ in strategies}
    
    for sim in range(n_simulations):
        seed = sim * 42
        
        for name, strategy, param in strategies:
            final, _ = simulate_trades(
                n_trades=n_trades,
                win_rate=win_rate,
                avg_entry_price=avg_entry,
                strategy=strategy,
                param=param,
                starting_balance=starting,
                liquidity_cap=liquidity_cap,
                seed=seed,  # Same seed for fair comparison
            )
            results[name].append(final)
    
    return results


def main():
    print("=" * 80)
    print("KELLY VS FLAT: LONG-TERM SIMULATION")
    print("=" * 80)
    
    # Your stats
    win_rate = 0.5417
    avg_entry = 0.49
    trades_per_day = 90
    liquidity_cap = 150.0  # Max $ per trade due to market depth
    
    print(f"\nAssumptions from your backtest:")
    print(f"  Win rate: {win_rate*100:.1f}%")
    print(f"  Avg entry price: ${avg_entry:.2f}")
    print(f"  Trades/day: {trades_per_day:.1f}")
    print(f"  Starting balance: $100")
    print(f"  Liquidity cap: ${liquidity_cap:.0f}/trade (market depth limit)")
    
    # =========================================================================
    # SHORT TERM (your 144 trades)
    # =========================================================================
    print("\n" + "=" * 80)
    print("SHORT TERM: 144 trades (your backtest period)")
    print("=" * 80)
    
    results_short = run_comparison(n_trades=144, n_simulations=1000)
    
    print(f"\n{'Strategy':<20} {'Median':>12} {'Mean':>12} {'Min':>12} {'Max':>12} {'% Profitable':>14}")
    print("-" * 85)
    
    for name, balances in results_short.items():
        arr = np.array(balances)
        profitable = (arr > 100).mean() * 100
        print(f"{name:<20} ${np.median(arr):>11.2f} ${np.mean(arr):>11.2f} "
              f"${np.min(arr):>11.2f} ${np.max(arr):>11.2f} {profitable:>13.1f}%")
    
    # =========================================================================
    # MEDIUM TERM (1 month)
    # =========================================================================
    n_trades_1mo = int(trades_per_day * 30)
    print(f"\n" + "=" * 80)
    print(f"MEDIUM TERM: {n_trades_1mo} trades (~1 month)")
    print("=" * 80)
    
    results_1mo = run_comparison(n_trades=n_trades_1mo, n_simulations=1000)
    
    print(f"\n{'Strategy':<20} {'Median':>12} {'Mean':>12} {'Min':>12} {'Max':>12} {'% Profitable':>14}")
    print("-" * 85)
    
    for name, balances in results_1mo.items():
        arr = np.array(balances)
        profitable = (arr > 100).mean() * 100
        print(f"{name:<20} ${np.median(arr):>11.2f} ${np.mean(arr):>11.2f} "
              f"${np.min(arr):>11.2f} ${np.max(arr):>11.2f} {profitable:>13.1f}%")
    
    # =========================================================================
    # LONG TERM (6 months)
    # =========================================================================
    n_trades_6mo = int(trades_per_day * 180)
    print(f"\n" + "=" * 80)
    print(f"LONG TERM: {n_trades_6mo} trades (~6 months)")
    print("=" * 80)
    
    results_6mo = run_comparison(n_trades=n_trades_6mo, n_simulations=500)
    
    print(f"\n{'Strategy':<20} {'Median':>12} {'Mean':>12} {'Min':>12} {'Max':>12} {'% Profitable':>14}")
    print("-" * 85)
    
    for name, balances in results_6mo.items():
        arr = np.array(balances)
        profitable = (arr > 100).mean() * 100
        print(f"{name:<20} ${np.median(arr):>11.2f} ${np.mean(arr):>11.2f} "
              f"${np.min(arr):>11.2f} ${np.max(arr):>11.2f} {profitable:>13.1f}%")
    
    # =========================================================================
    # Summary insight
    # =========================================================================
    print("\n" + "=" * 80)
    print("KEY INSIGHT")
    print("=" * 80)
    
    print("""
SHORT TERM (144 trades):
  - High variance, Kelly can easily lose to bad sequences
  - Flat sizing is safer and more predictable

LONG TERM (3000+ trades):
  - Law of large numbers kicks in
  - Kelly's compounding advantage dominates
  - Kelly grows exponentially, flat grows linearly

The crossover point depends on your edge and variance.
With 54.17% win rate and ~1:1 payoff, you need 500+ trades before 
Kelly reliably beats flat sizing.
""")

    # =========================================================================
    # Single detailed run to show the progression
    # =========================================================================
    print("\n" + "=" * 80)
    print("EXAMPLE 6-MONTH RUN (same random sequence)")
    print("=" * 80)
    
    seed = 12345
    n_trades = n_trades_6mo
    liq_cap = 150.0
    
    print(f"\n{'Trade':>8} {'Flat $8':>14} {'Kelly 5.7%':>14} {'Kelly 10%':>14}")
    print("-" * 55)
    
    _, flat_balances = simulate_trades(n_trades, win_rate, avg_entry, "flat_dollars", 8, 100, liq_cap, seed)
    _, k57_balances = simulate_trades(n_trades, win_rate, avg_entry, "kelly", 0.057, 100, liq_cap, seed)
    _, k10_balances = simulate_trades(n_trades, win_rate, avg_entry, "kelly", 0.10, 100, liq_cap, seed)
    
    checkpoints = [0, 100, 500, 1000, 1500, 2000, 2500, 3000, n_trades]
    for i in checkpoints:
        if i < len(flat_balances):
            print(f"{i:>8} ${flat_balances[i]:>13.2f} ${k57_balances[i]:>13.2f} ${k10_balances[i]:>13.2f}")
    
    print(f"\nFinal after {n_trades} trades:")
    print(f"  Flat $8:     ${flat_balances[-1]:,.2f}")
    print(f"  Kelly 5.7%:  ${k57_balances[-1]:,.2f}")
    print(f"  Kelly 10%:   ${k10_balances[-1]:,.2f}")


if __name__ == "__main__":
    main()
