"""
Compare position sizing strategies on best_config_trades.csv

1. Flat contracts (1 contract per trade)
2. Flat dollars ($8 per trade, no compounding)  
3. Kelly 8% with compounding
"""

import pandas as pd
import numpy as np
from pathlib import Path
import math


def kalshi_taker_fee(price: float, contracts: float) -> float:
    """Kalshi taker fee: ceil(0.07 * C * P * (1-P) * 100) / 100"""
    if price <= 0 or price >= 1:
        return 0.0
    raw_fee = 0.07 * contracts * price * (1 - price)
    return math.ceil(raw_fee * 100) / 100


def run_flat_contracts(df: pd.DataFrame, contracts: int = 1) -> dict:
    """Flat N contracts per trade"""
    total_pnl = 0
    total_fees = 0
    
    for _, row in df.iterrows():
        entry = row["entry_price"]
        win = row["win"]
        
        fee = kalshi_taker_fee(entry, contracts)
        if win:
            gross = (1 - entry) * contracts
        else:
            gross = -entry * contracts
        
        pnl = gross - fee
        total_pnl += pnl
        total_fees += fee
    
    return {
        "strategy": f"Flat {contracts} contract(s)",
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "avg_pnl": total_pnl / len(df),
    }


def run_flat_dollars(df: pd.DataFrame, dollars: float = 8.0) -> dict:
    """Flat $X per trade (no compounding)"""
    total_pnl = 0
    total_fees = 0
    
    for _, row in df.iterrows():
        entry = row["entry_price"]
        win = row["win"]
        
        contracts = dollars / entry
        fee = kalshi_taker_fee(entry, contracts)
        
        if win:
            gross = (1 - entry) * contracts
        else:
            gross = -entry * contracts
        
        pnl = gross - fee
        total_pnl += pnl
        total_fees += fee
    
    return {
        "strategy": f"Flat ${dollars:.0f} per trade",
        "total_pnl": total_pnl,
        "total_fees": total_fees,
        "avg_pnl": total_pnl / len(df),
    }


def run_kelly_compounding(df: pd.DataFrame, kelly_pct: float = 0.08, starting_balance: float = 100.0) -> dict:
    """Kelly % with compounding"""
    balance = starting_balance
    total_fees = 0
    balances = [balance]
    
    for _, row in df.iterrows():
        entry = row["entry_price"]
        win = row["win"]
        
        # Position size = kelly% of current balance
        position_dollars = balance * kelly_pct
        contracts = position_dollars / entry
        fee = kalshi_taker_fee(entry, contracts)
        
        if win:
            gross = (1 - entry) * contracts
        else:
            gross = -entry * contracts
        
        pnl = gross - fee
        balance += pnl
        balance = max(balance, 0.01)  # Floor
        balances.append(balance)
        total_fees += fee
    
    return {
        "strategy": f"Kelly {kelly_pct*100:.0f}% (compounding)",
        "starting_balance": starting_balance,
        "final_balance": balance,
        "total_pnl": balance - starting_balance,
        "total_fees": total_fees,
        "avg_pnl": (balance - starting_balance) / len(df),
        "roi": (balance - starting_balance) / starting_balance * 100,
        "balances": balances,
    }


def main():
    # Load trades
    trades_path = Path(__file__).parent / "best_config_trades.csv"
    df = pd.read_csv(trades_path)
    
    print("=" * 70)
    print("POSITION SIZING COMPARISON")
    print("=" * 70)
    print(f"\nTrades: {len(df)}")
    print(f"Win rate: {df['win'].mean()*100:.1f}%")
    print(f"Avg entry price: ${df['entry_price'].mean():.3f}")
    
    # Run all strategies
    results = []
    
    # Flat contracts
    for n in [1, 5, 10, 20]:
        results.append(run_flat_contracts(df, n))
    
    # Flat dollars
    for d in [5, 8, 10, 15]:
        results.append(run_flat_dollars(df, d))
    
    # Kelly compounding
    for k in [0.04, 0.08, 0.10, 0.15]:
        results.append(run_kelly_compounding(df, k, 100.0))
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    
    print(f"\n{'Strategy':<35} {'P&L':>12} {'Fees':>10} {'Avg P&L':>10}")
    print("-" * 70)
    
    for r in results:
        pnl = r["total_pnl"]
        fees = r["total_fees"]
        avg = r["avg_pnl"]
        print(f"{r['strategy']:<35} ${pnl:>+10.2f} ${fees:>8.2f} ${avg:>+9.4f}")
    
    # Detailed Kelly comparison
    print("\n" + "=" * 70)
    print("KELLY COMPOUNDING DETAIL (starting with $100)")
    print("=" * 70)
    
    for k in [0.04, 0.08, 0.10]:
        r = run_kelly_compounding(df, k, 100.0)
        print(f"\n{k*100:.0f}% Kelly:")
        print(f"  Final balance: ${r['final_balance']:.2f}")
        print(f"  Total P&L: ${r['total_pnl']:+.2f}")
        print(f"  ROI: {r['roi']:+.1f}%")
        print(f"  Total fees: ${r['total_fees']:.2f}")
    
    # Compare apples to apples
    print("\n" + "=" * 70)
    print("APPLES-TO-APPLES COMPARISON")
    print("=" * 70)
    print("\nStarting with $100, what's the best approach?")
    
    # Flat $8 (no compounding) - same avg risk as 8% Kelly at start
    flat_8 = run_flat_dollars(df, 8.0)
    kelly_8 = run_kelly_compounding(df, 0.08, 100.0)
    
    print(f"\nFlat $8/trade (no compounding):")
    print(f"  Final balance: ${100 + flat_8['total_pnl']:.2f}")
    print(f"  Total P&L: ${flat_8['total_pnl']:+.2f}")
    
    print(f"\n8% Kelly (compounding):")
    print(f"  Final balance: ${kelly_8['final_balance']:.2f}")
    print(f"  Total P&L: ${kelly_8['total_pnl']:+.2f}")
    
    print(f"\nDifference: Kelly earns ${kelly_8['total_pnl'] - flat_8['total_pnl']:+.2f} more")
    
    # Why?
    print("\n" + "=" * 70)
    print("WHY THE DIFFERENCE?")
    print("=" * 70)
    print("""
With FLAT $ sizing:
  - You risk the same $ amount every trade
  - Cheap entries = more contracts = bigger wins/losses in unit terms
  - Expensive entries = fewer contracts = smaller wins/losses
  - No adjustment for bankroll growth

With KELLY % compounding:
  - After wins, you bet MORE (bigger bankroll)
  - After losses, you bet LESS (smaller bankroll)
  - This "lets winners run" and "cuts losses"
  
The order of wins/losses matters for Kelly but not for flat!
""")

    # Show balance progression
    print("\n" + "=" * 70)
    print("BALANCE PROGRESSION (8% Kelly)")
    print("=" * 70)
    
    kelly_result = run_kelly_compounding(df, 0.08, 100.0)
    balances = kelly_result["balances"]
    
    # Show every 10th trade
    print(f"\n{'Trade':>6} {'Balance':>12} {'Change':>12}")
    print("-" * 35)
    for i in range(0, len(balances), 10):
        if i == 0:
            change = 0
        else:
            change = balances[i] - balances[i-10]
        print(f"{i:>6} ${balances[i]:>11.2f} ${change:>+11.2f}")
    
    # Final
    print(f"{len(balances)-1:>6} ${balances[-1]:>11.2f} ${balances[-1]-balances[-11]:>+11.2f}")


if __name__ == "__main__":
    main()
