import csv
from collections import defaultdict

trades = list(csv.DictReader(open('runs/paper_trades_20260109_151007.csv')))
equity = list(csv.DictReader(open('runs/paper_equity_20260109_151007.csv')))

print('=' * 60)
print('2-HOUR PAPER TRADING RUN ANALYSIS')
print('=' * 60)

print(f'\nTotal trades: {len(trades)}')
buys = [t for t in trades if t['action']=='buy']
sells = [t for t in trades if t['action']=='sell']
print(f'Buys: {len(buys)}, Sells: {len(sells)}')

auto_flatten = [t for t in trades if 'auto-flatten' in t.get('note','')]
print(f'\n🚨 AUTO-FLATTEN EVENTS: {len(auto_flatten)}')
for t in auto_flatten:
    print(f"   {t['ticker']}: bought {t['count']} contracts @ {t['price_cents']}c, fee=${t['fee_dollars']}")

# Total fees
total_fees = sum(float(t['fee_dollars']) for t in trades)
print(f'\n💸 Total fees paid: ${total_fees:.2f}')

# Volume analysis
buy_value = sum(int(t['price_cents'])*int(t['count'])/100 for t in buys)
sell_value = sum(int(t['price_cents'])*int(t['count'])/100 for t in sells)
print(f'\n📊 Trade Volume:')
print(f'   Buy volume: ${buy_value:.2f}')
print(f'   Sell volume: ${sell_value:.2f}')

# Final results
last = trades[-1]
final_equity = float(last['equity_after'])
final_pnl = final_equity - 100
print(f'\n📉 FINAL RESULTS:')
print(f'   Starting: $100.00')
print(f'   Ending: ${final_equity:.2f}')
print(f'   PnL: ${final_pnl:.2f}')
print(f'   Return: {final_pnl:.1f}%')

# Equity curve analysis
equities = [float(e['equity']) for e in equity]
pnls = [float(e['pnl']) for e in equity]
min_pnl = min(pnls)
max_pnl = max(pnls)
print(f'\n📈 Equity Curve:')
print(f'   Max PnL: ${max_pnl:.2f}')
print(f'   Min PnL (Max Drawdown): ${min_pnl:.2f}')

# Per-market analysis
market_pnl = defaultdict(lambda: {'buys': 0, 'sells': 0, 'buy_val': 0, 'sell_val': 0})
for t in trades:
    ticker = t['ticker']
    if t['action'] == 'buy':
        market_pnl[ticker]['buys'] += int(t['count'])
        market_pnl[ticker]['buy_val'] += int(t['price_cents']) * int(t['count']) / 100
    else:
        market_pnl[ticker]['sells'] += int(t['count'])
        market_pnl[ticker]['sell_val'] += int(t['price_cents']) * int(t['count']) / 100

print(f'\n📋 Per-Market Breakdown:')
for ticker, data in sorted(market_pnl.items()):
    net_pos = data['buys'] - data['sells']
    net_flow = data['sell_val'] - data['buy_val']
    print(f"   {ticker}:")
    print(f"      Buys: {data['buys']} (${data['buy_val']:.2f})")
    print(f"      Sells: {data['sells']} (${data['sell_val']:.2f})")
    print(f"      Net position: {net_pos}, Cash flow: ${net_flow:.2f}")

# Problem diagnosis
print('\n' + '=' * 60)
print('🔍 PROBLEM DIAGNOSIS')
print('=' * 60)

# Check for one-sided fills
total_buys = sum(int(t['count']) for t in buys)
total_sells = sum(int(t['count']) for t in sells)
print(f'\nTotal contracts bought: {total_buys}')
print(f'Total contracts sold: {total_sells}')
print(f'Imbalance: {total_buys - total_sells} contracts')

# Check auto-flatten impact
flatten_cost = 0
for t in auto_flatten:
    flatten_cost += int(t['price_cents']) * int(t['count']) / 100
print(f'\nAuto-flatten total cost: ${flatten_cost:.2f}')

# Identify the pattern
print('\n⚠️  KEY ISSUES IDENTIFIED:')
print('   1. One-sided fills: We kept selling at high prices, accumulating')
print('      short positions that then got bought back at even higher prices')
print('      during auto-flatten.')
print('   2. Auto-flatten is a TAKER order (7% fee vs 1.75% maker)')
print('   3. Markets trending strongly against our positions')
print('   4. Adverse selection: our orders got filled when price moved against us')
