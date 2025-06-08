import ccxt
import pandas as pd
from itertools import product
import numpy as np

# ----- Basic Data Load -----
def fetch_ohlcv(limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def atr(df, period=14):
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
    df['ATR'] = df['tr'].rolling(period).mean()
    return df

def get_fibonacci_level(df, fib_pct):
    high = df['high'].max()
    low = df['low'].min()
    return high - (high - low) * fib_pct

def get_support_resistance(df, window=20):
    support = df['low'].rolling(window).min().iloc[-1]
    resistance = df['high'].rolling(window).max().iloc[-1]
    return support, resistance

def is_bullish_pin_bar(df, i):
    o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
    body = abs(c - o)
    lower_wick = min(c, o) - l
    upper_wick = h - max(c, o)
    return lower_wick > body and upper_wick < body * 1.5 and c > o

def is_bearish_engulfing(df, i):
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (prev['close'] > prev['open'] and
            curr['open'] > curr['close'] and
            curr['open'] > prev['close'] and
            curr['close'] < prev['open'])

def generate_signals(df, fib_pct, atr_mult):
    fib_level = get_fibonacci_level(df, fib_pct)
    support, resistance = get_support_resistance(df)
    signals = []

    for i in range(2, len(df)):
        price = df.iloc[i]['close']
        if df['ATR'].iloc[i] < df['close'].iloc[i] * atr_mult:
            continue

        if price <= fib_level + 10 and df.iloc[i]['low'] <= support + 10 and is_bullish_pin_bar(df, i):
            signals.append({'type': 'buy', 'price': price, 'index': i})
        elif price >= fib_level - 10 and df.iloc[i]['high'] >= resistance - 10 and is_bearish_engulfing(df, i):
            signals.append({'type': 'sell', 'price': price, 'index': i})
    return signals

def backtest(df, signals, sl_pct, tp_pct,fib_pct,  capital=10000, atr_mult=0.002):
    balance = capital
    trades, win = [], 0
    risk_per_trade = 0.01

    for s in signals:
        entry = s['price']
        direction = 1 if s['type'] == 'buy' else -1
        sl = entry * (1 - sl_pct) if direction == 1 else entry * (1 + sl_pct)
        tp = entry * (1 + tp_pct) if direction == 1 else entry * (1 - tp_pct)
        stake = balance * risk_per_trade

        for i in range(s['index'] + 1, len(df)):
            price = df.iloc[i]['close']
            if (direction == 1 and price <= sl) or (direction == -1 and price >= sl):
                trades.append(-stake)
                balance -= stake
                break
            elif (direction == 1 and price >= tp) or (direction == -1 and price <= tp):
                profit = stake * tp_pct
                trades.append(profit)
                balance += profit
                win += 1
                break

    if not trades:
        return None

    win_rate = round(win / len(trades), 2)
    total_return = round(sum(trades), 2)
    profit_factor = round(
        (sum(x for x in trades if x > 0) / abs(sum(x for x in trades if x < 0)))
        if any(x < 0 for x in trades) else float('inf'), 2
    )
    return {
        'Return': total_return,
        'Balance': round(balance, 2),
        'Trades': len(trades),
        'WinRate': win_rate,
        'PF': profit_factor,
        'SL': sl_pct,
        'TP': tp_pct,
        'FIB': fib_pct,
        'ATR_mult': atr_mult
    }


# ---------- Optimization ----------
if __name__ == "__main__":
    df = fetch_ohlcv()
    df = atr(df)

    fib_levels = [0.382, 0.50, 0.558, 0.618]
    sl_pcts = np.arange(0.001, 0.01, 0.001)  # 0.1% to 1%
    tp_pcts = np.arange(0.005, 0.03, 0.005)  # 0.5% to 3%
    atr_thresholds = [0.0015, 0.002, 0.003]

    combos = product(fib_levels, sl_pcts, tp_pcts, atr_thresholds)
    results = []

    for fib, sl, tp, atr_mult in combos:
        signals = generate_signals(df, fib, atr_mult)
        res = backtest(df, signals, sl, tp, fib, atr_mult)
        if res:
            results.append(res)

    results_df = pd.DataFrame(results)
    best = results_df.sort_values(by='Return', ascending=False).head(10)

    print("\n🎯 Top 10 Optimized Strategies:\n")
    print(best.to_string(index=False))
