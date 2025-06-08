import ccxt
import pandas as pd
import time

binance = ccxt.binance({
    'apiKey': 'JchPY86EVWY8B4zLZh8Il1WZtL3n7RMZ3sI78Mw6s11MhDgEatFdjImvSeoMJjtu',
    'secret': 'OhGpuHAmhvxKkXyVPRjdcahzNFIlaDQCuJ2V6ZinJewkLazBL6JynVaLYV9pe3VM',
    'enableRateLimit': True
})

def fetch_ohlcv(symbol="BTC/USDT", timeframe='1h', limit=500):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df


def draw_fibonacci(df):
    high = df['high'].max()
    low = df['low'].min()
    levels = {
        'fib_38.2': high - (high - low) * 0.382,
        'fib_61.8': high - (high - low) * 0.618
    }
    return levels

def is_bullish_pin_bar(df, i):
    o, h, l, c = df.iloc[i][['open', 'high', 'low', 'close']]
    body = abs(o - c)
    lower_wick = min(o, c) - l
    upper_wick = h - max(o, c)
    return lower_wick > body and upper_wick < body * 1.5 and c > o


def atr(df, period=14):
    tr = df[['high', 'low', 'close']].copy()
    tr['prev_close'] = tr['close'].shift(1)
    tr['tr'] = tr[['high', 'prev_close']].max(axis=1) - tr[['low', 'prev_close']].min(axis=1)
    return tr['tr'].rolling(period).mean()


def is_bearish_engulfing(df, i):
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    return (
        prev['close'] > prev['open'] and
        curr['open'] > curr['close'] and
        curr['open'] > prev['close'] and
        curr['close'] < prev['open']
    )


def generate_signals(df, fib_levels):
    signals = []
    for i in range(2, len(df)):
        price = df.iloc[i]['close']
        if price <= fib_levels['fib_61.8'] and is_bullish_pin_bar(df, i):
            signals.append({'type': 'buy', 'price': price, 'index': i})
        elif price >= fib_levels['fib_38.2'] and is_bearish_engulfing(df, i):
            signals.append({'type': 'sell', 'price': price, 'index': i})
    return signals
def backtest(df, signals, sl_pct=0.003, tp_pct=0.01, capital=10000, risk_per_trade=0.01, fee_pct=0.001):
    balance = capital
    trades = []
    win = 0

    for s in signals:
        entry = s['price']
        sl = entry * (1 - sl_pct) if s['type'] == 'buy' else entry * (1 + sl_pct)
        tp = entry * (1 + tp_pct) if s['type'] == 'buy' else entry * (1 - tp_pct)
        stake = balance * risk_per_trade
        direction = 1 if s['type'] == 'buy' else -1

        for j in range(s['index'] + 1, len(df)):
            price = df.iloc[j]['close']
            if (direction == 1 and price <= sl) or (direction == -1 and price >= sl):
                balance -= stake
                trades.append(-stake)
                break
            elif (direction == 1 and price >= tp) or (direction == -1 and price <= tp):
                profit = stake * tp_pct - stake * fee_pct
                balance += profit
                trades.append(profit)
                win += 1
                break

    total_return = sum(trades)
    return {
        'Total Return': total_return,
        'Ending Balance': balance,
        'Trades': len(trades),
        'Wins': win,
        'Win Rate': win / len(trades) if trades else 0,
        'Profit Factor': (sum(x for x in trades if x > 0) /
                          abs(sum(x for x in trades if x < 0))) if any(x < 0 for x in trades) else float('inf')
    }

def backtest_sniper(df, signals, sl_pct=0.002, tp_pct=0.006, capital=10000):
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

    return {
        'Total Return': round(sum(trades), 2),
        'Ending Balance': round(balance, 2),
        'Trades': len(trades),
        'Wins': win,
        'Win Rate': round(win / len(trades), 2) if trades else 0,
        'Profit Factor': round(
            (sum(x for x in trades if x > 0) / abs(sum(x for x in trades if x < 0)))
            if any(x < 0 for x in trades) else float('inf'), 2
        )
    }


def plot_sniper_signals(df, signals):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(df['timestamp'], df['close'], label='Close Price')
    for s in signals:
        plt.scatter(df.iloc[s['index']]['timestamp'], s['price'],
                    color='green' if s['type'] == 'buy' else 'red', label=s['type'])
    plt.legend()
    plt.grid(True)
    plt.show()


def generate_sniper_signals(df):
    levels = get_fibonacci_levels(df)
    support, resistance = get_support_resistance(df)
    signals = []

    for i in range(2, len(df)):
        price = df.iloc[i]['close']
        
        if (price <= levels['61.8'] + 10 and 
            df.iloc[i]['low'] <= support + 10 and 
            is_bullish_pin_bar(df, i)):
            signals.append({'type': 'buy', 'price': price, 'index': i})

        elif (price >= levels['38.2'] - 10 and 
              df.iloc[i]['high'] >= resistance - 10 and 
              is_bearish_engulfing(df, i)):
            signals.append({'type': 'sell', 'price': price, 'index': i})

    return signals


def get_fibonacci_levels(df):
    swing_high = df['high'].max()
    swing_low = df['low'].min()
    return {
        '38.2': swing_high - (swing_high - swing_low) * 0.382,
        '61.8': swing_high - (swing_high - swing_low) * 0.618
    }

def get_support_resistance(df, window=20):
    support = df['low'].rolling(window).min().iloc[-1]
    resistance = df['high'].rolling(window).max().iloc[-1]
    return support, resistance



df = fetch_ohlcv(limit=1000)
signals = generate_sniper_signals(df)
results = backtest_sniper(df, signals)
print("Backtest Results:")
for k, v in results.items():
    print(f"{k}: {v}")
plot_sniper_signals(df, signals)

