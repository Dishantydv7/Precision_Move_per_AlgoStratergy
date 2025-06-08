import ccxt
import pandas as pd
import numpy as np
from itertools import product
import warnings
warnings.filterwarnings('ignore')

def fetch_ohlcv(limit=1000):
    exchange = ccxt.binance()
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', timeframe='1h', limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

def calculate_indicators(df, fib_lookback=100, sr_window=20, atr_period=14):
    # ATR Calculation
    df['prev_close'] = df['close'].shift(1)
    df['tr'] = df[['high', 'prev_close']].max(axis=1) - df[['low', 'prev_close']].min(axis=1)
    df['ATR'] = df['tr'].rolling(atr_period).mean()
    
    # Support/Resistance
    df['support'] = df['low'].rolling(sr_window).min()
    df['resistance'] = df['high'].rolling(sr_window).max()
    
    # Rolling Fibonacci levels
    df['fib_high'] = df['high'].rolling(fib_lookback).max()
    df['fib_low'] = df['low'].rolling(fib_lookback).min()
    return df.drop(columns=['prev_close', 'tr'])

# ----- Enhanced Pattern Detection -----
def is_bullish_pin_bar(row):
    o, h, l, c = row[['open', 'high', 'low', 'close']]
    body = abs(c - o)
    lower_wick = min(c, o) - l
    upper_wick = h - max(c, o)
    return (lower_wick > 1.5 * body and 
            upper_wick < 0.5 * body and 
            c > o)

def is_bearish_engulfing(df, i):
    if i < 1: 
        return False
    prev, curr = df.iloc[i-1], df.iloc[i]
    return (prev['close'] > prev['open'] and
            curr['open'] > curr['close'] and
            curr['open'] > prev['close'] and
            curr['close'] < prev['open'])

# ----- Signal Generation without Look-Ahead -----
def generate_signals(df, fib_pct, atr_mult):
    df = df.copy()
    df['fib_level'] = df['fib_high'] - (df['fib_high'] - df['fib_low']) * fib_pct
    signals = []
    
    for i in range(100, len(df)):
        # Skip low volatility periods
        if df['ATR'].iloc[i] < df['close'].iloc[i] * atr_mult:
            continue
            
        row = df.iloc[i]
        tolerance = 0.5 * row['ATR']
        price = row['close']
        
        # Buy signal conditions
        if (abs(price - row['fib_level']) <= tolerance and
            abs(price - row['support']) <= tolerance and
            is_bullish_pin_bar(row)):
            signals.append({'type': 'buy', 'price': price, 'index': i})
        
        # Sell signal conditions
        elif (abs(price - row['fib_level']) <= tolerance and
              abs(price - row['resistance']) <= tolerance and
              is_bearish_engulfing(df, i)):
            signals.append({'type': 'sell', 'price': price, 'index': i})
            
    return signals

def backtest(df, signals, sl_pct, tp_pct, capital=10000, max_hold=24):
    if not signals:
        return None
        
    balance = capital
    trades = []
    win_trades = 0
    risk_per_trade = 0.01  # Risk 1% of capital per trade
    
    for s in signals:
        entry = s['price']
        direction = 1 if s['type'] == 'buy' else -1
        
        # Calculate position size based on risk
        risk_amount = balance * risk_per_trade
        if direction == 1:
            sl_price = entry * (1 - sl_pct)
            position_size = risk_amount / (entry - sl_price)
        else:
            sl_price = entry * (1 + sl_pct)
            position_size = risk_amount / (sl_price - entry)
        
        # Check if we have enough capital
        if position_size * entry > balance:
            continue
            
        # Exit conditions
        exit_price = None
        for j in range(1, max_hold + 1):
            idx = s['index'] + j
            if idx >= len(df):
                break
                
            price = df.iloc[idx]['close']
            
            # Check stop loss
            if (direction == 1 and price <= sl_price) or (direction == -1 and price >= sl_price):
                exit_price = price
                break
                
            # Check take profit
            if direction == 1:
                if price >= entry * (1 + tp_pct):
                    exit_price = price
                    break
            else:
                if price <= entry * (1 - tp_pct):
                    exit_price = price
                    break
        else:
            # Time-based exit if no SL/TP hit
            exit_price = df.iloc[idx]['close']
        
        # Calculate P&L
        if direction == 1:
            pl = (exit_price - entry) * position_size
        else:
            pl = (entry - exit_price) * position_size
            
        balance += pl
        trades.append(pl)
        
        if pl > 0:
            win_trades += 1
    
    if not trades:
        return None
        
    # Performance metrics
    win_rate = win_trades / len(trades)
    total_return = balance - capital
    profit_factor = (sum(p for p in trades if p > 0) / 
                    abs(sum(l for l in trades if l < 0))) if any(t < 0 for t in trades) else float('inf')
    
    return {
        'Return': total_return,
        'Balance': balance,
        'Trades': len(trades),
        'WinRate': win_rate,
        'PF': profit_factor,
        'SL': sl_pct,
        'TP': tp_pct,
        'FIB': fib_pct
    }

# ----- Optimization -----
if __name__ == "__main__":
    # Load and prepare data
    df = fetch_ohlcv(1500)  # Get extra data for rolling calculations
    df = calculate_indicators(df)
    df = df.dropna().reset_index(drop=True)
    
    # Expanded parameter ranges
    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
    sl_pcts = np.arange(0.005, 0.051, 0.005)  # 0.5% to 5%
    tp_pcts = np.arange(0.01, 0.11, 0.01)     # 1% to 10%
    atr_thresholds = [0.001, 0.002, 0.003]
    
    results = []
    for fib, sl, tp, atr_mult in product(fib_levels, sl_pcts, tp_pcts, atr_thresholds):
        signals = generate_signals(df, fib, atr_mult)
        res = backtest(df, signals, sl, tp)
        if res:
            res['ATR_mult'] = atr_mult
            results.append(res)
    
    if not results:
        print("No successful strategies found")
    else:
        results_df = pd.DataFrame(results)
        best = results_df.sort_values(by='Return', ascending=False).head(10)
        print("\n🎯 Top 10 Optimized Strategies:\n")
        print(best.to_string(index=False))
