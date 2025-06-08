
# BTC/USDT Strategy Backtesting and Optimization

This repository contains three main Python scripts for backtesting and optimizing trading strategies on the BTC/USDT pair using historical OHLCV data from Binance via the `ccxt` library. Each script explores different approaches to signal generation, backtesting, and parameter optimization.

---

## 1. [main.py](main.py)

### **Core Idea**
- Implements a basic backtesting framework for BTC/USDT using Fibonacci retracement levels and candlestick patterns (bullish pin bar and bearish engulfing).
- Includes a "sniper" strategy that combines Fibonacci, support/resistance, and candlestick patterns for more precise entries.

### **Implementation Details**
- **Data Fetching:** Uses `ccxt` to fetch 1-hour OHLCV data from Binance.
- **Fibonacci Levels:** Calculates 38.2% and 61.8% retracement levels from the highest and lowest prices in the dataset.
- **Pattern Detection:** 
  - `is_bullish_pin_bar` and `is_bearish_engulfing` functions detect specific candlestick patterns.
- **Signal Generation:** 
  - `generate_signals` uses Fibonacci levels and patterns to generate buy/sell signals.
  - `generate_sniper_signals` adds support/resistance proximity to the entry criteria.
- **Backtesting:** 
  - `backtest` and `backtest_sniper` simulate trades, applying stop-loss and take-profit logic, and track performance metrics (return, win rate, profit factor).
- **Visualization:** 
  - `plot_sniper_signals` plots the close price and marks buy/sell signals.
- **Execution:** 
  - Fetches data, generates signals, runs the backtest, prints results, and plots signals.

---

## 2. [main3.py](main3.py)

### **Core Idea**
- Provides a parameterized and vectorized approach to backtesting, allowing for grid search optimization over multiple strategy parameters (Fibonacci level, stop-loss, take-profit, ATR threshold).

### **Implementation Details**
- **Data Fetching:** Similar to `main.py`, fetches OHLCV data.
- **ATR Calculation:** Adds an ATR (Average True Range) column for volatility filtering.
- **Fibonacci & Support/Resistance:** 
  - `get_fibonacci_level` computes a dynamic Fibonacci retracement level.
  - `get_support_resistance` computes rolling support and resistance.
- **Pattern Detection:** 
  - Uses the same candlestick pattern logic as `main.py`.
- **Signal Generation:** 
  - `generate_signals` creates signals when price is near Fibonacci/support/resistance and patterns are detected, filtered by ATR.
- **Backtesting:** 
  - `backtest` simulates trades, applying risk management and tracking performance.
- **Optimization:** 
  - Uses `itertools.product` to grid search over ranges of Fibonacci levels, stop-loss, take-profit, and ATR thresholds.
  - Collects and prints the top 10 strategies by return.

---

## 3. [main4.py](main4.py)

### **Core Idea**
- Further enhances the strategy by using rolling windows for Fibonacci, support/resistance, and ATR calculations, and implements a more robust optimization and backtesting framework.

### **Implementation Details**
- **Data Fetching:** Fetches a larger dataset to support rolling calculations.
- **Indicator Calculation:** 
  - `calculate_indicators` computes rolling ATR, support, resistance, and Fibonacci high/low for each row.
- **Pattern Detection:** 
  - `is_bullish_pin_bar` and `is_bearish_engulfing` are implemented for row-wise and index-based checks.
- **Signal Generation:** 
  - `generate_signals` checks if the current price is within a tolerance (based on ATR) of the rolling Fibonacci/support/resistance levels and if the relevant pattern is present.
- **Backtesting:** 
  - `backtest` uses position sizing based on risk per trade, simulates holding trades for a maximum period, and calculates P&L, win rate, and profit factor.
- **Optimization:** 
  - Performs a grid search over a wide range of Fibonacci levels, stop-loss, take-profit, and ATR multipliers.
  - Collects and prints the top 10 strategies by return.
