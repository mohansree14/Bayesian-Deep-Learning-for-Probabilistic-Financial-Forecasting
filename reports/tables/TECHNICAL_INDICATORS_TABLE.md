# Technical Indicators Reference Table

## Complete List of 21 Technical Indicators

| # | Indicator Name | Category | Formula | Parameters | Purpose | Interpretation |
|---|---|---|---|---|---|---|
| **1** | **Open** | Price | Raw OHLC data | - | Opening price of trading period | Market opening sentiment |
| **2** | **High** | Price | Raw OHLC data | - | Highest price during period | Intraday peak, resistance level |
| **3** | **Low** | Price | Raw OHLC data | - | Lowest price during period | Intraday trough, support level |
| **4** | **Close** | Price | Raw OHLC data | - | Closing price of trading period | Market closing sentiment, most important |
| **5** | **Adj Close** | Price | Close × Adjustment Factor | - | Price adjusted for splits/dividends | Accurate historical price comparison |
| **6** | **Volume** | Volume | Raw trading volume | - | Number of shares traded | Market activity, liquidity indicator |
| **7** | **Return (1-day)** | Returns | ret = (Close_t - Close_{t-1}) / Close_{t-1} | window=1 | Daily percentage change | Short-term price momentum |
| **8** | **Log Return (1-day)** | Returns | log_ret = ln(Close_t / Close_{t-1}) | window=1 | Logarithmic daily return | Statistical properties, compounding |
| **9** | **SMA (10-day)** | Trend | SMA = (1/n) × Σ(Close_i) for i=1 to n | window=10 | Simple Moving Average (10 days) | Short-term trend direction |
| **10** | **SMA (20-day)** | Trend | SMA = (1/n) × Σ(Close_i) for i=1 to n | window=20 | Simple Moving Average (20 days) | Medium-term trend direction |
| **11** | **EMA (12-day)** | Trend | EMA_t = α × Close_t + (1-α) × EMA_{t-1}<br>where α = 2/(n+1) | window=12 | Exponential Moving Average (12 days) | Fast-reacting trend indicator |
| **12** | **EMA (26-day)** | Trend | EMA_t = α × Close_t + (1-α) × EMA_{t-1}<br>where α = 2/(n+1) | window=26 | Exponential Moving Average (26 days) | Slow-reacting trend indicator |
| **13** | **RSI (14-day)** | Momentum | RSI = 100 - (100 / (1 + RS))<br>RS = Avg_Gain / Avg_Loss | window=14 | Relative Strength Index | Overbought (>70) / Oversold (<30) |
| **14** | **MACD Line** | Momentum | MACD = EMA(12) - EMA(26) | fast=12, slow=26 | Moving Average Convergence Divergence | Trend strength and direction |
| **15** | **MACD Signal** | Momentum | Signal = EMA(MACD, 9) | signal=9 | MACD signal line (9-day EMA of MACD) | Buy/sell signal crossovers |
| **16** | **MACD Histogram** | Momentum | Histogram = MACD - Signal | - | Difference between MACD and signal | Momentum strength, divergence |
| **17** | **BB Middle** | Volatility | BB_mid = SMA(Close, 20) | window=20 | Bollinger Band middle line | Mean reversion reference |
| **18** | **BB Upper** | Volatility | BB_upper = BB_mid + (2 × σ) | window=20, std=2 | Bollinger Band upper boundary | Overbought threshold, volatility expansion |
| **19** | **BB Lower** | Volatility | BB_lower = BB_mid - (2 × σ) | window=20, std=2 | Bollinger Band lower boundary | Oversold threshold, volatility expansion |
| **20** | **Stochastic %K** | Momentum | %K = 100 × (Close - Low_n) / (High_n - Low_n) | window=14 | Stochastic oscillator %K line | Current price vs. recent range |
| **21** | **Stochastic %D** | Momentum | %D = SMA(%K, 3) | window=3 | Stochastic oscillator %D line (signal) | Smoothed %K, buy/sell signals |

---

## Detailed Formulas and Calculations

### 1-6. Price and Volume Features (Raw Data)

#### Basic OHLCV
```python
Open:       Raw opening price
High:       Raw highest price  
Low:        Raw lowest price
Close:      Raw closing price
Adj_Close:  Close × (Adjustment for splits/dividends)
Volume:     Total shares traded
```

**Purpose:** Fundamental price action data; foundation for all technical indicators.

---

### 7-8. Return Features

#### Daily Return
```python
ret_1d = (Close_t - Close_{t-1}) / Close_{t-1}
       = (Close_t / Close_{t-1}) - 1
```
**Range:** (-∞, +∞), typically [-10%, +10%] for stocks  
**Purpose:** Measure daily percentage change; used in risk/return calculations

#### Log Return
```python
log_ret_1d = ln(Close_t / Close_{t-1})
           = ln(Close_t) - ln(Close_{t-1})
```
**Range:** (-∞, +∞), approximately equal to ret_1d for small changes  
**Purpose:** 
- Time-additive (can sum across periods)
- Symmetric (±10% log returns are equal magnitude)
- Better statistical properties (more normal distribution)

---

### 9-12. Moving Averages (Trend Indicators)

#### Simple Moving Average (SMA)
```python
SMA_n = (1/n) × Σ(Close_i) for i=t-n+1 to t
      = (Close_{t-n+1} + Close_{t-n+2} + ... + Close_t) / n
```

**SMA(10):** Short-term trend (2 weeks)  
**SMA(20):** Medium-term trend (1 month)

**Interpretation:**
- Price > SMA → Uptrend
- Price < SMA → Downtrend
- SMA(10) crosses above SMA(20) → Golden Cross (bullish)
- SMA(10) crosses below SMA(20) → Death Cross (bearish)

#### Exponential Moving Average (EMA)
```python
EMA_t = α × Close_t + (1 - α) × EMA_{t-1}
where α = 2 / (n + 1)  # Smoothing factor

For n=12: α = 2/13 = 0.1538
For n=26: α = 2/27 = 0.0741
```

**EMA(12):** Fast-reacting, more weight on recent prices  
**EMA(26):** Slow-reacting, more weight on historical prices

**Advantages over SMA:**
- More responsive to recent price changes
- Weights decrease exponentially (not equally)
- Used in MACD calculation

---

### 13. Relative Strength Index (RSI)

#### Full Calculation
```python
# Step 1: Calculate price changes
Gain_t = max(Close_t - Close_{t-1}, 0)
Loss_t = max(Close_{t-1} - Close_t, 0)

# Step 2: Calculate average gains and losses (14-day)
Avg_Gain = EMA(Gain, 14)
Avg_Loss = EMA(Loss, 14)

# Step 3: Calculate Relative Strength
RS = Avg_Gain / Avg_Loss

# Step 4: Calculate RSI
RSI = 100 - (100 / (1 + RS))
    = 100 × (Avg_Gain / (Avg_Gain + Avg_Loss))
```

**Range:** 0 to 100  
**Interpretation:**
- RSI > 70 → Overbought (potential sell signal)
- RSI < 30 → Oversold (potential buy signal)
- RSI = 50 → Neutral (equal buying/selling pressure)

**Divergences:**
- Bullish: Price makes lower low, RSI makes higher low
- Bearish: Price makes higher high, RSI makes lower high

---

### 14-16. MACD (Moving Average Convergence Divergence)

#### MACD Line
```python
MACD = EMA(Close, 12) - EMA(Close, 26)
```
**Purpose:** Shows difference between fast and slow trend  
**Positive MACD:** Uptrend (fast EMA > slow EMA)  
**Negative MACD:** Downtrend (fast EMA < slow EMA)

#### MACD Signal Line
```python
Signal = EMA(MACD, 9)
```
**Purpose:** Smoothed version of MACD for crossover signals

#### MACD Histogram
```python
Histogram = MACD - Signal
```
**Purpose:** Visual representation of MACD-Signal difference  
**Interpretation:**
- Histogram > 0 → Bullish momentum
- Histogram < 0 → Bearish momentum
- Histogram increasing → Momentum strengthening
- Histogram decreasing → Momentum weakening

**Trading Signals:**
- MACD crosses above Signal → Buy signal
- MACD crosses below Signal → Sell signal
- Histogram crosses zero line → Trend change

---

### 17-19. Bollinger Bands

#### BB Middle (SMA)
```python
BB_mid = SMA(Close, 20)
```

#### BB Upper Band
```python
BB_upper = BB_mid + (k × σ)
where σ = StdDev(Close, 20)
      k = 2 (standard multiplier)
```

#### BB Lower Band
```python
BB_lower = BB_mid - (k × σ)
```

**Full Calculation:**
```python
# 1. Calculate 20-day SMA
BB_mid = (1/20) × Σ(Close_i)

# 2. Calculate 20-day standard deviation
σ = sqrt((1/20) × Σ(Close_i - BB_mid)²)

# 3. Calculate bands
BB_upper = BB_mid + 2σ
BB_lower = BB_mid - 2σ

# 4. Calculate bandwidth
BB_width = (BB_upper - BB_lower) / BB_mid
```

**Interpretation:**
- **Price touches upper band:** Overbought, potential reversal
- **Price touches lower band:** Oversold, potential reversal
- **Bands narrow (squeeze):** Low volatility, breakout likely
- **Bands widen (expansion):** High volatility, trend continuation
- **Price consistently above mid:** Uptrend
- **Price consistently below mid:** Downtrend

**Statistical Meaning:**
- ~68% of prices fall within ±1σ
- ~95% of prices fall within ±2σ
- ~99.7% of prices fall within ±3σ

---

### 20-21. Stochastic Oscillator

#### Stochastic %K
```python
%K = 100 × (Close - Low_n) / (High_n - Low_n)

where:
Low_n = Lowest low over past n periods (typically 14)
High_n = Highest high over past n periods (typically 14)
```

**Range:** 0 to 100  
**Interpretation:**
- Measures where current close is relative to recent range
- %K = 100 → Close at highest high
- %K = 0 → Close at lowest low
- %K = 50 → Close at midpoint

#### Stochastic %D (Signal Line)
```python
%D = SMA(%K, 3)
   = (%K_{t-2} + %K_{t-1} + %K_t) / 3
```

**Purpose:** Smoothed version of %K to reduce noise

**Trading Signals:**
- %K > 80 → Overbought
- %K < 20 → Oversold
- %K crosses above %D → Buy signal
- %K crosses below %D → Sell signal

**Divergences:**
- Bullish: Price makes lower low, %K makes higher low
- Bearish: Price makes higher high, %K makes lower high

---

## Indicator Categories

### By Purpose

| Category | Count | Indicators | Primary Use |
|---|---|---|---|
| **Price** | 5 | Open, High, Low, Close, Adj Close | Raw price data |
| **Volume** | 1 | Volume | Market activity |
| **Returns** | 2 | ret_1d, log_ret_1d | Price changes |
| **Trend** | 4 | SMA(10), SMA(20), EMA(12), EMA(26) | Trend direction |
| **Momentum** | 6 | RSI, MACD, MACD Signal, MACD Hist, %K, %D | Price momentum |
| **Volatility** | 3 | BB Upper, BB Mid, BB Lower | Price volatility |

### By Time Sensitivity

| Speed | Indicators | Responsiveness |
|---|---|---|
| **Instant** | OHLC, Volume, Returns | Real-time |
| **Fast** | EMA(12), %K | 1-2 weeks |
| **Medium** | SMA(10), RSI(14), Stochastic | 2-3 weeks |
| **Slow** | SMA(20), EMA(26), Bollinger(20) | 3-4 weeks |

---

## Correlation Analysis

### Expected High Correlations (|r| > 0.9)

| Indicator Pair | Correlation | Reason |
|---|---|---|
| Open ↔ Close | ~1.00 | Same day prices |
| Close ↔ Adj Close | 1.00 | Adjustment factor only |
| SMA(20) ↔ EMA(12) | ~0.99 | Similar timeframes |
| BB_mid ↔ SMA(20) | 1.00 | BB_mid IS SMA(20) |
| BB_upper ↔ BB_lower | ~0.99 | Parallel bands |
| %K ↔ %D | ~0.92 | %D is smoothed %K |
| ret_1d ↔ log_ret_1d | ~1.00 | Approximation for small changes |

### Expected Low Correlations (|r| < 0.3)

| Indicator Pair | Correlation | Reason |
|---|---|---|
| Volume ↔ Close | ~0.1-0.3 | Independent factors |
| RSI ↔ Volume | ~0.1 | Different dimensions |
| MACD ↔ Volume | ~0.2 | Price vs. activity |

---

## Implementation Code Reference

### Python Calculation Example

```python
import pandas as pd
import numpy as np

def add_technical_indicators(df):
    """Add all 21 technical indicators to dataframe."""
    
    # 1-6: Raw OHLCV (already present)
    
    # 7-8: Returns
    df['ret_1d'] = df['close'].pct_change()
    df['log_ret_1d'] = np.log(df['close'] / df['close'].shift(1))
    
    # 9-10: Simple Moving Averages
    df['sma_10'] = df['close'].rolling(window=10).mean()
    df['sma_20'] = df['close'].rolling(window=20).mean()
    
    # 11-12: Exponential Moving Averages
    df['ema_12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema_26'] = df['close'].ewm(span=26, adjust=False).mean()
    
    # 13: RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 14-16: MACD
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_signal']
    
    # 17-19: Bollinger Bands
    df['bb_mid'] = df['close'].rolling(window=20).mean()
    bb_std = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['bb_mid'] + (2 * bb_std)
    df['bb_lower'] = df['bb_mid'] - (2 * bb_std)
    
    # 20-21: Stochastic Oscillator
    low_14 = df['low'].rolling(window=14).min()
    high_14 = df['high'].rolling(window=14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
    df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()
    
    return df
```

---

## Trading Strategy Combinations

### Common Indicator Combinations

| Strategy | Indicators Used | Logic |
|---|---|---|
| **Trend Following** | SMA(10), SMA(20), MACD | Buy when SMA(10) > SMA(20) and MACD > 0 |
| **Mean Reversion** | RSI, Bollinger Bands | Buy when RSI < 30 and Close < BB_lower |
| **Momentum** | RSI, MACD, Stochastic | Buy when all three show bullish signals |
| **Breakout** | Bollinger Bands, Volume | Buy when Close breaks BB_upper with high volume |
| **Multi-Timeframe** | SMA(10), SMA(20), EMA(26) | Align short, medium, long-term trends |

---

## Key Insights for ML Models

### Feature Engineering Considerations

1. **Multicollinearity:**
   - Many indicators highly correlated (SMA/EMA variations)
   - Consider regularization (L1/L2) or feature selection
   - Principal Component Analysis (PCA) may help

2. **Scaling Requirements:**
   - Price features: Large values ($100-300)
   - RSI/Stochastic: Already bounded [0, 100]
   - Returns: Small values (-0.1 to 0.1)
   - Recommend: StandardScaler or MinMaxScaler

3. **Temporal Dependencies:**
   - All indicators use historical windows
   - Already capture past information
   - LSTM benefits from sequential nature

4. **Missing Values:**
   - First n days will have NaN (e.g., SMA(20) needs 20 days)
   - Use forward-fill or drop initial rows
   - Affects train/valid/test split

---

## References

1. **Bollinger Bands:** Bollinger, J. (2001). "Bollinger on Bollinger Bands"
2. **MACD:** Appel, G. (2005). "Technical Analysis: Power Tools for Active Investors"
3. **RSI:** Wilder, J. W. (1978). "New Concepts in Technical Trading Systems"
4. **Stochastic:** Lane, G. (1950s). Stochastic Oscillator
5. **Moving Averages:** Murphy, J. J. (1999). "Technical Analysis of Financial Markets"

---

**Document Version:** 1.0  
**Last Updated:** October 14, 2025  
**Source Code:** `src/data/indicators.py`  
**Feature Engineering:** `src/data/preprocess.py`
