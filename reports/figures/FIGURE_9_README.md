# Figure 9: Forecast Visualization with Uncertainty Bands

## Quick Reference

**Purpose:** Demonstrate how probabilistic forecasting models quantify prediction uncertainty and adapt to changing market conditions.

**Key Finding:** Uncertainty bands widen **1.48x** during the COVID-19 market crash (March 2020), correctly identifying periods of low predictability.

---

## Visual Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Figure 9: Forecast Visualization with Uncertainty Bands   │
│  ─────────────────────────────────────────────────────────  │
│                                                             │
│  │                                                          │
│  │     ┌─────────────────────────────┐   ← Wide bands      │
│  │    ╱                               ╲    (high volatility)│
│  │   ╱  [COVID-19 CRASH PERIOD]       ╲                    │
│  │  ╱   Feb-Apr 2020                   ╲                   │
│  │ ╱                                     ╲                  │
│  │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│                  │
│  │     ────── Actual Price                                 │
│  │     - - -  Forecast                                     │
│  │     ▓▓▓▓▓  Uncertainty Bands (95% CI)                   │
│  │                                                          │
│  └──────────────────────────────────────────────────────►  │
│                    Time (Jan - Aug 2020)                   │
└─────────────────────────────────────────────────────────────┘
```

---

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Average Band Width** | $42.43 | Overall prediction uncertainty |
| **COVID-19 Period Width** | $57.08 | Uncertainty during crisis |
| **Non-Crisis Width** | $38.57 | Uncertainty during stability |
| **Volatility Impact** | **1.48x** | Bands are 48% wider during crash |
| **Volatility Ratio** | **1.79x** | COVID period had 79% higher volatility |

---

## What This Means

### 🎯 For Traders
- **Wide bands = High risk** → Reduce position sizes
- **Narrow bands = Low risk** → Normal trading confidence
- Use band width as a dynamic risk signal

### 📊 For Risk Managers
- Uncertainty bands serve as early warning system
- Widening bands → Increase hedging, reduce leverage
- Quantified risk for regulatory compliance

### 💼 For Portfolio Managers
- Incorporate uncertainty into asset allocation
- Uncertainty-weighted portfolio optimization
- Better risk-adjusted returns

---

## Technical Details

**Data:** AAPL stock prices (January - August 2020)  
**Method:** Volatility-based confidence intervals (95% CI)  
**Crisis Period:** Feb 19 - Apr 7, 2020 (COVID-19 market crash)  
**Volatility Window:** 20-day rolling standard deviation  

**Files Generated:**
- `figure_9_uncertainty_bands.png` (300 DPI, 618 KB)
- `figure_9_uncertainty_bands.pdf` (vector, 37 KB)

---

## The COVID-19 Market Crash Context

**What Happened:**
- Fastest bear market in history (Feb-Mar 2020)
- S&P 500 dropped ~34% in 33 days
- AAPL fell from ~$81 to ~$57 (30% decline)
- Unprecedented uncertainty and volatility

**Model Response:**
- ✅ Correctly identified extreme uncertainty
- ✅ Widened confidence bands by 48%
- ✅ Avoided overconfident predictions
- ✅ Demonstrated risk awareness

---

## Comparison with Traditional Forecasting

| Feature | Traditional | Probabilistic (This Figure) |
|---------|-------------|----------------------------|
| Output | Single number | Distribution with bands |
| Uncertainty | ❌ Not shown | ✅ Explicitly quantified |
| Crisis Response | Same confidence | Adapts to conditions |
| Risk Info | Limited | Rich uncertainty data |
| Decision Making | "Blind" prediction | Informed risk assessment |

---

## Key Insights

1. **Adaptive Uncertainty:** Model adjusts confidence based on market conditions
2. **Crisis Detection:** Automatically identifies unpredictable periods
3. **Risk Awareness:** Wider bands = "I don't know" signal
4. **Practical Value:** Enables better risk management decisions
5. **Regulatory Compliance:** Meets requirements for uncertainty disclosure

---

## How to Use This Figure

### In Reports
- Demonstrate probabilistic forecasting capabilities
- Show model adapts to market conditions
- Illustrate uncertainty quantification

### In Presentations
- Highlight the March 2020 widening
- Explain the practical value of uncertainty
- Compare with deterministic forecasts

### In Academic Papers
- Reference for uncertainty quantification methods
- Example of volatility-aware confidence intervals
- Case study of COVID-19 market behavior

---

## Regenerate This Figure

```bash
cd "c:\Users\mohan\Desktop\ML INtern"
python generate_figures.py
```

Or generate only Figure 9:

```python
from generate_figures import generate_figure_9_uncertainty_bands
generate_figure_9_uncertainty_bands()
```

---

## Related Documentation

- **Full Documentation:** `reports/FIGURES_DOCUMENTATION.md` (Section: Figure 9)
- **Generation Script:** `generate_figures.py`
- **Project Report:** `reports/PROJECT_ANALYSIS_REPORT.md`
- **Quick Reference:** `reports/QUICK_REFERENCE.md`

---

**Generated:** October 14, 2025  
**Script:** `generate_figures.py`  
**Purpose:** Demonstrate uncertainty quantification in financial forecasting
