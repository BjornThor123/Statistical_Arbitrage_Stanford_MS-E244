# Project Proposal: Systematic Cross-Sectional Equity Skew Arbitrage

## 1. Executive Summary
**Objective:** To systematically identify and monetize inefficiencies in the volatility surface slope (Skew) across a diversified universe of liquid equities.

**Thesis:** Unlike broad indices, single-name equities exhibit dynamic skew profiles driven by idiosyncratic factors (leverage, earnings, M&A speculation). Market participants often overpay for "lottery tickets" (OTM Calls) or "crash protection" (OTM Puts) relative to realized moves. By applying a **Surface SVI (SSVI)** model to enforce no-arbitrage constraints, we can isolate pure sentiment dislocations—selling expensive tails (skew) while neutralizing directional exposure (delta).

---

## 2. Theoretical Framework

### The Model: Surface SVI (SSVI)
To compare volatility surfaces across hundreds of tickers without "Calendar Arbitrage," we fit **Jim Gatheral’s Surface SVI** for each asset. This guarantees that the volatility surface is consistent across both strike ($k$) and time ($\tau$).

$$w(k, \theta_\tau) = \frac{\theta_\tau}{2} \left( 1 + \rho \phi(\theta_\tau) k + \sqrt{( \phi(\theta_\tau) k + \rho )^2 + (1-\rho^2)} \right)$$

* **$\rho$ (Correlation):** The primary signal.
    * $\rho < 0$: Typical "Smirk" (Downside fear).
    * $\rho > 0$: "Inverted Skew" (Upside speculation/M&A risk).
* **Constraint:** We apply the Heston-like constraint on $\phi(\theta_\tau)$ to ensure no butterfly arbitrage, which is critical when dealing with less liquid single-name chains.

### Regime: Sticky Delta vs. Jump Risk
Single equities are prone to "Gaps" (e.g., overnight news). While we hedge assuming **Sticky Delta** (volatility moves with price), we must account for **Jump Diffusion** risk.
* *Implication:* The "smoothness" of the SSVI fit is a quality filter. If the market prices discrete jumps (e.g., earnings), SSVI fits will degrade. We filter these out to avoid model error.

---

## 3. Methodology

### A. Universe Selection & Data Ingestion
* **Universe:** Top 500 Liquid Global Equities (by ADV > $50M).
* **Filtering:**
    * **Earnings Filter:** Exclude any ticker with an earnings release within the target option expiry (to avoid binary event risk).
    * **Liquidity Filter:** Exclude chains where the ATM Bid-Ask spread > 15bps of spot.
    * **Hard-to-Borrow (HTB) Check:** Exclude stocks with high borrow costs (makes shorting the delta hedge expensive/impossible).

### B. Surface Calibration (The "Global Fit")
For each ticker $i$ at time $t$:
1.  **Fit SSVI:** Minimize the weighted RMSE for the entire surface.
2.  **Quality Score ($Q_i$):** If the fit error (RMSE) > Threshold, the surface is deemed "broken" (likely due to illiquidity or discrete dividends). **Discard Ticker.**
3.  **Interpolate:** Extract constant-maturity (e.g., 45-day) Implied Vols for:
    * $IV_{25\Delta P}$ (Put)
    * $IV_{25\Delta C}$ (Call)

### C. Signal Generation (Bi-Directional)
We look for statistical outliers in the **Skew Premium**.

1.  **Metric:** Normalized Skew Z-Score:
    $$Z_{Skew} = \frac{(\text{Skew}_{i,t} - \mu_{i, \text{hist}})}{\sigma_{i, \text{hist}}}$$
    Where $\text{Skew} = IV_{25\Delta P} - IV_{25\Delta C}$.

2.  **Entry Signals:**
    * **Case A: Panic (Overpriced Puts):** $Z_{Skew} > 2.0$. The market is paying too much for downside.
        * *Action:* **Sell Put Skew** (Short OTM Puts / Buy OTM Calls).
    * **Case B: Mania (Overpriced Calls):** $Z_{Skew} < -2.0$. The market is paying too much for upside.
        * *Action:* **Sell Call Skew** (Short OTM Calls / Buy OTM Puts).

---

## 4. Execution Strategy: The "Dynamic Seagull"

We structure the trade to cap the tail that we are shorting.

### Scenario A: Shorting Downside Skew (The "Put Seagull")
* **View:** Fear is overstated; stock will stabilize or rise.
* **Structure:**
    1.  **Sell** 25$\Delta$ Put (Income).
    2.  **Buy** 25$\Delta$ Call (Funded by Leg 1).
    3.  **Buy** 10$\Delta$ Put (The Cap).
* **Hedge:** Short Underlying Stock (Delta Neutral).

### Scenario B: Shorting Upside Skew (The "Call Seagull")
* **View:** Speculation is overheated; stock will stabilize or fall.
* **Structure:**
    1.  **Sell** 25$\Delta$ Call (Income).
    2.  **Buy** 25$\Delta$ Put (Funded by Leg 1).
    3.  **Buy** 10$\Delta$ Call (The Cap).
* **Hedge:** Long Underlying Stock (Delta Neutral).

---

## 5. Risk Management

* **Idiosyncratic Risk:**
    * **Position Limit:** Max 2% NAV per ticker.
    * **Sector Limit:** Max 15% NAV per sector (to avoid correlation in a sector-wide crash).
* **Gamma Risk:**
    * **Gamma Scalping:** Re-hedge delta if spot moves > 1 standard deviation.
    * **Expiration Limit:** Roll or close positions 7 days prior to expiry to avoid "Pin Risk."
* **Stop Loss:**
    * **Hard Stop:** If the spread widens by 2$\sigma$ against the position.
    * **Soft Stop:** If Borrow Cost increases > 50bps (squeeze risk).

---

## 6. Tools & Implementation
* **Language:** Python.
* **Libraries:**
    * `scipy.optimize` (SSVI Calibration).
    * `QSTrader` or Custom Execution Algo (for basket orders).
* **Data:** Tick-level option data (OPRA) required for accurate Bid/Ask modeling in single names.
* **Visualization:** Rank-ordered Bar Chart of $Z_{Skew}$ across the universe (Top 10 "Rich" vs Top 10 "Cheap").