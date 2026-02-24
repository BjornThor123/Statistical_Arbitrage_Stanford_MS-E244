# Project Proposal: Systematic Equity Skew Arbitrage (SSVI)

## 1. Executive Summary
**Objective:** To isolate and monetize the "Variance Risk Premium" embedded in the structural skewness of the S&P 500 implied volatility surface, while explicitly capping tail risk.

**Thesis:** The equity volatility surface exhibits a persistent "Smirk"—a structural overpricing of downside puts relative to upside calls due to institutional hedging demand. This "Fear Premium" fluctuates with sentiment. We treat the **slope of the volatility surface** (Skew) as a mean-reverting signal. By using a **Surface SVI (SSVI)** model to guarantee no-arbitrage across time, we can systematically sell overpriced skew when fear is statistically excessive, hedging delta exposure to remain market-neutral.

---

## 2. Theoretical Framework

### The Model: Surface SVI (SSVI)
To avoid "Calendar Arbitrage" (where independent slices imply negative forward variance), we utilize **Jim Gatheral’s Surface SVI** parameterization. This ensures the volatility surface is consistent across both strike ($k$) and time to maturity ($\tau$).

The total implied variance $w(k, \theta_\tau)$ is modeled as:
$$w(k, \theta_\tau) = \frac{\theta_\tau}{2} \left( 1 + \rho \phi(\theta_\tau) k + \sqrt{( \phi(\theta_\tau) k + \rho )^2 + (1-\rho^2)} \right)$$

* **$\theta_\tau$**: ATM total variance for maturity $\tau$.
* **$\phi(\theta_\tau)$**: Function controlling the skew backbone.
* **$\rho$**: Correlation parameter (determines the asymmetry/skew direction).
* **Constraint:** The function $\phi$ must satisfy Heston-like constraints (e.g., $\phi(\theta) \le \frac{1}{\rho^2}(1+|\rho|)$) to prevent **Butterfly Arbitrage** (negative probability density).

### Hedging Assumption: Sticky Delta
Unlike rates, equity volatility surfaces move with the spot price. We calculate all Greeks (Delta, Vega) assuming a **Sticky Delta** regime:
* *Implication:* As the market falls, the volatility surface "slides" to the left. The implied volatility of a fixed strike put increases.
* *Action:* Our Delta hedge must account for this `dVol/dSpot` correlation to prevent under-hedging during a crash.

---

## 3. Methodology

### A. Data Ingestion & Cleaning
* **Input:** SPX Option Chains (1-minute intervals or EOD).
* **Filtering:**
    * Remove quotes with Spread > 5% of Mid.
    * Remove strikes with Volume < 100 or Open Interest < 500.
    * **Exclusion:** Do not trade options within 7 days of expiration (Gamma Week risk).

### B. Surface Calibration (The "Arb-Free Fit")
For each observation time $t$:
1.  **Fit SSVI:** Minimize the weighted RMSE between model variance and market mid-prices across **all maturities simultaneously**.
2.  **Arb Check:** Verify **Durrleman’s Condition** (non-negative second derivative of Call Price with respect to Strike) to ensure no Butterfly Arbitrage exists in the fitted curve.
3.  **Interpolate:** Extract constant-maturity (e.g., 45-day) Implied Volatility for:
    * $IV_{25\Delta P}$ (Put)
    * $IV_{25\Delta C}$ (Call)
    * $IV_{10\Delta P}$ (Tail Put)

### C. Signal Generation (The Composite Score)
We avoid the "Low Vol Trap" by requiring both Relative and Absolute skew to be elevated.

1.  **Metric 1: Relative Skew Z-Score:**
    $$Z_{Rel} = \text{Z-Score} \left( \frac{IV_{25\Delta P} - IV_{25\Delta C}}{IV_{ATM}} \right)$$
2.  **Metric 2: Absolute Skew Threshold:**
    $$\text{Spread}_{Abs} = IV_{25\Delta P} - IV_{25\Delta C}$$

* **Entry Signal:** **SELL SKEW** if:
    * $Z_{Rel} > 2.0$ (Skew is statistically steep) **AND**
    * $\text{Spread}_{Abs} > \text{40th Percentile}$ (Premium is absolutely rich).

---

## 4. Execution Strategy: The "Seagull" / Capped Risk Reversal

To monetize the skew while strictly capping the downside tail risk (Black Swan protection), we structure the trade as a **Ratio Spread** or **Seagull**:

| Leg | Instrument | Action | Greeks Impact |
| :--- | :--- | :--- | :--- |
| **Leg 1** | **25$\Delta$ Put** | **Sell 1x** | Short Vega, Short Gamma (The Alpha Source). |
| **Leg 2** | **25$\Delta$ Call** | **Buy 1x** | Long Vega, Long Gamma (Offsets Leg 1 locally). |
| **Leg 3** | **10$\Delta$ Put** | **Buy 1x** | **The Cap.** Stops the loss if market crashes >10%. |
| **Hedge** | **SPX Futures** | **Dynamic** | Delta hedge daily to maintain neutrality. |

* **Net Exposure:** Short Skew (profit if Put Vol falls relative to Call Vol).
* **Max Loss:** Defined by the spread width between the 25$\Delta$ Put and 10$\Delta$ Put.
* **Exit Condition:**
    * **Target:** Skew reverts to mean ($Z_{Rel} < 0.5$).
    * **Stop Loss:** Portfolio drawdown > 3% or VIX spike > 5 points in a single session.

---

## 5. Expected Outcomes & Risk Profile

* **Classification:** Statistical Relative Value (Volatility Risk Premium).
* **Return Driver:** Mean reversion of the implied volatility slope ($\rho$).
* **Primary Risks:**
    * **"Crash-Up":** A violent rally where Call Skew explodes (short squeeze).
    * **Path Dependency:** Choppy markets where daily delta hedging bleeds P&L (Gamma scalping loss).
* **Key Metric:** Sortino Ratio (focus on downside volatility).

---

## 6. Tools & Implementation stack
* **Core:** Python 3.11+.
* **Optimization:** `scipy.optimize.minimize` (SLSQP solver for constrained SSVI fitting).
* **Pricing:** `QuantLib` (for accurate Greeks handling Sticky Delta).
* **Visualization:** `plotly` for 3D interactive Volatility Surfaces (Time x Strike x Vol).