# Project Proposal: SABR Skew Term Structure Arbitrage

## 1. Executive Summary
**Objective:** To systematically identify and monetize dislocations in the **Inter-Temporal Skew** of the Swaption Volatility Surface.

**Thesis:** While the absolute level of volatility (Alpha) is efficient, the market often misprices the **speed of mean reversion** for skewness (Rho). Panic regarding rate moves is often priced aggressively in the short end of the curve (e.g., 1-year expiry) while the long end (e.g., 5-year expiry) remains complacent. This project constructs a delta-neutral, vega-neutral strategy to arbitrage these inconsistencies.

---

## 2. Theoretical Framework

### The Model: SABR (Stochastic Alpha, Beta, Rho)
We utilize the Hagan et al. (2002) SABR model to decompose the volatility surface into its constituent risk factors.
* **$\alpha$ (Alpha):** The level of ATM volatility.
* **$\beta$ (Beta):** The backbone (relationship between rate levels and vol).
* **$\rho$ (Rho):** The **Skew** (correlation between the rate and its volatility).
* **$\nu$ (Nu):** The "Vol of Vol" (convexity/smile curvature).

### The Opportunity: Skew Term Structure
In standard markets, the skew parameter ($\rho$) should exhibit a smooth term structure across expiries (e.g., the skew for a 1y10y swaption should be highly correlated with a 2y10y swaption).
* **Dislocation:** During idiosyncratic macro events (e.g., specific central bank meeting fears), short-dated skew ($\rho_{1y}$) often disconnects from long-dated skew ($\rho_{5y}$).
* **The Trade:** We view the spread $\Delta \rho = \rho_{1y} - \rho_{5y}$ as a mean-reverting signal.

---

## 3. Methodology

### A. Data Ingestion
* **Input:** Swaption Volatility Grid (ATM, 25-Delta Risk Reversals, 25-Delta Butterflies).
* **Dimensions:** Expiry (1M to 10Y) $\times$ Tenor (2Y to 30Y).

### B. Calibration Engine
For every observation date $t$ and every tenor $T$:
1.  Solve the SABR parameters $(\alpha_t, \rho_t, \nu_t)$ that minimize the squared error between model implied volatilities and market quotes.
2.  **Output:** A time-series of the **Skew Parameter $\rho$** for each point on the grid.

### C. Signal Generation (The "Z-Score Map")
We calculate the rolling Z-Score of the skew spread between maturities:
$$Z_{spread} = \frac{(\rho_{Expiry A} - \rho_{Expiry B}) - \mu_{spread}}{\sigma_{spread}}$$

* **Entry Signal:** If $|Z_{spread}| > 2.0$, execute a **Calendar Skew Box**.
* **Direction:**
    * If Short-End Skew is historically *expensive* vs Long-End: **Sell Short-End Skew / Buy Long-End Skew**.

---

## 4. Execution Strategy: The "Skew Box"

To isolate the skew ($\rho$) while hedging out level ($\alpha$) and convexity ($\nu$), we utilize a **Risk Reversal Calendar Spread**:

| Leg | Instrument | Action | View |
| :--- | :--- | :--- | :--- |
| **Leg 1** | **Short-Term (1y10y)** | **Sell Risk Reversal** | Sell the "expensive" fear of rates dropping/rising in the near term. |
| **Leg 2** | **Medium-Term (5y10y)** | **Buy Risk Reversal** | Buy "cheap" protection for the same move in the longer term. |

* **Delta Hedge:** Continuously hedged with underlying Swaps.
* **Vega Hedge:** Net Vega is minimized by weighting the notionals (contracts) by their respective Vegas ($\text{Vega}_{1y} \times N_1 = \text{Vega}_{5y} \times N_2$).

---

## 5. Expected Outcomes & Risk Profile

* **Return Profile:** Negatively skewed "Carry" style returns. The strategy earns premium as the "panic" in the short-end skew subsides and the term structure normalizes.
* **Risk:** A structural regime change where the short-end and long-end skews permanently de-correlate (e.g., a permanent inversion of the yield curve volatility surface).
* **Key Metric:** Information Ratio (IR) of the Skew Spread mean reversion.

---

## 6. Tools & Implementation
* **Language:** Python.
* **Libraries:** `QuantLib` (for Swaption pricing/Greeks), `scipy.optimize` (for SABR calibration).
* **Visualization:** Heatmaps of $\rho$ Z-scores across the Expiry $\times$ Tenor grid.