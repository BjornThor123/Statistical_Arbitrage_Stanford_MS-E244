# Project Proposal: Robust Statistical Arbitrage of Equity Volatility Surfaces

## 1. Executive Summary
**Objective:** To generate uncorrelated alpha by capturing mean-reversion in the **idiosyncratic skewness** of equity volatility surfaces, while explicitly filtering for liquidity cost and binary event risk.

**Core Intuition:**
Market participants often overpay for tail risk (skew) in single stocks due to temporary liquidity demands or panic, rather than fundamental shifts. While a stock's skew generally moves in lockstep with its sector (systematic fear), deviations from this relationship often represent inefficient pricing. By isolating this "residual noise"—the portion of skew *not* explained by sector movements—we can construct a portfolio that profits as these pricing errors normalize.

---

## 2. Universe Selection & Data Hygiene
*Assumption: Statistical patterns are only valid in liquid markets where transaction costs do not consume the theoretical edge.*

To ensure tradeability, the eligible universe is dynamically filtered:
1.  **Liquidity Gate:** Stocks must be in the top percentile of Average Daily Option Volume (ADOV).
2.  **Spread Constraint:** The average bid-ask spread on 25-Delta options must be $\le X\%$ of the theoretical signal strength.
3.  **Event Blackout:** Any stock with an earnings release, analyst day, or major corporate action within the trade horizon is **hard-excluded**.
    * *Reasoning:* Skew pre-earnings is a rational pricing of binary risk, not an inefficiency to be arbitraged.

---

## 3. Signal Generation: The Beta-Adjusted Residual

*Assumption: A simple comparison (Stock Skew minus Sector Skew) is flawed because high-beta stocks naturally have more volatile skew. We must compare a stock to its "expected" self.*

### Step 1: Signal Generation (The "No-Arbitrage" Smoothing)

*Assumption: Raw market data is discrete and noisy (e.g., strikes at 22$\Delta$ and 28$\Delta$). To trade a constant 25-Delta exposure, we must interpolate a smooth, continuous surface that forbids static arbitrage.*

We fit **Jim Gatheral’s Surface SVI (SSVI)** to the raw option chain of every stock in the universe.

* **Why SSVI?** It guarantees free-of-arbitrage conditions (no negative butterfly spreads or calendar arbitrage).
* **The Output:** A parametrized volatility surface $\sigma(k, t)$ that allows us to solve for the exact Implied Volatility at the 25-Delta point, regardless of available strikes.

### Step 2: The Rolling Regression
Instead of arithmetic subtraction, we calculate the **Idiosyncratic Residual** via a rolling linear regression over a moving window (e.g., 60 days):

$$\text{Skew}_{i,t} = \alpha_i + \beta_i (\text{Skew}_{\text{Sector}, t}) + \epsilon_{i,t}$$

* $\text{Skew}_{i,t}$: The 25-Delta Put/Call spread for Stock $i$.
* $\beta_i$: The sensitivity of Stock $i$'s skew to the Sector's skew.
* $\epsilon_{i,t}$ **(The Signal):** The unexplained residual.

### Step 3: Z-Score Ranking
We standardize the residuals to identify statistical outliers:
$$Z_{i,t} = \frac{\epsilon_{i,t} - \mu_{\epsilon}}{\sigma_{\epsilon}}$$

* **Signal:** We trade stocks where $|Z_{i,t}| > 2.0$ (statistically significant dislocation).

---

## 4. Execution Strategy: Asymmetric Risk Profiles

*Assumption: "Long Skew" and "Short Skew" are not symmetrical risks. Shorting skew exposes the portfolio to unbounded gap risk (e.g., fraud, lawsuits) and requires structural protection.*

### A. The "Cheap Skew" Leg (Long Vol-of-Vol)
* **Signal:** $Z < -2.0$ (Skew is too flat relative to sector).
* **Structure:** **Risk Reversal.**
    * Buy OTM Put / Sell OTM Call.
    * *Intuition:* We profit if the skew steepens (puts get more expensive relative to calls).

### B. The "Expensive Skew" Leg (Short Vol-of-Vol)
* **Signal:** $Z > 2.0$ (Skew is too steep relative to sector).
* **Structure:** **Short Seagull (Capped Risk).**
    * Sell OTM Put / Buy OTM Call / **Buy Deep OTM Put**.
    * *Intuition:* We profit if the skew flattens. The "wing" put (Deep OTM) acts as a catastrophe insurance policy, capping losses if the stock crashes.

---

## 5. Portfolio Construction Constraints

The portfolio is constructed using an optimizer to solve for weights $w_i$ that maximize expected convergence while satisfying:

1.  **Delta Neutrality:** Net portfolio Delta is constrained to zero (hedged daily or via bands).
2.  **Sector Neutrality:** Exposure to any single sector (e.g., Tech, Energy) is capped at a fixed $\%$ of Gross Market Value (GMV).
3.  **Proportional Weights:** Position sizes are inversely proportional to the stock's realized volatility ($\frac{1}{\sigma_{stock}}$).

---

## 6. Transaction Cost Modelling (TCM)

*Assumption: A theoretical arb is a loss if the cost to enter/exit exceeds the convergence profit.*

We define the **Net Expected Value (NEV)** of a trade as:

$$E[\text{P\&L}] = (\text{Expected Convergence} \times \text{Vega}) - \text{Total Cost}$$

Where **Total Cost** is modeled explicitly:
$$\text{Total Cost} = 2 \times (\text{Half-Spread} + \text{Impact} + \text{Commission}) + \text{Hedge Drag}$$

* **Half-Spread:** The distance from Mid-Price to Bid/Ask.
* **Impact:** Modeled as a function of trade size relative to Average Daily Volume (ADV).
* **Hedge Drag:** Estimated cost of re-balancing the Delta hedge over the holding period (Gamma $\times$ Volatility $\times$ Spread).

**Rule:** Trades are only executed if $E[\text{P\&L}] > \text{Hurdle Rate}$.

---

## 7. Risk Management & Stop-Losses

1.  **Correlation Breakdown:** If the average pairwise correlation of the universe spikes above $0.7$ (systemic crash), the "idiosyncratic" assumption fails. **Action:** Halve position sizes or liquidate.
2.  **Gap Risk:** The Short Seagull structure structurally caps gap risk on individual names.
3.  **Drawdown Limit:** If Strategy Drawdown $> Y\%$ of NAV, trading is paused to recalibrate volatility regimes.