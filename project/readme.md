# Cross-Sectional Volatility Mispricing in Equity Options

### A Statistical Arbitrage Approach

## 1. Introduction

Options markets provide forward-looking information about volatility, tail risk, and asymmetry in asset returns. A well-established empirical result is that implied volatility (IV) is a biased predictor of future realized volatility (RV), reflecting the existence of a volatility risk premium. However, this bias is not uniform across assets or time and varies with market conditions, liquidity, and risk perceptions.

This project proposes a **cross-sectional statistical arbitrage strategy** that exploits relative mispricing of volatility across equity options. Rather than attempting to forecast volatility in absolute terms, we model expected realized volatility and identify assets where implied volatility appears high or low relative to model-based forecasts. The strategy constructs approximately delta-neutral option positions, primarily at-the-money (ATM), and takes long and short volatility exposures in a market-neutral manner.

The emphasis of the project is on **modeling, structure, and economic interpretation**, rather than maximizing returns. Extensions explore the relationship between volatility mispricing, skew, and tail-risk regimes.

---

## 2. Related Literature

The project builds on several strands of literature:

* **Implied vs. Realized Volatility**
  Andersen & Bollerslev (1998); Christensen & Prabhala (1998); Poon & Granger (2003)

* **Volatility Risk Premia and Option Returns**
  Bakshi & Kapadia (2003); Carr & Wu (2009); Bollerslev, Tauchen & Zhou (2009)

* **Skewness and Tail Risk**
  Harvey & Siddique (2000); Buraschi, Trojani & Vedolin (2014); Kelly & Jiang (2014)

* **Machine Learning in Volatility Modeling**
  Sirignano & Cont (2019); Dixon, Polson & Sokolov (2020)

These studies motivate both the existence of volatility mispricing and the use of richer models to capture nonlinear and regime-dependent dynamics.

---

## 3. Methodology

### 3.1 Volatility Forecasting

Let  $\sigma^{\text{imp}}_{i,t}$ denote the implied volatility of an ATM option on asset $i$ at time $t$, and let
$\mathbb{E}_t[\sigma^{\text{real}}_{i,t+1}]$ denote the forecast of realized volatility over the option’s maturity.

We define the volatility mispricing signal as:
$$
M_{i,t} = \sigma^{\text{imp}}_{i,t} - \mathbb{E}_t[\sigma^{\text{real}}_{i,t+1}]
$$

Realized volatility forecasts will be generated using multiple approaches:

* GARCH(1,1) and GJR-GARCH models
* HAR-RV models using realized variance
* Machine learning models (e.g., XGBoost), potentially using GARCH forecasts as inputs alongside market variables

Model performance will be evaluated based on **cross-sectional ranking accuracy**, rather than solely on forecasting error metrics.

---

### 3.2 Trading Strategy

At each rebalancing date:

* Rank assets by $M_{i,t}$
* Go **long volatility** on assets with low implied relative to expected realized volatility
* Go **short volatility** on assets with high implied relative to expected realized volatility

Positions are constructed using ATM option structures (e.g., straddles) and are designed to be:

* Approximately delta-neutral
* Cross-sectionally market-neutral
* Concentrated in liquid options to reduce execution noise

---

### 3.3 Skew and Regime Diagnostics

As an extension, the project examines whether volatility mispricing is systematically related to:

* Implied skew (e.g., risk reversals)
* Market stress indicators
* Tail-risk regimes

This analysis helps interpret periods of strategy underperformance and links volatility arbitrage to skew and crash risk without explicitly trading skew in the baseline strategy.

---

## 4. Data Requirements

* Daily equity prices
* Option prices and implied volatilities (ATM)
* Option Greeks
* Risk-free interest rates
* Optional: intraday data for realized volatility, VIX/VVIX, market factor returns

---

## 5. Project Phases

### Phase 1: Proof of Concept

* Implement a baseline volatility forecasting model (e.g., GARCH or HAR-RV)
* Construct a cross-sectional volatility arbitrage strategy
* Ignore transaction costs and execution frictions
* Evaluate signal stability and basic performance

---

### Phase 2: Enhanced Realism

* Introduce transaction costs and bid–ask spreads
* Restrict universe to highly liquid options
* Incorporate execution delays
* Control turnover and position sizes

---

### Phase 3: Toward Production-Level Research

* Joint modeling of volatility and skew
* Regime-switching or state-dependent volatility forecasts
* Capital and liquidity constraints
* Stress testing during crisis periods
* Risk budgeting and portfolio-level constraints

This phase outlines potential extensions if additional time and data were available.

---

## 6. Expected Contributions

The project aims to demonstrate:

* A structured approach to cross-sectional volatility arbitrage
* The role of advanced volatility modeling in trading decisions
* The interaction between implied volatility, realized volatility, and skew
* A strategy evaluated on conceptual rigor rather than return maximization