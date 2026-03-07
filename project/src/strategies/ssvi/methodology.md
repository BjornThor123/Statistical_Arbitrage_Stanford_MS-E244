# SSVI Skew Arbitrage: Complete Methodology

This document describes every step of the strategy as it was actually run. The universe is 5 US financial stocks — BAC, BRK, C, GS, JPM — with XLF as the sector proxy. The sample period is January 3, 2006 through December 31, 2020. The factor model is PCA with 2 components.


## The trade

Every trading day, we fit a parametric volatility surface to each stock's listed options, extract the implied volatility skew at 30-day ATM maturity, remove the component of that skew that's explained by common cross-sectional factors (via PCA on the entire universe of skew series), and trade the residual when it deviates far enough from its recent norm.

In a real implementation, "trading the residual" would mean constructing an option portfolio that has exposure to the implied volatility skew — for instance, buying OTM puts and selling OTM calls (or vice versa), delta-hedged daily, with the option strikes and notionals chosen so the portfolio's value moves with the skew. **This backtest does not simulate that.** It does not buy or sell any options, does not compute greeks, does not maintain or rebalance a delta hedge, and does not track the mark-to-market of any actual option position.

Instead, the backtest treats the skew residual as a directly tradeable quantity. The P&L on any given day, for a single name, is simply:

    gross_pnl = weight × (−Δresidual)

where Δresidual = residual_tomorrow − residual_today. If we are "long skew" on BAC (positive weight) because BAC's skew residual is −2 standard deviations, and tomorrow the residual moves from −0.08 to −0.05 (mean-reverting toward zero), then Δresidual = +0.03 and the gross P&L is weight × (−0.03) = a positive number. We profit when the residual reverts.

This is an idealization. It assumes: (1) you can construct an option portfolio whose daily P&L tracks the skew residual one-for-one, (2) the mapping between skew change and option P&L is linear and has unit coefficient, and (3) all other greeks (delta, vega to parallel shifts, theta) are perfectly hedged or cancel out. In practice, the actual P&L would also depend on realized volatility, the gamma/theta tradeoff of the option positions, discrete hedging slippage, and the vega profile of the specific strikes chosen. The backtest approximates some of these missing costs via a flat "gamma drag" term in the cost model (see Step 16), but this is a rough proxy, not a simulation of hedging.


## Step 1: Load and filter option data

We read daily option observations from OptionMetrics-format data. Each observation has a date, expiration, strike, call/put flag, implied volatility, bid, ask, open interest, and volume. The data comes either from zipped CSVs or a DuckDB table — the processing is identical either way.

Initial filters discard observations with implied volatility outside [0.01, 5.0], zero or negative strikes, non-positive option mid-prices, and illiquid contracts (open interest + volume = 0).


## Step 2: Estimate forward prices

For each (date, expiration) pair, we estimate the forward price F using put-call parity. Given matched call-put pairs at the same strikes, the relationship C − P = S·e^(−q·t) − K·e^(−r·t) simplifies to a linear relationship between C − P and K. We regress C − P on K across all available strikes:

    C − P = slope · K + intercept

The forward price is F = −intercept / slope. This works because in the risk-neutral relationship C − P = F − K (after absorbing discounting into the implied vol), the intercept equals F and the slope equals −1. We require at least 4 unique strikes, a positive slope, and a finite positive F; otherwise we discard that expiration for that day.


## Step 3: Compute moneyness and maturity

With the forward price, we compute log-moneyness k = ln(K/F) and time to maturity t = (expiration − date) / 365 in years. We discard observations where t < 1/365 (less than one day) or t > 2.5 years, and where |k| > 1.5 (far out-of-the-money). This leaves a panel of (k, t, σ) triples for each stock-day.


## Step 4: Robust trimming

Before fitting, we remove outlier implied volatilities. For each stock-day, we compute the median and MAD (median absolute deviation) of σ across all observations, and discard any observation where σ falls more than 5 × 1.4826 × MAD from the median. (The factor 1.4826 makes the MAD consistent with the standard deviation under normality.) We require at least 120 surviving data points and at least 3 distinct expiration dates to attempt a surface fit. If a stock-day doesn't meet these thresholds, we skip it — that day gets no skew value for that stock.


## Step 5: Fit the SSVI surface

The SSVI (Surface Stochastic Volatility Inspired) parameterization models total implied variance w as a function of log-moneyness k and ATM total variance θ at each maturity:

    φ(θ) = η · θ^(−γ)
    w(k, θ) = ½ · θ · (1 + ρ·φ·k + √((φ·k + ρ)² + (1 − ρ²)))

Three global parameters — ρ, η, γ — govern the entire surface across all maturities simultaneously. ρ controls skew direction and magnitude (how tilted the smile is; ρ < 0 means OTM puts are more expensive, the standard equity pattern). η controls how much curvature the smile has. γ controls how curvature decays with maturity (γ near 1 means short-dated smiles are much more curved than long-dated ones). The only maturity-specific degree of freedom is the ATM total variance term structure θ(t), represented as a set of nodes interpolated linearly between observed maturities.

**Parameter domains and arbitrage constraint.** We require ρ ∈ (−1, 1), η > 0, γ ∈ (0, 1). The SSVI no-butterfly arbitrage sufficient condition is η · (1 + |ρ|) ≤ 2 — this ensures the fitted surface does not admit butterfly spread arbitrage at any strike or maturity. All three domain constraints and the butterfly condition are enforced during optimization via quadratic penalty terms added to the loss function: if η · (1 + |ρ|) exceeds 2, a penalty of 10 · (violation)² is added; if ρ hits the boundary, 10³ · (violation)² is added; and so on.

**Objective function.** The calibration minimizes weighted mean squared error between the SSVI model total variance and the market total variance w_market = σ² · t:

    loss = weighted_average((w_model − w_market)²) + constraint_penalties

Observation weights are (1/√t) · 1/(1 + (|k|/0.5)²). The 1/√t term upweights short-dated options, which are more relevant for our 30-day skew signal. The 1/(1 + (|k|/0.5)²) term is a Cauchy kernel that downweights far-OTM options where bid-ask noise is larger and the skew signal is less relevant.

**Optimization algorithm.** The algorithm alternates between two stages in an inner-outer loop:

*Stage A — parameter search.* Hold θ(t) nodes fixed and search over (ρ, η, γ). We generate 400 random proposals by perturbing the current best parameters: ρ is shifted by a Gaussian with initial standard deviation 0.08, η is multiplied by exp(Gaussian with std 0.18), γ is shifted by Gaussian with std 0.08. All proposals are clipped to the domain. We evaluate all 400 in parallel (on GPU this is a single batched operation) and keep the best. Then we do coordinate-wise refinement: try ±0.02 for ρ, ±0.05 in log-space for η, ±0.02 for γ, keeping any improvement.

*Stage B — θ optimization.* Hold (ρ, η, γ) fixed and optimize each θ node independently. For each maturity, create a geometric grid of 31 candidates between 0.5× and 1.5× the current value, evaluate the loss for each, and pick the best. Then enforce monotonicity: θ values must be non-decreasing in maturity (θ(t₁) ≤ θ(t₂) when t₁ < t₂), which we enforce by taking the cumulative maximum.

These two stages alternate 3 times per "restart," and step sizes decay by a factor of 0.85–0.88 after each alternation. We run 4 restarts from different initial seeds (three hardcoded seeds with ρ ∈ {−0.4, −0.5, −0.6}, η ∈ {0.35, 0.5, 0.8}, γ ∈ {0.4, 0.55, 0.65}, plus the previous day's parameters as a warm start). The best result across all restarts wins.

**Initial θ guess.** For each maturity, we find the "near-ATM" observations (those with |k| in the smallest 25th percentile, or the 3 closest to ATM if the percentile gives fewer), take the median of their total variance, and enforce monotonicity. This gives a reasonable starting point for the ATM term structure.


## Step 6: Extract the skew number

Given the calibrated surface, we compute the volatility skew at the target point: 30-day maturity (t* = 30/365) and ATM log-moneyness (k₀ = 0). The skew is the derivative of implied volatility with respect to log-moneyness:

    dw/dk = ½ · θ · φ · (ρ + (φ·k₀ + ρ) / √((φ·k₀ + ρ)² + (1 − ρ²)))

    skew = (dw/dk) / (2 · √w · √t*)

where w is the SSVI total variance at the target point. This is the slope of the implied volatility smile at ATM for 30-day options. A large negative value means OTM puts are much more expensive than OTM calls (steep skew); a value near zero means a flat smile. We compute one skew number per stock per day.

We also store the fitted parameters (ρ, η, γ), the calibration RMSE in both total variance and implied volatility space, the no-butterfly constraint score (2 − η(1+|ρ|), positive means arbitrage-free), the number of data points and maturities used, and the θ term structure bounds.


## Step 7: Build the cross-sectional factor model

We want to isolate the part of each stock's skew that is idiosyncratic — the part that moves independently of common factors affecting all financial stocks. This run uses PCA (principal component analysis) on the universe of 5 stock skew series.

On each day d, we take the trailing 60-day window of skew values for all 5 stocks, arranged as a 60 × 5 matrix. We drop any ticker that has more than 40% missing values in this window. We fill remaining gaps with the column mean, standardize each column to zero mean and unit variance, and compute the singular value decomposition (SVD). The first 2 right singular vectors (V₁, V₂) are the factor loadings — they describe the two dominant patterns of co-movement in the cross-section of skews.

We project today's standardized skew vector onto these loadings to get today's factor scores:

    z_today = (skew_today − mean) / std    (standardized, 1 × 5 vector)
    [pc1, pc2] = z_today · [V₁ V₂]        (2 factor scores)

Then, for each individual stock, we regress its skew history against these two factors using ordinary least squares on the same 60-day window. We winsorize both the dependent and independent variables at the 2nd and 98th percentiles before regression to limit the influence of extreme observations:

    stock_skew_t = α + β₁·PC1_t + β₂·PC2_t + ε_t

The regression uses at least 40 observations (the `min_regression_obs` parameter). The residual for today is:

    residual = stock_skew_today − (α̂ + β̂₁·PC1_today + β̂₂·PC2_today)

This residual is what we trade. The PCA factors capture the common modes of skew variation — if a macro event (bank stress, rate shock, regulatory announcement) moves all financial skews together, that co-movement is absorbed by the factors. The residual is what's left: the stock-specific skew dislocation. The regression is re-estimated every day on a rolling window, so the factor loadings adapt to the evolving correlation structure.

Why PCA instead of simply regressing on the XLF sector ETF's skew? The sector ETF approach (stock_skew = α + β·sector_skew + ε) uses a single factor. PCA uses two factors extracted from the actual data, so it can capture richer common structure — for instance, if BAC and C move together in a way that's different from how GS and JPM move, PCA can separate those two patterns, while a single sector regression would blend them. The tradeoff is that PCA requires the full universe of skew series as input and is more prone to noise when the universe is small.


## Step 8: Z-score the residual

We standardize the residual using a 60-day rolling window. The standardization uses the median and MAD (median absolute deviation) rather than the mean and standard deviation, because MAD is robust to occasional large residual spikes that would inflate the standard deviation and suppress subsequent z-scores:

    center = median(residual over last 60 days)
    scale = 1.4826 · MAD(residual over last 60 days)
    z = (residual_today − center) / scale

The factor 1.4826 makes the MAD consistent with the standard deviation under a Gaussian distribution. We call this `residual_scale` and store it alongside the z-score. We need at least 60 observations in the residual history before we can compute a z-score; before that, the stock gets no signal.


## Step 9: Regime filter

We don't trade when the sector is in a stress regime, because factor model relationships tend to break down during dislocations. The regime filter uses the XLF sector ETF's skew series:

1. Compute the 21-day rolling mean and standard deviation of sector skew.
2. Compute sector_abs_z = |sector_skew − rolling_mean| / rolling_std.
3. Compute the 21-day rolling volatility of sector skew itself, then z-score that volatility against its own rolling mean and std to get sector_vol_z.
4. The regime passes if sector_abs_z ≤ 3.0 AND sector_vol_z ≤ 2.0.

When sector skew is spiking (sector_abs_z > 3) or when sector skew volatility itself is elevated (sector_vol_z > 2), we flag the regime as stressed and block all trades.


## Step 10: Liquidity filter

A trade is blocked if the SSVI calibration for that stock on that day used fewer than 150 data points, or if the RMSE in implied volatility exceeded 0.60. The first condition ensures the option market was liquid enough to give a reliable skew estimate. The second ensures the SSVI surface actually fit the data well — a high RMSE means the parametric model couldn't capture the observed smile shape, so the extracted skew number is unreliable.


## Step 11: Edge-cost filter

Even if the z-score is extreme and the regime and liquidity filters pass, we only trade if the expected profit exceeds the estimated cost with a margin. The expected edge is the rate at which we expect the residual to revert, approximated as:

    expected_edge = |residual| / 5

This assumes the residual will mean-revert over roughly 5 days (the `edge_horizon_days` parameter), and divides by 5 to get a daily profit rate.

The estimated unit cost adapts to the quality and liquidity of the option data:

    liquidity_penalty = 0.5 · √(150 / n_points)
    fit_penalty = min(1.5, rmse_iv / 0.25)
    estimated_unit_cost = (round_trip_cost / 2) · (1 + liquidity_penalty + fit_penalty)

The round-trip cost is 2 × (half_spread + impact + commission) = 2 × (0.02 + 0 + 0.005) = 0.05, so the base per-side cost is 0.025. The liquidity penalty increases costs when there are fewer option data points (the denominator 150 is the minimum threshold; if n_points = 150 the penalty is 0.5; if n_points = 600 the penalty is 0.25). The fit penalty increases costs when the SSVI surface fits poorly (the denominator 0.25 means that at RMSE = 0.25 the penalty is 1.0, capped at 1.5).

A trade is allowed only if:

    expected_edge > estimated_unit_cost × 1.25

The 1.25 multiplier (from `edge_cost_buffer = 0.25`) requires the expected edge to exceed cost by 25%, providing a buffer against estimation error and execution slippage.


## Step 12: Direction inference

By default we mean-revert: a negative z-score (cheap skew) gets a positive weight (long skew), and vice versa. But this run uses `signal_direction = "auto"`, which checks whether mean-reversion actually worked over the recent past before committing to it.

The auto-direction logic looks back 63 days. For each stock-day in that window, it computes z × Δresidual (the product of the z-score and the next-day change in residual). If z and Δresidual tend to have the same sign — meaning that when z is positive (skew is rich) the residual tends to increase further, not revert — then mean-reversion is not working and the strategy flips to momentum (direction_multiplier = −1). If the average is non-negative (mean-reversion is working), direction_multiplier = 1.

This check requires at least 20 observations; before that, it defaults to mean-reversion. The direction multiplier is applied to z before candidate selection, so a multiplier of −1 effectively inverts all entry signals.


## Step 13: Portfolio construction

On each day, we look at all stocks that have valid signals (z-score computed, all filters passed). After applying the direction multiplier, long candidates are those with z ≤ −1.0 (entry_z) and short candidates are those with z ≥ 1.0. We sort each group by |z| (signal strength, largest first) and take the top 2 from each side (cross_section_top_k = cross_section_bottom_k = 2).

Weight allocation within each side is proportional to |z|. The total long allocation starts at 0.5 and total short at −0.5. For example, if BAC has z = −1.8 and GS has z = −1.3 after direction adjustment, the long allocations are BAC: 0.5 × 1.8/3.1 = 0.290, GS: 0.5 × 1.3/3.1 = 0.210. If JPM has z = +1.5 and C has z = +1.1, the short allocations are JPM: −0.5 × 1.5/2.6 = −0.288, C: −0.5 × 1.1/2.6 = −0.212.


## Step 14: Risk scaling

The raw candidate weights go through three transformations, applied in this exact order:

**Inverse-volatility scaling.** Each weight is multiplied by 1/residual_scale (the inverse of that stock's rolling residual volatility). If BAC's residual has a scale of 0.04 and GS's has a scale of 0.08, BAC's weight gets twice the multiplier of GS's. The rationale: a z-score of 2 in a low-volatility name represents a more reliable dislocation than the same z-score in a high-volatility name. After this multiplication, all weights are renormalized so their absolute values sum to 1.

**Per-name and leverage caps.** Each weight is clipped to [−0.35, +0.35] (max_name_weight). Then, if the gross leverage (sum of absolute weights) exceeds 1.5, all weights are scaled down proportionally to bring gross leverage to exactly 1.5.

**Volatility targeting.** We estimate the portfolio's daily P&L volatility as:

    vol_proxy = √(Σᵢ (|wᵢ| · residual_scaleᵢ)²)

This treats each name's contribution as independent (no cross-correlations). If vol_proxy exceeds the 2% daily target, we compute a scaling factor f = 0.02 / vol_proxy and multiply all weights by f. The scaling factor is also capped so that it doesn't push gross leverage above 1.5 — specifically, f = min(max_gross_leverage / gross, vol_target / vol_proxy). If the leverage cap binds (the portfolio is already at 1.5× leverage before vol targeting), the leverage cap takes precedence.


## Step 15: Position hysteresis

After computing the target weights, we check against the previous day's actual weights. If a position was entered fewer than 3 days ago (min_hold_days), we block two actions: (a) exiting the position entirely (target weight = 0 but previous weight ≠ 0), and (b) flipping direction (target and previous weights have opposite signs). In both cases, we keep the previous day's weight instead. This prevents whipsawing from short-lived z-score fluctuations and reduces turnover. After 3 days, the position is free to change as the target dictates.


## Step 16: Daily P&L and the cost model

For each name held in the portfolio, the gross P&L is:

    gross_pnl = weight × (−(residual_tomorrow − residual_today))

This is the idealized profit from the residual changing, as described in "The trade" above. The portfolio gross P&L is the sum across all names.

The cost for each name on each day is:

    cost = turnover × (estimated_unit_cost + 0.01 × |weight|)

where turnover = |weight_today − weight_yesterday|. There are two components:

The first component, turnover × estimated_unit_cost, represents execution cost — what you'd pay to enter or exit an option position. It includes a 2% per-side half-spread (approximating the option bid-ask), 0.5% per-side commission, and the adaptive penalties from Step 11 that inflate costs for illiquid or poorly-fit names. This component is only incurred when the position changes (turnover > 0).

The second component, turnover × 0.01 × |weight|, is labeled "gamma drag" and is meant to approximate the cost of delta-hedging. In a real implementation, holding a skew position through option spreads means holding options whose delta changes with the stock price. Rebalancing that delta daily costs approximately ½Γ(ΔS)² in expectation, which scales with position size and realized volatility. The backtest does not simulate this — it adds a flat 1% per unit of |weight| per unit of turnover as a cost proxy. This is charged only when the position changes, not as a daily running cost, which understates the true hedging drag for positions held over multiple days. For a 30% position entering from flat (turnover = 0.3), the gamma drag component is 0.3 × 0.01 × 0.3 = 0.0009.

The net P&L for each name is gross_pnl − cost. The portfolio net P&L sums across all names. The portfolio cumulative P&L is the running sum of daily net P&L.


## What the backtest does and does not simulate

The backtest simulates: (1) the signal — fitting SSVI surfaces, extracting skew, computing PCA residuals, z-scoring; (2) portfolio construction — candidate selection, risk scaling, hysteresis; (3) P&L from skew residual changes, treated as a directly tradeable quantity; and (4) a cost model with adaptive spread/commission and a gamma drag proxy.

The backtest does NOT simulate: (a) constructing an actual option portfolio (no specific strikes, expirations, or notionals are chosen), (b) delta-hedging (no stock trades, no gamma P&L from realized moves, no discrete hedging slippage), (c) theta decay (the option positions would bleed theta that partially offsets gamma gains — this is entirely absent), (d) vega risk from non-parallel implied vol shifts (the skew trade is not perfectly vega-neutral in practice), (e) rolling the option portfolio as options approach expiration, or (f) margin requirements and funding costs.

These omissions mean the backtest overstates the achievable P&L. The gross P&L line assumes perfect, costless translation between skew changes and portfolio value. The cost model partially compensates via the spread, commission, and gamma drag terms, but it cannot capture the path-dependent, nonlinear reality of maintaining a delta-hedged option book. A more realistic simulation would price specific option portfolios, mark them to market daily, simulate discrete delta-hedging against realized stock moves, and account for theta/gamma P&L and roll costs.


## What the strategy bets on

The strategy bets that the idiosyncratic component of implied volatility skew for individual financial stocks mean-reverts over roughly 5-day horizons. This is the residual after removing the common cross-sectional factors via PCA.

It does not intentionally bet on the direction of stock prices, the level of implied volatility, or the direction of sector-wide skew. The PCA regression removes the common factor from skew, and the long/short portfolio construction makes the net skew exposure roughly zero. However, because the backtest does not actually construct or hedge option positions, these neutrality properties are assumed rather than enforced. In a real implementation, achieving delta-neutrality and vega-neutrality would require careful option portfolio construction and daily rebalancing, each of which introduces additional costs and risks not captured here.

The economic rationale: demand for OTM puts on individual stocks fluctuates with stock-specific hedging needs — earnings events, corporate actions, concentrated holder activity, flow imbalances. These demand-driven skew dislocations are temporary. The factor model strips out sector-wide moves (like a broad risk-off event that reprices all financial skew simultaneously), leaving only the stock-specific transient component.
