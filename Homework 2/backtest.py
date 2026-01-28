import hashlib
import json
import os
import pickle
import warnings
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
import arch.unitroot
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import petname
import seaborn as sns
import statsmodels.api as sm
from IPython.display import display
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf
from arch.unitroot import ADF


def load_data(
    config: dict, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes the Nikkei 225 data.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters, containing NIKKEI_CSV_PATH, the filepath to the Nikkei 225 CSV file
    verbose: bool, optional
        Whether to print verbose output
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]
        Tuple containing the prices, returns, tickers, and metadata DataFrames
    """
    # Configure pandas display options (feel free to change these)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.max_rows', 20)
    pd.set_option('display.precision', 4)
 
    # Load the data
    prices = pd.read_csv(config['NIKKEI_CSV_PATH'], low_memory=False)
    if verbose:
        print("Price df shape at load:", prices.shape)
 
    # Slice the prices to only view data up to and including 2017
    first_2018_idx = prices[prices["Ticker"].str.contains("2018", na=False)].index[0].item()
    prices = prices.iloc[:first_2018_idx]
    if verbose:
        print("Price df shape after slicing time axis:", prices.shape)
 
    # Drop columns containing only NaNs (not considering the metadata rows)
    # These correspond to equities which only come into existence post-2017
    prices = prices.loc[:, prices.isna().sum() < prices.shape[0] - 3]
    if verbose:
        print("Price df shape after removing future asset columns:", prices.shape)
 
    # Extract the metadata: industrial classification, sector, and company names
    metadata = pd.DataFrame(prices.iloc[:3])
    metadata = metadata.T
    metadata.columns = metadata.iloc[0]
    metadata = metadata.iloc[1:]
    metadata.rename(columns={"Nikkei Industrial Classification": "Industry"}, inplace=True)
 
    # Drop the metadata rows and process date
    prices = prices.iloc[3:]
    prices.rename(columns={'Ticker':'Date'}, inplace=True)
    prices['Date'] = pd.to_datetime(prices['Date'])
    prices.set_index('Date', inplace=True, drop=True)
    prices = prices.astype(float)
    tickers = prices.columns
 
    # Calculate returns
    returns = prices.pct_change(fill_method=None)
    # Set initial return to zero
    returns.iloc[0] = 0
    
    if verbose:
        print("\nPrices head:")
        display(prices.head())
        print("\nMetadata head:")
        display(metadata.head())
 
        # Plot NaNs
        plt.imshow(prices.isna(), aspect='auto', cmap='viridis', interpolation=None)
        plt.xlabel('Stock')
        plt.ylabel('Date')
        plt.title('Missing Data in Nikkei 225 Prices')
        plt.grid(False)
        plt.show()
    
    return prices, returns, tickers, metadata


def plot_asset_with_max_return(returns, prices, max_rank=0):
    """
    Plots the asset with the highest return in the returns dataframe.
    
    Parameters:
    -----------
    returns: pd.DataFrame
        Dataframe of returns with tickers as columns and dates as index
    prices: pd.DataFrame
        Dataframe of prices with tickers as columns and dates as index
    max_rank: int
        Rank of the asset to plot
        
    Returns:
    --------
    None. Plots the asset with the highest return in the returns dataframe
    """
    # find asset index with the specified rank of return
    max_returns = pd.DataFrame({'max_return': returns.max(axis=1), 'idx': returns.idxmax(axis=1)}, columns=['max_return', 'idx'])
    asset_idx = max_returns.sort_values(by='max_return', ascending=False).iloc[max_rank]['idx']
    max_return_date = max_returns.sort_values(by='max_return', ascending=False).index[max_rank]

    # plot this asset
    prices[asset_idx].plot()
    # plot the date of the max return
    plt.axvline(max_return_date, color='r', linestyle='--')
    # Plot text of the price of the asset at the max return date
    # Add a text box with information on the side of the chart
    info_text = (
        f'Rank: {max_rank}\n'
        f'Ticker: {asset_idx}\n'
        f'Date: {max_return_date.strftime("%Y-%m-%d")}\n'
        f'Price t:    {prices[asset_idx][max_return_date]:.2f}\n'
    )
    
    try:
        # Find the index position of max_return_date in the prices dataframe
        date_idx = prices.index.get_loc(max_return_date)
        if date_idx > 0:  # Make sure there is a previous trading day
            prev_date = prices.index[date_idx - 1]
            info_text += f'Price t-1: {prices[asset_idx][prev_date]:.2f}\n'
        else:
            info_text += 'Price t-1: N/A\n'
    except:
        info_text += 'Price t-1: N/A\n'
        
    info_text += f'Return: {returns[asset_idx][max_return_date]:.1%}'
    
    # Position the text box outside the main plot area
    plt.figtext(0.85, 0.5, info_text, bbox=dict(facecolor='white', alpha=0.8), 
                verticalalignment='center', color='r')
    plt.axhline(0, color='r', linestyle='--')
    plt.show()


def compute_half_life(spread: pd.Series) -> int:
    """
    Computes the half-life of mean reversion for a spread using an
    error-correction (OU) model estimated by OLS.

    Parameters:
    -----------
    spread: pd.Series
        Time series of the spread (assumed to be mean-reverting, e.g. after
        cointegration/stationarity filtering).

    Returns:
    --------
    float
        Estimated half-life in number of periods (rounded to an integer and
        bounded below by 1).
    """

    Y = spread.diff().iloc[1:]
    X = spread.shift().iloc[1:].to_frame().assign(const=1)

    beta = np.linalg.lstsq(X, Y, rcond=None)[0][0]
    halflife = int(round(-np.log(2) / beta, 0))
    return max(halflife, 1)


def select_asset_universe(
    prices: pd.DataFrame, 
    returns: pd.DataFrame, 
    date: pd.Timestamp, 
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Index]:
    """
    Reduces the cross-section to only those stocks which were members of the asset universe at the given time
    with sufficient non-missing data and valid returns over the lookback period.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    returns: pd.DataFrame
        Dataframe of stock returns with dates as index and tickers as columns
    date: pd.Timestamp
        The reference date
    config: dict
        Configuration parameters, including 
        - FORMATION_PERIOD: int, the number of trading days to use for forming pairs, and thus also 
            for checking data availability, e.g. 252 days
        - FILTER_MAX_ABS_RETURN: float, the maximum absolute return allowed, e.g. 0.5
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Index]
        Tuple containing the selected historical prices, returns, and valid stocks
    """
    # Your code here
    lookback_period = config['FORMATION_PERIOD']
    prices, returns = prices.loc[prices.index < date, :], returns.loc[returns.index < date, :]
    prices, returns = prices.tail(lookback_period), returns.tail(lookback_period)

    # Remove all stocks with missing price data
    filter_tickers = ~prices.isna().any(axis=0)
    prices, returns = prices.loc[:, filter_tickers], returns.loc[:, filter_tickers]

    # Filter out stocks with too high return
    max_abs_return = config['FILTER_MAX_ABS_RETURN']
    filter_tickers = ~(abs(returns) > max_abs_return).any(axis=0)
    prices, returns = prices.loc[:, filter_tickers], returns.loc[:, filter_tickers]

    return (prices, returns, prices.columns)
    # e.g. return historical_prices[valid_stocks], historical_returns[valid_stocks], valid_stocks


def form_pairs(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Computes the distance between all pairs of stocks based on their normalized price series
    and returns information on the top `NUM_PAIRS` pairs with lowest distances.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    config: dict
        Configuration parameters, containing the key `NUM_PAIRS`, with a value such as 20
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with top `NUM_PAIRS` pairs sorted by distance
    """
    # Your code here
    allowed_distance_metrics = ['ssd', 'wsd']
    distance_metric = config.get('DISTANCE_METRIC')
    if distance_metric not in allowed_distance_metrics:
        raise ValueError('Distance metric not allowed')
    
    prices = prices / prices.iloc[0, :]
    returns = prices.pct_change().dropna()

    data = []
    already_present = []
    for company1 in prices.columns:
        for company2 in prices.columns:
            if company1 == company2 or (company2, company1) in already_present:
                continue
            if distance_metric == 'ssd':
                distance = ((prices.loc[:, company1] - prices.loc[:, company2])**2).sum()
            elif distance_metric == 'wsd':
                distance = stats.wasserstein_distance(returns.loc[:, company1], returns.loc[:, company2])
            data.append([company1, company2, distance])
            already_present.append((company1, company2))
            
    pairs = pd.DataFrame(data, columns=['stock1', 'stock2', 'distance'])
    pairs = pairs.sort_values(by='distance').head(config['NUM_PAIRS']).reset_index(drop=True)
    return pairs
    # e.g. return pairs_df


def estimate_hedge_ratio(prices: pd.DataFrame, pairs: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Estimates the betas and cointegrating relationships between the top K pairs and returns this information.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    pairs: pd.DataFrame
        Dataframe with pairs information from form_pairs()
        
    Returns:
    --------
    pd.DataFrame
        Dataframe with betas and cointegration information for each pair
    """
    # Your code here
    idxs, hedge_ratios, adf_stats, adf_pvalues, is_cointegrated = [], [], [], [], []
    returns = prices.pct_change().dropna()
    for idx, (s1, s2, _) in pairs.iterrows():
        s1_price, s2_price = prices.loc[:, s1], prices.loc[:, s2]
        if config['HEDGE_RATIO_METHOD'] == 'unit':
            hedge_ratio = 1.0
        elif config['HEDGE_RATIO_METHOD'] == 'ols':
            hedge_ratio = sm.OLS(s2_price, s1_price).fit().params.iloc[0]
        elif config['HEDGE_RATIO_METHOD'] == 'tls':
            r1, r2 = returns.loc[:, s1], returns.loc[:, s2]
            r1_var, r2_var, r1r2_cov = r1.var(), r2.var(), np.cov(r1, r2)[0,1]
            hedge_ratio = (r2_var - r1_var + np.sqrt((r2_var - r1_var)**2 + 4*r1r2_cov**2)) / (2*r1r2_cov)
        hedge_ratios.append(hedge_ratio)
        spread = s2_price - hedge_ratio*s1_price
        adf_test = ADF(spread)
        adf_stats.append(adf_test.stat)
        adf_pvalues.append(adf_test.pvalue)
        is_cointegrated.append(adf_test.pvalue < config['COINT_THRESHOLD'])
        idxs.append(idx) # not strictly necessary, but more robust to have the same idx since we merge on idx
    concat_df = pd.DataFrame(
        data=np.array([hedge_ratios, adf_stats, adf_pvalues, is_cointegrated]).T,
        columns=['hedge_ratio', 'adf_stat', 'adf_pvalue', 'is_cointegrated'],
        index = idxs
    )
    return_df = pd.merge(pairs, concat_df, left_index=True, right_index=True)
    return_df = return_df.loc[return_df['is_cointegrated'] == 1, :].reset_index(drop=True)
    return return_df
    # e.g. return results


def compute_signal(prices: pd.DataFrame, hedge_ratios: pd.DataFrame, config: dict) -> dict:
    """
    Computes trading signals for each spread based on z-score.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    hedge_ratios: pd.DataFrame
        Dataframe with hedge ratios and cointegration information from estimate_hedge_ratio
    config: dict
        Configuration parameters, containing: e.g. signal threshold Z_THRESHOLD, such as 2.0
        
    Returns:
    --------
    dict
        Dictionary with signals for each pair
    """
    # Your code here
    signals = {}
    for _, row in hedge_ratios.iterrows():
        s1 = row["stock1"]
        s2 = row["stock2"]
        hr = row["hedge_ratio"]

        price1 = prices.loc[:, s1]
        price2 = prices.loc[:, s2]

        spread = price2 - hr * price1

        window_size = config["ESTIMATION_PERIOD"]
        rolling_mean = spread.rolling(window = window_size).mean()
        rolling_stdev = spread.rolling(window = window_size).std()

        rolling_z = (spread - rolling_mean) / rolling_stdev

        norm_p1 = price1 / price1.iloc[0]
        norm_p2 = price2 / price2.iloc[0]

        signal = pd.Series(0, index = prices.index, dtype = int)
        trade = False
        entry_signal = 0
        p2_higher = None

        for t in range(len(prices)):
            z = rolling_z.iloc[t]

            if not trade:
                if pd.isna(z):
                    signal.iloc[t] = 0 
                elif z < -config["Z_THRESHOLD"]:
                    trade = True
                    entry_signal = 1
                    if norm_p2.iloc[t] > norm_p1.iloc[t]:
                        p2_higher = True
                    else:
                        p2_higher = False
                    signal.iloc[t] = 1
                elif z > config["Z_THRESHOLD"]:
                    trade = True
                    entry_signal = -1
                    if norm_p2.iloc[t] > norm_p1.iloc[t]:
                        p2_higher = True
                    else:
                        p2_higher = False
                    signal.iloc[t] = -1
                
                else:
                    signal.iloc[t] = 0
            
            else:
                signal.iloc[t] = entry_signal

                if p2_higher and norm_p2.iloc[t] <= norm_p1.iloc[t]:
                    trade = False
                    entry_signal = 0
                    p2_higher = None
                    signal.iloc[t] = 0
                elif not p2_higher and norm_p2.iloc[t] >= norm_p1.iloc[t]:
                    trade = False
                    entry_signal = 0
                    p2_higher = None
                    signal.iloc[t] = 0

        signals[f"{s1}_{s2}"] = {
            "stock1": s1,
            "stock2": s2,
            "hedge_ratio": hr,
            "signal": signal,
            "z_score": rolling_z
        }

    return signals

    # e.g. return signals


def allocate_positions(
    signals: dict,
    config: dict,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    portfolio_cash: float,
) -> dict:
    """
    Simple position sizing that allocates cash equally across all active pairs.
    
    Parameters:
    -----------
    signals: dict
        Dictionary with signal information for each pair
    config: dict
        Configuration parameters, containing MAX_LEVERAGE
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    date: pd.Timestamp
        Current date for price reference
    portfolio_cash: float
        Available cash in the portfolio
    
    Returns:
    --------
    dict
        Dictionary with position information for each stock
    """
    # Your code here
    positions = {}
    active_pairs = {}
    if date not in prices.index:
        return positions
    # if date == prices.index[0]:
    #     return positions # cannot make trade the first day as we rely on lagged signal
    if portfolio_cash == None or portfolio_cash <= 0:
        return positions

    for _, data in signals.items():
        s1, s2, hedge_ratio, signal_series = data['stock1'], data['stock2'], data['hedge_ratio'], data['signal']
        try: 
            p1, p2 = prices.loc[date, s1], prices.loc[date, s2]
        except KeyError:
            continue
        lagged_signal = signal_series.shift(1).loc[date]
        if np.isnan(lagged_signal) or lagged_signal == 0: continue
        active_pairs[f'{s1}_{s2}'] = {
            'stock1': s1,
            'stock2': s2,
            'price1': p1,
            'price2': p2,
            'hedge_ratio': hedge_ratio,
            'lagged_signal': lagged_signal
        }

    if len(active_pairs) == 0: return positions

    capital_per_pair = (portfolio_cash*config['MAX_LEVERAGE'])/len(active_pairs)
    for _, pair_data in active_pairs.items():
        s1, s2, p1, p2, hedge_ratio, lagged_signal = pair_data.values()
        if not positions.get(s1): positions[s1] = 0
        if not positions.get(s2): positions[s2] = 0
        notional_ratio = hedge_ratio*p2/p1
        shares1 = capital_per_pair/ (p1*(1 + notional_ratio))
        shares2 = shares1*hedge_ratio
        positions[s1] += -lagged_signal*shares1
        positions[s2] += lagged_signal*shares2

    return positions

    # e.g. return positions


def update_portfolio(
    current_portfolio: dict,
    target_positions: dict,
    prices: pd.DataFrame,
    date: pd.Timestamp,
    config: dict,
) -> dict:
    """
    Updates the portfolio based on target positions, accounting for transaction costs.
    
    Parameters:
    -----------
    current_portfolio: dict
        Current portfolio positions and cash
    target_positions: dict
        Target positions for each stock, in number of shares
    prices: pd.DataFrame
        Dataframe of stock prices with dates as index and tickers as columns
    date: pd.Timestamp
        Current date
    config: dict
        Configuration parameters
        
    Returns:
    --------
    dict
        Dictionary with updated portfolio and performance metrics
        - portfolio: Updated portfolio
        - transaction_cost: Transaction costs
        - turnover: Turnover rate
        - portfolio_value_before_costs: Portfolio value before costs
        - portfolio_value_after_txn_costs: Portfolio value after txn costs
        - portfolio_value_after_costs: Portfolio value after all costs
        - long_exposure: Long exposure
        - short_exposure: Short exposure
        - leverage: Leverage ratio (total absolute exposure / portfolio value)
        - active_positions: Number of active positions
        - max_position_size: Maximum position size as percentage of portfolio
    """
    # Initialize new positions dictionary
    new_positions = {}
    new_cash = current_portfolio.get('cash', config['INITIAL_CASH'])
    
    # Get current prices
    current_prices = prices.loc[date]
    
    # Calculate transaction costs
    total_transaction_cost_pct = 0
    total_transaction_cost_amount = 0
    total_turnover = 0
    
    # Update positions for each stock
    for stock, target_position in target_positions.items():
        # Get target position for current date
        target_shares = target_position  # Now we expect target_position to be a simple number
        
        # Get current position (default to 0 if not already in portfolio)
        current_shares = current_portfolio.get(stock, 0)
        
        # Calculate the change in position
        shares_diff = target_shares - current_shares
        
        # Calculate the transaction cost if there's a change in position
        if shares_diff != 0:
            price = current_prices[stock]
            
            # Calculate the value of the transaction
            transaction_value = abs(shares_diff * price)
            
            # Calculate the transaction cost using specified percentage
            transaction_cost = transaction_value * config['TRANSACTION_COST']
            
            # Add to total transaction costs
            total_transaction_cost_pct += transaction_cost / transaction_value if transaction_value > 0 else 0
            total_transaction_cost_amount += transaction_cost
            
            # Add to total turnover
            total_turnover += transaction_value
            
            # Calculate the cash flow from the transaction
            cash_flow = -shares_diff * price
            
            # Update cash balance
            new_cash += cash_flow
        
        # Update the position if it's non-zero
        if target_shares != 0:
            new_positions[stock] = target_shares
    
    # Remove stocks with zero positions
    new_positions = {k: v for k, v in new_positions.items() if v != 0}
    
    # Calculate portfolio value before costs
    positions_value = (pd.Series(new_positions) * current_prices).sum()
    portfolio_value_before_costs = positions_value + new_cash
    
    # Calculate portfolio value after txn costs
    portfolio_value_after_txn_costs = portfolio_value_before_costs - total_transaction_cost_amount
    
    # Ensure that total portfolio value is positive
    # if portfolio_value_after_txn_costs <= 0:
    #     raise ValueError(f"Portfolio value after costs is negative or zero: {portfolio_value_after_txn_costs}")
    
    # Calculate exposures
    long_positions = {k: v for k, v in new_positions.items() if v > 0}
    short_positions = {k: v for k, v in new_positions.items() if v < 0}
    
    long_exposure = sum(v * current_prices[k] for k, v in long_positions.items())
    short_exposure = abs(sum(v * current_prices[k] for k, v in short_positions.items()))
    
    # Calculate leverage as total exposure relative to portfolio value
    leverage = (long_exposure + short_exposure) / portfolio_value_after_txn_costs
    
    # Calculate borrowing costs based on the current leverage
    leverage_cost = 0
    margin_cost = 0
    
    if leverage > 1.0:
        # Calculate the cost of using leverage
        leverage_cost = (leverage - 1.0) * portfolio_value_after_txn_costs * config['LEVERAGE_RATE_DAILY']
        
        # If we have excess long or short exposure, apply margin cost to the excess exposure
        excess_long_exposure = max(0, long_exposure - portfolio_value_after_txn_costs)
        excess_short_exposure = max(0, short_exposure - portfolio_value_after_txn_costs)
        margin_cost = (excess_long_exposure + excess_short_exposure) * config['MARGIN_RATE_DAILY']
    
    # Calculate borrowing cost based on short positions
    borrowing_cost = short_exposure * config['BORROW_RATE_DAILY']
    
    # Calculate total financing costs
    total_financing_cost = leverage_cost + margin_cost + borrowing_cost
    
    # Calculate total cost (transaction + financing)
    total_cost = total_transaction_cost_amount + total_financing_cost
    
    # Calculate portfolio value after all costs
    portfolio_value_after_costs = portfolio_value_after_txn_costs - total_financing_cost
    
    # Update cash balance after financing and transaction costs
    new_cash = new_cash - total_financing_cost - total_transaction_cost_amount
    
    # Create the new portfolio
    new_portfolio = new_positions.copy()
    new_portfolio['cash'] = new_cash
    
    # Calculate portfolio statistics
    active_positions = len(new_positions)
    max_position_size = 0
    if portfolio_value_after_costs > 0 and active_positions > 0:
        position_sizes = [(abs(pos * current_prices[stock]) / portfolio_value_after_costs) for stock, pos in new_positions.items()]
        max_position_size = max(position_sizes) if position_sizes else 0
    
    # Calculate turnover as a percentage of portfolio value
    turnover_pct = total_turnover / portfolio_value_after_costs if portfolio_value_after_costs > 0 else 0
    
    return {
        'portfolio': new_portfolio,
        'transaction_cost': total_transaction_cost_amount,
        'turnover': turnover_pct,
        'portfolio_value_before_costs': portfolio_value_before_costs,
        'portfolio_value_after_txn_costs': portfolio_value_after_txn_costs,
        'portfolio_value_after_costs': portfolio_value_after_costs,
        'leverage': leverage,
        'long_exposure': long_exposure,
        'short_exposure': short_exposure,
        'active_positions': active_positions,
        'max_position_size': max_position_size,
        'borrowing_cost': borrowing_cost,
        'leverage_cost': leverage_cost,
        'margin_cost': margin_cost,
        'total_financing_cost': total_financing_cost,
        'total_cost': total_cost
    }


def implement_strategy(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    config: dict,
) -> tuple:
    """
    Implements the pairs trading strategy over the entire dataset.
    
    Parameters:
    -----------
    prices: pd.DataFrame
        Dataframe of stock prices
    config: dict
        Configuration parameters
        
    Returns:
    --------
    tuple
        Portfolio history, metrics dataframe, and pairs dataframe
    """
    # Get unique dates
    trading_dates = prices.index.unique()
    
    # Initialize variables
    portfolio_history = {}
    metrics_data = []
    pairs_history = []
    current_portfolio = {'cash': config['INITIAL_CASH']}  # Initialize with just our initial cash
    prev_portfolio_value = None
    position_timestamps = {}  # Tracks when each position was entered
    
    # Split the data into formation and trading periods
    for i in range(0, len(trading_dates) - config['FORMATION_PERIOD'] - config['TRADING_PERIOD'], config['TRADING_PERIOD']):
        # Define formation and trading periods
        formation_start = trading_dates[i]
        formation_end = trading_dates[i + config['FORMATION_PERIOD'] - 1]
        trading_start = trading_dates[i + config['FORMATION_PERIOD']]
        trading_end = trading_dates[i + config['FORMATION_PERIOD'] + config['TRADING_PERIOD'] - 1]
        
        print("Formation period: %s to %s" % (formation_start, formation_end))
        print("Trading period: %s to %s" % (trading_start, trading_end))
        
        # Select stocks for the formation period
        formation_stocks, _, _ = select_asset_universe(prices, returns, formation_end, config)
        
        if formation_stocks.empty or formation_stocks.shape[1] < 2:
            print("Not enough stocks with complete data in the formation period. Skipping.")
            continue
        
        # Compute distances between pairs
        pairs = form_pairs(formation_stocks, config)
        
        # Estimate hedge ratios
        hedge_ratios = estimate_hedge_ratio(formation_stocks, pairs, config)
        
        if hedge_ratios.empty:
            print("No pairs found. Skipping.")
            continue
        else: 
            print(f"Pairs found: {len(hedge_ratios)}")
            
            for _, pair_row in hedge_ratios.iterrows():
                pairs_history.append({
                    'formation_start': formation_start,
                    'formation_end': formation_end,
                    'trading_start': trading_start,
                    'trading_end': trading_end,
                    'pair': f"{pair_row['stock1']}_{pair_row['stock2']}",
                    'stock1': pair_row['stock1'],
                    'stock2': pair_row['stock2'],
                    'distance': pair_row['distance'],
                    'hedge_ratio': pair_row['hedge_ratio'],
                    'half_life': pair_row.get('half_life', None),
                    'adf_pvalue': pair_row['adf_pvalue']
                })
        
        # Get trading period data INCLUDING the last ESTIMATION_PERIOD days from formation period
        # This allows us to calculate signals immediately at the start of the trading period
        estimation_start_idx = max(0, i + config['FORMATION_PERIOD'] - config['ESTIMATION_PERIOD'])
        estimation_start = trading_dates[estimation_start_idx]
        
        # Get combined data: last ESTIMATION_PERIOD days from formation + regular trading period
        trading_data = prices.loc[estimation_start:trading_end]
        
        print(f"Using data from {estimation_start} to {trading_end} for signal calculation")
        print(f"Actual trading period: {trading_start} to {trading_end}")
        
        # Compute signals for the trading period (including estimation lookback)
        signals = compute_signal(trading_data, hedge_ratios, config)
        
        # Reset position timestamps for new trading period
        active_position_timestamps = {}
        
        # NOTE: Even though we include formation data for signal calculation,
        # we still only start trading at the proper trading_start date
        trading_dates_only = trading_data.loc[trading_start:trading_end].index
        
        # Compute positions for the first trading day
        positions = allocate_positions(
            signals,
            config,
            trading_data,
            trading_dates_only[0],
            current_portfolio.get('cash')
        )
        
        # Update portfolio for each day in the trading period (not including estimation data)
        for date in trading_dates_only:
            # Recompute positions for each day to account for lagged signals and leverage constraints
            # Only do this after the first day, since we already did it for the first day
            if date != trading_dates_only[0]:
                positions = allocate_positions(
                    signals,
                    config,
                    trading_data,
                    date,
                    current_portfolio.get('cash')
                )
            
            # Update portfolio
            portfolio_update = update_portfolio(
                current_portfolio,
                positions,
                trading_data,
                date,
                config
            )
            
            # Extract values from the portfolio update
            current_portfolio = portfolio_update['portfolio']
            portfolio_value_before_costs = portfolio_update['portfolio_value_before_costs']
            portfolio_value_after_costs = portfolio_update['portfolio_value_after_costs']
            
            # Store portfolio state
            portfolio_history[date] = current_portfolio.copy()
            
            # Calculate return including transaction costs and borrowing costs
            if prev_portfolio_value is not None:
                # Calculate gross return (before any costs)
                gross_return = (portfolio_value_before_costs - prev_portfolio_value) / prev_portfolio_value
                
                # Calculate net return (after all costs)
                # This uses portfolio_value_after_costs which already has transaction and borrowing costs deducted
                period_return = (portfolio_value_after_costs - prev_portfolio_value) / prev_portfolio_value
            else:
                period_return = 0.0
                gross_return = 0.0
            
            # Store all metrics in a single data structure
            metrics_data.append({
                'date': date,
                'portfolio_value': portfolio_value_after_costs,
                'gross_portfolio_value': portfolio_value_before_costs,
                'cash': current_portfolio['cash'],
                'net_returns': period_return,
                'gross_returns': gross_return,
                'transaction_cost': portfolio_update['transaction_cost'],
                'borrowing_cost': portfolio_update['borrowing_cost'],
                'leverage_cost': portfolio_update['leverage_cost'],
                'margin_cost': portfolio_update['margin_cost'],
                'total_cost': portfolio_update['total_cost'],
                'total_financing_cost': portfolio_update['total_financing_cost'],
                'leverage': portfolio_update['leverage'],
                'long_exposure': portfolio_update['long_exposure'],
                'short_exposure': portfolio_update['short_exposure'],
                'num_stocks': portfolio_update['active_positions'],
                'max_position_pct': portfolio_update['max_position_size'],
                'turnover': portfolio_update['turnover'],
            })
            
            prev_portfolio_value = portfolio_value_after_costs
        
        # Merge the position timestamps from this trading period into the global set
        position_timestamps.update(active_position_timestamps)
    
    # Convert all metrics to a single DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    if not metrics_df.empty:
        metrics_df.set_index('date', inplace=True)
    
    # Convert pairs history to DataFrame
    pairs_df = pd.DataFrame(pairs_history)
    
    return (
        portfolio_history, 
        metrics_df,
        pairs_df
    )


def analyze_performance(
    config: dict,
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    tickers: list,
    metadata: pd.DataFrame,
    portfolio_history: dict, 
    metrics_df: pd.DataFrame, 
    pairs_df: pd.DataFrame,
) -> dict:
    """
    Analyzes the performance of the strategy.
    
    Parameters:
    -----------
    config: dict
        Configuration dictionary
    prices: pd.DataFrame
        Dataframe with stock prices
    returns: pd.DataFrame
        Dataframe with stock returns
    tickers: list
        List of stock tickers
    metadata: pd.DataFrame
        Dataframe with stock metadata
    portfolio_history: dict
        Dictionary with portfolio history
    metrics_df: pd.DataFrame
        Dataframe with all metrics (returns, costs, exposures, etc.)
    pairs_df: pd.DataFrame
        Pairs information
        
    Returns:
    --------
    dict
        Dictionary with performance metrics
    """
    # Initialize variables
    initial_cash = config['INITIAL_CASH']
    num_pairs = config['NUM_PAIRS']
    transaction_cost = config['TRANSACTION_COST']
    borrow_rate_daily = config['BORROW_RATE_DAILY']
    metrics = {}
    
    # Calculate basic metrics
    net_returns = metrics_df['net_returns']
    metrics['Net Total Return'] = (1 + net_returns).prod() - 1
    metrics['Net Annualized Return'] = (1 + metrics['Net Total Return']) ** (252 / len(net_returns)) - 1
    metrics['Net Volatility'] = net_returns.std() * np.sqrt(252)
    metrics['Net Sharpe Ratio'] = metrics['Net Annualized Return'] / metrics['Net Volatility'] if metrics['Net Volatility'] > 0 else 0
    metrics['Gross Total Return'] = (1 + metrics_df['gross_returns']).prod() - 1
    metrics['Gross Annualized Return'] = (1 + metrics['Gross Total Return']) ** (252 / len(metrics_df['gross_returns'])) - 1
    metrics['Gross Volatility'] = metrics_df['gross_returns'].std() * np.sqrt(252)
    metrics['Gross Sharpe Ratio'] = metrics['Gross Annualized Return'] / metrics['Gross Volatility'] if metrics['Gross Volatility'] > 0 else 0
    
    # Calculate drawdown
    cum_returns = (1 + net_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    metrics['Net Max Drawdown'] = drawdown.min()
    metrics['Net Calmar Ratio'] = metrics['Net Annualized Return'] / abs(metrics['Net Max Drawdown'])
    gross_cum_returns = (1 + metrics_df['gross_returns']).cumprod()
    running_max = gross_cum_returns.cummax()
    drawdown = (gross_cum_returns / running_max) - 1
    metrics['Gross Max Drawdown'] = drawdown.min()
    metrics['Gross Calmar Ratio'] = metrics['Gross Annualized Return'] / abs(metrics['Gross Max Drawdown'])
        
    # Calculate win rate
    win_days = (net_returns > 0).sum()
    total_days = len(net_returns)
    metrics['Net Win Rate'] = win_days / total_days if total_days > 0 else 0
    gross_win_days = (metrics_df['gross_returns'] > 0).sum()
    metrics['Gross Win Rate'] = gross_win_days / total_days if total_days > 0 else 0
    
    # Calculate skewness and kurtosis
    metrics['Net Skewness'] = net_returns.skew()
    metrics['Net Kurtosis'] = net_returns.kurtosis()
    metrics['Gross Skewness'] = metrics_df['gross_returns'].skew()
    metrics['Gross Kurtosis'] = metrics_df['gross_returns'].kurtosis()
        
    # Visualize cumulative returns
    plt.figure(figsize=(12, 6))
        
    # Plot net returns
    ((1 + metrics_df['net_returns']).cumprod() - 1).plot(label='Net Returns')
    
    # Plot gross returns if available
    ((1 + metrics_df['gross_returns']).cumprod() - 1).plot(label='Gross Returns')
    
    # TODO: Plot benchmark returns if available
        
    plt.title('Cumulative Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.PercentFormatter(xmax=1))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize leverage if available
    plt.figure(figsize=(12, 6))
    metrics_df['leverage'].plot(label='Leverage')
    plt.title('Portfolio Leverage Over Time')
    plt.xlabel('Date')
    plt.ylabel('Leverage Ratio')
    plt.axhline(y=1, color='r', linestyle='--', label='Leverage = 1')
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Plot exposure
    plt.figure(figsize=(12, 6))
    metrics_df['long_exposure'].plot(label='Long Exposure', color='green')
    (-metrics_df['short_exposure']).plot(label='Short Exposure', color='red')
    plt.title('Long and Short Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Exposure')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Plot exposure
    plt.figure(figsize=(12, 6))
    (metrics_df['long_exposure'] + metrics_df['short_exposure']).plot(label='Total Exposure', color='black', alpha=0.7)
    (metrics_df['long_exposure'] - metrics_df['short_exposure']).plot(label='Net Exposure', color='purple', alpha=0.7)
    plt.title('Net and Total Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Exposure')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.legend()
    plt.grid(True)
    plt.show()
        
    # Plot portfolio value if available
    plt.figure(figsize=(12, 6))
    metrics_df['portfolio_value'].plot(label='Portfolio Value', color='blue')
    plt.title('Total Portfolio Value Over Time (Positions + Cash)')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value')
    plt.grid(True)
    plt.show()
    
    # Visualize cash balance if available
    plt.figure(figsize=(12, 6))
    metrics_df['cash'].plot(label='Cash Balance', color='green')
    plt.title('Cash Balance Over Time')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.xlabel('Date')
    plt.ylabel('Cash Amount')
    plt.grid(True)
    plt.show()
        
    # Plot cash as percentage of portfolio value
    cash_percentage = metrics_df['cash'] / metrics_df['portfolio_value'] * 100
    plt.figure(figsize=(12, 6))
    cash_percentage.plot(label='Cash as % of Portfolio', color='blue')
    plt.title('Cash as Percentage of Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.axhline(y=100, color='r', linestyle='--', label='100% Cash')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Visualize number of stocks in portfolio if available
    plt.figure(figsize=(12, 6))
    metrics_df['num_stocks'].plot(label='Number of Stocks', color='blue')
    plt.title('Number of Stocks in Portfolio Over Time')
    plt.xlabel('Date')
    plt.ylabel('Number of Stocks')
    plt.grid(True)
    plt.show()
    
    # Visualize maximum position size as percentage of portfolio value
    plt.figure(figsize=(12, 6))
    metrics_df['max_position_pct'].plot(label='Max Position Size (%)', color='orange')
    plt.title('Maximum Position Size as % of Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Percentage (%)')
    plt.grid(True)
    plt.show()
    
    # Visualize turnover
    plt.figure(figsize=(12, 6))
    metrics_df['turnover'].plot(label='Turnover', color='purple')
    plt.title('Portfolio Turnover Over Time')
    plt.xlabel('Date')
    plt.ylabel('Turnover')
    plt.ylim(0, min(10, metrics_df['turnover'].max()))
    plt.grid(True)
    plt.show()
    
    # Analyze pairs data if available
    if pairs_df is not None:
        # Count occurrences of each pair
        pair_counts = pairs_df['pair'].value_counts()
        
        # Plot top 10 most frequent pairs
        plt.figure(figsize=(12, 6))
        pair_counts.head(10).plot(kind='bar')
        plt.title('Top 10 Most Frequent Pairs')
        plt.xlabel('Pair')
        plt.ylabel('Frequency')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        
        # Display the metadata and the of the top 10 pairs
        top_pairs_info = pair_counts.head(10).to_frame()
        top_pairs_info['ticker1'] = top_pairs_info.index.to_series().str.split('_').str[0]
        top_pairs_info['ticker2'] = top_pairs_info.index.to_series().str.split('_').str[1]
        top_pairs_info = pd.merge(top_pairs_info, metadata, left_on='ticker1', right_index=True, how='left')
        top_pairs_info = pd.merge(top_pairs_info, metadata, left_on='ticker2', right_index=True, how='left')
        top_pairs_info['ticker1_name'] = top_pairs_info['Company_x']
        top_pairs_info['ticker2_name'] = top_pairs_info['Company_y']
        top_pairs_info['ticker1_sector'] = top_pairs_info['Sector_x']
        top_pairs_info['ticker2_sector'] = top_pairs_info['Sector_y']
        top_pairs_info['ticker1_industry'] = top_pairs_info['Industry_x']
        top_pairs_info['ticker2_industry'] = top_pairs_info['Industry_y']
        
        top_pairs_info = top_pairs_info[[
            'ticker1', 'ticker1_name', 'ticker1_sector', 'ticker1_industry', 
            'ticker2', 'ticker2_name', 'ticker2_sector', 'ticker2_industry', 
            'count',
        ]]
        
        print("\nTop 10 Most Frequent Pairs:")
        with pd.option_context('display.max_columns', None, 'display.max_rows', None):
            display(top_pairs_info)
        
        # Count hedge ratios
        def categorize_hedge_ratio(x):
            normalized = x if x >= 1 else 1/x
            if normalized < 10:
                return f"{int(normalized)}-{int(normalized)+1}"
            else:
                return "and 10+"
            
        categorized_ratios = pairs_df['hedge_ratio'].apply(categorize_hedge_ratio)
        plt.figure(figsize=(12, 6))
        ratio_counts = categorized_ratios.value_counts().sort_index()
        ratio_counts.plot(kind='bar')
        plt.title('Distribution of Hedge Ratios')
        plt.xlabel('Hedge Ratio')
        plt.ylabel('Frequency')
        plt.xticks(rotation=0)
        plt.grid(True)
        plt.show()

        # Add pair statistics to metrics
        metrics['Total Unique Pairs'] = len(pair_counts)
        metrics['Most Frequent Pair'] = pair_counts.index[0] if len(pair_counts) > 0 else "None"
        metrics['Avg Hedge Ratio'] = pairs_df['hedge_ratio'].mean()

        # Print summary of pairs
        print("\nPairs Trading Summary:")
        print(f"Total unique pairs: {len(pair_counts)}")
        print(f"Average hedge ratio: {pairs_df['hedge_ratio'].mean():.2f}")

    # Analyze costs if available
    # Initialize cost variables
    total_transaction_costs = 0
    avg_transaction_cost_per_day = 0
    transaction_cost_total_impact = 0
    transaction_cost_annualized_impact = 0
    
    total_borrowing_costs = 0
    avg_borrowing_cost_per_day = 0
    borrowing_cost_total_impact = 0
    borrowing_cost_annualized_impact = 0
    
    # Calculate transaction cost metrics if available
    if 'transaction_cost' in metrics_df.columns and not metrics_df['transaction_cost'].isna().all():
        # Handle NaN values safely
        clean_tc = metrics_df['transaction_cost'].fillna(0)
        total_transaction_costs = clean_tc.sum()
        avg_transaction_cost_per_day = total_transaction_costs / len(metrics_df) if len(metrics_df) > 0 else 0
        
        # Calculate impact on returns
        if 'gross_returns' in metrics_df.columns:
            # Use safe calculation methods to avoid overflow
            try:
                # Calculate gross return
                gross_returns = metrics_df['gross_returns'].fillna(0)
                gross_return = (1 + gross_returns).prod() - 1
                
                # Convert transaction costs to percentage of portfolio value
                portfolio_values = metrics_df['portfolio_value'].fillna(0)
                tc_pct = clean_tc / portfolio_values.shift(1)
                tc_pct = tc_pct.fillna(0)
                
                # Calculate net return after transaction costs
                net_returns_after_tc = gross_returns - tc_pct
                net_return_after_tc = (1 + net_returns_after_tc).prod() - 1
                
                # Calculate impact
                transaction_cost_total_impact = gross_return - net_return_after_tc
                
                # Annualized impact - only calculate if we have reasonable values
                if np.isfinite(gross_return) and np.isfinite(net_return_after_tc) and len(metrics_df) > 0:
                    gross_annual = (1 + gross_return) ** (252 / len(metrics_df)) - 1
                    net_annual_after_tc = (1 + net_return_after_tc) ** (252 / len(metrics_df)) - 1
                    transaction_cost_annualized_impact = gross_annual - net_annual_after_tc
            except Exception as e:
                print(f"Warning: Error calculating transaction cost impact: {e}")
                transaction_cost_total_impact = 0
                transaction_cost_annualized_impact = 0        
    
    # Calculate borrowing cost metrics if available
    if 'borrowing_cost' in metrics_df.columns and not metrics_df['borrowing_cost'].isna().all():
        # Handle NaN values safely
        clean_bc = metrics_df['borrowing_cost'].fillna(0)
        total_borrowing_costs = clean_bc.sum()
        avg_borrowing_cost_per_day = total_borrowing_costs / len(metrics_df) if len(metrics_df) > 0 else 0
        
        # Calculate impact on returns
        if 'gross_returns' in metrics_df.columns and 'transaction_cost' in metrics_df.columns:
            try:
                # Use the already cleaned values from above
                gross_returns = metrics_df['gross_returns'].fillna(0)
                portfolio_values = metrics_df['portfolio_value'].fillna(0)
                
                # Convert transaction costs to percentage of portfolio value
                tc_pct = clean_tc / portfolio_values.shift(1)
                tc_pct = tc_pct.fillna(0)
                
                # Convert borrowing costs to percentage of portfolio value
                bc_pct = clean_bc / portfolio_values.shift(1)
                bc_pct = bc_pct.fillna(0)
                
                # Calculate net return after transaction costs
                net_returns_after_tc = gross_returns - tc_pct
                net_return_after_tc = (1 + net_returns_after_tc).prod() - 1
                
                # Calculate net return after transaction costs and borrowing costs
                net_returns_after_tc_bc = gross_returns - tc_pct - bc_pct
                net_return_after_tc_bc = (1 + net_returns_after_tc_bc).prod() - 1
                
                # Calculate impact
                borrowing_cost_total_impact = net_return_after_tc - net_return_after_tc_bc
                
                # Annualized impact - only calculate if we have reasonable values
                if np.isfinite(net_return_after_tc) and np.isfinite(net_return_after_tc_bc) and len(metrics_df) > 0:
                    net_annual_after_tc = (1 + net_return_after_tc) ** (252 / len(metrics_df)) - 1
                    net_annual_after_tc_bc = (1 + net_return_after_tc_bc) ** (252 / len(metrics_df)) - 1
                    borrowing_cost_annualized_impact = net_annual_after_tc - net_annual_after_tc_bc
            except Exception as e:
                print(f"Warning: Error calculating borrowing cost impact: {e}")
                borrowing_cost_total_impact = 0
                borrowing_cost_annualized_impact = 0
        
    if 'leverage_cost' in metrics_df.columns and not metrics_df['leverage_cost'].isna().all():
        clean_lc = metrics_df['leverage_cost'].fillna(0)  
        total_leverage_cost = clean_lc.sum()
        avg_leverage_cost_per_day = total_leverage_cost / len(metrics_df) if len(metrics_df) > 0 else 0
        
    if 'margin_cost' in metrics_df.columns and not metrics_df['margin_cost'].isna().all():
        clean_mc = metrics_df['margin_cost'].fillna(0)
        total_margin_cost = clean_mc.sum()
        avg_margin_cost_per_day = total_margin_cost / len(metrics_df) if len(metrics_df) > 0 else 0
        
        
    # Calculate/plot total costs
    # Plot costs
    plt.figure(figsize=(12, 6))
    clean_tc.cumsum().plot(label='Cumulative Transaction Costs') if 'transaction_cost' in metrics_df.columns and not metrics_df['transaction_cost'].isna().all() else None
    clean_bc.cumsum().plot(label='Cumulative Borrowing Costs') if 'borrowing_cost' in metrics_df.columns and not metrics_df['borrowing_cost'].isna().all() else None
    clean_lc.cumsum().plot(label='Cumulative Leverage Costs') if 'leverage_cost' in metrics_df.columns and not metrics_df['leverage_cost'].isna().all() else None
    clean_mc.cumsum().plot(label='Cumulative Margin Costs') if 'margin_cost' in metrics_df.columns and not metrics_df['margin_cost'].isna().all() else None
        
    if (
        'transaction_cost' in metrics_df.columns and not metrics_df['transaction_cost'].isna().all()
        and 'borrowing_cost' in metrics_df.columns and not metrics_df['borrowing_cost'].isna().all()
        and 'leverage_cost' in metrics_df.columns and not metrics_df['leverage_cost'].isna().all()
        and 'margin_cost' in metrics_df.columns and not metrics_df['margin_cost'].isna().all()
    ):
        # Create a total cost series
        total_costs = pd.DataFrame({
                'transaction_cost': clean_tc,
                'borrowing_cost': clean_bc,
                'leverage_cost': clean_lc,
                'margin_cost': clean_mc
            }).sum(axis=1)
        total_costs.cumsum().plot(label='Cumulative Total Costs', linewidth=2)
    
    plt.title('Cumulative Costs Over Time')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Cost')
    plt.gca().yaxis.set_major_formatter(mpl_ticker.FuncFormatter(lambda x, _: f'${x:,.0f}'))
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Add leverage metrics if available
    if 'leverage' in metrics_df.columns:
        metrics['Average Leverage'] = metrics_df['leverage'].mean()
        metrics['Max Leverage'] = metrics_df['leverage'].max()
        metrics['Min Leverage'] = metrics_df['leverage'].min()
    
    # Add profit
    metrics['Total Gross Profit'] = initial_cash * ((1 + metrics_df['gross_returns']).prod() - 1)
    metrics['Total Net Profit'] = initial_cash * ((1 + metrics_df['net_returns']).prod() - 1)
    
    # Add transaction cost metrics
    metrics['Total Transaction Costs'] = total_transaction_costs
    metrics['Avg Transaction Cost/Day'] = avg_transaction_cost_per_day
    metrics['Total Transaction Cost Impact'] = transaction_cost_total_impact
    metrics['Annual Transaction Cost Impact'] = transaction_cost_annualized_impact
    
    # Add borrowing cost metrics
    metrics['Total Borrowing Costs'] = total_borrowing_costs
    metrics['Avg Borrowing Cost/Day'] = avg_borrowing_cost_per_day
    metrics['Total Borrowing Cost Impact'] = borrowing_cost_total_impact
    metrics['Annual Borrowing Cost Impact'] = borrowing_cost_annualized_impact
    
    # Add margin and leverage cost metrics
    metrics['Total Leverage Costs'] = total_leverage_cost
    metrics['Avg Leverage Cost/Day'] = avg_leverage_cost_per_day
    metrics['Total Margin Costs'] = total_margin_cost
    metrics['Avg Margin Cost/Day'] = avg_margin_cost_per_day
    
    # Combined cost metrics
    metrics['Total Costs'] = total_transaction_costs + total_borrowing_costs + total_leverage_cost + total_margin_cost
    metrics['Annual Txn+Borrow Cost Impact'] = transaction_cost_annualized_impact + borrowing_cost_annualized_impact
    
    # Basic monthly returns heatmap (using net returns)
    try:
        monthly_returns = metrics_df['net_returns'].resample('ME').apply(lambda x: (1 + x).prod() - 1)
        
        # Create a DataFrame with the right structure for pivot
        monthly_df = pd.DataFrame({
            'month': monthly_returns.index.month,
            'year': monthly_returns.index.year,
            'returns': monthly_returns.values
        })
        
        # Now create the pivot table
        monthly_table = pd.pivot_table(
            data=monthly_df,
            index='month',
            columns='year',
            values='returns'
        )
        
        # Plot monthly returns as a heatmap
        plt.figure(figsize=(14, 8))
        sns.heatmap(monthly_table.T, annot=True, cmap='RdYlGn', center=0, fmt='.2%')
        plt.title('Monthly Returns Heatmap')
        plt.xlabel('Month')
        plt.ylabel('Year')
        plt.show()
        
        # Annual returns
        annual_returns = metrics_df['net_returns'].resample('YE').apply(lambda x: (1 + x).prod() - 1)
        # Remove time part of date (otherwise it looks bad)
        annual_returns.index = annual_returns.index.year
        
        # Plot annual returns bar chart
        plt.figure(figsize=(12, 6))
        annual_returns.plot(kind='bar', color=annual_returns.map(lambda x: 'green' if x > 0 else 'red'))
        plt.title('Annual Returns')
        plt.xlabel('Year')
        plt.ylabel('Return')
        plt.xticks(rotation=45)
        plt.grid(True, axis='y')
        plt.axhline(0, color='black', linewidth=1)
        
        # Add value labels on top of bars
        for i, v in enumerate(annual_returns):
            # Position labels based on the range of values in the data
            value_range = annual_returns.max() - annual_returns.min()
            label_position = v + np.sign(v) * value_range * (0.05 if v < 0 else 0.02)
            plt.text(i, label_position, f'{v:.2%}', ha='center')
            
        plt.show()
        
    except Exception as e:
        print(f"Could not generate monthly/annual return tables: {e}")
        raise e
    
    # TODO: Add more metrics/tests from quantstats and Gatev et al. 2006
    
    # Display performance metrics as a table
    summary_metrics_df = pd.DataFrame({
        'Metric': list(metrics.keys()),
        'Value': list(metrics.values())
    })
    summary_metrics_df['Formatted Value'] = summary_metrics_df.apply(
        lambda row: (
            f'{row["Value"]:.2%}' if any(x in row["Metric"] for x in ["Return", "Volatility", "Drawdown", "Win Rate"]) else
            f'{row["Value"]:.2f}' if any(x in row["Metric"] for x in ["Sharpe Ratio", "Calmar Ratio", "Skewness", "Kurtosis", "Hedge Ratio"]) or (row["Metric"] == "Leverage") else
            f'${row["Value"]:,.2f}' if any(x in row["Metric"] for x in ["Profit", "Total Transaction Costs", "Avg Transaction Cost/Day", "Total Borrowing Costs", "Avg Borrowing Cost/Day", "Total Costs", "Total Leverage Costs", "Total Margin Costs", "Avg Leverage Cost/Day", "Avg Margin Cost/Day"]) else
            f'{row["Value"]:.2%}' if any(x in row["Metric"] for x in ["Transaction Cost Impact", "Annual Transaction Cost Impact", "Borrowing Cost Impact", "Annual Borrowing Cost Impact", "Annual Cost Impact"]) else
            f'{int(row["Value"])}' if any(x in row["Metric"] for x in ["Pairs"]) else
            f'{row["Value"]:.2%}' if isinstance(row["Value"], (int, float)) and abs(row["Value"]) <= 1 else
            f'{row["Value"]:.2f}' if isinstance(row["Value"], (int, float)) else str(row["Value"])
        ), axis=1
    )
    
    try:
        with pd.option_context('display.max_columns', None, 'display.max_rows', None):
            display(summary_metrics_df[['Metric', 'Value', 'Formatted Value']])
    except Exception as e:
        print(f"Could not display summary metrics table: {e}")
        print(summary_metrics_df[['Metric', 'Value', 'Formatted Value']])
        raise e
    
    return metrics


def save_results(
    unique_id: str,
    config: dict,
    portfolio_history: dict,
    metrics_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
) -> None:
    """
    Saves all results with the unique identifier.
    
    Parameters:
    -----------
    unique_id: str
        Unique identifier for the results
    config: dict
        Configuration parameters
    portfolio_history: dict
        Portfolio history
    metrics_df: pd.DataFrame
        Dataframe with all metrics (returns, costs, exposures, etc.)
    pairs_df: pd.DataFrame
        Pairs information
        
    Returns:
    --------
    None
    """    
    # Create results directory if it doesn't exist
    os.makedirs('results', exist_ok=True)
    
    # Save metrics dataframe
    metrics_df.to_csv(f'results/{unique_id}_metrics.csv')
    
    # Save pairs dataframe
    pairs_df.to_csv(f'results/{unique_id}_pairs.csv')
    
    # Save portfolio history and config
    import pickle
    with open(f'results/{unique_id}_portfolio_history.pkl', 'wb') as f:
        pickle.dump(portfolio_history, f)
    with open(f'results/{unique_id}_config.json', 'w') as f:
        json.dump(config, f)
    
    print(f"Results saved with ID: {unique_id}")


def get_unique_id(config: dict) -> str:
    """
    Generates a unique identifier from the config.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters
        
    Returns:
    --------
    str
        Unique identifier
    """
    # Normalize the config (convert all numbers to the same format); 
    # Could perhaps use json.dumps(config) instead after sorting
    normalized_config = {}
    for k, v in config.items():
        if isinstance(v, (int, float)):
            normalized_config[k] = float(v)
        elif isinstance(v, (list, tuple)):
            for i, item in enumerate(v):
                if not isinstance(item, (int, float, str, bool)):
                    raise ValueError("Unsupported type for config value: "
                                     f"'{type(item)}'; value: {item}; parent: {k}")
                v[i] = float(item) if isinstance(item, int) else item
            normalized_config[k] = tuple(v)
        elif isinstance(v, set):
            for item in v:
                if not isinstance(item, (int, float, str, bool)):
                    raise ValueError("Unsupported type for config value: "
                                     f"'{type(item)}'; value: {item}; parent: {k}")
                if isinstance(item, int):
                    v.remove(item)
                    v.add(float(item))
            normalized_config[k] = tuple(v)
        elif isinstance(v, dict):
            for k, vv in v.items():
                if not isinstance(vv, (int, float, str, bool)):
                    raise ValueError("Unsupported type for config value: "
                                     f"'{type(vv)}'; value: {vv}; parent: {k}")
                v[k] = float(vv) if isinstance(vv, int) else vv
            normalized_config[k] = tuple(v.items())
        elif isinstance(v, (bool, str)):
            normalized_config[k] = v
        else:
            raise ValueError("Unsupported type for config value: "
                             f"'{type(v)}'; value: {v}")
    
    # Sort config        
    sorted_config = sorted(normalized_config.items(), key=lambda item: item[0])
    config_hash = hashlib.sha256(str(sorted_config).encode()).hexdigest()
    
    # Break hash into three numbers
    hash_len = len(config_hash)
    substring_len = hash_len // 3
    lookups = [
        int(config_hash[0:substring_len], 16),
        int(config_hash[substring_len:2*substring_len], 16),
        int(config_hash[2*substring_len:], 16)
    ]
    
    # Generate petname
    tables = [
        petname.english.adverbs,
        petname.english.adjectives,
        petname.english.names,
    ]
    part_ids = [
        tables[i][lookup % len(tables[i])] for i, lookup in enumerate(lookups)
    ]
    unique_id = '-'.join(part_ids)
    
    print(f"Generated unique ID: {unique_id}")
    
    return unique_id


def load_results(unique_id: str):
    if os.path.exists(f'results/{unique_id}_metrics.csv'):
        print(f"Results already exist for '{unique_id}'; loading")
        # load results
        with open(f'results/{unique_id}_portfolio_history.pkl', 'rb') as f:
            portfolio_history = pickle.load(f)
        # load CSVs
        metrics_df = pd.read_csv(f'results/{unique_id}_metrics.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        pairs_df = pd.read_csv(f'results/{unique_id}_pairs.csv', index_col=0, parse_dates=True, date_format='%Y-%m-%d')
        return (
            portfolio_history,
            metrics_df,
            pairs_df
        )
    else:
        return None 


def validate_config(config: dict) -> None:
    """
    Validates the configuration parameters.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters
    
    Returns:
    --------
    None
    """
    assert config['NIKKEI_CSV_PATH'], "Nikkei CSV path must be specified"
    assert config['FILTER_MAX_ABS_RETURN'] >= 0, "Filter max absolute return must be non-negative"
    assert config['FORMATION_PERIOD'] > 0, "Formation period must be positive"
    assert config['TRADING_PERIOD'] > 0, "Trading period must be positive"
    assert config['NUM_PAIRS'] > 0, "Number of pairs must be positive"
    assert config['DISTANCE_METRIC'] in ['ssd'], "Distance metric must be 'ssd'"
    assert config['HEDGE_RATIO_METHOD'] in ['ols', 'unit'], "Hedge ratio method must be 'ols' or 'unit'"
    assert 0 < config['COINT_THRESHOLD'] < 1, "Cointegration threshold must be in (0, 1)"
    assert config['ESTIMATION_PERIOD'] > 0, "Estimation period must be positive"
    assert config['ESTIMATION_PERIOD'] < config['TRADING_PERIOD'], "Estimation period must fit inside trading period"
    assert config['Z_THRESHOLD'] > 0, "Z-score threshold must be positive"
    assert config['TRANSACTION_COST'] >= 0, "Transaction cost must be non-negative"
    assert config['BORROW_RATE_DAILY'] >= 0, "Borrowing rate must be non-negative"
    assert config['LEVERAGE_RATE_DAILY'] >= 0, "Leverage rate must be non-negative"
    assert config['MARGIN_RATE_DAILY'] >= 0, "Margin rate must be non-negative"
    assert config['INITIAL_CASH'] > 0, "Initial cash must be positive"
    assert config['MAX_LEVERAGE'] > 0, "Max leverage must be positive"
    # assert config['CAPITAL_PER_PAIR'] > 0, "Capital per pair must be positive"
    # assert config['CLOSE_OLD_POSITIONS'] in [True, False], "Close old positions must be True or False"


def run_backtest(config=None):
    """
    Runs the full strategy and analyzes performance.
    """
    # Parameters
    config = config or {
        # Save results
        'SAVE_RESULTS': True,
        # Path to the Nikkei 225 CSV file
        'NIKKEI_CSV_PATH': 'N225.csv',
        # The maximum absolute return allowed, e.g. 0.5
        'FILTER_MAX_ABS_RETURN': 0.5,
        # Length of formation period, in trading days; also used as period to select asset universe based on data availability and returns
        'FORMATION_PERIOD': 252,
        # Length of trading period, in trading days
        'TRADING_PERIOD': 126,
        # Top number of pairs to select for trading
        'NUM_PAIRS': 50,
        # Distance metric to use for selecting pairs
        'DISTANCE_METRIC': 'ssd',
        # Method to use for hedge ratio estimation ("ols" or "unit" or your own)
        'HEDGE_RATIO_METHOD': 'unit',
        # Cointegration p-value threshold for choosing pairs (set to 0.05 for 5% significance level)
        'COINT_THRESHOLD': 0.05,
        # Length of rolling mean/volatility estimation period for spreads, in trading days
        'ESTIMATION_PERIOD': 20,
        # Z-score threshold for generating signals
        'Z_THRESHOLD': 2.0,
        # Transaction cost as a fraction of trade value in percentage (0.0001 = 1 bps)
        'TRANSACTION_COST': 0.0005,
        # Daily borrowing cost for leverage in percentage (0.0001 = 1 bps); multiply by 252 to convert to annual rate
        'BORROW_RATE_DAILY': 0.03 / 252,
        # Leverage rate daily on balanced leverage in percentage (0.0001 = 1 bps); multiply by 252 to convert to annual rate
        'LEVERAGE_RATE_DAILY': 0.01 / 252,
        # Margin rate daily on unbalanced (excess) leverage in percentage (0.0001 = 1 bps); multiply by 252 to convert to annual rate
        'MARGIN_RATE_DAILY': 0.05 / 252,
        # Initial cash amount
        'INITIAL_CASH': 1_000_000.0,
        # Maximum leverage to target
        'MAX_LEVERAGE': 0.05,
    }
    
    print(f"Starting from '{os.getcwd()}'")
    
    # Validate configuration
    validate_config(config)
    
    # Generate unique identifier for config
    unique_id = get_unique_id(config)
    
    # Load the data
    prices, returns, tickers, metadata = load_data(config)
    
    # Load results if extant, otherwise run strategy
    results = load_results(unique_id)
    if not results:
        results = implement_strategy(
            prices,
            returns,
            config,
        )
        if config['SAVE_RESULTS']:
            save_results(
                unique_id,
                config,
                *results,
            )
    
    # Analyze performance and plot/print results
    performance_metrics = analyze_performance(
        config,
        prices, 
        returns, 
        tickers, 
        metadata,
        *results,
    )
        
    return (
        prices,
        returns,
        tickers,
        metadata,
        *results
    )


if __name__ == "__main__":  
    run_backtest()