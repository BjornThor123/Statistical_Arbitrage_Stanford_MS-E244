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
from IPython.display import display
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


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
 
    # Slice the prices to only view data up to and including the year we want to end at
    first_post_end_year_idx = prices[prices["Ticker"].str.contains(str(config['END_YEAR'] + 1), na=False)].index[0].item()
    prices = prices.iloc[:first_post_end_year_idx]
    if verbose:
        print("Price df shape after slicing time axis:", prices.shape)
 
    # Drop columns containing only NaNs (not considering the metadata rows)
    # These correspond to equities which only come into existence post-end year
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
        plt.xlabel('Stock Index')
        plt.ylabel('Date')
        plt.yticks(np.arange(len(prices.index))[::252], prices.index.strftime('%Y')[::252])
        plt.title('Missing Data in Nikkei 225 Prices')
        plt.grid(False)
        plt.show()
    
    return prices, returns, tickers, metadata


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
        The reference date to select the asset universe (i.e. the day on which we want to form the universe)
    config: dict
        Configuration parameters, including 
        - LOOKBACK_PERIOD: int, the number of trading days to use for checking data availability, e.g. 252 days
        - FILTER_MAX_ABS_RETURN: float, the maximum absolute return allowed, e.g. 0.5
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.Index]
        Tuple containing the selected historical prices, returns, and valid stocks
    """
    # Define the lookback period
    # Get the exact date that is lookback_period trading days before the reference date
    all_dates = prices.index.sort_values()
    date_idx = all_dates.get_loc(date)
    if date_idx < config['LOOKBACK_PERIOD']:
        # Not enough history available
        return pd.DataFrame(), pd.DataFrame(), pd.Index([])
    start_date = all_dates[date_idx - config['LOOKBACK_PERIOD']]
    
    # Filter the prices dataframe for the lookback period
    # Drop the last day to avoid look ahead bias
    historical_prices = prices.loc[start_date:date].iloc[:-1]
    
    # Filter stocks that have complete price data and valid returns in the lookback period 
    # Drop the last day to avoid look ahead bias
    historical_returns = returns.loc[start_date:date].iloc[:-1]
    
    # Create masks for both conditions
    complete_data_mask = historical_prices.notna().all()
    valid_returns_mask = historical_returns.abs().max() <= config['FILTER_MAX_ABS_RETURN']
    
    # Find stocks that satisfy both conditions
    valid_stocks = historical_prices.columns[complete_data_mask & valid_returns_mask]
    
    return historical_prices[valid_stocks], historical_returns[valid_stocks], valid_stocks