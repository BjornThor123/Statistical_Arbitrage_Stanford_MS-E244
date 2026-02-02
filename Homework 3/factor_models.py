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
import statsmodels.api as sm
from sklearn.decomposition import PCA

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


class NikkeiSectorFactorModel():
    def __init__(
        self,
        metadata: pd.DataFrame,
        intercept: bool = False,
    ):
        """
        Initialize the Nikkei sector factor model.
        
        Parameters:
        - metadata (pd.DataFrame): An Nx3 matrix containing the metadata
          with columns [Industry, Sector, Company].
        - intercept (bool): Perform the estimation with an intercept term.
        """
        self.metadata = metadata
        self.intercept = intercept
        self.is_fit = False
        
        self.sectors = metadata['Sector'].unique()
        self.sectors.sort()
        self.sector_counts = self.metadata['Sector'].value_counts().sort_index()
        self.num_factors = len(self.sectors)
        
        self.factor_weights = None
        self.betas = None
        self.rhat = None
        self.comp_mtx = None
        self.residuals = None
        self.alphas = None
        assert(self.num_factors > 0)

    def __compute_factors(self, returns):
        # We compute factors using the stored weights (S matrix)
        # Note: self.factor_weights must be set in fit() using the training universe
        if self.factor_weights is None:
             raise Exception("Model not fit yet")
        
        # F = R @ S
        factors = returns.values @ self.factor_weights
        return pd.DataFrame(factors, index=returns.index, columns=self.sectors)
        
        
    def fit(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit the factor model to the given returns data and return the 
        residuals, estimated returns, and composition matrix.
        
        Parameters:
        - returns (pd.DataFrame): A TxN matrix containing the returns data.

        Returns:
        - (pd.DataFrame) a TxN matrix of residuals with proper index/columns.
        - (pd.DataFrame) a TxN matrix of estimated returns with proper index/columns.
        - (pd.DataFrame) an NxN composition matrix with proper index/columns.
        """
        self.returns = returns
        self.T, self.N = returns.shape
        assert(self.num_factors < self.N)
        assert(self.N < self.T)

        # 1. Compute Sector Weights Matrix (S)
        # Map tickers to sectors
        sector_series = self.metadata.loc[returns.columns]['Sector']
        dummies = pd.get_dummies(sector_series)
        # Ensure all sectors are present as columns (sorted order)
        dummies = dummies.reindex(columns=self.sectors, fill_value=0)
        
        # Normalize columns to sum to 1 (calculating mean return of sector)
        col_sums = dummies.sum(axis=0)
        col_sums[col_sums == 0] = 1.0 # Avoid division by zero
        # S matrix (N x K)
        self.factor_weights = (dummies / col_sums).values

        # 2. Compute factors
        # F = R @ S
        factors_val = returns.values @ self.factor_weights
        factors_df = pd.DataFrame(factors_val, index=returns.index, columns=self.sectors)

        # 3. Estimate Betas
        if self.intercept:
            factors_with_const = sm.add_constant(factors_df)
        else:
            factors_with_const = factors_df

        # Run the vectorized regression
        model = sm.OLS(returns, factors_with_const).fit()
        
        if self.intercept:
            self.betas = model.params.iloc[1:, :]
            self.alphas = model.params.iloc[0, :]
        else:
            self.betas = model.params
            self.alphas = pd.Series(0, index=returns.columns)

        # 4. Compute residuals and estimated returns
        self.rhat = model.fittedvalues
        
        # Critical: If we want comp_mtx to reconstruct residuals, we must define residuals as R - F @ Beta
        # If intercept is present, model.resid = R - F @ Beta - Alpha
        # Traded residuals portfolio usually includes Alpha component if it's mean-reverting
        # Here we follow the definition: epsilon = R - F @ Beta (includes alpha)
        # But wait, typically epsilon is expected to be mean zero.
        # But comp_mtx corresponds to traded portfolio P = I - W Beta.
        # Returns of P = R - R W Beta = R - F Beta = Alpha + epsilon_OLS.
        # The user's error check likely compares R @ comp_mtx with (R - F @ Beta).
        # We will return the manual residual calculation to be consistent with comp_mtx.
        
        predicted_no_alpha = factors_val @ self.betas.values
        self.residuals = returns - predicted_no_alpha
        # Note: self.residuals will contain alpha if self.intercept is True.

        # 5. Composition matrix
        # Phi = I - S @ Beta
        # This assumes F = R @ S is essentially the factor definition used.
        # Check: F = R @ S. 
        # R_approx = F @ Beta = R @ S @ Beta.
        # Residuals = R - R @ S @ Beta = R @ (I - S @ Beta).
        # So Phi = I - S @ Beta.
        self.comp_mtx_df = np.eye(self.N) - self.factor_weights @ self.betas.values
            
        self.is_fit = True
        return (self.residuals, self.rhat, self.comp_mtx_df)
    
    def predict(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate the out-of-sample residuals and factors using from the 
        estimated factor model given the new returns.
        
        Parameters:
        - returns (pd.DataFrame): A T2xN matrix containing the new returns data.

        Returns:
        - (pd.DataFrame) a T2xN matrix of residuals with proper index/columns.
        - (pd.DataFrame) an T2xK matrix of factors with proper index/columns.
        """
        if not self.is_fit:
            raise Exception("Must call fit() on model first.")

        # Compute factors using stored weights
        # sectors = self.__compute_factors(returns) -> this uses self.factor_weights now
        factors = returns.values @ self.factor_weights
        factors_df = pd.DataFrame(factors, index=returns.index, columns=self.sectors)
        
        # Predict
        if self.intercept:
            predictions = factors @ self.betas.values + self.alphas.values
        else:
            predictions = factors @ self.betas.values
        
        residuals_df = returns - predictions
        
        return (residuals_df, factors_df)


class NikkeiPCAFactorModel():
    def __init__(
        self,
        num_factors: int = 6, 
        intercept: bool = False,
    ):
        """
        Initialize the Nikkei PCA factor model with the given parameters.
        
        Parameters:
        - num_factors (int): The number of factors to estimate.
        - intercept (bool): Perform the estimation with an intercept term.
        """
        assert(num_factors > 0)
        self.num_factors = num_factors
        self.intercept = intercept
        self.is_fit = False
        
        self.factor_weights = None 
        self.factors = None
        self.betas = None
        self.alphas = None
        self.rhat = None
        self.comp_mtx = None
        self.residuals = None
    
    def fit(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit the factor model to the given returns data and return the 
        residuals, estimated returns, and composition matrix.
        
        Parameters:
        - returns (pd.DataFrame): A TxN matrix containing the returns data.

        Returns:
        - (pd.DataFrame) a TxN matrix of residuals with proper index/columns.
        - (pd.DataFrame) a TxN matrix of estimated returns with proper index/columns.
        - (pd.DataFrame) an NxN composition matrix with proper index/columns.
        """
        self.returns = returns
        self.T, self.N = returns.shape
        assert(self.num_factors < self.N)
        assert(self.N < self.T)
        
        # 1. Compute PCA
        # Use sklearn PCA
        pca = PCA(n_components=self.num_factors)
        pca.fit(returns)
        
        # 2. Compute Factor Weights
        # components_ is (K, N). Transpose to (N, K)
        self.factor_weights = pca.components_.T
        
        # 3. Compute Factors
        # Explicit projection of Returns onto Weights: F = R @ W
        # Note: default PCA transform does F = (R - mu) @ W.
        # We want F = R @ W to maintain R Phi = Epsilon identity easily if intercept is OFF.
        # If intercept is ON, it absorbs the mean shift.
        # But manual projection is safest for consistency with R @ W.
        factors_val = returns.values @ self.factor_weights
        self.factors = pd.DataFrame(factors_val, index=returns.index)
        
        # 4. Estimate Betas (Time-series regression)
        if self.intercept:
            factors_with_const = sm.add_constant(self.factors)
        else:
            factors_with_const = self.factors
            
        model = sm.OLS(returns, factors_with_const).fit()
        
        if self.intercept:
            self.betas = model.params.iloc[1:, :]
            self.alphas = model.params.iloc[0, :]
        else:
            self.betas = model.params
            self.alphas = pd.Series(0, index=returns.columns)
            
        # 5. Compute residuals and estimated returns
        self.rhat = model.fittedvalues
        
        # Manual residuals calculation to be consistent with composition matrix
        # Epsilon = R - F @ Beta (includes alpha if any)
        predicted_no_alpha = factors_val @ self.betas.values
        self.residuals = returns - predicted_no_alpha

        # 6. Compute composition matrix
        # Phi = I - W @ Beta
        self.comp_mtx = np.eye(self.N) - self.factor_weights @ self.betas.values
        
        self.is_fit = True
        return (self.residuals, self.rhat, self.comp_mtx)
    
    def predict(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate the out-of-sample residuals and factors using from the 
        estimated factor model given the new returns.
        
        Parameters:
        - returns (pd.DataFrame): A T2xN matrix containing the new returns data.

        Returns:
        - (pd.DataFrame) a T2xN matrix of residuals with proper index/columns.
        - (pd.DataFrame) an T2xK matrix of factors with proper index/columns.
        """
        if not self.is_fit:
            raise Exception("Must call fit() on model first.")
        
        # Compute factors using stored weights
        # F = R @ W
        factors = returns.values @ self.factor_weights
        factors_df = pd.DataFrame(factors, index=returns.index)
        
        # Compute predicted returns
        if self.intercept:
            predictions = factors @ self.betas.values + self.alphas.values
        else:
            predictions = factors @ self.betas.values
            
        residuals_df = returns - predictions
        
        return (residuals_df, factors_df)