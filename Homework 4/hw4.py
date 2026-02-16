import hashlib
import json
import os
import pickle
import warnings
from abc import abstractmethod, ABC
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

import arch.unitroot
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib.ticker as mpl_ticker
import numpy as np
import pandas as pd
import petname
import seaborn as sns
import scipy as sp
from IPython.display import display
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf


class FactorModel(ABC):
    def __init__(
        self,
        intercept: bool = False,
    ):
        self.intercept = intercept
        self.is_fit = False

    @abstractmethod
    def fit(self, returns: pd.DataFrame):
        raise NotImplementedError

    @abstractmethod
    def predict(self, returns: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class NikkeiSectorFactorModel(FactorModel):
    def __init__(
        self,
        metadata: pd.DataFrame,
        intercept: bool = False,
    ):
        """
        Initialize the Nikkei sector factor model.
        
        Parameters:
        -----------
        metadata (pd.DataFrame): An Nx3 matrix containing the metadata
          with columns [Industry, Sector, Company].
        intercept (bool): Perform the estimation with an intercept term.
        """
        self.metadata = metadata
        self.intercept = intercept
        self.is_fit = False
        
        self.sectors = metadata['Sector'].unique()
        self.sectors.sort()
        self.sector_counts = self.metadata['Sector'].value_counts().sort_index()
        self.num_factors = len(self.sectors)
        assert(self.num_factors > 0)
        
    def fit(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit the factor model to the given returns data and return the residuals and
        composition matrix.
        
        Parameters:
        -----------
        returns: pd.DataFrame
            A TxN matrix containing the returns data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - residuals: (pd.DataFrame) a TxN matrix of residuals with the proper index and columns.
            - rhat: (pd.DataFrame) a TxN matrix of estimated returns with the proper index and columns.
            - comp_mtx: (pd.DataFrame) an NxN composition matrix with the proper index and columns.
        """
        self.returns = returns
        self.T, self.N = returns.shape

        self.sector_returns = self.returns.T.groupby(self.metadata['Sector']).mean().T
        self.sector_returns = self.sector_returns.fillna(0)
        self.factor_weights = pd.DataFrame(
            np.zeros((returns.shape[1], self.num_factors)),
            index=returns.columns,
            columns=self.sectors
        )
        for sector in self.sectors:
            self.factor_weights[sector] = (self.metadata['Sector'] == sector)
        self.factor_weights = self.factor_weights / self.factor_weights.sum(axis=0)
        assert np.allclose(self.returns @ self.factor_weights, self.sector_returns)
        self.factors = self.sector_returns
        
        self.factors = np.dot(self.returns, self.factor_weights)
        if self.intercept:
            self.factors = np.column_stack([np.ones(self.factors.shape[0]), self.factors])
        
        self.betas = np.linalg.lstsq(self.factors, self.returns.values, rcond=None)[0]
        
        if self.intercept:
            self.alphas = self.betas[0:1,:]
            self.factors = self.factors[:,1:]
            self.betas = self.betas[1:,:]
        self.rhat = np.dot(self.factors, self.betas)
        self.residuals = self.returns.values - self.rhat
        self.comp_mtx = np.eye(self.N) - np.dot(self.factor_weights, self.betas)  # returns @ comp_mtx = residuals
        self.r_squared = 1 - np.var(self.residuals, axis=0) / np.var(self.returns.values, axis=0)
        self.rmse = np.sqrt(np.mean(self.residuals**2, axis=0))
        residual_cols = [f"Residual{c}" for c in self.returns.columns]

        residuals_df = pd.DataFrame(self.residuals, index=self.returns.index, columns=self.returns.columns)
        rhat_df = pd.DataFrame(self.rhat, index=self.returns.index, columns=self.returns.columns)
        comp_mtx_df = pd.DataFrame(self.comp_mtx, index=self.returns.columns, columns=residual_cols)
        
        self.is_fit = True
        return (residuals_df, rhat_df, comp_mtx_df)
    
    def predict(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate the out-of-sample residuals and factors using from the estimated 
        factor model given the new returns.
        
        Parameters:
        -----------
        returns: pd.DataFrame
            A T2xN matrix containing the new returns data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]
            A tuple containing:
            - residuals: (pd.DataFrame) a T2xN matrix of residuals with the proper index and columns.
            - factors: (pd.DataFrame) a T2xK matrix of factors with the proper index and columns.
        """
        if not self.is_fit:
            raise Exception("Must call fit() on model first.")
        
        new_factors = np.dot(returns, self.factor_weights)
        if self.intercept:
            new_factors = np.column_stack([np.ones(new_factors.shape[0]), new_factors])
        
        if self.intercept:
            parameters = np.concatenate((self.alphas, self.betas), axis=0)
            new_rhat = np.dot(new_factors, parameters)
        else:
            new_rhat = np.dot(new_factors, self.betas)
        
        new_residuals = returns.values - new_rhat
        
        residuals_df = pd.DataFrame(new_residuals, index=returns.index, columns=returns.columns)
        if self.intercept:
            new_factors = new_factors[:,1:]
        factors_df = pd.DataFrame(new_factors, index=returns.index, columns=[f"Factor{i}" for i in range(self.num_factors)])
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
        -----------
        num_factors: int
            The number of factors to estimate.
        intercept: bool
            Perform the estimation with an intercept term.
        """
        assert(num_factors > 0)
        self.num_factors = num_factors
        self.intercept = intercept
        self.is_fit = False
        
    def fit(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Fit the factor model to the given returns data and return the 
        residuals, estimated returns, and composition matrix.
        
        Parameters:
        -----------
        returns: pd.DataFrame
            A TxN matrix containing the returns data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            A tuple containing:
            - pd.DataFrame: a TxN matrix of residuals with proper index/columns.
            - pd.DataFrame: a TxN matrix of estimated returns with proper index/columns.
            - pd.DataFrame: an NxN composition matrix with proper index/columns.
        """
        self.returns = returns
        self.T, self.N = returns.shape
        assert(self.num_factors < self.N)

        # Use correlation matrix for better accuracy
        Cov = returns.corr()
        Lambda, V = sp.linalg.eigh(Cov, subset_by_index=[self.N-self.num_factors, self.N-1])
        self.factor_weights = V[:,::-1]
        self.normalized_factor_weights = self.factor_weights / np.abs(self.factor_weights).sum(axis=0)
        self.factors = np.dot(self.returns, self.factor_weights)
        self.normalized_factors = np.dot(self.returns, self.normalized_factor_weights)
        if self.intercept:
            self.factors = np.column_stack([np.ones(self.factors.shape[0]), self.factors])
        self.betas = np.linalg.lstsq(self.factors, self.returns.values, rcond=None)[0]
        if self.intercept:
            self.alphas = self.betas[0:1,:]
            self.factors = self.factors[:,1:]
            self.betas = self.betas[1:,:]
        self.rhat = np.dot(self.factors, self.betas)
        self.residuals = self.returns.values - self.rhat
        self.comp_mtx = np.eye(self.N) - np.dot(self.factor_weights, self.betas)  # returns @ comp_mtx = residuals
        self.r_squared = 1 - np.var(self.residuals, axis=0) / np.var(self.returns.values, axis=0)
        self.rmse = np.sqrt(np.mean(self.residuals**2, axis=0))
        residuals_df = pd.DataFrame(self.residuals, index=self.returns.index, columns=self.returns.columns)
        rhat_df = pd.DataFrame(self.rhat, index=self.returns.index, columns=self.returns.columns)
        comp_mtx_df = pd.DataFrame(self.comp_mtx, index=self.returns.columns, columns=self.returns.columns)
        
        self.is_fit = True
        return (residuals_df, rhat_df, comp_mtx_df)
    
    def predict(self, returns: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Estimate the out-of-sample residuals and factors using from the 
        estimated factor model given the new returns.
        
        Parameters:
        -----------
        returns: pd.DataFrame
            A T2xN matrix containing the new returns data.

        Returns:
        --------
        Tuple[pd.DataFrame, pd.DataFrame]:
            A tuple containing:
            - pd.DataFrame: a T2xN matrix of residuals with proper index/columns.
            - pd.DataFrame: an T2xK matrix of factors with proper index/columns.
        """
        if not self.is_fit:
            raise Exception("Must call fit() on model first.")
        
        new_factors = np.dot(returns, self.factor_weights)
        if self.intercept:
            new_factors = np.column_stack([np.ones(new_factors.shape[0]), new_factors])
        
        if self.intercept:
            parameters = np.concatenate((self.alphas, self.betas), axis=0)
            new_rhat = np.dot(new_factors, parameters)
        else:
            new_rhat = np.dot(new_factors, self.betas)
        
        new_residuals = returns.values - new_rhat
        
        residuals_df = pd.DataFrame(new_residuals, index=returns.index, columns=returns.columns)
        if self.intercept:
            new_factors = new_factors[:,1:]
        factors_df = pd.DataFrame(new_factors, index=returns.index, columns=[f"Factor{i}" for i in range(self.num_factors)])
        return (residuals_df, factors_df)


def load_data(
    config: dict, verbose: bool = False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads and processes the Nikkei 225 data.
    
    Parameters:
    -----------
    config: dict
        Configuration parameters, containing:
        - NIKKEI_CSV_PATH, the filepath to the Nikkei 225 CSV file
        - END_YEAR, the year to end at (inclusive)
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


def estimate_oos_residuals(returns: pd.DataFrame, prices: pd.DataFrame, metadata: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Estimates all out-of-sample residuals of the factor model and returns this information.
    
    Parameters:
    -----------
    reutrns: pd.DataFrame
        Dataframe of returns with dates as index and tickers as columns
    prices: pd.DataFrame
        Dataframe of prices with dates as index and tickers as columns
    metadata: pd.DataFrame
        Dataframe of metadata with columns [Industry, Sector, Company]
    config: dict
        Configuration parameters, containing:
        - FACTOR_MODEL: str, the factor model to use
        - N_FACTORS: int, the number of factors to use (if applicable)
        - RESIDUAL_ESTIMATION_LOOKBACK_DAYS: int, the number of days to use for residual estimation
        - FACTOR_ESTIMATION_FREQUENCY_DAYS: int, the frequency of factor estimation
        - USE_INTERCEPT: bool, whether to use an intercept in the factor model
        
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp], pd.DataFrame]
        A tuple containing:
        - residual_returns: pd.DataFrame, a DataFrame of residual returns
        - composition_matrices: pd.DataFrame, a DataFrame of composition matrices
        - estimation_dates: List[pd.Timestamp], a list of estimation dates
        - alphas: pd.DataFrame, a DataFrame of alphas (if applicable), otherwise None
    """
    T = len(returns)
    N = len(returns.columns)

    assert(config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS'] > 0)
    assert(config['FACTOR_ESTIMATION_FREQUENCY_DAYS'] > 0)
    assert(config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS'] % config['FACTOR_ESTIMATION_FREQUENCY_DAYS'] == 0)
    assert(config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS'] < T)
    assert(config['FACTOR_ESTIMATION_FREQUENCY_DAYS'] < T)
    assert(config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS'] > config['FACTOR_ESTIMATION_FREQUENCY_DAYS'])

    estimation_dates = []
    residual_prices = []
    residual_returns = []
    composition_matrices = []
    if config['USE_INTERCEPT']:
        alphas = []
    oos_rmses = []
    oos_stds = []

    for t in range(config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS'], T):    
        if t % config['FACTOR_ESTIMATION_FREQUENCY_DAYS'] == 0 or t == config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS']:
            estimation_dates.append(returns.index[t])
            if config.get('VERBOSE', False):
                print(f"Estimating residuals at time {t} (date {returns.index[t]})...")
            if config['FACTOR_MODEL'] == 'sector':
                model = NikkeiSectorFactorModel(metadata, intercept=config['USE_INTERCEPT'])
            elif config['FACTOR_MODEL'] == 'pca':
                model = NikkeiPCAFactorModel(num_factors=config['N_FACTORS'], intercept=config['USE_INTERCEPT'])
            else:
                raise ValueError(f"Invalid FACTOR_MODEL '{config['FACTOR_MODEL']}' specified in config")
        
            valid_assets = returns.iloc[t-config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS']:t].columns
            if config.get('VERBOSE', False):
                print(f"Number of valid assets: {len(valid_assets)}")
            _, _, comp_mtx_t = model.fit(returns[valid_assets].iloc[t-config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS']:t])
            if config.get('VERBOSE', False):
                print(
                    f"In-sample min/mean/median/max R^2: "
                    f"{model.r_squared.min():0.4f}, "
                    f"{model.r_squared.mean():0.4f}, "
                    f"{np.median(model.r_squared):0.4f}, "
                    f"{model.r_squared.max():0.4f}"
                )
                print(
                    f"In-sample min/mean/median/max RMSE: "
                    f"{model.rmse.min():0.4f}, "
                    f"{model.rmse.mean():0.4f}, "
                    f"{np.median(model.rmse):0.4f}, "
                    f"{model.rmse.max():0.4f}"
                )
            composition_matrices.append(comp_mtx_t)
            if config['USE_INTERCEPT']:
                alphas_t = model.alphas.ravel()
                alphas.append(alphas_t)
                print(
                    f"In-sample min/mean/median/max Alphas: "
                    f"{alphas_t.min():0.4f}, "
                    f"{alphas_t.mean():0.4f}, "
                    f"{np.median(alphas_t):0.4f}, "
                    f"{alphas_t.max():0.4f}"
                )
        # TODO: check if this is right for residual prices; what if alpha != 0?
        residual_prices_t = prices[valid_assets].iloc[t:t+1].fillna(0) @ comp_mtx_t
        residual_prices.append(residual_prices_t)
        residual_returns_t, _ = model.predict(returns[valid_assets].iloc[t:t+1].fillna(0))
        residual_returns.append(residual_returns_t)
        oos_rmse = np.sqrt(np.mean(residual_returns_t**2))
        oos_std = np.sqrt(np.mean(returns[valid_assets].iloc[t:t+1].fillna(0)**2))
        oos_rmses.append(oos_rmse)
        oos_stds.append(oos_std)
        if config.get('VERBOSE', False) and t % config['FACTOR_ESTIMATION_FREQUENCY_DAYS'] == 0:
            print(f"Out-of-sample day t RMSE: {oos_rmse:0.4f}")
            print(f"Day t returns std dev: {oos_std:0.4f}")
    
    residual_returns = pd.concat(residual_returns)
    residual_prices = pd.concat(residual_prices)
    alphas = pd.DataFrame(alphas, columns=valid_assets, index=estimation_dates) if config['USE_INTERCEPT'] else None
    oos_rmses = pd.DataFrame(oos_rmses, index=returns.index[config['RESIDUAL_ESTIMATION_LOOKBACK_DAYS']:T])
    oos_rmses.columns = ['RMSE']
    composition_matrices = pd.concat(composition_matrices, keys=estimation_dates)
    composition_matrices.index.names = ['date', 'ticker']
    
    return residual_returns, residual_prices, composition_matrices, estimation_dates, alphas


def estimate_ou_parameters(
    residual_returns: pd.DataFrame, 
    config: dict,
) -> pd.DataFrame:
    """
    Estimates OU parameters from a DataFrame of residual returns.

    Parameters:
    -----------
    residual_returns: pd.DataFrame
        A TxN DataFrame of residual returns.
    config: dict
        Configuration parameters for OU parameter estimation, containing:
        - R_SQUARED_THRESHOLD: float, the minimum threshold for the R^2, e.g. 0.25
        - B_THRESHOLD: float, the maximum threshold for beta, e.g. 0.99

    Returns:
    --------
    pd.DataFrame
        A DataFrame containing the estimated parameters (all floats)
        with index as the tickers of the residuals and columns as the variables:
        - y_last: last value of the residual process
        - mu: mean of the residual process
        - sigma: equilibrium standard deviation of the residual process
        - kappa: mean reversion speed
        - r_squared: R^2 value of the regression
        - b: beta coefficient of the regression
        - a: intercept of the regression
        - mask: boolean mask indicating valid residual processes
        - signal: standardized signal for the residual process
    """
    variables = ['y_last', 'mu', 'sigma', 'kappa', 'r_squared', 'b', 'a', 'mask', 'signal']
    
    T, N = residual_returns.shape
    cumulative_sums = residual_returns.cumsum(skipna=True, axis=0)
    params = np.zeros((N, len(variables)), dtype=np.float32)
    
    x = cumulative_sums.values.T

    # ===> YOUR CODE BELOW <===
    for idx, asset in enumerate(residual_returns.columns):
        # Avoid using loops?
        X_t: pd.Series = cumulative_sums[asset]

        X_t_1 = X_t.shift(1)

        mean, var = X_t.mean(), X_t.var()

        mean_1, var_1 = X_t_1.mean(), X_t_1.var()

        cov = X_t.cov(X_t_1)

        b = cov / X_t_1.var()
        a = mean_1 - b * X_t_1.mean()

        mu = a/(1-b)
        kappa = -np.log(b)/1 # dt is 1 day?
        var_zeta = (X_t - a - b * X_t_1).var()
        sigma = np.sqrt(var_zeta * 2 * kappa / (1 - b**2))
        var_eq = var_zeta / (1 - b**2)

        r_squared = 1 - (X_t - a - b * X_t_1).var() / X_t.var()

        mask = (b > 0) & (b <= config['B_THRESHOLD']) & (r_squared >= config['R_SQUARED_THRESHOLD'])

        signal = (X_t.iloc[-1] - mu)/np.sqrt(var_eq)
        
        if not mask:
            mu, sigma, kappa, signal = 0, 0, 0, 0

        params[idx, :] = [X_t.iloc[-1], mu, sigma, kappa, r_squared, b, a, mask, signal]
    # ===> YOUR CODE ABOVE <===

    result_df = pd.DataFrame(params, columns=variables, index=residual_returns.columns)
    
    return result_df


def forecast_residual_returns_ou_signal(
    residual_returns: pd.DataFrame,
    signals: pd.DataFrame,
    config: dict,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Forecasts residual returns from OU signals by regressing future cumulative residual returns
    on the signal. Fits one regression model across all tickers on a monthly basis.
    
    Parameters:
    -----------
    residual_returns: pd.DataFrame
        Dataframe of residual returns with dates as index and tickers as columns
    signals: pd.DataFrame
        Multiindex Dataframe of signals with (dates, tickers) as index and signal variables as columns;
        signal variables are the following:
        - y_last, float: the last value of the residual process (cumulative sum of returns)
        - mu, float: the mean of the residual process
        - sigma, float: the equilibrium standard deviation of the residual process
        - kappa, float: the mean reversion speed
        - r_squared, float: the R^2 value of the regression used to estimate the OU parameters
        - b, float: the slope of the regression used to estimate the OU parameters
        - a, float: the intercept of the regression used to estimate the OU parameters
        - mask, float: an indicator of whether the residual passes the mean reversion filter (1 if valid, 0 otherwise)
        - signal, float: the standardized signal for the residual process
    config: dict
        Configuration parameters, containing:
        - RETURN_FORECAST_HORIZON: int, the return forecast horizon (number of days into the future we treat as regression target)
        - RETURN_FORECAST_LOOKBACK: int, the number of days to look back to compute the regression coefficients
        - RETURN_FORECAST_REFIT_PERIOD: str, the period to refit the regression (e.g. 'M' for monthly; no need to implement others)
        - VERBOSE: bool, optional, whether to print verbose output
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        - pd.DataFrame: Dataframe containing forecasted returns
        - pd.DataFrame: Dataframe containing future horizon residual returns
        - pd.DataFrame: Dataframe containing regression coefficients for each estimation date
    """
    # Extract configuration parameters
    horizon = config['RETURN_FORECAST_HORIZON']
    lookback = config['RETURN_FORECAST_LOOKBACK']
    verbose = config.get('VERBOSE', False)
    
    # Create a DataFrame to store the forecasted returns with the same index and columns as returns
    forecasted_returns = pd.DataFrame(np.nan, index=residual_returns.index, columns=residual_returns.columns)
    
    # Get all dates from the returns DataFrame
    dates = residual_returns.index.tolist()
    
    # Extract the signals and masks into regular DataFrames
    signal_values = signals.reset_index().pivot(index='date', columns='ticker', values='signal')
    mask_values = signals.reset_index().pivot(index='date', columns='ticker', values='mask')
    
    # Ensure they align with returns dates
    signal_values = signal_values.reindex(residual_returns.index)
    mask_values = mask_values.reindex(residual_returns.index)
    
    # Precompute future cumulative returns for all dates
    future_cumul_returns = pd.DataFrame(np.nan, index=residual_returns.index, columns=residual_returns.columns)
    for i, date in enumerate(dates):
        if i + 1 + horizon <= len(dates):
            # Get returns from t+1 to t+horizon
            future_rets = residual_returns.iloc[i+1:i+1+horizon]
            # Calculate cumulative return
            future_cumul_returns.loc[date] = (1 + future_rets).prod() - 1
    
    # Initialize regression coefficients
    beta_0 = 0.0  # intercept
    beta_1 = 0.0  # slope
    last_refit_month = None
    
    # Create a DataFrame to store the regression coefficients
    beta_coefficients = []
    beta_dates = []
    
    # For each date in the returns DataFrame
    for i, current_date in enumerate(dates):
        # ===> YOUR CODE BELOW <===
        if i < lookback:
            continue

        if i + horizon >= len(dates):
            continue
        

        current_month = pd.Timestamp(current_date).to_period("M")
        if last_refit_month is None or current_month != last_refit_month:
            last_refit_month = current_month
        
            X = []
            y = []

            lb_dates = dates[i - lookback : i]

            for d in lb_dates:
                signals = signal_values.loc[d]
                masks = mask_values.loc[d]
                fut_ret = future_cumul_returns.loc[d]

                for t in residual_returns.columns:
                    s = signals[t]
                    m = masks[t]
                    f = fut_ret[t]

                    if (m == 1) and pd.notna(s) and pd.notna(f):
                        X.append(s)
                        y.append(f)
            
            if len(X) >= 2: 
                model = linregress(X, y)
                beta_1 = model.slope
                beta_0 = model.intercept
                r = model.rvalue
            else:
                beta_1, beta_0 = np.nan, np.nan
            
            beta_coefficients.append({"beta_0" : beta_0, "beta_1" : beta_1})
            beta_dates.append(current_date)


        sig = signal_values.loc[current_date]

        for tick in residual_returns.columns:
            st = sig[tick]

            if pd.notna(st):
                forecast = beta_0 + beta_1 * st
                forecasted_returns.loc[current_date, tick] = forecast
            else:
                forecasted_returns.loc[current_date, tick] = np.nan
        # ===> YOUR CODE ABOVE <===
    
    # Create DataFrame of beta coefficients
    beta_df = pd.DataFrame(beta_coefficients, index=beta_dates)
    
    return forecasted_returns, future_cumul_returns, beta_df


def forecast_returns_noisy_oracle(residual_returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Forecasts returns from a noisy oracle of the future residual information.
    
    Parameters:
    -----------
    residual_returns: pd.DataFrame
        Dataframe with residual returns and other information from estimate_residuals
    config: dict
        Configuration parameters, containing:
        - RETURN_FORECAST_HORIZON: int, the return forecast horizon (number of days into the future we treat as regression target)
        - INFORMATION_COEFFICIENT: float, the information coefficient that the estimated returns will have
        - SEED: int, random seed for reproducibility
        
    Returns:
    --------
    pd.DataFrame:
        a DataFrame of noisy oracle return predictions computed as alpha * (future_returns + noise)
    """
    horizon = config['RETURN_FORECAST_HORIZON']
    information_coefficient = config['INFORMATION_COEFFICIENT']
    seed = config['SEED']
    
    np.random.seed(seed)
    
    # ===> YOUR CODE BELOW <===
    raise NotImplementedError("forecast_returns_noisy_oracle not yet implemented")
    # ===> YOUR CODE ABOVE <===


def run_portfolio_optimization(
    residual_returns: pd.DataFrame,
    config: dict,
    forecasted_residual_returns: pd.DataFrame,
    forecasted_residual_covariance: Optional[Union[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Run portfolio optimization over given forecast time series to compute 
    a time series of approximate portfolio allocation weights. Output 
    approximate performance metrics for the portfolio.
    
    Parameters:
    -----------
    residual_returns: pd.DataFrame
        DataFrame with residual returns for each residual
    config: dict
        Configuration parameters, containing:
        - RISK_AVERSION: float, the risk aversion parameter
        - MAX_LEVERAGE: float, the maximum leverage allowed
        - MAX_TURNOVER: float, the maximum turnover allowed
        - RISK_FREE_RATE: float, the risk-free rate
        - MAX_WEIGHT: float, the maximum absolute value weight allowed for each residual
        - TRANSACTION_COST: float, the proportional transaction cost per sleeve of weight change
        - BORROW_COST: float, the short sell borrow cost per sleeve of borrowed weight
        - REBALANCING_PERIOD: str, optional, the rebalancing period ('W' for weekly, 'M' for monthly);
                              if not provided, daily rebalancing is used
    forecasted_residual_returns: pd.DataFrame
        DataFrame with forecasted returns for each residual
    forecasted_residual_covariance: Optional[Union[pd.DataFrame, Dict[pd.Timestamp, pd.DataFrame]]]
        DataFrame with forecasted covariance matrix for each residual, can be None
    
    Returns:
    --------
    Tuple[pd.DataFrame, pd.Series, pd.DataFrame]
        Tuple of:
        - DataFrame with the strategy's portfolio allocation weights for each residual
        - Series with strategy returns
        - DataFrame with strategy's performance metrics
    """
    # Extract configuration parameters
    risk_aversion = config['RISK_AVERSION']
    max_leverage = config['MAX_LEVERAGE']
    max_turnover = config['MAX_TURNOVER']
    max_weight = config['MAX_WEIGHT']
    transaction_cost = config['TRANSACTION_COST']
    borrow_cost = config['BORROW_COST']
    rebalancing_period = config.get('REBALANCING_PERIOD', None)  # Default to daily rebalancing
    verbose = config.get('VERBOSE', False)
    
    # Common data for all days
    tickers = residual_returns.columns
    n_assets = len(tickers)
    dates = forecasted_residual_returns.index
    
    # Initialize portfolio weights DataFrame
    portfolio_weights = pd.DataFrame(0.0, index=dates, columns=tickers)
    
    # For each date, run optimization to get portfolio weights
    last_rebalance_period = None
    
    for i, date in enumerate(dates):
        if verbose and i % 100 == 0:
            print(f"[{datetime.now()}] Processing date {i}/{len(dates)}")
        
        if i == 0:
            # Skip the first date as we need historical data
            continue
            
        # Get the forecasted returns for current date
        if date not in forecasted_residual_returns.index:
            # No forecast available, keep previous weights
            if i > 1:
                prev_date = dates[i-1]
                portfolio_weights.loc[date] = portfolio_weights.loc[prev_date]
            continue
        
        # Check if all forecast values are NaN
        current_forecast = forecasted_residual_returns.loc[date]
        if current_forecast.isna().all():
            # All forecast values are NaN, so this is either start or end of forecast period
            # Consequently, set weights to 0
            portfolio_weights.loc[date] = 0
            continue
            
        # Determine if we should rebalance based on the rebalancing period
        should_rebalance = True
        
        if rebalancing_period is not None:
            current_period = pd.Timestamp(date).to_period(rebalancing_period)
            
            if last_rebalance_period == current_period and i > 1:
                # Not time to rebalance yet, keep previous weights
                prev_date = dates[i-1]
                portfolio_weights.loc[date] = portfolio_weights.loc[prev_date]
                should_rebalance = False
            else:
                # New period, rebalance!
                last_rebalance_period = current_period
        
        if not should_rebalance:
            continue
            
        # Get the forecasted returns for current date
        mu = forecasted_residual_returns.loc[date].values
        
        # Use historical covariance if forecasted covariance is None
        if forecasted_residual_covariance is None:
            # Use last 125 days of data up to (but not including) current date
            historical_window = 125
            end_idx = residual_returns.index.get_indexer([date])[0]
            if end_idx < historical_window:
                # Not enough historical data
                continue
                
            start_idx = end_idx - historical_window
            historical_returns = residual_returns.iloc[start_idx:end_idx]
            Sigma = historical_returns.cov().values
        else:
            # Get forecasted covariance for current date
            if isinstance(forecasted_residual_covariance, dict) and date in forecasted_residual_covariance:
                Sigma = forecasted_residual_covariance[date]
            elif isinstance(forecasted_residual_covariance, pd.DataFrame) and date in forecasted_residual_covariance.index:
                # Assuming a multiindex DataFrame with date and ticker combinations
                date_cov = forecasted_residual_covariance.loc[date]
                Sigma = date_cov.values.reshape(n_assets, n_assets)
            else:
                raise ValueError(f"Forecasted covariance has invalid type '{type(forecasted_residual_covariance)}'")
        
        # Get previous weights for turnover constraint
        if i > 1:
            prev_date = dates[i-1]
            prev_weights = portfolio_weights.loc[prev_date].values
        else:
            prev_weights = np.zeros(n_assets)
            
        # Setup portfolio optimization problem
        w = cp.Variable(n_assets)
        
        # Maximize expected return - risk_aversion/2 * risk
        objective = cp.Maximize(mu @ w - risk_aversion/2 * cp.quad_form(w, Sigma))
        
        # Constraints
        constraints = [
            cp.sum(w) == 0,  # Dollar-neutral
            cp.sum(cp.abs(w)) <= max_leverage,  # Leverage constraint
            cp.abs(w) <= max_weight,  # Position size constraints
        ]
        
        # Add turnover constraint if not the first optimization
        if i > 1:
            constraints.append(cp.sum(cp.abs(w - prev_weights)) <= max_turnover)
        
        # Solve the optimization problem (using Clarabel; use a better solver if you have one)
        problem = cp.Problem(objective, constraints)
        try:
            problem.solve(solver=cp.CLARABEL)
            
            if problem.status in ['optimal', 'optimal_inaccurate']:
                # Store the optimal weights
                portfolio_weights.loc[date] = w.value
            else:
                print(f"Warning: Optimization failed for {date} with status {problem.status}")
                # Keep previous weights in case of failure
                if i > 1:
                    portfolio_weights.loc[date] = portfolio_weights.loc[prev_date]
        except Exception as e:
            print(f"Error in optimization for {date}: {e}")
            # Keep previous weights in case of error
            if i > 1:
                portfolio_weights.loc[date] = portfolio_weights.loc[prev_date]
    
    # Calculate portfolio returns
    # Shift weights by 1 day (we trade at the close of day t, positions take effect on day t+1)
    shifted_weights = portfolio_weights.shift(1)
    
    # Calculate gross returns: weights * next day's returns
    gross_returns = pd.Series(0.0, index=residual_returns.index)
    valid_days = shifted_weights.index.intersection(residual_returns.index)
    
    for date in valid_days:
        if pd.isna(shifted_weights.loc[date]).all():
            continue
        weights = shifted_weights.loc[date]
        if date in residual_returns.index:
            ret = residual_returns.loc[date]
            gross_returns.loc[date] = (weights * ret).sum()
    
    # Calculate transaction costs and borrowing costs
    turnover = shifted_weights.diff().abs().sum(axis=1)
    short_proportion = shifted_weights.clip(upper=0).abs().sum(axis=1)
    
    transaction_costs = transaction_cost * turnover
    borrowing_costs = borrow_cost * short_proportion
    
    # Net returns after costs
    net_returns = gross_returns - transaction_costs - borrowing_costs
    
    # Calculate performance metrics
    performance_metrics = pd.DataFrame()
    
    # Calculate annualized metrics assuming 252 trading days
    ann_factor = 252
    
    # Calculate basic stats for gross and net returns
    gross_mean = gross_returns.mean() * ann_factor
    gross_std = gross_returns.std() * np.sqrt(ann_factor)
    gross_sharpe = gross_mean / gross_std if gross_std > 0 else 0
    
    net_mean = net_returns.mean() * ann_factor
    net_std = net_returns.std() * np.sqrt(ann_factor)
    net_sharpe = net_mean / net_std if net_std > 0 else 0
    
    # Max drawdown
    gross_cum_returns = (1 + gross_returns).cumprod()
    gross_max_dd = (gross_cum_returns / gross_cum_returns.cummax() - 1).min()
    
    net_cum_returns = (1 + net_returns).cumprod()
    net_max_dd = (net_cum_returns / net_cum_returns.cummax() - 1).min()
    
    # Store metrics in DataFrame
    performance_metrics['Metric'] = [
        'Annualized Return (%)', 
        'Annualized Volatility (%)',
        'Sharpe Ratio', 
        'Maximum Drawdown (%)',
        'Average Turnover (%)',
        'Average Short Exposure (%)'
    ]
    
    performance_metrics['Gross'] = [
        gross_mean * 100,
        gross_std * 100,
        gross_sharpe,
        gross_max_dd * 100,
        turnover.mean() * 100,
        short_proportion.mean() * 100
    ]
    
    performance_metrics['Net'] = [
        net_mean * 100,
        net_std * 100,
        net_sharpe,
        net_max_dd * 100,
        turnover.mean() * 100,
        short_proportion.mean() * 100
    ]
    
    # Print performance metrics
    print("\nPortfolio Performance Metrics:\n")
    print(performance_metrics)
    
    # Plot the cumulative returns
    plt.figure(figsize=(12, 6))
    ((1 + gross_returns).cumprod() - 1).plot(label='Gross Returns')
    ((1 + net_returns).cumprod() - 1).plot(label='Net Returns')
    plt.title('Cumulative Portfolio Returns')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Plot the turnover over time
    plt.figure(figsize=(12, 6))
    turnover.plot()
    plt.title('Turnover Over Time')
    plt.xlabel('Date')
    plt.ylabel('Turnover')
    plt.grid(True)
    plt.show()
    
    # Plot the short exposure over time
    plt.figure(figsize=(12, 6))
    short_proportion.plot()
    plt.title('Short Exposure Over Time')
    plt.xlabel('Date')
    plt.ylabel('Short Exposure')
    plt.grid(True)
    plt.show()
    
    # Plot the largest weight in the portfolio over time
    plt.figure(figsize=(12, 6))
    portfolio_weights.abs().max(axis=1).plot()
    plt.title('Largest Weight Over Time')
    plt.xlabel('Date')
    plt.ylabel('Largest Weight')
    plt.grid(True)
    plt.show()
    
    # Make a bar plot of annual returns in each year
    plt.figure(figsize=(12, 6))
    ar = net_returns.resample('YE').apply(lambda x: (1 + x).prod() - 1)
    plt.bar(ar.index.year, ar.values)
    plt.title('Annual Returns')
    plt.xlabel('Year')
    plt.ylabel('Annual Return')
    plt.grid(True)
    plt.show()
    
    return portfolio_weights, net_returns, performance_metrics