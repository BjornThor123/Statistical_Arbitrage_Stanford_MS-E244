import pandas as pd
import numpy as np
import time

from pathlib import Path
import zipfile

from concurrent.futures import ThreadPoolExecutor

from typing import Union, List, Optional, Tuple, Any
from functools import wraps

def timer(func):
    """Decorator to time function execution if verbose mode is on"""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        if self.verbose:
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            print(f"{func.__name__} took {end_time - start_time:.2f} seconds")
        else:
            result = func(self, *args, **kwargs)
        return result
    return wrapper

class DataLoader:
    def __init__(self, data_path: str, verbose: bool = False):
        self.data_path = Path(data_path)
        self.options_path = self.data_path / "options"
        self.options_data_path = self.options_path / "data"
        self.options_metadata_path = self.options_path / "metadata"
        
        self.equities_path = self.data_path / "equities"
        self.equities_data_path = self.equities_path / "data"
        self.equities_metadata_path = self.equities_path / "metadata"
        
        self.rf_path = self.data_path / "risk_free"
        self.rf_data_path = self.rf_path / "data"
        self.rf_metadata_path = self.rf_path / "metadata"

        self.verbose = verbose
        
    def unzip(self, *zip_files: Path, extract_to: Path) -> None:
        def extract_single(zip_file: Path):
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
        
        with ThreadPoolExecutor() as executor:
            executor.map(extract_single, zip_files)
            
    def ensure_col_name(self, metadata_path: Path, col_names: Union[str, List[str]]) -> bool:
        # Convert string to list
        if isinstance(col_names, str):
            col_names = [col_names]

        metadata_file = next(metadata_path.glob("*dictionary.csv"))
        metadata = pd.read_csv(metadata_file)
        variable_names = metadata["Variable Name"].values

        if not all(col in variable_names for col in col_names):
            raise ValueError(f"Not all columns {col_names} found in metadata at {metadata_path}")
        return True
    
    @timer
    def load_options_data(self,
                            tickers: List = None,
                            start_date: pd.Timestamp = None,
                            end_date: pd.Timestamp = None,
                            ) -> pd.DataFrame:
        
        # Unzip options data if not already unzipped
        zip_files = list(self.options_data_path.glob("*.zip"))
        if zip_files:
            self.unzip(*zip_files, extract_to=self.options_data_path)
        
        # Load all CSV files into a single DataFrame
        csv_files = list(self.options_data_path.glob("**/*.csv"))

        # Check metadata once before processing
        self.ensure_col_name(self.options_metadata_path, ['ticker', 'date'])

        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, low_memory=False)
            if tickers is not None:
                df = df[df['ticker'].isin(tickers)]
            
            df['date'] = pd.to_datetime(df['date'])
            if start_date is not None:
                df = df[df['date'] >= start_date]
            if end_date is not None:
                df = df[df['date'] <= end_date]
            
            data_frames.append(df)
        
        if not data_frames:
            return pd.DataFrame()
        options_data = pd.concat(data_frames, ignore_index=True)
        return options_data
    
    
    @timer
    def load_equities_data(self, tickers: List = None,
                           start_date: pd.Timestamp = None,
                           end_date: pd.Timestamp = None) -> pd.DataFrame:

        # Unzip equities data if not already unzipped
        zip_files = list(self.equities_data_path.glob("*.zip"))
        if zip_files:
            self.unzip(*zip_files, extract_to=self.equities_data_path)

        # Load all CSV files into a single DataFrame
        csv_files = list(self.equities_data_path.glob("**/*.csv"))

        # Check metadata once before processing
        self.ensure_col_name(self.equities_metadata_path, ['ticker', 'date'])

        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, low_memory=False)
            if tickers is not None:
                df = df[df['ticker'].isin(tickers)]

            df['date'] = pd.to_datetime(df['date'])
            if start_date is not None:
                df = df[df['date'] >= start_date]
            if end_date is not None:
                df = df[df['date'] <= end_date]

            data_frames.append(df)

        if not data_frames:
            return pd.DataFrame()
        equities_data = pd.concat(data_frames, ignore_index=True)
        return equities_data

    @timer
    def load_rf_data(self, start_date: pd.Timestamp = None,
                     end_date: pd.Timestamp = None) -> pd.DataFrame:

        # Unzip risk-free data if not already unzipped
        zip_files = list(self.rf_data_path.glob("*.zip"))
        if zip_files:
            self.unzip(*zip_files, extract_to=self.rf_data_path)

        # Load all CSV files into a single DataFrame
        csv_files = list(self.rf_data_path.glob("**/*.csv"))

        # Check metadata once before processing
        self.ensure_col_name(self.rf_metadata_path, 'date')

        data_frames = []
        for csv_file in csv_files:
            df = pd.read_csv(csv_file, low_memory=False)

            df['date'] = pd.to_datetime(df['date'])
            if start_date is not None:
                df = df[df['date'] >= start_date]
            if end_date is not None:
                df = df[df['date'] <= end_date]

            data_frames.append(df)

        if not data_frames:
            return pd.DataFrame()
        rf_data = pd.concat(data_frames, ignore_index=True)
        return rf_data

if __name__ == "__main__":
    data_path = "/Users/bjorn/Documents/Skóli/Stanford/Skóli/Q2/StatArb/Statistical_Arbitrage_Stanford_MS-E244/project/data"
    data_loader = DataLoader(data_path, verbose=True)
    options_data = data_loader.load_options_data()
    print(options_data.head())

        
    
        
        
    