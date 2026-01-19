# data_manager.py
"""
Data Manager for Brazilian Financial Analysis
Handles fetching, caching, and sharing historical data across scripts
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')

from config import get_config, ConfigManager


class DataManager:
    """Centralized data manager for fetching and caching market data."""
    
    def __init__(self, config: ConfigManager = None):
        self.config = config or get_config()
        self._cache = {}
        self._data_dir = Path(self.config.config_path).parent / "data"
        self._data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_historical_data(
        self, 
        ticker: str, 
        start_date: str = None, 
        end_date: str = None,
        period: str = '2y',
        force_refresh: bool = False,
        save_to_file: bool = True
    ) -> Optional[pd.DataFrame]:
        """Fetch historical data for a ticker"""
        ticker = self.config.normalize_ticker(ticker)
        
        # Check memory cache
        if not force_refresh and ticker in self._cache:
            return self._cache[ticker].copy()
        
        # Check file cache
        file_path = self._data_dir / f"{ticker.replace('.', '_')}_historical.csv"
        
        if not force_refresh and file_path.exists():
            try:
                data = pd.read_csv(file_path, index_col=0, parse_dates=True)
                if self._is_cache_valid(data, period):
                    self._cache[ticker] = data
                    return data.copy()
            except Exception:
                pass
        
        # Fetch from Yahoo Finance
        print(f"  Fetching {ticker} from Yahoo Finance...")
        try:
            stock = yf.Ticker(ticker)
            
            if start_date and end_date:
                data = stock.history(start=start_date, end=end_date)
            else:
                data = stock.history(period=period)
            
            if data.empty:
                print(f"    Warning: No data for {ticker}")
                return None
            
            # Calculate returns
            data['Returns'] = data['Close'].pct_change()
            data['Log_Returns'] = np.log(data['Close'] / data['Close'].shift(1))
            
            # Save to file
            if save_to_file:
                data.to_csv(file_path)
            
            # Cache in memory
            self._cache[ticker] = data
            
            return data.copy()
            
        except Exception as e:
            print(f"    Error fetching {ticker}: {e}")
            return None
    
    def fetch_multiple_tickers(
        self,
        tickers: List[str] = None,
        period: str = '2y',
        force_refresh: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers"""
        if tickers is None:
            tickers = self.config.get_tickers()
        
        results = {}
        print(f"Fetching data for {len(tickers)} tickers...")
        
        for ticker in tickers:
            data = self.fetch_historical_data(ticker, period=period, force_refresh=force_refresh)
            if data is not None:
                results[self.config.normalize_ticker(ticker)] = data
        
        return results
    
    def get_returns_matrix(
        self,
        tickers: List[str] = None,
        period: str = '2y',
        return_type: str = 'simple'
    ) -> Optional[pd.DataFrame]:
        """Get returns matrix for multiple tickers"""
        if tickers is None:
            tickers = self.config.get_tickers()
        
        all_data = self.fetch_multiple_tickers(tickers, period=period)
        
        if not all_data:
            return None
        
        returns_col = 'Returns' if return_type == 'simple' else 'Log_Returns'
        
        returns_dict = {}
        for ticker, data in all_data.items():
            if returns_col in data.columns:
                returns_dict[ticker] = data[returns_col]
        
        if not returns_dict:
            return None
        
        returns_df = pd.DataFrame(returns_dict).dropna()
        return returns_df
    
    def get_covariance_matrix(
        self,
        tickers: List[str] = None,
        period: str = '2y',
        annualize: bool = True
    ) -> Optional[pd.DataFrame]:
        """Get covariance matrix for tickers"""
        returns = self.get_returns_matrix(tickers, period)
        if returns is None:
            return None
        
        cov = returns.cov()
        if annualize:
            cov = cov * 252
        return cov
    
    def get_stock_info(self, ticker: str) -> Optional[dict]:
        """Get stock info from Yahoo Finance"""
        ticker = self.config.normalize_ticker(ticker)
        try:
            stock = yf.Ticker(ticker)
            return stock.info
        except Exception:
            return None
    
    def get_financials(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get financial statements"""
        ticker = self.config.normalize_ticker(ticker)
        try:
            stock = yf.Ticker(ticker)
            return {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow
            }
        except Exception:
            return {}
    
    def load_from_file(self, ticker: str) -> Optional[pd.DataFrame]:
        """Load data from cached file"""
        ticker = self.config.normalize_ticker(ticker)
        file_path = self._data_dir / f"{ticker.replace('.', '_')}_historical.csv"
        
        if file_path.exists():
            try:
                return pd.read_csv(file_path, index_col=0, parse_dates=True)
            except Exception:
                pass
        return None
    
    def list_cached_tickers(self) -> List[str]:
        """List all cached tickers"""
        files = list(self._data_dir.glob("*_historical.csv"))
        return [f.stem.replace('_historical', '').replace('_SA', '.SA') for f in files]
    
    def clear_cache(self, ticker: str = None):
        """Clear cache"""
        if ticker:
            ticker = self.config.normalize_ticker(ticker)
            self._cache.pop(ticker, None)
            file_path = self._data_dir / f"{ticker.replace('.', '_')}_historical.csv"
            if file_path.exists():
                file_path.unlink()
        else:
            self._cache.clear()
            for f in self._data_dir.glob("*_historical.csv"):
                f.unlink()
        print(f"Cache cleared: {'all' if ticker is None else ticker}")
    
    def _is_cache_valid(self, data: pd.DataFrame, period: str) -> bool:
        """Check if cached data is still valid"""
        if data.empty:
            return False
        
        last_date = data.index[-1]
        if isinstance(last_date, str):
            last_date = pd.to_datetime(last_date)
        
        # Allow 3 days of staleness (weekends, holidays)
        return (datetime.now() - last_date.to_pydatetime().replace(tzinfo=None)).days < 3


def get_data_manager(config: ConfigManager = None) -> DataManager:
    """Get data manager instance"""
    return DataManager(config)


if __name__ == "__main__":
    dm = get_data_manager()
    config = get_config()
    
    print("\n=== Data Manager Test ===")
    
    ticker = config.get_default_ticker()
    data = dm.fetch_historical_data(ticker, period='6mo')
    
    if data is not None:
        print(f"\n{ticker}: {len(data)} rows")
        print(data.tail(3))
    
    print("\n--- Fetching all tickers ---")
    all_data = dm.fetch_multiple_tickers(period='3mo')
    print(f"Fetched: {list(all_data.keys())}")
