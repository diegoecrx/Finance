# config.py
"""
Configuration Manager for Brazilian Financial Analysis
Simplified: Only requires ticker names - metadata is fetched automatically
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

# Default paths
DEFAULT_CONFIG_PATH = Path(__file__).parent / "stocks.json"
DEFAULT_DATA_DIR = Path(__file__).parent / "data"

# Brazilian benchmark indices
IBOV = "^BVSP"  # Bovespa Index for stocks
IFIX = "IFIX11.SA"  # FII Index (use ETF as proxy)


class ConfigManager:
    """
    Simplified configuration manager.
    Users only provide ticker names - all other info is auto-fetched.
    """
    _instance = None
    _config = None
    _metadata_manager = None
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self, config_path: str = None):
        """Initialize configuration manager"""
        if ConfigManager._config is None:
            self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
            self._load_config()
            self._ensure_data_dir()
    
    def _load_config(self):
        """Load configuration from JSON file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8-sig') as f:
                    ConfigManager._config = json.load(f)
            else:
                ConfigManager._config = self._default_config()
                self.save_config()
        except Exception as e:
            print(f"Error loading config: {e}")
            ConfigManager._config = self._default_config()
    
    def _default_config(self) -> dict:
        return {
            "tickers": ["PETR4.SA", "VALE3.SA", "ITUB4.SA"],
            "default_ticker": "PETR4.SA",
            "default_benchmark": "^BVSP",
            "fii_benchmark": "IFIX11.SA"
        }
    
    def _ensure_data_dir(self):
        DEFAULT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    def _get_metadata_manager(self):
        if ConfigManager._metadata_manager is None:
            from stock_metadata import get_metadata_manager
            ConfigManager._metadata_manager = get_metadata_manager()
        return ConfigManager._metadata_manager
    
    def save_config(self):
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(ConfigManager._config, f, indent=4, ensure_ascii=False)
    
    def reload(self):
        ConfigManager._config = None
        ConfigManager._metadata_manager = None
        self._load_config()
    
    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        if ticker is None:
            return "PETR4.SA"
        ticker = ticker.upper().strip()
        if not ticker.endswith('.SA') and not ticker.startswith('^'):
            ticker = f"{ticker}.SA"
        return ticker
    
    def get_tickers(self) -> List[str]:
        tickers = ConfigManager._config.get("tickers", [])
        return [self.normalize_ticker(t) for t in tickers]
    
    def add_ticker(self, ticker: str):
        ticker = self.normalize_ticker(ticker)
        if ticker not in ConfigManager._config.get("tickers", []):
            ConfigManager._config.setdefault("tickers", []).append(ticker)
            self.save_config()
    
    def remove_ticker(self, ticker: str):
        ticker = self.normalize_ticker(ticker)
        if ticker in ConfigManager._config.get("tickers", []):
            ConfigManager._config["tickers"].remove(ticker)
            self.save_config()
    
    def set_tickers(self, tickers: List[str]):
        ConfigManager._config["tickers"] = [self.normalize_ticker(t) for t in tickers]
        self.save_config()
    
    def get_stocks(self) -> List[str]:
        manager = self._get_metadata_manager()
        types = manager.get_stocks_by_type(self.get_tickers())
        return types.get("stocks", [])
    
    def get_fiis(self) -> List[str]:
        manager = self._get_metadata_manager()
        types = manager.get_stocks_by_type(self.get_tickers())
        return types.get("fiis", [])
    
    def get_sectors(self) -> Dict[str, List[str]]:
        manager = self._get_metadata_manager()
        return manager.get_stocks_by_sector(self.get_tickers())
    
    def get_sector_tickers(self, sector: str) -> List[str]:
        return self.get_sectors().get(sector.lower(), [])
    
    def get_metadata(self, ticker: str) -> Optional[Dict]:
        manager = self._get_metadata_manager()
        return manager.fetch_metadata(self.normalize_ticker(ticker))
    
    def get_all_metadata(self) -> Dict[str, Dict]:
        manager = self._get_metadata_manager()
        return manager.get_all_metadata(self.get_tickers())
    
    def is_fii(self, ticker: str) -> bool:
        ticker = self.normalize_ticker(ticker)
        if ticker.replace('.SA', '').endswith('11'):
            return True
        manager = self._get_metadata_manager()
        meta = manager.get_cached_metadata(ticker)
        if meta:
            return meta.get('type') == 'FII'
        return False
    
    def get_portfolio(self, custom_weights: Dict[str, float] = None) -> Dict[str, float]:
        if custom_weights:
            return {self.normalize_ticker(k): v for k, v in custom_weights.items()}
        tickers = self.get_tickers()
        if not tickers:
            return {}
        weight = 1.0 / len(tickers)
        return {t: weight for t in tickers}
    
    def get_default_ticker(self) -> str:
        default = ConfigManager._config.get("default_ticker", "")
        if not default:
            tickers = self.get_tickers()
            default = tickers[0] if tickers else "PETR4.SA"
        return self.normalize_ticker(default)
    
    def set_default_ticker(self, ticker: str):
        ConfigManager._config["default_ticker"] = self.normalize_ticker(ticker)
        self.save_config()
    
    def get_default_benchmark(self) -> str:
        return ConfigManager._config.get("default_benchmark", IBOV)
    
    def get_fii_benchmark(self) -> str:
        return ConfigManager._config.get("fii_benchmark", IFIX)
    
    def get_benchmark_for_ticker(self, ticker: str) -> str:
        if self.is_fii(ticker):
            return self.get_fii_benchmark()
        return self.get_default_benchmark()
    
    def set_default_benchmark(self, benchmark: str):
        ConfigManager._config["default_benchmark"] = benchmark
        self.save_config()
    
    def get_data_path(self, ticker: str) -> Path:
        ticker = self.normalize_ticker(ticker).replace('.', '_')
        return DEFAULT_DATA_DIR / f"{ticker}_historical.csv"
    
    # Backward Compatibility
    def get_watchlist_stocks(self) -> List[str]:
        return self.get_stocks()
    
    def get_watchlist_fiis(self) -> List[str]:
        return self.get_fiis()
    
    def get_all_watchlist(self) -> List[str]:
        return self.get_tickers()
    
    def get_all_sectors(self) -> Dict[str, List[str]]:
        return self.get_sectors()
    
    def refresh_metadata(self, force: bool = True):
        manager = self._get_metadata_manager()
        manager.get_all_metadata(self.get_tickers(), force_refresh=force)
    
    def print_summary(self):
        print("\n" + "="*60)
        print("CONFIGURATION SUMMARY")
        print("="*60)
        tickers = self.get_tickers()
        print(f"\nTotal Tickers: {len(tickers)}")
        print(f"First 5: {tickers[:5]}")
        print(f"\nDefault Ticker: {self.get_default_ticker()}")
        print(f"Stock Benchmark: {self.get_default_benchmark()}")
        print(f"FII Benchmark: {self.get_fii_benchmark()}")


def get_config(config_path: str = None) -> ConfigManager:
    return ConfigManager(config_path)


if __name__ == "__main__":
    config = get_config()
    config.print_summary()