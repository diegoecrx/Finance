# stock_metadata.py
"""
Stock Metadata Manager
Automatically fetches and caches stock information (sector, category, etc.)
"""

import json
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

# Paths
METADATA_FILE = Path(__file__).parent / "stock_cache.json"
CACHE_EXPIRY_DAYS = 7  # Re-fetch metadata after 7 days


class StockMetadataManager:
    """
    Manages stock metadata - fetches from Yahoo Finance and caches locally.
    Users only need to provide ticker symbols.
    """
    
    def __init__(self):
        self._cache = self._load_cache()
    
    def _load_cache(self) -> dict:
        """Load cached metadata from file"""
        if METADATA_FILE.exists():
            try:
                with open(METADATA_FILE, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception:
                return {"stocks": {}, "last_updated": {}}
        return {"stocks": {}, "last_updated": {}}
    
    def _save_cache(self):
        """Save metadata cache to file"""
        try:
            with open(METADATA_FILE, 'w', encoding='utf-8') as f:
                json.dump(self._cache, f, indent=4, ensure_ascii=False)
        except Exception as e:
            print(f"Warning: Could not save metadata cache: {e}")
    
    @staticmethod
    def normalize_ticker(ticker: str) -> str:
        """Normalize ticker to Brazilian format"""
        ticker = ticker.upper().strip()
        if not ticker.endswith('.SA') and not ticker.startswith('^'):
            ticker = f"{ticker}.SA"
        return ticker
    
    def _is_cache_valid(self, ticker: str) -> bool:
        """Check if cached data is still valid"""
        last_updated = self._cache.get("last_updated", {}).get(ticker)
        if not last_updated:
            return False
        
        try:
            update_date = datetime.fromisoformat(last_updated)
            return (datetime.now() - update_date).days < CACHE_EXPIRY_DAYS
        except:
            return False
    
    def fetch_metadata(self, ticker: str, force_refresh: bool = False) -> Optional[Dict]:
        """
        Fetch stock metadata from Yahoo Finance
        
        Returns dict with: sector, industry, type (stock/fii), name, etc.
        """
        ticker = self.normalize_ticker(ticker)
        
        # Return cached if valid
        if not force_refresh and ticker in self._cache["stocks"] and self._is_cache_valid(ticker):
            return self._cache["stocks"][ticker]
        
        print(f"  Fetching metadata for {ticker}...")
        
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            if not info or info.get('regularMarketPrice') is None:
                print(f"    Warning: No data available for {ticker}")
                return None
            
            # Determine if it's a FII (Real Estate Fund)
            quote_type = info.get('quoteType', 'EQUITY')
            long_name = info.get('longName', '') or ''
            short_name = info.get('shortName', '') or ''
            
            is_fii = (
                'FII' in long_name.upper() or 
                'FUNDO' in long_name.upper() or
                '11' in ticker.replace('.SA', '')[-2:] or  # FIIs usually end in 11
                quote_type == 'ETF'
            )
            
            # Build metadata
            metadata = {
                "ticker": ticker,
                "name": info.get('longName') or info.get('shortName') or ticker,
                "short_name": info.get('shortName', ticker),
                "type": "FII" if is_fii else "Stock",
                "sector": info.get('sector', 'Unknown'),
                "industry": info.get('industry', 'Unknown'),
                "market_cap": info.get('marketCap', 0),
                "currency": info.get('currency', 'BRL'),
                "exchange": info.get('exchange', 'SAO'),
                "beta": info.get('beta', 1.0),
                "dividend_yield": info.get('dividendYield', 0),
                "pe_ratio": info.get('trailingPE', 0),
                "pb_ratio": info.get('priceToBook', 0),
                "profit_margin": info.get('profitMargins', 0),
                "roe": info.get('returnOnEquity', 0),
                "current_price": info.get('currentPrice') or info.get('regularMarketPrice', 0),
            }
            
            # Classify sector for Brazilian stocks
            metadata["sector_normalized"] = self._normalize_sector(
                metadata["sector"], 
                metadata["industry"],
                metadata["name"]
            )
            
            # Save to cache
            self._cache["stocks"][ticker] = metadata
            self._cache["last_updated"][ticker] = datetime.now().isoformat()
            self._save_cache()
            
            return metadata
            
        except Exception as e:
            print(f"    Error fetching {ticker}: {e}")
            return None
    
    def _normalize_sector(self, sector: str, industry: str, name: str) -> str:
        """Normalize sector names for Brazilian market"""
        sector_lower = (sector or '').lower()
        industry_lower = (industry or '').lower()
        name_lower = (name or '').lower()
        
        # Banking
        if any(x in sector_lower + industry_lower for x in ['bank', 'financ']):
            return 'banks'
        
        # Mining
        if any(x in sector_lower + industry_lower + name_lower for x in ['mining', 'metal', 'steel', 'mineração', 'siderurg']):
            return 'mining'
        
        # Energy / Oil & Gas
        if any(x in sector_lower + industry_lower for x in ['oil', 'gas', 'petro', 'energy']):
            return 'energy'
        
        # Utilities
        if any(x in sector_lower + industry_lower for x in ['utilities', 'electric', 'elétric']):
            return 'utilities'
        
        # Retail
        if any(x in sector_lower + industry_lower for x in ['retail', 'varejo', 'consumer cyclical']):
            return 'retail'
        
        # Consumer
        if any(x in sector_lower + industry_lower for x in ['consumer', 'food', 'beverage', 'aliment']):
            return 'consumer'
        
        # Real Estate / FII
        if any(x in sector_lower + industry_lower + name_lower for x in ['real estate', 'fii', 'fundo', 'imobili']):
            return 'real_estate'
        
        return 'other'
    
    def get_all_metadata(self, tickers: List[str], force_refresh: bool = False) -> Dict[str, Dict]:
        """Fetch metadata for multiple tickers"""
        results = {}
        
        print(f"Loading metadata for {len(tickers)} tickers...")
        
        for ticker in tickers:
            normalized = self.normalize_ticker(ticker)
            metadata = self.fetch_metadata(normalized, force_refresh)
            if metadata:
                results[normalized] = metadata
        
        return results
    
    def get_stocks_by_sector(self, tickers: List[str] = None) -> Dict[str, List[str]]:
        """Group tickers by sector"""
        if tickers is None:
            # Use all cached tickers
            tickers = list(self._cache.get("stocks", {}).keys())
        
        sectors = {}
        
        for ticker in tickers:
            ticker = self.normalize_ticker(ticker)
            metadata = self.fetch_metadata(ticker)
            
            if metadata:
                sector = metadata.get("sector_normalized", "other")
                if sector not in sectors:
                    sectors[sector] = []
                sectors[sector].append(ticker)
        
        return sectors
    
    def get_stocks_by_type(self, tickers: List[str] = None) -> Dict[str, List[str]]:
        """Group tickers by type (Stock vs FII)"""
        if tickers is None:
            tickers = list(self._cache.get("stocks", {}).keys())
        
        result = {"stocks": [], "fiis": []}
        
        for ticker in tickers:
            ticker = self.normalize_ticker(ticker)
            metadata = self.fetch_metadata(ticker)
            
            if metadata:
                if metadata.get("type") == "FII":
                    result["fiis"].append(ticker)
                else:
                    result["stocks"].append(ticker)
        
        return result
    
    def get_cached_metadata(self, ticker: str) -> Optional[Dict]:
        """Get metadata from cache without fetching"""
        ticker = self.normalize_ticker(ticker)
        return self._cache.get("stocks", {}).get(ticker)
    
    def list_cached_tickers(self) -> List[str]:
        """List all tickers in cache"""
        return list(self._cache.get("stocks", {}).keys())
    
    def get_portfolio_weights(self, tickers: List[str]) -> Dict[str, float]:
        """Generate equal weights for a list of tickers"""
        tickers = [self.normalize_ticker(t) for t in tickers]
        weight = 1.0 / len(tickers) if tickers else 0
        return {t: weight for t in tickers}
    
    def clear_cache(self, ticker: str = None):
        """Clear cache for specific ticker or all"""
        if ticker:
            ticker = self.normalize_ticker(ticker)
            self._cache["stocks"].pop(ticker, None)
            self._cache["last_updated"].pop(ticker, None)
        else:
            self._cache = {"stocks": {}, "last_updated": {}}
        
        self._save_cache()
        print(f"Cache cleared: {'all' if ticker is None else ticker}")
    
    def print_summary(self, tickers: List[str] = None):
        """Print summary of stock metadata"""
        if tickers is None:
            tickers = self.list_cached_tickers()
        
        print("\n" + "="*80)
        print("STOCK METADATA SUMMARY")
        print("="*80)
        
        for ticker in tickers:
            ticker = self.normalize_ticker(ticker)
            meta = self.fetch_metadata(ticker)
            
            if meta:
                print(f"\n{meta['ticker']} - {meta['name'][:40]}")
                print(f"  Type: {meta['type']} | Sector: {meta['sector_normalized']}")
                print(f"  Industry: {meta['industry']}")
                print(f"  Price: R$ {meta['current_price']:.2f} | Beta: {meta['beta']:.2f}")


# Convenience function
def get_metadata_manager() -> StockMetadataManager:
    """Get metadata manager instance"""
    return StockMetadataManager()


if __name__ == "__main__":
    # Test
    manager = get_metadata_manager()
    
    test_tickers = ["PETR4", "VALE3", "BTLG11", "ITUB4"]
    
    manager.print_summary(test_tickers)
    
    print("\n--- By Sector ---")
    sectors = manager.get_stocks_by_sector(test_tickers)
    for sector, tickers in sectors.items():
        print(f"{sector}: {tickers}")
    
    print("\n--- By Type ---")
    types = manager.get_stocks_by_type(test_tickers)
    print(f"Stocks: {types['stocks']}")
    print(f"FIIs: {types['fiis']}")
