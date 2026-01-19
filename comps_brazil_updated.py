# comps_brazil_updated.py
"""
Comparable Company Analysis for Brazilian Market
Updated to use dynamic configuration and shared data manager
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from config import get_config, ConfigManager
from data_manager import get_data_manager, DataManager


class BrazilianCompsAnalysis:
    def __init__(
        self,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        """
        Comparable Company Analysis for Brazilian Market
        
        Parameters:
        -----------
        config : ConfigManager
            Configuration manager instance
        data_manager : DataManager
            Data manager instance
        """
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        self.companies = {}
        self.comps_data = pd.DataFrame()
        
        # Sector mappings from config
        self.sector_mappings = self.config.get_all_sectors()
    
    def fetch_sector_companies(
        self, 
        sector: str, 
        include_target: bool = True, 
        target_ticker: str = None,
        custom_tickers: List[str] = None
    ) -> bool:
        """
        Fetch data for companies in a specific sector
        
        Parameters:
        -----------
        sector : str
            Sector name (e.g., 'banks', 'mining')
        include_target : bool
            Whether to include target ticker
        target_ticker : str
            Target company for comparison
        custom_tickers : List[str]
            Custom list of tickers (overrides sector)
        """
        if custom_tickers:
            tickers = [self.config.normalize_ticker(t) for t in custom_tickers]
        else:
            tickers = self.sector_mappings.get(sector.lower(), [])
        
        if not tickers:
            print(f"No tickers found for sector: {sector}")
            return False
        
        if target_ticker:
            target_ticker = self.config.normalize_ticker(target_ticker)
            if include_target and target_ticker not in tickers:
                tickers.append(target_ticker)
        
        print(f"Analyzing {len(tickers)} companies in {sector} sector...")
        
        all_data = []
        
        for ticker in tickers:
            try:
                print(f"  Fetching {ticker}...")
                stock = yf.Ticker(ticker)
                info = stock.info
                
                # Also cache historical data
                self.data_manager.fetch_historical_data(ticker, period='1y')
                
                # Extract key metrics
                market_cap = info.get('marketCap', 0)
                if market_cap == 0:
                    continue
                
                company_data = {
                    'ticker': ticker,
                    'name': info.get('longName', ticker),
                    'sector': sector,
                    'market_cap_brl': market_cap / 1e9,
                    'enterprise_value': info.get('enterpriseValue', 0),
                    'ev_brl': info.get('enterpriseValue', 0) / 1e9,
                    'revenue_brl': info.get('totalRevenue', 0) / 1e9,
                    'ebitda_brl': info.get('ebitda', 0) / 1e9,
                    'net_income_brl': info.get('netIncomeToCommon', 0) / 1e9,
                    'total_debt': info.get('totalDebt', 0) / 1e9,
                    'cash': info.get('totalCash', 0) / 1e9,
                    'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                    'pe_ratio': info.get('trailingPE', 0),
                    'pb_ratio': info.get('priceToBook', 0),
                    'ev_revenue': info.get('enterpriseToRevenue', 0),
                    'ev_ebitda': info.get('enterpriseToEbitda', 0),
                    'profit_margin': info.get('profitMargins', 0),
                    'roe': info.get('returnOnEquity', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 1.0)
                }
                
                # Calculate derived metrics
                if company_data['ebitda_brl'] > 0:
                    company_data['debt_ebitda'] = company_data['total_debt'] / company_data['ebitda_brl']
                else:
                    company_data['debt_ebitda'] = 0
                
                all_data.append(company_data)
                
            except Exception as e:
                print(f"    Error fetching {ticker}: {e}")
                continue
        
        if all_data:
            self.comps_data = pd.DataFrame(all_data)
            # Clean up invalid values
            self.comps_data = self.comps_data.replace([np.inf, -np.inf], np.nan)
            print(f"Successfully loaded {len(self.comps_data)} companies")
            return True
        
        return False
    
    def calculate_comps_valuation(
        self, 
        target_ticker: str, 
        revenue: float = None, 
        ebitda: float = None, 
        earnings: float = None
    ) -> Optional[Dict]:
        """
        Value target company using comparable multiples
        
        Parameters:
        -----------
        target_ticker : str
            Target company ticker
        revenue : float
            Target revenue (BRL billions)
        ebitda : float
            Target EBITDA (BRL billions)
        earnings : float
            Target net income (BRL billions)
        """
        if self.comps_data.empty:
            print("No comps data available. Fetch sector companies first.")
            return None
        
        target_ticker = self.config.normalize_ticker(target_ticker)
        
        # Calculate median multiples (excluding target)
        comps_ex_target = self.comps_data[self.comps_data['ticker'] != target_ticker]
        
        def clean_median(series):
            """Remove outliers and calculate median"""
            q1, q3 = series.quantile(0.25), series.quantile(0.75)
            iqr = q3 - q1
            clean = series[(series >= q1 - 1.5*iqr) & (series <= q3 + 1.5*iqr)]
            return clean.median() if not clean.empty else series.median()
        
        median_ev_revenue = clean_median(comps_ex_target['ev_revenue'].dropna())
        median_ev_ebitda = clean_median(comps_ex_target['ev_ebitda'].dropna())
        median_pe = clean_median(comps_ex_target['pe_ratio'].dropna())
        
        valuations = {}
        
        if revenue and revenue > 0 and median_ev_revenue > 0:
            ev_from_revenue = revenue * median_ev_revenue
            valuations['EV/Revenue'] = {
                'multiple': median_ev_revenue,
                'implied_ev': ev_from_revenue
            }
        
        if ebitda and ebitda > 0 and median_ev_ebitda > 0:
            ev_from_ebitda = ebitda * median_ev_ebitda
            valuations['EV/EBITDA'] = {
                'multiple': median_ev_ebitda,
                'implied_ev': ev_from_ebitda
            }
        
        if earnings and earnings > 0 and median_pe > 0:
            equity_from_pe = earnings * median_pe
            valuations['P/E'] = {
                'multiple': median_pe,
                'implied_equity': equity_from_pe
            }
        
        if valuations:
            # Calculate average implied EV
            ev_values = [v['implied_ev'] for v in valuations.values() if 'implied_ev' in v]
            avg_implied_ev = np.mean(ev_values) if ev_values else 0
            
            return {
                'target': target_ticker,
                'valuations': valuations,
                'avg_implied_ev': avg_implied_ev,
                'median_multiples': {
                    'EV/Revenue': median_ev_revenue,
                    'EV/EBITDA': median_ev_ebitda,
                    'P/E': median_pe
                }
            }
        
        return None
    
    def generate_comps_table(self) -> pd.DataFrame:
        """Generate formatted comparables table"""
        if self.comps_data.empty:
            return pd.DataFrame()
        
        formatted_df = self.comps_data.copy()
        
        display_cols = [
            'ticker', 'name', 'market_cap_brl', 'ev_brl', 
            'ev_ebitda', 'pe_ratio', 'pb_ratio', 'roe', 
            'debt_ebitda', 'dividend_yield'
        ]
        
        available_cols = [c for c in display_cols if c in formatted_df.columns]
        formatted_df = formatted_df[available_cols]
        
        return formatted_df
    
    def plot_comps_analysis(
        self, 
        target_ticker: str = None, 
        save_path: str = None
    ):
        """Visualize comparable company analysis"""
        if self.comps_data.empty:
            print("No data to plot")
            return None
        
        if target_ticker:
            target_ticker = self.config.normalize_ticker(target_ticker)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. EV/EBITDA Comparison
        ax1 = axes[0, 0]
        comps_sorted = self.comps_data.dropna(subset=['ev_ebitda']).sort_values('ev_ebitda')
        
        colors = ['red' if t == target_ticker else 'steelblue' for t in comps_sorted['ticker']]
        ax1.barh(comps_sorted['ticker'], comps_sorted['ev_ebitda'], color=colors, alpha=0.7)
        
        median_val = comps_sorted['ev_ebitda'].median()
        ax1.axvline(x=median_val, color='black', linestyle='--', 
                   label=f'Median: {median_val:.1f}x')
        ax1.set_xlabel('EV/EBITDA (x)')
        ax1.set_title('EV/EBITDA Multiples')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. P/E Ratio Comparison
        ax2 = axes[0, 1]
        comps_pe = self.comps_data.dropna(subset=['pe_ratio'])
        comps_pe = comps_pe[(comps_pe['pe_ratio'] > 0) & (comps_pe['pe_ratio'] < 100)]
        comps_pe_sorted = comps_pe.sort_values('pe_ratio')
        
        colors = ['red' if t == target_ticker else 'orange' for t in comps_pe_sorted['ticker']]
        ax2.barh(comps_pe_sorted['ticker'], comps_pe_sorted['pe_ratio'], color=colors, alpha=0.7)
        
        if not comps_pe_sorted.empty:
            median_pe = comps_pe_sorted['pe_ratio'].median()
            ax2.axvline(x=median_pe, color='black', linestyle='--', 
                       label=f'Median: {median_pe:.1f}x')
        ax2.set_xlabel('P/E Ratio (x)')
        ax2.set_title('P/E Multiples')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Market Cap Distribution
        ax3 = axes[0, 2]
        colors = ['red' if t == target_ticker else 'green' for t in self.comps_data['ticker']]
        ax3.bar(self.comps_data['ticker'], self.comps_data['market_cap_brl'], 
               color=colors, alpha=0.7)
        ax3.set_ylabel('Market Cap (BRL bi)')
        ax3.set_title('Market Capitalization')
        ax3.tick_params(axis='x', rotation=45)
        ax3.grid(True, alpha=0.3)
        
        # 4. ROE Comparison
        ax4 = axes[1, 0]
        comps_roe = self.comps_data.dropna(subset=['roe']).sort_values('roe')
        
        colors_roe = ['green' if roe > 0.15 else 'orange' if roe > 0 else 'red' 
                     for roe in comps_roe['roe']]
        
        ax4.barh(comps_roe['ticker'], comps_roe['roe'] * 100, color=colors_roe, alpha=0.7)
        ax4.axvline(x=15, color='blue', linestyle='--', alpha=0.5, label='15% ROE Target')
        ax4.set_xlabel('Return on Equity (%)')
        ax4.set_title('ROE Comparison')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. Debt/EBITDA Comparison
        ax5 = axes[1, 1]
        comps_debt = self.comps_data.dropna(subset=['debt_ebitda'])
        comps_debt = comps_debt[comps_debt['debt_ebitda'] < 10].sort_values('debt_ebitda')
        
        colors_debt = ['green' if debt < 3 else 'orange' if debt < 5 else 'red' 
                      for debt in comps_debt['debt_ebitda']]
        
        ax5.barh(comps_debt['ticker'], comps_debt['debt_ebitda'], color=colors_debt, alpha=0.7)
        ax5.axvline(x=3, color='blue', linestyle='--', alpha=0.5, label='3x Debt/EBITDA')
        ax5.set_xlabel('Debt/EBITDA (x)')
        ax5.set_title('Leverage Comparison')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Valuation Summary (if target provided)
        ax6 = axes[1, 2]
        
        if target_ticker:
            target_data = self.comps_data[self.comps_data['ticker'] == target_ticker]
            if not target_data.empty:
                target = target_data.iloc[0]
                
                valuation = self.calculate_comps_valuation(
                    target_ticker,
                    revenue=target.get('revenue_brl'),
                    ebitda=target.get('ebitda_brl'),
                    earnings=target.get('net_income_brl')
                )
                
                if valuation:
                    methods = list(valuation['valuations'].keys())
                    values = [
                        valuation['valuations'][m].get('implied_ev', 
                            valuation['valuations'][m].get('implied_equity', 0))
                        for m in methods
                    ]
                    
                    ax6.bar(methods, values, color='teal', alpha=0.7)
                    ax6.axhline(y=valuation['avg_implied_ev'], color='red', 
                               linestyle='--', label=f"Avg: {valuation['avg_implied_ev']:.1f}B")
                    ax6.set_ylabel('Implied Value (BRL Billions)')
                    ax6.set_title(f'{target_ticker} Valuation')
                    ax6.legend()
                    ax6.grid(True, alpha=0.3)
                else:
                    ax6.text(0.5, 0.5, 'Insufficient data\nfor valuation', 
                            ha='center', va='center', fontsize=12)
                    ax6.set_title('Valuation Summary')
            else:
                ax6.text(0.5, 0.5, f'Target {target_ticker}\nnot in dataset', 
                        ha='center', va='center', fontsize=12)
                ax6.set_title('Valuation Summary')
        else:
            # Show sector summary
            summary_text = f"Sector Statistics:\n\n"
            summary_text += f"Companies: {len(self.comps_data)}\n"
            summary_text += f"Median EV/EBITDA: {self.comps_data['ev_ebitda'].median():.1f}x\n"
            summary_text += f"Median P/E: {self.comps_data['pe_ratio'].median():.1f}x\n"
            summary_text += f"Median ROE: {self.comps_data['roe'].median()*100:.1f}%\n"
            
            ax6.text(0.5, 0.5, summary_text, ha='center', va='center', 
                    fontsize=11, family='monospace')
            ax6.set_title('Sector Summary')
            ax6.axis('off')
        
        sector = self.comps_data['sector'].iloc[0] if not self.comps_data.empty else 'Unknown'
        plt.suptitle(f'Comparable Company Analysis: Brazilian {sector.title()} Sector', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*100)
        print("COMPARABLE COMPANY ANALYSIS")
        print("="*100)
        
        table = self.generate_comps_table()
        if not table.empty:
            print("\n" + table.to_string(index=False))
        
        print("\n--- Sector Median Multiples ---")
        print(f"EV/EBITDA: {self.comps_data['ev_ebitda'].median():.1f}x")
        print(f"P/E Ratio: {self.comps_data['pe_ratio'].median():.1f}x")
        print(f"P/B Ratio: {self.comps_data['pb_ratio'].median():.1f}x")
        print(f"EV/Revenue: {self.comps_data['ev_revenue'].median():.1f}x")
        
        return fig


if __name__ == "__main__":
    config = get_config()
    
    print("="*80)
    print("BRAZILIAN COMPARABLE COMPANY ANALYSIS")
    print("="*80)
    
    comps = BrazilianCompsAnalysis()
    
    # Analyze banking sector
    target = 'ITUB4.SA'
    sector = 'banks'
    
    print(f"\nAnalyzing {sector} sector with target: {target}")
    
    if comps.fetch_sector_companies(sector, target_ticker=target):
        comps.plot_comps_analysis(target_ticker=target)
    
    # Analyze custom list
    print("\n" + "="*80)
    print("CUSTOM COMPARABLES")
    print("="*80)
    
    custom_comps = BrazilianCompsAnalysis()
    custom_tickers = config.get_watchlist_stocks()[:5]
    
    print(f"\nAnalyzing custom list: {custom_tickers}")
    
    if custom_comps.fetch_sector_companies('custom', custom_tickers=custom_tickers):
        table = custom_comps.generate_comps_table()
        print("\n" + table.to_string(index=False))
