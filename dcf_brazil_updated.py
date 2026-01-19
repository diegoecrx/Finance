# dcf_brazil_updated.py
"""
Discounted Cash Flow Analysis for Brazilian Stocks
Updated to use dynamic configuration and shared data manager
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from config import get_config, ConfigManager
from data_manager import get_data_manager, DataManager


class BrazilianDCF:
    def __init__(
        self, 
        ticker: str = None,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        """
        DCF Analysis for Brazilian Stocks
        
        Parameters:
        -----------
        ticker : str
            Stock ticker (e.g., 'PETR4' or 'PETR4.SA'). If None, uses default from config.
        config : ConfigManager
            Configuration manager instance
        data_manager : DataManager
            Data manager instance for fetching/caching data
        """
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        if ticker is None:
            ticker = self.config.get_default_ticker()
        
        self.ticker = self.config.normalize_ticker(ticker)
        self.company = None
        self.financials = {}
        
    def fetch_data(self) -> bool:
        """Fetch Brazilian stock data and financials"""
        print(f"Fetching data for {self.ticker}...")
        
        try:
            self.company = yf.Ticker(self.ticker)
            info = self.company.info
            
            # Store historical data using data manager
            historical = self.data_manager.fetch_historical_data(self.ticker, period='2y')
            
            if historical is None or historical.empty:
                print(f"Warning: No historical data for {self.ticker}")
            
            # Extract financials
            self.financials = {
                'market_cap': info.get('marketCap', 0),
                'current_price': info.get('currentPrice', info.get('regularMarketPrice', 0)),
                'beta': info.get('beta', 1.0),
                'shares_outstanding': info.get('sharesOutstanding', 0),
                'currency': info.get('currency', 'BRL'),
                'profit_margins': info.get('profitMargins', 0.10),
                'revenue': info.get('totalRevenue', 0),
                'ebit': info.get('ebit', 0),
                'net_income': info.get('netIncomeToCommon', 0),
                'total_debt': info.get('totalDebt', 0),
                'cash': info.get('totalCash', 0),
                'free_cash_flow': info.get('freeCashflow', 0),
                'enterprise_value': info.get('enterpriseValue', 0),
                'ebitda': info.get('ebitda', 0),
            }
            
            # Estimate missing data if needed
            if self.financials['revenue'] == 0 or self.financials['ebit'] == 0:
                self._estimate_financials()
            
            print(f"Successfully loaded data for {self.ticker}")
            return True
            
        except Exception as e:
            print(f"Error fetching data: {e}")
            return False
    
    def _estimate_financials(self):
        """Estimate financials when data is limited"""
        market_cap = self.financials.get('market_cap', 1e9)
        
        if self.financials.get('revenue', 0) == 0:
            self.financials['revenue'] = market_cap / 2
        
        if self.financials.get('ebit', 0) == 0:
            self.financials['ebit'] = self.financials['revenue'] * 0.15
        
        if self.financials.get('net_income', 0) == 0:
            self.financials['net_income'] = self.financials['ebit'] * 0.66
        
        if self.financials.get('total_debt', 0) == 0:
            self.financials['total_debt'] = market_cap * 0.4
        
        if self.financials.get('cash', 0) == 0:
            self.financials['cash'] = market_cap * 0.1
        
        if self.financials.get('free_cash_flow', 0) == 0:
            self.financials['free_cash_flow'] = self.financials['net_income'] * 0.8
        
        self.financials['capex'] = self.financials['revenue'] * 0.08
    
    def calculate_wacc(
        self, 
        risk_free_rate: float = 0.1065, 
        market_risk_premium: float = 0.055
    ) -> float:
        """
        Calculate Weighted Average Cost of Capital for Brazilian company
        
        Parameters:
        -----------
        risk_free_rate : float
            Brazilian risk-free rate (SELIC ~10.65%)
        market_risk_premium : float
            Brazil market risk premium (~5.5%)
            
        Returns:
        --------
        WACC as decimal
        """
        rf = risk_free_rate
        
        # Cost of equity (CAPM)
        beta = self.financials.get('beta', 1.0)
        if beta is None or beta <= 0:
            beta = 1.0
        
        cost_of_equity = rf + beta * market_risk_premium
        
        # Cost of debt (estimate: risk-free + spread)
        cost_of_debt = rf + 0.04  # Company spread over risk-free
        
        # Weights
        market_cap = self.financials.get('market_cap', 1e9)
        total_debt = self.financials.get('total_debt', market_cap * 0.4)
        
        total_capital = market_cap + total_debt
        if total_capital == 0:
            return 0.12  # Default WACC for Brazil
        
        equity_weight = market_cap / total_capital
        debt_weight = total_debt / total_capital
        
        # Brazilian corporate tax rate
        tax_rate = 0.34
        
        # WACC formula
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        
        return wacc
    
    def project_cash_flows(
        self, 
        years: int = 5, 
        growth_rate: float = 0.07, 
        margin_change: float = 0.01
    ) -> pd.DataFrame:
        """
        Project free cash flows
        
        Parameters:
        -----------
        years : int
            Projection period
        growth_rate : float
            Revenue growth rate
        margin_change : float
            Annual margin improvement
            
        Returns:
        --------
        DataFrame with projections
        """
        current_revenue = self.financials.get('revenue', 1e9)
        current_margin = self.financials.get('profit_margins', 0.15)
        if current_margin is None or current_margin <= 0:
            current_margin = 0.15
        
        capex_ratio = 0.08
        nwc_ratio = 0.05
        depreciation_ratio = 0.05
        tax_rate = 0.34
        
        projections = []
        
        for year in range(1, years + 1):
            # Growing revenue with declining growth
            year_growth = growth_rate * (1 - 0.1 * (year - 1))
            revenue = current_revenue * ((1 + year_growth) ** year)
            
            # Improving margins
            margin = min(current_margin + margin_change * year, 0.35)
            
            # Calculate EBIT and adjustments
            ebit = revenue * margin
            depreciation = revenue * depreciation_ratio
            capex = revenue * capex_ratio
            nwc_change = revenue * nwc_ratio * 0.1  # Only change, not absolute
            
            # Tax expense
            taxes = ebit * tax_rate
            
            # Free Cash Flow to Firm
            fcff = ebit - taxes + depreciation - capex - nwc_change
            
            projections.append({
                'year': year,
                'revenue': revenue,
                'ebit': ebit,
                'ebit_margin': margin,
                'depreciation': depreciation,
                'capex': capex,
                'nwc_change': nwc_change,
                'taxes': taxes,
                'fcff': fcff
            })
        
        return pd.DataFrame(projections)
    
    def calculate_terminal_value(
        self, 
        last_fcf: float, 
        wacc: float, 
        perpetual_growth: float = 0.025
    ) -> float:
        """
        Calculate terminal value using Gordon Growth Model
        """
        if wacc <= perpetual_growth:
            perpetual_growth = wacc - 0.02
        
        terminal_value = last_fcf * (1 + perpetual_growth) / (wacc - perpetual_growth)
        return terminal_value
    
    def perform_dcf(self) -> Optional[Dict]:
        """Complete DCF valuation"""
        if not self.fetch_data():
            return None
        
        # Calculate WACC
        wacc = self.calculate_wacc()
        
        # Project cash flows
        projections = self.project_cash_flows(years=5)
        
        # Discount projected cash flows
        discounted_cf = []
        for i, row in projections.iterrows():
            year = row['year']
            dcf = row['fcff'] / ((1 + wacc) ** year)
            discounted_cf.append(dcf)
        
        # Calculate terminal value
        last_fcf = projections.iloc[-1]['fcff']
        terminal_value = self.calculate_terminal_value(last_fcf, wacc)
        discounted_tv = terminal_value / ((1 + wacc) ** 5)
        
        # Enterprise value
        enterprise_value = sum(discounted_cf) + discounted_tv
        
        # Equity value
        cash = self.financials.get('cash', 0)
        debt = self.financials.get('total_debt', 0)
        equity_value = enterprise_value + cash - debt
        
        # Per share value
        shares = self.financials.get('shares_outstanding', 1)
        if shares and shares > 0:
            fair_value_per_share = equity_value / shares
        else:
            fair_value_per_share = 0
        
        # Current price
        current_price = self.financials.get('current_price', 0)
        
        # Handle currency
        if self.financials.get('currency') == 'BRL':
            current_price_brl = current_price
            fair_value_brl = fair_value_per_share
        else:
            exchange_rate = 5.0  # BRL/USD approximate
            current_price_brl = current_price * exchange_rate
            fair_value_brl = fair_value_per_share * exchange_rate
        
        # Upside calculation
        if current_price_brl > 0:
            upside = ((fair_value_brl / current_price_brl) - 1) * 100
        else:
            upside = 0
        
        # Recommendation
        if fair_value_brl > current_price_brl * 1.15:
            recommendation = 'STRONG BUY'
        elif fair_value_brl > current_price_brl * 1.05:
            recommendation = 'BUY'
        elif fair_value_brl < current_price_brl * 0.85:
            recommendation = 'STRONG SELL'
        elif fair_value_brl < current_price_brl * 0.95:
            recommendation = 'SELL'
        else:
            recommendation = 'HOLD'
        
        results = {
            'ticker': self.ticker,
            'current_price_brl': current_price_brl,
            'fair_value_brl': fair_value_brl,
            'upside_pct': upside,
            'wacc': wacc,
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'terminal_value': terminal_value,
            'recommendation': recommendation,
            'projections': projections
        }
        
        return results
    
    def sensitivity_analysis(
        self, 
        wacc_range: tuple = (-0.02, 0.02), 
        growth_range: tuple = (-0.01, 0.01),
        steps: int = 5
    ) -> pd.DataFrame:
        """Perform sensitivity analysis on WACC and growth rate"""
        if not hasattr(self, 'financials') or not self.financials:
            if not self.fetch_data():
                return None
        
        base_wacc = self.calculate_wacc()
        base_growth = 0.025
        
        wacc_values = np.linspace(
            base_wacc + wacc_range[0], 
            base_wacc + wacc_range[1], 
            steps
        )
        growth_values = np.linspace(
            base_growth + growth_range[0], 
            base_growth + growth_range[1], 
            steps
        )
        
        sensitivity_matrix = np.zeros((steps, steps))
        
        projections = self.project_cash_flows(years=5)
        last_fcf = projections.iloc[-1]['fcff']
        
        discounted_cf_sum = sum([
            row['fcff'] / ((1 + base_wacc) ** row['year']) 
            for _, row in projections.iterrows()
        ])
        
        for i, wacc in enumerate(wacc_values):
            for j, growth in enumerate(growth_values):
                if wacc <= growth:
                    growth = wacc - 0.01
                
                tv = last_fcf * (1 + growth) / (wacc - growth)
                discounted_tv = tv / ((1 + wacc) ** 5)
                
                ev = discounted_cf_sum + discounted_tv
                eq_value = ev + self.financials.get('cash', 0) - self.financials.get('total_debt', 0)
                
                shares = self.financials.get('shares_outstanding', 1)
                if shares and shares > 0:
                    fair_value = eq_value / shares
                else:
                    fair_value = 0
                
                sensitivity_matrix[i, j] = fair_value
        
        return pd.DataFrame(
            sensitivity_matrix,
            index=[f"{w*100:.1f}%" for w in wacc_values],
            columns=[f"{g*100:.1f}%" for g in growth_values]
        )
    
    def plot_dcf_results(self, results: Dict = None, save_path: str = None):
        """Visualize DCF results"""
        if results is None:
            results = self.perform_dcf()
        
        if results is None:
            print("No results to plot")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        projections = results['projections']
        
        # 1. Revenue and FCF Projections
        ax1 = axes[0, 0]
        years = projections['year']
        revenues = projections['revenue'] / 1e9
        fcfs = projections['fcff'] / 1e9
        
        ax1.plot(years, revenues, 'b-o', label='Revenue (BRL bi)', linewidth=2, markersize=8)
        ax1.plot(years, fcfs, 'g-s', label='FCF (BRL bi)', linewidth=2, markersize=8)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('BRL Billions')
        ax1.set_title(f'{self.ticker} - Revenue & FCF Projections')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Valuation Comparison
        ax2 = axes[0, 1]
        labels = ['Current Price', 'Fair Value']
        values = [results['current_price_brl'], results['fair_value_brl']]
        
        if results['upside_pct'] >= 0:
            colors = ['gray', 'green']
        else:
            colors = ['gray', 'red']
        
        bars = ax2.bar(labels, values, color=colors, alpha=0.7, edgecolor='black')
        ax2.set_ylabel('BRL per Share')
        ax2.set_title(f'Valuation - Upside: {results["upside_pct"]:.1f}%\nRecommendation: {results["recommendation"]}')
        
        for bar, value in zip(bars, values):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    f'R${value:.2f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Margin Evolution
        ax3 = axes[1, 0]
        margins = projections['ebit_margin'] * 100
        ax3.bar(years, margins, color='teal', alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('EBIT Margin (%)')
        ax3.set_title('Projected EBIT Margin')
        ax3.grid(True, alpha=0.3)
        
        for i, (year, margin) in enumerate(zip(years, margins)):
            ax3.text(year, margin + 0.3, f'{margin:.1f}%', ha='center', fontsize=10)
        
        # 4. Sensitivity Heatmap
        ax4 = axes[1, 1]
        sensitivity = self.sensitivity_analysis()
        
        if sensitivity is not None:
            sns.heatmap(sensitivity, annot=True, fmt='.1f', cmap='RdYlGn', 
                       ax=ax4, cbar_kws={'label': 'Fair Value (BRL)'})
            ax4.set_xlabel('Terminal Growth Rate')
            ax4.set_ylabel('WACC')
            ax4.set_title('Sensitivity Analysis')
        
        plt.suptitle(f'DCF Valuation Analysis: {self.ticker}', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print(f"DCF VALUATION SUMMARY: {self.ticker}")
        print("="*80)
        print(f"\nCurrent Price: R$ {results['current_price_brl']:.2f}")
        print(f"Fair Value: R$ {results['fair_value_brl']:.2f}")
        print(f"Upside: {results['upside_pct']:.1f}%")
        print(f"WACC: {results['wacc']*100:.1f}%")
        print(f"Enterprise Value: R$ {results['enterprise_value']/1e9:.2f} billion")
        print(f"Equity Value: R$ {results['equity_value']/1e9:.2f} billion")
        print(f"\n*** RECOMMENDATION: {results['recommendation']} ***")
        
        return fig


def analyze_multiple_stocks(
    tickers: List[str] = None,
    config: ConfigManager = None
) -> pd.DataFrame:
    """Analyze multiple stocks with DCF"""
    config = config or get_config()
    
    if tickers is None:
        tickers = config.get_watchlist_stocks()
    
    results = []
    
    for ticker in tickers:
        print(f"\n--- Analyzing {ticker} ---")
        dcf = BrazilianDCF(ticker=ticker, config=config)
        result = dcf.perform_dcf()
        
        if result:
            results.append({
                'Ticker': result['ticker'],
                'Current Price': result['current_price_brl'],
                'Fair Value': result['fair_value_brl'],
                'Upside %': result['upside_pct'],
                'WACC %': result['wacc'] * 100,
                'Recommendation': result['recommendation']
            })
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    config = get_config()
    
    print("="*80)
    print("BRAZILIAN DCF VALUATION ANALYSIS")
    print("="*80)
    
    # Single stock analysis using default ticker
    dcf = BrazilianDCF()
    results = dcf.perform_dcf()
    
    if results:
        dcf.plot_dcf_results(results)
    
    # Multiple stock analysis
    print("\n" + "="*80)
    print("WATCHLIST DCF ANALYSIS")
    print("="*80)
    
    summary = analyze_multiple_stocks()
    if not summary.empty:
        print("\n" + summary.to_string(index=False))
