# credit_analysis_brazil_updated.py
"""
Credit Analysis & Covenant Modeling for Brazilian Companies
Updated to use dynamic configuration and shared data manager
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from config import get_config, ConfigManager
from data_manager import get_data_manager, DataManager


class BrazilianCreditAnalysis:
    def __init__(
        self, 
        company_data: Dict = None,
        ticker: str = None,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        """
        Credit Analysis & Covenant Modeling for Brazilian Companies
        
        Parameters:
        -----------
        company_data : Dict
            Company financial data. If None, uses sample data or fetches from ticker.
        ticker : str
            Ticker to fetch data from (alternative to company_data)
        config : ConfigManager
            Configuration manager instance
        data_manager : DataManager
            Data manager instance
        """
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        if company_data is not None:
            self.company_data = company_data
        elif ticker:
            self.company_data = self._fetch_company_data(ticker)
        else:
            # Sample Brazilian company data
            self.company_data = {
                'name': 'Sample Brazilian Company',
                'ebitda': 2500,  # BRL millions
                'revenue': 8000,
                'total_debt': 6000,
                'cash': 800,
                'interest_expense': 480,
                'current_assets': 2000,
                'current_liabilities': 1200,
                'total_assets': 10000,
                'total_equity': 4000,
                'net_income': 800
            }
        
        # Brazilian market specific parameters
        self.selic_rate = 0.1065  # 10.65%
        self.cdi_rate = 0.104  # 10.4% (Brazilian interbank rate)
        self.country_risk_spread = 0.035  # 3.5%
    
    def _fetch_company_data(self, ticker: str) -> Dict:
        """Fetch company data from market"""
        ticker = self.config.normalize_ticker(ticker)
        info = self.data_manager.get_stock_info(ticker)
        
        if info:
            return {
                'name': info.get('longName', ticker),
                'ticker': ticker,
                'ebitda': info.get('ebitda', 0) / 1e6,
                'revenue': info.get('totalRevenue', 0) / 1e6,
                'total_debt': info.get('totalDebt', 0) / 1e6,
                'cash': info.get('totalCash', 0) / 1e6,
                'interest_expense': info.get('interestExpense', info.get('ebitda', 0) * 0.05) / 1e6,
                'current_assets': info.get('totalCurrentAssets', 0) / 1e6,
                'current_liabilities': info.get('totalCurrentLiabilities', 0) / 1e6,
                'total_assets': info.get('totalAssets', 0) / 1e6,
                'total_equity': info.get('totalStockholderEquity', 0) / 1e6,
                'net_income': info.get('netIncomeToCommon', 0) / 1e6
            }
        
        return self.company_data  # Return default if fetch fails
    
    def calculate_credit_ratios(self) -> Dict:
        """Calculate key credit ratios"""
        data = self.company_data
        
        ebitda = data.get('ebitda', 0)
        revenue = data.get('revenue', 0)
        total_debt = data.get('total_debt', 0)
        cash = data.get('cash', 0)
        interest = data.get('interest_expense', 1)  # Avoid division by zero
        current_assets = data.get('current_assets', 0)
        current_liabilities = data.get('current_liabilities', 1)
        total_assets = data.get('total_assets', 1)
        total_equity = data.get('total_equity', 1)
        net_income = data.get('net_income', 0)
        
        # Leverage ratios
        net_debt_ebitda = (total_debt - cash) / ebitda if ebitda > 0 else float('inf')
        total_debt_ebitda = total_debt / ebitda if ebitda > 0 else float('inf')
        debt_equity = total_debt / total_equity if total_equity > 0 else float('inf')
        debt_assets = total_debt / total_assets if total_assets > 0 else float('inf')
        
        # Coverage ratios
        interest_coverage = ebitda / interest if interest > 0 else float('inf')
        ebit_interest = (ebitda * 0.8) / interest if interest > 0 else float('inf')
        
        # Liquidity ratios
        current_ratio = current_assets / current_liabilities if current_liabilities > 0 else float('inf')
        quick_ratio = (current_assets * 0.7) / current_liabilities if current_liabilities > 0 else float('inf')
        cash_ratio = cash / current_liabilities if current_liabilities > 0 else float('inf')
        
        # Profitability ratios
        ebitda_margin = ebitda / revenue if revenue > 0 else 0
        roa = net_income / total_assets if total_assets > 0 else 0
        roe = net_income / total_equity if total_equity > 0 else 0
        
        # Brazil-specific
        cdi_coverage = (ebitda / (total_debt * self.cdi_rate)) if total_debt > 0 else float('inf')
        
        return {
            # Leverage
            'Net_Debt_EBITDA': net_debt_ebitda,
            'Total_Debt_EBITDA': total_debt_ebitda,
            'Debt_Equity': debt_equity,
            'Debt_Assets': debt_assets,
            # Coverage
            'Interest_Coverage': interest_coverage,
            'EBIT_Interest': ebit_interest,
            # Liquidity
            'Current_Ratio': current_ratio,
            'Quick_Ratio': quick_ratio,
            'Cash_Ratio': cash_ratio,
            # Profitability
            'EBITDA_Margin': ebitda_margin,
            'ROA': roa,
            'ROE': roe,
            # Brazil-specific
            'CDI_Coverage': cdi_coverage
        }
    
    def assess_credit_risk(self, ratios: Dict = None) -> Dict:
        """Assess credit risk based on ratios"""
        if ratios is None:
            ratios = self.calculate_credit_ratios()
        
        thresholds = {
            'Net_Debt_EBITDA': {
                'Investment_Grade': (0, 2.5),
                'Speculative': (2.5, 4.0),
                'Distressed': (4.0, float('inf'))
            },
            'Interest_Coverage': {
                'Investment_Grade': (5.0, float('inf')),
                'Speculative': (2.0, 5.0),
                'Distressed': (0, 2.0)
            },
            'Current_Ratio': {
                'Strong': (1.5, float('inf')),
                'Adequate': (1.0, 1.5),
                'Weak': (0, 1.0)
            },
            'EBITDA_Margin': {
                'Strong': (0.25, float('inf')),
                'Average': (0.15, 0.25),
                'Weak': (0, 0.15)
            }
        }
        
        risk_assessment = {}
        
        for ratio_name, value in ratios.items():
            if ratio_name in thresholds:
                for category, (low, high) in thresholds[ratio_name].items():
                    if low <= value < high:
                        risk_assessment[ratio_name] = category
                        break
            else:
                risk_assessment[ratio_name] = 'N/A'
        
        return risk_assessment
    
    def calculate_credit_score(self, ratios: Dict = None) -> Dict:
        """Calculate composite credit score"""
        if ratios is None:
            ratios = self.calculate_credit_ratios()
        
        # Scoring weights
        weights = {
            'Net_Debt_EBITDA': 25,
            'Interest_Coverage': 20,
            'Current_Ratio': 15,
            'EBITDA_Margin': 15,
            'Debt_Equity': 10,
            'ROE': 10,
            'Cash_Ratio': 5
        }
        
        # Scoring functions (higher score = better)
        def score_leverage(value):
            if value < 2: return 100
            elif value < 3: return 80
            elif value < 4: return 60
            elif value < 5: return 40
            else: return 20
        
        def score_coverage(value):
            if value > 6: return 100
            elif value > 4: return 80
            elif value > 2: return 60
            elif value > 1: return 40
            else: return 20
        
        def score_liquidity(value):
            if value > 2: return 100
            elif value > 1.5: return 80
            elif value > 1: return 60
            elif value > 0.8: return 40
            else: return 20
        
        def score_margin(value):
            if value > 0.30: return 100
            elif value > 0.20: return 80
            elif value > 0.15: return 60
            elif value > 0.10: return 40
            else: return 20
        
        def score_roe(value):
            if value > 0.20: return 100
            elif value > 0.15: return 80
            elif value > 0.10: return 60
            elif value > 0.05: return 40
            else: return 20
        
        scores = {
            'Net_Debt_EBITDA': score_leverage(ratios.get('Net_Debt_EBITDA', 0)),
            'Interest_Coverage': score_coverage(ratios.get('Interest_Coverage', 0)),
            'Current_Ratio': score_liquidity(ratios.get('Current_Ratio', 0)),
            'EBITDA_Margin': score_margin(ratios.get('EBITDA_Margin', 0)),
            'Debt_Equity': score_leverage(ratios.get('Debt_Equity', 0)),
            'ROE': score_roe(ratios.get('ROE', 0)),
            'Cash_Ratio': score_liquidity(ratios.get('Cash_Ratio', 0) * 2)  # Scale up
        }
        
        # Calculate weighted score
        total_weight = sum(weights.values())
        weighted_score = sum(scores[k] * weights.get(k, 0) for k in scores) / total_weight
        
        # Determine rating
        if weighted_score >= 80:
            rating = 'AAA/AA'
            outlook = 'Investment Grade - Strong'
        elif weighted_score >= 70:
            rating = 'A/BBB'
            outlook = 'Investment Grade'
        elif weighted_score >= 60:
            rating = 'BB'
            outlook = 'Speculative Grade'
        elif weighted_score >= 50:
            rating = 'B'
            outlook = 'Highly Speculative'
        else:
            rating = 'CCC/D'
            outlook = 'Distressed'
        
        return {
            'scores': scores,
            'weighted_score': weighted_score,
            'rating': rating,
            'outlook': outlook
        }
    
    def model_financial_covenants(
        self, 
        ebitda_growth: float = 0.05, 
        revenue_growth: float = 0.06,
        capex_ratio: float = 0.08, 
        dividend_payout: float = 0.3
    ) -> pd.DataFrame:
        """Model financial covenants over projection period"""
        years = 5
        projections = []
        
        ebitda = self.company_data['ebitda']
        revenue = self.company_data['revenue']
        total_debt = self.company_data['total_debt']
        cash = self.company_data['cash']
        
        for year in range(1, years + 1):
            # Project financials
            proj_revenue = revenue * ((1 + revenue_growth) ** year)
            proj_ebitda = ebitda * ((1 + ebitda_growth) ** year)
            
            # Capex and FCF
            capex = proj_revenue * capex_ratio
            fcf = proj_ebitda * 0.7 - capex
            
            # Debt amortization (assume 10% per year)
            debt_payment = total_debt * 0.10
            proj_debt = total_debt - debt_payment * year
            
            # Interest expense (CDI + spread)
            interest_rate = self.cdi_rate + 0.025
            interest_expense = proj_debt * interest_rate
            
            # Cash accumulation
            proj_cash = cash + fcf * year * 0.5  # Simplified
            
            # Calculate covenant metrics
            net_debt_ebitda = (proj_debt - proj_cash) / proj_ebitda
            interest_coverage = proj_ebitda / interest_expense
            debt_equity = proj_debt / self.company_data['total_equity']
            
            projections.append({
                'Year': year,
                'Revenue': proj_revenue,
                'EBITDA': proj_ebitda,
                'Total_Debt': proj_debt,
                'Cash': proj_cash,
                'Net_Debt': proj_debt - proj_cash,
                'Interest_Expense': interest_expense,
                'FCF': fcf,
                'Net_Debt_EBITDA': net_debt_ebitda,
                'Interest_Coverage': interest_coverage,
                'Debt_Equity': debt_equity,
                'Covenant_Compliant': net_debt_ebitda < 3.5 and interest_coverage > 3.0
            })
        
        return pd.DataFrame(projections)
    
    def stress_test_covenants(self, scenarios: Dict = None) -> pd.DataFrame:
        """Stress test covenants under different scenarios"""
        if scenarios is None:
            scenarios = {
                'Base Case': {'ebitda_shock': 0, 'rate_increase': 0},
                'Mild Stress': {'ebitda_shock': -0.10, 'rate_increase': 0.02},
                'Moderate Stress': {'ebitda_shock': -0.20, 'rate_increase': 0.03},
                'Severe Stress': {'ebitda_shock': -0.30, 'rate_increase': 0.05},
                'Recession': {'ebitda_shock': -0.40, 'rate_increase': 0.04}
            }
        
        results = []
        
        for scenario_name, params in scenarios.items():
            ebitda_shock = params.get('ebitda_shock', 0)
            rate_increase = params.get('rate_increase', 0)
            
            # Apply stress
            stressed_ebitda = self.company_data['ebitda'] * (1 + ebitda_shock)
            stressed_rate = self.cdi_rate + 0.025 + rate_increase
            stressed_interest = self.company_data['total_debt'] * stressed_rate
            
            # Calculate stressed ratios
            net_debt = self.company_data['total_debt'] - self.company_data['cash']
            net_debt_ebitda = net_debt / stressed_ebitda if stressed_ebitda > 0 else float('inf')
            interest_coverage = stressed_ebitda / stressed_interest if stressed_interest > 0 else float('inf')
            
            # Covenant thresholds
            covenant_breach = net_debt_ebitda > 3.5 or interest_coverage < 3.0
            
            results.append({
                'Scenario': scenario_name,
                'EBITDA_Shock': f"{ebitda_shock*100:.0f}%",
                'Rate_Increase': f"+{rate_increase*100:.0f}bps",
                'Stressed_EBITDA': stressed_ebitda,
                'Stressed_Interest': stressed_interest,
                'Net_Debt_EBITDA': net_debt_ebitda,
                'Interest_Coverage': interest_coverage,
                'Covenant_Breach': 'YES' if covenant_breach else 'NO',
                'Risk_Level': 'High' if covenant_breach else 'Low'
            })
        
        return pd.DataFrame(results)
    
    def calculate_pd_lgd(self, rating_category: str = 'Speculative') -> Dict:
        """Calculate Probability of Default and Loss Given Default"""
        pd_by_rating = {
            'Investment_Grade': {'PD': 0.005, 'LGD': 0.40},
            'Speculative': {'PD': 0.025, 'LGD': 0.55},
            'Distressed': {'PD': 0.10, 'LGD': 0.70}
        }
        
        # Adjust for Brazilian country risk
        country_adjustment = 1.3  # 30% higher PD in Brazil
        
        base = pd_by_rating.get(rating_category, {'PD': 0.05, 'LGD': 0.60})
        
        adjusted_pd = base['PD'] * country_adjustment
        adjusted_lgd = base['LGD'] * 1.1  # 10% higher LGD
        
        # Expected Loss
        total_debt = self.company_data['total_debt']
        expected_loss = adjusted_pd * adjusted_lgd * total_debt
        
        return {
            'rating_category': rating_category,
            'base_pd': base['PD'],
            'adjusted_pd': adjusted_pd,
            'base_lgd': base['LGD'],
            'adjusted_lgd': adjusted_lgd,
            'exposure': total_debt,
            'expected_loss': expected_loss
        }
    
    def plot_credit_analysis(self, save_path: str = None):
        """Visualize credit analysis"""
        ratios = self.calculate_credit_ratios()
        assessment = self.assess_credit_risk(ratios)
        credit_score = self.calculate_credit_score(ratios)
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. Credit Ratios Bar Chart
        ax1 = axes[0, 0]
        key_ratios = ['Net_Debt_EBITDA', 'Interest_Coverage', 'Current_Ratio', 'EBITDA_Margin']
        values = [ratios.get(r, 0) for r in key_ratios]
        
        # Normalize for display
        display_values = [min(v, 10) if i < 2 else min(v * 100, 50) for i, v in enumerate(values)]
        
        colors = []
        for r in key_ratios:
            cat = assessment.get(r, 'N/A')
            if cat in ['Investment_Grade', 'Strong']:
                colors.append('green')
            elif cat in ['Speculative', 'Adequate', 'Average']:
                colors.append('orange')
            else:
                colors.append('red')
        
        bars = ax1.bar(key_ratios, display_values, color=colors, alpha=0.7)
        ax1.set_ylabel('Value')
        ax1.set_title('Key Credit Ratios')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        for bar, val in zip(bars, values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{val:.2f}', ha='center', fontsize=9)
        
        # 2. Credit Score Gauge
        ax2 = axes[0, 1]
        score = credit_score['weighted_score']
        
        # Create gauge visualization
        theta = np.linspace(0, np.pi, 100)
        r = 1
        
        # Background arc
        for i, (color, start, end) in enumerate([
            ('red', 0, 0.5), 
            ('orange', 0.5, 0.7), 
            ('green', 0.7, 1.0)
        ]):
            ax2.fill_between(
                np.linspace(start*np.pi, end*np.pi, 30),
                0.6, 1.0,
                alpha=0.3, color=color
            )
        
        # Score indicator
        score_angle = (score / 100) * np.pi
        ax2.arrow(0, 0, np.cos(np.pi - score_angle) * 0.5, 
                 np.sin(np.pi - score_angle) * 0.5, 
                 head_width=0.1, color='black')
        
        ax2.set_xlim(-1.2, 1.2)
        ax2.set_ylim(-0.2, 1.2)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f'Credit Score: {score:.0f}/100\nRating: {credit_score["rating"]}')
        
        # 3. Covenant Projections
        ax3 = axes[0, 2]
        projections = self.model_financial_covenants()
        
        ax3.plot(projections['Year'], projections['Net_Debt_EBITDA'], 
                'b-o', label='Net Debt/EBITDA', linewidth=2)
        ax3.axhline(y=3.5, color='red', linestyle='--', label='Covenant (3.5x)')
        ax3.set_xlabel('Year')
        ax3.set_ylabel('Net Debt/EBITDA')
        ax3.set_title('Covenant Projection')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Stress Test Results
        ax4 = axes[1, 0]
        stress_results = self.stress_test_covenants()
        
        scenarios = stress_results['Scenario'].tolist()
        coverage = stress_results['Interest_Coverage'].tolist()
        colors = ['green' if c > 3 else 'orange' if c > 2 else 'red' for c in coverage]
        
        ax4.barh(scenarios, coverage, color=colors, alpha=0.7)
        ax4.axvline(x=3.0, color='red', linestyle='--', label='Covenant (3.0x)')
        ax4.set_xlabel('Interest Coverage')
        ax4.set_title('Stress Test: Interest Coverage')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # 5. PD/LGD Analysis
        ax5 = axes[1, 1]
        
        categories = ['Investment_Grade', 'Speculative', 'Distressed']
        pds = []
        lgds = []
        
        for cat in categories:
            result = self.calculate_pd_lgd(cat)
            pds.append(result['adjusted_pd'] * 100)
            lgds.append(result['adjusted_lgd'] * 100)
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax5.bar(x - width/2, pds, width, label='PD (%)', color='coral')
        ax5.bar(x + width/2, lgds, width, label='LGD (%)', color='steelblue')
        ax5.set_xticks(x)
        ax5.set_xticklabels(['Inv Grade', 'Speculative', 'Distressed'])
        ax5.set_ylabel('Percentage (%)')
        ax5.set_title('PD & LGD by Rating (Brazil-Adjusted)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Summary Box
        ax6 = axes[1, 2]
        
        summary = f"""
CREDIT ANALYSIS SUMMARY
═══════════════════════════════

Company: {self.company_data.get('name', 'N/A')}

Key Metrics:
  Net Debt/EBITDA: {ratios['Net_Debt_EBITDA']:.2f}x
  Interest Coverage: {ratios['Interest_Coverage']:.2f}x
  Current Ratio: {ratios['Current_Ratio']:.2f}x
  EBITDA Margin: {ratios['EBITDA_Margin']*100:.1f}%

Credit Score: {credit_score['weighted_score']:.0f}/100
Rating: {credit_score['rating']}
Outlook: {credit_score['outlook']}

Brazilian Market:
  SELIC Rate: {self.selic_rate*100:.2f}%
  CDI Rate: {self.cdi_rate*100:.2f}%
  Country Risk: {self.country_risk_spread*100:.1f}%
"""
        
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax6.axis('off')
        
        plt.suptitle('Credit Analysis & Covenant Modeling - Brazilian Company', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
        
        return fig, ratios, credit_score


if __name__ == "__main__":
    config = get_config()
    
    print("="*80)
    print("BRAZILIAN CREDIT ANALYSIS")
    print("="*80)
    
    # Example with sample data
    credit = BrazilianCreditAnalysis()
    
    fig, ratios, score = credit.plot_credit_analysis()
    
    # Stress test
    print("\n--- Covenant Stress Test ---")
    stress = credit.stress_test_covenants()
    print(stress.to_string(index=False))
    
    # Example with real ticker
    print("\n" + "="*80)
    print("MARKET DATA ANALYSIS")
    print("="*80)
    
    default_ticker = config.get_default_ticker()
    print(f"\nAnalyzing: {default_ticker}")
    
    market_credit = BrazilianCreditAnalysis(ticker=default_ticker)
    market_ratios = market_credit.calculate_credit_ratios()
    market_score = market_credit.calculate_credit_score(market_ratios)
    
    print(f"\nCredit Score: {market_score['weighted_score']:.0f}")
    print(f"Rating: {market_score['rating']}")
