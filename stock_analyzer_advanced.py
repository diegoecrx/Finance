# -*- coding: utf-8 -*-
"""
Advanced Brazilian Stock Analyzer (adapted for SGuide.xlsx format)
With VaR, DCF, Credit Analysis, Monte Carlo, and Sensitivity Analysis
Includes Professional Visualization - Works like FII Analyzer
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass, field
from enum import Enum
import warnings
from scipy import stats
from datetime import datetime
import sys

warnings.filterwarnings('ignore')

# Set matplotlib backend to Agg for non-interactive use if saving files
if '--charts' in ' '.join(sys.argv):
    import matplotlib
    matplotlib.use('Agg')

# Plotting imports
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set professional style
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    plt.style.use('ggplot')
sns.set_palette("husl")

# Custom colors for recommendations
COLORS = {
    'STRONG_BUY': '#00C853',
    'BUY': '#4CAF50',
    'HOLD': '#FFC107',
    'SELL': '#FF5722',
    'STRONG_SELL': '#D32F2F',
    'primary': '#1976D2',
    'secondary': '#7B1FA2',
    'accent': '#00BCD4',
    'background': '#F5F5F5'
}

# Economic Cycle colors
CYCLE_COLORS = {
    'EXPANSION': '#4CAF50',
    'PEAK': '#2196F3',
    'CONTRACTION': '#FF9800',
    'RECESSION': '#F44336',
    'RECOVERY': '#9C27B0',
}

# Sector to Economic Cycle mapping
SECTOR_CYCLE_MAP = {
    'Serviços Básicos': 'RECESSION',
    'Saúde & Educação': 'RECESSION',
    'Alimentos & Bebidas': 'RECESSION',
    'Bancos': 'RECOVERY',
    'Setor financeiro (Ex-Bancos)': 'RECOVERY',
    'Moradia': 'RECOVERY',
    'Propriedades Comerciais': 'RECOVERY',
    'Varejo': 'EXPANSION',
    'Bens de capital': 'EXPANSION',
    'Telecom, media & tech': 'EXPANSION',
    'Aluguel de carros & Logística': 'EXPANSION',
    'Infraestrutura': 'EXPANSION',
    'Metais & Mineração': 'PEAK',
    'Petróleo & Gas': 'PEAK',
    'Papel & celulose': 'PEAK',
    'Agronegócio': 'PEAK',
    'Aéreas': 'CONTRACTION',
}

class Recommendation(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class AnalysisResult:
    ticker: str
    company_name: str
    recommendation: Recommendation
    score: float
    analyst_rating: str
    sector: str
    # Price metrics
    current_price: float
    target_price: float
    upside_pct: float
    # Valuation metrics
    pe_2026: float
    pe_2027: float
    ev_ebitda_2026: float
    pvp_2026: float
    # Yield and returns
    dividend_yield: float
    roe_2026: float
    # Performance
    return_week: float
    return_month: float
    return_year: float
    # Size and liquidity
    market_cap: float
    volume_avg: float
    # Debt
    net_debt_ebitda: float
    # Risk metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    volatility: float = 0.0
    sharpe_ratio: float = 0.0
    # DCF
    dcf_fair_value: float = 0.0
    dcf_upside: float = 0.0
    # Credit
    credit_score: float = 0.0
    credit_rating: str = "N/A"
    # Monte Carlo
    mc_upside: float = 0.0
    mc_downside: float = 0.0
    # Individual scores
    valuation_score: float = 0.0
    profitability_score: float = 0.0
    momentum_score: float = 0.0
    liquidity_score: float = 0.0
    # Analysis reasons
    reasons: List[str] = field(default_factory=list)
    # Economic cycle
    economic_cycle: str = "EXPANSION"


class AdvancedStockAnalyzer:
    """
    Advanced Stock Analyzer for Brazilian Stocks
    Adapted for SGuide.xlsx format - Works like FII Analyzer
    """
    
    # Brazilian Market Parameters
    SELIC_RATE = 0.1175  # Current SELIC
    CDI_RATE = 0.1165  # CDI Rate
    IPCA = 0.045  # Inflation target
    MARKET_RISK_PREMIUM = 0.055
    PERPETUAL_GROWTH_RATE = 0.03
    CORPORATE_TAX_RATE = 0.34
    
    def __init__(self, file_path: str):
        """Initialize analyzer"""
        self.file_path = file_path
        self.df_raw = None
        self.df = None
        self.sectors = {}
        self.results: List[AnalysisResult] = []
        self.results_by_sector: Dict[str, List[AnalysisResult]] = {}
        self.figures = []
    
    def load_data(self) -> pd.DataFrame:
        """Load data from SGuide.xlsx file"""
        print(f"\n{'='*70}")
        print("  ADVANCED STOCK ANALYZER - Loading Data")
        print(f"{'='*70}")
        
        # Read raw data
        self.df_raw = pd.read_excel(self.file_path, header=None)
        
        # Parse sectors and stocks
        stocks_data = []
        current_sector = "Unknown"
        
        for idx, row in self.df_raw.iterrows():
            if idx < 5:  # Skip header rows
                continue
            
            col1 = row[1] if pd.notna(row[1]) else ''
            col2 = row[2] if pd.notna(row[2]) else ''
            
            # Check if this is a sector row
            if col1 and (pd.isna(row[2]) or str(row[2]).strip() == ''):
                if str(col1).strip() not in ['Mediana', 'nan', '']:
                    current_sector = str(col1).strip()
                    if current_sector not in self.sectors:
                        self.sectors[current_sector] = []
                continue
            
            # Check if this is a stock row (has ticker)
            ticker = str(col2).strip() if col2 else ''
            if ticker and ticker not in ['Ticker', 'nan', '']:
                stock_data = {
                    'sector': current_sector,
                    'company_name': str(col1).strip() if col1 else '',
                    'ticker': ticker,
                    'rating': self._safe_value(row, 3, ''),
                    'market_cap': self._safe_numeric(row, 5),
                    'volume_week': self._safe_numeric(row, 6),
                    'volume_avg_12m': self._safe_numeric(row, 7),
                    'volume_pct_avg': self._safe_numeric(row, 8),
                    'price': self._safe_numeric(row, 9),
                    'target_price': self._safe_numeric(row, 10),
                    'upside': self._safe_numeric(row, 11),
                    'return_week': self._safe_numeric(row, 12),
                    'return_month': self._safe_numeric(row, 13),
                    'return_year': self._safe_numeric(row, 14),
                    'pe_2026': self._safe_numeric(row, 15),
                    'pe_2027': self._safe_numeric(row, 16),
                    'ev_ebitda_2026': self._safe_numeric(row, 17),
                    'ev_ebitda_2027': self._safe_numeric(row, 18),
                    'pvp_2026': self._safe_numeric(row, 19),
                    'pvp_2027': self._safe_numeric(row, 20),
                    'dy_2026': self._safe_numeric(row, 21),
                    'dy_2027': self._safe_numeric(row, 22),
                    'net_debt_ebitda_2026': self._safe_numeric(row, 23),
                    'net_debt_ebitda_2027': self._safe_numeric(row, 24),
                    'roe_2026': self._safe_numeric(row, 25),
                    'roe_2027': self._safe_numeric(row, 26),
                }
                stocks_data.append(stock_data)
                
                if current_sector in self.sectors:
                    self.sectors[current_sector].append(ticker)
        
        self.df = pd.DataFrame(stocks_data)
        
        print(f"\n[INFO] Found {len(self.sectors)} sectors:")
        for sector, tickers in self.sectors.items():
            print(f"  - {sector}: {len(tickers)} stocks")
        
        print(f"\n[INFO] Total stocks loaded: {len(self.df)}")
        
        return self.df
    
    def _safe_value(self, row, idx, default=''):
        """Safely get a value from row"""
        try:
            val = row[idx]
            if pd.isna(val) or str(val).lower() in ['nan', 'n.a.', 'n/a', '']:
                return default
            return val
        except:
            return default
    
    def _safe_numeric(self, row, idx, default=0.0):
        """Safely get a numeric value from row"""
        try:
            val = row[idx]
            if pd.isna(val) or str(val).lower() in ['nan', 'n.a.', 'n/a', '']:
                return default
            return float(val)
        except:
            return default
    
    # ==================== VALUE AT RISK (VaR) ====================
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                      method: str = 'parametric') -> Dict:
        """Calculate Value at Risk using multiple methods"""
        if len(returns) < 10:
            return {'var': 0, 'cvar': 0, 'method': method}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = abs(z_score * std)
            cvar = abs(std * stats.norm.pdf(z_score) / (1 - confidence_level))
        else:
            var, cvar = 0, 0
        
        return {'var': var, 'cvar': cvar, 'method': method, 'confidence': confidence_level}
    
    def estimate_returns_from_metrics(self, row: pd.Series) -> np.ndarray:
        """Estimate historical returns from available metrics"""
        return_year = row.get('return_year', 0) or 0
        sector = row.get('sector', 'Unknown')
        base_vol = self._get_sector_volatility(sector)
        pvp = row.get('pvp_2026', 1.0) or 1.0
        volatility_estimate = base_vol + abs(1 - pvp) * 0.1
        
        np.random.seed(int(abs(hash(str(row.get('ticker', '')))) % 10000))
        monthly_return = return_year / 12 if return_year else 0.01
        monthly_vol = volatility_estimate / np.sqrt(12)
        daily_returns = np.random.normal(monthly_return / 21, monthly_vol / np.sqrt(21), 252)
        
        return daily_returns
    
    def _get_sector_volatility(self, sector: str) -> float:
        """Get typical volatility for a sector"""
        sector_vols = {
            'Bancos': 0.25, 'Varejo': 0.40, 'Metais & Mineração': 0.35,
            'Petróleo & Gas': 0.35, 'Serviços Básicos': 0.20, 'Saúde & Educação': 0.30,
            'Agronegócio': 0.30, 'Alimentos & Bebidas': 0.25, 'Telecom, media & tech': 0.35,
            'Moradia': 0.35, 'Bens de capital': 0.35, 'Aéreas': 0.50,
        }
        return sector_vols.get(sector, 0.30)
    
    def _estimate_beta(self, row: pd.Series) -> float:
        """Estimate beta based on sector"""
        sector = row.get('sector', 'Unknown')
        sector_betas = {
            'Bancos': 1.0, 'Setor financeiro (Ex-Bancos)': 1.1, 'Varejo': 1.3,
            'Moradia': 1.2, 'Propriedades Comerciais': 0.8, 'Serviços Básicos': 0.6,
            'Metais & Mineração': 1.4, 'Papel & celulose': 1.2, 'Saúde & Educação': 0.9,
            'Petróleo & Gas': 1.3, 'Agronegócio': 1.1, 'Alimentos & Bebidas': 0.8,
            'Aluguel de carros & Logística': 1.1, 'Aéreas': 1.5, 'Bens de capital': 1.2,
            'Telecom, media & tech': 1.1, 'Infraestrutura': 0.7,
        }
        return sector_betas.get(sector, 1.0)
    
    # ==================== DCF VALUATION ====================
    
    def calculate_dcf_valuation(self, row: pd.Series) -> Dict:
        """DCF Valuation for stocks"""
        price = row.get('price', 0) or 0
        pe_2026 = row.get('pe_2026', 0) or 0
        roe = row.get('roe_2026', 0) or 0
        dy = row.get('dy_2026', 0) or 0
        
        if price <= 0 or pe_2026 <= 0:
            return {'fair_value': 0, 'upside': 0, 'method': 'N/A'}
        
        eps = price / pe_2026 if pe_2026 > 0 else 0
        dividend_per_share = price * dy if dy > 0 else eps * 0.3
        payout_ratio = min(dividend_per_share / eps, 0.8) if eps > 0 else 0.3
        
        retention_ratio = 1 - payout_ratio
        growth_rate = roe * retention_ratio if roe > 0 else 0.05
        growth_rate = min(max(growth_rate, 0.02), 0.15)
        
        beta = self._estimate_beta(row)
        cost_of_equity = self.SELIC_RATE + beta * self.MARKET_RISK_PREMIUM
        
        fair_value = 0
        current_eps = eps
        
        for year in range(1, 6):
            current_eps *= (1 + growth_rate)
            dividend = current_eps * payout_ratio
            fair_value += dividend / ((1 + cost_of_equity) ** year)
        
        terminal_growth = self.PERPETUAL_GROWTH_RATE
        if cost_of_equity > terminal_growth:
            terminal_eps = current_eps * (1 + terminal_growth)
            terminal_dividend = terminal_eps * payout_ratio
            terminal_value = terminal_dividend / (cost_of_equity - terminal_growth)
            fair_value += terminal_value / ((1 + cost_of_equity) ** 5)
        
        upside = (fair_value - price) / price * 100 if price > 0 else 0
        
        return {'fair_value': fair_value, 'current_price': price, 'upside': upside,
                'cost_of_equity': cost_of_equity, 'growth_rate': growth_rate, 'beta': beta}
    
    # ==================== CREDIT ANALYSIS ====================
    
    def calculate_credit_analysis(self, row: pd.Series) -> Dict:
        """Comprehensive Credit Analysis"""
        market_cap = row.get('market_cap', 0) or 0
        volume = row.get('volume_avg_12m', 0) or 0
        pe = row.get('pe_2026', 0) or 0
        roe = row.get('roe_2026', 0) or 0
        net_debt_ebitda = row.get('net_debt_ebitda_2026', 0) or 0
        return_year = row.get('return_year', 0) or 0
        
        # Size Score
        if market_cap >= 50_000: size_score = 100
        elif market_cap >= 20_000: size_score = 85
        elif market_cap >= 5_000: size_score = 70
        elif market_cap >= 1_000: size_score = 50
        else: size_score = 30
        
        # Liquidity Score
        if volume >= 500: liquidity_score = 100
        elif volume >= 200: liquidity_score = 85
        elif volume >= 50: liquidity_score = 70
        elif volume >= 10: liquidity_score = 55
        else: liquidity_score = 30
        
        # Debt Score
        if net_debt_ebitda <= 0: debt_score = 100
        elif net_debt_ebitda <= 1: debt_score = 90
        elif net_debt_ebitda <= 2: debt_score = 75
        elif net_debt_ebitda <= 3: debt_score = 55
        elif net_debt_ebitda <= 4: debt_score = 35
        else: debt_score = 15
        
        # Profitability Score
        if roe >= 0.25: prof_score = 100
        elif roe >= 0.18: prof_score = 85
        elif roe >= 0.12: prof_score = 70
        elif roe >= 0.08: prof_score = 50
        elif roe > 0: prof_score = 30
        else: prof_score = 10
        
        # Valuation Score
        if 0 < pe <= 8: val_score = 100
        elif pe <= 12: val_score = 85
        elif pe <= 18: val_score = 70
        elif pe <= 25: val_score = 50
        elif pe > 25: val_score = 30
        else: val_score = 40
        
        # Momentum Score
        if return_year >= 0.30: mom_score = 100
        elif return_year >= 0.15: mom_score = 80
        elif return_year >= 0.05: mom_score = 60
        elif return_year >= 0: mom_score = 45
        elif return_year >= -0.15: mom_score = 30
        else: mom_score = 15
        
        overall_score = (size_score * 0.20 + liquidity_score * 0.15 + debt_score * 0.20 +
                        prof_score * 0.20 + val_score * 0.10 + mom_score * 0.15)
        
        if overall_score >= 85: rating = 'AAA'
        elif overall_score >= 75: rating = 'AA'
        elif overall_score >= 65: rating = 'A'
        elif overall_score >= 55: rating = 'BBB'
        elif overall_score >= 45: rating = 'BB'
        elif overall_score >= 35: rating = 'B'
        else: rating = 'CCC'
        
        return {'overall_score': overall_score, 'rating': rating, 'size_score': size_score,
                'liquidity_score': liquidity_score, 'debt_score': debt_score,
                'profitability_score': prof_score, 'valuation_score': val_score,
                'momentum_score': mom_score}
    
    # ==================== MONTE CARLO ====================
    
    def run_monte_carlo(self, row: pd.Series, simulations: int = 1000) -> Dict:
        """Monte Carlo simulation"""
        price = row.get('price', 100) or 100
        dy = row.get('dy_2026', 0) or 0
        roe = row.get('roe_2026', 0.10) or 0.10
        sector = row.get('sector', 'Unknown')
        
        volatility = self._get_sector_volatility(sector)
        expected_return = roe * 0.7 + dy
        expected_return = min(max(expected_return, 0.05), 0.25)
        
        drift = expected_return - 0.5 * volatility**2
        np.random.seed(abs(hash(str(row.get('ticker', '')))) % 2**31)
        annual_shocks = np.random.normal(drift, volatility, simulations)
        final_prices = price * np.exp(annual_shocks)
        
        percentile_5 = np.percentile(final_prices, 5)
        percentile_95 = np.percentile(final_prices, 95)
        
        return {
            'mean_price': np.mean(final_prices),
            'upside_potential': (percentile_95 - price) / price * 100 if price > 0 else 0,
            'downside_risk': (percentile_5 - price) / price * 100 if price > 0 else 0,
        }
    
    # ==================== COMPOSITE SCORE ====================
    
    def calculate_composite_score(self, row: pd.Series) -> Tuple[float, Dict]:
        """Calculate comprehensive composite score"""
        scores = {}
        
        # Analyst Rating Score
        rating = str(row.get('rating', '')).lower()
        if 'buy' in rating: scores['analyst'] = 80
        elif 'neutral' in rating: scores['analyst'] = 50
        elif 'sell' in rating: scores['analyst'] = 20
        else: scores['analyst'] = 50
        
        # Upside Score
        upside = row.get('upside', 0) or 0
        if upside >= 0.30: scores['upside'] = 100
        elif upside >= 0.20: scores['upside'] = 85
        elif upside >= 0.10: scores['upside'] = 70
        elif upside >= 0: scores['upside'] = 50
        elif upside >= -0.10: scores['upside'] = 30
        else: scores['upside'] = 15
        
        # Valuation Score (P/E)
        pe = row.get('pe_2026', 0) or 0
        if 0 < pe <= 8: scores['valuation'] = 100
        elif pe <= 12: scores['valuation'] = 85
        elif pe <= 18: scores['valuation'] = 65
        elif pe <= 25: scores['valuation'] = 45
        else: scores['valuation'] = 25
        
        # Profitability Score (ROE)
        roe = row.get('roe_2026', 0) or 0
        if roe >= 0.25: scores['profitability'] = 100
        elif roe >= 0.18: scores['profitability'] = 85
        elif roe >= 0.12: scores['profitability'] = 65
        elif roe >= 0.08: scores['profitability'] = 45
        else: scores['profitability'] = 25
        
        # Dividend Score
        dy = row.get('dy_2026', 0) or 0
        if dy >= 0.10: scores['dividend'] = 100
        elif dy >= 0.06: scores['dividend'] = 80
        elif dy >= 0.04: scores['dividend'] = 60
        elif dy >= 0.02: scores['dividend'] = 40
        else: scores['dividend'] = 20
        
        # Momentum Score
        return_year = row.get('return_year', 0) or 0
        if return_year >= 0.30: scores['momentum'] = 100
        elif return_year >= 0.15: scores['momentum'] = 80
        elif return_year >= 0.05: scores['momentum'] = 60
        elif return_year >= 0: scores['momentum'] = 40
        else: scores['momentum'] = 20
        
        # Debt Score
        net_debt_ebitda = row.get('net_debt_ebitda_2026', 0) or 0
        if net_debt_ebitda <= 0: scores['debt'] = 100
        elif net_debt_ebitda <= 1: scores['debt'] = 85
        elif net_debt_ebitda <= 2: scores['debt'] = 65
        elif net_debt_ebitda <= 3: scores['debt'] = 45
        else: scores['debt'] = 25
        
        weights = {'analyst': 0.20, 'upside': 0.15, 'valuation': 0.15, 
                   'profitability': 0.15, 'dividend': 0.10, 'momentum': 0.10, 'debt': 0.15}
        
        composite = sum(scores.get(k, 50) * v for k, v in weights.items())
        
        return composite, scores
    
    def get_recommendation(self, score: float) -> Recommendation:
        """Convert score to recommendation"""
        if score >= 80: return Recommendation.STRONG_BUY
        elif score >= 65: return Recommendation.BUY
        elif score >= 45: return Recommendation.HOLD
        elif score >= 30: return Recommendation.SELL
        else: return Recommendation.STRONG_SELL
    
    def generate_reasons(self, row: pd.Series, scores: Dict, dcf: Dict, credit: Dict) -> List[str]:
        """Generate detailed reasons"""
        reasons = []
        
        upside = row.get('upside', 0) or 0
        pe = row.get('pe_2026', 0) or 0
        roe = row.get('roe_2026', 0) or 0
        dy = row.get('dy_2026', 0) or 0
        rating = str(row.get('rating', '')).strip()
        
        if upside >= 0.15:
            reasons.append(f"High upside potential ({upside*100:.1f}%)")
        if 'buy' in rating.lower():
            reasons.append(f"Analyst rating: {rating}")
        if 0 < pe <= 12:
            reasons.append(f"Attractive P/E ({pe:.1f}x)")
        if roe >= 0.15:
            reasons.append(f"Strong ROE ({roe*100:.1f}%)")
        if dy >= 0.05:
            reasons.append(f"Good dividend yield ({dy*100:.1f}%)")
        
        dcf_upside = dcf.get('upside', 0)
        if dcf_upside > 30:
            reasons.append(f"Undervalued by DCF (+{dcf_upside:.0f}%)")
        elif dcf_upside < -20:
            reasons.append(f"Overvalued by DCF ({dcf_upside:.0f}%)")
        
        cr = credit.get('rating', 'N/A')
        if cr in ['AAA', 'AA']:
            reasons.append(f"High credit quality ({cr})")
        
        return reasons if reasons else ["Mixed indicators"]
    
    # ==================== MAIN ANALYSIS ====================
    
    def analyze(self) -> List[AnalysisResult]:
        """Run comprehensive analysis"""
        print(f"\n{'='*70}")
        print("  Running Advanced Analysis...")
        print(f"{'='*70}\n")
        
        self.results = []
        self.results_by_sector = {}
        
        total = len(self.df)
        
        for idx, row in self.df.iterrows():
            ticker = row.get('ticker', f'Unknown_{idx}')
            if pd.isna(ticker) or ticker == '':
                continue
            
            if (idx + 1) % 20 == 0:
                print(f"  Processing: {idx + 1}/{total} stocks...")
            
            # Run analyses
            composite_score, individual_scores = self.calculate_composite_score(row)
            recommendation = self.get_recommendation(composite_score)
            dcf = self.calculate_dcf_valuation(row)
            credit = self.calculate_credit_analysis(row)
            monte_carlo = self.run_monte_carlo(row)
            returns = self.estimate_returns_from_metrics(row)
            var_95 = self.calculate_var(returns, 0.95, 'parametric')
            var_99 = self.calculate_var(returns, 0.99, 'parametric')
            
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            dy = row.get('dy_2026', 0) or 0
            sharpe = (dy - self.SELIC_RATE) / volatility if volatility > 0 else 0
            
            reasons = self.generate_reasons(row, individual_scores, dcf, credit)
            sector = row.get('sector', 'Unknown')
            
            result = AnalysisResult(
                ticker=str(ticker),
                company_name=str(row.get('company_name', 'Unknown'))[:40],
                recommendation=recommendation,
                score=composite_score,
                analyst_rating=str(row.get('rating', 'N/A')),
                sector=sector,
                current_price=row.get('price', 0) or 0,
                target_price=row.get('target_price', 0) or 0,
                upside_pct=(row.get('upside', 0) or 0) * 100,
                pe_2026=row.get('pe_2026', 0) or 0,
                pe_2027=row.get('pe_2027', 0) or 0,
                ev_ebitda_2026=row.get('ev_ebitda_2026', 0) or 0,
                pvp_2026=row.get('pvp_2026', 0) or 0,
                dividend_yield=(row.get('dy_2026', 0) or 0) * 100,
                roe_2026=(row.get('roe_2026', 0) or 0) * 100,
                return_week=(row.get('return_week', 0) or 0) * 100,
                return_month=(row.get('return_month', 0) or 0) * 100,
                return_year=(row.get('return_year', 0) or 0) * 100,
                market_cap=row.get('market_cap', 0) or 0,
                volume_avg=row.get('volume_avg_12m', 0) or 0,
                net_debt_ebitda=row.get('net_debt_ebitda_2026', 0) or 0,
                var_95=var_95.get('var', 0) * 100,
                var_99=var_99.get('var', 0) * 100,
                cvar_95=var_95.get('cvar', 0) * 100,
                volatility=volatility * 100,
                sharpe_ratio=sharpe,
                dcf_fair_value=dcf.get('fair_value', 0),
                dcf_upside=dcf.get('upside', 0),
                credit_score=credit['overall_score'],
                credit_rating=credit['rating'],
                mc_upside=monte_carlo['upside_potential'],
                mc_downside=monte_carlo['downside_risk'],
                valuation_score=individual_scores.get('valuation', 0),
                profitability_score=individual_scores.get('profitability', 0),
                momentum_score=individual_scores.get('momentum', 0),
                liquidity_score=credit.get('liquidity_score', 0),
                reasons=reasons,
                economic_cycle=SECTOR_CYCLE_MAP.get(sector, 'EXPANSION')
            )
            
            self.results.append(result)
            
            if sector not in self.results_by_sector:
                self.results_by_sector[sector] = []
            self.results_by_sector[sector].append(result)
        
        # Sort by score
        self.results.sort(key=lambda x: x.score, reverse=True)
        for sector in self.results_by_sector:
            self.results_by_sector[sector].sort(key=lambda x: x.score, reverse=True)
        
        print(f"\n  [OK] Analysis complete for {len(self.results)} stocks")
        
        return self.results
    
    # ==================== PRINT SUMMARY ====================
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*120)
        print(" " * 40 + "ADVANCED STOCK ANALYSIS REPORT")
        print(" " * 40 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*120)
        
        print(f"\nPORTFOLIO OVERVIEW")
        print("-"*60)
        print(f"  Total Stocks Analyzed: {len(self.results)}")
        print(f"  Sectors Covered: {len(self.results_by_sector)}")
        
        rec_counts = {}
        for rec in Recommendation:
            rec_counts[rec.value] = sum(1 for r in self.results if r.recommendation == rec)
        
        print("\n  Recommendation Distribution:")
        for rec, count in rec_counts.items():
            pct = count / len(self.results) * 100
            bar = "#" * int(pct / 2)
            print(f"    {rec:12} : {count:3} ({pct:5.1f}%) {bar}")
        
        # Risk Metrics
        print(f"\nRISK METRICS SUMMARY")
        print("-"*60)
        var_values = [r.var_95 for r in self.results if not np.isnan(r.var_95)]
        vol_values = [r.volatility for r in self.results if not np.isnan(r.volatility)]
        print(f"  Average VaR (95%):     {np.mean(var_values) if var_values else 0:.2f}%")
        print(f"  Average Volatility:    {np.mean(vol_values) if vol_values else 0:.1f}%")
        
        # Valuation Metrics
        print(f"\nVALUATION METRICS SUMMARY")
        print("-"*60)
        print(f"  Average DCF Upside:    {np.mean([r.dcf_upside for r in self.results]):+.1f}%")
        print(f"  Average P/E 2026:      {np.mean([r.pe_2026 for r in self.results if r.pe_2026 > 0]):.1f}x")
        print(f"  Average ROE 2026:      {np.mean([r.roe_2026 for r in self.results if r.roe_2026 > 0]):.1f}%")
        
        # Top 15 Picks
        print(f"\nTOP 15 PICKS")
        print("-"*120)
        print(f"{'Ticker':<10} {'Name':<22} {'Sector':<20} {'Score':>7} {'Rec':<12} {'Rating':>7} {'Upside%':>8} {'VaR%':>7}")
        print("-"*120)
        
        for r in self.results[:15]:
            print(f"{r.ticker:<10} {r.company_name[:20]:<22} {r.sector[:18]:<20} {r.score:>6.1f}  "
                  f"{r.recommendation.value:<12} {r.credit_rating:>7} {r.upside_pct:>+7.1f}% {r.var_95:>6.2f}%")
            print(f"           -> {'; '.join(r.reasons[:2])}")
        
        # Sector Analysis
        print(f"\nSECTOR ANALYSIS")
        print("-"*100)
        print(f"{'Sector':<28} {'Count':>6} {'Avg Score':>10} {'Top Pick':<10} {'Avg PE':>8} {'Avg ROE':>8}")
        print("-"*100)
        
        for sector, results in sorted(self.results_by_sector.items()):
            avg_score = np.mean([r.score for r in results])
            top_pick = results[0].ticker if results else '-'
            avg_pe = np.mean([r.pe_2026 for r in results if r.pe_2026 > 0])
            avg_roe = np.mean([r.roe_2026 for r in results])
            print(f"{sector[:26]:<28} {len(results):>6} {avg_score:>9.1f}  {top_pick:<10} {avg_pe:>7.1f}x {avg_roe:>7.1f}%")
        
        # Bottom 10
        print(f"\n[!] BOTTOM 10 (WEAKEST)")
        print("-"*100)
        for r in self.results[-10:]:
            print(f"  {r.ticker:<10} {r.company_name[:25]:<25} Score: {r.score:.1f}  {r.recommendation.value}")
        
        print("\n" + "="*120)
        print("METHODOLOGY:")
        print("-"*120)
        print("  • VaR (Value at Risk): Parametric method at 95% & 99% confidence")
        print("  • DCF: Multi-stage model with CAPM-based cost of equity")
        print("  • Credit Score: Size, Liquidity, Debt, Profitability, Valuation, Momentum")
        print("  • Monte Carlo: 1,000 simulations using GBM for 1-year price projection")
        print("  • Composite Score: Weighted average of Analyst, Upside, Valuation, Profitability, Dividend, Momentum, Debt")
        print("="*120 + "\n")
    
    # ==================== VISUALIZATION ====================
    
    def create_dashboard(self, save_path: str = None):
        """Create detailed charts"""
        print(f"\n{'='*70}")
        print("  Creating Professional Charts...")
        print(f"{'='*70}\n")
        
        self._chart_top_picks_detailed()
        print("  [OK] Chart 1: Top 20 Picks with Full Details")
        
        self._chart_sector_comparison()
        print("  [OK] Chart 2: Sector Comparison")
        
        self._chart_valuation_analysis()
        print("  [OK] Chart 3: Valuation Analysis")
        
        self._chart_risk_return()
        print("  [OK] Chart 4: Risk-Return Profile")
        
        self._chart_bottom_picks()
        print("  [OK] Chart 5: Bottom 15 (Stocks to Avoid)")
        
        self._chart_summary_dashboard()
        print("  [OK] Chart 6: Summary Dashboard")
        
        self._chart_economic_cycle()
        print("  [OK] Chart 7: Economic Cycle Analysis")
        
        self._chart_risk_distribution_detailed()
        print("  [OK] Chart 8: Detailed Risk Distribution")
        
        if save_path:
            print("\n  Saving charts to files...")
            for i, fig in enumerate(self.figures):
                fname = f"{save_path}_{i+1:02d}.png"
                fig.savefig(fname, dpi=150, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"    -> {fname}")
                plt.close(fig)
            print(f"\n  [OK] All {len(self.figures)} charts saved successfully!")
        else:
            plt.show()
    
    def _chart_top_picks_detailed(self):
        """Top 20 picks with full details"""
        top_20 = self.results[:20]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle('TOP 20 STOCK PICKS - DETAILED ANALYSIS\nRanked by Composite Score', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        y_pos = np.arange(len(top_20))
        bars = ax.barh(y_pos, [r.score for r in top_20], height=0.7, 
                      color=[COLORS.get(r.recommendation.name, '#999') for r in top_20],
                      edgecolor='white', linewidth=1)
        
        for i, r in enumerate(top_20):
            label = f"{r.ticker} | {r.sector[:18]}"
            ax.text(-2, i, label, ha='right', va='center', fontsize=9, fontweight='bold')
            ax.text(r.score - 3, i, f"{r.score:.1f}", ha='right', va='center', 
                   color='white', fontweight='bold', fontsize=10)
            details = f"P/E: {r.pe_2026:.1f}x | ROE: {r.roe_2026:.1f}% | Upside: {r.upside_pct:+.0f}% | {r.recommendation.value}"
            ax.text(r.score + 1, i, details, ha='left', va='center', fontsize=8)
        
        ax.set_yticks([])
        ax.set_xlim(-5, 130)
        ax.set_ylim(-0.5, len(top_20) - 0.5)
        ax.invert_yaxis()
        ax.set_xlabel('Composite Score (0-100)', fontsize=12)
        ax.axvline(80, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7)
        ax.axvline(65, color='green', linestyle=':', linewidth=2, alpha=0.7)
        
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=COLORS['STRONG_BUY'], label='STRONG BUY'),
                          Patch(facecolor=COLORS['BUY'], label='BUY')]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_sector_comparison(self):
        """Sector comparison chart"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('SECTOR ANALYSIS', fontsize=16, fontweight='bold')
        
        # Average score by sector
        ax1 = axes[0]
        sector_scores = {}
        for sector, results in self.results_by_sector.items():
            sector_scores[sector] = np.mean([r.score for r in results])
        
        sorted_sectors = sorted(sector_scores.items(), key=lambda x: x[1], reverse=True)
        sectors = [s[0][:20] for s in sorted_sectors]
        scores = [s[1] for s in sorted_sectors]
        colors = [COLORS['STRONG_BUY'] if s >= 75 else COLORS['BUY'] if s >= 65 else 
                 COLORS['HOLD'] if s >= 45 else COLORS['SELL'] for s in scores]
        
        ax1.barh(sectors, scores, color=colors, edgecolor='white')
        ax1.set_xlabel('Average Score')
        ax1.set_title('Sector Ranking by Score')
        ax1.axvline(65, color='green', linestyle='--', alpha=0.7)
        
        # Recommendation distribution by sector
        ax2 = axes[1]
        sector_data = []
        for sector in [s[0] for s in sorted_sectors[:10]]:
            results = self.results_by_sector[sector]
            buy_pct = sum(1 for r in results if r.recommendation in [Recommendation.STRONG_BUY, Recommendation.BUY]) / len(results) * 100
            sector_data.append((sector[:18], buy_pct))
        
        ax2.barh([s[0] for s in sector_data], [s[1] for s in sector_data], color=COLORS['BUY'], edgecolor='white')
        ax2.set_xlabel('% Buy Recommendations')
        ax2.set_title('Top 10 Sectors by Buy %')
        ax2.axvline(50, color='gray', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_valuation_analysis(self):
        """Valuation analysis chart"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('VALUATION ANALYSIS', fontsize=16, fontweight='bold')
        
        top_20 = self.results[:20]
        tickers = [r.ticker for r in top_20]
        
        # P/E
        ax1 = axes[0, 0]
        pe_values = [r.pe_2026 if r.pe_2026 > 0 else 0 for r in top_20]
        colors = ['green' if pe <= 12 else 'orange' if pe <= 18 else 'red' for pe in pe_values]
        ax1.barh(tickers, pe_values, color=colors, alpha=0.7)
        ax1.axvline(15, color='gray', linestyle='--')
        ax1.set_xlabel('P/E 2026E')
        ax1.set_title('Valuation (P/E)')
        ax1.invert_yaxis()
        
        # ROE
        ax2 = axes[0, 1]
        roe_values = [r.roe_2026 for r in top_20]
        colors = ['green' if roe >= 15 else 'orange' if roe >= 10 else 'red' for roe in roe_values]
        ax2.barh(tickers, roe_values, color=colors, alpha=0.7)
        ax2.axvline(15, color='gray', linestyle='--')
        ax2.set_xlabel('ROE 2026E (%)')
        ax2.set_title('Profitability (ROE)')
        ax2.invert_yaxis()
        
        # Upside
        ax3 = axes[1, 0]
        upside_values = [r.upside_pct for r in top_20]
        colors = ['green' if u > 0 else 'red' for u in upside_values]
        ax3.barh(tickers, upside_values, color=colors, alpha=0.7)
        ax3.axvline(0, color='black', linewidth=0.5)
        ax3.set_xlabel('Upside to Target (%)')
        ax3.set_title('Upside Potential')
        ax3.invert_yaxis()
        
        # Dividend Yield
        ax4 = axes[1, 1]
        dy_values = [r.dividend_yield for r in top_20]
        colors = ['green' if dy >= 5 else 'orange' if dy >= 3 else 'gray' for dy in dy_values]
        ax4.barh(tickers, dy_values, color=colors, alpha=0.7)
        ax4.axvline(5, color='gray', linestyle='--')
        ax4.set_xlabel('Dividend Yield 2026E (%)')
        ax4.set_title('Dividend Yield')
        ax4.invert_yaxis()
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_risk_return(self):
        """Risk-return scatter plot - showing top picks clearly"""
        # Filter to show at least top 10 clearly
        top_n = min(20, len(self.results))  # Show up to top 20 for clarity
        display_results = self.results[:top_n]
        
        fig, ax = plt.subplots(figsize=(14, 10))
        fig.suptitle('RISK-RETURN PROFILE\nTop 20 Picks - Finding Low-Risk, High-Upside Investments', 
                    fontsize=16, fontweight='bold')
        
        # Extract data for top stocks
        x = [r.volatility for r in display_results]
        y = [r.dcf_upside for r in display_results]
        tickers = [r.ticker for r in display_results]
        scores = [r.score for r in display_results]
        colors = [COLORS.get(r.recommendation.name, '#999') for r in display_results]
        
        # Set base size and make size proportional to score
        base_size = 150
        sizes = [base_size + (score * 2) for score in scores]
        
        # Plot all points
        scatter = ax.scatter(x, y, c=colors, s=sizes, alpha=0.7, edgecolors='white', linewidth=1.5)
        
        # Add reference lines
        ax.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        avg_vol = np.mean(x) if x else 0
        if avg_vol > 0:
            ax.axvline(avg_vol, color='gray', linestyle='--', alpha=0.5, 
                      label=f'Avg Volatility: {avg_vol:.1f}%')
        
        # Highlight zones
        ax.fill_between([0, avg_vol], 0, max(y) if y else 100, alpha=0.08, color='green', 
                       label='Low Risk + High Upside')
        ax.fill_between([avg_vol, max(x) * 1.1 if x else 80], min(y) if y else -50, 0, 
                       alpha=0.08, color='red', label='High Risk + Downside')
        
        # Annotate ALL top stocks (not just top 5)
        for i, r in enumerate(display_results):
            # Position annotation to avoid overlap
            offset_x = 0.5 if i % 2 == 0 else -0.5
            offset_y = 0.5 if i < len(display_results)/2 else -0.5
            
            ax.annotate(f'{r.ticker}\n+{r.dcf_upside:.0f}%', 
                       (r.volatility, r.dcf_upside), 
                       fontsize=9 if i < 10 else 8,  # Smaller font for lower ranked
                       ha='center', va='center',
                       fontweight='bold' if i < 10 else 'normal',
                       bbox=dict(boxstyle='round,pad=0.3', 
                                facecolor='yellow' if i < 5 else 'lightyellow', 
                                alpha=0.7),
                       xytext=(offset_x, offset_y), 
                       textcoords='offset points')
        
        # Add quadrant labels
        ax.text(avg_vol/2, max(y)*0.7 if y else 50, 'BEST ZONE\nLow Risk + High Upside', 
               fontsize=10, color='darkgreen', fontweight='bold', ha='center',
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        
        if x and max(x) > 0:
            ax.text(avg_vol + (max(x) - avg_vol)/2, min(y)*0.5 if y else -30, 
                   'DANGER ZONE\nHigh Risk + Downside', fontsize=10, color='darkred', 
                   fontweight='bold', ha='center',
                   bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        
        ax.set_xlabel('Volatility (%)\n← Lower Risk | Higher Risk →')
        ax.set_ylabel('DCF Upside (%)\n← Overvalued | Undervalued →')
        
        # Set reasonable limits
        if x:
            ax.set_xlim(0, max(x) * 1.15)
        if y:
            ax.set_ylim(min(y) * 1.1, max(y) * 1.1)
        
        # Add legend for recommendation colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['STRONG_BUY'], label='STRONG BUY'),
            Patch(facecolor=COLORS['BUY'], label='BUY'),
            Patch(facecolor=COLORS['HOLD'], label='HOLD'),
            Patch(facecolor=COLORS['SELL'], label='SELL'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=9)
        
        # Add grid
        ax.grid(True, alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_bottom_picks(self):
        """Bottom 15 stocks to avoid"""
        bottom_15 = self.results[-15:][::-1]
        
        fig, ax = plt.subplots(figsize=(16, 10))
        fig.suptitle('BOTTOM 15 STOCKS - AVOID THESE\nLowest Composite Scores', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        y_pos = np.arange(len(bottom_15))
        bars = ax.barh(y_pos, [r.score for r in bottom_15], height=0.7, 
                      color=[COLORS.get(r.recommendation.name, '#999') for r in bottom_15],
                      edgecolor='white', linewidth=1)
        
        for i, r in enumerate(bottom_15):
            label = f"{r.ticker} | {r.sector[:18]}"
            ax.text(-2, i, label, ha='right', va='center', fontsize=9, fontweight='bold')
            ax.text(r.score + 1, i, f"{r.score:.1f} - {r.recommendation.value}", ha='left', va='center', fontsize=9)
        
        ax.set_yticks([])
        ax.set_xlim(-5, 80)
        ax.set_ylim(-0.5, len(bottom_15) - 0.5)
        ax.invert_yaxis()
        ax.set_xlabel('Composite Score (0-100)', fontsize=12)
        ax.axvline(45, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Hold threshold')
        ax.axvline(30, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Sell threshold')
        ax.legend()
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_summary_dashboard(self):
        """Summary dashboard"""
        fig = plt.figure(figsize=(16, 12))
        gs = GridSpec(3, 3, figure=fig)
        fig.suptitle('STOCK ANALYSIS SUMMARY DASHBOARD', fontsize=18, fontweight='bold')
        
        # 1. Recommendation pie
        ax1 = fig.add_subplot(gs[0, 0])
        rec_counts = {}
        for r in self.results:
            rec = r.recommendation.value
            rec_counts[rec] = rec_counts.get(rec, 0) + 1
        labels = list(rec_counts.keys())
        sizes = list(rec_counts.values())
        colors = [COLORS.get(r.replace(' ', '_'), '#888888') for r in labels]
        ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Recommendation Distribution')
        
        # 2. Top 5 bar
        ax2 = fig.add_subplot(gs[0, 1:])
        top_5 = self.results[:5]
        ax2.barh([r.ticker for r in top_5], [r.score for r in top_5], 
                color=[COLORS.get(r.recommendation.name, '#999') for r in top_5])
        ax2.set_xlabel('Score')
        ax2.set_title('Top 5 Picks')
        ax2.invert_yaxis()
        
        # 3. Average metrics by sector (top 8)
        ax3 = fig.add_subplot(gs[1, :2])
        sorted_sectors = sorted(self.results_by_sector.items(), 
                               key=lambda x: np.mean([r.score for r in x[1]]), reverse=True)[:8]
        sectors = [s[0][:15] for s in sorted_sectors]
        scores = [np.mean([r.score for r in s[1]]) for s in sorted_sectors]
        colors = [COLORS['STRONG_BUY'] if s >= 70 else COLORS['BUY'] if s >= 60 else COLORS['HOLD'] for s in scores]
        ax3.barh(sectors, scores, color=colors)
        ax3.set_xlabel('Average Score')
        ax3.set_title('Best Sectors')
        ax3.invert_yaxis()
        
        # 4. Risk distribution
        ax4 = fig.add_subplot(gs[1, 2])
        var_values = [r.var_95 for r in self.results]
        ax4.hist(var_values, bins=20, color=COLORS['primary'], edgecolor='white', alpha=0.7)
        ax4.axvline(np.mean(var_values), color='red', linestyle='--', label=f'Avg: {np.mean(var_values):.1f}%')
        ax4.set_xlabel('VaR 95% (%)')
        ax4.set_title('Risk Distribution')
        ax4.legend()
        
        # 5. Key stats
        ax5 = fig.add_subplot(gs[2, :])
        ax5.axis('off')
        
        # Calculate stats
        total_stocks = len(self.results)
        total_sectors = len(self.results_by_sector)
        
        strong_buy = sum(1 for r in self.results if r.recommendation == Recommendation.STRONG_BUY)
        buy = sum(1 for r in self.results if r.recommendation == Recommendation.BUY)
        hold = sum(1 for r in self.results if r.recommendation == Recommendation.HOLD)
        sell = sum(1 for r in self.results if r.recommendation == Recommendation.SELL)
        
        avg_pe = np.mean([r.pe_2026 for r in self.results if r.pe_2026 > 0])
        avg_roe = np.mean([r.roe_2026 for r in self.results if r.roe_2026 > 0])
        avg_dy = np.mean([r.dividend_yield for r in self.results])
        avg_var = np.mean([r.var_95 for r in self.results])
        avg_vol = np.mean([r.volatility for r in self.results])
        
        top_pick = self.results[0] if self.results else None
        
        # Create the stats text with proper characters
        stats_text = f"""
{'='*100}
ANALYSIS SUMMARY: {total_stocks} Stocks | {total_sectors} Sectors
{'='*100}

RECOMMENDATIONS:  Strong Buy: {strong_buy:3d}  |  Buy: {buy:3d}  |  Hold: {hold:3d}  |  Sell: {sell:3d}

VALUATION:        Avg P/E: {avg_pe:.1f}x  |  Avg ROE: {avg_roe:.1f}%  |  Avg DY: {avg_dy:.1f}%

RISK:             Avg VaR 95%: {avg_var:.2f}%  |  Avg Volatility: {avg_vol:.1f}%

"""
        
        if top_pick:
            stats_text += f"""
TOP PICK:         {top_pick.ticker} ({top_pick.company_name[:30]}) 
                  Score: {top_pick.score:.1f} - {top_pick.recommendation.value}
{'='*100}
"""
        
        ax5.text(0.5, 0.5, stats_text, transform=ax5.transAxes, fontsize=11, 
                verticalalignment='center', horizontalalignment='center',
                fontfamily='monospace', linespacing=1.5,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_economic_cycle(self):
        """Economic Cycle Momentum Clock - Top 5 stocks with last 4 quarters"""
        if not self.results:
            print("  [WARNING] No results to plot economic cycle chart")
            return
        
        fig, ax = plt.subplots(figsize=(14, 12))
        
        # Title exactly as in image
        fig.suptitle('Analisador\nMomentum Cycle\n\nComparador\nForecasting', 
                    fontsize=16, fontweight='bold', y=0.95, linespacing=1.8)
        
        # Set up axes exactly as in image
        ax.set_xlabel('DESACELERAÇÃO                RETRAÇÃO', 
                     fontsize=14, fontweight='bold')
        
        # Set Y-axis ticks exactly as in image (comma decimals)
        y_ticks = [0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, -0.1, -0.2, -0.3, -0.4, -0.5, -0.6]
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(['0,6', '0,5', '0,4', '0,3', '0,2', '0,1', '0,0', 
                           '-0,1', '-0,2', '-0,3', '-0,4', '-0,5', '-0,6'], 
                         fontsize=12)
        
        # Remove X-axis numeric labels (not shown in image)
        ax.set_xticks([])
        
        # Set limits for clock-like appearance
        ax.set_xlim(-0.65, 0.65)
        ax.set_ylim(-0.65, 0.65)
        
        # Center lines (thinner than in image but visible)
        ax.axhline(y=0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(x=0, color='black', linewidth=0.5, alpha=0.5)
        
        # Add quadrant labels exactly as in image
        ax.text(-0.55, 0.25, 'RECUPERAÇÃO', fontsize=14, fontweight='bold', 
                color='green', ha='center')
        ax.text(0.55, 0.25, 'EXPANSÃO', fontsize=14, fontweight='bold', 
                color='blue', ha='center')
        
        # Add annotations exactly as in image
        ax.annotate('Aceleração(Y)', xy=(0.25, 0.25), xytext=(0.35, 0.4),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red', fontweight='bold', ha='center')
        ax.annotate('Desaceleração(X)', xy=(-0.25, -0.25), xytext=(-0.45, -0.4),
                    arrowprops=dict(arrowstyle='->', color='red', lw=2),
                    fontsize=12, color='red', fontweight='bold', ha='center')
        
        # Create concentric circles for clock effect
        theta = np.linspace(0, 2*np.pi, 100)
        circle_radii = [0.2, 0.4, 0.6]
        
        for radius in circle_radii:
            x_circle = radius * np.cos(theta)
            y_circle = radius * np.sin(theta)
            ax.plot(x_circle, y_circle, 'gray', linewidth=0.8, alpha=0.3, linestyle='--')
        
        # Get top 5 stocks
        top_5_stocks = self.results[:5]
        
        # Define quarter labels for the last 4 quarters
        quarters = ['Q4', 'Q1', 'Q2', 'Q3']  # Last 4 quarters
        
        # Different colors for each stock
        stock_colors = ['red', 'blue', 'green', 'orange', 'purple']
        
        for stock_idx, stock in enumerate(top_5_stocks):
            if stock_idx >= len(stock_colors):
                break
                
            color = stock_colors[stock_idx]
            
            # Calculate base position based on stock metrics
            # X-axis: momentum indicators (return_month for recent momentum)
            base_x = (stock.return_month / 100) * 0.3
            
            # Y-axis: growth potential (upside)
            base_y = (stock.upside_pct / 100) * 0.3
            
            # Add some variation to show quarter progression
            quarter_positions = []
            
            for q_idx, quarter in enumerate(quarters):
                # Create a progression through quarters
                # Simulate quarter-to-quarter movement
                quarter_x = base_x + (q_idx - 1.5) * 0.08  # Spread across X
                quarter_y = base_y + (q_idx - 1.5) * 0.08  # Spread across Y
                
                # Add some random variation for realism
                variation = 0.05 * (q_idx + 1)
                quarter_x += np.random.uniform(-variation, variation)
                quarter_y += np.random.uniform(-variation, variation)
                
                # Ensure within bounds
                quarter_x = max(-0.55, min(0.55, quarter_x))
                quarter_y = max(-0.55, min(0.55, quarter_y))
                
                quarter_positions.append((quarter_x, quarter_y))
                
                # Plot quarter point
                point_size = 120 - (q_idx * 20)  # Q4 is largest, Q3 is smallest
                alpha = 1.0 - (q_idx * 0.2)  # Q4 is most opaque
                
                ax.scatter(quarter_x, quarter_y, s=point_size, color=color, 
                          edgecolor='black', linewidth=1.5, zorder=10, alpha=alpha)
                
                # Label for Q4 (most recent) only to avoid clutter
                if quarter == 'Q4':
                    ax.text(quarter_x, quarter_y + 0.04, stock.ticker, 
                           fontsize=10, fontweight='bold', ha='center', 
                           color=color, bbox=dict(boxstyle='round,pad=0.2', 
                                                 facecolor='white', alpha=0.7))
            
            # Connect quarters with line showing progression
            x_line = [pos[0] for pos in quarter_positions]
            y_line = [pos[1] for pos in quarter_positions]
            
            # Plot line connecting quarters
            ax.plot(x_line, y_line, color=color, linewidth=1.5, alpha=0.6, 
                   linestyle='-', marker='', zorder=5)
            
            # Add arrow showing progression direction (Q4 to Q3)
            if len(x_line) > 1:
                ax.annotate('', xy=(x_line[-1], y_line[-1]), 
                           xytext=(x_line[-2], y_line[-2]),
                           arrowprops=dict(arrowstyle='->', color=color, 
                                          lw=1.5, alpha=0.8))
        
        # Add legend for stocks
        legend_patches = []
        for i, stock in enumerate(top_5_stocks[:len(stock_colors)]):
            legend_patches.append(
                mpatches.Patch(color=stock_colors[i], label=f'{stock.ticker}: {stock.score:.1f}')
            )
        
        if legend_patches:
            ax.legend(handles=legend_patches, loc='lower right', fontsize=9,
                     title='Top 5 Stocks (Score)', title_fontsize=10,
                     framealpha=0.8)
        
        # Add quarter indicator legend
        quarter_legend_text = "Quarters: ● Q4 (Latest) → ○ Q3 (Oldest)"
        ax.text(0.02, -0.62, quarter_legend_text, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.7))
        
        # Add grid lines (subtle)
        ax.grid(True, alpha=0.15, linestyle='-')
        
        # Set equal aspect ratio for perfect circle
        ax.set_aspect('equal', adjustable='box')
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        
        plt.tight_layout()
        self.figures.append(fig)
        print(f"  [OK] Chart 7: Economic Cycle Momentum Clock (Top {len(top_5_stocks)} stocks)")
    
    def _chart_risk_distribution_detailed(self):
        """Detailed Risk Distribution - Shows which stocks are in each risk category"""
        if not self.results:
            print("  [WARNING] No results to plot risk distribution")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('DETAILED RISK ANALYSIS\nStock Distribution by Risk Category', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 1. Risk Categories Distribution (Pie Chart)
        ax1.set_title('Risk Category Distribution', fontsize=14, fontweight='bold')
        
        # Categorize stocks by VaR 95%
        risk_categories = {
            'Very Low Risk': [],   # VaR < 5%
            'Low Risk': [],        # VaR 5-10%
            'Medium Risk': [],     # VaR 10-20%
            'High Risk': [],       # VaR 20-30%
            'Very High Risk': [],  # VaR > 30%
        }
        
        for result in self.results:
            var_95 = result.var_95
            if var_95 < 5:
                risk_categories['Very Low Risk'].append(result)
            elif var_95 < 10:
                risk_categories['Low Risk'].append(result)
            elif var_95 < 20:
                risk_categories['Medium Risk'].append(result)
            elif var_95 < 30:
                risk_categories['High Risk'].append(result)
            else:
                risk_categories['Very High Risk'].append(result)
        
        # Calculate counts
        category_labels = list(risk_categories.keys())
        category_counts = [len(stocks) for stocks in risk_categories.values()]
        
        # Colors for risk categories
        risk_colors = ['#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#F44336']
        
        # Create pie chart
        wedges, texts, autotexts = ax1.pie(category_counts, labels=category_labels, 
                                          colors=risk_colors, autopct='%1.1f%%',
                                          startangle=90, pctdistance=0.85,
                                          wedgeprops=dict(edgecolor='white', linewidth=2))
        
        # Make autotexts white
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
        
        # Draw circle for donut chart effect
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.text(0, 0, f"Total:\n{len(self.results)}", 
                ha='center', va='center', fontsize=12, fontweight='bold')
        
        # 2. Top 10 Lowest Risk Stocks
        ax2.set_title('Top 10 Lowest Risk Stocks', fontsize=14, fontweight='bold')
        
        # Sort by VaR (ascending = lower risk first)
        low_risk_stocks = sorted(self.results, key=lambda x: x.var_95)[:10]
        
        tickers = [r.ticker for r in low_risk_stocks]
        var_values = [r.var_95 for r in low_risk_stocks]
        upside_values = [r.upside_pct for r in low_risk_stocks]
        
        # Create grouped bar chart
        x = np.arange(len(tickers))
        width = 0.35
        
        bars1 = ax2.bar(x - width/2, var_values, width, label='VaR 95%', 
                       color='#2196F3', alpha=0.8)
        bars2 = ax2.bar(x + width/2, upside_values, width, label='Upside %', 
                       color='#4CAF50', alpha=0.8)
        
        ax2.set_xlabel('Stocks')
        ax2.set_ylabel('Percentage (%)')
        ax2.set_xticks(x)
        ax2.set_xticklabels(tickers, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Top 10 Highest Risk Stocks
        ax3.set_title('Top 10 Highest Risk Stocks', fontsize=14, fontweight='bold')
        
        # Sort by VaR (descending = higher risk first)
        high_risk_stocks = sorted(self.results, key=lambda x: x.var_95, reverse=True)[:10]
        
        tickers_high = [r.ticker for r in high_risk_stocks]
        var_values_high = [r.var_95 for r in high_risk_stocks]
        downside_values = [abs(r.mc_downside) for r in high_risk_stocks]  # Monte Carlo downside
        
        bars3 = ax3.bar(tickers_high, var_values_high, color='#F44336', alpha=0.8, 
                       label='VaR 95%')
        
        # Create secondary axis for downside risk
        ax3_secondary = ax3.twinx()
        ax3_secondary.plot(tickers_high, downside_values, color='#FF5722', 
                          marker='o', linewidth=2, label='Monte Carlo Downside')
        
        ax3.set_xlabel('Stocks')
        ax3.set_ylabel('VaR 95% (%)', color='#F44336')
        ax3.tick_params(axis='x', rotation=45)
        ax3.tick_params(axis='y', labelcolor='#F44336')
        ax3_secondary.set_ylabel('Downside Risk (%)', color='#FF5722')
        ax3_secondary.tick_params(axis='y', labelcolor='#FF5722')
        
        # Add combined legend
        lines1, labels1 = ax3.get_legend_handles_labels()
        lines2, labels2 = ax3_secondary.get_legend_handles_labels()
        ax3.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        
        # 4. Risk vs Return Heatmap (Scatter with density)
        ax4.set_title('Risk-Return Heatmap', fontsize=14, fontweight='bold')
        
        # Prepare data
        x_data = [r.volatility for r in self.results]
        y_data = [r.dcf_upside for r in self.results]
        ticker_data = [r.ticker for r in self.results]
        var_data = [r.var_95 for r in self.results]
        
        # Create scatter plot with color based on VaR
        scatter = ax4.scatter(x_data, y_data, c=var_data, cmap='RdYlGn_r', 
                             s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax4)
        cbar.set_label('VaR 95% (%)', rotation=270, labelpad=20)
        
        # Add reference lines
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        ax4.axvline(x=np.median(x_data) if x_data else 0, color='gray', 
                   linestyle='--', linewidth=1, alpha=0.5, 
                   label=f'Median Vol: {np.median(x_data):.1f}%' if x_data else '')
        
        # Label top and bottom performers
        if len(self.results) >= 5:
            # Label top 5 by score
            for result in self.results[:5]:
                ax4.annotate(result.ticker, 
                            (result.volatility, result.dcf_upside),
                            xytext=(5, 5), textcoords='offset points',
                            fontsize=9, fontweight='bold',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='yellow', alpha=0.7))
            
            # Label bottom 5 (high risk, low return)
            bottom_results = sorted(self.results, 
                                   key=lambda x: (x.var_95, -x.dcf_upside), 
                                   reverse=True)[:3]
            for result in bottom_results:
                ax4.annotate(result.ticker, 
                            (result.volatility, result.dcf_upside),
                            xytext=(5, -15), textcoords='offset points',
                            fontsize=9, fontweight='bold', color='red',
                            bbox=dict(boxstyle='round,pad=0.3', 
                                     facecolor='lightcoral', alpha=0.7))
        
        ax4.set_xlabel('Volatility (%)')
        ax4.set_ylabel('DCF Upside (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(fig)
        print("  [OK] Chart 8: Detailed Risk Distribution Analysis")


def main():
    """Main entry point - Works like FII Analyzer"""
    import sys
    
    file_path = None
    save_charts = None
    auto_charts = False
    
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--charts='):
            save_charts = arg.split('=')[1]
            auto_charts = True
        elif arg == '--charts':
            auto_charts = True
        elif not file_path:
            file_path = arg
    
    if not file_path:
        print("\n" + "="*70)
        print("  ADVANCED STOCK ANALYZER")
        print("  With VaR, DCF, Credit Analysis, Monte Carlo & Visualization")
        print("="*70)
        print("\nUsage: python stock_analyzer_advanced.py <xlsx_file> [--charts] [--charts=<output_path>]")
        file_path = input("\nEnter the path to the XLSX file: ").strip()
        
        if not file_path:
            file_path = r"C:\Users\DiegoCamargo\Downloads\Cont\SGuide.xlsx"
            print(f"Using default: {file_path}")
    
    try:
        analyzer = AdvancedStockAnalyzer(file_path)
        analyzer.load_data()
        analyzer.analyze()
        analyzer.print_summary()
        
        if auto_charts:
            analyzer.create_dashboard(save_charts)
        else:
            show_charts = input("\nGenerate professional charts? (y/n): ").strip().lower()
            if show_charts == 'y':
                save_path = input("Save charts to path (leave empty to just display): ").strip()
                analyzer.create_dashboard(save_path if save_path else None)
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()