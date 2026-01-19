"""
Advanced FII (Fundos de Investimento Imobiliário) Analyzer
With VaR, DCF, Credit Analysis, Monte Carlo, and Sensitivity Analysis
Includes Professional Visualization
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


class Recommendation(Enum):
    STRONG_BUY = "STRONG BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG SELL"


@dataclass
class AnalysisResult:
    ticker: str
    fund_name: str
    recommendation: Recommendation
    score: float
    pvpa_score: float
    yield_score: float
    momentum_score: float
    liquidity_score: float
    credit_score: float
    dcf_upside: float
    reasons: List[str]
    segment: str = "Unknown"
    # Extended metrics
    var_95: float = 0.0
    var_99: float = 0.0
    cvar_95: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    volatility: float = 0.0
    fair_value: float = 0.0
    current_price: float = 0.0
    credit_rating: str = "N/A"
    monte_carlo_upside: float = 0.0
    monte_carlo_downside: float = 0.0
    # Raw data metrics
    dividend_yield: float = 0.0
    pvpa: float = 1.0
    market_cap: float = 0.0
    volume: float = 0.0


class AdvancedFIIAnalyzer:
    """
    Advanced FII Analyzer with comprehensive financial analysis methods
    """
    
    # Brazilian Market Parameters
    SELIC_RATE = 0.1175  # Current SELIC
    CDI_RATE = 0.1165  # CDI Rate
    IPCA = 0.045  # Inflation target
    MARKET_RISK_PREMIUM = 0.055
    PERPETUAL_GROWTH_RATE = 0.03
    TAX_RATE = 0.0  # FIIs are tax-exempt for individuals
    
    # Positional column mapping
    POSITIONAL_MAPPING = {
        0: 'ticker', 1: 'fund_name', 2: 'price', 3: 'ifix_weight',
        4: 'avg_daily_volume', 5: 'market_cap', 6: 'book_value',
        7: 'pvpa_current', 8: 'pvpa_projected', 9: 'dy_ltm',
        10: 'dy_annualized', 11: 'last_dividend', 12: 'return_month',
        13: 'return_year', 14: 'return_ltm'
    }
    
    def __init__(self, file_path: str):
        """Initialize analyzer"""
        self.file_path = file_path
        self.df = None
        self.all_sheets_data = {}
        self.results: List[AnalysisResult] = []
        self.results_by_segment: Dict[str, List[AnalysisResult]] = {}
        self.figures = []
    
    def load_data(self) -> pd.DataFrame:
        """Load data from all sheets in XLSX file"""
        print(f"\n{'='*70}")
        print("  ADVANCED FII ANALYZER - Loading Data")
        print(f"{'='*70}")
        
        xl = pd.ExcelFile(self.file_path)
        sheet_names = xl.sheet_names
        print(f"\n[INFO] Found {len(sheet_names)} segments: {', '.join(sheet_names)}")
        
        all_dfs = []
        
        for sheet_name in sheet_names:
            df_raw = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)
            
            # Find header row with 'Código'
            header_row = None
            for idx, row in df_raw.iterrows():
                if 'Código' in ' '.join([str(x) for x in row.values]):
                    header_row = idx
                    break
            
            if header_row is not None:
                df_sheet = pd.read_excel(self.file_path, sheet_name=sheet_name, header=header_row)
            else:
                df_sheet = pd.read_excel(self.file_path, sheet_name=sheet_name, header=5)
            
            # Clean and rename columns
            df_sheet.columns = [str(c).strip() if pd.notna(c) else f'Col_{i}' 
                               for i, c in enumerate(df_sheet.columns)]
            
            # Positional mapping
            rename_dict = {col: self.POSITIONAL_MAPPING[idx] 
                          for idx, col in enumerate(df_sheet.columns) 
                          if idx in self.POSITIONAL_MAPPING}
            df_sheet = df_sheet.rename(columns=rename_dict)
            
            # Convert percentages
            for col in ['ifix_weight', 'dy_ltm', 'dy_annualized', 'return_month', 'return_year', 'return_ltm']:
                if col in df_sheet.columns:
                    df_sheet[col] = pd.to_numeric(df_sheet[col], errors='coerce')
                    if df_sheet[col].median() > 1:
                        df_sheet[col] = df_sheet[col] / 100
            
            # Filter valid tickers
            if 'ticker' in df_sheet.columns:
                df_sheet = df_sheet[df_sheet['ticker'].astype(str).str.match(r'^[A-Z]{4}\d{2}$', na=False)]
            
            df_sheet['segment'] = sheet_name
            self.all_sheets_data[sheet_name] = df_sheet.copy()
            all_dfs.append(df_sheet)
            print(f"  [OK] {sheet_name}: {len(df_sheet)} FIIs")
        
        self.df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n[INFO] Total FIIs loaded: {len(self.df)}")
        
        return self.df
    
    # ==================== VALUE AT RISK (VaR) ====================
    
    def calculate_var(self, returns: np.ndarray, confidence_level: float = 0.95, 
                      method: str = 'parametric') -> Dict:
        """
        Calculate Value at Risk using multiple methods
        
        Methods:
        - historical: Uses historical simulation
        - parametric: Uses variance-covariance approach (Normal distribution)
        - monte_carlo: Uses Monte Carlo simulation
        """
        if len(returns) < 10:
            return {'var': 0, 'cvar': 0, 'method': method}
        
        returns = np.array(returns)
        returns = returns[~np.isnan(returns)]
        
        if method == 'historical':
            var = np.percentile(returns, (1 - confidence_level) * 100)
            var = abs(var)
            # Conditional VaR (Expected Shortfall)
            threshold = np.percentile(returns, (1 - confidence_level) * 100)
            cvar = abs(returns[returns <= threshold].mean()) if len(returns[returns <= threshold]) > 0 else var
            
        elif method == 'parametric':
            mean = np.mean(returns)
            std = np.std(returns)
            z_score = stats.norm.ppf(1 - confidence_level)
            var = abs(z_score * std)
            # CVaR for normal distribution
            cvar = abs(std * stats.norm.pdf(z_score) / (1 - confidence_level))
            
        elif method == 'monte_carlo':
            np.random.seed(42)
            mean = np.mean(returns)
            std = np.std(returns)
            simulated = np.random.normal(mean, std, 10000)
            var = abs(np.percentile(simulated, (1 - confidence_level) * 100))
            threshold = np.percentile(simulated, (1 - confidence_level) * 100)
            cvar = abs(simulated[simulated <= threshold].mean())
        else:
            var, cvar = 0, 0
        
        return {
            'var': var,
            'cvar': cvar,
            'method': method,
            'confidence': confidence_level
        }
    
    def estimate_returns_from_metrics(self, row: pd.Series) -> np.ndarray:
        """Estimate historical returns from available metrics"""
        # Use available metrics to simulate return distribution
        return_ltm = row.get('return_ltm', 0) or 0
        return_year = row.get('return_year', 0) or 0
        return_month = row.get('return_month', 0) or 0
        dy = row.get('dy_annualized', 0.12) or 0.12
        
        # Estimate daily volatility from P/VPA dispersion
        pvpa = row.get('pvpa_current', 1.0) or 1.0
        volatility_estimate = abs(1 - pvpa) * 0.5 + 0.15  # Base volatility + P/VPA deviation
        
        # Generate synthetic returns based on known metrics
        np.random.seed(int(abs(hash(str(row.get('ticker', '')))) % 10000))
        
        # Monthly return with estimated volatility
        monthly_return = return_ltm / 12 if return_ltm else dy / 12
        monthly_vol = volatility_estimate / np.sqrt(12)
        
        # Generate 252 trading days of returns
        daily_returns = np.random.normal(monthly_return / 21, monthly_vol / np.sqrt(21), 252)
        
        return daily_returns
    
    # ==================== DCF VALUATION ====================
    
    def calculate_dcf_valuation(self, row: pd.Series) -> Dict:
        """
        Enhanced DCF Valuation using Gordon Growth Model and Multi-Stage DCF
        """
        price = row.get('price', 0) or 0
        dividend = row.get('last_dividend', 0) or 0
        dy = row.get('dy_annualized', 0) or 0
        pvpa = row.get('pvpa_current', 1.0) or 1.0
        market_cap = row.get('market_cap', 0) or 0
        book_value = row.get('book_value', 0) or 0
        
        if price <= 0:
            return {'fair_value': 0, 'upside': 0, 'method': 'N/A'}
        
        # Estimate annual dividend
        if dividend > 0:
            annual_dividend = dividend * 12
        elif dy > 0:
            annual_dividend = price * dy
        else:
            return {'fair_value': price, 'upside': 0, 'method': 'Insufficient Data'}
        
        # Calculate cost of equity (CAPM-like for FIIs)
        # FII Beta is typically lower than market (0.3-0.7)
        fii_beta = 0.5 + (pvpa - 1) * 0.2  # Adjust beta based on P/VPA
        fii_beta = max(0.2, min(1.0, fii_beta))
        
        cost_of_equity = self.SELIC_RATE + fii_beta * self.MARKET_RISK_PREMIUM
        
        # Stage 1: High growth (3 years)
        growth_stage1 = min(self.IPCA + 0.02, 0.08)  # Inflation + 2%
        
        # Stage 2: Transition (2 years)
        growth_stage2 = (growth_stage1 + self.PERPETUAL_GROWTH_RATE) / 2
        
        # Terminal growth
        terminal_growth = self.PERPETUAL_GROWTH_RATE
        
        # Multi-stage DCF
        fair_value = 0
        current_dividend = annual_dividend
        
        # Stage 1 (Years 1-3)
        for year in range(1, 4):
            current_dividend *= (1 + growth_stage1)
            fair_value += current_dividend / ((1 + cost_of_equity) ** year)
        
        # Stage 2 (Years 4-5)
        for year in range(4, 6):
            current_dividend *= (1 + growth_stage2)
            fair_value += current_dividend / ((1 + cost_of_equity) ** year)
        
        # Terminal value (Gordon Growth)
        if cost_of_equity > terminal_growth:
            terminal_dividend = current_dividend * (1 + terminal_growth)
            terminal_value = terminal_dividend / (cost_of_equity - terminal_growth)
            fair_value += terminal_value / ((1 + cost_of_equity) ** 5)
        
        upside = (fair_value - price) / price * 100 if price > 0 else 0
        
        return {
            'fair_value': fair_value,
            'current_price': price,
            'upside': upside,
            'cost_of_equity': cost_of_equity,
            'terminal_growth': terminal_growth,
            'method': 'Multi-Stage DCF'
        }
    
    # ==================== CREDIT ANALYSIS ====================
    
    def calculate_credit_analysis(self, row: pd.Series) -> Dict:
        """
        Comprehensive Credit Analysis for FIIs
        """
        market_cap = row.get('market_cap', 0) or 0
        book_value = row.get('book_value', 0) or 0
        volume = row.get('avg_daily_volume', 0) or 0
        pvpa = row.get('pvpa_current', 1.0) or 1.0
        dy = row.get('dy_annualized', 0) or 0
        return_ltm = row.get('return_ltm', 0) or 0
        
        # Size Score (larger = more stable)
        if market_cap >= 2_000_000_000:
            size_score = 100
            size_tier = 'MEGA CAP'
        elif market_cap >= 1_000_000_000:
            size_score = 85
            size_tier = 'LARGE CAP'
        elif market_cap >= 500_000_000:
            size_score = 70
            size_tier = 'MID CAP'
        elif market_cap >= 100_000_000:
            size_score = 50
            size_tier = 'SMALL CAP'
        else:
            size_score = 30
            size_tier = 'MICRO CAP'
        
        # Liquidity Score
        if volume >= 2_000_000:
            liquidity_score = 100
            liquidity_tier = 'EXCELLENT'
        elif volume >= 1_000_000:
            liquidity_score = 85
            liquidity_tier = 'VERY HIGH'
        elif volume >= 500_000:
            liquidity_score = 70
            liquidity_tier = 'HIGH'
        elif volume >= 200_000:
            liquidity_score = 55
            liquidity_tier = 'MEDIUM'
        elif volume >= 100_000:
            liquidity_score = 40
            liquidity_tier = 'LOW'
        else:
            liquidity_score = 20
            liquidity_tier = 'VERY LOW'
        
        # P/VPA Quality Score (fair value = 1.0)
        if 0.95 <= pvpa <= 1.05:
            pvpa_quality = 100
            pvpa_status = 'FAIR VALUE'
        elif 0.85 <= pvpa < 0.95 or 1.05 < pvpa <= 1.15:
            pvpa_quality = 80
            pvpa_status = 'SLIGHT DEVIATION'
        elif 0.70 <= pvpa < 0.85:
            pvpa_quality = 70
            pvpa_status = 'DISCOUNT'
        elif pvpa < 0.70:
            pvpa_quality = 50  # Deep discount may indicate problems
            pvpa_status = 'DEEP DISCOUNT - VERIFY'
        elif 1.15 < pvpa <= 1.30:
            pvpa_quality = 60
            pvpa_status = 'PREMIUM'
        else:
            pvpa_quality = 40
            pvpa_status = 'HIGH PREMIUM'
        
        # Dividend Stability Score (based on yield vs SELIC)
        yield_spread = dy - self.SELIC_RATE
        if yield_spread >= 0.05:
            dividend_score = 100
            dividend_status = 'EXCELLENT'
        elif yield_spread >= 0.03:
            dividend_score = 85
            dividend_status = 'VERY GOOD'
        elif yield_spread >= 0.01:
            dividend_score = 70
            dividend_status = 'GOOD'
        elif yield_spread >= 0:
            dividend_score = 55
            dividend_status = 'FAIR'
        else:
            dividend_score = 35
            dividend_status = 'BELOW RISK-FREE'
        
        # Momentum Score
        if return_ltm >= 0.25:
            momentum_score = 100
        elif return_ltm >= 0.15:
            momentum_score = 80
        elif return_ltm >= 0.05:
            momentum_score = 60
        elif return_ltm >= 0:
            momentum_score = 45
        elif return_ltm >= -0.10:
            momentum_score = 30
        else:
            momentum_score = 15
        
        # Overall Credit Score (weighted)
        overall_score = (
            size_score * 0.25 +
            liquidity_score * 0.25 +
            pvpa_quality * 0.15 +
            dividend_score * 0.20 +
            momentum_score * 0.15
        )
        
        # Credit Rating
        if overall_score >= 85:
            rating = 'AAA'
        elif overall_score >= 75:
            rating = 'AA'
        elif overall_score >= 65:
            rating = 'A'
        elif overall_score >= 55:
            rating = 'BBB'
        elif overall_score >= 45:
            rating = 'BB'
        elif overall_score >= 35:
            rating = 'B'
        else:
            rating = 'CCC'
        
        return {
            'overall_score': overall_score,
            'rating': rating,
            'size_score': size_score,
            'size_tier': size_tier,
            'liquidity_score': liquidity_score,
            'liquidity_tier': liquidity_tier,
            'pvpa_quality': pvpa_quality,
            'pvpa_status': pvpa_status,
            'dividend_score': dividend_score,
            'dividend_status': dividend_status,
            'momentum_score': momentum_score
        }
    
    # ==================== MONTE CARLO SIMULATION ====================
    
    def run_monte_carlo(self, row: pd.Series, simulations: int = 1000) -> Dict:
        """
        Monte Carlo simulation for price projections (vectorized for speed)
        """
        price = row.get('price', 100) or 100
        dy = row.get('dy_annualized', 0.12) or 0.12
        pvpa = row.get('pvpa_current', 1.0) or 1.0
        
        # Estimate volatility from available data
        base_vol = 0.20  # Base FII volatility
        pvpa_adj = abs(pvpa - 1) * 0.1  # P/VPA deviation adds volatility
        volatility = base_vol + pvpa_adj
        
        # Expected return (dividend yield + price appreciation)
        expected_return = dy + self.IPCA
        
        # Vectorized GBM simulation (1-year horizon, simplified)
        # Using annual parameters directly for speed
        drift = expected_return - 0.5 * volatility**2
        
        # Generate all random shocks at once
        np.random.seed(abs(hash(str(row.get('ticker', '')))) % 2**31)
        annual_shocks = np.random.normal(drift, volatility, simulations)
        final_prices = price * np.exp(annual_shocks)
        
        # Calculate statistics
        mean_price = np.mean(final_prices)
        median_price = np.median(final_prices)
        percentile_5 = np.percentile(final_prices, 5)
        percentile_95 = np.percentile(final_prices, 95)
        std_price = np.std(final_prices)
        
        upside_prob = np.mean(final_prices > price) * 100
        
        return {
            'mean_price': mean_price,
            'median_price': median_price,
            'percentile_5': percentile_5,
            'percentile_95': percentile_95,
            'std_price': std_price,
            'upside_potential': (percentile_95 - price) / price * 100 if price > 0 else 0,
            'downside_risk': (percentile_5 - price) / price * 100 if price > 0 else 0,
            'upside_probability': upside_prob
        }
    
    # ==================== SENSITIVITY ANALYSIS ====================
    
    def sensitivity_analysis(self, row: pd.Series) -> Dict:
        """
        Sensitivity analysis for key variables
        """
        price = row.get('price', 100) or 100
        dy = row.get('dy_annualized', 0.12) or 0.12
        
        # Base case fair value
        base_dcf = self.calculate_dcf_valuation(row)
        base_fair_value = base_dcf.get('fair_value', price)
        
        # Sensitivity to yield changes
        yield_sensitivity = {}
        for yield_change in [-0.02, -0.01, 0, 0.01, 0.02]:
            new_dy = max(0.01, dy + yield_change)
            # Simplified: fair value inversely proportional to required return
            implied_value = (price * dy) / (self.SELIC_RATE + self.MARKET_RISK_PREMIUM * 0.5 - self.PERPETUAL_GROWTH_RATE)
            adjusted_value = implied_value * (1 + yield_change / dy) if dy > 0 else base_fair_value
            yield_sensitivity[f"{yield_change*100:+.0f}%"] = adjusted_value
        
        # Sensitivity to SELIC changes
        selic_sensitivity = {}
        for selic_change in [-0.02, -0.01, 0, 0.01, 0.02]:
            new_selic = max(0.02, self.SELIC_RATE + selic_change)
            cost_of_equity = new_selic + self.MARKET_RISK_PREMIUM * 0.5
            if cost_of_equity > self.PERPETUAL_GROWTH_RATE:
                annual_div = price * dy
                adjusted_value = annual_div * (1 + self.PERPETUAL_GROWTH_RATE) / (cost_of_equity - self.PERPETUAL_GROWTH_RATE)
            else:
                adjusted_value = base_fair_value
            selic_sensitivity[f"{selic_change*100:+.0f}%"] = adjusted_value
        
        return {
            'base_fair_value': base_fair_value,
            'yield_sensitivity': yield_sensitivity,
            'selic_sensitivity': selic_sensitivity
        }
    
    # ==================== COMPOSITE SCORE ====================
    
    def calculate_composite_score(self, row: pd.Series) -> Tuple[float, Dict]:
        """Calculate comprehensive composite score"""
        weights = {
            'pvpa': 0.20, 'yield': 0.20, 'momentum': 0.15,
            'liquidity': 0.15, 'credit': 0.10, 'dcf': 0.10, 'var': 0.10
        }
        
        scores = {}
        
        # P/VPA Score
        pvpa = row.get('pvpa_current', 1.0) or 1.0
        if pvpa <= 0.80:
            scores['pvpa'] = 100
        elif pvpa <= 0.90:
            scores['pvpa'] = 85
        elif pvpa <= 1.00:
            scores['pvpa'] = 70
        elif pvpa <= 1.10:
            scores['pvpa'] = 55
        elif pvpa <= 1.20:
            scores['pvpa'] = 40
        else:
            scores['pvpa'] = 25
        
        # Yield Score
        dy = row.get('dy_annualized', 0) or 0
        spread = dy - self.SELIC_RATE
        if spread >= 0.06:
            scores['yield'] = 100
        elif spread >= 0.04:
            scores['yield'] = 85
        elif spread >= 0.02:
            scores['yield'] = 70
        elif spread >= 0:
            scores['yield'] = 50
        else:
            scores['yield'] = 25
        
        # Momentum Score
        return_ltm = row.get('return_ltm', 0) or 0
        if return_ltm >= 0.25:
            scores['momentum'] = 100
        elif return_ltm >= 0.15:
            scores['momentum'] = 80
        elif return_ltm >= 0.05:
            scores['momentum'] = 60
        elif return_ltm >= 0:
            scores['momentum'] = 45
        elif return_ltm >= -0.10:
            scores['momentum'] = 30
        else:
            scores['momentum'] = 15
        
        # Liquidity Score
        volume = row.get('avg_daily_volume', 0) or 0
        if volume >= 2_000_000:
            scores['liquidity'] = 100
        elif volume >= 1_000_000:
            scores['liquidity'] = 85
        elif volume >= 500_000:
            scores['liquidity'] = 70
        elif volume >= 200_000:
            scores['liquidity'] = 55
        elif volume >= 100_000:
            scores['liquidity'] = 40
        else:
            scores['liquidity'] = 20
        
        # Credit Score
        credit = self.calculate_credit_analysis(row)
        scores['credit'] = credit.get('overall_score', 50)
        
        # DCF Score
        dcf = self.calculate_dcf_valuation(row)
        upside = dcf.get('upside', 0)
        if upside >= 40:
            scores['dcf'] = 100
        elif upside >= 25:
            scores['dcf'] = 85
        elif upside >= 10:
            scores['dcf'] = 70
        elif upside >= 0:
            scores['dcf'] = 55
        elif upside >= -15:
            scores['dcf'] = 40
        else:
            scores['dcf'] = 25
        
        # VaR Score (lower risk = higher score)
        returns = self.estimate_returns_from_metrics(row)
        var_result = self.calculate_var(returns, 0.95, 'parametric')
        var_95 = var_result.get('var', 0.02)
        if var_95 <= 0.015:
            scores['var'] = 100
        elif var_95 <= 0.02:
            scores['var'] = 80
        elif var_95 <= 0.03:
            scores['var'] = 60
        elif var_95 <= 0.04:
            scores['var'] = 40
        else:
            scores['var'] = 20
        
        composite = sum(scores[k] * weights[k] for k in weights.keys())
        
        return composite, scores
    
    def get_recommendation(self, score: float) -> Recommendation:
        """Convert score to recommendation"""
        if score >= 80:
            return Recommendation.STRONG_BUY
        elif score >= 65:
            return Recommendation.BUY
        elif score >= 45:
            return Recommendation.HOLD
        elif score >= 30:
            return Recommendation.SELL
        else:
            return Recommendation.STRONG_SELL
    
    def generate_reasons(self, row: pd.Series, scores: Dict, dcf: Dict, credit: Dict) -> List[str]:
        """Generate detailed reasons for recommendation"""
        reasons = []
        
        pvpa = row.get('pvpa_current', 1.0) or 1.0
        dy = row.get('dy_annualized', 0) or 0
        return_ltm = row.get('return_ltm', 0) or 0
        
        # P/VPA insights
        if pvpa < 0.85:
            reasons.append(f"Deep discount to NAV (P/VPA: {pvpa:.2f}x)")
        elif pvpa < 0.95:
            reasons.append(f"Trading below NAV (P/VPA: {pvpa:.2f}x)")
        elif pvpa > 1.15:
            reasons.append(f"Premium valuation (P/VPA: {pvpa:.2f}x)")
        
        # Yield insights
        spread = dy - self.SELIC_RATE
        if spread >= 0.04:
            reasons.append(f"Excellent yield spread ({dy*100:.1f}% vs SELIC {self.SELIC_RATE*100:.1f}%)")
        elif spread >= 0.02:
            reasons.append(f"Attractive yield ({dy*100:.1f}%)")
        elif spread < 0:
            reasons.append(f"Yield below risk-free rate ({dy*100:.1f}%)")
        
        # DCF insights
        upside = dcf.get('upside', 0)
        if upside > 30:
            reasons.append(f"Significantly undervalued by DCF (+{upside:.0f}%)")
        elif upside < -20:
            reasons.append(f"Overvalued by DCF ({upside:.0f}%)")
        
        # Credit insights
        rating = credit.get('rating', 'N/A')
        if rating in ['AAA', 'AA']:
            reasons.append(f"High credit quality ({rating})")
        elif rating in ['B', 'CCC']:
            reasons.append(f"Credit concern ({rating})")
        
        # Momentum insights
        if return_ltm > 0.20:
            reasons.append(f"Strong momentum (+{return_ltm*100:.0f}% LTM)")
        elif return_ltm < -0.15:
            reasons.append(f"Negative momentum ({return_ltm*100:.0f}% LTM)")
        
        return reasons if reasons else ["Mixed indicators"]
    
    # ==================== MAIN ANALYSIS ====================
    
    def analyze(self) -> List[AnalysisResult]:
        """Run comprehensive analysis"""
        print(f"\n{'='*70}")
        print("  Running Advanced Analysis...")
        print(f"{'='*70}\n")
        
        self.results = []
        self.results_by_segment = {}
        
        total = len(self.df)
        
        for idx, row in self.df.iterrows():
            ticker = row.get('ticker', f'Unknown_{idx}')
            
            if pd.isna(ticker) or ticker == '':
                continue
            
            # Progress indicator
            if (idx + 1) % 20 == 0:
                print(f"  Processing: {idx + 1}/{total} FIIs...")
            
            # Run all analyses
            composite_score, individual_scores = self.calculate_composite_score(row)
            recommendation = self.get_recommendation(composite_score)
            dcf = self.calculate_dcf_valuation(row)
            credit = self.calculate_credit_analysis(row)
            monte_carlo = self.run_monte_carlo(row)
            returns = self.estimate_returns_from_metrics(row)
            var_95 = self.calculate_var(returns, 0.95, 'parametric')
            var_99 = self.calculate_var(returns, 0.99, 'parametric')
            
            # Calculate additional metrics
            volatility = np.std(returns) * np.sqrt(252) if len(returns) > 0 else 0
            sharpe = (row.get('dy_annualized', 0.12) - self.SELIC_RATE) / volatility if volatility > 0 else 0
            
            reasons = self.generate_reasons(row, individual_scores, dcf, credit)
            segment = row.get('segment', 'Unknown')
            
            result = AnalysisResult(
                ticker=str(ticker),
                fund_name=str(row.get('fund_name', 'Unknown'))[:40],
                recommendation=recommendation,
                score=composite_score,
                pvpa_score=individual_scores.get('pvpa', 0),
                yield_score=individual_scores.get('yield', 0),
                momentum_score=individual_scores.get('momentum', 0),
                liquidity_score=individual_scores.get('liquidity', 0),
                credit_score=individual_scores.get('credit', 0),
                dcf_upside=dcf.get('upside', 0),
                reasons=reasons,
                segment=segment,
                var_95=var_95.get('var', 0) * 100,
                var_99=var_99.get('var', 0) * 100,
                cvar_95=var_95.get('cvar', 0) * 100,
                sharpe_ratio=sharpe,
                volatility=volatility * 100,
                fair_value=dcf.get('fair_value', 0),
                current_price=row.get('price', 0) or 0,
                credit_rating=credit.get('rating', 'N/A'),
                monte_carlo_upside=monte_carlo.get('upside_potential', 0),
                monte_carlo_downside=monte_carlo.get('downside_risk', 0),
                dividend_yield=row.get('dy_annualized', 0) or 0,
                pvpa=row.get('pvpa_current', 1.0) or 1.0,
                market_cap=row.get('market_cap', 0) or 0,
                volume=row.get('avg_daily_volume', 0) or 0
            )
            
            self.results.append(result)
            
            if segment not in self.results_by_segment:
                self.results_by_segment[segment] = []
            self.results_by_segment[segment].append(result)
        
        # Sort by score
        self.results.sort(key=lambda x: x.score, reverse=True)
        for segment in self.results_by_segment:
            self.results_by_segment[segment].sort(key=lambda x: x.score, reverse=True)
        
        print(f"\n  [OK] Analysis complete for {len(self.results)} FIIs")
        
        return self.results
    
    # ==================== VISUALIZATION ====================
    
    def create_dashboard(self, save_path: str = None):
        """Create detailed individual charts with full stock information"""
        print(f"\n{'='*70}")
        print("  Creating Detailed Individual Charts...")
        print(f"{'='*70}\n")
        
        # Create individual detailed charts
        self._chart_top_picks_detailed()
        print("  [OK] Chart 1: Top 20 Picks with Full Details")
        
        self._chart_dcf_valuation_detailed()
        print("  [OK] Chart 2: DCF Valuation Analysis")
        
        self._chart_risk_analysis_detailed()
        print("  [OK] Chart 3: Risk Analysis (VaR, Volatility)")
        
        self._chart_yield_analysis_detailed()
        print("  [OK] Chart 4: Yield & Income Analysis")
        
        self._chart_value_matrix_detailed()
        print("  [OK] Chart 5: Value Matrix with Labels")
        
        self._chart_segment_comparison()
        print("  [OK] Chart 6: Segment Comparison")
        
        self._chart_bottom_picks_detailed()
        print("  [OK] Chart 7: Bottom 15 (Stocks to Avoid)")
        
        self._chart_portfolio_summary()
        print("  [OK] Chart 8: Portfolio Summary")
        
        if save_path:
            print("\n  Saving charts to files...")
            for i, fig in enumerate(self.figures):
                fname = f"{save_path}_{i+1:02d}.png"
                fig.savefig(fname, dpi=150, bbox_inches='tight', 
                           facecolor='white', edgecolor='none')
                print(f"    -> {fname}")
                plt.close(fig)
            print(f"\n  [OK] All {len(self.figures)} charts saved successfully!")
        else:
            plt.show()
    
    def _chart_top_picks_detailed(self):
        """Chart 1: Top 20 picks with full details - one stock per row"""
        top_20 = self.results[:20]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle('TOP 20 INVESTMENT PICKS - DETAILED ANALYSIS\nRanked by Composite Score (Higher = Better)', 
                    fontsize=16, fontweight='bold', y=0.98)
        
        # Create horizontal bars
        y_pos = np.arange(len(top_20))
        bars = ax.barh(y_pos, [r.score for r in top_20], height=0.7, 
                      color=[COLORS.get(r.recommendation.name, '#999') for r in top_20],
                      edgecolor='white', linewidth=1)
        
        # Add detailed labels for each stock
        for i, r in enumerate(top_20):
            # Left side: Full stock info
            label = f"{r.ticker} | {r.segment[:20]}"
            ax.text(-2, i, label, ha='right', va='center', fontsize=10, fontweight='bold')
            
            # On the bar: Score
            ax.text(r.score - 3, i, f"{r.score:.1f}", ha='right', va='center', 
                   color='white', fontweight='bold', fontsize=11)
            
            # Right side: Key metrics
            details = f"DY: {r.dividend_yield*100:.1f}% | P/VPA: {r.pvpa:.2f}x | DCF: {r.dcf_upside:+.0f}% | VaR: {r.var_95:.1f}% | {r.recommendation.value}"
            ax.text(r.score + 1, i, details, ha='left', va='center', fontsize=9)
        
        ax.set_yticks([])
        ax.set_xlim(-5, 130)
        ax.set_ylim(-0.5, len(top_20) - 0.5)
        ax.invert_yaxis()
        ax.set_xlabel('Composite Score (0-100)', fontsize=12)
        
        # Add threshold lines
        ax.axvline(80, color='darkgreen', linestyle='--', linewidth=2, alpha=0.7, label='Strong Buy (80+)')
        ax.axvline(65, color='green', linestyle=':', linewidth=2, alpha=0.7, label='Buy (65+)')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['STRONG_BUY'], label='STRONG BUY'),
            Patch(facecolor=COLORS['BUY'], label='BUY'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Add interpretation box
        textstr = 'INTERPRETATION:\n- DY = Dividend Yield (higher = more income)\n- P/VPA = Price to Book (lower = cheaper)\n- DCF = Fair Value Upside (positive = undervalued)\n- VaR = Risk measure (lower = safer)'
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_dcf_valuation_detailed(self):
        """Chart 2: DCF Valuation with stock labels"""
        # Sort by DCF upside
        sorted_by_dcf = sorted(self.results, key=lambda x: x.dcf_upside, reverse=True)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('DCF VALUATION ANALYSIS - Finding Undervalued Stocks\nPositive DCF Upside = Stock is UNDERVALUED = BUY Signal', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Left: Top 25 Most Undervalued
        top_undervalued = sorted_by_dcf[:25]
        y_pos = np.arange(len(top_undervalued))
        colors = ['#00C853' if r.dcf_upside > 50 else '#4CAF50' if r.dcf_upside > 20 else '#8BC34A' 
                  for r in top_undervalued]
        
        bars1 = ax1.barh(y_pos, [r.dcf_upside for r in top_undervalued], color=colors, edgecolor='white')
        
        for i, r in enumerate(top_undervalued):
            ax1.text(-5, i, f"{r.ticker}", ha='right', va='center', fontsize=9, fontweight='bold')
            ax1.text(r.dcf_upside + 2, i, 
                    f"Fair: R${r.fair_value:.0f} | Now: R${r.current_price:.0f} | {r.recommendation.value}",
                    ha='left', va='center', fontsize=8)
        
        ax1.set_yticks([])
        ax1.set_xlim(-10, max([r.dcf_upside for r in top_undervalued]) + 80)
        ax1.invert_yaxis()
        ax1.set_xlabel('DCF Upside (%) - How much stock could rise to fair value')
        ax1.set_title('TOP 25 MOST UNDERVALUED\n(Biggest Upside Potential)', fontweight='bold', color='darkgreen')
        ax1.axvline(0, color='black', linewidth=2)
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Most Overvalued (bottom 25)
        bottom_overvalued = sorted_by_dcf[-25:][::-1]
        y_pos = np.arange(len(bottom_overvalued))
        colors = ['#D32F2F' if r.dcf_upside < -20 else '#FF5722' if r.dcf_upside < -10 else '#FF9800' 
                  for r in bottom_overvalued]
        
        bars2 = ax2.barh(y_pos, [r.dcf_upside for r in bottom_overvalued], color=colors, edgecolor='white')
        
        for i, r in enumerate(bottom_overvalued):
            ax2.text(min(r.dcf_upside, -5) - 2, i, f"{r.ticker}", ha='right', va='center', fontsize=9, fontweight='bold')
            ax2.text(2, i, 
                    f"Fair: R${r.fair_value:.0f} | Now: R${r.current_price:.0f} | {r.recommendation.value}",
                    ha='left', va='center', fontsize=8)
        
        ax2.set_yticks([])
        ax2.set_xlim(min([r.dcf_upside for r in bottom_overvalued]) - 20, max([r.dcf_upside for r in bottom_overvalued]) + 60)
        ax2.invert_yaxis()
        ax2.set_xlabel('DCF Upside (%) - Negative means OVERVALUED')
        ax2.set_title('TOP 25 MOST OVERVALUED\n(Avoid - Downside Risk)', fontweight='bold', color='darkred')
        ax2.axvline(0, color='black', linewidth=2)
        ax2.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_risk_analysis_detailed(self):
        """Chart 3: Risk analysis with stock labels"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        fig.suptitle('RISK ANALYSIS - Identifying Safe vs Risky Investments\nLower VaR and Volatility = Safer Investment', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Top Left: Safest Stocks (Lowest VaR)
        sorted_by_var = sorted([r for r in self.results if r.var_95 > 0], key=lambda x: x.var_95)
        safest_20 = sorted_by_var[:20]
        
        y_pos = np.arange(len(safest_20))
        colors = ['#4CAF50' if r.var_95 < 2.0 else '#8BC34A' if r.var_95 < 2.5 else '#FFC107' 
                  for r in safest_20]
        ax1.barh(y_pos, [r.var_95 for r in safest_20], color=colors, edgecolor='white')
        
        for i, r in enumerate(safest_20):
            ax1.text(-0.1, i, f"{r.ticker}", ha='right', va='center', fontsize=9, fontweight='bold')
            ax1.text(r.var_95 + 0.1, i, f"Vol: {r.volatility:.0f}% | {r.segment[:15]} | {r.recommendation.value}",
                    ha='left', va='center', fontsize=8)
        
        ax1.set_yticks([])
        ax1.invert_yaxis()
        ax1.set_xlabel('VaR 95% (Daily) - Max expected loss with 95% confidence')
        ax1.set_title('TOP 20 SAFEST STOCKS\n(Lowest Risk - Best for Conservative Investors)', 
                     fontweight='bold', color='darkgreen')
        ax1.axvline(2.0, color='green', linestyle='--', linewidth=2, label='Low Risk Threshold')
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        
        # Top Right: Riskiest Stocks (Highest VaR)
        riskiest_20 = sorted_by_var[-20:][::-1]
        
        y_pos = np.arange(len(riskiest_20))
        colors = ['#D32F2F' if r.var_95 > 4.0 else '#FF5722' if r.var_95 > 3.0 else '#FF9800' 
                  for r in riskiest_20]
        ax2.barh(y_pos, [r.var_95 for r in riskiest_20], color=colors, edgecolor='white')
        
        for i, r in enumerate(riskiest_20):
            ax2.text(-0.1, i, f"{r.ticker}", ha='right', va='center', fontsize=9, fontweight='bold')
            ax2.text(r.var_95 + 0.1, i, f"Vol: {r.volatility:.0f}% | {r.segment[:15]} | {r.recommendation.value}",
                    ha='left', va='center', fontsize=8)
        
        ax2.set_yticks([])
        ax2.invert_yaxis()
        ax2.set_xlabel('VaR 95% (Daily) - Higher = More Risky')
        ax2.set_title('TOP 20 RISKIEST STOCKS\n(High Volatility - Only for Aggressive Investors)', 
                     fontweight='bold', color='darkred')
        ax2.axvline(3.0, color='red', linestyle='--', linewidth=2, label='High Risk Threshold')
        ax2.legend(loc='lower right')
        ax2.grid(axis='x', alpha=0.3)
        
        # Bottom Left: Best Sharpe Ratios
        sorted_by_sharpe = sorted([r for r in self.results if not np.isnan(r.sharpe_ratio)], 
                                  key=lambda x: x.sharpe_ratio, reverse=True)
        best_sharpe_20 = sorted_by_sharpe[:20]
        
        y_pos = np.arange(len(best_sharpe_20))
        colors = ['#00C853' if r.sharpe_ratio > 0.5 else '#4CAF50' if r.sharpe_ratio > 0.2 else '#FFC107' 
                  for r in best_sharpe_20]
        ax3.barh(y_pos, [r.sharpe_ratio for r in best_sharpe_20], color=colors, edgecolor='white')
        
        for i, r in enumerate(best_sharpe_20):
            ax3.text(-0.05, i, f"{r.ticker}", ha='right', va='center', fontsize=9, fontweight='bold')
            ax3.text(r.sharpe_ratio + 0.02, i, f"DY: {r.dividend_yield*100:.1f}% | VaR: {r.var_95:.1f}% | {r.recommendation.value}",
                    ha='left', va='center', fontsize=8)
        
        ax3.set_yticks([])
        ax3.invert_yaxis()
        ax3.set_xlabel('Sharpe Ratio - Return per unit of risk (higher = better risk-adjusted return)')
        ax3.set_title('BEST RISK-ADJUSTED RETURNS\n(Highest Sharpe Ratio - Best Return for Risk Taken)', 
                     fontweight='bold', color='darkgreen')
        ax3.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Good Threshold')
        ax3.axvline(0, color='gray', linestyle='-', linewidth=1)
        ax3.legend(loc='lower right')
        ax3.grid(axis='x', alpha=0.3)
        
        # Bottom Right: Credit Rating Distribution with examples
        ratings = ['AAA', 'AA', 'A', 'BBB', 'BB', 'B', 'CCC']
        rating_counts = {r: 0 for r in ratings}
        rating_examples = {r: [] for r in ratings}
        
        for r in self.results:
            if r.credit_rating in rating_counts:
                rating_counts[r.credit_rating] += 1
                if len(rating_examples[r.credit_rating]) < 5:
                    rating_examples[r.credit_rating].append(r.ticker)
        
        colors = ['#00C853', '#4CAF50', '#8BC34A', '#FFC107', '#FF9800', '#FF5722', '#D32F2F']
        bars = ax4.bar(ratings, [rating_counts[r] for r in ratings], color=colors, edgecolor='white')
        
        for i, (rating, bar) in enumerate(zip(ratings, bars)):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    str(rating_counts[rating]), ha='center', fontweight='bold')
            examples = rating_examples[rating][:3]
            if examples:
                ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                        '\n'.join(examples), ha='center', va='center', fontsize=7, color='white')
        
        ax4.set_xlabel('Credit Rating')
        ax4.set_ylabel('Number of FIIs')
        ax4.set_title('CREDIT RATING DISTRIBUTION\nAAA-A = Investment Grade (Safe) | BBB-CCC = Speculative (Risky)', 
                     fontweight='bold')
        ax4.axvline(2.5, color='gray', linestyle='--', linewidth=2, label='Investment Grade Cutoff')
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_yield_analysis_detailed(self):
        """Chart 4: Yield analysis with stock details"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 12))
        fig.suptitle('DIVIDEND YIELD ANALYSIS - Finding High Income Investments\nDividend Yield = Annual income as percentage of investment', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Left: Highest Yielding Stocks
        sorted_by_yield = sorted(self.results, key=lambda x: x.dividend_yield, reverse=True)
        top_yield_25 = sorted_by_yield[:25]
        
        y_pos = np.arange(len(top_yield_25))
        colors = ['#00C853' if r.dividend_yield > 0.15 else '#4CAF50' if r.dividend_yield > 0.12 else '#8BC34A' 
                  for r in top_yield_25]
        
        ax1.barh(y_pos, [r.dividend_yield * 100 for r in top_yield_25], color=colors, edgecolor='white')
        
        for i, r in enumerate(top_yield_25):
            ax1.text(-0.5, i, f"{r.ticker}", ha='right', va='center', fontsize=9, fontweight='bold')
            yield_vs_selic = r.dividend_yield * 100 - 11.75
            status = f"+{yield_vs_selic:.1f}% vs SELIC" if yield_vs_selic > 0 else f"{yield_vs_selic:.1f}% vs SELIC"
            ax1.text(r.dividend_yield * 100 + 0.5, i, 
                    f"{r.segment[:15]} | P/VPA: {r.pvpa:.2f}x | {status} | {r.recommendation.value}",
                    ha='left', va='center', fontsize=8)
        
        ax1.set_yticks([])
        ax1.invert_yaxis()
        ax1.set_xlabel('Dividend Yield (%) - Annual income return')
        ax1.set_title('TOP 25 HIGHEST YIELDING\n(Best for Income Investors)', fontweight='bold', color='darkgreen')
        ax1.axvline(11.75, color='blue', linestyle='--', linewidth=2, label=f'SELIC Rate: 11.75%')
        ax1.axvline(15, color='green', linestyle=':', linewidth=2, label='Excellent Yield: 15%+')
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Yield vs P/VPA Scatter (Finding Value + Income)
        for r in self.results:
            color = COLORS.get(r.recommendation.name, '#999')
            ax2.scatter(r.pvpa, r.dividend_yield * 100, c=color, s=100, alpha=0.6, edgecolors='white')
        
        # Add quadrants
        ax2.axhline(11.75, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='SELIC 11.75%')
        ax2.axvline(1.0, color='gray', linestyle='--', linewidth=2, alpha=0.7, label='Fair Value (P/VPA=1)')
        
        # Highlight ideal zone
        ax2.fill_between([0, 1.0], 11.75, 40, alpha=0.15, color='green')
        ax2.text(0.5, 25, 'IDEAL ZONE\nHigh Yield +\nDiscount to NAV', fontsize=10, ha='center', 
                fontweight='bold', color='darkgreen')
        
        # Label top picks in ideal zone
        ideal_picks = [r for r in self.results if r.pvpa < 1.0 and r.dividend_yield > 0.1175]
        ideal_picks = sorted(ideal_picks, key=lambda x: x.score, reverse=True)[:10]
        for r in ideal_picks:
            ax2.annotate(r.ticker, (r.pvpa, r.dividend_yield * 100), fontsize=8, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='yellow', alpha=0.7))
        
        ax2.set_xlabel('P/VPA (Price to Book) - Lower = Cheaper stock')
        ax2.set_ylabel('Dividend Yield (%) - Higher = More income')
        ax2.set_title('YIELD vs P/VPA: Finding Value + Income\nIdeal: Low P/VPA + High Yield', fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 2.5)
        ax2.set_ylim(0, 40)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_value_matrix_detailed(self):
        """Chart 5: Value Matrix with all stock labels"""
        fig, ax = plt.subplots(figsize=(18, 14))
        fig.suptitle('VALUE MATRIX - P/VPA Score vs DCF Upside\nTop-Right = BEST (Discounted + Undervalued) | Bottom-Left = WORST', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Create scatter with stock labels
        for r in self.results:
            color = COLORS.get(r.recommendation.name, '#999')
            ax.scatter(r.pvpa_score, r.dcf_upside, c=color, s=150, alpha=0.6, edgecolors='white', linewidth=1)
        
        # Add quadrant zones
        ax.axhline(0, color='black', linestyle='-', linewidth=2)
        ax.axvline(50, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        # Highlight zones
        ax.fill_between([60, 100], 0, 150, alpha=0.1, color='green')
        ax.fill_between([0, 40], -60, 0, alpha=0.1, color='red')
        
        # Zone labels
        ax.text(80, 100, 'BEST ZONE\nDiscounted + Undervalued\nSTRONG BUY', fontsize=11, ha='center', 
               fontweight='bold', color='darkgreen', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        ax.text(20, -40, 'WORST ZONE\nExpensive + Overvalued\nAVOID', fontsize=11, ha='center', 
               fontweight='bold', color='darkred', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))
        
        # Label all stocks in best zone (top 15)
        best_zone = [r for r in self.results if r.pvpa_score > 60 and r.dcf_upside > 20]
        best_zone = sorted(best_zone, key=lambda x: x.score, reverse=True)[:15]
        for r in best_zone:
            ax.annotate(f"{r.ticker}\nDCF:{r.dcf_upside:+.0f}%", (r.pvpa_score, r.dcf_upside), 
                       fontsize=7, fontweight='bold', ha='center',
                       xytext=(0, 8), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightgreen', alpha=0.8))
        
        # Label stocks in worst zone
        worst_zone = [r for r in self.results if r.pvpa_score < 40 and r.dcf_upside < -10]
        worst_zone = sorted(worst_zone, key=lambda x: x.dcf_upside)[:10]
        for r in worst_zone:
            ax.annotate(f"{r.ticker}\nDCF:{r.dcf_upside:+.0f}%", (r.pvpa_score, r.dcf_upside), 
                       fontsize=7, fontweight='bold', ha='center',
                       xytext=(0, -12), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='lightcoral', alpha=0.8))
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['STRONG_BUY'], label='STRONG BUY'),
            Patch(facecolor=COLORS['BUY'], label='BUY'),
            Patch(facecolor=COLORS['HOLD'], label='HOLD'),
            Patch(facecolor=COLORS['SELL'], label='SELL'),
            Patch(facecolor=COLORS['STRONG_SELL'], label='STRONG SELL'),
        ]
        ax.legend(handles=legend_elements, loc='lower left', fontsize=10, title='Recommendation')
        
        ax.set_xlabel('P/VPA Score (0-100)\nExpensive (Premium to NAV) <-- --> Cheap (Discount to NAV)', fontsize=11)
        ax.set_ylabel('DCF Upside (%)\nOvervalued <-- --> Undervalued', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 100)
        
        # Add interpretation box
        textstr = 'HOW TO READ:\n- X-axis: P/VPA Score - Higher = stock trades at discount to book value\n- Y-axis: DCF Upside - Positive = stock is undervalued vs fair value\n- Color: Recommendation based on composite score\n- IDEAL: Top-right quadrant (high P/VPA score + positive DCF)'
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_segment_comparison(self):
        """Chart 6: Segment comparison table"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))
        fig.suptitle('SEGMENT ANALYSIS - Comparing FII Sectors\nWhich segments offer best value?', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        # Prepare segment data
        segment_data = []
        for segment, results in self.results_by_segment.items():
            segment_data.append({
                'segment': segment[:18],
                'count': len(results),
                'avg_score': np.mean([r.score for r in results]),
                'avg_yield': np.mean([r.dividend_yield for r in results]) * 100,
                'avg_pvpa': np.mean([r.pvpa for r in results]),
                'avg_dcf': np.mean([r.dcf_upside for r in results]),
                'avg_var': np.mean([r.var_95 for r in results]),
                'top_pick': max(results, key=lambda x: x.score).ticker,
                'buy_pct': sum(1 for r in results if r.recommendation.name in ['STRONG_BUY', 'BUY']) / len(results) * 100
            })
        
        # Sort by average score
        segment_data = sorted(segment_data, key=lambda x: x['avg_score'], reverse=True)
        
        # Left: Score ranking
        segments = [d['segment'] for d in segment_data]
        scores = [d['avg_score'] for d in segment_data]
        colors = [COLORS['STRONG_BUY'] if s >= 75 else COLORS['BUY'] if s >= 65 else 
                 COLORS['HOLD'] if s >= 50 else COLORS['SELL'] for s in scores]
        
        y_pos = np.arange(len(segments))
        bars = ax1.barh(y_pos, scores, color=colors, edgecolor='white')
        
        for i, d in enumerate(segment_data):
            ax1.text(-2, i, d['segment'], ha='right', va='center', fontsize=10, fontweight='bold')
            ax1.text(d['avg_score'] + 1, i, 
                    f"Top: {d['top_pick']} | {d['buy_pct']:.0f}% Buy | DY: {d['avg_yield']:.0f}%",
                    ha='left', va='center', fontsize=9)
        
        ax1.set_yticks([])
        ax1.invert_yaxis()
        ax1.set_xlim(-5, 110)
        ax1.set_xlabel('Average Composite Score')
        ax1.set_title('SEGMENT RANKING BY SCORE\n(Higher = Better Sector)', fontweight='bold')
        ax1.axvline(65, color='green', linestyle='--', linewidth=2, label='Buy Threshold')
        ax1.legend(loc='lower right')
        ax1.grid(axis='x', alpha=0.3)
        
        # Right: Key metrics comparison table
        ax2.axis('off')
        
        # Create table data
        table_data = []
        headers = ['Segment', 'Count', 'Score', 'Yield', 'P/VPA', 'DCF%', 'VaR%', 'Top Pick']
        for d in segment_data:
            table_data.append([
                d['segment'][:15],
                str(d['count']),
                f"{d['avg_score']:.0f}",
                f"{d['avg_yield']:.1f}%",
                f"{d['avg_pvpa']:.2f}x",
                f"{d['avg_dcf']:+.0f}%",
                f"{d['avg_var']:.1f}%",
                d['top_pick']
            ])
        
        table = ax2.table(cellText=table_data, colLabels=headers, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.8)
        
        # Color header
        for j, header in enumerate(headers):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        
        # Color rows by score
        for i, d in enumerate(segment_data):
            color = '#E2EFDA' if d['avg_score'] >= 65 else '#FFF2CC' if d['avg_score'] >= 50 else '#FCE4D6'
            for j in range(len(headers)):
                table[(i+1, j)].set_facecolor(color)
        
        ax2.set_title('SEGMENT METRICS COMPARISON TABLE', fontweight='bold', pad=20)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_bottom_picks_detailed(self):
        """Chart 7: Bottom picks to avoid"""
        bottom_20 = self.results[-20:][::-1]
        
        fig, ax = plt.subplots(figsize=(16, 12))
        fig.suptitle('BOTTOM 20 STOCKS TO AVOID - Weakest Investments\nLow scores indicate poor value, high risk, or overvaluation', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        y_pos = np.arange(len(bottom_20))
        colors = [COLORS.get(r.recommendation.name, '#999') for r in bottom_20]
        bars = ax.barh(y_pos, [r.score for r in bottom_20], color=colors, edgecolor='white')
        
        for i, r in enumerate(bottom_20):
            ax.text(-2, i, f"{r.ticker} | {r.segment[:18]}", ha='right', va='center', fontsize=10, fontweight='bold')
            ax.text(r.score + 1, i, 
                   f"DY: {r.dividend_yield*100:.1f}% | P/VPA: {r.pvpa:.2f}x | DCF: {r.dcf_upside:+.0f}% | VaR: {r.var_95:.1f}% | {r.recommendation.value}",
                   ha='left', va='center', fontsize=9)
        
        ax.set_yticks([])
        ax.set_xlim(-5, 100)
        ax.invert_yaxis()
        ax.set_xlabel('Composite Score (0-100) - Lower = Worse Investment')
        
        # Add threshold lines
        ax.axvline(45, color='orange', linestyle='--', linewidth=2, label='Hold/Sell Boundary')
        ax.axvline(30, color='red', linestyle='--', linewidth=2, label='Sell/Strong Sell Boundary')
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=COLORS['HOLD'], label='HOLD'),
            Patch(facecolor=COLORS['SELL'], label='SELL'),
            Patch(facecolor=COLORS['STRONG_SELL'], label='STRONG SELL'),
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=10)
        
        # Warning box
        textstr = 'WARNING: These stocks scored poorly due to:\n- High P/VPA (trading at premium to book value)\n- Low or negative DCF upside (overvalued)\n- High VaR (risky/volatile)\n- Poor dividend yield relative to risk\n- Weak momentum or liquidity issues'
        ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        self.figures.append(fig)
    
    def _chart_portfolio_summary(self):
        """Chart 8: Overall portfolio summary"""
        fig = plt.figure(figsize=(18, 12))
        fig.suptitle('PORTFOLIO SUMMARY - Complete Investment Overview\n' + 
                    f'Analysis Date: {datetime.now().strftime("%Y-%m-%d")} | Total FIIs Analyzed: {len(self.results)}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Recommendation Distribution
        ax1 = fig.add_subplot(gs[0, 0])
        rec_counts = {}
        for rec in Recommendation:
            count = sum(1 for r in self.results if r.recommendation == rec)
            if count > 0:
                rec_counts[rec.value] = count
        
        colors = [COLORS.get(rec.name, '#999') for rec in Recommendation if rec.value in rec_counts]
        wedges, texts, autotexts = ax1.pie(
            rec_counts.values(), labels=None, autopct='', colors=colors, 
            explode=[0.05] * len(rec_counts), shadow=True, startangle=90
        )
        
        # Add legend with counts
        legend_labels = [f"{k}: {v} ({v/len(self.results)*100:.0f}%)" for k, v in rec_counts.items()]
        ax1.legend(wedges, legend_labels, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
        ax1.set_title('Recommendation Distribution', fontweight='bold')
        
        # 2. Key Metrics Summary
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.axis('off')
        
        metrics = [
            ['METRIC', 'AVERAGE', 'INTERPRETATION'],
            ['Dividend Yield', f"{np.mean([r.dividend_yield for r in self.results])*100:.1f}%", 
             'Above SELIC (11.75%) = Good'],
            ['P/VPA', f"{np.mean([r.pvpa for r in self.results]):.2f}x", 
             'Below 1.0 = Discount to NAV'],
            ['DCF Upside', f"{np.mean([r.dcf_upside for r in self.results]):+.0f}%", 
             'Positive = Undervalued'],
            ['VaR 95%', f"{np.mean([r.var_95 for r in self.results]):.2f}%", 
             'Below 2.5% = Low Risk'],
            ['Sharpe Ratio', f"{np.nanmean([r.sharpe_ratio for r in self.results]):.2f}", 
             'Above 0.5 = Good'],
            ['Composite Score', f"{np.mean([r.score for r in self.results]):.0f}/100", 
             'Above 65 = Buy'],
        ]
        
        table = ax2.table(cellText=metrics[1:], colLabels=metrics[0], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.8)
        for j in range(3):
            table[(0, j)].set_facecolor('#4472C4')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        ax2.set_title('Key Metrics Summary', fontweight='bold', pad=20)
        
        # 3. Score Distribution
        ax3 = fig.add_subplot(gs[0, 2])
        scores = [r.score for r in self.results]
        n, bins, patches = ax3.hist(scores, bins=15, edgecolor='white', alpha=0.8)
        for i, patch in enumerate(patches):
            if bins[i] >= 80:
                patch.set_facecolor(COLORS['STRONG_BUY'])
            elif bins[i] >= 65:
                patch.set_facecolor(COLORS['BUY'])
            elif bins[i] >= 45:
                patch.set_facecolor(COLORS['HOLD'])
            else:
                patch.set_facecolor(COLORS['SELL'])
        ax3.axvline(65, color='green', linestyle='--', linewidth=2)
        ax3.axvline(45, color='orange', linestyle='--', linewidth=2)
        ax3.set_xlabel('Score')
        ax3.set_ylabel('Count')
        ax3.set_title('Score Distribution', fontweight='bold')
        
        # 4. Top 10 List
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.axis('off')
        
        top_10_data = [['Rank', 'Ticker', 'Segment', 'Score', 'Recommendation']]
        for i, r in enumerate(self.results[:10]):
            top_10_data.append([str(i+1), r.ticker, r.segment[:15], f"{r.score:.0f}", r.recommendation.value])
        
        table = ax4.table(cellText=top_10_data[1:], colLabels=top_10_data[0], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)
        for j in range(5):
            table[(0, j)].set_facecolor('#00C853')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        ax4.set_title('TOP 10 PICKS', fontweight='bold', color='darkgreen', pad=20)
        
        # 5. Bottom 10 List  
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.axis('off')
        
        bottom_10_data = [['Rank', 'Ticker', 'Segment', 'Score', 'Recommendation']]
        for i, r in enumerate(self.results[-10:][::-1]):
            bottom_10_data.append([str(i+1), r.ticker, r.segment[:15], f"{r.score:.0f}", r.recommendation.value])
        
        table = ax5.table(cellText=bottom_10_data[1:], colLabels=bottom_10_data[0], loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.2, 1.6)
        for j in range(5):
            table[(0, j)].set_facecolor('#D32F2F')
            table[(0, j)].set_text_props(color='white', fontweight='bold')
        ax5.set_title('BOTTOM 10 (AVOID)', fontweight='bold', color='darkred', pad=20)
        
        # 6. Investment Action Summary
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.axis('off')
        
        buy_count = sum(1 for r in self.results if r.recommendation.name in ['STRONG_BUY', 'BUY'])
        hold_count = sum(1 for r in self.results if r.recommendation.name == 'HOLD')
        sell_count = sum(1 for r in self.results if r.recommendation.name in ['SELL', 'STRONG_SELL'])
        
        summary = f"""
INVESTMENT ACTION SUMMARY
{'='*40}

[+] BUY SIGNALS: {buy_count} FIIs ({buy_count/len(self.results)*100:.0f}%)
    Strong Buy: {sum(1 for r in self.results if r.recommendation.name == 'STRONG_BUY')}
    Buy: {sum(1 for r in self.results if r.recommendation.name == 'BUY')}

[~] HOLD: {hold_count} FIIs ({hold_count/len(self.results)*100:.0f}%)
    Monitor these positions

[-] SELL SIGNALS: {sell_count} FIIs ({sell_count/len(self.results)*100:.0f}%)
    Sell: {sum(1 for r in self.results if r.recommendation.name == 'SELL')}
    Strong Sell: {sum(1 for r in self.results if r.recommendation.name == 'STRONG_SELL')}

{'='*40}
BEST SECTOR: {max(self.results_by_segment.items(), key=lambda x: np.mean([r.score for r in x[1]]))[0]}
WORST SECTOR: {min(self.results_by_segment.items(), key=lambda x: np.mean([r.score for r in x[1]]))[0]}
"""
        ax6.text(0.1, 0.95, summary, transform=ax6.transAxes, fontsize=10,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax6.set_title('Action Summary', fontweight='bold', pad=20)
        
        plt.tight_layout()
        self.figures.append(fig)
    
    # Keep old methods for compatibility (deprecated)
    def _create_executive_summary(self):
        """DEPRECATED - Use new chart methods instead"""
        pass
        
    def _create_segment_analysis(self):
        """DEPRECATED - Use new chart methods instead"""
        pass
        
    def _create_risk_analysis(self):
        """DEPRECATED - Use new chart methods instead"""
        pass
    
    def _create_top_picks_analysis(self):
        """DEPRECATED - Use new chart methods instead"""
        pass
    
    def _create_valuation_analysis(self):
        """DEPRECATED - Use new chart methods instead"""
        pass
    
    # ==================== PRINT SUMMARY ====================
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*120)
        print(" " * 40 + "ADVANCED FII ANALYSIS REPORT")
        print(" " * 40 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*120)
        
        colors = [COLORS.get(rec.name, '#999') for rec in Recommendation if rec.value in rec_counts]
        wedges, texts, autotexts = ax1.pie(
            rec_counts.values(), labels=rec_counts.keys(), autopct='%1.0f%%',
            colors=colors, explode=[0.05] * len(rec_counts), shadow=True
        )
        ax1.set_title('Recommendation Distribution\nGreen=BUY | Yellow=HOLD | Red=SELL', fontweight='bold', fontsize=10)
        
        # 2. Score Distribution (Histogram)
        ax2 = fig.add_subplot(gs[0, 1])
        scores = [r.score for r in self.results]
        n, bins, patches = ax2.hist(scores, bins=20, edgecolor='white', alpha=0.8)
        # Color bars by score range
        for i, patch in enumerate(patches):
            if bins[i] >= 80:
                patch.set_facecolor(COLORS['STRONG_BUY'])
            elif bins[i] >= 65:
                patch.set_facecolor(COLORS['BUY'])
            elif bins[i] >= 45:
                patch.set_facecolor(COLORS['HOLD'])
            elif bins[i] >= 30:
                patch.set_facecolor(COLORS['SELL'])
            else:
                patch.set_facecolor(COLORS['STRONG_SELL'])
        ax2.axvline(80, color='darkgreen', linestyle='-', linewidth=2, alpha=0.7)
        ax2.axvline(65, color='green', linestyle='--', linewidth=2, alpha=0.7)
        ax2.axvline(45, color='orange', linestyle='--', linewidth=2, alpha=0.7)
        ax2.axvline(30, color='red', linestyle='--', linewidth=2, alpha=0.7)
        ax2.text(82, ax2.get_ylim()[1]*0.9, 'STRONG\nBUY', fontsize=7, color='darkgreen', fontweight='bold')
        ax2.text(67, ax2.get_ylim()[1]*0.9, 'BUY', fontsize=8, color='green', fontweight='bold')
        ax2.text(47, ax2.get_ylim()[1]*0.9, 'HOLD', fontsize=8, color='orange', fontweight='bold')
        ax2.text(32, ax2.get_ylim()[1]*0.9, 'SELL', fontsize=8, color='red', fontweight='bold')
        ax2.set_xlabel('Composite Score (0-100)\n↑ Higher = Better Investment')
        ax2.set_ylabel('Number of FIIs')
        ax2.set_title('Score Distribution by Rating Zone', fontweight='bold')
        
        # 3. Top 10 by Score (Horizontal Bar)
        ax3 = fig.add_subplot(gs[0, 2])
        top_10 = self.results[:10]
        y_pos = np.arange(len(top_10))
        colors_bar = [COLORS.get(r.recommendation.name, '#999') for r in top_10]
        bars = ax3.barh(y_pos, [r.score for r in top_10], color=colors_bar, edgecolor='white')
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels([f"{r.ticker} ({r.credit_rating})" for r in top_10])
        ax3.invert_yaxis()
        # Add score labels on bars
        for i, (bar, r) in enumerate(zip(bars, top_10)):
            ax3.text(bar.get_width() - 5, bar.get_y() + bar.get_height()/2, 
                    f'{r.score:.0f}', va='center', ha='right', color='white', fontweight='bold', fontsize=9)
        ax3.set_xlabel('Score (Higher = Better)')
        ax3.set_title('TOP 10 FIIs - Best Picks\nAll rated BUY or STRONG BUY', fontweight='bold', fontsize=10)
        ax3.axvline(80, color='darkgreen', linestyle=':', alpha=0.5)
        
        # 4. Yield vs P/VPA Scatter
        ax4 = fig.add_subplot(gs[1, :2])
        for rec in Recommendation:
            subset = [r for r in self.results if r.recommendation == rec]
            if subset:
                x = [r.pvpa_score for r in subset]
                y = [r.yield_score for r in subset]
                ax4.scatter(x, y, c=COLORS.get(rec.name, '#999'), label=rec.value, 
                           alpha=0.7, s=80, edgecolors='white')
        
        # Add quadrant labels with interpretations
        ax4.axhline(50, color='gray', linestyle='--', alpha=0.4)
        ax4.axvline(50, color='gray', linestyle='--', alpha=0.4)
        ax4.fill_between([70, 100], 70, 100, alpha=0.1, color='green')
        ax4.fill_between([0, 30], 0, 30, alpha=0.1, color='red')
        ax4.text(85, 85, 'IDEAL\nHigh Yield +\nGood Discount', fontsize=9, ha='center', 
                color='darkgreen', fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        ax4.text(15, 15, 'AVOID\nLow Yield +\nOverpriced', fontsize=9, ha='center', 
                color='darkred', fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        ax4.text(85, 15, 'REVIEW\nGood Discount\nLow Yield', fontsize=8, ha='center', color='gray')
        ax4.text(15, 85, 'REVIEW\nHigh Yield\nOverpriced', fontsize=8, ha='center', color='gray')
        ax4.set_xlabel('P/VPA Score\n← Overpriced (BAD) | Discounted (GOOD) →')
        ax4.set_ylabel('Yield Score\n← Low Yield (BAD) | High Yield (GOOD) →')
        ax4.set_title('Yield vs P/VPA Analysis - Finding Value Investments', fontweight='bold')
        ax4.legend(loc='center left', bbox_to_anchor=(1.01, 0.5), title='Recommendation')
        ax4.grid(True, alpha=0.3)
        ax4.set_xlim(0, 100)
        ax4.set_ylim(0, 100)
        
        # 5. Segment Performance
        ax5 = fig.add_subplot(gs[1, 2])
        segment_scores = {}
        for segment, results in self.results_by_segment.items():
            segment_scores[segment[:15]] = np.mean([r.score for r in results])
        
        # Sort by score
        sorted_segments = sorted(segment_scores.items(), key=lambda x: x[1], reverse=True)
        segments = [s[0] for s in sorted_segments]
        avg_scores = [s[1] for s in sorted_segments]
        
        # Color by score
        colors_seg = [COLORS['STRONG_BUY'] if s >= 75 else COLORS['BUY'] if s >= 65 else 
                     COLORS['HOLD'] if s >= 45 else COLORS['SELL'] for s in avg_scores]
        
        bars = ax5.barh(segments, avg_scores, color=colors_seg, edgecolor='white')
        ax5.set_xlabel('Average Score')
        ax5.set_title('Segment Ranking\n(Best to Worst)', fontweight='bold')
        ax5.axvline(65, color='green', linestyle='--', alpha=0.7, linewidth=2)
        ax5.axvline(45, color='orange', linestyle='--', alpha=0.7, linewidth=2)
        ax5.text(66, len(segments)-0.5, 'BUY', fontsize=8, color='green', fontweight='bold')
        ax5.text(40, len(segments)-0.5, '←HOLD', fontsize=8, color='orange', fontweight='bold')
        
        # 6. Risk-Return Profile
        ax6 = fig.add_subplot(gs[2, :])
        x = [r.volatility for r in self.results]
        y = [r.dcf_upside for r in self.results]
        sizes = [r.score * 2 for r in self.results]
        colors_scatter = [COLORS.get(r.recommendation.name, '#999') for r in self.results]
        
        scatter = ax6.scatter(x, y, c=colors_scatter, s=sizes, alpha=0.6, edgecolors='white')
        ax6.axhline(0, color='black', linestyle='-', linewidth=2, alpha=0.5)
        avg_vol = np.mean(x)
        ax6.axvline(avg_vol, color='gray', linestyle='--', alpha=0.5)
        
        # Add quadrant zones with descriptions
        ax6.fill_between([0, avg_vol], 0, 100, alpha=0.08, color='green')
        ax6.fill_between([avg_vol, 50], -50, 0, alpha=0.08, color='red')
        ax6.text(5, 60, 'BEST ZONE\nLow Risk + High Upside', fontsize=10, color='darkgreen', 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))
        ax6.text(35, -35, 'DANGER ZONE\nHigh Risk + Downside', fontsize=10, color='darkred', 
                fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
        ax6.text(avg_vol+1, 2, f'Avg Vol: {avg_vol:.1f}%', fontsize=8, color='gray')
        
        ax6.set_xlabel('Volatility (%)\n← Lower Risk (GOOD) | Higher Risk (BAD) →')
        ax6.set_ylabel('DCF Upside (%)\n← Overvalued (BAD) | Undervalued (GOOD) →')
        ax6.set_title('Risk-Return Profile: Finding Low-Risk, High-Upside Investments\n(Bubble Size = Overall Score)', fontweight='bold')
        
        # Add annotations for top picks
        for r in self.results[:5]:
            ax6.annotate(f'{r.ticker}\n+{r.dcf_upside:.0f}%', (r.volatility, r.dcf_upside), fontsize=8, 
                        ha='left', va='bottom', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        self.figures.append(fig)
    
    # ==================== PRINT SUMMARY ====================
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*120)
        print(" " * 40 + "ADVANCED FII ANALYSIS REPORT")
        print(" " * 40 + f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        print("="*120)
        
        # Overall Statistics
        print(f"\nPORTFOLIO OVERVIEW")
        print("-"*60)
        print(f"  Total FIIs Analyzed: {len(self.results)}")
        print(f"  Segments Covered: {len(self.results_by_segment)}")
        
        rec_counts = {}
        for rec in Recommendation:
            rec_counts[rec.value] = sum(1 for r in self.results if r.recommendation == rec)
        
        print("\n  Recommendation Distribution:")
        for rec, count in rec_counts.items():
            pct = count / len(self.results) * 100
            bar = "█" * int(pct / 2)
            print(f"    {rec:12} : {count:3} ({pct:5.1f}%) {bar}")
        
        # Risk Metrics
        print(f"\nRISK METRICS SUMMARY")
        print("-"*60)
        var_values = [r.var_95 for r in self.results if not np.isnan(r.var_95)]
        cvar_values = [r.cvar_95 for r in self.results if not np.isnan(r.cvar_95)]
        vol_values = [r.volatility for r in self.results if not np.isnan(r.volatility)]
        sharpe_values = [r.sharpe_ratio for r in self.results if not np.isnan(r.sharpe_ratio)]
        print(f"  Average VaR (95%):     {np.mean(var_values) if var_values else 0:.2f}%")
        print(f"  Average CVaR (95%):    {np.mean(cvar_values) if cvar_values else 0:.2f}%")
        print(f"  Average Volatility:    {np.mean(vol_values) if vol_values else 0:.1f}%")
        print(f"  Average Sharpe Ratio:  {np.mean(sharpe_values) if sharpe_values else 0:.2f}")
        
        # Valuation Metrics
        print(f"\nVALUATION METRICS SUMMARY")
        print("-"*60)
        print(f"  Average DCF Upside:    {np.mean([r.dcf_upside for r in self.results]):+.1f}%")
        print(f"  Median DCF Upside:     {np.median([r.dcf_upside for r in self.results]):+.1f}%")
        print(f"  Undervalued (>10%):    {sum(1 for r in self.results if r.dcf_upside > 10)}")
        print(f"  Overvalued (<-10%):    {sum(1 for r in self.results if r.dcf_upside < -10)}")
        
        # Top 15 Picks
        print(f"\nTOP 15 PICKS")
        print("-"*120)
        print(f"{'Ticker':<10} {'Name':<25} {'Segment':<18} {'Score':>7} {'Rec':<12} {'Rating':>7} {'DCF%':>8} {'VaR%':>7} {'Sharpe':>7}")
        print("-"*120)
        
        for r in self.results[:15]:
            print(f"{r.ticker:<10} {r.fund_name[:23]:<25} {r.segment[:16]:<18} {r.score:>6.1f}  "
                  f"{r.recommendation.value:<12} {r.credit_rating:>7} {r.dcf_upside:>+7.1f}% {r.var_95:>6.2f}% {r.sharpe_ratio:>7.2f}")
            print(f"           └─ {'; '.join(r.reasons[:2])}")
        
        # Segment Analysis
        print(f"\nSEGMENT ANALYSIS")
        print("-"*100)
        print(f"{'Segment':<22} {'Count':>6} {'Avg Score':>10} {'Top Pick':<10} {'Avg DY':>8} {'Avg VaR':>8}")
        print("-"*100)
        
        for segment, results in sorted(self.results_by_segment.items()):
            avg_score = np.mean([r.score for r in results])
            top_pick = results[0].ticker if results else '-'
            avg_dy = np.mean([r.yield_score for r in results])
            avg_var = np.nanmean([r.var_95 for r in results])
            avg_var_str = f"{avg_var:>7.2f}%" if not np.isnan(avg_var) else "    N/A"
            print(f"{segment[:20]:<22} {len(results):>6} {avg_score:>9.1f}  {top_pick:<10} {avg_dy:>7.0f}% {avg_var_str}")
        
        # Bottom 10
        print(f"\n[!] BOTTOM 10 (WEAKEST)")
        print("-"*100)
        for r in self.results[-10:]:
            print(f"  {r.ticker:<10} {r.fund_name[:25]:<25} Score: {r.score:.1f}  {r.recommendation.value}")
        
        print("\n" + "="*120)
        print("METHODOLOGY:")
        print("-"*120)
        print("  • VaR (Value at Risk): Parametric method at 95% & 99% confidence")
        print("  • CVaR (Expected Shortfall): Average loss beyond VaR threshold")
        print("  • DCF: Multi-stage model with CAPM-based cost of equity")
        print("  • Credit Score: Size, Liquidity, P/VPA Quality, Dividend Stability, Momentum")
        print("  • Monte Carlo: 10,000 simulations using GBM for 1-year price projection")
        print("  • Composite Score: Weighted average of P/VPA, Yield, Momentum, Liquidity, Credit, DCF, VaR")
        print("="*120 + "\n")


def main():
    """Main entry point"""
    import sys
    
    # Parse arguments
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
        print("  ADVANCED FII ANALYZER")
        print("  With VaR, DCF, Credit Analysis, Monte Carlo & Visualization")
        print("="*70)
        print("\nUsage: python fii_analyzer_advanced.py <xlsx_file> [--charts] [--charts=<output_path>]")
        file_path = input("\nEnter the path to the XLSX file: ").strip()
        
        if not file_path:
            file_path = r"C:\Users\DiegoCamargo\Downloads\Cont\SAMPLE.xlsx"
            print(f"Using default: {file_path}")
    
    try:
        analyzer = AdvancedFIIAnalyzer(file_path)
        analyzer.load_data()
        analyzer.analyze()
        analyzer.print_summary()
        
        # Handle charts
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
