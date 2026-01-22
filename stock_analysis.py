"""
Institutional-Grade Stock Portfolio Analysis
Advanced Analytics Used by Top Investment Banks (JPMorgan, Goldman Sachs, Morgan Stanley)
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set professional style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ffbb00',
    'info': '#17becf',
    'dark': '#2c3e50',
    'gradient': ['#d73027', '#fc8d59', '#fee090', '#e0f3f8', '#91bfdb', '#4575b4']
}

RISK_FREE_RATE = 13.75  # SELIC rate

# ============================================================================
# LOAD DATA
# ============================================================================
def load_data():
    """Load and prepare the stock data from StatusInvest export"""
    df = pd.read_excel(r'C:\Users\shodan\Desktop\statusinvest-stocks.xlsx')
    
    # StatusInvest column mapping - comprehensive mapping of all 30 columns
    COLUMN_MAP = {
        'TICKER': 'Ticker',
        'PRECO': 'Price',
        'DY': 'Dividend_Yield',
        'P/L': 'PE_Ratio',
        'P/VP': 'PB_Ratio',
        'P/ATIVOS': 'P_Assets',
        'MARGEM BRUTA': 'Gross_Margin',
        'MARGEM EBIT': 'EBIT_Margin',
        'MARG. LIQUIDA': 'Net_Margin',
        'P/EBIT': 'P_EBIT',
        'EV/EBIT': 'EV_EBIT',
        'DIVIDA LIQUIDA / EBIT': 'Net_Debt_EBIT',
        'DIV. LIQ. / PATRI.': 'Net_Debt_Equity',
        'PSR': 'PS_Ratio',
        'P/CAP. GIRO': 'P_Working_Capital',
        'P. AT CIR. LIQ.': 'P_Net_Current_Assets',
        'LIQ. CORRENTE': 'Current_Ratio',
        'ROE': 'ROE',
        'ROA': 'ROA',
        'ROIC': 'ROIC',
        'PATRIMONIO / ATIVOS': 'Equity_Assets',
        'PASSIVOS / ATIVOS': 'Liabilities_Assets',
        'GIRO ATIVOS': 'Asset_Turnover',
        'CAGR RECEITAS 5 ANOS': 'Revenue_CAGR_5Y',
        'CAGR LUCROS 5 ANOS': 'Earnings_CAGR_5Y',
        ' LIQUIDEZ MEDIA DIARIA': 'Avg_Daily_Liquidity',
        ' VPA': 'Book_Value_Per_Share',
        ' LPA': 'EPS',
        ' PEG Ratio': 'PEG_Ratio',
        ' VALOR DE MERCADO': 'Market_Cap'
    }
    
    df = df.rename(columns=COLUMN_MAP)
    
    # =========================================================================
    # DERIVED METRICS FROM STATUSINVEST DATA
    # =========================================================================
    
    # Net Debt/EBITDA estimated from Net Debt/EBIT (EBITDA ~ EBIT x 1.25)
    df['Net_Debt_EBITDA'] = df['Net_Debt_EBIT'] * 0.8
    
    # EBITDA Margin estimated from EBIT Margin (add back ~20% for D&A)
    df['EBITDA_Margin'] = df['EBIT_Margin'] * 1.2
    
    # =========================================================================
    # BETA ESTIMATION using Hamada Equation
    # Beta = Unlevered Beta x [1 + (1-Tax) x (Debt/Equity)]
    # =========================================================================
    base_beta = 0.85  # Average unlevered beta for Brazilian stocks
    tax_rate = 0.34   # Brazilian corporate tax rate
    
    df['Beta'] = base_beta * (1 + (1 - tax_rate) * df['Net_Debt_Equity'].clip(-0.5, 3))
    df['Beta'] = df['Beta'].clip(0.3, 2.5)
    
    # Adjust beta for growth companies (higher growth = higher beta)
    growth_adjustment = df['Earnings_CAGR_5Y'].clip(-20, 50) / 100
    df['Beta'] = df['Beta'] + growth_adjustment * 0.3
    df['Beta'] = df['Beta'].clip(0.3, 2.5)
    
    # =========================================================================
    # IMPLIED VOLATILITY ESTIMATION
    # Using relationship: Vol = Beta x Market_Vol x Idiosyncratic_Factor
    # =========================================================================
    market_volatility = 25.0  # Ibovespa average volatility
    
    # Base volatility from beta
    df['Implied_Volatility'] = df['Beta'] * market_volatility
    
    # Adjust for leverage (higher debt = higher vol)
    leverage_adjustment = df['Net_Debt_Equity'].clip(-0.5, 2) * 8
    df['Implied_Volatility'] = df['Implied_Volatility'] + leverage_adjustment
    
    # Adjust for size (smaller companies = higher vol)
    size_factor = 1 - df['Market_Cap'].rank(pct=True) * 0.3
    df['Implied_Volatility'] = df['Implied_Volatility'] * (1 + size_factor * 0.5)
    
    # Adjust for profitability (unprofitable = higher vol)
    df.loc[df['Net_Margin'] < 0, 'Implied_Volatility'] *= 1.3
    df.loc[df['ROE'] < 0, 'Implied_Volatility'] *= 1.2
    
    df['Implied_Volatility'] = df['Implied_Volatility'].clip(15, 90)
    
    # =========================================================================
    # CORRELATION ESTIMATION
    # Based on sector characteristics and market cap
    # =========================================================================
    df['Correlation'] = 0.50  # Base correlation
    # Large caps have higher correlation with market
    df.loc[df['Market_Cap'] > df['Market_Cap'].quantile(0.75), 'Correlation'] = 0.65
    df.loc[df['Market_Cap'] < df['Market_Cap'].quantile(0.25), 'Correlation'] = 0.35
    
    # =========================================================================
    # ADDITIONAL DERIVED METRICS
    # =========================================================================
    
    # Volume from daily liquidity
    df['Volume'] = df['Avg_Daily_Liquidity']
    
    # Shares Outstanding
    df['Shares_Outstanding'] = df['Market_Cap'] / df['Price']
    
    # Earnings Yield (inverse of P/E)
    df['Earnings_Yield_%'] = (1 / df['PE_Ratio'].replace(0, np.nan)) * 100
    
    # Enterprise Value / EBITDA (more accurate than P/E for leveraged companies)
    df['EV_EBITDA'] = df['EV_EBIT'] * 0.8
    
    # Free Cash Flow Yield estimate (Net Margin - Capex proxy)
    df['FCF_Yield_%'] = df['Earnings_Yield_%'] * (1 - 0.3)  # Assume 30% reinvestment
    
    # Sustainable Growth Rate (SGR = ROE x Retention Ratio)
    payout_ratio = df['Dividend_Yield'] / df['Earnings_Yield_%'].replace(0, np.nan)
    df['Retention_Ratio'] = (1 - payout_ratio.clip(0, 1)).fillna(0.7)
    df['Sustainable_Growth'] = df['ROE'] * df['Retention_Ratio']
    
    # DuPont Analysis Components
    df['Profit_Margin'] = df['Net_Margin']
    df['Asset_Efficiency'] = df['Asset_Turnover']
    df['Financial_Leverage'] = 1 / df['Equity_Assets'].replace(0, np.nan)
    
    # Altman Z-Score Components (adapted for available data)
    # Z = 1.2*WC/TA + 1.4*RE/TA + 3.3*EBIT/TA + 0.6*MVE/TL + 1.0*S/TA
    df['Working_Capital_Ratio'] = (df['Current_Ratio'] - 1) / df['Current_Ratio']
    df['Retained_Earnings_Ratio'] = df['Equity_Assets'] * 0.8  # Proxy
    df['EBIT_Assets'] = df['EBIT_Margin'] * df['Asset_Turnover'] / 100
    df['Market_Equity_Liabilities'] = df['PB_Ratio'] * df['Equity_Assets'] / df['Liabilities_Assets'].replace(0, np.nan)
    
    # Quality of Earnings (Accrual Ratio proxy)
    df['Earnings_Quality'] = df['Net_Margin'] / df['Gross_Margin'].replace(0, np.nan)
    
    return df


# ============================================================================
# CALCULATION FUNCTIONS WITH DETAILED EXPLANATIONS
# ============================================================================

def calculate_graham_number(df, show_details=True):
    """
    GRAHAM NUMBER - Benjamin Graham's Intrinsic Value Formula
    
    Used by: Value investors, Warren Buffett, Berkshire Hathaway
    
    Formula: Graham Number = sqrt(22.5 x EPS x Book Value Per Share)
    
    Where:
    - 22.5 = Graham's constant (derived from P/E of 15 x P/B of 1.5)
    - EPS = Earnings Per Share (company's profit divided by shares outstanding)
    - Book Value Per Share = (Total Assets - Total Liabilities) / Shares Outstanding
    
    Interpretation:
    - If Current Price < Graham Number: Stock is UNDERVALUED
    - If Current Price > Graham Number: Stock is OVERVALUED
    - Upside % = ((Graham Number / Price) - 1) x 100
    """
    df['Price'] = df['Market_Cap'] / df['Shares_Outstanding']
    df['Graham_Number'] = np.sqrt(22.5 * df['EPS'].clip(lower=0) * df['Book_Value_Per_Share'].clip(lower=0))
    df['Graham_Upside_%'] = ((df['Graham_Number'] / df['Price']) - 1) * 100
    
    if show_details:
        print("\n" + "=" * 80)
        print("GRAHAM NUMBER CALCULATION")
        print("=" * 80)
        print("""
FORMULA: Graham Number = sqrt(22.5 x EPS x Book Value Per Share)

ORIGIN: Benjamin Graham, "The Intelligent Investor" (1949)
        Known as the "father of value investing"

DERIVATION OF 22.5:
    - Graham considered a fair P/E ratio to be 15
    - Graham considered a fair P/B ratio to be 1.5
    - 15 x 1.5 = 22.5

CALCULATION STEPS:
    1. Get EPS (Earnings Per Share) from financial statements
    2. Get Book Value Per Share = (Total Equity) / Shares Outstanding
    3. Multiply: 22.5 x EPS x BVPS
    4. Take square root of result
    5. Compare to current market price

EXAMPLE CALCULATION:
    Stock: RECV3
    - EPS = R$ 2.04
    - Book Value Per Share = R$ 15.14
    - Graham Number = sqrt(22.5 x 2.04 x 15.14)
    - Graham Number = sqrt(693.22) = R$ 26.33
    - Current Price = R$ 10.60
    - Upside = ((26.33 / 10.60) - 1) x 100 = 148.4%
    - INTERPRETATION: Stock trades at 60% discount to intrinsic value
        """)
        
        print("\nTOP 10 STOCKS BY GRAHAM UPSIDE:")
        print("-" * 70)
        result = df[['Ticker', 'Price', 'EPS', 'Book_Value_Per_Share', 'Graham_Number', 'Graham_Upside_%']].copy()
        result = result[result['Graham_Upside_%'] > 0].nlargest(10, 'Graham_Upside_%')
        print(result.to_string(index=False))
    
    return df


def calculate_earnings_yield(df, show_details=True):
    """
    EARNINGS YIELD - Inverse of P/E Ratio
    
    Used by: Warren Buffett, Joel Greenblatt (Magic Formula)
    
    Formula: Earnings Yield = (1 / P/E Ratio) x 100 = (EPS / Price) x 100
    
    Interpretation:
    - Higher is better (more earnings per dollar invested)
    - Compare to bond yields to assess attractiveness
    - If Earnings Yield > Bond Yield: Stocks may be attractive
    """
    df['Earnings_Yield_%'] = (1 / df['PE_Ratio'].replace(0, np.nan)) * 100
    
    if show_details:
        print("\n" + "=" * 80)
        print("EARNINGS YIELD CALCULATION")
        print("=" * 80)
        print("""
FORMULA: Earnings Yield = (1 / P/E) x 100 = (EPS / Price) x 100

ORIGIN: Concept popularized by Warren Buffett and Joel Greenblatt

RATIONALE:
    - P/E tells you how much you pay for each $1 of earnings
    - Earnings Yield tells you how much you EARN for each $1 invested
    - Allows comparison with bond yields (fixed income)

CALCULATION STEPS:
    1. Get P/E Ratio (Price / EPS)
    2. Calculate inverse: 1 / P/E
    3. Multiply by 100 to get percentage

EXAMPLE CALCULATION:
    Stock: RECV3
    - P/E Ratio = 5.19
    - Earnings Yield = (1 / 5.19) x 100 = 19.27%
    
    Stock: WEGE3
    - P/E Ratio = 35.42
    - Earnings Yield = (1 / 35.42) x 100 = 2.82%
    
COMPARISON WITH SELIC (Risk-Free Rate = 13.75%):
    - RECV3 Earnings Yield (19.27%) > SELIC (13.75%) = ATTRACTIVE
    - WEGE3 Earnings Yield (2.82%) < SELIC (13.75%) = Must justify with growth

INTERPRETATION:
    - > 15%: Excellent (deep value)
    - 10-15%: Good
    - 5-10%: Fair (needs growth story)
    - < 5%: Expensive (requires high growth)
        """)
        
        print(f"\nCurrent SELIC (Risk-Free Rate): {RISK_FREE_RATE}%")
        print("\nTOP 10 STOCKS BY EARNINGS YIELD:")
        print("-" * 70)
        result = df[['Ticker', 'PE_Ratio', 'Earnings_Yield_%']].copy()
        result = result.nlargest(10, 'Earnings_Yield_%')
        result['Spread_vs_SELIC'] = result['Earnings_Yield_%'] - RISK_FREE_RATE
        print(result.to_string(index=False))
    
    return df


def calculate_peg_ratio(df, show_details=True):
    """
    PEG RATIO - Price/Earnings to Growth
    
    Used by: Peter Lynch (Fidelity Magellan Fund)
    
    Formula: PEG = P/E Ratio / Earnings Growth Rate
    
    StatusInvest provides the actual PEG Ratio calculated with real CAGR data
    """
    # Use the actual PEG from StatusInvest, but also calculate our own for comparison
    df['PEG_Ratio_CAGR'] = df['PE_Ratio'] / df['Earnings_CAGR_5Y'].replace(0, np.nan)
    # PEG_Ratio is already loaded from StatusInvest data
    
    if show_details:
        print("\n" + "=" * 80)
        print("PEG RATIO CALCULATION")
        print("=" * 80)
        print("""
FORMULA: PEG Ratio = P/E Ratio / Expected Earnings Growth Rate

ORIGIN: Peter Lynch, "One Up on Wall Street" (1989)
        Managed Fidelity Magellan Fund with 29.2% annual return

DATA SOURCE: StatusInvest provides pre-calculated PEG Ratio
             We also calculate using 5-Year Earnings CAGR

CALCULATION METHODS:
    1. StatusInvest PEG: P/E / Forward Growth Estimate
    2. Our CAGR PEG: P/E / 5-Year Earnings CAGR

EXAMPLE CALCULATION:
    Stock: ALLD3
    - P/E Ratio = 2.24
    - Earnings CAGR 5Y = 28.55%
    - PEG (CAGR) = 2.24 / 28.55 = 0.08
    - StatusInvest PEG = 0.02

INTERPRETATION (Peter Lynch's Guidelines):
    - PEG < 0.5: EXTREMELY UNDERVALUED
    - PEG 0.5 - 1.0: UNDERVALUED  
    - PEG = 1.0: FAIRLY VALUED
    - PEG 1.0 - 2.0: OVERVALUED
    - PEG > 2.0: EXTREMELY OVERVALUED

NOTE: Negative CAGR or negative PEG should be ignored as they indicate
      declining earnings which requires different analysis.
        """)
        
        print("\nTOP 15 STOCKS BY PEG RATIO (StatusInvest data):")
        print("-" * 90)
        result = df[['Ticker', 'PE_Ratio', 'Earnings_CAGR_5Y', 'PEG_Ratio', 'PEG_Ratio_CAGR']].copy()
        result = result[(result['PEG_Ratio'] > 0) & (result['PEG_Ratio'] < 5)]
        result = result.nsmallest(15, 'PEG_Ratio')
        print(result.round(2).to_string(index=False))
        
        print("\n\nGROWTH AT REASONABLE PRICE (GARP) - Best PEG with High Growth:")
        print("-" * 90)
        garp = df[(df['PEG_Ratio'] > 0) & (df['PEG_Ratio'] < 1.5) & (df['Earnings_CAGR_5Y'] > 10)]
        garp = garp[['Ticker', 'PE_Ratio', 'Earnings_CAGR_5Y', 'ROE', 'PEG_Ratio']].nsmallest(10, 'PEG_Ratio')
        print(garp.round(2).to_string(index=False))
    
    return df


def calculate_var(df, show_details=True):
    """
    VALUE AT RISK (VaR) - Risk Management Metric
    
    Used by: JPMorgan (invented it), Goldman Sachs, all major banks
    
    Formula: VaR(95%) = Portfolio Value x Volatility x Z-score x sqrt(Time)
    
    For daily VaR with 95% confidence: VaR = Vol x 1.645 / sqrt(252)
    """
    df['VaR_95_Daily_%'] = df['Implied_Volatility'] * 1.645 / np.sqrt(252)
    df['VaR_99_Daily_%'] = df['Implied_Volatility'] * 2.326 / np.sqrt(252)
    df['CVaR_95_%'] = df['VaR_95_Daily_%'] * 1.25
    df['Max_Drawdown_Est_%'] = df['Implied_Volatility'] * 2.5
    
    if show_details:
        print("\n" + "=" * 80)
        print("VALUE AT RISK (VaR) CALCULATION")
        print("=" * 80)
        print("""
FORMULA: VaR = Volatility x Z-score / sqrt(Trading Days)

ORIGIN: JPMorgan, 1990s - Created by the RiskMetrics team
        Now industry standard for risk management

COMPONENTS:
    - Volatility (sigma): Standard deviation of returns (using Implied Vol)
    - Z-score: Statistical confidence level
        - 95% confidence: Z = 1.645
        - 99% confidence: Z = 2.326
    - Trading Days: 252 (standard trading days per year)

CALCULATION STEPS:
    1. Get Implied Volatility (annualized)
    2. Multiply by Z-score for desired confidence
    3. Divide by sqrt(252) to convert to daily

EXAMPLE CALCULATION:
    Stock: MOVI3
    - Implied Volatility = 55.47%
    - Daily VaR (95%) = 55.47% x 1.645 / sqrt(252)
    - Daily VaR (95%) = 55.47% x 1.645 / 15.87
    - Daily VaR (95%) = 5.75%
    
    INTERPRETATION: 
    There is a 5% probability that MOVI3 will lose MORE than 5.75% 
    in a single day, or equivalently, 95% of the time daily losses 
    will be LESS than 5.75%.

EXPECTED SHORTFALL (CVaR):
    Formula: CVaR = VaR x 1.25 (approximation)
    
    CVaR answers: "If we exceed VaR, what is the AVERAGE loss?"
    This is also called "Conditional VaR" or "Expected Shortfall"

MAXIMUM DRAWDOWN ESTIMATE:
    Formula: Max DD = Volatility x 2.5 (rule of thumb)
    
    Represents the worst peak-to-trough decline expected

RISK CLASSIFICATION:
    Daily VaR 95%:
    - < 2%: LOW RISK
    - 2-4%: MODERATE RISK
    - > 4%: HIGH RISK
        """)
        
        print("\nRISK METRICS SUMMARY:")
        print("-" * 70)
        stats_df = df[['VaR_95_Daily_%', 'VaR_99_Daily_%', 'CVaR_95_%', 'Max_Drawdown_Est_%']].describe()
        print(stats_df.to_string())
        
        print("\n\nLOWEST RISK STOCKS (by VaR 95%):")
        print("-" * 70)
        result = df[df['Implied_Volatility'] > 0][['Ticker', 'Implied_Volatility', 'Beta', 'VaR_95_Daily_%', 'VaR_99_Daily_%']].copy()
        result = result.nsmallest(10, 'VaR_95_Daily_%')
        print(result.to_string(index=False))
        
        print("\n\nHIGHEST RISK STOCKS (by VaR 95%):")
        print("-" * 70)
        result = df[['Ticker', 'Implied_Volatility', 'Beta', 'VaR_95_Daily_%', 'VaR_99_Daily_%']].copy()
        result = result.nlargest(10, 'VaR_95_Daily_%')
        print(result.to_string(index=False))
    
    return df


def calculate_sharpe_ratio(df, show_details=True):
    """
    SHARPE RATIO - Risk-Adjusted Return
    
    Used by: All institutional investors, hedge funds, mutual funds
    
    Formula: Sharpe = (Return - Risk-Free Rate) / Volatility
    """
    df['Sharpe_Ratio'] = (df['Earnings_Yield_%'] - RISK_FREE_RATE) / df['Implied_Volatility'].replace(0, np.nan)
    df['Sortino_Ratio'] = (df['Earnings_Yield_%'] - RISK_FREE_RATE) / (df['Implied_Volatility'].replace(0, np.nan) * 0.7)
    df['Treynor_Ratio'] = (df['Earnings_Yield_%'] - RISK_FREE_RATE) / df['Beta'].replace(0, np.nan)
    
    if show_details:
        print("\n" + "=" * 80)
        print("SHARPE RATIO CALCULATION")
        print("=" * 80)
        print("""
FORMULA: Sharpe Ratio = (Expected Return - Risk-Free Rate) / Standard Deviation

ORIGIN: William Sharpe, Nobel Prize in Economics (1990)
        Foundation of Modern Portfolio Theory

COMPONENTS:
    - Expected Return: Using Earnings Yield as return proxy
    - Risk-Free Rate: SELIC = 13.75%
    - Standard Deviation: Implied Volatility

CALCULATION STEPS:
    1. Calculate excess return: Return - Risk-Free Rate
    2. Divide by volatility (standard deviation)

EXAMPLE CALCULATION:
    Stock: RECV3
    - Earnings Yield = 19.27%
    - Risk-Free Rate (SELIC) = 13.75%
    - Implied Volatility = 20% (example)
    - Sharpe = (19.27 - 13.75) / 20 = 0.28

RELATED METRICS:

    SORTINO RATIO (1980s, Frank Sortino):
    Formula: Sortino = (Return - Rf) / Downside Deviation
    
    Improvement over Sharpe: Only penalizes downside volatility
    We approximate: Sortino = Sharpe / 0.7

    TREYNOR RATIO (1965, Jack Treynor):
    Formula: Treynor = (Return - Rf) / Beta
    
    Uses systematic risk (Beta) instead of total risk (volatility)

INTERPRETATION:
    Sharpe Ratio:
    - > 1.0: EXCELLENT
    - 0.5 - 1.0: GOOD
    - 0 - 0.5: ACCEPTABLE
    - < 0: POOR (return below risk-free rate)
        """)
        
        print(f"\nUsing SELIC Risk-Free Rate: {RISK_FREE_RATE}%")
        print("\nTOP 10 STOCKS BY SHARPE RATIO:")
        print("-" * 70)
        result = df[np.isfinite(df['Sharpe_Ratio'])][['Ticker', 'Earnings_Yield_%', 'Implied_Volatility', 'Sharpe_Ratio', 'Sortino_Ratio', 'Treynor_Ratio']].copy()
        result = result.nlargest(10, 'Sharpe_Ratio')
        print(result.round(4).to_string(index=False))
    
    return df


def calculate_quality_score(df, show_details=True):
    """
    QUALITY SCORE - Composite Quality Factor
    
    Used by: AQR Capital, BlackRock, Vanguard Factor Funds
    
    Components: Profitability, Safety, Earnings Quality, Growth
    """
    # Profitability Score (using all available margin and return metrics)
    df['Profitability_Score'] = (
        df['ROE'].rank(pct=True) * 0.20 +
        df['ROA'].rank(pct=True) * 0.15 +
        df['ROIC'].rank(pct=True) * 0.20 +
        df['Gross_Margin'].rank(pct=True) * 0.15 +
        df['EBIT_Margin'].rank(pct=True) * 0.15 +
        df['Net_Margin'].rank(pct=True) * 0.15
    ) * 100
    
    # Safety Score (comprehensive leverage and liquidity analysis)
    df['Safety_Score'] = (
        (1 - df['Net_Debt_EBITDA'].rank(pct=True)) * 0.25 +
        (1 - df['Net_Debt_Equity'].rank(pct=True)) * 0.20 +
        df['Current_Ratio'].rank(pct=True) * 0.20 +
        df['Equity_Assets'].rank(pct=True) * 0.20 +
        (1 - df['Liabilities_Assets'].rank(pct=True)) * 0.15
    ) * 100
    
    # Growth Score (using CAGR data)
    df['Growth_Score'] = (
        df['Revenue_CAGR_5Y'].rank(pct=True) * 0.40 +
        df['Earnings_CAGR_5Y'].rank(pct=True) * 0.60
    ) * 100
    
    # Efficiency Score (asset utilization)
    df['Efficiency_Score'] = (
        df['Asset_Turnover'].rank(pct=True) * 0.50 +
        df['ROIC'].rank(pct=True) * 0.50
    ) * 100
    
    # Overall Quality Score (balanced weighting)
    df['Quality_Score'] = (
        df['Profitability_Score'] * 0.35 +
        df['Safety_Score'] * 0.30 +
        df['Growth_Score'] * 0.20 +
        df['Efficiency_Score'] * 0.15
    )
    
    if show_details:
        print("\n" + "=" * 80)
        print("QUALITY SCORE CALCULATION")
        print("=" * 80)
        print("""
FORMULA: Quality Score = Profitability x 0.35 + Safety x 0.30 + Growth x 0.20 + Efficiency x 0.15

ORIGIN: Factor investing research by AQR Capital Management
        Cliff Asness, "Quality Minus Junk" (2013)

PROFITABILITY SCORE (35% weight):
    Components:
    - ROE (20%): Return on Equity - Net Income / Shareholders' Equity
    - ROA (15%): Return on Assets - Net Income / Total Assets  
    - ROIC (20%): Return on Invested Capital - NOPAT / Invested Capital
    - Gross Margin (15%): Gross Profit / Revenue
    - EBIT Margin (15%): Operating Income / Revenue
    - Net Margin (15%): Net Income / Revenue

SAFETY SCORE (30% weight):
    Components:
    - Net Debt/EBITDA (25%): Lower is better (inverted)
    - Net Debt/Equity (20%): Lower leverage is safer (inverted)
    - Current Ratio (20%): Current Assets / Current Liabilities
    - Equity/Assets (20%): Higher equity ratio = more stable
    - Liabilities/Assets (15%): Lower is better (inverted)

GROWTH SCORE (20% weight):
    Components:
    - Revenue CAGR 5Y (40%): 5-year compound revenue growth
    - Earnings CAGR 5Y (60%): 5-year compound earnings growth

EFFICIENCY SCORE (15% weight):
    Components:
    - Asset Turnover (50%): Revenue / Total Assets
    - ROIC (50%): Capital efficiency

INTERPRETATION:
    - > 75: HIGH QUALITY (top quartile)
    - 55-75: GOOD QUALITY
    - 40-55: AVERAGE QUALITY
    - < 40: LOW QUALITY
        """)
        
        print("\nTOP 15 HIGHEST QUALITY STOCKS:")
        print("-" * 100)
        result = df[['Ticker', 'ROE', 'ROA', 'ROIC', 'Gross_Margin', 'Net_Margin', 
                     'Profitability_Score', 'Safety_Score', 'Growth_Score', 'Quality_Score']].copy()
        result = result.nlargest(15, 'Quality_Score')
        print(result.round(2).to_string(index=False))
        
        print("\n\nQUALITY SCORE BREAKDOWN - TOP 10:")
        print("-" * 80)
        breakdown = df[['Ticker', 'Profitability_Score', 'Safety_Score', 'Growth_Score', 
                        'Efficiency_Score', 'Quality_Score']].nlargest(10, 'Quality_Score')
        print(breakdown.round(2).to_string(index=False))
    
    return df


def calculate_piotroski_score(df, show_details=True):
    """
    PIOTROSKI F-SCORE - Financial Health Score
    
    Used by: Academic research, value investors
    
    Formula: Sum of 9 binary signals (0 or 1)
    """
    # PROFITABILITY (4 signals)
    df['F_ROA'] = (df['ROA'] > 0).astype(int)  # Using actual ROA
    df['F_CFO'] = (df['Net_Margin'] > 0).astype(int)  # Positive cash flow proxy
    df['F_Delta_ROA'] = (df['Earnings_CAGR_5Y'] > 0).astype(int)  # Using CAGR as growth proxy
    df['F_Accrual'] = (df['Net_Margin'] / df['Gross_Margin'].replace(0, np.nan) > 0.2).astype(int)  # Quality of earnings
    
    # LEVERAGE/LIQUIDITY (3 signals)
    df['F_Leverage'] = (df['Net_Debt_Equity'] < df['Net_Debt_Equity'].median()).astype(int)  # Lower than median leverage
    df['F_Liquidity'] = (df['Current_Ratio'] > 1).astype(int)  # Can pay short-term debts
    df['F_Equity'] = (df['Equity_Assets'] > df['Equity_Assets'].median()).astype(int)  # Strong equity position
    
    # OPERATING EFFICIENCY (2 signals)
    df['F_Margin'] = (df['Gross_Margin'] > df['Gross_Margin'].median()).astype(int)  # Above median margin
    df['F_Turnover'] = (df['Asset_Turnover'] > df['Asset_Turnover'].median()).astype(int)  # Above median turnover
    
    df['Piotroski_Score'] = (
        df['F_ROA'] + df['F_CFO'] + df['F_Delta_ROA'] +
        df['F_Accrual'] + df['F_Leverage'] + df['F_Liquidity'] +
        df['F_Equity'] + df['F_Margin'] + df['F_Turnover']
    )
    
    if show_details:
        print("\n" + "=" * 80)
        print("PIOTROSKI F-SCORE CALCULATION")
        print("=" * 80)
        print("""
FORMULA: F-Score = Sum of 9 binary signals (each 0 or 1, max = 9)

ORIGIN: Joseph Piotroski, "Value Investing: The Use of Historical 
        Financial Statement Information to Separate Winners from Losers" (2000)
        University of Chicago research

THE 9 SIGNALS (using StatusInvest data):

PROFITABILITY (4 points):
    1. ROA > 0 (positive return on assets)
       Data used: ROA from StatusInvest
    
    2. Operating Cash Flow > 0 (positive cash generation)
       Proxy: Net Margin > 0
    
    3. ROA/Earnings Improving
       Data used: Earnings CAGR 5Y > 0
    
    4. Quality of Earnings (Cash > Accruals)
       Proxy: Net Margin / Gross Margin > 0.2

LEVERAGE/LIQUIDITY (3 points):
    5. Leverage below median (improving debt position)
       Data used: Net Debt/Equity < Median
    
    6. Current Ratio > 1 (can pay short-term debts)
       Data used: LIQ. CORRENTE from StatusInvest
    
    7. Strong Equity Position
       Data used: Equity/Assets > Median

OPERATING EFFICIENCY (2 points):
    8. Gross Margin above median
       Data used: MARGEM BRUTA from StatusInvest
    
    9. Asset Turnover above median
       Data used: GIRO ATIVOS from StatusInvest

INTERPRETATION:
    8-9: STRONG (high probability of outperformance)
    5-7: NEUTRAL
    0-4: WEAK (high probability of underperformance)

RESEARCH FINDINGS:
    Piotroski found that high F-Score stocks outperformed low F-Score 
    stocks by 7.5% annually (1976-1996)
    8-9: STRONG (high probability of outperformance)
    5-7: NEUTRAL
    0-4: WEAK (high probability of underperformance)

RESEARCH FINDINGS:
    Piotroski found that high F-Score stocks outperformed low F-Score 
    stocks by 7.5% annually (1976-1996)
        """)
        
        print("\nPIOTROSKI SCORE DISTRIBUTION:")
        print("-" * 70)
        print(df['Piotroski_Score'].value_counts().sort_index())
        
        print("\n\nSTOCKS WITH HIGHEST F-SCORE (8-9):")
        print("-" * 70)
        result = df[df['Piotroski_Score'] >= 8][['Ticker', 'ROE', 'Net_Margin', 'Net_Debt_EBITDA', 'Current_Ratio', 'Piotroski_Score']].copy()
        print(result.to_string(index=False))
    
    return df


def calculate_magic_formula(df, show_details=True):
    """
    MAGIC FORMULA - Joel Greenblatt's Ranking System
    
    Used by: Gotham Capital (Greenblatt's hedge fund)
    
    Formula: Combined rank of Earnings Yield + ROIC
    """
    df['Earnings_Yield_Rank'] = df['Earnings_Yield_%'].rank(ascending=False)
    df['ROIC_Rank'] = df['ROIC'].rank(ascending=False)
    df['Magic_Formula_Rank'] = df['Earnings_Yield_Rank'] + df['ROIC_Rank']
    
    if show_details:
        print("\n" + "=" * 80)
        print("MAGIC FORMULA CALCULATION")
        print("=" * 80)
        print("""
FORMULA: Magic Formula Rank = Earnings Yield Rank + ROIC Rank

ORIGIN: Joel Greenblatt, "The Little Book That Beats the Market" (2005)
        Gotham Capital achieved 40%+ annual returns (1985-2005)

CONCEPT:
    Find companies that are both:
    1. CHEAP (high earnings yield)
    2. GOOD (high return on capital)
    
    "Buy good companies at bargain prices"

CALCULATION STEPS:
    1. Rank all stocks by Earnings Yield (highest = rank 1)
    2. Rank all stocks by ROIC (highest = rank 1)
    3. Add both ranks together
    4. Lowest combined rank = best stock

EXAMPLE CALCULATION:
    Stock: DEXP3
    - Earnings Yield = 17.42% (Rank: 3rd out of 63)
    - ROIC = 14.90% (Rank: 10th out of 63)
    - Magic Formula Rank = 3 + 10 = 13
    
    Stock: ODPV3
    - Earnings Yield = 9.47% (Rank: 15th out of 63)
    - ROIC = 63.90% (Rank: 1st out of 63)
    - Magic Formula Rank = 15 + 1 = 16

GREENBLATT'S STRATEGY:
    1. Rank stocks by Magic Formula
    2. Buy top 20-30 stocks
    3. Hold for 1 year
    4. Rebalance annually
    
    Backtested returns: 30.8% annually (1988-2004) vs 12.4% S&P 500

WHY IT WORKS:
    - Combines VALUE (cheap stocks) with QUALITY (profitable companies)
    - Avoids "value traps" (cheap but unprofitable)
    - Simple, systematic, removes emotion
        """)
        
        print("\nTOP 15 MAGIC FORMULA STOCKS (Lowest Rank = Best):")
        print("-" * 70)
        result = df[['Ticker', 'Earnings_Yield_%', 'ROIC', 'Earnings_Yield_Rank', 'ROIC_Rank', 'Magic_Formula_Rank']].copy()
        result = result.nsmallest(15, 'Magic_Formula_Rank')
        print(result.round(2).to_string(index=False))
    
    return df


def calculate_value_score(df, show_details=True):
    """
    VALUE SCORE - Composite Value Factor
    
    Used by: Fama-French research, dimensional funds, value ETFs
    
    Components: Multiple valuation ratios from StatusInvest
    """
    # Comprehensive Value Score using all available valuation metrics
    df['Value_Score'] = (
        (1 - df['PE_Ratio'].rank(pct=True)) * 0.15 +          # P/E - lower is cheaper
        (1 - df['PB_Ratio'].rank(pct=True)) * 0.15 +          # P/B - lower is cheaper
        (1 - df['PS_Ratio'].rank(pct=True)) * 0.10 +          # P/Sales - lower is cheaper
        (1 - df['P_EBIT'].rank(pct=True)) * 0.15 +            # P/EBIT - lower is cheaper
        (1 - df['EV_EBIT'].rank(pct=True)) * 0.15 +           # EV/EBIT - lower is cheaper
        (1 - df['P_Assets'].rank(pct=True)) * 0.10 +          # P/Assets - lower is cheaper
        df['Dividend_Yield'].rank(pct=True) * 0.10 +          # Dividend Yield - higher is better
        df['Earnings_Yield_%'].rank(pct=True) * 0.10          # Earnings Yield - higher is better
    ) * 100
    
    # Also create an EV-based value score (better for comparing leveraged companies)
    df['EV_Value_Score'] = (
        (1 - df['EV_EBIT'].rank(pct=True)) * 0.50 +
        (1 - df['PS_Ratio'].rank(pct=True)) * 0.25 +
        df['ROIC'].rank(pct=True) * 0.25
    ) * 100
    
    if show_details:
        print("\n" + "=" * 80)
        print("VALUE SCORE CALCULATION")
        print("=" * 80)
        print("""
FORMULA: Value Score = Weighted average of valuation metric percentile ranks

ORIGIN: Fama-French Three-Factor Model (1992)
        Eugene Fama and Kenneth French, Nobel Prize research

COMPONENTS (using StatusInvest data):
    1. P/E Ratio (15%):     Price / Earnings - lower is cheaper
    2. P/B Ratio (15%):     Price / Book Value - lower is cheaper  
    3. P/S Ratio (10%):     Price / Sales (PSR) - lower is cheaper
    4. P/EBIT (15%):        Price / Operating Income - lower is cheaper
    5. EV/EBIT (15%):       Enterprise Value / EBIT - lower is cheaper
    6. P/Assets (10%):      Price / Total Assets - lower is cheaper
    7. Dividend Yield (10%): Higher yield = better value
    8. Earnings Yield (10%): Higher yield = better value

EV-BASED VALUE SCORE (for leveraged companies):
    - EV/EBIT (50%): Better than P/E for comparing companies with different debt
    - P/Sales (25%): Revenue-based valuation
    - ROIC (25%): Return on invested capital

WHY EV/EBIT IS SUPERIOR:
    - Accounts for debt (EV = Market Cap + Debt - Cash)
    - Compares operating performance regardless of capital structure
    - Preferred by private equity and M&A professionals

INTERPRETATION:
    - > 75: DEEP VALUE
    - 55-75: VALUE
    - 40-55: FAIR VALUE
    - < 40: GROWTH/EXPENSIVE
        """)
        
        print("\nTOP 15 VALUE STOCKS (Composite Score):")
        print("-" * 100)
        result = df[['Ticker', 'PE_Ratio', 'PB_Ratio', 'EV_EBIT', 'Dividend_Yield', 'Earnings_Yield_%', 'Value_Score']].copy()
        result = result.nlargest(15, 'Value_Score')
        print(result.round(2).to_string(index=False))
        
        print("\n\nTOP 15 VALUE STOCKS (EV-Based Score - Better for Leveraged Companies):")
        print("-" * 80)
        ev_result = df[['Ticker', 'EV_EBIT', 'PS_Ratio', 'ROIC', 'Net_Debt_Equity', 'EV_Value_Score']].copy()
        ev_result = ev_result.nlargest(15, 'EV_Value_Score')
        print(ev_result.round(2).to_string(index=False))
    
    return df


def calculate_dividend_metrics(df, show_details=True):
    """
    DIVIDEND METRICS - Income Investing Analysis
    
    Used by: Income investors, pension funds, retirees
    
    Includes: Payout ratio, Gordon Model, Dividend Quality
    """
    df['Payout_Ratio_Est'] = df['Dividend_Yield'] / df['Earnings_Yield_%'].replace(0, np.nan) * 100
    df['Dividend_Safety'] = 100 - df['Payout_Ratio_Est'].clip(upper=100)
    
    # Use actual CAGR data for growth estimation
    df['Implied_Growth'] = df['Earnings_CAGR_5Y'].clip(lower=0)
    df['Sustainable_Growth_Rate'] = df['ROE'] * (1 - df['Payout_Ratio_Est'].clip(0, 100) / 100)
    
    # Gordon Model with actual growth data
    df['Gordon_Expected_Return_%'] = df['Dividend_Yield'] + df['Implied_Growth']
    
    # Enhanced Dividend Quality Score
    df['Dividend_Quality'] = (
        df['Dividend_Yield'].rank(pct=True) * 0.25 +
        (1 - df['Payout_Ratio_Est'].rank(pct=True)) * 0.20 +
        df['ROE'].rank(pct=True) * 0.15 +
        df['Earnings_CAGR_5Y'].rank(pct=True) * 0.15 +
        (1 - df['Net_Debt_Equity'].rank(pct=True)) * 0.15 +
        df['Current_Ratio'].rank(pct=True) * 0.10
    ) * 100
    
    if show_details:
        print("\n" + "=" * 80)
        print("DIVIDEND METRICS CALCULATION")
        print("=" * 80)
        print("""
FORMULAS:

1. PAYOUT RATIO:
   Formula: Payout Ratio = Dividends / Net Income = Div Yield / Earnings Yield
   
   Interpretation:
   - < 50%: SAFE (room for dividend growth)
   - 50-75%: MODERATE (sustainable but limited growth)
   - > 75%: RISKY (may not be sustainable)
   - > 100%: UNSUSTAINABLE (paying more than earned)

2. GORDON GROWTH MODEL (Dividend Discount Model):
   Formula: Expected Return = Dividend Yield + Growth Rate
   
   Origin: Myron Gordon, 1962
   Also known as: Gordon-Shapiro Model
   
   Sustainable Growth Rate = ROE x (1 - Payout Ratio)
   
   Example:
   - Dividend Yield = 8%
   - ROE = 20%
   - Payout Ratio = 40%
   - Sustainable Growth = 20% x (1 - 0.40) = 12%
   - Expected Return = 8% + 12% = 20%

3. DIVIDEND QUALITY SCORE:
   Components:
   - Dividend Yield (30%): Higher is better
   - Payout Ratio (30%): Lower is better (inverted)
   - ROE (20%): Higher means better dividend coverage
   - Leverage (20%): Lower debt = safer dividends

CALCULATION EXAMPLE:
    Stock: RANI3
    - Dividend Yield = 7.94%
    - Earnings Yield = 18.35%
    - Payout Ratio = 7.94 / 18.35 x 100 = 43.27%
    - ROE = 27.22%
    - Growth = 27.22% x (1 - 0.4327) = 15.44%
    - Gordon Return = 7.94% + 15.44% = 23.38%
        """)
        
        print("\nTOP 10 DIVIDEND QUALITY STOCKS:")
        print("-" * 70)
        result = df[['Ticker', 'Dividend_Yield', 'Payout_Ratio_Est', 'ROE', 'Gordon_Expected_Return_%', 'Dividend_Quality']].copy()
        result = result.nlargest(10, 'Dividend_Quality')
        print(result.round(2).to_string(index=False))
        
        print("\n\nHIGHEST GORDON EXPECTED RETURNS:")
        print("-" * 70)
        result = df[['Ticker', 'Dividend_Yield', 'Implied_Growth', 'Gordon_Expected_Return_%']].copy()
        result = result.nlargest(10, 'Gordon_Expected_Return_%')
        print(result.round(2).to_string(index=False))
    
    return df


def calculate_liquidity_metrics(df, show_details=True):
    """
    LIQUIDITY METRICS - Trading Analysis
    
    Used by: Institutional traders, market makers
    
    Includes: Turnover, Amihud illiquidity, Liquidity score
    """
    df['Turnover_Ratio'] = df['Volume'] / df['Market_Cap']
    df['Amihud_Illiquidity'] = df['Implied_Volatility'] / (df['Volume'] / 1e9)
    df['Liquidity_Score'] = (
        df['Volume'].rank(pct=True) * 0.5 +
        df['Market_Cap'].rank(pct=True) * 0.5
    ) * 100
    
    if show_details:
        print("\n" + "=" * 80)
        print("LIQUIDITY METRICS CALCULATION")
        print("=" * 80)
        print("""
FORMULAS:

1. TURNOVER RATIO:
   Formula: Turnover = Daily Volume / Market Cap
   
   Interpretation:
   - Higher turnover = more actively traded
   - Typical values: 0.001 to 0.05 (0.1% to 5%)

2. AMIHUD ILLIQUIDITY RATIO:
   Formula: Illiquidity = |Return| / Volume (in R$)
   Our proxy: Volatility / (Volume in Billions)
   
   Origin: Yakov Amihud, "Illiquidity and Stock Returns" (2002)
   
   Interpretation:
   - Higher value = more illiquid (harder to trade)
   - Lower value = more liquid (easier to trade)
   
   Measures price impact: How much does the price move per R$ traded?

3. LIQUIDITY SCORE:
   Components:
   - Volume rank (50%)
   - Market Cap rank (50%)
   
   Higher score = more liquid

WHY LIQUIDITY MATTERS:
    - Large orders can move illiquid stocks significantly
    - Bid-ask spreads are wider for illiquid stocks
    - Institutional investors require liquid stocks
    - Easier to enter/exit positions quickly
        """)
        
        print("\nTOP 10 MOST LIQUID STOCKS:")
        print("-" * 70)
        result = df[['Ticker', 'Market_Cap', 'Volume', 'Turnover_Ratio', 'Liquidity_Score']].copy()
        result['Market_Cap'] = (result['Market_Cap'] / 1e9).round(2)
        result['Volume'] = (result['Volume'] / 1e6).round(2)
        result = result.nlargest(10, 'Liquidity_Score')
        print("(Market Cap in R$ Billions, Volume in R$ Millions)")
        print(result.to_string(index=False))
    
    return df


def calculate_beta_analysis(df, show_details=True):
    """
    BETA ANALYSIS - Systematic Risk
    
    Used by: All institutional investors, CAPM model
    
    Includes: Raw beta, Adjusted beta (Bloomberg method)
    """
    df['Adj_Beta'] = (2/3 * df['Beta']) + (1/3 * 1)
    
    if show_details:
        print("\n" + "=" * 80)
        print("BETA ANALYSIS")
        print("=" * 80)
        print("""
FORMULA: Beta = Covariance(Stock, Market) / Variance(Market)

ORIGIN: Capital Asset Pricing Model (CAPM)
        William Sharpe, John Lintner, Jan Mossin (1960s)

INTERPRETATION:
    Beta = 1.0: Stock moves WITH the market (same volatility)
    Beta > 1.0: Stock is MORE volatile than market (aggressive)
    Beta < 1.0: Stock is LESS volatile than market (defensive)
    Beta < 0:   Stock moves OPPOSITE to market (rare, hedging)

EXAMPLES:
    Beta = 1.5: If market rises 10%, stock rises ~15%
                If market falls 10%, stock falls ~15%
    
    Beta = 0.5: If market rises 10%, stock rises ~5%
                If market falls 10%, stock falls ~5%

ADJUSTED BETA (Bloomberg Method):
    Formula: Adjusted Beta = (2/3 x Raw Beta) + (1/3 x 1.0)
    
    Rationale: Betas tend to regress toward 1.0 over time
    This adjustment improves forward-looking predictions
    
    Example:
    - Raw Beta = 1.5
    - Adjusted = (2/3 x 1.5) + (1/3 x 1.0) = 1.0 + 0.33 = 1.33

CLASSIFICATION:
    Beta > 1.5:  HIGH BETA (aggressive, cyclical)
    1.0 - 1.5:   MODERATE HIGH BETA
    0.5 - 1.0:   MODERATE LOW BETA (defensive)
    Beta < 0.5:  LOW BETA (very defensive, utilities)
        """)
        
        print("\nBETA DISTRIBUTION:")
        print("-" * 70)
        print(f"Mean Beta:     {df['Beta'].mean():.2f}")
        print(f"Median Beta:   {df['Beta'].median():.2f}")
        print(f"Std Dev:       {df['Beta'].std():.2f}")
        print(f"Min Beta:      {df['Beta'].min():.2f}")
        print(f"Max Beta:      {df['Beta'].max():.2f}")
        
        print("\n\nLOWEST BETA STOCKS (Defensive):")
        print("-" * 70)
        result = df[['Ticker', 'Beta', 'Adj_Beta', 'Implied_Volatility']].copy()
        result = result.nsmallest(10, 'Beta')
        print(result.round(2).to_string(index=False))
        
        print("\n\nHIGHEST BETA STOCKS (Aggressive):")
        print("-" * 70)
        result = df[['Ticker', 'Beta', 'Adj_Beta', 'Implied_Volatility']].copy()
        result = result.nlargest(10, 'Beta')
        print(result.round(2).to_string(index=False))
    
    return df


def calculate_composite_score(df, show_details=True):
    """
    COMPOSITE SCORE - Multifactor Ranking
    
    Used by: BlackRock, Vanguard, factor ETFs
    
    Combines: Quality, Value, Low Volatility, Liquidity, Dividend
    """
    df['Composite_Score'] = (
        df['Quality_Score'] * 0.25 +
        df['Value_Score'] * 0.25 +
        (100 - df['VaR_95_Daily_%'].rank(pct=True) * 100) * 0.20 +
        df['Liquidity_Score'] * 0.15 +
        df['Dividend_Quality'] * 0.15
    )
    
    df['Risk_Adj_Score'] = df['Composite_Score'] / (1 + df['Adj_Beta'])
    
    if show_details:
        print("\n" + "=" * 80)
        print("COMPOSITE SCORE CALCULATION")
        print("=" * 80)
        print("""
FORMULA: 
    Composite Score = Quality x 0.25 + Value x 0.25 + LowVol x 0.20 + 
                      Liquidity x 0.15 + DivQuality x 0.15

ORIGIN: Modern Factor Investing (BlackRock, Vanguard, State Street)
        Combines multiple academically-proven factors

FACTOR WEIGHTS:
    1. QUALITY (25%):     Profitable, safe, well-managed companies
    2. VALUE (25%):       Cheap relative to fundamentals
    3. LOW VOLATILITY (20%): Lower risk, smoother returns
    4. LIQUIDITY (15%):   Easy to trade, institutional grade
    5. DIVIDEND (15%):    Sustainable income stream

RISK-ADJUSTED SCORE:
    Formula: Risk-Adj Score = Composite Score / (1 + Adjusted Beta)
    
    Penalizes high-beta stocks for their additional risk

CALCULATION EXAMPLE:
    Stock: PETR4
    - Quality Score = 62.08
    - Value Score = 79.05
    - Low Vol Score = 80 (based on VaR rank)
    - Liquidity Score = 99.21
    - Dividend Quality = 70.16
    
    Composite = 62.08 x 0.25 + 79.05 x 0.25 + 80 x 0.20 + 
                99.21 x 0.15 + 70.16 x 0.15
    Composite = 15.52 + 19.76 + 16.00 + 14.88 + 10.52 = 76.68
    
    Risk-Adj = 76.68 / (1 + 0.55) = 49.47

INTERPRETATION:
    - > 70: EXCELLENT (top quartile)
    - 55-70: GOOD
    - 40-55: AVERAGE
    - < 40: BELOW AVERAGE
        """)
        
        print("\nTOP 20 STOCKS BY COMPOSITE SCORE:")
        print("-" * 70)
        result = df[['Ticker', 'Quality_Score', 'Value_Score', 'Liquidity_Score', 'Dividend_Quality', 'Composite_Score', 'Risk_Adj_Score']].copy()
        result = result.nlargest(20, 'Composite_Score')
        print(result.round(2).to_string(index=False))
    
    return df


def calculate_portfolio_stats(df, show_details=True):
    """
    PORTFOLIO STATISTICS - Aggregate Analysis
    
    Includes: Weighted metrics, Portfolio VaR, Diversification benefit
    """
    total_market_cap = df['Market_Cap'].sum()
    df['Portfolio_Weight_%'] = (df['Market_Cap'] / total_market_cap) * 100
    
    weighted_pe = (df['PE_Ratio'] * df['Portfolio_Weight_%']).sum() / 100
    weighted_pb = (df['PB_Ratio'] * df['Portfolio_Weight_%']).sum() / 100
    weighted_div = (df['Dividend_Yield'] * df['Portfolio_Weight_%']).sum() / 100
    weighted_beta = (df['Beta'] * df['Portfolio_Weight_%']).sum() / 100
    weighted_roe = (df['ROE'] * df['Portfolio_Weight_%']).sum() / 100
    
    avg_vol = df['Implied_Volatility'].mean()
    avg_corr = df['Correlation'].mean()
    n_stocks = len(df)
    diversification_factor = np.sqrt((1/n_stocks) + ((n_stocks-1)/n_stocks) * avg_corr)
    portfolio_var_95 = avg_vol * 1.645 / np.sqrt(252) * diversification_factor
    
    if show_details:
        print("\n" + "=" * 80)
        print("PORTFOLIO STATISTICS")
        print("=" * 80)
        print("""
PORTFOLIO RISK FORMULA:
    
    For a portfolio of N stocks with average correlation rho:
    
    Portfolio Volatility = Individual Vol x sqrt((1/N) + ((N-1)/N) x rho)
    
    Where:
    - N = number of stocks
    - rho = average pairwise correlation
    
    Diversification Benefit = 1 - sqrt((1/N) + ((N-1)/N) x rho)

WEIGHTED METRICS:
    Each metric is weighted by market cap:
    Weighted Metric = Sum(Metric_i x Weight_i)
        """)
        
        print(f"""
PORTFOLIO SUMMARY:
{'=' * 50}
Total Market Cap:        R$ {total_market_cap/1e12:.2f} Trillion
Number of Stocks:        {n_stocks}
Average Correlation:     {avg_corr:.2%}

WEIGHTED METRICS:
{'=' * 50}
P/E Ratio:               {weighted_pe:.2f}x
P/B Ratio:               {weighted_pb:.2f}x
Dividend Yield:          {weighted_div:.2f}%
Beta:                    {weighted_beta:.2f}
ROE:                     {weighted_roe:.2f}%

PORTFOLIO RISK:
{'=' * 50}
Average Volatility:      {avg_vol:.2f}%
Portfolio VaR (95%):     {portfolio_var_95:.2f}% daily
Portfolio VaR (Annual):  {portfolio_var_95 * np.sqrt(252):.2f}% yearly
Diversification Benefit: {(1 - diversification_factor) * 100:.1f}%
        """)
    
    return df


def run_all_calculations(df, show_details=False):
    """Run all calculations"""
    df = calculate_graham_number(df, show_details)
    df = calculate_earnings_yield(df, show_details)
    df = calculate_peg_ratio(df, show_details)
    df = calculate_var(df, show_details)
    df = calculate_sharpe_ratio(df, show_details)
    df = calculate_quality_score(df, show_details)
    df = calculate_piotroski_score(df, show_details)
    df = calculate_magic_formula(df, show_details)
    df = calculate_value_score(df, show_details)
    df = calculate_dividend_metrics(df, show_details)
    df = calculate_liquidity_metrics(df, show_details)
    df = calculate_beta_analysis(df, show_details)
    df = calculate_composite_score(df, show_details)
    df = calculate_portfolio_stats(df, show_details)
    return df


def display_charts(df):
    """Display all 12 charts"""
    print("\n" + "=" * 80)
    print("GENERATING CHARTS")
    print("=" * 80)
    print("\nClose each chart window to see the next one...")
    
    # Chart 1: Risk vs Return
    print("\n[1/12] Risk vs Return Analysis...")
    fig1, ax1 = plt.subplots(figsize=(14, 10))
    valid_mask = np.isfinite(df['Sharpe_Ratio']) & np.isfinite(df['Implied_Volatility'])
    df_plot = df[valid_mask].copy()
    
    scatter1 = ax1.scatter(
        df_plot['Implied_Volatility'], df_plot['Earnings_Yield_%'],
        s=df_plot['Market_Cap'] / 1e9 * 5, c=df_plot['Quality_Score'],
        cmap='RdYlGn', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    ax1.axhline(y=RISK_FREE_RATE, color='red', linestyle='--', alpha=0.7, linewidth=2, 
                label=f'SELIC ({RISK_FREE_RATE}%)')
    for _, row in df_plot.nlargest(12, 'Earnings_Yield_%').iterrows():
        ax1.annotate(row['Ticker'], (row['Implied_Volatility'], row['Earnings_Yield_%']),
                    fontsize=9, alpha=0.9, fontweight='bold')
    ax1.set_xlabel('Implied Volatility (%)', fontsize=12)
    ax1.set_ylabel('Earnings Yield (%)', fontsize=12)
    ax1.set_title('RISK vs RETURN ANALYSIS\nBubble Size = Market Cap | Color = Quality Score', fontsize=14, fontweight='bold')
    plt.colorbar(scatter1, ax=ax1, label='Quality Score', shrink=0.8)
    ax1.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Chart 2: Value-Quality Quadrant
    print("[2/12] Value vs Quality Quadrant...")
    fig2, ax2 = plt.subplots(figsize=(14, 10))
    scatter2 = ax2.scatter(
        df['Value_Score'], df['Quality_Score'],
        s=df['Market_Cap'] / 1e9 * 5, c=df['Composite_Score'],
        cmap='RdYlGn', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    med_value = df['Value_Score'].median()
    med_quality = df['Quality_Score'].median()
    ax2.axhline(y=med_quality, color='gray', linestyle='--', alpha=0.5)
    ax2.axvline(x=med_value, color='gray', linestyle='--', alpha=0.5)
    ax2.text(80, 85, 'QUALITY VALUE\nSTRONG BUY', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='green', alpha=0.3), fontweight='bold')
    ax2.text(20, 85, 'QUALITY GROWTH', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='blue', alpha=0.2))
    ax2.text(80, 25, 'DEEP VALUE', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='orange', alpha=0.3))
    ax2.text(20, 25, 'AVOID', fontsize=12, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='red', alpha=0.3))
    for _, row in df.nlargest(12, 'Composite_Score').iterrows():
        ax2.annotate(row['Ticker'], (row['Value_Score'], row['Quality_Score']),
                    fontsize=9, alpha=0.9, fontweight='bold')
    ax2.set_xlabel('Value Score', fontsize=12)
    ax2.set_ylabel('Quality Score', fontsize=12)
    ax2.set_title('VALUE-QUALITY MATRIX', fontsize=14, fontweight='bold')
    plt.colorbar(scatter2, ax=ax2, label='Composite Score', shrink=0.8)
    plt.tight_layout()
    plt.show()
    
    # Chart 3: Risk Map
    print("[3/12] Risk Map...")
    fig3, ax3 = plt.subplots(figsize=(14, 10))
    df_risk = df[df['Implied_Volatility'] > 0].copy()
    scatter3 = ax3.scatter(
        df_risk['Beta'], df_risk['Implied_Volatility'],
        s=df_risk['VaR_95_Daily_%'] * 80, c=df_risk['Net_Debt_EBITDA'],
        cmap='RdYlGn_r', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    ax3.axvline(x=1.0, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Market Beta = 1')
    for _, row in df_risk.nlargest(8, 'VaR_95_Daily_%').iterrows():
        ax3.annotate(row['Ticker'], (row['Beta'], row['Implied_Volatility']),
                    fontsize=9, color='red', fontweight='bold')
    ax3.set_xlabel('Beta (Systematic Risk)', fontsize=12)
    ax3.set_ylabel('Implied Volatility (%)', fontsize=12)
    ax3.set_title('RISK MAP: SYSTEMATIC vs TOTAL RISK\nBubble Size = VaR | Color = Leverage', fontsize=14, fontweight='bold')
    plt.colorbar(scatter3, ax=ax3, label='Net Debt/EBITDA', shrink=0.8)
    ax3.legend(loc='upper left', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Chart 4: P/E vs ROE
    print("[4/12] Valuation: P/E vs Profitability...")
    fig4, ax4 = plt.subplots(figsize=(14, 10))
    df_pe = df[(df['PE_Ratio'] > 0) & (df['PE_Ratio'] < 50)].copy()
    scatter4 = ax4.scatter(
        df_pe['PE_Ratio'], df_pe['ROE'],
        s=df_pe['Market_Cap'] / 1e9 * 5, c=df_pe['PEG_Ratio'].clip(-5, 5),
        cmap='RdYlGn_r', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    x_line = np.linspace(0, 50, 100)
    ax4.plot(x_line, x_line, 'g--', alpha=0.5, linewidth=2, label='P/E = ROE')
    for _, row in df_pe.nlargest(10, 'ROE').iterrows():
        ax4.annotate(row['Ticker'], (row['PE_Ratio'], row['ROE']),
                    fontsize=9, fontweight='bold')
    ax4.set_xlabel('P/E Ratio', fontsize=12)
    ax4.set_ylabel('ROE (%)', fontsize=12)
    ax4.set_title('VALUATION: P/E vs PROFITABILITY\nBubble Size = Market Cap | Color = PEG Ratio', fontsize=14, fontweight='bold')
    ax4.set_xlim(0, 50)
    plt.colorbar(scatter4, ax=ax4, label='PEG Ratio', shrink=0.8)
    ax4.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Chart 5: Dividend Sustainability
    print("[5/12] Dividend Sustainability...")
    fig5, ax5 = plt.subplots(figsize=(14, 10))
    df_div = df[df['Payout_Ratio_Est'] < 200].copy()
    scatter5 = ax5.scatter(
        df_div['Payout_Ratio_Est'], df_div['Dividend_Yield'],
        s=df_div['ROE'] * 12, c=df_div['Net_Debt_EBITDA'],
        cmap='RdYlGn_r', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    ax5.axvline(x=75, color='orange', linestyle='--', alpha=0.7, linewidth=2, label='Warning (75%)')
    ax5.axvline(x=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Unsustainable (100%)')
    for _, row in df_div.nlargest(12, 'Dividend_Quality').iterrows():
        ax5.annotate(row['Ticker'], (row['Payout_Ratio_Est'], row['Dividend_Yield']),
                    fontsize=9, fontweight='bold')
    ax5.set_xlabel('Payout Ratio (%)', fontsize=12)
    ax5.set_ylabel('Dividend Yield (%)', fontsize=12)
    ax5.set_title('DIVIDEND SUSTAINABILITY\nBubble Size = ROE | Color = Leverage', fontsize=14, fontweight='bold')
    plt.colorbar(scatter5, ax=ax5, label='Net Debt/EBITDA', shrink=0.8)
    ax5.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Chart 6: Magic Formula
    print("[6/12] Magic Formula...")
    fig6, ax6 = plt.subplots(figsize=(14, 10))
    scatter6 = ax6.scatter(
        df['ROIC'], df['Earnings_Yield_%'],
        s=df['Market_Cap'] / 1e9 * 5, c=df['Magic_Formula_Rank'],
        cmap='RdYlGn_r', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    for _, row in df.nsmallest(15, 'Magic_Formula_Rank').iterrows():
        ax6.annotate(row['Ticker'], (row['ROIC'], row['Earnings_Yield_%']),
                    fontsize=9, fontweight='bold', color='darkgreen')
    ax6.set_xlabel('ROIC (%)', fontsize=12)
    ax6.set_ylabel('Earnings Yield (%)', fontsize=12)
    ax6.set_title("GREENBLATT'S MAGIC FORMULA\nBubble Size = Market Cap | Color = Rank", fontsize=14, fontweight='bold')
    plt.colorbar(scatter6, ax=ax6, label='Magic Formula Rank', shrink=0.8)
    plt.tight_layout()
    plt.show()
    
    # Chart 7: Market Cap
    print("[7/12] Market Cap Distribution...")
    fig7, ax7 = plt.subplots(figsize=(14, 10))
    df_top20 = df.nlargest(20, 'Market_Cap').copy()
    df_top20['MC_Billions'] = df_top20['Market_Cap'] / 1e9
    circles = ax7.scatter(
        range(len(df_top20)), df_top20['MC_Billions'],
        s=df_top20['MC_Billions'] * 5, c=df_top20['Quality_Score'],
        cmap='RdYlGn', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    for i, (_, row) in enumerate(df_top20.iterrows()):
        ax7.annotate(f"{row['Ticker']}\nR${row['MC_Billions']:.0f}B", 
                    (i, row['MC_Billions']), fontsize=9, ha='center', va='bottom', fontweight='bold')
    ax7.set_ylabel('Market Cap (R$ Billions)', fontsize=12)
    ax7.set_title('TOP 20 STOCKS BY MARKET CAP', fontsize=14, fontweight='bold')
    ax7.set_xticks([])
    plt.colorbar(circles, ax=ax7, label='Quality Score', shrink=0.8)
    plt.tight_layout()
    plt.show()
    
    # Chart 8: Correlation Heatmap
    print("[8/12] Factor Correlation Matrix...")
    fig8, ax8 = plt.subplots(figsize=(14, 12))
    corr_cols = ['PE_Ratio', 'PB_Ratio', 'ROE', 'ROIC', 'Beta', 'Implied_Volatility', 
                 'Dividend_Yield', 'Net_Debt_EBITDA', 'Quality_Score', 'Value_Score']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=0.5, ax=ax8, annot_kws={'size': 11})
    ax8.set_title('FACTOR CORRELATION MATRIX', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Chart 9: Composite Ranking
    print("[9/12] Composite Score Ranking...")
    fig9, ax9 = plt.subplots(figsize=(14, 12))
    top25 = df.nlargest(25, 'Composite_Score')
    colors = ['#2ca02c' if x >= 60 else '#ff7f0e' if x >= 50 else '#d62728' for x in top25['Composite_Score']]
    ax9.barh(top25['Ticker'], top25['Composite_Score'], color=colors, edgecolor='darkgray')
    ax9.axvline(x=df['Composite_Score'].median(), color='blue', linestyle='--', 
                linewidth=2, label=f"Median: {df['Composite_Score'].median():.1f}")
    ax9.set_xlabel('Composite Score', fontsize=12)
    ax9.set_title('TOP 25 STOCKS - MULTIFACTOR RANKING', fontsize=14, fontweight='bold')
    ax9.legend(loc='lower right', fontsize=10)
    ax9.invert_yaxis()
    plt.tight_layout()
    plt.show()
    
    # Chart 10: VaR Distribution
    print("[10/12] VaR Distribution...")
    fig10, ax10 = plt.subplots(figsize=(14, 10))
    df_var = df[df['VaR_95_Daily_%'] > 0].copy()
    n, bins, patches = ax10.hist(df_var['VaR_95_Daily_%'], bins=20, alpha=0.7, edgecolor='white')
    for i, patch in enumerate(patches):
        if bins[i] < 2:
            patch.set_facecolor('#2ca02c')
        elif bins[i] < 3.5:
            patch.set_facecolor('#ff7f0e')
        else:
            patch.set_facecolor('#d62728')
    ax10.axvline(x=df_var['VaR_95_Daily_%'].median(), color='blue', linestyle='--', 
                 linewidth=2, label=f"Median: {df_var['VaR_95_Daily_%'].median():.2f}%")
    ax10.set_xlabel('Daily VaR 95% (%)', fontsize=12)
    ax10.set_ylabel('Number of Stocks', fontsize=12)
    ax10.set_title('VALUE AT RISK DISTRIBUTION', fontsize=14, fontweight='bold')
    ax10.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
    
    # Chart 11: Liquidity vs Return
    print("[11/12] Liquidity vs Return...")
    fig11, ax11 = plt.subplots(figsize=(14, 10))
    scatter11 = ax11.scatter(
        np.log10(df['Volume'] + 1), df['Earnings_Yield_%'],
        s=df['Turnover_Ratio'] * 8000, c=df['Liquidity_Score'],
        cmap='Blues', alpha=0.7, edgecolors='darkgray', linewidth=1
    )
    for _, row in df.nlargest(12, 'Liquidity_Score').iterrows():
        ax11.annotate(row['Ticker'], (np.log10(row['Volume'] + 1), row['Earnings_Yield_%']),
                     fontsize=9, fontweight='bold')
    ax11.set_xlabel('Log10(Volume)', fontsize=12)
    ax11.set_ylabel('Earnings Yield (%)', fontsize=12)
    ax11.set_title('LIQUIDITY vs RETURN ANALYSIS', fontsize=14, fontweight='bold')
    plt.colorbar(scatter11, ax=ax11, label='Liquidity Score', shrink=0.8)
    plt.tight_layout()
    plt.show()
    
    # Chart 12: Radar Chart
    print("[12/12] Factor Exposure Radar...")
    fig12, ax12 = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))
    top5 = df.nlargest(5, 'Composite_Score')
    categories = ['Quality', 'Value', 'Dividend', 'Liquidity', 'Low Risk']
    n_cats = len(categories)
    angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
    angles += angles[:1]
    stock_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for idx, (_, row) in enumerate(top5.iterrows()):
        values = [
            row['Quality_Score'] / 100,
            row['Value_Score'] / 100,
            row['Dividend_Quality'] / 100,
            row['Liquidity_Score'] / 100,
            1 - (row['VaR_95_Daily_%'] / df['VaR_95_Daily_%'].max()) if df['VaR_95_Daily_%'].max() > 0 else 0.5
        ]
        values += values[:1]
        ax12.plot(angles, values, 'o-', linewidth=2, label=row['Ticker'], color=stock_colors[idx], markersize=8)
        ax12.fill(angles, values, alpha=0.15, color=stock_colors[idx])
    ax12.set_xticks(angles[:-1])
    ax12.set_xticklabels(categories, fontsize=12)
    ax12.set_title('FACTOR EXPOSURE: TOP 5 STOCKS', fontsize=14, fontweight='bold', pad=20)
    ax12.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=11)
    plt.tight_layout()
    plt.show()
    
    print("\nAll 12 charts displayed.")


def calculate_growth_analysis(df, show_details=True):
    """
    GROWTH ANALYSIS - CAGR and Sustainable Growth Metrics
    
    Used by: Growth investors, Peter Lynch, T. Rowe Price
    
    Uses actual StatusInvest CAGR data for revenue and earnings growth
    """
    # Sustainable Growth Rate = ROE * Retention Ratio
    df['Retention_Ratio'] = np.clip(1 - (df['Dividend_Yield'] * df['PE_Ratio'] / 100), 0, 1)
    df['Sustainable_Growth_Rate'] = df['ROE'] * df['Retention_Ratio']
    
    # Growth Score calculation (already done in quality score, but let's be explicit)
    revenue_growth_norm = (df['Revenue_CAGR_5Y'].clip(-50, 100) + 50) / 150 * 100
    earnings_growth_norm = (df['Earnings_CAGR_5Y'].clip(-50, 150) + 50) / 200 * 100
    sustainable_growth_norm = df['Sustainable_Growth_Rate'].clip(0, 30) / 30 * 100
    
    df['Growth_Score'] = (
        0.40 * revenue_growth_norm +
        0.40 * earnings_growth_norm +
        0.20 * sustainable_growth_norm
    ).fillna(0)
    
    # Growth Quality = Consistency (earnings growth > revenue growth is quality)
    df['Growth_Quality'] = np.where(
        df['Earnings_CAGR_5Y'] > df['Revenue_CAGR_5Y'],
        'MARGIN_EXPANSION',
        np.where(
            df['Earnings_CAGR_5Y'] < df['Revenue_CAGR_5Y'] * 0.5,
            'MARGIN_COMPRESSION',
            'STABLE_MARGINS'
        )
    )
    
    if show_details:
        print("\n" + "=" * 80)
        print("GROWTH ANALYSIS (StatusInvest CAGR Data)")
        print("=" * 80)
        print("""
FORMULAS:

1. REVENUE CAGR (5 Years):
   Formula: CAGR = (Ending Value / Beginning Value)^(1/n) - 1
   
   Source: StatusInvest historical data
   
   Interpretation:
   - > 15%: High growth company
   - 5-15%: Moderate growth
   - 0-5%:  Slow growth
   - < 0%:  Declining revenue

2. EARNINGS CAGR (5 Years):
   Formula: Same as above, but for Net Income
   
   More volatile than revenue due to operating leverage
   
   Quality Signal:
   - Earnings CAGR > Revenue CAGR = Margin expansion (positive)
   - Earnings CAGR < Revenue CAGR = Margin compression (negative)

3. SUSTAINABLE GROWTH RATE (SGR):
   Formula: SGR = ROE x Retention Ratio
            Retention Ratio = 1 - Dividend Payout Ratio
   
   Origin: DuPont Analysis extension
   
   Meaning: Maximum growth rate achievable without external financing
   
   Example:
   - ROE = 20%, Payout = 40%
   - Retention = 60%
   - SGR = 20% x 0.60 = 12%

4. GROWTH SCORE:
   Components (Weighted Average):
   - Revenue CAGR 5Y (40%)
   - Earnings CAGR 5Y (40%)
   - Sustainable Growth Rate (20%)
   
   Range: 0-100, higher is better

GROWTH QUALITY SIGNALS:
    MARGIN_EXPANSION:   Earnings growing faster than revenue (improving efficiency)
    STABLE_MARGINS:     Earnings and revenue growing at similar rates
    MARGIN_COMPRESSION: Earnings growing slower than revenue (losing efficiency)

PETER LYNCH'S PEG RATIO:
    PEG = P/E / Earnings Growth Rate
    
    - PEG < 1: Stock may be undervalued relative to growth
    - PEG = 1: Fair valued
    - PEG > 1: May be overvalued relative to growth
        """)
        
        print("\nGROWTH CLASSIFICATION BY REVENUE CAGR:")
        print("-" * 70)
        high_growth = (df['Revenue_CAGR_5Y'] > 15).sum()
        moderate_growth = ((df['Revenue_CAGR_5Y'] >= 5) & (df['Revenue_CAGR_5Y'] <= 15)).sum()
        slow_growth = ((df['Revenue_CAGR_5Y'] >= 0) & (df['Revenue_CAGR_5Y'] < 5)).sum()
        declining = (df['Revenue_CAGR_5Y'] < 0).sum()
        
        print(f"High Growth (>15%):      {high_growth:3d} stocks")
        print(f"Moderate Growth (5-15%): {moderate_growth:3d} stocks")
        print(f"Slow Growth (0-5%):      {slow_growth:3d} stocks")
        print(f"Declining (<0%):         {declining:3d} stocks")
        
        print("\nGROWTH QUALITY DISTRIBUTION:")
        print("-" * 70)
        print(df['Growth_Quality'].value_counts().to_string())
        
        print("\nTOP 10 GROWTH STOCKS:")
        print("-" * 80)
        result = df[['Ticker', 'Revenue_CAGR_5Y', 'Earnings_CAGR_5Y', 
                     'Sustainable_Growth_Rate', 'PEG_Ratio', 'Growth_Score', 'Growth_Quality']].copy()
        result = result.nlargest(10, 'Growth_Score')
        print(result.round(2).to_string(index=False))
        
        print("\nBEST PEG RATIOS (Value + Growth):")
        print("-" * 80)
        peg_valid = df[df['PEG_Ratio'] > 0].nsmallest(10, 'PEG_Ratio')
        result = peg_valid[['Ticker', 'PE_Ratio', 'Earnings_CAGR_5Y', 'PEG_Ratio', 'Growth_Score']].copy()
        print(result.round(2).to_string(index=False))
    
    return df


def calculate_efficiency_analysis(df, show_details=True):
    """
    EFFICIENCY ANALYSIS - Asset Utilization and Capital Structure
    
    Used by: Warren Buffett, DuPont Analysis practitioners
    
    Uses actual StatusInvest efficiency metrics
    """
    # DuPont Analysis components
    df['Profit_Margin_DuPont'] = df['Net_Margin'] / 100
    df['Asset_Turnover_DuPont'] = df['Asset_Turnover']
    df['Financial_Leverage_DuPont'] = 1 / df['Equity_Assets'].replace(0, np.nan).fillna(0.01)
    
    # DuPont ROE Decomposition
    df['DuPont_ROE'] = df['Profit_Margin_DuPont'] * df['Asset_Turnover_DuPont'] * df['Financial_Leverage_DuPont'] * 100
    
    # Efficiency Score
    asset_turnover_norm = df['Asset_Turnover'].clip(0, 2) / 2 * 100
    equity_ratio_norm = df['Equity_Assets'].clip(0, 100)
    low_liabilities_norm = (100 - df['Liabilities_Assets'].clip(0, 100))
    roic_norm = df['ROIC'].clip(-10, 40) / 40 * 100
    
    df['Efficiency_Score'] = (
        0.30 * asset_turnover_norm +
        0.25 * equity_ratio_norm +
        0.25 * low_liabilities_norm +
        0.20 * roic_norm
    ).fillna(0)
    
    # Capital Intensity Classification
    df['Capital_Intensity'] = np.where(
        df['Asset_Turnover'] > 1.5,
        'ASSET_LIGHT',
        np.where(
            df['Asset_Turnover'] < 0.5,
            'CAPITAL_INTENSIVE',
            'MODERATE'
        )
    )
    
    if show_details:
        print("\n" + "=" * 80)
        print("EFFICIENCY ANALYSIS (DuPont Decomposition)")
        print("=" * 80)
        print("""
FORMULAS:

1. DuPONT ANALYSIS (3-Factor):
   Formula: ROE = Profit Margin x Asset Turnover x Financial Leverage
   
   Origin: DuPont Corporation, 1920s
   
   Components:
   a) Profit Margin = Net Income / Revenue
      - Measures pricing power and cost control
      - Higher = more efficient at converting sales to profit
   
   b) Asset Turnover = Revenue / Total Assets
      - Measures asset utilization efficiency
      - Higher = generating more revenue per R$ of assets
   
   c) Financial Leverage = Total Assets / Shareholders' Equity
      - Measures use of debt financing
      - Higher = more debt-financed (higher risk)

2. ASSET TURNOVER (from StatusInvest):
   Formula: Asset Turnover = Revenue / Total Assets
   
   Interpretation:
   - > 1.5: Asset-light business (retail, tech)
   - 0.5-1.5: Moderate capital needs
   - < 0.5: Capital-intensive (utilities, manufacturing)

3. EQUITY/ASSETS RATIO (from StatusInvest):
   Formula: Equity Ratio = Shareholders' Equity / Total Assets
   
   Interpretation:
   - > 60%: Very strong balance sheet
   - 40-60%: Healthy
   - 20-40%: Moderate leverage
   - < 20%: Highly leveraged

4. ROIC (Return on Invested Capital):
   Formula: ROIC = NOPAT / Invested Capital
   
   Key Insight:
   - ROIC > WACC: Creates shareholder value
   - ROIC < WACC: Destroys shareholder value

5. EFFICIENCY SCORE:
   Components:
   - Asset Turnover (30%): Revenue generation efficiency
   - Equity Ratio (25%): Balance sheet strength
   - Low Liabilities (25%): Conservative financing
   - ROIC (20%): Capital efficiency
   
   Range: 0-100, higher is better
        """)
        
        print("\nDuPONT ROE DECOMPOSITION (Top 10 by ROE):")
        print("-" * 90)
        result = df[['Ticker', 'Net_Margin', 'Asset_Turnover', 'Equity_Assets', 
                     'ROE', 'DuPont_ROE']].copy()
        result['Leverage'] = 100 / result['Equity_Assets'].replace(0, 1)
        result = result.nlargest(10, 'ROE')
        print(result.round(2).to_string(index=False))
        
        print("\nCAPITAL INTENSITY CLASSIFICATION:")
        print("-" * 70)
        print(df['Capital_Intensity'].value_counts().to_string())
        
        print("\nMOST EFFICIENT STOCKS (High Turnover, Low Leverage):")
        print("-" * 80)
        result = df[['Ticker', 'Asset_Turnover', 'Equity_Assets', 'ROIC', 
                     'Efficiency_Score', 'Capital_Intensity']].copy()
        result = result.nlargest(10, 'Efficiency_Score')
        print(result.round(2).to_string(index=False))
        
        print("\nCAPITAL-INTENSIVE STOCKS (Require Close Monitoring):")
        print("-" * 80)
        capital_intensive = df[df['Capital_Intensity'] == 'CAPITAL_INTENSIVE'].nsmallest(10, 'Asset_Turnover')
        result = capital_intensive[['Ticker', 'Asset_Turnover', 'Equity_Assets', 'ROIC', 
                                    'Efficiency_Score']].copy()
        print(result.round(2).to_string(index=False))
    
    return df


def export_results(df):
    """Export results to Excel with comprehensive StatusInvest data"""
    # Core valuation metrics
    export_cols = [
        'Ticker', 'Price', 'Market_Cap', 'Portfolio_Weight_%',
        # Valuation ratios (from StatusInvest)
        'PE_Ratio', 'PB_Ratio', 'PS_Ratio', 'P_EBIT', 'EV_EBIT', 'P_Assets',
        'Dividend_Yield', 'Earnings_Yield_%', 'PEG_Ratio',
        # Profitability metrics (from StatusInvest)
        'ROE', 'ROA', 'ROIC', 'Gross_Margin', 'EBIT_Margin', 'Net_Margin',
        # Growth metrics (from StatusInvest)
        'Revenue_CAGR_5Y', 'Earnings_CAGR_5Y',
        # Efficiency metrics (from StatusInvest)
        'Asset_Turnover', 'Equity_Assets', 'Liabilities_Assets',
        # Leverage & Liquidity (from StatusInvest)
        'Net_Debt_Equity', 'Net_Debt_EBIT', 'Net_Debt_EBITDA', 'Current_Ratio',
        # Risk metrics (calculated)
        'Beta', 'Adj_Beta', 'Implied_Volatility',
        'VaR_95_Daily_%', 'VaR_99_Daily_%', 'CVaR_95_%', 'Max_Drawdown_Est_%',
        'Sharpe_Ratio', 'Sortino_Ratio', 'Treynor_Ratio',
        # Value metrics (calculated)
        'Graham_Number', 'Graham_Upside_%', 'EPS', 'Book_Value_Per_Share',
        # Composite scores (calculated)
        'Quality_Score', 'Value_Score', 'EV_Value_Score', 'Growth_Score',
        'Profitability_Score', 'Safety_Score', 'Efficiency_Score',
        'Liquidity_Score', 'Dividend_Quality', 'Piotroski_Score',
        'Magic_Formula_Rank', 'Composite_Score', 'Risk_Adj_Score'
    ]
    
    # Filter to only include columns that exist
    export_cols = [col for col in export_cols if col in df.columns]
    
    output_path = r'C:\Users\shodan\Desktop\STOCKS_ANALYSIS.xlsx'
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        df[export_cols].round(4).to_excel(writer, sheet_name='Full_Analysis', index=False)
        df.nlargest(20, 'Composite_Score')[export_cols].round(4).to_excel(writer, sheet_name='Top_20_Picks', index=False)
        df.nsmallest(20, 'VaR_95_Daily_%')[export_cols].round(4).to_excel(writer, sheet_name='Low_Risk_Stocks', index=False)
        df.nlargest(20, 'Dividend_Quality')[export_cols].round(4).to_excel(writer, sheet_name='Dividend_Stocks', index=False)
        df.nsmallest(20, 'Magic_Formula_Rank')[export_cols].round(4).to_excel(writer, sheet_name='Magic_Formula', index=False)
        # Growth stocks
        df.nlargest(20, 'Growth_Score')[export_cols].round(4).to_excel(writer, sheet_name='Growth_Stocks', index=False)
        # Quality stocks
        df.nlargest(20, 'Quality_Score')[export_cols].round(4).to_excel(writer, sheet_name='Quality_Stocks', index=False)
        # Value stocks
        df.nlargest(20, 'Value_Score')[export_cols].round(4).to_excel(writer, sheet_name='Value_Stocks', index=False)
        # Summary statistics
        df[export_cols].describe().round(4).to_excel(writer, sheet_name='Summary_Stats')
    
    print(f"\nResults exported to: {output_path}")
    print(f"Sheets: Full_Analysis, Top_20_Picks, Low_Risk_Stocks, Dividend_Stocks,")
    print(f"        Magic_Formula, Growth_Stocks, Quality_Stocks, Value_Stocks, Summary_Stats")


def display_menu():
    """Display the main menu"""
    print("\n" + "=" * 80)
    print("STOCK ANALYSIS MENU")
    print("=" * 80)
    print("""
VALUATION FORMULAS:
    1.  Graham Number          - Benjamin Graham's intrinsic value
    2.  Earnings Yield         - Inverse of P/E ratio
    3.  PEG Ratio              - Peter Lynch's growth-adjusted P/E

RISK METRICS:
    4.  Value at Risk (VaR)    - JPMorgan's risk measurement
    5.  Sharpe Ratio           - Risk-adjusted return metrics
    6.  Beta Analysis          - Systematic risk measurement

QUALITY FACTORS:
    7.  Quality Score          - Profitability, Safety, Growth, Efficiency
    8.  Piotroski F-Score      - Financial health score (0-9)

RANKING SYSTEMS:
    9.  Magic Formula          - Joel Greenblatt's ranking
    10. Value Score            - Fama-French value factor
    11. Composite Score        - Multifactor ranking

INCOME ANALYSIS:
    12. Dividend Metrics       - Payout, Gordon Model, Quality

GROWTH & EFFICIENCY (StatusInvest Data):
    13. Growth Analysis        - CAGR, Sustainable Growth, PEG
    14. Efficiency Analysis    - DuPont, Asset Turnover, ROIC

PORTFOLIO:
    15. Liquidity Metrics      - Trading and liquidity analysis
    16. Portfolio Statistics   - Aggregate portfolio metrics

ACTIONS:
    17. Run ALL Calculations   - Execute all formulas with details
    18. Display Charts         - Show all 12 visualization charts
    19. Export to Excel        - Save results (9 sheets)
    
    0.  Exit
    """)


def main():
    """Main program loop"""
    print("=" * 80)
    print("INSTITUTIONAL STOCK PORTFOLIO ANALYSIS")
    print("Advanced Analytics Engine - Bank-Grade Calculations")
    print("Using StatusInvest Data: 30 Metrics per Stock")
    print("=" * 80)
    
    # Load data
    print("\nLoading data...")
    df = load_data()
    print(f"Loaded {len(df)} stocks with StatusInvest data")
    
    # Run all calculations silently first
    df = run_all_calculations(df, show_details=False)
    # Also run new analysis functions
    df = calculate_growth_analysis(df, show_details=False)
    df = calculate_efficiency_analysis(df, show_details=False)
    print("All calculations completed (including Growth & Efficiency analysis).")
    
    while True:
        display_menu()
        
        try:
            choice = input("\nEnter your choice (0-19): ").strip()
            
            if choice == '0':
                print("\nExiting. Goodbye!")
                break
            elif choice == '1':
                df = calculate_graham_number(df, show_details=True)
            elif choice == '2':
                df = calculate_earnings_yield(df, show_details=True)
            elif choice == '3':
                df = calculate_peg_ratio(df, show_details=True)
            elif choice == '4':
                df = calculate_var(df, show_details=True)
            elif choice == '5':
                df = calculate_sharpe_ratio(df, show_details=True)
            elif choice == '6':
                df = calculate_beta_analysis(df, show_details=True)
            elif choice == '7':
                df = calculate_quality_score(df, show_details=True)
            elif choice == '8':
                df = calculate_piotroski_score(df, show_details=True)
            elif choice == '9':
                df = calculate_magic_formula(df, show_details=True)
            elif choice == '10':
                df = calculate_value_score(df, show_details=True)
            elif choice == '11':
                df = calculate_composite_score(df, show_details=True)
            elif choice == '12':
                df = calculate_dividend_metrics(df, show_details=True)
            elif choice == '13':
                df = calculate_growth_analysis(df, show_details=True)
            elif choice == '14':
                df = calculate_efficiency_analysis(df, show_details=True)
            elif choice == '15':
                df = calculate_liquidity_metrics(df, show_details=True)
            elif choice == '16':
                df = calculate_portfolio_stats(df, show_details=True)
            elif choice == '17':
                df = run_all_calculations(df, show_details=True)
                df = calculate_growth_analysis(df, show_details=True)
                df = calculate_efficiency_analysis(df, show_details=True)
            elif choice == '18':
                display_charts(df)
            elif choice == '19':
                export_results(df)
            else:
                print("\nInvalid choice. Please enter a number between 0 and 19.")
        
        except KeyboardInterrupt:
            print("\n\nExiting. Goodbye!")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    main()
