"""
FII (Fundos de Investimento Imobiliário) Analyzer
Reads XLSX file with FII data and performs comprehensive financial analysis
Provides buy/hold/sell recommendations based on multiple factors
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List
from dataclasses import dataclass
from enum import Enum


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


class FIIAnalyzer:
    """
    Comprehensive FII Analyzer with multiple valuation methods
    """
    
    # Column mapping (Portuguese to English)
    COLUMN_MAPPING = {
        'Código': 'ticker',
        'Recebível': 'fund_name',
        'Fundo': 'fund_name',
        'Fechamento': 'price',
        'Part. IFIX': 'ifix_weight',
        'Volume de Negociação': 'avg_daily_volume',
        'Valor de Mercado': 'market_cap',
        'Valor Patrimonial': 'book_value',
        'P/VPA Atual': 'pvpa_current',
        'P/VPA 2025': 'pvpa_projected',
        'Dividend Yield LTM': 'dy_ltm',
        'Dividend Yield Anualizado': 'dy_annualized',
        'Último Dividendo': 'last_dividend',
        'Retorno no Mês': 'return_month',
        'Retorno no Ano': 'return_year',
        'Retorno LTM': 'return_ltm'
    }
    
    # Positional column mapping (for files with generic column names)
    POSITIONAL_MAPPING = {
        0: 'ticker',           # Código
        1: 'fund_name',        # Recebível/Fundo
        2: 'price',            # R$ (Fechamento)
        3: 'ifix_weight',      # (%) Part. IFIX
        4: 'avg_daily_volume', # Média Diária (R$) - Volume
        5: 'market_cap',       # R$.1 - Valor de Mercado
        6: 'book_value',       # R$.2 - Valor Patrimonial
        7: 'pvpa_current',     # - P/VPA Atual
        8: 'pvpa_projected',   # -.1 P/VPA 2025
        9: 'dy_ltm',           # (% ao ano) - Dividend Yield LTM
        10: 'dy_annualized',   # (% ao ano).1 - Dividend Yield Anualizado
        11: 'last_dividend',   # R$/cota - Último Dividendo
        12: 'return_month',    # (%).1 - Retorno no Mês
        13: 'return_year',     # (%).2 - Retorno no Ano
        14: 'return_ltm'       # (%).3 - Retorno LTM
    }
    
    # Analysis parameters
    RISK_FREE_RATE = 0.1175  # SELIC rate (approximate)
    MARKET_RISK_PREMIUM = 0.05
    PERPETUAL_GROWTH_RATE = 0.03
    
    def __init__(self, file_path: str):
        """Initialize analyzer with XLSX file path"""
        self.file_path = file_path
        self.df = None
        self.all_sheets_data = {}  # Store data from all sheets
        self.results: List[AnalysisResult] = []
        self.results_by_segment: Dict[str, List[AnalysisResult]] = {}
        
    def load_data(self) -> pd.DataFrame:
        """Load and preprocess XLSX data from all sheets"""
        print(f"\n{'='*60}")
        print("Loading FII data from XLSX file...")
        print(f"{'='*60}")
        
        # Get all sheet names
        xl = pd.ExcelFile(self.file_path)
        sheet_names = xl.sheet_names
        print(f"Found {len(sheet_names)} sheets: {sheet_names}")
        
        all_dfs = []
        
        for sheet_name in sheet_names:
            print(f"\n  Processing sheet: {sheet_name}")
            
            # Read sheet without headers first to detect structure
            df_raw = pd.read_excel(self.file_path, sheet_name=sheet_name, header=None)
            
            # Find the row that contains 'Código' (the actual header row)
            header_row = None
            for idx, row in df_raw.iterrows():
                row_str = ' '.join([str(x) for x in row.values])
                if 'Código' in row_str:
                    header_row = idx
                    break
            
            if header_row is not None:
                # Read again with correct header row
                df_sheet = pd.read_excel(self.file_path, sheet_name=sheet_name, header=header_row)
            else:
                # Try different header positions
                df_sheet = pd.read_excel(self.file_path, sheet_name=sheet_name, header=5)
            
            # Clean up column names
            df_sheet.columns = [str(c).strip() if pd.notna(c) else f'Col_{i}' 
                              for i, c in enumerate(df_sheet.columns)]
            
            # Rename first two columns if they match the pattern
            if len(df_sheet.columns) >= 2:
                cols = list(df_sheet.columns)
                if cols[0] == 'Código' or 'Código' in str(cols[0]):
                    cols[0] = 'Código'
                if any(x in str(cols[1]) for x in ['Recebível', 'Fundo', 'Laje', 'Galpão', 'Híbrido']):
                    cols[1] = 'Fundo'
                df_sheet.columns = cols
            
            # Standardize column names for this sheet
            df_sheet = self._standardize_sheet_columns(df_sheet)
            
            # Convert percentages and multipliers
            df_sheet = self._convert_sheet_percentages(df_sheet)
            df_sheet = self._convert_sheet_multipliers(df_sheet)
            
            # Filter valid tickers
            if 'ticker' in df_sheet.columns:
                df_sheet = df_sheet[df_sheet['ticker'].notna() & (df_sheet['ticker'] != '')]
                df_sheet = df_sheet[df_sheet['ticker'].astype(str).str.match(r'^[A-Z]{4}\d{2}$', na=False)]
            
            # Add segment column
            df_sheet['segment'] = sheet_name
            
            # Store in dictionary
            self.all_sheets_data[sheet_name] = df_sheet.copy()
            
            print(f"    Loaded {len(df_sheet)} FII records")
            
            all_dfs.append(df_sheet)
        
        # Combine all sheets
        self.df = pd.concat(all_dfs, ignore_index=True)
        
        print(f"\n{'='*60}")
        print(f"Total FIIs loaded from all sheets: {len(self.df)}")
        print(f"{'='*60}")
        
        return self.df
    
    def _standardize_sheet_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Map Portuguese columns to English names for a single sheet"""
        rename_dict = {}
        mapped_count = 0
        
        # First try name-based mapping
        for old_col in df.columns:
            for pt_key, en_key in self.COLUMN_MAPPING.items():
                if pt_key.lower() in str(old_col).lower():
                    rename_dict[old_col] = en_key
                    mapped_count += 1
                    break
        
        # If name-based mapping didn't work well, use positional mapping
        if mapped_count < 5 and len(df.columns) >= 10:
            rename_dict = {}
            for idx, col in enumerate(df.columns):
                if idx in self.POSITIONAL_MAPPING:
                    rename_dict[col] = self.POSITIONAL_MAPPING[idx]
        
        return df.rename(columns=rename_dict)
    
    def _convert_sheet_percentages(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert percentage columns from string to float for a single sheet"""
        pct_columns = ['ifix_weight', 'dy_ltm', 'dy_annualized', 
                       'return_month', 'return_year', 'return_ltm']
        
        for col in pct_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('%', '').str.replace(',', '.'),
                    errors='coerce'
                )
                # Check if values are already in decimal format
                median_val = df[col].median()
                if pd.notna(median_val) and abs(median_val) > 1:
                    df[col] = df[col] / 100
        
        return df
    
    def _convert_sheet_multipliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert multiplier columns (e.g., 0.87x) to float for a single sheet"""
        mult_columns = ['pvpa_current', 'pvpa_projected']
        
        for col in mult_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(
                    df[col].astype(str).str.replace('x', '').str.replace(',', '.'),
                    errors='coerce'
                )
        
        return df
    
    def _standardize_columns(self):
        """Map Portuguese columns to English names"""
        rename_dict = {}
        mapped_count = 0
        
        # First try name-based mapping
        for old_col in self.df.columns:
            for pt_key, en_key in self.COLUMN_MAPPING.items():
                if pt_key.lower() in str(old_col).lower():
                    rename_dict[old_col] = en_key
                    mapped_count += 1
                    break
        
        # If name-based mapping didn't work well, use positional mapping
        if mapped_count < 5 and len(self.df.columns) >= 10:
            print("Using positional column mapping...")
            rename_dict = {}
            for idx, col in enumerate(self.df.columns):
                if idx in self.POSITIONAL_MAPPING:
                    rename_dict[col] = self.POSITIONAL_MAPPING[idx]
        
        self.df.rename(columns=rename_dict, inplace=True)
    
    def _convert_percentages(self):
        """Convert percentage columns from string to float"""
        pct_columns = ['ifix_weight', 'dy_ltm', 'dy_annualized', 
                       'return_month', 'return_year', 'return_ltm']
        
        for col in pct_columns:
            if col in self.df.columns:
                # Convert to numeric first
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace('%', '').str.replace(',', '.'),
                    errors='coerce'
                )
                # Check if values are already in decimal format (e.g., 0.12 for 12%)
                # If median is > 1, assume they are percentages that need dividing
                median_val = self.df[col].median()
                if pd.notna(median_val) and abs(median_val) > 1:
                    self.df[col] = self.df[col] / 100
    
    def _convert_multipliers(self):
        """Convert multiplier columns (e.g., 0.87x) to float"""
        mult_columns = ['pvpa_current', 'pvpa_projected']
        
        for col in mult_columns:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(
                    self.df[col].astype(str).str.replace('x', '').str.replace(',', '.'),
                    errors='coerce'
                )
    
    def calculate_variance_analysis(self, row: pd.Series) -> Dict:
        """
        Variance Analysis: Compare current metrics with averages and benchmarks
        """
        variance = {}
        
        # P/VPA Variance (compared to fair value of 1.0)
        pvpa = row.get('pvpa_current', 1.0)
        if pd.notna(pvpa):
            variance['pvpa_variance'] = (pvpa - 1.0) / 1.0 * 100  # % deviation from fair value
            variance['pvpa_status'] = 'DISCOUNT' if pvpa < 1.0 else 'PREMIUM'
        
        # Yield Variance (compared to risk-free rate)
        dy = row.get('dy_annualized', 0)
        if pd.notna(dy):
            variance['yield_spread'] = (dy - self.RISK_FREE_RATE) * 100  # Spread over SELIC
            variance['yield_status'] = 'ATTRACTIVE' if dy > self.RISK_FREE_RATE * 1.2 else 'FAIR'
        
        # Momentum Variance
        return_month = row.get('return_month', 0)
        return_year = row.get('return_year', 0)
        if pd.notna(return_month) and pd.notna(return_year):
            variance['momentum'] = (return_month * 12 + return_year) / 2 * 100
            variance['momentum_status'] = 'POSITIVE' if variance['momentum'] > 0 else 'NEGATIVE'
        
        return variance
    
    def calculate_credit_analysis(self, row: pd.Series) -> Dict:
        """
        Credit Analysis for Receivables FIIs (Recebíveis)
        Evaluates the credit quality based on available metrics
        """
        credit = {}
        
        # Market Cap as proxy for size/stability (larger = more stable)
        market_cap = row.get('market_cap', 0)
        if pd.notna(market_cap) and market_cap > 0:
            # Score based on market cap tiers
            if market_cap >= 1_000_000_000:  # > 1 billion
                credit['size_score'] = 100
                credit['size_tier'] = 'LARGE CAP'
            elif market_cap >= 500_000_000:  # > 500 million
                credit['size_score'] = 75
                credit['size_tier'] = 'MID CAP'
            elif market_cap >= 100_000_000:  # > 100 million
                credit['size_score'] = 50
                credit['size_tier'] = 'SMALL CAP'
            else:
                credit['size_score'] = 25
                credit['size_tier'] = 'MICRO CAP'
        
        # Liquidity Score based on daily volume
        volume = row.get('avg_daily_volume', 0)
        if pd.notna(volume) and volume > 0:
            if volume >= 1_000_000:  # > 1 million daily
                credit['liquidity_score'] = 100
                credit['liquidity_tier'] = 'HIGH'
            elif volume >= 500_000:
                credit['liquidity_score'] = 75
                credit['liquidity_tier'] = 'MEDIUM-HIGH'
            elif volume >= 100_000:
                credit['liquidity_score'] = 50
                credit['liquidity_tier'] = 'MEDIUM'
            else:
                credit['liquidity_score'] = 25
                credit['liquidity_tier'] = 'LOW'
        
        # P/VPA as credit quality indicator
        pvpa = row.get('pvpa_current', 1.0)
        if pd.notna(pvpa):
            # Very low P/VPA might indicate market concerns about credit quality
            if pvpa >= 0.95 and pvpa <= 1.10:
                credit['valuation_score'] = 100
                credit['valuation_status'] = 'FAIRLY VALUED'
            elif pvpa >= 0.85 or pvpa <= 1.20:
                credit['valuation_score'] = 75
                credit['valuation_status'] = 'ACCEPTABLE'
            elif pvpa < 0.85:
                credit['valuation_score'] = 50  # Deep discount may indicate issues
                credit['valuation_status'] = 'DEEP DISCOUNT - VERIFY CREDIT'
            else:
                credit['valuation_score'] = 50
                credit['valuation_status'] = 'PREMIUM - OVERVALUED'
        
        # Overall Credit Score (weighted average)
        scores = [credit.get('size_score', 50), 
                  credit.get('liquidity_score', 50),
                  credit.get('valuation_score', 50)]
        credit['overall_score'] = np.mean(scores)
        
        return credit
    
    def calculate_dcf_valuation(self, row: pd.Series) -> Dict:
        """
        Simplified DCF (Discounted Cash Flow) Valuation
        Uses dividend yield as proxy for cash flow
        """
        dcf = {}
        
        price = row.get('price', 0)
        dividend = row.get('last_dividend', 0)
        dy = row.get('dy_annualized', 0)
        
        if pd.isna(price) or price <= 0:
            return {'fair_value': 0, 'upside': 0}
        
        # Estimate annual dividend
        if pd.notna(dividend) and dividend > 0:
            annual_dividend = dividend * 12  # Assuming monthly distribution
        elif pd.notna(dy) and dy > 0:
            annual_dividend = price * dy
        else:
            return {'fair_value': price, 'upside': 0}
        
        # Gordon Growth Model: Fair Value = D1 / (r - g)
        # Where D1 = expected dividend, r = required return, g = growth rate
        required_return = self.RISK_FREE_RATE + self.MARKET_RISK_PREMIUM
        
        if required_return > self.PERPETUAL_GROWTH_RATE:
            # D1 = D0 * (1 + g)
            d1 = annual_dividend * (1 + self.PERPETUAL_GROWTH_RATE)
            fair_value = d1 / (required_return - self.PERPETUAL_GROWTH_RATE)
            
            dcf['fair_value'] = fair_value
            dcf['upside'] = (fair_value - price) / price * 100
            dcf['status'] = 'UNDERVALUED' if dcf['upside'] > 10 else ('OVERVALUED' if dcf['upside'] < -10 else 'FAIRLY VALUED')
        else:
            dcf['fair_value'] = price
            dcf['upside'] = 0
            dcf['status'] = 'N/A'
        
        return dcf
    
    def calculate_composite_score(self, row: pd.Series) -> Tuple[float, Dict]:
        """
        Calculate composite score based on multiple factors
        Returns score (0-100) and individual component scores
        """
        scores = {}
        weights = {
            'pvpa': 0.25,
            'yield': 0.25,
            'momentum': 0.15,
            'liquidity': 0.15,
            'credit': 0.10,
            'dcf': 0.10
        }
        
        # 1. P/VPA Score (lower is better for value investors)
        pvpa = row.get('pvpa_current', 1.0)
        if pd.notna(pvpa):
            if pvpa <= 0.85:
                scores['pvpa'] = 100
            elif pvpa <= 0.95:
                scores['pvpa'] = 80
            elif pvpa <= 1.05:
                scores['pvpa'] = 60
            elif pvpa <= 1.15:
                scores['pvpa'] = 40
            else:
                scores['pvpa'] = 20
        else:
            scores['pvpa'] = 50
        
        # 2. Yield Score (higher is better)
        dy = row.get('dy_annualized', 0)
        if pd.notna(dy):
            # Score based on yield spread over SELIC
            spread = dy - self.RISK_FREE_RATE
            if spread >= 0.05:  # 5% or more above SELIC
                scores['yield'] = 100
            elif spread >= 0.03:
                scores['yield'] = 80
            elif spread >= 0.01:
                scores['yield'] = 60
            elif spread >= 0:
                scores['yield'] = 40
            else:
                scores['yield'] = 20
        else:
            scores['yield'] = 50
        
        # 3. Momentum Score
        return_ltm = row.get('return_ltm', 0)
        return_year = row.get('return_year', 0)
        if pd.notna(return_ltm):
            if return_ltm >= 0.20:
                scores['momentum'] = 100
            elif return_ltm >= 0.10:
                scores['momentum'] = 80
            elif return_ltm >= 0:
                scores['momentum'] = 60
            elif return_ltm >= -0.10:
                scores['momentum'] = 40
            else:
                scores['momentum'] = 20
        elif pd.notna(return_year):
            if return_year >= 0.10:
                scores['momentum'] = 80
            elif return_year >= 0:
                scores['momentum'] = 60
            else:
                scores['momentum'] = 40
        else:
            scores['momentum'] = 50
        
        # 4. Liquidity Score
        volume = row.get('avg_daily_volume', 0)
        if pd.notna(volume) and volume > 0:
            if volume >= 1_000_000:
                scores['liquidity'] = 100
            elif volume >= 500_000:
                scores['liquidity'] = 80
            elif volume >= 200_000:
                scores['liquidity'] = 60
            elif volume >= 100_000:
                scores['liquidity'] = 40
            else:
                scores['liquidity'] = 20
        else:
            scores['liquidity'] = 50
        
        # 5. Credit Score
        credit_analysis = self.calculate_credit_analysis(row)
        scores['credit'] = credit_analysis.get('overall_score', 50)
        
        # 6. DCF Score
        dcf_analysis = self.calculate_dcf_valuation(row)
        upside = dcf_analysis.get('upside', 0)
        if upside >= 30:
            scores['dcf'] = 100
        elif upside >= 15:
            scores['dcf'] = 80
        elif upside >= 0:
            scores['dcf'] = 60
        elif upside >= -15:
            scores['dcf'] = 40
        else:
            scores['dcf'] = 20
        
        # Calculate weighted composite score
        composite = sum(scores[k] * weights[k] for k in weights.keys())
        
        return composite, scores
    
    def get_recommendation(self, score: float) -> Recommendation:
        """Convert composite score to recommendation"""
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
    
    def generate_reasons(self, row: pd.Series, scores: Dict) -> List[str]:
        """Generate human-readable reasons for the recommendation"""
        reasons = []
        
        pvpa = row.get('pvpa_current', 1.0)
        if pd.notna(pvpa):
            if pvpa < 0.90:
                reasons.append(f"Trading at significant discount (P/VPA: {pvpa:.2f}x)")
            elif pvpa > 1.10:
                reasons.append(f"Trading at premium (P/VPA: {pvpa:.2f}x)")
        
        dy = row.get('dy_annualized', 0)
        if pd.notna(dy):
            if dy > self.RISK_FREE_RATE * 1.3:
                reasons.append(f"Attractive yield ({dy*100:.1f}% vs SELIC {self.RISK_FREE_RATE*100:.1f}%)")
            elif dy < self.RISK_FREE_RATE:
                reasons.append(f"Yield below risk-free rate ({dy*100:.1f}%)")
        
        return_ltm = row.get('return_ltm', 0)
        if pd.notna(return_ltm):
            if return_ltm > 0.15:
                reasons.append(f"Strong momentum (LTM return: {return_ltm*100:.1f}%)")
            elif return_ltm < -0.10:
                reasons.append(f"Negative momentum (LTM return: {return_ltm*100:.1f}%)")
        
        if scores.get('liquidity', 50) < 40:
            reasons.append("Low liquidity - exercise caution")
        
        return reasons if reasons else ["Neutral indicators"]
    
    def analyze(self) -> List[AnalysisResult]:
        """Run complete analysis on all FIIs from all segments"""
        print(f"\n{'='*60}")
        print("Running Comprehensive FII Analysis...")
        print(f"{'='*60}\n")
        
        self.results = []
        self.results_by_segment = {}
        
        for idx, row in self.df.iterrows():
            ticker = row.get('ticker', f'Unknown_{idx}')
            fund_name = row.get('fund_name', 'Unknown Fund')
            segment = row.get('segment', 'Unknown')
            
            if pd.isna(ticker) or ticker == '':
                continue
            
            # Calculate composite score
            composite_score, individual_scores = self.calculate_composite_score(row)
            
            # Get recommendation
            recommendation = self.get_recommendation(composite_score)
            
            # Calculate DCF upside
            dcf = self.calculate_dcf_valuation(row)
            
            # Generate reasons
            reasons = self.generate_reasons(row, individual_scores)
            
            result = AnalysisResult(
                ticker=str(ticker),
                fund_name=str(fund_name)[:40],  # Truncate long names
                recommendation=recommendation,
                score=composite_score,
                pvpa_score=individual_scores.get('pvpa', 0),
                yield_score=individual_scores.get('yield', 0),
                momentum_score=individual_scores.get('momentum', 0),
                liquidity_score=individual_scores.get('liquidity', 0),
                credit_score=individual_scores.get('credit', 0),
                dcf_upside=dcf.get('upside', 0),
                reasons=reasons,
                segment=segment
            )
            
            self.results.append(result)
            
            # Group by segment
            if segment not in self.results_by_segment:
                self.results_by_segment[segment] = []
            self.results_by_segment[segment].append(result)
        
        # Sort by composite score
        self.results.sort(key=lambda x: x.score, reverse=True)
        
        # Sort within each segment
        for segment in self.results_by_segment:
            self.results_by_segment[segment].sort(key=lambda x: x.score, reverse=True)
        
        return self.results
    
    def print_summary(self):
        """Print comprehensive analysis summary"""
        print("\n" + "="*110)
        print(" " * 40 + "FII ANALYSIS SUMMARY")
        print("="*110)
        
        # Statistics
        recommendations_count = {}
        for r in Recommendation:
            recommendations_count[r.value] = sum(1 for res in self.results if res.recommendation == r)
        
        print(f"\nTotal FIIs Analyzed: {len(self.results)}")
        print(f"Total Segments: {len(self.results_by_segment)}")
        print("\nOverall Recommendation Distribution:")
        for rec, count in recommendations_count.items():
            bar = "█" * count
            print(f"  {rec:12} : {count:3} {bar}")
        
        # Segment Summary
        print("\n" + "-"*110)
        print(" " * 35 + "SUMMARY BY SEGMENT")
        print("-"*110)
        print(f"{'Segment':<25} {'Total':>7} {'Strong Buy':>12} {'Buy':>7} {'Hold':>7} {'Sell':>7} {'Top Pick':<12} {'Score':>7}")
        print("-"*110)
        
        for segment, results in sorted(self.results_by_segment.items()):
            strong_buy = sum(1 for r in results if r.recommendation == Recommendation.STRONG_BUY)
            buy = sum(1 for r in results if r.recommendation == Recommendation.BUY)
            hold = sum(1 for r in results if r.recommendation == Recommendation.HOLD)
            sell = sum(1 for r in results if r.recommendation in [Recommendation.SELL, Recommendation.STRONG_SELL])
            top_pick = results[0].ticker if results else '-'
            top_score = results[0].score if results else 0
            
            print(f"{segment:<25} {len(results):>7} {strong_buy:>12} {buy:>7} {hold:>7} {sell:>7} {top_pick:<12} {top_score:>6.1f}")
        
        # Top 15 Picks Overall
        print("\n" + "-"*110)
        print(" " * 40 + "TOP 15 PICKS (ALL SEGMENTS)")
        print("-"*110)
        print(f"{'Ticker':<10} {'Fund Name':<30} {'Segment':<20} {'Score':>7} {'Rec':<12} {'P/VPA':>7} {'Yield':>7} {'DCF':>10}")
        print("-"*110)
        
        for result in self.results[:15]:
            segment = getattr(result, 'segment', 'Unknown')[:18]
            print(f"{result.ticker:<10} {result.fund_name[:28]:<30} {segment:<20} {result.score:>6.1f}  "
                  f"{result.recommendation.value:<12} {result.pvpa_score:>6.0f}  {result.yield_score:>6.0f}  "
                  f"{result.dcf_upside:>+9.1f}%")
            if result.reasons:
                print(f"           └─ {'; '.join(result.reasons[:2])}")
        
        # Results by Segment
        print("\n" + "="*110)
        print(" " * 35 + "DETAILED ANALYSIS BY SEGMENT")
        print("="*110)
        
        for segment, results in sorted(self.results_by_segment.items()):
            print(f"\n{'─'*110}")
            print(f" 📊 {segment.upper()} ({len(results)} FIIs)")
            print(f"{'─'*110}")
            print(f"{'Ticker':<10} {'Recommendation':<15} {'Score':>8} {'P/VPA':>8} {'Yield':>8} {'Momentum':>10} {'Liquidity':>10} {'Credit':>8}")
            print("-"*110)
            
            for result in results:
                rec_color = {
                    Recommendation.STRONG_BUY: "★★",
                    Recommendation.BUY: "★ ",
                    Recommendation.HOLD: "─ ",
                    Recommendation.SELL: "▼ ",
                    Recommendation.STRONG_SELL: "▼▼"
                }
                symbol = rec_color.get(result.recommendation, "  ")
                
                print(f"{result.ticker:<10} {symbol} {result.recommendation.value:<12} {result.score:>7.1f}  "
                      f"{result.pvpa_score:>7.0f}  {result.yield_score:>7.0f}  {result.momentum_score:>9.0f}  "
                      f"{result.liquidity_score:>9.0f}  {result.credit_score:>7.0f}")
        
        # Bottom Picks (potential sells)
        print("\n" + "-"*110)
        print(" " * 35 + "BOTTOM 10 (WEAKEST PERFORMERS)")
        print("-"*110)
        
        for result in self.results[-10:]:
            segment = getattr(result, 'segment', 'Unknown')[:18]
            print(f"{result.ticker:<10} {result.fund_name[:28]:<30} {segment:<20} {result.score:>6.1f}  "
                  f"{result.recommendation.value:<12}")
            if result.reasons:
                print(f"           └─ {'; '.join(result.reasons[:2])}")
        
        print("\n" + "="*110)
        print("SCORING METHODOLOGY:")
        print("-"*110)
        print("  • P/VPA Score (25%): Lower P/VPA = Higher score (value investing)")
        print("  • Yield Score (25%): Higher dividend yield spread over SELIC = Higher score")
        print("  • Momentum Score (15%): Based on LTM and YTD returns")
        print("  • Liquidity Score (15%): Higher daily volume = Higher score")
        print("  • Credit Score (10%): Based on market cap, liquidity, and valuation")
        print("  • DCF Score (10%): Based on Gordon Growth Model fair value vs current price")
        print("\nRECOMMENDATION THRESHOLDS:")
        print("  • STRONG BUY: Score ≥ 80  |  BUY: Score ≥ 65  |  HOLD: Score ≥ 45")
        print("  • SELL: Score ≥ 30  |  STRONG SELL: Score < 30")
        print("="*110 + "\n")


def main():
    """Main entry point"""
    import sys
    
    # Get file path from command line or use default
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Prompt for file path
        print("\n" + "="*60)
        print("FII (Fundos de Investimento Imobiliário) Analyzer")
        print("="*60)
        file_path = input("\nEnter the path to the XLSX file: ").strip()
        
        if not file_path:
            # Use sample data for demonstration
            print("\nNo file provided. Creating sample data for demonstration...")
            file_path = create_sample_data()
    
    try:
        # Initialize and run analyzer
        analyzer = FIIAnalyzer(file_path)
        analyzer.load_data()
        analyzer.analyze()
        analyzer.print_summary()
        
    except FileNotFoundError:
        print(f"\nError: File not found: {file_path}")
        print("Please provide a valid XLSX file path.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def create_sample_data() -> str:
    """Create sample XLSX file with FII data for demonstration"""
    sample_data = {
        'Código': ['AFHI11', 'ALZC11', 'ARRI11', 'ARXD11', 'BCRI11', 
                   'BTCI11', 'CACR11', 'CLIN11', 'CYCR11', 'DAMA11',
                   'DEVA11', 'EQIR11', 'EXES11', 'FLCR11', 'FYTO11'],
        'Fundo': ['AF Invest CRI Recebíveis FII', 'Aliança Crédito Imobiliário FII',
                  'Open K Ativos e Recebíveis FII', 'ARX Dover Recebíveis FII',
                  'Banestes Recebíveis Imobiliários FII', 'BTG Pactual Crédito Imobiliário FII',
                  'Cartesia Recebíveis Imobiliários FII', 'Clave Índices de Preços FII',
                  'Cyrela Crédito FII', 'Dama FII', 'Devant Recebíveis Imobiliários FII',
                  'EQI Recebíveis Imobiliários FII', 'Êxes FII',
                  'Faria Lima Capital Recebíveis FII', 'Fyto Recebíveis Imobiliários FII'],
        'Fechamento': [95.96, 7.74, 6.59, 7.69, 69.21, 9.25, 79.80, 89.99, 8.80, 6.99,
                       26.71, 8.46, 9.49, 95.46, 8.66],
        'Part. IFIX': [0.35, 0.00, 0.00, 0.00, 0.29, 0.62, 0.26, 0.26, 0.22, 0.00,
                       0.25, 0.00, 0.00, 0.00, 0.00],
        'Volume de Negociação': [651125, 410355, 211233, 62990, 498330, 1800584, 1529330,
                                  1210881, 584436, 35740, 635689, 111538, 142615, 142820, 155616],
        'Valor de Mercado': [437157103, 155188780, 136590752, 69295897, 433107390,
                             920570841, 385938655, 391165202, 321635116, 38703847,
                             375139493, 92292805, 133822808, 71771028, 132336820],
        'Valor Patrimonial': [430678521, 188174497, 174164421, 80927743, 529481781,
                              1002606708, 357460076, 425159593, 341626786, 45626139,
                              1373366960, 104549110, 137700933, 71254745, 151046129],
        'P/VPA Atual': [1.02, 0.82, 0.78, 0.86, 0.82, 0.92, 1.08, 0.92, 0.94, 0.85,
                        0.27, 0.88, 0.97, 1.01, 0.88],
        'P/VPA 2025': [1.01, 0.83, 0.83, 0.85, 0.80, 0.93, 1.06, 0.90, 0.94, 0.95,
                       0.27, 0.86, 0.97, 1.02, 0.86],
        'Dividend Yield LTM': [12.55, 16.39, 14.56, 14.04, 14.75, 12.55, 20.28, 13.75,
                               14.58, 13.87, 17.97, 14.19, 16.23, 13.98, 14.33],
        'Dividend Yield Anualizado': [12.63, 16.39, 14.04, 15.78, 12.58, 20.30, 12.67,
                                       14.45, 12.44, 17.97, 14.18, 16.44, 15.08, 15.24, 12.55],
        'Último Dividendo': [1.01, 0.09, 0.09, 0.91, 0.10, 1.35, 0.95, 0.11, 0.07,
                             0.40, 0.10, 0.13, 1.20, 0.11, 0.09],
        'Retorno no Mês': [0.18, -2.51, 0.39, 1.64, 0.08, 1.54, 3.36, 0.64, -9.49,
                           3.33, 2.55, 0.32, -0.03, 3.18, 0.18],
        'Retorno no Ano': [0.18, -2.51, 0.39, 1.64, 0.08, 1.54, 3.36, 0.64, -9.49,
                           4.33, 2.55, 0.32, -0.03, 3.18, 0.18],
        'Retorno LTM': [19.69, -1.91, 26.44, 35.91, 19.08, 11.74, 25.00, 26.63, -16.83,
                        11.74, 26.66, 17.55, 11.65, 33.79, 19.08]
    }
    
    df = pd.DataFrame(sample_data)
    
    # Convert percentages to proper format
    pct_cols = ['Part. IFIX', 'Dividend Yield LTM', 'Dividend Yield Anualizado',
                'Retorno no Mês', 'Retorno no Ano', 'Retorno LTM']
    for col in pct_cols:
        df[col] = df[col] / 100
    
    sample_file = 'sample_fii_data.xlsx'
    df.to_excel(sample_file, index=False)
    print(f"Created sample file: {sample_file}")
    
    return sample_file


if __name__ == "__main__":
    main()
