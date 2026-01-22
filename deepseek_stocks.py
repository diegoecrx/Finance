import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def analyze_stocks_top10(file_path):
    """
    Analyze stocks and return top 10 ranked stocks based on composite scoring
    """
    # Read the Excel file
    df = pd.read_excel(file_path, sheet_name='statusinvest-stocks.xlsx')
    
    # Clean column names
    df.columns = [col.strip() for col in df.columns]
    
    # Create a copy for calculations
    df_clean = df.copy()
    
    # Handle missing/invalid values
    for col in df_clean.columns[1:]:  # Skip TICKER
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
    
    # ========================
    # 1. DATA CLEANING & PREPARATION
    # ========================
    
    # Handle infinite values and extreme outliers for key metrics
    key_metrics = ['P/L', 'P/VP', 'DY', 'ROE', 'ROIC', 'P/EBIT', 'EV/EBIT', 
                   'MARG. LIQUIDA', 'CAGR RECEITAS 5 ANOS', 'CAGR LUCROS 5 ANOS',
                   'DIV. LIQ. / PATRI.', 'LIQ. CORRENTE', 'PATRIMONIO / ATIVOS']
    
    for col in key_metrics:
        if col in df_clean.columns:
            # Replace infinities with NaN
            df_clean[col] = df_clean[col].replace([np.inf, -np.inf], np.nan)
            
            # Remove extreme outliers (keep 5th to 95th percentile)
            q_low = df_clean[col].quantile(0.05)
            q_high = df_clean[col].quantile(0.95)
            df_clean[col] = df_clean[col].clip(q_low, q_high)
    
    # Fill remaining NaN with median values
    for col in df_clean.columns[1:]:
        if df_clean[col].isnull().any():
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # ========================
    # 2. NORMALIZATION FUNCTION
    # ========================
    def normalize_series(series, inverse=False, higher_better=True):
        """
        Normalize a series with options for inverse relationship
        """
        if series.isnull().all():
            return np.zeros(len(series))
        
        if inverse and higher_better:
            # For metrics where lower is better (P/L, P/VP, etc.)
            series_clean = series.replace(0, np.nan)  # Avoid division by zero
            series_inv = 1 / series_clean
            series_inv = series_inv.fillna(0)
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(series_inv.values.reshape(-1, 1))
        elif not higher_better:
            # For metrics where lower is better and we don't want inverse
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(series.values.reshape(-1, 1))
            normalized = 1 - normalized  # Flip so higher is better
        else:
            # For metrics where higher is better (DY, ROE, etc.)
            scaler = MinMaxScaler()
            normalized = scaler.fit_transform(series.values.reshape(-1, 1))
        
        return normalized.flatten()
    
    # ========================
    # 3. CALCULATE COMPONENT SCORES
    # ========================
    
    # Valuation Score (lower is better for these metrics)
    valuation_metrics = {
        'P/L': 0.25,
        'P/VP': 0.25,
        'P/EBIT': 0.20,
        'EV/EBIT': 0.20,
        'PSR': 0.10
    }
    
    valuation_score = np.zeros(len(df_clean))
    for metric, weight in valuation_metrics.items():
        if metric in df_clean.columns:
            norm_score = normalize_series(df_clean[metric], inverse=True, higher_better=True)
            valuation_score += norm_score * weight
    
    df_clean['Valuation_Score'] = valuation_score
    
    # Profitability Score (higher is better)
    profitability_metrics = {
        'ROE': 0.30,
        'ROIC': 0.30,
        'MARG. LIQUIDA': 0.20,
        'ROA': 0.10,
        'MARGEM BRUTA': 0.10
    }
    
    profitability_score = np.zeros(len(df_clean))
    for metric, weight in profitability_metrics.items():
        if metric in df_clean.columns:
            norm_score = normalize_series(df_clean[metric], inverse=False, higher_better=True)
            profitability_score += norm_score * weight
    
    df_clean['Profitability_Score'] = profitability_score
    
    # Dividend Score
    if 'DY' in df_clean.columns:
        df_clean['Dividend_Score'] = normalize_series(df_clean['DY'], inverse=False, higher_better=True)
    else:
        df_clean['Dividend_Score'] = 0
    
    # Growth Score
    growth_metrics = {
        'CAGR RECEITAS 5 ANOS': 0.50,
        'CAGR LUCROS 5 ANOS': 0.50
    }
    
    growth_score = np.zeros(len(df_clean))
    for metric, weight in growth_metrics.items():
        if metric in df_clean.columns:
            norm_score = normalize_series(df_clean[metric], inverse=False, higher_better=True)
            growth_score += norm_score * weight
    
    df_clean['Growth_Score'] = growth_score
    
    # Financial Health Score
    health_metrics = {
        'DIV. LIQ. / PATRI.': 0.30,  # Lower is better (inverse)
        'LIQ. CORRENTE': 0.25,       # Higher is better
        'PATRIMONIO / ATIVOS': 0.20, # Higher is better
        'DIVIDA LIQUIDA / EBIT': 0.15, # Lower is better (inverse)
        'PASSIVOS / ATIVOS': 0.10    # Lower is better (inverse)
    }
    
    health_score = np.zeros(len(df_clean))
    for metric, weight in health_metrics.items():
        if metric in df_clean.columns:
            if metric in ['DIV. LIQ. / PATRI.', 'DIVIDA LIQUIDA / EBIT', 'PASSIVOS / ATIVOS']:
                # For debt metrics, use absolute value and inverse
                norm_score = normalize_series(df_clean[metric].abs(), inverse=True, higher_better=True)
            else:
                norm_score = normalize_series(df_clean[metric], inverse=False, higher_better=True)
            health_score += norm_score * weight
    
    df_clean['Health_Score'] = health_score
    
    # ========================
    # 4. COMPOSITE SCORE (Adjusted Weights)
    # ========================
    weights_composite = {
        'Valuation_Score': 0.25,      # 25% - Value investing focus
        'Profitability_Score': 0.25,  # 25% - Profitability is key
        'Dividend_Score': 0.20,       # 20% - Income generation
        'Health_Score': 0.15,         # 15% - Financial stability
        'Growth_Score': 0.15          # 15% - Future potential
    }
    
    composite_score = np.zeros(len(df_clean))
    for score_name, weight in weights_composite.items():
        composite_score += df_clean[score_name] * weight
    
    df_clean['Composite_Score'] = composite_score
    
    # ========================
    # 5. QUALITY FILTERS (More Flexible for Top 10)
    # ========================
    filtered_df = df_clean.copy()
    
    # Apply basic quality filters (less restrictive for top 10)
    filters_passed = pd.Series(True, index=filtered_df.index)
    
    # Minimum dividend yield (4% instead of 5% for more options)
    if 'DY' in filtered_df.columns:
        filters_passed = filters_passed & (filtered_df['DY'] >= 4)
    
    # Reasonable P/L (allow up to 20 for growth companies)
    if 'P/L' in filtered_df.columns:
        filters_passed = filters_passed & (filtered_df['P/L'] <= 25)
        filters_passed = filters_passed & (filtered_df['P/L'] > 0)  # Positive earnings
    
    # Minimum ROE (8% instead of 10%)
    if 'ROE' in filtered_df.columns:
        filters_passed = filters_passed & (filtered_df['ROE'] >= 8)
    
    # Debt control (more flexible for top 10)
    if 'DIV. LIQ. / PATRI.' in filtered_df.columns:
        filters_passed = filters_passed & (filtered_df['DIV. LIQ. / PATRI.'] <= 2)
    
    # Minimum liquidity
    if 'LIQUIDEZ MEDIA DIARIA' in filtered_df.columns:
        filters_passed = filters_passed & (filtered_df['LIQUIDEZ MEDIA DIARIA'] >= 500000)
    
    # Positive equity
    if 'P/VP' in filtered_df.columns:
        filters_passed = filters_passed & (filtered_df['P/VP'] > 0.1)  # Some equity value
    
    # Apply all filters
    filtered_df = filtered_df[filters_passed]
    
    # ========================
    # 6. FINAL RANKING (TOP 10)
    # ========================
    
    # Calculate enhanced final score with tie-breakers
    def calculate_enhanced_score(row):
        base_score = row['Composite_Score']
        
        # Tie-breaker adjustments (small weights to break ties)
        tie_adjustment = 0
        
        # Favor higher dividend yield (1% = 0.01 adjustment)
        if 'DY' in row:
            tie_adjustment += (row['DY'] / 100) * 0.05
        
        # Favor lower P/L ratio
        if 'P/L' in row and row['P/L'] > 0:
            tie_adjustment += (1 / row['P/L']) * 0.03
        
        # Favor higher ROIC
        if 'ROIC' in row:
            tie_adjustment += (row['ROIC'] / 100) * 0.02
        
        # Favorable debt position
        if 'DIV. LIQ. / PATRI.' in row:
            debt_ratio = abs(row['DIV. LIQ. / PATRI.'])
            if debt_ratio <= 0.5:
                tie_adjustment += 0.02  # Bonus for low debt
            elif debt_ratio > 2:
                tie_adjustment -= 0.01  # Penalty for high debt
        
        # Bonus for growth consistency
        if 'CAGR LUCROS 5 ANOS' in row and row['CAGR LUCROS 5 ANOS'] > 10:
            tie_adjustment += 0.01
        
        return base_score + tie_adjustment
    
    filtered_df['Enhanced_Score'] = filtered_df.apply(calculate_enhanced_score, axis=1)
    
    # Sort by enhanced score and get top 10
    top_10 = filtered_df.sort_values('Enhanced_Score', ascending=False).head(10).copy()
    top_10['Rank'] = range(1, len(top_10) + 1)
    
    # ========================
    # 7. CATEGORIZE STOCKS
    # ========================
    
    def categorize_stock(row):
        """Categorize stock based on characteristics"""
        categories = []
        
        if 'DY' in row and row['DY'] >= 8:
            categories.append('High Dividend')
        elif 'DY' in row and row['DY'] >= 5:
            categories.append('Dividend')
        
        if 'P/L' in row and row['P/L'] <= 10:
            categories.append('Value')
        elif 'P/L' in row and row['P/L'] <= 15:
            categories.append('Fair Value')
        
        if 'ROE' in row and row['ROE'] >= 15:
            categories.append('High ROE')
        
        if 'CAGR LUCROS 5 ANOS' in row and row['CAGR LUCROS 5 ANOS'] >= 15:
            categories.append('Growth')
        
        if 'DIV. LIQ. / PATRI.' in row and row['DIV. LIQ. / PATRI.'] <= 0.5:
            categories.append('Low Debt')
        
        return ', '.join(categories) if categories else 'Balanced'
    
    top_10['Category'] = top_10.apply(categorize_stock, axis=1)
    
    # ========================
    # 8. PREPARE OUTPUT
    # ========================
    
    # Select columns for final output
    output_columns = [
        'Rank', 'TICKER', 'PRECO', 'DY', 'P/L', 'P/VP', 'ROE', 'ROIC',
        'Composite_Score', 'Enhanced_Score', 'Category',
        'MARG. LIQUIDA', 'CAGR LUCROS 5 ANOS', 'DIV. LIQ. / PATRI.',
        'LIQUIDEZ MEDIA DIARIA', 'VALOR DE MERCADO'
    ]
    
    # Filter to available columns
    available_columns = [col for col in output_columns if col in top_10.columns]
    results_df = top_10[available_columns].copy()
    
    # Format numeric columns
    if 'VALOR DE MERCADO' in results_df.columns:
        results_df['VALOR DE MERCADO'] = results_df['VALOR DE MERCADO'].apply(
            lambda x: f'R${x:,.0f}' if pd.notnull(x) else 'N/A'
        )
    
    if 'LIQUIDEZ MEDIA DIARIA' in results_df.columns:
        results_df['LIQUIDEZ MEDIA DIARIA'] = results_df['LIQUIDEZ MEDIA DIARIA'].apply(
            lambda x: f'{x:,.0f}' if pd.notnull(x) else 'N/A'
        )
    
    # Round numeric columns
    float_columns = results_df.select_dtypes(include=[np.float64]).columns
    results_df[float_columns] = results_df[float_columns].round(3)
    
    return results_df, df_clean, filtered_df

# ========================
# 9. VISUALIZATION FUNCTION
# ========================
def visualize_results(results_df):
    """
    Create visualizations for the top 10 stocks
    """
    import matplotlib.pyplot as plt
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # 1. Composite Score by Ticker
    axes[0, 0].bar(results_df['TICKER'], results_df['Composite_Score'], color='skyblue')
    axes[0, 0].set_title('Composite Score - Top 10 Stocks')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # 2. Dividend Yield Comparison
    axes[0, 1].bar(results_df['TICKER'], results_df['DY'], color='green', alpha=0.7)
    axes[0, 1].axhline(y=5, color='r', linestyle='--', alpha=0.5, label='5% Threshold')
    axes[0, 1].set_title('Dividend Yield (%)')
    axes[0, 1].set_ylabel('DY %')
    axes[0, 1].legend()
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 3. P/L vs ROE Scatter
    scatter = axes[1, 0].scatter(results_df['P/L'], results_df['ROE'], 
                                c=results_df['Composite_Score'], s=100, cmap='viridis')
    axes[1, 0].set_title('P/L vs ROE (color=Composite Score)')
    axes[1, 0].set_xlabel('P/L Ratio')
    axes[1, 0].set_ylabel('ROE %')
    plt.colorbar(scatter, ax=axes[1, 0])
    
    # 4. Category Distribution
    category_counts = results_df['Category'].str.split(', ').explode().value_counts()
    axes[1, 1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Investment Categories in Top 10')
    
    plt.tight_layout()
    plt.savefig('top10_stocks_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

# ========================
# 10. MAIN EXECUTION
# ========================
if __name__ == "__main__":
    # Example usage
    print("="*80)
    print("STOCK ANALYSIS SYSTEM - TOP 10 RANKING")
    print("="*80)
    
    print("\nTo use this script:")
    print("1. Save your Excel file as 'statusinvest-stocks.xlsx'")
    print("2. Ensure it has the correct sheet name")
    print("3. Run: results, full_df, filtered = analyze_stocks_top10('your_file.xlsx')")
    
    print("\n" + "="*80)
    print("SAMPLE OUTPUT FORMAT (Top 10 Stocks):")
    print("="*80)
    
    # Create sample output
    sample_data = {
        'Rank': range(1, 11),
        'TICKER': ['GRND3', 'BALM4', 'SOND3', 'VULC3', 'RECV3', 
                  'TRIS3', 'POMO4', 'CSUD3', 'EZTC3', 'BLAU3'],
        'PRECO': [4.55, 17.72, 38.66, 18.06, 10.60, 6.12, 5.97, 17.45, 13.85, 8.80],
        'DY': [37.43, 26.18, 36.89, 30.20, 18.15, 21.01, 21.11, 15.97, 11.22, 11.64],
        'P/L': [5.66, 7.02, 3.42, 4.86, 5.02, 6.96, 6.20, 7.74, 7.15, 6.48],
        'P/VP': [1.01, 0.95, 1.31, 2.28, 0.68, 0.94, 1.72, 1.43, 0.76, 0.87],
        'ROE': [17.92, 13.52, 38.39, 46.97, 13.59, 13.44, 27.78, 18.49, 10.60, 13.47],
        'ROIC': [9.10, 19.28, 20.76, 11.49, 9.43, 6.72, 12.81, 13.85, 6.18, 7.66],
        'Composite_Score': [0.85, 0.82, 0.80, 0.78, 0.76, 0.74, 0.72, 0.70, 0.68, 0.66],
        'Enhanced_Score': [0.872, 0.841, 0.823, 0.801, 0.785, 0.769, 0.751, 0.732, 0.714, 0.698],
        'Category': ['High Dividend, Value', 'High Dividend, Value', 
                    'High Dividend, Value, High ROE, Growth', 'High Dividend, High ROE',
                    'Dividend, Value', 'High Dividend, Fair Value',
                    'High Dividend, Fair Value, High ROE', 'Dividend, Fair Value',
                    'Dividend, Fair Value', 'Dividend, Fair Value'],
        'MARG. LIQUIDA': [26.50, 11.28, 12.59, 34.01, 18.81, 15.71, 13.14, 15.58, 32.85, 17.89],
        'CAGR LUCROS 5 ANOS': [7.96, 31.66, 44.65, 52.39, 57.65, 8.77, 42.97, 28.57, 14.12, 9.59],
        'DIV. LIQ. / PATRI.': [-0.25, -0.39, -0.48, 0.18, 0.32, 0.29, 0.29, -0.10, 0.03, 0.10],
        'LIQUIDEZ MEDIA DIARIA': ['19,588,004', '28,823', '3,866', '22,801,400', 
                                 '47,343,333', '2,093,494', '57,518,586', 
                                 '2,227,186', '28,119,660', '6,417,717'],
        'VALOR DE MERCADO': ['R$4,104,828,000', 'R$233,828,000', 'R$138,161,581',
                            'R$5,716,029,190', 'R$3,110,804,536', 'R$1,484,819,333',
                            'R$7,272,497,347', 'R$729,410,000', 'R$3,891,850,000',
                            'R$2,052,266,664']
    }
    
    sample_df = pd.DataFrame(sample_data)
    print(sample_df.to_string(index=False))
    
    print("\n" + "="*80)
    print("ANALYSIS SUMMARY:")
    print("="*80)
    print("✓ Top 10 stocks selected based on composite scoring")
    print("✓ Minimum filters: DY ≥ 4%, P/L ≤ 25, ROE ≥ 8%")
    print("✓ Enhanced scoring with tie-breakers")
    print("✓ Investment categorization for each stock")
    print("✓ Visualizations available with visualize_results() function")
    
    print("\nKey Metrics Considered:")
    print("- Valuation: P/L, P/VP, P/EBIT, EV/EBIT")
    print("- Profitability: ROE, ROIC, Margins")
    print("- Dividends: DY")
    print("- Growth: CAGR Receitas & Lucros")
    print("- Financial Health: Debt ratios, Liquidity, Equity")
    
    print("\nTo get full analysis:")
    print("1. results, all_data, filtered = analyze_stocks_top10('your_file.xlsx')")
    print("2. visualize_results(results)")
    print("3. all_data.to_excel('full_analysis.xlsx')")