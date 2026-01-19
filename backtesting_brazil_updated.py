# backtesting_brazil_updated.py
"""
Stress Testing and Backtesting for Brazilian Market
Updated to use dynamic configuration and shared data manager
Uses IFIX benchmark for FIIs and IBOV for stocks
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

from config import get_config, ConfigManager
from data_manager import get_data_manager, DataManager


class BrazilianStressTesting:
    def __init__(
        self, 
        ticker: str = None, 
        initial_investment: float = 1000000,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        if ticker is None:
            ticker = self.config.get_default_ticker()
        
        # Handle index tickers
        if ticker.startswith('^'):
            self.ticker = ticker
        else:
            self.ticker = self.config.normalize_ticker(ticker)
        
        self.initial_investment = initial_investment
        self.historical_data = None
        self.returns = None
        
        # Get appropriate benchmark based on ticker type
        self.benchmark = self.config.get_benchmark_for_ticker(self.ticker)
        
    def fetch_data(self, start_date: str = '2020-01-01') -> bool:
        print(f"Fetching data for {self.ticker}...")
        
        try:
            data = self.data_manager.fetch_historical_data(
                self.ticker, start_date=start_date, period='5y'
            )
            
            if data is None or data.empty:
                print(f"No data for {self.ticker}")
                return False
            
            self.historical_data = data
            self.returns = data['Returns'].dropna()
            
            print(f"Loaded {len(data)} records from {data.index.min()} to {data.index.max()}")
            return True
            
        except Exception as e:
            print(f"Error: {e}")
            return False
    
    def historical_stress_test(self, stress_periods: Dict = None) -> Optional[pd.DataFrame]:
        if self.historical_data is None:
            self.fetch_data()
        
        if stress_periods is None:
            stress_periods = {
                'COVID-19 Crash': ('2020-02-20', '2020-03-23'),
                'Brazil Political 2021': ('2021-09-01', '2021-12-01'),
                'Rate Hike 2022': ('2022-01-01', '2022-06-30'),
                'Fiscal Concerns 2023': ('2023-03-01', '2023-05-31'),
            }
        
        results = []
        
        for period_name, (start_date, end_date) in stress_periods.items():
            try:
                start = pd.to_datetime(start_date)
                end = pd.to_datetime(end_date)
                
                mask = (self.historical_data.index >= start) & (self.historical_data.index <= end)
                period_data = self.historical_data.loc[mask]
                
                if len(period_data) < 5:
                    continue
                
                period_returns = period_data['Returns'].dropna()
                cum_return = (1 + period_returns).prod() - 1
                
                results.append({
                    'Period': period_name,
                    'Start': start_date,
                    'End': end_date,
                    'Days': len(period_data),
                    'Cumulative_Return': cum_return * 100,
                    'Max_Drawdown': period_returns.min() * 100,
                    'Volatility': period_returns.std() * np.sqrt(252) * 100,
                    'Loss_BRL': self.initial_investment * cum_return
                })
            except Exception:
                continue
        
        return pd.DataFrame(results) if results else None
    
    def scenario_stress_test(self, scenarios: Dict = None) -> Optional[pd.DataFrame]:
        if scenarios is None:
            scenarios = {
                'Base Case': {'shock': 0, 'vol_mult': 1.0},
                'Mild Stress (-10%)': {'shock': -0.10, 'vol_mult': 1.3},
                'Moderate Stress (-20%)': {'shock': -0.20, 'vol_mult': 1.5},
                'Severe Stress (-30%)': {'shock': -0.30, 'vol_mult': 2.0},
                'Extreme Crisis (-50%)': {'shock': -0.50, 'vol_mult': 3.0},
                'SELIC Spike': {'shock': -0.15, 'vol_mult': 1.8},
                'FX Crisis (BRL)': {'shock': -0.25, 'vol_mult': 2.0},
            }
        
        if self.returns is None:
            self.fetch_data()
        
        base_vol = self.returns.std() * np.sqrt(252)
        
        results = []
        
        for scenario_name, params in scenarios.items():
            shock = params.get('shock', 0)
            vol_mult = params.get('vol_mult', 1.0)
            
            stressed_value = self.initial_investment * (1 + shock)
            stressed_vol = base_vol * vol_mult
            var_95 = stressed_value * 1.645 * (stressed_vol / np.sqrt(252))
            
            results.append({
                'Scenario': scenario_name,
                'Shock': f"{shock*100:.0f}%",
                'Vol_Multiplier': f"{vol_mult:.1f}x",
                'Portfolio_Value': stressed_value,
                'Loss_BRL': self.initial_investment - stressed_value,
                'VaR_95_Daily': var_95,
                'Stressed_Vol': f"{stressed_vol*100:.1f}%"
            })
        
        return pd.DataFrame(results)
    
    def backtest_strategy(self, strategy: str = 'buy_hold', benchmark: str = None) -> Optional[Dict]:
        if self.historical_data is None:
            self.fetch_data()
        
        # Use appropriate benchmark
        if benchmark is None:
            benchmark = self.benchmark
        
        print(f"Using benchmark: {benchmark}")
        
        # Fetch benchmark data
        try:
            bench_data = self.data_manager.fetch_historical_data(benchmark, period='5y')
            if bench_data is not None and not bench_data.empty:
                bench_returns = bench_data['Returns'].dropna()
            else:
                bench_returns = None
        except Exception:
            bench_returns = None
        
        portfolio_returns = self.returns.copy()
        
        if strategy == 'buy_hold':
            strategy_returns = portfolio_returns
        elif strategy == 'momentum':
            lookback = 20
            signal = portfolio_returns.rolling(lookback).mean() > 0
            strategy_returns = portfolio_returns * signal.shift(1).fillna(0)
        elif strategy == 'mean_reversion':
            lookback = 20
            z_score = (portfolio_returns - portfolio_returns.rolling(lookback).mean()) / portfolio_returns.rolling(lookback).std()
            signal = z_score < -1
            strategy_returns = portfolio_returns * signal.shift(1).fillna(0)
        else:
            strategy_returns = portfolio_returns
        
        # Calculate metrics
        cum_return = (1 + strategy_returns).prod() - 1
        annual_return = (1 + cum_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        
        # Max drawdown
        cum_values = (1 + strategy_returns).cumprod()
        rolling_max = cum_values.expanding().max()
        drawdowns = cum_values / rolling_max - 1
        max_drawdown = drawdowns.min()
        
        results = {
            'strategy': strategy,
            'ticker': self.ticker,
            'benchmark': benchmark,
            'total_return': cum_return * 100,
            'annual_return': annual_return * 100,
            'volatility': volatility * 100,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown * 100,
            'final_value': self.initial_investment * (1 + cum_return)
        }
        
        # Add benchmark comparison if available
        if bench_returns is not None:
            aligned = pd.DataFrame({'strategy': strategy_returns, 'benchmark': bench_returns}).dropna()
            if not aligned.empty:
                bench_cum = (1 + aligned['benchmark']).prod() - 1
                results['benchmark_return'] = bench_cum * 100
                results['alpha'] = results['total_return'] - results['benchmark_return']
        
        return results
    
    def calculate_var_backtest(self, confidence_level: float = 0.95, window: int = 252) -> Dict:
        if self.returns is None:
            self.fetch_data()
        
        if len(self.returns) < window + 20:
            window = max(20, len(self.returns) // 2)
        
        var_predictions = []
        actual_returns = []
        
        for i in range(window, len(self.returns)):
            historical = self.returns.iloc[i-window:i]
            var = np.percentile(historical, (1 - confidence_level) * 100)
            var_predictions.append(var)
            actual_returns.append(self.returns.iloc[i])
        
        var_predictions = np.array(var_predictions)
        actual_returns = np.array(actual_returns)
        
        breaches = actual_returns < var_predictions
        breach_rate = breaches.mean()
        expected_rate = 1 - confidence_level
        
        return {
            'confidence_level': confidence_level,
            'expected_breach_rate': expected_rate * 100,
            'actual_breach_rate': breach_rate * 100,
            'total_breaches': breaches.sum(),
            'total_observations': len(breaches),
            'model_accuracy': 'Good' if abs(breach_rate - expected_rate) < 0.02 else 'Needs Review'
        }
    
    def plot_stress_test_results(self, save_path: str = None):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Price history
        ax1 = axes[0, 0]
        if self.historical_data is not None:
            ax1.plot(self.historical_data.index, self.historical_data['Close'], 'b-', linewidth=1)
            ax1.set_title(f'{self.ticker} Price History')
            ax1.set_ylabel('Price (BRL)')
            ax1.grid(True, alpha=0.3)
        
        # 2. Scenario stress test
        ax2 = axes[0, 1]
        scenarios = self.scenario_stress_test()
        if scenarios is not None:
            colors = ['green' if v >= self.initial_investment else 'red' 
                     for v in scenarios['Portfolio_Value']]
            ax2.barh(scenarios['Scenario'], scenarios['Portfolio_Value'], color=colors, alpha=0.7)
            ax2.axvline(x=self.initial_investment, color='black', linestyle='--', label='Initial')
            ax2.set_xlabel('Portfolio Value (BRL)')
            ax2.set_title('Scenario Stress Test')
            ax2.legend()
        
        # 3. Backtest results
        ax3 = axes[1, 0]
        bt_result = self.backtest_strategy()
        if bt_result:
            metrics = ['total_return', 'annual_return', 'volatility', 'max_drawdown']
            values = [bt_result.get(m, 0) for m in metrics]
            labels = ['Total Ret%', 'Annual Ret%', 'Volatility%', 'Max DD%']
            colors = ['green' if v > 0 and m != 'max_drawdown' else 'red' if m == 'max_drawdown' else 'orange' 
                     for m, v in zip(metrics, values)]
            ax3.bar(labels, values, color=colors, alpha=0.7)
            ax3.set_title(f'Backtest: {bt_result["strategy"]} (vs {self.benchmark})')
            ax3.axhline(y=0, color='black', linewidth=0.5)
            ax3.grid(True, alpha=0.3)
        
        # 4. VaR backtest
        ax4 = axes[1, 1]
        var_bt = self.calculate_var_backtest()
        if var_bt:
            labels = ['Expected', 'Actual']
            values = [var_bt['expected_breach_rate'], var_bt['actual_breach_rate']]
            colors = ['blue', 'green' if var_bt['model_accuracy'] == 'Good' else 'red']
            ax4.bar(labels, values, color=colors, alpha=0.7)
            ax4.set_ylabel('Breach Rate (%)')
            ax4.set_title(f'VaR Backtest - {var_bt["model_accuracy"]}')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Stress Testing: {self.ticker} (Benchmark: {self.benchmark})', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        # Print summary
        print("\n" + "="*70)
        print("STRESS TEST SUMMARY")
        print("="*70)
        print(f"Ticker: {self.ticker}")
        print(f"Benchmark: {self.benchmark}")
        print(f"Initial Investment: R$ {self.initial_investment:,.2f}")
        
        if bt_result:
            print(f"\nBacktest Results ({bt_result['strategy']}):")
            print(f"  Total Return: {bt_result['total_return']:.2f}%")
            print(f"  Annual Return: {bt_result['annual_return']:.2f}%")
            print(f"  Volatility: {bt_result['volatility']:.2f}%")
            print(f"  Sharpe Ratio: {bt_result['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {bt_result['max_drawdown']:.2f}%")
            if 'benchmark_return' in bt_result:
                print(f"  Benchmark Return: {bt_result['benchmark_return']:.2f}%")
                print(f"  Alpha: {bt_result['alpha']:.2f}%")
        
        return fig


def analyze_sector_stress(sector: str = None, config: ConfigManager = None) -> pd.DataFrame:
    config = config or get_config()
    
    if sector is None:
        tickers = config.get_tickers()[:5]  # Limit for performance
    else:
        tickers = config.get_sector_tickers(sector)
    
    results = []
    
    for ticker in tickers:
        try:
            stress = BrazilianStressTesting(ticker=ticker, initial_investment=100000)
            stress.fetch_data()
            bt = stress.backtest_strategy()
            
            if bt:
                results.append({
                    'Ticker': ticker,
                    'Benchmark': stress.benchmark,
                    'Total_Return': bt['total_return'],
                    'Volatility': bt['volatility'],
                    'Sharpe': bt['sharpe_ratio'],
                    'Max_DD': bt['max_drawdown']
                })
        except Exception as e:
            print(f"Error with {ticker}: {e}")
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    config = get_config()
    
    print("="*70)
    print("BRAZILIAN STRESS TESTING & BACKTESTING")
    print("="*70)
    
    # Test with default ticker (FII)
    stress_tester = BrazilianStressTesting(initial_investment=100_000)
    stress_tester.plot_stress_test_results()