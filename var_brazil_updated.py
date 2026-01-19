# var_brazil_updated.py
"""
Value at Risk Analysis for Brazilian Market
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


class BrazilianVaR:
    def __init__(
        self, 
        portfolio: Dict[str, float] = None, 
        portfolio_value: float = 10000000,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        """
        Value at Risk Analysis for Brazilian Portfolio
        
        Parameters:
        -----------
        portfolio : Dict[str, float]
            Portfolio weights. If None, loads from config file.
        portfolio_value : float
            Total portfolio value in BRL
        config : ConfigManager
            Configuration manager instance
        data_manager : DataManager
            Data manager instance for fetching/caching data
        """
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        if portfolio is None:
            self.portfolio = self.config.get_portfolio()
        else:
            # Normalize ticker names
            self.portfolio = {
                self.config.normalize_ticker(k): v 
                for k, v in portfolio.items()
            }
        
        self.portfolio_value = portfolio_value
        self.returns_data = None
        self.cov_matrix = None
        
    def fetch_historical_data(self, period: str = '2y') -> bool:
        """Fetch historical data using data manager"""
        print("Fetching historical data...")
        
        tickers = list(self.portfolio.keys())
        self.returns_data = self.data_manager.get_returns_matrix(tickers, period=period)
        
        if self.returns_data is None or self.returns_data.empty:
            print("Failed to fetch returns data")
            return False
        
        self.cov_matrix = self.returns_data.cov()
        
        print(f"Successfully loaded data for {len(self.returns_data.columns)} assets")
        print(f"Date range: {self.returns_data.index.min()} to {self.returns_data.index.max()}")
        
        return True
        
    def calculate_var(
        self, 
        method: str = 'parametric', 
        confidence_level: float = 0.95, 
        time_horizon: int = 1
    ) -> Dict:
        """
        Calculate Value at Risk using different methods
        
        Parameters:
        -----------
        method : str
            'historical', 'parametric', or 'monte_carlo'
        confidence_level : float
            VaR confidence level (e.g., 0.95 for 95%)
        time_horizon : int
            Time horizon in days
            
        Returns:
        --------
        Dict with VaR results
        """
        if self.returns_data is None:
            if not self.fetch_historical_data():
                return None
        
        portfolio_weights = np.array(list(self.portfolio.values()))
        
        # Ensure weights are aligned with returns columns
        aligned_weights = []
        for col in self.returns_data.columns:
            if col in self.portfolio:
                aligned_weights.append(self.portfolio[col])
            else:
                aligned_weights.append(0)
        portfolio_weights = np.array(aligned_weights)
        
        # Calculate portfolio returns
        portfolio_returns = self.returns_data.dot(portfolio_weights)
        
        # Initialize variables
        portfolio_std = None
        portfolio_mean = None
        
        if method == 'historical':
            method_name = "Historical Simulation"
            var = np.percentile(portfolio_returns.dropna(), (1 - confidence_level) * 100)
            var = abs(var)
            
        elif method == 'parametric':
            method_name = "Parametric (Variance-Covariance)"
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            
            z_score = stats.norm.ppf(1 - confidence_level)
            var = abs(z_score * portfolio_std)
            
        elif method == 'monte_carlo':
            method_name = "Monte Carlo Simulation"
            n_simulations = 10000
            
            portfolio_mean = portfolio_returns.mean()
            portfolio_std = portfolio_returns.std()
            
            # Simulate returns
            np.random.seed(42)
            simulated_returns = np.random.normal(
                portfolio_mean, portfolio_std, n_simulations
            )
            
            var = abs(np.percentile(simulated_returns, (1 - confidence_level) * 100))
            
        else:
            raise ValueError(f"Unknown VaR method: {method}")
        
        # Scale to portfolio value and time horizon
        var_daily = var * self.portfolio_value
        var_scaled = var_daily * np.sqrt(time_horizon)
        
        # Calculate Conditional VaR (Expected Shortfall)
        if method == 'historical':
            threshold = np.percentile(portfolio_returns.dropna(), (1 - confidence_level) * 100)
            cvar = abs(portfolio_returns[portfolio_returns <= threshold].mean())
        elif method == 'monte_carlo':
            threshold = np.percentile(simulated_returns, (1 - confidence_level) * 100)
            cvar = abs(simulated_returns[simulated_returns <= threshold].mean())
        else:  # parametric
            z_score = stats.norm.ppf(1 - confidence_level)
            cvar = abs(portfolio_std * stats.norm.pdf(z_score) / (1 - confidence_level))
        
        cvar_daily = cvar * self.portfolio_value
        cvar_scaled = cvar_daily * np.sqrt(time_horizon)
        
        # Calculate portfolio volatility (annualized)
        if portfolio_std is not None:
            portfolio_volatility = portfolio_std * np.sqrt(252)
        else:
            portfolio_volatility = portfolio_returns.std() * np.sqrt(252)
        
        results = {
            'method': method_name,
            'confidence_level': confidence_level,
            'time_horizon_days': time_horizon,
            'var_daily_pct': var * 100,
            'var_daily_brl': var_daily,
            'var_scaled_brl': var_scaled,
            'cvar_daily_pct': cvar * 100,
            'cvar_daily_brl': cvar_daily,
            'cvar_scaled_brl': cvar_scaled,
            'portfolio_volatility': portfolio_volatility
        }
        
        return results
    
    def calculate_component_var(self, confidence_level: float = 0.95) -> pd.DataFrame:
        """Calculate component VaR and marginal VaR"""
        if self.returns_data is None or self.cov_matrix is None:
            if not self.fetch_historical_data():
                return None
        
        # Align portfolio weights with data columns
        asset_names = list(self.returns_data.columns)
        portfolio_weights = np.array([
            self.portfolio.get(asset, 0) for asset in asset_names
        ])
        
        # Portfolio variance
        cov_values = self.cov_matrix.values
        portfolio_variance = portfolio_weights.T @ cov_values @ portfolio_weights
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Parametric VaR
        z_score = stats.norm.ppf(1 - confidence_level)
        portfolio_var = abs(z_score * portfolio_std * self.portfolio_value)
        
        # Marginal VaR
        marginal_var = (cov_values @ portfolio_weights) / portfolio_std
        marginal_var = marginal_var * abs(z_score)
        
        # Component VaR
        component_var = marginal_var * portfolio_weights * self.portfolio_value
        
        # Incremental VaR (approximation)
        incremental_var = []
        for i in range(len(asset_names)):
            temp_weights = portfolio_weights.copy()
            temp_weights[i] = 0
            if temp_weights.sum() > 0:
                temp_weights = temp_weights / temp_weights.sum()
            temp_var = np.sqrt(temp_weights.T @ cov_values @ temp_weights)
            temp_var = abs(z_score * temp_var * self.portfolio_value)
            incremental_var.append(portfolio_var - temp_var)
        
        results_df = pd.DataFrame({
            'Asset': asset_names,
            'Weight': portfolio_weights,
            'Marginal_VaR_BRL': marginal_var * self.portfolio_value,
            'Component_VaR_BRL': component_var,
            'Incremental_VaR_BRL': incremental_var,
            'Percent_of_Total_VaR': (component_var / portfolio_var) * 100 if portfolio_var != 0 else 0
        })
        
        return results_df, portfolio_var
    
    def stress_test_var(self, stress_scenarios: Dict = None) -> pd.DataFrame:
        """Stress test VaR under different scenarios"""
        if self.returns_data is None:
            if not self.fetch_historical_data():
                return None
        
        if stress_scenarios is None:
            stress_scenarios = {
                'Brazil_2008': {'volatility_multiplier': 2.5, 'return_shock': -0.15},
                'Brazil_2015': {'volatility_multiplier': 1.8, 'return_shock': -0.10},
                'COVID_2020': {'volatility_multiplier': 3.0, 'return_shock': -0.25},
                'Rate_Hike': {'volatility_multiplier': 1.5, 'return_shock': -0.08},
                'BRL_Crash': {'volatility_multiplier': 2.0, 'return_shock': -0.12},
                'Commodity_Crash': {'volatility_multiplier': 2.2, 'return_shock': -0.18}
            }
        
        portfolio_weights = np.array([
            self.portfolio.get(col, 0) for col in self.returns_data.columns
        ])
        portfolio_returns = self.returns_data.dot(portfolio_weights)
        
        base_std = portfolio_returns.std()
        base_var = abs(stats.norm.ppf(0.05) * base_std * self.portfolio_value)
        
        stress_results = []
        
        for scenario_name, params in stress_scenarios.items():
            vol_mult = params.get('volatility_multiplier', 1.0)
            return_shock = params.get('return_shock', 0.0)
            
            stressed_std = base_std * vol_mult
            stressed_var = abs(stats.norm.ppf(0.05) * stressed_std * self.portfolio_value)
            
            portfolio_loss = abs(return_shock) * self.portfolio_value
            
            stress_results.append({
                'Scenario': scenario_name,
                'Description': self._get_scenario_description(scenario_name),
                'Vol_Multiplier': vol_mult,
                'Return_Shock': f"{return_shock*100:.1f}%",
                'Stressed_VaR_BRL': stressed_var,
                'VaR_Increase': f"{((stressed_var/base_var)-1)*100:.1f}%",
                'Portfolio_Loss_BRL': portfolio_loss
            })
        
        return pd.DataFrame(stress_results)
    
    def _get_scenario_description(self, scenario_name: str) -> str:
        """Get description for stress scenarios"""
        descriptions = {
            'Brazil_2008': 'Global Financial Crisis impact on Brazil',
            'Brazil_2015': 'Brazil recession and political crisis',
            'COVID_2020': 'COVID-19 pandemic market crash',
            'Rate_Hike': 'Brazilian central bank aggressive rate hikes',
            'BRL_Crash': 'Brazilian Real depreciation > 30%',
            'Commodity_Crash': 'Commodity prices collapse'
        }
        return descriptions.get(scenario_name, 'Custom stress scenario')
    
    def backtest_var(self, window: int = 252, confidence_level: float = 0.95) -> Dict:
        """Backtest VaR model"""
        if self.returns_data is None:
            if not self.fetch_historical_data():
                return None
        
        portfolio_weights = np.array([
            self.portfolio.get(col, 0) for col in self.returns_data.columns
        ])
        portfolio_returns = self.returns_data.dot(portfolio_weights)
        
        var_breaches = []
        actual_returns = []
        var_estimates = []
        
        for i in range(window, len(portfolio_returns)):
            historical_returns = portfolio_returns.iloc[i-window:i]
            
            var_estimate = abs(stats.norm.ppf(1 - confidence_level) * historical_returns.std())
            actual_return = portfolio_returns.iloc[i]
            
            var_estimates.append(var_estimate)
            actual_returns.append(actual_return)
            var_breaches.append(1 if actual_return < -var_estimate else 0)
        
        n_breaches = sum(var_breaches)
        n_observations = len(var_breaches)
        expected_breaches = (1 - confidence_level) * n_observations
        breach_ratio = n_breaches / n_observations if n_observations > 0 else 0
        
        # Kupiec test for VaR model validity
        if expected_breaches > 0 and 0 < breach_ratio < 1:
            try:
                lr_stat = 2 * (
                    n_breaches * np.log(breach_ratio / (1 - confidence_level)) +
                    (n_observations - n_breaches) * np.log((1 - breach_ratio) / confidence_level)
                )
                kupiec_pvalue = 1 - stats.chi2.cdf(lr_stat, 1)
                kupiec_stat = lr_stat
            except:
                kupiec_stat = np.nan
                kupiec_pvalue = np.nan
        else:
            kupiec_stat = np.nan
            kupiec_pvalue = np.nan
        
        return {
            'observations': n_observations,
            'breaches': n_breaches,
            'expected_breaches': expected_breaches,
            'breach_ratio': breach_ratio,
            'kupiec_statistic': kupiec_stat,
            'kupiec_pvalue': kupiec_pvalue,
            'model_valid': kupiec_pvalue > 0.05 if not np.isnan(kupiec_pvalue) else False,
            'var_estimates': var_estimates,
            'actual_returns': actual_returns,
            'var_breaches': var_breaches
        }
    
    def plot_var_analysis(self, save_path: str = None):
        """Visualize VaR analysis"""
        if not self.fetch_historical_data():
            print("Failed to fetch data for plotting")
            return None
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # Get portfolio returns
        portfolio_weights = np.array([
            self.portfolio.get(col, 0) for col in self.returns_data.columns
        ])
        portfolio_returns = self.returns_data.dot(portfolio_weights)
        
        # 1. Portfolio Returns Distribution
        ax1 = axes[0, 0]
        ax1.hist(portfolio_returns * 100, bins=50, alpha=0.7, color='steelblue', 
                edgecolor='black', density=True)
        
        # Add VaR lines
        var_95 = np.percentile(portfolio_returns, 5) * 100
        var_99 = np.percentile(portfolio_returns, 1) * 100
        
        ax1.axvline(x=var_95, color='orange', linestyle='--', linewidth=2, label=f'VaR 95%: {var_95:.2f}%')
        ax1.axvline(x=var_99, color='red', linestyle='--', linewidth=2, label=f'VaR 99%: {var_99:.2f}%')
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Density')
        ax1.set_title('Portfolio Returns Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. VaR Methods Comparison
        ax2 = axes[0, 1]
        methods = ['historical', 'parametric', 'monte_carlo']
        var_values = []
        cvar_values = []
        
        for method in methods:
            result = self.calculate_var(method=method)
            if result:
                var_values.append(result['var_daily_brl'] / 1e6)
                cvar_values.append(result['cvar_daily_brl'] / 1e6)
        
        x = np.arange(len(methods))
        width = 0.35
        
        ax2.bar(x - width/2, var_values, width, label='VaR', color='steelblue', alpha=0.7)
        ax2.bar(x + width/2, cvar_values, width, label='CVaR', color='coral', alpha=0.7)
        ax2.set_xticks(x)
        ax2.set_xticklabels(['Historical', 'Parametric', 'Monte Carlo'])
        ax2.set_ylabel('Risk (BRL Millions)')
        ax2.set_title('VaR Methods Comparison (95%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Component VaR
        ax3 = axes[0, 2]
        component_result = self.calculate_component_var()
        if component_result is not None:
            comp_df, total_var = component_result
            comp_df_sorted = comp_df.sort_values('Component_VaR_BRL', ascending=True)
            
            colors = ['green' if x > 0 else 'red' for x in comp_df_sorted['Component_VaR_BRL']]
            ax3.barh(comp_df_sorted['Asset'], comp_df_sorted['Component_VaR_BRL'] / 1e6, 
                    color=colors, alpha=0.7)
            ax3.set_xlabel('Component VaR (BRL Millions)')
            ax3.set_title('Component VaR by Asset')
            ax3.grid(True, alpha=0.3)
        
        # 4. Stress Test Results
        ax4 = axes[1, 0]
        stress_results = self.stress_test_var()
        if stress_results is not None:
            scenarios = stress_results['Scenario'].tolist()
            stressed_vars = stress_results['Stressed_VaR_BRL'].tolist()
            
            colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(scenarios)))
            ax4.barh(scenarios, [v/1e6 for v in stressed_vars], color=colors, alpha=0.8)
            ax4.set_xlabel('Stressed VaR (BRL Millions)')
            ax4.set_title('Stress Test Scenarios')
            ax4.grid(True, alpha=0.3)
        
        # 5. Rolling VaR
        ax5 = axes[1, 1]
        window = 60
        rolling_std = portfolio_returns.rolling(window=window).std()
        rolling_var = abs(stats.norm.ppf(0.05) * rolling_std * self.portfolio_value / 1e6)
        
        ax5.plot(rolling_var.index, rolling_var.values, 'b-', linewidth=1, label='Rolling VaR (60d)')
        ax5.fill_between(rolling_var.index, 0, rolling_var.values, alpha=0.3)
        ax5.set_xlabel('Date')
        ax5.set_ylabel('VaR (BRL Millions)')
        ax5.set_title('Rolling 60-Day VaR (95%)')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. VaR Backtest
        ax6 = axes[1, 2]
        backtest = self.backtest_var(window=252)
        if backtest and backtest['var_estimates']:
            dates = portfolio_returns.index[252:]
            ax6.plot(dates, backtest['actual_returns'], 'b-', alpha=0.5, linewidth=0.5, label='Actual Returns')
            ax6.plot(dates, [-v for v in backtest['var_estimates']], 'r--', linewidth=1, label='VaR (95%)')
            
            # Mark breaches
            breach_dates = dates[np.array(backtest['var_breaches']) == 1]
            breach_returns = np.array(backtest['actual_returns'])[np.array(backtest['var_breaches']) == 1]
            ax6.scatter(breach_dates, breach_returns, color='red', s=20, zorder=5, label='Breaches')
            
            ax6.set_xlabel('Date')
            ax6.set_ylabel('Return')
            ax6.set_title(f'VaR Backtest (Breaches: {backtest["breaches"]}, Expected: {backtest["expected_breaches"]:.0f})')
            ax6.legend(fontsize=8)
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Value at Risk Analysis - Brazilian Portfolio (BRL {self.portfolio_value/1e6:.1f}M)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print("VALUE AT RISK ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\nPortfolio Value: BRL {self.portfolio_value:,.0f}")
        print(f"\nPortfolio Composition:")
        for asset, weight in self.portfolio.items():
            print(f"  {asset}: {weight*100:.1f}%")
        
        print("\n--- VaR Comparison (95% Confidence, 1-Day) ---")
        for method in methods:
            result = self.calculate_var(method=method)
            if result:
                print(f"{result['method']}:")
                print(f"  VaR: BRL {result['var_daily_brl']:,.0f} ({result['var_daily_pct']:.2f}%)")
                print(f"  CVaR: BRL {result['cvar_daily_brl']:,.0f} ({result['cvar_daily_pct']:.2f}%)")
        
        if backtest:
            print(f"\n--- Backtest Results ---")
            print(f"Observations: {backtest['observations']}")
            print(f"Breaches: {backtest['breaches']} (Expected: {backtest['expected_breaches']:.1f})")
            print(f"Model Valid: {'Yes' if backtest['model_valid'] else 'No'}")
        
        return fig


if __name__ == "__main__":
    # Example usage with dynamic configuration
    config = get_config()
    
    print("="*80)
    print("BRAZILIAN PORTFOLIO VALUE AT RISK ANALYSIS")
    print("="*80)
    
    # Use portfolio from config file
    var_analyzer = BrazilianVaR(
        portfolio=None,  # Will load from config
        portfolio_value=10_000_000
    )
    
    print(f"\nUsing portfolio from config:")
    for ticker, weight in var_analyzer.portfolio.items():
        print(f"  {ticker}: {weight*100:.1f}%")
    
    # Run analysis
    var_analyzer.plot_var_analysis()
    
    # Example with custom portfolio
    print("\n" + "="*80)
    print("CUSTOM PORTFOLIO ANALYSIS")
    print("="*80)
    
    custom_portfolio = {
        'PETR4': 0.40,
        'VALE3': 0.30,
        'BTHF11': 0.30
    }
    
    custom_analyzer = BrazilianVaR(
        portfolio=custom_portfolio,
        portfolio_value=5_000_000
    )
    
    result = custom_analyzer.calculate_var(method='parametric')
    if result:
        print(f"\nParametric VaR (95%): BRL {result['var_daily_brl']:,.0f}")
        print(f"Portfolio Volatility: {result['portfolio_volatility']*100:.1f}%")
