# montecarlo_brazil_updated.py
"""
Monte Carlo Simulation for Brazilian Investments
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


class BrazilianMonteCarlo:
    def __init__(
        self, 
        base_npv: float = 100, 
        simulations: int = 10000,
        ticker: str = None,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        """
        Monte Carlo Simulation for Brazilian Investment Analysis
        
        Parameters:
        -----------
        base_npv : float
            Base NPV in BRL millions
        simulations : int
            Number of Monte Carlo simulations
        ticker : str
            Optional ticker to use for volatility calibration
        config : ConfigManager
            Configuration manager instance
        data_manager : DataManager
            Data manager instance
        """
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        self.simulations = simulations
        self.base_npv = base_npv
        self.ticker = ticker
        
        # Brazilian market parameters
        self.selic_rate = 0.1065  # 10.65%
        self.inflation = 0.045  # 4.5%
        self.exchange_rate_vol = 0.15  # 15% BRL/USD volatility
        self.country_risk_premium = 0.055  # 5.5%
        
        self.simulated_npv = None
        
        # Calibrate from market data if ticker provided
        if ticker:
            self._calibrate_from_market()
    
    def _calibrate_from_market(self):
        """Calibrate simulation parameters from market data"""
        data = self.data_manager.fetch_historical_data(self.ticker, period='2y')
        
        if data is not None and not data.empty:
            returns = data['Returns'].dropna()
            
            # Update volatility estimate
            self.market_vol = returns.std() * np.sqrt(252)
            self.market_mean = returns.mean() * 252
            
            print(f"Calibrated from {self.ticker}:")
            print(f"  Annualized volatility: {self.market_vol*100:.1f}%")
            print(f"  Annualized mean return: {self.market_mean*100:.1f}%")
    
    def simulate_npv(
        self, 
        revenue_growth_mean: float = 0.07,
        revenue_growth_std: float = 0.04,
        ebitda_margin_mean: float = 0.22,
        ebitda_margin_std: float = 0.03,
        wacc_mean: float = 0.12,
        wacc_std: float = 0.015
    ) -> np.ndarray:
        """
        Simulate NPV using Monte Carlo with Brazilian market factors
        
        Parameters:
        -----------
        revenue_growth_mean : float
            Mean revenue growth rate
        revenue_growth_std : float
            Std dev of revenue growth
        ebitda_margin_mean : float
            Mean EBITDA margin
        ebitda_margin_std : float
            Std dev of EBITDA margin
        wacc_mean : float
            Mean WACC
        wacc_std : float
            Std dev of WACC
            
        Returns:
        --------
        Array of simulated NPV values
        """
        np.random.seed(42)
        
        # Generate random variables
        revenue_growth = np.random.normal(revenue_growth_mean, revenue_growth_std, self.simulations)
        ebitda_margin = np.random.normal(ebitda_margin_mean, ebitda_margin_std, self.simulations)
        wacc = np.random.normal(wacc_mean, wacc_std, self.simulations)
        terminal_growth = np.random.normal(0.025, 0.005, self.simulations)
        
        # Brazil-specific factors
        fx_change = np.random.normal(0.02, self.exchange_rate_vol, self.simulations)
        
        # Country risk events (5% probability of major event)
        country_risk_shock = np.random.choice([0, 1], size=self.simulations, p=[0.95, 0.05])
        country_risk_impact = country_risk_shock * np.random.uniform(-0.15, -0.25, self.simulations)
        
        simulated_npv = []
        
        for i in range(self.simulations):
            # Base revenue projection (5 years)
            base_revenue = 100  # Normalized
            projected_revenue = base_revenue * (1 + revenue_growth[i]) ** 5
            
            # EBITDA
            ebitda = projected_revenue * max(0.05, min(0.40, ebitda_margin[i]))
            
            # Ensure WACC > terminal growth
            sim_wacc = max(0.08, wacc[i])
            sim_terminal_growth = min(terminal_growth[i], sim_wacc - 0.02)
            
            # Terminal value
            if sim_wacc > sim_terminal_growth:
                terminal_value = ebitda * (1 + sim_terminal_growth) / (sim_wacc - sim_terminal_growth)
            else:
                terminal_value = ebitda * 10
            
            # Discounted value
            discounted_tv = terminal_value / ((1 + sim_wacc) ** 5)
            
            # DCF of interim cash flows
            dcf_interim = sum([
                ebitda * 0.7 / ((1 + sim_wacc) ** year) 
                for year in range(1, 6)
            ])
            
            # Total NPV
            npv = (dcf_interim + discounted_tv) * (1 + fx_change[i])
            
            # Apply country risk shock
            npv = npv * (1 + country_risk_impact[i])
            
            # Scale to base NPV
            npv = npv * self.base_npv / 100
            
            simulated_npv.append(npv)
        
        self.simulated_npv = np.array(simulated_npv)
        return self.simulated_npv
    
    def calculate_risk_metrics(self, confidence_level: float = 0.95) -> Dict:
        """Calculate VaR, CVaR, and other risk metrics"""
        if self.simulated_npv is None:
            self.simulate_npv()
        
        sorted_npv = np.sort(self.simulated_npv)
        
        var_index = int((1 - confidence_level) * len(sorted_npv))
        var = sorted_npv[var_index]
        cvar = sorted_npv[:var_index].mean()
        
        return {
            'mean': np.mean(self.simulated_npv),
            'median': np.median(self.simulated_npv),
            'std': np.std(self.simulated_npv),
            'var_95': var,
            'cvar_95': cvar,
            'min': sorted_npv[0],
            'max': sorted_npv[-1],
            'prob_negative': np.mean(self.simulated_npv < 0),
            'prob_under_base': np.mean(self.simulated_npv < self.base_npv),
            'percentile_25': np.percentile(self.simulated_npv, 25),
            'percentile_75': np.percentile(self.simulated_npv, 75)
        }
    
    def sensitivity_analysis(self) -> pd.DataFrame:
        """Analyze sensitivity to different parameters"""
        base_stats = self.calculate_risk_metrics()
        
        parameters = {
            'Revenue Growth': {'param': 'revenue_growth_mean', 'low': 0.03, 'high': 0.11},
            'EBITDA Margin': {'param': 'ebitda_margin_mean', 'low': 0.15, 'high': 0.28},
            'WACC': {'param': 'wacc_mean', 'low': 0.10, 'high': 0.15},
            'FX Volatility': {'param': 'exchange_rate_vol', 'low': 0.08, 'high': 0.22}
        }
        
        results = []
        
        for param_name, settings in parameters.items():
            # Low case
            self.simulated_npv = None
            if settings['param'] == 'exchange_rate_vol':
                self.exchange_rate_vol = settings['low']
                self.simulate_npv()
                self.exchange_rate_vol = 0.15
            else:
                kwargs = {settings['param']: settings['low']}
                self.simulate_npv(**kwargs)
            
            low_stats = self.calculate_risk_metrics()
            
            # High case
            self.simulated_npv = None
            if settings['param'] == 'exchange_rate_vol':
                self.exchange_rate_vol = settings['high']
                self.simulate_npv()
                self.exchange_rate_vol = 0.15
            else:
                kwargs = {settings['param']: settings['high']}
                self.simulate_npv(**kwargs)
            
            high_stats = self.calculate_risk_metrics()
            
            results.append({
                'Parameter': param_name,
                'Low_Value': settings['low'],
                'High_Value': settings['high'],
                'NPV_Low': low_stats['mean'],
                'NPV_High': high_stats['mean'],
                'NPV_Range': high_stats['mean'] - low_stats['mean']
            })
        
        # Reset to base case
        self.simulated_npv = None
        self.simulate_npv()
        
        return pd.DataFrame(results)
    
    def scenario_analysis(self, scenarios: Dict = None) -> pd.DataFrame:
        """Run scenario analysis"""
        if scenarios is None:
            scenarios = {
                'Base Case': {'selic': 0.1065, 'inflation': 0.045, 'fx_vol': 0.15},
                'High Inflation': {'selic': 0.15, 'inflation': 0.08, 'fx_vol': 0.20},
                'Economic Crisis': {'selic': 0.18, 'inflation': 0.10, 'fx_vol': 0.25},
                'Stability': {'selic': 0.09, 'inflation': 0.035, 'fx_vol': 0.10},
                'Strong BRL': {'selic': 0.08, 'inflation': 0.03, 'fx_vol': 0.08}
            }
        
        results = []
        
        for scenario_name, params in scenarios.items():
            # Adjust parameters
            self.selic_rate = params['selic']
            self.inflation = params['inflation']
            self.exchange_rate_vol = params['fx_vol']
            
            # Adjust WACC based on SELIC
            wacc_mean = self.selic_rate + 0.02
            
            self.simulated_npv = None
            self.simulate_npv(wacc_mean=wacc_mean)
            stats = self.calculate_risk_metrics()
            
            results.append({
                'Scenario': scenario_name,
                'SELIC': f"{params['selic']*100:.1f}%",
                'Inflation': f"{params['inflation']*100:.1f}%",
                'FX_Vol': f"{params['fx_vol']*100:.0f}%",
                'Mean_NPV': stats['mean'],
                'VaR_95': stats['var_95'],
                'Prob_Negative': f"{stats['prob_negative']*100:.1f}%"
            })
        
        # Reset to defaults
        self.selic_rate = 0.1065
        self.inflation = 0.045
        self.exchange_rate_vol = 0.15
        self.simulated_npv = None
        
        return pd.DataFrame(results)
    
    def plot_results(self, save_path: str = None):
        """Visualize Monte Carlo results"""
        if self.simulated_npv is None:
            self.simulate_npv()
        
        stats = self.calculate_risk_metrics()
        
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        
        # 1. NPV Distribution
        ax1 = axes[0, 0]
        ax1.hist(self.simulated_npv, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax1.axvline(x=stats['mean'], color='red', linestyle='--', linewidth=2, 
                   label=f"Mean: {stats['mean']:.1f}")
        ax1.axvline(x=stats['median'], color='green', linestyle='--', linewidth=2,
                   label=f"Median: {stats['median']:.1f}")
        ax1.axvline(x=stats['var_95'], color='orange', linestyle='--', linewidth=2,
                   label=f"VaR 95%: {stats['var_95']:.1f}")
        ax1.axvline(x=self.base_npv, color='purple', linestyle='--', linewidth=2,
                   label=f"Base NPV: {self.base_npv:.1f}")
        ax1.set_xlabel('NPV (BRL Millions)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Monte Carlo: NPV Distribution')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # 2. Cumulative Distribution
        ax2 = axes[0, 1]
        sorted_npv = np.sort(self.simulated_npv)
        cumulative_prob = np.arange(1, len(sorted_npv) + 1) / len(sorted_npv)
        
        ax2.plot(sorted_npv, cumulative_prob, 'b-', linewidth=2)
        ax2.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='5% VaR Level')
        ax2.axvline(x=stats['var_95'], color='red', linestyle='--', alpha=0.7)
        ax2.set_xlabel('NPV (BRL Millions)')
        ax2.set_ylabel('Cumulative Probability')
        ax2.set_title('Cumulative Distribution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Box Plot
        ax3 = axes[0, 2]
        bp = ax3.boxplot([self.simulated_npv], vert=True, patch_artist=True, labels=['NPV'])
        bp['boxes'][0].set_facecolor('lightblue')
        ax3.set_ylabel('NPV (BRL Millions)')
        ax3.set_title('Box Plot: NPV Distribution')
        ax3.grid(True, alpha=0.3)
        
        # 4. Risk Metrics Summary
        ax4 = axes[1, 0]
        metrics = ['Mean', 'Median', 'Std Dev', 'VaR 95%', 'CVaR 95%', 'Min', 'Max']
        values = [stats['mean'], stats['median'], stats['std'], 
                 stats['var_95'], stats['cvar_95'], stats['min'], stats['max']]
        
        colors = ['blue', 'green', 'orange', 'red', 'darkred', 'purple', 'darkgreen']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7)
        ax4.set_ylabel('BRL Millions')
        ax4.set_title('Risk Metrics Summary')
        ax4.tick_params(axis='x', rotation=45)
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, values):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{value:.1f}', ha='center', fontsize=9)
        
        # 5. Probability Analysis
        ax5 = axes[1, 1]
        probs = [
            stats['prob_negative'] * 100,
            stats['prob_under_base'] * 100,
            (1 - stats['prob_under_base']) * 100,
            np.mean(self.simulated_npv > self.base_npv * 1.5) * 100
        ]
        labels = ['Negative NPV', '< Base NPV', '> Base NPV', '> 1.5x Base']
        
        colors = ['red', 'orange', 'lightgreen', 'green']
        ax5.bar(labels, probs, color=colors, alpha=0.7, edgecolor='black')
        ax5.set_ylabel('Probability (%)')
        ax5.set_title('Probability Analysis')
        ax5.tick_params(axis='x', rotation=15)
        ax5.grid(True, alpha=0.3)
        
        for i, (label, prob) in enumerate(zip(labels, probs)):
            ax5.text(i, prob + 1, f'{prob:.1f}%', ha='center', fontsize=10)
        
        # 6. Sensitivity Tornado
        ax6 = axes[1, 2]
        sensitivity = self.sensitivity_analysis()
        
        if sensitivity is not None and not sensitivity.empty:
            params = sensitivity['Parameter'].tolist()
            ranges = sensitivity['NPV_Range'].tolist()
            
            colors = plt.cm.coolwarm(np.linspace(0.2, 0.8, len(params)))
            bars = ax6.barh(params, ranges, color=colors, alpha=0.8)
            ax6.set_xlabel('NPV Range (BRL Millions)')
            ax6.set_title('Sensitivity Analysis')
            ax6.grid(True, alpha=0.3)
        
        plt.suptitle(f'Monte Carlo Simulation: Brazilian Investment (Base NPV: BRL {self.base_npv}M)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print("MONTE CARLO SIMULATION SUMMARY")
        print("="*80)
        print(f"\nSimulations: {self.simulations:,}")
        print(f"Base NPV: BRL {self.base_npv:.1f} million")
        print(f"\nKey Statistics:")
        print(f"  Mean NPV: BRL {stats['mean']:.1f} million")
        print(f"  Median NPV: BRL {stats['median']:.1f} million")
        print(f"  Std Dev: BRL {stats['std']:.1f} million")
        print(f"\nRisk Metrics:")
        print(f"  VaR (95%): BRL {stats['var_95']:.1f} million")
        print(f"  CVaR (95%): BRL {stats['cvar_95']:.1f} million")
        print(f"  Prob. Negative: {stats['prob_negative']*100:.1f}%")
        print(f"  Prob. Below Base: {stats['prob_under_base']*100:.1f}%")
        
        return fig, stats


if __name__ == "__main__":
    config = get_config()
    
    print("="*80)
    print("BRAZILIAN MONTE CARLO SIMULATION")
    print("="*80)
    
    # Basic simulation
    mc = BrazilianMonteCarlo(
        base_npv=150,
        simulations=50000
    )
    
    fig, stats = mc.plot_results()
    
    # Scenario analysis
    print("\n" + "="*80)
    print("SCENARIO ANALYSIS")
    print("="*80)
    
    scenarios = mc.scenario_analysis()
    print("\n" + scenarios.to_string(index=False))
    
    # Calibrated from market data
    print("\n" + "="*80)
    print("MARKET-CALIBRATED SIMULATION")
    print("="*80)
    
    default_ticker = config.get_default_ticker()
    mc_calibrated = BrazilianMonteCarlo(
        base_npv=100,
        simulations=10000,
        ticker=default_ticker
    )
    
    mc_calibrated.simulate_npv()
    calibrated_stats = mc_calibrated.calculate_risk_metrics()
    print(f"\nCalibrated from {default_ticker}:")
    print(f"  Mean NPV: BRL {calibrated_stats['mean']:.1f} million")
    print(f"  VaR 95%: BRL {calibrated_stats['var_95']:.1f} million")
