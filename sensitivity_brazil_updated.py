# sensitivity_brazil_updated.py
"""
Sensitivity Analysis for Brazilian Investments
Updated to use dynamic configuration and shared data manager
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from typing import Dict, List, Optional
import warnings

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')

from config import get_config, ConfigManager
from data_manager import get_data_manager, DataManager


class BrazilianSensitivityAnalysis:
    def __init__(
        self, 
        base_price: float = 50, 
        base_volume: float = 1000000, 
        base_margin: float = 0.25,
        ticker: str = None,
        config: ConfigManager = None,
        data_manager: DataManager = None
    ):
        """
        Sensitivity Analysis for Brazilian Investments
        
        Parameters:
        -----------
        base_price : float
            Base price in BRL per unit
        base_volume : float
            Base volume in units
        base_margin : float
            Base EBITDA margin
        ticker : str
            Optional ticker to calibrate from market data
        config : ConfigManager
            Configuration manager instance
        data_manager : DataManager
            Data manager instance
        """
        self.config = config or get_config()
        self.data_manager = data_manager or get_data_manager(self.config)
        
        self.base_price = base_price
        self.base_volume = base_volume
        self.base_margin = base_margin
        self.base_capex = 50000000  # BRL 50 million
        self.base_wacc = 0.12  # 12% for Brazil
        
        # Brazilian market specific factors
        self.exchange_rate = 5.0  # BRL/USD
        self.selic_rate = 0.1065  # 10.65%
        self.inflation = 0.045  # 4.5% target
        
        self.ticker = ticker
        if ticker:
            self._calibrate_from_market()
    
    def _calibrate_from_market(self):
        """Calibrate parameters from market data"""
        info = self.data_manager.get_stock_info(self.ticker)
        
        if info:
            self.base_margin = info.get('profitMargins', self.base_margin) or self.base_margin
            beta = info.get('beta', 1.0) or 1.0
            self.base_wacc = self.selic_rate + beta * 0.055  # CAPM
            
            print(f"Calibrated from {self.ticker}:")
            print(f"  Margin: {self.base_margin*100:.1f}%")
            print(f"  WACC: {self.base_wacc*100:.1f}%")
    
    def calculate_base_npv(self) -> float:
        """Calculate base case NPV"""
        revenue = self.base_price * self.base_volume
        ebitda = revenue * self.base_margin
        capex = self.base_capex
        fcf = ebitda * 0.7 - capex * 0.3  # Simplified FCF
        
        npv = 0
        for year in range(1, 6):
            npv += fcf / ((1 + self.base_wacc) ** year)
        
        # Terminal value
        terminal_growth = 0.025
        terminal_value = fcf * (1 + terminal_growth) / (self.base_wacc - terminal_growth)
        npv += terminal_value / ((1 + self.base_wacc) ** 5)
        
        return npv
    
    def price_volume_sensitivity(
        self, 
        price_range: tuple = (-0.2, 0.2), 
        volume_range: tuple = (-0.15, 0.15),
        steps: int = 20
    ) -> tuple:
        """Two-way sensitivity: Price vs Volume"""
        price_changes = np.linspace(price_range[0], price_range[1], steps)
        volume_changes = np.linspace(volume_range[0], volume_range[1], steps)
        
        sensitivity_matrix = np.zeros((steps, steps))
        
        for i, price_pct in enumerate(price_changes):
            for j, volume_pct in enumerate(volume_changes):
                price = self.base_price * (1 + price_pct)
                volume = self.base_volume * (1 + volume_pct)
                
                revenue = price * volume
                ebitda = revenue * self.base_margin
                fcf = ebitda * 0.7 - self.base_capex * 0.3
                
                npv = 0
                for year in range(1, 6):
                    npv += fcf / ((1 + self.base_wacc) ** year)
                
                terminal_value = fcf * 1.025 / (self.base_wacc - 0.025)
                npv += terminal_value / ((1 + self.base_wacc) ** 5)
                
                sensitivity_matrix[i, j] = npv / 1e6  # In millions
        
        return price_changes, volume_changes, sensitivity_matrix
    
    def wacc_growth_sensitivity(
        self, 
        wacc_range: tuple = (0.08, 0.16), 
        growth_range: tuple = (0.01, 0.05),
        steps: int = 20
    ) -> tuple:
        """Two-way sensitivity: WACC vs Terminal Growth Rate"""
        wacc_values = np.linspace(wacc_range[0], wacc_range[1], steps)
        growth_values = np.linspace(growth_range[0], growth_range[1], steps)
        
        sensitivity_matrix = np.zeros((steps, steps))
        
        base_fcf = self.base_price * self.base_volume * self.base_margin * 0.7 - self.base_capex * 0.3
        
        for i, wacc in enumerate(wacc_values):
            for j, growth in enumerate(growth_values):
                if wacc <= growth:
                    growth = wacc - 0.01
                
                npv = 0
                for year in range(1, 6):
                    npv += base_fcf / ((1 + wacc) ** year)
                
                terminal_value = base_fcf * (1 + growth) / (wacc - growth)
                npv += terminal_value / ((1 + wacc) ** 5)
                
                sensitivity_matrix[i, j] = npv / 1e6
        
        return wacc_values, growth_values, sensitivity_matrix
    
    def margin_capex_sensitivity(
        self, 
        margin_range: tuple = (0.15, 0.35), 
        capex_range: tuple = (0.5, 1.5),
        steps: int = 20
    ) -> tuple:
        """Two-way sensitivity: EBITDA Margin vs CAPEX"""
        margin_values = np.linspace(margin_range[0], margin_range[1], steps)
        capex_multipliers = np.linspace(capex_range[0], capex_range[1], steps)
        
        sensitivity_matrix = np.zeros((steps, steps))
        
        base_revenue = self.base_price * self.base_volume
        
        for i, margin in enumerate(margin_values):
            for j, capex_mult in enumerate(capex_multipliers):
                ebitda = base_revenue * margin
                capex = self.base_capex * capex_mult
                fcf = ebitda * 0.7 - capex * 0.3
                
                npv = 0
                for year in range(1, 6):
                    npv += fcf / ((1 + self.base_wacc) ** year)
                
                terminal_value = fcf * 1.025 / (self.base_wacc - 0.025)
                npv += terminal_value / ((1 + self.base_wacc) ** 5)
                
                sensitivity_matrix[i, j] = npv / 1e6
        
        return margin_values, capex_multipliers, sensitivity_matrix
    
    def scenario_analysis(self, scenarios: Dict = None) -> pd.DataFrame:
        """Multi-scenario analysis"""
        if scenarios is None:
            scenarios = {
                'Base Case': {},
                'Bull Case': {
                    'price': self.base_price * 1.15,
                    'volume': self.base_volume * 1.1,
                    'margin': 0.28
                },
                'Bear Case': {
                    'price': self.base_price * 0.85,
                    'volume': self.base_volume * 0.9,
                    'margin': 0.22,
                    'wacc': 0.14
                },
                'High Inflation': {
                    'price': self.base_price * 1.1,
                    'wacc': 0.15,
                    'margin': 0.23
                },
                'Strong BRL': {
                    'price': self.base_price * 1.05,
                    'margin': 0.27,
                    'wacc': 0.11
                },
                'Commodity Boom': {
                    'price': self.base_price * 1.3,
                    'margin': 0.30,
                    'wacc': 0.12
                }
            }
        
        results = []
        
        for scenario_name, params in scenarios.items():
            price = params.get('price', self.base_price)
            volume = params.get('volume', self.base_volume)
            margin = params.get('margin', self.base_margin)
            wacc = params.get('wacc', self.base_wacc)
            terminal_growth = params.get('terminal_growth', 0.025)
            
            revenue = price * volume
            ebitda = revenue * margin
            fcf = ebitda * 0.7 - self.base_capex * 0.3
            
            npv = 0
            for year in range(1, 6):
                npv += fcf / ((1 + wacc) ** year)
            
            if wacc > terminal_growth:
                terminal_value = fcf * (1 + terminal_growth) / (wacc - terminal_growth)
            else:
                terminal_value = fcf * 10
            
            npv += terminal_value / ((1 + wacc) ** 5)
            
            base_npv = self.calculate_base_npv()
            change_vs_base = ((npv / base_npv) - 1) * 100
            
            results.append({
                'Scenario': scenario_name,
                'Price': f"R${price:.0f}",
                'Volume': f"{volume/1e6:.2f}M",
                'Margin': f"{margin*100:.0f}%",
                'WACC': f"{wacc*100:.1f}%",
                'NPV_BRL_M': npv / 1e6,
                'Change_vs_Base': f"{change_vs_base:+.1f}%"
            })
        
        return pd.DataFrame(results)
    
    def tornado_diagram(self, base_npv: float = None) -> pd.DataFrame:
        """Generate tornado diagram data"""
        if base_npv is None:
            base_npv = self.calculate_base_npv() / 1e6
        
        parameters = {
            'Price': {'low': -0.15, 'high': 0.15, 'base': self.base_price},
            'Volume': {'low': -0.10, 'high': 0.10, 'base': self.base_volume},
            'Margin': {'low': -0.05, 'high': 0.05, 'base': self.base_margin},
            'WACC': {'low': 0.02, 'high': -0.02, 'base': self.base_wacc},  # Inverted
            'Terminal Growth': {'low': -0.01, 'high': 0.01, 'base': 0.025}
        }
        
        results = []
        
        for param_name, settings in parameters.items():
            low_change = settings['low']
            high_change = settings['high']
            
            # Calculate NPV for low and high cases
            for case, change in [('low', low_change), ('high', high_change)]:
                price = self.base_price
                volume = self.base_volume
                margin = self.base_margin
                wacc = self.base_wacc
                terminal_growth = 0.025
                
                if param_name == 'Price':
                    price = self.base_price * (1 + change)
                elif param_name == 'Volume':
                    volume = self.base_volume * (1 + change)
                elif param_name == 'Margin':
                    margin = self.base_margin + change
                elif param_name == 'WACC':
                    wacc = self.base_wacc + change
                elif param_name == 'Terminal Growth':
                    terminal_growth = 0.025 + change
                
                revenue = price * volume
                ebitda = revenue * margin
                fcf = ebitda * 0.7 - self.base_capex * 0.3
                
                npv = 0
                for year in range(1, 6):
                    npv += fcf / ((1 + wacc) ** year)
                
                if wacc > terminal_growth:
                    terminal_value = fcf * (1 + terminal_growth) / (wacc - terminal_growth)
                else:
                    terminal_value = fcf * 10
                
                npv += terminal_value / ((1 + wacc) ** 5)
                npv = npv / 1e6
                
                if case == 'low':
                    low_npv = npv
                else:
                    high_npv = npv
            
            results.append({
                'Parameter': param_name,
                'Low_NPV': min(low_npv, high_npv),
                'High_NPV': max(low_npv, high_npv),
                'Range': abs(high_npv - low_npv),
                'Base_NPV': base_npv
            })
        
        return pd.DataFrame(results).sort_values('Range', ascending=True)
    
    def plot_sensitivity_analysis(self, save_path: str = None):
        """Create comprehensive sensitivity visualization"""
        fig = plt.figure(figsize=(18, 12))
        
        # 1. Price vs Volume Sensitivity (3D)
        ax1 = fig.add_subplot(231, projection='3d')
        price_changes, volume_changes, matrix1 = self.price_volume_sensitivity()
        
        X, Y = np.meshgrid(volume_changes * 100, price_changes * 100)
        surf1 = ax1.plot_surface(X, Y, matrix1, cmap=cm.coolwarm, alpha=0.8)
        ax1.set_xlabel('Volume Change (%)')
        ax1.set_ylabel('Price Change (%)')
        ax1.set_zlabel('NPV (BRL M)')
        ax1.set_title('Price vs Volume Sensitivity')
        
        # 2. WACC vs Growth Sensitivity (3D)
        ax2 = fig.add_subplot(232, projection='3d')
        wacc_values, growth_values, matrix2 = self.wacc_growth_sensitivity()
        
        X2, Y2 = np.meshgrid(growth_values * 100, wacc_values * 100)
        surf2 = ax2.plot_surface(X2, Y2, matrix2, cmap=cm.viridis, alpha=0.8)
        ax2.set_xlabel('Growth Rate (%)')
        ax2.set_ylabel('WACC (%)')
        ax2.set_zlabel('NPV (BRL M)')
        ax2.set_title('WACC vs Growth Sensitivity')
        
        # 3. Price-Volume Heatmap
        ax3 = fig.add_subplot(233)
        price_changes, volume_changes, matrix1 = self.price_volume_sensitivity()
        
        sns.heatmap(matrix1, ax=ax3, cmap='RdYlGn', 
                   xticklabels=[f"{v*100:.0f}%" for v in volume_changes[::4]],
                   yticklabels=[f"{p*100:.0f}%" for p in price_changes[::4]])
        ax3.set_xlabel('Volume Change')
        ax3.set_ylabel('Price Change')
        ax3.set_title('Price-Volume Sensitivity Heatmap')
        
        # 4. Scenario Analysis
        ax4 = fig.add_subplot(234)
        scenario_df = self.scenario_analysis()
        
        scenarios = scenario_df['Scenario'].tolist()
        npv_values = scenario_df['NPV_BRL_M'].tolist()
        
        colors = ['gray' if s == 'Base Case' else 
                  'green' if npv_values[i] > npv_values[0] else 'red' 
                  for i, s in enumerate(scenarios)]
        
        bars = ax4.barh(scenarios, npv_values, color=colors, alpha=0.7)
        ax4.set_xlabel('NPV (BRL Millions)')
        ax4.set_title('Scenario Analysis')
        ax4.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, npv_values):
            ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                    f'{value:.1f}', va='center', fontsize=9)
        
        # 5. Tornado Diagram
        ax5 = fig.add_subplot(235)
        base_npv = self.calculate_base_npv() / 1e6
        tornado = self.tornado_diagram(base_npv)
        
        params = tornado['Parameter'].tolist()
        low_npvs = tornado['Low_NPV'].tolist()
        high_npvs = tornado['High_NPV'].tolist()
        
        y_pos = np.arange(len(params))
        
        # Draw bars
        for i, param in enumerate(params):
            ax5.barh(y_pos[i], high_npvs[i] - base_npv, left=base_npv, 
                    height=0.5, color='green', alpha=0.7)
            ax5.barh(y_pos[i], low_npvs[i] - base_npv, left=base_npv, 
                    height=0.5, color='red', alpha=0.7)
        
        ax5.axvline(x=base_npv, color='black', linewidth=2, linestyle='--', label='Base NPV')
        ax5.set_yticks(y_pos)
        ax5.set_yticklabels(params)
        ax5.set_xlabel('NPV (BRL Millions)')
        ax5.set_title('Tornado Diagram')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. Margin vs CAPEX Heatmap
        ax6 = fig.add_subplot(236)
        margin_values, capex_mult, matrix3 = self.margin_capex_sensitivity()
        
        sns.heatmap(matrix3, ax=ax6, cmap='RdYlGn',
                   xticklabels=[f"{c:.1f}x" for c in capex_mult[::4]],
                   yticklabels=[f"{m*100:.0f}%" for m in margin_values[::4]])
        ax6.set_xlabel('CAPEX Multiplier')
        ax6.set_ylabel('EBITDA Margin')
        ax6.set_title('Margin vs CAPEX Sensitivity')
        
        plt.suptitle(f'Sensitivity Analysis - Brazilian Investment', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Chart saved to {save_path}")
        
        plt.show()
        
        # Print summary
        print("\n" + "="*80)
        print("SENSITIVITY ANALYSIS SUMMARY")
        print("="*80)
        print(f"\nBase Case NPV: BRL {base_npv:.1f} million")
        print(f"\nBase Parameters:")
        print(f"  Price: R$ {self.base_price:.0f}")
        print(f"  Volume: {self.base_volume/1e6:.2f} million units")
        print(f"  Margin: {self.base_margin*100:.0f}%")
        print(f"  WACC: {self.base_wacc*100:.1f}%")
        
        print("\n--- Scenario Analysis ---")
        print(scenario_df.to_string(index=False))
        
        print("\n--- Tornado Analysis ---")
        print(tornado[['Parameter', 'Low_NPV', 'High_NPV', 'Range']].to_string(index=False))
        
        return fig


if __name__ == "__main__":
    config = get_config()
    
    print("="*80)
    print("BRAZILIAN SENSITIVITY ANALYSIS")
    print("="*80)
    
    # Basic sensitivity analysis
    sensitivity = BrazilianSensitivityAnalysis(
        base_price=75,
        base_volume=1500000,
        base_margin=0.22
    )
    
    sensitivity.plot_sensitivity_analysis()
    
    # Calibrated from market
    print("\n" + "="*80)
    print("MARKET-CALIBRATED ANALYSIS")
    print("="*80)
    
    default_ticker = config.get_default_ticker()
    calibrated = BrazilianSensitivityAnalysis(
        base_price=50,
        base_volume=1000000,
        ticker=default_ticker
    )
    
    scenarios = calibrated.scenario_analysis()
    print("\n" + scenarios.to_string(index=False))
