# filepath: c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino_ready\Arduino_Version\hardware_validation_comparison.py
"""
Hardware Validation Comparison Tool
===================================

Vergleicht theoretische Berechnungen mit echten Arduino Hardware-Messungen:
- Theoretische vs. gemessene Inference-Zeiten
- Theoretische vs. gemessene RAM-Verbrauch
- Theoretische vs. gemessene Energie-Verbrauch
- Wissenschaftliche Analyse der Abweichungen
- Validierung der Deployment-Feasibility

Verwendet Daten von:
1. Theoretischen Berechnungen (wie im wissenschaftlichen Script)
2. Echten Arduino Hardware-Messungen (arduino_hardware_monitor.py)

Erstellt wissenschaftliche Validierungs-Plots für Papers!
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import statistics

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HardwareValidationComparison:
    """
    Hardware Validation & Comparison Tool
    ====================================
    
    Analysiert Abweichungen zwischen theoretischen Berechnungen und
    echten Hardware-Messungen für wissenschaftliche Validierung
    """
    
    def __init__(self):
        self.theoretical_data = {}
        self.measured_data = {}
        self.comparison_results = {}
        
        # Arduino Spezifikationen (anpassen an dein Board)
        self.arduino_specs = {
            'model': 'Arduino Uno R3',  # oder Due, ESP32, etc.
            'cpu_frequency_mhz': 16,    # 16 MHz für Uno
            'total_ram_bytes': 2048,    # 2 KB für Uno
            'flash_bytes': 32768,       # 32 KB für Uno
            'voltage': 5.0,             # 5V für Uno
            'typical_power_w': 0.2      # ~200mW typisch
        }
        
        logger.info("Hardware Validation Tool initialisiert")
        logger.info(f"Target Arduino: {self.arduino_specs['model']}")
    
    def load_theoretical_calculations(self):
        """
        Lade theoretische Berechnungen
        (Hier würdest du die Werte aus deinem wissenschaftlichen Script nehmen)
        """
        
        # LSTM Model Annahmen (anpassen an dein Modell)
        model_config = {
            'input_size': 4,      # Voltage, Current, SOH, Q_c
            'hidden_size': 128,   # wie in deinem Config
            'num_layers': 2,
            'sequence_length': 720,  # wie in deinem Config
            'dropout': 0.3
        }
        
        # Theoretische Berechnungen (vereinfacht - aus deinem wissenschaftlichen Script)
        total_params = (
            # LSTM Layer 0: input-to-hidden + hidden-to-hidden
            4 * (4 * model_config['input_size'] * model_config['hidden_size'] + 
                 4 * model_config['hidden_size'] * model_config['hidden_size']) +
            # LSTM Layer 1: hidden-to-hidden
            4 * (4 * model_config['hidden_size'] * model_config['hidden_size'] + 
                 4 * model_config['hidden_size'] * model_config['hidden_size']) +
            # FC Layers
            model_config['hidden_size'] * 64 + 64 + 64 * 1 + 1
        )
        
        # Memory requirements (32-bit floats = 4 bytes)
        model_memory_bytes = total_params * 4
        
        # Inference operations (MAC operations)
        lstm_ops_per_timestep = (
            8 * model_config['input_size'] * model_config['hidden_size'] +  # Layer 0
            8 * model_config['hidden_size'] * model_config['hidden_size'] +
            8 * model_config['hidden_size'] * model_config['hidden_size'] +  # Layer 1
            8 * model_config['hidden_size'] * model_config['hidden_size']
        )
        total_lstm_ops = lstm_ops_per_timestep * model_config['sequence_length']
        fc_ops = model_config['hidden_size'] * 64 + 64 * 1
        total_ops = total_lstm_ops + fc_ops
        
        # Timing estimates (pessimistische Annahme: 2-4 Zyklen pro MAC)
        cycles_per_op = 3  # Mittelwert
        cycles_per_inference = total_ops * cycles_per_op
        theoretical_inference_time_us = (cycles_per_inference / (self.arduino_specs['cpu_frequency_mhz'] * 1e6)) * 1e6
        
        # Runtime memory (input window + hidden states + intermediate)
        input_window_bytes = model_config['sequence_length'] * model_config['input_size'] * 4
        hidden_states_bytes = model_config['num_layers'] * model_config['hidden_size'] * 2 * 4  # h und c
        intermediate_bytes = model_config['sequence_length'] * model_config['hidden_size'] * 4
        total_runtime_memory_bytes = input_window_bytes + hidden_states_bytes + intermediate_bytes
        
        # Energy estimates
        theoretical_energy_per_inference_j = (self.arduino_specs['typical_power_w'] * 
                                             theoretical_inference_time_us / 1e6)
        theoretical_energy_per_inference_uj = theoretical_energy_per_inference_j * 1e6
        
        self.theoretical_data = {
            'model_config': model_config,
            'total_parameters': total_params,
            'model_memory_bytes': model_memory_bytes,
            'total_operations': total_ops,
            'lstm_operations': total_lstm_ops,
            'fc_operations': fc_ops,
            'cycles_per_inference': cycles_per_inference,
            'theoretical_inference_time_us': theoretical_inference_time_us,
            'input_window_bytes': input_window_bytes,
            'hidden_states_bytes': hidden_states_bytes,
            'intermediate_bytes': intermediate_bytes,
            'total_runtime_memory_bytes': total_runtime_memory_bytes,
            'theoretical_energy_per_inference_uj': theoretical_energy_per_inference_uj,
            'theoretical_max_sampling_rate_hz': 1e6 / theoretical_inference_time_us,
            'memory_feasible': (model_memory_bytes + total_runtime_memory_bytes) < self.arduino_specs['total_ram_bytes'],
            'total_memory_required_bytes': model_memory_bytes + total_runtime_memory_bytes
        }
        
        logger.info("📊 Theoretische Berechnungen geladen:")
        logger.info(f"  - Model Parameters: {total_params:,}")
        logger.info(f"  - Model Memory: {model_memory_bytes:,} bytes")
        logger.info(f"  - Runtime Memory: {total_runtime_memory_bytes:,} bytes")
        logger.info(f"  - Total Memory: {self.theoretical_data['total_memory_required_bytes']:,} bytes")
        logger.info(f"  - Theoretical Inference Time: {theoretical_inference_time_us:.0f} μs")
        logger.info(f"  - Memory Feasible: {self.theoretical_data['memory_feasible']}")
        
        return True
    
    def load_measured_data(self, measurement_file=None):
        """
        Lade gemessene Hardware-Daten von arduino_hardware_monitor.py
        """
        if measurement_file is None:
            # Suche nach neuester Messung
            measurement_files = list(Path('.').glob('arduino_hardware_analysis_*.json'))
            if not measurement_files:
                logger.error("Keine Hardware-Messdaten gefunden!")
                logger.info("Führe zuerst arduino_hardware_monitor.py aus")
                return False
            measurement_file = max(measurement_files, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(measurement_file, 'r') as f:
                measured_data = json.load(f)
            
            self.measured_data = {
                'total_samples': measured_data.get('total_samples', 0),
                'runtime_seconds': measured_data.get('runtime_seconds', 0),
                'sample_rate_hz': measured_data.get('sample_rate_hz', 0),
                'final_mae': measured_data.get('final_mae', 0),
                'hardware_monitoring_enabled': measured_data.get('hardware_monitoring_enabled', False)
            }
            
            # Inference Statistics
            if measured_data.get('inference_statistics'):
                inf_stats = measured_data['inference_statistics']
                self.measured_data.update({
                    'measured_mean_inference_us': inf_stats.get('mean_us', 0),
                    'measured_std_inference_us': inf_stats.get('std_us', 0),
                    'measured_min_inference_us': inf_stats.get('min_us', 0),
                    'measured_max_inference_us': inf_stats.get('max_us', 0),
                    'inference_samples': inf_stats.get('samples', 0)
                })
            
            # RAM Statistics
            if measured_data.get('ram_statistics'):
                ram_stats = measured_data['ram_statistics']
                self.measured_data.update({
                    'measured_mean_free_ram_bytes': ram_stats.get('mean_free_bytes', 0),
                    'measured_mean_used_ram_bytes': ram_stats.get('mean_used_bytes', 0),
                    'measured_mean_ram_usage_percent': ram_stats.get('mean_usage_percent', 0),
                    'measured_max_ram_usage_percent': ram_stats.get('max_usage_percent', 0)
                })
                
                if ram_stats.get('mean_free_bytes') and ram_stats.get('mean_used_bytes'):
                    self.measured_data['measured_total_ram_bytes'] = (
                        ram_stats['mean_free_bytes'] + ram_stats['mean_used_bytes']
                    )
            
            # CPU Statistics
            if measured_data.get('cpu_statistics'):
                cpu_stats = measured_data['cpu_statistics']
                self.measured_data.update({
                    'measured_mean_cpu_load_percent': cpu_stats.get('mean_load_percent', 0),
                    'measured_max_cpu_load_percent': cpu_stats.get('max_load_percent', 0),
                    'measured_cpu_load_std_percent': cpu_stats.get('std_load_percent', 0)
                })
            
            # Energy Statistics
            if measured_data.get('energy_estimation'):
                energy_stats = measured_data['energy_estimation']
                self.measured_data.update({
                    'measured_energy_per_inference_uj': energy_stats.get('energy_per_inference_uj', 0),
                    'measured_power_estimation_w': energy_stats.get('estimated_power_w', 0),
                    'measured_inference_time_s': energy_stats.get('avg_inference_time_s', 0)
                })
            
            logger.info(f"📊 Hardware-Messdaten geladen: {measurement_file}")
            logger.info(f"  - Total Samples: {self.measured_data['total_samples']}")
            logger.info(f"  - Runtime: {self.measured_data['runtime_seconds']:.1f} seconds")
            logger.info(f"  - Hardware Monitoring: {self.measured_data['hardware_monitoring_enabled']}")
            
            if self.measured_data['hardware_monitoring_enabled']:
                logger.info(f"  - Measured Inference Time: {self.measured_data.get('measured_mean_inference_us', 0):.0f} μs")
                logger.info(f"  - Measured RAM Usage: {self.measured_data.get('measured_mean_ram_usage_percent', 0):.1f}%")
                logger.info(f"  - Measured CPU Load: {self.measured_data.get('measured_mean_cpu_load_percent', 0):.1f}%")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Messdaten: {e}")
            return False
    
    def calculate_comparison_metrics(self):
        """
        Berechne Vergleichsmetriken zwischen Theorie und Messung
        """
        if not self.theoretical_data or not self.measured_data:
            logger.error("Theorie- oder Messdaten nicht verfügbar")
            return False
        
        self.comparison_results = {}
        
        # Inference Time Comparison
        if self.measured_data.get('measured_mean_inference_us'):
            theoretical_us = self.theoretical_data['theoretical_inference_time_us']
            measured_us = self.measured_data['measured_mean_inference_us']
            
            self.comparison_results['inference_time'] = {
                'theoretical_us': theoretical_us,
                'measured_us': measured_us,
                'absolute_error_us': abs(measured_us - theoretical_us),
                'relative_error_percent': abs(measured_us - theoretical_us) / theoretical_us * 100,
                'factor_difference': measured_us / theoretical_us,
                'measured_faster': measured_us < theoretical_us
            }
        
        # RAM Usage Comparison
        if self.measured_data.get('measured_total_ram_bytes'):
            theoretical_total_memory = self.theoretical_data['total_memory_required_bytes']
            measured_total_ram = self.measured_data['measured_total_ram_bytes']
            measured_used_ram = self.measured_data.get('measured_mean_used_ram_bytes', 0)
            
            self.comparison_results['memory_usage'] = {
                'theoretical_required_bytes': theoretical_total_memory,
                'measured_total_ram_bytes': measured_total_ram,
                'measured_used_ram_bytes': measured_used_ram,
                'theoretical_feasible': self.theoretical_data['memory_feasible'],
                'measured_feasible': measured_used_ram < measured_total_ram,
                'memory_overhead_bytes': measured_used_ram - theoretical_total_memory,
                'memory_overhead_percent': ((measured_used_ram - theoretical_total_memory) / 
                                          theoretical_total_memory * 100) if theoretical_total_memory > 0 else 0
            }
        
        # Energy Comparison
        if (self.measured_data.get('measured_energy_per_inference_uj') and 
            self.theoretical_data.get('theoretical_energy_per_inference_uj')):
            theoretical_uj = self.theoretical_data['theoretical_energy_per_inference_uj']
            measured_uj = self.measured_data['measured_energy_per_inference_uj']
            
            self.comparison_results['energy_consumption'] = {
                'theoretical_energy_uj': theoretical_uj,
                'measured_energy_uj': measured_uj,
                'energy_error_uj': abs(measured_uj - theoretical_uj),
                'energy_error_percent': abs(measured_uj - theoretical_uj) / theoretical_uj * 100,
                'measured_more_efficient': measured_uj < theoretical_uj
            }
        
        # Performance Analysis
        if self.measured_data.get('measured_mean_inference_us'):
            measured_max_rate = 1e6 / self.measured_data['measured_mean_inference_us']
            theoretical_max_rate = self.theoretical_data['theoretical_max_sampling_rate_hz']
            
            self.comparison_results['performance'] = {
                'theoretical_max_rate_hz': theoretical_max_rate,
                'measured_max_rate_hz': measured_max_rate,
                'rate_improvement_factor': measured_max_rate / theoretical_max_rate,
                'measured_sample_rate_hz': self.measured_data['sample_rate_hz'],
                'headroom_percent': (measured_max_rate - self.measured_data['sample_rate_hz']) / measured_max_rate * 100
            }
        
        logger.info("🔍 Vergleichsanalyse abgeschlossen")
        return True
    
    def create_validation_plots(self, save_dir='hardware_validation_results'):
        """
        Erstelle wissenschaftliche Validierungs-Plots
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # Plot 1: Inference Time Comparison
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        if 'inference_time' in self.comparison_results:
            inf_comp = self.comparison_results['inference_time']
            
            # Bar chart: Theoretical vs Measured
            categories = ['Theoretical\nCalculation', 'Measured\nHardware']
            values = [inf_comp['theoretical_us'], inf_comp['measured_us']]
            colors = ['lightblue', 'orange']
            
            bars = ax1.bar(categories, values, color=colors, alpha=0.8)
            ax1.set_ylabel('Inference Time (μs)')
            ax1.set_title('Inference Time: Theory vs Reality')
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.0f} μs', ha='center', va='bottom', fontweight='bold')
            
            # Error analysis
            error_text = f"Error: {inf_comp['absolute_error_us']:.0f} μs ({inf_comp['relative_error_percent']:.1f}%)\n"
            error_text += f"Factor: {inf_comp['factor_difference']:.2f}x\n"
            error_text += f"Result: {'✓ Faster' if inf_comp['measured_faster'] else '✗ Slower'}"
            
            ax1.text(0.02, 0.98, error_text, transform=ax1.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                    fontsize=10)
        else:
            ax1.text(0.5, 0.5, 'Inference Time Data\nNot Available', ha='center', va='center', 
                    transform=ax1.transAxes, fontsize=14, style='italic')
            ax1.set_title('Inference Time: Theory vs Reality')
        
        # Plot 2: Memory Usage Comparison
        if 'memory_usage' in self.comparison_results:
            mem_comp = self.comparison_results['memory_usage']
            
            # Stacked bar chart
            labels = ['Theoretical\nRequirement', 'Measured\nUsage', 'Available\nRAM']
            theoretical_req = mem_comp['theoretical_required_bytes']
            measured_used = mem_comp['measured_used_ram_bytes']
            measured_total = mem_comp['measured_total_ram_bytes']
            
            # Convert to KB for readability
            theoretical_kb = theoretical_req / 1024
            used_kb = measured_used / 1024
            total_kb = measured_total / 1024
            free_kb = total_kb - used_kb
            
            x = np.arange(len(labels))
            ax2.bar(x[0], theoretical_kb, color='lightblue', alpha=0.8, label='Required')
            ax2.bar(x[1], used_kb, color='red', alpha=0.8, label='Used')
            ax2.bar(x[1], free_kb, bottom=used_kb, color='green', alpha=0.8, label='Free')
            ax2.bar(x[2], total_kb, color='gray', alpha=0.8, label='Total Available')
            
            ax2.set_ylabel('Memory (KB)')
            ax2.set_title('Memory Usage: Theory vs Reality')
            ax2.set_xticks(x)
            ax2.set_xticklabels(labels)
            ax2.legend()
            
            # Feasibility info
            feasibility_text = f"Theoretical: {'✓ Feasible' if mem_comp['theoretical_feasible'] else '✗ Not Feasible'}\n"
            feasibility_text += f"Measured: {'✓ Feasible' if mem_comp['measured_feasible'] else '✗ Not Feasible'}\n"
            feasibility_text += f"Overhead: {mem_comp['memory_overhead_percent']:.1f}%"
            
            ax2.text(0.02, 0.98, feasibility_text, transform=ax2.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if mem_comp['measured_feasible'] else "lightcoral", alpha=0.7),
                    fontsize=10)
        else:
            ax2.text(0.5, 0.5, 'Memory Usage Data\nNot Available', ha='center', va='center', 
                    transform=ax2.transAxes, fontsize=14, style='italic')
            ax2.set_title('Memory Usage: Theory vs Reality')
        
        # Plot 3: Energy Consumption Comparison
        if 'energy_consumption' in self.comparison_results:
            energy_comp = self.comparison_results['energy_consumption']
            
            categories = ['Theoretical\nEstimation', 'Measured\nHardware']
            values = [energy_comp['theoretical_energy_uj'], energy_comp['measured_energy_uj']]
            colors = ['lightblue', 'orange']
            
            bars = ax3.bar(categories, values, color=colors, alpha=0.8)
            ax3.set_ylabel('Energy per Inference (μJ)')
            ax3.set_title('Energy Consumption: Theory vs Reality')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.1f} μJ', ha='center', va='bottom', fontweight='bold')
            
            energy_text = f"Error: {energy_comp['energy_error_uj']:.1f} μJ ({energy_comp['energy_error_percent']:.1f}%)\n"
            energy_text += f"Result: {'✓ More Efficient' if energy_comp['measured_more_efficient'] else '✗ Less Efficient'}"
            
            ax3.text(0.02, 0.98, energy_text, transform=ax3.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen" if energy_comp['measured_more_efficient'] else "lightyellow", alpha=0.7),
                    fontsize=10)
        else:
            ax3.text(0.5, 0.5, 'Energy Data\nNot Available', ha='center', va='center', 
                    transform=ax3.transAxes, fontsize=14, style='italic')
            ax3.set_title('Energy Consumption: Theory vs Reality')
        
        # Plot 4: Performance Summary
        if 'performance' in self.comparison_results:
            perf_comp = self.comparison_results['performance']
            
            # Max sampling rates
            categories = ['Theoretical\nMax Rate', 'Measured\nMax Rate', 'Actual\nSample Rate']
            values = [perf_comp['theoretical_max_rate_hz'], 
                     perf_comp['measured_max_rate_hz'],
                     perf_comp['measured_sample_rate_hz']]
            colors = ['lightblue', 'orange', 'green']
            
            bars = ax4.bar(categories, values, color=colors, alpha=0.8)
            ax4.set_ylabel('Sampling Rate (Hz)')
            ax4.set_title('Performance: Theoretical vs Measured')
            
            for bar, value in zip(bars, values):
                height = bar.get_height()
                ax4.text(bar.get_x() + bar.get_width()/2., height + max(values)*0.01,
                        f'{value:.2f} Hz', ha='center', va='bottom', fontweight='bold')
            
            perf_text = f"Improvement: {perf_comp['rate_improvement_factor']:.2f}x\n"
            perf_text += f"Headroom: {perf_comp['headroom_percent']:.1f}%"
            
            ax4.text(0.02, 0.98, perf_text, transform=ax4.transAxes, va='top', ha='left',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7),
                    fontsize=10)
        else:
            ax4.text(0.5, 0.5, 'Performance Data\nNot Available', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14, style='italic')
            ax4.set_title('Performance: Theoretical vs Measured')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'hardware_validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot 2: Error Analysis
        if any(key in self.comparison_results for key in ['inference_time', 'memory_usage', 'energy_consumption']):
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            
            # Relative errors
            metrics = []
            errors = []
            colors = []
            
            if 'inference_time' in self.comparison_results:
                metrics.append('Inference\nTime')
                errors.append(self.comparison_results['inference_time']['relative_error_percent'])
                colors.append('orange')
            
            if 'memory_usage' in self.comparison_results:
                metrics.append('Memory\nOverhead')
                errors.append(abs(self.comparison_results['memory_usage']['memory_overhead_percent']))
                colors.append('red')
            
            if 'energy_consumption' in self.comparison_results:
                metrics.append('Energy\nConsumption')
                errors.append(self.comparison_results['energy_consumption']['energy_error_percent'])
                colors.append('purple')
            
            if metrics:
                bars = ax1.bar(metrics, errors, color=colors, alpha=0.8)
                ax1.set_ylabel('Relative Error (%)')
                ax1.set_title('Validation Errors: Theory vs Hardware')
                
                for bar, error in zip(bars, errors):
                    height = bar.get_height()
                    ax1.text(bar.get_x() + bar.get_width()/2., height + max(errors)*0.01,
                            f'{error:.1f}%', ha='center', va='bottom', fontweight='bold')
                
                # Error interpretation
                max_error = max(errors)
                if max_error < 10:
                    accuracy_text = "Validation Result: ✓ EXCELLENT\nTheory matches hardware very well"
                    box_color = "lightgreen"
                elif max_error < 25:
                    accuracy_text = "Validation Result: ✓ GOOD\nTheory reasonably matches hardware"
                    box_color = "lightyellow"
                else:
                    accuracy_text = "Validation Result: ⚠ NEEDS IMPROVEMENT\nSignificant theory-hardware gap"
                    box_color = "lightcoral"
                
                ax1.text(0.02, 0.98, accuracy_text, transform=ax1.transAxes, va='top', ha='left',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=box_color, alpha=0.7),
                        fontsize=12, fontweight='bold')
            
            # Feasibility comparison
            feasibility_data = []
            
            if self.theoretical_data.get('memory_feasible') is not None:
                theoretical_feasible = self.theoretical_data['memory_feasible']
                measured_feasible = self.comparison_results.get('memory_usage', {}).get('measured_feasible', None)
                
                if measured_feasible is not None:
                    categories = ['Theoretical\nFeasibility', 'Measured\nFeasibility']
                    values = [1 if theoretical_feasible else 0, 1 if measured_feasible else 0]
                    colors = ['green' if v == 1 else 'red' for v in values]
                    
                    bars = ax2.bar(categories, values, color=colors, alpha=0.8)
                    ax2.set_ylabel('Feasible (1) / Not Feasible (0)')
                    ax2.set_title('Deployment Feasibility Assessment')
                    ax2.set_ylim(0, 1.2)
                    
                    for bar, value, feasible in zip(bars, values, [theoretical_feasible, measured_feasible]):
                        ax2.text(bar.get_x() + bar.get_width()/2., value + 0.05,
                                '✓ Feasible' if feasible else '✗ Not Feasible',
                                ha='center', va='bottom', fontweight='bold')
                    
                    # Overall conclusion
                    if theoretical_feasible and measured_feasible:
                        conclusion = "✅ DEPLOYMENT READY\nBoth theory and hardware confirm feasibility"
                        conclusion_color = "lightgreen"
                    elif measured_feasible:
                        conclusion = "⚠️ THEORY CONSERVATIVE\nHardware works despite theoretical concerns"
                        conclusion_color = "lightyellow"
                    else:
                        conclusion = "❌ DEPLOYMENT NOT READY\nHardware limitations confirmed"
                        conclusion_color = "lightcoral"
                    
                    ax2.text(0.5, 0.5, conclusion, transform=ax2.transAxes, va='center', ha='center',
                            bbox=dict(boxstyle="round,pad=0.3", facecolor=conclusion_color, alpha=0.8),
                            fontsize=12, fontweight='bold')
            
            plt.tight_layout()
            plt.savefig(save_dir / 'hardware_validation_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"📊 Validierungs-Plots erstellt in: {save_dir}")
    
    def save_validation_report(self, save_dir='hardware_validation_results'):
        """
        Speichere detaillierten Validierungs-Report
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        # Kombiniere alle Daten
        validation_report = {
            'timestamp': datetime.now().isoformat(),
            'arduino_specifications': self.arduino_specs,
            'theoretical_calculations': self.theoretical_data,
            'measured_hardware_data': self.measured_data,
            'comparison_results': self.comparison_results,
            'validation_summary': self.generate_validation_summary()
        }
        
        # Speichere als JSON
        report_file = save_dir / f'hardware_validation_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        # Erstelle auch Markdown-Report für Menschen
        markdown_report = self.generate_markdown_report()
        markdown_file = save_dir / 'hardware_validation_summary.md'
        with open(markdown_file, 'w') as f:
            f.write(markdown_report)
        
        logger.info(f"📄 Validierungs-Report gespeichert:")
        logger.info(f"  - JSON: {report_file}")
        logger.info(f"  - Markdown: {markdown_file}")
        
        return report_file, markdown_file
    
    def generate_validation_summary(self):
        """
        Generiere kompakte Validierungs-Zusammenfassung
        """
        summary = {
            'overall_validation_status': 'unknown',
            'key_findings': [],
            'recommendations': [],
            'deployment_ready': False
        }
        
        if 'inference_time' in self.comparison_results:
            inf_comp = self.comparison_results['inference_time']
            if inf_comp['relative_error_percent'] < 25:
                summary['key_findings'].append(f"✓ Inference time prediction accurate ({inf_comp['relative_error_percent']:.1f}% error)")
            else:
                summary['key_findings'].append(f"⚠ Inference time prediction off by {inf_comp['relative_error_percent']:.1f}%")
        
        if 'memory_usage' in self.comparison_results:
            mem_comp = self.comparison_results['memory_usage']
            if mem_comp['measured_feasible']:
                summary['key_findings'].append("✓ Memory requirements feasible on hardware")
                summary['deployment_ready'] = True
            else:
                summary['key_findings'].append("❌ Memory requirements exceed hardware capacity")
                summary['deployment_ready'] = False
        
        if 'energy_consumption' in self.comparison_results:
            energy_comp = self.comparison_results['energy_consumption']
            if energy_comp['measured_more_efficient']:
                summary['key_findings'].append("✓ Hardware more energy efficient than predicted")
            else:
                summary['key_findings'].append("⚠ Hardware consumes more energy than predicted")
        
        # Overall status
        if len([f for f in summary['key_findings'] if f.startswith('✓')]) >= 2:
            summary['overall_validation_status'] = 'excellent'
        elif len([f for f in summary['key_findings'] if f.startswith('❌')]) > 0:
            summary['overall_validation_status'] = 'needs_improvement'
        else:
            summary['overall_validation_status'] = 'good'
        
        # Recommendations
        if summary['deployment_ready']:
            summary['recommendations'].append("Proceed with hardware deployment")
            summary['recommendations'].append("Consider optimizations for better performance")
        else:
            summary['recommendations'].append("Optimize model or use more powerful hardware")
            summary['recommendations'].append("Consider model quantization or pruning")
        
        return summary
    
    def generate_markdown_report(self):
        """
        Generiere Human-readable Markdown Report
        """
        report = f"""# Hardware Validation Report

Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Arduino Specifications
- **Model**: {self.arduino_specs['model']}
- **CPU Frequency**: {self.arduino_specs['cpu_frequency_mhz']} MHz
- **Total RAM**: {self.arduino_specs['total_ram_bytes']:,} bytes
- **Flash Memory**: {self.arduino_specs['flash_bytes']:,} bytes

## Validation Summary

"""
        
        summary = self.generate_validation_summary()
        
        report += f"**Overall Status**: {summary['overall_validation_status'].upper()}\n\n"
        report += f"**Deployment Ready**: {'✅ YES' if summary['deployment_ready'] else '❌ NO'}\n\n"
        
        report += "### Key Findings\n"
        for finding in summary['key_findings']:
            report += f"- {finding}\n"
        
        report += "\n### Recommendations\n"
        for rec in summary['recommendations']:
            report += f"- {rec}\n"
        
        # Detailed comparisons
        if 'inference_time' in self.comparison_results:
            inf_comp = self.comparison_results['inference_time']
            report += f"""
## Inference Time Analysis

| Metric | Theoretical | Measured | Error |
|--------|-------------|----------|-------|
| Inference Time | {inf_comp['theoretical_us']:.0f} μs | {inf_comp['measured_us']:.0f} μs | {inf_comp['relative_error_percent']:.1f}% |
| Factor Difference | - | {inf_comp['factor_difference']:.2f}x | - |
| Performance | - | {'✓ Faster' if inf_comp['measured_faster'] else '✗ Slower'} | - |
"""
        
        if 'memory_usage' in self.comparison_results:
            mem_comp = self.comparison_results['memory_usage']
            report += f"""
## Memory Usage Analysis

| Metric | Value |
|--------|-------|
| Theoretical Requirement | {mem_comp['theoretical_required_bytes']:,} bytes |
| Measured Total RAM | {mem_comp['measured_total_ram_bytes']:,} bytes |
| Measured Used RAM | {mem_comp['measured_used_ram_bytes']:,} bytes |
| Memory Overhead | {mem_comp['memory_overhead_percent']:.1f}% |
| Theoretical Feasible | {'✅ YES' if mem_comp['theoretical_feasible'] else '❌ NO'} |
| Measured Feasible | {'✅ YES' if mem_comp['measured_feasible'] else '❌ NO'} |
"""
        
        return report
    
    def run_complete_validation(self, measurement_file=None):
        """
        Führe komplette Hardware-Validierung durch
        """
        logger.info("🔍 Starte komplette Hardware-Validierung...")
        
        # 1. Lade theoretische Berechnungen
        if not self.load_theoretical_calculations():
            logger.error("Theoretische Berechnungen fehlgeschlagen")
            return False
        
        # 2. Lade Hardware-Messungen
        if not self.load_measured_data(measurement_file):
            logger.error("Hardware-Messungen laden fehlgeschlagen")
            return False
        
        # 3. Berechne Vergleiche
        if not self.calculate_comparison_metrics():
            logger.error("Vergleichsberechnungen fehlgeschlagen")
            return False
        
        # 4. Erstelle Plots
        self.create_validation_plots()
        
        # 5. Speichere Report
        json_file, md_file = self.save_validation_report()
        
        # 6. Zeige Zusammenfassung
        summary = self.generate_validation_summary()
        
        logger.info("🎉 Hardware-Validierung abgeschlossen!")
        logger.info(f"📊 Status: {summary['overall_validation_status'].upper()}")
        logger.info(f"🚀 Deployment Ready: {'YES' if summary['deployment_ready'] else 'NO'}")
        
        print("\n" + "="*60)
        print("HARDWARE VALIDATION RESULTS")
        print("="*60)
        
        for finding in summary['key_findings']:
            print(f"  {finding}")
        
        print(f"\n🎯 CONCLUSION: {summary['overall_validation_status'].upper()}")
        print(f"🚀 DEPLOYMENT: {'READY' if summary['deployment_ready'] else 'NOT READY'}")
        
        return True

def main():
    """Hauptfunktion für Hardware-Validierung"""
    try:
        logger.info("=" * 80)
        logger.info("Hardware Validation & Comparison Tool")
        logger.info("=" * 80)
        
        # Hardware Validation Tool erstellen
        validator = HardwareValidationComparison()
        
        # Komplette Validierung durchführen
        success = validator.run_complete_validation()
        
        if success:
            logger.info("✅ Hardware-Validierung erfolgreich abgeschlossen")
        else:
            logger.error("❌ Hardware-Validierung fehlgeschlagen")
        
    except Exception as e:
        logger.error(f"Fehler in Hardware-Validierung: {e}")
    finally:
        logger.info("Hardware Validation Tool beendet")

if __name__ == "__main__":
    main()
