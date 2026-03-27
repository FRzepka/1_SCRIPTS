"""
Hardware Validation & Comparison Tool
====================================

Vergleicht theoretische Berechnungen mit echten Arduino Hardware-Messungen:
- Theoretical vs Measured Inference Times
- Theoretical vs Measured RAM Usage  
- Theoretical vs Measured Energy Consumption
- Scientific Plots for Paper Publication
- Deployment Feasibility Assessment

Verwendet Daten aus:
1. Theoretische Berechnungen (wie im original Script)
2. Echte Hardware-Messungen (aus arduino_hardware_monitor.py)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from datetime import datetime

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# === THEORETICAL MICROCONTROLLER SPECIFICATIONS ===
# Basierend auf dem originalen theoretical Script
THEORETICAL_STM32_RAM_BYTES = 1024 * 1024  # 1MB RAM
THEORETICAL_STM32_FLASH_BYTES = 2 * 1024 * 1024  # 2MB Flash 
THEORETICAL_STM32_CORE_FREQ_HZ = 480_000_000  # 480MHz Cortex-M7
THEORETICAL_STM32_POWER_ACTIVE_W = 0.5  # Geschätzte aktive Leistung in Watt

# === ARDUINO ACTUAL SPECIFICATIONS ===
# Wird aus Hardware-Messungen ermittelt
ARDUINO_ACTUAL_RAM_BYTES = 32 * 1024  # 32KB RAM (Arduino Uno R4)
ARDUINO_ACTUAL_FLASH_BYTES = 256 * 1024  # 256KB Flash
ARDUINO_ACTUAL_CORE_FREQ_HZ = 16_000_000  # 16MHz (Arduino Uno R4)

class HardwareValidationComparison:
    """
    Hardware Validation & Comparison Tool
    ====================================
    
    Analysiert und vergleicht:
    - Theoretische vs. gemessene Performance
    - Hardware-Spezifikations-Validierung
    - Deployment-Feasibility Assessment
    """
    
    def __init__(self):
        self.theoretical_data = {}
        self.measured_data = {}
        self.comparison_results = {}
        
        logger.info("🔬 Hardware Validation & Comparison Tool initialisiert")
    
    def calculate_theoretical_performance(self, config):
        """Berechnet theoretische Performance-Metriken"""
        logger.info("🧮 Berechne theoretische Performance...")
        
        # LSTM Architecture aus Config
        input_size = 4  # V, I, SOH, Q_c
        hidden_size = getattr(config, 'HIDDEN_SIZE', 32)
        num_layers = getattr(config, 'NUM_LAYERS', 2)
        sequence_length = getattr(config, 'SEQUENCE_LENGTH', 720)
        
        # Theoretische MAC Operations pro Inferenz
        # LSTM: 8 operations per gate * 4 gates * sequence_length * layers
        lstm_ops_per_timestep = 8 * input_size * hidden_size + 8 * hidden_size * hidden_size
        total_lstm_ops = lstm_ops_per_timestep * sequence_length * num_layers
        
        # MLP: Linear layers
        mlp_ops = hidden_size * 32 + 32 * 32 + 32 * 1  # 32->32->32->1
        
        total_ops = total_lstm_ops + mlp_ops
        
        # Theoretical Timing (pessimistisch: 2 Zyklen pro MAC)
        cycles_per_inference = total_ops * 2
        theoretical_inference_time_us = (cycles_per_inference / THEORETICAL_STM32_CORE_FREQ_HZ) * 1_000_000
        
        # Theoretical Memory Usage
        model_params = self.estimate_model_parameters(input_size, hidden_size, num_layers)
        model_memory_bytes = model_params * 4  # 32-bit float
        
        # Window Memory
        window_memory_bytes = sequence_length * input_size * 4  # Input window
        hidden_states_memory = num_layers * hidden_size * 2 * 4  # h and c states
        intermediate_memory = sequence_length * hidden_size * 4  # LSTM outputs
        
        total_inference_memory = model_memory_bytes + window_memory_bytes + hidden_states_memory + intermediate_memory
        
        # Energy Consumption
        energy_per_inference_j = THEORETICAL_STM32_POWER_ACTIVE_W * (theoretical_inference_time_us / 1_000_000)
        
        self.theoretical_data = {
            'total_mac_operations': total_ops,
            'lstm_operations': total_lstm_ops,
            'mlp_operations': mlp_ops,
            'inference_time_us': theoretical_inference_time_us,
            'model_parameters': model_params,
            'model_memory_bytes': model_memory_bytes,
            'window_memory_bytes': window_memory_bytes,
            'total_inference_memory_bytes': total_inference_memory,
            'energy_per_inference_j': energy_per_inference_j,
            'energy_per_inference_uj': energy_per_inference_j * 1_000_000,
            'target_cpu_freq_hz': THEORETICAL_STM32_CORE_FREQ_HZ,
            'target_ram_bytes': THEORETICAL_STM32_RAM_BYTES,
            'target_flash_bytes': THEORETICAL_STM32_FLASH_BYTES
        }
        
        logger.info(f"✅ Theoretische Berechnungen abgeschlossen:")
        logger.info(f"  - Total MAC Ops: {total_ops:,}")
        logger.info(f"  - Theoretical Inference Time: {theoretical_inference_time_us:.0f}μs")
        logger.info(f"  - Model Memory: {model_memory_bytes/1024:.1f}KB")
        logger.info(f"  - Total Memory: {total_inference_memory/1024:.1f}KB")
        
        return self.theoretical_data
    
    def estimate_model_parameters(self, input_size, hidden_size, num_layers):
        """Schätzt Anzahl der Modell-Parameter"""
        # LSTM Parameters
        lstm_params = 0
        for layer in range(num_layers):
            if layer == 0:
                # Erste Schicht: input_size -> hidden_size
                lstm_params += 4 * (input_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
            else:
                # Weitere Schichten: hidden_size -> hidden_size
                lstm_params += 4 * (hidden_size * hidden_size + hidden_size * hidden_size + 2 * hidden_size)
        
        # MLP Parameters (32->32->32->1)
        mlp_params = (hidden_size * 32 + 32) + (32 * 32 + 32) + (32 * 1 + 1)
        
        total_params = lstm_params + mlp_params
        return total_params
    
    def load_measured_hardware_data(self, hardware_results_dir):
        """Lädt gemessene Hardware-Daten"""
        results_path = Path(hardware_results_dir)
        
        if not results_path.exists():
            logger.error(f"Hardware results directory nicht gefunden: {results_path}")
            return False
        
        logger.info(f"📊 Lade gemessene Hardware-Daten: {results_path}")
        
        try:
            # Hardware Performance Metrics
            hardware_metrics_file = results_path / "hardware_performance_metrics.csv"
            if hardware_metrics_file.exists():
                hardware_df = pd.read_csv(hardware_metrics_file)
                
                self.measured_data = {
                    'inference_times_us': hardware_df['inference_time_us'].tolist(),
                    'free_ram_bytes': hardware_df['free_ram_bytes'].tolist(),
                    'used_ram_bytes': hardware_df['used_ram_bytes'].tolist(),
                    'total_ram_bytes': hardware_df['total_ram_bytes'].tolist(),
                    'ram_usage_percent': hardware_df['ram_fragmentation_percent'].tolist(),
                    'cpu_load_percent': hardware_df['cpu_load_percent'].tolist(),
                    'temperature_celsius': hardware_df['temperature_celsius'].tolist(),
                    
                    # Statistiken
                    'avg_inference_time_us': np.mean(hardware_df['inference_time_us']),
                    'min_inference_time_us': np.min(hardware_df['inference_time_us']),
                    'max_inference_time_us': np.max(hardware_df['inference_time_us']),
                    'std_inference_time_us': np.std(hardware_df['inference_time_us']),
                    
                    'avg_free_ram_bytes': np.mean(hardware_df['free_ram_bytes']),
                    'avg_used_ram_bytes': np.mean(hardware_df['used_ram_bytes']),
                    'avg_total_ram_bytes': np.mean(hardware_df['total_ram_bytes']),
                    'avg_ram_usage_percent': np.mean(hardware_df['ram_fragmentation_percent']),
                    
                    'avg_cpu_load_percent': np.mean(hardware_df['cpu_load_percent']),
                    'avg_temperature_celsius': np.mean(hardware_df['temperature_celsius']),
                    
                    'total_measurements': len(hardware_df)
                }
                
                logger.info(f"✅ Hardware-Metriken geladen: {len(hardware_df)} Messungen")
                logger.info(f"  - Avg Inference Time: {self.measured_data['avg_inference_time_us']:.0f}μs")
                logger.info(f"  - Avg RAM Usage: {self.measured_data['avg_ram_usage_percent']:.1f}%")
                logger.info(f"  - Avg CPU Load: {self.measured_data['avg_cpu_load_percent']:.1f}%")
                
            # Summary JSON
            summary_file = results_path / "hardware_performance_summary.json"
            if summary_file.exists():
                with open(summary_file, 'r') as f:
                    summary_data = json.load(f)
                    self.measured_data.update(summary_data.get('hardware_performance_metrics', {}))
                    
                logger.info("✅ Hardware Summary geladen")
            
            return True
            
        except Exception as e:
            logger.error(f"Fehler beim Laden der Hardware-Daten: {e}")
            return False
    
    def perform_comparison_analysis(self):
        """Führt Vergleichsanalyse durch"""
        if not self.theoretical_data or not self.measured_data:
            logger.error("Theoretical oder Measured Data fehlt!")
            return
        
        logger.info("🔬 Führe Vergleichsanalyse durch...")
        
        # Performance Vergleich
        theoretical_inference_us = self.theoretical_data['inference_time_us']
        measured_inference_us = self.measured_data['avg_inference_time_us']
        inference_ratio = measured_inference_us / theoretical_inference_us if theoretical_inference_us > 0 else 0
        
        # Memory Vergleich
        theoretical_memory_kb = self.theoretical_data['total_inference_memory_bytes'] / 1024
        measured_memory_kb = self.measured_data['avg_used_ram_bytes'] / 1024 if 'avg_used_ram_bytes' in self.measured_data else 0
        memory_ratio = measured_memory_kb / theoretical_memory_kb if theoretical_memory_kb > 0 else 0
        
        # Hardware Platform Vergleich
        theoretical_target_ram_mb = self.theoretical_data['target_ram_bytes'] / (1024 * 1024)
        actual_arduino_ram_kb = self.measured_data['avg_total_ram_bytes'] / 1024 if 'avg_total_ram_bytes' in self.measured_data else ARDUINO_ACTUAL_RAM_BYTES / 1024
        
        theoretical_target_freq_mhz = self.theoretical_data['target_cpu_freq_hz'] / 1_000_000
        actual_arduino_freq_mhz = ARDUINO_ACTUAL_CORE_FREQ_HZ / 1_000_000
        
        # Feasibility Assessment
        ram_utilization_percent = (measured_memory_kb / (actual_arduino_ram_kb)) * 100 if actual_arduino_ram_kb > 0 else 0
        cpu_feasible = measured_inference_us < 1_000_000  # < 1 second
        ram_feasible = ram_utilization_percent < 80  # < 80% RAM usage
        
        self.comparison_results = {
            'performance_comparison': {
                'theoretical_inference_us': theoretical_inference_us,
                'measured_inference_us': measured_inference_us,
                'inference_time_ratio': inference_ratio,
                'performance_delta_percent': ((measured_inference_us - theoretical_inference_us) / theoretical_inference_us * 100) if theoretical_inference_us > 0 else 0
            },
            'memory_comparison': {
                'theoretical_memory_kb': theoretical_memory_kb,
                'measured_memory_kb': measured_memory_kb,
                'memory_ratio': memory_ratio,
                'memory_delta_percent': ((measured_memory_kb - theoretical_memory_kb) / theoretical_memory_kb * 100) if theoretical_memory_kb > 0 else 0
            },
            'platform_comparison': {
                'theoretical_target_platform': 'STM32H757',
                'actual_platform': 'Arduino Uno R4',
                'theoretical_ram_mb': theoretical_target_ram_mb,
                'actual_ram_kb': actual_arduino_ram_kb,
                'theoretical_freq_mhz': theoretical_target_freq_mhz,
                'actual_freq_mhz': actual_arduino_freq_mhz,
                'ram_scaling_factor': theoretical_target_ram_mb * 1024 / actual_arduino_ram_kb,
                'freq_scaling_factor': theoretical_target_freq_mhz / actual_arduino_freq_mhz
            },
            'feasibility_assessment': {
                'ram_utilization_percent': ram_utilization_percent,
                'cpu_load_percent': self.measured_data.get('avg_cpu_load_percent', 0),
                'inference_time_feasible': cpu_feasible,
                'ram_usage_feasible': ram_feasible,
                'overall_feasible': cpu_feasible and ram_feasible,
                'deployment_recommendation': self.get_deployment_recommendation(cpu_feasible, ram_feasible, ram_utilization_percent)
            },
            'accuracy_assessment': {
                'theoretical_model_accurate': abs(inference_ratio - 1.0) < 0.5,  # Within 50%
                'prediction_accuracy_rating': self.rate_prediction_accuracy(inference_ratio, memory_ratio)
            }
        }
        
        logger.info("✅ Vergleichsanalyse abgeschlossen:")
        logger.info(f"  - Inference Time: Theoretical {theoretical_inference_us:.0f}μs vs Measured {measured_inference_us:.0f}μs")
        logger.info(f"  - Memory Usage: Theoretical {theoretical_memory_kb:.1f}KB vs Measured {measured_memory_kb:.1f}KB")
        logger.info(f"  - RAM Utilization: {ram_utilization_percent:.1f}%")
        logger.info(f"  - Overall Feasible: {self.comparison_results['feasibility_assessment']['overall_feasible']}")
        
        return self.comparison_results
    
    def get_deployment_recommendation(self, cpu_feasible, ram_feasible, ram_utilization):
        """Gibt Deployment-Empfehlung basierend auf Analyse"""
        if cpu_feasible and ram_feasible and ram_utilization < 50:
            return "EXCELLENT - Ready for deployment with room for optimization"
        elif cpu_feasible and ram_feasible and ram_utilization < 80:
            return "GOOD - Deployment possible with careful resource management"
        elif cpu_feasible and not ram_feasible:
            return "MEMORY_LIMITED - Consider model compression or larger MCU"
        elif not cpu_feasible and ram_feasible:
            return "CPU_LIMITED - Consider faster MCU or model optimization"
        else:
            return "NOT_FEASIBLE - Major optimizations or hardware upgrade required"
    
    def rate_prediction_accuracy(self, inference_ratio, memory_ratio):
        """Bewertet Genauigkeit der theoretischen Vorhersagen"""
        inference_accuracy = 1.0 - min(abs(inference_ratio - 1.0), 1.0)
        memory_accuracy = 1.0 - min(abs(memory_ratio - 1.0), 1.0) if memory_ratio > 0 else 0
        
        overall_accuracy = (inference_accuracy + memory_accuracy) / 2
        
        if overall_accuracy > 0.8:
            return "EXCELLENT (>80%)"
        elif overall_accuracy > 0.6:
            return "GOOD (60-80%)"
        elif overall_accuracy > 0.4:
            return "MODERATE (40-60%)"
        else:
            return "POOR (<40%)"
    
    def create_comparison_plots(self, save_dir):
        """Erstellt wissenschaftliche Vergleichsplots"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        logger.info("📊 Erstelle Vergleichsplots...")
        
        # Set scientific plot style
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
        
        # === PLOT 1: Performance Comparison ===
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Inference Time Comparison
        categories = ['Theoretical\\n(STM32H757)', 'Measured\\n(Arduino R4)']
        inference_times = [
            self.comparison_results['performance_comparison']['theoretical_inference_us'],
            self.comparison_results['performance_comparison']['measured_inference_us']
        ]
        
        bars1 = ax1.bar(categories, inference_times, color=['#4CAF50', '#FF5722'], alpha=0.8)
        ax1.set_ylabel('Inference Time (μs)')
        ax1.set_title('Inference Time: Theoretical vs Measured')
        ax1.set_yscale('log')
        
        # Add value labels
        for bar, value in zip(bars1, inference_times):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height * 1.1,
                    f'{value:.0f}μs', ha='center', va='bottom', fontweight='bold')
        
        # Memory Usage Comparison
        memory_usage = [
            self.comparison_results['memory_comparison']['theoretical_memory_kb'],
            self.comparison_results['memory_comparison']['measured_memory_kb']
        ]
        
        bars2 = ax2.bar(categories, memory_usage, color=['#2196F3', '#FF9800'], alpha=0.8)
        ax2.set_ylabel('Memory Usage (KB)')
        ax2.set_title('Memory Usage: Theoretical vs Measured')
        
        for bar, value in zip(bars2, memory_usage):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + max(memory_usage)*0.02,
                    f'{value:.1f}KB', ha='center', va='bottom', fontweight='bold')
        
        # Platform Specifications Comparison
        platforms = ['STM32H757\\n(Theoretical)', 'Arduino R4\\n(Actual)']
        ram_sizes = [
            self.comparison_results['platform_comparison']['theoretical_ram_mb'] * 1024,  # Convert to KB
            self.comparison_results['platform_comparison']['actual_ram_kb']
        ]
        cpu_freqs = [
            self.comparison_results['platform_comparison']['theoretical_freq_mhz'],
            self.comparison_results['platform_comparison']['actual_freq_mhz']
        ]
        
        ax3_twin = ax3.twinx()
        
        # RAM bars
        bars3 = ax3.bar([p + ' (RAM)' for p in platforms], ram_sizes, 
                       color=['#9C27B0', '#E91E63'], alpha=0.8, width=0.6)
        ax3.set_ylabel('RAM (KB)', color='#9C27B0')
        ax3.set_title('Hardware Platform Comparison')
        ax3.tick_params(axis='x', rotation=45)
        
        # CPU frequency line
        cpu_x = [0, 1]
        line = ax3_twin.plot(cpu_x, cpu_freqs, 'o-', color='#FF5722', linewidth=3, markersize=8, label='CPU Frequency')
        ax3_twin.set_ylabel('CPU Frequency (MHz)', color='#FF5722')
        ax3_twin.set_ylim(0, max(cpu_freqs) * 1.2)
        
        # Add value labels
        for i, (ram, freq) in enumerate(zip(ram_sizes, cpu_freqs)):
            ax3.text(i, ram + max(ram_sizes)*0.02, f'{ram:.0f}KB', ha='center', va='bottom', fontweight='bold')
            ax3_twin.text(i, freq + max(cpu_freqs)*0.05, f'{freq:.0f}MHz', ha='center', va='bottom', fontweight='bold', color='#FF5722')
        
        # Feasibility Assessment
        feasibility_metrics = [
            'RAM\\nUtilization', 'CPU\\nLoad', 'Inference\\nTime', 'Overall\\nFeasibility'
        ]
        feasibility_values = [
            self.comparison_results['feasibility_assessment']['ram_utilization_percent'],
            self.comparison_results['feasibility_assessment']['cpu_load_percent'],
            min(self.comparison_results['performance_comparison']['measured_inference_us'] / 10000 * 100, 100),  # Scale to percentage
            100 if self.comparison_results['feasibility_assessment']['overall_feasible'] else 0
        ]
        
        colors = ['green' if v < 80 else 'orange' if v < 95 else 'red' for v in feasibility_values[:-1]]
        colors.append('green' if feasibility_values[-1] > 50 else 'red')
        
        bars4 = ax4.bar(feasibility_metrics, feasibility_values, color=colors, alpha=0.8)
        ax4.set_ylabel('Percentage / Feasibility Score')
        ax4.set_title('Deployment Feasibility Assessment')
        ax4.set_ylim(0, 110)
        ax4.axhline(y=80, color='orange', linestyle='--', alpha=0.7, label='Warning Threshold')
        ax4.axhline(y=95, color='red', linestyle='--', alpha=0.7, label='Critical Threshold')
        
        for bar, value, metric in zip(bars4, feasibility_values, feasibility_metrics):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 2,
                    f'{value:.1f}%' if 'Feasibility' not in metric else ('✅' if value > 50 else '❌'),
                    ha='center', va='bottom', fontweight='bold')
        
        ax4.legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'hardware_validation_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # === PLOT 2: Detailed Performance Analysis ===
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Inference Time Distribution (if available)
        if 'inference_times_us' in self.measured_data:
            inference_times = self.measured_data['inference_times_us']
            ax1.hist(inference_times, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
            ax1.axvline(np.mean(inference_times), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(inference_times):.0f}μs')
            ax1.axvline(self.comparison_results['performance_comparison']['theoretical_inference_us'], 
                       color='green', linestyle='--', linewidth=2, label=f'Theoretical: {self.comparison_results["performance_comparison"]["theoretical_inference_us"]:.0f}μs')
            ax1.set_xlabel('Inference Time (μs)')
            ax1.set_ylabel('Frequency')
            ax1.set_title('Measured Inference Time Distribution')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Error Analysis
        performance_delta = self.comparison_results['performance_comparison']['performance_delta_percent']
        memory_delta = self.comparison_results['memory_comparison']['memory_delta_percent']
        
        metrics = ['Inference Time', 'Memory Usage']
        deltas = [performance_delta, memory_delta]
        colors = ['green' if abs(d) < 25 else 'orange' if abs(d) < 50 else 'red' for d in deltas]
        
        bars = ax2.bar(metrics, deltas, color=colors, alpha=0.8)
        ax2.set_ylabel('Prediction Error (%)')
        ax2.set_title('Theoretical Model Prediction Accuracy')
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=25, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=-25, color='orange', linestyle='--', alpha=0.7)
        ax2.axhline(y=50, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=-50, color='red', linestyle='--', alpha=0.7)
        
        for bar, delta in zip(bars, deltas):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + (5 if height >= 0 else -8),
                    f'{delta:+.1f}%', ha='center', va='bottom' if height >= 0 else 'top', fontweight='bold')
        
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'performance_analysis_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("✅ Vergleichsplots erstellt!")
        
    def save_comparison_report(self, save_dir):
        """Speichert vollständigen Vergleichsreport"""
        save_dir = Path(save_dir)
        save_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Full Comparison Report
        report = {
            'timestamp': timestamp,
            'analysis_type': 'Hardware Validation & Comparison',
            'theoretical_data': self.theoretical_data,
            'measured_data': {k: v for k, v in self.measured_data.items() if not isinstance(v, list)},  # Exclude large lists
            'comparison_results': self.comparison_results,
            'summary': {
                'theoretical_model_accuracy': self.comparison_results['accuracy_assessment']['prediction_accuracy_rating'],
                'deployment_feasibility': self.comparison_results['feasibility_assessment']['overall_feasible'],
                'deployment_recommendation': self.comparison_results['feasibility_assessment']['deployment_recommendation'],
                'key_findings': [
                    f"Inference time prediction was {self.comparison_results['performance_comparison']['performance_delta_percent']:.1f}% off",
                    f"Memory usage prediction was {self.comparison_results['memory_comparison']['memory_delta_percent']:.1f}% off",
                    f"RAM utilization on actual hardware: {self.comparison_results['feasibility_assessment']['ram_utilization_percent']:.1f}%",
                    f"Platform scaling factor: {self.comparison_results['platform_comparison']['freq_scaling_factor']:.1f}x frequency, {self.comparison_results['platform_comparison']['ram_scaling_factor']:.1f}x RAM"
                ]
            }
        }
        
        with open(save_dir / f'hardware_validation_report_{timestamp}.json', 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # CSV Summary for easy analysis
        summary_df = pd.DataFrame([{
            'Metric': 'Inference Time',
            'Theoretical': f"{self.comparison_results['performance_comparison']['theoretical_inference_us']:.0f}μs",
            'Measured': f"{self.comparison_results['performance_comparison']['measured_inference_us']:.0f}μs",
            'Delta': f"{self.comparison_results['performance_comparison']['performance_delta_percent']:.1f}%"
        }, {
            'Metric': 'Memory Usage',
            'Theoretical': f"{self.comparison_results['memory_comparison']['theoretical_memory_kb']:.1f}KB",
            'Measured': f"{self.comparison_results['memory_comparison']['measured_memory_kb']:.1f}KB",
            'Delta': f"{self.comparison_results['memory_comparison']['memory_delta_percent']:.1f}%"
        }, {
            'Metric': 'RAM Utilization',
            'Theoretical': 'N/A',
            'Measured': f"{self.comparison_results['feasibility_assessment']['ram_utilization_percent']:.1f}%",
            'Delta': 'N/A'
        }, {
            'Metric': 'Deployment Feasible',
            'Theoretical': 'Unknown',
            'Measured': 'Yes' if self.comparison_results['feasibility_assessment']['overall_feasible'] else 'No',
            'Delta': 'N/A'
        }])
        
        summary_df.to_csv(save_dir / f'validation_summary_{timestamp}.csv', index=False)
        
        logger.info(f"💾 Comparison Report gespeichert: {save_dir}")
        return save_dir / f'hardware_validation_report_{timestamp}.json'

# Mock Config Class für Tests
class MockConfig:
    def __init__(self):
        self.HIDDEN_SIZE = 32
        self.NUM_LAYERS = 2
        self.SEQUENCE_LENGTH = 720

def main():
    """Main Validation Function"""
    logger.info("🔬 Hardware Validation & Comparison Analysis")
    logger.info("=" * 50)
    
    # Initialisierung
    validator = HardwareValidationComparison()
    
    # Mock Config (in echter Anwendung aus trainiertem Modell laden)
    config = MockConfig()
    
    # Theoretische Performance berechnen
    validator.calculate_theoretical_performance(config)
    
    # Suche nach neuesten Hardware-Messungen
    results_base_dir = Path(__file__).parent
    hardware_results_dirs = list(results_base_dir.glob("hardware_monitoring_results_*"))
    
    if not hardware_results_dirs:
        logger.error("❌ Keine Hardware-Messergebnisse gefunden!")
        logger.info("Führe zuerst arduino_hardware_monitor.py aus, um Messdaten zu sammeln.")
        return
    
    # Neueste Ergebnisse verwenden
    latest_results_dir = max(hardware_results_dirs, key=lambda x: x.stat().st_mtime)
    logger.info(f"📊 Verwende Hardware-Ergebnisse: {latest_results_dir.name}")
    
    # Hardware-Daten laden
    if not validator.load_measured_hardware_data(latest_results_dir):
        logger.error("❌ Fehler beim Laden der Hardware-Daten")
        return
    
    # Vergleichsanalyse durchführen
    comparison_results = validator.perform_comparison_analysis()
    
    if not comparison_results:
        logger.error("❌ Vergleichsanalyse fehlgeschlagen")
        return
    
    # Ergebnisse speichern und Plots erstellen
    output_dir = results_base_dir / f"validation_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(exist_ok=True)
    
    validator.create_comparison_plots(output_dir)
    report_file = validator.save_comparison_report(output_dir)
    
    # Final Summary
    logger.info("🎉 Hardware Validation & Comparison abgeschlossen!")
    logger.info(f"📊 Ergebnisse gespeichert in: {output_dir}")
    logger.info("📋 Key Findings:")
    
    for finding in comparison_results['summary']['key_findings']:
        logger.info(f"  • {finding}")
    
    logger.info(f"🎯 Deployment Recommendation: {comparison_results['feasibility_assessment']['deployment_recommendation']}")
    logger.info(f"✅ Overall Feasible: {comparison_results['feasibility_assessment']['overall_feasible']}")

if __name__ == "__main__":
    main()
