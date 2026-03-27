#!/usr/bin/env python3
"""
Arduino Memory Calculator with Uncertainty Analysis - Improved Visualization
Berechnet Flash- und RAM-Verbrauch mit realistischen Unsicherheitsbereichen
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from dataclasses import dataclass
from typing import Dict, Tuple

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

@dataclass
class UncertaintyRange:
    """Represents a value with uncertainty"""
    nominal: float
    min_val: float
    max_val: float
    factor_name: str
    confidence: str  # "high", "medium", "low"
    
    @classmethod
    def from_percent(cls, nominal: float, percent: float, factor_name: str, confidence: str = "medium"):
        return cls(nominal, nominal * (1 - percent/100), nominal * (1 + percent/100), factor_name, confidence)
    
    @classmethod
    def from_absolute(cls, nominal: float, plus_minus: float, factor_name: str, confidence: str = "medium"):
        return cls(nominal, nominal - plus_minus, nominal + plus_minus, factor_name, confidence)

def calculate_flash_with_uncertainty(hidden_size: int, input_size: int = 4) -> Dict[str, UncertaintyRange]:
    """Berechnet Flash-Verbrauch mit Unsicherheitsbereichen"""
    
    # 1. MODELL-GEWICHTE (sehr sicher - bekannte Parameter)
    lstm_weights = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
    mlp_weights = (hidden_size * hidden_size + hidden_size) + (hidden_size * hidden_size + hidden_size) + (hidden_size * 1 + 1)
    total_weights_bytes = (lstm_weights + mlp_weights) * 4
    
    # 2. ARDUINO FRAMEWORK CODE (mittlere Unsicherheit - compiler-abhängig)
    base_framework = 75000  # Nominal 75KB
    
    # 3. ANWENDUNGSCODE (hohe Unsicherheit - stark implementierungs-abhängig)
    application_code = 20000  # Nominal 20KB
    
    # 4. KONSTANTEN & STRINGS (mittlere Unsicherheit)
    constants = 3000  # Nominal 3KB
    
    return {
        'model_weights': UncertaintyRange.from_percent(
            total_weights_bytes / 1024, 2, "Model Weights", "high"
        ),
        'arduino_framework': UncertaintyRange.from_percent(
            base_framework / 1024, 25, "Arduino Framework", "medium"
        ),
        'application_code': UncertaintyRange.from_percent(
            application_code / 1024, 40, "Application Code", "low"
        ),
        'constants': UncertaintyRange.from_percent(
            constants / 1024, 30, "Constants & Strings", "medium"
        )
    }

def calculate_ram_with_uncertainty(hidden_size: int, input_size: int = 4) -> Dict[str, UncertaintyRange]:
    """Berechnet RAM-Verbrauch mit Unsicherheitsbereichen"""
    
    # 1. ARDUINO SYSTEM (sehr sicher - hardware-abhängig)
    arduino_system = 7500  # 7.5KB nominal
    
    # 2. LSTM STATES (sicher - bekannte Größe)
    lstm_states = hidden_size * 2 * 4  # h_state + c_state, float32
    
    # 3. TEMPORÄRE ARRAYS (mittlere Unsicherheit - compiler/optimierung-abhängig)
    temp_arrays = hidden_size * 16  # Schätzung für temporäre Berechnungen
    
    # 4. MLP BUFFERS (niedrige Unsicherheit)
    mlp_buffers = hidden_size * 4 * 3  # 3 Layer outputs
    
    # 5. I/O BUFFERS (mittlere Unsicherheit - konfigurationsabhängig)
    io_buffers = 1000  # 1KB nominal
    
    return {
        'arduino_system': UncertaintyRange.from_percent(
            arduino_system / 1024, 8, "Arduino System", "high"
        ),
        'lstm_states': UncertaintyRange.from_percent(
            lstm_states / 1024, 5, "LSTM States", "high"
        ),
        'temp_arrays': UncertaintyRange.from_percent(
            temp_arrays / 1024, 35, "Temp Arrays", "low"
        ),
        'mlp_buffers': UncertaintyRange.from_percent(
            mlp_buffers / 1024, 15, "MLP Buffers", "medium"
        ),
        'io_buffers': UncertaintyRange.from_percent(
            io_buffers / 1024, 25, "I/O Buffers", "medium"
        )
    }

def sum_uncertainty_ranges(ranges: Dict[str, UncertaintyRange]) -> UncertaintyRange:
    """Summiert Unsicherheitsbereiche"""
    total_nominal = sum(r.nominal for r in ranges.values())
    total_min = sum(r.min_val for r in ranges.values())
    total_max = sum(r.max_val for r in ranges.values())
    return UncertaintyRange(total_nominal, total_min, total_max, "Total", "combined")

def create_comprehensive_visualization():
    """Erstellt umfassende Visualisierung mit 4 Subplots"""
    
    # Testdaten
    architectures = ['16×16', '32×32', '64×64']
    hidden_sizes = [16, 32, 64]
    measured_flash = [48.8, 106.9, 123.0]
    measured_ram = [7.7, 8.9, 9.8]
    
    # Berechne Unsicherheitsbereiche
    flash_results = []
    ram_results = []
    
    for hidden_size in hidden_sizes:
        flash_ranges = calculate_flash_with_uncertainty(hidden_size)
        ram_ranges = calculate_ram_with_uncertainty(hidden_size)
        
        flash_total = sum_uncertainty_ranges(flash_ranges)
        ram_total = sum_uncertainty_ranges(ram_ranges)
        
        flash_results.append((flash_total, flash_ranges))
        ram_results.append((ram_total, ram_ranges))
    
    # Erstelle Figure mit 4 Subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # ========== 1. BALKEN MIT FEHLERBEREICHEN ==========
    x = np.arange(len(architectures))
    width = 0.35
    
    # Flash Balken
    flash_nominals = [r[0].nominal for r in flash_results]
    flash_errors_low = [r[0].nominal - r[0].min_val for r in flash_results]
    flash_errors_high = [r[0].max_val - r[0].nominal for r in flash_results]
    
    bars1 = ax1.bar(x - width/2, flash_nominals, width, 
                   yerr=[flash_errors_low, flash_errors_high],
                   label='Flash (Berechnet)', alpha=0.7, color='coral',
                   capsize=5, error_kw={'linewidth': 2})
    
    # RAM Balken  
    ram_nominals = [r[0].nominal for r in ram_results]
    ram_errors_low = [r[0].nominal - r[0].min_val for r in ram_results]
    ram_errors_high = [r[0].max_val - r[0].nominal for r in ram_results]
    
    bars2 = ax1.bar(x + width/2, ram_nominals, width,
                   yerr=[ram_errors_low, ram_errors_high], 
                   label='RAM (Berechnet)', alpha=0.7, color='lightblue',
                   capsize=5, error_kw={'linewidth': 2})
    
    # Gemessene Werte als Punkte
    ax1.scatter(x - width/2, measured_flash, color='darkred', s=100, 
               label='Flash (Gemessen)', marker='o', zorder=5)
    ax1.scatter(x + width/2, measured_ram, color='darkblue', s=100,
               label='RAM (Gemessen)', marker='s', zorder=5)
    
    ax1.set_xlabel('LSTM Architektur')
    ax1.set_ylabel('Speicher (kB)')
    ax1.set_title('Flash & RAM Verbrauch mit Unsicherheitsbereichen')
    ax1.set_xticks(x)
    ax1.set_xticklabels(architectures)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. SKALIERUNGSKURVEN MIT BÄNDERN ==========
    hidden_range = np.arange(10, 80, 2)
    
    flash_curve_nom = []
    flash_curve_min = []
    flash_curve_max = []
    ram_curve_nom = []
    ram_curve_min = []
    ram_curve_max = []
    
    for h in hidden_range:
        flash_ranges = calculate_flash_with_uncertainty(h)
        ram_ranges = calculate_ram_with_uncertainty(h)
        
        flash_total = sum_uncertainty_ranges(flash_ranges)
        ram_total = sum_uncertainty_ranges(ram_ranges)
        
        flash_curve_nom.append(flash_total.nominal)
        flash_curve_min.append(flash_total.min_val)
        flash_curve_max.append(flash_total.max_val)
        
        ram_curve_nom.append(ram_total.nominal)
        ram_curve_min.append(ram_total.min_val)
        ram_curve_max.append(ram_total.max_val)
    
    # Flash Kurve
    ax2.plot(hidden_range, flash_curve_nom, 'r-', linewidth=2, label='Flash (Nominal)')
    ax2.fill_between(hidden_range, flash_curve_min, flash_curve_max, 
                    color='red', alpha=0.2, label='Flash (Unsicherheit)')
    
    # RAM Kurve
    ax2_twin = ax2.twinx()
    ax2_twin.plot(hidden_range, ram_curve_nom, 'b-', linewidth=2, label='RAM (Nominal)')
    ax2_twin.fill_between(hidden_range, ram_curve_min, ram_curve_max,
                         color='blue', alpha=0.2, label='RAM (Unsicherheit)')
    
    # Gemessene Punkte
    ax2.scatter(hidden_sizes, measured_flash, color='darkred', s=100, zorder=5)
    ax2_twin.scatter(hidden_sizes, measured_ram, color='darkblue', s=100, zorder=5)
    
    ax2.set_xlabel('Hidden Size')
    ax2.set_ylabel('Flash (kB)', color='red')
    ax2_twin.set_ylabel('RAM (kB)', color='blue')
    ax2.set_title('Speicher-Skalierung mit Unsicherheitsbändern')
    ax2.grid(True, alpha=0.3)
    
    # Legenden kombinieren
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    # ========== 3. RAM KOMPONENTEN KREISDIAGRAMM ==========
    # Verwende 32×32 als Beispiel
    ram_example = ram_results[1][1]  # 32×32 RAM components
    
    ram_labels = []
    ram_sizes = []
    ram_colors = []
    confidence_colors = {'high': '#2E8B57', 'medium': '#FFD700', 'low': '#FF6347'}
    
    for name, range_obj in ram_example.items():
        ram_labels.append(f"{name}\n{range_obj.nominal:.1f} kB")
        ram_sizes.append(range_obj.nominal)
        ram_colors.append(confidence_colors[range_obj.confidence])
    
    wedges, texts, autotexts = ax3.pie(ram_sizes, labels=ram_labels, colors=ram_colors,
                                      autopct='%1.1f%%', startangle=90)
    ax3.set_title('RAM Komponenten (32×32)\nFarben: Grün=Sicher, Gelb=Mittel, Rot=Unsicher')
    
    # ========== 4. FLASH KOMPONENTEN KREISDIAGRAMM ==========
    # Verwende 32×32 als Beispiel  
    flash_example = flash_results[1][1]  # 32×32 Flash components
    
    flash_labels = []
    flash_sizes = []
    flash_colors = []
    
    for name, range_obj in flash_example.items():
        flash_labels.append(f"{name}\n{range_obj.nominal:.1f} kB")
        flash_sizes.append(range_obj.nominal)
        flash_colors.append(confidence_colors[range_obj.confidence])
    
    wedges, texts, autotexts = ax4.pie(flash_sizes, labels=flash_labels, colors=flash_colors,
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title('Flash Komponenten (32×32)\nFarben: Grün=Sicher, Gelb=Mittel, Rot=Unsicher')
    
    plt.tight_layout()
    
    # Speichere Diagramm
    filename = "arduino_memory_uncertainty_analysis.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"✅ Comprehensive Analysis saved: {filename}")
    
    # ========== VALIDIERUNG AUSGEBEN ==========
    print("\n" + "="*80)
    print("🎯 UNSICHERHEITSANALYSE - VALIDIERUNG")
    print("="*80)
    
    for i, (arch, h_size, meas_f, meas_r) in enumerate(zip(architectures, hidden_sizes, measured_flash, measured_ram)):
        flash_total, _ = flash_results[i]
        ram_total, _ = ram_results[i]
        
        flash_in_range = flash_total.min_val <= meas_f <= flash_total.max_val
        ram_in_range = ram_total.min_val <= meas_r <= ram_total.max_val
        
        print(f"\n📊 {arch} ARCHITEKTUR:")
        print(f"   Flash: {meas_f} kB {'✅' if flash_in_range else '❌'} ∈ [{flash_total.min_val:.1f}, {flash_total.max_val:.1f}] kB")
        print(f"   RAM:   {meas_r} kB {'✅' if ram_in_range else '❌'} ∈ [{ram_total.min_val:.1f}, {ram_total.max_val:.1f}] kB")
    
    plt.show()
    return fig

if __name__ == "__main__":
    print("🚀 Starting Arduino Memory Uncertainty Analysis...")
    create_comprehensive_visualization()
    print("✅ Analysis complete!")
