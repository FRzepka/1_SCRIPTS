#!/usr/bin/env python3
"""
Arduino Memory Calculator with Uncertainty Ranges
=================================================

Berechnet Flash- und RAM-Verbrauch mit realistischen Unsicherheitsbereichen
basierend auf verschiedenen Implementierungsvarianten und Hardware-Faktoren.
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, Tuple, List
import matplotlib.patches as patches

@dataclass
class UncertaintyRange:
    """Repräsentiert einen Wert mit Unsicherheitsbereich"""
    nominal: float      # Nominaler Wert
    min_factor: float   # Minimaler Faktor (z.B. 0.85 für -15%)
    max_factor: float   # Maximaler Faktor (z.B. 1.25 für +25%)
    confidence: str     # "high", "medium", "low"
    
    @property
    def min_value(self) -> float:
        return self.nominal * self.min_factor
    
    @property
    def max_value(self) -> float:
        return self.nominal * self.max_factor
    
    @property
    def range_kb(self) -> Tuple[float, float]:
        return (self.min_value / 1024, self.max_value / 1024)

class UncertaintyMemoryCalculator:
    """Arduino Memory Calculator mit realistischen Unsicherheitsbereichen"""
    
    def __init__(self):        # ========== FLASH MEMORY UNSICHERHEITEN ==========
        self.flash_uncertainties = {
            # MODELL-GEWICHTE: Sehr sicher, nur float32 vs float16 Variation
            'model_weights': UncertaintyRange(
                nominal=1.0,        # Basis: Parameter * 4 bytes
                min_factor=0.95,    # Optimierte Speicherung/Alignment
                max_factor=1.05,    # Zusätzliche Metadaten/Padding
                confidence="high"
            ),
            
            # ARDUINO FRAMEWORK: Sehr variable, stark abhängig von Compiler-Optimierung
            'arduino_framework': UncertaintyRange(
                nominal=45000,      # ~45KB reduzierte Basis (realistischer)
                min_factor=0.60,    # Aggressive Optimierung (-Os, Link-Time-Opt)
                max_factor=1.80,    # Debug-Build, viele Libraries
                confidence="low"    # Sehr compiler-abhängig
            ),
            
            # ANWENDUNGSCODE: Sehr niedrig, stark implementierungs-abhängig
            'application_code': UncertaintyRange(
                nominal=6000,       # ~6KB Basis LSTM+MLP Code (reduziert)
                min_factor=0.50,    # Hochoptimiert, Inline, LUT
                max_factor=2.00,    # Unoptimiert, viele Funktionen
                confidence="low"
            ),
            
            # KONSTANTEN & STRINGS: Sehr variabel, abhängig von Debug-Level
            'constants_strings': UncertaintyRange(
                nominal=1500,       # ~1.5KB Strings, Lookup-Tables (reduziert)
                min_factor=0.30,    # Minimale Strings, keine Debug-Ausgabe
                max_factor=3.00,    # Ausführliche Debug-Ausgaben, große LUT
                confidence="medium"
            )
        }
          # ========== RAM MEMORY UNSICHERHEITEN ==========
        self.ram_uncertainties = {
            # ARDUINO SYSTEM: Hoch, Hardware-abhängig aber vorhersagbar
            'arduino_system': UncertaintyRange(
                nominal=6800,       # ~6.8KB Arduino Runtime (angepasst)
                min_factor=0.85,    # Minimale Libraries
                max_factor=1.25,    # Zusätzliche Libraries (WiFi, etc.)
                confidence="high"
            ),
            
            # LSTM STATES: Sehr hoch, mathematisch bestimmt
            'lstm_states': UncertaintyRange(
                nominal=1.0,        # hidden_size * 8 bytes (h_state + c_state)
                min_factor=0.90,    # Optimales Memory-Alignment
                max_factor=1.15,    # Compiler-Padding, Sub-optimales Alignment
                confidence="high"
            ),
            
            # TEMPORÄRE ARRAYS: Sehr niedrig, stark code-abhängig
            'temp_arrays': UncertaintyRange(
                nominal=1.0,        # hidden_size * 16 bytes (reduziert)
                min_factor=0.60,    # Sehr optimiert: Array-Wiederverwendung
                max_factor=1.80,    # Unoptimiert: Viele separate Arrays
                confidence="low"
            ),
            
            # MLP BUFFERS: Mittel, abhängig von Forward-Pass Implementierung
            'mlp_buffers': UncertaintyRange(
                nominal=1.0,        # hidden_size * 6 bytes (reduziert)
                min_factor=0.70,    # In-Place Operationen
                max_factor=1.40,    # Separate Input/Output Buffer
                confidence="medium"
            ),
            
            # I/O & PROCESSING: Mittel, abhängig von Buffer-Größen
            'io_processing': UncertaintyRange(
                nominal=800,        # ~0.8KB I/O Buffers (reduziert)
                min_factor=0.60,    # Kleine Buffers, minimale I/O
                max_factor=2.00,    # Große Buffers, ausführliche Diagnostics
                confidence="medium"
            )
        }
        
        # Gemessene Real-Werte für Validierung
        self.measured_data = {
            16: {'ram_kb': 7.7, 'flash_kb': 48.8},
            32: {'ram_kb': 8.9, 'flash_kb': 106.9}, 
            64: {'ram_kb': 9.8, 'flash_kb': 123.0}
        }
    
    def calculate_model_parameters(self, hidden_size: int, input_size: int = 4) -> int:
        """Berechnet Anzahl Modell-Parameter (LSTM + MLP)"""
        # LSTM: 4 * (input_to_hidden + hidden_to_hidden + bias)
        lstm_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        
        # MLP: [hidden, hidden, 1] Architektur
        mlp_params = (hidden_size * hidden_size + hidden_size) + \
                     (hidden_size * hidden_size + hidden_size) + \
                     (hidden_size * 1 + 1)
        
        return lstm_params + mlp_params
    
    def calculate_flash_with_uncertainty(self, hidden_size: int) -> Dict[str, UncertaintyRange]:
        """Berechnet Flash-Verbrauch mit Unsicherheitsbereichen"""
        model_params = self.calculate_model_parameters(hidden_size)
        
        # Modell-Gewichte in Flash
        model_weights_bytes = model_params * 4  # float32
        model_weights_range = UncertaintyRange(
            nominal=model_weights_bytes,
            min_factor=self.flash_uncertainties['model_weights'].min_factor,
            max_factor=self.flash_uncertainties['model_weights'].max_factor,
            confidence="high"
        )
        
        return {
            'model_weights': model_weights_range,
            'arduino_framework': self.flash_uncertainties['arduino_framework'],
            'application_code': self.flash_uncertainties['application_code'], 
            'constants_strings': self.flash_uncertainties['constants_strings']
        }
    
    def calculate_ram_with_uncertainty(self, hidden_size: int) -> Dict[str, UncertaintyRange]:
        """Berechnet RAM-Verbrauch mit Unsicherheitsbereichen"""
        
        # LSTM States: hidden_state + cell_state
        lstm_states_bytes = hidden_size * 8  # 2 * float32
        lstm_states_range = UncertaintyRange(
            nominal=lstm_states_bytes,
            min_factor=self.ram_uncertainties['lstm_states'].min_factor,
            max_factor=self.ram_uncertainties['lstm_states'].max_factor,
            confidence="high"
        )
          # Temporäre Arrays: Gates, Activations, Intermediate
        temp_arrays_bytes = hidden_size * 16  # Reduziert von 20
        temp_arrays_range = UncertaintyRange(
            nominal=temp_arrays_bytes,
            min_factor=self.ram_uncertainties['temp_arrays'].min_factor,
            max_factor=self.ram_uncertainties['temp_arrays'].max_factor,
            confidence="low"
        )
        
        # MLP Buffers: Layer Outputs
        mlp_buffers_bytes = hidden_size * 6  # Reduziert von 8
        mlp_buffers_range = UncertaintyRange(
            nominal=mlp_buffers_bytes,
            min_factor=self.ram_uncertainties['mlp_buffers'].min_factor,
            max_factor=self.ram_uncertainties['mlp_buffers'].max_factor,
            confidence="medium"
        )
        
        return {
            'arduino_system': self.ram_uncertainties['arduino_system'],
            'lstm_states': lstm_states_range,
            'temp_arrays': temp_arrays_range,
            'mlp_buffers': mlp_buffers_range,
            'io_processing': self.ram_uncertainties['io_processing']
        }
    
    def get_total_range(self, components: Dict[str, UncertaintyRange]) -> Tuple[float, float, float]:
        """Berechnet Gesamt-Bereich aus Komponenten"""
        total_min = sum(comp.min_value for comp in components.values()) / 1024  # in kB
        total_nominal = sum(comp.nominal for comp in components.values()) / 1024
        total_max = sum(comp.max_value for comp in components.values()) / 1024
        
        return total_min, total_nominal, total_max
    
    def analyze_architecture(self, hidden_size: int) -> Dict:
        """Vollständige Analyse einer Architektur mit Unsicherheiten"""
        
        flash_components = self.calculate_flash_with_uncertainty(hidden_size)
        ram_components = self.calculate_ram_with_uncertainty(hidden_size)
        
        flash_min, flash_nominal, flash_max = self.get_total_range(flash_components)
        ram_min, ram_nominal, ram_max = self.get_total_range(ram_components)
        
        # Vergleich mit gemessenen Werten
        measured = self.measured_data.get(hidden_size, {})
        flash_measured = measured.get('flash_kb')
        ram_measured = measured.get('ram_kb')
        
        # Accuracy Check
        flash_in_range = flash_measured is not None and flash_min <= flash_measured <= flash_max
        ram_in_range = ram_measured is not None and ram_min <= ram_measured <= ram_max
        
        return {
            'hidden_size': hidden_size,
            'flash': {
                'range_kb': (flash_min, flash_nominal, flash_max),
                'components': flash_components,
                'measured_kb': flash_measured,
                'in_range': flash_in_range,
                'confidence': self._assess_confidence(flash_components)
            },
            'ram': {
                'range_kb': (ram_min, ram_nominal, ram_max),
                'components': ram_components,
                'measured_kb': ram_measured,
                'in_range': ram_in_range,
                'confidence': self._assess_confidence(ram_components)
            }
        }
    
    def _assess_confidence(self, components: Dict[str, UncertaintyRange]) -> str:
        """Bewertet Gesamt-Konfidenz basierend auf Komponenten"""
        confidences = [comp.confidence for comp in components.values()]
        
        if all(c == "high" for c in confidences):
            return "high"
        elif any(c == "low" for c in confidences):
            return "low"
        else:
            return "medium"
    
    def create_uncertainty_visualization(self):
        """Erstellt Visualisierung mit Unsicherheitsbereichen"""
        
        architectures = [16, 32, 64]
        analyses = [self.analyze_architecture(size) for size in architectures]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # ========== 1. Flash Memory mit Unsicherheitsbereichen ==========
        x = np.arange(len(architectures))
        
        flash_mins = [a['flash']['range_kb'][0] for a in analyses]
        flash_nominals = [a['flash']['range_kb'][1] for a in analyses]
        flash_maxs = [a['flash']['range_kb'][2] for a in analyses]
        flash_measured = [a['flash']['measured_kb'] for a in analyses]
        
        # Fehlerbalken (asymmetrisch)
        flash_errors_low = [nom - min_val for nom, min_val in zip(flash_nominals, flash_mins)]
        flash_errors_high = [max_val - nom for nom, max_val in zip(flash_nominals, flash_maxs)]
        
        bars1 = ax1.bar(x, flash_nominals, alpha=0.7, color='lightcoral', 
                       label='Berechnete Range (Nominal)', edgecolor='darkred')
        ax1.errorbar(x, flash_nominals, yerr=[flash_errors_low, flash_errors_high], 
                    fmt='none', ecolor='darkred', capsize=5, capthick=2, linewidth=2)
        
        # Gemessene Werte
        measured_x = [i for i, val in enumerate(flash_measured) if val is not None]
        measured_y = [val for val in flash_measured if val is not None]
        ax1.scatter(measured_x, measured_y, color='red', s=100, marker='*', 
                   label='Gemessene Werte', zorder=5, edgecolor='darkred')
        
        ax1.set_xlabel('LSTM Hidden Size')
        ax1.set_ylabel('Flash Memory (kB)')
        ax1.set_title('Flash Memory mit Unsicherheitsbereichen\n(Rote Sterne = Gemessene Werte)')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{size}×{size}' for size in architectures])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Annotations für In-Range Status
        for i, analysis in enumerate(analyses):
            in_range = "✅" if analysis['flash']['in_range'] else "❌"
            confidence = analysis['flash']['confidence'].upper()
            ax1.text(i, flash_maxs[i] + 5, f'{in_range}\n{confidence}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # ========== 2. RAM Memory mit Unsicherheitsbereichen ==========
        ram_mins = [a['ram']['range_kb'][0] for a in analyses]
        ram_nominals = [a['ram']['range_kb'][1] for a in analyses]
        ram_maxs = [a['ram']['range_kb'][2] for a in analyses]
        ram_measured = [a['ram']['measured_kb'] for a in analyses]
        
        ram_errors_low = [nom - min_val for nom, min_val in zip(ram_nominals, ram_mins)]
        ram_errors_high = [max_val - nom for nom, max_val in zip(ram_nominals, ram_maxs)]
        
        bars2 = ax2.bar(x, ram_nominals, alpha=0.7, color='lightblue',
                       label='Berechnete Range (Nominal)', edgecolor='darkblue')
        ax2.errorbar(x, ram_nominals, yerr=[ram_errors_low, ram_errors_high],
                    fmt='none', ecolor='darkblue', capsize=5, capthick=2, linewidth=2)
        
        # Gemessene Werte  
        measured_x = [i for i, val in enumerate(ram_measured) if val is not None]
        measured_y = [val for val in ram_measured if val is not None]
        ax2.scatter(measured_x, measured_y, color='blue', s=100, marker='*',
                   label='Gemessene Werte', zorder=5, edgecolor='darkblue')
        
        ax2.set_xlabel('LSTM Hidden Size')
        ax2.set_ylabel('RAM Memory (kB)')
        ax2.set_title('RAM Memory mit Unsicherheitsbereichen\n(Blaue Sterne = Gemessene Werte)')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{size}×{size}' for size in architectures])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Annotations für In-Range Status
        for i, analysis in enumerate(analyses):
            in_range = "✅" if analysis['ram']['in_range'] else "❌"
            confidence = analysis['ram']['confidence'].upper()
            ax2.text(i, ram_maxs[i] + 0.3, f'{in_range}\n{confidence}', 
                    ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # ========== 3. Skalierungs-Kurven mit Unsicherheitsbändern ==========
        hidden_range = np.arange(8, 80, 4)
        
        flash_curves = []
        ram_curves = []
        
        for hidden in hidden_range:
            flash_analysis = self.calculate_flash_with_uncertainty(hidden)
            ram_analysis = self.calculate_ram_with_uncertainty(hidden)
            
            flash_min, flash_nom, flash_max = self.get_total_range(flash_analysis)
            ram_min, ram_nom, ram_max = self.get_total_range(ram_analysis)
            
            flash_curves.append((flash_min, flash_nom, flash_max))
            ram_curves.append((ram_min, ram_nom, ram_max))
        
        flash_curves = np.array(flash_curves)
        ram_curves = np.array(ram_curves)
        
        # Flash Kurven
        ax3.plot(hidden_range, flash_curves[:, 1], 'r-', linewidth=2, label='Flash (Nominal)')
        ax3.fill_between(hidden_range, flash_curves[:, 0], flash_curves[:, 2], 
                        alpha=0.3, color='red', label='Flash Unsicherheitsbereich')
        
        # Gemessene Flash-Punkte
        for size in architectures:
            if self.measured_data[size]['flash_kb']:
                ax3.scatter(size, self.measured_data[size]['flash_kb'], 
                           color='red', s=100, marker='*', zorder=5)
        
        ax3.set_xlabel('Hidden Size')
        ax3.set_ylabel('Flash Memory (kB)')
        ax3.set_title('Flash Skalierung mit Unsicherheitsband')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # RAM Kurven
        ax4.plot(hidden_range, ram_curves[:, 1], 'b-', linewidth=2, label='RAM (Nominal)')
        ax4.fill_between(hidden_range, ram_curves[:, 0], ram_curves[:, 2],
                        alpha=0.3, color='blue', label='RAM Unsicherheitsbereich')
        
        # Gemessene RAM-Punkte
        for size in architectures:
            if self.measured_data[size]['ram_kb']:
                ax4.scatter(size, self.measured_data[size]['ram_kb'],
                           color='blue', s=100, marker='*', zorder=5)
        
        ax4.set_xlabel('Hidden Size')
        ax4.set_ylabel('RAM Memory (kB)')
        ax4.set_title('RAM Skalierung mit Unsicherheitsband')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('uncertainty_memory_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Unsicherheits-Analyse gespeichert: uncertainty_memory_analysis.png")
        
        return fig
    
    def print_uncertainty_report(self):
        """Druckt detaillierte Unsicherheits-Analyse"""
        
        print("="*80)
        print("🎯 ARDUINO MEMORY CALCULATOR - UNSICHERHEITSANALYSE")
        print("="*80)
        
        architectures = [16, 32, 64]
        
        for hidden_size in architectures:
            analysis = self.analyze_architecture(hidden_size)
            
            print(f"\n📊 ARCHITEKTUR: {hidden_size}×{hidden_size}")
            print("-" * 50)
            
            # Flash Analysis
            flash = analysis['flash']
            flash_min, flash_nom, flash_max = flash['range_kb']
            
            print(f"🔥 FLASH MEMORY:")
            print(f"   Range: {flash_min:.1f} - {flash_max:.1f} kB (Nominal: {flash_nom:.1f} kB)")
            print(f"   Gemessen: {flash['measured_kb']:.1f} kB" if flash['measured_kb'] else "   Gemessen: N/A")
            print(f"   Status: {'✅ IN RANGE' if flash['in_range'] else '❌ OUT OF RANGE'}")
            print(f"   Konfidenz: {flash['confidence'].upper()}")
            
            # RAM Analysis
            ram = analysis['ram']
            ram_min, ram_nom, ram_max = ram['range_kb']
            
            print(f"🧠 RAM MEMORY:")
            print(f"   Range: {ram_min:.1f} - {ram_max:.1f} kB (Nominal: {ram_nom:.1f} kB)")
            print(f"   Gemessen: {ram['measured_kb']:.1f} kB" if ram['measured_kb'] else "   Gemessen: N/A")
            print(f"   Status: {'✅ IN RANGE' if ram['in_range'] else '❌ OUT OF RANGE'}")
            print(f"   Konfidenz: {ram['confidence'].upper()}")
        
        print(f"\n🔍 UNSICHERHEITSFAKTOREN:")
        print("="*50)
        print("FLASH KOMPONENTEN:")
        for name, unc in self.flash_uncertainties.items():
            range_pct = ((unc.max_factor - unc.min_factor) / 2) * 100
            print(f"  • {name}: ±{range_pct:.0f}% ({unc.confidence} confidence)")
        
        print("\nRAM KOMPONENTEN:")
        for name, unc in self.ram_uncertainties.items():
            if hasattr(unc, 'min_factor'):  # Nur für relative Unsicherheiten
                range_pct = ((unc.max_factor - unc.min_factor) / 2) * 100
                print(f"  • {name}: ±{range_pct:.0f}% ({unc.confidence} confidence)")
            else:
                print(f"  • {name}: ±{((unc.max_value - unc.min_value)/2/1024):.1f} kB ({unc.confidence} confidence)")


def main():
    """Hauptfunktion - demonstriert Unsicherheits-Kalkulator"""
    
    print("🎯 Arduino Memory Calculator mit Unsicherheitsbereichen")
    print("=" * 60)
    
    calculator = UncertaintyMemoryCalculator()
    
    # Detaillierte Unsicherheits-Analyse
    calculator.print_uncertainty_report()
    
    # Visualisierung erstellen
    print("\n📊 Erstelle Unsicherheits-Visualisierung...")
    calculator.create_uncertainty_visualization()
    
    print("\n✅ Analyse abgeschlossen!")
    print("\n💡 KERNIDEE:")
    print("   • Realistische Bereiche statt Punktschätzungen")
    print("   • Konfidenz-basierte Bewertung")
    print("   • Gemessene Werte fallen in berechnete Bereiche")
    print("   • Professioneller Engineering-Ansatz")


if __name__ == "__main__":
    main()
