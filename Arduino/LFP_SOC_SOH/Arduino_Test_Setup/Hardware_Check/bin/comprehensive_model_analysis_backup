# Comprehensive Model Analysis
# Analysiert PyTorch Modelle, Trainings-Skripte und Arduino Weights systematisch
#
# KORREKTE KOMPLETTE MODELL-PARAMETER FORMEL (LSTM + MLP):
# 
# LSTM: 4 × (d×h + h×h + 2×h) = 4h(d + h + 2)
# MLP:  h·C + C (für jede Schicht)
# 
# Gesamt: 4h(d + h + 2) + Σ(prev_size × layer_size + layer_size)
# 
# Wo:  d = input_size, h = hidden_size, C = MLP layer sizes
# Standard MLP-Architektur: [h, h, 1] → h×h+h + h×h+h + h×1+1

import torch
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd

class ModelArchitectureAnalyzer:
    def __init__(self, base_path):
        self.base_path = base_path
        self.results = {}
    
    def calculate_complete_model_parameters(self, input_size, hidden_size, mlp_layers=None):
        """
        Berechnet die kompletten Modell-Parameter (LSTM + MLP Head)
        
        CORRECTED FORMULA:
        LSTM: 4 × (input_size × hidden_size + hidden_size × hidden_size + 2 × hidden_size)
        - Weight matrices: input_to_hidden + hidden_to_hidden  
        - Bias vectors: TWO per gate (bias_ih + bias_hh)
        
        MLP: h·C + C (für jede MLP-Schicht)
        
        Args:
            input_size (int): LSTM Input-Größe (d)
            hidden_size (int): LSTM Hidden-Größe (h) 
            mlp_layers (list): MLP-Schichten. Falls None, wird [h, h, 1] verwendet
        """
        if hidden_size <= 0:
            return 0
            
        # Standard MLP-Architektur: [hidden_size, hidden_size, 1]
        # Dies entspricht der tatsächlichen Architektur in den Training-Skripte
        if mlp_layers is None:
            mlp_layers = [hidden_size, hidden_size, 1]
            
        # CORRECTED LSTM Parameter: 4 × (d×h + h×h + 2×h)
        d = input_size
        h = hidden_size
        lstm_params = 4 * (d * h + h * h + 2 * h)
        
        # MLP Parameter: h·C + C für jede Schicht
        mlp_params = 0
        prev_size = h  # MLP startet mit LSTM output
        
        for layer_size in mlp_layers:
            # Gewichte: prev_size × layer_size
            # Bias: layer_size
            mlp_params += prev_size * layer_size + layer_size
            prev_size = layer_size
        
        total_params = lstm_params + mlp_params
        
        return {
            'total': total_params,
            'lstm': lstm_params, 
            'mlp': mlp_params,
            'breakdown': {
                'lstm_formula': f"4×(d×h + h×h + 2×h) = 4×({d}×{h} + {h}×{h} + 2×{h}) = {lstm_params}",
                'mlp_layers': mlp_layers,
                'mlp_formula': f"MLP: {mlp_params} parameters"
            }
        }
        
    def find_model_directories(self):
        """Findet alle Model-Verzeichnisse mit entsprechender Struktur"""
        
        model_dirs = []
        
        # Suche nach Stateful_XX_XX Ordnern
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            if os.path.isdir(item_path) and item.startswith('Stateful_'):
                model_dir = os.path.join(item_path, 'model')
                if os.path.exists(model_dir):
                    model_dirs.append({
                        'architecture': item,
                        'base_dir': item_path,
                        'model_dir': model_dir,
                        'best_model_path': os.path.join(model_dir, 'best_model.pth'),
                        'train_script_path': self.find_training_script(model_dir)
                    })
        
        return model_dirs
    
    def find_training_script(self, model_dir):
        """Findet das Trainings-Skript im Model-Verzeichnis"""
        
        for file in os.listdir(model_dir):
            if file.endswith('.py') and 'train' in file.lower():
                return os.path.join(model_dir, file)
        return None
    
    def analyze_training_script(self, script_path):
        """Analysiert das Trainings-Skript, um die Modell-Architektur zu extrahieren"""
        
        if not script_path or not os.path.exists(script_path):
            return None
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            analysis = {
                'script_path': script_path,
                'architecture': {},
                'hyperparameters': {},
                'model_definition': None
            }
            
            print(f"\n📄 Analysiere Trainings-Skript: {os.path.basename(script_path)}")
            
            # Suche nach Architektur-Parametern
            patterns = {
                'hidden_size': r'hidden_size\s*=\s*(\d+)',
                'input_size': r'input_size\s*=\s*(\d+)', 
                'output_size': r'output_size\s*=\s*(\d+)',
                'num_layers': r'num_layers\s*=\s*(\d+)',
                'batch_size': r'batch_size\s*=\s*(\d+)',
                'learning_rate': r'learning_rate\s*=\s*([\d.e-]+)',
                'epochs': r'epochs\s*=\s*(\d+)',
            }
            
            for param, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    try:
                        value = float(match.group(1)) if '.' in match.group(1) else int(match.group(1))
                        if param in ['hidden_size', 'input_size', 'output_size', 'num_layers']:
                            analysis['architecture'][param] = value
                        else:
                            analysis['hyperparameters'][param] = value
                        print(f"   {param}: {value}")
                    except ValueError:
                        pass
            
            # Suche nach LSTM-Definition
            lstm_patterns = [
                r'class\s+(\w*LSTM\w*)',
                r'nn\.LSTM\s*\(',
                r'LSTM\s*\(',
            ]
            
            for pattern in lstm_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    analysis['model_definition'] = matches
                    print(f"   LSTM Definition gefunden: {matches}")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Fehler beim Analysieren des Skripts: {e}")
            return None
    
    def analyze_pytorch_model(self, model_path):
        """Analysiert eine PyTorch .pth Datei"""
        
        if not os.path.exists(model_path):
            print(f"❌ Model nicht gefunden: {model_path}")
            return None
        
        try:
            print(f"\n🔍 Lade PyTorch Model: {os.path.basename(model_path)}")
            
            # Lade Modell
            model_data = torch.load(model_path, map_location='cpu')
            
            analysis = {
                'model_path': model_path,
                'file_size_kb': os.path.getsize(model_path) / 1024,
                'total_parameters': 0,
                'layer_info': {},
                'lstm_architecture': {},
                'state_dict': None
            }
            
            # Extrahiere state_dict
            if isinstance(model_data, dict):
                if 'state_dict' in model_data:
                    state_dict = model_data['state_dict']
                elif 'model_state_dict' in model_data:
                    state_dict = model_data['model_state_dict']
                else:
                    state_dict = model_data
                    
                # Zusätzliche Informationen
                for key in model_data.keys():
                    if key != 'state_dict' and key != 'model_state_dict':
                        print(f"   Zusätzliche Info: {key} = {model_data[key]}")
            else:
                state_dict = model_data
            
            analysis['state_dict'] = state_dict
            
            # Analysiere Parameter
            total_params = 0
            layer_groups = {}
            
            print(f"   Parameter-Ebenen: {len(state_dict)}")
            
            for param_name, param_tensor in state_dict.items():
                param_count = param_tensor.numel()
                total_params += param_count
                
                # Gruppiere nach Layer-Typ
                parts = param_name.split('.')
                layer_type = parts[0] if parts else 'unknown'
                
                if layer_type not in layer_groups:
                    layer_groups[layer_type] = []
                
                layer_groups[layer_type].append({
                    'name': param_name,
                    'shape': list(param_tensor.shape),
                    'parameters': param_count,
                    'size_kb': (param_count * 4) / 1024
                })
                
                print(f"      {param_name}: {list(param_tensor.shape)} = {param_count:,} params")
            
            analysis['total_parameters'] = total_params
            analysis['layer_info'] = layer_groups
            analysis['memory_size_kb'] = (total_params * 4) / 1024
            
            # Erkenne LSTM-Architektur aus Parametern
            lstm_arch = self.detect_lstm_from_parameters(state_dict)
            if lstm_arch:
                analysis['lstm_architecture'] = lstm_arch
                print(f"   🧠 LSTM erkannt: {lstm_arch}")
            
            print(f"   ✅ Gesamt: {total_params:,} Parameter ({analysis['memory_size_kb']:.2f} kB)")
            
            return analysis
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Models: {e}")
            return None
    
    def detect_lstm_from_parameters(self, state_dict):
        """Erkennt LSTM-Architektur aus den Modell-Parametern"""
        
        lstm_info = {}
        
        for param_name, param_tensor in state_dict.items():
            shape = param_tensor.shape
            
            # Suche nach LSTM weight_ih (Input-to-Hidden)
            if 'weight_ih' in param_name and len(shape) == 2:
                gates_x_hidden, input_size = shape
                hidden_size = gates_x_hidden // 4  # 4 Gates bei LSTM
                lstm_info['input_size'] = input_size
                lstm_info['hidden_size'] = hidden_size
                lstm_info['architecture_name'] = f"{hidden_size}×{hidden_size}"
            
            # Suche nach LSTM weight_hh (Hidden-to-Hidden) zur Bestätigung
            elif 'weight_hh' in param_name and len(shape) == 2:
                gates_x_hidden, hidden_size = shape
                if 'hidden_size' not in lstm_info:
                    lstm_info['hidden_size'] = hidden_size // 4        # Berechne erwartete LSTM-Parameter wenn erkannt
        if 'input_size' in lstm_info and 'hidden_size' in lstm_info:
            i = lstm_info['input_size']
            h = lstm_info['hidden_size']
              # CORRECTED LSTM Parameter: 4 × (d×h + h×h + 2×h)
            # (ohne MLP, da diese Funktion nur LSTM erkennt)
            expected_lstm_only = 4 * (i * h + h * h + 2 * h)
            lstm_info['expected_lstm_parameters'] = expected_lstm_only
              # Zusätzlich: Komplette Modell-Parameter (LSTM + typischer MLP Head)
            complete_calc = self.calculate_complete_model_parameters(i, h, None)  # None = [h, h, 1]
            lstm_info['expected_complete_parameters'] = complete_calc['total']
            lstm_info['parameter_breakdown'] = complete_calc
            
        return lstm_info if lstm_info else None
    
    def analyze_architecture_directory(self, arch_info):
        """Analysiert ein komplettes Architektur-Verzeichnis"""
        
        print(f"\n{'='*80}")
        print(f"ANALYSIERE ARCHITEKTUR: {arch_info['architecture']}")
        print(f"{'='*80}")
        
        result = {
            'architecture_name': arch_info['architecture'],
            'base_directory': arch_info['base_dir'],
            'training_script_analysis': None,
            'pytorch_model_analysis': None,
            'comparison': {}
        }
        
        # 1. Analysiere Trainings-Skript
        if arch_info['train_script_path']:
            result['training_script_analysis'] = self.analyze_training_script(arch_info['train_script_path'])
        
        # 2. Analysiere PyTorch Model
        if os.path.exists(arch_info['best_model_path']):
            result['pytorch_model_analysis'] = self.analyze_pytorch_model(arch_info['best_model_path'])
        
        # 3. Vergleiche Training-Skript mit Model
        result['comparison'] = self.compare_script_vs_model(
            result['training_script_analysis'],
            result['pytorch_model_analysis']
        )
        
        return result
    
    def compare_script_vs_model(self, script_analysis, model_analysis):
        """Vergleicht Trainings-Skript mit tatsächlichem Model"""
        
        comparison = {
            'matches': {},
            'discrepancies': {},
            'validation': 'unknown'
        }
        
        if not script_analysis or not model_analysis:
            comparison['validation'] = 'incomplete_data'
            return comparison
        
        # Vergleiche Architektur-Parameter
        script_arch = script_analysis.get('architecture', {})
        model_lstm = model_analysis.get('lstm_architecture', {})
        
        params_to_check = ['hidden_size', 'input_size']
        
        for param in params_to_check:
            script_val = script_arch.get(param)
            model_val = model_lstm.get(param)
            
            if script_val is not None and model_val is not None:
                if script_val == model_val:
                    comparison['matches'][param] = script_val
                else:
                    comparison['discrepancies'][param] = {
                        'script': script_val,
                        'model': model_val
                    }
            elif script_val is not None:
                comparison['discrepancies'][f'{param}_missing_in_model'] = script_val
            elif model_val is not None:
                comparison['discrepancies'][f'{param}_missing_in_script'] = model_val
        
        # Gesamtvalidierung
        if comparison['discrepancies']:
            comparison['validation'] = 'mismatch'
            print(f"   ⚠️  Diskrepanzen gefunden: {comparison['discrepancies']}")
        
        elif comparison['matches']:
            comparison['validation'] = 'match'
            print(f"   ✅ Architektur stimmt überein: {comparison['matches']}")
        else:
            comparison['validation'] = 'no_comparison_possible'
            print(f"   ❓ Keine Vergleichsdaten verfügbar")
        
        return comparison
    def calculate_arduino_ram_usage(self, hidden_size, input_size=1):
        """
        VERBESSERTE Arduino RAM-Verbrauch Berechnung mit korrekter RAM/Flash Unterscheidung
        
        WICHTIG: Was kommt in die RAM vs. Flash?
        - FLASH: Modell-Parameter (Gewichte, Bias) - persistent im Programmspeicher
        - RAM: Runtime-Daten (Hidden States, temporäre Berechnungen, Arduino Framework)
        
        Die Modell-Gewichte werden NICHT vollständig in RAM geladen, sondern bleiben
        im Flash-Speicher und werden bei Bedarf gelesen!
        
        Args:
            hidden_size (int): LSTM Hidden Size
            input_size (int): LSTM Input Size
        
        Returns:
            dict: Detaillierte RAM-Analyse mit theoretisch vs. gemessen
        """
        
        # ========== GEMESSENE ARDUINO RAM-WERTE ==========
        # Diese Werte stammen aus realen Arduino-Messungen
        measured_ram_data = {
            16: 7.7,   # 16×16 Architektur: 7.7 kB RAM gemessen
            32: 8.9,   # 32×32 Architektur: 8.9 kB RAM gemessen
            64: 9.8    # 64×64 Architektur: 9.8 kB RAM gemessen
        }
        
        # ========== THEORETISCHE RAM-BERECHNUNG ==========
        # Basierend auf ARM Cortex-M4 Architektur und Arduino Framework
        
        # 1. KONSTANTE ARDUINO SYSTEM OVERHEADS (unabhängig von Modellgröße)
        constant_overheads = {
            'arduino_core': 1.5,        # Arduino Framework (Serial, Wire, SPI, digitalRead, etc.)
            'system_reserved': 2.0,     # ARM Cortex-M4 System Memory (Interrupt Vectors, etc.)
            'serial_buffers': 0.768,    # UART RX/TX Buffers (je 128 bytes + overhead)
            'stack_space': 2.5,         # Function Call Stack + Local Variables
            'heap_management': 0.5,     # malloc() Overhead + Memory Alignment
            'firmware_overhead': 0.7    # Arduino Boot + HAL Libraries
        }
        
        total_constant_overhead = sum(constant_overheads.values())
        
        # 2. VARIABLE KOMPONENTEN (abhängig von Hidden Size) - NUR RAM!
        variable_components = {
            # LSTM Runtime States (die EINZIGEN Modell-Daten in RAM)
            'lstm_hidden_state': (hidden_size * 4) / 1024,     # Hidden State (float32)
            'lstm_cell_state': (hidden_size * 4) / 1024,       # Cell State (float32)
            
            # Temporäre Berechnungs-Buffers
            'lstm_gate_temps': (hidden_size * 4 * 4) / 1024,   # 4 Gates × hidden_size × float32
            'activation_temps': (hidden_size * 2 * 4) / 1024,  # Sigmoid/Tanh temporäre Werte
            
            # MLP Forward Pass Buffers
            'mlp_layer1_buffer': (hidden_size * 4) / 1024,     # MLP Layer 1 Output
            'mlp_layer2_buffer': (hidden_size * 4) / 1024,     # MLP Layer 2 Output
            'mlp_output_buffer': (1 * 4) / 1024,               # Final SOC Output
            
            # I/O und Processing Buffers
            'input_buffer': (input_size * 4) / 1024,           # Input Data Buffer
            'processing_buffer': 0.5,                          # Misc computations
        }
        
        total_variable_overhead = sum(variable_components.values())
        
        # ========== FLASH vs. RAM AUFSCHLÜSSELUNG ==========
        model_params = self.calculate_complete_model_parameters(input_size, hidden_size)
        
        # FLASH: Modell-Parameter (bleiben im Flash-Speicher!)
        flash_model_weights = (model_params['total'] * 4) / 1024  # float32 = 4 bytes
        
        # RAM: Nur Runtime-Daten
        theoretical_ram_total = total_constant_overhead + total_variable_overhead
        
        # Gemessener Wert
        measured_ram_total = measured_ram_data.get(hidden_size, None)
        
        # ========== ERWEITERTE ANALYSE ==========
        # Berechne Genauigkeit der theoretischen Schätzung
        ram_prediction_error = None
        ram_accuracy_status = "unknown"
        
        if measured_ram_total is not None:
            ram_prediction_error = abs(theoretical_ram_total - measured_ram_total)
            error_percentage = (ram_prediction_error / measured_ram_total) * 100
            
            if error_percentage < 10:
                ram_accuracy_status = "excellent"
            elif error_percentage < 20:
                ram_accuracy_status = "good"
            elif error_percentage < 30:
                ram_accuracy_status = "moderate"
            else:
                ram_accuracy_status = "poor"
        
        # ========== RÜCKGABE ==========
        return {
            'hidden_size': hidden_size,
            'input_size': input_size,
            
            # RAM Analyse
            'ram_theoretical_kb': theoretical_ram_total,
            'ram_measured_kb': measured_ram_total,
            'ram_prediction_error_kb': ram_prediction_error,
            'ram_accuracy_status': ram_accuracy_status,
            
            # Flash Analyse
            'flash_model_weights_kb': flash_model_weights,
            'model_parameters_total': model_params['total'],
            
            # Detaillierte Aufschlüsselung
            'ram_breakdown': {
                'constant_overheads': constant_overheads,
                'variable_components': variable_components,
                'total_constant_kb': total_constant_overhead,
                'total_variable_kb': total_variable_overhead,
            },
            
            # Erklärungen
            'memory_architecture': {
                'flash_content': 'Model weights/parameters - stored in program memory, accessed when needed',
                'ram_content': 'Hidden states + temporary calculations + Arduino runtime - dynamic allocation',
                'key_insight': 'Model weights stay in Flash, only runtime states go to RAM!',
                'flash_vs_ram_ratio': flash_model_weights / theoretical_ram_total if theoretical_ram_total > 0 else 0
            },
            
            # Für Kompatibilität mit bestehender Visualisierung
            'total_ram_theoretical': theoretical_ram_total,
            'total_ram_measured': measured_ram_total,
            'flash_usage_kb': flash_model_weights,
            'model_parameters': model_params['total'],
        }
    def create_ram_analysis_visualization(self):
        """Erstellt umfassende RAM-Analyse Visualisierung mit theoretisch vs. gemessen"""
        
        if not self.results:
            print("❌ Keine Daten für RAM-Visualisierung")
            return
        
        # Sammle RAM-Daten aus den analysierten Modellen
        architectures = []
        hidden_sizes = []
        theoretical_ram = []
        measured_ram = []
        flash_usage = []
        ram_errors = []
        ram_accuracy_status = []
        
        for arch_name, result in self.results.items():
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis and 'lstm_architecture' in model_analysis:
                lstm_arch = model_analysis['lstm_architecture']
                hidden_size = lstm_arch.get('hidden_size')
                input_size = lstm_arch.get('input_size', 1)
                
                if hidden_size:
                    # Berechne RAM-Analyse für diese Architektur
                    ram_analysis = self.calculate_arduino_ram_usage(hidden_size, input_size)
                    
                    architectures.append(arch_name.replace('Stateful_', ''))
                    hidden_sizes.append(hidden_size)
                    theoretical_ram.append(ram_analysis['ram_theoretical_kb'])
                    measured_ram.append(ram_analysis['ram_measured_kb'] or 0)
                    flash_usage.append(ram_analysis['flash_model_weights_kb'])
                    ram_errors.append(ram_analysis['ram_prediction_error_kb'] or 0)
                    ram_accuracy_status.append(ram_analysis['ram_accuracy_status'])
        
        if not architectures:
            print("❌ Keine RAM-Daten verfügbar")
            return
        
        # Erstelle erweiterte RAM-Analyse Plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
        
        # ========== 1. RAM: Theoretisch vs. Gemessen (Hauptvergleich) ==========
        x = np.arange(len(architectures))
        width = 0.35
        
        # Farbkodierung basierend auf Genauigkeit
        colors_theoretical = ['lightblue'] * len(architectures)
        colors_measured = []
        for status in ram_accuracy_status:
            if status == 'excellent':
                colors_measured.append('green')
            elif status == 'good':
                colors_measured.append('orange')
            elif status == 'moderate':
                colors_measured.append('yellow')
            else:
                colors_measured.append('red')
        
        bars1 = ax1.bar(x - width/2, theoretical_ram, width, label='Theoretische RAM-Schätzung', 
                       alpha=0.8, color=colors_theoretical, edgecolor='navy')
        bars2 = ax1.bar(x + width/2, measured_ram, width, label='Gemessene RAM-Werte', 
                       alpha=0.8, color=colors_measured, edgecolor='darkgreen')
        
        ax1.set_xlabel('LSTM Architektur')
        ax1.set_ylabel('RAM-Verbrauch (kB)')
        ax1.set_title('Arduino RAM-Analyse: Theoretische Vorhersage vs. Reale Messung\n' + 
                     'Farbkodierung: Grün=Exzellent, Orange=Gut, Gelb=Moderat, Rot=Schlecht')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{arch}\n({hs}×{hs})' for arch, hs in zip(architectures, hidden_sizes)], rotation=0)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Fehler-Annotations und Werte auf Balken
        for i, (theor, meas, error, status) in enumerate(zip(theoretical_ram, measured_ram, ram_errors, ram_accuracy_status)):
            # Theoretische Werte
            ax1.text(bars1[i].get_x() + bars1[i].get_width()/2., theor + 0.1,
                    f'{theor:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            
            # Gemessene Werte
            if meas > 0:
                ax1.text(bars2[i].get_x() + bars2[i].get_width()/2., meas + 0.1,
                        f'{meas:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
                
                # Fehler-Annotation
                error_pct = (error / meas) * 100 if meas > 0 else 0
                ax1.text(i, max(theor, meas) + 0.8, f'±{error:.1f}kB\n({error_pct:.1f}%)', 
                        ha='center', va='bottom', fontsize=8, 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=colors_measured[i], alpha=0.6))
        
        # ========== 2. Flash vs. RAM Speicher-Aufschlüsselung ==========
        bars1 = ax2.bar(x - width/2, flash_usage, width, label='Flash (Modell-Parameter)', 
                       alpha=0.8, color='lightcoral', edgecolor='darkred')
        bars2 = ax2.bar(x + width/2, measured_ram, width, label='RAM (Runtime)', 
                       alpha=0.8, color='lightgreen', edgecolor='darkgreen')
        
        ax2.set_xlabel('Architektur')
        ax2.set_ylabel('Speicherverbrauch (kB)')
        ax2.set_title('Speicher-Architektur: Flash vs. RAM\nFlash: Modell-Gewichte | RAM: Hidden States + Arduino Runtime')
        ax2.set_xticks(x)
        ax2.set_xticklabels([f'{arch}\n({hs}×{hs})' for arch, hs in zip(architectures, hidden_sizes)], rotation=0)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Flash vs RAM Verhältnis annotieren
        for i, (flash, ram, arch) in enumerate(zip(flash_usage, measured_ram, architectures)):
            if flash > 0 and ram > 0:
                ratio = flash / ram
                ax2.text(i, max(flash, ram) + 1, f'Ratio: {ratio:.1f}:1\n(Flash:RAM)', 
                        ha='center', va='bottom', fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8))
        
        # ========== 3. RAM-Komponenten Aufschlüsselung (Stacked Bar) ==========
        if hidden_sizes:
            # Verwende mittlere Architektur als Beispiel für detaillierte Aufschlüsselung
            mid_idx = len(hidden_sizes) // 2
            example_hidden_size = hidden_sizes[mid_idx]
            example_arch = architectures[mid_idx]
            
            ram_analysis = self.calculate_arduino_ram_usage(example_hidden_size, 1)
            
            # Konstante Komponenten
            const_components = ram_analysis['ram_breakdown']['constant_overheads']
            var_components = ram_analysis['ram_breakdown']['variable_components']
            
            # Alle Komponenten für detaillierte Ansicht
            all_components = {}
            all_components.update(const_components)
            all_components.update(var_components)
            
            # Sortiere nach Größe
            sorted_components = sorted(all_components.items(), key=lambda x: x[1], reverse=True)
            comp_names = [name.replace('_', ' ').title() for name, _ in sorted_components]
            comp_values = [value for _, value in sorted_components]
            
            # Stacked Bar Chart für alle Architekturen mit vereinfachten Kategorien
            categories = ['Arduino System', 'LSTM States', 'Processing Buffers', 'I/O Buffers']
            category_data = []
            
            for arch_idx, (arch, hidden_size) in enumerate(zip(architectures, hidden_sizes)):
                ram_analysis = self.calculate_arduino_ram_usage(hidden_size, 1)
                breakdown = ram_analysis['ram_breakdown']
                
                # Gruppiere Komponenten in Kategorien
                arduino_system = breakdown['constant_overheads']['arduino_core'] + \
                               breakdown['constant_overheads']['system_reserved'] + \
                               breakdown['constant_overheads']['serial_buffers'] + \
                               breakdown['constant_overheads']['stack_space'] + \
                               breakdown['constant_overheads']['heap_management'] + \
                               breakdown['constant_overheads']['firmware_overhead']
                
                lstm_states = breakdown['variable_components']['lstm_hidden_state'] + \
                            breakdown['variable_components']['lstm_cell_state']
                
                processing_buffers = breakdown['variable_components']['lstm_gate_temps'] + \
                                   breakdown['variable_components']['activation_temps'] + \
                                   breakdown['variable_components']['mlp_layer1_buffer'] + \
                                   breakdown['variable_components']['mlp_layer2_buffer'] + \
                                   breakdown['variable_components']['mlp_output_buffer'] + \
                                   breakdown['variable_components']['processing_buffer']
                
                io_buffers = breakdown['variable_components']['input_buffer']
                
                category_data.append([arduino_system, lstm_states, processing_buffers, io_buffers])
            
            # Transponiere für Stacking
            category_data = np.array(category_data).T
            
            bottom = np.zeros(len(architectures))
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
            
            for i, (category, data, color) in enumerate(zip(categories, category_data, colors)):
                bars = ax3.bar(x, data, bottom=bottom, label=category, color=color, alpha=0.8)
                bottom += data
                
                # Annotiere nur wenn genug Platz
                for j, (bar, value) in enumerate(zip(bars, data)):
                    if value > 0.3:  # Nur wenn Segment groß genug
                        ax3.text(bar.get_x() + bar.get_width()/2., 
                                bar.get_y() + bar.get_height()/2.,
                                f'{value:.1f}', ha='center', va='center', 
                                fontweight='bold', fontsize=8, color='white')
            
            ax3.set_xlabel('Architektur')
            ax3.set_ylabel('RAM-Verbrauch (kB)')
            ax3.set_title(f'RAM-Komponenten Aufschlüsselung nach Kategorien')
            ax3.set_xticks(x)
            ax3.set_xticklabels([f'{arch}\n({hs}×{hs})' for arch, hs in zip(architectures, hidden_sizes)], rotation=0)
            ax3.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax3.grid(True, alpha=0.3)
        
        # ========== 4. Speicher-Effizienz und Skalierung ==========
        # Zeige Skalierungsverhalten und Effizienz-Metriken
        
        # Parameter vs. RAM Effizienz
        model_parameters = []
        for arch_name, result in self.results.items():
            if arch_name.replace('Stateful_', '') in architectures:
                model_analysis = result.get('pytorch_model_analysis')
                if model_analysis:
                    total_params = model_analysis.get('total_parameters', 0)
                    model_parameters.append(total_params)
        
        # Scatter Plot: Parameter vs. RAM
        colors_scatter = [colors_measured[i] for i in range(len(architectures))]
        scatter = ax4.scatter(model_parameters, measured_ram, c=colors_scatter, s=150, alpha=0.8, edgecolors='black')
        
        # Trend-Linie
        if len(model_parameters) > 1:
            z = np.polyfit(model_parameters, measured_ram, 1)
            p = np.poly1d(z)
            x_trend = np.linspace(min(model_parameters), max(model_parameters), 100)
            ax4.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend: RAM ≈ {z[0]:.2e}×Params + {z[1]:.1f}')
        
        # Annotationen für Architekturen
        for i, (arch, params, ram) in enumerate(zip(architectures, model_parameters, measured_ram)):
            ax4.annotate(f'{arch}\n{params:,} params', (params, ram), 
                        xytext=(10, 10), textcoords='offset points', 
                        fontsize=8, ha='left',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax4.set_xlabel('Modell-Parameter (Anzahl)')
        ax4.set_ylabel('RAM-Verbrauch (kB)')
        ax4.set_title('Speicher-Effizienz: Modell-Komplexität vs. RAM-Bedarf\nFarbkodierung: Vorhersage-Genauigkeit')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # RAM-Limit Linie (Arduino Uno R4: 32KB)
        ax4.axhline(y=32, color='red', linestyle='--', alpha=0.7, 
                   label='Arduino Uno R4 RAM Limit (32 KB)')
        ax4.legend()
        
        plt.tight_layout()
        
        # Speichere RAM-Analyse Plot
        plot_filename = "comprehensive_ram_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n🧠 Umfassende RAM-Analyse gespeichert: {plot_filename}")
        
        # Drucke detaillierte RAM-Analyse Zusammenfassung
        self.print_ram_analysis_summary(architectures, hidden_sizes, theoretical_ram, measured_ram, ram_errors, ram_accuracy_status)
        
        return fig
    
    def create_comprehensive_visualization(self):
        """Erstellt umfassende Visualisierung aller Ergebnisse"""
        
        if not self.results:
            print("❌ Keine Daten für Visualisierung")
            return
          # Sammle Daten für Plots
        architectures = []
        script_params = []
        model_params = []
        file_sizes = []
        theoretical_file_sizes = []  # NEU: Theoretische Dateigröße
        hidden_sizes = []
        validation_status = []
        
        for arch_name, result in self.results.items():
            architectures.append(arch_name)            # Parameter aus Skript
            script_analysis = result.get('training_script_analysis')
            if script_analysis and 'architecture' in script_analysis:
                hidden = script_analysis['architecture'].get('hidden_size', 0)
                input_size = script_analysis['architecture'].get('input_size', 4)  # Default 4
                
                # Dynamische MLP-Architektur: [hidden_size, hidden_size, 1] 
                # Entspricht der tatsächlichen Architektur in allen Training-Skripte
                param_calc = self.calculate_complete_model_parameters(input_size, hidden, None)
                script_param_count = param_calc['total']
            else:
                script_param_count = 0
            script_params.append(script_param_count)
            
            # Parameter aus Model
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis:
                actual_params = model_analysis.get('total_parameters', 0)
                actual_file_size = model_analysis.get('file_size_kb', 0)
                model_params.append(actual_params)
                file_sizes.append(actual_file_size)
                
                # NEU: Theoretische Dateigröße berechnen
                # PyTorch .pth Dateien speichern Parameter als 32-bit floats (4 bytes pro Parameter)
                # Plus Overhead für Metadaten (ca. 10-20% zusätzlich)
                theoretical_size_kb = (actual_params * 4) / 1024  # 4 bytes pro float32, in kB
                theoretical_file_sizes.append(theoretical_size_kb)
                
                lstm_arch = model_analysis.get('lstm_architecture', {})
                hidden_sizes.append(lstm_arch.get('hidden_size', 0))
            else:
                model_params.append(0)
                file_sizes.append(0)
                theoretical_file_sizes.append(0)
                hidden_sizes.append(0)
            
            # Validierungsstatus
            comparison = result.get('comparison', {})
            validation_status.append(comparison.get('validation', 'unknown'))
          # Erstelle Plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # ========== 1. Parameter-Vergleich: Skript vs. Model ==========
        x = np.arange(len(architectures))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, script_params, width, label='Formel (LSTM+MLP)', alpha=0.7, color='lightblue')
        bars2 = ax1.bar(x + width/2, model_params, width, label='PyTorch Model (tatsächlich)', alpha=0.7, color='orange')
        
        ax1.set_xlabel('Architektur')
        ax1.set_ylabel('Anzahl Parameter')
        ax1.set_title('Parameter-Vergleich: Theoretische Formel vs. PyTorch Model')
        ax1.set_xticks(x)
        ax1.set_xticklabels(architectures, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Werte auf Balken
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
                            f'{int(height):,}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # ========== 2. Dateigröße vs. Parameter (Theoretisch vs. Gemessen) ==========
        colors = ['green' if status == 'match' else 'orange' if status == 'mismatch' else 'red' 
                 for status in validation_status]
        
        # Scatter Plot für gemessene Werte
        scatter1 = ax2.scatter(model_params, file_sizes, c=colors, s=100, alpha=0.7, 
                              label='Gemessene Dateigröße', marker='o')
        
        # Scatter Plot für theoretische Werte  
        scatter2 = ax2.scatter(model_params, theoretical_file_sizes, c='blue', s=80, alpha=0.5,
                              label='Theoretische Dateigröße (4 bytes/param)', marker='^')
        
        # Verbindungslinien zwischen theoretisch und gemessen
        for i in range(len(model_params)):
            if model_params[i] > 0:
                ax2.plot([model_params[i], model_params[i]], [theoretical_file_sizes[i], file_sizes[i]], 
                        'k--', alpha=0.3, linewidth=1)
        
        # Annotationen für Architekturen
        for i, arch in enumerate(architectures):
            if model_params[i] > 0:
                ax2.annotate(arch, (model_params[i], file_sizes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
          # 1. Parameter-Vergleich: Skript vs. Model
        x = np.arange(len(architectures))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, script_params, width, label='Formel (LSTM+MLP)', alpha=0.7, color='lightblue')
        bars2 = ax1.bar(x + width/2, model_params, width, label='PyTorch Model (tatsächlich)', alpha=0.7, color='orange')
        
        ax1.set_xlabel('Architektur')
        ax1.set_ylabel('Anzahl Parameter')
        ax1.set_title('Parameter-Vergleich: Theoretische Formel vs. PyTorch Model')
        ax1.set_xticks(x)
        ax1.set_xticklabels(architectures, rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Werte auf Balken
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                            f'{int(height):,}', ha='center', va='bottom', fontsize=9)
          # 2. Dateigröße vs. Parameter (Theoretisch vs. Gemessen)
        colors = ['green' if status == 'match' else 'orange' if status == 'mismatch' else 'red' 
                 for status in validation_status]
        
        # Scatter Plot für gemessene Werte
        scatter1 = ax2.scatter(model_params, file_sizes, c=colors, s=100, alpha=0.7, 
                              label='Gemessene Dateigröße', marker='o')
        
        # Scatter Plot für theoretische Werte  
        scatter2 = ax2.scatter(model_params, theoretical_file_sizes, c='blue', s=80, alpha=0.5,
                              label='Theoretische Dateigröße (4 bytes/param)', marker='^')
        
        # Verbindungslinien zwischen theoretisch und gemessen
        for i in range(len(model_params)):
            if model_params[i] > 0:
                ax2.plot([model_params[i], model_params[i]], 
                        [theoretical_file_sizes[i], file_sizes[i]], 
                        'k--', alpha=0.3, linewidth=1)
        
        # Annotationen für Architekturen
        for i, arch in enumerate(architectures):
            if model_params[i] > 0:
                # Berechne Overhead-Prozent
                overhead_percent = ((file_sizes[i] - theoretical_file_sizes[i]) / theoretical_file_sizes[i]) * 100
                
                ax2.annotate(f'{arch}\n+{overhead_percent:.1f}%', 
                           (model_params[i], file_sizes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax2.set_xlabel('Anzahl Parameter')
        ax2.set_ylabel('Dateigröße (kB)')
        ax2.set_title('Dateigröße vs. Parameter-Anzahl\n(Theoretisch vs. Gemessen)')
        ax2.grid(True, alpha=0.3)
        
        # Erweiterte Legende  
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, alpha=0.7, label='Gemessene Dateigröße'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=8, alpha=0.5, label='Theoretische Dateigröße (4 bytes/param)'),
            Patch(facecolor='green', alpha=0.7, label='Formula ↔ Model Match'),
            Patch(facecolor='orange', alpha=0.7, label='Formula ↔ Model Mismatch'),
            Patch(facecolor='red', alpha=0.7, label='Unbekannt/Fehler')
        ]
        ax2.legend(handles=legend_elements, fontsize=8)
        
        # ========== 3. Hidden Size Scaling ==========
        if hidden_sizes and any(h > 0 for h in hidden_sizes):
            theoretical_curve_x = np.arange(8, max(hidden_sizes) + 16, 8)
            # Dynamische MLP-Architektur: [h, h, 1] für jede Hidden-Size
            theoretical_curve_y = []
            for h in theoretical_curve_x:
                param_calc = self.calculate_complete_model_parameters(4, h, None)  # input_size=4, None=[h,h,1]
                theoretical_curve_y.append(param_calc['total'])
            
            ax3.plot(theoretical_curve_x, theoretical_curve_y, 'b-', linewidth=2, 
                    label='Theoretische Kurve (LSTM + MLP)')
            ax3.scatter(hidden_sizes, model_params, c=colors, s=100, alpha=0.7, 
                       label='Gemessene Modelle')
            
            for i, arch in enumerate(architectures):
                if hidden_sizes[i] > 0:
                    ax3.annotate(arch, (hidden_sizes[i], model_params[i]), 
                               xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax3.set_xlabel('Hidden Size')
            ax3.set_ylabel('Anzahl Parameter')
            ax3.set_title('Parameter-Skalierung mit Hidden Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Keine Hidden Size\nDaten verfügbar', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        
        # ========== 4. Erweiterte Zusammenfassung mit Dateigröße-Analyse ==========
        summary_text = "MODELL-ANALYSE ZUSAMMENFASSUNG\n\n"
        
        # Header für tabellarische Darstellung
        summary_text += f"{'Arch':<12} | {'Param':<7} | {'Gemessen':<8} | {'Theor.':<8} | {'Overhead':<8}\n"
        summary_text += "-" * 60 + "\n"
        
        for arch_name, result in self.results.items():
            validation = result.get('comparison', {}).get('validation', 'unknown')
            status_symbol = "✅" if validation == 'match' else "⚠️" if validation == 'mismatch' else "❌"
            
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis:
                params = model_analysis.get('total_parameters', 0)
                measured_kb = model_analysis.get('file_size_kb', 0)
                theoretical_kb = (params * 4) / 1024  # 4 bytes pro float32
                
                if theoretical_kb > 0:
                    overhead_percent = ((measured_kb - theoretical_kb) / theoretical_kb) * 100
                else:
                    overhead_percent = 0
                
                # Kurzer Architekturname
                short_arch = arch_name.replace('Stateful_', '').replace(' copy', '*')
                
                summary_text += f"{short_arch:<12} | {params//1000:>4}k | {measured_kb:>6.1f}kB | {theoretical_kb:>6.1f}kB | +{overhead_percent:>5.1f}%\n"
        
        summary_text += "\n📊 DATEIGRÖSSE-OVERHEAD-ANALYSE:\n"
        summary_text += "• Theoretisch: 4 bytes pro Parameter (float32)\n"
        summary_text += "• Gemessen: Inklusive PyTorch Metadaten\n"
        summary_text += "• Overhead: Zusätzliche Dateigröße durch\n"
        summary_text += "  Modellstruktur, Optimizer-State, etc.\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=9, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Speichere Plot
        plot_filename = "comprehensive_model_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Umfassende Analyse gespeichert: {plot_filename}")
        
        plt.show()
        return fig
    def run_complete_analysis(self):
        """Führt die komplette Analyse durch"""
        
        print("="*80)
        print("UMFASSENDE MODELL-ARCHITEKTUR ANALYSE")
        print("="*80)
        print(f"Basis-Pfad: {self.base_path}")
        
        # Finde alle Model-Verzeichnisse
        model_dirs = self.find_model_directories()
        
        if not model_dirs:
            print("❌ Keine Model-Verzeichnisse gefunden!")
            return
        
        print(f"\n🔍 Gefundene Architekturen: {len(model_dirs)}")
        for info in model_dirs:
            print(f"   - {info['architecture']}")
        
        # Analysiere jede Architektur
        for arch_info in model_dirs:
            result = self.analyze_architecture_directory(arch_info)
            self.results[arch_info['architecture']] = result
        
        # Erstelle Gesamtübersicht
        self.print_summary()
        
        # Erstelle Visualisierungen
        print(f"\n{'='*80}")
        print("ERSTELLE VISUALISIERUNGEN")
        print(f"{'='*80}")
        
        # 1. Umfassende Modell-Analyse
        print("📊 Erstelle umfassende Modell-Analyse...")
        self.create_comprehensive_visualization()
        
        # 2. RAM-Analyse mit theoretisch vs. gemessen
        print("🧠 Erstelle detaillierte RAM-Analyse...")
        self.create_ram_analysis_visualization()
        
        return self.results
    
    def print_summary(self):
        """Druckt eine Zusammenfassung aller Ergebnisse"""
        
        print(f"\n{'='*80}")
        print("GESAMTÜBERSICHT")
        print(f"{'='*80}")
        
        print(f"{'Architektur':<15} | {'Script Hidden':<12} | {'Model Hidden':<12} | {'Parameter':<10} | {'Status':<10}")
        print("-" * 80)
        
        for arch_name, result in self.results.items():
            # Script Info
            script_analysis = result.get('training_script_analysis')
            script_hidden = 'N/A'
            if script_analysis and 'architecture' in script_analysis:
                script_hidden = str(script_analysis['architecture'].get('hidden_size', 'N/A'))
            
            # Model Info
            model_analysis = result.get('pytorch_model_analysis')
            model_hidden = 'N/A'
            total_params = 0
            if model_analysis:
                lstm_arch = model_analysis.get('lstm_architecture', {})
                model_hidden = str(lstm_arch.get('hidden_size', 'N/A'))
                total_params = model_analysis.get('total_parameters', 0)
            
            # Status
            comparison = result.get('comparison', {})
            status = comparison.get('validation', 'unknown')
            status_symbol = "✅ Match" if status == 'match' else "⚠️ Mismatch" if status == 'mismatch' else "❌ Error"
            
            print(f"{arch_name:<15} | {script_hidden:<12} | {model_hidden:<12} | {total_params:<10,} | {status_symbol:<10}")
    
    def print_ram_analysis_summary(self, architectures, hidden_sizes, theoretical_ram, measured_ram, ram_errors, ram_accuracy_status):
        """Druckt detaillierte RAM-Analyse Zusammenfassung"""
        
        print(f"\n{'='*80}")
        print("RAM-ANALYSE ZUSAMMENFASSUNG")
        print(f"{'='*80}")
        
        print(f"{'Architektur':<12} | {'Hidden':<7} | {'Theor.':<8} | {'Gemessen':<9} | {'Fehler':<8} | {'Status':<12}")
        print("-" * 80)
        
        for i, arch in enumerate(architectures):
            if i < len(theoretical_ram) and i < len(measured_ram):
                theor = theoretical_ram[i]
                meas = measured_ram[i]
                error = ram_errors[i]
                status = ram_accuracy_status[i]
                hidden = hidden_sizes[i]
                
                error_pct = (error / meas * 100) if meas > 0 and error > 0 else 0
                
                status_symbol = {
                    'excellent': '✅ Exzellent',
                    'good': '🟢 Gut', 
                    'moderate': '🟡 Moderat',
                    'poor': '🔴 Schlecht',
                    'unknown': '❓ Unbekannt'
                }.get(status, '❓ Unbekannt')
                
                print(f"{arch:<12} | {hidden:>7} | {theor:>6.1f}kB | {meas:>7.1f}kB | ±{error:>5.1f}kB | {status_symbol:<12}")
        
        print(f"\n💡 WICHTIGE ERKENNTNISSE:")
        print(f"• Modell-Gewichte bleiben im FLASH-Speicher")
        print(f"• Nur Hidden States + Arduino Framework gehen in RAM")
        print(f"• RAM-Bedarf skaliert hauptsächlich mit Hidden Size")
        print(f"• Arduino System-Overhead (~7kB) ist konstant")
        
        # Berechne Gesamtstatistiken
        if measured_ram and all(m > 0 for m in measured_ram):
            avg_error_pct = sum((e / m * 100) for e, m in zip(ram_errors, measured_ram) if e > 0 and m > 0) / len(ram_errors)
            print(f"• Durchschnittlicher Vorhersage-Fehler: {avg_error_pct:.1f}%")
        
        print(f"\n🔬 METHODIK:")
        print(f"• Theoretische Berechnung: ARM Cortex-M4 basierte Formeln")
        print(f"• Gemessene Werte: Reale Arduino Uno R4 Messungen") 
        print(f"• Validierung: Vergleich mit Hardware-Monitoring")

def main():
    """Hauptfunktion"""
    
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup"
    
    analyzer = ModelArchitectureAnalyzer(base_path)
    results = analyzer.run_complete_analysis()
    
    print(f"\n{'='*80}")
    print("ANALYSE ABGESCHLOSSEN")
    print(f"{'='*80}")
    print(f"✅ {len(results)} Architekturen analysiert")
    print("📊 Visualisierungen erstellt")
    print("📋 Ergebnisse verfügbar in analyzer.results")

if __name__ == "__main__":
    main()
