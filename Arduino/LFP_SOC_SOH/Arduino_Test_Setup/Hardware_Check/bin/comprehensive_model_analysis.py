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
    def calculate_arduino_flash_ram_estimation(self, hidden_size, input_size=1):
        """
        PRÄZISE Arduino Flash/RAM-Abschätzung basierend auf realen Compiler-Messungen
        
        Analysiert basierend auf deinen gemessenen Werten:
        32×32 Modell: 93 kB Flash (35%) / 8.4 kB RAM (25%)
        
        FLASH-Komponenten:
        - Arduino Code Base (~65 kB konstant)
        - LSTM Gewichte (Parameter × 4 bytes als const float arrays)
        - Bibliotheken (Math, Serial, etc.)
        
        RAM-Komponenten:
        - LSTM Hidden/Cell States (2 × hidden_size × 4 bytes)
        - I/O Buffers, globale Variablen
        - Arduino Framework Runtime
        - Stack Space für lokale Variablen
        
        Args:
            hidden_size (int): LSTM Hidden Size
            input_size (int): LSTM Input Size (default 1)
        
        Returns:
            dict: Detaillierte Flash/RAM-Abschätzung mit Compiler-Vergleich
        """
          # ========== REALE COMPILER-MESSUNGEN ==========
        # Aus deinen Arduino IDE Outputs
        real_measurements = {
            16: {  # 16×16 Architektur
                'flash_used_kb': 48.8,       # Gemessene Flash-Nutzung
                'ram_used_kb': 7.7,          # Gemessene RAM-Nutzung
                'parameters': 1969,          # Aus Modell-Berechnung
                'flash_percent': 18.6,       # "Flash percentage"
                'ram_percent': 24.1,         # "RAM percentage"
            },
            32: {  # 32×32 Architektur
                'flash_used_kb': 106.9,      # Korrigierte Flash-Nutzung
                'ram_used_kb': 8.9,          # Korrigierte RAM-Nutzung
                'parameters': 7009,          # Aus Modell-Berechnung
                'flash_percent': 40.8,       # "Flash percentage"
                'ram_percent': 27.8,         # "RAM percentage"
            },
            64: {  # 64×64 Architektur
                'flash_used_kb': 123.0,      # Gemessene Flash-Nutzung
                'ram_used_kb': 9.8,          # Gemessene RAM-Nutzung
                'parameters': 26305,         # Aus Modell-Berechnung
                'flash_percent': 46.9,       # "Flash percentage"
                'ram_percent': 30.6,         # "RAM percentage"
            }
            # Weitere Messungen können hier ergänzt werden
        }
        
        # ========== FLASH-ABSCHÄTZUNG ==========
        # Berechne Modell-Parameter
        model_params = self.calculate_complete_model_parameters(input_size, hidden_size)
        total_parameters = model_params['total']        # Flash-Komponenten (FINE-TUNED ADAPTIVE SCHÄTZUNG basierend auf realen Messungen)
        # Target: 16x16=48.8kB, 32x32=106.9kB, 64x64=123.0kB
        # Balanciere lineare und quadratische Terme für beste Genauigkeit
        
        # Basis-Code: moderater quadratischer Anstieg 
        adaptive_base = 20.0 + (hidden_size * 0.8) + (hidden_size * hidden_size * 0.008)
        
        flash_components = {
            'arduino_adaptive_base': adaptive_base,
            'model_weights_kb': (total_parameters * 4) / 1024,
            'compiler_optimization': hidden_size * 0.35,  # Moderater Overhead
        }
        
        estimated_flash_kb = sum(flash_components.values())
          # ========== RAM-ABSCHÄTZUNG (VERBESSERTE MODELL-ABHÄNGIGE SCHÄTZUNG) ==========
        # RAM-Komponenten (nur Runtime-Daten!)
        ram_components = {
            # 1. ARDUINO SYSTEM-OVERHEADS (größtenteils konstant)
            'arduino_framework': 2.2,       # Arduino Core Runtime
            'serial_buffers': 0.8,          # UART TX/RX Buffers (größer für Debug)
            'system_reserved': 1.3,         # ARM Cortex-M4 System RAM
            'stack_space': 1.8 + (hidden_size * 0.01),  # Stack wächst mit Modellkomplexität
            'heap_overhead': 0.4,           # malloc/free Management
            'interrupt_vectors': 0.5,       # Interrupt Handler Memory
            
            # 2. MODELL-RUNTIME (stark abhängig von Hidden Size)
            'lstm_hidden_state': (hidden_size * 4) / 1024,    # float32 Hidden State
            'lstm_cell_state': (hidden_size * 4) / 1024,      # float32 Cell State
            'lstm_temp_buffers': (hidden_size * 12) / 1024,   # Temporäre Gate-Berechnungen (4 Gates * 3 Temp Arrays)
            'mlp_activation_buffers': (hidden_size * 6) / 1024,  # MLP Layer Activations
            
            # 3. I/O UND PROCESSING BUFFERS
            'input_buffer': (input_size * 8 * 4) / 1024,      # Rolling Input Buffer (8 Steps)
            'output_buffer': 0.15,          # SOC Output + Debug Info
            'global_variables': 0.3 + (hidden_size * 0.005), # Globale Vars (skaliert leicht)
        }
        
        estimated_ram_kb = sum(ram_components.values())
        
        # ========== VALIDIERUNG MIT REALEN MESSUNGEN ==========
        validation = {}
        measured_data = real_measurements.get(hidden_size)
        
        if measured_data:
            # Flash-Vergleich
            flash_error_kb = abs(estimated_flash_kb - measured_data['flash_used_kb'])
            flash_error_percent = (flash_error_kb / measured_data['flash_used_kb']) * 100
            
            # RAM-Vergleich  
            ram_error_kb = abs(estimated_ram_kb - measured_data['ram_used_kb'])
            ram_error_percent = (ram_error_kb / measured_data['ram_used_kb']) * 100
            
            # Parameter-Vergleich
            param_match = (total_parameters == measured_data['parameters'])
            
            validation = {
                'has_real_data': True,
                'flash_measured_kb': measured_data['flash_used_kb'],
                'flash_error_kb': flash_error_kb,
                'flash_error_percent': flash_error_percent,
                'flash_accuracy': 'excellent' if flash_error_percent < 5 else 
                                'good' if flash_error_percent < 10 else 
                                'moderate' if flash_error_percent < 20 else 'poor',
                
                'ram_measured_kb': measured_data['ram_used_kb'], 
                'ram_error_kb': ram_error_kb,
                'ram_error_percent': ram_error_percent,
                'ram_accuracy': 'excellent' if ram_error_percent < 10 else
                              'good' if ram_error_percent < 20 else
                              'moderate' if ram_error_percent < 30 else 'poor',
                
                'parameter_match': param_match,
                'measured_flash_percent': measured_data['flash_percent'],
                'measured_ram_percent': measured_data['ram_percent'],
            }
        else:
            validation = {
                'has_real_data': False,
                'note': f'Keine realen Messungen für {hidden_size}×{hidden_size} verfügbar'
            }
        
        # ========== ARDUINO UNO R4 LIMITS ==========
        arduino_r4_specs = {
            'flash_total_kb': 256,          # 256 kB Flash
            'ram_total_kb': 32,             # 32 kB SRAM
            'flash_available_percent': (estimated_flash_kb / 256) * 100,
            'ram_available_percent': (estimated_ram_kb / 32) * 100,
            'flash_fits': estimated_flash_kb < 256,
            'ram_fits': estimated_ram_kb < 32,
            'overall_fits': (estimated_flash_kb < 256) and (estimated_ram_kb < 32)        }        # ========== SKALIERUNGS-VORHERSAGEN ==========
        # Für andere Hidden Sizes basierend auf der IDENTISCHEN fine-tuned adaptiven Formel
        scaling_predictions = {}
        
        for test_hidden in [16, 24, 32, 48, 64, 96, 128]:
            test_params = self.calculate_complete_model_parameters(input_size, test_hidden)['total']
            
            # IDENTISCHE Flash-Formel wie in der Hauptberechnung verwenden
            test_adaptive_base = 20.0 + (test_hidden * 0.8) + (test_hidden * test_hidden * 0.008)
            test_flash_components = {
                'arduino_adaptive_base': test_adaptive_base,
                'model_weights_kb': (test_params * 4) / 1024,
                'compiler_optimization': test_hidden * 0.35,
            }
            test_flash = sum(test_flash_components.values())
            
            # IDENTISCHE RAM-Formel wie in der Hauptberechnung
            test_arduino_base = 2.2 + 0.8 + 1.3 + 1.8 + 0.4 + 0.5  # Konstant ~6.9kB
            test_ram_components = {
                'arduino_base': test_arduino_base,
                'lstm_hidden_state': (test_hidden * 4) / 1024,
                'lstm_cell_state': (test_hidden * 4) / 1024,
                'lstm_temp_vars': (test_hidden * 6 * 4) / 1024,
                'mlp_intermediate': (test_hidden * 2 * 4) / 1024,
                'input_output_buffers': 0.5,
            }
            test_ram = sum(test_ram_components.values())
            
            scaling_predictions[test_hidden] = {
                'flash_kb': test_flash,
                'ram_kb': test_ram,
                'parameters': test_params,
                'fits_arduino_r4': (test_flash < 256) and (test_ram < 32)
            }
        
        # ========== RÜCKGABE ==========
        return {
            'architecture': f'{hidden_size}×{hidden_size}',
            'input_size': input_size,
            'total_parameters': total_parameters,
            
            # Flash-Analyse
            'flash_estimated_kb': estimated_flash_kb,
            'flash_components': flash_components,
            
            # RAM-Analyse  
            'ram_estimated_kb': estimated_ram_kb,
            'ram_components': ram_components,
            
            # Arduino R4 Kompatibilität
            'arduino_r4_compatibility': arduino_r4_specs,
            
            # Validierung mit realen Messungen
            'validation': validation,
            
            # Skalierungs-Vorhersagen
            'scaling_predictions': scaling_predictions,
            
            # Schlüssel-Erkenntnisse
            'key_insights': {
                'flash_content': 'Arduino Code (~78 kB) + Model Weights (const arrays)',
                'ram_content': 'LSTM States + I/O Buffers + Arduino Runtime (NO model weights!)',
                'weight_storage': 'Model weights stay in Flash as const float arrays',
                'critical_bottleneck': 'Flash' if estimated_flash_kb > estimated_ram_kb * 3 else 'RAM',
                'max_recommended_hidden_size': max([h for h, pred in scaling_predictions.items() 
                                                  if pred['fits_arduino_r4']], default=0)
            }
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
        """Erstellt umfassende Visualisierung aller Ergebnisse - das 4-Panel Diagramm"""
        
        if not self.results:
            print("❌ Keine Daten für Visualisierung")
            return
        
        # Sammle Daten für Plots
        architectures = []
        script_params = []
        model_params = []
        file_sizes = []
        theoretical_file_sizes = []
        hidden_sizes = []
        validation_status = []
        
        for arch_name, result in self.results.items():
            architectures.append(arch_name.replace('Stateful_', ''))
            
            # Parameter aus Skript
            script_analysis = result.get('training_script_analysis')
            if script_analysis and 'architecture' in script_analysis:
                hidden = script_analysis['architecture'].get('hidden_size', 0)
                input_size = script_analysis['architecture'].get('input_size', 4)
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
                
                # Theoretische Dateigröße (4 bytes pro float32 Parameter)
                theoretical_size_kb = (actual_params * 4) / 1024
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
        
        # Erstelle 4-Panel Plot wie im Original
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # ========== 1. Parameter-Vergleich: Formel vs. PyTorch Model ==========
        x = np.arange(len(architectures))
        width = 0.35
        
        bars1 = ax1.bar(x - width/2, script_params, width, label='Formel (LSTM+MLP)', 
                       alpha=0.7, color='lightblue', edgecolor='darkblue')
        bars2 = ax1.bar(x + width/2, model_params, width, label='PyTorch Model (tatsächlich)', 
                       alpha=0.7, color='orange', edgecolor='darkorange')
        
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
                ax2.plot([model_params[i], model_params[i]], 
                        [theoretical_file_sizes[i], file_sizes[i]], 
                        'k--', alpha=0.3, linewidth=1)
        
        # Annotationen für Architekturen
        for i, arch in enumerate(architectures):
            if model_params[i] > 0:
                overhead_percent = ((file_sizes[i] - theoretical_file_sizes[i]) / theoretical_file_sizes[i]) * 100
                ax2.annotate(f'{arch}\n+{overhead_percent:.1f}%', 
                           (model_params[i], file_sizes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax2.set_xlabel('Anzahl Parameter')
        ax2.set_ylabel('Dateigröße (kB)')
        ax2.set_title('Dateigröße vs. Parameter-Anzahl\n(Theoretisch vs. Gemessen)')
        ax2.grid(True, alpha=0.3)
        
        # Legende
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, 
                   alpha=0.7, label='Gemessene Dateigröße'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=8, 
                   alpha=0.5, label='Theoretische Dateigröße (4 bytes/param)'),
            Patch(facecolor='green', alpha=0.7, label='Formula ↔ Model Match'),
            Patch(facecolor='orange', alpha=0.7, label='Formula ↔ Model Mismatch'),
            Patch(facecolor='red', alpha=0.7, label='Unbekannt/Fehler')
        ]
        ax2.legend(handles=legend_elements, fontsize=8)
        
        # ========== 3. Parameter-Skalierung mit Hidden Size ==========
        if hidden_sizes and any(h > 0 for h in hidden_sizes):
            # Theoretische Kurve
            theoretical_curve_x = np.arange(8, max(hidden_sizes) + 16, 8)
            theoretical_curve_y = []
            for h in theoretical_curve_x:
                param_calc = self.calculate_complete_model_parameters(4, h, None)
                theoretical_curve_y.append(param_calc['total'])
            
            ax3.plot(theoretical_curve_x, theoretical_curve_y, 'b-', linewidth=2, 
                    label='Theoretische Kurve (LSTM + MLP)')
            ax3.scatter(hidden_sizes, model_params, c=colors, s=100, alpha=0.7, 
                       label='Gemessene Modelle')
            
            # Annotationen
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
        
        # ========== 4. Modell-Analyse Zusammenfassung ==========
        summary_text = "MODELL-ANALYSE ZUSAMMENFASSUNG\n\n"
        
        # Header
        summary_text += f"{'Arch':<12} | {'Param':<7} | {'Gemessen':<8} | {'Theor.':<8} | {'Overhead':<8}\n"
        summary_text += "-" * 60 + "\n"
        
        # Daten für jede Architektur
        for arch_name, result in self.results.items():
            validation = result.get('comparison', {}).get('validation', 'unknown')
            status_symbol = "✅" if validation == 'match' else "⚠️" if validation == 'mismatch' else "❌"
            
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis:
                params = model_analysis.get('total_parameters', 0)
                measured_kb = model_analysis.get('file_size_kb', 0)
                theoretical_kb = (params * 4) / 1024
                
                if theoretical_kb > 0:
                    overhead_percent = ((measured_kb - theoretical_kb) / theoretical_kb) * 100
                else:
                    overhead_percent = 0
                
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
        ax2.legend(handles=legend_elements, fontsize=8)# 3. Hidden Size Scaling
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
          # 4. Erweiterte Zusammenfassung mit Dateigröße-Analyse
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
    
    def create_detailed_memory_breakdown_visualization(self):
        """Erstellt detaillierte Memory-Breakdown Visualisierung mit Erklärungen"""
        
        if not self.results:
            print("❌ Keine Daten für Memory-Breakdown")
            return
        
        # Sammle Daten
        architectures = []
        memory_data = []
        
        for arch_name, result in self.results.items():
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis and 'lstm_architecture' in model_analysis:
                lstm_arch = model_analysis['lstm_architecture']
                hidden_size = lstm_arch.get('hidden_size')
                input_size = lstm_arch.get('input_size', 4)
                
                if hidden_size:
                    # Berechne Flash/RAM Estimation
                    estimation = self.calculate_arduino_flash_ram_estimation(hidden_size, input_size)
                    ram_analysis = self.calculate_arduino_ram_usage(hidden_size, input_size)
                    
                    architectures.append(arch_name.replace('Stateful_', ''))
                    memory_data.append({
                        'hidden_size': hidden_size,
                        'total_params': estimation['total_parameters'],
                        'flash_est': estimation['flash_estimated_kb'],
                        'ram_est': estimation['ram_estimated_kb'],
                        'flash_components': estimation['flash_components'],
                        'ram_components': estimation['ram_components'],
                        'ram_measured': ram_analysis['ram_measured_kb'] or 0,
                        'flash_measured': estimation['validation'].get('flash_measured_kb', 0)
                    })
        
        if not memory_data:
            print("❌ Keine Memory-Daten verfügbar")
            return
        
        # Erstelle detaillierte Memory-Breakdown Plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # ========== 1. Flash-Komponenten Aufschlüsselung ==========
        # Beispiel: 32×32 Architektur für detaillierte Aufschlüsselung
        example_data = memory_data[1] if len(memory_data) > 1 else memory_data[0]  # 32×32
        flash_comp = example_data['flash_components']
          # Flash Pie Chart
        flash_labels = []
        flash_values = []
        flash_colors = []
        
        for key, value in flash_comp.items():
            if key == 'model_weights_kb':
                flash_labels.append(f'Modell-Gewichte\n{value:.1f} kB')
                flash_colors.append('#FF6B6B')
            elif key == 'arduino_core_framework':
                flash_labels.append(f'Arduino Core\n{value:.1f} kB')
                flash_colors.append('#4ECDC4')
            elif key == 'lstm_algorithm_code':
                flash_labels.append(f'LSTM Code\n{value:.1f} kB')
                flash_colors.append('#45B7D1')
            elif key == 'mlp_algorithm_code':
                flash_labels.append(f'MLP Code\n{value:.1f} kB')
                flash_colors.append('#96CEB4')
            elif key == 'math_functions':
                flash_labels.append(f'Math Libs\n{value:.1f} kB')
                flash_colors.append('#FFA07A')
            else:
                flash_labels.append(f'{key.replace("_", " ").title()}\n{value:.1f} kB')
                flash_colors.append('#D3D3D3')
            flash_values.append(value)
        
        wedges1, texts1, autotexts1 = ax1.pie(flash_values, labels=flash_labels, autopct='%1.1f%%',
                                              colors=flash_colors, startangle=90)
        ax1.set_title(f'Flash Memory Breakdown\n{example_data["hidden_size"]}×{example_data["hidden_size"]} Architektur\n'
                     f'Gesamt: {example_data["flash_est"]:.1f} kB', fontsize=14, fontweight='bold')
        
        # Füge Erklärung hinzu
        explanation_flash = (
            "FLASH-SPEICHER KOMPONENTEN:\n"
            "• Modell-Gewichte: Parameter × 4 bytes (const float arrays)\n"
            "• Arduino Code: Framework + Bibliotheken (~78 kB konstant)\n"
            "• Alle Gewichte bleiben permanent im Flash!"
        )
        ax1.text(1.3, 0.5, explanation_flash, transform=ax1.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8),
                verticalalignment='center')
        
        # ========== 2. RAM-Komponenten Aufschlüsselung ==========
        ram_comp = example_data['ram_components']
        
        # RAM Pie Chart
        ram_labels = []
        ram_values = []
        ram_colors = []
        
        for key, value in ram_comp.items():
            if 'lstm' in key.lower():
                if 'hidden' in key or 'cell' in key:
                    ram_labels.append(f'LSTM States\n{value:.2f} kB')
                    ram_colors.append('#FF9999')
                else:
                    ram_labels.append(f'LSTM Buffers\n{value:.2f} kB')
                    ram_colors.append('#FFB366')
            elif key == 'arduino_framework':
                ram_labels.append(f'Arduino Runtime\n{value:.1f} kB')
                ram_colors.append('#66B2FF')
            elif key == 'stack_space':
                ram_labels.append(f'Stack Space\n{value:.1f} kB')
                ram_colors.append('#99FF99')
            else:
                ram_labels.append(f'{key.replace("_", " ").title()}\n{value:.2f} kB')
                ram_colors.append('#E6E6FA')
            ram_values.append(value)
        
        wedges2, texts2, autotexts2 = ax2.pie(ram_values, labels=ram_labels, autopct='%1.1f%%',
                                              colors=ram_colors, startangle=90)
        ax2.set_title(f'RAM Memory Breakdown\n{example_data["hidden_size"]}×{example_data["hidden_size"]} Architektur\n'
                     f'Gesamt: {example_data["ram_est"]:.1f} kB (Gemessen: {example_data["ram_measured"]:.1f} kB)', 
                     fontsize=14, fontweight='bold')
        
        # Füge Erklärung hinzu
        explanation_ram = (
            "RAM-SPEICHER KOMPONENTEN:\n"
            "• LSTM States: Hidden + Cell State (Runtime)\n"
            "• Arduino System: Framework + Stack (~6.8 kB konstant)\n"
            "• Temp Buffers: Berechnungs-Zwischenspeicher\n"
            "• KEINE Gewichte in RAM!"
        )
        ax2.text(1.3, 0.5, explanation_ram, transform=ax2.transAxes, fontsize=10,
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgreen", alpha=0.8),
                verticalalignment='center')
        
        # ========== 3. Skalierung: Flash vs. RAM mit Hidden Size ==========
        hidden_sizes = [data['hidden_size'] for data in memory_data]
        flash_sizes = [data['flash_est'] for data in memory_data]
        ram_sizes = [data['ram_est'] for data in memory_data]
        ram_measured_sizes = [data['ram_measured'] for data in memory_data]
        
        # Erweiterte Skalierung für Vorhersagen
        extended_hidden = list(range(8, 129, 8))
        extended_flash = []
        extended_ram = []
        
        for h in extended_hidden:
            est = self.calculate_arduino_flash_ram_estimation(h, 4)
            extended_flash.append(est['flash_estimated_kb'])
            extended_ram.append(est['ram_estimated_kb'])
        
        # Plot Skalierung
        ax3.plot(extended_hidden, extended_flash, 'r-', linewidth=3, label='Flash (Geschätzt)', alpha=0.8)
        ax3.plot(extended_hidden, extended_ram, 'b-', linewidth=3, label='RAM (Geschätzt)', alpha=0.8)
        
        # Tatsächliche Datenpunkte
        ax3.scatter(hidden_sizes, flash_sizes, c='red', s=150, marker='o', 
                   label='Flash (Architektur)', alpha=0.9, edgecolors='darkred', linewidth=2)
        ax3.scatter(hidden_sizes, ram_sizes, c='blue', s=150, marker='s', 
                   label='RAM (Geschätzt)', alpha=0.9, edgecolors='darkblue', linewidth=2)
        ax3.scatter(hidden_sizes, ram_measured_sizes, c='green', s=150, marker='^', 
                   label='RAM (Gemessen)', alpha=0.9, edgecolors='darkgreen', linewidth=2)
        
        # Arduino Limits
        ax3.axhline(y=256, color='red', linestyle='--', alpha=0.7, linewidth=2,
                   label='Arduino R4 Flash Limit')
        ax3.axhline(y=32, color='blue', linestyle='--', alpha=0.7, linewidth=2,
                   label='Arduino R4 RAM Limit')
        
        # Annotationen für unsere Architekturen
        for i, (h, f, r, rm, arch) in enumerate(zip(hidden_sizes, flash_sizes, ram_sizes, ram_measured_sizes, architectures)):
            ax3.annotate(f'{arch}\n{h}×{h}', (h, f), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax3.set_xlabel('Hidden Size', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Speicher (kB)', fontsize=12, fontweight='bold')
        ax3.set_title('Speicher-Skalierung mit Hidden Size\nFlash wächst quadratisch, RAM linear', 
                     fontsize=14, fontweight='bold')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # ========== 4. Berechnung-Formeln Erklärung ==========
        ax4.axis('off')
        
        formula_text = """
ARDUINO MEMORY BERECHNUNG - FORMELN & METHODIK

🔥 FLASH-BERECHNUNG (wo Gewichte landen):
Flash = Arduino Base + Modell-Gewichte
      = 78 kB + (Parameter × 4 bytes)
      
Komponenten:
• Arduino Code Base: ~65 kB (konstant)
• Bibliotheken: ~8 kB (Serial, Math, etc.)
• System Overhead: ~5 kB (Vectors, Boot)
• Modell-Gewichte: Parameter × 4 bytes (const float arrays)

Beispiel 32×32: 78 kB + (7009 × 4)/1024 = 78 + 27.4 = 105.4 kB

🧠 RAM-BERECHNUNG (nur Runtime):
RAM = Arduino System + LSTM States + Temp Buffers
    = 6.8 kB + Hidden Size abhängige Komponenten

Konstante Komponenten (~6.8 kB):
• Arduino Framework: 2.5 kB
• System Reserved: 1.5 kB  
• Stack Space: 2.0 kB
• Serial Buffers: 0.5 kB
• Heap Management: 0.3 kB

Variable Komponenten (Hidden Size abhängig):
• Hidden State: hidden_size × 4 bytes
• Cell State: hidden_size × 4 bytes
• Temp Buffers: hidden_size × 8 bytes (Gate-Berechnungen)

Beispiel 32×32: 6.8 kB + (32×4 + 32×4 + 32×8)/1024 = 6.8 + 0.5 = 7.3 kB

✅ VALIDIERUNG:
• 32×32 Gemessen: 93 kB Flash, 8.4 kB RAM
• 32×32 Berechnet: 105 kB Flash, 7.3 kB RAM
• Flash Fehler: ~14% (konservativ geschätzt)
• RAM Fehler: ~13% (sehr gut!)

🔑 SCHLÜSSEL-ERKENNTNISSE:
• Gewichte bleiben IMMER im Flash (const arrays)
• RAM enthält NUR Runtime-Daten
• Flash wächst mit Parameter² (quadratisch)
• RAM wächst mit Hidden Size (linear)
• Arduino System-Overhead ist konstant
        """
        
        ax4.text(0.05, 0.95, formula_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.8", facecolor="lightyellow", alpha=0.9))
        
        plt.tight_layout()
          # Speichere Plot
        plot_filename = "detailed_memory_breakdown_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n🔍 Detaillierte Memory-Breakdown Analyse gespeichert: {plot_filename}")
        
        return fig
      def create_arduino_memory_estimation_visualization(self):
        """Erstellt umfassende Arduino Memory Estimation Visualisierung mit realen Messwerten"""
        
        if not self.results:
            print("❌ Keine Daten für Arduino Memory Estimation")
            return
        
        # Sammle sowohl Estimations als auch reale Messwerte
        architectures = []
        estimation_data = []
        hidden_sizes = []
        
        # Reale Messwerte (aus Arduino IDE)
        real_measurements = {
            16: {
                'flash_used_kb': 48.8,
                'ram_used_kb': 7.7,
                'parameters': 1969,
                'flash_percent': 18.6,
                'ram_percent': 24.1,
            },
            32: {
                'flash_used_kb': 106.9,
                'ram_used_kb': 8.9,
                'parameters': 7009,
                'flash_percent': 40.8,
                'ram_percent': 27.8,
            },
            64: {
                'flash_used_kb': 123.0,
                'ram_used_kb': 9.8,
                'parameters': 26305,
                'flash_percent': 46.9,
                'ram_percent': 30.6,
            }
        }
        
        for arch_name, result in self.results.items():
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis and 'lstm_architecture' in model_analysis:
                lstm_arch = model_analysis['lstm_architecture']
                hidden_size = lstm_arch.get('hidden_size')
                input_size = lstm_arch.get('input_size', 1)
                
                if hidden_size:
                    # Berechne Arduino Memory Estimation
                    estimation = self.calculate_arduino_flash_ram_estimation(hidden_size, input_size)
                    
                    architectures.append(arch_name.replace('Stateful_', ''))
                    estimation_data.append(estimation)
                    hidden_sizes.append(hidden_size)
        
        if not estimation_data:
            print("❌ Keine Arduino Memory Estimation verfügbar")
            return
        
        # Erstelle erweiterte Plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # ========== 1. Flash vs. RAM: Estimation vs. Reale Messwerte ==========
        flash_est = [data['flash_estimated_kb'] for data in estimation_data]
        ram_est = [data['ram_estimated_kb'] for data in estimation_data]
        
        # Reale Messwerte sammeln
        flash_real = []
        ram_real = []
        for hs in hidden_sizes:
            if hs in real_measurements:
                flash_real.append(real_measurements[hs]['flash_used_kb'])
                ram_real.append(real_measurements[hs]['ram_used_kb'])
            else:
                flash_real.append(None)
                ram_real.append(None)
        
        x = np.arange(len(architectures))
        width = 0.2
        
        # Flash Estimation vs. Real
        bars1 = ax1.bar(x - width*1.5, flash_est, width, label='Flash Estimation', 
                       alpha=0.8, color='lightcoral', edgecolor='darkred')
        bars2 = ax1.bar(x - width*0.5, [f for f in flash_real if f is not None], width, 
                       label='Flash Real', alpha=0.8, color='red', edgecolor='darkred')
        
        # RAM Estimation vs. Real  
        bars3 = ax1.bar(x + width*0.5, ram_est, width, label='RAM Estimation', 
                       alpha=0.8, color='lightblue', edgecolor='darkblue')
        bars4 = ax1.bar(x + width*1.5, [r for r in ram_real if r is not None], width,
                       label='RAM Real', alpha=0.8, color='blue', edgecolor='darkblue')
        
        # Arduino Limits
        ax1.axhline(y=256, color='red', linestyle='--', alpha=0.7, 
                   label='Arduino R4 Flash Limit (256 kB)')
        ax1.axhline(y=32, color='blue', linestyle='--', alpha=0.7,
                   label='Arduino R4 RAM Limit (32 kB)')
        
        ax1.set_xlabel('Architektur')
        ax1.set_ylabel('Speicher (kB)')
        ax1.set_title('Arduino Flash/RAM Abschätzung vs. Reale Messwerte')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{arch}\n({hs}×{hs})' for arch, hs in zip(architectures, hidden_sizes)])
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Werte auf Balken
        for i, (f_est, f_real, r_est, r_real) in enumerate(zip(flash_est, flash_real, ram_est, ram_real)):
            ax1.text(i - width*1.5, f_est + 5, f'{f_est:.1f}', ha='center', va='bottom', fontsize=8)
            if f_real is not None:
                ax1.text(i - width*0.5, f_real + 5, f'{f_real:.1f}', ha='center', va='bottom', fontsize=8)
            ax1.text(i + width*0.5, r_est + 1, f'{r_est:.1f}', ha='center', va='bottom', fontsize=8)
            if r_real is not None:
                ax1.text(i + width*1.5, r_real + 1, f'{r_real:.1f}', ha='center', va='bottom', fontsize=8)
        
        # ========== 2. Logarithmische Skalierung über Hidden Size ==========
        all_hidden_sizes = sorted(list(set(hidden_sizes + list(real_measurements.keys()))))
        
        # Berechne Estimations für alle Größen
        log_flash_est = []
        log_ram_est = []
        log_flash_real = []
        log_ram_real = []
        
        for hs in all_hidden_sizes:
            estimation = self.calculate_arduino_flash_ram_estimation(hs, 1)
            log_flash_est.append(estimation['flash_estimated_kb'])
            log_ram_est.append(estimation['ram_estimated_kb'])
            
            if hs in real_measurements:
                log_flash_real.append(real_measurements[hs]['flash_used_kb'])
                log_ram_real.append(real_measurements[hs]['ram_used_kb'])
            else:
                log_flash_real.append(None)
                log_ram_real.append(None)
        
        # Logarithmische Kurven
        ax2.plot(all_hidden_sizes, log_flash_est, 'o-', color='red', alpha=0.7, 
                label='Flash Estimation', linewidth=2, markersize=8)
        ax2.plot(all_hidden_sizes, log_ram_est, 'o-', color='blue', alpha=0.7,
                label='RAM Estimation', linewidth=2, markersize=8)
        
        # Reale Messwerte als Punkte
        real_hs = [hs for hs in all_hidden_sizes if hs in real_measurements]
        real_flash_vals = [real_measurements[hs]['flash_used_kb'] for hs in real_hs]
        real_ram_vals = [real_measurements[hs]['ram_used_kb'] for hs in real_hs]
        
        ax2.scatter(real_hs, real_flash_vals, color='darkred', s=100, marker='s', 
                   label='Flash Real', zorder=5, edgecolors='black')
        ax2.scatter(real_hs, real_ram_vals, color='darkblue', s=100, marker='s',
                   label='RAM Real', zorder=5, edgecolors='black')
        
        ax2.set_xscale('log', base=2)
        ax2.set_yscale('log')
        ax2.axhline(y=256, color='red', linestyle='--', alpha=0.7, label='Flash Limit')
        ax2.axhline(y=32, color='blue', linestyle='--', alpha=0.7, label='RAM Limit')
        ax2.set_xlabel('Hidden Size (Log Scale)')
        ax2.set_ylabel('Speicher (kB, Log Scale)')
        ax2.set_title('Logarithmische Skalierung: Flash/RAM vs. Hidden Size')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ========== 3. Flash-Komponenten Tortendiagramm ==========
        if estimation_data:
            # Verwende die erste Estimation für Beispiel-Breakdown
            sample_estimation = estimation_data[0]
            flash_components = sample_estimation['flash_components']
            
            sizes = list(flash_components.values())
            labels = [label.replace('_', ' ').title() for label in flash_components.keys()]
            colors = plt.cm.Reds(np.linspace(0.3, 0.8, len(sizes)))
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             colors=colors, startangle=90)
            ax3.set_title(f'Flash-Komponenten Breakdown\n({architectures[0]} Architektur)')
            
            # Legende mit absoluten Werten
            legend_labels = [f'{label}: {size:.1f} kB' for label, size in zip(labels, sizes)]
            ax3.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        # ========== 4. RAM-Komponenten Tortendiagramm ==========
        if estimation_data:
            ram_components = sample_estimation['ram_components']
            
            sizes = list(ram_components.values())
            labels = [label.replace('_', ' ').title() for label in ram_components.keys()]
            colors = plt.cm.Blues(np.linspace(0.3, 0.8, len(sizes)))
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
            ax4.set_title(f'RAM-Komponenten Breakdown\n({architectures[0]} Architektur)')
            
            # Legende mit absoluten Werten
            legend_labels = [f'{label}: {size:.1f} kB' for label, size in zip(labels, sizes)]
            ax4.legend(wedges, legend_labels, loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
        
        plt.tight_layout()
        
        # Speichere Plot
        plot_filename = "arduino_memory_estimation_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n🔋 Arduino Memory Estimation Analyse gespeichert: {plot_filename}")
        
        # Drucke Zusammenfassung
        self.print_arduino_memory_summary(architectures, estimation_data, hidden_sizes, real_measurements)
        
        return fig
        ax1.set_xticklabels(architectures, rotation=0)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Werte auf Balken annotieren
        for i, (flash, ram) in enumerate(zip(flash_est, ram_est)):
            ax1.text(bars1[i].get_x() + bars1[i].get_width()/2., flash + 2,
                    f'{flash:.0f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
            ax1.text(bars2[i].get_x() + bars2[i].get_width()/2., ram + 0.3,
                    f'{ram:.1f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
        
        # ========== 2. Validierung mit realen Messungen ==========
        # Sammle Validierungsdaten
        has_validation = []
       

        flash_errors = []
        ram_errors = []
        
        for data in estimation_data:
            validation = data['validation']
            if validation['has_real_data']:
                has_validation.append(True)
                flash_errors.append(validation['flash_error_percent'])
                ram_errors.append(validation['ram_error_percent'])
            else:
                has_validation.append(False)
                flash_errors.append(0)
                ram_errors.append(0)
        
        # Plot nur Architekturen mit Validierungsdaten
        valid_archs = [arch for i, arch in enumerate(architectures) if has_validation[i]]
        valid_flash_errors = [err for i, err in enumerate(flash_errors) if has_validation[i]]
        valid_ram_errors = [err for i, err in enumerate(ram_errors) if has_validation[i]]
        
        if valid_archs:
            x_valid = np.arange(len(valid_archs))
            bars1 = ax2.bar(x_valid - width/2, valid_flash_errors, width, 
                           label='Flash Fehler (%)', alpha=0.8, color='orange')
            bars2 = ax2.bar(x_valid + width/2, valid_ram_errors, width, 
                           label='RAM Fehler (%)', alpha=0.8, color='lightgreen')
            
            ax2.set_xlabel('Architektur (mit Validierung)')
            ax2.set_ylabel('Abweichung (%)')
            ax2.set_title('Vorhersage-Genauigkeit vs. reale Messungen')
            ax2.set_xticks(x_valid)
            ax2.set_xticklabels(valid_archs)
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Keine Validierungsdaten\nverfügbar', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        
        # ========== 3. Arduino R4 Kompatibilität ==========
        compatibility = []
        for data in estimation_data:
            compat = data['arduino_r4_compatibility']
            compatibility.append({
                'flash_percent': compat['flash_available_percent'],
                'ram_percent': compat['ram_available_percent'],
                'fits': compat['overall_fits']
            })
        
        flash_percents = [c['flash_percent'] for c in compatibility]
        ram_percents = [c['ram_percent'] for c in compatibility]
        colors = ['green' if c['fits'] else 'red' for c in compatibility]
        
        # Scatter Plot
        scatter = ax3.scatter(flash_percents, ram_percents, c=colors, s=200, alpha=0.7, edgecolors='black')
        
        # Arduino Limits
        ax3.axvline(x=100, color='red', linestyle='--', alpha=0.7, linewidth=2, label='Flash Limit (100%)')
        ax3.axhline(y=100, color='blue', linestyle='--', alpha=0.7, linewidth=2, label='RAM Limit (100%)')
        
        # Annotationen
        for i, (fp, rp, arch) in enumerate(zip(flash_percents, ram_percents, architectures)):
            ax3.annotate(arch, (fp, rp), xytext=(5, 5), textcoords='offset points',
                        fontsize=9, ha='left')
        
        ax3.set_xlabel('Flash Nutzung (%)')
        ax3.set_ylabel('RAM Nutzung (%)')
        ax3.set_title('Arduino R4 Kompatibilität\nGrün = Passt, Rot = Zu groß')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # ========== 4. Zusammenfassung ==========
        summary_text = "ARDUINO MEMORY ESTIMATION ZUSAMMENFASSUNG\n\n"
        
        for i, (arch, data) in enumerate(zip(architectures, estimation_data)):
            params = data['total_parameters']
            flash_kb = data['flash_estimated_kb']
            ram_kb = data['ram_estimated_kb'] 
            fits = data['arduino_r4_compatibility']['overall_fits']
            status = "✅ Ja" if fits else "❌ Nein"
            
            validation = data['validation']
            if validation['has_real_data']:
                flash_acc = validation['flash_accuracy']
                ram_acc = validation['ram_accuracy']
                accuracy_str = f"{flash_acc[:4]}/{ram_acc[:4]}"
            else:
                accuracy_str = "N/A"
            
            summary_text += f"{arch:<8} | {params//1000:>3}k | {flash_kb:>7.1f}kB | {ram_kb:>6.1f}kB | {accuracy_str:<12} | {status}\n"
        
        summary_text += "\n💡 SCHLÜSSEL-ERKENNTNISSE:\n"
        summary_text += "• Flash = Arduino Code (~78kB) + const float Gewichte\n"
        summary_text += "• RAM = LSTM States + I/O Buffer + Arduino Runtime\n"
        summary_text += "• Gewichte bleiben in Flash, nur States gehen in RAM!\n"
        summary_text += "• Flash ist meist der limitierende Faktor\n"
        
        max_hidden = max([data['key_insights']['max_recommended_hidden_size'] for data in estimation_data])
        summary_text += f"• Max empfohlene Hidden Size für Arduino R4: {max_hidden}"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=10, fontfamily='monospace',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        
        plt.tight_layout()
        
        # Speichere Plot
        plot_filename = "arduino_memory_estimation_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n💾 Arduino Memory Estimation Analyse gespeichert: {plot_filename}")
          # Drucke detaillierte Zusammenfassung
        self.print_arduino_memory_summary(architectures, estimation_data, hidden_sizes, real_measurements)
        
        return fig

    def print_arduino_memory_summary(self, architectures, estimation_data, hidden_sizes, real_measurements):
        """Druckt Arduino Memory Estimation Zusammenfassung mit realen Messwerten"""
        
        print(f"\n{'='*100}")
        print("ARDUINO MEMORY ESTIMATION ZUSAMMENFASSUNG")
        print(f"{'='*100}")
        
        header = f"{'Arch':<8} | {'Param':<6} | {'Flash Est':<10} | {'Flash Real':<11} | {'RAM Est':<8} | {'RAM Real':<9} | {'Flash Err':<10} | {'RAM Err':<9} | {'Passt?'}"
        print(header)
        print("-" * len(header))
        
        for arch, data, hs in zip(architectures, estimation_data, hidden_sizes):
            params = data['total_parameters']
            flash_kb = data['flash_estimated_kb']
            ram_kb = data['ram_estimated_kb'] 
            fits = data['arduino_r4_compatibility']['overall_fits']
            fits_str = "✅ Ja" if fits else "❌ Nein"
            
            # Reale Messwerte
            if hs in real_measurements:
                real_data = real_measurements[hs]
                flash_real = real_data['flash_used_kb']
                ram_real = real_data['ram_used_kb']
                flash_err = abs(flash_kb - flash_real) / flash_real * 100
                ram_err = abs(ram_kb - ram_real) / ram_real * 100
                flash_real_str = f"{flash_real:.1f} kB"
                ram_real_str = f"{ram_real:.1f} kB"
                flash_err_str = f"{flash_err:.1f}%"
                ram_err_str = f"{ram_err:.1f}%"
            else:
                flash_real_str = "N/A"
                ram_real_str = "N/A"
                flash_err_str = "N/A"
                ram_err_str = "N/A"
            
            row = f"{arch:<8} | {params:<6} | {flash_kb:<10.1f} | {flash_real_str:<11} | {ram_kb:<8.1f} | {ram_real_str:<9} | {flash_err_str:<10} | {ram_err_str:<9} | {fits_str}"
            print(row)
        
        print("\n" + "="*100)
        print("SCHLÜSSEL-ERKENNTNISSE:")
        print("• Flash: Modell-Gewichte + Arduino Code (~65-80 kB Base)")
        print("• RAM: Hidden States + Temporary Buffers + Arduino Runtime")
        print("• Modell-Gewichte bleiben im Flash, nur Runtime-Daten gehen ins RAM")
        print("• Arduino Uno R4: 256 kB Flash / 32 kB RAM")
        
        # Empfehlungen
        compatible_archs = [arch for arch, data in zip(architectures, estimation_data) 
                          if data['arduino_r4_compatibility']['overall_fits']]
        if compatible_archs:
            print(f"• ✅ KOMPATIBEL: {', '.join(compatible_archs)}")
        
        incompatible_archs = [arch for arch, data in zip(architectures, estimation_data) 
                            if not data['arduino_r4_compatibility']['overall_fits']]
        if incompatible_archs:
            print(f"• ❌ INKOMPATIBEL: {', '.join(incompatible_archs)}")
        
        print("="*100)
            
            print(f"{arch:<8} | {params//1000:>4}k | {flash_kb:>6.1f}kB | {ram_kb:>6.1f}kB | {measured_str:<16} | {accuracy_str:<10} | {fits_str}")
        
        print(f"\n💡 SCHLÜSSEL-ERKENNTNISSE:")
        print("• Flash = Arduino Code (~78kB) + const float Gewichte")
        print("• RAM = LSTM States + I/O Buffer + Arduino Runtime")
        print("• Gewichte bleiben in Flash, nur States gehen in RAM!")
        print("• Flash ist meist der limitierende Faktor")
        
        # Validierungs-Info
        validated_data = [data for data in estimation_data if data['validation']['has_real_data']]
        if validated_data:
            print(f"\n🔬 VALIDIERUNG:")
            for data in validated_data:
                val = data['validation']
                print(f"• {arch}: Flash ±{val['flash_error_percent']:.1f}%, RAM ±{val['ram_error_percent']:.1f}%")

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
    
    def create_enhanced_memory_estimation_with_uncertainty(self):
        """
        Erstellt erweiterte Arduino Memory Visualisierung mit Uncertainty/Error Bars
        im Stil von arduino_memory_estimation_analysis.png
        
        Features:
        - 4-Subplot Layout: RAM & Flash bars mit error bars, scaling curves, pie charts
        - Uncertainty ranges für robuste Memory Berechnungen
        - Error bar caps (horizontal lines) mit matplotlib capsize parameter
        - Einheitliche Achsen-Skalierung für RAM und Flash
        - Stil aus arduino_memory_estimation_analysis.png
        """
        
        if not self.results:
            print("❌ Keine Daten für erweiterte Memory-Visualisierung")
            return
        
        # ========== DATEN SAMMELN ==========
        architectures = []
        hidden_sizes = []
        
        # RAM Daten mit Uncertainty
        ram_theoretical = []
        ram_measured = []
        ram_error_lower = []  # Untere Grenze
        ram_error_upper = []  # Obere Grenze
        ram_uncertainty = []  # ±Bereich
        
        # Flash Daten mit Uncertainty
        flash_theoretical = []
        flash_error_lower = []
        flash_error_upper = []
        flash_uncertainty = []
        
        # Komponenten für Pie Charts
        ram_components_all = []
        flash_components_all = []
        
        for arch_name, result in self.results.items():
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis and 'lstm_architecture' in model_analysis:
                lstm_arch = model_analysis['lstm_architecture']
                hidden_size = lstm_arch.get('hidden_size')
                input_size = lstm_arch.get('input_size', 1)
                
                if hidden_size:
                    # RAM Analyse mit Uncertainty
                    ram_analysis = self.calculate_arduino_ram_usage(hidden_size, input_size)
                    
                    # Flash Analyse mit erweiterten Uncertainty Berechnungen
                    flash_analysis = self.calculate_arduino_flash_ram_estimation(hidden_size, input_size)
                    
                    architectures.append(arch_name.replace('Stateful_', ''))
                    hidden_sizes.append(hidden_size)
                    
                    # ========== RAM UNCERTAINTY BERECHNUNG ==========
                    ram_base = ram_analysis['ram_theoretical_kb']
                    
                    # Uncertainty-Faktoren für RAM-Komponenten:
                    # - System components: ±10% (relativ stabil)
                    # - LSTM states: ±15% (variiert mit Optimierungen)
                    # - Temporary buffers: ±20% (stark implementierungsabhängig)
                    system_uncertainty = 0.10
                    lstm_uncertainty = 0.15
                    buffer_uncertainty = 0.20
                    
                    # Gewichtete Uncertainty basierend auf Komponenten
                    breakdown = ram_analysis['ram_breakdown']
                    system_weight = breakdown['total_constant_kb'] / ram_base if ram_base > 0 else 0
                    variable_weight = breakdown['total_variable_kb'] / ram_base if ram_base > 0 else 0
                    
                    # Kombinierte Uncertainty (weighted average)
                    ram_uncertainty_factor = (system_weight * system_uncertainty + 
                                            variable_weight * lstm_uncertainty + 
                                            0.3 * buffer_uncertainty)  # 30% buffer contribution
                    
                    ram_uncertainty_range = ram_base * ram_uncertainty_factor
                    
                    ram_theoretical.append(ram_base)
                    ram_measured.append(ram_analysis['ram_measured_kb'] or ram_base)
                    ram_error_lower.append(ram_uncertainty_range)
                    ram_error_upper.append(ram_uncertainty_range)
                    ram_uncertainty.append(ram_uncertainty_range)
                    
                    # ========== FLASH UNCERTAINTY BERECHNUNG ==========
                    flash_base = flash_analysis['estimated_flash_kb']
                    
                    # Flash Uncertainty-Faktoren:
                    # - Model weights: ±5% (sehr stabil, nur float32 precision)
                    # - Code size: ±12% (Compiler-abhängig)
                    # - Constants: ±8% (relativ stabil)
                    weights_uncertainty = 0.05
                    code_uncertainty = 0.12
                    constants_uncertainty = 0.08
                    
                    # Gewichtete Flash Uncertainty
                    flash_weights = ram_analysis['flash_model_weights_kb']
                    code_size = flash_base - flash_weights - 2.0  # Estimate code portion
                    constants_size = 2.0  # Typical constants
                    
                    weights_weight = flash_weights / flash_base if flash_base > 0 else 0
                    code_weight = code_size / flash_base if flash_base > 0 else 0
                    constants_weight = constants_size / flash_base if flash_base > 0 else 0
                    
                    flash_uncertainty_factor = (weights_weight * weights_uncertainty + 
                                              code_weight * code_uncertainty + 
                                              constants_weight * constants_uncertainty)
                    
                    flash_uncertainty_range = flash_base * flash_uncertainty_factor
                    
                    flash_theoretical.append(flash_base)
                    flash_error_lower.append(flash_uncertainty_range)
                    flash_error_upper.append(flash_uncertainty_range)
                    flash_uncertainty.append(flash_uncertainty_range)
                    
                    # ========== KOMPONENTEN FÜR PIE CHARTS ==========
                    # RAM Komponenten (detailliert)
                    ram_comp = breakdown['constant_overheads'].copy()
                    ram_comp.update(breakdown['variable_components'])
                    ram_components_all.append(ram_comp)
                    
                    # Flash Komponenten
                    flash_comp = {
                        'LSTM Weights': flash_weights * 0.7,  # Hauptteil
                        'MLP Weights': flash_weights * 0.3,   # Kleinerer Teil
                        'Arduino Code': code_size,
                        'Constants': constants_size
                    }
                    flash_components_all.append(flash_comp)
        
        if not architectures:
            print("❌ Keine Daten für erweiterte Visualisierung verfügbar")
            return
        
        # ========== 4-SUBPLOT LAYOUT ERSTELLEN ==========
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        fig.suptitle('Enhanced Arduino Memory Analysis with Uncertainty Quantification\n' +
                    'Style: arduino_memory_estimation_analysis.png', fontsize=16, fontweight='bold')
        
        # ========== 1. RAM & FLASH BARS MIT ERROR BARS (Links oben) ==========
        x = np.arange(len(architectures))
        width = 0.35
        
        # Unified scale for both RAM and Flash (wichtig für Vergleichbarkeit)
        max_ram = max(ram_theoretical) + max(ram_uncertainty) if ram_theoretical else 10
        max_flash = max(flash_theoretical) + max(flash_uncertainty) if flash_theoretical else 50
        unified_max = max(max_ram, max_flash) * 1.1  # 10% margin
        
        # RAM Bars mit Error Bars (capsize für horizontale Linien!)
        bars_ram = ax1.bar(x - width/2, ram_theoretical, width, 
                          label='RAM (Theoretical)', 
                          alpha=0.8, color='#4A90E2', edgecolor='#2E5F8A')
        
        # RAM Error Bars mit capsize parameter
        ax1.errorbar(x - width/2, ram_theoretical, 
                    yerr=[ram_error_lower, ram_error_upper],
                    fmt='none', color='#2E5F8A', capsize=8, capthick=2, linewidth=2)
        
        # Flash Bars mit Error Bars
        bars_flash = ax1.bar(x + width/2, flash_theoretical, width,
                           label='Flash (Theoretical)',
                           alpha=0.8, color='#E74C3C', edgecolor='#A93226')
        
        # Flash Error Bars mit capsize parameter
        ax1.errorbar(x + width/2, flash_theoretical,
                    yerr=[flash_error_lower, flash_error_upper],
                    fmt='none', color='#A93226', capsize=8, capthick=2, linewidth=2)
        
        # Styling
        ax1.set_xlabel('LSTM Architecture', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Memory Usage (kB)', fontsize=12, fontweight='bold')
        ax1.set_title('Memory Usage with Uncertainty Ranges\n(Error bars show ±uncertainty)', 
                     fontsize=14, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{arch}\n({hs}×{hs})' for arch, hs in zip(architectures, hidden_sizes)])
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.set_ylim(0, unified_max)
        
        # Werte auf Balken annotieren
        for i, (ram_val, flash_val, ram_unc, flash_unc) in enumerate(zip(ram_theoretical, flash_theoretical, ram_uncertainty, flash_uncertainty)):
            # RAM annotation
            ax1.text(bars_ram[i].get_x() + bars_ram[i].get_width()/2., ram_val + ram_unc + 1,
                    f'{ram_val:.1f}±{ram_unc:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9, color='#2E5F8A')
            
            # Flash annotation
            ax1.text(bars_flash[i].get_x() + bars_flash[i].get_width()/2., flash_val + flash_unc + 1,
                    f'{flash_val:.1f}±{flash_unc:.1f}', ha='center', va='bottom', 
                    fontweight='bold', fontsize=9, color='#A93226')
        
        # ========== 2. SCALING CURVES MIT UNCERTAINTY BANDS (Rechts oben) ==========
        # Erstelle Skalierungskurven für verschiedene Hidden Sizes
        hidden_range = np.arange(8, 129, 8)  # 8, 16, 24, ..., 128
        
        ram_scaling = []
        ram_scaling_lower = []
        ram_scaling_upper = []
        flash_scaling = []
        flash_scaling_lower = []
        flash_scaling_upper = []
        
        for h_size in hidden_range:
            # RAM scaling
            ram_analysis_scale = self.calculate_arduino_ram_usage(h_size, 1)
            ram_base_scale = ram_analysis_scale['ram_theoretical_kb']
            ram_uncertainty_scale = ram_base_scale * 0.15  # 15% uncertainty
            
            ram_scaling.append(ram_base_scale)
            ram_scaling_lower.append(ram_base_scale - ram_uncertainty_scale)
            ram_scaling_upper.append(ram_base_scale + ram_uncertainty_scale)
            
            # Flash scaling
            flash_analysis_scale = self.calculate_arduino_flash_ram_estimation(h_size, 1)
            flash_base_scale = flash_analysis_scale['estimated_flash_kb']
            flash_uncertainty_scale = flash_base_scale * 0.10  # 10% uncertainty
            
            flash_scaling.append(flash_base_scale)
            flash_scaling_lower.append(flash_base_scale - flash_uncertainty_scale)
            flash_scaling_upper.append(flash_base_scale + flash_uncertainty_scale)
        
        # Plot scaling curves with uncertainty bands
        ax2.plot(hidden_range, ram_scaling, 'o-', color='#4A90E2', linewidth=3, 
                label='RAM Scaling', markersize=6)
        ax2.fill_between(hidden_range, ram_scaling_lower, ram_scaling_upper, 
                        color='#4A90E2', alpha=0.3, label='RAM Uncertainty Band')
        
        ax2.plot(hidden_range, flash_scaling, 's-', color='#E74C3C', linewidth=3,
                label='Flash Scaling', markersize=6)
        ax2.fill_between(hidden_range, flash_scaling_lower, flash_scaling_upper,
                        color='#E74C3C', alpha=0.3, label='Flash Uncertainty Band')
        
        # Mark current architectures
        for arch, h_size, ram_val, flash_val in zip(architectures, hidden_sizes, ram_theoretical, flash_theoretical):
            ax2.plot(h_size, ram_val, 'o', color='#2E5F8A', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax2.plot(h_size, flash_val, 's', color='#A93226', markersize=10, markeredgecolor='white', markeredgewidth=2)
            ax2.annotate(arch, (h_size, max(ram_val, flash_val) + 5), 
                        ha='center', va='bottom', fontsize=8, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        # Arduino limits
        ax2.axhline(y=32, color='red', linestyle='--', alpha=0.8, linewidth=2, label='Arduino R4 RAM Limit')
        ax2.axhline(y=256, color='orange', linestyle='--', alpha=0.8, linewidth=2, label='Arduino R4 Flash Limit')
        
        ax2.set_xlabel('Hidden Size', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Memory Usage (kB)', fontsize=12, fontweight='bold')
        ax2.set_title('Memory Scaling with Uncertainty Bands\n(Shaded areas show uncertainty ranges)', 
                     fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_xlim(0, 130)
        
        # ========== 3. RAM COMPONENTS PIE CHART (Links unten) ==========
        if ram_components_all:
            # Verwende mittlere Architektur als Beispiel
            mid_idx = len(ram_components_all) // 2
            ram_components = ram_components_all[mid_idx]
            example_arch = architectures[mid_idx]
            
            # Gruppiere ähnliche Komponenten
            grouped_ram = {
                'Arduino Core': ram_components.get('arduino_core', 0) + 
                              ram_components.get('system_reserved', 0) +
                              ram_components.get('firmware_overhead', 0),
                'Stack & Heap': ram_components.get('stack_space', 0) + 
                               ram_components.get('heap_management', 0),
                'I/O Buffers': ram_components.get('serial_buffers', 0) + 
                              ram_components.get('input_buffer', 0),
                'LSTM States': ram_components.get('lstm_hidden_state', 0) + 
                              ram_components.get('lstm_cell_state', 0),
                'Processing': ram_components.get('lstm_gate_temps', 0) + 
                             ram_components.get('activation_temps', 0) +
                             ram_components.get('mlp_layer1_buffer', 0) +
                             ram_components.get('mlp_layer2_buffer', 0) +
                             ram_components.get('mlp_output_buffer', 0) + \
                             ram_components.get('processing_buffer', 0)
            }
            
            # Filtere kleine Komponenten heraus
            grouped_ram = {k: v for k, v in grouped_ram.items() if v > 0.1}
            
            labels_ram = list(grouped_ram.keys())
            sizes_ram = list(grouped_ram.values())
            colors_ram = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57']
            
            wedges, texts, autotexts = ax3.pie(sizes_ram, labels=labels_ram, colors=colors_ram,
                                             autopct=lambda pct: f'{pct:.1f}%\n({pct/100*sum(sizes_ram):.1f}kB)',
                                             startangle=90, textprops={'fontsize': 10})
            
            ax3.set_title(f'RAM Components Breakdown\n{example_arch} Architecture ({hidden_sizes[mid_idx]}×{hidden_sizes[mid_idx]})',
                         fontsize=14, fontweight='bold')
        
        # ========== 4. FLASH COMPONENTS PIE CHART (Rechts unten) ==========
        if flash_components_all:
            flash_components = flash_components_all[mid_idx]
            
            labels_flash = list(flash_components.keys())
            sizes_flash = list(flash_components.values())
            colors_flash = ['#E74C3C', '#F39C12', '#8E44AD', '#27AE60']
            
            wedges, texts, autotexts = ax4.pie(sizes_flash, labels=labels_flash, colors=colors_flash,
                                             autopct=lambda pct: f'{pct:.1f}%\n({pct/100*sum(sizes_flash):.1f}kB)',
                                             startangle=90, textprops={'fontsize': 10})
            
            ax4.set_title(f'Flash Components Breakdown\n{example_arch} Architecture ({hidden_sizes[mid_idx]}×{hidden_sizes[mid_idx]})',
                         fontsize=14, fontweight='bold')
        
        # ========== FINAL STYLING ==========
        plt.tight_layout()
          # Speichere enhanced Memory-Analyse Plot
        plot_filename = "arduino_memory_estimation_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"\n🎯 Enhanced Arduino Memory Analysis mit Uncertainty gespeichert: {plot_filename}")
        print("✅ Features implementiert:")
        print("   📊 4-Subplot Layout: RAM & Flash bars + scaling curves + pie charts")
        print("   📏 Error bars mit capsize parameter (horizontale Linien)")
        print("   🎨 Einheitliche Achsen-Skalierung für RAM und Flash")
        print("   🔬 Uncertainty quantification mit gewichteten Faktoren")
        print("   🎭 Stil basierend auf arduino_memory_estimation_analysis.png")
        
        return fig

    def run_complete_analysis(self):
        """
        Führt die komplette Modellanalyse durch und erstellt alle Visualisierungen
        """
        print(f"\n{'='*80}")
        print("STARTING COMPLETE MODEL ANALYSIS")
        print(f"{'='*80}")
        
        # 1. Finde alle Model-Verzeichnisse
        model_dirs = self.find_model_directories()
        print(f"📁 Found {len(model_dirs)} model architectures")
        
        # 2. Analysiere jede Architektur
        results = []
        for arch_info in model_dirs:
            try:
                result = self.analyze_architecture_directory(arch_info)
                results.append(result)
                self.results[arch_info['architecture']] = result
            except Exception as e:
                print(f"❌ Error analyzing {arch_info['architecture']}: {e}")
                continue
        
        # 3. Erstelle RAM-Analyse-Visualisierung
        print(f"\n📊 Creating RAM analysis visualization...")
        try:
            self.create_ram_analysis_visualization()
            print("✅ RAM analysis visualization created")
        except Exception as e:
            print(f"❌ Error creating RAM visualization: {e}")
        
        # 4. Erstelle umfassende Visualisierung
        print(f"\n📊 Creating comprehensive visualization...")
        try:
            self.create_comprehensive_visualization()
            print("✅ Comprehensive visualization created")
        except Exception as e:
            print(f"❌ Error creating comprehensive visualization: {e}")
        
        # 5. Erstelle detaillierte Memory-Breakdown-Visualisierung
        print(f"\n📊 Creating detailed memory breakdown visualization...")
        try:
            self.create_detailed_memory_breakdown_visualization()
            print("✅ Detailed memory breakdown visualization created")
        except Exception as e:
            print(f"❌ Error creating detailed memory visualization: {e}")
        
        # 6. Erstelle Arduino Memory Estimation Visualisierung
        print(f"\n📊 Creating Arduino memory estimation visualization...")
        try:
            self.create_arduino_memory_estimation_visualization()
            print("✅ Arduino memory estimation visualization created")
        except Exception as e:
            print(f"❌ Error creating Arduino memory visualization: {e}")
        
        # 7. Erstelle Enhanced Memory Estimation mit Uncertainty (DIE NEUE FUNKTION!)
        print(f"\n📊 Creating enhanced memory estimation with uncertainty...")
        try:
            self.create_enhanced_memory_estimation_with_uncertainty()
            print("✅ Enhanced memory estimation with uncertainty created")
            print("🎯 Enhanced visualization saved as 'arduino_memory_estimation_analysis.png'")
        except Exception as e:
            print(f"❌ Error creating enhanced memory visualization: {e}")
        
        print(f"\n{'='*80}")
        print("COMPLETE ANALYSIS FINISHED")
        print(f"{'='*80}")
        print(f"📊 Total architectures analyzed: {len(results)}")
        print(f"📈 Visualizations created:")
        print(f"   - RAM analysis")
        print(f"   - Comprehensive analysis")  
        print(f"   - Detailed memory breakdown")
        print(f"   - Arduino memory estimation")
        print(f"   - Enhanced memory estimation with uncertainty ⭐")
        
        return results

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
