# Comprehensive Model Analysis - FIXED VERSION
# Analysiert PyTorch Modelle, Trainings-Skripte und Arduino Weights systematisch

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
        """Berechnet die kompletten Modell-Parameter (LSTM + MLP Head)"""
        if hidden_size <= 0:
            return {'total': 0, 'lstm': 0, 'mlp': 0}
            
        if mlp_layers is None:
            mlp_layers = [hidden_size, hidden_size, 1]
            
        # LSTM Parameter: 4 × (d×h + h×h + 2×h)
        d = input_size
        h = hidden_size
        lstm_params = 4 * (d * h + h * h + 2 * h)
        
        # MLP Parameter
        mlp_params = 0
        prev_size = h
        
        for layer_size in mlp_layers:
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
        """Findet alle Model-Verzeichnisse"""
        model_dirs = []
        
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
                        'train_script_path': self.find_training_script(item_path)
                    })
        
        return model_dirs
    
    def find_training_script(self, model_dir):
        """Findet das Trainings-Skript"""
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.py') and 'train' in file.lower():
                    return os.path.join(root, file)
        return None
    
    def analyze_training_script(self, script_path):
        """Analysiert das Trainings-Skript"""
        if not script_path or not os.path.exists(script_path):
            return None
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
              # Extrahiere Architektur-Parameter (sowohl Groß- als auch Kleinbuchstaben)
            hidden_pattern = r'(?:HIDDEN_SIZE|hidden_size)\s*=\s*(\d+)'
            input_pattern = r'(?:INPUT_SIZE|input_size)\s*=\s*(\d+)'
            
            hidden_match = re.search(hidden_pattern, content, re.IGNORECASE)
            input_match = re.search(input_pattern, content, re.IGNORECASE)
            
            architecture = {}
            if hidden_match:
                architecture['hidden_size'] = int(hidden_match.group(1))
            if input_match:
                architecture['input_size'] = int(input_match.group(1))
            
            return {'architecture': architecture}
            
        except Exception as e:
            print(f"❌ Fehler beim Analysieren des Skripts: {e}")
            return None
    
    def analyze_pytorch_model(self, model_path):
        """Analysiert eine PyTorch .pth Datei"""
        if not os.path.exists(model_path):
            return None
        
        try:
            # Lade Model State Dict
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Berechne Parameter
            total_params = 0
            layer_groups = {}
            
            for param_name, param_tensor in state_dict.items():
                param_count = param_tensor.numel()
                total_params += param_count
                
                layer_type = param_name.split('.')[0]
                if layer_type not in layer_groups:
                    layer_groups[layer_type] = 0
                layer_groups[layer_type] += param_count
            
            # Dateigröße
            file_size_bytes = os.path.getsize(model_path)
            file_size_kb = file_size_bytes / 1024
            
            analysis = {
                'total_parameters': total_params,
                'layer_info': layer_groups,
                'memory_size_kb': (total_params * 4) / 1024,
                'file_size_kb': file_size_kb
            }
            
            # LSTM-Architektur erkennen
            lstm_arch = self.detect_lstm_from_parameters(state_dict)
            if lstm_arch:
                analysis['lstm_architecture'] = lstm_arch
            
            return analysis
            
        except Exception as e:
            print(f"❌ Fehler beim Laden des Models: {e}")
            return None
    
    def detect_lstm_from_parameters(self, state_dict):
        """Erkennt LSTM-Architektur aus Parametern"""
        lstm_info = {}
        
        for param_name, param_tensor in state_dict.items():
            shape = param_tensor.shape
            
            if 'weight_ih' in param_name and len(shape) == 2:
                # Input-to-Hidden: [4*hidden_size, input_size]
                lstm_info['hidden_size'] = shape[0] // 4
                lstm_info['input_size'] = shape[1]
            elif 'weight_hh' in param_name and len(shape) == 2:
                # Hidden-to-Hidden: [4*hidden_size, hidden_size]
                lstm_info['hidden_size'] = shape[0] // 4
        
        if 'input_size' in lstm_info and 'hidden_size' in lstm_info:
            i = lstm_info['input_size']
            h = lstm_info['hidden_size']
            
            # Erwartete Parameter berechnen
            complete_calc = self.calculate_complete_model_parameters(i, h, None)
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
        
        # Analysiere Trainings-Skript
        if arch_info['train_script_path']:
            result['training_script_analysis'] = self.analyze_training_script(arch_info['train_script_path'])
        
        # Analysiere PyTorch Model
        if os.path.exists(arch_info['best_model_path']):
            result['pytorch_model_analysis'] = self.analyze_pytorch_model(arch_info['best_model_path'])
        
        # Vergleiche
        result['comparison'] = self.compare_script_vs_model(
            result['training_script_analysis'],
            result['pytorch_model_analysis']
        )
        
        return result
    
    def compare_script_vs_model(self, script_analysis, model_analysis):
        """Vergleicht Trainings-Skript mit Model"""
        comparison = {
            'matches': {},
            'discrepancies': {},
            'validation': 'unknown'
        }
        
        if not script_analysis or not model_analysis:
            comparison['validation'] = 'incomplete_data'
            return comparison
        
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
                    comparison['discrepancies'][param] = {'script': script_val, 'model': model_val}
        
        if comparison['discrepancies']:
            comparison['validation'] = 'mismatch'
        elif comparison['matches']:
            comparison['validation'] = 'match'
        else:
            comparison['validation'] = 'no_comparison_possible'
        
        return comparison
    
    def calculate_arduino_flash_ram_estimation(self, hidden_size, input_size=1):
        """Arduino Flash/RAM-Abschätzung mit realen Messwerten"""
        
        # Reale Messungen
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
        
        # Parameter berechnen
        model_params = self.calculate_complete_model_parameters(input_size, hidden_size)
        total_parameters = model_params['total']
        
        # Flash-Abschätzung
        adaptive_base = 20.0 + (hidden_size * 0.8) + (hidden_size * hidden_size * 0.008)
        
        flash_components = {
            'arduino_adaptive_base': adaptive_base,
            'model_weights_kb': (total_parameters * 4) / 1024,
            'compiler_optimization': hidden_size * 0.35,
        }
        
        estimated_flash_kb = sum(flash_components.values())
        
        # RAM-Abschätzung
        ram_components = {
            'arduino_framework': 2.2,
            'serial_buffers': 0.8,
            'system_reserved': 1.3,
            'stack_space': 1.8 + (hidden_size * 0.01),
            'heap_overhead': 0.4,
            'interrupt_vectors': 0.5,
            'lstm_hidden_state': (hidden_size * 4) / 1024,
            'lstm_cell_state': (hidden_size * 4) / 1024,
            'lstm_temp_buffers': (hidden_size * 12) / 1024,
            'mlp_activation_buffers': (hidden_size * 6) / 1024,
            'input_buffer': (input_size * 8 * 4) / 1024,
            'output_buffer': 0.15,
            'global_variables': 0.3 + (hidden_size * 0.005),
        }
        
        estimated_ram_kb = sum(ram_components.values())
        
        # Validierung
        validation = {}
        measured_data = real_measurements.get(hidden_size)
        
        if measured_data:
            validation['has_real_data'] = True
            validation['flash_measured_kb'] = measured_data['flash_used_kb']
            validation['ram_measured_kb'] = measured_data['ram_used_kb']
            validation['flash_error_percent'] = abs(estimated_flash_kb - measured_data['flash_used_kb']) / measured_data['flash_used_kb'] * 100
            validation['ram_error_percent'] = abs(estimated_ram_kb - measured_data['ram_used_kb']) / measured_data['ram_used_kb'] * 100
        else:
            validation['has_real_data'] = False
        
        # Arduino R4 Kompatibilität
        arduino_r4_specs = {
            'flash_total_kb': 256,
            'ram_total_kb': 32,
            'flash_available_percent': (estimated_flash_kb / 256) * 100,
            'ram_available_percent': (estimated_ram_kb / 32) * 100,
            'flash_fits': estimated_flash_kb < 256,
            'ram_fits': estimated_ram_kb < 32,
            'overall_fits': (estimated_flash_kb < 256) and (estimated_ram_kb < 32)
        }
        
        return {
            'architecture': f'{hidden_size}×{hidden_size}',
            'input_size': input_size,
            'total_parameters': total_parameters,
            'flash_estimated_kb': estimated_flash_kb,
            'flash_components': flash_components,
            'ram_estimated_kb': estimated_ram_kb,
            'ram_components': ram_components,
            'arduino_r4_compatibility': arduino_r4_specs,
            'validation': validation,
        }
    
    def create_comprehensive_visualization(self):
        """Erstellt das umfassende 4-Panel Diagramm wie im Bild"""
        
        if not self.results:
            print("❌ Keine Daten für Visualisierung")
            return
          # Sammle Daten
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
                input_size = script_analysis['architecture'].get('input_size', 4)  # Default 4 wie in copy
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
                
                # Theoretische Dateigröße
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
        
        # Erstelle 4-Panel Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
          # ========== 1. Parameter-Vergleich: Skript vs. Model MIT FEHLERBALKEN ==========
        x = np.arange(len(architectures))
        width = 0.35
        
        # Berechne Fehler (Abweichung zwischen theoretisch und gemessen)
        errors = []
        for theoretical, measured in zip(script_params, model_params):
            if theoretical > 0 and measured > 0:
                error = abs(measured - theoretical)
                errors.append(error)
            else:
                errors.append(0)
        
        # Balken mit Fehlerbalken
        bars1 = ax1.bar(x - width/2, script_params, width, label='Formel (LSTM+MLP)', alpha=0.7, color='lightblue')
        bars2 = ax1.bar(x + width/2, model_params, width, label='PyTorch Model (tatsächlich)', alpha=0.7, color='orange')
        
        # FEHLERBALKEN hinzufügen (T-förmig mit horizontalen Enden)
        for i, (x_pos, measured, theoretical, error) in enumerate(zip(x, model_params, script_params, errors)):
            if error > 0 and measured > 0:
                # Fehlerbalken auf den gemessenen Balken
                ax1.errorbar(x_pos + width/2, measured, yerr=error, 
                           fmt='none', color='black', capsize=8, capthick=2, linewidth=2)
        
        ax1.set_xlabel('Architektur')
        ax1.set_ylabel('Anzahl Parameter')
        ax1.set_title('Parameter-Vergleich: Theoretische Formel vs. PyTorch Model (mit Fehlerbalken)')
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
        
        # ========== 2. Dateigröße vs. Parameter ==========
        colors = ['green' if status == 'match' else 'orange' if status == 'mismatch' else 'red' 
                 for status in validation_status]
        
        # Scatter Plots
        scatter1 = ax2.scatter(model_params, file_sizes, c=colors, s=100, alpha=0.7, 
                              label='Gemessene Dateigröße', marker='o')
        scatter2 = ax2.scatter(model_params, theoretical_file_sizes, c='blue', s=80, alpha=0.5,
                              label='Theoretische Dateigröße (4 bytes/param)', marker='^')
          # Verbindungslinien
        for i in range(len(model_params)):
            if model_params[i] > 0:
                ax2.plot([model_params[i], model_params[i]], 
                        [theoretical_file_sizes[i], file_sizes[i]], 
                        'k--', alpha=0.3, linewidth=1)
        
        # Annotationen
        for i, arch in enumerate(architectures):
            if model_params[i] > 0:
                # Berechne Overhead-Prozent
                overhead_percent = ((file_sizes[i] - theoretical_file_sizes[i]) / theoretical_file_sizes[i]) * 100 if theoretical_file_sizes[i] > 0 else 0
                
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
            Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=8, alpha=0.7, label='Gemessene Dateigröße'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', markersize=8, alpha=0.5, label='Theoretische Dateigröße (4 bytes/param)'),
            Patch(facecolor='green', alpha=0.7, label='Formula ↔ Model Match'),
            Patch(facecolor='orange', alpha=0.7, label='Formula ↔ Model Mismatch'),
            Patch(facecolor='red', alpha=0.7, label='Unbekannt/Fehler')
        ]
        ax2.legend(handles=legend_elements, fontsize=8)
          # ========== 3. Hidden Size Scaling ==========
        if hidden_sizes and any(h > 0 for h in hidden_sizes):
            # Glatte theoretische Kurve wie in copy-Version
            max_hidden = max([h for h in hidden_sizes if h > 0])
            theoretical_curve_x = np.arange(8, max_hidden + 16, 8)  # Feinere Auflösung
            theoretical_curve_y = []
            
            for h in theoretical_curve_x:
                param_calc = self.calculate_complete_model_parameters(4, h, None)  # input_size=4 wie in copy
                theoretical_curve_y.append(param_calc['total'])
            
            ax3.plot(theoretical_curve_x, theoretical_curve_y, 'b-', linewidth=2, 
                    label='Theoretische Kurve (LSTM + MLP)')
            
            # Gemessene Modelle - verwende die richtigen Farben
            valid_hidden = [h for h in hidden_sizes if h > 0]
            valid_params = [p for h, p in zip(hidden_sizes, model_params) if h > 0]
            valid_archs = [a for h, a in zip(hidden_sizes, architectures) if h > 0]
            valid_colors = [c for h, c in zip(hidden_sizes, colors) if h > 0]
            
            ax3.scatter(valid_hidden, valid_params, c=valid_colors, s=100, alpha=0.8, 
                       marker='o', label='Gemessene Modelle', zorder=5)
            
            # Annotationen
            for h, p, arch in zip(valid_hidden, valid_params, valid_archs):
                ax3.annotate(arch, (h, p), xytext=(5, 5), textcoords='offset points', fontsize=9)
            
            ax3.set_xlabel('Hidden Size')
            ax3.set_ylabel('Anzahl Parameter')
            ax3.set_title('Parameter-Skalierung mit Hidden Size')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Keine Hidden Size\nDaten verfügbar', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=14)
          # ========== 4. Zusammenfassung ==========
        summary_text = "MODELL-ANALYSE ZUSAMMENFASSUNG\n\n"
        
        # Header
        summary_text += f"{'Arch':<12} | {'Param':<7} | {'Gemessen':<8} | {'Theor.':<8} | {'Overhead':<8}\n"
        summary_text += "-" * 60 + "\n"
        
        # Daten für jede Architektur
        for arch_name, result in self.results.items():
            arch_short = arch_name.replace('Stateful_', '').replace(' copy', '*')
            
            # Validierungsstatuts
            validation = result.get('comparison', {}).get('validation', 'unknown')
            status_symbol = "✅" if validation == 'match' else "⚠️" if validation == 'mismatch' else "❌"
            
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis:
                params = model_analysis.get('total_parameters', 0)
                file_size = model_analysis.get('file_size_kb', 0)
                theoretical_size = (params * 4) / 1024
                
                if theoretical_size > 0:
                    overhead_pct = ((file_size - theoretical_size) / theoretical_size) * 100
                else:
                    overhead_pct = 0
                
                summary_text += f"{arch_short:<12} | {params//1000:>4}k | {file_size:>6.1f}kB | {theoretical_size:>6.1f}kB | +{overhead_pct:>5.1f}%\n"
        
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
    
    def create_arduino_memory_estimation_visualization(self):
        """Erstellt Arduino Memory Estimation mit realen Messwerten und Tortendiagrammen"""
        
        if not self.results:
            print("❌ Keine Daten für Arduino Memory Estimation")
            return
        
        # Sammle Daten
        architectures = []
        estimation_data = []
        hidden_sizes = []
        
        # Reale Messwerte
        real_measurements = {
            16: {'flash_used_kb': 48.8, 'ram_used_kb': 7.7},
            32: {'flash_used_kb': 106.9, 'ram_used_kb': 8.9},
            64: {'flash_used_kb': 123.0, 'ram_used_kb': 9.8}
        }
        
        for arch_name, result in self.results.items():
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis and 'lstm_architecture' in model_analysis:
                lstm_arch = model_analysis['lstm_architecture']
                hidden_size = lstm_arch.get('hidden_size')
                input_size = lstm_arch.get('input_size', 1)
                
                if hidden_size:
                    estimation = self.calculate_arduino_flash_ram_estimation(hidden_size, input_size)
                    architectures.append(arch_name.replace('Stateful_', ''))
                    estimation_data.append(estimation)
                    hidden_sizes.append(hidden_size)
        
        if not estimation_data:
            print("❌ Keine Arduino Memory Estimation verfügbar")
            return
        
        # Erstelle 4-Panel Plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # ========== 1. Flash vs. RAM: Estimation vs. Real ==========
        flash_est = [data['flash_estimated_kb'] for data in estimation_data]
        ram_est = [data['ram_estimated_kb'] for data in estimation_data]
        
        # Reale Messwerte
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
        
        # Flash Bars
        bars1 = ax1.bar(x - width*1.5, flash_est, width, label='Flash Estimation', 
                       alpha=0.8, color='lightcoral', edgecolor='darkred')
        flash_real_filtered = [f for f in flash_real if f is not None]
        if flash_real_filtered:
            bars2 = ax1.bar(x[:len(flash_real_filtered)] - width*0.5, flash_real_filtered, width, 
                           label='Flash Real', alpha=0.8, color='red', edgecolor='darkred')
        
        # RAM Bars  
        bars3 = ax1.bar(x + width*0.5, ram_est, width, label='RAM Estimation', 
                       alpha=0.8, color='lightblue', edgecolor='darkblue')
        ram_real_filtered = [r for r in ram_real if r is not None]
        if ram_real_filtered:
            bars4 = ax1.bar(x[:len(ram_real_filtered)] + width*1.5, ram_real_filtered, width,
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
        ax1.grid(True, alpha=0.3)        # ========== 2. Logarithmische Skalierung mit DURCHGEHENDEN FEHLERKURVEN ==========
        # Erstelle Zukunftsprognose: 16, 32, 64, 128, 256, 512
        future_hidden_sizes = [16, 32, 64, 128, 256, 512]  # X-Achse NICHT logarithmisch!
        
        log_flash_est = []
        log_ram_est = []
        
        # Berechne Schätzungen für alle Größen
        for hs in future_hidden_sizes:
            estimation = self.calculate_arduino_flash_ram_estimation(hs, 1)
            log_flash_est.append(estimation['flash_estimated_kb'])
            log_ram_est.append(estimation['ram_estimated_kb'])
        
        # BERECHNE DURCHGEHENDE FEHLERKURVEN basierend auf realen Abweichungen
        flash_errors_percent = []  # Sammle relative Fehler von realen Messwerten
        ram_errors_percent = []
        
        for hs in [16, 32, 64]:  # Nur für die mit realen Messwerten
            if hs in real_measurements:
                estimation = self.calculate_arduino_flash_ram_estimation(hs, 1)
                real_flash = real_measurements[hs]['flash_used_kb']
                real_ram = real_measurements[hs]['ram_used_kb']
                
                flash_error_pct = abs(estimation['flash_estimated_kb'] - real_flash) / real_flash
                ram_error_pct = abs(estimation['ram_estimated_kb'] - real_ram) / real_ram
                
                flash_errors_percent.append(flash_error_pct)
                ram_errors_percent.append(ram_error_pct)
        
        # Mittlere relative Fehler für Zukunftsprognose
        avg_flash_error_pct = np.mean(flash_errors_percent) if flash_errors_percent else 0.15
        avg_ram_error_pct = np.mean(ram_errors_percent) if ram_errors_percent else 0.12
        
        print(f"📊 Durchschnittliche Fehler: Flash {avg_flash_error_pct:.1%}, RAM {avg_ram_error_pct:.1%}")
        
        # Berechne obere und untere Grenzen für ALLE Punkte
        flash_upper = [est * (1 + avg_flash_error_pct) for est in log_flash_est]
        flash_lower = [est * (1 - avg_flash_error_pct) for est in log_flash_est]
        ram_upper = [est * (1 + avg_ram_error_pct) for est in log_ram_est]
        ram_lower = [est * (1 - avg_ram_error_pct) for est in log_ram_est]
          # HAUPTKURVEN (Mittlere Schätzung)
        ax2.plot(future_hidden_sizes, log_flash_est, 'o-', color='red', alpha=0.9, 
                label='Flash Estimation', linewidth=3, markersize=10)
        ax2.plot(future_hidden_sizes, log_ram_est, 'o-', color='blue', alpha=0.9,
                label='RAM Estimation', linewidth=3, markersize=10)
        
        # FLASH: Schraffierter Bereich + Fehlerstriche (große Unsicherheit 33.9%)
        ax2.fill_between(future_hidden_sizes, flash_lower, flash_upper, 
                        color='red', alpha=0.2, label='Flash Unsicherheitsbereich')
        
        # Flash Fehlerstriche (zusätzlich zum Bereich)
        flash_errors_absolute = [est * avg_flash_error_pct for est in log_flash_est]
        ax2.errorbar(future_hidden_sizes, log_flash_est, yerr=flash_errors_absolute,
                    fmt='none', color='darkred', alpha=0.8, 
                    capsize=8, capthick=3, linewidth=2, label='Flash Error Bars')
        
        # RAM: NUR Fehlerstriche (kleine Unsicherheit 3.2%, kein schraffierter Bereich)
        ram_errors_absolute = [est * avg_ram_error_pct for est in log_ram_est]
        ax2.errorbar(future_hidden_sizes, log_ram_est, yerr=ram_errors_absolute,
                    fmt='none', color='darkblue', alpha=0.8,
                    capsize=8, capthick=3, linewidth=2, label='RAM Error Bars')
        
        # OBERE UND UNTERE GRENZLINIEN nur für Flash (gestrichelt)
        ax2.plot(future_hidden_sizes, flash_upper, '--', color='red', alpha=0.6, linewidth=2)
        ax2.plot(future_hidden_sizes, flash_lower, '--', color='red', alpha=0.6, linewidth=2)
        
        # Reale Messwerte als Referenz (große schwarze Quadrate)
        real_hs = list(real_measurements.keys())
        real_flash_vals = [real_measurements[hs]['flash_used_kb'] for hs in real_hs]
        real_ram_vals = [real_measurements[hs]['ram_used_kb'] for hs in real_hs]
        
        ax2.scatter(real_hs, real_flash_vals, color='darkred', s=200, marker='s', 
                   label='Flash Real Measurements', zorder=5, edgecolors='black', linewidth=2)
        ax2.scatter(real_hs, real_ram_vals, color='darkblue', s=200, marker='s',
                   label='RAM Real Measurements', zorder=5, edgecolors='black', linewidth=2)
        
        # Arduino Limits als horizontale Linien
        ax2.axhline(y=256, color='red', linestyle='--', alpha=0.8, linewidth=3, 
                   label='Arduino R4 Flash Limit (256 kB)')
        ax2.axhline(y=32, color='blue', linestyle='--', alpha=0.8, linewidth=3,
                   label='Arduino R4 RAM Limit (32 kB)')        # X-Achse linear, Y-Achse logarithmisch
        ax2.set_xticks(future_hidden_sizes)
        ax2.set_xticklabels([f'{hs}' for hs in future_hidden_sizes])
        ax2.set_yscale('log')
        ax2.set_xlabel('Hidden Size (Linear Scale)')
        ax2.set_ylabel('Speicher (kB, Log Scale)')
        ax2.set_title('Logarithmische Skalierung: Flash/RAM vs. Hidden Size\n(mit durchgehenden Unsicherheitsbereichen)')
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # ========== 3. Flash-Komponenten Tortendiagramm ==========
        if estimation_data:
            sample_estimation = estimation_data[0]
            flash_components = sample_estimation['flash_components']
            
            sizes = list(flash_components.values())
            labels = [label.replace('_', ' ').title() for label in flash_components.keys()]
            colors_pie = plt.cm.Reds(np.linspace(0.3, 0.8, len(sizes)))
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             colors=colors_pie, startangle=90)
            ax3.set_title(f'Flash-Komponenten Breakdown\n({architectures[0]} Architektur)')
        
        # ========== 4. RAM-Komponenten Tortendiagramm ==========
        if estimation_data:
            ram_components = sample_estimation['ram_components']
            
            sizes = list(ram_components.values())
            labels = [label.replace('_', ' ').title() for label in ram_components.keys()]
            colors_pie = plt.cm.Blues(np.linspace(0.3, 0.8, len(sizes)))
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                             colors=colors_pie, startangle=90)
            ax4.set_title(f'RAM-Komponenten Breakdown\n({architectures[0]} Architektur)')
        
        plt.tight_layout()
        
        # Speichere Plot
        plot_filename = "arduino_memory_estimation_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n🔋 Arduino Memory Estimation Analyse gespeichert: {plot_filename}")
        
        return fig
    
    def run_complete_analysis(self):
        """Führt die komplette Analyse durch"""
        print("🚀 Starte umfassende Modell-Analyse...")
        
        # Finde alle Modell-Verzeichnisse
        model_dirs = self.find_model_directories()
        
        if not model_dirs:
            print("❌ Keine Modell-Verzeichnisse gefunden!")
            return {}
        
        print(f"📁 Gefunden: {len(model_dirs)} Modell-Verzeichnisse")
        
        # Analysiere jedes Verzeichnis
        for arch_info in model_dirs:
            result = self.analyze_architecture_directory(arch_info)
            self.results[arch_info['architecture']] = result
        
        # Erstelle Visualisierungen
        print(f"\n📊 Erstelle Visualisierungen...")
        
        self.create_comprehensive_visualization()
        self.create_arduino_memory_estimation_visualization()
        
        print(f"\n✅ Analyse abgeschlossen!")
        return self.results

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

if __name__ == "__main__":
    main()
