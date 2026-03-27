# Comprehensive Model Analysis - FIXED VERSION
# Analyzes PyTorch models, training scripts and Arduino weights systematically

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
          # New color palette - mainly green & blue, minimal purple accents
        self.color_scheme = {
            # 🌿 Main Stateful LSTM colors - green/mint tones
            'stateful_lstm': '#A8E6CF',      # Soft mint green for Stateful LSTM (kept)
            'stateful_secondary': '#C8F0DB', # Very light mint for secondary elements (kept)
            
            # 🔵 Main Windows LSTM colors - blue tones (replaced purple/lavender)
            'windows_lstm': '#87CEEB',       # Sky blue for Windows LSTM (NEW - replaces lavender)
            'windows_secondary': '#B0E0E6',  # Powder blue for secondary elements (NEW - replaces light lavender)
            
            # Architecture-specific colors - green variants for Stateful
            'stateful_16': '#A8E6CF',        # Base soft mint green (kept)
            'stateful_32': '#7DD3C0',        # Medium mint green for variation (NEW)
            'stateful_64': '#A8E6CF',        # Base soft mint green (kept)
            'stateful_16_secondary': '#C8F0DB',  # Light mint (kept)
            'stateful_32_secondary': '#C8F0DB',  # Light mint (kept)
            'stateful_64_secondary': '#C8F0DB',  # Light mint (kept)
              # Architecture-specific colors - blue variants for Windows
            'windows_16': '#87CEEB',         # Sky blue (NEW - replaces lavender)
            'windows_32': '#6BB6DD',         # Medium blue (NEW - replaces lavender)
            'windows_64': '#87CEEB',         # Sky blue (NEW - replaces lavender)
            'windows_16_secondary': '#B0E0E6',   # Powder blue (NEW - replaces light lavender)
            'windows_32_secondary': '#B0E0E6',   # Powder blue (NEW - replaces light lavender)
            'windows_64_secondary': '#B0E0E6',   # Powder blue (NEW - replaces light lavender)
              # 🔷 Blue accent - professional accent color
            'blue_accent': '#5B9BD5',         # Professional blue for accents
            
            # ⚡ Kräftige Akzent-Farben für Details
            'accent_blue': '#2091C9',        # Lebendiges Blau für Linien, Punkte, Verbindungen
            'accent_violet': '#BB76F7',      # Elegantes Violett für spezielle Highlights
            'error_color': '#D9140E'         # Signalfarbe Rot für Fehler (beibehalten)
        }
    
    def calculate_complete_model_parameters(self, input_size, hidden_size, mlp_layers=None):
        """Calculates the complete model parameters (LSTM + MLP Head)"""
        if hidden_size <= 0:
            return {'total': 0, 'lstm': 0, 'mlp': 0}
            
        if mlp_layers is None:
            mlp_layers = [hidden_size, hidden_size, 1]
            
        # LSTM Parameters: 4 × (d×h + h×h + 2×h)
        d = input_size
        h = hidden_size
        lstm_params = 4 * (d * h + h * h + 2 * h)
        
        # MLP Parameters
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
            }        }
        
    def find_model_directories(self):
        """Finds all model directories"""
        model_dirs = []
        
        for item in os.listdir(self.base_path):
            item_path = os.path.join(self.base_path, item)
            # Skip copy directories
            if 'copy' in item.lower():
                continue
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
        """Finds the training script"""
        for root, dirs, files in os.walk(model_dir):
            for file in files:
                if file.endswith('.py') and 'train' in file.lower():
                    return os.path.join(root, file)
        return None
    
    def analyze_training_script(self, script_path):
        """Analyzes the training script"""
        if not script_path or not os.path.exists(script_path):
            return None
        
        try:
            with open(script_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Extract architecture parameters (both upper and lowercase)
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
            print(f"❌ Error analyzing script: {e}")
            return None
    
    def analyze_pytorch_model(self, model_path):
        """Analyzes a PyTorch .pth file"""
        if not os.path.exists(model_path):
            return None
        
        try:
            # Load model state dict
            state_dict = torch.load(model_path, map_location='cpu')
            
            # Calculate parameters
            total_params = 0
            layer_groups = {}
            
            for param_name, param_tensor in state_dict.items():
                param_count = param_tensor.numel()
                total_params += param_count
                
                layer_type = param_name.split('.')[0]
                if layer_type not in layer_groups:
                    layer_groups[layer_type] = 0
                layer_groups[layer_type] += param_count
            
            # File size
            file_size_bytes = os.path.getsize(model_path)
            file_size_kb = file_size_bytes / 1024
            
            analysis = {
                'total_parameters': total_params,
                'layer_info': layer_groups,
                'memory_size_kb': (total_params * 4) / 1024,
                'file_size_kb': file_size_kb
            }
              # Detect LSTM architecture
            lstm_arch = self.detect_lstm_from_parameters(state_dict)
            if lstm_arch:
                analysis['lstm_architecture'] = lstm_arch
            
            return analysis
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None
    
    def detect_lstm_from_parameters(self, state_dict):
        """Detects LSTM architecture from parameters"""
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
            
            # Calculate expected parameters
            complete_calc = self.calculate_complete_model_parameters(i, h, None)
            lstm_info['expected_complete_parameters'] = complete_calc['total']
            lstm_info['parameter_breakdown'] = complete_calc
            
        return lstm_info if lstm_info else None
    
    def get_model_type_and_color(self, architecture_name):
        """Determine model type and return appropriate colors"""
        arch_lower = architecture_name.lower()
        
        # Determine architecture size
        if '16_16' in arch_lower or '16x16' in arch_lower:
            size = '16'
        elif '32_32' in arch_lower or '32x32' in arch_lower:
            size = '32'
        elif '64_64' in arch_lower or '64x64' in arch_lower:
            size = '64'
        else:
            size = '16'  # Default
        
        if 'stateful' in arch_lower:
            primary_key = f'stateful_{size}'
            secondary_key = f'stateful_{size}_secondary'
            model_type = 'stateful_lstm'
        elif 'window' in arch_lower:
            primary_key = f'windows_{size}'
            secondary_key = f'windows_{size}_secondary'
            model_type = 'windows_lstm'
        else:
            # Default to stateful LSTM
            primary_key = f'stateful_{size}'
            secondary_key = f'stateful_{size}_secondary'
            model_type = 'stateful_lstm'
        
        # Get colors with fallback
        primary_color = self.color_scheme.get(primary_key, self.color_scheme['stateful_lstm'])
        secondary_color = self.color_scheme.get(secondary_key, self.color_scheme['stateful_secondary'])
        
        return model_type, primary_color, secondary_color
    
    def analyze_architecture_directory(self, arch_info):
        """Analyzes a complete architecture directory"""
        print(f"\n{'='*80}")
        print(f"ANALYZING ARCHITECTURE: {arch_info['architecture']}")
        print(f"{'='*80}")
        
        result = {
            'architecture_name': arch_info['architecture'],
            'base_directory': arch_info['base_dir'],
            'training_script_analysis': None,
            'pytorch_model_analysis': None,
            'comparison': {}
        }
        
        # Analyze training script
        if arch_info['train_script_path']:
            result['training_script_analysis'] = self.analyze_training_script(arch_info['train_script_path'])
        
        # Analyze PyTorch model
        if os.path.exists(arch_info['best_model_path']):
            result['pytorch_model_analysis'] = self.analyze_pytorch_model(arch_info['best_model_path'])
        
        # Compare
        result['comparison'] = self.compare_script_vs_model(
            result['training_script_analysis'],
            result['pytorch_model_analysis']
        )
        
        return result
    
    def compare_script_vs_model(self, script_analysis, model_analysis):
        """Compares training script with model"""
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
        """Arduino Flash/RAM estimation with real measurements"""
        
        # Real measurements
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
        
        # Calculate parameters
        model_params = self.calculate_complete_model_parameters(input_size, hidden_size)
        total_parameters = model_params['total']
        
        # Flash estimation - simplified to English categories  
        adaptive_base = 20.0 + (hidden_size * 0.8) + (hidden_size * hidden_size * 0.008)
        
        flash_components = {
            'Arduino Framework Base': adaptive_base,
            'Model Weights': (total_parameters * 4) / 1024,
        }
        
        estimated_flash_kb = sum(flash_components.values())
        
        # RAM estimation - simplified categories
        
        # Fixed system overhead components (constant for Arduino)
        fixed_overhead = (2.2 + 0.8 + 1.3 + 0.4 + 0.5 + 0.15 + 0.3)  # arduino_framework + serial_buffers + system_reserved + heap_overhead + interrupt_vectors + output_buffer + base_variables
        
        # Variable components that scale with model size
        stack_space_kb = 1.8 + (hidden_size * 0.01)
        lstm_states_kb = ((hidden_size * 4) + (hidden_size * 4)) / 1024  # hidden + cell state
        lstm_temp_buffers_kb = (hidden_size * 12) / 1024
        mlp_buffers_kb = (hidden_size * 6) / 1024
        input_buffer_kb = (input_size * 8 * 4) / 1024
        additional_variables_kb = (hidden_size * 0.005)
        
        variable_values = stack_space_kb + lstm_states_kb + lstm_temp_buffers_kb + mlp_buffers_kb + input_buffer_kb + additional_variables_kb
        
        ram_components = {
            'Fixed System Overhead': fixed_overhead,
            'Variable Model Components': variable_values,
        }
        
        estimated_ram_kb = sum(ram_components.values())
        
        # Validation
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
        
        # Arduino R4 compatibility
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
        """Creates the comprehensive 4-panel diagram as in the image"""
        
        if not self.results:
            print("❌ No data for visualization")
            return        # Collect data
        architectures = []
        script_params = []
        model_params = []
        file_sizes = []
        theoretical_file_sizes = []
        hidden_sizes = []
        validation_status = []
        model_colors = []  # Store colors for each model
        
        for arch_name, result in self.results.items():
            architectures.append(arch_name.replace('Stateful_', ''))
            
            # Get model type and colors
            model_type, primary_color, secondary_color = self.get_model_type_and_color(arch_name)
            model_colors.append((primary_color, secondary_color))
            
            # Parameters from script
            script_analysis = result.get('training_script_analysis')
            if script_analysis and 'architecture' in script_analysis:
                hidden = script_analysis['architecture'].get('hidden_size', 0)
                input_size = script_analysis['architecture'].get('input_size', 4)  # Default 4 as in copy
                param_calc = self.calculate_complete_model_parameters(input_size, hidden, None)
                script_param_count = param_calc['total']
            else:
                script_param_count = 0
            script_params.append(script_param_count)
            
            # Parameters from model
            model_analysis = result.get('pytorch_model_analysis')
            if model_analysis:
                actual_params = model_analysis.get('total_parameters', 0)
                actual_file_size = model_analysis.get('file_size_kb', 0)
                model_params.append(actual_params)
                file_sizes.append(actual_file_size)
                
                # Theoretical file size
                theoretical_size_kb = (actual_params * 4) / 1024
                theoretical_file_sizes.append(theoretical_size_kb)
                
                lstm_arch = model_analysis.get('lstm_architecture', {})
                hidden_sizes.append(lstm_arch.get('hidden_size', 0))
            else:
                model_params.append(0)
                file_sizes.append(0)
                theoretical_file_sizes.append(0)
                hidden_sizes.append(0)            # Validation status
            comparison = result.get('comparison', {})
            validation_status.append(comparison.get('validation', 'unknown'))
        
        # Create 4-panel plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # ========== 1. Parameter Comparison: Script vs. Model WITH ERROR BARS ==========
        x = np.arange(len(architectures))
        width = 0.35
        
        # Calculate errors (deviation between theoretical and measured)
        errors = []
        for theoretical, measured in zip(script_params, model_params):
            if theoretical > 0 and measured > 0:
                error = abs(measured - theoretical)
                errors.append(error)
            else:
                errors.append(0)
        
        # Bars with error bars - use consistent colors
        # Use only 2 colors: one for theoretical, one for measured
        theoretical_color = '#B0E0E6'  # Light blue for theoretical
        measured_color = '#5B9BD5'     # Professional blue for measured
        
        bars1 = ax1.bar(x - width/2, script_params, width, label='Theoretical (Formula)', 
                       alpha=0.7, color=theoretical_color, edgecolor='black', linewidth=0.5)
        bars2 = ax1.bar(x + width/2, model_params, width, label='Measured (PyTorch Model)', 
                       alpha=0.8, color=measured_color, edgecolor='black', linewidth=0.5)
        
        # Add ERROR BARS (T-shaped with horizontal ends)
        for i, (x_pos, measured, theoretical, error) in enumerate(zip(x, model_params, script_params, errors)):
            if error > 0 and measured > 0:
                # Error bars on measured bars
                ax1.errorbar(x_pos + width/2, measured, yerr=error, 
                           fmt='none', color='black', capsize=8, capthick=2, linewidth=2)
        
        ax1.set_xlabel('Architecture')
        ax1.set_ylabel('Parameters Count')
        ax1.set_title('Parameter Comparison: Theoretical vs. Measured\n(Formula vs. PyTorch Model)', fontsize=20) # Updated title
        ax1.set_xticks(x)
        ax1.set_xticklabels([arch.replace('_', 'x') for arch in architectures], rotation=45, fontsize=15) # Change 16_16 to 16x16
        
        # Simple legend with only 2 colors
        ax1.legend(fontsize=12, loc='upper left') # Simplified legend
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('Architecture', fontsize=18) # Increased
        ax1.set_ylabel('Parameters Count', fontsize=18) # Increased
        
        # Values on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.02,                            f'{int(height):,}', ha='center', va='bottom', fontsize=12) # Increased, removed bold
        
        # ========== 2. File Size vs. Parameter ==========
        # Create scatter colors based on validation status and model type
        scatter_colors = []
        for i, (status, (primary_color, secondary_color)) in enumerate(zip(validation_status, model_colors)):
            if status == 'match':
                scatter_colors.append(primary_color)  # Use model's primary color for matches
            elif status == 'mismatch':
                scatter_colors.append(secondary_color)  # Use secondary color for mismatches
            else:
                scatter_colors.append('red')  # Red for errors/unknown

        # Scatter plots with model-specific colors
        scatter1 = ax2.scatter(model_params, file_sizes, c=scatter_colors, s=120, alpha=0.8, 
                              label='Measured File Size', marker='o', edgecolors='black', linewidth=1)
        scatter2 = ax2.scatter(model_params, theoretical_file_sizes, c='#87CEEB', s=80, alpha=0.6,
                              label='Theoretical File Size (4 bytes/param)', marker='^', edgecolors='black')

        # Connection lines - softer color from your diagram
        for i in range(len(model_params)):
            if model_params[i] > 0:
                ax2.plot([model_params[i], model_params[i]], 
                        [theoretical_file_sizes[i], file_sizes[i]], 
                        '--', color='#B0E0E6', alpha=0.4, linewidth=1)  # Powder blue
        
        # Annotations
        for i, arch in enumerate(architectures):
            if model_params[i] > 0:                # Calculate overhead percent
                overhead_percent = ((file_sizes[i] - theoretical_file_sizes[i]) / theoretical_file_sizes[i]) * 100 if theoretical_file_sizes[i] > 0 else 0
                
                ax2.annotate(f'{arch}\n+{overhead_percent:.1f}%', # Fixed newline
                           (model_params[i], file_sizes[i]), 
                           xytext=(5, 5), textcoords='offset points', fontsize=12) # Increased
        
        ax2.set_xlabel('Number of Parameters', fontsize=18) # Increased
        ax2.set_ylabel('File Size (kB)', fontsize=18) # Increased
        ax2.set_title('File Size vs. Parameter Count\n(Theoretical vs. Measured)', fontsize=20) # Increased, removed bold
        ax2.grid(True, alpha=0.3)
          # Legend with color coding information - using your harmonious palette
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#87CEEB', markersize=8, alpha=0.7, label='Measured File Size'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='#87CEEB', markersize=8, alpha=0.6, label='Theoretical File Size (4 bytes/param)'),
            Patch(facecolor=self.color_scheme['stateful_16'], alpha=0.7, label='16×16 Stateful (Match)'),
            Patch(facecolor=self.color_scheme['stateful_32'], alpha=0.7, label='32×32 Stateful (Match)'),
            Patch(facecolor=self.color_scheme['stateful_64'], alpha=0.7, label='64×64 Stateful (Match)')
        ]
        ax2.legend(handles=legend_elements, fontsize=12, loc='upper left') # Increased
        
        # ========== 3. Hidden Size Scaling ==========
        if hidden_sizes and any(h > 0 for h in hidden_sizes):
            # Smooth theoretical curve as in copy version
            max_hidden = max([h for h in hidden_sizes if h > 0])
            theoretical_curve_x = np.arange(8, max_hidden + 16, 8)  # Finer resolution
            theoretical_curve_y = []
            for h in theoretical_curve_x:
                param_calc = self.calculate_complete_model_parameters(4, h, None)  # input_size=4 as in copy
                theoretical_curve_y.append(param_calc['total'])
            
            ax3.plot(theoretical_curve_x, theoretical_curve_y, '-', color='#87CEEB', linewidth=2,
                    label='Theoretical Curve (LSTM + MLP)')  # Sky blue
            
            # Measured models - use model-specific colors
            valid_hidden = [h for h in hidden_sizes if h > 0]
            valid_params = [p for h, p in zip(hidden_sizes, model_params) if h > 0]
            valid_archs = [a for h, a in zip(hidden_sizes, architectures) if h > 0]
            valid_model_colors = [colors[0] for h, colors in zip(hidden_sizes, model_colors) if h > 0]  # Primary colors for valid models
            
            ax3.scatter(valid_hidden, valid_params, c=valid_model_colors, s=120, alpha=0.8, 
                       marker='o', label='Measured Models', zorder=5, edgecolors='black', linewidth=1)
            
            # Annotations
            for h, p, arch in zip(valid_hidden, valid_params, valid_archs):
                ax3.annotate(arch, (h, p), xytext=(5, 5), textcoords='offset points', fontsize=12) # Increased
            
            ax3.set_xlabel('Hidden Size', fontsize=18) # Increased
            ax3.set_ylabel('Number of Parameters', fontsize=18) # Increased
            ax3.set_title('Parameter Scaling with Hidden Size', fontsize=20) # Increased, removed bold
            ax3.legend(fontsize=14) # Increased
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'No Hidden Size\nData Available', 
                    ha='center', va='center', transform=ax3.transAxes, fontsize=18) # Increased
        
        # ========== 4. Summary ==========
        summary_text = "MODEL ANALYSIS SUMMARY\n\n"
        
        # Header
        summary_text += f"{'Arch':<12} | {'Param':<7} | {'Measured':<8} | {'Theory':<8} | {'Overhead':<8}\n"
        summary_text += "-" * 60 + "\n"
        
        # Data for each architecture
        for arch_name, result in self.results.items():
            arch_short = arch_name.replace('Stateful_', '').replace(' copy', '*')
            
            # Validation status
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
        summary_text += "\n📊 FILE SIZE OVERHEAD ANALYSIS:\n"
        summary_text += "• Theoretical: 4 bytes per parameter (float32)\n"
        summary_text += "• Measured: Including PyTorch metadata\n"
        summary_text += "• Overhead: Additional file size from\n"
        summary_text += "  model structure, optimizer state, etc.\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
                verticalalignment='top', fontsize=12, fontfamily='monospace', # Increased
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#B0E0E6", alpha=0.3))
        ax4.set_xlim(0, 1)
        ax4.set_ylim(0, 1)
        ax4.axis('off')
        plt.tight_layout()
        
        # Save plot
        plot_filename = "comprehensive_model_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n📊 Comprehensive analysis saved: {plot_filename}")
        
        # Show plot without blocking
        plt.show(block=False)
        return fig
    
    def create_arduino_memory_estimation_visualization(self):
        """Creates Arduino Memory Estimation with real measurements and pie charts"""
        
        if not self.results:
            print("❌ No data for Arduino Memory Estimation")
            return
        # Collect data
        architectures = []
        estimation_data = []
        hidden_sizes = []
        memory_model_colors = []  # Store colors for memory visualization
        
        # Real measurements
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
                    
                    # Get model colors for memory visualization
                    model_type, primary_color, secondary_color = self.get_model_type_and_color(arch_name)
                    memory_model_colors.append((primary_color, secondary_color))
        
        if not estimation_data:
            print("❌ No Arduino Memory Estimation available")
            return
        
        # Create 4-panel plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 16))
        
        # ========== 1. Flash vs. RAM: Estimation vs. Real ==========
        flash_est = [data['flash_estimated_kb'] for data in estimation_data]
        ram_est = [data['ram_estimated_kb'] for data in estimation_data]        # Real measurements
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
        
        # Use elegant color schemes for Flash and RAM bars - inspired by your diagram
        flash_colors = ["#C8F0DB", '#C8F0DB', '#C8F0DB']  # Light mint for Flash (from mint_blue)
        ram_colors = ['#B0E0E6', '#B0E0E6', '#B0E0E6']    # Light blue for RAM (from mint_blue)
        
        # Assign colors based on architecture size
        flash_bar_colors = []
        ram_bar_colors = []
        for i, hs in enumerate(hidden_sizes):
            if hs == 16:
                flash_bar_colors.append(flash_colors[0])
                ram_bar_colors.append(ram_colors[0])
            elif hs == 32:
                flash_bar_colors.append(flash_colors[1])
                ram_bar_colors.append(ram_colors[1])
            elif hs == 64:
                flash_bar_colors.append(flash_colors[2])
                ram_bar_colors.append(ram_colors[2])
            else:
                flash_bar_colors.append(flash_colors[0])  # Default
                ram_bar_colors.append(ram_colors[0])
        
        # Flash bars
        bars1 = ax1.bar(x - width*1.5, flash_est, width, label='Flash Estimation', 
                       alpha=0.8, color=flash_bar_colors, edgecolor='black', linewidth=1)  # Black edge
        flash_real_filtered = [f for f in flash_real if f is not None]
        if flash_real_filtered:
            # Use medium mint for real measurements - your diagram style
            flash_real_bar_colors = ['#A8E6CF'] * len(flash_real_filtered)  # Your soft mint (from mint_blue)
            bars2 = ax1.bar(x[:len(flash_real_filtered)] - width*0.5, flash_real_filtered, width, 
                           label='Flash Real', alpha=0.9, color=flash_real_bar_colors, 
                           edgecolor='black', linewidth=1)  # Black edge
        
        # RAM bars
        bars3 = ax1.bar(x + width*0.5, ram_est, width, label='RAM Estimation', 
                       alpha=0.8, color=ram_bar_colors, edgecolor='black', linewidth=1)  # Black edge
        ram_real_filtered = [r for r in ram_real if r is not None]
        if ram_real_filtered:            # Use medium blue for real RAM measurements - your diagram style  
            ram_real_bar_colors = ['#5B9BD5'] * len(ram_real_filtered)  # Professional blue (from mint_blue)
            bars4 = ax1.bar(x[:len(ram_real_filtered)] + width*1.5, ram_real_filtered, width,
                           label='RAM Real', alpha=0.9, color=ram_real_bar_colors, 
                           edgecolor='black', linewidth=1)  # Black edge
        
        # Arduino limits - use elegant soft colors from your diagram
        ax1.axhline(y=256, color='#A8E6CF', linestyle='--', alpha=0.7,
                   label='Arduino R4 Flash Limit (256 kB)')
        ax1.axhline(y=32, color='#5B9BD5', linestyle='--', alpha=0.7,
                   label='Arduino R4 RAM Limit (32 kB)')
        ax1.set_xlabel('Architecture', fontsize=18) # Increased
        ax1.set_ylabel('Memory (kB)', fontsize=18) # Increased
        ax1.set_title('Arduino Flash/RAM Estimation vs. Real Measurements', fontsize=20) # Increased, removed bold
        ax1.set_xticks(x)
        ax1.set_xticklabels([f'{arch}\n({hs}×{hs})' for arch, hs in zip(architectures, hidden_sizes)], fontsize=16) # Increased
        ax1.legend(fontsize=14) # Increased
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('Memory (kB)', fontsize=16) # Increased
        
        # ========== 2. Logarithmic Scaling with CONTINUOUS ERROR CURVES ==========
        # Future prediction: 16, 32, 64, 128, 256, 512
        future_hidden_sizes = [16, 32, 64, 128, 256, 512]  # X-axis NOT logarithmic!
        
        log_flash_est = []
        log_ram_est = []
        
        # Calculate estimations for all sizes
        for hs in future_hidden_sizes:
            estimation = self.calculate_arduino_flash_ram_estimation(hs, 1)
            log_flash_est.append(estimation['flash_estimated_kb'])
            log_ram_est.append(estimation['ram_estimated_kb'])
        
        # CALCULATE CONTINUOUS ERROR CURVES based on real deviations
        flash_errors_percent = []  # Collect relative errors from real measurements
        ram_errors_percent = []
        
        for hs in [16, 32, 64]:  # Only for those with real measurements
            if hs in real_measurements:
                estimation = self.calculate_arduino_flash_ram_estimation(hs, 1)
                real_flash = real_measurements[hs]['flash_used_kb']
                real_ram = real_measurements[hs]['ram_used_kb']
                
                flash_error_pct = abs(estimation['flash_estimated_kb'] - real_flash) / real_flash
                ram_error_pct = abs(estimation['ram_estimated_kb'] - real_ram) / real_ram
                
                flash_errors_percent.append(flash_error_pct)
                ram_errors_percent.append(ram_error_pct)
        
        # Average relative errors for future prediction
        avg_flash_error_pct = np.mean(flash_errors_percent) if flash_errors_percent else 0.15
        avg_ram_error_pct = np.mean(ram_errors_percent) if ram_errors_percent else 0.12
        
        print(f"📊 Average errors: Flash {avg_flash_error_pct:.1%}, RAM {avg_ram_error_pct:.1%}")
        
        # Calculate upper and lower bounds for ALL points
        flash_upper = [est * (1 + avg_flash_error_pct) for est in log_flash_est]
        flash_lower = [est * (1 - avg_flash_error_pct) for est in log_flash_est]
        ram_upper = [est * (1 + avg_ram_error_pct) for est in log_ram_est]
        ram_lower = [est * (1 - avg_ram_error_pct) for est in log_ram_est]        # MAIN CURVES - elegant mint and blue from mint_blue diagram
        flash_color = '#A8E6CF'  # Soft mint green for Flash (from mint_blue)
        ram_color = '#5B9BD5'    # Professional blue for RAM (from mint_blue)
        ax2.plot(future_hidden_sizes, log_flash_est, 'o-', color=flash_color, alpha=0.9, 
                label='Flash Estimation', linewidth=3, markersize=10)
        ax2.plot(future_hidden_sizes, log_ram_est, 'o-', color=ram_color, alpha=0.9,
                label='RAM Estimation', linewidth=3, markersize=10)
        
        # FLASH: Shaded area + error bars
        ax2.fill_between(future_hidden_sizes, flash_lower, flash_upper, 
                        color=flash_color, alpha=0.2, label='Flash Uncertainty Range')
        
        # Flash error bars (additional to the area)
        flash_errors_absolute = [est * avg_flash_error_pct for est in log_flash_est]
        ax2.errorbar(future_hidden_sizes, log_flash_est, yerr=flash_errors_absolute,
                    fmt='none', color=flash_color, alpha=0.8, 
                    capsize=8, capthick=3, linewidth=2, label='Flash Error Bars')
        
        # RAM: ONLY error bars
        ram_errors_absolute = [est * avg_ram_error_pct for est in log_ram_est]
        ax2.errorbar(future_hidden_sizes, log_ram_est, yerr=ram_errors_absolute,
                    fmt='none', color=ram_color, alpha=0.8,
                    capsize=8, capthick=3, linewidth=2, label='RAM Error Bars')
        
        # UPPER AND LOWER BOUNDARY LINES only for Flash (dashed)
        ax2.plot(future_hidden_sizes, flash_upper, '--', color=flash_color, alpha=0.6, linewidth=2)
        ax2.plot(future_hidden_sizes, flash_lower, '--', color=flash_color, alpha=0.6, linewidth=2)
        
        # Real measurements as reference - use different colors for Flash vs RAM
        real_hs = list(real_measurements.keys())
        real_flash_vals = [real_measurements[hs]['flash_used_kb'] for hs in real_hs]
        real_ram_vals = [real_measurements[hs]['ram_used_kb'] for hs in real_hs]        # Use elegant colors from mint_blue diagram for real measurements
        real_flash_color = '#C8F0DB'  # Very light mint for real Flash (from mint_blue)
        real_ram_color = '#87CEEB'   # Sky blue for real RAM (from mint_blue)
        
        ax2.scatter(real_hs, real_flash_vals, color=real_flash_color, s=200, marker='^', 
                   label='Flash Real Measurements', zorder=5, edgecolors='black', linewidth=2)
        ax2.scatter(real_hs, real_ram_vals, color=real_ram_color, s=200, marker='^',
                   label='RAM Real Measurements', zorder=5, edgecolors='black', linewidth=2)
        
        # Arduino limits as horizontal lines - elegant colors from mint_blue diagram
        ax2.axhline(y=256, color='#A8E6CF', linestyle='--', alpha=0.8, linewidth=3, 
                   label='Arduino R4 Flash Limit (256 kB)')
        ax2.axhline(y=32, color='#5B9BD5', linestyle='--', alpha=0.8, linewidth=3,
                   label='Arduino R4 RAM Limit (32 kB)')
        
        # X-axis linear, Y-axis logarithmic
        ax2.set_xticks(future_hidden_sizes)
        ax2.set_xticklabels([f'{hs}' for hs in future_hidden_sizes], fontsize=13) # Increased
        ax2.set_yscale('log')
        ax2.set_xlabel('Hidden Size (Linear Scale)', fontsize=16) # Increased
        ax2.set_ylabel('Memory (kB, Log Scale)', fontsize=16) # Increased
        ax2.set_title('Logarithmic Scaling: Flash/RAM vs. Hidden Size\n(with continuous uncertainty ranges)', fontsize=20) # Increased, removed bold
        ax2.legend(loc='upper left', fontsize=12) # Increased
        ax2.grid(True, alpha=0.3)
        
        # ========== 3. Flash Components Pie Chart ==========
        if estimation_data:
            sample_estimation = estimation_data[0]
            flash_components = sample_estimation['flash_components']
            
            sizes = list(flash_components.values())
            labels = [label.replace('_', ' ').title() for label in flash_components.keys()]            # Use elegant colors from mint_blue diagram for Flash
            colors_pie = ['#A8E6CF', '#C8F0DB']  # Mint green palette from mint_blue
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.1f%%', 
                                             colors=colors_pie, startangle=90,
                                             textprops={'fontsize': 16}) # Increased label fontsize, removed bold
            ax3.set_title(f'Flash Components Breakdown\n({architectures[0]} Architecture)', 
                         fontsize=18) # Increased, removed bold
        
        # ========== 4. RAM Components Pie Chart ==========
        if estimation_data:
            ram_components = sample_estimation['ram_components']
            
            sizes = list(ram_components.values())
            labels = [label.replace('_', ' ').title() for label in ram_components.keys()]
              # Use elegant colors from mint_blue diagram for RAM
            colors_pie = ['#5B9BD5', '#87CEEB']  # Professional blue palette from mint_blue
            
            wedges, texts, autotexts = ax4.pie(sizes, labels=labels, autopct='%1.1f%%',
                                             colors=colors_pie, startangle=90,
                                             textprops={'fontsize': 16}) # Increased, removed bold
            ax4.set_title(f'RAM Components Breakdown\n({architectures[0]} Architecture)', 
                         fontsize=18) # Increased, removed bold
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = "arduino_memory_estimation_analysis.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"\n🔋 Arduino Memory Estimation analysis saved: {plot_filename}")
          # Show plot without blocking
        plt.show(block=False)
        return fig
    
    def create_ram_comparison_visualization(self):
        """A1: RAM-Vergleich: Stateful vs. Window-basiert Architektur - Integrated with mint_blue color scheme"""
        fig, ax = plt.subplots(figsize=(12, 7)) # Increased figure size for better label fitting
        
        # Data (from plotting_script.py)
        models = ['16×16 stateful', '32×32 stateful', '64×64 stateful', '32×32 window']
        ram_values = [7.7, 8.9, 9.8, 80.0]  # RAM in kB
        
        # Use harmonized colors with the beautiful red from A5 for 32x32 window
        colors = [
            self.color_scheme['stateful_16'],
            self.color_scheme['stateful_32'],
            self.color_scheme['stateful_64'],
            '#FF6B6B'  # Beautiful red from A5 for the 32x32 window
        ]
        
        # Arduino R4 has 32kB RAM
        arduino_ram = 32
        percentages = [(ram/arduino_ram)*100 for ram in ram_values]
        
        bars = ax.bar(models, ram_values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
          # Add percentage labels on bars
        for bar, pct, ram_val in zip(bars, percentages, ram_values):
            height = bar.get_height()
            # Display RAM value and percentage of Arduino RAM, correct newlines
            bar_label = f'{ram_val:.1f} kB'
            if ram_val <= arduino_ram: # Only show percentage if it fits
                 bar_label += f'\n({pct:.1f}% des\nArduino RAM)' # Fixed newlines
            
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5, bar_label,
                    ha='center', va='bottom', fontsize=16, color='black') # Increased fontsize, removed bold
        
        ax.set_ylabel('RAM Verbrauch (kB)', fontsize=20) # Increased fontsize, removed bold
        ax.set_title('RAM-Vergleich: Stateful vs. Window-basiert Architektur', fontsize=24) # Increased fontsize, removed bold
        ax.grid(True, alpha=0.3)
        
        # Add Arduino RAM limit line
        ax.axhline(y=arduino_ram, color='red', linestyle='--', alpha=0.7, 
                   label='Arduino R4 RAM Limit (32kB)', linewidth=2)
        ax.legend(fontsize=16) # Increased fontsize
        
        plt.xticks(rotation=45, ha="right", fontsize=18) # Increased fontsize and ha for better alignment
        plt.tight_layout()
        
        # Save plot
        plot_filename = "A1_RAM_Comparison_stateful_vs_window.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 A1 RAM comparison saved: {plot_filename}")
        
        # Show plot without blocking
        plt.show(block=False)
        return fig

    def create_window_pie_chart_visualization(self):
        """A4: Window-RAM als Kreisdiagramm - Corrected values with harmonized mint_blue colors"""
        fig, ax = plt.subplots(figsize=(10, 8))
          # Corrected window-based breakdown for 5000 seconds
        # Based on task requirement: Basis LSTM ~8.9 kB, Window Array ~71.1 kB (80-8.9), total 80 kB
        total_ram = 80.0  # kB
        basis_lstm = 8.9  # Basis LSTM model (corrected from 3.0)
        window_array = 71.1  # Window array (corrected from 75.0 to 80-8.9)
        
        sizes = [window_array, basis_lstm]
        labels = [f'Window Array\n({window_array:.1f} kB)', 
                  f'Basis LSTM\n({basis_lstm:.1f} kB)'] # Fixed newlines
          # Use the beautiful red from A5 for the large slice (Window Array) and harmonized mint_blue for LSTM
        colors = ['#FF6B6B', self.color_scheme['stateful_lstm']]  # Beautiful red for Window Array
        
        explode = (0.1, 0)  # explode the largest slice (window array)
        
        wedges, texts, autotexts = ax.pie(sizes, explode=explode, labels=labels, colors=colors,
                                         autopct='%1.1f%%', shadow=True, startangle=90,
                                         textprops={'fontsize': 16}) # Increased label fontsize, removed bold
        # Make percentage text larger and white, without bold
        for autotext in autotexts:
            autotext.set_fontsize(18) # Increased fontsize
            autotext.set_color('white')
        
        ax.set_title('RAM-Verteilung: Window-basiert Architektur (5000s Zeitfenster)\nKorrigierte Werte: Basis LSTM ~8.9 kB, Window Array ~71.1 kB', 
                     fontsize=22, pad=20) # Increased fontsize, removed bold
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = "A4_Window_RAM_Pie_Chart_corrected.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 A4 Window pie chart saved: {plot_filename}")
        
        # Show plot without blocking
        plt.show(block=False)
        return fig

    def create_mcu_overview_visualization(self):
        """A3: Mikrocontroller-Übersicht & Modell-Fit - Integrated with mint_blue color scheme"""
        fig, ax = plt.subplots(figsize=(16, 10)) # Changed to more rectangular shape (16:10 ratio)
        
        # MCU specifications (from plotting_script.py)
        mcu_specs = {
            'Arduino R4': {'RAM': 32, 'Flash': 256},
            'ESP32': {'RAM': 520, 'Flash': 4096},
            'STM32H7': {'RAM': 1024, 'Flash': 2048},
            'RP2040': {'RAM': 264, 'Flash': 2048},
            'Arduino Nano': {'RAM': 2, 'Flash': 32},
            'Arduino Uno': {'RAM': 2, 'Flash': 32},
            'STM32F4': {'RAM': 192, 'Flash': 1024}
        }
        
        # MCU families and harmonized colors from mint_blue scheme
        families = {
            'Arduino': ['Arduino R4', 'Arduino Nano', 'Arduino Uno'],
            'ESP': ['ESP32'],
            'STM32': ['STM32H7', 'STM32F4'],
            'RP': ['RP2040']
        }
        
        # Use harmonized mint_blue color palette
        family_colors = {
            'Arduino': self.color_scheme['stateful_32'],      # Medium mint green for Arduino
            'ESP': self.color_scheme['windows_lstm'],         # Sky blue for ESP
            'STM32': self.color_scheme['blue_accent'],        # Professional blue for STM32
            'RP': self.color_scheme['stateful_lstm']          # Soft mint for RP
        }
        
        # Plot MCUs
        for family, mcus_in_family in families.items():
            for mcu_name in mcus_in_family:
                if mcu_name in mcu_specs:
                    ax.scatter(mcu_specs[mcu_name]['RAM'], mcu_specs[mcu_name]['Flash'], 
                               s=200, label=f"{mcu_name} ({family})", 
                               color=family_colors[family], alpha=0.8, edgecolor='black')
                    ax.annotate(mcu_name, (mcu_specs[mcu_name]['RAM'], mcu_specs[mcu_name]['Flash']),
                                textcoords="offset points", xytext=(5,5), ha='left', fontsize=14) # Increased fontsize

        # Add requirement lines with harmonized colors
        max_stateful_ram = 9.8   # 64x64 stateful
        max_stateful_flash = 123.0  # 64x64 stateful
        window_ram = 80.0        # window-based
        
        # Use mint_blue colors for requirement lines
        ax.axhline(y=max_stateful_flash, color=self.color_scheme['stateful_32'], 
                   linestyle='-', alpha=0.8, label='Flash Requirement (64×64 stateful)', linewidth=3)
        ax.axvline(x=max_stateful_ram, color=self.color_scheme['stateful_32'], 
                   linestyle='-', alpha=0.8, label='RAM Requirement (64×64 stateful)', linewidth=3)
        ax.axvline(x=window_ram, color='#FF6B6B', linestyle='--', alpha=0.8, 
                   label='RAM Requirement (window-based, 80kB)', linewidth=3) # Clarified label
        ax.set_xlabel('RAM (kB) - Log Scale', fontsize=20) # Increased fontsize, removed bold
        ax.set_ylabel('Flash (kB) - Log Scale', fontsize=20) # Increased fontsize, removed bold
        ax.set_title('Microcontroller Overview: Memory Requirements vs. MCU Capabilities', 
                     fontsize=24) # Increased fontsize, removed bold        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(True, which="both", ls="-", alpha=0.3) # Grid for both major and minor ticks on log scale
        ax.legend(fontsize=16, loc='lower right') # Increased fontsize and moved legend to bottom right
        
        plt.tight_layout() # Use standard tight layout without space reservation
        
        # Save plot
        plot_filename = "A3_MCU_Overview_Memory_Requirements.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 A3 MCU overview saved: {plot_filename}")
        
        # Show plot without blocking
        plt.show(block=False)
        return fig

    def create_ram_headroom_heatmap_visualization(self):
        """A5: RAM-Headroom-Ampel - Integrated with mint_blue color scheme"""
        fig, ax = plt.subplots(figsize=(14, 10)) # Increased figure size
        
        # MCU specifications (from plotting_script.py)
        mcu_specs = {
            'Arduino Uno': {'RAM': 2, 'Flash': 32},
            'Arduino Nano': {'RAM': 2, 'Flash': 32},
            'Arduino R4': {'RAM': 32, 'Flash': 256},
            'RP2040': {'RAM': 264, 'Flash': 2048},
            'STM32F4': {'RAM': 192, 'Flash': 1024},
            'ESP32': {'RAM': 520, 'Flash': 4096},
            'STM32H7': {'RAM': 1024, 'Flash': 2048}
        }
        
        # MCUs to analyze
        mcus = ['Arduino Uno', 'Arduino Nano', 'Arduino R4', 'RP2040', 'STM32F4', 'ESP32', 'STM32H7']
        models = ['16×16 stateful', '32×32 stateful', '64×64 stateful', '32×32 window']
        model_ram = [7.7, 8.9, 9.8, 80.0]
          # Calculate headroom matrix
        headroom_matrix = []
        for mcu in mcus:
            mcu_ram = mcu_specs[mcu]['RAM']
            headroom_row = []
            for ram_need in model_ram:
                if ram_need > mcu_ram:
                    headroom = -100  # Impossible
                else:
                    headroom = ((mcu_ram - ram_need) / mcu_ram) * 100
                headroom_row.append(headroom)
            headroom_matrix.append(headroom_row)
        
        headroom_matrix = np.array(headroom_matrix)        # Create harmonious colormap: Red (bad) -> Blue -> Green (good)
        from matplotlib.colors import LinearSegmentedColormap
        
        colors_custom = [
            '#FF6B6B',                                  # Red (from A2 plotting_script, for <0%, "Impossible")
            '#FFB3B3',                                  # Soft Red/Pink (for Low 0-15%) - Transition
            self.color_scheme['windows_secondary'],     # Powder Blue for Okay (15-35%)
            self.color_scheme['windows_lstm'],          # Sky Blue for Good (35-55%)
            self.color_scheme['stateful_secondary'],    # Light Mint for Very Good (55-70%)
            self.color_scheme['stateful_lstm'],         # Soft Mint for Excellent (70-85%)
            '#96CEB4',                                  # Greenish (from A2 plotting_script, for Ample 85-95%)
            '#96CEB4'                                   # Greenish (for Abundant 95-100%)
        ]
        
        # Create the custom colormap
        custom_cmap = LinearSegmentedColormap.from_list(
            'custom_gradient_red_to_green', colors_custom, N=256
        )
        
        # Create heatmap with custom color gradient
        im = ax.imshow(headroom_matrix, cmap=custom_cmap, aspect='auto', vmin=-50, vmax=100)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(mcus)))        # Set ticks and labels
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(mcus)))
        ax.set_xticklabels(models, rotation=45, ha='right', fontsize=16) # Increased fontsize further
        ax.set_yticklabels(mcus, fontsize=16) # Increased fontsize further
        
        # Add text annotations with enhanced styling
        for i in range(len(mcus)):
            for j in range(len(models)):
                value = headroom_matrix[i, j]
                text_color = 'black' # Default to black
                if value < 0: # Impossible
                    text_content = "IMPOSSIBLE"
                else:
                    text_content = f'{value:.0f}%\nfree'
                
                ax.text(j, i, text_content, ha='center', va='center', 
                        fontsize=16, color=text_color) # Increased fontsize, removed bold
        
        ax.set_title('RAM Headroom Matrix: Available Memory by MCU and Model', 
                     fontsize=22, pad=20) # Increased fontsize, removed bold
        ax.set_xlabel('Model Architecture', fontsize=18) # Increased fontsize, removed bold
        ax.set_ylabel('Microcontroller', fontsize=18) # Increased fontsize, removed bold
        
        # Add colorbar with enhanced styling
        cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
        cbar.set_label('Free RAM (%)', rotation=270, labelpad=25, fontsize=16) # Increased fontsize, removed bold
        cbar.ax.tick_params(labelsize=14) # Increased cbar tick fontsize
        
        # Remove the interpretation text legend that was in the way
        plt.tight_layout() # Simple tight layout without text interference
        
        # Save plot
        plot_filename = "A5_RAM_Headroom_Heatmap.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 A5 RAM headroom heatmap saved: {plot_filename}")
        
        # Show plot without blocking
        plt.show(block=False)
        return fig

    def run_complete_analysis(self):
        """Runs the complete analysis"""
        print("🚀 Starting comprehensive model analysis...")
        
        # Find all model directories
        model_dirs = self.find_model_directories()
        
        if not model_dirs:
            print("❌ No model directories found!")
            return {}
        
        print(f"📁 Found: {len(model_dirs)} model directories")
        
        # Analyze each directory
        for arch_info in model_dirs:
            result = self.analyze_architecture_directory(arch_info)
            self.results[arch_info['architecture']] = result        # Create visualizations
        print(f"\n📊 Creating visualizations...")
        
        self.create_comprehensive_visualization()
        self.create_arduino_memory_estimation_visualization()
        
        # Create integrated A1, A3, A4, and A5 diagrams
        print(f"\n📊 Creating integrated A1, A3, A4, and A5 diagrams...")
        self.create_ram_comparison_visualization()
        self.create_mcu_overview_visualization()
        self.create_window_pie_chart_visualization()
        self.create_ram_headroom_heatmap_visualization()
        
        print(f"\n✅ Analysis completed!")
        print(f"📊 All plots are displayed and saved. Close plot windows to continue.")
        return self.results

def main():
    """Main function"""
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup"
    
    analyzer = ModelArchitectureAnalyzer(base_path)
    results = analyzer.run_complete_analysis()
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETED")
    print(f"{'='*80}")
    print(f"✅ {len(results)} architectures analyzed")
    print("📊 Visualizations created")

if __name__ == "__main__":
    main()
