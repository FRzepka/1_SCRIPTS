"""
Arduino Hardware Requirements Calculator
========================================

Berechnet theoretische Hardware-Anforderungen (Flash, RAM, CPU) für Arduino LSTM Deployment
basierend auf dem PyTorch Modell und konvertiert diese in C/Arduino spezifische Werte.

Features:
- Flash Memory Berechnung (Modellgröße + C-Code Overhead)
- RAM Memory Berechnung (LSTM States + Arbeitspeicher)
- CPU Performance Schätzung (Inference Time)
- Arduino Board Kompatibilitäts-Check
- Optimierungsvorschläge
"""

import torch
import torch.nn as nn
import os
import json
import numpy as np
from pathlib import Path
import struct

class ArduinoHardwareCalculator:
    def __init__(self, model_path=None):
        """
        Initialize hardware calculator
        
        Args:
            model_path: Path to PyTorch model (.pth file)
        """
        self.model_path = model_path
        self.model = None
        self.model_info = {}
        
        # Arduino board specifications
        self.arduino_boards = {
            'uno': {
                'flash_kb': 32,
                'ram_bytes': 2048,
                'eeprom_bytes': 1024,
                'cpu_mhz': 16,
                'architecture': 'AVR',
                'float_size': 4
            },
            'nano': {
                'flash_kb': 32,
                'ram_bytes': 2048,
                'eeprom_bytes': 1024,
                'cpu_mhz': 16,
                'architecture': 'AVR',
                'float_size': 4
            },
            'leonardo': {
                'flash_kb': 32,
                'ram_bytes': 2560,
                'eeprom_bytes': 1024,
                'cpu_mhz': 16,
                'architecture': 'AVR',
                'float_size': 4
            },
            'mega2560': {
                'flash_kb': 256,
                'ram_bytes': 8192,
                'eeprom_bytes': 4096,
                'cpu_mhz': 16,
                'architecture': 'AVR',
                'float_size': 4
            },
            'due': {
                'flash_kb': 512,
                'ram_bytes': 98304,  # 96KB
                'eeprom_bytes': 0,
                'cpu_mhz': 84,
                'architecture': 'ARM',
                'float_size': 4
            },
            'esp32': {
                'flash_kb': 4096,  # 4MB (varies)
                'ram_bytes': 327680,  # 320KB
                'eeprom_bytes': 0,
                'cpu_mhz': 240,
                'architecture': 'ESP32',
                'float_size': 4
            },
            'teensy40': {
                'flash_kb': 2048,
                'ram_bytes': 1048576,  # 1MB
                'eeprom_bytes': 1080,
                'cpu_mhz': 600,
                'architecture': 'ARM',
                'float_size': 4
            }
        }
        
        # C-Code overhead factors
        self.c_overhead_factors = {
            'bootloader_kb': 0.5,  # Bootloader overhead
            'arduino_core_kb': 4.0,  # Arduino core libraries
            'lstm_code_kb': 8.0,  # LSTM implementation code
            'math_lib_kb': 2.0,  # Math library overhead
            'serial_comm_kb': 1.0,  # Serial communication code
            'monitoring_kb': 2.0,  # Hardware monitoring code
            'safety_margin': 1.2  # 20% safety margin
        }
        
        # RAM overhead factors  
        self.ram_overhead_factors = {
            'stack_bytes': 512,  # Stack space
            'heap_bytes': 256,  # Heap space
            'arduino_runtime_bytes': 512,  # Arduino runtime
            'serial_buffer_bytes': 128,  # Serial buffers
            'temp_variables_bytes': 256,  # Temporary calculation variables
            'safety_margin': 1.3  # 30% safety margin
        }

    def load_model(self, model_path=None):
        """Load PyTorch model and extract information"""
        if model_path:
            self.model_path = model_path
            
        if not self.model_path or not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
        print(f"Loading model: {self.model_path}")
          # Load model
        try:
            # First try with weights_only=False for full model
            self.model = torch.load(self.model_path, map_location='cpu', weights_only=False)
            if hasattr(self.model, 'state_dict'):
                self.model = self.model.state_dict()
        except Exception as e:
            print(f"Error loading full model: {e}")
            try:
                # Try loading as state dict directly with weights_only=True
                self.model = torch.load(self.model_path, map_location='cpu', weights_only=True)
            except Exception as e2:
                print(f"Error loading weights only: {e2}")
                try:
                    # Last attempt: load without weights_only restriction
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore", category=FutureWarning)
                        self.model = torch.load(self.model_path, map_location='cpu')
                        if hasattr(self.model, 'state_dict'):
                            self.model = self.model.state_dict()
                except Exception as e3:
                    raise Exception(f"Could not load model with any method. Final error: {e3}")
            
        self._analyze_model()
        
    def _analyze_model(self):
        """Analyze model architecture and parameters"""
        if self.model is None:
            raise ValueError("Model not loaded")
            
        total_params = 0
        layer_info = {}
        
        print("\n=== Model Architecture Analysis ===")
        
        for name, param in self.model.items():
            if isinstance(param, torch.Tensor):
                param_count = param.numel()
                total_params += param_count
                param_size_bytes = param_count * 4  # float32 = 4 bytes
                
                layer_info[name] = {
                    'shape': list(param.shape),
                    'param_count': param_count,
                    'size_bytes': param_size_bytes,
                    'size_kb': param_size_bytes / 1024
                }
                
                print(f"{name:40s} | Shape: {str(param.shape):20s} | Params: {param_count:8d} | Size: {param_size_bytes:8d} bytes ({param_size_bytes/1024:.2f} KB)")
        
        # Extract LSTM architecture info
        self.model_info = {
            'total_parameters': total_params,
            'total_size_bytes': total_params * 4,
            'total_size_kb': total_params * 4 / 1024,
            'layer_info': layer_info
        }
        
        # Try to detect LSTM architecture
        self._detect_lstm_architecture()
        
        print(f"\nTotal Parameters: {total_params:,}")
        print(f"Total Model Size: {self.model_info['total_size_bytes']:,} bytes ({self.model_info['total_size_kb']:.2f} KB)")
        
    def _detect_lstm_architecture(self):
        """Detect LSTM architecture from model parameters"""
        lstm_layers = 0
        hidden_size = 0
        input_size = 0
        
        # Look for LSTM layer patterns
        for name, info in self.model_info['layer_info'].items():
            if 'lstm' in name.lower():
                if 'weight_ih_l0' in name:
                    # Input-hidden weight matrix: (4*hidden_size, input_size)
                    shape = info['shape']
                    if len(shape) == 2:
                        hidden_size = shape[0] // 4
                        input_size = shape[1]
                        
                if 'weight_ih_l' in name:
                    layer_num = int(name.split('_l')[1].split('.')[0])
                    lstm_layers = max(lstm_layers, layer_num + 1)
        
        self.model_info.update({
            'lstm_layers': lstm_layers,
            'hidden_size': hidden_size,
            'input_size': input_size
        })
        
        print(f"\nDetected LSTM Architecture:")
        print(f"  Input Size: {input_size}")
        print(f"  Hidden Size: {hidden_size}")
        print(f"  Number of Layers: {lstm_layers}")

    def calculate_flash_requirements(self):
        """Calculate Flash memory requirements"""
        if not self.model_info:
            raise ValueError("Model not analyzed yet")
            
        print("\n=== Flash Memory Requirements ===")
        
        # Base model size
        model_size_kb = self.model_info['total_size_kb']
        print(f"Base Model Size: {model_size_kb:.2f} KB")
        
        # C-Code overhead
        overhead_kb = 0
        print("\nC-Code Overhead:")
        for component, size_kb in self.c_overhead_factors.items():
            if component != 'safety_margin':
                overhead_kb += size_kb
                print(f"  {component:20s}: {size_kb:6.1f} KB")
        
        # Total before safety margin
        total_before_margin = model_size_kb + overhead_kb
        print(f"\nSubtotal: {total_before_margin:.2f} KB")
        
        # Apply safety margin
        safety_margin = self.c_overhead_factors['safety_margin']
        total_flash_kb = total_before_margin * safety_margin
        
        print(f"Safety Margin ({safety_margin}x): {total_flash_kb:.2f} KB")
        
        flash_requirements = {
            'model_size_kb': model_size_kb,
            'overhead_kb': overhead_kb,
            'safety_margin_factor': safety_margin,
            'total_flash_kb': total_flash_kb,
            'total_flash_bytes': total_flash_kb * 1024
        }
        
        return flash_requirements

    def calculate_ram_requirements(self):
        """Calculate RAM requirements"""
        if not self.model_info:
            raise ValueError("Model not analyzed yet")
            
        print("\n=== RAM Memory Requirements ===")
        
        # LSTM State memory
        lstm_layers = self.model_info.get('lstm_layers', 2)
        hidden_size = self.model_info.get('hidden_size', 128)
        
        # Each LSTM layer needs: hidden_state + cell_state = 2 * hidden_size * 4 bytes
        lstm_state_bytes = lstm_layers * 2 * hidden_size * 4
        print(f"LSTM States ({lstm_layers} layers): {lstm_state_bytes} bytes ({lstm_state_bytes/1024:.2f} KB)")
        
        # Working memory for computations
        # Need space for intermediate calculations (gates, activations, etc.)
        working_memory_bytes = hidden_size * 4 * 8  # Rough estimate for intermediate values
        print(f"Working Memory: {working_memory_bytes} bytes ({working_memory_bytes/1024:.2f} KB)")
        
        # Input/Output buffers
        input_size = self.model_info.get('input_size', 4)
        io_buffer_bytes = (input_size + 1) * 4 * 2  # Input + output, double buffered
        print(f"I/O Buffers: {io_buffer_bytes} bytes")
        
        # System overhead
        overhead_bytes = 0
        print("\nSystem Overhead:")
        for component, size_bytes in self.ram_overhead_factors.items():
            if component != 'safety_margin':
                overhead_bytes += size_bytes
                print(f"  {component:25s}: {size_bytes:6d} bytes")
        
        # Total before safety margin
        total_before_margin = lstm_state_bytes + working_memory_bytes + io_buffer_bytes + overhead_bytes
        print(f"\nSubtotal: {total_before_margin} bytes ({total_before_margin/1024:.2f} KB)")
        
        # Apply safety margin
        safety_margin = self.ram_overhead_factors['safety_margin']
        total_ram_bytes = int(total_before_margin * safety_margin)
        
        print(f"Safety Margin ({safety_margin}x): {total_ram_bytes} bytes ({total_ram_bytes/1024:.2f} KB)")
        
        ram_requirements = {
            'lstm_state_bytes': lstm_state_bytes,
            'working_memory_bytes': working_memory_bytes,
            'io_buffer_bytes': io_buffer_bytes,
            'overhead_bytes': overhead_bytes,
            'safety_margin_factor': safety_margin,
            'total_ram_bytes': total_ram_bytes,
            'total_ram_kb': total_ram_bytes / 1024
        }
        
        return ram_requirements

    def estimate_performance(self):
        """Estimate CPU performance requirements"""
        if not self.model_info:
            raise ValueError("Model not analyzed yet")
            
        print("\n=== Performance Estimation ===")
        
        hidden_size = self.model_info.get('hidden_size', 128)
        lstm_layers = self.model_info.get('lstm_layers', 2)
        input_size = self.model_info.get('input_size', 4)
        
        # Estimate operations per inference
        # LSTM cell: ~8 * hidden_size^2 + 4 * hidden_size * input_size operations per layer
        ops_per_layer = 8 * hidden_size * hidden_size + 4 * hidden_size * input_size
        total_lstm_ops = ops_per_layer * lstm_layers
        
        # FC layers operations (estimate)
        fc_ops = 0
        for name, info in self.model_info['layer_info'].items():
            if 'fc' in name.lower() or 'linear' in name.lower():
                if 'weight' in name:
                    fc_ops += info['param_count']
        
        total_ops = total_lstm_ops + fc_ops
        
        print(f"LSTM Operations: {total_lstm_ops:,}")
        print(f"FC Operations: {fc_ops:,}")
        print(f"Total Operations: {total_ops:,}")
        
        # Estimate timing for different Arduino boards
        performance_estimates = {}
        
        for board_name, board_specs in self.arduino_boards.items():
            # Very rough estimate: operations per second = cpu_mhz * 1M * efficiency_factor
            # Efficiency factor depends on architecture and instruction complexity
            if board_specs['architecture'] == 'AVR':
                efficiency_factor = 0.1  # AVR is slower for float operations
            elif board_specs['architecture'] == 'ARM':
                efficiency_factor = 0.5  # ARM has better float performance
            else:  # ESP32
                efficiency_factor = 0.3
                
            ops_per_second = board_specs['cpu_mhz'] * 1000000 * efficiency_factor
            inference_time_us = (total_ops / ops_per_second) * 1000000
            
            performance_estimates[board_name] = {
                'inference_time_us': inference_time_us,
                'inference_time_ms': inference_time_us / 1000,
                'inferences_per_second': 1000000 / inference_time_us if inference_time_us > 0 else 0
            }
            
            print(f"{board_name:10s}: {inference_time_us:8.0f} µs ({inference_time_us/1000:6.1f} ms) - {1000000/inference_time_us:5.1f} inf/sec")
        
        return performance_estimates

    def check_board_compatibility(self, flash_req, ram_req):
        """Check which Arduino boards can support the model"""
        print("\n=== Board Compatibility Analysis ===")
        
        compatible_boards = {}
        
        for board_name, board_specs in self.arduino_boards.items():
            flash_available_kb = board_specs['flash_kb']
            ram_available_bytes = board_specs['ram_bytes']
            
            flash_ok = flash_req['total_flash_kb'] <= flash_available_kb
            ram_ok = ram_req['total_ram_bytes'] <= ram_available_bytes
            
            flash_usage_percent = (flash_req['total_flash_kb'] / flash_available_kb) * 100
            ram_usage_percent = (ram_req['total_ram_bytes'] / ram_available_bytes) * 100
            
            compatible = flash_ok and ram_ok
            
            status = "✓ COMPATIBLE" if compatible else "✗ INCOMPATIBLE"
            print(f"{board_name:10s} | Flash: {flash_usage_percent:5.1f}% | RAM: {ram_usage_percent:5.1f}% | {status}")
            
            compatible_boards[board_name] = {
                'compatible': compatible,
                'flash_usage_percent': flash_usage_percent,
                'ram_usage_percent': ram_usage_percent,
                'flash_ok': flash_ok,
                'ram_ok': ram_ok,
                'board_specs': board_specs
            }
        
        return compatible_boards

    def generate_optimization_suggestions(self, flash_req, ram_req, compatibility):
        """Generate optimization suggestions for incompatible boards"""
        print("\n=== Optimization Suggestions ===")
        
        suggestions = {}
        
        for board_name, board_info in compatibility.items():
            if not board_info['compatible']:
                board_suggestions = []
                
                if not board_info['flash_ok']:
                    flash_reduction_needed = flash_req['total_flash_kb'] - board_info['board_specs']['flash_kb']
                    board_suggestions.append(f"Reduce Flash usage by {flash_reduction_needed:.1f} KB:")
                    board_suggestions.append("  - Use quantization (int8 instead of float32)")
                    board_suggestions.append("  - Remove unnecessary monitoring features")
                    board_suggestions.append("  - Optimize LSTM implementation")
                    
                if not board_info['ram_ok']:
                    ram_reduction_needed = ram_req['total_ram_bytes'] - board_info['board_specs']['ram_bytes']
                    board_suggestions.append(f"Reduce RAM usage by {ram_reduction_needed} bytes:")
                    board_suggestions.append("  - Reduce hidden size")
                    board_suggestions.append("  - Use fewer LSTM layers")
                    board_suggestions.append("  - Optimize memory allocation")
                
                suggestions[board_name] = board_suggestions
                
                print(f"\n{board_name.upper()}:")
                for suggestion in board_suggestions:
                    print(f"  {suggestion}")
        
        return suggestions

    def save_report(self, output_path="arduino_hardware_analysis.json"):
        """Save complete analysis report"""
        if not self.model_info:
            raise ValueError("No analysis data to save")
            
        flash_req = self.calculate_flash_requirements()
        ram_req = self.calculate_ram_requirements()
        performance = self.estimate_performance()
        compatibility = self.check_board_compatibility(flash_req, ram_req)
        suggestions = self.generate_optimization_suggestions(flash_req, ram_req, compatibility)
        
        report = {
            'model_info': self.model_info,
            'flash_requirements': flash_req,
            'ram_requirements': ram_req,
            'performance_estimates': performance,
            'board_compatibility': compatibility,
            'optimization_suggestions': suggestions,
            'arduino_boards': self.arduino_boards
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
            
        print(f"\n=== Report saved to: {output_path} ===")
        return report

    def run_complete_analysis(self, model_path=None, save_report=True):
        """Run complete hardware analysis"""
        print("=" * 80)
        print("ARDUINO HARDWARE REQUIREMENTS ANALYSIS")
        print("=" * 80)
        
        # Load and analyze model
        self.load_model(model_path)
        
        # Calculate requirements
        flash_req = self.calculate_flash_requirements()
        ram_req = self.calculate_ram_requirements()
        performance = self.estimate_performance()
        compatibility = self.check_board_compatibility(flash_req, ram_req)
        suggestions = self.generate_optimization_suggestions(flash_req, ram_req, compatibility)
        
        # Save report
        if save_report:
            report = self.save_report()
            return report
        
        return {
            'flash_requirements': flash_req,
            'ram_requirements': ram_req,
            'performance_estimates': performance,
            'board_compatibility': compatibility,
            'optimization_suggestions': suggestions
        }


def main():
    """Main function for command line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Arduino Hardware Requirements Calculator')
    parser.add_argument('--model', '-m', type=str, help='Path to PyTorch model file (.pth)')
    parser.add_argument('--output', '-o', type=str, default='arduino_hardware_analysis.json',
                       help='Output file for analysis report')
    
    args = parser.parse_args()
    
    # Default model paths to try
    default_model_paths = [
        "best_model_optimized.pth",
        "best_model.pth",
        "../best_model_optimized.pth",
        "../best_model.pth",
        "../../best_model_optimized.pth",
        "../../best_model.pth"
    ]
    
    model_path = args.model
    if not model_path:
        # Try to find model automatically
        for path in default_model_paths:
            if os.path.exists(path):
                model_path = path
                break
    
    if not model_path:
        print("No model file specified and no default model found.")
        print("Please specify model path with --model option")
        print(f"Searched paths: {default_model_paths}")
        return
    
    # Run analysis
    calculator = ArduinoHardwareCalculator()
    try:
        report = calculator.run_complete_analysis(model_path, save_report=True)
        print(f"\nAnalysis complete! Report saved to: {args.output}")
    except Exception as e:
        print(f"Error during analysis: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
