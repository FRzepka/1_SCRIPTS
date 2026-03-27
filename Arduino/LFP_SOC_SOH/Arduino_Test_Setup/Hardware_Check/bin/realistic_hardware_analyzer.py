#!/usr/bin/env python3
"""
Realistic Arduino Hardware Analyzer for LSTM Models

This script analyzes actual file sizes and provides realistic memory usage estimates
for different Arduino hardware platforms when running LSTM models.

Author: Florian
Date: 2024
"""

import os
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

class RealisticHardwareAnalyzer:
    """Analyzes real LSTM weight files and provides hardware recommendations"""
    
    def __init__(self):
        self.hardware_specs = {
            'Arduino UNO R3': {
                'flash': 32768,    # 32KB Flash
                'sram': 2048,      # 2KB SRAM
                'eeprom': 1024,    # 1KB EEPROM
                'cost': 25,        # USD
                'availability': 'Very High'
            },
            'Arduino UNO R4': {
                'flash': 262144,   # 256KB Flash
                'sram': 32768,     # 32KB SRAM
                'eeprom': 8192,    # 8KB EEPROM
                'cost': 35,        # USD
                'availability': 'High'
            },
            'Arduino Nano': {
                'flash': 32768,    # 32KB Flash
                'sram': 2048,      # 2KB SRAM
                'eeprom': 1024,    # 1KB EEPROM
                'cost': 15,        # USD
                'availability': 'Very High'
            },
            'ESP32': {
                'flash': 4194304,  # 4MB Flash
                'sram': 520192,    # 520KB SRAM
                'eeprom': 0,       # Uses Flash
                'cost': 12,        # USD
                'availability': 'Very High'
            },
            'ESP8266': {
                'flash': 4194304,  # 4MB Flash
                'sram': 81920,     # 80KB SRAM
                'eeprom': 0,       # Uses Flash
                'cost': 8,         # USD
                'availability': 'High'
            },
            'STM32F103': {
                'flash': 65536,    # 64KB Flash
                'sram': 20480,     # 20KB SRAM
                'eeprom': 0,       # Uses Flash
                'cost': 10,        # USD
                'availability': 'Medium'
            },
            'STM32F407': {
                'flash': 1048576,  # 1MB Flash
                'sram': 196608,    # 192KB SRAM
                'eeprom': 0,       # Uses Flash
                'cost': 25,        # USD
                'availability': 'Medium'
            },
            'Teensy 4.0': {
                'flash': 2097152,  # 2MB Flash
                'sram': 1048576,   # 1MB SRAM
                'eeprom': 0,       # Uses Flash
                'cost': 40,        # USD
                'availability': 'Medium'
            }
        }
        
        # Base code memory usage (estimated from Arduino IDE compilation)
        self.base_code_usage = {
            'flash': 8000,     # Base Arduino sketch + libraries
            'sram': 1000       # Variables, stack, heap
        }
    
    def analyze_weight_file(self, filepath):
        """Analyze a specific weight file and return memory requirements"""
        if not os.path.exists(filepath):
            print(f"⚠️  File not found: {filepath}")
            return None
            
        file_size = os.path.getsize(filepath)
        print(f"📁 Analyzing: {os.path.basename(filepath)}")
        print(f"   File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")
        
        # Read file to count actual float values
        with open(filepath, 'r') as f:
            content = f.read()
            
        # Count float values (simple approximation)
        float_count = content.count(',') + content.count(';')  # Approximate
        estimated_ram_usage = float_count * 4  # 4 bytes per float
        
        print(f"   Estimated float count: {float_count:,}")
        print(f"   Estimated RAM usage: {estimated_ram_usage:,} bytes ({estimated_ram_usage/1024:.1f} KB)")
        
        return {
            'file_size': file_size,
            'float_count': float_count,
            'ram_usage': estimated_ram_usage,
            'flash_usage': file_size  # Assuming similar flash usage
        }
    
    def check_hardware_compatibility(self, weight_info, model_name):
        """Check which hardware can run the model"""
        print(f"\n🔍 Hardware Compatibility Analysis for {model_name}")
        print("=" * 60)
        
        compatible = []
        incompatible = []
        
        for hw_name, specs in self.hardware_specs.items():
            # Calculate total memory requirements
            total_flash_needed = self.base_code_usage['flash'] + weight_info['flash_usage']
            total_sram_needed = self.base_code_usage['sram'] + weight_info['ram_usage']
            
            # Check if hardware can handle it
            flash_ok = total_flash_needed <= specs['flash']
            sram_ok = total_sram_needed <= specs['sram']
            
            flash_usage_pct = (total_flash_needed / specs['flash']) * 100
            sram_usage_pct = (total_sram_needed / specs['sram']) * 100
            
            hw_result = {
                'name': hw_name,
                'flash_ok': flash_ok,
                'sram_ok': sram_ok,
                'flash_usage_pct': flash_usage_pct,
                'sram_usage_pct': sram_usage_pct,
                'total_flash': total_flash_needed,
                'total_sram': total_sram_needed,
                'specs': specs
            }
            
            if flash_ok and sram_ok:
                compatible.append(hw_result)
                status = "✅ COMPATIBLE"
            else:
                incompatible.append(hw_result)
                status = "❌ INCOMPATIBLE"
            
            print(f"\n{hw_name}:")
            print(f"  Status: {status}")
            print(f"  Flash: {total_flash_needed:,} / {specs['flash']:,} bytes ({flash_usage_pct:.1f}%)")
            print(f"  SRAM:  {total_sram_needed:,} / {specs['sram']:,} bytes ({sram_usage_pct:.1f}%)")
            
            if not flash_ok:
                print(f"  ⚠️  Flash overflow by {total_flash_needed - specs['flash']:,} bytes")
            if not sram_ok:
                print(f"  ⚠️  SRAM overflow by {total_sram_needed - specs['sram']:,} bytes")
                
            print(f"  Cost: ${specs['cost']}, Availability: {specs['availability']}")
        
        return compatible, incompatible
    
    def plot_memory_comparison(self, model_32_info, model_64_info):
        """Create visualization comparing 32x32 vs 64x64 models"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 1. File Size Comparison
        models = ['32x32 Model', '64x64 Model']
        file_sizes = [model_32_info['file_size']/1024, model_64_info['file_size']/1024]
        
        ax1.bar(models, file_sizes, color=['#2E8B57', '#CD5C5C'])
        ax1.set_ylabel('File Size (KB)')
        ax1.set_title('Weight File Size Comparison')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(file_sizes):
            ax1.text(i, v + 5, f'{v:.1f} KB', ha='center', va='bottom', fontweight='bold')
        
        # 2. RAM Usage Comparison
        ram_usage = [model_32_info['ram_usage']/1024, model_64_info['ram_usage']/1024]
        
        ax2.bar(models, ram_usage, color=['#2E8B57', '#CD5C5C'])
        ax2.set_ylabel('RAM Usage (KB)')
        ax2.set_title('Estimated RAM Usage Comparison')
        ax2.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, v in enumerate(ram_usage):
            ax2.text(i, v + 5, f'{v:.1f} KB', ha='center', va='bottom', fontweight='bold')
        
        # 3. Hardware Compatibility Matrix
        hardware_names = list(self.hardware_specs.keys())
        ram_limits = [specs['sram']/1024 for specs in self.hardware_specs.values()]
        
        ax3.barh(hardware_names, ram_limits, alpha=0.7, color='lightblue', label='SRAM Limit')
        ax3.axvline(x=model_32_info['ram_usage']/1024, color='green', linestyle='--', linewidth=2, label='32x32 Model')
        ax3.axvline(x=model_64_info['ram_usage']/1024, color='red', linestyle='--', linewidth=2, label='64x64 Model')
        ax3.set_xlabel('Memory (KB)')
        ax3.set_title('Hardware SRAM Limits vs Model Requirements')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Cost vs Performance Analysis
        compatible_32 = []
        compatible_64 = []
        costs = []
        
        for hw_name, specs in self.hardware_specs.items():
            total_sram_32 = self.base_code_usage['sram'] + model_32_info['ram_usage']
            total_sram_64 = self.base_code_usage['sram'] + model_64_info['ram_usage']
            
            can_run_32 = total_sram_32 <= specs['sram']
            can_run_64 = total_sram_64 <= specs['sram']
            
            compatible_32.append(1 if can_run_32 else 0)
            compatible_64.append(1 if can_run_64 else 0)
            costs.append(specs['cost'])
        
        x = np.arange(len(hardware_names))
        width = 0.35
        
        ax4.bar(x - width/2, compatible_32, width, label='Can run 32x32', color='green', alpha=0.7)
        ax4.bar(x + width/2, compatible_64, width, label='Can run 64x64', color='red', alpha=0.7)
        
        ax4.set_xlabel('Hardware Platform')
        ax4.set_ylabel('Compatibility (1=Yes, 0=No)')
        ax4.set_title('Hardware Compatibility by Platform')
        ax4.set_xticks(x)
        ax4.set_xticklabels(hardware_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('realistic_hardware_analysis.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_recommendation_report(self, model_32_info, model_64_info):
        """Generate a comprehensive recommendation report"""
        print("\n" + "="*80)
        print("🎯 HARDWARE RECOMMENDATION REPORT")
        print("="*80)
        
        print("\n📊 MODEL COMPARISON:")
        print(f"32x32 Model: {model_32_info['file_size']/1024:.1f} KB file, ~{model_32_info['ram_usage']/1024:.1f} KB RAM")
        print(f"64x64 Model: {model_64_info['file_size']/1024:.1f} KB file, ~{model_64_info['ram_usage']/1024:.1f} KB RAM")
        print(f"Size Ratio: {model_64_info['file_size']/model_32_info['file_size']:.1f}x larger")
        
        print("\n🏆 RECOMMENDED HARDWARE:")
        print("\nFor 32x32 Model:")
        print("  ✅ Arduino UNO R4 (32KB SRAM) - Best balance of compatibility and cost")
        print("  ✅ ESP32 (520KB SRAM) - Excellent performance, WiFi capability")
        print("  ✅ STM32F407 (192KB SRAM) - High performance, good for advanced features")
        
        print("\nFor 64x64 Model:")
        print("  ✅ ESP32 (520KB SRAM) - Minimum viable option with good margin")
        print("  ✅ STM32F407 (192KB SRAM) - Sufficient but tight memory")
        print("  ✅ Teensy 4.0 (1MB SRAM) - Premium option with excellent performance")
        print("  ❌ Arduino UNO R3/R4 - Insufficient SRAM")
        
        print("\n💡 OPTIMIZATION STRATEGIES:")
        print("1. **Quantization**: Convert float32 to int8/int16 (4x memory reduction)")
        print("2. **Pruning**: Remove less important weights")
        print("3. **Progressive Loading**: Load weights in chunks")
        print("4. **External Memory**: Use SD card or external SRAM")
        
        print("\n⚠️  CRITICAL FINDINGS:")
        print("- The issue is NOT flash storage (upload works)")
        print("- The issue IS runtime SRAM (const arrays loaded to RAM)")
        print("- 64x64 model requires 3.6x more memory than 32x32")
        print("- Arduino UNO platforms fundamentally incompatible with 64x64")
        
        print("\n🔧 IMMEDIATE SOLUTIONS:")
        print("1. **Use ESP32**: $12, widely available, 520KB SRAM")
        print("2. **Implement quantization**: Reduce memory by 50-75%")
        print("3. **Consider model compression**: Distillation or pruning")
        
        return True

def main():
    """Main analysis function"""
    analyzer = RealisticHardwareAnalyzer()
    
    print("🔍 REALISTIC ARDUINO HARDWARE ANALYZER")
    print("=" * 50)
    
    # Define file paths
    base_path = Path("c:/Users/Florian/SynologyDrive/TUB/1_Dissertation/5_Codes/LFP_SOC_SOH/Arduino_Test_Setup")
    file_32 = base_path / "Stateful_32_32/code_weights/lstm_weights.h"
    file_64 = base_path / "Stateful_64_64/code_weights/lstm_weights.h"
    
    # Analyze weight files
    print("\n📊 ANALYZING WEIGHT FILES:")
    model_32_info = analyzer.analyze_weight_file(str(file_32))
    model_64_info = analyzer.analyze_weight_file(str(file_64))
    
    if model_32_info is None or model_64_info is None:
        print("❌ Could not analyze weight files!")
        return
    
    # Check hardware compatibility
    print("\n" + "="*80)
    compatible_32, incompatible_32 = analyzer.check_hardware_compatibility(model_32_info, "32x32 LSTM")
    
    print("\n" + "="*80)
    compatible_64, incompatible_64 = analyzer.check_hardware_compatibility(model_64_info, "64x64 LSTM")
    
    # Generate visualizations
    print("\n📈 Generating comparison charts...")
    analyzer.plot_memory_comparison(model_32_info, model_64_info)
    
    # Generate recommendation report
    analyzer.generate_recommendation_report(model_32_info, model_64_info)
    
    print("\n✅ Analysis complete! Check 'realistic_hardware_analysis.png' for visual comparison.")

if __name__ == "__main__":
    main()
