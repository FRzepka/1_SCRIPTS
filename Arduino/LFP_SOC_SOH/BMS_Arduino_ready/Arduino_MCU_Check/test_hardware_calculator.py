"""
Test Script für Arduino Hardware Calculator
==========================================

Demonstriert die Verwendung des Arduino Hardware Calculators
und erstellt einen detaillierten Bericht über die Hardware-Anforderungen.
"""

import sys
import os
from pathlib import Path

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from arduino_hardware_calculator import ArduinoHardwareCalculator

def test_hardware_calculator():
    """Test the hardware calculator with available models"""
    
    print("🔧 Testing Arduino Hardware Calculator")
    print("=" * 60)
    
    # Try to find model files
    possible_model_paths = [
        "../../best_model_optimized.pth",
        "../../best_model.pth", 
        "../../../best_model_optimized.pth",
        "../../../best_model.pth",
        "../../../../best_model_optimized.pth",
        "../../../../best_model.pth"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if os.path.exists(path):
            model_path = path
            print(f"✅ Found model: {path}")
            break
    
    if not model_path:
        print("❌ No model file found! Searching in:")
        for path in possible_model_paths:
            print(f"   {os.path.abspath(path)} - {'EXISTS' if os.path.exists(path) else 'NOT FOUND'}")
        return
    
    # Initialize calculator
    calculator = ArduinoHardwareCalculator()
    
    try:
        # Run complete analysis
        print(f"\n🚀 Starting analysis of: {model_path}")
        report = calculator.run_complete_analysis(model_path)
        
        # Print summary
        print("\n" + "=" * 60)
        print("📊 ANALYSIS SUMMARY")
        print("=" * 60)
        
        flash_req = report['flash_requirements']
        ram_req = report['ram_requirements']
        
        print(f"Flash Requirements: {flash_req['total_flash_kb']:.1f} KB")
        print(f"RAM Requirements:   {ram_req['total_ram_kb']:.1f} KB")
        
        print("\n🎯 Recommended Arduino Boards:")
        for board_name, board_info in report['board_compatibility'].items():
            if board_info['compatible']:
                print(f"  ✅ {board_name.upper()}: Flash {board_info['flash_usage_percent']:.1f}%, RAM {board_info['ram_usage_percent']:.1f}%")
        
        print("\n⚠️  Incompatible Boards:")
        for board_name, board_info in report['board_compatibility'].items():
            if not board_info['compatible']:
                reasons = []
                if not board_info['flash_ok']:
                    reasons.append(f"Flash {board_info['flash_usage_percent']:.1f}%")
                if not board_info['ram_ok']:
                    reasons.append(f"RAM {board_info['ram_usage_percent']:.1f}%")
                print(f"  ❌ {board_name.upper()}: {', '.join(reasons)}")
        
        print(f"\n💾 Detailed report saved to: arduino_hardware_analysis.json")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

def create_comparison_table():
    """Create a simple comparison table for Arduino boards"""
    
    calculator = ArduinoHardwareCalculator()
    
    print("\n" + "=" * 80)
    print("📋 ARDUINO BOARD SPECIFICATIONS")
    print("=" * 80)
    
    print(f"{'Board':<12} | {'Flash (KB)':<10} | {'RAM (bytes)':<12} | {'CPU (MHz)':<10} | {'Architecture':<12}")
    print("-" * 80)
    
    for board_name, specs in calculator.arduino_boards.items():
        print(f"{board_name:<12} | {specs['flash_kb']:<10} | {specs['ram_bytes']:<12} | {specs['cpu_mhz']:<10} | {specs['architecture']:<12}")

if __name__ == "__main__":
    test_hardware_calculator()
    create_comparison_table()
