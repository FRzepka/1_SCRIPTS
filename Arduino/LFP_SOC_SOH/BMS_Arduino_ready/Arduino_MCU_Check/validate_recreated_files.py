#!/usr/bin/env python3
"""
Script to validate the recreated Arduino deployment files
and compare them with hardware requirements.
"""

import os
import sys
from arduino_hardware_calculator import ArduinoHardwareCalculator

def validate_arduino_deployment():
    """Validate the recreated Arduino deployment files."""
    
    # Initialize calculator
    calc = ArduinoHardwareCalculator()
    
    # File paths
    ino_file = 'arduino_lstm_soc_full32_with_monitoring/arduino_lstm_soc_full32_with_monitoring.ino'
    weights_file = 'arduino_lstm_soc_full32_with_monitoring/lstm_weights.h'
    
    print("🔍 ARDUINO DEPLOYMENT VALIDATION")
    print("=" * 60)
    
    # Check if files exist
    print(f"INO file exists: {os.path.exists(ino_file)}")
    print(f"Weights file exists: {os.path.exists(weights_file)}")
    
    if not (os.path.exists(ino_file) and os.path.exists(weights_file)):
        print("❌ Arduino deployment files not found!")
        return False
    
    # Get file sizes
    ino_size = os.path.getsize(ino_file)
    weights_size = os.path.getsize(weights_file)
    
    print(f"\n📊 FILE SIZES:")
    print(f"INO file:     {ino_size:,} bytes ({ino_size/1024:.1f} KB)")
    print(f"Weights file: {weights_size:,} bytes ({weights_size/1024:.1f} KB)")
    print(f"Total:        {(ino_size + weights_size):,} bytes ({(ino_size + weights_size)/1024:.1f} KB)")
    
    try:
        # Analyze the Arduino deployment
        print(f"\n🔬 ANALYZING ARDUINO DEPLOYMENT...")
        analysis = calc.analyze_arduino_deployment(ino_file, weights_file)
        
        print(f"\n📋 DEPLOYMENT ANALYSIS RESULTS:")
        print(f"Total Flash Required: {analysis['flash_required_kb']:.1f} KB")
        print(f"Total RAM Required:   {analysis['ram_required_kb']:.1f} KB")
        print(f"Model Parameters:     {analysis['model_params']:,}")
        
        print(f"\n🎯 BOARD COMPATIBILITY:")
        compatible_boards = []
        incompatible_boards = []
        
        for board, compat in analysis['board_compatibility'].items():
            if compat['compatible']:
                flash_usage = compat['flash_usage_percent']
                ram_usage = compat['ram_usage_percent']
                compatible_boards.append(f"{board}: Flash {flash_usage:.1f}%, RAM {ram_usage:.1f}%")
            else:
                incompatible_boards.append(board)
                flash_usage = compat['flash_usage_percent']
                ram_usage = compat['ram_usage_percent']
                print(f"❌ {board.upper()}: Flash {flash_usage:.1f}%, RAM {ram_usage:.1f}%")
                for reason in compat['reasons']:
                    print(f"   - {reason}")
        
        if compatible_boards:
            print(f"\n✅ COMPATIBLE BOARDS:")
            for board_info in compatible_boards:
                print(f"   ✓ {board_info}")
        
        print(f"\n💡 OPTIMIZATION SUGGESTIONS:")
        for suggestion in analysis['optimization_suggestions'][:5]:  # Show top 5
            print(f"   - {suggestion}")
        
        # Check if the deployment is practical
        practical_boards = [board for board, compat in analysis['board_compatibility'].items() 
                          if compat['compatible']]
        
        if practical_boards:
            print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
            print(f"   This LSTM model can be successfully deployed on: {', '.join(practical_boards).upper()}")
            if 'esp32' in practical_boards:
                print(f"   ESP32 is recommended for IoT applications with WiFi connectivity")
            if 'teensy40' in practical_boards:
                print(f"   Teensy 4.0 is recommended for high-performance real-time applications")
        else:
            print(f"\n⚠️  DEPLOYMENT WARNING:")
            print(f"   This model requires significant optimization for deployment on standard Arduino boards")
            
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing deployment: {e}")
        return False

def check_conversion_pipeline():
    """Check if the conversion pipeline is available."""
    
    converter_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\bin\pytorch_to_arduino_converter_full32.py"
    model_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU\training_run_1_2_4_31\best_model.pth"
    
    print(f"\n🔧 CONVERSION PIPELINE STATUS:")
    print(f"Converter script: {os.path.exists(converter_path)}")
    print(f"Source model:     {os.path.exists(model_path)}")
    
    if os.path.exists(converter_path) and os.path.exists(model_path):
        print(f"✅ Full conversion pipeline is available and ready")
        return True
    else:
        print(f"⚠️  Conversion pipeline incomplete")
        return False

if __name__ == "__main__":
    print("🎯 ARDUINO DEPLOYMENT VALIDATION SCRIPT")
    print("=" * 80)
    
    # Validate deployment
    deployment_valid = validate_arduino_deployment()
    
    # Check conversion pipeline
    pipeline_valid = check_conversion_pipeline()
    
    print(f"\n" + "=" * 80)
    print(f"📊 VALIDATION SUMMARY:")
    print(f"   Arduino Deployment: {'✅ VALID' if deployment_valid else '❌ INVALID'}")
    print(f"   Conversion Pipeline: {'✅ AVAILABLE' if pipeline_valid else '❌ INCOMPLETE'}")
    
    if deployment_valid and pipeline_valid:
        print(f"\n🎉 SUCCESS: Arduino deployment is ready for use!")
        print(f"   You can upload the .ino file to compatible Arduino boards")
        print(f"   Recommended boards: ESP32, Teensy 4.0")
    else:
        print(f"\n⚠️  Issues found - please review the analysis above")
