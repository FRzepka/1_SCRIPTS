#!/usr/bin/env python3
"""
Script to recreate Arduino deployment files from PyTorch model
Uses the exact conversion pipeline identified in the codebase
"""

import sys
import os
from pathlib import Path
import subprocess
import json

# Add paths to sys.path for imports
project_root = Path(__file__).parent.parent.parent
bin_path = project_root / "bin"
sys.path.insert(0, str(bin_path))
sys.path.insert(0, str(project_root))

def main():
    """Main function to recreate Arduino deployment files"""
    print("🔄 Arduino Deployment Files Recreation")
    print("=" * 50)
    
    # Define paths
    converter_script = project_root / "bin" / "pytorch_to_arduino_converter_full32.py"
    model_path = project_root / "BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU" / "training_run_1_2_4_31" / "best_model.pth"
    arduino_dir = Path(__file__).parent / "arduino_lstm_soc_full32_with_monitoring"
    
    print(f"📁 Project root: {project_root}")
    print(f"🔧 Converter script: {converter_script}")
    print(f"🤖 Model path: {model_path}")
    print(f"📱 Arduino directory: {arduino_dir}")
    
    # Check if required files exist
    files_to_check = [
        (converter_script, "PyTorch-to-Arduino converter"),
        (model_path, "Trained PyTorch model")
    ]
    
    missing_files = []
    for file_path, description in files_to_check:
        if file_path.exists():
            print(f"✅ {description}: Found")
        else:
            print(f"❌ {description}: Missing - {file_path}")
            missing_files.append((file_path, description))
    
    if missing_files:
        print(f"\n⚠️  Missing {len(missing_files)} required files:")
        for file_path, description in missing_files:
            print(f"   - {description}: {file_path}")
        return False
    
    # Create Arduino directory if it doesn't exist
    arduino_dir.mkdir(exist_ok=True)
    
    # Run the converter
    print(f"\n🚀 Running PyTorch-to-Arduino converter...")
    try:
        # Set up environment
        env = os.environ.copy()
        env['PYTHONPATH'] = str(bin_path)
        
        # Run converter
        result = subprocess.run([
            sys.executable, str(converter_script)
        ], capture_output=True, text=True, cwd=str(bin_path), env=env)
        
        if result.returncode == 0:
            print("✅ Converter executed successfully")
            print("📤 Converter output:")
            print(result.stdout)
            
            # Check if files were generated
            generated_files = []
            for file_pattern in ["*.h", "*.ino"]:
                for file_path in bin_path.glob(file_pattern):
                    generated_files.append(file_path)
            
            if generated_files:
                print(f"\n📄 Generated files found in {bin_path}:")
                for file_path in generated_files:
                    print(f"   - {file_path.name} ({file_path.stat().st_size} bytes)")
                    
                    # Move to Arduino directory if it's a weights file
                    if "lstm_weights" in file_path.name:
                        dest_path = arduino_dir / "lstm_weights.h"
                        file_path.rename(dest_path)
                        print(f"   📱 Moved to Arduino directory: {dest_path}")
            
            # Check existing Arduino files
            existing_arduino_files = list(arduino_dir.glob("*"))
            if existing_arduino_files:
                print(f"\n📱 Arduino deployment files in {arduino_dir}:")
                for file_path in existing_arduino_files:
                    size = file_path.stat().st_size if file_path.is_file() else "dir"
                    print(f"   - {file_path.name} ({size} bytes)")
            
            return True
            
        else:
            print("❌ Converter failed:")
            print(f"Return code: {result.returncode}")
            print(f"Error output:\n{result.stderr}")
            return False
            
    except Exception as e:
        print(f"❌ Error running converter: {e}")
        return False

def check_arduino_compatibility():
    """Check hardware compatibility using existing calculator"""
    print(f"\n🔍 Checking Arduino hardware compatibility...")
    
    try:
        # Import and use hardware calculator
        from arduino_hardware_calculator import ArduinoHardwareCalculator
        
        calc = ArduinoHardwareCalculator()
        
        # Try to load the real model
        model_path = project_root / "BMS_SOC_LSTM_stateful_1.2.4.31_Train_CPU" / "training_run_1_2_4_31" / "best_model.pth"
        
        if model_path.exists():
            try:
                analysis = calc.analyze_model(str(model_path))
                calc.generate_report(analysis, "arduino_real_model_analysis.json")
                print("✅ Hardware analysis completed with real model")
            except Exception as e:
                print(f"⚠️  Real model analysis failed: {e}")
                # Fallback to synthetic model
                print("🔄 Using synthetic model for analysis...")
                analysis = calc.create_synthetic_analysis()
                calc.generate_report(analysis, "arduino_synthetic_analysis.json")
        else:
            print("⚠️  Model file not found, using synthetic analysis")
            analysis = calc.create_synthetic_analysis()
            calc.generate_report(analysis, "arduino_synthetic_analysis.json")
            
        return True
        
    except Exception as e:
        print(f"❌ Hardware analysis failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    if success:
        check_arduino_compatibility()
        print(f"\n🎉 Arduino deployment files recreation completed!")
        print(f"🔗 Files ready for Arduino deployment")
    else:
        print(f"\n❌ Recreation failed - check error messages above")
