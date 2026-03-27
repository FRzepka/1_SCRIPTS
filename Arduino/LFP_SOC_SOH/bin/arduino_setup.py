"""
Arduino Setup Helper Script

Automatisiert die Vorbereitung und den Upload des Arduino LSTM SOC Systems:
1. Überprüft verfügbare COM-Ports
2. Konvertiert PyTorch-Modell zu Arduino-Gewichten
3. Überprüft Arduino IDE Installation
4. Gibt Upload-Instruktionen

Verwendung:
    python arduino_setup.py --scan-ports
    python arduino_setup.py --prepare-model
    python arduino_setup.py --full-setup
"""

import serial.tools.list_ports
import subprocess
import sys
import os
from pathlib import Path
import argparse
import platform
import json

class ArduinoSetup:
    def __init__(self):
        self.project_dir = Path(__file__).parent
        self.arduino_sketch = self.project_dir / "arduino_lstm_soc_v2.ino"
        self.weights_file = self.project_dir / "lstm_weights.h"
        self.converter_script = self.project_dir / "pytorch_to_arduino_converter.py"
        
    def scan_ports(self):
        """Scanne verfügbare COM-Ports"""
        print("🔍 Scanning for available COM ports...")
        
        ports = serial.tools.list_ports.comports()
        
        if not ports:
            print("❌ No COM ports found!")
            return []
        
        available_ports = []
        for port in ports:
            try:
                # Test if port is available
                ser = serial.Serial(port.device, timeout=1)
                ser.close()
                available_ports.append(port)
                print(f"✅ {port.device}: {port.description}")
            except:
                print(f"⚠️ {port.device}: {port.description} (busy or unavailable)")
        
        print(f"\n📊 Found {len(available_ports)} available ports")
        return available_ports
    
    def check_arduino_ide(self):
        """Überprüfe Arduino IDE Installation"""
        print("🔍 Checking Arduino IDE installation...")
        
        # Common Arduino IDE paths
        arduino_paths = []
        
        if platform.system() == "Windows":
            possible_paths = [
                "C:/Program Files (x86)/Arduino/arduino_debug.exe",
                "C:/Program Files/Arduino/arduino_debug.exe",
                "C:/Users/*/AppData/Local/Arduino15/arduino_debug.exe",
                "C:/Program Files (x86)/Arduino/arduino.exe",
                "C:/Program Files/Arduino/arduino.exe"
            ]
        elif platform.system() == "Darwin":  # macOS
            possible_paths = [
                "/Applications/Arduino.app/Contents/MacOS/Arduino"
            ]
        else:  # Linux
            possible_paths = [
                "/usr/bin/arduino",
                "/usr/local/bin/arduino"
            ]
        
        for path in possible_paths:
            if os.path.exists(path):
                arduino_paths.append(path)
        
        if arduino_paths:
            print(f"✅ Arduino IDE found: {arduino_paths[0]}")
            return arduino_paths[0]
        else:
            print("❌ Arduino IDE not found!")
            print("💡 Please install Arduino IDE from: https://arduino.cc/downloads")
            return None
    
    def check_libraries(self):
        """Überprüfe erforderliche Arduino-Bibliotheken"""
        print("📚 Checking Arduino libraries...")
        
        # Arduino library paths
        if platform.system() == "Windows":
            lib_path = Path.home() / "Documents" / "Arduino" / "libraries"
        else:
            lib_path = Path.home() / "Arduino" / "libraries"
        
        required_libs = {
            "ArduinoJson": "ArduinoJson"
        }
        
        missing_libs = []
        for lib_name, lib_folder in required_libs.items():
            lib_full_path = lib_path / lib_folder
            if lib_full_path.exists():
                print(f"✅ {lib_name} library found")
            else:
                print(f"❌ {lib_name} library missing")
                missing_libs.append(lib_name)
        
        if missing_libs:
            print("\n💡 Install missing libraries via Arduino IDE:")
            print("   Tools → Manage Libraries → Search for:")
            for lib in missing_libs:
                print(f"   - {lib}")
        
        return len(missing_libs) == 0
    
    def prepare_model_weights(self):
        """Konvertiere PyTorch-Modell zu Arduino-Gewichten"""
        print("🧠 Preparing model weights...")
        
        if not self.converter_script.exists():
            print(f"❌ Converter script not found: {self.converter_script}")
            return False
        
        try:
            # Run converter script
            result = subprocess.run([
                sys.executable, str(self.converter_script)
            ], capture_output=True, text=True, cwd=str(self.project_dir))
            
            if result.returncode == 0:
                print("✅ Model weights converted successfully")
                
                if self.weights_file.exists():
                    size = self.weights_file.stat().st_size
                    print(f"📁 Weights file: {self.weights_file.name} ({size} bytes)")
                    return True
                else:
                    print("❌ Weights file not generated")
                    return False
            else:
                print(f"❌ Converter failed: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"❌ Error running converter: {e}")
            return False
    
    def check_sketch_files(self):
        """Überprüfe Arduino-Sketch-Dateien"""
        print("📝 Checking Arduino sketch files...")
        
        required_files = [
            ("arduino_lstm_soc_v2.ino", "Main Arduino sketch"),
            ("lstm_weights.h", "Model weights header")
        ]
        
        all_present = True
        for filename, description in required_files:
            filepath = self.project_dir / filename
            if filepath.exists():
                size = filepath.stat().st_size
                print(f"✅ {filename}: {description} ({size} bytes)")
            else:
                print(f"❌ {filename}: {description} (missing)")
                all_present = False
        
        return all_present
    
    def estimate_memory_usage(self):
        """Schätze Speicherverbrauch für verschiedene Arduino-Boards"""
        print("💾 Estimating memory usage...")
        
        # Geschätzte Speichernutzung basierend auf unserem Modell
        memory_usage = {
            "LSTM weights": 1408 * 4,      # float32 weights
            "LSTM states": 8 * 2 * 4,      # hidden + cell states
            "Input buffer": 10 * 4 * 4,    # sequence buffer
            "Temp arrays": 8 * 4 * 4,      # calculation arrays
            "JSON buffer": 1024,           # JSON processing
            "Program variables": 500       # misc variables
        }
        
        total_sram = sum(memory_usage.values())
        
        print("\n📊 Estimated SRAM usage:")
        for component, bytes_used in memory_usage.items():
            print(f"  {component}: {bytes_used} bytes")
        print(f"  Total: ~{total_sram} bytes")
        
        # Arduino board memory comparison
        board_memory = {
            "Arduino Uno": 2048,
            "Arduino Nano": 2048,
            "Arduino Micro": 2560,
            "Arduino Leonardo": 2560,
            "Arduino Mega 2560": 8192,
            "ESP32": 520192,
            "ESP8266": 81920
        }
        
        print(f"\n🎯 Board compatibility:")
        for board, ram in board_memory.items():
            percentage = (total_sram / ram) * 100
            status = "✅" if percentage < 80 else "⚠️" if percentage < 95 else "❌"
            print(f"  {status} {board}: {ram} bytes ({percentage:.1f}% used)")
        
        return total_sram
    
    def generate_upload_instructions(self, port=None):
        """Erstelle Upload-Instruktionen"""
        print("\n📋 Arduino Upload Instructions:")
        print("=" * 50)
        
        print("1. 🔧 PREPARE ARDUINO IDE:")
        print("   - Open Arduino IDE")
        print("   - Install ArduinoJson library (Tools → Manage Libraries)")
        print("   - Select your board type (Tools → Board)")
        
        if port:
            print(f"   - Select port: {port} (Tools → Port)")
        else:
            print("   - Select your Arduino port (Tools → Port)")
        
        print(f"\n2. 📁 OPEN PROJECT:")
        print(f"   - Open: {self.arduino_sketch}")
        print(f"   - Verify lstm_weights.h is in same folder")
        
        print(f"\n3. ⚡ RECOMMENDED BOARDS:")
        print("   - ESP32 (preferred): Full model, 520KB RAM")
        print("   - Arduino Mega: Reduced model, 8KB RAM") 
        print("   - Arduino Uno/Nano: NOT RECOMMENDED (2KB RAM)")
        
        print(f"\n4. 🚀 UPLOAD:")
        print("   - Click Upload button (→) in Arduino IDE")
        print("   - Wait for compilation and upload")
        print("   - Open Serial Monitor (115200 baud)")
        print("   - Look for 'Neural network ready!' message")
        
        print(f"\n5. 🧪 TEST:")
        print("   - Run: python hardware_validation.py --quick-test")
        print("   - Or: python test_arduino_system.py --test-connection")
        
    def full_setup(self):
        """Vollständige Setup-Routine"""
        print("🚀 Arduino LSTM SOC System - Full Setup")
        print("=" * 50)
        
        success = True
        
        # 1. Scan ports
        ports = self.scan_ports()
        recommended_port = ports[0].device if ports else None
        
        # 2. Check Arduino IDE
        arduino_ide = self.check_arduino_ide()
        if not arduino_ide:
            success = False
        
        # 3. Check libraries
        libs_ok = self.check_libraries()
        if not libs_ok:
            success = False
        
        # 4. Prepare model weights
        weights_ok = self.prepare_model_weights()
        if not weights_ok:
            success = False
        
        # 5. Check sketch files
        files_ok = self.check_sketch_files()
        if not files_ok:
            success = False
        
        # 6. Memory estimation
        memory_usage = self.estimate_memory_usage()
        
        # 7. Generate instructions
        self.generate_upload_instructions(recommended_port)
        
        print(f"\n🏁 Setup Status: {'✅ READY' if success else '❌ ISSUES FOUND'}")
        
        if success:
            print("\n🎉 System ready for Arduino upload!")
            if recommended_port:
                print(f"💡 Recommended port: {recommended_port}")
        else:
            print("\n🔧 Please resolve the issues above before uploading")
        
        return success

def main():
    parser = argparse.ArgumentParser(description='Arduino LSTM Setup Helper')
    parser.add_argument('--scan-ports', action='store_true', help='Scan available COM ports')
    parser.add_argument('--prepare-model', action='store_true', help='Convert PyTorch model to Arduino')
    parser.add_argument('--check-ide', action='store_true', help='Check Arduino IDE installation')
    parser.add_argument('--full-setup', action='store_true', help='Run complete setup process')
    
    args = parser.parse_args()
    
    setup = ArduinoSetup()
    
    if args.scan_ports:
        setup.scan_ports()
    elif args.prepare_model:
        setup.prepare_model_weights()
    elif args.check_ide:
        setup.check_arduino_ide()
        setup.check_libraries()
    elif args.full_setup:
        setup.full_setup()
    else:
        # Default: run full setup
        setup.full_setup()

if __name__ == "__main__":
    main()
