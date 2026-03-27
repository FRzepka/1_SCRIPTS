import sys
import re
import os
sys.path.append(r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Hardware_Check")

from comprehensive_model_analysis_fixed import ModelArchitectureAnalyzer

# Teste die Regex-Pattern direkt
def test_regex_patterns():
    print("🧪 Teste Regex-Pattern...")
    test_files = [
        r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_32_32\model\BMS_SOC_LSTM_stateful_1.2.4_Train.py",
        r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup\Stateful_64_64\model\BMS_SOC_LSTM_stateful_1.2.4_Train.py"
    ]
    
    hidden_pattern = r'(?:HIDDEN_SIZE|hidden_size)\s*=\s*(\d+)'
    input_pattern = r'(?:INPUT_SIZE|input_size)\s*=\s*(\d+)'
    
    for filepath in test_files:
        if os.path.exists(filepath):
            print(f"\n📄 Teste: {os.path.basename(filepath)}")
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
            
            hidden_match = re.search(hidden_pattern, content, re.IGNORECASE)
            input_match = re.search(input_pattern, content, re.IGNORECASE)
            
            if hidden_match:
                print(f"   ✅ HIDDEN_SIZE gefunden: {hidden_match.group(1)}")
            else:
                print(f"   ❌ HIDDEN_SIZE nicht gefunden")
                
            if input_match:
                print(f"   ✅ INPUT_SIZE gefunden: {input_match.group(1)}")
            else:
                print(f"   ❌ INPUT_SIZE nicht gefunden")
        else:
            print(f"❌ Datei nicht gefunden: {filepath}")

test_regex_patterns()

# Debug
base_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\Arduino_Test_Setup"
analyzer = ModelArchitectureAnalyzer(base_path)

# Lade die Daten
model_dirs = analyzer.find_model_directories()
print(f"Gefunden: {len(model_dirs)} Verzeichnisse")

for arch_info in model_dirs[:2]:  # Nur die ersten 2 testen
    result = analyzer.analyze_architecture_directory(arch_info)
    analyzer.results[arch_info['architecture']] = result
    
    print(f"\n=== {arch_info['architecture']} ===")
    
    # Script Analyse
    script_analysis = result.get('training_script_analysis')
    if script_analysis:
        print(f"Script-Analyse: {script_analysis}")
        arch = script_analysis.get('architecture', {})
        if arch:
            hidden = arch.get('hidden_size', 0)
            input_size = arch.get('input_size', 4)
            print(f"Hidden: {hidden}, Input: {input_size}")
            
            # Teste Parameter-Berechnung
            if hidden > 0:
                param_calc = analyzer.calculate_complete_model_parameters(input_size, hidden, None)
                print(f"Berechnete Parameter: {param_calc['total']}")
            else:
                print("❌ Hidden size ist 0!")
        else:
            print("❌ Keine Architektur in Script-Analyse")
    else:
        print("❌ Keine Script-Analyse")
    
    # Model Analyse
    model_analysis = result.get('pytorch_model_analysis')
    if model_analysis:
        print(f"Model Parameter: {model_analysis.get('total_parameters', 0)}")
    else:
        print("❌ Keine Model-Analyse")
