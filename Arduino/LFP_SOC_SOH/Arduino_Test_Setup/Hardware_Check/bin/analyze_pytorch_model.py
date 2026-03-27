# PyTorch Model Parameter Analysis
# Analysiert die Parameter aus best_model.pth Dateien

import torch
import numpy as np
import os
import matplotlib.pyplot as plt
from collections import OrderedDict

def analyze_pytorch_model(model_path):
    """
    Analysiert ein PyTorch Modell und extrahiert alle Parameter
    """
    
    if not os.path.exists(model_path):
        print(f"❌ Datei nicht gefunden: {model_path}")
        return None
    
    try:
        # Lade das Modell (CPU-only für Kompatibilität)
        model_data = torch.load(model_path, map_location='cpu')
        
        print(f"✅ Modell geladen: {os.path.basename(model_path)}")
        print(f"Dateigröße: {os.path.getsize(model_path) / 1024:.1f} kB")
        
        # Prüfe, was in der Datei gespeichert ist
        if isinstance(model_data, dict):
            print(f"\nDatei-Struktur: Dictionary mit {len(model_data)} Einträgen")
            for key in model_data.keys():
                print(f"  - {key}: {type(model_data[key])}")
            
            # Suche nach state_dict
            if 'state_dict' in model_data:
                state_dict = model_data['state_dict']
                print("\n📋 Verwende 'state_dict' für Parameter-Analyse")
            elif 'model_state_dict' in model_data:
                state_dict = model_data['model_state_dict']
                print("\n📋 Verwende 'model_state_dict' für Parameter-Analyse")
            else:
                # Annahme: Das Dictionary IST das state_dict
                state_dict = model_data
                print("\n📋 Verwende gesamtes Dictionary als state_dict")
        else:
            print(f"\nDatei-Typ: {type(model_data)}")
            return None
        
        return analyze_state_dict(state_dict, model_path)
        
    except Exception as e:
        print(f"❌ Fehler beim Laden: {e}")
        return None

def analyze_state_dict(state_dict, model_path):
    """
    Analysiert ein PyTorch state_dict
    """
    
    analysis = {
        'model_path': model_path,
        'total_parameters': 0,
        'layers': {},
        'architecture_info': {},
        'memory_usage': {}
    }
    
    print(f"\n{'='*80}")
    print(f"PYTORCH MODEL PARAMETER ANALYSE")
    print(f"{'='*80}")
    print(f"Modell: {os.path.basename(model_path)}")
    print(f"Parameter-Ebenen: {len(state_dict)}")
    
    # Analysiere jede Ebene
    total_params = 0
    layer_groups = {}
    
    for param_name, param_tensor in state_dict.items():
        param_count = param_tensor.numel()
        param_shape = list(param_tensor.shape)
        param_size_kb = (param_count * 4) / 1024  # float32 = 4 bytes
        
        total_params += param_count
        
        # Gruppiere nach Layer-Typ
        layer_type = param_name.split('.')[0] if '.' in param_name else param_name
        if layer_type not in layer_groups:
            layer_groups[layer_type] = []
        layer_groups[layer_type].append({
            'name': param_name,
            'shape': param_shape,
            'parameters': param_count,
            'size_kb': param_size_kb,
            'tensor': param_tensor
        })
        
        print(f"\n📊 {param_name}")
        print(f"   Shape: {param_shape}")
        print(f"   Parameter: {param_count:,}")
        print(f"   Größe: {param_size_kb:.3f} kB")
        print(f"   Min/Max: {param_tensor.min().item():.6f} / {param_tensor.max().item():.6f}")
        print(f"   Mean/Std: {param_tensor.mean().item():.6f} / {param_tensor.std().item():.6f}")
    
    analysis['total_parameters'] = total_params
    analysis['layers'] = layer_groups
    analysis['memory_usage'] = {
        'total_size_kb': (total_params * 4) / 1024,
        'total_size_mb': (total_params * 4) / (1024 * 1024)
    }
    
    # Zusammenfassung
    print(f"\n{'='*50}")
    print(f"ZUSAMMENFASSUNG")
    print(f"{'='*50}")
    print(f"Gesamte Parameter: {total_params:,}")
    print(f"Speicherverbrauch: {analysis['memory_usage']['total_size_kb']:.2f} kB")
    print(f"Speicherverbrauch: {analysis['memory_usage']['total_size_mb']:.2f} MB")
    
    print(f"\nParameter pro Layer-Gruppe:")
    for layer_type, params in layer_groups.items():
        layer_total = sum(p['parameters'] for p in params)
        layer_size_kb = sum(p['size_kb'] for p in params)
        print(f"  {layer_type}: {layer_total:,} Parameter ({layer_size_kb:.2f} kB)")
    
    return analysis

def detect_lstm_architecture(analysis):
    """
    Versucht die LSTM-Architektur aus den Parametern zu erkennen
    """
    
    if not analysis or 'layers' not in analysis:
        return None
    
    lstm_info = {}
    
    # Suche nach LSTM-Parametern
    for layer_type, params in analysis['layers'].items():
        if 'lstm' in layer_type.lower() or 'rnn' in layer_type.lower():
            print(f"\n🔍 LSTM Layer gefunden: {layer_type}")
            
            for param in params:
                name = param['name']
                shape = param['shape']
                
                if 'weight_ih' in name:  # Input-to-Hidden weights
                    if len(shape) == 2:
                        gates_x_hidden, input_size = shape
                        hidden_size = gates_x_hidden // 4  # 4 gates in LSTM
                        lstm_info['input_size'] = input_size
                        lstm_info['hidden_size'] = hidden_size
                        print(f"   Input→Hidden: {shape} → Input={input_size}, Hidden={hidden_size}")
                
                elif 'weight_hh' in name:  # Hidden-to-Hidden weights
                    if len(shape) == 2:
                        gates_x_hidden, hidden_size = shape
                        print(f"   Hidden→Hidden: {shape} → Hidden={hidden_size}")
                
                elif 'bias' in name:
                    print(f"   Bias: {shape}")
    
    # Vergleiche mit bekannten Architekturen
    if 'input_size' in lstm_info and 'hidden_size' in lstm_info:
        input_size = lstm_info['input_size']
        hidden_size = lstm_info['hidden_size']
        
        # Berechne erwartete LSTM Parameter
        expected_lstm_params = 4 * (input_size * hidden_size + hidden_size * hidden_size + hidden_size)
        
        print(f"\n🧮 LSTM Architektur-Erkennung:")
        print(f"   Input Size: {input_size}")
        print(f"   Hidden Size: {hidden_size}")
        print(f"   Erwartete LSTM Parameter: {expected_lstm_params:,}")
        
        lstm_info['expected_parameters'] = expected_lstm_params
        
        # Architektur-Bezeichnung
        arch_name = f"{hidden_size}×{hidden_size}"
        if arch_name in ['16×16', '32×32', '64×64']:
            print(f"   ✅ Erkannte Architektur: {arch_name}")
            lstm_info['architecture'] = arch_name
        else:
            print(f"   ⚠️  Unbekannte Architektur: {arch_name}")
    
    return lstm_info

def compare_with_arduino_weights(analysis, arduino_weight_path=None):
    """
    Vergleicht PyTorch Parameter mit Arduino Weight-Dateien
    """
    
    if not arduino_weight_path or not os.path.exists(arduino_weight_path):
        print(f"\n⚠️  Arduino Weight-Datei nicht gefunden: {arduino_weight_path}")
        return
    
    print(f"\n🔗 VERGLEICH MIT ARDUINO WEIGHTS")
    print(f"Arduino Datei: {os.path.basename(arduino_weight_path)}")
    
    # Hier könntest du die Arduino Header-Datei parsen und vergleichen
    # Das wäre eine Erweiterung für später

def create_parameter_visualization(analysis):
    """
    Erstellt Visualisierungen der Modell-Parameter
    """
    
    if not analysis or 'layers' not in analysis:
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Parameter pro Layer
    layer_names = []
    layer_params = []
    layer_colors = []
    
    color_map = {
        'lstm': 'lightblue',
        'mlp': 'lightcoral', 
        'linear': 'lightgreen',
        'fc': 'lightyellow'
    }
    
    for layer_type, params in analysis['layers'].items():
        layer_total = sum(p['parameters'] for p in params)
        layer_names.append(layer_type)
        layer_params.append(layer_total)
        
        # Farbe basierend auf Layer-Typ
        color = 'lightgray'
        for key, col in color_map.items():
            if key in layer_type.lower():
                color = col
                break
        layer_colors.append(color)
    
    bars1 = ax1.bar(layer_names, layer_params, color=layer_colors)
    ax1.set_xlabel('Layer')
    ax1.set_ylabel('Anzahl Parameter')
    ax1.set_title(f'Parameter pro Layer\n(Gesamt: {analysis["total_parameters"]:,})')
    ax1.tick_params(axis='x', rotation=45)
    
    for bar, params in zip(bars1, layer_params):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{params:,}', ha='center', va='bottom', fontsize=9)
    
    # 2. Speicherverbrauch
    layer_sizes = [sum(p['size_kb'] for p in params) for params in analysis['layers'].values()]
    
    wedges, texts, autotexts = ax2.pie(layer_sizes, labels=layer_names, autopct='%1.1f%%', 
                                       colors=layer_colors)
    ax2.set_title(f'Speicherverbrauch pro Layer\n({analysis["memory_usage"]["total_size_kb"]:.1f} kB total)')
    
    # 3. Parameter-Verteilung (falls LSTM erkannt)
    lstm_info = detect_lstm_architecture(analysis)
    if lstm_info and 'hidden_size' in lstm_info:
        h = lstm_info['hidden_size']
        i = lstm_info['input_size']
        
        # LSTM Komponenten
        components = ['Input→Hidden', 'Hidden→Hidden', 'Bias', 'Andere']
        ih_params = 4 * i * h
        hh_params = 4 * h * h  
        bias_params = 4 * h
        other_params = analysis['total_parameters'] - (ih_params + hh_params + bias_params)
        
        comp_values = [ih_params, hh_params, bias_params, max(0, other_params)]
        comp_colors = ['skyblue', 'lightcoral', 'lightgreen', 'lightyellow']
        
        wedges3, texts3, autotexts3 = ax3.pie(comp_values, labels=components, autopct='%1.1f%%',
                                              colors=comp_colors)
        ax3.set_title(f'LSTM Parameter-Verteilung\n({lstm_info["architecture"]} Architektur)')
    else:
        ax3.text(0.5, 0.5, 'Keine LSTM\nArchitektur erkannt', 
                ha='center', va='center', transform=ax3.transAxes, fontsize=14)
        ax3.set_title('Architektur-Analyse')
    
    # 4. Modell-Übersicht
    info_text = f"""Modell-Informationen:

Datei: {os.path.basename(analysis['model_path'])}
Parameter: {analysis['total_parameters']:,}
Speicher: {analysis['memory_usage']['total_size_kb']:.1f} kB
Layers: {len(analysis['layers'])}

Memory Breakdown:
Flash (Weights): {analysis['memory_usage']['total_size_kb']:.1f} kB
"""
    
    if lstm_info and 'hidden_size' in lstm_info:
        hidden_ram = (lstm_info['hidden_size'] * 2 * 4) / 1024
        info_text += f"RAM (Hidden State): {hidden_ram:.3f} kB"
    
    ax4.text(0.05, 0.95, info_text, transform=ax4.transAxes, 
             verticalalignment='top', fontsize=11, fontfamily='monospace',
             bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.7))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Modell-Übersicht')
    
    plt.tight_layout()
    
    # Speichere Plot
    plot_filename = f"pytorch_model_analysis_{os.path.splitext(os.path.basename(analysis['model_path']))[0]}.png"
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"\n📊 Plot gespeichert: {plot_filename}")
    
    plt.show()
    return fig

def find_model_files():
    """
    Sucht nach .pth Dateien im Workspace
    """
    
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes"
    model_files = []
    
    print("🔍 Suche nach PyTorch Modell-Dateien (.pth)...")
    
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith('.pth'):
                full_path = os.path.join(root, file)
                model_files.append(full_path)
                print(f"   Gefunden: {full_path}")
    
    return model_files

def main():
    """
    Hauptfunktion - analysiert alle gefundenen PyTorch Modelle
    """
    
    print("="*80)
    print("PYTORCH MODEL PARAMETER ANALYZER")
    print("="*80)
    
    # Suche nach Modell-Dateien
    model_files = find_model_files()
    
    if not model_files:
        print("❌ Keine .pth Dateien gefunden!")
        
        # Fallback: Frage nach manueller Eingabe
        manual_path = input("\nBitte gib den Pfad zu einer .pth Datei ein (oder Enter zum Beenden): ").strip()
        if manual_path and os.path.exists(manual_path):
            model_files = [manual_path]
        else:
            return
    
    # Analysiere jede gefundene Datei
    for model_path in model_files:
        print(f"\n{'='*80}")
        print(f"ANALYSIERE: {os.path.basename(model_path)}")
        print(f"{'='*80}")
        
        analysis = analyze_pytorch_model(model_path)
        
        if analysis:
            # Erkenne LSTM Architektur
            lstm_info = detect_lstm_architecture(analysis)
            
            # Erstelle Visualisierung
            create_parameter_visualization(analysis)
            
            print(f"\n✅ Analyse abgeschlossen für {os.path.basename(model_path)}")
        else:
            print(f"\n❌ Analyse fehlgeschlagen für {os.path.basename(model_path)}")

if __name__ == "__main__":
    main()
