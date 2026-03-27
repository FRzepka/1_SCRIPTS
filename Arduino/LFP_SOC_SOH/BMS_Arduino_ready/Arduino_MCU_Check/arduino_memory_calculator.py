#!/usr/bin/env python3
"""
Arduino Memory Usage Calculator & Validator
===========================================

Theoretische Berechnung von RAM und Flash Verbrauch für Arduino LSTM SOC Predictor
basierend auf statischer Code-Analyse und Gewichts-Größen.

Vergleicht theoretische Werte mit realen Arduino Messungen für präzise 
Memory-Management Analyse.

Features:
- Statische Analyse der .ino Arduino Datei
- LSTM Weights Größen-Berechnung aus lstm_weights.h
- Stateful vs Window-based Memory-Modelle
- RAM/Flash Verbrauch Vorhersage
- Validierung gegen echte Hardware-Messungen

Author: Arduino BMS Memory Analysis System
Version: 1.0.0
"""

import re
import os
import numpy as np
import pandas as pd
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging
from enum import Enum

# Logging Setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MemoryType(Enum):
    """Speicher-Typen für Arduino"""
    SRAM = "SRAM"
    FLASH = "Flash"
    EEPROM = "EEPROM"

class ProcessingMode(Enum):
    """LSTM Processing Modi"""
    STATEFUL = "Stateful"
    WINDOW_BASED = "Window-based"

@dataclass
class MemoryUsage:
    """Memory Usage Datenstruktur"""
    variables: int = 0
    weights: int = 0
    buffers: int = 0
    stack: int = 0
    heap: int = 0
    total: int = 0
    
    def __add__(self, other):
        return MemoryUsage(
            variables=self.variables + other.variables,
            weights=self.weights + other.weights,
            buffers=self.buffers + other.buffers,
            stack=self.stack + other.stack,
            heap=self.heap + other.heap,
            total=self.total + other.total
        )

@dataclass
class ArduinoSpecs:
    """Arduino Hardware Spezifikationen"""
    name: str
    flash_kb: int
    sram_kb: int
    eeprom_kb: int
    cpu_mhz: int
    
# Arduino Hardware Datenbank
ARDUINO_SPECS = {
    "UNO_R4_WIFI": ArduinoSpecs("Arduino Uno R4 WiFi", 256, 32, 4, 48),
    "UNO_R3": ArduinoSpecs("Arduino Uno R3", 32, 2, 1, 16),
    "NANO": ArduinoSpecs("Arduino Nano", 32, 2, 1, 16),
    "MEGA": ArduinoSpecs("Arduino Mega", 256, 8, 4, 16),
    "DUE": ArduinoSpecs("Arduino Due", 512, 96, 0, 84),
}

class ArduinoMemoryCalculator:
    """
    Arduino Memory Usage Calculator
    
    Analysiert Arduino .ino Code und lstm_weights.h um theoretischen
    RAM und Flash Verbrauch zu berechnen.
    """
    
    def __init__(self, 
                 ino_path: str,
                 weights_path: str,
                 arduino_type: str = "UNO_R4_WIFI",
                 processing_mode: ProcessingMode = ProcessingMode.STATEFUL):
        
        self.ino_path = Path(ino_path)
        self.weights_path = Path(weights_path)
        self.arduino_specs = ARDUINO_SPECS[arduino_type]
        self.processing_mode = processing_mode
        
        # Memory tracking
        self.sram_usage = MemoryUsage()
        self.flash_usage = MemoryUsage()
        
        # Code analysis results
        self.variables = {}
        self.arrays = {}
        self.weights_info = {}
        
        logger.info(f"Arduino Memory Calculator initialisiert")
        logger.info(f"Target: {self.arduino_specs.name}")
        logger.info(f"Mode: {self.processing_mode.value}")

    def analyze_ino_file(self) -> Dict:
        """
        Analysiert die Arduino .ino Datei für Memory Usage
        """
        logger.info(f"Analysiere Arduino Code: {self.ino_path}")
        
        if not self.ino_path.exists():
            raise FileNotFoundError(f"Arduino Datei nicht gefunden: {self.ino_path}")
        
        with open(self.ino_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        results = {
            'variables': self._extract_variables(code),
            'arrays': self._extract_arrays(code),
            'buffers': self._extract_buffers(code),
            'functions': self._extract_functions(code),
            'includes': self._extract_includes(code),
            'defines': self._extract_defines(code)
        }
        
        logger.info(f"Code Analyse abgeschlossen:")
        logger.info(f"  - Variablen: {len(results['variables'])}")
        logger.info(f"  - Arrays: {len(results['arrays'])}")
        logger.info(f"  - Buffer: {len(results['buffers'])}")
        logger.info(f"  - Funktionen: {len(results['functions'])}")
        
        return results

    def _extract_variables(self, code: str) -> Dict[str, Dict]:
        """Extrahiert Variable Definitionen"""
        variables = {}
        
        # Variable patterns (float, int, unsigned long, etc.)
        patterns = [
            r'(float|double)\s+(\w+)\s*(?:=\s*([^;]+))?;',
            r'(int|long|short)\s+(\w+)\s*(?:=\s*([^;]+))?;',
            r'(unsigned\s+(?:int|long|short))\s+(\w+)\s*(?:=\s*([^;]+))?;',
            r'(bool|byte|char)\s+(\w+)\s*(?:=\s*([^;]+))?;'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            for match in matches:
                var_type = match[0].strip()
                var_name = match[1].strip()
                var_value = match[2].strip() if len(match) > 2 and match[2] else "0"
                
                size = self._get_type_size(var_type)
                variables[var_name] = {
                    'type': var_type,
                    'size': size,
                    'value': var_value,
                    'location': 'SRAM'
                }
        
        return variables

    def _extract_arrays(self, code: str) -> Dict[str, Dict]:
        """Extrahiert Array Definitionen"""
        arrays = {}
        
        # Array patterns
        patterns = [
            r'(float|int|unsigned\s+\w+|char)\s+(\w+)\[([^\]]+)\](?:\[([^\]]+)\])?\s*(?:=\s*\{[^}]*\})?;',
            r'(float|int|unsigned\s+\w+|char)\s+(\w+)\[([^\]]+)\](?:\[([^\]]+)\])?(?:\[([^\]]+)\])?\s*(?:=\s*\{[^}]*\})?;'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, code, re.MULTILINE)
            for match in matches:
                array_type = match[0].strip()
                array_name = match[1].strip()
                
                # Calculate dimensions
                dimensions = [d.strip() for d in match[2:] if d.strip()]
                
                # Calculate total size
                element_size = self._get_type_size(array_type)
                total_elements = 1
                
                for dim in dimensions:
                    try:
                        # Try to evaluate dimension (e.g., "32", "HIDDEN_SIZE")
                        dim_value = self._evaluate_dimension(dim, code)
                        total_elements *= dim_value
                    except:
                        logger.warning(f"Kann Dimension nicht evaluieren: {dim}")
                        total_elements *= 32  # Default fallback
                
                total_size = element_size * total_elements
                
                arrays[array_name] = {
                    'type': array_type,
                    'dimensions': dimensions,
                    'elements': total_elements,
                    'element_size': element_size,
                    'total_size': total_size,
                    'location': 'SRAM'
                }
        
        return arrays

    def _extract_buffers(self, code: str) -> Dict[str, Dict]:
        """Extrahiert Buffer Definitionen"""
        buffers = {}
        
        # Buffer patterns (oft char arrays für Serial communication)
        pattern = r'char\s+(\w+)\[([^\]]+)\]'
        matches = re.findall(pattern, code)
        
        for match in matches:
            buffer_name = match[0]
            buffer_size = self._evaluate_dimension(match[1], code)
            
            buffers[buffer_name] = {
                'type': 'char',
                'size': buffer_size,
                'location': 'SRAM'
            }
        
        return buffers

    def _extract_functions(self, code: str) -> List[str]:
        """Extrahiert Funktionsnamen"""
        pattern = r'(?:void|int|float|bool|char\*?)\s+(\w+)\s*\([^)]*\)\s*\{'
        matches = re.findall(pattern, code)
        return matches

    def _extract_includes(self, code: str) -> List[str]:
        """Extrahiert Include Statements"""
        pattern = r'#include\s*[<"]([^>"]+)[>"]'
        matches = re.findall(pattern, code)
        return matches

    def _extract_defines(self, code: str) -> Dict[str, str]:
        """Extrahiert #define Statements"""
        defines = {}
        pattern = r'#define\s+(\w+)\s+(.+)'
        matches = re.findall(pattern, code)
        
        for match in matches:
            defines[match[0]] = match[1].strip()
        
        return defines

    def analyze_weights_file(self) -> Dict:
        """
        Analysiert lstm_weights.h für Gewichts-Größen
        """
        logger.info(f"Analysiere LSTM Weights: {self.weights_path}")
        
        if not self.weights_path.exists():
            raise FileNotFoundError(f"Weights Datei nicht gefunden: {self.weights_path}")
        
        with open(self.weights_path, 'r', encoding='utf-8') as f:
            weights_code = f.read()
        
        weights_info = {}
        
        # Pattern für Gewichts-Arrays
        pattern = r'const\s+float\s+(\w+)\[([^\]]+)\](?:\[([^\]]+)\])?\s*='
        matches = re.findall(pattern, weights_code, re.MULTILINE)
        
        total_weights = 0
        total_size = 0
        
        for match in matches:
            weight_name = match[0]
            dim1 = self._evaluate_dimension(match[1], weights_code)
            dim2 = self._evaluate_dimension(match[2], weights_code) if match[2] else 1
            
            elements = dim1 * dim2
            size_bytes = elements * 4  # float = 4 bytes
            
            weights_info[weight_name] = {
                'dimensions': [dim1, dim2] if dim2 > 1 else [dim1],
                'elements': elements,
                'size_bytes': size_bytes,
                'location': 'FLASH'  # const weights go to Flash
            }
            
            total_weights += elements
            total_size += size_bytes
        
        logger.info(f"Weights Analyse abgeschlossen:")
        logger.info(f"  - Gewichts-Arrays: {len(weights_info)}")
        logger.info(f"  - Gesamt Weights: {total_weights:,}")
        logger.info(f"  - Gesamt Größe: {total_size:,} Bytes ({total_size/1024:.1f} KB)")
        
        return {
            'weights': weights_info,
            'total_weights': total_weights,
            'total_size': total_size
        }

    def _get_type_size(self, var_type: str) -> int:
        """Gibt die Größe eines Datentyps in Bytes zurück"""
        size_map = {
            'char': 1, 'byte': 1, 'bool': 1,
            'int': 2, 'short': 2,
            'long': 4, 'unsigned long': 4,
            'float': 4, 'double': 4,
            'unsigned int': 2, 'unsigned short': 2
        }
        return size_map.get(var_type.strip(), 4)  # Default: 4 bytes

    def _evaluate_dimension(self, dim_str: str, code: str) -> int:
        """Evaluiert Dimension Strings (z.B. HIDDEN_SIZE, 32)"""
        dim_str = dim_str.strip()
        
        # Direkte Zahl
        if dim_str.isdigit():
            return int(dim_str)
        
        # Bekannte Konstanten
        constants = {
            'HIDDEN_SIZE': 32,
            'INPUT_SIZE': 4,
            'OUTPUT_SIZE': 1,
            'BUFFER_SIZE': 256,
            'ARDUINO_HIDDEN_SIZE': 32,
            'MLP_HIDDEN_SIZE': 32
        }
        
        if dim_str in constants:
            return constants[dim_str]
        
        # Versuche aus #define zu extrahieren
        pattern = rf'#define\s+{re.escape(dim_str)}\s+(\d+)'
        match = re.search(pattern, code)
        if match:
            return int(match.group(1))
        
        # Fallback
        logger.warning(f"Kann Dimension nicht auflösen: {dim_str}, verwende 32")
        return 32

    def calculate_theoretical_usage(self) -> Dict:
        """
        Berechnet theoretischen RAM und Flash Verbrauch
        """
        logger.info("Berechne theoretischen Memory Usage...")
        # Analysiere Code und Weights
        code_analysis = self.analyze_ino_file()
        weights_analysis = self.analyze_weights_file()
        
        # SRAM Berechnung - VERBESSERTE SCHÄTZUNG
        sram_variables = sum(var['size'] for var in code_analysis['variables'].values())
        sram_arrays = sum(arr['total_size'] for arr in code_analysis['arrays'].values())
        sram_buffers = sum(buf['size'] for buf in code_analysis['buffers'].values())
        
        # Stack estimation (function calls, local variables) - ERHÖHT
        stack_estimate = len(code_analysis['functions']) * 64  # ~64 bytes per function call (war 32)
        
        # LSTM Hidden States - KRITISCH FÜR KORREKTE SCHÄTZUNG
        hidden_size = 32  # HIDDEN_SIZE aus Arduino Code
        lstm_states_sram = hidden_size * 2 * 4 * 2  # h_state + c_state, 2 layers, 4 bytes each, safety factor
        
        # Temporary arrays für LSTM Berechnungen - SEHR WICHTIG
        lstm_temp_arrays = hidden_size * 4 * 4 * 4  # 4 gates * 4 bytes * safety factor
        
        # Arduino Framework Overhead - DEUTLICH ERHÖHT
        arduino_framework_overhead = 8192  # Arduino runtime, interrupts, etc.
        
        # Compiler padding und Alignment - WICHTIG
        alignment_overhead = 1024
        
        # Heap (normalerweise nicht verwendet, aber reserve)
        heap_estimate = 2048
        sram_total = (sram_variables + sram_arrays + sram_buffers + stack_estimate + 
                     lstm_states_sram + lstm_temp_arrays + arduino_framework_overhead + 
                     alignment_overhead + heap_estimate)

        # Flash Berechnung  
        flash_weights = weights_analysis['total_size']
        flash_code = self._estimate_code_size(code_analysis)
        flash_constants = 1024  # Estimate für String literals, etc.
        
        flash_total = flash_weights + flash_code + flash_constants
        
        # Processing Mode Adjustments
        if self.processing_mode == ProcessingMode.WINDOW_BASED:
            # Window-based benötigt zusätzlichen Buffer für Fenster
            window_size = 64  # Typical window size
            input_size = 4
            additional_sram = window_size * input_size * 4  # float
            sram_total += additional_sram
            logger.info(f"Window-based Modus: +{additional_sram} Bytes für Input Window")
        
        results = {
            'sram': {
                'variables': sram_variables,
                'arrays': sram_arrays,
                'buffers': sram_buffers,
                'stack': stack_estimate,
                'heap': heap_estimate,
                'total': sram_total,
                'percent': (sram_total / (self.arduino_specs.sram_kb * 1024)) * 100
            },
            'flash': {
                'weights': flash_weights,
                'code': flash_code,
                'constants': flash_constants,
                'total': flash_total,
                'percent': (flash_total / (self.arduino_specs.flash_kb * 1024)) * 100
            },
            'weights_count': weights_analysis['total_weights'],
            'arduino_specs': self.arduino_specs
        }
        
        logger.info(f"Theoretischer Memory Usage berechnet:")
        logger.info(f"  SRAM: {sram_total:,} Bytes ({results['sram']['percent']:.1f}%)")
        logger.info(f"  Flash: {flash_total:,} Bytes ({results['flash']['percent']:.1f}%)")
        
        return results

    def _estimate_code_size(self, code_analysis: Dict) -> int:
        """
        Schätzt Flash-Größe für kompilierten Code
        """
        # Rough estimation basierend auf Funktionsanzahl und Komplexität
        base_size = 8192  # Arduino framework overhead
        function_size = len(code_analysis['functions']) * 128  # ~128 bytes per function
        # LSTM functions sind komplexer
        lstm_complexity_bonus = 4096  # LSTM forward pass ist groß
        mlp_complexity_bonus = 2048   # MLP forward pass
        
        return base_size + function_size + lstm_complexity_bonus + mlp_complexity_bonus

    def compare_with_real_measurements(self, real_ram_total: int, real_ram_used: int,
                                     real_flash_total: int, real_flash_used: int,
                                     theoretical_data: Dict = None) -> Dict:
        """
        Vergleicht theoretische Berechnung mit realen Arduino Messungen
        """
        # Use provided theoretical data to avoid recalculation
        if theoretical_data is None:
            theoretical = self.calculate_theoretical_usage()
        else:
            theoretical = theoretical_data
        
        comparison = {
            'theoretical': theoretical,
            'real': {
                'sram': {
                    'total': real_ram_total,
                    'used': real_ram_used,
                    'free': real_ram_total - real_ram_used,
                    'percent': (real_ram_used / real_ram_total) * 100
                },
                'flash': {
                    'total': real_flash_total,
                    'used': real_flash_used,
                    'free': real_flash_total - real_flash_used,
                    'percent': (real_flash_used / real_flash_total) * 100
                }
            },
            'differences': {
                'sram_diff': abs(theoretical['sram']['total'] - real_ram_used),
                'sram_diff_percent': abs(theoretical['sram']['percent'] - 
                                       (real_ram_used / real_ram_total) * 100),
                'flash_diff': abs(theoretical['flash']['total'] - real_flash_used),
                'flash_diff_percent': abs(theoretical['flash']['percent'] - 
                                        (real_flash_used / real_flash_total) * 100)
            }
        }
        
        return comparison

    def generate_report(self, comparison_data: Dict = None, save_path: Optional[str] = None) -> str:
        """
        Generiert detaillierten Memory Analysis Report
        """
        if comparison_data is None:
            theoretical = self.calculate_theoretical_usage()
            comparison_data = {'theoretical': theoretical}
        
        report = f"""
# Arduino Memory Usage Analysis Report
{'='*50}

## Target Hardware
- **Board**: {self.arduino_specs.name}
- **SRAM**: {self.arduino_specs.sram_kb} KB
- **Flash**: {self.arduino_specs.flash_kb} KB  
- **CPU**: {self.arduino_specs.cpu_mhz} MHz
- **Processing Mode**: {self.processing_mode.value}

## Theoretische Analyse
### SRAM Usage
"""
        
        sram = comparison_data['theoretical']['sram']
        report += f"""
- **Variables**: {sram['variables']:,} Bytes
- **Arrays**: {sram['arrays']:,} Bytes  
- **Buffers**: {sram['buffers']:,} Bytes
- **Stack**: {sram['stack']:,} Bytes
- **Total**: {sram['total']:,} Bytes ({sram['percent']:.1f}%)
"""
        
        flash = comparison_data['theoretical']['flash']
        report += f"""
### Flash Usage
- **LSTM Weights**: {flash['weights']:,} Bytes
- **Code**: {flash['code']:,} Bytes
- **Constants**: {flash['constants']:,} Bytes  
- **Total**: {flash['total']:,} Bytes ({flash['percent']:.1f}%)

### Model Information
- **Total Weights**: {comparison_data['theoretical']['weights_count']:,}
- **Architecture**: LSTM(4→32) + MLP(32→32→32→1)
"""
        
        if 'real' in comparison_data:
            real = comparison_data['real']
            diff = comparison_data['differences']
            
            report += f"""
## Real Arduino Measurements
### SRAM (Real)
- **Total**: {real['sram']['total']:,} Bytes
- **Used**: {real['sram']['used']:,} Bytes ({real['sram']['percent']:.1f}%)
- **Free**: {real['sram']['free']:,} Bytes

### Flash (Real)  
- **Total**: {real['flash']['total']:,} Bytes
- **Used**: {real['flash']['used']:,} Bytes ({real['flash']['percent']:.1f}%)
- **Free**: {real['flash']['free']:,} Bytes

## Theoretical vs Real Comparison
### SRAM Accuracy
- **Difference**: {diff['sram_diff']:,} Bytes ({diff['sram_diff_percent']:.1f}%)
- **Accuracy**: {'✅ SEHR GUT' if diff['sram_diff_percent'] < 10 else '⚠️ MODERAT' if diff['sram_diff_percent'] < 25 else '❌ SCHLECHT'}

### Flash Accuracy  
- **Difference**: {diff['flash_diff']:,} Bytes ({diff['flash_diff_percent']:.1f}%)
- **Accuracy**: {'✅ SEHR GUT' if diff['flash_diff_percent'] < 15 else '⚠️ MODERAT' if diff['flash_diff_percent'] < 30 else '❌ SCHLECHT'}
"""
        
        report += f"""
## Memory Optimization Recommendations
"""
        
        # Recommendations basierend auf Usage
        if sram['percent'] > 80:
            report += "- ⚠️ **SRAM KRITISCH**: Erwäge LSTM State Komprimierung\n"
        elif sram['percent'] > 60:
            report += "- 🔶 **SRAM HOCH**: Optimiere Array-Größen\n"
        else:
            report += "- ✅ **SRAM OK**: Ausreichend Speicher verfügbar\n"
            
        if flash['percent'] > 90:
            report += "- ⚠️ **FLASH KRITISCH**: Gewichts-Quantisierung notwendig\n"
        elif flash['percent'] > 70:
            report += "- 🔶 **FLASH HOCH**: Erwäge Model-Komprimierung\n"
        else:
            report += "- ✅ **FLASH OK**: Ausreichend Speicher verfügbar\n"
        
        report += f"""
---
*Report generiert: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Calculator Version: 1.0.0*
"""
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            logger.info(f"Report gespeichert: {save_path}")
        
        return report


def main():
    """
    Hauptfunktion für Memory Analysis
    """
    # Pfade definieren
    base_path = Path(r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino_ready\Arduino_MCU_Check")
    
    ino_path = base_path / "arduino_lstm_soc_full32_with_monitoring" / "arduino_lstm_soc_full32_with_monitoring.ino"
    weights_path = base_path / "arduino_lstm_soc_full32_with_monitoring" / "lstm_weights.h"
    
    # Calculator initialisieren
    calculator = ArduinoMemoryCalculator(
        ino_path=str(ino_path),
        weights_path=str(weights_path),
        arduino_type="UNO_R4_WIFI",
        processing_mode=ProcessingMode.STATEFUL
    )
    
    try:
        # Theoretische Analyse
        logger.info("Starte theoretische Memory Analyse...")
        theoretical_results = calculator.calculate_theoretical_usage()
        
        # Report generieren
        report = calculator.generate_report()
        print(report)
        
        # Report speichern
        report_path = base_path / "memory_analysis_report.md"
        calculator.generate_report(save_path=str(report_path))
        
        # Beispiel für Vergleich mit realen Werten (falls verfügbar)
        example_real_values = {
            'ram_total': 32768,  # 32KB wie korrigiert
            'ram_used': 24576,   # Beispiel: 24KB verwendet
            'flash_total': 262144,  # 256KB
            'flash_used': 122880    # Beispiel: 120KB verwendet
        }
        logger.info("Führe Vergleich mit Beispiel-Messwerten durch...")
        comparison = calculator.compare_with_real_measurements(
            real_ram_total=example_real_values['ram_total'],
            real_ram_used=example_real_values['ram_used'], 
            real_flash_total=example_real_values['flash_total'],
            real_flash_used=example_real_values['flash_used'],
            theoretical_data=theoretical_results  # Pass existing data to avoid recalculation
        )
        
        # Vergleichsreport
        comparison_report = calculator.generate_report(comparison)
        print("\n" + "="*60)
        print("VERGLEICH MIT BEISPIEL-MESSWERTEN:")
        print("="*60)
        print(comparison_report)
        
        # Vergleichsreport speichern
        comparison_report_path = base_path / "memory_comparison_report.md"
        calculator.generate_report(comparison, save_path=str(comparison_report_path))
        
        return calculator, theoretical_results, comparison
        
    except Exception as e:
        logger.error(f"Fehler bei Memory Analyse: {e}")
        raise


if __name__ == "__main__":
    main()
