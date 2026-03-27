
# Arduino Memory Usage Analysis Report
==================================================

## Target Hardware
- **Board**: Arduino Uno R4 WiFi
- **SRAM**: 32 KB
- **Flash**: 256 KB  
- **CPU**: 48 MHz
- **Processing Mode**: Stateful

## Theoretische Analyse
### SRAM Usage

- **Variables**: 144 Bytes
- **Arrays**: 1,552 Bytes  
- **Buffers**: 256 Bytes
- **Stack**: 1,216 Bytes
- **Total**: 16,992 Bytes (51.9%)

### Flash Usage
- **LSTM Weights**: 28,036 Bytes
- **Code**: 16,768 Bytes
- **Constants**: 1,024 Bytes  
- **Total**: 45,828 Bytes (17.5%)

### Model Information
- **Total Weights**: 7,009
- **Architecture**: LSTM(4→32) + MLP(32→32→32→1)

## Memory Optimization Recommendations
- ✅ **SRAM OK**: Ausreichend Speicher verfügbar
- ✅ **FLASH OK**: Ausreichend Speicher verfügbar

---
*Report generiert: 2025-06-08 14:44:01*
*Calculator Version: 1.0.0*
