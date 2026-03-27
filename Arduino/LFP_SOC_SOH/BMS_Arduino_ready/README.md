# BMS Arduino Ready - Clean Project Structure

## 🎯 Projekt Übersicht

Saubere, organisierte Implementierung des Arduino LSTM-basierten SOC Vorhersagesystems mit Ground Truth Validierung und MAE Testing.

### 📁 Ordnerstruktur

```
BMS_Arduino_ready/
├── PC_Version/
│   ├── arduino_live_soc_prediction.py    # Live SOC Vorhersage mit Ground Truth
│   ├── live_soc_results.png              # Ergebnisse Visualisierung  
│   └── live_soc_metrics.json             # Metriken Export
├── Arduino_Version/
│   ├── arduino_lstm_soc_full32.ino       # Arduino LSTM Implementation
│   ├── lstm_weights.h                    # Trainierte LSTM Gewichte
│   └── arduino_comm.py                   # Arduino Kommunikation Scripts
└── README.md                             # Diese Datei
```

## 🖥️ PC Version

### Features
- **Live SOC Vorhersage**: Real-time Kommunikation mit Arduino Hardware
- **Ground Truth Vergleich**: Validierung gegen MGFarm C19 Zelldaten  
- **MAE Berechnung**: Kontinuierliches Mean Absolute Error Monitoring
- **Daten Skalierung**: Exakt identisch mit 1.2.4 Train Script (StandardScaler)
- **Real-time Metriken**: Voltage, Current, Inference Time, Error Tracking
- **Visualisierung**: Live Plots und Export von Ergebnissen

### Verwendung
```python
python arduino_live_soc_prediction.py
```

**Was passiert:**
1. Initialisiert StandardScaler exakt wie im Training (über alle 9 Zellen)
2. Lädt C19 Ground Truth Daten und skaliert Features
3. Verbindet mit Arduino auf COM13
4. Führt Live SOC Vorhersagen durch (500 Samples, 50ms Delay)
5. Berechnet kontinuierlich MAE und Inference Times
6. Exportiert Metriken und Visualisierungen

### Technische Details
- **Features**: `["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]`
- **Target**: `"SOC_ZHU"`
- **Skalierung**: StandardScaler mit partial_fit über alle 9 Trainingszellen
- **Validierung**: MGFarm C19 Cell (gleiche Zelle wie im Training)
- **Kommunikation**: CSV Format über Serial (115200 baud)

## 🔧 Arduino Version

### Hardware Setup
- **Mikrocontroller**: Arduino kompatibel (ESP32 empfohlen)
- **LSTM Modell**: 4→32→32→32→1 (7009 Parameter, 27.38KB Memory)
- **Kommunikation**: Serial 115200 baud, reaktiver Modus

### Arduino Files

#### arduino_lstm_soc_full32.ino
Vollständige Arduino LSTM Implementation:
- LSTM Layer (4 input → 32 hidden)
- MLP Head (32→32→32→1) 
- Serial Kommunikation
- Commands: INFO, RESET, CSV-Daten

#### lstm_weights.h
Trainierte Gewichte vom Python Modell:
- LSTM Gewichte und Biases
- MLP Gewichte und Biases  
- Exakt übertragen ohne Genauigkeitsverlust

#### arduino_comm.py
Kommunikations-Scripts für PC:

**Basis Test:**
```bash
python arduino_comm.py test
```

**Interaktiver Modus:**
```bash
python arduino_comm.py interactive
```

**Kontinuierlicher Test:**
```bash
python arduino_comm.py continuous
```

## 🔬 Daten Pipeline

### 1. Training Data Scaling (1.2.4 Train Script)
```python
# StandardScaler über alle 9 Zellen
train_cells = ["C01", "C03", "C05", "C11", "C17", "C23"]
val_cells = ["C07", "C19", "C21"]

scaler = StandardScaler()
for cell in all_cells:
    scaler.partial_fit(cell_data[features])
```

### 2. Live Prediction Scaling
```python
# EXAKT gleicher Scaler für Live Daten
features = ["Voltage[V]", "Current[A]", "SOH_ZHU", "Q_c"]
scaled_data = scaler.transform(live_data[features])
```

### 3. Arduino Communication
```
Format: voltage,current,soh,q_c
Beispiel: 0.5,-0.2,1.0,0.8
Response: 0.987630
```

## 📊 Erwartete Ergebnisse

### Performance Metriken
- **MAE**: < 0.02 (Target: ~0.015)
- **RMSE**: < 0.03
- **Inference Time**: ~10-50ms pro Vorhersage
- **Communication Overhead**: ~5-10ms

### Validierung
- **Ground Truth**: MGFarm C19 Cell Data (~50,000 Datenpunkte)
- **Vergleich**: Arduino vs Python LSTM Modell
- **Tolerance**: MAE Differenz < 0.005

## 🚀 Quick Start

### 1. Hardware Setup
```bash
# Arduino mit USB verbinden (COM13)
# arduino_lstm_soc_full32.ino hochladen
```

### 2. PC Test
```bash
cd PC_Version
python arduino_live_soc_prediction.py
```

### 3. Ergebnisse
```bash
# Automatisch generiert:
# - live_soc_results.png
# - live_soc_metrics.json
```

## 🛠️ Troubleshooting

### Arduino Verbindung
```python
# Port prüfen
import serial.tools.list_ports
ports = serial.tools.list_ports.comports()
for port in ports:
    print(f"{port.device}: {port.description}")
```

### Daten Scaling
```python
# Scaler Parameter prüfen
print("Scaler scale_:", dict(zip(features, scaler.scale_)))
print("Scaler mean_:", dict(zip(features, scaler.mean_)))
```

### Memory Issues
- Arduino: 32KB SRAM minimum empfohlen
- PC: ~2GB RAM für Ground Truth Daten

## 📈 Erweiterungen

### Mögliche Verbesserungen
1. **Batch Prediction**: Multiple Samples gleichzeitig
2. **Real-time Streaming**: Kontinuierliche Datenaufnahme
3. **Web Interface**: Browser-basierte Überwachung
4. **Data Logging**: Kontinuierliche Aufzeichnung
5. **Multiple Cells**: Parallel Testing verschiedener Zellen

### Hardware Optimierungen
1. **Faster MCU**: ESP32-S3 für höhere Performance
2. **Quantization**: INT8 für schnellere Inferenz
3. **Parallel Processing**: Multi-Core für Batch Processing

## 🔄 Empfohlene Nutzung und Aufräumen

- **PC_Version/arduino_live_soc_prediction.py**: Simuliert den kompletten SOC-Live-Test auf dem PC (ohne Arduino, nur Simulation).
- **PC_Version/test_pc_version_fixed.py**: Testet den Workflow und bereitet alles für den späteren Hardware-Test vor.
- **Arduino_Version/arduino_comm.py**: Skript für den echten Arduino-Test (Kommunikation mit Arduino).

**Empfohlene Nutzung:**
1. Zuerst `test_pc_version_fixed.py` ausführen, um sicherzustellen, dass alles korrekt vorbereitet ist.
2. Dann `arduino_live_soc_prediction.py` für den PC-Live-Test (ohne Hardware).
3. Für den Hardware-Test: Skripte in `Arduino_Version` nutzen.

**Hinweis:**
Unnötige Dateien wie alte Testergebnisse, temporäre Plots oder nicht benötigte Skripte sollten entfernt werden, damit die Struktur klar bleibt.

## 🏆 Projekt Status

✅ **COMPLETED:**
- Saubere Projektstruktur
- PC Live Vorhersage System
- Arduino LSTM Implementation  
- Kommunikations Scripts
- Ground Truth Validierung
- MAE Testing Framework

🎯 **READY FOR:**
- Live Testing und Validierung
- Performance Benchmarking  
- Hardware Optimierung
- Production Deployment