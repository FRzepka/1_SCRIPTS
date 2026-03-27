# 🔧 Arduino Hardware Monitoring Test Guide

## Phase 1: Arduino Setup

### 1.1 Upload Enhanced Firmware
```bash
# Upload arduino_lstm_soc_hardware_monitor.ino to your Arduino
# Using Arduino IDE or command line:
arduino-cli compile --fqbn arduino:avr:uno arduino_lstm_soc_hardware_monitor.ino
arduino-cli upload -p COM13 --fqbn arduino:avr:uno arduino_lstm_soc_hardware_monitor.ino
```

### 1.2 Verify Hardware Monitoring
```bash
# Test basic connection and hardware monitoring commands
python -c "
import serial
import time
ser = serial.Serial('COM13', 115200, timeout=3)
time.sleep(3)
ser.write(b'STATS\n')
print('STATS Response:', ser.readline().decode().strip())
ser.write(b'RAM\n') 
print('RAM Response:', ser.readline().decode().strip())
ser.write(b'BENCHMARK\n')
print('BENCHMARK Response:', ser.readline().decode().strip())
ser.close()
"
```

## Phase 2: Hardware Performance Collection

### 2.1 Run Hardware Monitoring
```bash
# Start real hardware monitoring (runs indefinitely)
python arduino_hardware_monitor.py
```

**Expected Output:**
- Live plots with 6 subplots showing real Arduino metrics
- Console logs with hardware performance data
- JSON export with detailed statistics

### 2.2 Collect Data for Different Durations
```bash
# Short test (5 minutes)
timeout 300 python arduino_hardware_monitor.py

# Medium test (30 minutes) 
timeout 1800 python arduino_hardware_monitor.py

# Long test (2 hours)
timeout 7200 python arduino_hardware_monitor.py
```

## Phase 3: Scientific Validation

### 3.1 Run Hardware Validation Analysis
```bash
# This compares theoretical calculations with real measurements
python hardware_validation_comparison.py
```

**This will generate:**
- ✅ Validation error analysis
- ✅ Scientific plots comparing theory vs reality
- ✅ Deployment feasibility assessment
- ✅ Hardware optimization recommendations

### 3.2 Expected Validation Metrics
- **Inference Time**: Theory vs Measured (should be within 10-20%)
- **Memory Usage**: Theory vs Actual RAM consumption
- **CPU Load**: Estimated vs Real processing load
- **Energy Analysis**: Theoretical vs Hardware-based estimates
- **Deployment Readiness**: Pass/Fail assessment

## Phase 4: Results and Optimization

### 4.1 Expected Output Files
- `arduino_hardware_analysis_YYYYMMDD_HHMMSS.json` - Raw hardware data
- `hardware_validation_YYYYMMDD_HHMMSS.json` - Validation results
- `hardware_validation_YYYYMMDD_HHMMSS.md` - Human-readable report
- Scientific plots for paper publication

### 4.2 Key Metrics to Analyze
1. **Performance Accuracy**:
   - Theoretical inference time vs measured
   - Memory predictions vs actual usage
   
2. **Hardware Efficiency**:
   - RAM utilization patterns
   - CPU load distribution
   - Memory fragmentation analysis
   
3. **Deployment Viability**:
   - Power consumption estimates
   - Real-time capability assessment
   - Resource constraint analysis

## Expected Results

Based on the theoretical model, we expect:
- **Inference Time**: ~1200-1500 μs per prediction
- **RAM Usage**: ~60-70% of Arduino's 2KB SRAM
- **CPU Load**: ~20-40% depending on sampling rate
- **Energy**: ~0.6-1.0 mJ per prediction

The hardware tests will validate these predictions and provide real-world deployment data!

## Troubleshooting

### Arduino Connection Issues
```bash
# Check available ports
python -c "import serial.tools.list_ports; print([p.device for p in serial.tools.list_ports.comports()])"

# Test basic serial communication
python -c "
import serial
ser = serial.Serial('COM13', 115200, timeout=5)
print('Arduino connected:', ser.is_open)
ser.close()
"
```

### Hardware Monitoring Not Working
1. Verify Arduino firmware uploaded correctly
2. Check serial port and baudrate settings
3. Ensure Arduino has sufficient power
4. Monitor Arduino Serial Monitor for debug output

Ready to collect real hardware performance data! 🚀
