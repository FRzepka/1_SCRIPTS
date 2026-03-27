# 🚀 Arduino LSTM SOC System - Hardware Testing Guide

## 📋 Pre-Testing Checklist

Before proceeding with hardware testing, ensure you have:

### ✅ **Hardware Requirements**
- [ ] **ESP32 Development Board** (recommended) OR Arduino Mega 2560
- [ ] USB Cable (compatible with your Arduino board)
- [ ] Computer with Arduino IDE installed
- [ ] Stable power supply (via USB is fine for testing)

### ✅ **Software Requirements**
- [ ] Arduino IDE (latest version)
- [ ] ArduinoJson library installed
- [ ] Python environment with required packages
- [ ] All project files in `BMS_Arduino` folder

### ✅ **Files Ready**
- [ ] `arduino_lstm_soc_v2.ino` - Main Arduino sketch
- [ ] `lstm_weights.h` - Generated model weights (17KB)
- [ ] `hardware_validation.py` - Testing script
- [ ] `monitoring_dashboard.py` - Real-time monitoring

## 🔧 Step 1: Arduino IDE Setup

### 1.1 Install Arduino IDE
```
Download from: https://arduino.cc/downloads
Install and launch Arduino IDE
```

### 1.2 Install ArduinoJson Library
```
1. Open Arduino IDE
2. Go to Tools → Manage Libraries
3. Search for "ArduinoJson"
4. Install version 6.21.0 or newer by Benoit Blanchon
```

### 1.3 Select Board and Port
```
1. Connect your Arduino/ESP32 via USB
2. Tools → Board → Select your board:
   - For ESP32: "ESP32 Dev Module" 
   - For Arduino Mega: "Arduino Mega or Mega 2560"
3. Tools → Port → Select your COM port (e.g., COM4, COM13)
```

## 📁 Step 2: Upload Arduino Code

### 2.1 Open Project
```
1. File → Open → Navigate to:
   C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino\arduino_lstm_soc_v2.ino

2. Verify that lstm_weights.h is in the same folder
```

### 2.2 Compile and Upload
```
1. Click the "Verify" button (checkmark) to compile
2. If compilation succeeds, click "Upload" button (arrow)
3. Wait for upload to complete
4. Open Serial Monitor (Tools → Serial Monitor)
5. Set baud rate to 115200
```

### 2.3 Expected Serial Output
```
Arduino LSTM SOC Predictor v2.0
Using real PyTorch weights!
Model: 4x8 LSTM, 10 sequence length
SRAM usage: ~1400 bytes
Neural network ready!
Waiting for data from PC...
```

## 🧪 Step 3: Basic Testing

### 3.1 Quick Connection Test
```powershell
# In PowerShell, navigate to BMS_Arduino folder
cd "C:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino"

# Run quick test (replace COM4 with your actual port)
python hardware_validation.py --quick-test --port COM4
```

**Expected Output:**
```
Testing Arduino Connection...
Test 1: SOC=0.532, Time=8.5ms
Test 2: SOC=0.612, Time=9.2ms  
Test 3: SOC=0.698, Time=8.8ms

Connection Test: 3/3 successful
```

### 3.2 Performance Benchmark
```powershell
# Test inference performance
python hardware_validation.py --benchmark --samples 100 --port COM4
```

**Expected Results:**
```
Performance Results:
  Mean inference time: 9.2 ± 1.5 ms
  Range: 7.5 - 12.8 ms
  LSTM avg: 10.1 ms (70 samples)
  Simple avg: 7.2 ms (30 samples)
  Throughput: 18.5 Hz
  Errors: 0/100
```

## 📊 Step 4: Real-Time Monitoring

### 4.1 Launch Monitoring Dashboard
```powershell
# Start real-time GUI monitor
python monitoring_dashboard.py --port COM4
```

This opens a GUI window with 4 real-time plots:
- SOC Predictions vs True Values
- Battery Voltage and Current
- Prediction Error
- Inference Time Performance

### 4.2 Alternative: Command Line Monitoring
```powershell
# Test with real battery data
python test_arduino_system.py --live-plot --cell MGFarm_18650_C19 --port COM4
```

## 🔍 Step 5: Comprehensive Validation

### 5.1 Full System Test
```powershell
# Run complete validation suite
python hardware_validation.py --full-test --port COM4 --samples 500 --stability 5
```

This performs:
- Connection testing
- Performance benchmarking  
- Memory usage analysis
- Accuracy validation
- 5-minute stability test

### 5.2 Expected Performance Targets

| Metric | Target | ESP32 Expected | Arduino Mega Expected |
|--------|---------|----------------|----------------------|
| **Inference Time** | <15ms | 8-12ms | 15-25ms |
| **Memory Usage** | <50% SRAM | 1.4% (7KB/520KB) | 91% (7KB/8KB) |
| **Accuracy (RMSE)** | <0.05 | 0.025-0.035 | 0.030-0.045 |
| **Communication Error Rate** | <1% | <0.1% | <0.5% |
| **Throughput** | >10 Hz | 15-20 Hz | 8-15 Hz |

## ❌ Troubleshooting

### Problem: Arduino won't connect
**Solution:**
```
1. Check COM port in Device Manager
2. Try different USB cable
3. Press Arduino reset button
4. Restart Arduino IDE
```

### Problem: Compilation errors
**Solution:**
```
1. Verify ArduinoJson library is installed
2. Check board selection matches your hardware
3. Ensure lstm_weights.h is in sketch folder
4. Try Arduino IDE restart
```

### Problem: Upload fails
**Solution:**
```
1. Close Serial Monitor if open
2. Check correct port selection
3. Press Arduino reset button during upload
4. Try lower upload speed (Tools → Upload Speed)
```

### Problem: No serial output
**Solution:**
```
1. Check baud rate is 115200
2. Try unplugging and reconnecting USB
3. Press Arduino reset button
4. Check power LED is on
```

### Problem: Poor inference performance
**Solution:**
```
1. ESP32: Should get 8-12ms inference time
2. Arduino Mega: 15-25ms expected (due to slower clock)
3. If much slower, check for memory issues
4. Reduce model size if needed
```

### Problem: Memory issues (Arduino Mega)
**Solution:**
```
Edit arduino_lstm_soc_v2.ino:
- Reduce TARGET_HIDDEN_SIZE from 8 to 4
- Reduce SEQUENCE_LENGTH from 10 to 5
- Recompile and upload
```

### Problem: Accuracy is poor
**Solution:**
```
1. Verify model weights loaded correctly
2. Check data scaling matches training
3. Run simple_model_inspector.py to debug
4. Ensure sequence length >= 10 for LSTM mode
```

## 📈 Performance Optimization

### For ESP32 (Recommended)
```cpp
// In arduino_lstm_soc_v2.ino - these are optimal settings
const int TARGET_HIDDEN_SIZE = 8;   // Full performance
const int SEQUENCE_LENGTH = 10;     // Good accuracy
```

### For Arduino Mega (Memory Constrained)
```cpp
// Reduce for memory limitations
const int TARGET_HIDDEN_SIZE = 4;   // Reduced memory
const int SEQUENCE_LENGTH = 5;      // Minimum for LSTM
```

### For Arduino Uno (Not Recommended)
```
Arduino Uno has insufficient SRAM (2KB) for this model.
Consider using ESP32 or significantly reduce model complexity.
```

## ✅ Success Criteria

### **Minimum Success (Arduino Working)**
- ✅ Arduino boots and shows "Neural network ready!"
- ✅ Communication established via Serial
- ✅ Inference time <25ms
- ✅ SOC predictions in valid range (0-1)
- ✅ <5% communication errors

### **Good Performance (Production Ready)**
- ✅ Inference time <15ms
- ✅ RMSE <0.04 (4% SOC error)
- ✅ <1% communication errors
- ✅ Stable operation >30 minutes
- ✅ Real-time visualization working

### **Excellent Performance (Research Quality)**
- ✅ Inference time <10ms (ESP32)
- ✅ RMSE <0.03 (3% SOC error)
- ✅ <0.1% communication errors
- ✅ 24/7 stable operation
- ✅ Matches PyTorch accuracy within 5%

## 📋 Test Report Template

After testing, document your results:

```
=== ARDUINO LSTM SOC SYSTEM TEST REPORT ===
Date: ___________
Hardware: ___________
Arduino IDE Version: ___________

BASIC FUNCTIONALITY:
[ ] Arduino boots successfully
[ ] Serial communication working
[ ] Model weights loaded
[ ] Inference running

PERFORMANCE METRICS:
- Mean inference time: _____ ms
- Memory usage: _____ KB / _____ KB
- Communication errors: _____ %
- RMSE accuracy: _____

STABILITY TEST:
- Test duration: _____ minutes
- Total samples: _____
- Error rate: _____ %

OVERALL RESULT: [ ] PASS / [ ] FAIL
Notes: _________________________________
```

## 🎯 Next Steps After Successful Testing

1. **Integration Planning**
   - Plan CAN bus or I2C integration with actual BMS
   - Design enclosure for production deployment
   - Consider power optimization

2. **Model Optimization**
   - Fine-tune model size vs accuracy trade-off
   - Implement quantization for better performance
   - Add temperature compensation

3. **Production Features**
   - Add watchdog timer for reliability
   - Implement data logging
   - Add wireless monitoring capability

4. **Validation**
   - Test with real battery packs
   - Long-term accuracy validation
   - Temperature and vibration testing

---

**Good luck with your hardware testing!** 🚀

The system is well-prepared and should work smoothly on ESP32 hardware. If you encounter any issues, refer to the troubleshooting section or run the diagnostic scripts for more detailed analysis.
