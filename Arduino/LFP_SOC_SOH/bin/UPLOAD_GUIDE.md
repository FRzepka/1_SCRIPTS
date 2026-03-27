# Quick Arduino Upload Guide

## ✅ Current Status
- ✅ Arduino connected on COM13
- ✅ Model weights generated (lstm_weights.h)
- ✅ Arduino sketch ready (arduino_lstm_soc_v2.ino)

## 🚀 Upload Steps

### 1. Install Arduino IDE
If you don't have Arduino IDE installed:
- Download from: https://www.arduino.cc/en/software
- Install the software

### 2. Install Required Library
1. Open Arduino IDE
2. Go to **Tools → Manage Libraries**
3. Search for "**ArduinoJson**"
4. Install "ArduinoJson by Benoit Blanchon"

### 3. Configure Arduino IDE
1. Connect your Arduino to COM13 (already connected ✅)
2. In Arduino IDE:
   - **Tools → Board** → Select your board type:
     - If ESP32: Select "ESP32 Dev Module" 
     - If Arduino Mega: Select "Arduino Mega or Mega 2560"
     - If Arduino Uno: ⚠️ **NOT RECOMMENDED** (insufficient memory)
   - **Tools → Port** → Select "COM13"

### 4. Open and Upload Code
1. Open file: `arduino_lstm_soc_v2.ino`
2. Verify `lstm_weights.h` is in the same folder ✅
3. Click **Upload** button (→) in Arduino IDE
4. Wait for compilation and upload to complete

### 5. Verify Upload
1. Open **Tools → Serial Monitor**
2. Set baud rate to **115200**
3. You should see:
   ```
   🤖 Arduino LSTM SOC Predictor v2.0
   ⚡ Using real PyTorch weights!
   📊 Model: 4x8 LSTM, 10 sequence length
   💾 SRAM usage: ~XXXX bytes
   ✅ Neural network ready!
   📡 Waiting for data from PC...
   ```

## 🔧 If Upload Fails

### Memory Issues (Arduino Uno/Nano)
If you get "sketch too large" error:
1. Use ESP32 or Arduino Mega instead
2. Or reduce model size by editing `arduino_lstm_soc_v2.ino`:
   ```cpp
   const int TARGET_HIDDEN_SIZE = 4;   // Reduce from 8
   const int SEQUENCE_LENGTH = 5;      // Reduce from 10
   ```

### Library Issues
If you get "ArduinoJson.h not found":
1. Install ArduinoJson library via Tools → Manage Libraries
2. Make sure to restart Arduino IDE after installation

### Port Issues
If COM13 is not available:
1. Run: `python arduino_setup.py --scan-ports`
2. Use the correct port shown in the scan

## ✅ After Successful Upload

Once uploaded successfully, test the system:

```bash
# Test basic connection
python hardware_validation.py --quick-test --port COM13

# If successful, run live test
python pc_arduino_interface.py
```

## 🎯 Expected Results

After upload, you should be able to:
- ✅ Connect to Arduino and see "Neural network ready!" message
- ✅ Send data and receive SOC predictions
- ✅ Achieve ~8-15ms inference times
- ✅ Get SOC predictions with ~3-5% accuracy

---

**What Arduino board are you using?** This will help me provide more specific instructions.
