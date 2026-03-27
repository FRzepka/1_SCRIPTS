# 🚀 ARDUINO LSTM SOC PREDICTION - DEPLOYMENT GUIDE

**Quick Start Guide for Arduino Hardware SOC Prediction**

---

## 📋 **PRE-DEPLOYMENT CHECKLIST**

### 1️⃣ **Hardware Setup**
- [ ] Arduino board connected to PC
- [ ] Upload `arduino_lstm_soc_full32.ino` to Arduino
- [ ] Verify COM port (default: COM13)
- [ ] Ensure Arduino has 32KB+ memory

### 2️⃣ **Software Dependencies**
```powershell
# Anaconda ML environment required
conda activate ml

# Required libraries (should be installed):
# - torch, matplotlib, pandas, scikit-learn, pyserial
```

### 3️⃣ **Data Files**
- [ ] Ensure `df.parquet` data file is accessible
- [ ] Verify scaler file paths in scripts
- [ ] Check model weight files are present

---

## 🎯 **QUICK START COMMANDS**

### **Run Arduino Live Prediction**
```powershell
cd "c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino_ready\Arduino_Version"
python arduino_live_soc_prediction.py
```

### **Run Validation Test**
```powershell
cd "c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino_ready\Arduino_Version"
python validate_arduino_vs_pc.py
```

### **Run Communication Test Only**
```powershell
cd "c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino_ready\Arduino_Version"
python arduino_comm_optimized.py
```

---

## ⚡ **EXPECTED PERFORMANCE**

### **Target Metrics**
- **MAE**: < 0.08 (Excellent if < 0.07)
- **Sample Rate**: > 60 samples/second
- **Communication**: < 0.5ms average
- **Memory Usage**: < 28KB on Arduino

### **Troubleshooting**
- **Arduino not responding**: Reset Arduino board, wait 2-3 seconds
- **COM port error**: Check Windows Device Manager for correct port
- **Import errors**: Ensure you're in correct directory with all files
- **Memory issues**: Verify Arduino has sufficient memory (32KB recommended)

---

## 📊 **LIVE MONITORING FEATURES**

### **Real-time Plots**
1. **SOC Prediction**: Arduino LSTM vs True SOC values
2. **MAE Evolution**: Running Mean Absolute Error
3. **Input Signals**: Voltage and Current measurements  
4. **Communication**: Response time monitoring

### **Performance Metrics**
- Sample processing rate (samples/second)
- Communication latency (milliseconds)
- Prediction accuracy (MAE, RMSE)
- Arduino memory utilization

---

## 🎯 **PRODUCTION DEPLOYMENT**

### **Integration Steps**
1. **Hardware Integration**: Connect Arduino to BMS system
2. **Data Pipeline**: Ensure real sensor data feeds to Arduino
3. **Monitoring Setup**: Configure live plotting for production use
4. **Alert System**: Implement SOC threshold alerts
5. **Data Logging**: Save predictions and metrics for analysis

### **Configuration Options**
- Adjust COM port in scripts
- Modify sample rate (currently 5ms delay)
- Change plot update frequency (currently every 5 samples)
- Configure max samples for continuous operation

---

**Deployment Guide Created**: June 1, 2025  
**Status**: Ready for Production Use ✅
