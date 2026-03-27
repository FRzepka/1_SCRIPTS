"""
Working Arduino MAE Test - Handles current Arduino protocol issues
"""

import serial
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import threading
import queue
from datetime import datetime

class ArduinoMAETest:
    def __init__(self, port='COM13', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.scaler = None
        self.ground_truth_data = None
        self.current_index = 0
        
        # Results storage
        self.predictions = []
        self.ground_truth_values = []
        self.timestamps = []
        self.errors = []
        self.communication_errors = 0
        
        # For real-time plotting
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Arduino LSTM SOC Prediction - Live MAE Test', fontsize=16)
        
        # Plot setup
        self.axes[0,0].set_title('SOC Predictions vs Ground Truth')
        self.axes[0,0].set_ylabel('SOC')
        self.axes[0,0].grid(True)
        
        self.axes[0,1].set_title('Absolute Error')
        self.axes[0,1].set_ylabel('|Error|')
        self.axes[0,1].grid(True)
        
        self.axes[1,0].set_title('Rolling MAE (Last 50 predictions)')
        self.axes[1,0].set_ylabel('MAE')
        self.axes[1,0].grid(True)
        
        self.axes[1,1].set_title('Input Signals')
        self.axes[1,1].set_ylabel('Voltage (V)')
        self.axes[1,1].grid(True)
        
        plt.tight_layout()
        
    def load_ground_truth_data(self):
        """Load and prepare ground truth data from C19 cell"""
        print("📊 Loading ground truth data from MGFarm C19...")
        
        # Load C19 data
        c19_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU\MGFarm_18650_C19\df.parquet"
        df_c19 = pd.read_parquet(c19_path)
        
        print(f"✅ Loaded {len(df_c19)} data points from C19")
        
        # Prepare features and target
        features = ['voltage', 'current', 'soh', 'q_c']
        target = 'soc'
        
        # Create scaler using the same procedure as training
        print("🔧 Creating StandardScaler...")
        
        # Load all training cells for scaler creation
        cell_names = ['C01', 'C03', 'C05', 'C11', 'C17', 'C23', 'C07', 'C19', 'C21']
        all_data = []
        
        base_path = r"c:\Users\Florian\SynologyDrive\TUB\3_Projekte\MG_Farm\5_Data\01_LFP\00_Data\MGFarm_18650_Dataframes_ZHU"
        
        for cell in cell_names:
            try:
                cell_path = f"{base_path}\MGFarm_18650_{cell}\df.parquet"
                df_cell = pd.read_parquet(cell_path)
                all_data.append(df_cell[features])
            except Exception as e:
                print(f"⚠️ Could not load {cell}: {e}")
        
        # Combine all data for scaler
        combined_data = pd.concat(all_data, ignore_index=True)
        
        # Create and fit scaler
        self.scaler = StandardScaler()
        self.scaler.fit(combined_data)
        
        print(f"✅ Scaler fitted on {len(combined_data)} samples from {len(all_data)} cells")
        print(f"   Mean: {self.scaler.mean_}")
        print(f"   Scale: {self.scaler.scale_}")
        
        # Prepare C19 test data
        X_c19 = df_c19[features].values
        y_c19 = df_c19[target].values
        
        # Scale the features
        X_c19_scaled = self.scaler.transform(X_c19)
        
        # Store for testing
        self.ground_truth_data = {\n            'features_raw': X_c19,\n            'features_scaled': X_c19_scaled,\n            'soc': y_c19,\n            'timestamps': df_c19.index if hasattr(df_c19.index, 'to_pydatetime') else range(len(df_c19))\n        }\n        \n        print(f\"✅ Prepared {len(X_c19)} test samples\")\n        print(f\"   SOC range: {y_c19.min():.3f} - {y_c19.max():.3f}\")\n        print(f\"   Voltage range: {X_c19[:, 0].min():.3f} - {X_c19[:, 0].max():.3f} V\")\n        \n    def connect_arduino(self):\n        \"\"\"Connect to Arduino and initialize\"\"\"\n        print(f\"🔌 Connecting to Arduino on {self.port}...\")\n        \n        try:\n            self.arduino = serial.Serial(self.port, self.baudrate, timeout=2)\n            time.sleep(3)  # Give Arduino time to initialize\n            \n            print(\"✅ Connected to Arduino\")\n            \n            # Clear any startup messages\n            self.arduino.flushInput()\n            time.sleep(0.5)\n            \n            # Send RESET command\n            print(\"🔄 Resetting Arduino LSTM states...\")\n            self.arduino.write(b'RESET\\n')\n            time.sleep(0.5)\n            \n            # Read reset response\n            response = self.arduino.readline().decode().strip()\n            print(f\"Reset response: {response}\")\n            \n            return True\n            \n        except Exception as e:\n            print(f\"❌ Failed to connect to Arduino: {e}\")\n            return False\n    \n    def get_arduino_prediction(self, voltage, current, soh, q_c, max_attempts=3):\n        \"\"\"Get SOC prediction from Arduino with robust error handling\"\"\"\n        \n        for attempt in range(max_attempts):\n            try:\n                # Clear input buffer\n                self.arduino.flushInput()\n                \n                # Send data\n                data_str = f\"{voltage},{current},{soh},{q_c}\\n\"\n                self.arduino.write(data_str.encode())\n                time.sleep(0.3)  # Give Arduino time to process\n                \n                # Collect all responses\n                responses = []\n                for _ in range(15):  # Read up to 15 lines\n                    if self.arduino.in_waiting:\n                        try:\n                            line = self.arduino.readline().decode().strip()\n                            if line:\n                                responses.append(line)\n                        except:\n                            continue\n                    else:\n                        break\n                \n                # Look for a valid SOC prediction (float between 0 and 1)\n                for response in responses:\n                    try:\n                        soc_val = float(response)\n                        if 0.0 <= soc_val <= 1.0:\n                            return soc_val\n                    except ValueError:\n                        continue\n                \n                # If no valid SOC found, try once more\n                if attempt < max_attempts - 1:\n                    time.sleep(0.2)\n                    continue\n                else:\n                    print(f\"⚠️ No valid SOC found in responses: {responses[:3]}\")\n                    return None\n                    \n            except Exception as e:\n                print(f\"⚠️ Arduino communication error (attempt {attempt+1}): {e}\")\n                if attempt < max_attempts - 1:\n                    time.sleep(0.5)\n                    continue\n                else:\n                    return None\n        \n        return None\n    \n    def run_mae_test(self, num_samples=100, update_interval=10):\n        \"\"\"Run the live MAE test\"\"\"\n        print(f\"\\n🚀 Starting Arduino MAE Test with {num_samples} samples...\")\n        print(\"📊 Press Ctrl+C to stop the test early\")\n        print(\"🔌 Unplug the Arduino cable to test disconnection detection\")\n        print(\"-\" * 60)\n        \n        try:\n            start_time = time.time()\n            \n            for i in range(num_samples):\n                # Get current test sample\n                if self.current_index >= len(self.ground_truth_data['soc']):\n                    self.current_index = 0  # Wrap around\n                \n                # Get raw features for Arduino\n                voltage = self.ground_truth_data['features_raw'][self.current_index, 0]\n                current = self.ground_truth_data['features_raw'][self.current_index, 1]\n                soh = self.ground_truth_data['features_raw'][self.current_index, 2]\n                q_c = self.ground_truth_data['features_raw'][self.current_index, 3]\n                \n                # Get ground truth SOC\n                true_soc = self.ground_truth_data['soc'][self.current_index]\n                \n                # Get Arduino prediction\n                pred_soc = self.get_arduino_prediction(voltage, current, soh, q_c)\n                \n                current_time = time.time() - start_time\n                \n                if pred_soc is not None:\n                    # Calculate error\n                    error = abs(pred_soc - true_soc)\n                    \n                    # Store results\n                    self.predictions.append(pred_soc)\n                    self.ground_truth_values.append(true_soc)\n                    self.timestamps.append(current_time)\n                    self.errors.append(error)\n                    \n                    # Print progress\n                    if (i + 1) % update_interval == 0:\n                        current_mae = np.mean(self.errors[-50:])  # MAE of last 50\n                        print(f\"[{i+1:3d}/{num_samples}] SOC: {pred_soc:.4f} | True: {true_soc:.4f} | Error: {error:.4f} | MAE-50: {current_mae:.4f}\")\n                        \n                        # Update plots\n                        self.update_plots()\n                        \n                else:\n                    self.communication_errors += 1\n                    print(f\"[{i+1:3d}/{num_samples}] ❌ Communication error (total: {self.communication_errors})\")\n                    \n                    # If too many communication errors, try to reconnect\n                    if self.communication_errors > 5:\n                        print(\"🔌 Too many communication errors, trying to reconnect...\")\n                        self.arduino.close()\n                        time.sleep(2)\n                        if not self.connect_arduino():\n                            print(\"❌ Failed to reconnect, stopping test\")\n                            break\n                        self.communication_errors = 0\n                \n                self.current_index += 1\n                time.sleep(0.1)  # Small delay between predictions\n                \n        except KeyboardInterrupt:\n            print(\"\\n⏹️ Test stopped by user\")\n        except Exception as e:\n            print(f\"\\n❌ Test failed: {e}\")\n        \n        # Final results\n        self.show_final_results()\n    \n    def update_plots(self):\n        \"\"\"Update real-time plots\"\"\"\n        if len(self.predictions) < 2:\n            return\n            \n        try:\n            # Clear axes\n            for ax in self.axes.flat:\n                ax.clear()\n            \n            # Plot 1: SOC Predictions vs Ground Truth\n            self.axes[0,0].plot(self.timestamps, self.predictions, 'b-', label='Arduino Predictions', alpha=0.7)\n            self.axes[0,0].plot(self.timestamps, self.ground_truth_values, 'r-', label='Ground Truth', alpha=0.7)\n            self.axes[0,0].set_title('SOC Predictions vs Ground Truth')\n            self.axes[0,0].set_ylabel('SOC')\n            self.axes[0,0].legend()\n            self.axes[0,0].grid(True)\n            \n            # Plot 2: Absolute Error\n            self.axes[0,1].plot(self.timestamps, self.errors, 'orange', label='Absolute Error')\n            self.axes[0,1].set_title('Absolute Error')\n            self.axes[0,1].set_ylabel('|Error|')\n            self.axes[0,1].grid(True)\n            \n            # Plot 3: Rolling MAE\n            if len(self.errors) >= 10:\n                rolling_mae = []\n                window_size = min(50, len(self.errors))\n                for i in range(window_size-1, len(self.errors)):\n                    mae = np.mean(self.errors[i-window_size+1:i+1])\n                    rolling_mae.append(mae)\n                \n                mae_times = self.timestamps[window_size-1:]\n                self.axes[1,0].plot(mae_times, rolling_mae, 'g-', label=f'Rolling MAE (window={window_size})')\n                self.axes[1,0].set_title(f'Rolling MAE (Last {window_size} predictions)')\n                self.axes[1,0].set_ylabel('MAE')\n                self.axes[1,0].grid(True)\n            \n            # Plot 4: Input signals\n            voltages = [self.ground_truth_data['features_raw'][i-len(self.predictions)+j, 0] \n                       for j in range(len(self.predictions))]\n            self.axes[1,1].plot(self.timestamps, voltages, 'purple', label='Voltage')\n            self.axes[1,1].set_title('Input Voltage')\n            self.axes[1,1].set_ylabel('Voltage (V)')\n            self.axes[1,1].set_xlabel('Time (s)')\n            self.axes[1,1].grid(True)\n            \n            plt.tight_layout()\n            plt.pause(0.01)\n            \n        except Exception as e:\n            print(f\"Plot update error: {e}\")\n    \n    def show_final_results(self):\n        \"\"\"Show final test results\"\"\"\n        print(\"\\n\" + \"=\"*60)\n        print(\"🎯 ARDUINO MAE TEST RESULTS\")\n        print(\"=\"*60)\n        \n        if len(self.predictions) > 0:\n            mae = np.mean(self.errors)\n            rmse = np.sqrt(np.mean(np.array(self.errors)**2))\n            max_error = np.max(self.errors)\n            \n            print(f\"📊 Total Predictions: {len(self.predictions)}\")\n            print(f\"❌ Communication Errors: {self.communication_errors}\")\n            print(f\"📈 Success Rate: {len(self.predictions)/(len(self.predictions)+self.communication_errors)*100:.1f}%\")\n            print(f\"\\n🎯 ACCURACY METRICS:\")\n            print(f\"   MAE (Mean Absolute Error): {mae:.6f}\")\n            print(f\"   RMSE (Root Mean Square Error): {rmse:.6f}\")\n            print(f\"   Maximum Error: {max_error:.6f}\")\n            print(f\"   SOC Range: {np.min(self.ground_truth_values):.3f} - {np.max(self.ground_truth_values):.3f}\")\n            \n            # Show final plot\n            self.update_plots()\n            plt.show()\n            \n        else:\n            print(\"❌ No successful predictions recorded!\")\n            \n        print(\"=\"*60)\n    \n    def close(self):\n        \"\"\"Clean up resources\"\"\"\n        if self.arduino:\n            self.arduino.close()\n            print(\"🔌 Arduino connection closed\")\n\ndef main():\n    # Create MAE test instance\n    mae_test = ArduinoMAETest()\n    \n    try:\n        # Load ground truth data\n        mae_test.load_ground_truth_data()\n        \n        # Connect to Arduino\n        if not mae_test.connect_arduino():\n            return\n        \n        # Run the test\n        mae_test.run_mae_test(num_samples=100, update_interval=5)\n        \n    except Exception as e:\n        print(f\"❌ Test failed: {e}\")\n    finally:\n        mae_test.close()\n\nif __name__ == \"__main__\":\n    main()
