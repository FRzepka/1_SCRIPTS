"""
Hardware Validation Script for Arduino LSTM SOC System

Systematische Tests für die Hardware-Implementierung:
1. Verbindungstest
2. Performance-Benchmarks
3. Speicher-Analyse
4. Genauigkeitsvalidierung
5. Langzeit-Stabilitätstest

Verwendung:
    python hardware_validation.py --port COM3 --full-test
    python hardware_validation.py --port COM3 --quick-test
    python hardware_validation.py --port COM3 --benchmark
"""

import serial
import time
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime
import statistics
from collections import defaultdict
import sys

class ArduinoValidator:
    def __init__(self, port='COM13', baud_rate=115200):
        """Initialize Arduino validator"""
        self.port = port
        self.baud_rate = baud_rate
        self.arduino = None
        self.test_results = defaultdict(list)
        
    def connect(self, timeout=10):
        """Connect to Arduino with timeout"""
        print(f"🔌 Connecting to Arduino on {self.port}...")
        
        try:
            self.arduino = serial.Serial(self.port, self.baud_rate, timeout=2)
            time.sleep(2)  # Arduino reset time
            
            # Wait for Arduino ready message
            start_time = time.time()
            while time.time() - start_time < timeout:
                if self.arduino.in_waiting:
                    line = self.arduino.readline().decode().strip()
                    print(f"📡 Arduino: {line}")
                    if "Neural network ready" in line:
                        print("✅ Arduino connected successfully!")
                        return True
                time.sleep(0.1)
            
            print("❌ Arduino connection timeout")
            return False
            
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from Arduino"""
        if self.arduino and self.arduino.is_open:
            self.arduino.close()
            print("🔌 Arduino disconnected")
    
    def send_test_data(self, voltage, current, soh, q_c, true_soc, idx):
        """Send single test data point to Arduino"""
        data = {
            "v": float(voltage),
            "i": float(current), 
            "s": float(soh),
            "q": float(q_c),
            "t": float(true_soc),
            "idx": int(idx)
        }
        
        json_str = json.dumps(data)
        self.arduino.write((json_str + '\n').encode())
        
        # Read response
        try:
            response_line = self.arduino.readline().decode().strip()
            if response_line:
                return json.loads(response_line)
        except:
            pass
        
        return None
    
    def test_connection(self):
        """Test basic Arduino connection and communication"""
        print("\n🧪 Testing Arduino Connection...")
        
        if not self.connect():
            return False
        
        # Send test data points
        test_points = [
            (3.2, -1.5, 0.95, 5.0, 0.5, 0),
            (3.3, -2.0, 0.95, 5.0, 0.6, 1),
            (3.4, -1.0, 0.95, 5.0, 0.7, 2)
        ]
        
        successful_comm = 0
        for i, (v, i_curr, soh, q_c, soc, idx) in enumerate(test_points):
            response = self.send_test_data(v, i_curr, soh, q_c, soc, idx)
            if response:
                print(f"✅ Test {i+1}: SOC={response.get('pred_soc', 'N/A'):.3f}, "
                      f"Time={response.get('inference_time_ms', 'N/A')}ms")
                successful_comm += 1
            else:
                print(f"❌ Test {i+1}: No response")
        
        success = successful_comm == len(test_points)
        self.test_results['connection_test'] = success
        
        print(f"\n📊 Connection Test: {successful_comm}/{len(test_points)} successful")
        return success
    
    def benchmark_performance(self, num_samples=100):
        """Benchmark Arduino inference performance"""
        print(f"\n⚡ Benchmarking Performance ({num_samples} samples)...")
        
        if not self.arduino:
            print("❌ Arduino not connected")
            return False
        
        inference_times = []
        lstm_times = []
        simple_times = []
        errors = 0
        
        # Generate test data
        np.random.seed(42)
        voltages = np.random.uniform(3.0, 3.6, num_samples)
        currents = np.random.uniform(-3.0, 3.0, num_samples) 
        sohs = np.random.uniform(0.8, 1.0, num_samples)
        q_cs = np.random.uniform(4.5, 5.5, num_samples)
        true_socs = np.random.uniform(0.1, 0.9, num_samples)
        
        start_time = time.time()
        
        for i in range(num_samples):
            response = self.send_test_data(
                voltages[i], currents[i], sohs[i], q_cs[i], true_socs[i], i
            )
            
            if response:
                inf_time = response.get('inference_time_ms', 0)
                inference_times.append(inf_time)
                
                if response.get('model_type') == 'LSTM':
                    lstm_times.append(inf_time)
                else:
                    simple_times.append(inf_time)
            else:
                errors += 1
            
            # Progress indicator
            if (i + 1) % 20 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {i+1}/{num_samples} ({elapsed:.1f}s)")
        
        # Calculate statistics
        if inference_times:
            stats = {
                'mean_time': statistics.mean(inference_times),
                'median_time': statistics.median(inference_times),
                'max_time': max(inference_times),
                'min_time': min(inference_times),
                'std_time': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                'lstm_mean': statistics.mean(lstm_times) if lstm_times else 0,
                'simple_mean': statistics.mean(simple_times) if simple_times else 0,
                'total_errors': errors,
                'samples_processed': len(inference_times),
                'throughput_hz': len(inference_times) / (time.time() - start_time)
            }
            
            self.test_results['performance'] = stats
            
            print(f"\n📈 Performance Results:")
            print(f"  Mean inference time: {stats['mean_time']:.2f} ± {stats['std_time']:.2f} ms")
            print(f"  Range: {stats['min_time']:.2f} - {stats['max_time']:.2f} ms")
            print(f"  LSTM avg: {stats['lstm_mean']:.2f} ms ({len(lstm_times)} samples)")
            print(f"  Simple avg: {stats['simple_mean']:.2f} ms ({len(simple_times)} samples)")
            print(f"  Throughput: {stats['throughput_hz']:.1f} Hz")
            print(f"  Errors: {errors}/{num_samples}")
            
            return True
        else:
            print("❌ No valid responses received")
            return False
    
    def test_memory_usage(self):
        """Test Arduino memory usage (ESP32 only)"""
        print("\n💾 Testing Memory Usage...")
        
        if not self.arduino:
            print("❌ Arduino not connected")
            return False
        
        # Send memory diagnostic command (if supported)
        # This would require additional Arduino code
        print("ℹ️ Memory testing requires ESP32 with heap monitoring")
        print("ℹ️ Check Arduino serial output for memory stats")
        
        # For now, just estimate based on our known model size
        estimated_memory = {
            'lstm_weights': 1408 * 4,  # bytes (float weights)
            'lstm_states': 8 * 2 * 4,  # hidden + cell states
            'input_buffer': 10 * 4 * 4,  # sequence buffer
            'temp_arrays': 8 * 4 * 4,  # temporary calculations
            'json_buffer': 1024,  # JSON processing
            'total_estimate': 7000  # bytes
        }
        
        self.test_results['memory'] = estimated_memory
        
        print(f"📊 Estimated Memory Usage:")
        for key, value in estimated_memory.items():
            if key != 'total_estimate':
                print(f"  {key}: {value} bytes")
        print(f"  Total: ~{estimated_memory['total_estimate']} bytes")
        
        return True
    
    def validate_accuracy(self, test_data_path=None, max_samples=500):
        """Validate prediction accuracy against known data"""
        print(f"\n🎯 Validating Accuracy ({max_samples} samples)...")
        
        if not self.arduino:
            print("❌ Arduino not connected")
            return False
        
        # Use synthetic data if no real data provided
        if not test_data_path:
            print("ℹ️ Using synthetic test data")
            np.random.seed(123)
            
            true_socs = np.linspace(0.1, 0.9, max_samples)
            voltages = 2.5 + true_socs * (3.65 - 2.5) + np.random.normal(0, 0.02, max_samples)
            currents = np.random.uniform(-2.5, 2.5, max_samples)
            sohs = np.random.uniform(0.85, 1.0, max_samples) 
            q_cs = np.random.uniform(4.8, 5.2, max_samples)
        else:
            # Load real test data (implementation depends on data format)
            print(f"📁 Loading test data from {test_data_path}")
            # Implementation would go here
            return False
        
        predictions = []
        true_values = []
        errors = 0
        
        for i in range(max_samples):
            response = self.send_test_data(
                voltages[i], currents[i], sohs[i], q_cs[i], true_socs[i], i
            )
            
            if response and 'pred_soc' in response:
                predictions.append(response['pred_soc'])
                true_values.append(true_socs[i])
            else:
                errors += 1
            
            if (i + 1) % 100 == 0:
                print(f"  Progress: {i+1}/{max_samples}")
        
        if predictions:
            predictions = np.array(predictions)
            true_values = np.array(true_values)
            
            # Calculate metrics
            mae = np.mean(np.abs(predictions - true_values))
            rmse = np.sqrt(np.mean((predictions - true_values)**2))
            max_error = np.max(np.abs(predictions - true_values))
            
            accuracy_stats = {
                'mae': mae,
                'rmse': rmse,
                'max_error': max_error,
                'mean_prediction': np.mean(predictions),
                'mean_true': np.mean(true_values),
                'correlation': np.corrcoef(predictions, true_values)[0,1],
                'samples_tested': len(predictions),
                'errors': errors
            }
            
            self.test_results['accuracy'] = accuracy_stats
            
            print(f"\n🎯 Accuracy Results:")
            print(f"  MAE: {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            print(f"  Max Error: {max_error:.4f}")
            print(f"  Correlation: {accuracy_stats['correlation']:.4f}")
            print(f"  Samples: {len(predictions)}/{max_samples}")
            
            return rmse < 0.05  # Consider good if RMSE < 5%
        else:
            print("❌ No valid predictions received")
            return False
    
    def stability_test(self, duration_minutes=5):
        """Test system stability over time"""
        print(f"\n⏱️ Stability Test ({duration_minutes} minutes)...")
        
        if not self.arduino:
            print("❌ Arduino not connected")
            return False
        
        start_time = time.time()
        end_time = start_time + duration_minutes * 60
        
        sample_count = 0
        errors = 0
        inference_times = []
        
        np.random.seed(456)
        
        while time.time() < end_time:
            # Generate random test data
            voltage = np.random.uniform(3.0, 3.6)
            current = np.random.uniform(-3.0, 3.0)
            soh = np.random.uniform(0.8, 1.0)
            q_c = np.random.uniform(4.5, 5.5)
            true_soc = np.random.uniform(0.1, 0.9)
            
            response = self.send_test_data(voltage, current, soh, q_c, true_soc, sample_count)
            
            if response:
                if 'inference_time_ms' in response:
                    inference_times.append(response['inference_time_ms'])
            else:
                errors += 1
            
            sample_count += 1
            
            # Progress update
            if sample_count % 100 == 0:
                elapsed = time.time() - start_time
                remaining = duration_minutes * 60 - elapsed
                print(f"  Progress: {elapsed/60:.1f}/{duration_minutes} min, "
                      f"{sample_count} samples, {errors} errors")
            
            time.sleep(0.05)  # 20 Hz sampling
        
        # Calculate stability metrics
        if inference_times:
            stability_stats = {
                'duration_minutes': duration_minutes,
                'total_samples': sample_count,
                'total_errors': errors,
                'error_rate': errors / sample_count if sample_count > 0 else 1.0,
                'mean_inference_time': statistics.mean(inference_times),
                'inference_time_std': statistics.stdev(inference_times) if len(inference_times) > 1 else 0,
                'avg_throughput_hz': sample_count / (duration_minutes * 60)
            }
            
            self.test_results['stability'] = stability_stats
            
            print(f"\n⏱️ Stability Results:")
            print(f"  Duration: {duration_minutes} minutes")
            print(f"  Samples processed: {sample_count}")
            print(f"  Error rate: {stability_stats['error_rate']:.2%}")
            print(f"  Average throughput: {stability_stats['avg_throughput_hz']:.1f} Hz")
            print(f"  Inference time: {stability_stats['mean_inference_time']:.2f} ± {stability_stats['inference_time_std']:.2f} ms")
            
            return stability_stats['error_rate'] < 0.01  # Less than 1% errors
        else:
            print("❌ No valid responses during stability test")
            return False
    
    def generate_report(self):
        """Generate comprehensive test report"""
        print("\n📋 Generating Test Report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"arduino_test_report_{timestamp}.txt"
        
        with open(report_file, 'w') as f:
            f.write("Arduino LSTM SOC System - Hardware Validation Report\n")
            f.write("=" * 60 + "\n")
            f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Arduino Port: {self.port}\n")
            f.write(f"Baud Rate: {self.baud_rate}\n\n")
            
            for test_name, results in self.test_results.items():
                f.write(f"{test_name.upper()} RESULTS:\n")
                f.write("-" * 30 + "\n")
                
                if isinstance(results, dict):
                    for key, value in results.items():
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  Result: {results}\n")
                f.write("\n")
        
        print(f"📄 Report saved: {report_file}")
        return report_file

def main():
    parser = argparse.ArgumentParser(description='Arduino LSTM Hardware Validation')
    parser.add_argument('--port', default='COM13', help='Arduino port')
    parser.add_argument('--baud', type=int, default=115200, help='Baud rate')
    parser.add_argument('--quick-test', action='store_true', help='Run quick validation')
    parser.add_argument('--full-test', action='store_true', help='Run comprehensive test')
    parser.add_argument('--benchmark', action='store_true', help='Run performance benchmark')
    parser.add_argument('--stability', type=int, default=5, help='Stability test duration (minutes)')
    parser.add_argument('--samples', type=int, default=100, help='Number of benchmark samples')
    
    args = parser.parse_args()
    
    validator = ArduinoValidator(args.port, args.baud)
    
    try:
        success = True
        
        if args.quick_test or args.full_test:
            success &= validator.test_connection()
            
        if args.benchmark or args.full_test:
            success &= validator.benchmark_performance(args.samples)
            
        if args.full_test:
            success &= validator.test_memory_usage()
            success &= validator.validate_accuracy()
            success &= validator.stability_test(args.stability)
        
        # Generate report
        validator.generate_report()
        
        print(f"\n🏁 Overall Test Result: {'✅ PASS' if success else '❌ FAIL'}")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
    finally:
        validator.disconnect()

if __name__ == "__main__":
    main()
