"""
SOC Performance Load Testing Tool
Generates realistic BMS data streams to test SOC live systems under load
"""

import socket
import time
import json
import threading
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import psutil
import subprocess
import sys
from collections import deque

class SOCLoadTester:
    def __init__(self, host='localhost', port=12346):
        self.host = host
        self.port = port
        self.running = False
        
    def generate_realistic_bms_data(self, duration_seconds=60, frequency_hz=10):
        """Generate realistic BMS data stream"""
        print(f"📊 Generating {duration_seconds}s of BMS data at {frequency_hz} Hz...")
        
        total_points = int(duration_seconds * frequency_hz)
        timestamps = np.linspace(0, duration_seconds, total_points)
        
        # Realistic battery discharge cycle
        base_voltage = 3.3 + 0.4 * np.exp(-timestamps / 3600)  # Voltage decay
        voltage_noise = np.random.normal(0, 0.01, total_points)
        voltages = base_voltage + voltage_noise
        
        # Current with discharge pattern and noise
        base_current = -2.5 * (1 + 0.3 * np.sin(timestamps / 600))  # Varying discharge
        current_noise = np.random.normal(0, 0.1, total_points)
        currents = base_current + current_noise
        
        # SOH slowly degrading
        soh_values = np.full(total_points, 0.95) + np.random.normal(0, 0.01, total_points)
        
        # Q_c values
        q_c_values = np.full(total_points, 5.0) + np.random.normal(0, 0.1, total_points)
        
        # True SOC (for validation)
        true_soc = 0.8 * np.exp(-timestamps / 7200)  # SOC decline
        
        # Create data packets
        data_packets = []
        for i in range(total_points):
            packet = {
                'voltage': round(float(voltages[i]), 4),
                'current': round(float(currents[i]), 4),
                'soh': round(float(soh_values[i]), 4),
                'q_c': round(float(q_c_values[i]), 4),
                'true_soc': round(float(true_soc[i]), 4),
                'timestamp': round(float(timestamps[i]), 3),
                'point_id': i
            }
            data_packets.append(packet)
            
        return data_packets
    
    def start_data_server(self, data_packets, send_frequency=10):
        """Start a server that sends BMS data to SOC test system"""
        def server_thread():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind((self.host, self.port))
                server.listen(1)
                print(f"🚀 BMS Data Server listening on {self.host}:{self.port}")
                
                while self.running:
                    try:
                        client, addr = server.accept()
                        print(f"📡 SOC Client connected: {addr}")
                        
                        packet_interval = 1.0 / send_frequency
                        start_time = time.time()
                        
                        for i, packet in enumerate(data_packets):
                            if not self.running:
                                break
                                
                            # Send data packet
                            message = json.dumps(packet) + '\n'
                            try:
                                client.send(message.encode())
                                
                                # Maintain sending frequency
                                target_time = start_time + (i + 1) * packet_interval
                                sleep_time = target_time - time.time()
                                if sleep_time > 0:
                                    time.sleep(sleep_time)
                                    
                            except:
                                print(f"📤 Client disconnected")
                                break
                                
                        client.close()
                        
                    except Exception as e:
                        if self.running:
                            print(f"Server error: {e}")
                        continue
                        
                server.close()
                
            except Exception as e:
                print(f"Server startup error: {e}")
        
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        return thread
    
    def run_soc_performance_test(self, script_path, data_packets, duration=60):
        """Run performance test on SOC script with realistic load"""
        print(f"🧪 Load testing {script_path} with {len(data_packets)} data points...")
        
        # Start data server
        self.running = True
        server_thread = self.start_data_server(data_packets, send_frequency=10)
        time.sleep(2)  # Wait for server to start
        
        # Performance metrics
        metrics = {
            'cpu_usage': deque(maxlen=2000),
            'memory_usage': deque(maxlen=2000),
            'timestamps': deque(maxlen=2000),
            'peak_memory': 0,
            'avg_cpu': 0,
            'memory_growth': 0,
            'start_memory': 0,
            'prediction_count': 0,
            'errors': 0
        }
        
        start_time = time.time()
        
        try:
            # Start SOC script
            process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitor process performance
            def monitor():
                try:
                    ps_process = psutil.Process(process.pid)
                    first_memory = True
                    
                    while time.time() - start_time < duration:
                        if process.poll() is not None:
                            break
                            
                        cpu = ps_process.cpu_percent()
                        memory = ps_process.memory_info().rss / 1024 / 1024  # MB
                        
                        if first_memory:
                            metrics['start_memory'] = memory
                            first_memory = False
                        
                        metrics['cpu_usage'].append(cpu)
                        metrics['memory_usage'].append(memory)
                        metrics['timestamps'].append(time.time())
                        metrics['peak_memory'] = max(metrics['peak_memory'], memory)
                        
                        time.sleep(0.2)  # Higher frequency monitoring
                        
                except psutil.NoSuchProcess:
                    pass
            
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            
            # Wait for test completion
            try:
                stdout, stderr = process.communicate(timeout=duration + 10)
            except subprocess.TimeoutExpired:
                process.terminate()
                stdout, stderr = process.communicate()
            
            # Stop data server
            self.running = False
            
            # Calculate metrics
            if metrics['memory_usage']:
                metrics['avg_cpu'] = np.mean(list(metrics['cpu_usage'])) if metrics['cpu_usage'] else 0
                memory_usage = list(metrics['memory_usage'])
                metrics['memory_growth'] = memory_usage[-1] - memory_usage[0] if len(memory_usage) > 1 else 0
                
                # Count predictions from output
                lines = stdout.split('\n') + stderr.split('\n')
                for line in lines:
                    if 'SOC:' in line or 'prediction' in line.lower() or 'Point' in line:
                        metrics['prediction_count'] += 1
                    elif 'error' in line.lower() or 'exception' in line.lower():
                        metrics['errors'] += 1
                
                metrics['throughput'] = metrics['prediction_count'] / max(duration, 1)
                
                # Memory stability calculation
                if len(memory_usage) > 10:
                    memory_variance = np.var(memory_usage[-10:])  # Variance of last 10 readings
                    metrics['memory_stability'] = memory_variance
                else:
                    metrics['memory_stability'] = 0
            
            print(f"✅ Load test completed:")
            print(f"   - Peak Memory: {metrics['peak_memory']:.1f} MB")
            print(f"   - Memory Growth: {metrics['memory_growth']:+.1f} MB")
            print(f"   - Avg CPU: {metrics['avg_cpu']:.1f}%")
            print(f"   - Throughput: {metrics['throughput']:.1f} predictions/s")
            print(f"   - Errors: {metrics['errors']}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Load test failed: {e}")
            self.running = False
            return metrics
    
    def compare_soc_systems(self, original_script, optimized_script, duration=60):
        """Compare original vs optimized SOC systems under load"""
        print("🔬 SOC PERFORMANCE COMPARISON UNDER REALISTIC LOAD")
        print("=" * 70)
        
        # Generate test data
        data_packets = self.generate_realistic_bms_data(duration + 10, frequency_hz=15)
        
        # Test original version
        print(f"\n📊 Load testing ORIGINAL SOC system...")
        original_results = self.run_soc_performance_test(original_script, data_packets, duration)
        
        time.sleep(5)  # Pause between tests
        
        # Test optimized version
        print(f"\n⚡ Load testing OPTIMIZED SOC system...")
        optimized_results = self.run_soc_performance_test(optimized_script, data_packets, duration)
        
        # Generate comparison report
        self.generate_soc_comparison_report(original_results, optimized_results)
    
    def generate_soc_comparison_report(self, original, optimized):
        """Generate detailed SOC performance comparison report"""
        print("\n" + "=" * 70)
        print("📊 SOC SYSTEM PERFORMANCE COMPARISON REPORT")
        print("=" * 70)
        
        # Memory analysis
        orig_peak = original.get('peak_memory', 0)
        opt_peak = optimized.get('peak_memory', 0)
        memory_reduction = ((orig_peak - opt_peak) / max(orig_peak, 0.1)) * 100
        
        orig_growth = original.get('memory_growth', 0)
        opt_growth = optimized.get('memory_growth', 0)
        growth_improvement = ((orig_growth - opt_growth) / max(abs(orig_growth), 0.1)) * 100
        
        print(f"\n💾 MEMORY PERFORMANCE:")
        print(f"  Original Peak:      {orig_peak:.1f} MB")
        print(f"  Optimized Peak:     {opt_peak:.1f} MB")
        print(f"  Peak Reduction:     {memory_reduction:+.1f}%")
        print(f"  Original Growth:    {orig_growth:+.1f} MB")
        print(f"  Optimized Growth:   {opt_growth:+.1f} MB")
        print(f"  Growth Improvement: {growth_improvement:+.1f}%")
        
        # CPU analysis
        orig_cpu = original.get('avg_cpu', 0)
        opt_cpu = optimized.get('avg_cpu', 0)
        cpu_improvement = ((orig_cpu - opt_cpu) / max(orig_cpu, 0.1)) * 100
        
        print(f"\n⚡ CPU PERFORMANCE:")
        print(f"  Original Avg CPU:   {orig_cpu:.1f}%")
        print(f"  Optimized Avg CPU:  {opt_cpu:.1f}%")
        print(f"  CPU Improvement:    {cpu_improvement:+.1f}%")
        
        # Throughput analysis
        orig_throughput = original.get('throughput', 0)
        opt_throughput = optimized.get('throughput', 0)
        throughput_improvement = ((opt_throughput - orig_throughput) / max(orig_throughput, 0.1)) * 100
        
        print(f"\n🚀 THROUGHPUT PERFORMANCE:")
        print(f"  Original Throughput: {orig_throughput:.1f} pred/s")
        print(f"  Optimized Throughput: {opt_throughput:.1f} pred/s")
        print(f"  Throughput Gain:     {throughput_improvement:+.1f}%")
        
        # Stability analysis
        orig_stability = original.get('memory_stability', 0)
        opt_stability = optimized.get('memory_stability', 0)
        
        print(f"\n📈 SYSTEM STABILITY:")
        print(f"  Original Memory Variance: {orig_stability:.2f}")
        print(f"  Optimized Memory Variance: {opt_stability:.2f}")
        if opt_stability < orig_stability:
            print("  ✅ Memory usage more stable!")
        else:
            print("  ⚠️ Memory stability needs improvement")
        
        # Error analysis
        orig_errors = original.get('errors', 0)
        opt_errors = optimized.get('errors', 0)
        
        print(f"\n❌ ERROR ANALYSIS:")
        print(f"  Original Errors:    {orig_errors}")
        print(f"  Optimized Errors:   {opt_errors}")
        print(f"  Error Change:       {opt_errors - orig_errors:+d}")
        
        # Overall assessment
        improvements = [memory_reduction, cpu_improvement, throughput_improvement, growth_improvement]
        valid_improvements = [x for x in improvements if not np.isnan(x) and abs(x) < 1000]
        overall_score = np.mean(valid_improvements) if valid_improvements else 0
        
        print(f"\n🏆 OVERALL OPTIMIZATION SCORE:")
        print(f"  Score: {overall_score:+.1f}%")
        
        if overall_score > 25:
            print("  ✅ OUTSTANDING OPTIMIZATION! Production ready.")
        elif overall_score > 10:
            print("  🟢 GOOD OPTIMIZATION! Significant improvements.")
        elif overall_score > 0:
            print("  🟡 MINOR IMPROVEMENTS. Consider further optimization.")
        else:
            print("  🔴 NEEDS MORE WORK. Review optimization strategy.")
        
        # Recommendations
        print(f"\n💡 OPTIMIZATION RECOMMENDATIONS:")
        if memory_reduction < 10:
            print("  • Implement stricter memory limits")
            print("  • Use bounded queues and data structures")
        if cpu_improvement < 10:
            print("  • Optimize processing loops")
            print("  • Reduce plot update frequency")
        if throughput_improvement < 20:
            print("  • Implement batch processing")
            print("  • Optimize I/O operations")
        if opt_growth > 10:
            print("  • Add periodic garbage collection")
            print("  • Review for memory leaks")

def main():
    parser = argparse.ArgumentParser(description='SOC Performance Load Tester')
    parser.add_argument('--duration', type=int, default=45, help='Test duration in seconds')
    parser.add_argument('--original', default='live_test_soc.py', help='Original SOC script')
    parser.add_argument('--optimized', default='live_test_soc_optimized.py', help='Optimized SOC script')
    parser.add_argument('--frequency', type=int, default=15, help='Data frequency (Hz)')
    
    args = parser.parse_args()
    
    print("🚀 SOC System Load Testing Framework")
    print("=" * 50)
    print(f"Duration: {args.duration}s")
    print(f"Data Frequency: {args.frequency} Hz")
    print()
    
    tester = SOCLoadTester()
    tester.compare_soc_systems(args.original, args.optimized, args.duration)

if __name__ == "__main__":
    main()
