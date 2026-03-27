"""
Performance Test Runner for PC-Arduino Interface
Simulates realistic data flow without requiring actual Arduino hardware
"""

import time
import sys
import subprocess
import psutil
import threading
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import json
import socket
import argparse

class PerformanceTestRunner:
    def __init__(self):
        self.results = {}
        
    def create_mock_arduino_server(self, port=12345):
        """Create a mock Arduino server that responds like real hardware"""
        def server_thread():
            try:
                server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                server.bind(('localhost', port))
                server.listen(1)
                print(f"🤖 Mock Arduino server listening on port {port}")
                
                while True:
                    try:
                        client, addr = server.accept()
                        print(f"📡 Client connected: {addr}")
                        
                        while True:
                            data = client.recv(1024).decode()
                            if not data:
                                break
                                
                            # Simulate Arduino processing time
                            time.sleep(0.008 + np.random.normal(0, 0.002))  # 8ms ± 2ms
                            
                            # Generate realistic SOC response
                            soc = 0.3 + 0.4 * np.random.random()  # 0.3-0.7 range
                            response = {
                                "pred_soc": round(soc, 6),
                                "inference_time_ms": round(8 + np.random.normal(0, 2), 2),
                                "model_type": "LSTM"
                            }
                            
                            client.send((json.dumps(response) + '\n').encode())
                            
                    except Exception as e:
                        print(f"Server error: {e}")
                        continue
                        
            except Exception as e:
                print(f"Server startup error: {e}")
        
        thread = threading.Thread(target=server_thread, daemon=True)
        thread.start()
        time.sleep(1)  # Wait for server to start
        
    def run_performance_test(self, script_path, duration=60, mock_server=True):
        """Run performance test on a script"""
        print(f"🧪 Testing {script_path} for {duration} seconds...")
        
        # Start mock server if needed
        if mock_server:
            self.create_mock_arduino_server()
        
        # Performance metrics
        metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000),
            'peak_memory': 0,
            'avg_cpu': 0,
            'messages_processed': 0,
            'errors': 0
        }
        
        start_time = time.time()
        
        try:
            # Start script
            process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitor process
            def monitor():
                try:
                    ps_process = psutil.Process(process.pid)
                    while time.time() - start_time < duration:
                        if process.poll() is not None:
                            break
                            
                        cpu = ps_process.cpu_percent()
                        memory = ps_process.memory_info().rss / 1024 / 1024  # MB
                        
                        metrics['cpu_usage'].append(cpu)
                        metrics['memory_usage'].append(memory)
                        metrics['timestamps'].append(time.time())
                        metrics['peak_memory'] = max(metrics['peak_memory'], memory)
                        
                        time.sleep(0.5)
                        
                except psutil.NoSuchProcess:
                    pass
            
            monitor_thread = threading.Thread(target=monitor, daemon=True)
            monitor_thread.start()
            
            # Wait for test to complete
            try:
                stdout, stderr = process.communicate(timeout=duration + 10)
            except subprocess.TimeoutExpired:
                process.terminate()
                stdout, stderr = process.communicate()
            
            # Calculate final metrics
            if metrics['cpu_usage']:
                metrics['avg_cpu'] = np.mean(list(metrics['cpu_usage']))
                metrics['memory_growth'] = list(metrics['memory_usage'])[-1] - list(metrics['memory_usage'])[0] if len(metrics['memory_usage']) > 1 else 0
                
            # Count processed messages from output
            lines = stdout.split('\n') + stderr.split('\n')
            for line in lines:
                if 'Point' in line or 'Processed' in line:
                    metrics['messages_processed'] += 1
                elif 'error' in line.lower() or 'failed' in line.lower():
                    metrics['errors'] += 1
            
            metrics['throughput'] = metrics['messages_processed'] / max(duration, 1)
            
            print(f"✅ Test completed:")
            print(f"   - Peak Memory: {metrics['peak_memory']:.1f} MB")
            print(f"   - Avg CPU: {metrics['avg_cpu']:.1f}%")
            print(f"   - Throughput: {metrics['throughput']:.1f} msg/s")
            print(f"   - Errors: {metrics['errors']}")
            
            return metrics
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return metrics
    
    def compare_versions(self, original_script, optimized_script, duration=60):
        """Compare original vs optimized versions"""
        print("🔬 PERFORMANCE COMPARISON WITH REALISTIC LOAD")
        print("=" * 60)
        
        # Test original version
        print("\n📊 Testing ORIGINAL version...")
        original_results = self.run_performance_test(original_script, duration)
        
        time.sleep(5)  # Pause between tests
        
        # Test optimized version
        print("\n⚡ Testing OPTIMIZED version...")
        optimized_results = self.run_performance_test(optimized_script, duration)
        
        # Generate comparison report
        self.generate_comparison_report(original_results, optimized_results)
        
    def generate_comparison_report(self, original, optimized):
        """Generate detailed comparison report"""
        print("\n" + "=" * 60)
        print("📊 DETAILED PERFORMANCE COMPARISON REPORT")
        print("=" * 60)
        
        # Memory comparison
        orig_memory = original.get('peak_memory', 0)
        opt_memory = optimized.get('peak_memory', 0)
        memory_improvement = ((orig_memory - opt_memory) / max(orig_memory, 0.1)) * 100
        
        print(f"\n💾 MEMORY USAGE:")
        print(f"  Original Peak:   {orig_memory:.1f} MB")
        print(f"  Optimized Peak:  {opt_memory:.1f} MB")
        print(f"  Improvement:     {memory_improvement:+.1f}%")
        
        # CPU comparison
        orig_cpu = original.get('avg_cpu', 0)
        opt_cpu = optimized.get('avg_cpu', 0)
        cpu_improvement = ((orig_cpu - opt_cpu) / max(orig_cpu, 0.1)) * 100
        
        print(f"\n⚡ CPU USAGE:")
        print(f"  Original Avg:    {orig_cpu:.1f}%")
        print(f"  Optimized Avg:   {opt_cpu:.1f}%")
        print(f"  Improvement:     {cpu_improvement:+.1f}%")
        
        # Throughput comparison
        orig_throughput = original.get('throughput', 0)
        opt_throughput = optimized.get('throughput', 0)
        throughput_improvement = ((opt_throughput - orig_throughput) / max(orig_throughput, 0.1)) * 100
        
        print(f"\n🚀 THROUGHPUT:")
        print(f"  Original:        {orig_throughput:.1f} msg/s")
        print(f"  Optimized:       {opt_throughput:.1f} msg/s")
        print(f"  Improvement:     {throughput_improvement:+.1f}%")
        
        # Error comparison
        orig_errors = original.get('errors', 0)
        opt_errors = optimized.get('errors', 0)
        
        print(f"\n❌ ERRORS:")
        print(f"  Original:        {orig_errors}")
        print(f"  Optimized:       {opt_errors}")
        print(f"  Change:          {opt_errors - orig_errors:+d}")
        
        # Overall assessment
        improvements = [memory_improvement, cpu_improvement, throughput_improvement]
        valid_improvements = [x for x in improvements if not np.isnan(x)]
        overall_score = np.mean(valid_improvements) if valid_improvements else 0
        
        print(f"\n🏆 OVERALL PERFORMANCE SCORE:")
        print(f"  Score: {overall_score:+.1f}%")
        
        if overall_score > 15:
            print("  ✅ EXCELLENT OPTIMIZATION!")
        elif overall_score > 5:
            print("  🟡 Good improvement")
        else:
            print("  🔴 Needs more optimization")
        
        # Memory growth comparison
        orig_growth = original.get('memory_growth', 0)
        opt_growth = optimized.get('memory_growth', 0)
        
        print(f"\n📈 MEMORY STABILITY:")
        print(f"  Original Growth: {orig_growth:+.1f} MB")
        print(f"  Optimized Growth: {opt_growth:+.1f} MB")
        if abs(opt_growth) < abs(orig_growth):
            print("  ✅ Memory leak reduced!")
        else:
            print("  ⚠️ Memory growth still present")

def main():
    parser = argparse.ArgumentParser(description='Performance Test Runner')
    parser.add_argument('--duration', type=int, default=45, help='Test duration in seconds')
    parser.add_argument('--original', default='pc_arduino_interface.py', help='Original script path')
    parser.add_argument('--optimized', default='pc_arduino_interface_optimized.py', help='Optimized script path')
    
    args = parser.parse_args()
    
    runner = PerformanceTestRunner()
    runner.compare_versions(args.original, args.optimized, args.duration)

if __name__ == "__main__":
    main()
