"""
Performance Comparison Tool for BMS SOC Live Test Systems
Compares original vs optimized versions of live_test_soc.py
"""

import subprocess
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
from collections import defaultdict
import numpy as np
import threading
import json
import os
from pathlib import Path

class SOCPerformanceComparator:
    def __init__(self):
        self.results = {
            'original': defaultdict(list),
            'optimized': defaultdict(list)
        }
        self.monitoring = False
        
    def monitor_process(self, process, version, duration=60):
        """Monitor a process for performance metrics"""
        start_time = time.time()
        
        while time.time() - start_time < duration and process.poll() is None:
            try:
                # Get process info
                proc = psutil.Process(process.pid)
                
                # CPU and Memory usage
                cpu_percent = proc.cpu_percent()
                memory_mb = proc.memory_info().rss / 1024 / 1024
                
                # Get all child processes (threads)
                children = proc.children(recursive=True)
                total_cpu = cpu_percent
                total_memory = memory_mb
                
                for child in children:
                    try:
                        total_cpu += child.cpu_percent()
                        total_memory += child.memory_info().rss / 1024 / 1024
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                # Store metrics
                timestamp = time.time() - start_time
                self.results[version]['timestamp'].append(timestamp)
                self.results[version]['cpu_percent'].append(total_cpu)
                self.results[version]['memory_mb'].append(total_memory)
                self.results[version]['thread_count'].append(proc.num_threads())
                
                time.sleep(1)  # Sample every second
                
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                break
                
    def run_comparison(self, test_duration=60):
        """Run performance comparison between original and optimized versions"""
        print("🚀 Starting SOC Performance Comparison...")
        print(f"⏱️ Test Duration: {test_duration} seconds each")
        
        # Paths to the scripts
        original_script = Path("live_test_soc.py")
        optimized_script = Path("live_test_soc_optimized.py")
        
        if not original_script.exists():
            print(f"❌ Original script not found: {original_script}")
            return
            
        if not optimized_script.exists():
            print(f"❌ Optimized script not found: {optimized_script}")
            return
        
        # Test Original Version
        print("\n📊 Testing Original Version...")
        self.test_version("original", str(original_script), test_duration)
        
        # Wait between tests
        print("⏸️ Waiting 10 seconds between tests...")
        time.sleep(10)
        
        # Test Optimized Version
        print("\n📊 Testing Optimized Version...")
        self.test_version("optimized", str(optimized_script), test_duration)
        
        # Generate comparison report
        self.generate_report()
        
    def test_version(self, version, script_path, duration):
        """Test a specific version of the script"""
        try:
            # Start the process
            process = subprocess.Popen(
                ["python", script_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=os.getcwd()
            )
            
            print(f"✅ Started {version} version (PID: {process.pid})")
            
            # Monitor the process
            monitor_thread = threading.Thread(
                target=self.monitor_process,
                args=(process, version, duration)
            )
            monitor_thread.start()
            
            # Wait for monitoring to complete
            monitor_thread.join()
            
            # Terminate the process
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                
            print(f"✅ Completed {version} version test")
            
        except Exception as e:
            print(f"❌ Error testing {version} version: {e}")
            
    def calculate_metrics(self, version):
        """Calculate performance metrics for a version"""
        data = self.results[version]
        
        if not data['cpu_percent']:
            return {}
            
        metrics = {
            'avg_cpu_percent': np.mean(data['cpu_percent']),
            'max_cpu_percent': np.max(data['cpu_percent']),
            'avg_memory_mb': np.mean(data['memory_mb']),
            'max_memory_mb': np.max(data['memory_mb']),
            'memory_growth': data['memory_mb'][-1] - data['memory_mb'][0] if len(data['memory_mb']) > 1 else 0,
            'avg_threads': np.mean(data['thread_count']),
            'cpu_stability': np.std(data['cpu_percent']),
            'memory_stability': np.std(data['memory_mb'])
        }
        
        return metrics
        
    def generate_report(self):
        """Generate detailed comparison report"""
        print("\n" + "="*60)
        print("📊 SOC PERFORMANCE COMPARISON REPORT")
        print("="*60)
        
        # Calculate metrics for both versions
        original_metrics = self.calculate_metrics('original')
        optimized_metrics = self.calculate_metrics('optimized')
        
        if not original_metrics or not optimized_metrics:
            print("❌ Insufficient data for comparison")
            return
            
        # Print comparison table
        print(f"\n{'Metric':<25} {'Original':<15} {'Optimized':<15} {'Improvement':<15}")
        print("-" * 70)
        
        metrics_to_compare = [
            ('Avg CPU %', 'avg_cpu_percent', lambda x: f"{x:.1f}%"),
            ('Max CPU %', 'max_cpu_percent', lambda x: f"{x:.1f}%"),
            ('Avg Memory (MB)', 'avg_memory_mb', lambda x: f"{x:.1f}"),
            ('Max Memory (MB)', 'max_memory_mb', lambda x: f"{x:.1f}"),
            ('Memory Growth (MB)', 'memory_growth', lambda x: f"{x:.1f}"),
            ('Avg Threads', 'avg_threads', lambda x: f"{x:.1f}"),
            ('CPU Stability', 'cpu_stability', lambda x: f"{x:.2f}"),
            ('Memory Stability', 'memory_stability', lambda x: f"{x:.2f}")
        ]
        
        for name, key, formatter in metrics_to_compare:
            orig_val = original_metrics[key]
            opt_val = optimized_metrics[key]
            
            # Calculate improvement
            if key in ['memory_growth', 'cpu_stability', 'memory_stability']:
                # Lower is better for these metrics
                if orig_val != 0:
                    improvement = ((orig_val - opt_val) / orig_val) * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"
            else:
                # Lower is better for CPU and memory usage too
                if orig_val != 0:
                    improvement = ((orig_val - opt_val) / orig_val) * 100
                    improvement_str = f"{improvement:+.1f}%"
                else:
                    improvement_str = "N/A"
                    
            print(f"{name:<25} {formatter(orig_val):<15} {formatter(opt_val):<15} {improvement_str:<15}")
        
        # Generate performance plots
        self.plot_comparison()
        
        # Summary assessment
        print(f"\n📈 PERFORMANCE SUMMARY:")
        
        cpu_improvement = ((original_metrics['avg_cpu_percent'] - optimized_metrics['avg_cpu_percent']) / 
                          original_metrics['avg_cpu_percent']) * 100
        memory_improvement = ((original_metrics['avg_memory_mb'] - optimized_metrics['avg_memory_mb']) / 
                             original_metrics['avg_memory_mb']) * 100
        memory_leak_reduction = original_metrics['memory_growth'] - optimized_metrics['memory_growth']
        
        print(f"   • CPU Usage: {cpu_improvement:+.1f}% improvement")
        print(f"   • Memory Usage: {memory_improvement:+.1f}% improvement")
        print(f"   • Memory Leak Reduction: {memory_leak_reduction:.1f} MB")
        
        if cpu_improvement > 0 and memory_improvement > 0:
            print("   ✅ Optimized version shows clear performance improvements!")
        elif cpu_improvement > 10 or memory_improvement > 10:
            print("   ✅ Optimized version shows significant improvements!")
        else:
            print("   ⚠️  Improvements are marginal or need further optimization")
            
    def plot_comparison(self):
        """Create comparison plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('SOC Live Test Performance Comparison', fontsize=16)
        
        # CPU Usage
        if self.results['original']['timestamp'] and self.results['optimized']['timestamp']:
            ax1.plot(self.results['original']['timestamp'], 
                    self.results['original']['cpu_percent'], 
                    'r-', label='Original', alpha=0.7)
            ax1.plot(self.results['optimized']['timestamp'], 
                    self.results['optimized']['cpu_percent'], 
                    'g-', label='Optimized', alpha=0.7)
            ax1.set_xlabel('Time (seconds)')
            ax1.set_ylabel('CPU Usage (%)')
            ax1.set_title('CPU Usage Over Time')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Memory Usage
            ax2.plot(self.results['original']['timestamp'], 
                    self.results['original']['memory_mb'], 
                    'r-', label='Original', alpha=0.7)
            ax2.plot(self.results['optimized']['timestamp'], 
                    self.results['optimized']['memory_mb'], 
                    'g-', label='Optimized', alpha=0.7)
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Memory Usage (MB)')
            ax2.set_title('Memory Usage Over Time')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Thread Count
            ax3.plot(self.results['original']['timestamp'], 
                    self.results['original']['thread_count'], 
                    'r-', label='Original', alpha=0.7)
            ax3.plot(self.results['optimized']['timestamp'], 
                    self.results['optimized']['thread_count'], 
                    'g-', label='Optimized', alpha=0.7)
            ax3.set_xlabel('Time (seconds)')
            ax3.set_ylabel('Thread Count')
            ax3.set_title('Thread Count Over Time')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            # Performance Summary Bar Chart
            metrics = ['Avg CPU %', 'Avg Memory (MB)', 'Memory Growth (MB)']
            original_vals = [
                np.mean(self.results['original']['cpu_percent']),
                np.mean(self.results['original']['memory_mb']),
                self.results['original']['memory_mb'][-1] - self.results['original']['memory_mb'][0]
            ]
            optimized_vals = [
                np.mean(self.results['optimized']['cpu_percent']),
                np.mean(self.results['optimized']['memory_mb']),
                self.results['optimized']['memory_mb'][-1] - self.results['optimized']['memory_mb'][0]
            ]
            
            x = np.arange(len(metrics))
            width = 0.35
            
            ax4.bar(x - width/2, original_vals, width, label='Original', color='red', alpha=0.7)
            ax4.bar(x + width/2, optimized_vals, width, label='Optimized', color='green', alpha=0.7)
            ax4.set_xlabel('Metrics')
            ax4.set_ylabel('Value')
            ax4.set_title('Performance Metrics Comparison')
            ax4.set_xticks(x)
            ax4.set_xticklabels(metrics, rotation=45, ha='right')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_filename = f"soc_performance_comparison_{int(time.time())}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"📊 Performance plots saved as: {plot_filename}")
        
        plt.show()

def main():
    """Main function to run the comparison"""
    comparator = SOCPerformanceComparator()
    
    print("🎯 SOC Live Test Performance Comparison Tool")
    print("=" * 50)
    print("This tool will test both original and optimized versions")
    print("of the SOC live test system and compare their performance.")
    print()
    
    try:
        duration = int(input("Enter test duration in seconds (default 60): ") or "60")
    except ValueError:
        duration = 60
        
    print(f"\n⚠️  Make sure both scripts are in the current directory:")
    print("   - live_test_soc.py (original)")
    print("   - live_test_soc_optimized.py (optimized)")
    print()
    
    input("Press Enter to start the comparison...")
    
    comparator.run_comparison(duration)

if __name__ == "__main__":
    main()
