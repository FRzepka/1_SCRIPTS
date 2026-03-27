"""
Performance Comparison Tool
Vergleicht Original vs Optimized PC-Arduino Interface
"""

import time
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import threading
import psutil
from collections import deque
import sys
import os

class PerformanceComparison:
    def __init__(self):
        self.results = {
            'original': {},
            'optimized': {}
        }
        
    def run_performance_test(self, script_path, test_duration=60):
        """Führe Performance-Test für ein Script aus"""
        print(f"🧪 Testing {script_path} for {test_duration} seconds...")
        
        # Performance-Metriken
        metrics = {
            'cpu_usage': deque(maxlen=1000),
            'memory_usage': deque(maxlen=1000),
            'timestamps': deque(maxlen=1000),
            'throughput': 0,
            'avg_latency': 0,
            'error_count': 0,
            'peak_memory': 0,
            'avg_cpu': 0
        }
        
        start_time = time.time()
        
        try:
            # Starte Script als Subprocess
            process = subprocess.Popen([
                sys.executable, script_path
            ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Monitoring-Thread
            def monitor_process():
                try:
                    ps_process = psutil.Process(process.pid)
                    
                    while time.time() - start_time < test_duration:
                        if process.poll() is not None:  # Process beendet
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
                    
            monitor_thread = threading.Thread(target=monitor_process, daemon=True)
            monitor_thread.start()
            
            # Warte auf Test-Ende
            try:
                stdout, stderr = process.communicate(timeout=test_duration + 10)
            except subprocess.TimeoutExpired:
                process.terminate()
                stdout, stderr = process.communicate()
                
            # Berechne finale Metriken
            if metrics['cpu_usage']:
                metrics['avg_cpu'] = np.mean(list(metrics['cpu_usage']))
                
            # Parse Output für Performance-Daten
            metrics.update(self._parse_output(stdout, stderr))
            
        except Exception as e:
            print(f"❌ Error testing {script_path}: {e}")
            
        return metrics
        
    def _parse_output(self, stdout, stderr):
        """Parse Output für Performance-Metriken"""
        parsed = {
            'throughput': 0,
            'avg_latency': 0,
            'error_count': 0,
            'total_messages': 0
        }
        
        try:
            # Suche nach Performance-Informationen in Output
            lines = stdout.split('\n') + stderr.split('\n')
            
            for line in lines:
                if 'Hz' in line and 'Throughput' in line:
                    # Extrahiere Throughput
                    parts = line.split('Throughput:')
                    if len(parts) > 1:
                        hz_part = parts[1].split('Hz')[0].strip()
                        parsed['throughput'] = float(hz_part)
                        
                elif 'Point' in line and 'Time' in line:
                    # Count processed messages
                    parsed['total_messages'] += 1
                    
                    # Extrahiere Latency falls vorhanden
                    if 'ms' in line:
                        try:
                            ms_parts = line.split('ms')[0].split()
                            if ms_parts:
                                latency = float(ms_parts[-1])
                                parsed['avg_latency'] = latency
                        except:
                            pass
                            
                elif 'error' in line.lower() or 'failed' in line.lower():
                    parsed['error_count'] += 1
                    
        except Exception as e:
            print(f"⚠️ Warning: Could not parse all performance data: {e}")
            
        return parsed
        
    def compare_systems(self, original_script, optimized_script, test_duration=60):
        """Vergleiche beide Systeme"""
        print("🔬 PERFORMANCE COMPARISON STARTING")
        print("="*50)
        
        # Teste Original System
        print("\n📊 Testing ORIGINAL system...")
        if os.path.exists(original_script):
            self.results['original'] = self.run_performance_test(original_script, test_duration)
        else:
            print(f"❌ Original script not found: {original_script}")
            self.results['original'] = self._empty_metrics()
            
        time.sleep(5)  # Pause zwischen Tests
        
        # Teste Optimized System
        print("\n⚡ Testing OPTIMIZED system...")
        if os.path.exists(optimized_script):
            self.results['optimized'] = self.run_performance_test(optimized_script, test_duration)
        else:
            print(f"❌ Optimized script not found: {optimized_script}")
            self.results['optimized'] = self._empty_metrics()
            
        # Erstelle Vergleichsreport
        self.create_comparison_report()
        self.plot_comparison()
        
    def _empty_metrics(self):
        """Leere Metriken für fehlende Systeme"""
        return {
            'cpu_usage': deque(),
            'memory_usage': deque(),
            'timestamps': deque(),
            'throughput': 0,
            'avg_latency': 0,
            'error_count': 0,
            'peak_memory': 0,
            'avg_cpu': 0,
            'total_messages': 0
        }
        
    def create_comparison_report(self):
        """Erstelle detaillierten Vergleichsreport"""
        orig = self.results['original']
        opt = self.results['optimized']
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE COMPARISON REPORT")
        print("="*60)
        
        # Throughput Vergleich
        orig_throughput = orig.get('throughput', 0)
        opt_throughput = opt.get('throughput', 0)
        throughput_improvement = ((opt_throughput - orig_throughput) / max(orig_throughput, 0.1)) * 100 if orig_throughput > 0 else 0
        
        print(f"\n🚀 THROUGHPUT:")
        print(f"  Original:   {orig_throughput:.1f} Hz")
        print(f"  Optimized:  {opt_throughput:.1f} Hz")
        print(f"  Improvement: {throughput_improvement:+.1f}%")
        
        # Memory Vergleich
        orig_memory = orig.get('peak_memory', 0)
        opt_memory = opt.get('peak_memory', 0)
        memory_improvement = ((orig_memory - opt_memory) / max(orig_memory, 0.1)) * 100 if orig_memory > 0 else 0
        
        print(f"\n💾 MEMORY USAGE:")
        print(f"  Original Peak:   {orig_memory:.1f} MB")
        print(f"  Optimized Peak:  {opt_memory:.1f} MB")
        print(f"  Reduction: {memory_improvement:+.1f}%")
        
        # CPU Vergleich
        orig_cpu = orig.get('avg_cpu', 0)
        opt_cpu = opt.get('avg_cpu', 0)
        cpu_improvement = ((orig_cpu - opt_cpu) / max(orig_cpu, 0.1)) * 100 if orig_cpu > 0 else 0
        
        print(f"\n⚡ CPU USAGE:")
        print(f"  Original Avg:   {orig_cpu:.1f}%")
        print(f"  Optimized Avg:  {opt_cpu:.1f}%")
        print(f"  Reduction: {cpu_improvement:+.1f}%")
        
        # Latency Vergleich
        orig_latency = orig.get('avg_latency', 0)
        opt_latency = opt.get('avg_latency', 0)
        latency_improvement = ((orig_latency - opt_latency) / max(orig_latency, 0.1)) * 100 if orig_latency > 0 else 0
        
        print(f"\n📡 LATENCY:")
        print(f"  Original Avg:   {orig_latency:.1f} ms")
        print(f"  Optimized Avg:  {opt_latency:.1f} ms")
        print(f"  Reduction: {latency_improvement:+.1f}%")
        
        # Error Vergleich
        orig_errors = orig.get('error_count', 0)
        opt_errors = opt.get('error_count', 0)
        
        print(f"\n❌ ERRORS:")
        print(f"  Original:   {orig_errors}")
        print(f"  Optimized:  {opt_errors}")
        print(f"  Change: {opt_errors - orig_errors:+d}")
        
        # Overall Score
        improvements = [throughput_improvement, memory_improvement, cpu_improvement, latency_improvement]
        valid_improvements = [x for x in improvements if not np.isnan(x) and x != 0]
        overall_score = np.mean(valid_improvements) if valid_improvements else 0
        
        print(f"\n🏆 OVERALL PERFORMANCE SCORE:")
        print(f"  Improvement: {overall_score:+.1f}%")
        
        if overall_score > 10:
            print("  ✅ SIGNIFICANT IMPROVEMENT!")
        elif overall_score > 0:
            print("  🟡 Minor improvement")
        else:
            print("  🔴 Needs more optimization")
            
        # Empfehlungen
        print(f"\n💡 RECOMMENDATIONS:")
        if throughput_improvement < 0:
            print("  - Check Arduino communication timing")
            print("  - Verify BAUD_RATE settings")
        if memory_improvement < 0:
            print("  - Implement stricter memory limits")
            print("  - Use more efficient data structures")
        if cpu_improvement < 0:
            print("  - Reduce plot update frequency")
            print("  - Optimize calculation loops")
        if overall_score > 20:
            print("  ✅ Great optimization! Consider this production-ready.")
            
    def plot_comparison(self):
        """Erstelle Vergleichs-Plots"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Original vs Optimized PC-Arduino Interface Comparison', fontsize=14)
        
        # Daten vorbereiten
        orig = self.results['original']
        opt = self.results['optimized']
        
        # Bar Chart für Key Metrics
        metrics = ['Throughput (Hz)', 'Peak Memory (MB)', 'Avg CPU (%)', 'Avg Latency (ms)']
        orig_values = [
            orig.get('throughput', 0),
            orig.get('peak_memory', 0),
            orig.get('avg_cpu', 0),
            orig.get('avg_latency', 0)
        ]
        opt_values = [
            opt.get('throughput', 0),
            opt.get('peak_memory', 0),
            opt.get('avg_cpu', 0),
            opt.get('avg_latency', 0)
        ]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        ax1.bar(x - width/2, orig_values, width, label='Original', color='red', alpha=0.7)
        ax1.bar(x + width/2, opt_values, width, label='Optimized', color='green', alpha=0.7)
        ax1.set_title('Key Performance Metrics')
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Memory Usage Over Time
        if len(orig['memory_usage']) > 0 and len(opt['memory_usage']) > 0:
            orig_times = np.linspace(0, len(orig['memory_usage']), len(orig['memory_usage']))
            opt_times = np.linspace(0, len(opt['memory_usage']), len(opt['memory_usage']))
            
            ax2.plot(orig_times, list(orig['memory_usage']), 'r-', label='Original', linewidth=2)
            ax2.plot(opt_times, list(opt['memory_usage']), 'g-', label='Optimized', linewidth=2)
            ax2.set_title('Memory Usage Over Time')
            ax2.set_xlabel('Sample')
            ax2.set_ylabel('Memory (MB)')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # CPU Usage Over Time
        if len(orig['cpu_usage']) > 0 and len(opt['cpu_usage']) > 0:
            orig_times = np.linspace(0, len(orig['cpu_usage']), len(orig['cpu_usage']))
            opt_times = np.linspace(0, len(opt['cpu_usage']), len(opt['cpu_usage']))
            
            ax3.plot(orig_times, list(orig['cpu_usage']), 'r-', label='Original', linewidth=2)
            ax3.plot(opt_times, list(opt['cpu_usage']), 'g-', label='Optimized', linewidth=2)
            ax3.set_title('CPU Usage Over Time')
            ax3.set_xlabel('Sample')
            ax3.set_ylabel('CPU (%)')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        # Improvement Summary (Radar Chart simplified as Bar)
        improvements = []
        labels = []
        
        if orig.get('throughput', 0) > 0:
            improvements.append(((opt.get('throughput', 0) - orig.get('throughput', 0)) / orig.get('throughput', 0.1)) * 100)
            labels.append('Throughput')
            
        if orig.get('peak_memory', 0) > 0:
            improvements.append(((orig.get('peak_memory', 0) - opt.get('peak_memory', 0)) / orig.get('peak_memory', 0.1)) * 100)
            labels.append('Memory')
            
        if orig.get('avg_cpu', 0) > 0:
            improvements.append(((orig.get('avg_cpu', 0) - opt.get('avg_cpu', 0)) / orig.get('avg_cpu', 0.1)) * 100)
            labels.append('CPU')
            
        if orig.get('avg_latency', 0) > 0:
            improvements.append(((orig.get('avg_latency', 0) - opt.get('avg_latency', 0)) / orig.get('avg_latency', 0.1)) * 100)
            labels.append('Latency')
        
        if improvements:
            colors = ['green' if x > 0 else 'red' for x in improvements]
            ax4.bar(labels, improvements, color=colors, alpha=0.7)
            ax4.set_title('Performance Improvements (%)')
            ax4.set_ylabel('Improvement (%)')
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.get_xticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        plt.show()

def main():
    """Hauptfunktion für Performance-Vergleich"""
    print("🔬 PC-Arduino Interface Performance Comparison")
    print("="*50)
    
    # Script-Pfade
    base_path = r"c:\Users\Florian\SynologyDrive\TUB\1_Dissertation\5_Codes\LFP_SOC_SOH\BMS_Arduino"
    original_script = os.path.join(base_path, "pc_arduino_interface.py")
    optimized_script = os.path.join(base_path, "pc_arduino_interface_optimized.py")
    
    print(f"Original script: {original_script}")
    print(f"Optimized script: {optimized_script}")
    
    # Überprüfe Script-Existenz
    if not os.path.exists(original_script):
        print(f"❌ Original script not found: {original_script}")
        return
        
    if not os.path.exists(optimized_script):
        print(f"❌ Optimized script not found: {optimized_script}")
        return
    
    # Performance-Vergleich
    comparator = PerformanceComparison()
    
    # Kurzer Test (30 Sekunden pro System)
    print("\n⏱️ Running short performance tests (30s each)...")
    print("💡 For full analysis, increase test_duration in script")
    
    comparator.compare_systems(original_script, optimized_script, test_duration=30)
    
    print("\n✅ Performance comparison completed!")
    print("📊 Check the plots and report above for detailed analysis")

if __name__ == "__main__":
    main()
