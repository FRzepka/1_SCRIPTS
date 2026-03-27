"""
Performance Analyser für PC-Arduino Interface
Identifiziert Bottlenecks und überwacht System-Performance
"""

import time
import psutil
import matplotlib.pyplot as plt
import numpy as np
from collections import deque
import threading
import queue
import gc

class PerformanceAnalyzer:
    def __init__(self, max_samples=1000):
        self.max_samples = max_samples
        self.running = False
        
        # Performance Metriken
        self.metrics = {
            'timestamps': deque(maxlen=max_samples),
            'cpu_usage': deque(maxlen=max_samples),
            'memory_usage': deque(maxlen=max_samples),
            'memory_mb': deque(maxlen=max_samples),
            'queue_size': deque(maxlen=max_samples),
            'communication_latency': deque(maxlen=max_samples),
            'processing_rate': deque(maxlen=max_samples),
            'error_rate': deque(maxlen=max_samples)
        }
        
        # Bottleneck Detection
        self.bottlenecks = {
            'memory_leak': False,
            'cpu_overload': False,
            'queue_overflow': False,
            'communication_timeout': False,
            'slow_processing': False
        }
        
        # Statistiken
        self.stats = {
            'start_time': time.time(),
            'total_samples': 0,
            'total_errors': 0,
            'peak_memory': 0,
            'peak_cpu': 0
        }
        
    def start_monitoring(self):
        """Starte Performance-Monitoring"""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        print("📈 Performance monitoring started")
        
    def stop_monitoring(self):
        """Stoppe Performance-Monitoring"""
        self.running = False
        print("📈 Performance monitoring stopped")
        
    def _monitor_loop(self):
        """Haupt-Monitoring Loop"""
        process = psutil.Process()
        
        while self.running:
            try:
                current_time = time.time()
                
                # System-Metriken sammeln
                cpu_percent = process.cpu_percent()
                memory_info = process.memory_info()
                memory_percent = process.memory_percent()
                memory_mb = memory_info.rss / 1024 / 1024  # MB
                
                # Speichere Metriken
                self.metrics['timestamps'].append(current_time)
                self.metrics['cpu_usage'].append(cpu_percent)
                self.metrics['memory_usage'].append(memory_percent)
                self.metrics['memory_mb'].append(memory_mb)
                
                # Update Statistiken
                self.stats['total_samples'] += 1
                self.stats['peak_memory'] = max(self.stats['peak_memory'], memory_mb)
                self.stats['peak_cpu'] = max(self.stats['peak_cpu'], cpu_percent)
                
                # Bottleneck Detection
                self._detect_bottlenecks(cpu_percent, memory_mb)
                
                time.sleep(0.5)  # 2Hz Monitoring
                
            except Exception as e:
                print(f"❌ Performance monitoring error: {e}")
                time.sleep(1)
                
    def _detect_bottlenecks(self, cpu_percent, memory_mb):
        """Erkenne Performance-Bottlenecks"""
        # CPU Overload
        if cpu_percent > 80:
            self.bottlenecks['cpu_overload'] = True
            
        # Memory Leak Detection
        if len(self.metrics['memory_mb']) > 100:
            recent_memory = list(self.metrics['memory_mb'])[-100:]
            memory_trend = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            if memory_trend > 0.1:  # Steigender Memory-Verbrauch
                self.bottlenecks['memory_leak'] = True
                
    def log_communication_event(self, latency_ms, success=True):
        """Logge Communication Event"""
        current_time = time.time()
        
        self.metrics['communication_latency'].append(latency_ms)
        
        if not success:
            self.stats['total_errors'] += 1
            
        # Error Rate berechnen
        if len(self.metrics['communication_latency']) > 0:
            recent_count = min(100, len(self.metrics['communication_latency']))
            error_rate = self.stats['total_errors'] / max(1, self.stats['total_samples']) * 100
            self.metrics['error_rate'].append(error_rate)
            
        # Communication Timeout Detection
        if latency_ms > 1000:  # 1 Sekunde
            self.bottlenecks['communication_timeout'] = True
            
    def log_queue_size(self, size):
        """Logge Queue-Größe"""
        self.metrics['queue_size'].append(size)
        
        # Queue Overflow Detection
        if size > 50:  # Zu große Queue
            self.bottlenecks['queue_overflow'] = True
            
    def log_processing_rate(self, rate_hz):
        """Logge Verarbeitungsrate"""
        self.metrics['processing_rate'].append(rate_hz)
        
        # Slow Processing Detection
        if rate_hz < 5:  # Weniger als 5 Hz
            self.bottlenecks['slow_processing'] = True
            
    def get_bottleneck_summary(self):
        """Erstelle Bottleneck-Zusammenfassung"""
        active_bottlenecks = [name for name, active in self.bottlenecks.items() if active]
        
        summary = {
            'has_bottlenecks': len(active_bottlenecks) > 0,
            'active_bottlenecks': active_bottlenecks,
            'recommendations': self._get_recommendations(active_bottlenecks)
        }
        
        return summary
        
    def _get_recommendations(self, bottlenecks):
        """Erstelle Performance-Empfehlungen"""
        recommendations = []
        
        if 'memory_leak' in bottlenecks:
            recommendations.extend([
                "🔧 Memory Leak detected:",
                "  - Limitiere Queue-Größen (maxsize parameter)",
                "  - Verwende deque mit maxlen statt unbegrenzte Listen",
                "  - Rufe gc.collect() regelmäßig auf",
                "  - Reduziere MAX_POINTS in Plot-Daten"
            ])
            
        if 'cpu_overload' in bottlenecks:
            recommendations.extend([
                "⚡ CPU Overload detected:",
                "  - Erhöhe SEND_INTERVAL (weniger Requests/Sekunde)",
                "  - Reduziere Plot-Update-Frequenz",
                "  - Verwende weniger Plot-Points",
                "  - Implementiere Batch-Processing"
            ])
            
        if 'queue_overflow' in bottlenecks:
            recommendations.extend([
                "📦 Queue Overflow detected:",
                "  - Arduino antwortet zu langsam",
                "  - Erhöhe Arduino BAUD_RATE",
                "  - Implementiere non-blocking Queue-Operations",
                "  - Reduziere JSON-Payload-Größe"
            ])
            
        if 'communication_timeout' in bottlenecks:
            recommendations.extend([
                "📡 Communication Timeout detected:",
                "  - Arduino überlastet oder blockiert",
                "  - Reduziere Modell-Komplexität auf Arduino",
                "  - Implementiere Timeout-Recovery",
                "  - Überprüfe Serial-Verbindung"
            ])
            
        if 'slow_processing' in bottlenecks:
            recommendations.extend([
                "🐌 Slow Processing detected:",
                "  - Optimiere Arduino LSTM-Implementierung",
                "  - Verwende Fixed-Point statt Float",
                "  - Reduziere Hidden-Size oder Sequence-Length",
                "  - Implementiere Hardware-Beschleunigung"
            ])
            
        return recommendations
        
    def create_performance_report(self):
        """Erstelle detaillierten Performance-Report"""
        current_time = time.time()
        runtime = current_time - self.stats['start_time']
        
        if len(self.metrics['memory_mb']) == 0:
            return "❌ No performance data collected"
            
        # Berechne Statistiken
        avg_memory = np.mean(list(self.metrics['memory_mb']))
        avg_cpu = np.mean(list(self.metrics['cpu_usage']))
        
        if self.metrics['communication_latency']:
            avg_latency = np.mean(list(self.metrics['communication_latency']))
            max_latency = max(self.metrics['communication_latency'])
        else:
            avg_latency = max_latency = 0
            
        if self.metrics['processing_rate']:
            avg_rate = np.mean(list(self.metrics['processing_rate']))
        else:
            avg_rate = 0
            
        error_rate = (self.stats['total_errors'] / max(1, self.stats['total_samples'])) * 100
        
        report = f"""
📊 PERFORMANCE ANALYSIS REPORT
{'='*50}
Runtime: {runtime:.1f} seconds
Samples collected: {self.stats['total_samples']}

SYSTEM PERFORMANCE:
  Average CPU: {avg_cpu:.1f}%
  Peak CPU: {self.stats['peak_cpu']:.1f}%
  Average Memory: {avg_memory:.1f} MB
  Peak Memory: {self.stats['peak_memory']:.1f} MB

COMMUNICATION PERFORMANCE:
  Average Latency: {avg_latency:.1f} ms
  Max Latency: {max_latency:.1f} ms
  Error Rate: {error_rate:.2f}%
  Processing Rate: {avg_rate:.1f} Hz

BOTTLENECKS DETECTED:
"""
        
        bottleneck_summary = self.get_bottleneck_summary()
        
        if bottleneck_summary['has_bottlenecks']:
            report += "  ❌ Issues found:\n"
            for bottleneck in bottleneck_summary['active_bottlenecks']:
                report += f"    - {bottleneck}\n"
                
            report += "\nRECOMMENDATIONS:\n"
            for rec in bottleneck_summary['recommendations']:
                report += f"{rec}\n"
        else:
            report += "  ✅ No major bottlenecks detected\n"
            
        return report
        
    def plot_performance_dashboard(self):
        """Erstelle Performance-Dashboard"""
        if len(self.metrics['timestamps']) < 10:
            print("❌ Not enough data for dashboard")
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('PC-Arduino Interface Performance Dashboard', fontsize=14)
        
        times = list(self.metrics['timestamps'])
        start_time = times[0]
        relative_times = [(t - start_time) / 60 for t in times]  # Minutes
        
        # CPU Usage
        ax1.plot(relative_times, list(self.metrics['cpu_usage']), 'r-', label='CPU %')
        ax1.set_ylabel('CPU Usage (%)')
        ax1.set_title('CPU Performance')
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.5, label='Critical Limit')
        ax1.legend()
        
        # Memory Usage
        ax2.plot(relative_times, list(self.metrics['memory_mb']), 'b-', label='Memory (MB)')
        ax2.set_ylabel('Memory (MB)')
        ax2.set_title('Memory Usage')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # Communication Latency
        if self.metrics['communication_latency']:
            latency_times = relative_times[-len(self.metrics['communication_latency']):]
            ax3.plot(latency_times, list(self.metrics['communication_latency']), 'g-', label='Latency (ms)')
            ax3.set_ylabel('Latency (ms)')
            ax3.set_title('Communication Performance')
            ax3.grid(True, alpha=0.3)
            ax3.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='Target')
            ax3.legend()
        
        # Processing Rate
        if self.metrics['processing_rate']:
            rate_times = relative_times[-len(self.metrics['processing_rate']):]
            ax4.plot(rate_times, list(self.metrics['processing_rate']), 'purple', label='Rate (Hz)')
            ax4.set_ylabel('Rate (Hz)')
            ax4.set_xlabel('Time (minutes)')
            ax4.set_title('Processing Rate')
            ax4.grid(True, alpha=0.3)
            ax4.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='Target')
            ax4.legend()
        
        plt.tight_layout()
        plt.show()

# Integration-Funktionen für optimized interface
def integrate_performance_monitoring(analyzer, arduino_data_queue):
    """Integriere Performance-Monitoring in main interface"""
    
    def monitor_queue():
        """Überwache Queue-Performance"""
        while analyzer.running:
            try:
                queue_size = arduino_data_queue.qsize()
                analyzer.log_queue_size(queue_size)
                time.sleep(1)
            except:
                pass
                
    threading.Thread(target=monitor_queue, daemon=True).start()

if __name__ == "__main__":
    # Standalone Performance Test
    print("🧪 Performance Analyzer Test")
    
    analyzer = PerformanceAnalyzer()
    analyzer.start_monitoring()
    
    # Simuliere Performance-Daten
    for i in range(20):
        # Simuliere Communication Events
        latency = np.random.normal(50, 20)  # 50ms ± 20ms
        success = np.random.random() > 0.05  # 5% Fehlerrate
        analyzer.log_communication_event(latency, success)
        
        # Simuliere Processing Rate
        rate = np.random.normal(15, 3)  # 15Hz ± 3Hz
        analyzer.log_processing_rate(rate)
        
        time.sleep(0.1)
    
    time.sleep(2)  # Sammle System-Metriken
    
    analyzer.stop_monitoring()
    
    # Zeige Report
    print(analyzer.create_performance_report())
    
    # Zeige Dashboard
    analyzer.plot_performance_dashboard()
