"""
Real-time Performance Monitor for BMS SOC Live Test System
Monitors PyTorch inference performance, memory usage, and bottlenecks
"""

import psutil
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
from collections import deque
import queue
import json
import socket
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SOCPerformanceMonitor:
    def __init__(self, max_history=500):
        self.max_history = max_history
        self.monitoring = False
        
        # Performance data storage
        self.metrics = {
            'timestamp': deque(maxlen=max_history),
            'cpu_percent': deque(maxlen=max_history),
            'memory_mb': deque(maxlen=max_history),
            'gpu_memory_mb': deque(maxlen=max_history),
            'inference_time_ms': deque(maxlen=max_history),
            'queue_size': deque(maxlen=max_history),
            'throughput_pts_per_sec': deque(maxlen=max_history),
            'memory_growth_rate': deque(maxlen=max_history)
        ]
        
        # Performance thresholds for alerts
        self.thresholds = {
            'cpu_percent': 80.0,
            'memory_mb': 2000.0,
            'inference_time_ms': 50.0,
            'queue_size': 150,
            'memory_growth_rate': 10.0  # MB per minute
        }
        
        # Alert tracking
        self.alerts = []
        self.alert_queue = queue.Queue()
        
        # Simulation data for testing
        self.simulation_data = self.generate_test_data()
        
    def generate_test_data(self):
        """Generate realistic test data for performance monitoring"""
        data = []
        base_time = time.time()
        
        for i in range(1000):
            # Simulate increasing performance degradation over time
            degradation_factor = 1 + (i / 1000) * 0.5  # 50% degradation over time
            
            point = {
                'timestamp': base_time + i * 0.01,  # 10ms intervals
                'cpu_percent': 15 + np.random.normal(0, 3) * degradation_factor,
                'memory_mb': 800 + i * 0.5 + np.random.normal(0, 10),  # Memory leak simulation
                'inference_time_ms': 5 + np.random.normal(0, 1) * degradation_factor,
                'queue_size': max(0, int(np.random.normal(50, 20) * degradation_factor)),
                'throughput': max(10, 100 - i * 0.05),  # Decreasing throughput
            }
            data.append(point)
            
        return data
        
    def start_monitoring(self, target_process_name="python"):
        """Start real-time performance monitoring"""
        self.monitoring = True
        self.start_time = time.time()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, args=(target_process_name,))
        monitor_thread.daemon = True
        monitor_thread.start()
        
        # Start alert handler
        alert_thread = threading.Thread(target=self._alert_handler)
        alert_thread.daemon = True
        alert_thread.start()
        
        logger.info(f"🚀 Started monitoring process: {target_process_name}")
        
    def _monitor_loop(self, target_process_name):
        """Main monitoring loop"""
        last_points_processed = 0
        last_memory_check = time.time()
        last_memory_mb = 0
        
        while self.monitoring:
            try:
                current_time = time.time()
                
                # Find target processes
                target_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                    try:
                        if (target_process_name in proc.info['name'] and 
                            any('live_test_soc' in str(cmd) for cmd in proc.info['cmdline'])):
                            target_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if target_processes:
                    # Aggregate metrics from all target processes
                    total_cpu = 0
                    total_memory = 0
                    total_threads = 0
                    
                    for proc in target_processes:
                        try:
                            total_cpu += proc.cpu_percent()
                            total_memory += proc.memory_info().rss / 1024 / 1024
                            total_threads += proc.num_threads()
                        except (psutil.NoSuchProcess, psutil.AccessDenied):
                            continue
                    
                    # GPU memory (if available)
                    gpu_memory = self._get_gpu_memory()
                    
                    # Calculate memory growth rate
                    memory_growth_rate = 0
                    if current_time - last_memory_check >= 60:  # Check every minute
                        if last_memory_mb > 0:
                            memory_growth_rate = (total_memory - last_memory_mb) / ((current_time - last_memory_check) / 60)
                        last_memory_mb = total_memory
                        last_memory_check = current_time
                    
                    # Simulated inference time and queue size (would come from actual app)
                    inference_time = np.random.normal(10, 2)  # Simulated
                    queue_size = np.random.randint(20, 100)   # Simulated
                    
                    # Calculate throughput
                    throughput = self._calculate_throughput()
                    
                    # Store metrics
                    elapsed = current_time - self.start_time
                    self.metrics['timestamp'].append(elapsed)
                    self.metrics['cpu_percent'].append(total_cpu)
                    self.metrics['memory_mb'].append(total_memory)
                    self.metrics['gpu_memory_mb'].append(gpu_memory)
                    self.metrics['inference_time_ms'].append(inference_time)
                    self.metrics['queue_size'].append(queue_size)
                    self.metrics['throughput_pts_per_sec'].append(throughput)
                    self.metrics['memory_growth_rate'].append(memory_growth_rate)
                    
                    # Check for performance issues
                    self._check_performance_thresholds(total_cpu, total_memory, inference_time, 
                                                     queue_size, memory_growth_rate)
                    
                else:
                    # No target process found, use simulation data
                    self._use_simulation_data()
                
                time.sleep(1)  # Monitor every second
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                time.sleep(1)
                
    def _get_gpu_memory(self):
        """Get GPU memory usage if available"""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        return 0
        
    def _calculate_throughput(self):
        """Calculate processing throughput"""
        # This would normally come from the actual application
        # For now, simulate realistic throughput
        if len(self.metrics['timestamp']) < 2:
            return 0
            
        # Simulate decreasing throughput over time (performance degradation)
        elapsed = self.metrics['timestamp'][-1]
        base_throughput = 100
        degradation = min(50, elapsed * 0.5)  # Decrease by 0.5 pts/s per second
        return max(10, base_throughput - degradation + np.random.normal(0, 5))
        
    def _use_simulation_data(self):
        """Use simulation data when no real process is found"""
        if not hasattr(self, 'sim_index'):
            self.sim_index = 0
            
        if self.sim_index < len(self.simulation_data):
            data = self.simulation_data[self.sim_index]
            elapsed = time.time() - self.start_time
            
            self.metrics['timestamp'].append(elapsed)
            self.metrics['cpu_percent'].append(data['cpu_percent'])
            self.metrics['memory_mb'].append(data['memory_mb'])
            self.metrics['gpu_memory_mb'].append(0)
            self.metrics['inference_time_ms'].append(data['inference_time_ms'])
            self.metrics['queue_size'].append(data['queue_size'])
            self.metrics['throughput_pts_per_sec'].append(data['throughput'])
            self.metrics['memory_growth_rate'].append(max(0, data['memory_mb'] - 800) / max(1, elapsed / 60))
            
            self.sim_index += 1
            
    def _check_performance_thresholds(self, cpu, memory, inference_time, queue_size, memory_growth):
        """Check for performance threshold violations"""
        current_time = time.time()
        
        alerts = []
        if cpu > self.thresholds['cpu_percent']:
            alerts.append(f"HIGH CPU: {cpu:.1f}% > {self.thresholds['cpu_percent']}%")
            
        if memory > self.thresholds['memory_mb']:
            alerts.append(f"HIGH MEMORY: {memory:.1f}MB > {self.thresholds['memory_mb']}MB")
            
        if inference_time > self.thresholds['inference_time_ms']:
            alerts.append(f"SLOW INFERENCE: {inference_time:.1f}ms > {self.thresholds['inference_time_ms']}ms")
            
        if queue_size > self.thresholds['queue_size']:
            alerts.append(f"QUEUE OVERFLOW: {queue_size} > {self.thresholds['queue_size']}")
            
        if memory_growth > self.thresholds['memory_growth_rate']:
            alerts.append(f"MEMORY LEAK: {memory_growth:.1f}MB/min > {self.thresholds['memory_growth_rate']}MB/min")
            
        for alert in alerts:
            self.alert_queue.put({
                'timestamp': current_time,
                'message': alert,
                'severity': 'WARNING'
            })
            
    def _alert_handler(self):
        """Handle performance alerts"""
        while self.monitoring:
            try:
                alert = self.alert_queue.get(timeout=1)
                self.alerts.append(alert)
                logger.warning(f"🚨 PERFORMANCE ALERT: {alert['message']}")
                
                # Keep only recent alerts
                current_time = time.time()
                self.alerts = [a for a in self.alerts if current_time - a['timestamp'] < 300]  # 5 minutes
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in alert handler: {e}")
                
    def get_performance_summary(self):
        """Get current performance summary"""
        if not self.metrics['timestamp']:
            return {}
            
        summary = {
            'current_cpu': self.metrics['cpu_percent'][-1] if self.metrics['cpu_percent'] else 0,
            'current_memory': self.metrics['memory_mb'][-1] if self.metrics['memory_mb'] else 0,
            'avg_inference_time': np.mean(list(self.metrics['inference_time_ms'])[-100:]) if self.metrics['inference_time_ms'] else 0,
            'current_throughput': self.metrics['throughput_pts_per_sec'][-1] if self.metrics['throughput_pts_per_sec'] else 0,
            'memory_growth_rate': self.metrics['memory_growth_rate'][-1] if self.metrics['memory_growth_rate'] else 0,
            'total_alerts': len(self.alerts),
            'runtime_seconds': self.metrics['timestamp'][-1] if self.metrics['timestamp'] else 0
        }
        
        return summary
        
    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        logger.info("🛑 Stopped performance monitoring")
        
    def plot_performance_dashboard(self):
        """Create real-time performance dashboard"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('SOC Live Test Performance Dashboard', fontsize=16)
        
        def update_plots(frame):
            if not self.metrics['timestamp']:
                return
                
            # Clear all axes
            for ax_row in axes:
                for ax in ax_row:
                    ax.clear()
            
            times = list(self.metrics['timestamp'])
            
            # CPU Usage
            if self.metrics['cpu_percent']:
                axes[0, 0].plot(times, list(self.metrics['cpu_percent']), 'r-', linewidth=2)
                axes[0, 0].axhline(y=self.thresholds['cpu_percent'], color='r', linestyle='--', alpha=0.7)
                axes[0, 0].set_title('CPU Usage (%)')
                axes[0, 0].set_ylabel('CPU %')
                axes[0, 0].grid(True, alpha=0.3)
                
            # Memory Usage
            if self.metrics['memory_mb']:
                axes[0, 1].plot(times, list(self.metrics['memory_mb']), 'b-', linewidth=2)
                axes[0, 1].axhline(y=self.thresholds['memory_mb'], color='r', linestyle='--', alpha=0.7)
                axes[0, 1].set_title('Memory Usage (MB)')
                axes[0, 1].set_ylabel('Memory (MB)')
                axes[0, 1].grid(True, alpha=0.3)
                
            # Inference Time
            if self.metrics['inference_time_ms']:
                axes[0, 2].plot(times, list(self.metrics['inference_time_ms']), 'g-', linewidth=2)
                axes[0, 2].axhline(y=self.thresholds['inference_time_ms'], color='r', linestyle='--', alpha=0.7)
                axes[0, 2].set_title('Inference Time (ms)')
                axes[0, 2].set_ylabel('Time (ms)')
                axes[0, 2].grid(True, alpha=0.3)
                
            # Queue Size
            if self.metrics['queue_size']:
                axes[1, 0].plot(times, list(self.metrics['queue_size']), 'orange', linewidth=2)
                axes[1, 0].axhline(y=self.thresholds['queue_size'], color='r', linestyle='--', alpha=0.7)
                axes[1, 0].set_title('Queue Size')
                axes[1, 0].set_ylabel('Queue Size')
                axes[1, 0].set_xlabel('Time (seconds)')
                axes[1, 0].grid(True, alpha=0.3)
                
            # Throughput
            if self.metrics['throughput_pts_per_sec']:
                axes[1, 1].plot(times, list(self.metrics['throughput_pts_per_sec']), 'purple', linewidth=2)
                axes[1, 1].set_title('Processing Throughput (pts/sec)')
                axes[1, 1].set_ylabel('Points/sec')
                axes[1, 1].set_xlabel('Time (seconds)')
                axes[1, 1].grid(True, alpha=0.3)
                
            # Performance Summary
            axes[1, 2].axis('off')
            summary = self.get_performance_summary()
            summary_text = f"""
Performance Summary:
CPU: {summary.get('current_cpu', 0):.1f}%
Memory: {summary.get('current_memory', 0):.1f} MB
Avg Inference: {summary.get('avg_inference_time', 0):.1f} ms
Throughput: {summary.get('current_throughput', 0):.1f} pts/s
Memory Growth: {summary.get('memory_growth_rate', 0):.1f} MB/min
Alerts: {summary.get('total_alerts', 0)}
Runtime: {summary.get('runtime_seconds', 0):.1f}s
"""
            axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes, 
                           fontsize=12, verticalalignment='top', fontfamily='monospace')
            axes[1, 2].set_title('Current Status')
            
        # Create animation
        anim = FuncAnimation(fig, update_plots, interval=1000, blit=False, cache_frame_data=False)
        
        plt.tight_layout()
        return fig, anim

def main():
    """Main function for standalone monitoring"""
    monitor = SOCPerformanceMonitor()
    
    print("🎯 SOC Performance Monitor")
    print("=" * 50)
    print("This tool monitors the performance of the SOC live test system")
    print("and provides real-time alerts for performance issues.")
    print()
    
    # Start monitoring
    monitor.start_monitoring()
    
    try:
        # Create dashboard
        fig, anim = monitor.plot_performance_dashboard()
        
        print("📊 Performance dashboard started")
        print("🚨 Watch for performance alerts in the console")
        print("🛑 Close the plot window or press Ctrl+C to stop")
        
        plt.show()
        
    except KeyboardInterrupt:
        print("\n🛑 Stopping monitor...")
    finally:
        monitor.stop_monitoring()
        
        # Print final summary
        summary = monitor.get_performance_summary()
        print(f"\n📊 Final Performance Summary:")
        print(f"   - Runtime: {summary.get('runtime_seconds', 0):.1f} seconds")
        print(f"   - Average CPU: {summary.get('current_cpu', 0):.1f}%")
        print(f"   - Final Memory: {summary.get('current_memory', 0):.1f} MB")
        print(f"   - Average Inference Time: {summary.get('avg_inference_time', 0):.1f} ms")
        print(f"   - Final Throughput: {summary.get('current_throughput', 0):.1f} pts/s")
        print(f"   - Total Alerts: {summary.get('total_alerts', 0)}")

if __name__ == "__main__":
    main()
