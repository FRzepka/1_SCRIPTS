"""
Real-time Arduino LSTM Monitoring Dashboard

Live-Überwachung des Arduino SOC-Vorhersagesystems:
- Echtzeitgraphen von SOC-Vorhersagen
- Performance-Metriken
- Kommunikationsstatistiken
- Fehlerüberwachung

Verwendung:
    python monitoring_dashboard.py --port COM3
    python monitoring_dashboard.py --port COM3 --demo-mode
"""

import tkinter as tk
from tkinter import ttk, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import serial
import json
import time
import threading
import queue
from collections import deque
import numpy as np
import argparse
from datetime import datetime

class ArduinoMonitor:
    def __init__(self, root, port='COM13', demo_mode=False):
        self.root = root
        self.port = port
        self.demo_mode = demo_mode
        self.arduino = None
        self.running = False
        
        # Data storage
        self.max_points = 500
        self.data = {
            'time': deque(maxlen=self.max_points),
            'true_soc': deque(maxlen=self.max_points),
            'pred_soc': deque(maxlen=self.max_points),
            'voltage': deque(maxlen=self.max_points),
            'current': deque(maxlen=self.max_points),
            'inference_time': deque(maxlen=self.max_points),
            'error': deque(maxlen=self.max_points)
        }
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'lstm_predictions': 0,
            'simple_predictions': 0,
            'avg_inference_time': 0,
            'max_inference_time': 0,
            'communication_errors': 0,
            'last_update': None,
            'uptime': 0
        }
        
        # Communication queue
        self.data_queue = queue.Queue()
        
        self.setup_ui()
        self.setup_plots()
        
        if not demo_mode:
            self.connect_arduino()
        
        self.start_monitoring()
    
    def setup_ui(self):
        """Setup main UI"""
        self.root.title("Arduino LSTM SOC Monitor")
        self.root.geometry("1200x800")
        
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="System Status")
        status_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Connection status
        self.status_label = ttk.Label(status_frame, text="🔌 Disconnected", font=('Arial', 12))
        self.status_label.pack(side=tk.LEFT, padx=10, pady=5)
        
        # Control buttons
        button_frame = ttk.Frame(status_frame)
        button_frame.pack(side=tk.RIGHT, padx=10, pady=5)
        
        self.connect_btn = ttk.Button(button_frame, text="Connect", command=self.toggle_connection)
        self.connect_btn.pack(side=tk.LEFT, padx=5)
        
        self.reset_btn = ttk.Button(button_frame, text="Reset Stats", command=self.reset_stats)
        self.reset_btn.pack(side=tk.LEFT, padx=5)
        
        # Statistics frame
        stats_frame = ttk.LabelFrame(main_frame, text="Performance Statistics")
        stats_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Create stats labels
        stats_grid = ttk.Frame(stats_frame)
        stats_grid.pack(fill=tk.X, padx=10, pady=5)
        
        self.stat_labels = {}
        
        # Row 1
        ttk.Label(stats_grid, text="Total Predictions:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.stat_labels['total'] = ttk.Label(stats_grid, text="0", font=('Arial', 10, 'bold'))
        self.stat_labels['total'].grid(row=0, column=1, sticky=tk.W, padx=5)\n        \n        ttk.Label(stats_grid, text=\"LSTM/Simple:\").grid(row=0, column=2, sticky=tk.W, padx=5)\n        self.stat_labels['model_split'] = ttk.Label(stats_grid, text=\"0/0\", font=('Arial', 10, 'bold'))\n        self.stat_labels['model_split'].grid(row=0, column=3, sticky=tk.W, padx=5)\n        \n        ttk.Label(stats_grid, text=\"Avg Inference:\").grid(row=0, column=4, sticky=tk.W, padx=5)\n        self.stat_labels['avg_time'] = ttk.Label(stats_grid, text=\"0.0 ms\", font=('Arial', 10, 'bold'))\n        self.stat_labels['avg_time'].grid(row=0, column=5, sticky=tk.W, padx=5)\n        \n        # Row 2\n        ttk.Label(stats_grid, text=\"Communication Errors:\").grid(row=1, column=0, sticky=tk.W, padx=5)\n        self.stat_labels['errors'] = ttk.Label(stats_grid, text=\"0\", font=('Arial', 10, 'bold'))\n        self.stat_labels['errors'].grid(row=1, column=1, sticky=tk.W, padx=5)\n        \n        ttk.Label(stats_grid, text=\"Max Inference:\").grid(row=1, column=2, sticky=tk.W, padx=5)\n        self.stat_labels['max_time'] = ttk.Label(stats_grid, text=\"0.0 ms\", font=('Arial', 10, 'bold'))\n        self.stat_labels['max_time'].grid(row=1, column=3, sticky=tk.W, padx=5)\n        \n        ttk.Label(stats_grid, text=\"Uptime:\").grid(row=1, column=4, sticky=tk.W, padx=5)\n        self.stat_labels['uptime'] = ttk.Label(stats_grid, text=\"00:00:00\", font=('Arial', 10, 'bold'))\n        self.stat_labels['uptime'].grid(row=1, column=5, sticky=tk.W, padx=5)\n        \n        # Plot frame\n        plot_frame = ttk.Frame(main_frame)\n        plot_frame.pack(fill=tk.BOTH, expand=True)\n        \n        self.plot_frame = plot_frame\n    \n    def setup_plots(self):\n        \"\"\"Setup matplotlib plots\"\"\"\n        # Create figure with subplots\n        self.fig, ((self.ax1, self.ax2), (self.ax3, self.ax4)) = plt.subplots(2, 2, figsize=(12, 8))\n        self.fig.suptitle('Arduino LSTM SOC Monitor', fontsize=14)\n        \n        # Plot 1: SOC Comparison\n        self.ax1.set_title('SOC Predictions vs True Values')\n        self.ax1.set_xlabel('Time')\n        self.ax1.set_ylabel('SOC')\n        self.ax1.grid(True, alpha=0.3)\n        self.line1_true, = self.ax1.plot([], [], 'b-', label='True SOC', linewidth=2)\n        self.line1_pred, = self.ax1.plot([], [], 'r-', label='Predicted SOC', linewidth=2)\n        self.ax1.legend()\n        self.ax1.set_ylim(0, 1)\n        \n        # Plot 2: Voltage and Current\n        self.ax2.set_title('Battery Voltage and Current')\n        self.ax2.set_xlabel('Time')\n        self.ax2_twin = self.ax2.twinx()\n        self.ax2.set_ylabel('Voltage [V]', color='blue')\n        self.ax2_twin.set_ylabel('Current [A]', color='red')\n        self.ax2.grid(True, alpha=0.3)\n        self.line2_voltage, = self.ax2.plot([], [], 'b-', label='Voltage', linewidth=2)\n        self.line2_current, = self.ax2_twin.plot([], [], 'r-', label='Current', linewidth=2)\n        \n        # Plot 3: Prediction Error\n        self.ax3.set_title('Prediction Error (True - Predicted)')\n        self.ax3.set_xlabel('Time')\n        self.ax3.set_ylabel('Error')\n        self.ax3.grid(True, alpha=0.3)\n        self.line3_error, = self.ax3.plot([], [], 'g-', linewidth=2)\n        self.ax3.axhline(y=0, color='k', linestyle='--', alpha=0.5)\n        \n        # Plot 4: Inference Time\n        self.ax4.set_title('Inference Time Performance')\n        self.ax4.set_xlabel('Time')\n        self.ax4.set_ylabel('Inference Time [ms]')\n        self.ax4.grid(True, alpha=0.3)\n        self.line4_time, = self.ax4.plot([], [], 'orange', linewidth=2)\n        \n        plt.tight_layout()\n        \n        # Embed in tkinter\n        self.canvas = FigureCanvasTkAgg(self.fig, self.plot_frame)\n        self.canvas.draw()\n        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)\n    \n    def connect_arduino(self):\n        \"\"\"Connect to Arduino\"\"\"\n        try:\n            self.arduino = serial.Serial(self.port, 115200, timeout=1)\n            time.sleep(2)  # Arduino reset time\n            \n            self.status_label.config(text=f\"🟢 Connected to {self.port}\")\n            self.connect_btn.config(text=\"Disconnect\")\n            return True\n            \n        except Exception as e:\n            messagebox.showerror(\"Connection Error\", f\"Failed to connect to {self.port}:\\n{e}\")\n            self.status_label.config(text=\"🔴 Connection Failed\")\n            return False\n    \n    def disconnect_arduino(self):\n        \"\"\"Disconnect from Arduino\"\"\"\n        if self.arduino and self.arduino.is_open:\n            self.arduino.close()\n        \n        self.arduino = None\n        self.status_label.config(text=\"🔌 Disconnected\")\n        self.connect_btn.config(text=\"Connect\")\n    \n    def toggle_connection(self):\n        \"\"\"Toggle Arduino connection\"\"\"\n        if self.arduino and self.arduino.is_open:\n            self.disconnect_arduino()\n        else:\n            self.connect_arduino()\n    \n    def reset_stats(self):\n        \"\"\"Reset all statistics\"\"\"\n        self.stats = {\n            'total_predictions': 0,\n            'lstm_predictions': 0,\n            'simple_predictions': 0,\n            'avg_inference_time': 0,\n            'max_inference_time': 0,\n            'communication_errors': 0,\n            'last_update': None,\n            'uptime': 0\n        }\n        \n        # Clear data\n        for key in self.data:\n            self.data[key].clear()\n        \n        self.update_stats_display()\n    \n    def update_stats_display(self):\n        \"\"\"Update statistics display\"\"\"\n        self.stat_labels['total'].config(text=str(self.stats['total_predictions']))\n        self.stat_labels['model_split'].config(\n            text=f\"{self.stats['lstm_predictions']}/{self.stats['simple_predictions']}\"\n        )\n        self.stat_labels['avg_time'].config(text=f\"{self.stats['avg_inference_time']:.1f} ms\")\n        self.stat_labels['max_time'].config(text=f\"{self.stats['max_inference_time']:.1f} ms\")\n        self.stat_labels['errors'].config(text=str(self.stats['communication_errors']))\n        \n        # Uptime\n        if self.stats['last_update']:\n            uptime_seconds = time.time() - self.stats['uptime']\n            hours = int(uptime_seconds // 3600)\n            minutes = int((uptime_seconds % 3600) // 60)\n            seconds = int(uptime_seconds % 60)\n            self.stat_labels['uptime'].config(text=f\"{hours:02d}:{minutes:02d}:{seconds:02d}\")\n    \n    def data_reader_thread(self):\n        \"\"\"Thread for reading Arduino data\"\"\"\n        while self.running:\n            try:\n                if self.demo_mode:\n                    # Generate demo data\n                    current_time = time.time()\n                    true_soc = 0.5 + 0.3 * np.sin(current_time * 0.1)\n                    pred_soc = true_soc + np.random.normal(0, 0.02)\n                    voltage = 3.2 + true_soc * 0.4 + np.random.normal(0, 0.01)\n                    current = np.random.uniform(-2, 2)\n                    inference_time = np.random.uniform(8, 15)\n                    \n                    data = {\n                        'pred_soc': pred_soc,\n                        'true_soc': true_soc,\n                        'voltage': voltage,\n                        'current': current,\n                        'inference_time_ms': inference_time,\n                        'model_type': 'LSTM' if np.random.random() > 0.3 else 'Simple'\n                    }\n                    \n                    self.data_queue.put(data)\n                    time.sleep(0.1)  # 10 Hz demo data\n                    \n                elif self.arduino and self.arduino.is_open:\n                    if self.arduino.in_waiting:\n                        line = self.arduino.readline().decode().strip()\n                        if line.startswith('{'):\n                            try:\n                                data = json.loads(line)\n                                self.data_queue.put(data)\n                            except json.JSONDecodeError:\n                                self.stats['communication_errors'] += 1\n                \n                else:\n                    time.sleep(0.1)\n                    \n            except Exception as e:\n                print(f\"Data reader error: {e}\")\n                time.sleep(1)\n    \n    def update_plots(self, frame):\n        \"\"\"Update plots with new data\"\"\"\n        # Process new data from queue\n        while not self.data_queue.empty():\n            try:\n                data = self.data_queue.get_nowait()\n                current_time = time.time()\n                \n                # Add to data storage\n                self.data['time'].append(current_time)\n                self.data['true_soc'].append(data.get('true_soc', 0))\n                self.data['pred_soc'].append(data.get('pred_soc', 0))\n                self.data['voltage'].append(data.get('voltage', 0))\n                self.data['current'].append(data.get('current', 0))\n                self.data['inference_time'].append(data.get('inference_time_ms', 0))\n                \n                # Calculate error\n                error = data.get('true_soc', 0) - data.get('pred_soc', 0)\n                self.data['error'].append(error)\n                \n                # Update statistics\n                self.stats['total_predictions'] += 1\n                if data.get('model_type') == 'LSTM':\n                    self.stats['lstm_predictions'] += 1\n                else:\n                    self.stats['simple_predictions'] += 1\n                \n                inference_time = data.get('inference_time_ms', 0)\n                if inference_time > 0:\n                    # Update average inference time\n                    total = self.stats['total_predictions']\n                    current_avg = self.stats['avg_inference_time']\n                    self.stats['avg_inference_time'] = (current_avg * (total - 1) + inference_time) / total\n                    \n                    # Update max inference time\n                    if inference_time > self.stats['max_inference_time']:\n                        self.stats['max_inference_time'] = inference_time\n                \n                self.stats['last_update'] = current_time\n                if self.stats['uptime'] == 0:\n                    self.stats['uptime'] = current_time\n                \n            except queue.Empty:\n                break\n        \n        # Update plots if we have data\n        if len(self.data['time']) > 0:\n            times = list(self.data['time'])\n            \n            # Convert to relative time (seconds from start)\n            if len(times) > 1:\n                start_time = times[0]\n                rel_times = [(t - start_time) for t in times]\n            else:\n                rel_times = [0]\n            \n            # Update SOC plot\n            self.line1_true.set_data(rel_times, self.data['true_soc'])\n            self.line1_pred.set_data(rel_times, self.data['pred_soc'])\n            \n            # Update voltage/current plot\n            self.line2_voltage.set_data(rel_times, self.data['voltage'])\n            self.line2_current.set_data(rel_times, self.data['current'])\n            \n            # Update error plot\n            self.line3_error.set_data(rel_times, self.data['error'])\n            \n            # Update inference time plot\n            self.line4_time.set_data(rel_times, self.data['inference_time'])\n            \n            # Auto-scale plots\n            for ax in [self.ax1, self.ax2, self.ax3, self.ax4]:\n                ax.relim()\n                ax.autoscale_view()\n            \n            # Auto-scale twin axis\n            self.ax2_twin.relim()\n            self.ax2_twin.autoscale_view()\n        \n        # Update statistics display\n        self.update_stats_display()\n        \n        return [self.line1_true, self.line1_pred, self.line2_voltage, \n                self.line2_current, self.line3_error, self.line4_time]\n    \n    def start_monitoring(self):\n        \"\"\"Start monitoring\"\"\"\n        self.running = True\n        \n        # Start data reader thread\n        self.reader_thread = threading.Thread(target=self.data_reader_thread, daemon=True)\n        self.reader_thread.start()\n        \n        # Start plot animation\n        self.animation = FuncAnimation(self.fig, self.update_plots, interval=500, blit=False)\n        \n        print(\"🚀 Monitoring started\")\n    \n    def stop_monitoring(self):\n        \"\"\"Stop monitoring\"\"\"\n        self.running = False\n        if hasattr(self, 'animation'):\n            self.animation.event_source.stop()\n        self.disconnect_arduino()\n        print(\"⏹️ Monitoring stopped\")\n    \n    def on_closing(self):\n        \"\"\"Handle window closing\"\"\"\n        self.stop_monitoring()\n        self.root.destroy()\n\ndef main():\n    parser = argparse.ArgumentParser(description='Arduino LSTM Monitoring Dashboard')\n    parser.add_argument('--port', default='COM3', help='Arduino COM port')\n    parser.add_argument('--demo-mode', action='store_true', help='Run with demo data (no Arduino needed)')\n    \n    args = parser.parse_args()\n    \n    root = tk.Tk()\n    monitor = ArduinoMonitor(root, args.port, args.demo_mode)\n    \n    # Handle window closing\n    root.protocol(\"WM_DELETE_WINDOW\", monitor.on_closing)\n    \n    print(f\"🖥️ Starting Arduino Monitor (Port: {args.port}, Demo: {args.demo_mode})\")\n    root.mainloop()\n\nif __name__ == \"__main__\":\n    main()
