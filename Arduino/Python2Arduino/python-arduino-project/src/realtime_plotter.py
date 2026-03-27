import serial
import time
import random
import math
import threading
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from collections import deque
import numpy as np

class ArduinoRealtimePlotter:
    def __init__(self, port='COM13', baudrate=9600, max_points=100):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.running = False
        self.max_points = max_points
        
        # Daten für den Plot
        self.sent_data = deque(maxlen=max_points)
        self.received_data = deque(maxlen=max_points)
        self.quadrat_data = deque(maxlen=max_points)
        self.time_data = deque(maxlen=max_points)
        
        # Plot Setup
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.line1, = self.ax.plot([], [], 'b-', label='Gesendete Werte', linewidth=2)
        self.line2, = self.ax.plot([], [], 'r-', label='Empfangene Werte', linewidth=2)
        self.line3, = self.ax.plot([], [], 'g-', label='Quadrat (skaliert)', linewidth=2)
        
        self.ax.set_xlabel('Zeit (Sekunden)')
        self.ax.set_ylabel('Wert')
        self.ax.set_title('Arduino-Python Echtzeit Kommunikation')
        self.ax.legend()
        self.ax.grid(True)
        
        self.start_time = time.time()
        
    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)
            print(f"✓ Arduino verbunden auf {self.port}")
            print("✓ Grafisches Plotting gestartet")
            print("Schließe das Plot-Fenster um zu beenden\n")
            return True
        except Exception as e:
            print(f"❌ Fehler: {e}")
            print("Schließe die Arduino IDE und versuche es nochmal!")
            return False
    
    def read_from_arduino(self):
        """Liest und verarbeitet Arduino Daten"""
        current_received = None
        current_quadrat = None
        
        while self.running and self.arduino:
            try:
                if self.arduino.in_waiting > 0:
                    line = self.arduino.readline().decode('utf-8').rstrip()
                    if line:
                        print(f"Arduino: {line}")
                        
                        # Parse empfangene Werte
                        if line.startswith("Empfangen:"):
                            try:
                                current_received = int(line.split(":")[1].strip())
                            except:
                                pass
                        elif line.startswith("Quadrat:"):
                            try:
                                quadrat_value = int(line.split(":")[1].strip())
                                current_quadrat = quadrat_value / 100  # Skalierung für Plot
                            except:
                                pass
                        elif line == "---" and current_received is not None:
                            # Ende eines Datensatzes - zu Plot hinzufügen
                            current_time = time.time() - self.start_time
                            self.time_data.append(current_time)
                            self.received_data.append(current_received)
                            self.quadrat_data.append(current_quadrat if current_quadrat else 0)
                            
                            current_received = None
                            current_quadrat = None
                            
            except Exception as e:
                print(f"Lesefehler: {e}")
                break
            time.sleep(0.01)
    
    def send_number(self, number):
        if self.arduino:
            message = f"{number}\n"
            self.arduino.write(message.encode())
            
            # Zu Plot hinzufügen
            current_time = time.time() - self.start_time
            if len(self.time_data) == 0 or current_time > self.time_data[-1]:
                self.time_data.append(current_time)
                self.sent_data.append(number)
                # Falls wir mehr sent als received haben, fülle mit None auf
                if len(self.received_data) < len(self.sent_data):
                    self.received_data.append(None)
                    self.quadrat_data.append(None)
            
            print(f"📤 Gesendet: {number}")
            time.sleep(0.1)
    
    def update_plot(self, frame):
        """Update Funktion für Animation"""
        if len(self.time_data) > 0:
            # Daten für Plot vorbereiten
            times = list(self.time_data)
            sent = list(self.sent_data)
            received = [x for x in self.received_data if x is not None]
            quadrat = [x for x in self.quadrat_data if x is not None]
            
            # Lines updaten
            if len(times) == len(sent):
                self.line1.set_data(times, sent)
            
            if len(times) >= len(received) and len(received) > 0:
                recv_times = times[-len(received):]
                self.line2.set_data(recv_times, received)
            
            if len(times) >= len(quadrat) and len(quadrat) > 0:
                quad_times = times[-len(quadrat):]
                self.line3.set_data(quad_times, quadrat)
            
            # Achsen anpassen
            if times:
                self.ax.set_xlim(max(0, times[-1] - 30), times[-1] + 2)  # Letzten 30 Sekunden
                
                all_values = []
                if sent: all_values.extend(sent)
                if received: all_values.extend(received)
                if quadrat: all_values.extend(quadrat)
                
                if all_values:
                    y_min = min(all_values) - 10
                    y_max = max(all_values) + 10
                    self.ax.set_ylim(y_min, y_max)
        
        return self.line1, self.line2, self.line3
    
    def start_communication(self):
        """Startet die Kommunikation und das Plotting"""
        if not self.connect():
            return
        
        # Lese-Thread starten
        self.running = True
        read_thread = threading.Thread(target=self.read_from_arduino)
        read_thread.daemon = True
        read_thread.start()
        
        # Sende-Thread starten
        def send_data():
            counter = 0
            try:
                while self.running:
                    # Verschiedene Datentypen senden
                    data_type = counter % 3
                    
                    if data_type == 0:
                        number = random.randint(10, 90)
                    elif data_type == 1:
                        number = int(50 + 40 * math.sin(counter * 0.2))
                    else:
                        number = int(50 + 30 * math.sin(counter * 0.1) * math.cos(counter * 0.05))
                    
                    self.send_number(number)
                    counter += 1
                    time.sleep(2.0)  # Alle 2 Sekunden
                    
            except Exception as e:
                print(f"Sende-Fehler: {e}")
        
        send_thread = threading.Thread(target=send_data)
        send_thread.daemon = True
        send_thread.start()
        
        # Animation starten
        try:
            ani = animation.FuncAnimation(self.fig, self.update_plot, interval=100, blit=False)
            plt.show()
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
    
    def stop(self):
        self.running = False
        if self.arduino:
            self.arduino.close()
            print("🔌 Verbindung getrennt")

def main():
    print("🚀 Arduino Realtime Plotter")
    print("=" * 40)
    print("Dieses Script zeigt LIVE Plots der Arduino-Kommunikation!")
    print("Schließe das Plot-Fenster um zu beenden.\n")
    
    plotter = ArduinoRealtimePlotter(port='COM13')
    plotter.start_communication()

if __name__ == "__main__":
    main()
