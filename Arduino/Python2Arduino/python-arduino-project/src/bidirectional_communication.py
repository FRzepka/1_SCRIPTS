import serial
import time
import random
import math
import threading

class ArduinoCommunicator:
    def __init__(self, port='COM13', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        self.running = False
        
    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino braucht Zeit zum Starten
            print(f"✓ Verbunden mit Arduino auf {self.port}")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Verbinden: {e}")
            print("Schließe die Arduino IDE (Serial Monitor) und versuche es nochmal!")
            return False
    
    def read_from_arduino(self):
        """Liest kontinuierlich Daten vom Arduino"""
        while self.running and self.arduino:
            try:
                if self.arduino.in_waiting > 0:
                    line = self.arduino.readline().decode('utf-8').rstrip()
                    if line:
                        print(f"🔴 Arduino: {line}")
            except Exception as e:
                print(f"Lesefehler: {e}")
                break
            time.sleep(0.01)
    
    def send_number(self, number):
        if self.arduino:
            message = f"{number}\n"
            self.arduino.write(message.encode())
            print(f"📤 Python: Gesendet -> {number}")
            time.sleep(0.1)
    
    def start_reading(self):
        """Startet einen separaten Thread zum Lesen"""
        self.running = True
        read_thread = threading.Thread(target=self.read_from_arduino)
        read_thread.daemon = True
        read_thread.start()
        return read_thread
    
    def stop(self):
        self.running = False
        if self.arduino:
            self.arduino.close()
            print("\n🔌 Verbindung getrennt")

def generate_sine_wave(counter, amplitude=50, offset=50):
    """Generiert eine Sinuswelle für schöne Plots"""
    return int(amplitude * math.sin(counter * 0.1) + offset)

def main():
    print("🚀 Arduino Bidirektionale Kommunikation")
    print("=" * 50)
    
    # Arduino verbinden
    comm = ArduinoCommunicator(port='COM13')
    
    if not comm.connect():
        input("Drücke Enter um zu beenden...")
        return
    
    # Starte das Lesen in einem separaten Thread
    read_thread = comm.start_reading()
    
    print("\n📡 Kommunikation gestartet!")
    print("Drücke Ctrl+C um zu stoppen...\n")
    
    counter = 0
    
    try:
        while True:
            # Verschiedene Datentypen zum Testen
            data_type = counter % 4
            
            if data_type == 0:
                # Zufallszahlen
                number = random.randint(0, 100)
            elif data_type == 1:
                # Sinuswelle
                number = generate_sine_wave(counter)
            elif data_type == 2:
                # Aufsteigende Zahlen
                number = (counter * 5) % 100
            else:
                # Random Walk
                number = max(0, min(100, 50 + random.randint(-10, 10)))
            
            comm.send_number(number)
            counter += 1
            
            # Pause zwischen Sendungen
            time.sleep(1.0)
            
    except KeyboardInterrupt:
        print("\n⏹️ Übertragung gestoppt...")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
    finally:
        comm.stop()
        # Kurz warten bis der Read-Thread beendet ist
        time.sleep(0.5)

if __name__ == "__main__":
    main()
