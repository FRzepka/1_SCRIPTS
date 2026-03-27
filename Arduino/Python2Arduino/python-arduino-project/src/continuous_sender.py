import serial
import time
import random
import math

class ContinuousArduinoSender:
    def __init__(self, port='COM13', baudrate=9600):
        self.port = port
        self.baudrate = baudrate
        self.arduino = None
        
    def connect(self):
        try:
            self.arduino = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2)  # Arduino braucht Zeit zum Starten
            print(f"✓ Verbunden mit Arduino auf {self.port}")
            print("Drücke Ctrl+C um zu stoppen...\n")
            return True
        except Exception as e:
            print(f"❌ Fehler beim Verbinden: {e}")
            print("Überprüfe ob:")
            print("1. Arduino angeschlossen ist")
            print("2. Der richtige COM-Port ausgewählt ist")
            print("3. Arduino IDE geschlossen ist")
            return False
    
    def send_number(self, number):
        if self.arduino:
            message = f"{number}\n"
            self.arduino.write(message.encode())
            print(f"📤 Gesendet: {number}")
            time.sleep(0.1)
    
    def disconnect(self):
        if self.arduino:
            self.arduino.close()
            print("\n🔌 Verbindung getrennt")

def generate_sine_wave(counter, amplitude=50, offset=50):
    """Generiert eine Sinuswelle für schöne Plots"""
    return int(amplitude * math.sin(counter * 0.1) + offset)

def generate_random_walk(current_value, step_size=5, min_val=0, max_val=100):
    """Generiert einen Random Walk"""
    change = random.randint(-step_size, step_size)
    new_value = current_value + change
    return max(min_val, min(max_val, new_value))

def main():
    # Arduino verbinden
    sender = ContinuousArduinoSender(port='COM13')
    
    if not sender.connect():
        return
    
    # Variablen für verschiedene Datentypen
    counter = 0
    random_walk_value = 50
    
    try:
        while True:
            # Wähle einen Datentyp (du kannst das ändern)
            data_type = input("\nWähle Datentyp:\n1 = Zufallszahlen\n2 = Sinuswelle\n3 = Random Walk\n4 = Zähler\nEnter = Standardmodus (Random): ").strip()
            
            if data_type == "":
                data_type = "1"
            
            print(f"\nStarte Übertragung... (Drücke Ctrl+C zum Stoppen)")
            
            while True:
                if data_type == "1":
                    # Zufallszahlen (0-100)
                    number = random.randint(0, 100)
                elif data_type == "2":
                    # Sinuswelle
                    number = generate_sine_wave(counter)
                elif data_type == "3":
                    # Random Walk
                    random_walk_value = generate_random_walk(random_walk_value)
                    number = random_walk_value
                elif data_type == "4":
                    # Einfacher Zähler
                    number = counter % 100
                else:
                    number = random.randint(0, 100)
                
                sender.send_number(number)
                counter += 1
                
                # Pause zwischen Sendungen (du kannst das anpassen)
                time.sleep(0.5)
                
    except KeyboardInterrupt:
        print("\n⏹️ Übertragung gestoppt...")
    except Exception as e:
        print(f"\n❌ Fehler: {e}")
    finally:
        sender.disconnect()

if __name__ == "__main__":
    main()
