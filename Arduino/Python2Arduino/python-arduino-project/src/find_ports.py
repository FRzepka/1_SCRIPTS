import serial.tools.list_ports

def find_arduino_ports():
    """Findet alle verfügbaren COM-Ports und zeigt Arduino-spezifische an"""
    print("🔍 Suche nach verfügbaren COM-Ports...\n")
    
    ports = serial.tools.list_ports.comports()
    arduino_ports = []
    
    if not ports:
        print("❌ Keine COM-Ports gefunden!")
        return
    
    print("📋 Verfügbare COM-Ports:")
    print("-" * 50)
    
    for port in ports:
        is_arduino = any(keyword in port.description.lower() for keyword in 
                        ['arduino', 'ch340', 'cp210', 'ftdi', 'usb serial'])
        
        status = "🔵 ARDUINO" if is_arduino else "⚪ Andere"
        print(f"{status} {port.device} - {port.description}")
        
        if is_arduino:
            arduino_ports.append(port.device)
    
    print("-" * 50)
    
    if arduino_ports:
        print(f"\n✅ Mögliche Arduino-Ports gefunden: {', '.join(arduino_ports)}")
        return arduino_ports[0]  # Gibt den ersten Arduino-Port zurück
    else:
        print("\n⚠️ Kein Arduino-spezifischer Port erkannt.")
        if ports:
            print(f"Versuche es mit: {ports[0].device}")
            return ports[0].device
    
    return None

def test_port_connection(port):
    """Testet ob ein Port verfügbar ist"""
    try:
        import serial
        import time
        
        print(f"\n🔧 Teste Verbindung zu {port}...")
        ser = serial.Serial(port, 9600, timeout=1)
        time.sleep(2)
        ser.close()
        print(f"✅ {port} ist verfügbar!")
        return True
    except Exception as e:
        print(f"❌ {port} ist nicht verfügbar: {e}")
        return False

if __name__ == "__main__":
    arduino_port = find_arduino_ports()
    
    if arduino_port:
        test_port_connection(arduino_port)
        print(f"\n💡 Verwende diesen Port in deinem Script: '{arduino_port}'")
