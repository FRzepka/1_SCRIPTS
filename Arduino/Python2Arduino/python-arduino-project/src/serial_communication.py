def send_data_to_arduino(data, port='COM13', baudrate=9600):
    import serial
    import time

    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            time.sleep(2)  # Wait for the connection to establish
            for number in data:
                ser.write(f"{number}\n".encode('utf-8'))  # Send data as bytes
                time.sleep(0.1)  # Small delay to ensure data is sent
    except serial.SerialException as e:
        print(f"Error: {e}")

def read_data_from_arduino(port='COM13', baudrate=9600):
    import serial

    try:
        with serial.Serial(port, baudrate, timeout=1) as ser:
            while True:
                if ser.in_waiting > 0:
                    line = ser.readline().decode('utf-8').rstrip()
                    print(f"Received: {line}")
    except serial.SerialException as e:
        print(f"Error: {e}")