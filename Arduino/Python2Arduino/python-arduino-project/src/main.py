from data_generator import generate_random_numbers
from serial_communication import send_data_to_arduino
import time

def main():
    # Specify the number of random values to generate
    num_values = 10
    
    # Generate random numbers
    random_numbers = generate_random_numbers(num_values)
    
    print(f"Generierte Zahlen: {random_numbers}")
    print("Sende Daten an Arduino...")
    
    # Send all numbers to the Arduino at once
    send_data_to_arduino(random_numbers, port='COM13')
    
    print("Daten erfolgreich gesendet!")

if __name__ == "__main__":
    main()