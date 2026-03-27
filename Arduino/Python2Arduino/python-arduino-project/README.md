# Python-Arduino Project

This project demonstrates how to generate random numbers in Python and send them to an Arduino for processing and visualization. The Python script generates random numbers, which are then transmitted to the Arduino via serial communication. The Arduino receives the data and can display it in the Serial Monitor.

## Project Structure

```
python-arduino-project
├── src
│   ├── data_generator.py       # Generates random numbers
│   ├── serial_communication.py  # Handles serial communication with Arduino
│   └── main.py                 # Entry point for the Python application
├── arduino
│   └── data_receiver.ino       # Arduino sketch to receive and process data
├── requirements.txt            # Python dependencies
└── README.md                   # Project documentation
```

## Setup Instructions

1. **Clone the repository:**
   ```
   git clone <repository-url>
   cd python-arduino-project
   ```

2. **Install Python dependencies:**
   Make sure you have Python installed. Then, install the required packages using pip:
   ```
   pip install -r requirements.txt
   ```

3. **Upload the Arduino sketch:**
   Open the `data_receiver.ino` file in the Arduino IDE and upload it to your Arduino board.

4. **Run the Python script:**
   Execute the `main.py` script to start generating and sending random numbers to the Arduino:
   ```
   python src/main.py
   ```

## Usage Example

After running the Python script, open the Serial Monitor in the Arduino IDE to see the received random numbers being printed. You can adjust the number of random values generated in the `main.py` file as needed.

## Notes

- Ensure that the baud rate in the Arduino sketch matches the baud rate set in the Python script.
- You may need to adjust the serial port settings based on your operating system and Arduino board.