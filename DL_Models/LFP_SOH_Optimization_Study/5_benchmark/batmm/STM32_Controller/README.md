# STM32 Microcontroller

## X_CUBE_AI

Contains the project for loading the microcontroller with the recreated SOH model and preparing it for data streamed by the contents of SOH_Evaluation.

## SOH_Evaluation

- `inference.py` – Runs inference with the given .parquet data file and the recreated model in Python for comparison, stores results in a csv
- `inference_on_microcontroller.py` – Streams data into the microcontroller (which needs to be set up with the project in X_CUBE_AI) and reads the output, stores results in a csv
- `plot_results.py` – Plots the SOH of the given csv

## USB_Tracker

- `tracker.py` – Starts tracking the main power and energy related metrics via the USB Meter and stores the results in a csv
- `plot_tracker.py` – Plots the metrics given the csv from tracker.py (whole timeline)
- `plot_tracker_live.py` – Plots the metrics given the csv from tracker.py while the measurements may still be running (only newest data plotted)
