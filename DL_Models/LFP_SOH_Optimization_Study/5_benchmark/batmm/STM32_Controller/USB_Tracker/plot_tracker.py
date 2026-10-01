import sys
import pandas as pd
import matplotlib.pyplot as plt
import re

def normalize_column(col):
    col = re.sub(r"\s*\(.*?\)", "", col)  # remove "(V)" etc.
    col = col.strip().replace(" ", "").lower()
    return col

def main():
    if len(sys.argv) < 2:
        print("Usage: python plot_tracker.py <data.csv>")
        sys.exit(1)

    csv_file = sys.argv[1]
    df = pd.read_csv(csv_file)
    df.columns = [normalize_column(c) for c in df.columns]

    print("Detected columns:", df.columns.tolist())

    if 'time' not in df.columns:
        print("Error: Could not find a 'time' column.")
        sys.exit(1)

    # Convert ms timestamp to elapsed seconds
    t0 = df['time'].iloc[0]
    df['time_s'] = (df['time'] - t0) / 1000.0

    # Plot configuration
    fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
    fig.suptitle("CT-3 Measurement Data")

    plot_map = [
        ("voltage", "Voltage (V)", "blue"),
        ("current", "Current (A)", "red"),
        ("power", "Power (W)", "green"),
        ("voltagedp", "USB D+ (V)", "orange"),
        ("voltagedm", "USB D- (V)", "purple"),
        ("energy", "Energy (Wh)", "brown")
    ]

    for ax, (col, ylabel, color) in zip(axs, plot_map):
        if col in df.columns:
            ax.plot(df['time_s'], df[col], color=color, linewidth=1.2)
        ax.set_ylabel(ylabel)
        ax.grid(True)

    axs[-1].set_xlabel("Time (s)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()

if __name__ == "__main__":
    main()
