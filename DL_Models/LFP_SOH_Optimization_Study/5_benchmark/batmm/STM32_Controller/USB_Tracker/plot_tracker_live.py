import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import re
import os

def normalize_column(col):
    """Remove units in parentheses and normalize column name."""
    col = re.sub(r"\s*\(.*?\)", "", col)
    col = col.strip().replace(" ", "").lower()
    return col


if len(sys.argv) < 2:
    print("Usage: python plot_tracker_live.py <data.csv>")
    sys.exit(1)

csv_file = sys.argv[1]

fig, axs = plt.subplots(6, 1, figsize=(12, 18), sharex=True)
fig.suptitle("CT-3 Live Measurement Data (last 2 minutes)")

plot_map = [
    ("voltage",    "Voltage (V)",   "blue"),
    ("current",    "Current (A)",   "red"),
    ("power",      "Power (W)",     "green"),
    ("voltagedp",  "USB D+ (V)",    "orange"),
    ("voltagedm",  "USB D- (V)",    "purple"),
    ("energy",     "Energy (Wh)",   "brown"),
]

lines = []
for ax, (_, ylabel, color) in zip(axs, plot_map):
    line, = ax.plot([], [], color=color, linewidth=1.2)
    lines.append(line)
    ax.set_ylabel(ylabel)
    ax.grid(True)

axs[-1].set_xlabel("Time (s)")
plt.tight_layout(rect=[0, 0, 1, 0.97])

WINDOW_SECONDS = 120.0  # rolling window size

def update(frame):
    if not os.path.exists(csv_file):
        return lines

    try:
        df = pd.read_csv(csv_file)
    except Exception:
        return lines  # skip if file mid-write

    df.columns = [normalize_column(c) for c in df.columns]

    if "time" not in df.columns:
        return lines

    # Convert timestamp (ms) to relative seconds
    t0 = df["time"].iloc[0]
    df["time_s"] = (df["time"] - t0) / 1000.0

    # Keep only last 2 minutes of data
    if not df.empty:
        t_max = df["time_s"].iloc[-1]
        df = df[df["time_s"] >= t_max - WINDOW_SECONDS]

    for idx, (col, _, _) in enumerate(plot_map):
        if col in df.columns:
            lines[idx].set_data(df["time_s"], df[col])
            axs[idx].relim()
            axs[idx].autoscale_view()

    # Fix x-axis window to last 2 min range
    for ax in axs:
        if not df.empty:
            ax.set_xlim(max(0, df["time_s"].iloc[-1] - WINDOW_SECONDS),
                        df["time_s"].iloc[-1])

    return lines

ani = animation.FuncAnimation(fig, update, interval=500, cache_frame_data=False)
plt.show()
