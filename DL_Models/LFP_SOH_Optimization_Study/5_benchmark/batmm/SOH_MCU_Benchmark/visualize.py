import os
import re
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ----- Config -----
INFERENCE_CSV = "inference_results.csv"
TRACKER_DIR = "tracker"

COLORS = {
    'cnn': '#5FCCC5',
    'tcn': '#253F57',
    'lstm': '#E1648C',
    'gru': '#4BFF21',
}


def extract_architecture(filename):
    """Infers architecture from the start of the filename."""
    fname_lower = str(filename).lower()
    for arch in ['cnn', 'tcn', 'lstm', 'gru']:
        if fname_lower.startswith(arch):
            return arch
    return 'unknown'


def extract_pruning_degree(filename):
    """Extracts the decimal pruning ratio from the filename. Defaults to 1.0."""
    match = re.search(r'(\d+\.\d+)', str(filename))
    if match:
        return float(match.group(1))
    return 1.0


def find_timestamped_tracker_file(tracker_dir, model_filename):
    """
    Searches the tracker directory for a file matching the new naming scheme:
    <model_filename>_<YYYYMMDD>_<HHMMSS>.csv
    """
    if not os.path.exists(tracker_dir):
        return None

    # Escape special characters in the filename (dots)
    escaped_filename = re.escape(model_filename)
    # Match pattern: model_name followed by underscore, 8 digits (date), underscore, 6 digits (time), .csv
    pattern = re.compile(rf"^{escaped_filename}_\d{{8}}_\d{{6}}\.csv$")

    for file in os.listdir(tracker_dir):
        if pattern.match(file):
            return os.path.join(tracker_dir, file)

    return None


def main():

    print(f"%%%%% Loading dataset from {INFERENCE_CSV}...")
    df = pd.read_csv(INFERENCE_CSV)

    # Parse filenames for Architecture and Pruning Degree
    df['architecture'] = df['filename'].apply(extract_architecture)
    df['pruning_degree'] = df['filename'].apply(extract_pruning_degree)

    print(f"%%%%% Processing tracker data from '{TRACKER_DIR}'...")
    avg_v, avg_c, avg_p = [], [], []

    for filename in df['filename']:
        tracker_path = find_timestamped_tracker_file(TRACKER_DIR, filename)

        if tracker_path and os.path.exists(tracker_path):
            try:
                # Read file lines manually to skip mangled rows
                valid_rows = []
                with open(tracker_path, 'r', errors='ignore') as f:
                    header = f.readline().strip().split(',')

                    for line in f:
                        parts = line.strip().split(',')
                        # Ensure row has the exact number of columns expected
                        if len(parts) != len(header):
                            continue  # Skip smashed-together string rows

                        try:
                            # Parse core values dynamically
                            v_val = float(parts[header.index('voltage (V)')])
                            c_val = float(parts[header.index('current (A)')])
                            p_val = float(parts[header.index('power (W)')])

                            # Exclude glitched values (usually first measurement point of tracker)
                            if 0.0 <= v_val <= 10.0 and -5.0 <= c_val <= 5.0 and -5.0 <= p_val <= 25.0:
                                valid_rows.append({
                                    'voltage (V)': v_val,
                                    'current (A)': c_val,
                                    'power (W)': p_val
                                })
                        except (ValueError, IndexError):
                            continue # Skip row if a value isn't a parseable number

                if valid_rows:
                    # Construct a clean dataframe out of filtered rows only
                    tdf = pd.DataFrame(valid_rows)
                    v = tdf['voltage (V)'].mean()
                    c = tdf['current (A)'].mean()
                    p = tdf['power (W)'].mean()
                else:
                    print(f"#####   File {tracker_path} contained only corrupted rows.")
                    v, c, p = pd.NA, pd.NA, pd.NA

                avg_v.append(v)
                avg_c.append(c)
                avg_p.append(p)

            except Exception as e:
                print(f"  Error reading {tracker_path}: {e}")
                avg_v.append(pd.NA)
                avg_c.append(pd.NA)
                avg_p.append(pd.NA)
        else:
            avg_v.append(pd.NA)
            avg_c.append(pd.NA)
            avg_p.append(pd.NA)

    df['voltage'] = avg_v
    df['current'] = avg_c
    df['power'] = avg_p

    # Convert model dimensions to plot-friendly scaling values
    df['model_size_mb'] = df['model_size_bytes'] / (1024 * 1024)
    df['inference_time'] = df['inference_time'] / (0.001 * df["chunks"])

    architectures = ['cnn', 'tcn', 'lstm', 'gru']

    metrics = [
        ('mae', 'Mean Absolute Error (MAE)'),
        ('max_abs_error', 'Max Absolute Error'),
        ('bias', 'Mean Error (Bias)'),
        ('inference_time', 'Inference Time per Datapoint (ms)'),
        ('model_size_mb', 'Model Size (MB)'),
        ('voltage', 'Average Voltage (V)'),
        ('current', 'Average Current (A)'),
        ('power', 'Average Power (W)')
    ]

    for arch in architectures:
        arch_df = df[df['architecture'] == arch].copy()
        if arch_df.empty:
            print(f"##### Skipping {arch.upper()} - No data found.")
            continue

        print(f"%%%%% Generating plots for {arch.upper()}...")

        fig, axes = plt.subplots(2, 4, figsize=(22, 10))
        fig.suptitle(f'Performance & Power Metrics: {arch.upper()}', fontsize=16, fontweight='bold')
        axes = axes.flatten()

        for i, (col_name, y_label) in enumerate(metrics):
            ax = axes[i]

            for is_quant in [False, True]:
                # Drop rows ONLY if the specific metric column being plotted is missing
                subset = arch_df[arch_df['quantized'] == is_quant].sort_values(by='pruning_degree')
                subset = subset.dropna(subset=['pruning_degree', col_name])

                if subset.empty:
                    continue

                color = COLORS[arch]
                alpha = 0.55 if is_quant else 1.0  # Slightly less opaque for quantized lines
                label = f"{arch.upper()} (INT8)" if is_quant else f"{arch.upper()} (FP32)"

                ax.plot(
                    subset['pruning_degree'],
                    subset[col_name],
                    marker='o',
                    markersize=10,
                    color=color,
                    alpha=alpha,
                    linewidth=4,
                    label=label
                )

            ax.set_xlabel('Pruning Degree (Parameters Remaining)')
            ax.set_ylabel(y_label)
            ax.set_title(y_label)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()

            # Invert X axis so 1.0 (Full Model) is on the left, 0.0 is on the right
            ax.set_xlim(1.05, -0.05)

            # Clip Y-axis
            if col_name == 'model_size_mb':
                ax.set_ylim(0, 2)
            if col_name == 'mae':
                ax.set_ylim(0, 0.1)
            if col_name == 'max_abs_error':
                ax.set_ylim(0, 0.25)
            if col_name == 'voltage':
                ax.set_ylim(5.0, 5.15)
            if col_name == 'current':
                ax.set_ylim(0.25, 0.3)
            if col_name == 'power':
                ax.set_ylim(1.25, 1.5)
            if col_name == 'inference_time':
                ax.set_ylim(0, 2000)
            if col_name == 'max_abs_error':
                ax.set_ylim(0, 1)
            if col_name == 'bias':
                ax.set_ylim(-0.5, 0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        output_file = f"%%%%% {arch}_metrics_dashboard.png"
        plt.savefig(output_file, dpi=300)
        plt.close()
        print(f"%%%%%  -> Saved {output_file}")

    print("%%%%% All plots generated successfully!")

if __name__ == "__main__":
    main()
