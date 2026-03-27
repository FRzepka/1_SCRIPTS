# SOH Pruned Model Step-by-Step Validation

This directory contains scripts to validate the pruned SOH LSTM model against the baseline using a step-by-step inference approach with online filtering.

## Scripts

*   `run_step_compare.py`: Runs the inference on the full dataset (or a limit) using the pruned model and baseline. It applies an online filter (EMA + Caps) to smooth the output.
*   `plot_results.py`: Generates plots from the `arrays.npz` file produced by the inference script.

## Usage

### 1. Run Inference

Run the comparison script. This will generate a timestamped output folder (e.g., `STEP_COMPARE_YYYYMMDD_HHMMSS`).

**Note:** The `--device cpu` flag is often sufficient and stable for this batched inference, but you can try `--device cuda` for potentially faster execution.

```bash
# Example command with recommended filter settings
PYTHONUNBUFFERED=1 stdbuf -oL -eL python -u DL_Models/LFP_LSTM_MLP/6_test/Python/pruned_SOH/run_step_compare.py \
  --device cpu \
  --pruned-ckpt DL_Models/LFP_LSTM_MLP/2_models/pruned/soh_2.1.0.0/prune_30pct_20251122_010142/soh_pruned_hidden90.pt \
  --data-root /home/florianr/MG_Farm/0_Data/MGFarm_18650_FE \
  --cell MGFarm_18650_C07 \
  --limit 0 \
  --strict-filter \
  --post-max-rel 1e-4 \
  --post-max-abs 1e-5 \
  --post-ema-alpha 0.02 \
  --calib-start-one \
  --calib-kind scale \
  --no-plots
```

**Parameters:**
*   `--limit 0`: Process the entire dataset (set to e.g. 50000 for a quick test).
*   `--post-max-rel 1e-4`: Maximum relative change allowed per step (0.01%).
*   `--post-max-abs 1e-5`: Maximum absolute change allowed per step.
*   `--post-ema-alpha 0.02`: Exponential Moving Average factor (smoothing).
*   `--calib-start-one`: Calibrate the first prediction to match the ground truth (or 1.0).
*   `--no-plots`: Skip plotting during the run (faster, use `plot_results.py` afterwards).

### 2. Plot Results

After the run completes, use `plot_results.py` to visualize the data. You need to provide the path to the specific cell folder inside the generated `STEP_COMPARE_...` directory.

```bash
# Syntax
python DL_Models/LFP_LSTM_MLP/6_test/Python/pruned_SOH/plot_results.py <PATH_TO_OUTPUT_DIR>

# Example
python DL_Models/LFP_LSTM_MLP/6_test/Python/pruned_SOH/plot_results.py DL_Models/LFP_LSTM_MLP/6_test/Python/pruned_SOH/STEP_COMPARE_20251124_091954/MGFarm_18650_C07
```

This will generate:
*   `overlay_full.png`: Full sequence comparison.
*   `overlay_first2000.png`: Zoom into the first 2000 steps.
*   `diff_hist.png`: Histogram of differences between baseline and pruned models.
