import csv
import os
import time
import numpy as np
import joblib
import time
from pathlib import Path
from stm_ai_runner import AiRunner

from src.tracker import AVHzY_CT3
from src.utils import data_utils
from config import SERIAL_PORT, METER_PORT, BAUD_RATE, SCALER_PATH, RESULTS_DIR, DATA_DIR, TRACK_EVERY


def quantize(x, scale, zp):
    return np.clip(np.round(x / scale + zp), -128, 127).astype(np.int8)


def dequantize(x, scale, zp):
    return (x.astype(np.float32) - zp) * scale


def infer_rnn(model_path: str, quantized: bool = True, chunk_size = 192):

    # Files and paths
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    model_filename = os.path.basename(model_path)
    tracker_csv_path = Path(f"{RESULTS_DIR}/tracker/{model_filename}_{timestamp}.csv")
    tracker_csv_path.parent.mkdir(parents=True, exist_ok=True)
    tracker_file = open(tracker_csv_path, "w")

    # Setup USB tracker
    tracker = AVHzY_CT3(
        device=METER_PORT, action="read", repeat=-1,
        time_ms=100, reads="all", output=tracker_file
    )

    try:
        # Connect to MCU
        desc   = f"serial:{SERIAL_PORT}:{BAUD_RATE}"
        runner = AiRunner()
        try:
            runner.connect(desc)
            print("%%%%% Connected successfully\n")
        except Exception as e:
            print(f"##### Failed to connect to STM32: {e}")
            return

        info = runner.get_info()
        print(info)

        # Zero point and scale transformations for quantized models
        if quantized:
            print("%%%%% Running in INT8 (quantized)")
            in_scales  = [float(inp["scale"]) for inp in info["inputs"]]
            in_zps     = [int(inp["zero_point"]) for inp in info["inputs"]]
            out_scales = [float(out["scale"]) for out in info["outputs"]]
            out_zps    = [int(out["zero_point"]) for out in info["outputs"]]
        else:
            print("%%%%% Running in FP32 (non-quantized)")

        # Need multiple scales and zero points for feature values and hidden/cell states
        in_idx_x       = [i for i, inp in enumerate(info["inputs"]) if inp["shape"][-1] == 20][0]   # 20 features
        in_idx_states  = [i for i, inp in enumerate(info["inputs"]) if inp["shape"][-1] == chunk_size]
        out_idx_soh    = [i for i, out in enumerate(info["outputs"]) if out["shape"][-1] == 1][0]
        out_idx_states = [i for i, out in enumerate(info["outputs"]) if out["shape"][-1] == chunk_size]

        # Manually map hidden/cell states output by MCU at step t and feed them back into the correct hidden/cell state inputs at step t+1
        STATE_ROUTING = {}
        # A list of all input indices that were identified as state variables (matching chunk_size)
        available_inputs = list(in_idx_states)
        for out_idx in out_idx_states:
            if quantized:
                out_scale = out_scales[out_idx]
                # Find the input tensor whose scale is closest to our output tensor's scale (direct mapping might not work due to rounding)
                best_in_idx = min(available_inputs, key=lambda idx: abs(in_scales[idx] - out_scale))
            else:
                # Fallback for float32: match by order of appearance
                best_in_idx = available_inputs[0]
            # Save the discovered link to our routing table
            STATE_ROUTING[out_idx] = best_in_idx
            # Remove assigned input from the pool so it can't be assigned to another output
            available_inputs.remove(best_in_idx)

        # Load scaler and dataloader
        print("%%%%% Loading scaler...")
        scaler = joblib.load(SCALER_PATH)

        print("%%%%% Building test dataloaders...")
        _, _, test_loaders = data_utils.build_dataloaders(
            data_root=DATA_DIR,
            scaler=scaler,
            chunk=1,
            batch_size=1,
            num_workers=0,
            load_train=False,
            load_val=False,
            load_test=True
        )

        all_results = []
        inference_time = 0

        try:
            print(f"%%%%% Starting power tracker... Logging to {tracker_csv_path}")
            tracker.start()
            start_time = time.perf_counter()
            # Loop through each evaluation cell in the test loaders
            for cell_idx, loader in enumerate(test_loaders):
                cell_name = data_utils.TEST_CELLS[cell_idx]
                print(f"\n%%%%% {'═'*72}")
                print(f"%%%%%   Cell: {cell_name}  ({len(loader)} steps)")
                print(f"%%%%% {'═'*72}")

                # Reset hidden states to zeros for each new battery cell sequence
                current_states = {in_idx: np.zeros((1, chunk_size), dtype=np.float32) for in_idx in in_idx_states}

                for t, (X, y) in enumerate(loader):
                    # Extract the single timestep
                    x_step = X.detach().cpu().numpy().astype(np.float32)
                    y_real  = y.detach().cpu().numpy().astype(np.float32).flatten()[0]

                    # Construct the raw float inputs buffer using dynamic positions
                    raw_inputs = [None] * len(info["inputs"])
                    raw_inputs[in_idx_x] = x_step
                    for in_idx, state_val in current_states.items():
                        raw_inputs[in_idx] = state_val

                    if quantized:
                        # Quantize each input array natively using its specific scale/zp
                        feed = [
                            quantize(raw_inputs[i], in_scales[i], in_zps[i])
                            for i in range(len(info["inputs"]))
                        ]
                        # Run inference on MCU
                        outputs, _ = runner.invoke(feed)
                        # Dequantize all outputs natively
                        pred_f32 = [
                            dequantize(outputs[i], out_scales[i], out_zps[i])
                            for i in range(len(info["outputs"]))
                        ]
                    else:
                        outputs, _ = runner.invoke(raw_inputs)
                        pred_f32 = [out.astype(np.float32) for out in outputs]

                    # State Bridge: Route dequantized outputs to their paired next-step inputs
                    for out_idx, in_idx in STATE_ROUTING.items():
                        current_states[in_idx] = pred_f32[out_idx]

                    # Only track inference if the buffer is "full" of real data
                    if t >= chunk_size - 1:

                        # Extract the SOH prediction
                        pred_soh = pred_f32[out_idx_soh].flatten()[0]

                        diff = pred_soh - y_real

                        all_results.append({
                            "cell":     cell_name,
                            "step":     t,
                            "y_real":   y_real,
                            "pred_mcu": pred_soh,
                            "diff":     diff,
                        })

                        if (t + 1) % 10 == 0:
                            print(f"%%%%%   Step {t+1:04d} | Real SOH: {y_real:.4f} | Predicted: {pred_soh:.4f} | Diff: {diff:+.4f}")

        except Exception as e:
            print(f"%%%%% Inference failed: {e}")

        finally:
            runner.disconnect()
            print("\n%%%%% Disconnected from STM32")
            inference_time = time.perf_counter() - start_time
            print(f"%%%%% Stopped power tracker. Total execution time: {inference_time:.4f}s")
    finally:
        # Stop Tracker safely regardless of errors
        tracker.stop()
        tracker_file.close()

    if not all_results:
        return

    # Print a summary
    diffs    = np.array([r["diff"] for r in all_results])
    mae      = float(np.mean(np.abs(diffs)))
    max_err  = float(np.max(np.abs(diffs)))
    bias     = float(np.mean(diffs))
    rmse   = float(np.sqrt((diffs ** 2).mean()))
    ss_res = float((diffs ** 2).sum())
    ss_tot = float(((y_real - y_real.mean()) ** 2).sum())
    r2     = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    model_size = os.path.getsize(model_path) if os.path.exists(model_path) else 0

    print(f"\n%%%%% {'═'*72}")
    print("%%%%%   SUMMARY")
    print(f"%%%%% {'═'*72}")
    print(f"%%%%%   Model               : {model_filename}")
    print(f"%%%%%   Quantized           : {quantized}")
    print(f"%%%%%   Mean Error          : {mae:.4f}")
    print(f"%%%%%   Max Error           : {max_err:.4f}")
    print(f"%%%%%   Root Mean Sq Error  : {rmse:.4f}")
    print(f"%%%%%   R2                  : {r2:.4f}")
    print(f"%%%%%   Mean Error (Bias)   : {bias:+.4f}")
    print(f"%%%%%   Chunks Evaluated    : {len(all_results)}")
    print(f"%%%%%   Inference Time      : {inference_time:.4f} s")
    print(f"%%%%%   Model Size (Bytes)  : {model_size:.4f}")

    # Append to CSV
    result_csv = Path(f"{RESULTS_DIR}/inference_results.csv")
    result_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = os.path.isfile(result_csv)

    with open(result_csv, "a", newline="") as csvfile:
        fieldnames = ["timestamp", "filename", "quantized", "mae", "max_abs_error", "rmse", "r2", "bias", "chunks", "inference_time", "model_size_bytes"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow({
            "timestamp":        timestamp,
            "filename":         model_filename,
            "quantized":        quantized,
            "mae":              round(mae, 6),
            "max_abs_error":    round(max_err, 6),
            "rmse":             round(rmse, 6),
            "r2":               round(r2, 6),
            "bias":             round(bias, 6),
            "chunks":           len(all_results),
            "inference_time":   round(inference_time, 4),
            "model_size_bytes": model_size
        })
    print(f"%%%%% Summary appended to {result_csv}")
