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


def infer_conv(model_path: str, quantized: bool = True, chunk_size = 128):

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
            in_scale  = float(info["inputs"][0]["scale"])
            in_zp  = int(info["inputs"][0]["zero_point"])
            out_scale = float(info["outputs"][0]["scale"])
            out_zp = int(info["outputs"][0]["zero_point"])
            print(f"%%%%% MCU input scale: {in_scale:.6f}, zero_point: {in_zp}")
            print(f"%%%%% MCU output scale: {out_scale:.6f}, zero_point: {out_zp}")
        else:
            print("%%%%% Running in FP32 (non-quantized)")

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
            load_test=True,
        )

        all_results = []
        inference_time = 0

        try:
            print(f"%%%%% Starting tracker... Logging to {tracker_csv_path}")
            tracker.start()
            start_time = time.perf_counter()
            for cell_idx, loader in enumerate(test_loaders):
                cell_name = data_utils.TEST_CELLS[cell_idx]
                print(f"\n%%%%% {'═'*72}")
                print(f"%%%%%   Cell: {cell_name}  ({len(loader)} chunks)")
                print(f"%%%%% {'═'*72}")

                history_buffer = np.zeros((chunk_size, 20), dtype=np.float32)   # keep track of #chunk_size datapoints

                for t, (X, y) in enumerate(loader):
                    x_step = X.detach().cpu().numpy().astype(np.float32).flatten() # (20,)
                    y_real  = y.detach().cpu().numpy().astype(np.float32).flatten()[0]

                    # Shift the buffer ALWAYS (pop oldest, push newest) so history stays accurate
                    history_buffer[:-1] = history_buffer[1:]
                    history_buffer[-1] = x_step

                    # Only run inference if the buffer is full AND it aligns with TRACK_EVERY
                    if t >= chunk_size - 1 and (t - (chunk_size - 1)) % TRACK_EVERY == 0:

                        # Reshape buffer for the MCU: (1, chunk_size, 20)
                        x_3d = history_buffer[np.newaxis, ...].astype(np.float32)

                        if quantized:
                            feed       = quantize(x_3d, in_scale, in_zp)
                            # Run inference on MCU
                            outputs, _ = runner.invoke([feed])
                            pred_f32   = dequantize(outputs[0].flatten(), out_scale, out_zp)
                        else:
                            outputs, _ = runner.invoke([x_3d])
                            pred_f32   = outputs[0].flatten().astype(np.float32)

                        # The CNN outputs a prediction for the whole sequence, we just want the current (last) SOH
                        pred_last = pred_f32[-1]
                        diff      = pred_last - y_real

                        all_results.append({
                            "cell":     cell_name,
                            "step":     t,
                            "y_real":   y_real,
                            "pred_mcu": pred_last,
                            "diff":     diff,
                        })

                        # Print progress based on the number of actually evaluated chunks
                        tracked_count = len(all_results)
                        if tracked_count % 10 == 0:
                            print(f"%%%%%   Step {t+1:04d} (Tracked {tracked_count}) | Real SOH: {y_real:.4f} | Predicted: {pred_last:.4f} | Diff: {diff:+.4f}")

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
    rmse     = float(np.sqrt((diffs ** 2).mean()))

    # R2 logic using only the tracked datapoints
    y_real_tracked = np.array([r["y_real"] for r in all_results])
    ss_res = float((diffs ** 2).sum())
    ss_tot = float(((y_real_tracked - y_real_tracked.mean()) ** 2).sum())
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
