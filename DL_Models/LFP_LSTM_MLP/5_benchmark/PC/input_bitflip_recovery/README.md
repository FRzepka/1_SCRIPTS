# Minimal transient input-bitflip recovery test

This folder contains a deliberately small software fault-injection test for
the SOC and SOH Base, Pruned, and Quantized C models.

## Question

Does structural pruning or weight quantization materially change the
short-term response of the recurrent estimator to a transient corruption of
its embedded input buffer?

## Protocol

- C07 feature data are replayed at the original 1 Hz sampling interval.
- Each model first processes at least 2,048 clean samples.
- At 30 distributed positions, one FP32 bit is flipped in voltage, current,
  or temperature for exactly one sample.
- Bit 22 is used by default. It is the most significant mantissa bit and
  therefore creates a finite but clearly visible numerical disturbance.
- The model then receives 60 clean samples without resetting its recurrent
  state.
- The disturbed trajectory is compared with a clean trajectory that starts
  from the identical copied LSTM state.

The reported transient-response metrics are the peak disturbed-clean
deviation, residual deviation after 60 seconds, and time to sustained
recovery. Accuracy against the target is reported separately for the clean
and disturbed 61-sample windows. Their difference,
`delta_window_mae_pp = fault_window_mae_pp - clean_window_mae_pp`, isolates
the change in target MAE caused by the injected event. All errors are stored
in percentage points, so a normalized MAE of `0.012` is reported as
`1.2 pp`.

This is an input-buffer single-event-upset test. It is not a complete sensor,
communication, weight-memory, or hardware fault campaign.

## Run

```powershell
./run_test.cmd
```

The console reports percentage, elapsed time, and estimated remaining time.
Compiled DLLs are cached in `build/`. Each run writes CSV, JSON, and an SVG
trace to a timestamped folder below `results/`.

Generate the review figure from the latest completed run:

```powershell
powershell -ExecutionPolicy Bypass -File ./generate_review_figure.ps1
```

This also replaces `rev_5_limited_fault_sensitivity.png` in the review
manuscript's `figures/Review_1_Additional` folder.

For a very small smoke test:

```powershell
./run_test.cmd --rows 3000 --warmup 512 --trials 6 --recovery 60
```
