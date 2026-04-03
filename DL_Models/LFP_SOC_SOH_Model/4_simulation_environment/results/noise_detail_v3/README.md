# Current-noise detail

- `Δ abs error = abs_err(noise_run) - abs_err(baseline_run)`
- Local spike metrics are based on `|Δ abs error|`.
- Drift check uses a 15-minute rolling mean of `Δ abs error` on 60 s downsampled traces.
- If the full-run rolling mean stays near zero, there is no evidence for a systematic noise-driven drift beyond baseline.