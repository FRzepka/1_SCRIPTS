# JES 2.0 manuscript handover

Date: 2026-08-30

## Protected original

The submitted manuscript PDF remains unchanged:

- `../Robustness_Benchmark_Manuscript.pdf`
- SHA-256: `d75d3c6039ad58bab336a98dde095d7eb0d1325498b53e1ce9df9f877272df42`

All revision work is confined to this `JES_2.0` directory.

## Revision artifacts

- Editable source: `Robustness_Benchmark_Manuscript_JES2_Updated.tex`
- Compiled manuscript: `Robustness_Benchmark_Manuscript_JES2_Updated.pdf`
- Source SHA-256: `94a405ef0cd64f12fc9a7ecb83cf95d2834b1eb4b698641e43863842ccd2465f`
- PDF SHA-256: `b8c893a75ce4566b842f2bc16b63cf2ed5361984217326a89df04b523269d1f2`
- Compiled length: 47 pages

## Build

Run from this directory:

```bash
/home/florianr/anaconda3/envs/ml/bin/tectonic \
  --keep-logs --keep-intermediates \
  Robustness_Benchmark_Manuscript_JES2_Updated.tex
```

The build completes without unresolved references or citations. Tectonic emits
non-blocking typographic box warnings inherited from the journal layout and dense
tables.

## Main revision scope

- Replaced the original illustrative results with the finalized six-cell JES2 figures.
- Reframed the title, abstract, introduction, methods, discussion, limitations, and conclusion around the central deployment question: why nominal accuracy alone is insufficient and how robustness, recovery, and microcontroller cost change estimator selection.
- Defined DM, HDM, HECM, and DD as four concrete representatives rather than universal proxies for complete estimator families, including their minimum class-defining mechanisms, optional broader extensions, and exact implementations used here.
- Kept run counts, seed schedules, window selection, and bootstrap details out of the abstract and in the reproducibility-focused methods and appendix.
- Documented 16 predeclared 24-hour windows, 19 cases, six holdout cells, 20 repeated seeds where applicable, 6,720 canonical runs, and 10,000 hierarchical bootstrap repetitions.
- Treated cells as the independent units and seeds/windows as nested observations.
- Added current-bias lifecycle/reset, initial-state recovery, missing-sample, jitter, burst-dropout, voltage-spike, ADC-quantization, and cross-scenario interpretations.
- Separated the robustness benchmark from the embedded performance benchmark and added isolated STM32H753ZI SOC-core results for numerical equivalence, latency, compiled flash occupancy, measured peak runtime RAM, and DD inference-mode tradeoffs.
- Moved coverage, test-matrix, evaluation-window, and detailed DD-latency material into Appendix A.
- Added a compact reviewer audit in `REVIEWER_COVERAGE_JES2.md`; the older `REVIEWER_TODO_STATUS.txt` is retained only as a historical implementation log.

## Interpretation boundaries retained in the manuscript

- The conclusions apply to the tested LFP cells, operating envelope, estimator implementations, and measurement-only disturbance protocol.
- The high-load class contains only C29 and is not interpreted as a population-level high-load estimate.
- Hardware measurements cover the SOC inference cores; the shared SOH LSTM and complete BMS scheduling are outside the on-device timing boundary.
- Peak runtime RAM combines statically allocated variable storage with on-device measurements of maximum call-stack and dynamic-memory use. The shared SOH-LSTM, concurrent BMS tasks, and board-level energy remain outside the measured hardware boundary.
- The causal LSTM-SOH trace remains the primary common input; the reference-SOH substitution is reported only as an ablation.

## Validation performed

- Visually inspected the rendered manuscript, including the main results and appendices.
- Confirmed all 24 finalized PNG figures are referenced exactly once.
- Confirmed that no missing-figure placeholders remain.
- Confirmed that all active labels and references resolve.
- Rechecked the protected original PDF hash after compilation.
