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
- Source SHA-256: `c8c250343ad17e2404811d20c01ec7d7fed2c216aec035f503433de07e082df1`
- PDF SHA-256: `72da9f2fd1d69d566b883182b45e9e07ea81c3fa102d1b4daedf900f8abad39c`
- Compiled length: 50 pages

## Build

Run from this directory:

```bash
/home/florianr/.local/bin/tectonic \
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
- Documented 16 predeclared 24-hour windows, 19 cases, six holdout cells, 10 sensor-noise seeds, 5 seeds for the other stochastic cases, 6,720 canonical runs, and 10,000 hierarchical bootstrap repetitions.
- Treated cells as the independent units and seeds/windows as nested observations.
- Recomputed every global result on the common DD-valid source-sample interval beginning at sample 2023. Each model now contributes 84,377 matched samples per 24-hour window.
- Replaced the inconsistent recovery outputs with one canonical paired trajectory endpoint, including recovery-or-censoring time, excess-error area, censoring, and later relapse.
- Included all eight declared disturbance families in the illustrative robustness score and added two alternative weighting analyses.
- Added current-gain lifecycle/reset, initial-state recovery, missing-sample, jitter, burst-dropout, voltage-spike, ADC-quantization, sensor-offset, and cross-scenario interpretations.
- Recomputed burst-dropout penalties against an undisturbed 48-hour reference and documented the maximum-absolute-net-charge placement rule.
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

- Recompiled the 50-page manuscript and inspected the revised core result figures, hardware figures, and Appendix protocol figure.
- Confirmed all 24 finalized PNG figures are referenced exactly once.
- Confirmed that no missing-figure placeholders remain.
- Confirmed that all active labels and references resolve.
- Ran the complete simulation-environment test suite: 32 tests passed.
- Confirmed 6,832 merged execution records with no failures, comprising 6,720 public benchmark runs and 112 internal duration-matched dropout-reference runs.
- Rechecked the protected original PDF hash after compilation.
