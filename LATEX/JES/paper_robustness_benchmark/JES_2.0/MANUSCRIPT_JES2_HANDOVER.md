# JES 2.0 manuscript handover

Date: 2026-08-31

## Protected original

The submitted manuscript PDF remains unchanged:

- `../Robustness_Benchmark_Manuscript.pdf`
- SHA-256: `d75d3c6039ad58bab336a98dde095d7eb0d1325498b53e1ce9df9f877272df42`

The submitted PDF remains protected. Revised manuscript artifacts are confined
to `JES_2.0`, while reproducibility scripts and finalized figures remain in their
existing project directories.

## Revision artifacts

- Editable source: `Robustness_Benchmark_Manuscript_JES2_Updated.tex`
- Compiled manuscript: `Robustness_Benchmark_Manuscript_JES2_Updated.pdf`
- Source SHA-256: `76363800e96230e76b91618b9e25eee90066a87f64431e72f98945ca9540bfd3`
- PDF SHA-256: `a5929f17796df90ecba7044e74e9957cddd52c9af6bc8d98fef8462622b551e7`
- Compiled length: 57 pages

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
- Structured the abstract, introduction, methodology, results, discussion, and conclusion around two complementary branches. The robustness benchmark reports accuracy, robustness, and recovery. The embedded performance benchmark reports numerical equivalence, inference time, flash occupancy, and peak runtime RAM.
- Retained the four nominal-accuracy MAE values in the abstract and added the matched current-offset result as the principal systematic sensor-error example. Recovery is summarized through the strongest observed representative.
- Reframed the title, abstract, introduction, methods, discussion, limitations, and conclusion around the central deployment question: why nominal accuracy alone is insufficient and how robustness, recovery, and microcontroller cost change estimator selection.
- Defined DM, HDM, HECM, and DD as four concrete representatives rather than universal proxies for complete estimator families, including their minimum class-defining mechanisms, optional broader extensions, and exact implementations used here.
- Kept run counts, seed schedules, window selection, and bootstrap details out of the abstract and in the reproducibility-focused methods and appendix.
- Documented 16 protocol-defined 24-hour windows, 20 final scenario definitions, six holdout cells, 10 sensor-noise seeds, 5 seeds for the other stochastic cases, 7,040 public model-window evaluations, and 10,000 hierarchical bootstrap repetitions.
- Retained the six-cell holdout summary and added the Dissertation SOH-aging overview as Figure 26. Its accompanying 15-cell table identifies every training, validation, and holdout cell and reports measured duration, SOH, temperature, and P95 absolute C-rate coverage.
- Defined each of the three current-gain magnitudes through matched positive and negative sign sublevels in the final benchmark build. The continuous C29 lifecycle replay remains a complementary single-cell mechanism analysis.
- Treated cells as the independent units and seeds/windows as nested observations.
- Recomputed every global result on the common DD-valid source-sample interval beginning at sample 2023. Each model now contributes 84,377 matched samples per 24-hour window.
- Rebuilt the initialization figure exclusively from the corrected dedicated paired campaign. The primary endpoint is persistent recovery through the remaining 24-hour horizon. First 300-second entry and later relapse are retained as separate diagnostics. Observation begins about 0.562 hours after intervention, and boundary endpoints are flagged as left-censored upper bounds.
- Removed relapse from the composite recovery score because it is undefined when a run never enters the recovery band and would otherwise reward non-recovery.
- Audited all 26 finalized PNG files against their manuscript captions and dependencies. Only Figures 09 and 15 depend on the corrected recovery analysis. Figure 14 deliberately excludes initialization mismatch because it is a measurement-disturbance heatmap.
- Included all eight evaluated disturbance families in the illustrative robustness score and added two alternative weighting analyses. The current-gain and additive current-offset components use adverse directions from their matched signed sweeps.
- Added current-gain lifecycle/reset, initial-state recovery, missing-sample, jitter, burst-dropout, voltage-spike, ADC-quantization, sensor-offset, and cross-scenario interpretations.
- Added a separate 9,184-run HECM lookup-table sensitivity over all 16 windows. The analysis crosses local resistance, time-constant, and OCV perturbations with all 22 measurement and signal-integrity subcases plus paired initialization recovery. The publication output remains compact through one interaction figure and a seven-row table. The largest absolute lookup--disturbance interaction is 0.006116 MAE and is small relative to the 0.1763 HECM current-offset penalty, while absolute accuracy and one cell-specific recovery case remain calibration dependent. The analysis remains outside the cross-model score.
- Replaced the legacy C07 current-noise illustration with a reset-free, sample-matched C29 mechanism example and retained the final six-cell hierarchical confidence intervals for the population-level noise result.
- Recomputed burst-dropout penalties against an undisturbed 48-hour reference and documented the maximum-absolute-net-charge placement rule.
- Separated the robustness benchmark from the embedded performance benchmark and added isolated STM32H753ZI SOC-core results for numerical equivalence, latency, compiled flash occupancy, measured peak runtime RAM, and DD inference-mode tradeoffs.
- Moved coverage, test-matrix, evaluation-window, detailed DD-latency, and HECM lookup-sensitivity material into Appendix A.
- Added a compact reviewer audit in `REVIEWER_COVERAGE_JES2.md`; the older `REVIEWER_TODO_STATUS.txt` is retained only as a historical implementation log.
- Added `verify_independent_audit.py` and `INDEPENDENT_AUDIT_VERIFICATION.md` so the four methodological audit corrections and three reporting corrections can be checked against the frozen result files rather than accepted from prose alone.

## Interpretation boundaries retained in the manuscript

- The conclusions apply to the tested LFP cells, operating envelope, estimator implementations, measurement disturbances, and model-specific initialization interventions.
- The high-load class contains only C29 and is not interpreted as a population-level high-load estimate.
- Nominal SOC metrics quantify accuracy against the common dataset ground truth, which is reconstructed offline by Coulomb counting and voltage-anchor logic.
- The HECM lookup surfaces originate from the training/validation development pool and exclude the holdout cells. Their HPPC identification records and fitted surfaces remain required release artifacts.
- Hardware measurements cover the SOC inference cores; the shared SOH LSTM and complete BMS scheduling are outside the on-device timing boundary.
- Peak runtime RAM combines statically allocated variable storage with on-device measurements of maximum call-stack and dynamic-memory use. The shared SOH-LSTM, concurrent BMS tasks, and board-level energy remain outside the measured hardware boundary.
- The causal LSTM-SOH trace remains the primary common input; the reference-SOH substitution is reported only as an ablation.

## Validation performed

- Recompiled the manuscript and inspected the revised recovery figure, core result figures, hardware figures, retained holdout summary, SOH-aging figure and 15-cell table, Appendix protocol figure, HECM lookup-sensitivity figure, and conclusion pages.
- The hardware PC must still export the raw per-cell STM32 summaries and timing records and rebuild the firmware from a clean tagged state. The current RAM record identifies `c3307581-dirty`.
- Confirmed all 26 finalized PNG figures are referenced exactly once.
- Confirmed that no missing-figure placeholders remain.
- Confirmed that all active labels and references resolve.
- Ran the complete simulation-environment test suite: 35 tests passed after adding the full HECM sensitivity regression tests.
- Ran `python verify_independent_audit.py`: all common-mask, recovery, robustness-score, offset/statistics, signed-build, lifecycle, HECM lookup-sensitivity, dataset-ground-truth, hardware-boundary, and manuscript-consistency checks passed.
- The test suite requires the `ml1` Conda libraries on this HPC: `LD_LIBRARY_PATH=/home/florianr/anaconda3/envs/ml1/lib /home/florianr/anaconda3/envs/ml1/bin/python -m pytest -q`.
- Confirmed 7,152 execution records with no failures, comprising 7,040 public benchmark evaluations and 112 internal duration-matched dropout-reference runs.
- Completed all 9,184 HECM lookup-sensitivity runs without failures and analyzed them with 10,000 cell-bootstrap repetitions.
- Rechecked the protected original PDF hash after compilation.
