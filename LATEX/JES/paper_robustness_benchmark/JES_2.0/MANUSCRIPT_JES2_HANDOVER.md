# JES 2.0 manuscript handover

Date: 2026-08-31

## Protected original

The submitted manuscript PDF remains unchanged:

- `../Robustness_Benchmark_Manuscript.pdf`
- SHA-256: `d75d3c6039ad58bab336a98dde095d7eb0d1325498b53e1ce9df9f877272df42`

All revision work is confined to this `JES_2.0` directory.

## Revision artifacts

- Editable source: `Robustness_Benchmark_Manuscript_JES2_Updated.tex`
- Compiled manuscript: `Robustness_Benchmark_Manuscript_JES2_Updated.pdf`
- Source SHA-256: `cd30aec1c26cc85264dd22e415fb6462555833154760ee3acafb0bb60c0a2e22`
- PDF SHA-256: `7dd684c152fb51df4a6c4b3fd1cfeea4822bbfa5245d628a639c591aa4cc1c39`
- Compiled length: 54 pages

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
- Documented 16 protocol-defined 24-hour windows, 19 final scenario definitions, six holdout cells, 10 sensor-noise seeds, 5 seeds for the other stochastic cases, 6,912 public model-window evaluations, and 10,000 hierarchical bootstrap repetitions.
- Defined each of the three current-gain magnitudes through matched positive and negative sign sublevels in the final benchmark build. The continuous C29 lifecycle replay remains a complementary single-cell mechanism analysis.
- Treated cells as the independent units and seeds/windows as nested observations.
- Recomputed every global result on the common DD-valid source-sample interval beginning at sample 2023. Each model now contributes 84,377 matched samples per 24-hour window.
- Rebuilt the initialization figure exclusively from the corrected dedicated paired campaign. The primary endpoint is persistent recovery through the remaining 24-hour horizon. First 300-second entry and later relapse are retained as separate diagnostics. Observation begins about 0.562 hours after intervention, and boundary endpoints are flagged as left-censored upper bounds.
- Removed relapse from the composite recovery score because it is undefined when a run never enters the recovery band and would otherwise reward non-recovery.
- Audited all 24 finalized PNG files against their manuscript captions and dependencies. Only Figures 09 and 15 depend on the corrected recovery analysis. Figure 14 deliberately excludes initialization mismatch because it is a measurement-disturbance heatmap.
- Included all eight evaluated disturbance families in the illustrative robustness score and added two alternative weighting analyses. The current-gain family uses the adverse direction from the matched signed sweep.
- Added current-gain lifecycle/reset, initial-state recovery, missing-sample, jitter, burst-dropout, voltage-spike, ADC-quantization, sensor-offset, and cross-scenario interpretations.
- Replaced the legacy C07 current-noise illustration with a reset-free, sample-matched C29 mechanism example and retained the final six-cell hierarchical confidence intervals for the population-level noise result.
- Recomputed burst-dropout penalties against an undisturbed 48-hour reference and documented the maximum-absolute-net-charge placement rule.
- Separated the robustness benchmark from the embedded performance benchmark and added isolated STM32H753ZI SOC-core results for numerical equivalence, latency, compiled flash occupancy, measured peak runtime RAM, and DD inference-mode tradeoffs.
- Moved coverage, test-matrix, evaluation-window, and detailed DD-latency material into Appendix A.
- Added a compact reviewer audit in `REVIEWER_COVERAGE_JES2.md`; the older `REVIEWER_TODO_STATUS.txt` is retained only as a historical implementation log.
- Added `verify_independent_audit.py` and `INDEPENDENT_AUDIT_VERIFICATION.md` so the four methodological audit corrections and three reporting corrections can be checked against the frozen result files rather than accepted from prose alone.

## Interpretation boundaries retained in the manuscript

- The conclusions apply to the tested LFP cells, operating envelope, estimator implementations, measurement disturbances, and model-specific initialization interventions.
- The high-load class contains only C29 and is not interpreted as a population-level high-load estimate.
- Nominal SOC metrics quantify accuracy against the common dataset ground truth, which is reconstructed offline by Coulomb counting and voltage-anchor logic.
- The HECM lookup table is not retuned in the benchmark, but its originating HPPC-cell metadata is absent and must be documented before resubmission.
- Hardware measurements cover the SOC inference cores; the shared SOH LSTM and complete BMS scheduling are outside the on-device timing boundary.
- Peak runtime RAM combines statically allocated variable storage with on-device measurements of maximum call-stack and dynamic-memory use. The shared SOH-LSTM, concurrent BMS tasks, and board-level energy remain outside the measured hardware boundary.
- The causal LSTM-SOH trace remains the primary common input; the reference-SOH substitution is reported only as an ablation.

## Validation performed

- Recompiled the 54-page manuscript and inspected the revised recovery figure, core result figures, hardware figures, Appendix protocol figure, and conclusion pages.
- The hardware PC must still export the raw per-cell STM32 summaries and timing records and rebuild the firmware from a clean tagged state. The current RAM record identifies `c3307581-dirty`.
- Confirmed all 24 finalized PNG figures are referenced exactly once.
- Confirmed that no missing-figure placeholders remain.
- Confirmed that all active labels and references resolve.
- Ran the complete simulation-environment test suite: 32 tests passed.
- Ran `python verify_independent_audit.py`: all common-mask, recovery, robustness-score, offset/statistics, signed-build, lifecycle, dataset-ground-truth, hardware-boundary, and manuscript-consistency checks passed.
- The test suite requires the `ml1` Conda libraries on this HPC: `LD_LIBRARY_PATH=/home/florianr/anaconda3/envs/ml1/lib /home/florianr/anaconda3/envs/ml1/bin/python -m pytest -q`.
- Confirmed 7,024 execution records with no failures, comprising 6,912 public benchmark evaluations and 112 internal duration-matched dropout-reference runs.
- Rechecked the protected original PDF hash after compilation.
