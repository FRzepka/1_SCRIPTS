# Paper-ready robustness results

This directory contains curated figures and tables for the robustness-benchmark manuscript.

## Global vs local evidence

- `paper_tables/table_key_results.md` contains the complete **global** scenario metrics used for ranking.
- `paper_tables/table_local_behaviour.md` contains the **local** recovery and drift metrics used when global MAE is not informative enough.
- `paper_tables/table_figure_scope.md` maps every paper figure to its global and local evidence.
- Figures that combine global and local evidence are explicitly marked below.

## Figures

- `Figure_1_baseline_performance.png`: baseline ranking on clean data (**global**)
- `Figure_2_current_bias.png`: current-bias penalty (**global**) plus early drift trajectory (**local**)
- `Figure_3_noise_robustness.png`: global noise penalties plus local rolling-error drift
- `Figure_4_signal_integrity.png`: global integrity penalties plus local burst-dropout recovery time
- `Figure_5_missing_gap_transition.png`: sensor and SOC transition around the dropout window (**local**)
- `Figure_6_missing_gap_recovery.png`: post-gap recovery trajectory (**local**)
- `Figure_7_initial_state_recovery.png`: CV-free initial-state recovery analysis (**local**; global MAE in key-results table)
- `Figure_8_spike_response.png`: global spike penalty plus local per-spike response

## Current-bias interpretation to preserve in the manuscript

- Figure 2 is intentionally mixed-scope.
- The left panel shows the **global** current-bias penalty via $\Delta$MAE on the complete run.
- The right panel shows a **local** early-window drift view (first 12 h) to expose the physical bias mechanism.
- This figure must not be described as if the right panel were the full-run behaviour.
- For the complete-run global ranking under current bias, use the values in `table_key_results.md`.
- Separate verification plots for the full run are stored under `results/figures_test/`.

Current-bias global ranking:
- `HECM`: global MAE `0.0945`, global $\Delta$MAE `0.0858`

## Tables

- `table_baseline.md`: clean-data baseline metrics (**global**)
- `table_key_results.md`: key scenario metrics and penalties (**global**)
- `table_local_behaviour.md`: local recovery and drift metrics (**local**)
- `table_figure_scope.md`: explicit global/local mapping for each figure

## Model classes

- `DM`: Direct measurement
- `HDM`: Hybrid direct measurement
- `HECM`: Hybrid ECM
- `DD`: Data-driven
