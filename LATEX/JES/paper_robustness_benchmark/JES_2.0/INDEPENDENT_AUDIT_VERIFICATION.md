# Independent audit verification

Verified: 2026-08-31

This record addresses the independent audit supplied after the JES 2.0 revision.
The paper remains concise. Detailed implementation evidence is retained here and
is checked by `python verify_independent_audit.py`.

| Audit issue | Verified resolution | Evidence |
|---|---|---|
| Different scored samples across estimators | All 6,720 public benchmark rows start at source sample 2023. A 24-h run has 84,377 scored points and a 48-h dropout run has 170,777 for every estimator. All run-by-dimension stratified counts sum to the same run-level count. | `results/jes2_run_metrics.csv`, `results/jes2_stratified_run_metrics.csv` |
| Conflicting recovery reference and endpoint | The only production recovery analysis compares each shifted trajectory with its matched correctly initialized trajectory. First entry requires 300 continuous seconds within 0.02 SOC. Persistent recovery requires remaining inside for the rest of 24 h and is the primary endpoint. Relapse is separate. | `results/jes2_paired_initial_recovery_method.txt`, `results/jes2_paired_initial_recovery_runs.csv`, `results/jes2_paired_initial_recovery_statistics.csv` |
| Recovery plot/statistics mismatch | Figure 09 reads the canonical paired statistics and uses hierarchical cell-level intervals. Figure 15 reads the same statistics. The legacy aggregate recovery plotter no longer emits a competing recovery figure. | `figures/build_figure_07_initial_recovery_corr.py`, `figures/build_revised_all_cells_figures.py`, `build_jes2_trajectory_figures.py` |
| Offsets omitted from the robustness score | The family-balanced synthesis contains all eight evaluated families, including the joint sensor-offset family. Equal-scenario and highest-level-plus-offset variants expose weighting sensitivity. | `results/jes2_robustness_family_penalties.csv`, `results/jes2_robustness_score_sensitivity.csv`, Appendix score table |
| Score presented as an objective winner | The abstract, methods, results, Figure 15 caption, discussion, and conclusion identify the synthesis as illustrative and weighting dependent. Scenario-level evidence remains primary. | Manuscript Sections 2.4, 4.7, 5.3, and 6 |
| Dropout placement described incorrectly | The protocol selects the eligible one-hour interval with maximum absolute integrated net charge using measured current only, with at least 12 h before and 24 h after. | Protocol table, implementation details, Figure 23 caption, `jes2_protocol.py` |
| Multiplicative current error called bias | Visible paper and final-figure terminology is `current-gain error`. Internal scenario aliases retain `current_bias_*` only to preserve result provenance. | Manuscript protocol and current-gain results, final figure captions |
| Voltage and temperature offsets underreported | Both are declared in the protocol, discussed in the results, included in Figure 14 and Figure 15, and listed numerically in the Appendix. | Protocol table, offset results paragraph, Appendix scenario table |
| Sign-flip/Holm results announced but absent | All six baseline model-pair comparisons are printed in the Appendix with mean difference, confidence interval, paired effect size, exact sign-flip value, and Holm-adjusted value. | Appendix baseline-pair table, `results/jes2_paired_model_tests.csv` |
| Signed current-gain build definition unclear | The final 19-scenario build defines three current-gain magnitudes with matched positive and negative sign sublevels. Adverse-direction summaries use the larger paired delta MAE at each magnitude. | Main and matched-sign execution manifests, protocol table, current-gain Results and Limitations |
| Recovery starts after the intervention | Observation begins at source sample 2023, about 0.562 h after intervention. Runs already satisfying an endpoint there are flagged as left-censored, and the boundary value is described as a conservative upper bound. | `results/jes2_paired_initial_recovery_runs.csv`, recovery method note, Figure 09, manuscript Methods and Limitations |
| C29 lifecycle interpretation contradicted its data | The text now reports HECM, not DD, as the smallest full-life adverse gain-error contribution on C29 and explicitly avoids a six-cell lifecycle claim. | `results/c29_bias_temporal_model_summary.csv`, Figure 06 and current-gain Results |
| Dataset ground-truth construction unclear | The manuscript consistently calls the reconstructed SOC the dataset ground truth and states its offline Coulomb-counting and voltage-anchor construction. | Ground-truth Methods, Baseline Results, Limitations and Conclusion |
| HECM lookup-table dependence unquantified | A separate 240-run analysis perturbs resistance by +/-10% and OCV by +/-10 mV under baseline and matched +/-3% current gain. All 240 runs use the common mask. The largest absolute macro lookup--gain interaction is 0.000260 MAE and all interaction intervals include zero. | `results/hecm_lookup_sensitivity_*.csv`, Figure 25, HECM Methods, Results and Discussion |
| Hardware and HECM provenance overstated | Runtime RAM is tied to the C09 replay, exact rolling-DD stack use to two valid calls, and SOC-core results are not called full-BMS feasibility. The HECM lookup surfaces originate from the development pool and exclude holdout cells. Raw HPPC identification artifacts, raw hardware records, and a clean firmware state remain release-packaging actions. | Model preparation, Hardware Methods and Limitations, Data availability, release actions below |

The original submitted PDF remains protected. Before resubmission, the hardware
PC must export the per-cell source summaries and timing records used for Figures
17, 18, 20, and 24, and the firmware must be rebuilt from a clean tagged source
state. The current RAM record identifies `c3307581-dirty`. The HPPC identification
records and fitted HECM surfaces from the training/validation development pool
must be included in the versioned release. These are release-packaging actions,
not reasons to alter the already recorded result values. The currently used HECM MAT file has SHA-256
`bba8bfb6d5946eb6e1fca965cbb6b2daa98af71b200c6b6585c4cb0f6ef56b08`,
and the runtime-memory record has SHA-256
`e5c9d8a7db770e5dee8df5ba6c5fd30a48385aa0f9b28659b43787cb086e2db6`.
