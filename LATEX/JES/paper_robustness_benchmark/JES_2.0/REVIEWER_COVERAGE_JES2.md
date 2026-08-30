# JES 2.0 reviewer coverage audit

Updated: 2026-08-30

This file maps the editor and reviewer comments to the current revised
manuscript. It is intentionally separate from the paper so that the manuscript
remains concise. `REVIEWER_TODO_STATUS.txt` is the historical implementation log.

## Reviewer 1

| Comment | Current response | Status |
|---|---|---|
| Citations/figure markers and figure placement | The compiled PDF has resolved citations/references and embeds every main figure near its result section. | Addressed |
| Equation (9) as standalone equation | The derivative definitions are now inline and unnumbered. | Addressed |
| Duplicate Section 2 headings | The headings are `SOC estimation methods` and `SOH estimation method`. | Addressed |

## Reviewer 2

| Comment | Current response | Status |
|---|---|---|
| Single validation trajectory | Six independent holdout cells and 16 predeclared Fresh/Mid-Life/Aged windows replace the single-trajectory comparison. External validity remains limited to the measured LFP envelope. | Addressed within available dataset |
| Independence of development and benchmark | Neural training, validation, tuning, scaling, pruning, and checkpoint selection exclude all six holdout cells. The imported HPPC-derived HECM surfaces predate the campaign and are not fitted or retuned with scored trajectories. Splits and hashes are stored in manifests. | Addressed; retain exact HPPC provenance in response letter |
| Unfair DD initialization proxy | All models receive a requested 10% SOC-equivalent intervention, realized output shifts are reported, and the text explicitly states that the interventions are not identical latent-state perturbations. The revised result no longer claims DD is the fastest; HECM is. | Addressed transparently, structural limitation retained |
| Estimator-specific recovery bands | One common 2% paired trajectory-difference threshold, 300-s hold time, and 24-h censor horizon are used. The endpoint compares perturbed and correctly initialized runs, not prediction error against dataset SOC. | Addressed |
| Embedded-deployment evidence | Six-cell STM32H753ZI tests report numerical equivalence, latency distributions, compiled flash occupancy, measured peak runtime RAM, and one-second deadline occupation. Energy and concurrent total-BMS CPU load are explicitly excluded. | Substantially addressed; energy and system-level measurements remain open |
| Missing statistical validation | Repeated seeds, equal-weight cell macro statistics, 10,000 hierarchical bootstrap repetitions, paired sign-flip tests, effect sizes, and Holm correction are reported. | Addressed |
| Shared SOH confounder | Selected scenarios are rerun with paired reference SOH. Results distinguish absolute SOH calibration effects from incremental disturbance effects. | Addressed |

## Reviewer 3

| Comment | Current response | Status |
|---|---|---|
| Representativeness of one trajectory and estimator classes | Six cells span different aging, load, thermal, and duration envelopes; load classes and missing strata are reported descriptively. A dedicated methods table defines each class envelope and the exact tested representative, and the abstract, discussion, limitations, and conclusion avoid family-wide claims. | Addressed within available dataset and implementation scope |
| Why LFP and NMC/NCA limits | Dataset suitability and LFP-specific flat-OCV/aging limitations are now explicit. No cross-chemistry claim is made. | Addressed |
| Shared SOH bottleneck | Same paired reference-SOH ablation as Reviewer 2.7. | Addressed |
| Hourly SOH cadence | The cadence is justified for gradual capacity fade; abrupt capacity-loss events are absent and explicitly outside scope. | Addressed with limitation |
| DD cold-start proxy | The accessible Qc intervention and its non-equivalence to hidden-state corruption are disclosed; a true hidden-state cold-start study remains future work. | Addressed with limitation |
| Long dropout recovery | Burst dropout is evaluated as a six-cell robustness penalty against a duration-matched 48-h baseline. A paired C29 trajectory illustrates post-gap behavior, but no six-cell dropout-recovery ranking is claimed because matched undisturbed trajectories were not retained for every original cell-window run. Canonical recovery is restricted to the paired initialization experiment. | Addressed without unsupported recovery claim |
| Larger timing jitter | The campaign includes +/-0.1, +/-0.5, and +/-0.9 s with repeated seeds. | Addressed |
| DD voltage-spike mechanism | Event alignment supports the voltage-plus-derivative/recurrent-context explanation; the manuscript labels it as an interpretation because no channel-removal ablation was run. | Addressed without overclaiming |
| Estimator-specific recovery thresholds | Replaced by the common recovery criterion. | Addressed |
| Runtime-memory lower bounds | Statically allocated variable storage is combined with on-device measurements of maximum call-stack and dynamic-memory use. The values cover the SOC firmware execution but exclude the shared SOH-LSTM and concurrent BMS tasks. | Addressed for the isolated SOC-core boundary |
| HECM dependence on lookup-table quality | The main result is explicitly conditional on the fixed HPPC tables. No full six-cell parameter-mismatch sensitivity campaign is included. | Open optional software extension |
| Temperature response through SOH LSTM | Paired reference-SOH temperature-noise runs show that the incremental substitution effect differs from baseline by 0.0015 SOC for HDM and less than 0.0001 for HECM/DD. | Addressed |
| Measurement-only exclusions | Parameter mismatch, hysteresis, gradients, pack imbalance, balancing/contactor/protocol/numerical/deadline faults are listed in limitations. | Addressed |
| Reproducibility artefacts | Public dataset DOI and repository are named; the minimum contents of the versioned release are listed. | Release packaging still required before resubmission |

## Independent audit corrections

| Audit finding | Implemented correction | Verification |
|---|---|---|
| DD and the other estimators used different scored samples | Every estimator now uses source samples 2023 onward. Each 24-h run contributes 84,377 matched points and each 48-h dropout run contributes 170,777. Non-DD runs were repeated and DD summaries were reused only where their rolling-window output already used this exact interval. | Corrected 6,720-run manifest has no failures. Every primary summary records the common start and sample count. |
| Recovery definitions and confidence intervals were inconsistent | One canonical paired initialization analysis compares shifted and correctly initialized trajectories. It uses a 0.02 SOC band held for 300 s, a 24-h censor horizon, a separate relapse endpoint, and hierarchical cell-level confidence intervals. | HECM has the strongest tested recovery profile. The same statistics file drives the text, Figure 09, Appendix table, and decision synthesis. |
| Robustness score omitted sensor offsets and implied an objective winner | The illustrative score now gives equal weight to eight declared families after averaging levels within each family. Voltage and temperature offsets are included. Equal-scenario and high-severity alternatives are reported as sensitivity analyses. | The manuscript states that normalized scores are weighting dependent and retains scenario-level evidence as the primary result. |
| Dropout placement was described as central | The method and Appendix now state that the eligible one-hour interval with maximum absolute net charge is selected with at least 12 h before and 24 h after the gap. | Protocol code, table, caption, and Appendix graphic use the same definition. |
| Multiplicative current error was called bias | Visible manuscript and figure terminology is `current-gain error`. Historical internal aliases remain unchanged for traceability. | Signed positive and negative reruns support the adverse-direction result. |
| Voltage and temperature offsets were underreported | Both offsets appear in the protocol, result discussion, heatmap, Appendix table, and robustness synthesis. | The DD temperature-offset penalty is 0.0074 SOC with a cell-bootstrap interval above zero. |
| Sign-flip and Holm results were announced but not shown | A compact Appendix table now reports all six baseline model-pair differences, confidence intervals, paired effect sizes, exact sign-flip values, and Holm-adjusted values. | Machine-readable scenario and model-pair tables remain in `JES_2.0/results`. |

## Remaining actions before resubmission

1. Package and archive the exact frozen weights/scalers, ECM tables, environment,
   manifests, summaries, and rebuild command as a versioned release.
2. Confirm the originating HPPC experiment/cell provenance of the imported HECM
   lookup surfaces in the response letter.
3. Retain the explicit distinction between the measured SOC-core peak RAM and the
   still-open memory requirement of a concurrent SOC--SOH BMS implementation.
4. Treat a full HECM parameter-mismatch sweep as optional additional evidence,
   not as part of the measurement-only primary ranking.

## Scope and length decision

The corrected audit reuses the existing figure sequence. Figure 09 is rebuilt
from the canonical paired recovery calculation, Figures 14 and 15 are rebuilt
from the complete disturbance set, and the Appendix protocol figure documents
the common scored interval and dropout placement. One compact scope table
distinguishes estimator families from the exact benchmark representatives.
Detailed reviewer mapping belongs in the response letter, while the paper keeps
only information required to understand or qualify the methods and results.
