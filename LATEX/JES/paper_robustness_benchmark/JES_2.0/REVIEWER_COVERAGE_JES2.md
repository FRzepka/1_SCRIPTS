# JES 2.0 reviewer coverage audit

Updated: 2026-08-28

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
| Estimator-specific recovery bands | One common 2% absolute-error threshold, 300-s hold time, and 24-h censor horizon are used. | Addressed |
| Embedded-deployment evidence | Six-cell STM32H753ZI tests report numerical equivalence, latency distributions, flash, static RAM, and one-second deadline occupation. Peak runtime RAM, energy, and concurrent total-BMS CPU load are explicitly excluded. | Substantially addressed; peak/system measurements remain open |
| Missing statistical validation | Repeated seeds, equal-weight cell macro statistics, 10,000 hierarchical bootstrap repetitions, paired sign-flip tests, effect sizes, and Holm correction are reported. | Addressed |
| Shared SOH confounder | Selected scenarios are rerun with paired reference SOH. Results distinguish absolute SOH calibration effects from incremental disturbance effects. | Addressed |

## Reviewer 3

| Comment | Current response | Status |
|---|---|---|
| Representativeness of one trajectory | Six cells span different aging, load, thermal, and duration envelopes; load classes and missing strata are reported descriptively. | Addressed within available dataset |
| Why LFP and NMC/NCA limits | Dataset suitability and LFP-specific flat-OCV/aging limitations are now explicit. No cross-chemistry claim is made. | Addressed |
| Shared SOH bottleneck | Same paired reference-SOH ablation as Reviewer 2.7. | Addressed |
| Hourly SOH cadence | The cadence is justified for gradual capacity fade; abrupt capacity-loss events are absent and explicitly outside scope. | Addressed with limitation |
| DD cold-start proxy | The accessible Qc intervention and its non-equivalence to hidden-state corruption are disclosed; a true hidden-state cold-start study remains future work. | Addressed with limitation |
| Long dropout recovery | The rerun uses a common criterion and now yields 4.13-12.45 h means. The paper explains unobserved charge, weak LFP voltage anchoring, and rolling-context replacement as model-specific mechanisms. | Addressed |
| Larger timing jitter | The campaign includes +/-0.1, +/-0.5, and +/-0.9 s with repeated seeds. | Addressed |
| DD voltage-spike mechanism | Event alignment supports the voltage-plus-derivative/recurrent-context explanation; the manuscript labels it as an interpretation because no channel-removal ablation was run. | Addressed without overclaiming |
| Estimator-specific recovery thresholds | Replaced by the common recovery criterion. | Addressed |
| Runtime-memory lower bounds | Measured static firmware allocation replaces analytical persistent-state estimates, but peak stack/dynamic memory remains unmeasured and is stated as such. | Partly open hardware boundary |
| HECM dependence on lookup-table quality | The main result is explicitly conditional on the fixed HPPC tables. No full six-cell parameter-mismatch sensitivity campaign is included. | Open optional software extension |
| Temperature response through SOH LSTM | Paired reference-SOH temperature-noise runs show that the incremental substitution effect differs from baseline by 0.0015 SOC for HDM and less than 0.0001 for HECM/DD. | Addressed |
| Measurement-only exclusions | Parameter mismatch, hysteresis, gradients, pack imbalance, balancing/contactor/protocol/numerical/deadline faults are listed in limitations. | Addressed |
| Reproducibility artefacts | Public dataset DOI and repository are named; the minimum contents of the versioned release are listed. | Release packaging still required before resubmission |

## Remaining actions before resubmission

1. Package and archive the exact frozen weights/scalers, ECM tables, environment,
   manifests, summaries, and rebuild command as a versioned release.
2. Confirm the originating HPPC experiment/cell provenance of the imported HECM
   lookup surfaces in the response letter.
3. Decide whether the existing measured static RAM is sufficient for the revised,
   explicitly limited SOC-core claim or whether peak-stack instrumentation should
   be added later.
4. Treat a full HECM parameter-mismatch sweep as optional additional evidence,
   not as part of the measurement-only primary ranking.

## Scope and length decision

No additional manuscript figures or tables were added in this audit. The revised
PDF remains 45 pages in review layout. Detailed reviewer mapping belongs in the
response letter; the paper retains only information required to understand or
qualify the methods and results.
