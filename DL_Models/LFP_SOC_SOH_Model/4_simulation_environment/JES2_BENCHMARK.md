# JES 2.0 Benchmark

`run_jes2_benchmark.py` keeps the existing SOC estimators unchanged and gives
HDM, HECM, and DD the same SOH input for every cell, disturbance, and seed.
DM remains the no-SOH baseline.

The primary test set contains the six model-development holdouts C09, C13, C15,
C25, C27, and C29. The runner rejects training/validation cells by default and
verifies that the frozen SOC and SOH configurations declare identical train and
validation exposure.

## SOH conditions

- `lstm`: primary full-system result using the frozen SOH JES2 model.
- `reference`: ideal-SOH ablation that isolates error propagation from the SOH
  estimator. It is not the primary hybrid result.
- SOH updates are causal: an interval estimate becomes available only after the
  interval is complete and is applied to the next interval.
- Fully missing intervals hold the previous SOH and do not update the LSTM.
- `--lstm_publish_intervals 1 6 24` tests SOH-context publication every 1, 6,
  and 24 hours without changing or retraining the hourly LSTM.

## Statistical protocol

- Deterministic disturbances run once per cell.
- Gaussian current/voltage/temperature noise uses 10 deterministic seeds.
- Random sample loss, voltage-spike signs, and timing jitter use 5 seeds. These
  repeat budgets still span every available cell/SOH window.
- Timing jitter covers +/-0.1 s, +/-0.5 s, and +/-0.9 s.
- The -0.10 initialization mismatch is mapped into every estimator's available
  online state: the coulomb-counting SOC state for DM/HDM, the EKF SOC state for
  HECM, and the capacity-equivalent `Q_c` feature offset for DD. All four use one
  common 0.02 SOC error band held for 300 s. Persistent recovery additionally
  requires remaining in the band through the 24 h horizon. Observation starts at
  source sample 2023, about 0.562 h after the intervention. Endpoints already
  satisfied there are left-censored and recorded at 0.562 h as conservative upper
  bounds. The difference at the first commonly scored sample is reported so the
  DD mapping is not assumed to be identical silently.
- Paper aggregation first averages protocol-defined windows inside each cell/seed,
  then gives every cell equal weight and calculates hierarchical bootstrap
  confidence intervals with seeds nested inside cells.
- Paired scenario effects and model-to-model differences are evaluated on
  cell-level means with exact two-sided sign-flip tests, paired `dz` effect
  sizes, bootstrap confidence intervals, and Holm-adjusted p-values.
- All global, stratified, and paired comparisons score every estimator on the
  common source-sample mask beginning at sample 2023. A 24 h window therefore
  contributes 84,377 matched points per model.
- Individual time samples are never treated as independent replicates.

## State-stratified tests

Every `summary.json` contains MAE, RMSE, bias, P95 error, sample count, and
coverage fraction for fixed reference-state strata, even in the default
`summary-only` mode:

- SOH: fresh (`>=0.90`), mid-life (`0.80-0.90`), aged (`<0.80`).
- Instantaneous load: `<0.5 C`, `0.5-1.5 C`, `>=1.5 C`.
- Temperature: `<=30 degC`, `30-35 degC`, `>35 degC`.
- SOC: `<0.20`, `0.20-0.80`, `>0.80`.

The six cells are also assigned to descriptive load groups based only on their
measured 95th-percentile absolute C-rate: low C25/C27, middle C09/C13/C15, and
high C29. These labels are not balanced treatment groups. High-load evidence has
one cell and is exploratory.

## Frozen evaluation windows

The primary benchmark does not replay every redundant second of each multi-month
aging trajectory. `select_jes2_evaluation_windows.py` freezes one representative
24 h window for every available Fresh/Mid-Life/Aged state in each holdout cell.
Every window starts at a measured full-charge anchor (SOC >= 0.98 and voltage >=
3.58 V), so all SOC estimators receive the same physically defined start. The
selector uses the medoid of measured SOH, temperature, C-rate, throughput, and SOC
coverage; it never sees estimator predictions or errors. C27 correctly contributes
only Fresh because its measured SOH never falls below 0.90.

The one-hour dropout scenario uses 48 h from the same anchor. The eligible gap
with maximum absolute integrated net charge is selected from measured current
only, with at least 12 h before and 24 h after the gap. Dropout is summarized as
a paired robustness penalty and is not mixed into the canonical initialization-
recovery score. The shared SOH LSTM is
initialized with the preceding 192 h of undisturbed causal measurements, matching
its trained sequence length. These context rows initialize SOH only and are not
included in SOC metrics. Window definitions are frozen in
`JES_2.0/tables/jes2_evaluation_windows.csv`.

## Six-cell pilot

Run this reduced campaign before freezing and launching the full protocol:

```bash
LD_LIBRARY_PATH=/home/florianr/anaconda3/envs/ml1/lib:${LD_LIBRARY_PATH:-} \
/home/florianr/anaconda3/envs/ml1/bin/python \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/run_jes2_benchmark.py \
  --cells C09 C13 C15 C25 C27 C29 \
  --aliases baseline current_noise_high initial_soc_error missing_gap_1h \
  --stochastic_repeats 3 \
  --secondary_stochastic_repeats 3 \
  --window_manifest LATEX/JES/paper_robustness_benchmark/JES_2.0/tables/jes2_evaluation_windows.csv \
  --tag jes2_six_cell_pilot \
  --trace_device cuda \
  --model_device cuda \
  --skip_existing
```

## Full holdout campaign

The local guarded scheduler `run_jes2_full_after_pilot.sh` waits for the pilot and
DD window pilot, reruns tests, validates the complete 6,720-run frozen-window
plan, executes a real all-scenario smoke, checks free disk
space, and only then launches the six full cell shards.

For HPC execution, run one command per cell as an independent GPU job and give
each job a unique tag (`jes2_full_C09`, ..., `jes2_full_C29`). The command below
shows one shard; replace `C09` and its tag for the remaining holdout cells:

```bash
LD_LIBRARY_PATH=/home/florianr/anaconda3/envs/ml1/lib:${LD_LIBRARY_PATH:-} \
/home/florianr/anaconda3/envs/ml1/bin/python \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/run_jes2_benchmark.py \
  --cells C09 \
  --tag jes2_full_C09 \
  --lstm_publish_intervals 1 6 24 \
  --window_manifest LATEX/JES/paper_robustness_benchmark/JES_2.0/tables/jes2_evaluation_windows.csv \
  --trace_device cuda \
  --model_device cuda \
  --skip_existing
```

After all six jobs finish, merge their manifests. The result builder performs a
strict cell/scenario/seed/SOH-condition/model completeness check on this merged
manifest before creating publication results:

```bash
/home/florianr/anaconda3/envs/ml1/bin/python \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/merge_jes2_manifests.py \
  --manifests \
    DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_C09/jes2_manifest.json \
    DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_C13/jes2_manifest.json \
    DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_C15/jes2_manifest.json \
    DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_C25/jes2_manifest.json \
    DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_C27/jes2_manifest.json \
    DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_C29/jes2_manifest.json \
  --out DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_holdout_merged.json \
  --tag jes2_full_holdout
```

Gaussian sensor-noise scenarios use 10 seeds, secondary stochastic integrity
scenarios use 5, and deterministic scenarios run once. Every run, window, trace,
and artifact path is recorded under
`campaigns/<tag>/jes2_manifest.json`.

The final benchmark build contains 19 scenario definitions. The 6,720-row main
manifest is combined with 192 matched negative-sign evaluations for the three
current-gain magnitudes. Each of these magnitudes therefore contains positive
and negative sign sublevels. Figures that report an adverse current-gain
direction take the larger paired delta MAE from each matched sign pair. This
prevents sign-dependent compensation with baseline error from being interpreted
as robustness.

Campaign runs write `summary.json` only by default. This avoids multi-gigabyte
per-run CSV/PNG output when full trajectories are repeated across cells and
seeds. Use `--keep_run_artifacts` only for a small diagnostic campaign whose
per-sample traces are actually needed. Shared causal SOH traces remain cached so
all SOH-dependent SOC branches consume exactly the same context.

Generate tables and dissertation-colored paper figures only after the campaign
has completed:

```bash
/home/florianr/anaconda3/envs/ml1/bin/python \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/results/build_jes2_paper_results.py \
  --manifest DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_holdout_merged.json \
  --out_dir LATEX/JES/paper_robustness_benchmark/JES_2.0/results \
  --figures_dir LATEX/JES/paper_robustness_benchmark/figures/Results
```

The result builder additionally writes state-stratified raw/aggregate tables,
paired scenario and model tests, the statistical-method note, and Figure 20-23
for SOH state, load-class x SOH interaction, SOH-dependent robustness, and
measured operating regimes.

Check a running campaign without `rg`:

```bash
watch -n 10 '/home/florianr/anaconda3/bin/python \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/jes2_campaign_progress.py \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/<tag>/jes2_manifest.json'
```

Run the separate HECM lookup-table sensitivity after the primary campaign. It
is explanatory model-parameter analysis and is not included in the
measurement-only robustness ranking:

```bash
/home/florianr/anaconda3/envs/ml1/bin/python \
  DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/run_jes2_hecm_parameter_sensitivity.py \
  --campaign_manifest DL_Models/LFP_SOC_SOH_Model/4_simulation_environment/campaigns/jes2_full_holdout_merged.json \
  --device cuda \
  --figures_dir LATEX/JES/paper_robustness_benchmark/figures/Results \
  --skip_existing
```

`--start_row` and `--max_rows` remain intended for smoke tests. Publication runs
must use the frozen window manifest so late windows receive the declared 192 h
causal SOH context rather than a cold start.
