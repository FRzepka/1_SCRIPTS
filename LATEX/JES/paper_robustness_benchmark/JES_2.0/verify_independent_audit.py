from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
MANUSCRIPT = ROOT / "Robustness_Benchmark_Manuscript_JES2_Updated.tex"
FIGURES = ROOT.parent / "figures" / "Results" / "All Cells"
WORKSPACE = ROOT.parents[3]
SIMULATION = WORKSPACE / "DL_Models" / "LFP_SOC_SOH_Model" / "4_simulation_environment"
FIGURE_SCRIPTS = ROOT.parent / "figures"
MODELS = {"DM", "HDM", "HECM", "DD"}
FAMILIES = [
    "Sensor noise",
    "Current-gain error",
    "Sensor offsets",
    "ADC quantization",
    "Missing samples",
    "Timing jitter",
    "Burst dropout",
    "Voltage spikes",
]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def check_common_mask() -> str:
    runs = pd.read_csv(RESULTS / "jes2_run_metrics.csv")
    require(len(runs) == 6720, f"Expected 6720 benchmark runs, found {len(runs)}")
    require(set(runs["model"]) == MODELS, "Run manifest does not contain all four models")
    require(
        set(runs["evaluation_start_sample"].astype(int)) == {2023},
        "Not every run starts scoring at source sample 2023",
    )
    expected = runs["max_rows"].astype(int) - 2023
    require(
        np.array_equal(runs["evaluation_samples"].astype(int).to_numpy(), expected.to_numpy()),
        "At least one run uses a non-matched evaluation sample count",
    )
    require(set(runs["max_rows"].astype(int)) == {86400, 172800}, "Unexpected run duration")

    stratified = pd.read_csv(RESULTS / "jes2_stratified_run_metrics.csv")
    require(not runs["summary_path"].duplicated().any(), "Run summary paths are not unique")
    expected_by_path = runs.set_index("summary_path")["evaluation_samples"].astype(int)
    grouped = stratified.groupby(["summary_path", "dimension"])["n_samples"].sum()
    require(set(grouped.index.get_level_values(0)) == set(expected_by_path.index), "Strata omit runs")
    grouped_expected = grouped.index.get_level_values(0).map(expected_by_path).to_numpy(dtype=int)
    require(
        np.array_equal(grouped.to_numpy(dtype=int), grouped_expected),
        "At least one stratified dimension does not sum to the common evaluation count",
    )
    common_source = (SIMULATION / "robustness_common.py").read_text(encoding="utf-8")
    campaign_source = (SIMULATION / "run_jes2_benchmark.py").read_text(encoding="utf-8")
    require("COMMON_EVALUATION_START_SAMPLE = 2023" in common_source, "Common-mask constant changed")
    require(
        '"--evaluation_start_sample", str(args.evaluation_start_sample)' in campaign_source,
        "Campaign commands no longer receive the common-mask argument",
    )
    return f"6720 runs and {len(grouped)} run-by-dimension strata use the common mask"


def check_recovery() -> str:
    runs = pd.read_csv(RESULTS / "jes2_paired_initial_recovery_runs.csv")
    stats = pd.read_csv(RESULTS / "jes2_paired_initial_recovery_statistics.csv")
    require(len(runs) == 64, f"Expected 64 paired recovery rows, found {len(runs)}")
    require(set(runs["model"]) == MODELS, "Recovery rows do not contain all four models")
    require(set(runs["evaluation_start_sample"].astype(int)) == {2023}, "Recovery mask differs")
    require(set(runs["paired_samples"].astype(int)) == {84377}, "Recovery pairs are not sample matched")
    required_metrics = {
        "recovery_or_censor_time_h",
        "recovery_excess_auc_soc_h",
        "recovery_censored",
        "first_entry_left_censored",
        "recovery_relapsed_after_first_hold",
        "persistent_recovery_or_censor_time_h",
        "persistent_recovery_censored",
        "persistent_recovery_left_censored",
    }
    require(required_metrics.issubset(set(stats["metric"])), "Canonical recovery metrics are incomplete")
    require(set(stats["model"]) == MODELS, "Recovery statistics omit a model")
    require(set(stats["n_cells"].astype(int)) == {6}, "Recovery statistics are not six-cell estimates")
    require(set(stats["n_windows"].astype(int)) == {16}, "Recovery statistics are not based on 16 windows")
    require(
        ((stats["ci_low"] <= stats["mean"]) & (stats["mean"] <= stats["ci_high"])).all(),
        "A recovery estimate lies outside its confidence interval",
    )
    method = (RESULTS / "jes2_paired_initial_recovery_method.txt").read_text(encoding="utf-8")
    require("correctly initialized trajectory" in method, "Recovery reference is not documented")
    require("remains inside the band for the rest" in method, "Persistent endpoint is not documented")
    require("left-censored" in method, "Recovery observation truncation is not documented")
    require(
        np.allclose(runs["observation_start_time_h"], 2023 / 3600, atol=0.001),
        "Recovery observation start changed",
    )
    require(runs["first_entry_left_censored"].any(), "Expected left-censored first entries are absent")
    require(
        runs["persistent_recovery_left_censored"].any(),
        "Expected left-censored persistent endpoints are absent",
    )
    analysis_source = (SIMULATION / "results" / "analyze_jes2_paired_recovery.py").read_text(
        encoding="utf-8"
    )
    figure_source = (FIGURE_SCRIPTS / "build_figure_07_initial_recovery_corr.py").read_text(
        encoding="utf-8"
    )
    synthesis_source = (FIGURE_SCRIPTS / "build_revised_all_cells_figures.py").read_text(
        encoding="utf-8"
    )
    legacy_source = (SIMULATION / "results" / "build_jes2_trajectory_figures.py").read_text(
        encoding="utf-8"
    )
    require(
        analysis_source.count("compute_common_recovery_metrics(") == 1,
        "Canonical recovery analyzer does not have exactly one metric call",
    )
    canonical_file = "jes2_paired_initial_recovery_statistics.csv"
    require(canonical_file in figure_source, "Figure 09 does not read canonical recovery statistics")
    require(canonical_file in synthesis_source, "Decision synthesis does not read canonical recovery statistics")
    require("ci_low" in figure_source and "ci_high" in figure_source, "Figure 09 intervals are not traceable")
    require("def plot_initial" not in legacy_source, "Legacy aggregate recovery plotter is active")
    require(
        "common_recovery_or_censor_time_h" not in legacy_source,
        "Legacy aggregate recovery endpoint remains in the trajectory plotter",
    )
    return "64 matched trajectories feed one six-cell paired persistent-recovery analysis"


def check_robustness_score() -> str:
    penalties = pd.read_csv(RESULTS / "jes2_robustness_family_penalties.csv", index_col=0)
    sensitivity = pd.read_csv(RESULTS / "jes2_robustness_score_sensitivity.csv", index_col=0)
    require(list(penalties.columns) == FAMILIES, "Robustness score does not use the eight evaluated families")
    require(set(penalties.index) == MODELS, "Robustness family table omits a model")
    require(set(sensitivity.index) == MODELS, "Robustness sensitivity table omits a model")
    require(sensitivity.shape[1] == 3, "Expected three declared robustness aggregations")
    source = (FIGURE_SCRIPTS / "build_revised_all_cells_figures.py").read_text(encoding="utf-8")
    require('"Sensor offsets": ["voltage_offset", "temperature_offset"]' in source, "Offset family changed")
    require("for family, aliases in ROBUSTNESS_FAMILIES.items()" in source, "Family balancing is not applied")
    return "eight families, including sensor offsets, are reported under three aggregations"


def check_protocol_and_statistics() -> str:
    runs = pd.read_csv(RESULTS / "jes2_run_metrics.csv")
    require({"voltage_offset", "temperature_offset"}.issubset(set(runs["alias"])), "Offset cases are absent")
    tests = pd.read_csv(RESULTS / "jes2_paired_model_tests.csv")
    baseline = tests[(tests["alias"] == "baseline") & (tests["metric"] == "mae")]
    require(len(baseline) == 6, f"Expected six baseline model-pair tests, found {len(baseline)}")
    require(baseline["p_exact"].notna().all(), "Exact sign-flip values are incomplete")
    require(baseline["p_holm"].notna().all(), "Holm-adjusted values are incomplete")
    protocol = (SIMULATION / "jes2_protocol.py").read_text(encoding="utf-8")
    require('"--missing_gap_placement", "max_abs_net_charge"' in protocol, "Dropout placement changed")
    require('"--missing_gap_min_pre_seconds", "43200"' in protocol, "Dropout pre-gap context changed")
    require('"--missing_gap_min_post_seconds", "86400"' in protocol, "Dropout post-gap context changed")
    return "both offsets and all six exact baseline pair tests with Holm correction are present"


def check_build_boundaries() -> str:
    main_path = SIMULATION / "campaigns" / "jes2_full_holdout_merged_20260825.json"
    signed_path = (
        SIMULATION
        / "campaigns"
        / "jes2_signed_gain_common_mask_20260830"
        / "jes2_manifest.json"
    )
    main = json.loads(main_path.read_text(encoding="utf-8"))["runs"]
    signed = json.loads(signed_path.read_text(encoding="utf-8"))["runs"]
    main_gain_aliases = {
        row["alias"] for row in main if row.get("scenario") == "current_offset"
    }
    signed_gain_aliases = {
        row["alias"] for row in signed if row.get("scenario") == "current_offset"
    }
    require(
        main_gain_aliases
        == {
            "current_bias_0p5pct",
            "current_bias_1p5pct",
            "current_bias_3p0pct",
        },
        "Main current-gain aliases changed",
    )
    require(
        signed_gain_aliases
        == {
            "current_bias_neg_0p5pct",
            "current_bias_neg_1p5pct",
            "current_bias_neg_3p0pct",
        },
        "Signed extension aliases changed",
    )
    require(len(main) + len(signed) == 6912, "Final public benchmark build size changed")

    lifecycle = pd.read_csv(RESULTS / "c29_bias_temporal_model_summary.csv")
    minimum_model = lifecycle.loc[lifecycle["delta_mae_full_life"].idxmin(), "model"]
    require(minimum_model == "HECM", "C29 lifecycle minimum is no longer HECM")

    hardware = json.loads(
        (
            WORKSPACE
            / "STM32"
            / "JES2_hardware_benchmark"
            / "results"
            / "runtime_memory"
            / "runtime_memory_measurements.json"
        ).read_text(encoding="utf-8")
    )
    require(hardware["cell"] == "C09", "Runtime-memory replay cell changed")
    rolling = next(row for row in hardware["models"] if row["model"] == "DD")
    require(rolling["valid_inferences"] == 2, "Rolling-DD RAM call count changed")
    require("dirty" in rolling["firmware_revision"], "Expected release blocker is no longer present")
    return "6912 public evaluations, signed gain pairs, C29 lifecycle, dataset ground truth, and hardware boundaries are traceable"


def check_manuscript() -> str:
    text = MANUSCRIPT.read_text(encoding="utf-8")
    lower = text.lower()
    required = [
        "common source-sample mask $k\\geq2023$",
        "persistent recovery is the primary time endpoint",
        "equal weight to eight evaluated disturbance families",
        "largest absolute integrated net charge",
        "at least 12~h of pre-gap observations and 24~h of post-gap observations",
        "voltage offset, $+\\si{0.02}{v}$",
        "temperature offset, $+\\si{3}{\\celsius}$",
        "holm correction is applied across the six model-pair tests",
        "final benchmark build",
        "dataset soc ground truth",
        "left-censored",
        "originating hppc cells",
        "representative c09 replay",
        "not complete bms feasibility or schedulability",
    ]
    for phrase in required:
        require(phrase in lower, f"Required manuscript statement is missing: {phrase}")
    forbidden = [
        "current bias",
        "current-bias",
        "one central frozen gap",
        "complete measurement-only campaign",
        "19 predeclared cases",
        "post-campaign",
        "preregistered",
        "not an independent electrochemical ground-truth measurement",
        "do not establish absolute soc truth",
        "dd retains the smallest adverse penalty",
    ]
    for phrase in forbidden:
        require(phrase not in lower, f"Obsolete manuscript wording remains: {phrase}")
    require(";" not in text, "Manuscript still contains a semicolon")

    references = re.findall(r"\\finalfigpath/([^}]+\.png)", text)
    require(len(references) == 24, f"Expected 24 figure references, found {len(references)}")
    require(len(set(references)) == 24, "A final figure is referenced more than once")
    missing = [name for name in references if not (FIGURES / name).is_file()]
    require(not missing, f"Referenced figures are missing: {missing}")
    return "required claims are present, obsolete wording is absent, and 24 unique figures resolve"


def main() -> None:
    checks = [
        ("Common evaluation mask", check_common_mask),
        ("Canonical recovery", check_recovery),
        ("Robustness synthesis", check_robustness_score),
        ("Offsets and paired tests", check_protocol_and_statistics),
        ("Final-build boundaries", check_build_boundaries),
        ("Manuscript consistency", check_manuscript),
    ]
    for label, check in checks:
        print(f"PASS  {label}: {check()}")
    print("PASS  Independent audit verification completed without failures")


if __name__ == "__main__":
    main()
