#!/usr/bin/env python3
"""Integrate the completed current-offset extension into paper-facing outputs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PAPER = Path(__file__).resolve().parents[1]
WORKSPACE = Path(__file__).resolve().parents[4]
RESULTS = PAPER / "JES_2.0" / "results"
OFFSET_RESULTS = RESULTS / "current_offset_extension"
TABLE = PAPER / "JES_2.0" / "tables" / "jes2_hecm_lookup_sensitivity_compact.tex"
FIGURE = (
    PAPER
    / "figures"
    / "Results"
    / "All Cells"
    / "Figure_25_APPENDIX_HECM_Lookup_Table_Sensitivity.png"
)
SIMULATION = (
    WORKSPACE
    / "DL_Models"
    / "LFP_SOC_SOH_Model"
    / "4_simulation_environment"
)

LOOKUP_LABELS = {
    "nominal_lookup": "Nominal lookup",
    "resistance_minus_10pct": r"Resistance $-10\%$",
    "resistance_plus_10pct": r"Resistance $+10\%$",
    "tau_minus_10pct": r"Time constants $-10\%$",
    "tau_plus_10pct": r"Time constants $+10\%$",
    "ocv_minus_10mV": r"OCV $-10$ mV",
    "ocv_plus_10mV": r"OCV $+10$ mV",
}

SOURCE_LABELS = {
    "current_offset_neg_50mA": r"Current offset $-50$ mA",
    "current_offset_pos_50mA": r"Current offset $+50$ mA",
}


def update_family_heatmap(offset: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS / "hecm_full_lookup_family_heatmap.csv"
    family = pd.read_csv(path)
    family["family"] = family["family"].replace(
        {
            "Sensor noise": "Random sensor noise",
            "Sensor offsets": "Additive sensor offsets",
        }
    )

    for lookup_condition, part in offset.groupby("lookup_condition"):
        if lookup_condition == "nominal_lookup":
            continue
        selected = part.loc[part["interaction_mae"].abs().idxmax()]
        mask = (
            family["family"].eq("Additive sensor offsets")
            & family["lookup_condition"].eq(lookup_condition)
        )
        if mask.sum() != 1:
            raise ValueError(f"Missing additive-offset family row for {lookup_condition}")
        current_value = float(family.loc[mask, "interaction_mean"].iloc[0])
        if abs(float(selected["interaction_mae"])) > abs(current_value):
            family.loc[mask, "interaction_mean"] = float(selected["interaction_mae"])
            family.loc[mask, "source_alias"] = selected["alias"]

    family.to_csv(path, index=False)
    return family


def update_summary(offset: pd.DataFrame) -> pd.DataFrame:
    path = RESULTS / "hecm_full_lookup_summary.csv"
    summary = pd.read_csv(path)
    for index, row in summary.iterrows():
        lookup_condition = row["lookup_condition"]
        if lookup_condition == "nominal_lookup":
            summary.loc[index, "scenario_count"] = 22
            continue

        part = offset[offset["lookup_condition"].eq(lookup_condition)]
        if len(part) != 2:
            raise ValueError(f"Expected two signed offset rows for {lookup_condition}")
        selected = part.loc[part["interaction_mae"].abs().idxmax()]
        if abs(float(selected["interaction_mae"])) > float(row["worst_absolute_interaction_mae"]):
            summary.loc[index, "worst_interaction_alias"] = selected["alias"]
            summary.loc[index, "worst_interaction_label"] = SOURCE_LABELS[selected["alias"]]
            summary.loc[index, "worst_absolute_interaction_mae"] = abs(
                float(selected["interaction_mae"])
            )

        if int(row["scenario_count"]) < 22:
            includes_zero = (
                part["interaction_ci_includes_zero"]
                .astype(str)
                .str.strip()
                .str.lower()
                .eq("true")
            )
            added_exclusions = int((~includes_zero).sum())
            summary.loc[index, "scenarios_ci_excluding_zero"] = int(
                row["scenarios_ci_excluding_zero"]
            ) + added_exclusions
        summary.loc[index, "scenario_count"] = 22

    summary.to_csv(path, index=False)
    return summary


def update_protocol() -> None:
    path = RESULTS / "hecm_full_lookup_protocol.json"
    protocol = json.loads(path.read_text(encoding="utf-8"))
    aliases = set(protocol["disturbance_aliases"])
    aliases.update({"current_offset_neg_50mA", "current_offset_pos_50mA"})
    protocol["disturbance_aliases"] = sorted(aliases)

    families = protocol["robustness_families"]
    noise = families.pop("Sensor noise", families.get("Random sensor noise", []))
    offsets = families.pop("Sensor offsets", families.get("Additive sensor offsets", []))
    offsets = sorted(set(offsets) | {"current_offset_neg_50mA", "current_offset_pos_50mA"})
    ordered = {
        "Current gain": families["Current gain"],
        "Random sensor noise": noise,
        "Additive sensor offsets": offsets,
        "ADC quantization": families["ADC quantization"],
        "Missing samples": families["Missing samples"],
        "Timing jitter": families["Timing jitter"],
        "Burst dropout": families["Burst dropout"],
        "Voltage spikes": families["Voltage spikes"],
    }
    protocol["robustness_families"] = ordered
    protocol["base_runs_before_current_offset"] = 8960
    protocol["current_offset_extension_runs"] = 224
    protocol["runs"] = 9184
    protocol["interpretation_boundary"] = (
        "The analysis covers all declared JES2 disturbance subcases for the fixed HECM "
        "implementation under local one-at-a-time lookup perturbations. The additive "
        "current-offset extension is combined with the original full sensitivity campaign. "
        "Combined lookup errors and other HECM structures are not covered."
    )
    path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")


def write_table(summary: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"    \centering",
        r"    \footnotesize",
        r"    \setlength{\tabcolsep}{4pt}",
        r"    \caption{HECM lookup sensitivity across the complete disturbance benchmark. The largest robustness interaction is selected from 22 measurement and signal-integrity subcases after equal-weight cell aggregation. Bracketed counts report subcases whose 95\% interaction interval excludes zero. Recovery is the persistent recovery-or-censor time from the paired initialization analysis, with the equal-weight censored-cell percentage in brackets.}",
        r"    \label{tab:hecm_lookup_sensitivity}",
        r"    \begin{tabularx}{0.99\textwidth}{@{}l>{\centering\arraybackslash}p{0.12\textwidth}>{\centering\arraybackslash}p{0.19\textwidth}Y>{\centering\arraybackslash}p{0.17\textwidth}@{}}",
        r"        \toprule",
        r"        Lookup condition & Baseline MAE & Max. robust $|\Delta\Delta\mathrm{MAE}|$ [CI] & Source subcase & Recovery/censor [h] [censored] \\",
        r"        \midrule",
    ]
    for row in summary.itertuples(index=False):
        if row.lookup_condition == "nominal_lookup":
            worst = "Reference"
            count = "--"
            source = "--"
        else:
            worst = f"{row.worst_absolute_interaction_mae:.5f}"
            count = f"{int(row.scenarios_ci_excluding_zero)}/{int(row.scenario_count)}"
            source = str(row.worst_interaction_label).replace("%", r"\%")
        lines.append(
            "        "
            f"{LOOKUP_LABELS[row.lookup_condition]} & {row.baseline_mae:.4f} & "
            f"{worst} [{count}] & {source} & {row.recovery_h:.2f} "
            f"[{100.0 * row.recovery_censored_fraction:.0f}\\%] "
            + r"\\"
        )
    lines.extend([r"        \bottomrule", r"    \end{tabularx}", r"\end{table*}", ""])
    TABLE.write_text("\n".join(lines), encoding="utf-8")


def plot_figure(family: pd.DataFrame) -> None:
    sys.path.insert(0, str(SIMULATION))
    import run_jes2_hecm_full_lookup_sensitivity as lookup

    lookup.ROBUSTNESS_FAMILIES = {
        "Current gain": [],
        "Random sensor noise": [],
        "Additive sensor offsets": [],
        "ADC quantization": [],
        "Missing samples": [],
        "Timing jitter": [],
        "Burst dropout": [],
        "Voltage spikes": [],
    }
    recovery_runs = pd.read_csv(RESULTS / "hecm_full_lookup_recovery_runs.csv")
    recovery_stats = pd.read_csv(RESULTS / "hecm_full_lookup_recovery_statistics.csv")
    lookup.plot_results(family, recovery_runs, recovery_stats, FIGURE)


def main() -> None:
    offset = pd.read_csv(
        OFFSET_RESULTS / "hecm_lookup_current_offset_statistics.csv"
    )
    family = update_family_heatmap(offset)
    summary = update_summary(offset)
    update_protocol()
    write_table(summary)
    plot_figure(family)
    print(
        json.dumps(
            {
                "figure": str(FIGURE),
                "table": str(TABLE),
                "maximum_lookup_interaction": float(
                    summary["worst_absolute_interaction_mae"].max()
                ),
                "sensitivity_runs": 9184,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
