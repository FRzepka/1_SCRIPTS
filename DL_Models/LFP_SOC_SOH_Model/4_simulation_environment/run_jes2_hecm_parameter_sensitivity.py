from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
RUNNER = ROOT / "ECM_0.0.3" / "run_ecm_scenario.py"
sys.path.insert(0, str(ROOT / "results"))
from jes2_plot_style import TU_RED, clean_axes, save_figure, setup_style

PERTURBATIONS = {
    "nominal": {},
    "resistance_minus_10pct": {"--ecm_resistance_scale": "0.90"},
    "resistance_plus_10pct": {"--ecm_resistance_scale": "1.10"},
    "tau_minus_20pct": {"--ecm_tau_scale": "0.80"},
    "tau_plus_20pct": {"--ecm_tau_scale": "1.20"},
    "ocv_minus_10mV": {"--ecm_ocv_offset_v": "-0.010"},
    "ocv_plus_10mV": {"--ecm_ocv_offset_v": "0.010"},
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main() -> None:
    parser = argparse.ArgumentParser(description="HECM lookup-parameter sensitivity outside the JES2 measurement-only ranking.")
    parser.add_argument("--campaign_manifest", type=Path, required=True)
    parser.add_argument("--data_root", default="/home/florianr/MG_Farm/0_Data/MGFarm_18650_FE")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--figures_dir", type=Path, default=None,
                        help="Optionally copy Figure 19 (PNG and PDF) into the paper figure directory.")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    args.campaign_manifest = args.campaign_manifest.resolve()

    campaign = json.loads(args.campaign_manifest.read_text(encoding="utf-8"))
    baseline = {}
    for record in campaign.get("runs", []):
        if (
            record.get("alias") == "baseline"
            and record.get("model") == "HECM"
            and record.get("soh_condition") == "lstm_h1"
        ):
            baseline[record["cell"]] = record
    missing = sorted(set(campaign["cells"]) - set(baseline))
    if missing:
        parser.error(f"Baseline lstm_h1 HECM traces missing for cells: {missing}")

    out_root = args.campaign_manifest.parent / "hecm_parameter_sensitivity"
    window = campaign.get("window", {})
    result = {
        "analysis": "HECM parameter sensitivity; excluded from primary measurement-only ranking",
        "started_utc": utc_now(),
        "source_campaign": str(args.campaign_manifest.resolve()),
        "perturbations": PERTURBATIONS,
        "runs": [],
    }
    for cell, source in baseline.items():
        for label, parameters in PERTURBATIONS.items():
            out_dir = out_root / cell / label
            summary_path = out_dir / "summary.json"
            record = {"cell": cell, "label": label, "out_dir": str(out_dir), "status": "running"}
            if args.skip_existing and summary_path.is_file():
                record["status"] = "skipped_existing"
            else:
                command = [
                    sys.executable,
                    str(RUNNER),
                    "--cell", cell,
                    "--scenario", "baseline",
                    "--seed", str(source["seed"]),
                    "--soh_trace", source["soh_trace"],
                    "--data_root", args.data_root,
                    "--device", args.device,
                    "--out_dir", str(out_dir),
                    "--start_row", str(int(window.get("start_row", 0))),
                    "--summary_only",
                ]
                if int(window.get("max_rows", 0)) > 0:
                    command.extend(["--max_rows", str(int(window["max_rows"]))])
                for key, value in parameters.items():
                    command.extend([key, value])
                record["command"] = command
                print("RUN", " ".join(command), flush=True)
                if not args.dry_run:
                    out_dir.mkdir(parents=True, exist_ok=True)
                    subprocess.run(command, cwd=WORKSPACE, check=True)
                record["status"] = "dry_run" if args.dry_run else "completed"
            result["runs"].append(record)
            out_root.mkdir(parents=True, exist_ok=True)
            (out_root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    result["finished_utc"] = utc_now()
    (out_root / "manifest.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if not args.dry_run:
        rows = []
        for record in result["runs"]:
            summary = json.loads((Path(record["out_dir"]) / "summary.json").read_text(encoding="utf-8"))
            rows.append({"cell": record["cell"], "label": record["label"], "mae": summary["mae"], "rmse": summary["rmse"]})
        metrics = pd.DataFrame(rows)
        nominal = metrics[metrics["label"] == "nominal"][["cell", "mae"]].rename(columns={"mae": "nominal_mae"})
        metrics = metrics.merge(nominal, on="cell", how="left")
        metrics["delta_mae"] = metrics["mae"] - metrics["nominal_mae"]
        metrics.to_csv(out_root / "hecm_parameter_sensitivity.csv", index=False)

        labels = [label for label in PERTURBATIONS if label != "nominal"]
        grouped = metrics[metrics["label"] != "nominal"].groupby("label")["delta_mae"]
        mean = grouped.mean().reindex(labels)
        std = grouped.std().fillna(0.0).reindex(labels)
        setup_style()
        fig, ax = plt.subplots(figsize=(8.6, 4.3))
        x = np.arange(len(labels))
        ax.bar(x, mean, yerr=std, capsize=3, color=TU_RED, alpha=0.9, edgecolor="#333333")
        ax.axhline(0.0, color="#444444", linewidth=0.8)
        ax.set_xticks(x, [label.replace("_", " ") for label in labels], rotation=20, ha="right")
        ax.set_ylabel(r"$\Delta$MAE versus nominal HECM [SOC]")
        ax.set_title("HECM lookup-parameter sensitivity (mean +/- cell SD)")
        clean_axes(ax)
        fig.tight_layout()
        figure_path = out_root / "Figure_19_HECM_Parameter_Sensitivity.png"
        save_figure(fig, figure_path)
        if args.figures_dir is not None:
            args.figures_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(figure_path, args.figures_dir / figure_path.name)
            shutil.copy2(figure_path.with_suffix(".pdf"), args.figures_dir / figure_path.with_suffix(".pdf").name)
    print(json.dumps({"manifest": str(out_root / "manifest.json"), "runs": len(result["runs"])}, indent=2))


if __name__ == "__main__":
    main()
