from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_jes2_hecm_full_lookup_sensitivity import (
    BASELINE_ALIAS,
    EXPECTED_DISTURBANCE_ALIASES,
    FAMILY_ALIASES,
    FULL_OUTPUT_ALIASES,
    LOOKUP_CONDITIONS,
    RECOVERY_BASELINE_ALIAS,
    ROBUSTNESS_FAMILIES,
    build_command,
    load_sources,
)


def test_full_lookup_protocol_covers_parameters_and_disturbance_subcases(tmp_path):
    assert len(LOOKUP_CONDITIONS) == 7
    assert LOOKUP_CONDITIONS["resistance_minus_10pct"] == {
        "--ecm_resistance_scale": "0.90"
    }
    assert LOOKUP_CONDITIONS["tau_plus_10pct"] == {"--ecm_tau_scale": "1.10"}
    assert LOOKUP_CONDITIONS["ocv_minus_10mV"] == {"--ecm_ocv_offset_v": "-0.010"}
    assert len(EXPECTED_DISTURBANCE_ALIASES) == 21
    assert EXPECTED_DISTURBANCE_ALIASES == set().union(*FAMILY_ALIASES.values())
    assert len(set().union(*ROBUSTNESS_FAMILIES.values())) == 20
    assert "Initialization MAE" not in ROBUSTNESS_FAMILIES

    source = {
        "alias": "current_noise_high",
        "command": [
            "/old/python",
            "/runner.py",
            "--device",
            "cuda",
            "--require_gpu",
            "--out_dir",
            "/old/output",
            "--summary_only",
        ],
    }
    command = build_command(
        source,
        tmp_path,
        LOOKUP_CONDITIONS["tau_plus_10pct"],
        evaluation_start_sample=2023,
        device="cpu",
    )
    assert "--require_gpu" not in command
    assert command.count("--summary_only") == 1
    assert command[command.index("--ecm_tau_scale") + 1] == "1.10"
    assert command[command.index("--evaluation_start_sample") + 1] == "2023"


def test_main_and_recovery_baselines_remain_separate():
    campaign = ROOT / "campaigns"
    sources = load_sources(
        campaign / "jes2_full_holdout_merged_20260825.json",
        campaign / "jes2_signed_gain_common_mask_20260830" / "jes2_manifest.json",
        campaign / "jes2_gap_baseline_common_mask_20260830" / "jes2_manifest.json",
        campaign / "jes2_initial_state_paired_sixcell_20260827_cuda" / "jes2_manifest.json",
    )
    aliases = [row["alias"] for row in sources]

    assert len(sources) == 1280
    assert aliases.count(BASELINE_ALIAS) == 16
    assert aliases.count(RECOVERY_BASELINE_ALIAS) == 16
    assert BASELINE_ALIAS not in FULL_OUTPUT_ALIASES
    assert RECOVERY_BASELINE_ALIAS in FULL_OUTPUT_ALIASES
