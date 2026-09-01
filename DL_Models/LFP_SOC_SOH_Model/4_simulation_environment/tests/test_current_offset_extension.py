from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_jes2_hecm_lookup_current_offset import ALIASES, build_records


def test_hecm_lookup_current_offset_build_is_complete_and_absolute(tmp_path):
    cells = ["C09", "C13", "C15", "C25", "C27", "C29"]
    windows = [f"W{index:02d}" for index in range(16)]
    sources = []
    for index, window in enumerate(windows):
        cell = cells[index % len(cells)]
        for alias, offset in (
            ("current_offset_neg_50mA", "-0.050"),
            ("current_offset_pos_50mA", "0.050"),
        ):
            sources.append(
                {
                    "cell": cell,
                    "window_id": window,
                    "soh_state": "fresh",
                    "cell_load_class": "middle",
                    "alias": alias,
                    "command": [
                        "/old/python",
                        "/runner.py",
                        "--device",
                        "cuda",
                        "--out_dir",
                        "/old/output",
                        "--current_offset_a",
                        offset,
                        "--summary_only",
                    ],
                }
            )

    records = build_records(sources, tmp_path.resolve(), 2023, "cpu")

    assert len(ALIASES) == 2
    assert len(records) == 224
    assert all(Path(record["out_dir"]).is_absolute() for record in records)
    assert all(
        record["command"][record["command"].index("--evaluation_start_sample") + 1]
        == "2023"
        for record in records
    )
    assert {
        record["command"][record["command"].index("--current_offset_a") + 1]
        for record in records
    } == {"-0.050", "0.050"}
