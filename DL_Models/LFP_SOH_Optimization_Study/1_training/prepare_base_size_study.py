#!/usr/bin/env python3
import argparse
import json
import shutil
from pathlib import Path

import yaml


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
STUDY_ROOT = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study"
SPECS_PATH = STUDY_ROOT / "base_size_study_specs.json"
TRAIN_ROOT = STUDY_ROOT / "1_training"


def load_specs() -> dict:
    with open(SPECS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def rewrite_out_root(cfg: dict, version: str) -> None:
    out_root = f"${{OUT_ROOT:-{TRAIN_ROOT / version / 'outputs' / 'soh'}}}"
    cfg.setdefault("paths", {})["out_root"] = str(out_root)
    tracking = cfg.setdefault("tracking", {})
    if tracking.get("csv_file") is not None:
        tracking["csv_file"] = f"{out_root}/training_log.csv"


def copy_variant_tree(src_dir: Path, dst_dir: Path, force: bool) -> None:
    if dst_dir.exists():
        if not force:
            return
        shutil.rmtree(dst_dir)
    for sub in ("config", "scripts", "test"):
        src = src_dir / sub
        if src.exists():
            shutil.copytree(src, dst_dir / sub)


def prepare_variant(family_spec: dict, variant: dict, force: bool) -> dict:
    src_dir = ROOT / family_spec["base_model_dir"]
    version = variant["version"]
    dst_dir = TRAIN_ROOT / version

    copy_variant_tree(src_dir, dst_dir, force=force)

    cfg_path = dst_dir / "config" / "train_soh.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Missing config after copy: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    rewrite_out_root(cfg, version)
    cfg.setdefault("study_variant", {})
    cfg["study_variant"]["family"] = family_spec["family"]
    cfg["study_variant"]["role"] = variant["role"]
    cfg["study_variant"]["tag"] = variant["tag"]
    cfg["study_variant"]["source_model_dir"] = family_spec["base_model_dir"]

    model_cfg = cfg.setdefault("model", {})
    for key, value in variant["model_overrides"].items():
        model_cfg[key] = value

    with open(cfg_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    meta = {
        "family": family_spec["family"],
        "version": version,
        "tag": variant["tag"],
        "role": variant["role"],
        "source_model_dir": family_spec["base_model_dir"],
        "training_dir": str(dst_dir.relative_to(ROOT)),
        "model_overrides": variant["model_overrides"],
    }
    with open(dst_dir / "size_study_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare SOH base-size-study training folders.")
    ap.add_argument("--force", action="store_true", help="Recreate existing variant folders.")
    args = ap.parse_args()

    specs = load_specs()
    manifest = {
        "specs_path": str(SPECS_PATH.relative_to(ROOT)),
        "test_cell": specs["test_cell"],
        "variants": [],
    }

    for family_spec in specs["families"]:
        for variant in family_spec["variants"]:
            if variant["role"] == "base":
                continue
            meta = prepare_variant(family_spec, variant, force=args.force)
            manifest["variants"].append(meta)

    out_dir = TRAIN_ROOT / "base_size_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "prepared_variants.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Prepared {len(manifest['variants'])} variants.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
