#!/usr/bin/env python3
import argparse
import json
import re
import shutil
from pathlib import Path


ROOT = Path("/home/florianr/MG_Farm/1_Scripts")
STUDY_ROOT = ROOT / "DL_Models" / "LFP_SOH_Optimization_Study"
SPECS_PATH = STUDY_ROOT / "base_size_study_specs.json"
TRAIN_ROOT = STUDY_ROOT / "1_training"
MODELS_ROOT = STUDY_ROOT / "2_models"


def load_specs() -> dict:
    with open(SPECS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def pick_best_ckpt(run_dir: Path) -> Path:
    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Missing checkpoints dir: {ckpt_dir}")

    best = list(ckpt_dir.glob("best_epoch*_rmse*.pt"))
    if best:
        def score(path: Path) -> float:
            match = re.search(r"rmse([0-9]+(?:\.[0-9]+)?)", path.name)
            return float(match.group(1)) if match else float("inf")

        best.sort(key=score)
        return best[0]

    for name in ("best_model.pt", "final_model.pt"):
        cand = ckpt_dir / name
        if cand.exists():
            return cand

    all_pts = sorted(ckpt_dir.glob("*.pt"))
    if not all_pts:
        raise FileNotFoundError(f"No checkpoint found in {ckpt_dir}")
    return all_pts[0]


def copy_tree(src: Path, dst: Path, force: bool) -> None:
    if dst.exists():
        if not force:
            return
        shutil.rmtree(dst)
    shutil.copytree(src, dst)


def copy_base_alias(family_spec: dict, variant: dict, force: bool) -> dict:
    src = ROOT / family_spec["base_model_dir"]
    dst = MODELS_ROOT / family_spec["family"] / "Base" / f"{variant['version']}_{variant['tag']}"
    copy_tree(src, dst, force=force)
    meta = {
        "family": family_spec["family"],
        "role": variant["role"],
        "version": variant["version"],
        "tag": variant["tag"],
        "model_dir": str(dst.relative_to(ROOT)),
        "source": str(src.relative_to(ROOT)),
    }
    with open(dst / "size_study_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    return meta


def copy_trained_variant(family_spec: dict, variant: dict, ts: str, force: bool) -> dict:
    version = variant["version"]
    tag = variant["tag"]
    train_dir = TRAIN_ROOT / version
    dst = MODELS_ROOT / family_spec["family"] / "Base" / f"{version}_{tag}"
    if dst.exists():
        if force:
            shutil.rmtree(dst)
        else:
            run_dir = train_dir / "outputs" / "soh" / f"size_{ts}_{tag}"
            best_ckpt = run_dir / "checkpoints" / "best_model.pt"
            return {
                "family": family_spec["family"],
                "role": variant["role"],
                "version": version,
                "tag": tag,
                "model_dir": str(dst.relative_to(ROOT)),
                "source_run": str(run_dir.relative_to(ROOT)),
                "best_ckpt": str(best_ckpt.relative_to(ROOT)),
                "skipped_existing": True,
            }

    run_dir = train_dir / "outputs" / "soh" / f"size_{ts}_{tag}"
    if not run_dir.exists():
        raise FileNotFoundError(f"Missing run dir: {run_dir}")

    best_ckpt = pick_best_ckpt(run_dir)
    scaler_path = run_dir / "scaler_robust.joblib"
    if not scaler_path.exists():
        raise FileNotFoundError(f"Missing scaler: {scaler_path}")

    (dst / "checkpoints").mkdir(parents=True, exist_ok=True)
    for sub in ("config", "scripts", "test"):
        src_sub = train_dir / sub
        if src_sub.exists():
            shutil.copytree(src_sub, dst / sub)

    shutil.copy2(scaler_path, dst / "scaler_robust.joblib")
    shutil.copy2(best_ckpt, dst / "checkpoints" / best_ckpt.name)
    shutil.copy2(best_ckpt, dst / "checkpoints" / "best_model.pt")

    for name in ("training_log.csv",):
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dst / name)

    meta = {
        "family": family_spec["family"],
        "role": variant["role"],
        "version": version,
        "tag": tag,
        "model_dir": str(dst.relative_to(ROOT)),
        "source_run": str(run_dir.relative_to(ROOT)),
        "best_ckpt": str(best_ckpt.relative_to(ROOT)),
        "model_overrides": variant["model_overrides"],
        "screen_timestamp": ts,
    }
    with open(dst / "size_study_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    with open(dst / "source_run.txt", "w", encoding="utf-8") as f:
        for key, value in meta.items():
            f.write(f"{key}={value}\n")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Copy trained SOH base-size-study models into 2_models.")
    ap.add_argument("--ts", required=True, help="Training screen timestamp from start_base_size_study_screens.sh")
    ap.add_argument("--force", action="store_true", help="Overwrite existing copied model folders.")
    args = ap.parse_args()

    specs = load_specs()
    copied = []
    for family_spec in specs["families"]:
        for variant in family_spec["variants"]:
            if variant["role"] == "base":
                copied.append(copy_base_alias(family_spec, variant, force=args.force))
            else:
                copied.append(copy_trained_variant(family_spec, variant, ts=args.ts, force=args.force))

    out_dir = TRAIN_ROOT / "base_size_study"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / f"copied_models_{args.ts}.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump({"screen_timestamp": args.ts, "models": copied}, f, indent=2)

    print(f"Copied {len(copied)} models.")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
