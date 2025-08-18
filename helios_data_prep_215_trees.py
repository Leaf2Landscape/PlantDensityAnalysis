#!/usr/bin/env python3
import os
import csv
import shutil
from pathlib import Path
import pandas as pd
import utils

# ========= EDIT THIS TO MATCH WHERE YOUR SCENES LIVE =========
# EITHER the /scratch/project root OR your /home/... root. Use ONE.
BASE_DIR = Path("/home/uqrarya1/Desktop/Scratch/veg3d/uqrarya1/phd_work/51_tree")
# BASE_DIR = Path("/home/uqrarya1/Desktop/Scratch/veg3d/uqrarya1/phd_work/51_tree")
# =============================================================

TREE_LIST_CSV = Path("/home/uqrarya1/Desktop/Scratch/veg3d/uqrarya1/phd_work/51_tree/tree_leaf_sizes_with_species_height.csv")
TREE_ID_COLUMN = "tree_id"
VOXEL_SIZES = ["0.2", "0.5", "1.0", "2.0"]
LEAF_OBJECT_IDS = [1]
WOOD_OBJECT_IDS = [0]

def load_tree_ids(csv_path: Path, col: str) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Tree list CSV not found: {csv_path}")
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in {csv_path}. Columns: {list(df.columns)}")
    ids = []
    for v in df[col].dropna().unique().tolist():
        s = str(v).strip()
        if s.endswith(".0"):
            s = s[:-2]
        ids.append(s)
    return ids

def resolve_scene_dir(base_dir: Path, tid: str) -> Path | None:
    exact = base_dir / tid
    if exact.is_dir():
        return exact
    candidates = [p for p in base_dir.glob(f"{tid}_*") if p.is_dir()]
    if not candidates:
        return None
    diamond = [p for p in candidates if p.name.endswith("_diamond")]
    if len(diamond) == 1:
        return diamond[0]
    if diamond:
        return sorted(diamond, key=lambda p: len(p.name))[0]
    return sorted(candidates, key=lambda p: len(p.name))[0]

def sniff_dialect(sample: str) -> csv.Dialect:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", ";", "\t", "|"])
    except Exception:
        return csv.excel

def normalize_header_to_voxel_cols(path: Path):
    """
    Ensure header has voxel_cx, voxel_cy, voxel_cz.
    If vox_* exists, rename to voxel_* in HEADER ONLY.
    """
    with open(path, "r", newline="") as f:
        sample = f.read(4096)
        f.seek(0)
        dialect = sniff_dialect(sample)
        reader = csv.reader(f, dialect)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError(f"Empty CSV (no header): {path}")

    cols = set(header)
    need = {"voxel_cx", "voxel_cy", "voxel_cz"}
    if need.issubset(cols):
        return
    if {"vox_cx", "vox_cy", "vox_cz"}.issubset(cols):
        rename_map = {"vox_cx":"voxel_cx","vox_cy":"voxel_cy","vox_cz":"voxel_cz"}
        new_header = [rename_map.get(c, c) for c in header]
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(path, "r", newline="") as src, open(tmp, "w", newline="") as dst:
            reader = csv.reader(src, dialect)
            writer = csv.writer(dst, dialect)
            next(reader, None)
            writer.writerow(new_header)
            for row in reader:
                writer.writerow(row)
        os.replace(tmp, path)
        print(f"[HEADER] vox_*  voxel_* in {path}")
        return
    raise ValueError(f"{path}: required columns not found. Header starts: {header[:10]}")

def rename_results_to_voxel_size(refs_dir: Path, scene_id: str):
    """
    Force filenames to the legacy pattern utils expects:
      <scene_id>_grouped_voxel_size_<vs>.csv
    by RENAMING any existing:
      <scene_id>_grouped_results_<vs>.csv
    Then normalize header to voxel_*.
    """
    for vs in VOXEL_SIZES:
        src = refs_dir / f"{scene_id}_grouped_results_{vs}.csv"
        dst = refs_dir / f"{scene_id}_grouped_voxel_size_{vs}.csv"

        # If only src exists -> rename to dst
        if src.exists() and not dst.exists():
            os.replace(src, dst)
            print(f"[RENAME] {src.name} -> {dst.name}")

        # If both exist -> remove the results* (it makes utils choke)
        if src.exists() and dst.exists():
            # keep dst (correct), remove src (incorrect name)
            src.unlink()
            print(f"[CLEAN ] removed {src.name} (duplicate wrong pattern)")

        # If dst exists, fix header
        if dst.exists():
            normalize_header_to_voxel_cols(dst)

def main():
    # SAFETY: show where we are looking
    print(f"[INFO] Using BASE_DIR = {BASE_DIR}")

    try:
        tree_ids = load_tree_ids(TREE_LIST_CSV, TREE_ID_COLUMN)
    except Exception as e:
        print(f"[FATAL] Could not load tree IDs: {e}")
        return
    print(f"[INFO] Loaded {len(tree_ids)} tree IDs from {TREE_LIST_CSV}")

    for tid in tree_ids:
        scene_dir = resolve_scene_dir(BASE_DIR, tid)
        if scene_dir is None:
            print(f"[SKIP] {tid}: no matching scene directory under {BASE_DIR}")
            continue

        scene_id = scene_dir.name
        helios_dir = scene_dir / "helios"
        refs_dir = scene_dir / "references"
        valid_rays_dir = scene_dir / "valid_rays"

        if not helios_dir.is_dir():
            print(f"[SKIP] {scene_id}: missing helios folder at {helios_dir}")
            continue
        if not refs_dir.is_dir():
            print(f"[SKIP] {scene_id}: missing references folder at {refs_dir}")
            continue

        # 1) Rename filenames so utils can parse voxel size; fix headers
        try:
            rename_results_to_voxel_size(refs_dir, scene_id)
        except Exception as e:
            print(f"[SKIP] {scene_id}: normalize failed: {e}")
            continue

        # 2) Ensure output folder
        valid_rays_dir.mkdir(parents=True, exist_ok=True)

        # 3) Run
        print(f"[RUNNING] prepare_helios_data for scene {scene_id}")
        try:
            utils.prepare_helios_data(
                input_dir=str(helios_dir),
                output_dir=str(valid_rays_dir),
                references_dir=str(refs_dir),
                leaf_object_ids=LEAF_OBJECT_IDS,   # e.g. [1]
                wood_object_ids=WOOD_OBJECT_IDS,   # e.g. [0]
                use_class=True,        # <- set True if IDs are in `class` (common for HELIOS)
            )

            print(f"[OK]     {scene_id}\n")
        except Exception as e:
            print(f"[ERROR]  {scene_id}: {e}\n")
            raise

        # Add Normals and Point Weighting to valid_rays
        try:
            import glob
            valid_rays_files = glob.glob(str(valid_rays_dir / "*valid_rays.parquet"))
            utils.add_normals_weights_to_valid_rays(
                valid_rays_files,
                knn=6
            )
            print(f"[OK] Normals and weights added to valid rays for {scene_id}")
        except Exception as e:
            print(f"[ERROR]  {scene_id}: {e}\n")

if __name__ == "__main__":
    main()
