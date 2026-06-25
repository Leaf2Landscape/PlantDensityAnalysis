#!/usr/bin/env python
"""
run_helios_analysis.py

Script ("run all") version of helios_analysis.ipynb.

Runs the complete workflow for analysing HELIOS simulation outputs against
reference voxel maps:

    Step 1  Configure paths / object IDs and generate a test classification
            image so the leaf/wood settings can be verified.
    Step 2  Extract and filter valid rays from the HELIOS data.
    Step 3  Compute ray-voxel intersections.
    Step 4  Calculate optical metrics and merge with reference data.
    (opt.)  Convert intersection Parquet files to CSV for inspection.

All inputs from the notebook's first code block (plus the analysis parameters
that were hard-coded further down) are exposed through argparse.

Use --test to stop right after the Step 1 test image is written, so the
leaf/wood classification can be checked before any heavy processing runs.
"""

import argparse
import glob
import os
import sys

import pandas as pd

from utils import (
    test_helios_settings,
    prepare_helios_data,
    voxel_ray_intersections,
    get_voxel_metrics,
    convert_parquet_to_csv,
)


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #
def _all_or_int_list(values):
    """Interpret an nargs list as the literal string 'all' or a list of ints."""
    if values is None:
        return None
    if len(values) == 1 and str(values[0]).lower() == "all":
        return "all"
    return [int(v) for v in values]


def _all_or_float_list(values):
    """Interpret an nargs list as the literal string 'all' or a list of floats."""
    if values is None:
        return None
    if len(values) == 1 and str(values[0]).lower() == "all":
        return "all"
    return [float(v) for v in values]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Run the full HELIOS analysis pipeline (script version of "
            "helios_analysis.ipynb)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Paths ------------------------------------------------------------- #
    parser.add_argument(
        "project_dir",
        help="Base directory for the analysis. Root for all subdirectories "
        "and used to name output files.",
    )
    parser.add_argument(
        "--helios_dir",
        default=None,
        help="HELIOS simulation output directory. Default: <project_dir>/helios",
    )
    parser.add_argument(
        "--references_dir",
        default=None,
        help="Reference voxel-map directory. Default: <project_dir>/references",
    )
    parser.add_argument(
        "--results_dir",
        default=None,
        help="Output directory for final merged CSVs. Default: <project_dir>/results",
    )
    parser.add_argument(
        "--valid_rays_dir",
        default=None,
        help="Working directory for intermediate Parquet files. "
        "Default: <project_dir>/valid_rays",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Project name stub used to prefix output CSV filenames. "
        "Default: basename of <project_dir>.",
    )

    # --- Object classification -------------------------------------------- #
    parser.add_argument(
        "--use_class",
        action="store_true",
        help="Use column 9 (class) instead of column 8 (hitObject_id) to "
        "classify leaf vs wood.",
    )
    parser.add_argument(
        "--leaf_object_ids",
        type=int,
        nargs="+",
        default=[0],
        help="Object/class IDs representing leaf geometry.",
    )
    parser.add_argument(
        "--wood_object_ids",
        type=int,
        nargs="+",
        default=[1],
        help="Object/class IDs representing wood/stem geometry.",
    )

    # --- Metrics selection (Step 4) --------------------------------------- #
    parser.add_argument(
        "--legs",
        nargs="+",
        default=["all"],
        help="Legs to include: 'all' or a list of integer leg IDs (e.g. 0 1 2).",
    )
    parser.add_argument(
        "--voxel_sizes",
        nargs="+",
        default=["all"],
        help="Voxel sizes to include: 'all' or a list of floats (e.g. 0.2 0.5 1.0 2.0).",
    )
    parser.add_argument(
        "--average_leaf_area",
        type=float,
        default=0.003582,
        help="Average leaf area (m^2) used for the lambda calculation.",
    )
    parser.add_argument(
        "--is_multireturn",
        action="store_true",
        help="Treat the data as multi-return when computing voxel metrics.",
    )

    # --- Optional Parquet -> CSV conversion (notebook utility cell) -------- #
    parser.add_argument(
        "--convert_csv",
        action="store_true",
        help="After analysis, convert intersection Parquet files to CSV for "
        "inspection (notebook utility cell).",
    )
    parser.add_argument(
        "--convert_csv_voxel_sizes",
        type=float,
        nargs="+",
        default=[2.0],
        help="Voxel sizes whose intersection Parquet files are converted to CSV "
        "when --convert_csv is set.",
    )

    # --- Behaviour flags -------------------------------------------------- #
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging in the pipeline stages.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Generate the Step 1 test classification image, then exit before "
        "any processing. Use to verify the configuration.",
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def step1_configure_and_test(args):
    """Resolve paths, validate inputs, and write the test classification image."""
    project_dir = args.project_dir
    helios_dir = args.helios_dir or os.path.join(project_dir, "helios")
    references_dir = args.references_dir or os.path.join(project_dir, "references")
    results_dir = args.results_dir or os.path.join(project_dir, "results")
    valid_rays_dir = args.valid_rays_dir or os.path.join(project_dir, "valid_rays")

    if not os.path.exists(helios_dir) or not os.path.exists(references_dir):
        raise FileNotFoundError(
            "The specified directories do not exist. Please check the paths "
            f"(helios_dir={helios_dir!r}, references_dir={references_dir!r})."
        )

    os.makedirs(valid_rays_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print("  Step 1 - Configuration")
    print("=" * 80)
    print(f"    project_dir:     {project_dir}")
    print(f"    helios_dir:      {helios_dir}")
    print(f"    references_dir:  {references_dir}")
    print(f"    results_dir:     {results_dir}")
    print(f"    valid_rays_dir:  {valid_rays_dir}")
    print(f"    use_class:       {args.use_class}")
    print(f"    leaf_object_ids: {args.leaf_object_ids}")
    print(f"    wood_object_ids: {args.wood_object_ids}")

    test_helios_settings(
        helios_dir=helios_dir,
        use_class=args.use_class,
        leaf_object_ids=args.leaf_object_ids,
        wood_object_ids=args.wood_object_ids,
        output_dir=project_dir,
    )

    return {
        "project_dir": project_dir,
        "helios_dir": helios_dir,
        "references_dir": references_dir,
        "results_dir": results_dir,
        "valid_rays_dir": valid_rays_dir,
    }


def step2_prepare_rays(args, paths):
    """Extract and filter valid rays from the HELIOS data (notebook Step 2)."""
    print("\n" + "=" * 80)
    print("  Step 2 - Extract and filter valid rays")
    print("=" * 80)
    prepare_helios_data(
        input_dir=paths["helios_dir"],
        output_dir=paths["valid_rays_dir"],
        references_dir=paths["references_dir"],
        leaf_object_ids=args.leaf_object_ids,
        wood_object_ids=args.wood_object_ids,
        use_class=args.use_class,
        debug=args.debug,
    )


def step3_intersections(args, paths):
    """Compute ray-voxel intersections (notebook Step 3)."""
    print("\n" + "=" * 80)
    print("  Step 3 - Compute ray-voxel intersections")
    print("=" * 80)
    voxel_ray_intersections(
        valid_rays_dir=paths["valid_rays_dir"],
        references_dir=paths["references_dir"],
        debug=args.debug,
    )


def _legs_for_voxel_size(valid_rays_dir, voxel_size):
    """Return the sorted list of leg IDs that have intersection files for a voxel size."""
    pattern = os.path.join(
        valid_rays_dir, f"leg_*_voxel_{voxel_size}_intersections.parquet"
    )
    legs = []
    for path in glob.glob(pattern):
        parts = os.path.basename(path).split("_")
        legs.append(int(parts[parts.index("leg") + 1]))
    return sorted(set(legs))


def step4_metrics(args, paths):
    """Calculate optical metrics and merge with reference data (notebook Step 4)."""
    print("\n" + "=" * 80)
    print("  Step 4 - Calculate optical metrics and merge with reference data")
    print("=" * 80)

    legs = _all_or_int_list(args.legs)
    voxel_sizes = _all_or_float_list(args.voxel_sizes)

    references_dir = paths["references_dir"]
    valid_rays_dir = paths["valid_rays_dir"]
    results_dir = paths["results_dir"]
    project_name = args.name or os.path.basename(os.path.normpath(paths["project_dir"]))

    # Compute per-voxel metrics. get_voxel_metrics discovers the intersection
    # Parquet files in valid_rays_dir, filtered by scan (leg) and voxel size,
    # and returns a single DataFrame spanning all processed voxel sizes.
    voxel_metrics_df = get_voxel_metrics(
        intersections_folder=valid_rays_dir,
        average_leaf_area=args.average_leaf_area,
        output_dir=results_dir,
        project_name=project_name,
        scan_ids=None if legs == "all" else [str(leg) for leg in legs],
        voxel_sizes=None if voxel_sizes == "all" else voxel_sizes,
        is_multireturn=args.is_multireturn,
        debug=args.debug,
    )

    if voxel_metrics_df is None or voxel_metrics_df.empty:
        print("No voxel metrics were computed. Please check the input parameters.")
        return

    # Merge each voxel size's metrics with its reference voxel map and save,
    # reproducing the notebook's validation output.
    for voxel_size, metrics in voxel_metrics_df.groupby("voxel_size", sort=False):
        metrics = metrics.reset_index(drop=True)

        reference_matches = glob.glob(
            os.path.join(references_dir, f"*voxel_size_{voxel_size}*")
        )
        if not reference_matches:
            print(
                f"  ! No reference file found for voxel size {voxel_size}; "
                "skipping merge for this size."
            )
            continue
        reference_file = reference_matches[0]

        df_ref = pd.read_csv(reference_file)
        df_ref = df_ref.groupby("voxel_id").mean(numeric_only=True).reset_index()
        df_ref = df_ref.add_suffix("_ref")
        df_ref.rename(
            columns={
                "voxel_id_ref": "voxel_id",
                "LAD_ref_ref": "LAD_ref",
                "PAD_ref_ref": "PAD_ref",
            },
            inplace=True,
        )

        merged = metrics.merge(df_ref, on="voxel_id", how="left")
        merged = merged.drop(
            columns=[
                col
                for col in ["voxel_cx_ref", "voxel_cy_ref", "voxel_cz_ref"]
                if col in merged.columns
            ]
        )

        legs_in_files = _legs_for_voxel_size(valid_rays_dir, voxel_size)
        leg_string = "_".join(map(str, legs_in_files))
        output_file = os.path.join(
            results_dir,
            f"{project_name}_leg_{leg_string}_voxel_size_{voxel_size}.csv",
        )

        if os.path.exists(output_file):
            os.remove(output_file)
        merged.to_csv(output_file, index=False)
        print(f"  Results saved to {output_file}")


def convert_intersections_to_csv(args, paths):
    """Convert intersection Parquet files to CSV (notebook utility cell)."""
    print("\n" + "=" * 80)
    print("  Utility - Convert intersection Parquet files to CSV")
    print("=" * 80)
    valid_rays_dir = paths["valid_rays_dir"]
    for voxel_size in args.convert_csv_voxel_sizes:
        input_files = glob.glob(
            os.path.join(
                valid_rays_dir, f"leg_*_voxel_{voxel_size}_intersections.parquet"
            )
        )
        for input_file in input_files:
            output_file = input_file.replace(".parquet", ".csv")
            convert_parquet_to_csv(input_file, output_file)
            print(f"  Converted {input_file} to {output_file}")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv=None):
    args = parse_args(argv)

    paths = step1_configure_and_test(args)

    if args.test:
        print(
            "\n--test set: test classification image written to "
            f"{paths['project_dir']}. Exiting before processing."
        )
        return 0

    step2_prepare_rays(args, paths)
    step3_intersections(args, paths)
    step4_metrics(args, paths)

    if args.convert_csv:
        convert_intersections_to_csv(args, paths)

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
