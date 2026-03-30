from utils import calculate_lambda_1, get_voxel_metrics

import os
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate voxel metrics from Helios single simulation data.")
    parser.add_argument("project_directory", type=str, help="Path to the project directory containing valid rays and results folders. This expects 'valid_rays' and 'results' subfolders inside, unless specified.")
    parser.add_argument("--average_leaf_area", type=float, default=None, help="Average leaf area in square centimeters (cm²) used for lambda_1 calculation.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=None, help="Voxel size in one-side size (i.e. 2.0 for 2x2x2 voxels). Default: None (will analyse all voxel sizes preparaed in project folder)")
    parser.add_argument("--legs", type=int, nargs='+', default=None, help="List of leg IDs to process (i.e. --legs 1 3 6 7 will limit the outputs to use voxel-ray intersections from legs 1, 3, 6, and 7 even if more were processed). Default: None (will analyse all legs in project folder)")
    parser.add_argument("--valid_rays_path", type=str, default=None, help="Path to the valid rays directory. Defaults to 'valid_rays' folder in project directory.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the results directory. Defaults to 'results' folder in project directory.")
    parser.add_argument("--same_normals", action='store_true', help="If set, uses the same normals for all voxel sizes (instead of calculating separate normals for each voxel size). This can speed up processing if you have many voxel sizes and want to use the same normals for all of them.")
    parser.add_argument("--debug", action='store_true', help="If set, runs in debug mode with verbose outputs and extra saved files")
    args = parser.parse_args()

    # Setup and validate paths
    project_dir = args.project_directory
    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"Project directory {project_dir} does not exist.")
    valid_rays_dir = os.path.join(project_dir, "valid_rays") if args.valid_rays_path is None else args.valid_rays_path
    if not os.path.exists(valid_rays_dir):
        raise FileNotFoundError(f"Valid rays directory {valid_rays_dir} does not exist.")
    results_dir = os.path.join(project_dir, "results") if args.results_path is None else args.results_path
    try:
        os.makedirs(results_dir, exist_ok=True)
    except Exception as e:
        raise OSError(f"Error creating results directory {results_dir}: {e}")

    # Setup chosen files to analyse
    legs = args.legs
    voxel_sizes = args.voxel_sizes

    # If voxel_ray_intersections parquet folder is found within valid_rays_dir, use it instead of looking for parquet files directly in valid_rays_dir
    voxel_ray_intersections_dir = os.path.join(valid_rays_dir, "voxel_ray_intersections")
    if os.path.exists(voxel_ray_intersections_dir) and os.path.isdir(voxel_ray_intersections_dir):
        intersection_folder = voxel_ray_intersections_dir
    else:
        intersection_folder = valid_rays_dir

    # look for leaf_area.csv if average_leaf_area is not provided as an argument, and read the value from there if found
    if args.average_leaf_area is None:
        leaf_area_csv_path_candidates = glob.glob(os.path.join(project_dir, "*leaf_area.csv"))
        if leaf_area_csv_path_candidates:
            leaf_area_csv_path = leaf_area_csv_path_candidates[0]
        if os.path.exists(leaf_area_csv_path):
            try:
                import pandas as pd
                leaf_area_df = pd.read_csv(leaf_area_csv_path)
                if "avg_leaf_area" in leaf_area_df.columns:
                    average_leaf_area = leaf_area_df["avg_leaf_area"].iloc[0]
                    print(f"Read average leaf area from {leaf_area_csv_path}: {average_leaf_area} cm²")
                else:
                    raise ValueError(f"'avg_leaf_area' column not found in {leaf_area_csv_path}. Please provide average leaf area as an argument or ensure the CSV has the correct format.")
            except Exception as e:
                raise ValueError(f"Error reading average leaf area from {leaf_area_csv_path}: {e}")
        else:
            raise ValueError("Average leaf area is required for lambda_1 calculation. Please provide it as an argument or include a 'leaf_area.csv' file in the project directory with an 'avg_leaf_area' column.")
    else:
        average_leaf_area = args.average_leaf_area
        print(f"Using average leaf area from argument: {average_leaf_area} cm²")

    # Compute metrics and pass in desired output path
    voxel_metrics_df = get_voxel_metrics(
        intersections_folder=intersection_folder, 
        average_leaf_area=average_leaf_area, 
        output_dir=results_dir,
        project_name=os.path.basename(os.path.normpath(project_dir)),
        scan_ids=legs if legs is not None else None,
        voxel_sizes=voxel_sizes if voxel_sizes is not None else None,
        debug=args.debug,
        optimal_threads=2
    )
