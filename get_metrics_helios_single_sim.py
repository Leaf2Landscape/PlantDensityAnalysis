from utils import calculate_lambda_1, get_voxel_metrics, get_voxel_metrics_dask

import os
import glob
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate voxel metrics from Helios single simulation data.")
    parser.add_argument("project_directory", type=str, help="Path to the project directory containing valid rays and results folders. This expects 'valid_rays' and 'results' subfolders inside, unless specified.")
    parser.add_argument("--average_leaf_area", type=float, required=True, help="Average leaf area in square centimeters (cm²) used for lambda_1 calculation.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=None, help="Voxel size in one-side size (i.e. 2.0 for 2x2x2 voxels). Default: None (will analyse all voxel sizes preparaed in project folder)")
    parser.add_argument("--legs", type=int, nargs='+', default=None, help="List of leg IDs to process (i.e. --legs 1 3 6 7 will limit the outputs to use voxel-ray intersections from legs 1, 3, 6, and 7 even if more were processed). Default: None (will analyse all legs in project folder)")
    parser.add_argument("--valid_rays_path", type=str, default=None, help="Path to the valid rays directory. Defaults to 'valid_rays' folder in project directory.")
    parser.add_argument("--results_path", type=str, default=None, help="Path to the results directory. Defaults to 'results' folder in project directory.")
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

    intersection_files = []

    # If voxel_ray_intersections parquet folder is found within valid_rays_dir, use it instead of looking for parquet files directly in valid_rays_dir
    voxel_ray_intersections_dir = os.path.join(valid_rays_dir, "voxel_ray_intersections")
    if os.path.exists(voxel_ray_intersections_dir) and os.path.isdir(voxel_ray_intersections_dir):
        intersection_files.append(voxel_ray_intersections_dir)
        
    else:
        leg_str = "all" if legs is None else "_".join(map(str, legs))
        if legs is None and voxel_sizes is None:
            # Use all files in valid_rays_dir
            intersection_files = glob.glob(os.path.join(valid_rays_dir, "*_intersections.parquet"))
        elif legs is None and isinstance(voxel_sizes, list):
            # Use all legs but only specified voxel sizes
            for voxel_size in voxel_sizes:
                pattern = os.path.join(valid_rays_dir, f"leg_*_voxel_{voxel_size}_intersections.parquet")
                intersection_files.extend(glob.glob(pattern))
        elif isinstance(legs, list) and voxel_sizes is None:
            # Use all voxel sizes but only specified legs
            for leg in legs:
                pattern = os.path.join(valid_rays_dir, f"*_leg_{leg}_*_intersections.parquet")
                intersection_files.extend(glob.glob(pattern))
        else:
            # Use only specified legs and voxel sizes
            for leg in legs:
                for voxel_size in voxel_sizes:
                    pattern = os.path.join(valid_rays_dir, f"*leg_{leg}_voxel_{voxel_size}_intersections.parquet")
                    intersection_files.extend(glob.glob(pattern))

    if len(intersection_files) == 0:
        raise RuntimeError(f"No intersection files found matching the specified criteria in {valid_rays_dir}.")
    
    # Compute metrics and pass in desired output path
    voxel_metrics_df = get_voxel_metrics(
        intersections_files=intersection_files, 
        average_leaf_area=args.average_leaf_area, 
        output_dir=results_dir,
        scan_ids=legs if legs is not None else None,
        voxel_sizes=voxel_sizes if voxel_sizes is not None else None,
        debug=args.debug,
        optimal_threads=2
    )
