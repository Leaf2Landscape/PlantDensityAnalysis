"""
Process Helios simulation data to extract voxel metrics from ray-voxel intersections.

This script orchestrates a three-step pipeline to analyze Helios LiDAR simulation data:
1. Prepares raw Helios data by filtering and validating ray intersections based on 
    material classification (leaf/wood) and generating valid ray datasets.
2. Computes surface normals and weighting factors for valid rays to support 
    subsequent spatial analysis.
3. Calculates intersections between valid rays and all available voxel grids 
    stored in the reference directory, producing voxel-level metrics.
The script expects a project structure with 'helios' and 'references' subdirectories
(or custom paths specified via arguments). Output is organized into 'valid_rays' 
and 'results' directories containing processed ray data and voxel intersection metrics.

Command-line Arguments:
     --project_directory (str, required): Root project path containing data subfolders.
     --output_path (str, optional): Custom output directory. Defaults to project_directory.
     --leaf_ids (int list): Object IDs or class values representing leaf material. Default: [1]
     --wood_ids (int list): Object IDs or class values representing wood material. Default: [0]
     --use_class (flag): If set, uses material classification; otherwise uses hit object IDs.
     --helios_data_path (str, optional): Custom path to Helios simulation data.
     --reference_data_path (str, optional): Custom path to reference voxel grids.
     --test_mode (flag): If set, validates settings and outputs classification visualization.
     --debug (flag): If set, enables verbose output and saves intermediate files.
Raises:
     FileNotFoundError: If project, Helios, or reference directories do not exist.
     OSError: If output directories cannot be created.
     RuntimeError: If test mode validation fails.

This code processes Helios simulation data to prep all the voxel-ray intersections completed in order to gather voxel metrics from various scanning inputs.
"""
from utils import test_helios_settings, prepare_helios_data, voxel_ray_intersections, voxel_ray_intersections_nodask, voxel_ray_intersections_nodask_04

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process Helios simulation data for voxel metrics.")
    parser.add_argument("project_directory", type=str, help="Path to the project. The project expects 'helios' and 'references' subfolders inside, unless specified.")
    parser.add_argument("--output_path", type=str, default=None, help="Set here for custom output path. By default, this will use the data_path as the output path.")
    parser.add_argument("--leaf_ids", type=int, nargs='+', default=[1], help="List of leaf IDs in Helios outputs. Default: [1]")
    parser.add_argument("--wood_ids", type=int, nargs='+', default=[0], help="List of wood IDs in Helios outputs. Default: [0]")
    parser.add_argument("--use_class", action='store_true', help="Whether the Helios simulation used material classification. Default uses hit object IDs")
    parser.add_argument("--helios_data_path", type=str, default=None, help="OPTIONAL: Path to the Helios simulation data. Defaults to 'helios' folder in project directory.")
    parser.add_argument("--reference_data_path", type=str, default=None, help="OPTIONAL: Path to the reference data. Defaults to 'references' folder in project directory.")
    parser.add_argument("--test_mode", action='store_true', help="If set, only outputs plot of wood and leaf points to check settings.")
    parser.add_argument("--overwrite_valid_rays", action='store_true', help="If set, will overwrite existing valid rays files in output directory. Default: False (will skip processing if valid rays files are found)")
    parser.add_argument("--debug", action='store_true', help="If set, runs in debug mode with verbose outputs and extra saved files")
    args = parser.parse_args()

    # Handle input paths
    project_dir = args.project_directory
    if not os.path.exists(project_dir):
        raise FileNotFoundError(f"Project directory {project_dir} does not exist.")
    helios_dir = os.path.join(project_dir,"helios") if args.helios_data_path is None else args.helios_data_path
    if not os.path.exists(helios_dir):
        raise FileNotFoundError(f"Helios data directory {helios_dir} does not exist.")
    reference_dir = os.path.join(project_dir,"references") if args.reference_data_path is None else args.reference_data_path
    if not os.path.exists(reference_dir):
        raise FileNotFoundError(f"Reference data directory {reference_dir} does not exist.")
    
    # Handle output paths
    valid_rays_dir = os.path.join(project_dir,"valid_rays") if args.output_path is None else os.path.join(args.output_path, "valid_rays")
    try:
        os.makedirs(valid_rays_dir, exist_ok=True)
    except Exception as e:
        raise OSError(f"Error creating output directory {valid_rays_dir}: {e}")

    # If in test mode, just test the Helios settings and exit
    if args.test_mode:
        success = test_helios_settings(
            helios_dir=helios_dir, 
            leaf_object_ids=args.leaf_ids, 
            wood_object_ids=args.wood_ids, 
            use_class=args.use_class, 
            output_dir=project_dir)
        if not success:
            raise RuntimeError("Test mode failed. Please check the Helios settings and output plot.")
        else:
            print("Test mode completed successfully. Please check the output plot for leaf and wood point classification.")
            exit(0)

    # Step 1: Prepare Helios data
    if args.overwrite_valid_rays or not any(fname.endswith("valid_rays.parquet") for fname in os.listdir(valid_rays_dir)):
        print("No existing valid rays files found or overwrite enabled. Processing Helios data to generate valid rays...")
        try:
            prepare_helios_data(
                input_dir=helios_dir,
                output_dir=valid_rays_dir,
                references_dir=reference_dir,
                leaf_object_ids=args.leaf_ids,
                wood_object_ids=args.wood_ids,
                use_class = args.use_class,
                debug=args.debug
            )
        except Exception as e:
            raise RuntimeError(f"Error during Helios data preparation: {e}")
        print("Helios data preparation completed. Proceeding to normals and voxel-ray intersections...")

    # Step 2: Calculate voxel-ray intersections for chosen voxel sizes
    try:
        voxel_ray_intersections_nodask(
            valid_rays_dir=valid_rays_dir,
            references_dir=reference_dir
        )
    except Exception as e:
        raise RuntimeError(f"Error during voxel-ray intersections computation: {e}")

    print("Helios data processing completed successfully.")


