import os
import argparse
import glob
import pandas as pd

from utils import (
    prepare_helios_data,
    add_normals_weights_to_valid_rays,
    voxel_ray_intersections,
    calculate_lambda_1,
    get_voxel_metrics,
)

def main(args):
    # Set up the project directory
    project_dir = args.project_dir
    helios_dir = os.path.join(project_dir, 'helios')
    references_dir = os.path.join(project_dir, 'references')
    results_dir = os.path.join(project_dir, 'results')
    valid_rays_dir = os.path.join(project_dir, 'valid_rays')

    if not os.path.exists(helios_dir) or not os.path.exists(references_dir):
        raise FileNotFoundError("The specified directories do not exist. Please check the paths.")

    os.makedirs(valid_rays_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # Run the data preparation script
    # prepare_helios_data(
    #    input_dir=helios_dir,
    #    output_dir=valid_rays_dir,
    #    references_dir=references_dir,
    #    leaf_object_ids=args.leaf_object_ids,
    #    wood_object_ids=args.wood_object_ids,
    #    use_class=args.use_class,
    #    debug=args.debug
    #)

    # Calculate normals and weights by loading valid rays
    #add_normals_weights_to_valid_rays(
    #    valid_rays_dir,
    #    debug=args.debug,
    #    knn=args.knn
    #)

    # Run intersections
    voxel_ray_intersections(
        valid_rays_dir=valid_rays_dir,
        references_dir=references_dir,
        debug=args.debug
    )

    # Select the desired legs and voxel_sizes to include in the analysis
    legs = args.legs
    voxel_sizes = args.voxel_sizes
    average_leaf_area = args.average_leaf_area

    # Get the list of all voxel sizes
    intersection_files = []
    if legs == 'all' and voxel_sizes == 'all':
        intersection_files = glob.glob(os.path.join(valid_rays_dir, '*_intersections.parquet'))
    elif legs == 'all' and isinstance(voxel_sizes, list):
        for voxel_size in voxel_sizes:
            intersection_files += glob.glob(os.path.join(valid_rays_dir, f'leg_*_voxel_{voxel_size}_intersections.parquet'))
    elif isinstance(legs, list) and voxel_sizes == 'all':
        for leg in legs:
            intersection_files += glob.glob(os.path.join(valid_rays_dir, f'leg_{leg}_*_intersections.parquet'))
    else:
        for leg in legs:
            for voxel_size in voxel_sizes:
                intersection_files += glob.glob(os.path.join(valid_rays_dir, f'leg_{leg}_voxel_{voxel_size}_intersections.parquet'))

    if not intersection_files:
        print("No intersection files found. Please check the input parameters.")
        return

    # Split intersection files into separate lists for each voxel_size
    voxel_size_files = {}
    for file in intersection_files:
        parts = file.split('_')
        voxel_size = float(parts[parts.index('voxel') + 1])
        voxel_size_files.setdefault(voxel_size, []).append(file)

    # Extract voxel information for each voxel size
    for voxel_size, files in voxel_size_files.items():
        legs_in_files = []
        for file in files:
            leg = os.path.basename(file)
            parts = leg.split('_')
            leg = int(parts[parts.index('leg') + 1])
            legs_in_files.append(leg)

        lambda_1 = calculate_lambda_1(voxel_size=voxel_size, average_leaf_area=average_leaf_area)
        print(f"Voxel size: {voxel_size}, Lambda_1: {lambda_1}")

        voxel_metrics_df = get_voxel_metrics(
            intersections_files=files,
            lambda_1=lambda_1,
            is_multireturn=False
        )

        reference_file = glob.glob(os.path.join(references_dir, f'*voxel_size_{voxel_size}*'))[0]
        df_ref = pd.read_csv(reference_file)
        df_ref = df_ref.groupby('voxel_id').mean(numeric_only=True).reset_index()
        df_ref = df_ref.add_suffix('_ref')
        df_ref.rename(columns={
            'voxel_id_ref': 'voxel_id',
            'LAD_ref_ref': 'LAD_ref',
            'PAD_ref_ref': 'PAD_ref'
        }, inplace=True)

        voxel_metrics_df = voxel_metrics_df.merge(df_ref, on='voxel_id', how='left')
        voxel_metrics_df.drop(columns=[
            "voxel_cx_ref",
            "voxel_cy_ref",
            "voxel_cz_ref"
        ], inplace=True)

        # Save outputs to csv
        project_name = os.path.basename(os.path.normpath(project_dir))
        legs_in_files.sort()
        leg_string = "_".join(map(str, legs_in_files))
        output_file = os.path.join(results_dir, f"{project_name}_leg_{leg_string}_voxel_size_{voxel_size}.csv")
        if os.path.exists(output_file):
            os.remove(output_file)
        voxel_metrics_df.to_csv(output_file, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Helios Plant Density Analysis")
    parser.add_argument('project_dir', type=str, help='Project directory')
    parser.add_argument('--leaf_object_ids', type=int, nargs='+', default=[1], help='Leaf object IDs')
    parser.add_argument('--wood_object_ids', type=int, nargs='+', default=[0], help='Wood object IDs')
    parser.add_argument('--use_class', action='store_true', help='Use class information')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--knn', type=int, default=6, help='KNN value for normals calculation')
    parser.add_argument('--legs', default='all', help='Legs to include (all or list of ints)')
    parser.add_argument('--voxel_sizes', default='all', help='Voxel sizes to include (all or list of floats)')
    parser.add_argument('--average_leaf_area', type=float, default=0.0146533, help='Average leaf area (m^2)')

    args = parser.parse_args()

    # Convert legs and voxel_sizes if they are lists
    if args.legs != 'all':
        args.legs = [int(x) for x in args.legs.split(',')]
    if args.voxel_sizes != 'all':
        args.voxel_sizes = [float(x) for x in args.voxel_sizes.split(',')]

    main(args)
