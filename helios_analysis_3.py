import os
import glob
import utils
import pandas as pd

# Select the desired legs and voxel_sizes to include in the analysis
# 1. load your table of (tree_id, average_leaf_area)
leaf_area_df = pd.DataFrame({
    'tree_id': [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 23, 24, 25, 26],
    'average_leaf_area': [
        0.002456154, 0.002456128, 0.002456134, 0.002456159, 0.002456129,
        0.002456163, 0.002456154, 0.002456139, 0.002456179, 0.002456127,
        0.002456079, 0.008347659, 0.002120369, 0.008350465, 0.008352368,
        0.005341984, 0.000228955, 0.000146533, 0.000329696, 0.0000667633
    ]
})

# 2. user-configurable legs/voxel_sizes
legs = 'all'
voxel_sizes = 'all'  # or [0.2, 0.5,&]

for _, row in leaf_area_df.iterrows():
    tree_id = str(row['tree_id']).zfill(3)
    avg_leaf = row['average_leaf_area']

    # Set up the project directory
    project_dir = '/home/uqrarya1/Desktop/Scratch/veg3d/uqrarya1/phd_work/blender2heliosScene'
    valid_rays_dir = os.path.join(project_dir, f'{tree_id}', 'Leaf&Wood', 'valid_rays')
    references_dir = os.path.join(project_dir, f'{tree_id}', 'Leaf&Wood', 'references')
    # Set up the output directory
    output_dir = os.path.join(project_dir, f'{tree_id}', 'Leaf&Wood', 'results')

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

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

    # Check if any intersection files were found
    if intersection_files == []:
        print("No intersection files found. Please check the input parameters.")

    # Split intersection files into separate lists for each voxel_size
    voxel_size_files = {}
    for file in intersection_files:
        # Extract the voxel size from the filename
        parts = file.split('_')
        voxel_size = float(parts[parts.index('voxel') + 1])
        
        # Add the file to the corresponding voxel size list
        if voxel_size not in voxel_size_files:
            voxel_size_files[voxel_size] = []
        voxel_size_files[voxel_size].append(file)

    # Extract voxel information for each voxel size
    for voxel_size, files in voxel_size_files.items():
        # Create a list of all legs in files
        legs = []
        for file in files:
            leg = os.path.basename(file)
            parts = leg.split('_')
            leg = int(parts[parts.index('leg') + 1])
            legs.append(leg)

        # Calculate the lambda_1 for average leaf area
        lambda_1 = utils.calculate_lambda_1(voxel_size=voxel_size, average_leaf_area=avg_leaf)
        print(f"Voxel size: {voxel_size}, Lambda_1: {lambda_1}")

        # Calculate per voxel information from all files
        voxel_metrics_df = utils.get_voxel_metrics(intersections_files=files, lambda_1=lambda_1, is_leaf_true=True)

        # Retrieve the reference file
        reference_file = glob.glob(os.path.join(references_dir, f'*voxel_size_{voxel_size}*'))[0]
        df_ref = pd.read_csv(reference_file)

        # CI_leaf_Corr, CI_lw_Corr
        # Ensure only numeric columns are included in the mean operation
        df_ref = df_ref.groupby('voxel_id').mean(numeric_only=True).reset_index()
        df_ref = df_ref.add_suffix('_ref')

        df_ref.rename(columns={
            'voxel_id_ref': 'voxel_id', 
            'voxel_cx_ref': 'voxel_cx',
            'voxel_cy_ref': 'voxel_cy',
            'voxel_cz_ref': 'voxel_cz',
            'LAD_ref_ref': 'LAD_ref', 
            'PAD_ref_ref': 'PAD_ref'
            }, inplace=True)

        # Merge to maintain voxel_id matching
        voxel_metrics_df = voxel_metrics_df.merge(df_ref, on='voxel_id', how='left')

        ### Add LAD calculations here if desired
        """Example, LAD_BL_TLS

        # Retrieve required variables
        I_leaf = voxel_metrics_df['I_leaf'].values
        mean_path_length = voxel_metrics_df['mean_path_length'].values  
        G_leaf = voxel_metrics_df['G_leaf'].values
        CI_leaf_ref = voxel_metrics_df['CI_leaf_corr_ref'].values

        LAD_BL_TLS = utils.BL_pimont_2018(I=I_leaf, mean_path_length=mean_path_length)
        LAD_BL_TLS_G = utils.BL_pimont_2018(I=I_leaf, mean_path_length=mean_path_length, G=G_leaf)
        LAD_BL_TLS_CI_ref = utils.BL_pimont_2018(I=I_leaf, mean_path_length=mean_path_length, G=G_leaf, CI=CI_leaf_ref)
        """

        # Save outputs to csv
        project_name = os.path.basename(os.path.normpath(project_dir))
        legs.sort()
        leg_string = "_".join(map(str, legs))
        output_file = os.path.join(output_dir, f"{project_name}_leg_{leg_string}_voxel_size_{voxel_size}.csv")
        if os.path.exists(output_file):
            os.remove(output_file)
        voxel_metrics_df.to_csv(output_file)
        print(f" {tree_id}, vs={voxel_size} Â {output_file}")