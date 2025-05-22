import numpy as np
import pandas as pd
import glob
import utils
import os
import shutil

# Set up the project directory
project_dir = '/home/uqrarya1/Desktop/Scratch/veg3d/uqrarya1/phd_work/blender2heliosScene/026_Trees/Leaf&Wood'
valid_rays_dir = os.path.join(project_dir, 'valid_rays')

# Establish the voxel_sizes you want to correct
voxel_sizes = [1.5] 

for voxel_size in voxel_sizes:
    intersection_files = glob.glob(os.path.join(valid_rays_dir, f'*{voxel_size}_intersections.parquet'))
    if intersection_files == []:
        raise FileNotFoundError("No intersection files found. Please check the directory.")
    
    # Copy the old intersection files to avoid data loss
    for file in intersection_files:
        new_file = file.replace('.parquet', '_old.parquet')
        shutil.copy(file, new_file)

    # Add normals and weights from the intersection files
    new_intersections_df = utils.add_normals_weights_from_intersection_files(files=intersection_files, knn=6)

    # Function to save the new intersection files
    def save_new_file(df):
        leg_id = df.name
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=utils.voxel_ray_intersection_schema)
        print(f"Saved new intersection file for leg {leg_id} at {output_filename}")
    
    new_intersections_df.groupby('leg_id').apply(lambda x: save_new_file(x))

print("Normals and weights added to the intersection files.")