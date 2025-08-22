import os
import glob
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
import shutil
import dask.dataframe as dd
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix incorrect intersections in ray data.")
    parser.add_argument("valid_rays_dir", type=str, help="Directory containing valid rays and intersection files.")
    parser.add_argument("--num_jobs", type=int, default=8, help="Number of parallel jobs.")
    args = parser.parse_args()

    valid_rays_dir = args.valid_rays_dir
    num_jobs = args.num_jobs
    intersection_files = glob.glob(os.path.join(valid_rays_dir, "*_intersections.parquet"))

    def process_file(file):
        df = pd.read_parquet(file, engine='pyarrow')
        leg_id = df['leg_id'].iloc[0]
        voxel_size = df['voxel_size'].iloc[0]
        valid_rays_file = os.path.join(valid_rays_dir, f'leg_{leg_id}_valid_rays.parquet')
        valid_rays_dd = dd.read_parquet(valid_rays_file, engine='pyarrow')
        valid_rays_dd = valid_rays_dd[['ray_id', 'origin_x', 'origin_y', 'origin_z']]
        valid_rays_dd = valid_rays_dd.compute()

        df = df.merge(valid_rays_dd, on='ray_id', how='left')

        del valid_rays_dd

        voxel = df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values
        entry = df[['t_entry_x', 't_entry_y', 't_entry_z']].values
        exit = df[['t_exit_x', 't_exit_y', 't_exit_z']].values
        point = df[['point_x', 'point_y', 'point_z']].values
        origin = df[['origin_x', 'origin_y', 'origin_z']].values

        # Identify rows where point_x, point_y, or point_z is nan (unbound rays)
        unbound_mask = np.isnan(point).any(axis=1)
        hit_mask = np.all((point >= (voxel - voxel_size / 2 - 1e-9)) & (point <= (voxel + voxel_size / 2 + 1e-9)), axis=1)

        dist_o_entry = np.sum((origin - entry) ** 2, axis=1)
        dist_o_exit = np.sum((origin - exit) ** 2, axis=1)
        dist_o_point = np.sum((point - origin) ** 2, axis=1)

        before_mask = (dist_o_entry > dist_o_point) & (~unbound_mask) & (~hit_mask)
        after_mask = (dist_o_exit < dist_o_point) & (~unbound_mask) & (~hit_mask)

        num_before = before_mask.sum()
        num_after = after_mask.sum()
        num_in = hit_mask.sum()
        num_unbound = unbound_mask.sum()
        df.loc[unbound_mask, 'hit_type'] = 0
        df.loc[before_mask, 'hit_type'] = 1
        df.loc[hit_mask, 'hit_type'] = 2
        df.loc[after_mask, 'hit_type'] = 3

        # Save backup and overwrite
        old_file = os.path.join(valid_rays_dir, os.path.basename(file).replace(".parquet", "_old.parquet"))
        shutil.copy2(file, old_file)
        output_file = os.path.join(valid_rays_dir, os.path.basename(file))
        df.to_parquet(output_file, engine='pyarrow')

    Parallel(n_jobs=num_jobs, prefer="threads")(
        delayed(process_file)(file) for file in tqdm(intersection_files, desc="Processing intersection files")
    )