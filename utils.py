"""
A Python script of commonly shared utilities for other scripts.
Includes schemas for i/o data, functions, and helpers.
"""

from fnvhash import fnv1a_32
import pyarrow as pa
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import gc
import dask
from dask.distributed import progress, get_client
import os
import tempfile
import uuid
import shutil
from scipy.sparse.csgraph import connected_components
import time


### CONSTANTS ###
beam_divergence = np.float32(0.001) # Beam divergence in radians

### SCHEMAS ###

# Voxel Ray Intersection Schema
"""
This schema is used to store ray intersections for each voxel.
It leverages the pyarrow library to maximise efficiency of dask, pandas, and parquet.

It is saved in the format:
    leg_{leg_id}_voxel_{voxel_size}_ray_intersections.parquet

And contains the information outlined in the following schema.
Each index corresponds to a ray that intersects a voxel.
"""

voxel_ray_intersection_schema_old = pa.schema([
    pa.field('voxel_size', pa.float32()),
    pa.field('voxel_id', pa.int64()),
    pa.field('leg_id', pa.int64()),
    pa.field('ray_id', pa.int64()),
    pa.field('t_entry_x', pa.float32()),
    pa.field('t_entry_y', pa.float32()),
    pa.field('t_entry_z', pa.float32()),
    pa.field('t_exit_x', pa.float32()),
    pa.field('t_exit_y', pa.float32()),
    pa.field('t_exit_z', pa.float32()),
    pa.field('t_entry_radius', pa.float32()),
    pa.field('t_exit_radius', pa.float32()),
    pa.field('point_x', pa.float32()),
    pa.field('point_y', pa.float32()),
    pa.field('point_z', pa.float32()),
    pa.field('echo_intensity', pa.float32()),
    pa.field('return_number', pa.int32()),
    pa.field('number_of_returns', pa.int32()),
    pa.field('normal_x', pa.float32()),
    pa.field('normal_y', pa.float32()),
    pa.field('normal_z', pa.float32()),
    pa.field('point_weight', pa.float32()),
    pa.field('viewing_angle', pa.float32()),
    pa.field('hit_ray', pa.bool_()),
    pa.field('is_leaf', pa.bool_())
])
voxel_ray_intersection_schema = pa.schema([
    pa.field('voxel_size', pa.float32()),
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_cx', pa.float64()),
    pa.field('voxel_cy', pa.float64()),
    pa.field('voxel_cz', pa.float64()),
    pa.field('leg_id', pa.uint64()),
    pa.field('ray_id', pa.uint64()),
    pa.field('t_entry_x', pa.float64()),
    pa.field('t_entry_y', pa.float64()),
    pa.field('t_entry_z', pa.float64()),
    pa.field('t_exit_x', pa.float64()),
    pa.field('t_exit_y', pa.float64()),
    pa.field('t_exit_z', pa.float64()),
    pa.field('distance_to_centre', pa.float64()),
    pa.field('point_x', pa.float64()),
    pa.field('point_y', pa.float64()),
    pa.field('point_z', pa.float64()),
    pa.field('echo_intensity', pa.float64()),
    pa.field('return_number', pa.int32()),
    pa.field('number_of_returns', pa.int32()),
    pa.field('normal_x', pa.float64()),
    pa.field('normal_y', pa.float64()),
    pa.field('normal_z', pa.float64()),
    pa.field('point_weight', pa.float64()),
    pa.field('viewing_angle', pa.float64()),
    pa.field('hit_type', pa.int32()),
    pa.field('is_leaf', pa.bool_())
])

# Voxel Metrics Schema
"""
This schema is used to store the metrics for each voxel, based on the selected legs and voxel size.
Since this one is only used to store to a csv file (for final output), it is not as important to be efficient.


"""

voxel_metrics_schema_oldcode = pa.schema([
    pa.field('voxel_id', pa.uint32()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('I_lw', pa.int64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.int64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('LIAD_leaf_bin_2.5', pa.float64()),
    pa.field('LIAD_leaf_bin_7.5', pa.float64()),
    pa.field('LIAD_leaf_bin_12.5', pa.float64()),
    pa.field('LIAD_leaf_bin_17.5', pa.float64()),
    pa.field('LIAD_leaf_bin_22.5', pa.float64()),
    pa.field('LIAD_leaf_bin_27.5', pa.float64()),
    pa.field('LIAD_leaf_bin_32.5', pa.float64()),
    pa.field('LIAD_leaf_bin_37.5', pa.float64()),
    pa.field('LIAD_leaf_bin_42.5', pa.float64()),
    pa.field('LIAD_leaf_bin_47.5', pa.float64()),
    pa.field('LIAD_leaf_bin_52.5', pa.float64()),
    pa.field('LIAD_leaf_bin_57.5', pa.float64()),
    pa.field('LIAD_leaf_bin_62.5', pa.float64()),
    pa.field('LIAD_leaf_bin_67.5', pa.float64()),
    pa.field('LIAD_leaf_bin_72.5', pa.float64()),
    pa.field('LIAD_leaf_bin_77.5', pa.float64()),
    pa.field('LIAD_leaf_bin_82.5', pa.float64()),
    pa.field('LIAD_leaf_bin_87.5', pa.float64()),
    # pa.field('LIAD_lw_bin_2.5', pa.float64()),
    # pa.field('LIAD_lw_bin_7.5', pa.float64()),
    # pa.field('LIAD_lw_bin_12.5', pa.float64()),
    # pa.field('LIAD_lw_bin_17.5', pa.float64()),
    # pa.field('LIAD_lw_bin_22.5', pa.float64()),
    # pa.field('LIAD_lw_bin_27.5', pa.float64()),
    # pa.field('LIAD_lw_bin_32.5', pa.float64()),
    # pa.field('LIAD_lw_bin_37.5', pa.float64()),
    # pa.field('LIAD_lw_bin_42.5', pa.float64()),
    # pa.field('LIAD_lw_bin_47.5', pa.float64()),
    # pa.field('LIAD_lw_bin_52.5', pa.float64()),
    # pa.field('LIAD_lw_bin_57.5', pa.float64()),
    # pa.field('LIAD_lw_bin_62.5', pa.float64()),
    # pa.field('LIAD_lw_bin_67.5', pa.float64()),
    # pa.field('LIAD_lw_bin_72.5', pa.float64()),
    # pa.field('LIAD_lw_bin_77.5', pa.float64()),
    # pa.field('LIAD_lw_bin_82.5', pa.float64()),
    # pa.field('LIAD_lw_bin_87.5', pa.float64()),
    pa.field('mean_angle_leaf', pa.float64()), # Mean angle of leaf hits only
    pa.field('mean_angle_all', pa.float64()), # Mean angle of all hits
    pa.field('mean_path_length', pa.float64()),
    pa.field('sum_path_length', pa.float64()),
    pa.field('mean_free_path_length', pa.float64()),
    pa.field('sum_free_path_length', pa.float64()),
    pa.field('sum_free_path_length_hit', pa.float64()),
    pa.field('sum_free_path_length_hit_leaf', pa.float64()),
    pa.field('mean_eff_path_length', pa.float64()),
    pa.field('var_eff_path_length', pa.float64()),
    pa.field('mean_eff_free_path_length', pa.float64()),
    pa.field('var_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length_hit', pa.float64()),  # Sum of z for all hits
    pa.field('sum_eff_free_path_length_hit_leaf', pa.float64()) # Sum of z for leaf hits only    
])

voxel_metrics_schema_singlereturn = pa.schema([
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_cx', pa.float64()),
    pa.field('voxel_cy', pa.float64()),
    pa.field('voxel_cz', pa.float64()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('I_lw', pa.float64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.float64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('lambda_1', pa.float64()),
    pa.field('LIAD_leaf_bin_2.5', pa.float32()),
    pa.field('LIAD_leaf_bin_7.5', pa.float32()),
    pa.field('LIAD_leaf_bin_12.5', pa.float32()),
    pa.field('LIAD_leaf_bin_17.5', pa.float32()),
    pa.field('LIAD_leaf_bin_22.5', pa.float32()),
    pa.field('LIAD_leaf_bin_27.5', pa.float32()),
    pa.field('LIAD_leaf_bin_32.5', pa.float32()),
    pa.field('LIAD_leaf_bin_37.5', pa.float32()),
    pa.field('LIAD_leaf_bin_42.5', pa.float32()),
    pa.field('LIAD_leaf_bin_47.5', pa.float32()),
    pa.field('LIAD_leaf_bin_52.5', pa.float32()),
    pa.field('LIAD_leaf_bin_57.5', pa.float32()),
    pa.field('LIAD_leaf_bin_62.5', pa.float32()),
    pa.field('LIAD_leaf_bin_67.5', pa.float32()),
    pa.field('LIAD_leaf_bin_72.5', pa.float32()),
    pa.field('LIAD_leaf_bin_77.5', pa.float32()),
    pa.field('LIAD_leaf_bin_82.5', pa.float32()),
    pa.field('LIAD_leaf_bin_87.5', pa.float32()),
    pa.field('mean_angle_leaf', pa.float32()), # Mean angle of leaf hits only
    pa.field('mean_angle_all', pa.float32()), # Mean angle of all hits
    pa.field('mean_path_length', pa.float64()),
    pa.field('sum_path_length', pa.float64()),
    pa.field('mean_free_path_length', pa.float64()),
    pa.field('sum_free_path_length', pa.float64()),
    pa.field('sum_free_path_length_hit', pa.float64()),
    pa.field('sum_free_path_length_hit_leaf', pa.float64()),
    pa.field('mean_eff_path_length', pa.float64()),
    pa.field('var_eff_path_length', pa.float64()),
    pa.field('sum_eff_path_length', pa.float64()),
    pa.field('mean_eff_free_path_length', pa.float64()),
    pa.field('mean_eff_free_path_length', pa.float64()),
    pa.field('var_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length_hit', pa.float64()),  # Sum of z for all hits
    pa.field('sum_eff_free_path_length_hit_leaf', pa.float64()) # Sum of z for leaf hits only    
])

voxel_metrics_schema_multireturn = pa.schema([
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_cx', pa.float64()),
    pa.field('voxel_cy', pa.float64()),
    pa.field('voxel_cz', pa.float64()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('I_lw', pa.float64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.float64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('lambda_1', pa.float64()),
    pa.field('P_first', pa.float64()),
    pa.field('P_equal', pa.float64()),
    pa.field('P_intensity', pa.float64()),
    pa.field('P_first_leaf', pa.float64()),
    pa.field('P_equal_leaf', pa.float64()),
    pa.field('P_intensity_leaf', pa.float64()),
    pa.field('LAD_BL_first', pa.float64()),
    pa.field('LAD_BL_equal', pa.float64()),
    pa.field('LAD_BL_intensity', pa.float64()),
    pa.field('LAD_MLE_nocorr', pa.float64()),
    pa.field('LAD_MLE_lambda1', pa.float64()),
    pa.field('LAD_MLE_bias', pa.float64()),
    pa.field('LAD_MLE_lambda1_bias', pa.float64()), 
    pa.field('LIAD_leaf_bin_2.5', pa.float32()),
    pa.field('LIAD_leaf_bin_7.5', pa.float32()),
    pa.field('LIAD_leaf_bin_12.5', pa.float32()),
    pa.field('LIAD_leaf_bin_17.5', pa.float32()),
    pa.field('LIAD_leaf_bin_22.5', pa.float32()),
    pa.field('LIAD_leaf_bin_27.5', pa.float32()),
    pa.field('LIAD_leaf_bin_32.5', pa.float32()),
    pa.field('LIAD_leaf_bin_37.5', pa.float32()),
    pa.field('LIAD_leaf_bin_42.5', pa.float32()),
    pa.field('LIAD_leaf_bin_47.5', pa.float32()),
    pa.field('LIAD_leaf_bin_52.5', pa.float32()),
    pa.field('LIAD_leaf_bin_57.5', pa.float32()),
    pa.field('LIAD_leaf_bin_62.5', pa.float32()),
    pa.field('LIAD_leaf_bin_67.5', pa.float32()),
    pa.field('LIAD_leaf_bin_72.5', pa.float32()),
    pa.field('LIAD_leaf_bin_77.5', pa.float32()),
    pa.field('LIAD_leaf_bin_82.5', pa.float32()),
    pa.field('LIAD_leaf_bin_87.5', pa.float32()),
    pa.field('mean_angle_leaf', pa.float32()), # Mean angle of leaf hits only
    pa.field('mean_angle_all', pa.float32()), # Mean angle of all hits
    pa.field('mean_path_length', pa.float64()),
    pa.field('sum_path_length', pa.float64()),
    pa.field('mean_free_path_length', pa.float64()),
    pa.field('sum_free_path_length', pa.float64()),
    pa.field('sum_free_path_length_hit', pa.float64()),
    pa.field('sum_free_path_length_hit_leaf', pa.float64()),
    pa.field('mean_eff_path_length', pa.float64()),
    pa.field('var_eff_path_length', pa.float64()),
    pa.field('sum_eff_path_length', pa.float64()),
    pa.field('mean_eff_free_path_length', pa.float64()),
    pa.field('var_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length_hit', pa.float64()),  # Sum of z for all hits
    pa.field('sum_eff_free_path_length_hit_leaf', pa.float64()) # Sum of z for leaf hits only    
])

# Occlusion metrics schema
"""
This schema is used to store the occlusion metrics for each voxel.

TEST ONLY at this stage.
"""

# Create occlusion metrics dataframe
voxel_occ_schema = pa.schema([
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_cx', pa.float64()),
    pa.field('voxel_cy', pa.float64()),
    pa.field('voxel_cz', pa.float64()),
    pa.field('theoretical_volume', pa.float64()),
    pa.field('actual_volume', pa.float64()),
    pa.field('volume_coverage', pa.float64()),
    pa.field('weighted_theoretical_volume', pa.float64()),
    pa.field('weighted_actual_volume', pa.float64()),
    pa.field('weighted_volume_coverage', pa.float64()),
    pa.field('theoretical_coverage_west', pa.float64()),
    pa.field('theoretical_coverage_east', pa.float64()),
    pa.field('theoretical_coverage_south', pa.float64()),
    pa.field('theoretical_coverage_north', pa.float64()),
    pa.field('theoretical_coverage_bottom', pa.float64()),
    pa.field('theoretical_coverage_top', pa.float64()),
    pa.field('actual_coverage_west', pa.float64()),
    pa.field('actual_coverage_east', pa.float64()),
    pa.field('actual_coverage_south', pa.float64()),
    pa.field('actual_coverage_north', pa.float64()),
    pa.field('actual_coverage_bottom', pa.float64()),
    pa.field('actual_coverage_top', pa.float64()),
    pa.field('weighted_theoretical_coverage_west', pa.float64()),
    pa.field('weighted_theoretical_coverage_east', pa.float64()),
    pa.field('weighted_theoretical_coverage_south', pa.float64()),
    pa.field('weighted_theoretical_coverage_north', pa.float64()),
    pa.field('weighted_theoretical_coverage_bottom', pa.float64()),
    pa.field('weighted_theoretical_coverage_top', pa.float64()),
    pa.field('weighted_actual_coverage_west', pa.float64()),
    pa.field('weighted_actual_coverage_east', pa.float64()),
    pa.field('weighted_actual_coverage_south', pa.float64()),
    pa.field('weighted_actual_coverage_north', pa.float64()),
    pa.field('weighted_actual_coverage_bottom', pa.float64()),
    pa.field('weighted_actual_coverage_top', pa.float64()),
])

# Reference Schema
"""
This schema is used to store the reference data for each voxel.
"""
reference_schema = pa.schema([
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_size', pa.float32()),
    pa.field('CI', pa.float32()),
    pa.field('woody_vol_proportion', pa.float32()),
    pa.field('G', pa.float32()),
    pa.field('G_leaf', pa.float32()),
    pa.field('LAD', pa.float32()),
    pa.field('PAD', pa.float32()),
])

# Valid Rays Schema
valid_rays_schema = pa.schema([
    pa.field('leg_id', pa.uint64()),
    pa.field('ray_id', pa.uint64()),
    pa.field('origin_x', pa.float64()),
    pa.field('origin_y', pa.float64()),
    pa.field('origin_z', pa.float64()),
    pa.field('direction_x', pa.float64()),
    pa.field('direction_y', pa.float64()),
    pa.field('direction_z', pa.float64()),
    pa.field('point_x', pa.float64()),
    pa.field('point_y', pa.float64()),
    pa.field('point_z', pa.float64()),
    pa.field('echo_intensity', pa.float64()),
    pa.field('return_number', pa.int32()),
    pa.field('number_of_returns', pa.int32()),
    pa.field('normal_x', pa.float64()),
    pa.field('normal_y', pa.float64()),
    pa.field('normal_z', pa.float64()),
    pa.field('point_weight', pa.float64()),
    pa.field('is_leaf', pa.bool_())
])

### HELPER FUNCTIONS ###
# Commonly used functions that offer small utilities for components of other scripts.

DASK_CLIENT = None

def _start_dask_client(memory_limit='4GB', n_workers=None, threads_per_worker=1):
    global DASK_CLIENT
    from dask.distributed import Client, LocalCluster
    import psutil
    import os

    try:
        running_dask_client = get_client()
        if running_dask_client is not None and not running_dask_client.status == 'closed':
            _close_dask_client(client=running_dask_client)
    except ValueError:
        running_dask_client = None

    if n_workers is None:
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK', psutil.cpu_count(logical=False)))

    # On some HPCs, Dask's LocalCluster expects memory_limit as an int (bytes), not a string.
    # Try to convert memory_limit to int if it's a string ending with 'GB' or 'MB'.
    if isinstance(memory_limit, str):
        mem = memory_limit.upper().strip()
        if mem.endswith('GB'):
            memory_limit = int(float(mem[:-2]) * 1024 ** 3)
        elif mem.endswith('MB'):
            memory_limit = int(float(mem[:-2]) * 1024 ** 2)
        else:
            try:
                memory_limit = int(memory_limit)
            except Exception:
                pass  # fallback to string if conversion fails

    cluster = LocalCluster(n_workers=n_workers, threads_per_worker=threads_per_worker, memory_limit=memory_limit)
    DASK_CLIENT = Client(cluster)
    return DASK_CLIENT

def _close_dask_client(client=None):
    global DASK_CLIENT
    if client is None:
        client = DASK_CLIENT
    if client is not None and not client.status == 'closed':
        client.shutdown()
        client.close()
        gc.collect()
        DASK_CLIENT = None

    # Delete any worker temp scratch space if it is present
    tmp_dir = os.environ.get("TMPDIR", "/tmp")
    dask_scratch_space = os.path.join(tmp_dir, "dask-scratch-space")
    if os.path.isdir(dask_scratch_space):
        shutil.rmtree(dask_scratch_space, ignore_errors=True)
        print(f"Deleted Dask worker scratch space at {dask_scratch_space}")

def _gen_dataframe(schema):
    fields = []
    for field in schema:
        dtype = field.type.to_pandas_dtype()
        if np.issubdtype(dtype, np.integer):
            dtype = 'Int64'
        fields.append((field.name, dtype))
    df = pd.DataFrame({name: pd.Series(dtype=dtype) for name, dtype in fields})
    return df

def compute_normals_weights_from_points(points, voxel_size=10.0, knn=6):
    from tqdm import tqdm
    import open3d as o3d
    import numpy as np
    from collections import defaultdict
    from joblib import Parallel, delayed
    """
    Get normals and weights from points.
    
    INPUTS:
        points: A numpy array of points
        knn: The number of nearest neighbours to consider

    OUTPUTS:
        normals: The normals of the points
        weights: The weights of the points
    """
    if len(points) < knn:
        return np.ones(len(points)), np.ones(len(points))

    # Fast voxel indexing
    print("Initialising voxels")
    voxel_indices = (points / voxel_size).astype(int)
    voxel_keys = np.char.add(
    np.char.add(voxel_indices[:, 0].astype(str), '_'),
    np.char.add(voxel_indices[:, 1].astype(str), '_')
    )
    voxel_keys = np.char.add(voxel_keys, voxel_indices[:, 2].astype(str))

    voxel_dict = defaultdict(list)

    def add_to_voxel_dict(idx_key_tuple):
        idx, key = idx_key_tuple
        return key, (idx, points[idx])

    # Use joblib to parallelize the assignment
    results = Parallel(n_jobs=-1)(
        delayed(add_to_voxel_dict)(item) for item in tqdm(enumerate(voxel_keys), total=len(voxel_keys), desc="Indexing voxels")
    )

    for key, value in results:
        voxel_dict[key].append(value)

    # # Compute the normals with open3d
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(points)
    # voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=10.0)

    normals = np.zeros((len(points), 3))
    weights = np.zeros(len(points))

    def process_voxel(voxel_data):
        idxs, pts = zip(*voxel_data)
        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(pts)
        voxel_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        distances = np.asarray(voxel_pcd.compute_nearest_neighbor_distance())
        voxel_normals = np.asarray(voxel_pcd.normals)
        voxel_weights = 1 / (distances + 1e-9)
        return idxs, voxel_normals, voxel_weights

    for key in tqdm(voxel_dict.keys(), desc="Processing voxels"):
        idxs, voxel_normals, voxel_weights = process_voxel(voxel_dict[key])
        for i, idx in enumerate(idxs):
            normals[idx] = voxel_normals[i]
            weights[idx] = voxel_weights[i]

    return normals, weights

    def unique_key_from_voxel_centre(voxel_centre):
        return f"{voxel_centre[0]}_{voxel_centre[1]}_{voxel_centre[2]}"

    # Helper to get voxel id for a point
    points_array = np.asarray(pcd.points)
    for idx, point in tqdm(enumerate(points_array), total=len(points_array), desc="Assigning points to voxels"):
        voxel_id = unique_key_from_voxel_centre(voxel_grid.get_voxel(point))
        if voxel_id not in voxel_dict:
            voxel_dict[voxel_id] = []
        voxel_dict[voxel_id].append((idx, point))

    # Iterate over voxels with progress bar
    for voxel, points in tqdm(voxel_dict.items(), total=len(voxel_dict), desc="Estimating normals/weights per voxel"):
        idxs, points = zip(*points)

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(points)
        voxel_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        distances = np.asarray(voxel_pcd.compute_nearest_neighbor_distance())
        voxel_normals = np.asarray(voxel_pcd.normals)
        voxel_weights = 1 / (distances + 1e-9)

        for i, idx in enumerate(idxs):
            normals[idx] = voxel_normals[i]
            weights[idx] = voxel_weights[i]

    return normals, weights

# Create a unique ID for a voxel
def create_voxel_id(voxel_size, x, y, z):
    """
    Create a unique ID for a voxel.
    
    INPUTS:
        nd_array: containing [voxel_size, x, y, z]

    OUTPUTS:
        voxel_id: A unique ID for the voxel
    """
    # Create a string representation of the voxel parameters
    voxel_string = f'{voxel_size}_{x}_{y}_{z}'

    # Encode the string and hash it using FNV-1a
    voxel_id = fnv1a_32(voxel_string.encode())
    # print(f"Created unique voxel_id: {voxel_id} for voxel {voxel_string}")

    return voxel_id

# Create a pandas dataframe from a pyarrow schema
def create_df_from_schema(schema):
    """
    Create a pandas dataframe from a pyarrow schema.
    
    INPUTS:
        schema: A pyarrow schema
        
    OUTPUTS:
        df: A pandas dataframe with the same columns and dtypes as the schema
    """
    new_df = pd.DataFrame(columns=schema.names)

    for field in schema:
        new_df[field.name] = new_df[field.name].astype(field.type.to_pandas_dtype())

    return new_df

# Calculate lambda_1
def calculate_lambda_1(average_leaf_area, voxel_size):
    """
    Calculate lambda_1 for a given voxel size.
    """
    lambda_1 = average_leaf_area / (voxel_size ** 3)

    return lambda_1

# Calculate the effective path length z
def effective_path_length_z(z, lambda_1):
    """
    Calculate the effective path length z.
    
    INPUTS:
        free_path_lengths: The free path lengths
        lambda_1: The calculated lambda_1
    
    OUTPUTS:
        z: The effective path length z
    """
    z = z.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = (lambda_1 * z) < 1
        eff_path_length_zs = np.full_like(z, fill_value=np.nan, dtype=np.float64)
        z[valid_mask] = -np.log(1 - lambda_1 * z[valid_mask]) / lambda_1
        
    return eff_path_length_zs

def calculate_LIAD(normals, weights, num_bins=18):
    """
    Calculate the Leaf Angle Distribution (LAD) for a set of normals and weights.
    
    INPUTS:
        normals: A numpy array of normals
        weights: A numpy array of weights
        num_bins: The number of bins to use for the histogram
        
    OUTPUTS:
        bin_centres: The bin centres
        LIAD_values: The LIAD values
        angles: The angles
    """
    # Compute the angles
    angles = np.arccos(np.dot(normals, np.array([0, 0, 1])))
    angles = np.where(angles > np.pi / 2, np.pi - angles, angles)
    angles = np.degrees(angles)

    # Compute LIAD for each voxel
    if len(angles) == 0 or np.all(np.isnan(angles)):
        return np.array([]), np.array([]), np.array([])

    if len(weights) == 0:
        weights = np.ones_like(angles)

    # Remove NaN angles and align weights
    valid_mask = ~np.isnan(angles)
    angles = angles[valid_mask]
    weights = weights[valid_mask].flatten()

    if len(angles) == 0:
        return np.array([]), np.array([]), np.array([])

    # Compute the histogram
    hist, bin_edges = np.histogram(angles, bins=num_bins, range=(0, 90), weights=weights)
    total_weight = np.sum(hist)
    if total_weight > 0:
        LIAD_values = hist / total_weight
    else:
        LIAD_values = np.zeros(num_bins)

    # Compute the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres, LIAD_values, angles

def calculate_LIAD_old(points, knn=6, radius=0.1, max_nn=10, num_bins=18):
    """
    Calculate the (LIAD) for a set of points.
    
    INPUTS:
        points: A numpy array of points
        knn: The number of nearest neighbours to consider
        radius: The radius to consider
        max_nn: The maximum number of nearest neighbours to consider
        num_bins: The number of bins to use for the histogram

    OUTPUTS:
        bin_centres: The bin centres
        LIAD_values: The LIAD values
        angles: The angles

    NOTE:
    1) Get Normals
    2) Weight based on proximity to other points
    3) Compute angles against up ([0, 0, 1]).
        Normalise to max 90 degrees.
    4) Bin anlges into 5 degree bins, and find weight ratio per bin against total weight 
        This varies from mesh-based approaches, since we use weighted volume instead of area
    5) Return bin centres, LIAD values, and angles
    """

    # Convert points to numpy
    points_copy = np.copy(points)
    points_copy = points_copy[~np.isnan(points_copy).any(axis=1)]

    # Compute the point density weights
    if len(points_copy) < knn:
        weights = np.ones(len(points_copy))
    else:
        nbrs = NearestNeighbors(n_neighbors=knn).fit(points_copy)
        distances, _ = nbrs.kneighbors(points_copy)
        # Inverse of the distance to the k nearest neighbours as weight
        weights = 1 / (distances[:, -1] + 1e-9) # Add a small value to avoid division by zero
    
    # Compute the normals
    if len(points_copy) < max_nn:
        return np.array([]), np.array([]), np.array([])
    
    # Create a KDTree for efficient neighbour search
    tree = NearestNeighbors(radius=radius, n_neighbors=max_nn).fit(points_copy)

    # Precompute neighbors for all points
    neighbors_indices = tree.radius_neighbors(points_copy, return_distance=False)

    # Initialize normals array
    normals = np.full((len(points_copy), 3), np.nan)

    for i, indices in enumerate(neighbors_indices):
        if len(indices) < max_nn and len(indices) < 3 and len(points_copy) > 0:
            continue

        # Ensure indices are within the bounds of 'points'
        valid_indices = [idx for idx in indices if 0 <= idx < len(points_copy)]
        if len(valid_indices) < max_nn:
            continue
        
        neighbours = points_copy[valid_indices]


        # Compute the covariance matrix
        covariance_matrix = np.cov(neighbours, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Get the normal
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normals[i] = normal
    
    # Compute the angles
    angles = np.arccos(np.dot(normals, np.array([0, 0, 1])))
    angles = np.where(angles > np.pi / 2, np.pi - angles, angles)
    angles = np.degrees(angles)

    # Compute LIAD for each voxel
    if len(angles) == 0 or np.all(np.isnan(angles)):
        pass
    
    if len(weights) == 0:
        weights = np.ones_like(angles)

    # Remove NaN angles and align weights
    valid_mask = ~np.isnan(angles)
    angles = angles[valid_mask]
    weights = weights[valid_mask]

    if len(angles) == 0:
        pass

    total_weights = np.sum(weights)
    if total_weights > 0:
        weights /= total_weights
    else:
        weights = np.ones_like(angles)

    # Compute the histogram
    hist, bin_edges = np.histogram(angles, bins=num_bins, range=(0, 90), weights=weights) # Switch 90 to np.pi to include upward angles
    total_weight = np.sum(hist)
    if total_weight > 0:
        LIAD_values = hist / total_weight
    else:
        LIAD_values = np.zeros(num_bins)

    # Compute the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres, LIAD_values, angles


# Calculate the G function mean
def calculate_G(viewing_angles, bin_centres, LIAD_values, epsilon=1e-9):
    """
    Calculate the G function mean.
    
    INPUTS:
        viewing_angle: The viewing angles
        bin_centres: The bin centres
        LIAD_values: The LIAD values
    
    OUTPUTS:
        G_mean: The G function mean
    """
    # Check for empty arrays
    if len(viewing_angles) == 0 or len(bin_centres) == 0 or len(LIAD_values) == 0:
        return np.nan
    
    # # Normalise LIAD
    # total_LIAD = LIAD_values.sum()
    # LIAD_norm = LIAD_values / total_LIAD if total_LIAD > 0 else LIAD_values
    LIAD_norm = LIAD_values

    # Ensure angles are clipped
    viewing_angles = np.clip(viewing_angles, epsilon, 90)
    bin_centres = np.clip(bin_centres, epsilon, 90)

    ### A(angle, leaf_angle)  ####
    theta_a = np.radians(viewing_angles)
    theta_b = np.radians(bin_centres)

    # Calculate the cotangent of the angles
    cos_theta_a = np.cos(theta_a)
    cot_theta_a = 1 / np.tan(theta_a)
    cos_theta_b = np.cos(theta_b)
    cot_theta_b = 1 / np.tan(theta_b)

    #
    cos_outer = np.outer(cos_theta_a, cos_theta_b)
    cot_outer = np.outer(cot_theta_a, cot_theta_b)

    A = np.zeros_like(cos_outer)
    mask_greater_1 = np.abs(cot_outer) > 1

    A[mask_greater_1] = cos_outer[mask_greater_1]

    inside = np.clip(cot_outer[~mask_greater_1], -1, 1)
    psi = np.arccos(inside)
    factor = 1.0 + (2.0 / np.pi) * (np.tan(psi) - psi)

    A[~mask_greater_1] = factor * cos_outer[~mask_greater_1]

    # Calculate the G function mean for all angles
    delta_bin = np.radians(bin_centres[1] - bin_centres[0])
    G = A @ LIAD_norm # (LIAD_norm * delta_bin)

    return G

### LAD/PAD Functions ###
def CI_adjusted(AD, CI):
    """
    This function takes an ADeff and CI and returns the AD.
    Where, AD = ADeff/CI
    """
    AD = AD/CI
    return AD

def nan_zero_to_default_G_CI(G, CI):
    """
    This function takes an array and a default value and returns the array with nans replaced by the default value.
    """
    if isinstance(G, np.ndarray):
        G = np.where(np.logical_or(np.isnan(G), G==0), 0.5, G)

    if isinstance(CI, np.ndarray):
        CI = np.where(np.logical_or(np.isnan(CI), CI==0), 1.0, CI)
    
    return G, CI

# Beer-Lambert Pimont et al. 2018, eq. 5
def BL_pimont_2018(P, mean_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate density using Beer-Lambert (Pimont et al. 2018), equation 5.
        BL = -log(P) / δ̄


    Calculate PAD by passing I/G values that use all hits, 
    and LAD by passing I/G values that use leaf hits only

    INPUTS:
        P:                  Pgap (probability gap fraction). Can be calculated in various methods.
        G:                  A provided G_mean value or default 0.5
        CI:                 A provided CI value or default 1.0
        mean_path_length:   Provided mean path length of voxel
        epsilon:            A condition to avoid issues with zero division

    OUTPUTS:
        ADeff:                 The calculated Leaf/Plant Area Density without corrected for CI

    """
    ### CI IS NOT CURRENTLY USED, BUT COULD BE LATER ###

    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        ADeff = np.where(
            (~np.isnan(P) & ~np.isnan(mean_path_length)),
            -(np.log(P) / (G * mean_path_length)),
            np.nan
        )  

        AD = np.where(
            (~np.isnan(ADeff) & (CI != 0)),
            ADeff / CI,
            np.nan
        )
    
    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def BL_EPL_UEPL_pimont_2018(I, mean_eff_path_length, var_eff_path_length, num_rays, G=0.5, epsilon=1e-9, CI=1.0):
    """
    Calculate density using Beer-Lambert (Pimont et al. 2018) with Effective Path Length, equation 25.
        Λ̂ = {
          -1 / δ̄ₑ * (log(1 - I) + I / (2N(1 - I)))      when I < 1
          log(2N + 2) / δ̄ₑ                              when I = 1
        }
    
    &

    Calculate the unbiased effective path length (UEPL) (Pimont et al. 2018, eq. 27), based on the shared EPL value before G correction
        Λ̅₂ = 1 / aₑ * (1 - sqrt(1 - 2 * aₑ * Λ̅))
        
        where:
        Λ̅₂ is the second Lambda with a bar over it
        aₑ is a subscripted 'a' with 'e'
        sqrt represents the square root

    Calculate PAD by passing I values that use all hits,
    and LAD by passing I values that use leaf hits only

    INPUTS:
        I:              A numpy array of Relative Density Indexes (num_hits/num_rays)
        mean_eff_path_length:   A numpy array of mean_eff_path_length
        num_rays:       A numpy array of num_rays
        epsilon:        A condition to avoid issues with zero division

    OUTPUTS:
        ADeff_EPL:          The calculated density, without correcting for CI from EPL
        ADeff_UEPL:         The calculated density, without correcting for CI from UEPL

    """
    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        # Check for nans in inputs
        valid_mask = (
            ~np.isnan(I) & 
            ~np.isnan(mean_eff_path_length) & 
            np.logical_and(~np.isnan(num_rays), num_rays > 0)
        )

        # Split I < 1 and I == 1 values to handle separate calculations
        I_lt_1_mask = I < 1
        I_eq_1_mask = I == 1

        # Calculate ADeff_EPL (L or P depending on inputs)
        ADeff_EPL = np.where(
            np.logical_and(I_lt_1_mask, valid_mask),    # I < 1
            -(1 / mean_eff_path_length) * (np.log(1 - I) + (I / (2 * num_rays * (1 - I)))),
            np.where(
                np.logical_and(I_eq_1_mask, valid_mask),    # I == 1
                np.log(2 * num_rays + 2) / mean_eff_path_length,
                np.nan          # Other
            )
        )

        # Calculate ADeff_UEPL (L or P depending on inputs)
        valid_UEPL_mask = (
            np.logical_and(~np.isnan(ADeff_EPL), (ADeff_EPL > 0)) &
            (mean_eff_path_length > 0) & 
            (var_eff_path_length > 0)
        ) 
        a_e = np.where(
            valid_UEPL_mask,
            var_eff_path_length / mean_eff_path_length,
            np.nan
        )
        ADeff_UEPL = np.where(
            valid_UEPL_mask,
            1 / a_e * (1 - np.sqrt(1 - 2 * a_e * ADeff_EPL)),
            np.nan
        )

        # Correct both ADeff values with G
        ADeff_EPL = np.where(
            ~np.isnan(ADeff_EPL) & (G > 0),
            ADeff_EPL / G,
            np.nan
        )
        ADeff_UEPL = np.where(
            ~np.isnan(ADeff_UEPL) & (G > 0),
            ADeff_UEPL / G,
            np.nan
        )

        AD_EPL = ADeff_EPL / CI
        AD_UEPL = ADeff_UEPL / CI

    except Exception as e:
        print(f"Error: {e}")
        return np.nan, np.nan
    
    return AD_EPL, AD_UEPL

def MCF_beland_2011(I, mean_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Modified Contact Frequency (MCF) using the formula from Pimont et al. 2018 (eq. 8).

    λ̃ = I / z̅  (See paper for more details about this simplification)
    and corrected for G (i.e. / G)
    
    INPUTS:
        mean_free_path_lengths: The mean z value
        I: = 1.0 - pgap
        G: The G function value
        epsilon: A condition to avoid issues with zero division

    OUTPUTS:
        ADeff: The calculated Mean Crown Fraction
    """
    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        # Calculate MCF
        AD = I / (mean_free_path_length * G) / CI

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def MCF_corrected_beland_2014(mean_free_path_length, I, lambda_1, mean_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the corrected Modified Contact Frequency (MCF) using the formula from Pimont et al. 2018 (eq. 9).

    λ̃ = I / z̅ * (1 + λ₁ * δ̄)  (See paper for more details about this simplification)
    and corrected for G (i.e. / G)
    
    INPUTS:
        mean_free_path_lengths: The mean z value
        I: The relative density index (num_hits/num_rays)
        lambda_1: The lambda_1 value
        mean_path_lengths: The mean path length
        G: The G function value
        epsilon: A condition to avoid issues with zero division

    OUTPUTS:
        ADeff: The calculated density from corrected Modified Contact Frequency 
    """
    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        valid_mask = (
            (mean_free_path_length > epsilon) & 
            (I > 0) & (I < 1) &
            (lambda_1 > 0) &
            (mean_path_length > 0)
        )

        ADeff = np.where(
            valid_mask,
            -1 * (lambda_1 * mean_path_length * I) / (np.log(1 - lambda_1 * mean_path_length) * mean_free_path_length),
            np.nan
        )

        # Correct for G
        ADeff = np.where(
            ~np.isnan(ADeff) & (G > 0),
            ADeff / G,
            np.nan
        )

        AD = np.where(
            ~np.isnan(ADeff) & ~np.isnan(CI) & (CI > 0),
            ADeff / CI,
            np.nan
        )

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def MLE_pimont_2019(woody_vol_proportion, num_hits, num_leaf_hits, sum_eff_free_path_length_hit_leaf, sum_eff_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Maximum Likelihood Estimation (MLE) using the formula from Pimont et al. 2018 (eq. 10).
    λ̃ = (1 - I) / (z̅ * G)  (See paper for more details about this simplification)

    For LAD, pass in the sum_hits_effective_path_length array for leaf only.
    
    """

    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        leaf_fraction = np.where(
            np.logical_and(num_hits > 0, num_leaf_hits > 0), 
            np.where(num_hits == num_leaf_hits, 1, num_leaf_hits / num_hits),
            0
        )

        valid_mask = (
            (woody_vol_proportion > 0) &
            (leaf_fraction > 0) &
            (G > 0) &
            (sum_eff_free_path_length_hit_leaf > 0) &
            (sum_eff_free_path_length > 0) &
            (num_hits > 0)
        )

        ADeff = np.where(
            valid_mask,
            (woody_vol_proportion * leaf_fraction / (G * sum_eff_free_path_length)) * (num_hits - sum_eff_free_path_length_hit_leaf / sum_eff_free_path_length),
            np.nan
        )

        AD = np.where(
            ~np.isnan(ADeff) & ~np.isnan(CI) & (CI > 0),
            ADeff / CI,
            np.nan
        )

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def MLE_soma_2021(num_hits, num_leaf_hits, sum_free_path_length_hit_leaf, sum_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Maximum Likelihood Estimation (MLE) using the formula from Soma et al. 2021 (eq. 10).
    λ̃ = (1 - I) / (z̅ * G)  (See paper for more details about this simplification)
    
    """

    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        leaf_fraction = np.where(
            np.logical_and(num_hits > 0, num_leaf_hits > 0), 
            np.where(num_hits == num_leaf_hits, 1, num_leaf_hits / num_hits),
            0
        )

        valid_mask = (
            (leaf_fraction > 0) &
            (G > 0) &
            (sum_free_path_length_hit_leaf > 0) &
            (sum_free_path_length > 0) &
            (num_hits > 0)
        )

        ADeff = np.where(
            valid_mask,
            (leaf_fraction / (G * sum_free_path_length)) * (num_hits - sum_free_path_length_hit_leaf / sum_free_path_length),
            np.nan
        )

        AD = np.where(
            ~np.isnan(ADeff) & ~np.isnan(CI) & (CI > 0),
            ADeff / CI,
            np.nan
        )

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def PAD_MLE(value):
    ### INSERT ###
    pass

# Multi-return MLE from AMAPvox
def MLE_vincent_2021(sum_ba_hit, sum_pl_all, G=0.5, CI=1.0, bias_corr=None):
    """
    Calculate the Maximum Likelihood Estimation (MLE) using the formula for multi-geometry correction.
    This code requires:
    - sum_ba_hit: Sum of expected beam areas for unique pulses at voxel centre that hit vegetation elements.         
    - sum_pl_all: Sum of expected beam areas for unique pulses at voxel centre.
    - bias_corr: The hit vs explored ratio outlined in Vincent 2021 (i.e. -(sum(ba_all * fraction_enter)/num_rays))
    - G: G function
    - CI: Clumping index

    To apply vegetation element size corrections, use the sum_eff_free_path_length_hit and sum_eff_path_length_exit values in the sum_fpl_h and sum_pl_e
    """

    k_hat = sum_ba_hit / sum_pl_all

    if bias_corr is not None:
        k_hat -= bias_corr

    return k_hat / G / CI       # NOT CONVINCED ON G CORRECTION HERE

# def LAD_MLE_geom_corr(num_hits,
#                       beam_areas_hit, beam_areas_all,       # scannerÃÂvoxel-centre ranges
#                       fpl_all,                      # free-path lengths
#                       G_leaf, CI=1.0,
#                       k1=0.0, bias_corr=True, eps=1e-9):

#     if num_hits == 0 or beam_areas_all.size == 0:
#         return np.nan

#     if k1 > 0:                                    # element-size bias
#         fpl_all = -np.log(1 - k1*np.clip(fpl_all, 0, 1-eps)) / k1

#     k_hat = beam_areas_hit.sum() / (beam_areas_all * fpl_all).sum()

#     if bias_corr:
#         N = beam_areas_all.size
#         k_hat -= (beam_areas_all.sum()/N) * (beam_areas_hit.sum() /
#                                     (beam_areas_all * fpl_all).sum())

#     return (k_hat / G_leaf) / CI


# -----------------------------------------------------------------
#  ENERGY weighting  (Bai 2024  + Vincent beam area)
# -----------------------------------------------------------------
def LAD_MLE_energy_corr(alpha_hit, alpha_all,
                        beam_areas_hit, beam_areas_all,
                        fpl_all,
                        G_leaf, CI=1.0,
                        k1=0.0, bias_corr=True, eps=1e-9):
    if alpha_hit.sum() <= eps or alpha_all.sum() <= eps:
        return np.nan

    if k1 > 0:
        fpl_all = -np.log(1 - k1*np.clip(fpl_all, 0, 1-eps)) / k1

    k_hat = (beam_areas_hit * alpha_hit).sum() / (beam_areas_all * alpha_all * fpl_all).sum()

    if bias_corr:
        N = beam_areas_all.size
        k_hat -= ((beam_areas_all * alpha_all).sum()/N) * \
                 ((beam_areas_hit * alpha_hit).sum() /
                  (beam_areas_all * alpha_all * fpl_all).sum())

    return (k_hat / G_leaf) / CI


# Functions used for voxel ray intersections

# Find viewing angles of the rays in comparison with straight up
# Normalise between 0 and 90 degrees
def find_viewing_angles(directions, reference_vector=np.array([0, 0, 1])):
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized_directions = directions / dir_norms
    dot_products = np.dot(normalized_directions, reference_vector)
    cos_thetas = np.clip(dot_products, -1, 1)
    viewing_angle = np.degrees(np.arccos(cos_thetas))
    viewing_angle = np.where(viewing_angle > 90, 180 - viewing_angle, viewing_angle)  # Adjust angles over 90 degrees
    return viewing_angle

# Function to traverse the voxels and find ray intersections
def traverse_voxels(voxel_references, ray_partition, voxels_per_chunk, temp_dir, debug=False, epsilon=1e-6):
    import logging
    logging.basicConfig(level=logging.INFO)

    if ray_partition.empty:
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)
    
    # Prep ray information
    leg_ids = ray_partition['leg_id'].values
    ray_ids = ray_partition['ray_id'].values
    origins = np.asarray(ray_partition[['origin_x', 'origin_y', 'origin_z']].values)
    directions = np.asarray(ray_partition[['direction_x', 'direction_y', 'direction_z']].values)
    points = np.asarray(ray_partition[['point_x', 'point_y', 'point_z']].values)
    normals = np.asarray(ray_partition[['normal_x', 'normal_y', 'normal_z']].values)
    point_weights = np.asarray(ray_partition['point_weight'].values)
    echo_intensities = np.asarray(ray_partition['echo_intensity'].values)
    is_leaf = np.asarray(ray_partition['is_leaf'].values)
    return_numbers = np.asarray(ray_partition['return_number'].values)
    number_of_returns = np.asarray(ray_partition['number_of_returns'].values)

    # print(f"Max Return Number: {np.nanmax(return_numbers)} Max Number of Returns: {np.nanmax(number_of_returns)} Num of points: {np.count_nonzero(~np.isnan(points).any(axis=1))}")

    num_rays = len(ray_partition)
    num_voxels = len(voxel_references)

    voxel_ids = voxel_references['voxel_id'].values
    voxel_sizes = voxel_references['voxel_size'].values
    voxel_centres = voxel_references[['voxel_cx', 'voxel_cy', 'voxel_cz']].values

    leg_ids = leg_ids[np.newaxis, :]
    ray_ids = ray_ids[np.newaxis, :]
    is_leaf = is_leaf[np.newaxis, :]
    origins = origins[np.newaxis, :, :]
    directions = directions[np.newaxis, :, :]
    points = points[np.newaxis, :, :]    
    normals = normals[np.newaxis, :, :]
    point_weights = point_weights[np.newaxis, :]
    echo_intensities = echo_intensities[np.newaxis, :]
    return_numbers = return_numbers[np.newaxis, :]
    number_of_returns = number_of_returns[np.newaxis, :]

    del voxel_references, ray_partition
    # gc.collect()

    ### TEST DATA ###
    # voxel_mins = np.array([
    #     [[0, 0, 0]],
    #     [[1, 1, 1]],
    #     [[2, 2, 2]]
    # ])  # Shape (3, 1, 3)

    # voxel_maxs = np.array([
    #     [[1, 1, 1]],
    #     [[2, 2, 2]],
    #     [[3, 3, 3]]
    # ])  # Shape (3, 1, 3)

    # origins = np.array([
    #     [[0.5, 0.5, -1],
    #      [1.5, 1.5, -1],
    #      [2.5, 2.5, -1]]
    # ])  # Shape (1, 3, 3)

    # directions = np.array([
    #     [[0, 0, 1],
    #      [0, 0, 1],
    #      [0, 0, 1]]
    # ])  # Shape (1, 3, 3)

    # Expected results:
    # - Ray 1 intersects voxel 1
    # - Ray 2 intersects voxel 2
    # - Ray 3 intersects voxel 3

    def broadcasted_ray_tracing(voxel_centres, voxel_sizes, origins, directions):
        # Establish empty mask for no-hit cases
        mask = np.zeros((voxel_centres.shape[0], origins.shape[1]), dtype=bool)

        # Use half the diagonal of the voxel as the radius, plus epsilon
        # This ensures the sphere fully contains the voxel, including edge cases
        # Use a slightly larger radius for pre-check to ensure all possible intersections are included
        voxel_radii = voxel_sizes * np.sqrt(3) * 0.5 + 0.05  # Diagonal radius for a cube, slightly expanded

        oc = origins - voxel_centres
        b = 2.0 * np.sum(oc * directions, axis=2)
        c = np.sum(oc * oc, axis=2) - voxel_radii**2
        discriminant = b**2 - 4 * c
        hit = discriminant >= -epsilon

        if np.any(hit):
            del oc, b, c, discriminant
            # Find min and max bounds
            voxel_mins = voxel_centres - (voxel_sizes[..., None] / 2 - epsilon)
            voxel_maxs = voxel_centres + (voxel_sizes[..., None] / 2 + epsilon)

            # Find indices where hit is True
            voxel_idx, ray_idx = np.nonzero(hit)

            del hit, voxel_centres, voxel_sizes

            # Select only the voxel/ray pairs where hit is True
            selected_voxel_mins = voxel_mins[voxel_idx, 0]
            selected_voxel_maxs = voxel_maxs[voxel_idx, 0]
            selected_origins = origins[0, ray_idx]
            selected_directions = directions[0, ray_idx]

            del voxel_mins, voxel_maxs, origins, directions

            # Optimized ray-AABB intersection for masking only
            # To avoid division by zero, set very small values for zero directions
            small_epsilon = 1e-9
            small_dir = np.abs(selected_directions) <= small_epsilon
            selected_directions = np.where(
                small_dir,
                np.where(selected_directions == 0, small_epsilon, np.sign(selected_directions) * small_epsilon),
                selected_directions
            )
            inv_directions = 1.0 / selected_directions

            del selected_directions

            # Compute intersection parameters
            t1 = (selected_voxel_mins - selected_origins) * inv_directions
            t2 = (selected_voxel_maxs - selected_origins) * inv_directions

            del selected_voxel_mins, selected_voxel_maxs, selected_origins, inv_directions

            # Find entry and exit points
            t_enter = np.max(np.minimum(t1, t2), axis=1)
            t_exit = np.min(np.maximum(t1, t2), axis=1)

            # Ray intersects if t_enter <= t_exit and t_exit >= 0
            valid = (t_enter <= t_exit + epsilon) & (t_exit >= -epsilon)

            # Set mask for valid voxel-ray pairs
            mask[voxel_idx[valid], ray_idx[valid]] = True

        return mask

    # Calculate mask, t_enter, and t_exit for max voxels that fit into memory
    masks = {}

    temp_dir = tempfile.mkdtemp(dir=temp_dir)

    # Generate a unique ID for the process
    process_id = uuid.uuid4().hex

    # print(f"Process {process_id}: Start {num_rays} rays, {num_voxels} voxels, in ({int(np.ceil(num_voxels / voxels_per_chunk))}) chunks.")

    # Find unique ray_ids and their indices
    _, unique_ray_idx, inverse_ray_idx = np.unique(ray_ids, return_index=True, return_inverse=True)
    # Get unique origins and directions
    unique_origins = origins[:, unique_ray_idx, :]
    unique_directions = directions[:, unique_ray_idx, :]

    for i in range(0, voxel_centres.shape[0], voxels_per_chunk):
        chunk_centres = voxel_centres[i:i + voxels_per_chunk, np.newaxis, :]
        chunk_sizes = voxel_sizes[i:i + voxels_per_chunk, np.newaxis]
        chunk_mask_unique = broadcasted_ray_tracing(
            voxel_centres=chunk_centres,
            voxel_sizes=chunk_sizes,
            origins=unique_origins,
            directions=unique_directions
        )

        # Map mask back to all rays using inverse_ray_idx
        chunk_mask = chunk_mask_unique[:, inverse_ray_idx]

        # Save chunk_mask and chunk_indices to disk with unique filenames
        chunk_mask_filename = os.path.join(temp_dir, f"chunk_mask_{i}_{process_id}.npy")
        np.save(chunk_mask_filename, chunk_mask)
        masks[i] = [chunk_mask_filename, chunk_mask.dtype, chunk_mask.shape]

        del chunk_mask, chunk_mask_unique, chunk_centres, chunk_sizes
        gc.collect()

    # Flatten mask and retrieve idx for rays and voxels
    # Combine masks into single array for further processing
    chunk_masks = []
    for key in sorted(masks.keys()):
        filename, dtype, shape = masks[key]
        chunk = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
        chunk_masks.append(chunk)
    mask = np.concatenate(chunk_masks, axis=0)

    # Flatten mask and retrieve idx for rays and voxels
    voxel_ref_idx, _, ray_ref_idx = np.nonzero(mask[:,:])

    # Delete temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)

    if len(voxel_ref_idx) == 0:
        del mask, voxel_ref_idx, ray_ref_idx
        del voxel_ids, voxel_sizes, voxel_centres, leg_ids, ray_ids, is_leaf, points, origins, directions
        gc.collect()
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

    del mask, chunk_masks, chunk
    gc.collect()

    # Flatten all values to match the mask
    filtered_voxel_ids = voxel_ids[voxel_ref_idx].reshape(-1)
    filtered_voxel_sizes = voxel_sizes[voxel_ref_idx]
    filtered_voxel_centres = voxel_centres[voxel_ref_idx].reshape(-1, 3)

    filtered_leg_ids = leg_ids[:, ray_ref_idx].reshape(-1)
    filtered_ray_ids = ray_ids[:, ray_ref_idx].reshape(-1)
    filtered_is_leaf = is_leaf[:, ray_ref_idx].reshape(-1)
    filtered_points = points[:, ray_ref_idx, :].reshape(-1, 3)

    filtered_normals = normals[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_point_weights = point_weights[:, ray_ref_idx].reshape(-1)
    filtered_origins = origins[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_directions = directions[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_echo_intensities = echo_intensities[:, ray_ref_idx].reshape(-1)
    filtered_return_numbers = return_numbers[:, ray_ref_idx].reshape(-1)
    filtered_number_of_returns = number_of_returns[:, ray_ref_idx].reshape(-1)

    # Backup points that are not in ray_ref_idx for each voxel_ref_idx (i.e., those not selected by the mask)
    # For each voxel_ref_idx, check rays not in ray_ref_idx to see if any points should have been included
    if debug:
        all_indices = np.arange(points.shape[1])
        for v_idx in np.unique(voxel_ref_idx):
            # Find all ray indices for this voxel
            mask_voxel = voxel_ref_idx == v_idx
            selected_ray_idx = ray_ref_idx[mask_voxel]
            not_selected_idx = np.setdiff1d(all_indices, selected_ray_idx)
            # Get the voxel min/max for this voxel
            voxel_min = (voxel_centres[v_idx] - (voxel_sizes[v_idx] / 2 - epsilon))
            voxel_max = (voxel_centres[v_idx] + (voxel_sizes[v_idx] / 2 + epsilon))
            # Get points for rays not selected
            backup_points = points[:, not_selected_idx, :].reshape(-1, 3)
            should_be_in_voxel = np.all((backup_points >= (voxel_min - epsilon)) & (backup_points <= (voxel_max + epsilon)), axis=1)
            if np.any(should_be_in_voxel):
                print(f"Voxel min: {voxel_min} max: {voxel_max}: Points that should be in voxel but not selected:", backup_points[should_be_in_voxel])

    # Find viewing angles based on the filtered directions
    filtered_viewing_angles = find_viewing_angles(directions=filtered_directions)

    # Cleanup memory
    del voxel_ids, voxel_sizes, voxel_centres, leg_ids, ray_ids, is_leaf, points, origins, directions, echo_intensities

    # Filter out points that are within each respective voxel  
    # Create hit_type variable
    # 0: unbound (point is nan)
    # 1: previous hit (point < min voxel)
    # 2: current hit (point in voxel)
    # 3: post hit (point > max voxel)
    filtered_voxel_mins = filtered_voxel_centres - (filtered_voxel_sizes[:, np.newaxis] / 2 - epsilon)
    filtered_voxel_maxs = filtered_voxel_centres + (filtered_voxel_sizes[:, np.newaxis] / 2 + epsilon)

    # Filter out rays which have previously hit a voxel (and therefore do not continue into this next one)
    # Also keep any NAN points (i.e. unbound rays)
    # distance_to_exit = np.linalg.norm(filtered_t_exit_coords - filtered_origins, axis=1)
    filtered_directions = np.where(
        np.abs(filtered_directions) <= epsilon,
        np.where(filtered_directions == 0, epsilon, np.sign(filtered_directions) * epsilon),
        filtered_directions
    )
    inv_filtered_directions = 1.0 / filtered_directions

    t_min = (filtered_voxel_mins - filtered_origins) * inv_filtered_directions
    t_max = (filtered_voxel_maxs - filtered_origins) * inv_filtered_directions
    t_enter = np.max(np.minimum(t_min, t_max), axis=1)
    t_exit = np.min(np.maximum(t_min, t_max), axis=1)
    del t_min, t_max
    gc.collect()

    filtered_exit_coords = filtered_origins + t_exit[:, np.newaxis] * filtered_directions
    filtered_entry_coords = filtered_origins + t_enter[:, np.newaxis] * filtered_directions

    # Assign hit type
    unbound = np.isnan(filtered_points).any(axis=1)
    in_voxel = np.all((filtered_points >= (filtered_voxel_mins - epsilon)) & (filtered_points <= (filtered_voxel_maxs + epsilon)), axis=1)

    # Use squared distances for faster computation on large datasets
    # All points should be in some voxel, but we need to classify their hit type
    dist_to_entry_sq = np.sum((filtered_origins -(filtered_entry_coords)) ** 2, axis=1)
    dist_to_exit_sq = np.sum((filtered_origins - (filtered_exit_coords)) ** 2, axis=1)
    dist_to_point_sq = np.sum((filtered_points - filtered_origins) ** 2, axis=1)

    # Classify hit types
    before_voxel = (dist_to_entry_sq > dist_to_point_sq) & ~in_voxel & ~unbound
    after_voxel = (dist_to_exit_sq < dist_to_point_sq) & ~in_voxel & ~unbound

    hit_type = np.full(filtered_points.shape[0], -1, dtype=np.int32)
    hit_type[unbound] = 0
    hit_type[before_voxel] = 1
    hit_type[in_voxel] = 2
    hit_type[after_voxel] = 3

    # If a point is not classified as in_voxel, before_voxel, after_voxel, or unbound, it is an error
    # But since all points are in a voxel, hit_type should never be -1 except for unbound rays

    invalid_points_mask = hit_type == -1
    if invalid_points_mask.sum() > 0:
        # print("Invalid points found")
        pass

    if debug:
        invalid_points_mask = hit_type == -1
        invalid_points = filtered_points[invalid_points_mask]
        import pyvista as pv
        pcd = pv.PolyData(invalid_points)
        pcd.save("invalid_points.ply")
        in_points = filtered_points[hit_type == 2]
        pcd = pv.PolyData(in_points)
        pcd.save("in_points.ply")
        bef_points = filtered_points[hit_type == 1]
        pcd = pv.PolyData(bef_points)
        pcd.save("bef_points.ply")
        aft_points = filtered_points[hit_type == 3]
        pcd = pv.PolyData(aft_points)
        pcd.save("aft_points.ply")
        unbound_origins = filtered_origins[hit_type == 0]
        pcd = pv.PolyData(unbound_origins)
        pcd.save("unbound_origins.ply")
    
    # Calculate distance to voxel centre used in metrics for beam divergence
    # Calculate voxel centre from min and max coordinates
    filtered_distances_to_voxel_centre = np.linalg.norm(filtered_origins - filtered_voxel_centres, axis=1)

    del t_enter, t_exit, filtered_origins, filtered_directions, filtered_voxel_maxs, filtered_voxel_mins
    gc.collect()

    # print(f"return_numbers: {np.nanmax(filtered_return_numbers)} number_of_returns: {np.nanmax(filtered_number_of_returns)} hit_type: {np.max(hit_type)}")

    # Construct the DataFrame directly from arrays
    data_dict = {
        'voxel_size': filtered_voxel_sizes.flatten(),
        'voxel_id': filtered_voxel_ids.flatten(),
        'voxel_cx': filtered_voxel_centres[:, 0],
        'voxel_cy': filtered_voxel_centres[:, 1],
        'voxel_cz': filtered_voxel_centres[:, 2],
        'leg_id': filtered_leg_ids.flatten(),
        'ray_id': filtered_ray_ids.flatten(),
        't_entry_x': filtered_entry_coords[:, 0],
        't_entry_y': filtered_entry_coords[:, 1],
        't_entry_z': filtered_entry_coords[:, 2],
        't_exit_x': filtered_exit_coords[:, 0],
        't_exit_y': filtered_exit_coords[:, 1],
        't_exit_z': filtered_exit_coords[:, 2],
        'distance_to_centre': filtered_distances_to_voxel_centre.flatten(),
        'point_x': filtered_points[:, 0],
        'point_y': filtered_points[:, 1],
        'point_z': filtered_points[:, 2],
        'echo_intensity': filtered_echo_intensities.flatten(),
        'return_number': filtered_return_numbers.flatten(),
        'number_of_returns': filtered_number_of_returns.flatten(),
        'normal_x': filtered_normals[:, 0],
        'normal_y': filtered_normals[:, 1],
        'normal_z': filtered_normals[:, 2],
        'point_weight': filtered_point_weights.flatten(),
        'viewing_angle': filtered_viewing_angles.flatten(),
        'hit_type': hit_type.flatten() if hasattr(hit_type, "flatten") else hit_type,
        'is_leaf': filtered_is_leaf.flatten() if hasattr(filtered_is_leaf, "flatten") else filtered_is_leaf
    }
    data_df = pd.DataFrame(data_dict)

    del filtered_voxel_sizes, filtered_voxel_ids, filtered_leg_ids, filtered_ray_ids, filtered_entry_coords, filtered_exit_coords, filtered_distances_to_voxel_centre, filtered_points, filtered_viewing_angles, hit_type, filtered_is_leaf
    gc.collect()

    print(f"Process {process_id}. Returning results...")

    return data_df

# Function to traverse the voxels and find ray intersections
def traverse_voxels_oldcode(voxel_references, ray_partition, chunks_per_compute, temp_dir, epsilon=1e-6):

    if ray_partition.empty:
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
    
    # Prep ray information
    leg_ids = ray_partition['leg_id'].values
    ray_ids = ray_partition['ray_id'].values
    origins = np.asarray(ray_partition[['origin_x', 'origin_y', 'origin_z']].values)
    directions = np.asarray(ray_partition[['direction_x', 'direction_y', 'direction_z']].values)
    points = np.asarray(ray_partition[['point_x', 'point_y', 'point_z']].values)
    normals = np.asarray(ray_partition[['normal_x', 'normal_y', 'normal_z']].values)
    weights = np.asarray(ray_partition['point_weight'].values)
    intensities = np.asarray(ray_partition['echo_intensity'].values)
    return_numbers = np.asarray(ray_partition['return_number'].values)
    number_of_returns = np.asarray(ray_partition['number_of_returns'].values)
    is_leaf = np.asarray(ray_partition['is_leaf'].values)

    num_rays = len(ray_partition)
    num_voxels = len(voxel_references)
    
    voxel_ids = voxel_references['voxel_id'].values
    voxel_sizes = voxel_references['voxel_size'].values
    voxel_mins = np.asarray(voxel_references[['voxel_min_x', 'voxel_min_y', 'voxel_min_z']].values) - epsilon
    voxel_maxs = np.asarray(voxel_references[['voxel_max_x', 'voxel_max_y', 'voxel_max_z']].values) + epsilon

    del voxel_references, ray_partition
    # gc.collect()

    voxel_ids = voxel_ids[:, np.newaxis]
    voxel_mins = voxel_mins[:, np.newaxis, :]
    voxel_maxs = voxel_maxs[:, np.newaxis, :]

    leg_ids = leg_ids[np.newaxis, :]
    ray_ids = ray_ids[np.newaxis, :]
    is_leaf = is_leaf[np.newaxis, :]
    origins = origins[np.newaxis, :, :]
    directions = directions[np.newaxis, :, :]
    points = points[np.newaxis, :, :]
    normals = normals[np.newaxis, :, :]
    weights = weights[np.newaxis, :]
    intensities = intensities[np.newaxis, :]
    return_numbers = return_numbers[np.newaxis, :]
    number_of_returns = number_of_returns[np.newaxis, :]

    ### TEST DATA ###
    # voxel_mins = np.array([
    #     [[0, 0, 0]],
    #     [[1, 1, 1]],
    #     [[2, 2, 2]]
    # ])  # Shape (3, 1, 3)

    # voxel_maxs = np.array([
    #     [[1, 1, 1]],
    #     [[2, 2, 2]],
    #     [[3, 3, 3]]
    # ])  # Shape (3, 1, 3)

    # origins = np.array([
    #     [[0.5, 0.5, -1],
    #      [1.5, 1.5, -1],
    #      [2.5, 2.5, -1]]
    # ])  # Shape (1, 3, 3)

    # directions = np.array([
    #     [[0, 0, 1],
    #      [0, 0, 1],
    #      [0, 0, 1]]
    # ])  # Shape (1, 3, 3)

    # Expected results:
    # - Ray 1 intersects voxel 1
    # - Ray 2 intersects voxel 2
    # - Ray 3 intersects voxel 3

    def broadcasted_ray_tracing(voxel_mins, voxel_maxs, origins, directions, beam_divergence = 0.001, epsilon=np.float64(1e-6)):
        # Calculate t_min and t_max for each dimension
        directions = np.where(
            np.abs(directions) <= epsilon,
            np.where(directions == 0, epsilon, np.sign(directions) * epsilon),
            directions
        )

        t_min = (voxel_mins - origins) / directions
        t_max = (voxel_maxs - origins) / directions

        t_enter = np.max(np.minimum(t_min, t_max), axis=2)
        t_exit = np.min(np.maximum(t_min, t_max), axis=2)

        # Check if t_enter is less than t_exit
        mask = (t_enter <= t_exit + epsilon) & (t_exit >= -epsilon)

        # Setup arrays for returned values
        # t_enter_coords = np.full((mask.shape[0], mask.shape[1], origins.shape[2]), np.nan, dtype=np.float64)
        # t_exit_coords = np.full((mask.shape[0], mask.shape[1], origins.shape[2]), np.nan, dtype=np.float64)
        # t_entry_radii = np.full_like(mask, np.nan, dtype=np.float64)
        # t_exit_radii = np.full_like(mask, np.nan, dtype=np.float64)

        # # If there are any true values in mask, run calculations
        # has_nonzero = np.any(mask, axis=(0,1))
        # if has_nonzero:
        #     # Calculate the entry and exit coordinates for valid rays
        #     origins = np.broadcast_to(origins, (mask.shape[0], mask.shape[1], origins.shape[2]))
        #     directions = np.broadcast_to(directions, (mask.shape[0], mask.shape[1], directions.shape[2]))
            
        #     origins = origins[mask]
        #     directions = directions[mask]
        #     t_enter = t_enter[mask]
        #     t_exit = t_exit[mask]
            
        #     t_enter_coords[mask] = origins + t_enter[:, np.newaxis] * directions
        #     t_exit_coords[mask] = origins + t_exit[:, np.newaxis] * directions

        #     # Calculate the radii from beam divergence using t_enter and t_exit as distances
        #     t_entry_radii[mask] = (t_enter * np.tan(beam_divergence)).astype(np.float64)
        #     t_exit_radii[mask] = (t_exit * np.tan(beam_divergence)).astype(np.float64)
        
        return mask # , t_enter_coords, t_exit_coords, t_entry_radii, t_exit_radii


        
    
    # Calculate mask, t_enter, and t_exit for max voxels that fit into memory
    masks = {}
    # t_enter_coords = {}
    # t_exit_coords = {}
    # t_entry_radiis = {}
    # t_exit_radiis = {}

    temp_dir = tempfile.mkdtemp(dir=temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a unique ID for the process
    process_id = uuid.uuid4().hex

    # print(f"Process {process_id}: Start {num_rays} rays, {num_voxels} voxels, in ({int(np.ceil(num_voxels / chunks_per_compute))}) chunks.")
    
    for i in range(0, voxel_mins.shape[0], chunks_per_compute):

        # Calculate the number of chunks to process in this iteration
        chunk_mask = broadcasted_ray_tracing( #, chunk_t_enter_coord, chunk_t_exit_coord, chunk_t_entry_radii, chunk_t_exit_radii = broadcasted_ray_tracing(
            voxel_mins[i:i+chunks_per_compute, :, :], 
            voxel_maxs[i:i+chunks_per_compute, :, :], 
            origins, 
            directions
        )

        # Save chunk_mask, t_enter, and t_exit to disk with unique filenames
        chunk_mask_filename = os.path.join(temp_dir, f"chunk_mask_{i}_{process_id}.npy")
        # t_enter_filename = os.path.join(temp_dir, f"t_enter_{i}_{process_id}.npy")
        # t_exit_filename = os.path.join(temp_dir, f"t_exit_{i}_{process_id}.npy")
        # t_entry_radii_filename = os.path.join(temp_dir, f"t_entry_radii_{i}_{process_id}.npy")
        # t_exit_radii_filename = os.path.join(temp_dir, f"t_exit_radii_{i}_{process_id}.npy")

        # Save arrays to disk
        np.save(chunk_mask_filename, chunk_mask)
        # np.save(t_enter_filename, chunk_t_enter_coord)
        # np.save(t_exit_filename, chunk_t_exit_coord)
        # np.save(t_entry_radii_filename, chunk_t_entry_radii)
        # np.save(t_exit_radii_filename, chunk_t_exit_radii)

        dtype = chunk_mask.dtype
        shape = chunk_mask.shape
        masks[i] = [chunk_mask_filename, dtype, shape]

        # dtype = chunk_t_enter_coord.dtype
        # shape = chunk_t_enter_coord.shape
        # t_enter_coords[i] = [t_enter_filename, dtype, shape]

        # dtype = chunk_t_exit_coord.dtype
        # shape = chunk_t_exit_coord.shape
        # t_exit_coords[i] = [t_exit_filename, dtype, shape]

        # dtype = chunk_t_entry_radii.dtype
        # shape = chunk_t_entry_radii.shape
        # t_entry_radiis[i] = [t_entry_radii_filename, dtype, shape]

        # dtype = chunk_t_exit_radii.dtype
        # shape = chunk_t_exit_radii.shape
        # t_exit_radiis[i] = [t_exit_radii_filename, dtype, shape]


        del chunk_mask #, chunk_t_enter_coord, chunk_t_exit_coord, chunk_t_entry_radii, chunk_t_exit_radii
        gc.collect()

    # print(f"Process {process_id}: Finished {num_rays} rays, {num_voxels} voxels. Concatenating results...")

    # Combine masks, t_enters, and t_exits into single arrays for further processing
    mask = None
    chunk_masks = []
    for key in sorted(masks.keys()):
        # Use np.memmap to map the saved files
        filename = masks[key][0]
        dtype = masks[key][1]
        shape = masks[key][2]

        # with open(filename, 'rb') as f:
        #     chunk_mask = pickle.load(f)
        chunk = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
        chunk_masks.append(chunk)
        
    mask = np.concatenate(chunk_masks, axis=0)

    # Flatten mask and retrieve idx for rays and voxels
    voxel_ref_idx, ray_ref_idx = np.nonzero(mask)
    if len(voxel_ref_idx) == 0:
        # print("No valid rays found.")
        # Cleanup memory
        del mask, voxel_ref_idx, ray_ref_idx
        del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, leg_ids, ray_ids, is_leaf, points, origins, directions
        gc.collect()
        # Delete the temporary folder and its contents
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Return an empty DataFrame with the same schema
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
    
    del mask, chunk_masks, chunk
    gc.collect()

    # # Combine t_enter and t_exit arrays
    # t_enters = []
    # for key in sorted(t_enter_coords.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_enter_coords[key][0]
    #     dtype = t_enter_coords[key][1]
    #     shape = t_enter_coords[key][2]

    #     chunk_t_enter = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_enters.append(chunk_t_enter)
    
    # filtered_t_entry_coords = np.concatenate(t_enters, axis=0)    
    # filtered_t_entry_coords = filtered_t_entry_coords[voxel_ref_idx, ray_ref_idx]
    # # Cleanup memory
    # del t_enter_coords, t_enters
    # gc.collect()

    # t_exits = []
    # for key in sorted(t_exit_coords.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_exit_coords[key][0]
    #     dtype = t_exit_coords[key][1]
    #     shape = t_exit_coords[key][2]

    #     chunk_t_exit = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_exits.append(chunk_t_exit)

    # filtered_t_exit_coords = np.concatenate(t_exits, axis=0)
    # filtered_t_exit_coords = filtered_t_exit_coords[voxel_ref_idx, ray_ref_idx]
    # # Cleanup memory
    # del t_exit_coords, t_exits
    # gc.collect()

    # # t_entry_radii
    # t_en_radiis = []
    # for key in sorted(t_entry_radiis.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_entry_radiis[key][0]
    #     dtype = t_entry_radiis[key][1]
    #     shape = t_entry_radiis[key][2]

    #     chunk_t_entry_radii = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_en_radiis.append(chunk_t_entry_radii)

    # filtered_t_entry_radii = np.concatenate(t_en_radiis, axis=0)
    # filtered_t_entry_radii = filtered_t_entry_radii[voxel_ref_idx, ray_ref_idx]
    # # Cleanup memory
    # del t_en_radiis, chunk_t_entry_radii, t_entry_radiis
    # gc.collect()

    # # t_exit_radii
    # t_ex_radiis = []
    # for key in sorted(t_exit_radiis.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_exit_radiis[key][0]
    #     dtype = t_exit_radiis[key][1]
    #     shape = t_exit_radiis[key][2]

    #     chunk_t_exit_radii = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_ex_radiis.append(chunk_t_exit_radii)

    # filtered_t_exit_radii = np.concatenate(t_ex_radiis, axis=0)
    # filtered_t_exit_radii = filtered_t_exit_radii[voxel_ref_idx, ray_ref_idx]

    # del t_ex_radiis, chunk_t_exit_radii, t_exit_radiis
    # gc.collect()

    # Delete the temporary folder and its contents
    shutil.rmtree(temp_dir, ignore_errors=True)
    

    # Flatten all values to match the mask
    filtered_voxel_ids = voxel_ids[voxel_ref_idx].reshape(-1)
    filtered_voxel_sizes = voxel_sizes[voxel_ref_idx]
    filtered_voxel_mins = voxel_mins[voxel_ref_idx].reshape(-1, 3)
    filtered_voxel_maxs = voxel_maxs[voxel_ref_idx].reshape(-1, 3)

    filtered_leg_ids = leg_ids[:, ray_ref_idx].reshape(-1)
    filtered_ray_ids = ray_ids[:, ray_ref_idx].reshape(-1)
    filtered_is_leaf = is_leaf[:, ray_ref_idx].reshape(-1)
    filtered_points = points[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_origins = origins[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_directions = directions[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_normals = normals[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_weights = weights[:, ray_ref_idx].reshape(-1)
    filtered_intensities = intensities[:, ray_ref_idx].reshape(-1)
    filtered_return_numbers = return_numbers[:, ray_ref_idx].reshape(-1)
    filtered_number_of_returns = number_of_returns[:, ray_ref_idx].reshape(-1)

    # Find viewing angles based on the filtered directions
    filtered_viewing_angles = find_viewing_angles(directions=filtered_directions)

    # Cleanup memory
    del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, leg_ids, ray_ids, is_leaf, points, origins, directions
    # gc.collect()

    # Filter out points that are within each respective voxel  
    filtered_hit_rays = np.all((filtered_points >= filtered_voxel_mins) & (filtered_points <= filtered_voxel_maxs), axis=1)
    
    # Filter out rays which have previously hit a voxel (and therefore do not continue into this next one)
    # Also keep any NAN points (i.e. unbound rays)
    # distance_to_exit = np.linalg.norm(filtered_t_exit_coords - filtered_origins, axis=1)
    filtered_directions = np.where(
        np.abs(filtered_directions) <= epsilon,
        np.where(filtered_directions == 0, epsilon, np.sign(filtered_directions) * epsilon),
        filtered_directions
    )
    t_min = (filtered_voxel_mins - filtered_origins) / filtered_directions
    t_max = (filtered_voxel_maxs - filtered_origins) / filtered_directions
    t_enter = np.max(np.minimum(t_min, t_max), axis=1)
    t_exit = np.min(np.maximum(t_min, t_max), axis=1)
    del t_min, t_max
    gc.collect()

    t_exit_coords = filtered_origins + t_exit[:, np.newaxis] * filtered_directions
    t_enter_coords = filtered_origins + t_enter[:, np.newaxis] * filtered_directions
    distance_to_exit_squared = np.sum((filtered_origins + t_exit[:, np.newaxis] * filtered_directions - filtered_origins) ** 2, axis=1)
    distance_to_point_squared = np.sum((filtered_points - filtered_origins) ** 2, axis=1)
    yet_to_hit_rays = np.logical_or(distance_to_point_squared > distance_to_exit_squared, np.isnan(filtered_points).any(axis=1)) 
    valid_ray_mask = np.logical_or(filtered_hit_rays, yet_to_hit_rays)

    if not np.any(valid_ray_mask):
        print("No valid rays intersect these voxels.")
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
        
    del filtered_voxel_mins, filtered_voxel_maxs
    gc.collect()
    
    # Ensure only hit ray points are kept per voxel
    filtered_points = np.where(
        filtered_hit_rays[:, None],
        filtered_points,
        np.full(filtered_points.shape, np.nan)
    )

    # Remove any invalid rays
    filtered_voxel_sizes = filtered_voxel_sizes[valid_ray_mask]
    filtered_voxel_ids = filtered_voxel_ids[valid_ray_mask]
    filtered_leg_ids = filtered_leg_ids[valid_ray_mask]
    filtered_ray_ids = filtered_ray_ids[valid_ray_mask]
    # filtered_t_entry_coords = filtered_t_entry_coords[valid_ray_mask]
    # filtered_t_exit_coords = filtered_t_exit_coords[valid_ray_mask]
    # filtered_t_entry_radii = filtered_t_entry_radii[valid_ray_mask]
    # filtered_t_exit_radii = filtered_t_exit_radii[valid_ray_mask]
    filtered_points = filtered_points[valid_ray_mask]
    filtered_viewing_angles = filtered_viewing_angles[valid_ray_mask]
    filtered_hit_rays = filtered_hit_rays[valid_ray_mask]
    filtered_is_leaf = filtered_is_leaf[valid_ray_mask] 
    filtered_origins = filtered_origins[valid_ray_mask]
    filtered_directions = filtered_directions[valid_ray_mask]
    filtered_normals = filtered_normals[valid_ray_mask]
    filtered_weights = filtered_weights[valid_ray_mask]
    filtered_intensities = filtered_intensities[valid_ray_mask]
    filtered_return_numbers = filtered_return_numbers[valid_ray_mask]
    filtered_number_of_returns = filtered_number_of_returns[valid_ray_mask]

    filtered_t_exit_coords = t_exit_coords[valid_ray_mask]
    filtered_t_entry_coords = t_enter_coords[valid_ray_mask]
    del t_exit_coords, t_enter_coords
    gc.collect()

    beam_divergence = 0.001
    filtered_t_entry_radii = t_enter[valid_ray_mask] * np.tan(beam_divergence)
    filtered_t_exit_radii = t_exit[valid_ray_mask] * np.tan(beam_divergence)
    del t_enter, t_exit, filtered_origins, filtered_directions
    gc.collect()

    # Ensure integer columns have no NaN and correct dtype
    filtered_return_numbers = np.nan_to_num(filtered_return_numbers, nan=0).astype(np.int32)
    filtered_number_of_returns = np.nan_to_num(filtered_number_of_returns, nan=0).astype(np.int32)
    filtered_leg_ids = np.nan_to_num(filtered_leg_ids, nan=-1).astype(np.int64)
    filtered_ray_ids = np.nan_to_num(filtered_ray_ids, nan=-1).astype(np.int64)
    filtered_voxel_ids = np.nan_to_num(filtered_voxel_ids, nan=-1).astype(np.int64)

    data = [
        pa.array(filtered_voxel_sizes),
        pa.array(filtered_voxel_ids),
        pa.array(filtered_leg_ids),
        pa.array(filtered_ray_ids),
        pa.array(filtered_t_entry_coords[:, 0]),
        pa.array(filtered_t_entry_coords[:, 1]),
        pa.array(filtered_t_entry_coords[:, 2]),
        pa.array(filtered_t_exit_coords[:, 0]),
        pa.array(filtered_t_exit_coords[:, 1]),
        pa.array(filtered_t_exit_coords[:, 2]),
        pa.array(filtered_t_entry_radii),
        pa.array(filtered_t_exit_radii),
        pa.array(filtered_points[:, 0]),
        pa.array(filtered_points[:, 1]),
        pa.array(filtered_points[:, 2]),
        pa.array(filtered_intensities),
        pa.array(filtered_return_numbers),
        pa.array(filtered_number_of_returns),
        pa.array(filtered_weights),
        pa.array(filtered_normals[:, 0]),
        pa.array(filtered_normals[:, 1]),
        pa.array(filtered_normals[:, 2]),
        pa.array(filtered_viewing_angles),
        pa.array(filtered_hit_rays),
        pa.array(filtered_is_leaf)
    ]
    result = pa.Table.from_arrays(data, schema=voxel_ray_intersection_schema_old)
    result = result.to_pandas()

    del filtered_voxel_sizes, filtered_voxel_ids, filtered_leg_ids, filtered_ray_ids, filtered_t_entry_coords, filtered_t_exit_coords, filtered_t_entry_radii, filtered_t_exit_radii, filtered_points, filtered_viewing_angles, filtered_hit_rays, filtered_is_leaf
    gc.collect()

    print(f"Process {process_id}. Returning results...")

    return result


    








    







### LARGE FUNCTIONS ###
# Functions that are used to perform large operations, such as calculating metrics or processing data.

# Prepare data from helios simulations
def prepare_helios_data(input_dir, output_dir, references_dir, leaf_object_ids, wood_object_ids, use_class=False, debug=False, epsilon=1e-6):
    """
    Main function to process helios simulation data.
    
    Args:
        input_dir (str): Path to the input folder containing helios simulation data.
        output_dir (str): Path to the output folder where processed data will be saved.
    """
    # Import modules
    import os
    import glob
    import shutil
    import logging
    import dask.delayed
    from dask.diagnostics import ProgressBar
    import pandas as pd
    import numpy as np
    import dask.array as da
    import dask.dataframe as dd
    import dask

    # Check if the input folder exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input folder '{input_dir}' does not exist.")
    if not os.path.exists(references_dir):
        raise FileNotFoundError(f"The references folder '{references_dir} does not exist.")

    # Check if the output folder exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output folder '{output_dir}' created.")
  
    log_file = os.path.join(output_dir, f"valid_rays.log")

    logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(filename=log_file, encoding='utf-8', level=level)

    logger.info(f"Preparing data from '{input_dir}' to '{output_dir}'...")

    # Setup valid rays filename template
    valid_rays_template = "leg_{leg:d}_valid_rays.parquet"

    ### PLOT BOUNDARY CALCULATION ###

    # Establish the plot boundaries of the plot, regardless of voxel size
    logger.info("Finding all voxel references to establish plot boundary.")
    voxel_references = glob.glob(os.path.join(references_dir, '*.csv'))

    dfs =[]
    for voxel_ref in voxel_references:
        voxel_size = os.path.basename(voxel_ref)
        voxel_size = os.path.splitext(voxel_size)[0]
        if "voxel_size_" in voxel_size:
            voxel_size = float(voxel_size.split("voxel_size_")[1])
        else:
            raise ValueError(f"Voxel size not found in {voxel_ref}. Please check the file name.")

        df = pd.read_csv(voxel_ref, index_col=None, header=0)
        

        if 'voxel_id' not in df.columns:
            logger.warning(f"No voxel_id found in {voxel_ref}. Updating csv now.")
            new_df = df[['voxel_cx', 'voxel_cy', 'voxel_cz']]
            new_df['voxel_size'] = voxel_size

            def parallel_voxel_id(pd_series):
                voxel_size = pd_series['voxel_size']
                x = pd_series['voxel_cx']
                y = pd_series['voxel_cy']
                z = pd_series['voxel_cz']
                voxel_id = create_voxel_id(voxel_size=voxel_size, x=x, y=y, z=z)

                return voxel_id

            # Add unique voxel_ids back to csv.
            voxel_ids = new_df.apply(parallel_voxel_id, axis=1)
            df['voxel_id'] = voxel_ids

            df.to_csv(voxel_ref)

            logger.info(f"Updated voxel_ids for {voxel_ref}")

        df = df[['voxel_cx', 'voxel_cy', 'voxel_cz']].astype(float)
        voxel_size = float(voxel_size)
        df['min_x'] = df['voxel_cx'].min() - (voxel_size / 2 + epsilon)
        df['max_x'] = df['voxel_cx'].max() + (voxel_size / 2 + epsilon)
        df['min_y'] = df['voxel_cy'].min() - (voxel_size / 2 + epsilon)
        df['max_y'] = df['voxel_cy'].max() + (voxel_size / 2 + epsilon)
        df['min_z'] = df['voxel_cz'].min() - (voxel_size / 2 + epsilon)
        df['max_z'] = df['voxel_cz'].max() + (voxel_size / 2 + epsilon)

        df = df[['min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']]
        dfs.append(df)

    plot_bounds = pd.concat(dfs, axis=0, ignore_index=True)
    buffer = 1e-4
    plot_min = np.array([plot_bounds['min_x'].min() - buffer, plot_bounds['min_y'].min() - buffer, plot_bounds['min_z'].min() - buffer])
    plot_max = np.array([plot_bounds['max_x'].max() + buffer, plot_bounds['max_y'].max() + buffer, plot_bounds['max_z'].max() + buffer])

    logger.info(f"Plot boundaries calculated as min: {plot_min} and max: {plot_max}")

    # Cleanup memory
    del plot_bounds, df, dfs, voxel_references

    ### DEFINE FUNCTIONS REQUIRED FOR VALID RAYS PREPARATION ###

    # Function to traverse plot and remove any rays which do not intersect the voxel plot
    # Plot min and max are already calculated
    def is_in_plot(ray_origins, ray_directions):
        t_min = da.divide(plot_min - ray_origins, ray_directions, out=da.full_like(ray_origins, np.inf), where=da.all(ray_directions!=0))
        t_max = da.divide(plot_max - ray_origins, ray_directions, out=da.full_like(ray_origins, np.inf), where=da.all(ray_directions!=0))
        t1 = da.minimum(t_min, t_max)
        t2 = da.maximum(t_min, t_max)
        t_enter = da.max(t1, axis=1)
        t_exit = da.min(t2, axis=1)
        mask = (t_enter <= t_exit) & (t_exit >= 0)
        return mask
    
    # Function to enable map_partitions functionality
    def valid_mask(df):
        origins = da.array([df['origin_x'].values, df['origin_y'].values, df['origin_z'].values]).T
        directions = da.array([df['direction_x'].values, df['direction_y'].values, df['direction_z'].values]).T
        mask = is_in_plot(origins, directions).compute()
        return df[mask]
    
    @dask.delayed
    def import_helios_leg(pulse_file, xyz_file):
        # Import all rays into dask dataframe
        leg_rays = dd.read_csv(pulse_file, delimiter=' ', header=None, names=['origin_x', 'origin_y', 'origin_z', 'direction_x', 'direction_y', 'direction_z', 'gps_time', 'ray_id', '_'])
        leg_rays = leg_rays.drop(columns=['gps_time', '_'])

        # Check for valid rays on partitions
        leg_rays = leg_rays.map_partitions(valid_mask, meta=leg_rays._meta)

        # Import all hits into dask dataframe
        leg_hits = dd.read_csv(xyz_file, delimiter=' ', header=None, names=['point_x', 'point_y', 'point_z', 'echo_intensity', 'echo_width', 'return_number', 'number_of_returns', 'ray_id', 'hit_object_id', 'class', 'gps_time'])

        leg_rays = leg_rays.merge(leg_hits, on='ray_id', how='left')
        leg_rays = leg_rays.drop(columns=['gps_time', 'echo_width']) # drop gps_time post merge

        return leg_rays
    

    ### START LEG RAY PROCESSING ###
    pulses = glob.glob(os.path.join(input_dir, '*_pulse.txt'))
    points = glob.glob(os.path.join(input_dir, '*_points.xyz'))
    pulses.sort()
    points.sort()

    ray_processing_list = []
    for i, pulse_file in enumerate(pulses):
        xyz_file = points[i]

        leg = pulse_file.split("leg")[1].split("_")[0]

        delayed_result = import_helios_leg(pulse_file=pulse_file, xyz_file=xyz_file)
        ray_processing_list.append((leg, delayed_result))

    # Count number of rays and points in output
    total_rays = 0
    total_points = 0

    # Process legs
    with ProgressBar():
        statement = "Processing dask delayed functions..."
        print(statement)
        logger.info(statement)
        results = dask.compute(*[ray_processing_list[i][1] for i in range(len(ray_processing_list))])
        
        for leg, result in enumerate(results):
            statement = f"Processing leg {leg}..."
            logger.info(statement)
            print(statement)

            rays = result.compute()
            logger.info("Dask computation complete.")
            
            leg = int(leg)
            rays_file = os.path.join(output_dir, valid_rays_template.format(leg=leg))
            if os.path.exists(rays_file):
                if os.path.isfile(rays_file):
                    os.remove(rays_file)
                else:
                    shutil.rmtree(rays_file)
            
            logger.info(f"Saving valid rays for leg {leg} to {rays_file}")
            rays['leg_id'] = leg
            hit_object_key = 'hit_object_id' if not use_class else 'class'
            rays['is_leaf'] = rays[hit_object_key].isin(leaf_object_ids)
            # Filter out points with unknown object ids
            rays = rays[
                pd.isna(rays[hit_object_key]) |
                rays[hit_object_key].isin(wood_object_ids + leaf_object_ids)
            ]

            rays = rays.drop(columns=['hit_object_id', 'class'])
            rays['normal_x'] = np.nan
            rays['normal_y'] = np.nan
            rays['normal_z'] = np.nan
            rays['point_weight'] = np.nan
            rays.to_parquet(rays_file, engine='pyarrow', compression='snappy', schema=valid_rays_schema)

            logger.info("Counting points...")

            num_rays = len(rays)
            num_points = (~rays['point_x'].isna()).sum()
            logger.info(f"Leg {leg} has {num_rays} valid rays and {num_points} points.")

            total_rays += int(num_rays)
            total_points += int(num_points)
            logger.info(f"Updated totals: {total_rays} rays and {total_points} points.")
    
    if debug:
        print("Debugging output...")
        import pyvista as pv
        import matplotlib.pyplot as plt
        # Plot a side on image of one leg of valid_rays with leaf_hits being green and wood_hits brown
        valid_ray_parquets = glob.glob(os.path.join(output_dir, '*valid_rays.parquet'))
        test_file = valid_ray_parquets[0] if valid_ray_parquets else None

        total_helios_points = 0
        helios_points_comb = []
        for file in points:
            helios_points = np.loadtxt(file, usecols=(0, 1, 2))
            helios_points_comb.append(helios_points)
            total_helios_points += helios_points.shape[0]
        helios_points_comb = np.concatenate(helios_points_comb, axis=0)

        if total_helios_points != total_points:
            print(f"Total Helios points {total_helios_points} do not match total valid points {total_points}")           

        else:
            print(f"Total Helios points {total_helios_points} match total valid points {total_points}")

        total_helios_rays = 0
        for file in pulses:
            rays = np.loadtxt(file, usecols=(-1))
            total_helios_rays += rays.shape[0]

        if total_helios_rays != total_rays:
            print(f"Total Helios rays {total_helios_rays} do not match total valid rays {total_rays}")
        else:
            print(f"Total Helios rays {total_helios_rays} match total valid rays {total_rays}")

        if test_file:
            df = pd.read_parquet(test_file)
            leg_id = df['leg_id'].iloc[0] if 'leg_id' in df.columns else 0
            # First, create the mask for non-NaN point_x, then filter the dataframe
            df = df[~df['point_x'].isna()][['point_x', 'point_y', 'point_z', 'is_leaf']]
            leaf_df = df[df['is_leaf']]
            wood_df = df[~df['is_leaf']]

            # Extract points and plot using matplotlib
            leaf_points = leaf_df[['point_x', 'point_y', 'point_z']].values
            del leaf_df, df  # Free up memory
            wood_points = wood_df[['point_x', 'point_y', 'point_z']].values
            del wood_df

            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            # Plot leaf points in green
            ax.scatter(leaf_points[:, 0], leaf_points[:, 2], c='green', s=1, label='Leaf')

            # Plot wood points in brown
            ax.scatter(wood_points[:, 0], wood_points[:, 2], c='saddlebrown', s=1, label='Wood')

            print("Plotting leaf and wood points to check classification...")
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'Leaf and Wood Point Check - Leg {leg_id}')
            ax.legend()
            plt.show()
            plt.savefig(os.path.join(output_dir, f'leg_{leg_id}_leaf_wood_check.png'))

            # Save 3d .ply
            print("Saving leaf and wood point clouds...")
            pcd_leaf = pv.PolyData(leaf_points)
            pcd_leaf.save(os.path.join(output_dir, f'leg_{leg_id}_leaf_points_test.ply'))
            pcd_wood = pv.PolyData(wood_points)
            pcd_wood.save(os.path.join(output_dir, f'leg_{leg_id}_wood_points_test.ply'))

    statement= "Helios data preparation complete."
    print(statement)
    logger.info(statement)

def potential_valid_rays_debug():
    import os
    import glob
    import pandas as pd
    import numpy as np

    helios_files = glob.glob(os.path.join(helios_dir, '*.xyz'))
    pulses = glob.glob(os.path.join(helios_dir, '*_pulse.txt'))
    valid_rays_files = glob.glob(os.path.join(valid_rays_dir, '*valid_rays.parquet'))

    total_helios_points = 0
    helios_points_comb = []
    for file in helios_files:
        # Read only the first three columns (point_x, point_y, point_z) using numpy for efficiency
        arr = np.loadtxt(file, usecols=(0, 1, 2))
        total_helios_points += arr.shape[0]
        helios_points_comb.append(arr)

    helios_points_comb = np.concatenate(helios_points_comb, axis=0)

    valid_rays_dfs = []
    for file in valid_rays_files:
        df = pd.read_parquet(file)
        valid_rays_dfs.append(df)

    valid_rays_df = pd.concat(valid_rays_dfs)
    valid_rays_points = valid_rays_df[['point_x', 'point_y', 'point_z']][valid_rays_df['point_x'].notna()].values

    total_valid_points = valid_rays_points.shape[0]

    print(f"Total Helios points: {total_helios_points}")
    print(f"Total valid rays points: {total_valid_points}")

    # Use matching logic similar to missing_valid_wood_points
    # Instead of looping, use broadcasting for efficiency
    missing_mask = np.array([
        not np.any(np.all(np.isclose(valid_rays_points.astype(np.float32), hp.astype(np.float32), atol=1e-6), axis=1))
        for hp in helios_points_comb
    ])
    missing_points = helios_points_comb[missing_mask]
    print(f"Number of missing points: {len(missing_points)}")
    if len(missing_points) > 0:
        print("Saving missing points to 'missing_points.xyz'")
        missing_points_file = os.path.join(valid_rays_dir, "missing_points.xyz")
        np.savetxt(missing_points_file, missing_points, fmt="%.6f")
    else:
        print("No missing points found.")

def potential_intersections_debug():
    import os
    import glob
    import numpy as np
    import pandas as pd

    voxel_sizes = 'all'

    if voxel_sizes == 'all':
        intersection_files = glob.glob(os.path.join(valid_rays_dir, '*_intersections.parquet'))
    else:
        intersection_files = []
        for vs in voxel_sizes:
            files = glob.glob(os.path.join(valid_rays_dir, f'*{vs}_intersections.parquet'))
            intersection_files.extend(files)

    for file in intersection_files:
        df = pd.read_parquet(file)
        leg_id = df['leg_id'].iloc[0]
        voxel_size = round(df['voxel_size'].iloc[0], 1)
        valid_rays = os.path.join(valid_rays_dir, f"leg_{leg_id}_valid_rays.parquet")
        
        reference = "/home/capheus/projects/51_tree_test/1001_etri_uniform_diamond/references/1001_etri_uniform_diamond_results_0.2.csv"

        if os.path.exists(valid_rays):
            print(f"Leg {leg_id}")
            valid_rays = pd.read_parquet(valid_rays, engine='pyarrow')
            hit_mask = valid_rays['point_x'].notna()
            leaf_hit_mask = valid_rays['is_leaf'] & hit_mask
            pre_num_hits = hit_mask.sum()
            pre_num_leaf_hits = leaf_hit_mask.sum()
            print(f"Pre-hits: {pre_num_hits}, Pre-leaf hits: {pre_num_leaf_hits}")

            hit_mask = df['hit_type'] == 2
            leaf_hit_mask = df['is_leaf'] & hit_mask
            post_num_hits = hit_mask.sum()
            post_num_leaf_hits = leaf_hit_mask.sum()
            print(f"Post-hits: {post_num_hits}, Post-leaf hits: {post_num_leaf_hits}")

            # Find ray_ids that have hits (any hit_type > 0) but never hit_type == 2
            rays_with_hits = df.loc[df['hit_type'] > 0, 'ray_id'].unique()
            rays_with_type2 = df.loc[df['hit_type'] == 2, 'ray_id'].unique()
            rays_with_hits_not_type2 = np.setdiff1d(rays_with_hits, rays_with_type2)

            rays_info = df[df['ray_id'].isin(rays_with_hits_not_type2)][['ray_id', 'point_x', 'point_y', 'point_z']]
            rays_info.drop_duplicates(subset=['ray_id', 'point_x', 'point_y', 'point_z'], inplace=True)
            print(f"Number of rays with hits but never hit_type == 2: {len(rays_with_hits_not_type2)}")
            if len(rays_with_hits_not_type2) > 0:
                points = rays_info[['point_x', 'point_y', 'point_z', 'ray_id']].values

                if reference is not None:
                    if os.path.exists(reference):
                        reference_df = pd.read_csv(reference)
                        reference_df.drop_duplicates(subset=['voxel_cx', 'voxel_cy', 'voxel_cz'], inplace=True)

                        # To fix, only append once:
                        points_meant_to_be_in_voxel = []
                        for point in points:
                            found_in_voxel = False
                            ray_id = point[3]
                            for _, voxel_row in reference_df.iterrows():
                                min_bound = np.array([voxel_row['voxel_cx'], voxel_row['voxel_cy'], voxel_row['voxel_cz']]) - voxel_size / 2.0 - 1e-6
                                max_bound = np.array([voxel_row['voxel_cx'], voxel_row['voxel_cy'], voxel_row['voxel_cz']]) + voxel_size / 2.0 + 1e-6
                                pt_xyz = point[:3]
                                # Check if point is inside voxel bounds
                                if np.all((pt_xyz > min_bound) & (pt_xyz < max_bound)):
                                    # Also check if ray_id is present in the voxel in df
                                    voxel_mask = (
                                        (df['voxel_cx'] == voxel_row['voxel_cx']) &
                                        (df['voxel_cy'] == voxel_row['voxel_cy']) &
                                        (df['voxel_cz'] == voxel_row['voxel_cz'])
                                    )
                                    if ray_id in df.loc[voxel_mask, 'ray_id'].values:
                                        found_in_voxel = True
                                        break
                            if found_in_voxel:
                                points_meant_to_be_in_voxel.append(point)
                        
                        if len(points_meant_to_be_in_voxel) > 0:
                            # Check if the ray_id is assigned to the voxel
                            print(f"Number of points meant to be in voxel: {len(points_meant_to_be_in_voxel)}")
                            out_file = os.path.join(valid_rays_dir, f"leg_{leg_id}_vs_{voxel_size}_hits_not_type2_points.xyz")
                            points = points_meant_to_be_in_voxel
                        else:
                            print("All missing points are not in reference voxels.")
                            out_file = None
                else:
                    out_file = os.path.join(valid_rays_dir, f"leg_{leg_id}_vs_{voxel_size}_hits_not_type2_points.xyz")
                
                if out_file is not None:
                    np.savetxt(out_file, points, fmt="%.6f")



# Function used for taking valid_rays parquet files and references to establish voxel_ray intersections per valid_rays file
def voxel_ray_intersections(valid_rays_dir, references_dir, temp_dir=None, debug=False, epsilon=1e-6):
    import os
    import glob
    import pandas as pd
    import numpy as np
    import psutil
    import shutil
    import dask
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    # Calculate the partition size that fits into memory alongside optimal voxel chunk size
    avail_cpu = 1
    if os.environ.get('SLURM_CPUS_PER_TASK') is not None:
        avail_cpu = max(avail_cpu, int(os.environ.get('SLURM_CPUS_PER_TASK')))
        num_threads = 2
    else:
        avail_cpu = max(avail_cpu, psutil.cpu_count(logical=True))
        num_threads = 1
    avail_mem = int(float(os.environ.get('SLURM_MEM_PER_NODE', psutil.virtual_memory().available // (1024 * 1024))))
    avail_mem *= 0.9  # Use 90% of available memory to be safe

    if temp_dir is None:
        temp_dir = os.environ.get("TMPDIR", "/tmp")

    avail_mem_string_for_dask = f"{int(avail_mem)}MB"
    _start_dask_client(memory_limit=avail_mem_string_for_dask, n_workers=avail_cpu, threads_per_worker=num_threads)

    # Compile the references files to establish a voxel dataframe of size and voxel_id
    voxel_references = glob.glob(os.path.join(references_dir, '*.csv'))

    if temp_dir == None:
        temp_dir = os.environ.get("TMPDIR", "/tmp")

    dfs = []
    max_voxels = 0
    for voxel_ref in voxel_references:
        # Read the csv
        df = pd.read_csv(voxel_ref, index_col=None, header=0)

        if 'voxel_id' not in df.columns:
            # Generate voxel_id using voxel_cx, voxel_cy, voxel_cz, and voxel_size
            df['voxel_id'] = df.apply(
                lambda row: create_voxel_id(
                    voxel_size=row['voxel_size'] if 'voxel_size' in row else float(os.path.splitext(voxel_ref)[0].split("_")[-1]),
                    x=row['voxel_cx'],
                    y=row['voxel_cy'],
                    z=row['voxel_cz']
                ),
                axis=1
            )

        # Filter out unnecessary columns and duplicates
        df = df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']].drop_duplicates()

        # Add voxel_size to dataframe for later grouping
        voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
        df['voxel_size'] = voxel_size

        dfs.append(df)

    # Combine into single pandas dataframe for later grouping
    voxel_references = pd.concat(dfs)

    # Compile all valid_rays parquets
    valid_rays_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))

    def map_ray_partition_to_function(ray_partition, voxel_group, temp_dir, voxels_per_chunk):
        result = traverse_voxels(ray_partition=ray_partition, voxel_references=voxel_group, voxels_per_chunk=voxels_per_chunk, temp_dir=temp_dir, debug=debug)
        return result

    voxel_ray_intersections = {}

    for file in valid_rays_files:
        # Read in parquet file
        df = dd.read_parquet(file, engine='pyarrow')

        # Get leg_id from filename
        leg_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])

        # Map partitions to traverse voxels
        meta = pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        mem_per_worker = avail_mem // avail_cpu

        num_rays = df['ray_id'].nunique().compute()
        num_voxels = len(voxel_references)
        ratio = num_rays / num_voxels if num_voxels > 0 else 1

        # Estimate the max elements that fit in memory per worker
        # Calculate bytes per ray from valid_rays_schema
        bytes_per_ray = 12 * 8 # sum(field.type.bit_width // 4 for field in valid_rays_schema)  # bit_width to bytes
        bytes_per_voxel = 6 * 8  # 6 float32 values per voxel (min/max for x, y, z), 4 bytes each
        max_elements_per_chunk = int(mem_per_worker // (bytes_per_voxel + bytes_per_ray))

        if avail_cpu * max_elements_per_chunk > avail_mem:
            rays_per_chunk = max(1, int(np.sqrt(max_elements_per_chunk * ratio)))
            voxels_per_chunk = max(1, int(max_elements_per_chunk // rays_per_chunk))
        else:
            rays_per_chunk = max(1, num_rays // avail_cpu)
            voxels_per_chunk = max(1, int(max_elements_per_chunk // rays_per_chunk))
        
        npartitions = max(1, int(np.ceil(num_rays / rays_per_chunk)))
        df = df.repartition(npartitions=npartitions)

        #### TEST
        npartitions = avail_cpu
        df = df.repartition(npartitions=npartitions)

        dd_results = []

        print(f"Initialising leg {leg_id} - {df.npartitions} partitions, {num_rays} rays, {num_voxels} voxels, {rays_per_chunk} rays/partition, {voxels_per_chunk} voxels/chunk, {max(1, num_voxels//voxels_per_chunk)} chunks")

        result = df.map_partitions(
            map_ray_partition_to_function,
            voxel_group=voxel_references,
            voxels_per_chunk=voxels_per_chunk,
            temp_dir=temp_dir,
            meta=meta
        )

        dd_results.append(result)

        voxel_ray_intersections[leg_id] = result

    def save_task(df, leg_id):
        if df.empty:
            return False
        
        # Get the voxel size from the dataframe
        voxel_size = round(float(group_df['voxel_size'].iloc[0]), 2)

        # Create the output filename
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")

        if debug:
            valid_rays = os.path.join(valid_rays_dir, f"leg_{leg_id}_valid_rays.parquet")
            if os.path.exists(valid_rays):
                valid_rays = pd.read_parquet(valid_rays, engine='pyarrow')
                hit_mask = valid_rays['point_x'].notna()
                leaf_hit_mask = valid_rays['is_leaf'] & hit_mask
                pre_num_hits = hit_mask.sum()
                pre_num_leaf_hits = leaf_hit_mask.sum()
                print(f"Pre-hits: {pre_num_hits}, Pre-leaf hits: {pre_num_leaf_hits}")

                hit_mask_point = df['point_x'].notna()
                hit_mask = df['hit_type'] == 2
                leaf_hit_mask_point = df['is_leaf'] & hit_mask_point
                leaf_hit_mask = df['is_leaf'] & hit_mask
                post_num_hits = hit_mask.sum()
                post_num_leaf_hits = leaf_hit_mask.sum()
                post_num_hits_point = hit_mask_point.sum()
                post_num_leaf_hits_point = leaf_hit_mask_point.sum()
                print(f"Post-hits: {post_num_hits}, Post-leaf hits: {post_num_leaf_hits}, Post-hits (point): {post_num_hits_point}, Post-leaf hits (point): {post_num_leaf_hits_point}")

        # Save the dataframe to parquet
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)

        return True


    # Schedule legs
    futures = []
    for leg_id, results in voxel_ray_intersections.items():
        future = DASK_CLIENT.compute(results)
        futures.append((leg_id, future))

    # Process legs
    start_time = time.time()
    for leg_id, future in futures:
        with ProgressBar():
            results = future.result()

        # Process and save each partition individually to release memory
        for voxel_size, group_df in results.groupby('voxel_size', group_keys=True):
            save_task(group_df, leg_id)
            del group_df
            gc.collect()

        del results
        gc.collect()

        print(f"Saved leg {leg_id}!")

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Voxel ray intersection processing complete in {elapsed_time:.2f} seconds.")

    # brief pause to ensure filesystem/dask flushes before shutting down
    time.sleep(1)
    _close_dask_client(DASK_CLIENT)

# Function used for taking valid_rays parquet files and references to establish voxel_ray intersections per valid_rays file
def voxel_ray_intersections_oldcode(valid_rays_dir, references_dir, temp_dir, cpus=None, mem=None, debug=True, epsilon=1e-6):
    import os
    import glob
    import pandas as pd
    import numpy as np
    import psutil
    import shutil
    import dask
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    # Compile the references files to establish a voxel dataframe of size and voxel_id
    voxel_references = glob.glob(os.path.join(references_dir, '*.csv'))

    dfs = []
    max_voxels = 0
    for voxel_ref in voxel_references:
        # Read the csv
        df = pd.read_csv(voxel_ref, index_col=None, header=0)

        # Filter out unnecessary columns and duplicates
        df = df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']].drop_duplicates()

        # Add voxel_size to dataframe for later grouping
        voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
        df['voxel_size'] = voxel_size

        # Prep dataframe as bounds of each voxel rather than centre (maintaining size and unique id)
        df['voxel_min_x'] = df['voxel_cx'].astype(float) - (voxel_size / 2 + epsilon)
        df['voxel_max_x'] = df['voxel_cx'].astype(float) + (voxel_size / 2 + epsilon)
        df['voxel_min_y'] = df['voxel_cy'].astype(float) - (voxel_size / 2 + epsilon)
        df['voxel_max_y'] = df['voxel_cy'].astype(float) + (voxel_size / 2 + epsilon)
        df['voxel_min_z'] = df['voxel_cz'].astype(float) - (voxel_size / 2 + epsilon)
        df['voxel_max_z'] = df['voxel_cz'].astype(float) + (voxel_size / 2 + epsilon)
        df = df.drop(columns=['voxel_cx', 'voxel_cy', 'voxel_cz'])

        dfs.append(df)

    # Combine into single pandas dataframe for later grouping
    voxel_references = pd.concat(dfs)

    # Calculate max chunk_size based on number of voxels
    # Get available memory in bytes

    def total_mem_required_per_ray(elements_per_ray=3*8+4, memory_per_element=8):
        return elements_per_ray * memory_per_element
    
    def calculate_chunk_size_mb(num_voxels, available_memory, min_chunk_size=100):
        mem_required_per_ray = total_mem_required_per_ray()
        total_mem_per_ray_per_voxel = num_voxels * mem_required_per_ray
        chunk_size = max(min_chunk_size, available_memory / total_mem_per_ray_per_voxel)
        return chunk_size


    # Compile all valid_rays parquets
    valid_rays_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))

    def map_ray_partition_to_function(ray_partition, voxel_group, temp_dir, chunks_per_compute):



        result = traverse_voxels_oldcode(ray_partition=ray_partition, voxel_references=voxel_group, chunks_per_compute=chunks_per_compute, temp_dir=temp_dir)
        return result

    voxel_ray_intersections = {}

    for file in valid_rays_files:
        # Read in parquet file
        df = dd.read_parquet(file, engine='pyarrow', blocksize=100 * 1024 * 1024)
        cpus = os.cpu_count() if cpus is None else cpus

        if df.npartitions < cpus:
            # Check for the size of resulting partition
            df_mem = df.memory_usage(deep=True).compute().sum()
            target_partition_mem = 25 * 1024 * 1024 # bytes
            if df_mem / cpus < target_partition_mem: # Avoid too small partitions
                cpus = int(df_mem / target_partition_mem)
            
            df = df.repartition(npartitions=cpus)


        # Get leg_id from filename
        leg_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])

        # Map partitions to traverse voxels
        meta = pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)

        if mem == None:
            mem = psutil.virtual_memory().available
        
        available_mem = mem * 0.8
        partition_mem = df.memory_usage_per_partition(deep=True).compute().max()
        num_voxels = len(voxel_references)

        voxel_memory = num_voxels * 12 * 8  # Voxels are broadcast to (n, 1, 8), each element is 8 bytes (float64)
        mem_per_compute = (partition_mem + voxel_memory) * 2
        # Calculate the memory required for broadcasting a single voxel to all rays
        chunks_per_compute = int((available_mem / (mem_per_compute * df.npartitions)))

        if chunks_per_compute < 1:
            required_partitions = int(np.ceil(partition_mem / mem_per_compute))
            max_partitions = int(np.floor(available_mem / (required_partitions * partition_mem)))
            df = df.repartition(npartitions=max_partitions)
            chunks_per_compute = int((available_mem / (mem_per_compute * df.npartitions)))
            while (max_partitions * partition_mem) > available_mem:
                max_partitions -= 1
                df = df.repartition(npartitions=max_partitions)
                partition_mem = df.memory_usage_per_partition(deep=True).compute().max()
            
            chunks_per_compute = int((available_mem / (mem_per_compute * df.npartitions)))

        dd_results = []

        result = df.map_partitions(
            map_ray_partition_to_function,
            voxel_group=voxel_references,
            chunks_per_compute=chunks_per_compute,
            temp_dir=temp_dir,
            meta=meta
        )

        dd_results.append(result)

        voxel_ray_intersections[leg_id] = result

    def save_task(df, leg_id):
        if df.empty:
            return False
        
        # Get the voxel size from the dataframe
        voxel_size = round(float(df.name), 2)

        # Create the output filename
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")

        # Save the dataframe to parquet
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema_old)

        return True


    with ProgressBar():

        for leg_id, results in voxel_ray_intersections.items():

            print(f"Processing leg {leg_id}...")

            results = results.compute()

            print(f"Saving results for leg {leg_id}...")
            results = results.groupby('voxel_size').apply(lambda x: save_task(x, leg_id))

            del results
            gc.collect()

            print(f"Completed leg {leg_id}!")


def get_voxel_metrics_oldcode(intersections_files, lambda_1, is_leaf_true=True, debug=True, epsilon=1e-9):
    """
    This function will take the voxel_ray_intersection files and calculate the metrics for each voxel.
    It will save the results to a parquet file.

    Args:
        intersections_files (list): List of paths to voxel_ray_intersection files.
        lambda_1 (float): This is calculated using (average leaf area / voxel size) and will need to be calculated and passed in.
        debug (bool): Whether to print debug information.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        None
    """
    import os
    import glob
    import pandas as pd
    import numpy as np
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    import logging
    import psutil

    

    # Setup logger
    # logger = logging.getLogger()
    # level = logging.DEBUG if debug else logging.WARNING
    # logging.basicConfig(filename=os.path.join(intersections_files, 'voxel_metrics.log'), encoding='utf-8', level=level)

    # Per voxel function
    def calculate_voxel_metrics_per_voxel(voxel_df, min_rays=6, epsilon=1e-9):
        """
        Calculate voxel metrics for a given voxel dataframe.
        
        INPUTS:
            voxel_df: A pandas dataframe containing voxel data
            min_rays: Minimum number of rays to consider a voxel valid
            epsilon: A small value to avoid division by zero

        OUTPUTS:
            voxel_metrics: A pandas dataframe containing the calculated metrics for each voxel
        """
        
        # Check if the dataframe is empty
        if len(voxel_df) == 0:
            return pd.DataFrame(columns=voxel_metrics_schema_oldcode.names)

        # Get voxel_id name
        voxel_id = voxel_df.name

        # Calculate the number of rays in each voxel
        num_rays = voxel_df['ray_id'].count()
        if num_rays <= 0:
            statement= f"Voxel {voxel_df['voxel_id'].values[0]} has no rays."
            print(statement)
            return pd.DataFrame(columns=voxel_metrics_schema_oldcode.names)
        
        num_hits = voxel_df['hit_ray'].sum()
        num_leaf_hits = voxel_df[(voxel_df['hit_ray']) & (voxel_df['is_leaf'])].shape[0] if is_leaf_true else voxel_df[(voxel_df['hit_ray']) & ~(voxel_df['is_leaf'])].shape[0]

        # Prepare common masks
        hit_mask = voxel_df['hit_ray'].values
        leaf_mask = hit_mask & voxel_df['is_leaf'].values if is_leaf_true else hit_mask & ~voxel_df['is_leaf'].values

        # Calculate mean viewing angles
        mean_angle_all = np.nanmean(voxel_df['viewing_angle'][hit_mask].values)
        mean_angle_leaf = np.nanmean(voxel_df['viewing_angle'][leaf_mask].values)

        # Calculate pgap_lw and I
        pgap_lw = (num_rays - num_hits) / num_rays
        I_lw = num_hits / num_rays
        pgap_leaf = (num_rays - num_leaf_hits) / num_rays
        I_leaf = num_leaf_hits / num_rays

        # Calcualte path lengths
        path_lengths = np.linalg.norm(voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values, axis=1)
        
        free_path_lengths = np.where(
            hit_mask,
            np.linalg.norm(voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values - voxel_df[['point_x', 'point_y', 'point_z']].values, axis=1),
            path_lengths
        )

        # Calculate the sums and means
        sum_path_length = np.nansum(path_lengths)
        mean_path_length = np.nanmean(path_lengths)
        sum_free_path_length = np.nansum(free_path_lengths)
        mean_free_path_length = np.nanmean(free_path_lengths)
        sum_free_path_length_hit = np.nansum(free_path_lengths[hit_mask])
        sum_free_path_length_hit_leaf = np.nansum(free_path_lengths[leaf_mask])
        # mean_free_path_length_leaf = np.nanmean(free_path_lengths_leaf)

        
        eff_path_lengths = calculate_effective_path_length(path_lengths, lambda_1)
        eff_free_path_lengths = calculate_effective_path_length(free_path_lengths, lambda_1)

        # Calculated the mean and var of effective path lengths and free path lengths
        mean_eff_path_length = np.nanmean(eff_path_lengths)
        var_eff_path_length = np.nanvar(eff_path_lengths)
        mean_eff_free_path_length = np.nanmean(eff_free_path_lengths)
        var_eff_free_path_length = np.nanvar(eff_free_path_lengths)

        # Calculate extra effective free path lengths values
        sum_eff_free_path_length = np.nansum(eff_free_path_lengths)
        sum_eff_free_path_lengths_hit = np.nansum(eff_free_path_lengths[hit_mask])
        sum_eff_free_path_lengths_hit_leaf = np.nansum(eff_free_path_lengths[leaf_mask])

        # Calculate LIAD and G using just leaf points
        valid_points_leaf = voxel_df[['point_x', 'point_y', 'point_z']][leaf_mask].values
        valid_normals_leaf = voxel_df[['normal_x', 'normal_y', 'normal_z']].values[leaf_mask]
        valid_weights_leaf = voxel_df['point_weight'].values[leaf_mask]
        valid_viewing_angles_leaf = voxel_df['viewing_angle'].values[leaf_mask]

        # Check for NAN points
        if np.isnan(valid_points_leaf).any():
            statement = f"Voxel {voxel_df['voxel_id'].values[0]} has NaN points."
            print(statement)

            mask = ~np.isnan(valid_points_leaf).any(axis=1)
            valid_points_leaf = valid_points_leaf[mask]
            valid_normals_leaf = valid_normals_leaf[mask]
            valid_weights_leaf = valid_weights_leaf[mask]
            valid_viewing_angles_leaf = valid_viewing_angles_leaf[mask]

        bin_centres, LIAD_leaf_values, angles = calculate_LIAD(normals=valid_normals_leaf, weights=valid_weights_leaf)
        G_leaf = calculate_G(viewing_angles=valid_viewing_angles_leaf, bin_centres=bin_centres, LIAD_values=LIAD_leaf_values)
        G_leaf = G_leaf.mean() if isinstance(G_leaf, np.ndarray) else G_leaf
       
        if len(LIAD_leaf_values) == 0:
            LIAD_leaf_values = [np.nan] * 18

        data = {
            'voxel_id': voxel_id,
            'num_rays': num_rays,
            'num_hits': num_hits,
            'num_leaf_hits': num_leaf_hits,
            'pgap_lw': pgap_lw,
            'pgap_leaf': pgap_leaf,
            'I_lw': I_lw,
            'I_leaf': I_leaf,
            'G_lw': np.nan,
            'G_leaf': G_leaf,
            'LIAD_leaf_bin_2.5': LIAD_leaf_values[0],
            'LIAD_leaf_bin_7.5': LIAD_leaf_values[1],
            'LIAD_leaf_bin_12.5': LIAD_leaf_values[2],
            'LIAD_leaf_bin_17.5': LIAD_leaf_values[3],
            'LIAD_leaf_bin_22.5': LIAD_leaf_values[4],
            'LIAD_leaf_bin_27.5': LIAD_leaf_values[5],
            'LIAD_leaf_bin_32.5': LIAD_leaf_values[6],
            'LIAD_leaf_bin_37.5': LIAD_leaf_values[7],
            'LIAD_leaf_bin_42.5': LIAD_leaf_values[8],
            'LIAD_leaf_bin_47.5': LIAD_leaf_values[9],
            'LIAD_leaf_bin_52.5': LIAD_leaf_values[10],
            'LIAD_leaf_bin_57.5': LIAD_leaf_values[11],
            'LIAD_leaf_bin_62.5': LIAD_leaf_values[12],
            'LIAD_leaf_bin_67.5': LIAD_leaf_values[13],
            'LIAD_leaf_bin_72.5': LIAD_leaf_values[14],
            'LIAD_leaf_bin_77.5': LIAD_leaf_values[15],
            'LIAD_leaf_bin_82.5': LIAD_leaf_values[16],
            'LIAD_leaf_bin_87.5': LIAD_leaf_values[17],
            # 'LIAD_lw_bin_2.5': LIAD_lw_values[0],
            # 'LIAD_lw_bin_7.5': LIAD_lw_values[1],
            # 'LIAD_lw_bin_12.5': LIAD_lw_values[2],
            # 'LIAD_lw_bin_17.5': LIAD_lw_values[3],
            # 'LIAD_lw_bin_22.5': LIAD_lw_values[4],
            # 'LIAD_lw_bin_27.5': LIAD_lw_values[5],
            # 'LIAD_lw_bin_32.5': LIAD_lw_values[6],
            # 'LIAD_lw_bin_37.5': LIAD_lw_values[7],
            # 'LIAD_lw_bin_42.5': LIAD_lw_values[8],
            # 'LIAD_lw_bin_47.5': LIAD_lw_values[9],
            # 'LIAD_lw_bin_52.5': LIAD_lw_values[10],
            # 'LIAD_lw_bin_57.5': LIAD_lw_values[11],
            # 'LIAD_lw_bin_62.5': LIAD_lw_values[12],
            # 'LIAD_lw_bin_67.5': LIAD_lw_values[13],
            # 'LIAD_lw_bin_72.5': LIAD_lw_values[14],
            # 'LIAD_lw_bin_77.5': LIAD_lw_values[15],
            # 'LIAD_lw_bin_82.5': LIAD_lw_values[16],
            # 'LIAD_lw_bin_87.5': LIAD_lw_values[17],
            # 'mean_angle_lw': mean_angle_lw,
            'mean_angle_leaf': mean_angle_leaf,
            'mean_angle_all': mean_angle_all,
            'mean_path_length': mean_path_length,
            'sum_path_length': sum_path_length,
            'mean_free_path_length': mean_free_path_length,
            'sum_free_path_length': sum_free_path_length,
            'sum_free_path_length_hit': sum_free_path_length_hit,
            'sum_free_path_length_hit_leaf': sum_free_path_length_hit_leaf,
            'mean_eff_path_length': mean_eff_path_length,
            'var_eff_path_length': var_eff_path_length,
            'mean_eff_free_path_length': mean_eff_free_path_length,
            'var_eff_free_path_length': var_eff_free_path_length,
            'sum_eff_free_path_length': sum_eff_free_path_length,
            'sum_eff_free_path_length_hit': sum_eff_free_path_lengths_hit,
            'sum_eff_free_path_length_hit_leaf': sum_eff_free_path_lengths_hit_leaf
        }
        voxel_metrics = pd.DataFrame(data, index=[0], columns=voxel_metrics_schema_oldcode.names)

        return voxel_metrics

    # # Find available memory
    # available_memory = psutil.virtual_memory().available
    # available_memory_mb = available_memory / (1024 * 1024)


    # Read all parquets into dask dataframe
    dfs = []
    for file in intersections_files:
        if os.path.exists(file):
            df = dd.read_parquet(file, engine='pyarrow') # add later if needed: blocksize=None)

            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    # Combine all dataframes into one
    voxel_intersections_df = dd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.repartition(npartitions=1)
    voxel_intersections_df = voxel_intersections_df.groupby('voxel_id')
    unique_voxel_ids = voxel_intersections_df['voxel_id'].unique().compute()
    num_voxels = len(unique_voxel_ids)

    # Extract requisite information for density calculations
    meta = pd.DataFrame(columns=voxel_metrics_schema_oldcode.names)
    voxel_metrics_df = voxel_intersections_df.apply(calculate_voxel_metrics_per_voxel, meta=meta)

    # Return the calculated metrics
    with ProgressBar():
        voxel_metrics_df = voxel_metrics_df.compute()
        voxel_metrics_df = voxel_metrics_df.reset_index(drop=True)
    return voxel_metrics_df

# Calculate effective path lengths and free path lengths
def calculate_effective_path_length(path_lengths, lambda_1):
    with np.errstate(divide='ignore', invalid='ignore'):
        mask = (lambda_1 * path_lengths) < 1
        effective_path_length = np.where(
            mask,
            -np.log(1 - lambda_1 * path_lengths) / lambda_1,
            np.nan
        )
    return effective_path_length

def lad_bl_suite(num_rays,
                 mean_path,              # 5_   (m)
                 G_leaf,                 # voxel-specific G(ÃÂ¸)
                 P_first, P_equal, P_int,
                 P_ideal, P_exact,
                 eps=1e-9):
    """
    Returns (LAD_first, LAD_equal, LAD_intensity, LAD_ideal, LAD_exact)

    Kent & Baileys P_* are *gap* (transmission) probabilities.
    Beer-Lambert inversion for gap probability is:

        LAD = -ln(P_gap) / (G_leaf ÃÂ· mean_path)

    Parameters
    ----------
    num_rays   : int    -# rays that crossed the voxel (used only for 0-ray shortcut)
    mean_path  : float  -ÃÂ¨path lengthÃÂ© inside the voxel   (m)
    G_leaf     : float  -G(ÃÂ¸) for leaves in this voxel
    P_*        : float  -Kent & Bailey Table-1 transmission probability
    eps        : float  -small constant to avoid log(0)
    """
    if (num_rays == 0 or
        not np.isfinite(mean_path) or mean_path <= 0 or
        not np.isfinite(G_leaf)   or G_leaf   <= 0):
        return (np.nan,)*5

    ##### helper #####################
    def _bl_gap(P_gap):
        if not np.isfinite(P_gap):
            return np.nan
        P_safe = np.clip(P_gap, eps, 1.0 - eps)   # keep in (0,1)
        return -np.log(P_safe) / (G_leaf * mean_path)

    return tuple(map(_bl_gap,
                     (P_first, P_equal, P_int, P_ideal, P_exact)))

def get_voxel_metrics(intersections_files, lambda_1, beam_divergence=0.35, is_multireturn=False, is_leaf_true=True, debug=True, epsilon=1e-9):
    """
    This function will take the voxel_ray_intersection files and calculate the metrics for each voxel.
    It will save the results to a parquet file.

    Args:
        intersections_files (list): List of paths to voxel_ray_intersection files.
        lambda_1 (float): This is calculated using (average leaf area / voxel size) and will need to be calculated and passed in.
        beam_divergence: The mrad beam divergence rating of your used lidar sensor (mrad) (0.35 default riegl vz400i mrad at 1/e2)
        debug (bool): Whether to print debug information.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        None
    """
    import os
    import glob
    import psutil
    import pandas as pd
    import numpy as np
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    from dask.distributed import Client
    from collections import defaultdict

    avail_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', psutil.cpu_count(logical=True)))
    threads = 1 if not os.environ.get('SLURM_CPUS_PER_TASK') else 2
    avail_mem = int(float(os.environ.get('SLURM_MEM_PER_NODE', psutil.virtual_memory().available // (1024 * 1024)))) # in MB
    avail_mem = avail_mem * 0.9 // 1024 # Use 80% of available memory and convert to GB
    mem_limit = avail_mem // avail_cpus
    mem_limit = f"{mem_limit}GB"
    _start_dask_client(memory_limit=mem_limit, n_workers=avail_cpus)

    # Per voxel function
    def calculate_voxel_metrics_per_voxel(voxel_df, min_rays=6, epsilon=1e-9):
        """
        Calculate voxel metrics for a given voxel dataframe.
        
        INPUTS:
            voxel_df: A pandas dataframe containing voxel data
            min_rays: Minimum number of rays to consider a voxel valid
            epsilon: A small value to avoid division by zero

        OUTPUTS:
            voxel_metrics: A pandas dataframe containing the calculated metrics for each voxel
        """
        # Check if the dataframe is empty
        schema = voxel_metrics_schema_singlereturn if not is_multireturn else voxel_metrics_schema_multireturn
        data_df = _gen_dataframe(schema)
        if len(voxel_df) == 0:
            return data_df

        voxel_id = voxel_df.name
        voxel_cx = voxel_df['voxel_cx'].min()
        voxel_cy = voxel_df['voxel_cy'].min()
        voxel_cz = voxel_df['voxel_cz'].min()

        hit_types = voxel_df['hit_type'].values
        unbound = hit_types == 0
        previous_hit = hit_types == 1
        current_hit = hit_types == 2
        yet_to_hit = hit_types == 3
        # Exclude rays that have no current hits or yet_to_hits
        valid_ray_mask = unbound | current_hit | yet_to_hit

        # print(f"hit_types = {unbound.sum() + previous_hit.sum() + current_hit.sum() + yet_to_hit.sum()} for {num_rays} rays")

        leaf_mask = voxel_df['is_leaf'].values if is_leaf_true else ~voxel_df['is_leaf'].values
        current_leaf_mask = current_hit & leaf_mask
        num_lw_hits = current_hit.sum()
        num_leaf_hits = current_leaf_mask.sum()
        num_before_hits = previous_hit.sum()
        num_after_hits = yet_to_hit.sum()
        num_unbound_rays = unbound.sum()


        num_rays = voxel_df['ray_id'][valid_ray_mask].nunique()
        if num_rays <= 0:
            statement= f"Voxel {voxel_df['voxel_id'].values[0]} has no rays."
            print(statement)
            return data_df

        # Calculate mean viewing angles
        mean_angle_all = np.nanmean(voxel_df['viewing_angle'][current_hit].values) if current_hit.sum() > 0 else np.nan
        mean_angle_leaf = np.nanmean(voxel_df['viewing_angle'][current_leaf_mask].values) if current_leaf_mask.sum() > 0 else np.nan

        # Calculate pgap_lw and I
        pgap_lw = (num_rays - num_lw_hits) / max(num_rays, 1)
        pgap_leaf = (num_rays - num_leaf_hits) / max(num_rays, 1)
        I_lw, I_leaf = 1.0 - pgap_lw, 1.0 - pgap_leaf

        # Calculate path lengths
        # print(f"Num valid paths = {valid_path_length_mask.sum()} for {num_rays} rays")
        ent = voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values
        ext = voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values
        path_lengths = np.full(len(voxel_df), np.nan, dtype=np.float64)
        path_lengths[valid_ray_mask] = np.linalg.norm(ext[valid_ray_mask] - ent[valid_ray_mask], axis=1)

        # Calculate free path lengths for multi-return rays
        # For each ray, if return_number is the minimum for that ray, use entry to point
        # Otherwise, use distance from previous point to current point
        free_path_lengths = path_lengths.copy()
        ray_ids = voxel_df['ray_id'].values
        return_numbers = voxel_df['return_number'].values if 'return_number' in voxel_df.columns else np.ones(len(voxel_df), dtype=int)
        points = voxel_df[['point_x', 'point_y', 'point_z']].values

        # Unbound fpl = pl
        free_path_lengths[unbound] = path_lengths[unbound]

        # Efficient method for single return rays
        single_return_mask = voxel_df['number_of_returns'] == 1
        free_path_lengths[single_return_mask & current_hit] = np.linalg.norm(points[single_return_mask & current_hit] - ent[single_return_mask & current_hit], axis=1)
        free_path_lengths[single_return_mask & yet_to_hit] = path_lengths[single_return_mask & yet_to_hit]
        free_path_lengths[single_return_mask & previous_hit] = np.nan
        

        multi_return_mask = voxel_df['number_of_returns'] > 1

        if multi_return_mask.sum() > 0:
            # For each ray, sort by return_number and compute distances
            # Simulataneously establish echo_types and echo_intensities
            # Group indices by ray_id
            inside_ray_indices = defaultdict(list)
            next_hit_ray_indices = defaultdict(list)
            original_indices = np.where(multi_return_mask)[0]  # Get indices of multi-return rays
            for idx, ray_id in zip(original_indices, ray_ids[multi_return_mask]):
                if hit_types[idx] == 2:  # Only consider current hits
                    inside_ray_indices[ray_id].append(idx)
                if hit_types[idx] == 3:  # Only consider yet_to_hits
                    next_hit_ray_indices[ray_id].append(idx)

            for inside_indices in inside_ray_indices.values():
                # Sort indices by return_number
                sorted_indices = sorted(inside_indices, key=lambda i: return_numbers[i])
                min_return_number = np.min(sorted_indices)
                max_return_number = np.max(sorted_indices)
                for i, idx in enumerate(sorted_indices):
                    if i == min_return_number:
                        # First return: entry to point
                        free_path_lengths[idx] = np.linalg.norm(points[idx] - ent[idx])
                    else:
                        # Subsequent returns: previous point to current point
                        prev_idx = sorted_indices[i - 1]
                        free_path_lengths[idx] = np.linalg.norm(points[idx] - points[prev_idx])
                    
                    # Check last point for continuing points
                    if i == max_return_number:
                        # For the last return, check if there is a next hit in next_hit_ray_indices for the same ray_id with the next return_number
                        next_indices = next_hit_ray_indices.get(ray_id, [])
                        next_return_number = return_numbers[idx] + 1
                        next_idx = None
                        for ni in next_indices:
                            if return_numbers[ni] == next_return_number:
                                next_idx = ni
                                break
                        if next_idx is not None:
                            free_path_lengths[next_idx] = np.linalg.norm(ext[idx] - points[idx])
        
        # Effective path lengths
        eff_path_lengths = calculate_effective_path_length(
            path_lengths=path_lengths, 
            lambda_1=lambda_1
        )

        eff_free_path_lengths = calculate_effective_path_length(
            path_lengths=free_path_lengths, 
            lambda_1=lambda_1
        )

        # Create aggregrate values
        mean_path_length = np.nanmean(path_lengths)
        sum_path_length = np.nansum(path_lengths)

        mean_free_path_length = np.nanmean(free_path_lengths)
        sum_free_path_length = np.nansum(free_path_lengths)
        sum_free_path_length_hit = np.nansum(free_path_lengths[current_hit])
        sum_free_path_length_exit = np.nansum(free_path_lengths[yet_to_hit])
        sum_free_path_length_hit_leaf = np.nansum(free_path_lengths[current_leaf_mask])

        mean_eff_path_length = np.nanmean(eff_path_lengths)
        var_eff_path_length = np.nanvar(eff_path_lengths)
        sum_eff_path_length = np.nansum(eff_path_lengths)

        mean_eff_free_path_length = np.nanmean(eff_free_path_lengths)
        sum_eff_free_path_length = np.nansum(eff_free_path_lengths)
        var_eff_free_path_length = np.nanvar(eff_free_path_lengths)
        sum_eff_free_path_length_hit = np.nansum(eff_free_path_lengths[current_hit])
        sum_eff_free_path_length_exit = np.nansum(eff_free_path_lengths[yet_to_hit])
        sum_eff_free_path_length_hit_leaf = np.nansum(eff_free_path_lengths[current_leaf_mask])


        # ------ LIAD & G(.) -------- #
        leaf_normals = voxel_df[['normal_x', 'normal_y', 'normal_z']].values[current_leaf_mask]
        leaf_weights = voxel_df['point_weight'].values[current_leaf_mask]
        bins, liad_vals, _ = calculate_LIAD(
            normals=leaf_normals, 
            weights=leaf_weights
        )

        view_angles_lw = voxel_df['viewing_angle'].values[current_hit]
        view_angles_leaf = voxel_df['viewing_angle'].values[current_leaf_mask]
        G_leaf = calculate_G(
            viewing_angles=view_angles_leaf, 
            bin_centres=bins, 
            LIAD_values=liad_vals
        )
        G_leaf = G_leaf.mean() if isinstance(G_leaf, np.ndarray) else G_leaf
        if not np.isfinite(G_leaf):
            G_leaf = np.nan
        
        G_lw = calculate_G(
            viewing_angles=view_angles_lw,
            bin_centres=bins,
            LIAD_values=liad_vals
        )
        G_lw = G_lw.mean() if isinstance(G_lw, np.ndarray) else G_lw
        if not np.isfinite(G_lw):
            G_lw = np.nan

        # If Multi-return, calculate probabilities and use appropriate LAD/PAD calcs
        if is_multireturn:

            # Kent & Bailey probabilities
            P_first = P_equal = P_int = np.nan
            LAD_first = LAD_equal = LAD_int = np.nan
            LAD_MLE_g = np.nan        

            # Establish the Tk and Wk values for first_hit, equal_hit, and intensity_hit weighting probability functions
            def _collapse(T, W):
                tot = W.sum()
                return float((T * W).sum() / tot) if tot else np.nan
            
            # -- First Hit Weighting -- #
            first_hit = voxel_df['return_number'] == 1
            current_first_hit = current_hit & first_hit
            Tk_first_lw = np.nansum(current_first_hit)
            Wk_first_lw = np.nansum(current_hit)        ### CHECK THIS. Might be rays rather than hits in the voxel

            current_first_hit_leaf = current_hit & leaf_mask
            Tk_first_leaf = np.nansum(current_first_hit_leaf)
            Wk_first_leaf = Wk_first_lw                 ### CHECK THIS

            # -- Equal Hit Weighting -- #
            echoes_before_lw = np.count_nonzero(previous_hit)
            echoes_during_lw = np.count_nonzero(current_hit)
            echoes_after_lw = np.count_nonzero(yet_to_hit)
            Tk_equal_lw = echoes_after_lw / np.clip((echoes_during_lw + echoes_after_lw), 1, None)
            Wk_equal_lw = (echoes_during_lw + echoes_after_lw) / np.clip((echoes_before_lw + echoes_during_lw + echoes_after_lw), 1, None)

            echoes_before_leaf = np.count_nonzero(previous_hit & leaf_mask)
            echoes_during_leaf = np.count_nonzero(current_hit & leaf_mask)
            echoes_after_leaf = np.count_nonzero(yet_to_hit & leaf_mask)
            Tk_equal_leaf = echoes_after_leaf / np.clip((echoes_during_leaf + echoes_after_leaf), 1, None)
            Wk_equal_leaf = (echoes_during_leaf + echoes_after_leaf) / np.clip((echoes_before_leaf + echoes_during_leaf + echoes_after_leaf), 1, None)

            # -- Intensity Hit Weighting -- #
            echo_intensities = voxel_df['echo_intensity'].values
            intensity_before_lw = np.nansum(echo_intensities[previous_hit])
            intensity_during_lw = np.nansum(echo_intensities[current_hit])
            intensity_after_lw = np.nansum(echo_intensities[yet_to_hit])

            denom_lw = intensity_during_lw + intensity_after_lw
            Tk_int_lw = intensity_after_lw / denom_lw if denom_lw != 0 else np.nan
            Wk_int_lw = (intensity_during_lw + intensity_after_lw) / (intensity_before_lw + intensity_during_lw + intensity_after_lw)

            intensity_before_leaf = np.nansum(echo_intensities[previous_hit & leaf_mask])
            intensity_during_leaf = np.nansum(echo_intensities[current_hit & leaf_mask])
            intensity_after_leaf = np.nansum(echo_intensities[yet_to_hit & leaf_mask])
            del echo_intensities
            denom_leaf = intensity_during_leaf + intensity_after_leaf
            Tk_int_leaf = intensity_after_leaf / denom_leaf if denom_leaf != 0 else np.nan
            Wk_int_leaf = (intensity_during_leaf + intensity_after_leaf) / (intensity_before_leaf + intensity_during_leaf + intensity_after_leaf)

            P_first_lw, P_equal_lw, P_int_lw, P_first_leaf, P_equal_leaf, P_int_leaf = (
                _collapse(*args) for args in [
                    (Tk_first_lw, Wk_first_lw), 
                    (Tk_equal_lw, Wk_equal_lw),
                    (Tk_int_lw,   Wk_int_lw),
                    (Tk_first_leaf, Wk_first_leaf),
                    (Tk_equal_leaf, Wk_equal_leaf),
                    (Tk_int_leaf,   Wk_int_leaf)
                ]
            )

            # These LAD calculations follow preliminary code from Raja, which used the P_{weight} values as substitutes for pgap in BL LAD estimation
            # They used P_first_lw (as calculated above) with G_leaf
            # # PRELIMINARY CODE
            # LAD_first, LAD_equal, LAD_int, LAD_ideal, LAD_exact = lad_bl_suite(
            #     num_rays, mean_path_length, G_leaf,
            #     P_first, P_equal, P_int, P_ideal, P_exact
            # )
            # lad_bl_suite --> -log(P_{weight} / (G_leaf * mean_path_length))

            LAD_first = BL_pimont_2018(
                P=P_first_lw,
                mean_path_length=mean_path_length,
                G=G_leaf,
                CI=1.0
            )

            LAD_equal = BL_pimont_2018(
                P=P_equal_lw,
                mean_path_length=mean_path_length,
                G=G_leaf,
                CI=1.0
            )

            LAD_int = BL_pimont_2018(
                P=P_int_lw,
                mean_path_length=mean_path_length,
                G=G_leaf,
                CI=1.0
            )

            # # -------- Vincent-2021 & Bai-2024 MLEs --------------------
            # This code uses the same LAD inputs as preliminary set by Raja
            # Preliminary code:
            # LAD_MLE_g = LAD_MLE_geom_corr(
            #     num_leaf_hits, beam_area_leaf, beam_area_all, free_path_lengths,
            #     G_leaf, CI=CI_vox, k1=k1_vox, bias_corr=True,
            # )


            distance_to_centre = voxel_df['distance_to_centre'].values
            ray_weights = 1.0 / voxel_df['number_of_returns'].values    # NOTE: unbound rays should have weight of 0, so any unbound ratio needs to ignore weights
            # Calculate the surface area of the beam cross section at the distance to the voxel centre
            # Surface area = π * (radius)^2, where radius = distance_to_centre * beam_divergence
            beam_surface_area_all = np.full(distance_to_centre.shape, np.nan, dtype=np.float64)
            beam_divergence_rad = beam_divergence * 1e-3  # Convert mrad to rad
            beam_radius = distance_to_centre[valid_ray_mask] * beam_divergence_rad
            beam_surface_area_all[valid_ray_mask] = np.pi * np.square(beam_radius)

            # Mask unique ray_ids (i.e. pulses)
            unique_rays_mask = np.unique(ray_ids, return_index=True)[1]

            ### CALCULATE NUMERATOR FOR MLE ###
            # Find sum(Sq * sum(αjq))

            ## Sq is unique_pulse_area
            unique_pulse_area = beam_surface_area_all[unique_rays_mask]

            # Precompute unique_ray_ids and sorter for mapping
            unique_ray_ids = ray_ids[unique_rays_mask]
            sorter = np.argsort(unique_ray_ids)

            # Precompute indices for each hit type
            idx_current_hit = np.searchsorted(unique_ray_ids, ray_ids[current_hit], sorter=sorter)
            idx_yet_to_hit = np.searchsorted(unique_ray_ids, ray_ids[yet_to_hit], sorter=sorter)
            idx_unbound = np.searchsorted(unique_ray_ids, ray_ids[unbound], sorter=sorter)
            idx_current_or_yet = np.searchsorted(unique_ray_ids, ray_ids[current_hit | yet_to_hit], sorter=sorter)

            # Precompute ray_weights for each hit type
            ray_weights_current_hit = ray_weights[current_hit]
            ray_weights_yet_to_hit = ray_weights[yet_to_hit]
            ray_weights_unbound = ray_weights[unbound]
            ray_weights_current_or_yet = ray_weights[current_hit | yet_to_hit]

            # Precompute path_lengths and free_path_lengths for each hit type
            path_lengths_yet_to_hit = path_lengths[yet_to_hit]
            path_lengths_unbound = path_lengths[unbound]
            free_path_lengths_current_hit = free_path_lengths[current_hit]
            eff_free_path_lengths_current_hit = eff_free_path_lengths[current_hit]
            eff_path_lengths_yet_to_hit = eff_path_lengths[yet_to_hit]
            eff_path_lengths_unbound = eff_path_lengths[unbound]

            # 1. sum(αjq) is hit_weights_per_pulse
            unique_weights_hit = np.bincount(idx_current_hit, weights=ray_weights_current_hit, minlength=unique_ray_ids.size)

            # Sq * sum(αjq) is sum_ba_hit
            sum_ba_hit = np.nansum(unique_pulse_area * unique_weights_hit)

            # 2. sum(αjq * FPLjq) is unique_fpl_hit
            unique_fpl_hit = np.bincount(
                idx_current_hit,
                weights=free_path_lengths_current_hit * ray_weights_current_hit,
                minlength=unique_ray_ids.size
            )

            # 3. αoutq * plq is unique_pl_exit
            sum_yet_to_hit = np.bincount(
                idx_yet_to_hit,
                weights=path_lengths_yet_to_hit * ray_weights_yet_to_hit,
                minlength=unique_ray_ids.size
            )
            sum_unbound = np.bincount(
                idx_unbound,
                weights=path_lengths_unbound * 1.0,
                minlength=unique_ray_ids.size
            )
            unique_pl_exit = sum_yet_to_hit + sum_unbound

            # sum(αjq * FPLjq) + αoutq * plq is sum_pl_all
            sum_pl_all = np.nansum(unique_pulse_area * (unique_fpl_hit + unique_pl_exit))

            # 4. sum(αjq * eff_FPLjq) is unique_eff_fpl_hit
            unique_eff_fpl_hit = np.bincount(
                idx_current_hit,
                weights=eff_free_path_lengths_current_hit * ray_weights_current_hit,
                minlength=unique_ray_ids.size
            )

            # 5. αoutq * eff_plq is unique_eff_pl_exit
            sum_yet_to_hit_eff = np.bincount(
                idx_yet_to_hit,
                weights=eff_path_lengths_yet_to_hit * ray_weights_yet_to_hit,
                minlength=unique_ray_ids.size
            )
            sum_unbound_eff = np.bincount(
                idx_unbound,
                weights=eff_path_lengths_unbound * 1.0,
                minlength=unique_ray_ids.size
            )
            unique_eff_pl_exit = sum_yet_to_hit_eff + sum_unbound_eff

            # sum(αjq * eff_FPLjq) + αoutq * eff_plq is sum_pl_all_eff
            sum_pl_all_eff = np.nansum(unique_pulse_area * (unique_eff_fpl_hit + unique_eff_pl_exit))

            # 6. sum(αinq) is unique_weights_enter
            sum_current_or_yet = np.bincount(
                idx_current_or_yet,
                weights=ray_weights_current_or_yet,
                minlength=unique_ray_ids.size
            )
            sum_unbound_enter = np.bincount(
                idx_unbound,
                weights=np.ones_like(idx_unbound, dtype=np.float64),
                minlength=unique_ray_ids.size
            )
            unique_weights_enter = sum_current_or_yet + sum_unbound_enter

            ## First part: sum(Sq * sum(αinq)) / num_rays is bias_pt_1
            bias_pt_1 = np.nansum(unique_pulse_area * unique_weights_enter) / num_rays

            ## Calculate Second Part: sum(Sq * (sum(αjq * FPLjq)) / sum(Sq * (sum(αjq * FPLjq) + αoutq * plq)))
            # This is simply (sum(pulse_area * unique_fpl_hit) / sum_pl_all from above
            bias_pt_2 = np.nansum(unique_pulse_area * unique_fpl_hit) / sum_pl_all if sum_pl_all != 0 else np.nan

            bias_corr = bias_pt_1 * bias_pt_2

            ### CALCULATE BIAS CORRECTION (with lambda_1 correction) ###
            # Find (sum(Sq * sum(αinq)) / num_rays) * (sum(Sq * (sum(αjq * eff_FPLjq)) / sum(Sq * (sum(αjq * eff_FPLjq) + αoutq * eff_plq)))
            # Sq is unique_pulse_area (as above)
            # num_rays is already calculated above
            
            ## bias_pt_1 is the same as above

            ## Calculate Second Part: sum(Sq * (sum(αjq * eff_FPLjq)) / sum(Sq * (sum(αjq * eff_FPLjq) + αoutq * eff_plq)))
            # This is simply (sum(pulse_area * unique_eff_fpl_hit) / sum_pl_all_eff from above
            bias_pt_2_eff = np.nansum(unique_pulse_area * unique_eff_fpl_hit) / sum_pl_all_eff if sum_pl_all_eff != 0 else np.nan

            bias_corr_eff = bias_pt_1 * bias_pt_2_eff

            LAD_MLE_nocorr = MLE_vincent_2021(
                sum_ba_hit=sum_ba_hit,      # Numerator
                sum_pl_all=sum_pl_all,      # Denominator
                G=G_leaf,
                CI=1.0,
                bias_corr=None
            )
            
            LAD_MLE_lambda1 = MLE_vincent_2021(
                sum_ba_hit = sum_ba_hit,        # Numerator. No change for lambda_1 correction
                sum_pl_all = sum_pl_all_eff,    # Denominator
                G=G_leaf,
                CI=1.0,
                bias_corr=None
            )

            LAD_MLE_bias = MLE_vincent_2021(
                sum_ba_hit=sum_ba_hit,          # Numerator
                sum_pl_all=sum_pl_all,          # Denominator
                G=G_leaf,
                CI=1.0,
                bias_corr=bias_corr
            )

            LAD_MLE_lambda1_bias = MLE_vincent_2021(
                sum_ba_hit=sum_ba_hit,          # Numerator
                sum_pl_all=sum_pl_all_eff,      # Denominator
                G=G_leaf,
                CI=1.0,
                bias_corr=bias_corr_eff         # Bias correction with lambda_1 correction
            )

            data_df.loc[0, 'P_first'] = P_first
            data_df.loc[0, 'P_equal'] = P_equal
            data_df.loc[0, 'P_intensity'] = P_int
            data_df.loc[0, 'P_first_leaf'] = P_first_leaf
            data_df.loc[0, 'P_equal_leaf'] = P_equal_leaf
            data_df.loc[0, 'P_intensity_leaf'] = P_int_leaf
            data_df.loc[0, 'LAD_first'] = LAD_first
            data_df.loc[0, 'LAD_equal'] = LAD_equal
            data_df.loc[0, 'LAD_intensity'] = LAD_int
            data_df.loc[0, 'LAD_MLE_nocorr'] = LAD_MLE_nocorr
            data_df.loc[0, 'LAD_MLE_lambda1'] = LAD_MLE_lambda1
            data_df.loc[0, 'LAD_MLE_bias'] = LAD_MLE_bias
            data_df.loc[0, 'LAD_MLE_lambda1_bias'] = LAD_MLE_lambda1_bias

        # Add values for both single and multi-return
        data_df.loc[0, 'voxel_id'] = voxel_id
        data_df.loc[0, 'voxel_cx'] = voxel_cx
        data_df.loc[0, 'voxel_cy'] = voxel_cy
        data_df.loc[0, 'voxel_cz'] = voxel_cz
        data_df.loc[0, 'num_rays'] = num_rays
        data_df.loc[0, 'num_hits'] = num_lw_hits
        data_df.loc[0, 'num_leaf_hits'] = num_leaf_hits
        data_df.loc[0, 'pgap_lw'] = pgap_lw
        data_df.loc[0, 'pgap_leaf'] = pgap_leaf
        data_df.loc[0, 'I_lw'] = I_lw
        data_df.loc[0, 'I_leaf'] = I_leaf
        data_df.loc[0, 'G_lw'] = G_lw
        data_df.loc[0, 'G_leaf'] = G_leaf
        data_df.loc[0, 'lambda_1'] = lambda_1
        data_df.loc[0, 'LIAD_leaf_bin_2.5'] = np.float32(liad_vals[0]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_7.5'] = np.float32(liad_vals[1]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_12.5'] = np.float32(liad_vals[2]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_17.5'] = np.float32(liad_vals[3]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_22.5'] = np.float32(liad_vals[4]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_27.5'] = np.float32(liad_vals[5]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_32.5'] = np.float32(liad_vals[6]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_37.5'] = np.float32(liad_vals[7]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_42.5'] = np.float32(liad_vals[8]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_47.5'] = np.float32(liad_vals[9]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_52.5'] = np.float32(liad_vals[10]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_57.5'] = np.float32(liad_vals[11]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_62.5'] = np.float32(liad_vals[12]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_67.5'] = np.float32(liad_vals[13]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_72.5'] = np.float32(liad_vals[14]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_77.5'] = np.float32(liad_vals[15]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_82.5'] = np.float32(liad_vals[16]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'LIAD_leaf_bin_87.5'] = np.float32(liad_vals[17]) if liad_vals.size > 0 else np.nan
        data_df.loc[0, 'mean_angle_leaf'] = np.float32(mean_angle_leaf)
        data_df.loc[0, 'mean_angle_all'] = np.float32(mean_angle_all)
        data_df.loc[0, 'mean_path_length'] = np.float64(mean_path_length)
        data_df.loc[0, 'sum_path_length'] = np.float64(sum_path_length)
        data_df.loc[0, 'mean_free_path_length'] = np.float64(mean_free_path_length)
        data_df.loc[0, 'sum_free_path_length'] = np.float64(sum_free_path_length)
        data_df.loc[0, 'sum_free_path_length_hit'] = np.float64(sum_free_path_length_hit)
        data_df.loc[0, 'sum_free_path_length_hit_leaf'] = np.float64(sum_free_path_length_hit_leaf)
        data_df.loc[0, 'mean_eff_path_length'] = np.float64(mean_eff_path_length)
        data_df.loc[0, 'var_eff_path_length'] = np.float64(var_eff_path_length)
        data_df.loc[0, 'sum_eff_path_length'] = np.float64(sum_eff_path_length)
        data_df.loc[0, 'mean_eff_free_path_length'] = np.float64(mean_eff_free_path_length)
        data_df.loc[0, 'var_eff_free_path_length'] = np.float64(var_eff_free_path_length)
        data_df.loc[0, 'sum_eff_free_path_length'] = np.float64(sum_eff_free_path_length)
        data_df.loc[0, 'sum_eff_free_path_length_hit'] = np.float64(sum_eff_free_path_length_hit)
        data_df.loc[0, 'sum_eff_free_path_length_hit_leaf'] = np.float64(sum_eff_free_path_length_hit_leaf)

        return data_df

    # Extract requisite information for density calculations
    schema = voxel_metrics_schema_singlereturn if not is_multireturn else voxel_metrics_schema_multireturn
    meta = _gen_dataframe(schema)

    # Read all parquets into dask dataframe
    dfs = []
    for file in intersections_files:
        if os.path.exists(file):
            df = dd.read_parquet(file, engine='pyarrow')

            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    # Combine all dataframes into one
    voxel_intersections_df = dd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.set_index('voxel_id')
    # Map partitions to voxel metrics calculation
    voxel_metrics_df = voxel_intersections_df.map_partitions(
        lambda df: df.groupby('voxel_id').apply(calculate_voxel_metrics_per_voxel), 
        meta=meta
    )

    # Return the calculated metrics
    # with ProgressBar():
    #     voxel_metrics_df = voxel_metrics_df.compute()
    #     voxel_metrics_df = voxel_metrics_df.reset_index(drop=True)

    future = DASK_CLIENT.compute(voxel_metrics_df)
    with ProgressBar():
        voxel_metrics_df = future.result()
        voxel_metrics_df = voxel_metrics_df.reset_index(drop=True)

    _close_dask_client(client=DASK_CLIENT)

    return voxel_metrics_df

def calculate_occlusion_metrics(intersections_files, reference_file, max_beam_distance=50, heat_map_resolution=0.01, debug=True, epsilon=1e-9):
    """
    This function will take the voxel_ray_intersection files and calculate the occlusion metrics for each voxel and group of points
    It will return a dataframe for voxel information:
        - Number of rays from each direction (i.e. North, South, East, West, Up, Down)
        - Total volume coverage percentave (i.e. using beam divergence, what percentage of voxel volume is explored)
        This can be used to create a voxel map .xyz which demonstrates all space of the chosen plot (i.e. not just explored voxels)
        
    It will also return a dataframe for point information, which is based on point groups:
        - Number of rays from each direction (i.e. North, South, East, West, Up, Down)
        This can be used to create a .laz file which includes extra classification information that demonstrates the points exploration metrics.
        """
    from sklearn.neighbors import NearestNeighbors
    
    dfs = []
    for file in intersections_files:
        if os.path.exists(file):
            df = pd.read_parquet(file, engine='pyarrow')
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    # Combine all dataframes into one
    voxel_intersections_df = pd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.reset_index(drop=True)

    # Retrieve reference information for voxel boundaries
    reference_df = pd.read_csv(reference_file, index_col=None, header=0)
    reference_df = reference_df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']]
    reference_df = reference_df.drop_duplicates()
    reference_df = reference_df.set_index('voxel_id')

    # Merge the voxel intersections with the reference dataframe
    voxel_intersections_df = voxel_intersections_df.merge(reference_df, left_on='voxel_id', right_index=True, how='left', suffixes=('', ''))

    del reference_df
    
    def get_occlusion_per_voxel(voxel_df, epsilon=1e-9):

        # Calculate the planes which constitute each face of the voxel
        voxel_min = voxel_df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values[0] - (voxel_df['voxel_size'].values[0] / 2)
        voxel_max = voxel_df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values[0] + (voxel_df['voxel_size'].values[0] / 2)

        # Define the six planes of the voxel using min/max for each axis
        # Each face is defined by a constant value on one axis
        # Map voxel face keys to real-world directions (assuming z is up):
        # x_min: West, x_max: East
        # y_min: South, y_max: North
        # z_min: Down, z_max: Up
        voxel_faces = {
            'west': voxel_min[0],   # x_min
            'east': voxel_max[0],   # x_max
            'south': voxel_min[1],  # y_min
            'north': voxel_max[1],  # y_max
            'bottom': voxel_min[2],   # z_min
            'top': voxel_max[2]      # z_max
        }

        points = voxel_df[['point_x', 'point_y', 'point_z']].values

        # Find which points land on each plane (plus tolerance)
        # point_planes = np.array([
        #     np.abs(points[:, 0] - voxel_planes[0, 0]) < epsilon,
        #     np.abs(points[:, 1] - voxel_planes[0, 1]) < epsilon,
        #     np.abs(points[:, 2] - voxel_planes[0, 2]) < epsilon
        # ]).T

        # Calculate the number of points on each plane
        # num_points_per_plane = np.sum(point_planes, axis=0)

        # Calculate the total volume coverage percentage
        total_volume = np.prod(voxel_max - voxel_min)
        entry_coords = voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values
        exit_coords = voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values
        entry_radii = voxel_df['t_entry_radius'].values
        exit_radii = voxel_df['t_exit_radius'].values

        # Calculate the weight of each beam, based on the distance from t_entry to sensor origin
        distance_to_exit = np.linalg.norm(exit_coords - entry_coords, axis=1)
        denom = (exit_radii - entry_radii)
        denom[denom == 0] = epsilon
        distance_to_sensor = (entry_radii * distance_to_exit) / denom

        # Weight beams linearly with max distance from sensor specified
        beam_weights = np.clip(1 - (distance_to_sensor / max_beam_distance), 0, 1)

        # Calculate the theoretical and actual beam volumes
        def calculate_beam_volume(s_coords, e_coords, s_radii, e_radii):
            distance_to_end = np.linalg.norm(s_coords - e_coords, axis=1)
            beam_volumes = ((1/3) * np.pi * distance_to_end) * (s_radii ** 2 + s_radii * e_radii + e_radii ** 2)
            return beam_volumes
        # Theoretical beam volumes
        theoretical_beam_volumes = calculate_beam_volume(entry_coords, exit_coords, entry_radii, exit_radii)
        weighted_theoretical_beam_volumes = theoretical_beam_volumes * beam_weights

        # Actual beam volumes
        hit_mask = voxel_df['hit_ray'].values
        points = voxel_df[['point_x', 'point_y', 'point_z']][hit_mask].values
        valid_entry_coords = entry_coords[hit_mask]
        valid_entry_radii = entry_radii[hit_mask]
        valid_exit_radii = exit_radii[hit_mask]
        actual_beam_volumes = theoretical_beam_volumes

        distance_to_point = np.linalg.norm(points - valid_entry_coords, axis=1)
        valid_distance_to_exit = distance_to_exit[hit_mask]
        radii_at_point = valid_entry_radii * ((valid_exit_radii - valid_entry_radii) / valid_distance_to_exit) * distance_to_point
        actual_beam_volumes[hit_mask] = calculate_beam_volume(valid_entry_coords, points, valid_entry_radii, radii_at_point)
        weighted_actual_beam_volumes = actual_beam_volumes * beam_weights

        #### INSERT DESIRED OCCLUSION METRICS HERE ####
        # At the moment, we'll just save the coverage volumes and work with the results

        # Calculate the heat map for each direction (i.e. face of the voxel intersected by t_entry)
        voxel_size = voxel_df['voxel_size'].values[0]
        bins_per_face = int(voxel_size / heat_map_resolution)
        face_heatmaps = {
            'west': np.zeros((bins_per_face, bins_per_face)),
            'east': np.zeros((bins_per_face, bins_per_face)),
            'south': np.zeros((bins_per_face, bins_per_face)),
            'north': np.zeros((bins_per_face, bins_per_face)),
            'bottom': np.zeros((bins_per_face, bins_per_face)),
            'top': np.zeros((bins_per_face, bins_per_face))
        }
        weighted_face_heatmaps = face_heatmaps.copy()

        plane_beam_theoretical_volumes = {
            'west': 0,
            'east': 0,
            'south': 0,
            'north': 0,
            'bottom': 0,
            'top': 0
        }
        plane_beam_actual_volumes = {
            'west': 0,
            'east': 0,
            'south': 0,
            'north': 0,
            'bottom': 0,
            'top': 0
        }
        plane_beam_weighted_theoretical_volumes = {
            'west': 0,
            'east': 0,
            'south': 0,
            'north': 0,
            'bottom': 0,
            'top': 0
        }
        plane_beam_weighted_actual_volumes = {
            'west': 0,
            'east': 0,
            'south': 0,
            'north': 0,
            'bottom': 0,
            'top': 0
        }

        for i in range(3):
            # Get the points that intersect with the plane
            for j in range(2):
                if j == 0:
                    face = voxel_faces[['west', 'south', 'bottom'][i]]
                else:
                    face = voxel_faces[['east', 'north', 'top'][i]]
                plane_mask = np.isclose(entry_coords[:, i % 3], face, atol=epsilon)
                hits = np.sum(plane_mask)
                if np.sum(plane_mask) > 0:
                    # Get the coordinates of the points on the plane
                    plane_points = entry_coords[plane_mask]
                    
                    # Calculate the bin indices for each point
                    # Use the two axes orthogonal to i for binning
                    axes = [0, 1, 2]
                    axes.remove(i)
                    bin_indices_x = ((plane_points[:, axes[0]] - voxel_min[axes[0]]) / heat_map_resolution).astype(int)
                    bin_indices_y = ((plane_points[:, axes[1]] - voxel_min[axes[1]]) / heat_map_resolution).astype(int)
                    # Update the heatmap
                    if j == 0:
                        face_heatmaps[['west', 'south', 'bottom'][i]] += np.histogram2d(bin_indices_x, bin_indices_y, bins=bins_per_face)[0]
                        weighted_face_heatmaps[['west', 'south', 'bottom'][i]] += np.histogram2d(bin_indices_x, bin_indices_y, bins=bins_per_face, weights=weighted_theoretical_beam_volumes[plane_mask])[0]
                        plane_beam_theoretical_volumes[['west', 'south', 'bottom'][i]] = np.sum(theoretical_beam_volumes[plane_mask])
                        plane_beam_actual_volumes[['west', 'south', 'bottom'][i]] = np.sum(actual_beam_volumes[plane_mask])
                        plane_beam_weighted_theoretical_volumes[['west', 'south', 'bottom'][i]] = np.sum(weighted_theoretical_beam_volumes[plane_mask])
                        plane_beam_weighted_actual_volumes[['west', 'south', 'bottom'][i]] = np.sum(weighted_actual_beam_volumes[plane_mask])
                    else:
                        face_heatmaps[['east', 'north', 'top'][i]] += np.histogram2d(bin_indices_x, bin_indices_y, bins=bins_per_face)[0]
                        weighted_face_heatmaps[['east', 'north', 'top'][i]] += np.histogram2d(bin_indices_x, bin_indices_y, bins=bins_per_face, weights=weighted_theoretical_beam_volumes[plane_mask])[0]
                        plane_beam_theoretical_volumes[['east', 'north', 'top'][i]] = np.sum(theoretical_beam_volumes[plane_mask])
                        plane_beam_actual_volumes[['east', 'north', 'top'][i]] = np.sum(actual_beam_volumes[plane_mask])
                        plane_beam_weighted_theoretical_volumes[['east', 'north', 'top'][i]] = np.sum(weighted_theoretical_beam_volumes[plane_mask])
                        plane_beam_weighted_actual_volumes[['east', 'north', 'top'][i]] = np.sum(weighted_actual_beam_volumes[plane_mask])

        # Calculate the volume coverages
        theoretical_volume = np.sum(theoretical_beam_volumes)
        actual_volume = np.sum(actual_beam_volumes)
        volume_coverage = (actual_volume / theoretical_volume)
        # Calculate the weighted volume coverages
        weighted_theoretical_volume = np.sum(weighted_theoretical_beam_volumes)
        weighted_actual_volume = np.sum(weighted_actual_beam_volumes)
        weighted_volume_coverage = (weighted_actual_volume / weighted_theoretical_volume)

        # Calculate the percentage of beam volume coverage for each direction
        # Vectorized calculation of beam volume per plane
        plane_beam_theoretical_volumes = np.array([
            plane_beam_theoretical_volumes['west'],
            plane_beam_theoretical_volumes['east'],
            plane_beam_theoretical_volumes['south'],
            plane_beam_theoretical_volumes['north'],
            plane_beam_theoretical_volumes['bottom'],
            plane_beam_theoretical_volumes['top']
        ])
        plane_beam_actual_volumes = np.array([
            plane_beam_actual_volumes['west'],
            plane_beam_actual_volumes['east'],
            plane_beam_actual_volumes['south'],
            plane_beam_actual_volumes['north'],
            plane_beam_actual_volumes['bottom'],
            plane_beam_actual_volumes['top']
        ])
        plane_beam_weighted_theoretical_volumes = np.array([
            plane_beam_weighted_theoretical_volumes['west'],
            plane_beam_weighted_theoretical_volumes['east'],
            plane_beam_weighted_theoretical_volumes['south'],
            plane_beam_weighted_theoretical_volumes['north'],
            plane_beam_weighted_theoretical_volumes['bottom'],
            plane_beam_weighted_theoretical_volumes['top']
        ])
        plane_beam_weighted_actual_volumes = np.array([
            plane_beam_weighted_actual_volumes['west'],
            plane_beam_weighted_actual_volumes['east'],
            plane_beam_weighted_actual_volumes['south'],
            plane_beam_weighted_actual_volumes['north'],
            plane_beam_weighted_actual_volumes['bottom'],
            plane_beam_weighted_actual_volumes['top']
        ])

        if theoretical_volume > 0:
            theoretical_coverage_per_plane = (plane_beam_theoretical_volumes / theoretical_volume)
        else:
            theoretical_coverage_per_plane = np.zeros(6)

        if actual_volume > 0:
            actual_coverage_per_plane = (plane_beam_actual_volumes / actual_volume)
        else:
            actual_coverage_per_plane = np.zeros(6)

        if weighted_theoretical_volume > 0:
            weighted_theoretical_coverage_per_plane = (plane_beam_weighted_theoretical_volumes / weighted_theoretical_volume)
        else:
            weighted_theoretical_coverage_per_plane = np.zeros(6)

        if weighted_actual_volume > 0:
            weighted_actual_coverage_per_plane = (plane_beam_weighted_actual_volumes / weighted_actual_volume)
        else:
            weighted_actual_coverage_per_plane = np.zeros(6)

        west_vertices = np.array([
            # west face
            [voxel_min[0], voxel_min[1], voxel_min[2]],
            [voxel_min[0], voxel_min[1], voxel_max[2]],
            [voxel_min[0], voxel_max[1], voxel_max[2]],
            [voxel_min[0], voxel_max[1], voxel_min[2]]
        ])
        east_vertices = np.array([
            # east face
            [voxel_max[0], voxel_min[1], voxel_min[2]],
            [voxel_max[0], voxel_min[1], voxel_max[2]],
            [voxel_max[0], voxel_max[1], voxel_max[2]],
            [voxel_max[0], voxel_max[1], voxel_min[2]]
        ])
        south_vertices = np.array([
            # south face
            [voxel_min[0], voxel_min[1], voxel_min[2]],
            [voxel_min[0], voxel_min[1], voxel_max[2]],
            [voxel_max[0], voxel_min[1], voxel_max[2]],
            [voxel_max[0], voxel_min[1], voxel_min[2]]
        ])
        north_vertices = np.array([
            # north face
            [voxel_min[0], voxel_max[1], voxel_min[2]],
            [voxel_min[0], voxel_max[1], voxel_max[2]],
            [voxel_max[0], voxel_max[1], voxel_max[2]],
            [voxel_max[0], voxel_max[1], voxel_min[2]]
        ])
        top_vertices = np.array([
            [voxel_min[0], voxel_min[1], voxel_max[2]],
            [voxel_min[0], voxel_max[1], voxel_max[2]],
            [voxel_max[0], voxel_max[1], voxel_max[2]],
            [voxel_max[0], voxel_min[1], voxel_max[2]]
        ])
        bottom_vertices = np.array([
            [voxel_min[0], voxel_min[1], voxel_min[2]],
            [voxel_min[0], voxel_max[1], voxel_min[2]],
            [voxel_max[0], voxel_max[1], voxel_min[2]],
            [voxel_max[0], voxel_min[1], voxel_min[2]]
        ])

        face_dict = {
            'west': {
                'vertices': west_vertices,
                'heatmap': face_heatmaps['west'],
                'weighted_heatmap': weighted_face_heatmaps['west']
            },
            'east': {
                'vertices': east_vertices,
                'heatmap': face_heatmaps['east'],
                'weighted_heatmap': weighted_face_heatmaps['east']
            },
            'south': {
                'vertices': south_vertices,
                'heatmap': face_heatmaps['south'],
                'weighted_heatmap': weighted_face_heatmaps['south']
            },
            'north': {
                'vertices': north_vertices,
                'heatmap': face_heatmaps['north'],
                'weighted_heatmap': weighted_face_heatmaps['north']
            },
            'top': {
                'vertices': top_vertices,
                'heatmap': face_heatmaps['top'],
                'weighted_heatmap': weighted_face_heatmaps['top']
            },
            'bottom': {
                'vertices': bottom_vertices,
                'heatmap': face_heatmaps['bottom'],
                'weighted_heatmap': weighted_face_heatmaps['bottom']
            }
        }
        
        # Create occlusion metrics dataframe
        voxel_id = voxel_df.name
        voxel_cx = voxel_df['voxel_cx'].values[0]
        voxel_cy = voxel_df['voxel_cy'].values[0]
        voxel_cz = voxel_df['voxel_cz'].values[0]
        data = {
            'voxel_id': voxel_id,
            'voxel_cx': voxel_cx,
            'voxel_cy': voxel_cy,
            'voxel_cz': voxel_cz,
            'theoretical_volume': float(theoretical_volume),
            'actual_volume': float(actual_volume),
            'volume_coverage': float(volume_coverage),
            'weighted_theoretical_volume': float(weighted_theoretical_volume),
            'weighted_actual_volume': float(weighted_actual_volume),
            'weighted_volume_coverage': float(weighted_volume_coverage),
            'theoretical_coverage_west': float(theoretical_coverage_per_plane[0]),
            'theoretical_coverage_east': float(theoretical_coverage_per_plane[1]),
            'theoretical_coverage_south': float(theoretical_coverage_per_plane[2]),
            'theoretical_coverage_north': float(theoretical_coverage_per_plane[3]),
            'theoretical_coverage_bottom': float(theoretical_coverage_per_plane[4]),
            'theoretical_coverage_top': float(theoretical_coverage_per_plane[5]),
            'actual_coverage_west': float(actual_coverage_per_plane[0]),
            'actual_coverage_east': float(actual_coverage_per_plane[1]),
            'actual_coverage_south': float(actual_coverage_per_plane[2]),
            'actual_coverage_north': float(actual_coverage_per_plane[3]),
            'actual_coverage_bottom': float(actual_coverage_per_plane[4]),
            'actual_coverage_top': float(actual_coverage_per_plane[5]),
            'weighted_theoretical_coverage_west': float(weighted_theoretical_coverage_per_plane[0]),
            'weighted_theoretical_coverage_east': float(weighted_theoretical_coverage_per_plane[1]),
            'weighted_theoretical_coverage_south': float(weighted_theoretical_coverage_per_plane[2]),
            'weighted_theoretical_coverage_north': float(weighted_theoretical_coverage_per_plane[3]),
            'weighted_theoretical_coverage_bottom': float(weighted_theoretical_coverage_per_plane[4]),
            'weighted_theoretical_coverage_top': float(weighted_theoretical_coverage_per_plane[5]),
            'weighted_actual_coverage_west': float(weighted_actual_coverage_per_plane[0]),
            'weighted_actual_coverage_east': float(weighted_actual_coverage_per_plane[1]),
            'weighted_actual_coverage_south': float(weighted_actual_coverage_per_plane[2]),
            'weighted_actual_coverage_north': float(weighted_actual_coverage_per_plane[3]),
            'weighted_actual_coverage_bottom': float(weighted_actual_coverage_per_plane[4]),
            'weighted_actual_coverage_top': float(weighted_actual_coverage_per_plane[5]),
            # 'face_dict': [face_dict]
        }
        # Create a dataframe for the occlusion metrics
        occ_df = pd.DataFrame(data, index=[0], columns=voxel_occ_schema.names)
        
        # # save point cloud of t_entry
        # import open3d as o3d
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(entry_coords)
        # pcd.colors = o3d.utility.Vector3dVector(np.ones((entry_coords.shape[0], 3)))
        # o3d.io.write_point_cloud(f"entry_coords_{voxel_df['voxel_id'].values[0]}.ply", pcd)

        return occ_df


    # Group by voxel_id and apply the occlusion function to each voxel
    # meta = pd.DataFrame(columns=voxel_occ_schema.names)

    voxel_grouped = voxel_intersections_df.groupby('voxel_id')
    first_voxel_id = voxel_intersections_df['voxel_id'].values[0]
    ### DEBUG ###
    # voxel_occ_df, voxel_heatmaps = get_occlusion_per_voxel(voxel_grouped.get_group(first_voxel_id))

    voxel_occ_df = voxel_intersections_df.groupby('voxel_id').apply(get_occlusion_per_voxel).reset_index(drop=True)
    # Ensure the datatypes for voxel_occ_df are consistent with the schema
    for col, field in zip(voxel_occ_schema.names, voxel_occ_schema):
        voxel_occ_df[col] = voxel_occ_df[col].astype(field.type.to_pandas_dtype())

    # # Extract the voxel heatmaps in a separate dataframe
    # voxel_heatmaps = voxel_occ_df[['voxel_id', 'face_dict']]
    # voxel_occ_df = voxel_occ_df.drop(columns=['face_dict'])
    # voxel_occ_df = voxel_occ_df.reset_index(drop=True)

    return voxel_occ_df



def convert_parquet_to_csv(parquet_file, output_file):
    """
    Convert a parquet file to a csv file.
    """
    
    import pandas as pd
    import pyarrow as pa

    # Read the parquet file
    df = pd.read_parquet(parquet_file, engine='pyarrow')

    df.to_csv(output_file, index=False)

def add_normals_weights_to_valid_rays(valid_rays_dir, knn=6, debug=False):
    """
    Add normals and weights to the points in the valid rays files.
    """
    import dask.dataframe as dd
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import os
    import glob
    import shutil
    from dask.diagnostics import ProgressBar

    # Read the valid rays files
    files = glob.glob(os.path.join(valid_rays_dir, "*valid_rays.parquet"))
    dfs = []
    for file in files:
        df = dd.read_parquet(file, engine='pyarrow')
        dfs.append(df)

    valid_ray_dir = os.path.dirname(files[0])

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    print(f"Loading {len(dfs)} files...")
    
    # Combine all dataframes into one
    valid_rays_df = dd.concat(dfs, axis=0, ignore_index=True)
    valid_rays_df = valid_rays_df.reset_index(drop=True)
    with ProgressBar():
        valid_rays_df = valid_rays_df.compute()
        valid_rays_df = valid_rays_df.reset_index(drop=True)  # Ensure indices are unique and sequential

        # Select only leaf points (where is_leaf is True and point_x is not NaN)
        leaf_mask = valid_rays_df['is_leaf'].values & ~valid_rays_df['point_x'].isna().values
        leaf_points = valid_rays_df.loc[leaf_mask, ['point_x', 'point_y', 'point_z']].to_numpy()
        leaf_idx = valid_rays_df.index[leaf_mask]

        # Check for duplicate leaf_idx
        if leaf_idx.duplicated().any():
            print("Duplicate indices found in leaf_idx. This may cause assignment errors.")
            raise ValueError("Duplicate indices found in leaf_idx. Please check your data for duplicates.")

        # Calculate normals and weights on all leaf hits
        normals, weights = compute_normals_weights_from_points(points=leaf_points, knn=knn)
        del leaf_points

        with ProgressBar():
            # Add normals and weights to the valid rays dataframe (only for leaf points)
            leaf_df = pd.DataFrame({
                'normal_x': normals[:, 0].astype(np.float64()),
                'normal_y': normals[:, 1].astype(np.float64()),
                'normal_z': normals[:, 2].astype(np.float64()),
                'point_weight': weights.astype(np.float64())
            }, index=leaf_idx)
            valid_rays_df.update(leaf_df)                                  
        # Save updated valid_rays parquet files per leg
        output_files = []

        print("Saving results...")

        grouped_df = valid_rays_df.groupby('leg_id')
        DEBUG_PRINT = False
        def save_group(group):
            nonlocal DEBUG_PRINT
            leg_id = group['leg_id'].iloc[0]
            output_file = os.path.join(valid_ray_dir, f"leg_{leg_id}_valid_rays.parquet")

            if debug and not DEBUG_PRINT:
                print("Debugging enabled:")
                DEBUG_PRINT = True
                print(group[~group['point_x'].isna() & group['is_leaf'] == True].head())

            group.to_parquet(output_file, engine='pyarrow', index=False, schema=valid_rays_schema)
            output_files.append(output_file)
            print(f"Saved {output_file}")

        grouped_df.apply(save_group, include_groups=True)

        print(f"Saved {len(output_files)} valid rays files with normals and weights.")

def add_normals_weights_from_intersection_files(files, knn=6):
    """
    Add normals and weights to the points in the intersection files.
    """
    import dask.dataframe as dd
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import os


    # Read the intersection files
    dfs = []
    for file in files:
        df = dd.read_parquet(file, engine='pyarrow')
        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    print(f"Adding normals and weights to {len(dfs)} files...")
    
    # Combine all dataframes into one
    voxel_intersections_df = dd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.reset_index(drop=True)

    # Filter out leaf hits (that definitely hit something)
    leaf_df = voxel_intersections_df[(voxel_intersections_df['is_leaf'] == True) & (voxel_intersections_df['hit_ray'] == True)]
    leaf_df = leaf_df.compute()
    leaf_points = leaf_df[['point_x', 'point_y', 'point_z']].values

    # Calculate normals and weights on all leaf hits
    normals, weights = compute_normals_weights_from_points(points=leaf_points, knn=knn)
    del leaf_points

    leaf_df["normal_x"] = normals[:, 0]
    leaf_df["normal_y"] = normals[:, 1]
    leaf_df["normal_z"] = normals[:, 2]
    leaf_df["point_weight"] = weights
    del normals, weights

    voxel_intersections_df = voxel_intersections_df.compute()

    voxel_intersections_df = voxel_intersections_df.merge(
        leaf_df[['ray_id', 'point_x', 'point_y', 'point_z', 'normal_x', 'normal_y', 'normal_z', 'point_weight']],
        on=['ray_id', 'point_x', 'point_y', 'point_z'],
        how='left'
    )
    del leaf_df

    return voxel_intersections_df

def fix_incorrect_intersections(valid_rays_dir, num_jobs=-1):
    import os
    import glob
    import pandas as pd
    import numpy as np
    from joblib import Parallel, delayed
    from tqdm import tqdm
    import shutil
    from dask.diagnostics import ProgressBar
    import dask.dataframe as dd

    intersection_files = glob.glob(os.path.join(valid_rays_dir, "*_intersections.parquet"))

    def process_file(file):
        df = pd.read_parquet(file, engine='pyarrow')
        leg_id = df['leg_id'].iloc[0]
        voxel_size = df['voxel_size'].iloc[0]
        valid_rays_file = os.path.join(valid_rays_dir, f'leg_{leg_id}_valid_rays.parquet')
        valid_rays_dd = dd.read_parquet(valid_rays_file, engine='pyarrow')
        valid_rays_dd = valid_rays_dd[['ray_id', 'origin_x', 'origin_y', 'origin_z', 'direction_x', 'direction_y', 'direction_z']]
        valid_rays_dd = valid_rays_dd.compute()

        df = df.merge(valid_rays_dd, on='ray_id', how='left')

        del valid_rays_dd

        voxel = df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values
        if 'origin_x_x' in df.columns:
            origin = df[['origin_x_x', 'origin_y_x', 'origin_z_x']].values
            direction = df[['direction_x_x', 'direction_y_x', 'direction_z_x']].values
        else:
            origin = df[['origin_x', 'origin_y', 'origin_z']].values
            direction = df[['direction_x', 'direction_y', 'direction_z']].values

        # Redo voxel-ray AABB intersection to remove any unwanted rays due to tolerance
        epsilon = 1e-6
        voxel_min = voxel - (voxel_size / 2) - epsilon
        voxel_max = voxel + (voxel_size / 2) + epsilon

        direction = np.where(
            np.abs(direction) <= epsilon,
            epsilon,
            direction
        )
        inv_direction = 1.0 / direction

        # Compute the intersection points
        t_ent_arr = (voxel_min - origin) * inv_direction
        t_ex_arr = (voxel_max - origin) * inv_direction
        t_ent = np.max(np.minimum(t_ent_arr, t_ex_arr), axis=1)
        t_ex = np.min(np.maximum(t_ent_arr, t_ex_arr), axis=1)

        # Mask out invalid rays 
        valid_ray_mask = (t_ent <= t_ex + epsilon) & (t_ex >= -epsilon)
        num_invalid_rays = len(valid_ray_mask) - valid_ray_mask.sum()
        print(f"Number of invalid rays: {num_invalid_rays}")
        df = df[valid_ray_mask]

        del t_ent, t_ex, direction

        origin = origin[valid_ray_mask]
        voxel = voxel[valid_ray_mask]
        entry = df[['t_entry_x', 't_entry_y', 't_entry_z']].values
        exit = df[['t_exit_x', 't_exit_y', 't_exit_z']].values
        point = df[['point_x', 'point_y', 'point_z']].values

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

        for col in [
            'origin_x', 'origin_y', 'origin_z', 
            'direction_x', 'direction_y', 'direction_z',
            'origin_x_x', 'origin_y_x', 'origin_z_x',
            'direction_x_x', 'direction_y_x', 'direction_z_x',
            'origin_x_y', 'origin_y_y', 'origin_z_y',
            'direction_x_y', 'direction_y_y', 'direction_z_y'
        ]:
            if col in df.columns:
                df.drop(columns=col, inplace=True)

        # Save backup and overwrite
        old_file = os.path.join(valid_rays_dir, os.path.basename(file).replace(".parquet", "_old.parquet"))
        shutil.copy2(file, old_file)
        output_file = os.path.join(valid_rays_dir, os.path.basename(file))
        df.to_parquet(output_file, engine='pyarrow')

    results = Parallel(n_jobs=num_jobs, prefer="threads")(
        delayed(process_file)(file) for file in intersection_files
    )
    for _ in tqdm(results, desc="Processing intersection files"):
        pass