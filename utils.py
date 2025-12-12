from __future__ import annotations

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
from dask.distributed import progress, get_client, get_worker
import os
import tempfile
import uuid
import shutil
from scipy.sparse.csgraph import connected_components
import time
import dask.dataframe as dd
from dask import delayed


# Dask test modules

import os
import glob
import gc
import time
import math
import tempfile
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

import psutil
import dask.dataframe as dd
from dask.distributed import as_completed, wait

from numba import typed, njit, prange, set_num_threads, get_num_threads
from collections.abc import Iterable
from collections import defaultdict


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

def _start_dask_client(memory_limit='4GB',
                       n_workers=None,
                       threads_per_worker=1,
                       memory_target_fraction=0.7,
                       memory_spill_fraction=0.85,
                       memory_pause_fraction=0.9,
                       memory_terminate=False,
                       temp_dir=None,
                       task_retries=3, 
                       worker_ttl="300s",
                       processes=True):
    """
    Start (or restart) a Dask LocalCluster with memory & temp directory controls.

    Parameters
    ----------
    memory_limit : str|int
        Per-worker memory limit. Accepts int (bytes) or 'XGB'/'XMB'.
    n_workers : int|None
        Number of workers. Defaults to SLURM_CPUS_PER_TASK or physical cores.
    threads_per_worker : int
        Threads per worker.
    memory_target_fraction : float
        Fraction of worker memory to start spilling.
    memory_spill_fraction : float
        Fraction of memory after spilling starts.
    memory_pause_fraction : float
        Fraction at which workers pause.
    memory_terminate : bool
        Whether to terminate workers that exceed terminate fraction.
    temp_dir : str|None
        Directory for dask worker local scratch (local_directory).
    task_retries : int|3
        How many retries to do
    worker_ttl : int|300
        Timeout
    


    Returns
    -------
    Client
    """
    global DASK_CLIENT
    from dask.distributed import Client, LocalCluster, get_client
    import dask
    import psutil
    import os

    # Close any existing client
    try:
        running = get_client()
        if running is not None and running.status != 'closed':
            _close_dask_client(running)
    except Exception:
        pass

    if n_workers is None:
        n_workers = int(os.environ.get('SLURM_CPUS_PER_TASK',
                                       psutil.cpu_count(logical=False)))

    # Normalize memory_limit
    if isinstance(memory_limit, str):
        mem = memory_limit.upper().strip()
        try:
            if mem.endswith('GB'):
                memory_limit = int(float(mem[:-2]) * 1024**3)
            elif mem.endswith('MB'):
                memory_limit = int(float(mem[:-2]) * 1024**2)
            else:
                memory_limit = int(float(mem))
        except Exception:
            pass  # leave as original if conversion fails

    # Temp dir fallback
    if temp_dir is None:
        tmp_env = os.environ.get("TMPDIR")
        if tmp_env and os.path.isdir(tmp_env):
            temp_dir = tmp_env
        else:
            import tempfile
            temp_dir = tempfile.gettempdir()

    # Set memory policies
    dask.config.set({
        "distributed.worker.memory.target": memory_target_fraction,
        "distributed.worker.memory.spill": memory_spill_fraction,
        "distributed.worker.memory.pause": memory_pause_fraction,
        "distributed.worker.memory.terminate": memory_pause_fraction if memory_terminate else False,
        "distributed.scheduler.default-task-retries": task_retries,
        "distributed.scheduler.worker-ttl": worker_ttl
    })

    cluster = LocalCluster(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit,
        local_directory=temp_dir,
        processes=processes
    )
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
def traverse_voxels_broadcasted(voxel_references, ray_partition, voxels_per_chunk, temp_dir, debug=False, epsilon=1e-6):
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
    import psutil
    import dask.dataframe as dd
    from dask.distributed import as_completed
    
    from dask.diagnostics import ProgressBar
    import tempfile
    import time

    print("[voxel_ray_intersections] Initialising Dask client...")
    if os.environ.get('SLURM_CPUS_PER_TASK') is not None:
        print(f"Detected SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK')}")
        avail_cpu = int(os.environ.get('SLURM_CPUS_PER_TASK'))
        threads_per_worker = 2 # hard code this for your system
        mem_threshold = 1.0
    else:
        avail_cpu = psutil.cpu_count(logical=False)
        threads_per_worker = psutil.cpu_count(logical=True) // avail_cpu
        mem_threshold = 0.8
        print(f"No SLURM_CPUS_PER_TASK detected, using system CPU count: {avail_cpu} physical cores with {threads_per_worker} threads per worker.")

    # avail_cpu = int(os.environ.get('SLURM_CPUS_PER_TASK', psutil.cpu_count(logical=True)))
    avail_mem = int(float(os.environ.get('SLURM_MEM_PER_NODE', psutil.virtual_memory().available // (1024 * 1024))) * mem_threshold) # in MB
    
    # Use Dask LocalCluster for memory management and spill configuration
    if temp_dir is None:
        # Prefer HPC scratch (TMPDIR) if it exists
        hpc_tmp = os.environ.get("TMPDIR")
        if hpc_tmp and os.path.isdir(hpc_tmp):
            temp_dir = hpc_tmp
            print(f"Using HPC temporary directory: {temp_dir}")
        else:
            # Fall back to OS default temp dir
            os_tmp = tempfile.gettempdir()
            if os_tmp and os.path.isdir(os_tmp):
                temp_dir = os_tmp
                print(f"Using OS temporary directory: {temp_dir}")
            else:
                # Final fallback
                temp_dir = "/tmp"
                print("Using fallback temporary directory: /tmp")

    total_threads = avail_cpu * threads_per_worker
    optimal_threads_per_worker = 8
    optimal_workers = max(1, total_threads // optimal_threads_per_worker)

    memory_worker = avail_mem / optimal_workers
    avail_mem_string_for_dask = f"{int(memory_worker)}MB"
    print(f"[voxel_ray_intersections] Starting Dask with memory_limit={avail_mem_string_for_dask}")

    client = _start_dask_client(
        memory_limit=avail_mem_string_for_dask,
        n_workers=optimal_workers,
        threads_per_worker=optimal_threads_per_worker,
        memory_target_fraction=0.6,
        memory_spill_fraction=0.75,
        memory_pause_fraction=0.95,
        memory_terminate=False,
        temp_dir=temp_dir,
        task_retries=3,
        worker_ttl="300s",
        processes=True
    )

    # Compile the references files to establish a voxel dataframe of size and voxel_id
    voxel_references = glob.glob(os.path.join(references_dir, '*.csv'))
    print(f"Found {len(voxel_references)} voxel reference files.")

    dfs = []
    for voxel_ref in voxel_references:
        df = pd.read_csv(voxel_ref, index_col=None, header=0)
        if 'voxel_id' not in df.columns:
            df['voxel_id'] = df.apply(
                lambda row: create_voxel_id(
                    voxel_size=row['voxel_size'] if 'voxel_size' in row else float(os.path.splitext(voxel_ref)[0].split("_")[-1]),
                    x=row['voxel_cx'],
                    y=row['voxel_cy'],
                    z=row['voxel_cz']
                ),
                axis=1
            )
        df = df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']].drop_duplicates()
        voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
        df['voxel_size'] = voxel_size
        dfs.append(df)
    voxel_references = pd.concat(dfs)
    print(f"Compiled voxel references with {voxel_references.shape[0]} entries.")

    valid_rays_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))
    print(f"Found {len(valid_rays_files)} valid rays files.")

    def map_ray_partition_to_function(ray_partition, voxel_group, temp_dir):
        # print(f"[map_ray_partition_to_function] Partition rows={len(ray_partition)} | voxels={len(voxel_group)}")
        return traverse_voxels(ray_partition=ray_partition, voxel_references=voxel_group, memory_limit_bytes=((memory_worker * 1024**2) // optimal_threads_per_worker), debug=debug)

    voxel_ray_intersections = {}

    for file in valid_rays_files:
        leg_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])
        print(f"[voxel_ray_intersections] Loading valid rays file for leg {leg_id}: {file}")
        df = dd.read_parquet(file, engine='pyarrow', blocksize="25MB")
        print(f"[voxel_ray_intersections] Leg {leg_id} partitions: {df.npartitions}")
        meta = pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        # # estimate memory per partition
        # num_rays = df.map_partitions(len).compute().max()
        # num_voxels = len(voxel_references)
        # estimated_memory_per_partition = estimate_broadcast_memory(num_rays=num_rays, num_voxels=num_voxels)
        result = df.map_partitions(
            map_ray_partition_to_function,
            voxel_group=voxel_references,
            temp_dir=temp_dir,
            meta=meta
        )


        # chunk_results = []
        # for start in range(0, voxel_references.shape[0], voxel_chunk_size):
        #     vchunk = voxel_references.iloc[start:start + voxel_chunk_size]
        #     r = df.map_partitions(
        #         map_ray_partition_to_function,
        #         voxel_group=vchunk,
        #         temp_dir=temp_dir,
        #         meta=meta
        #     )
        #     chunk_results.append(r)
    
        # result = dd.concat(chunk_results, axis=0, interleave_partitions=True)
        voxel_ray_intersections[leg_id] = result
        print(f"[voxel_ray_intersections] Mapped partitions for leg {leg_id}")# with memory: {(estimated_memory_per_partition / (1024**2)):.2f} MB each.")

    def save_task(df, leg_id):
        if df.empty:
            print(f"No data to save for leg_id: {leg_id}.")
            return False
        voxel_size = round(float(df['voxel_size'].iloc[0]), 2)
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
        print(f"Saved intersections for leg_id: {leg_id} to {output_filename}.")
        return True

    print("[voxel_ray_intersections] Submitting Dask compute jobs...")
    # futures = []
    start_time = time.time()
    # for leg_id, results in voxel_ray_intersections.items():
    #     future = client.compute(results)
    #     futures.append((leg_id, future))

        # out_dir = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_intersections")
        # if os.path.exists(out_dir):
        #     if os.path.isfile(out_dir):
        #         os.remove(out_dir)
        #     else:
        #         shutil.rmtree(out_dir)
        
        # # Compute and save using Dask's to_parquet before computing
        # results.to_parquet(
        #     out_dir,
        #     engine='pyarrow',
        #     compression='snappy',
        #     schema=voxel_ray_intersection_schema,
        #     partition_on=['voxel_size']
        # )

        # Submit all
    futures_dict = {}
    start_time = time.time()
    futures = []
    for leg_id, results in voxel_ray_intersections.items():
        future = client.compute(results)
        futures_dict[future] = leg_id
        futures.append((leg_id, future))

    # Process as they complete
    for future in as_completed(futures_dict):
        leg_id = futures_dict[future]
        results = future.result()
        # ... process and save
        grouped = results.groupby('voxel_size', group_keys=True)
        print(f"[voxel_ray_intersections] Leg {leg_id} grouped into {len(grouped)} voxel_size groups.")
        for voxel_size, group_df in grouped:
            print(f"[voxel_ray_intersections] Saving group voxel_size={voxel_size} (rows={len(group_df)}) for leg {leg_id}")
            save_task(group_df, leg_id)
            del group_df
        del results
        print(f"[voxel_ray_intersections] Completed save for leg {leg_id}")
    # for leg_id, future in futures:
    #     with ProgressBar():
    #         results = future.result()
    #     for voxel_size, group_df in results.groupby('voxel_size', group_keys=True):
    #         save_task(group_df, leg_id)
    #         del group_df
    #     del results

    # start_time = time.time()
    # print("[voxel_ray_intersections] Awaiting computation results...")
    # for leg_id, future in futures:
    #     print(f"[voxel_ray_intersections] Waiting on leg {leg_id} future...")
    #     with ProgressBar():
    #         results = future.result()
    #     print(f"[voxel_ray_intersections] Result received for leg {leg_id} (rows={len(results)})")
    #     grouped = results.groupby('voxel_size', group_keys=True)
    #     print(f"[voxel_ray_intersections] Leg {leg_id} grouped into {len(grouped)} voxel_size groups.")
    #     for voxel_size, group_df in grouped:
    #         print(f"[voxel_ray_intersections] Saving group voxel_size={voxel_size} (rows={len(group_df)}) for leg {leg_id}")
    #         save_task(group_df, leg_id)
    #         del group_df
    #     del results
    #     print(f"[voxel_ray_intersections] Completed save for leg {leg_id}")

    end_time = time.time()
    print(f"Voxel ray intersection processing complete in {end_time - start_time:.2f} seconds.")
    time.sleep(1)
    _close_dask_client(client)
    print("[voxel_ray_intersections] Dask client closed.")

def traverse_voxels(voxel_references, ray_partition, memory_limit_bytes, min_chunk_size=1, max_chunk_size=1000, debug=False, epsilon=1e-6):
    import logging
    from scipy.sparse import lil_matrix
    logging.basicConfig(level=logging.INFO)

    if ray_partition.empty:
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

    # Prepare requisite ray and voxel data
    ray_ids = ray_partition['ray_id'].values
    origins = np.asarray(ray_partition[['origin_x', 'origin_y', 'origin_z']].values)
    directions = np.asarray(ray_partition[['direction_x', 'direction_y', 'direction_z']].values)

    voxel_ids = voxel_references['voxel_id'].values
    voxel_sizes = voxel_references['voxel_size'].values
    voxel_centres = voxel_references[['voxel_cx', 'voxel_cy', 'voxel_cz']].values
    unique_sizes, size_group_ids = np.unique(voxel_sizes, return_inverse=True)

    # Find unique rays and create mapping back to original indices
    _, unique_ray_idx, inverse_indices = np.unique(ray_ids, return_index=True, return_inverse=True)
    
    # Extract unique ray data
    unique_origins = origins[unique_ray_idx]
    unique_directions = directions[unique_ray_idx]
    U = unique_origins.shape[0]
    # del unique_ray_idx, unique_ray_mask, unique_ray_indices
    gc.collect()

    ### Traversal 1: Rough cull to reduce to potential voxel-ray intersections using ray-sphere intersection ###

    # Calculate broadcast memory for ray-sphere intersection
    broadcast_memory = 3 * U * 3 * 8    # Broadcast arrays: 3 arrays of (V_chunk, U, 3) × 8 bytes (float64)
    intermediate_memory = 4 * U * 8     # Intermediate float arrays: oc, b, c, discriminant (conservative estimate)
    hit_mask_memory = U * 1             # Hit mask: (V_chunk, U) × 1 byte (bool)
    # Calculate broadcast memory for box intersection
    box_intersection_memory = (2 * 3 * 8) + (2 * U * 0.5 * 3 * 8)   # Box intersection (when hits exist, worst case all hit):
    mask_memory = 2 * U * 1      # Unique mask and final mask: (V_chunk, U) × 1 byte (bool)
    buffer = 1.25  # Safety buffer
    
    # Total memory per voxel in chunk
    memory_per_voxel = (
        broadcast_memory + 
        intermediate_memory + 
        hit_mask_memory + 
        box_intersection_memory + 
        mask_memory
    ) * buffer
    optimal_chunk_size = int(memory_limit_bytes / (memory_per_voxel))
    optimal_chunk_size = max(min_chunk_size, min(optimal_chunk_size, max_chunk_size))

    print(f"[traverse_voxels] Memory diagnostics:")
    print(f"  - Number of unique rays (U): {U}")
    # print(f"  - Number of voxels: {len(voxel_centres)}")
    # print(f"  - Memory limit (bytes): {memory_limit_bytes}")
    # print(f"  - Broadcast memory per voxel: {broadcast_memory} bytes")
    # print(f"  - Intermediate memory per voxel: {intermediate_memory} bytes")
    # print(f"  - Hit mask memory per voxel: {hit_mask_memory} bytes")
    # print(f"  - Box intersection memory per voxel: {box_intersection_memory} bytes")
    # print(f"  - Mask memory per voxel: {mask_memory} bytes")
    # print(f"  - Total memory per voxel: {memory_per_voxel} bytes ({memory_per_voxel / (1024**2):.2f} MB)")
    print(f"  - Optimal chunk size: {optimal_chunk_size} voxels")
    # print(f"  - Min chunk size: {min_chunk_size}, Max chunk size: {max_chunk_size}")

    # Iterate through voxel sizes and subsequently chunks
    hit_masks = []
    for s_idx, s in enumerate(unique_sizes):
        # Indices of voxels with size s
        group_mask = (size_group_ids == s_idx)
        group_centres = voxel_centres[group_mask]
        voxel_radius_sq = (s * np.sqrt(3) * 0.5 + 0.05) ** 2  # Precompute squared radius for sphere intersection

        for start in range(0, len(group_centres), optimal_chunk_size):
            vc_chunk = group_centres[start:start+optimal_chunk_size]

            # Vectorized ray-voxel intersection (only unique rays)
            voxel_centres_b = vc_chunk[:, np.newaxis, :]
            origins_b = unique_origins[np.newaxis, :, :]
            directions_b = unique_directions[np.newaxis, :, :]

            # Use half the diagonal of the voxel as the radius, plus epsilon
            oc = origins_b - voxel_centres_b
            b = 2.0 * np.sum(oc * directions_b, axis=2)
            c = np.sum(oc * oc, axis=2) - voxel_radius_sq
            discriminant = b**2 - 4 * c
            potential_hit = discriminant >= -epsilon
            
            del oc, b, c, discriminant
            gc.collect()

            hit_pairs = []
            if np.any(potential_hit):
                size_half = s / 2.0
                voxel_mins = voxel_centres_b - (size_half - epsilon)
                voxel_maxs = voxel_centres_b + (size_half + epsilon)
                potential_voxel_idx, potential_unique_ray_idx = np.nonzero(potential_hit)
                potential_voxel_mins = voxel_mins[potential_voxel_idx, 0]
                potential_voxel_maxs = voxel_maxs[potential_voxel_idx, 0]
                potential_origins = unique_origins[potential_unique_ray_idx]
                potential_directions = unique_directions[potential_unique_ray_idx]
                
                del voxel_mins, voxel_maxs, potential_hit
                gc.collect()
                
                small_epsilon = 1e-9
                small_dir = np.abs(potential_directions) <= small_epsilon
                potential_directions = np.where(
                    small_dir,
                    np.where(potential_directions == 0, small_epsilon, np.sign(potential_directions) * small_epsilon),
                    potential_directions
                )
                inv_potential_directions = 1.0 / potential_directions
                
                del small_dir
                gc.collect()
                
                t1 = (potential_voxel_mins - potential_origins) * inv_potential_directions
                t2 = (potential_voxel_maxs - potential_origins) * inv_potential_directions
                
                del potential_voxel_mins, potential_voxel_maxs, potential_origins, inv_potential_directions
                gc.collect()
                
                t_enter = np.max(np.minimum(t1, t2), axis=1)
                t_exit = np.min(np.maximum(t1, t2), axis=1)
                
                del t1, t2
                gc.collect()
                
                valid = (t_enter <= t_exit + epsilon) & (t_exit >= -epsilon)
                hit_pairs.extend(zip(potential_voxel_idx[valid], potential_unique_ray_idx[valid]))
                
                del t_enter, t_exit, valid, potential_voxel_idx, potential_unique_ray_idx
                gc.collect()
            
            del voxel_centres_b, origins_b, directions_b
            gc.collect()
        
            hit_masks.append((group_mask, start, len(vc_chunk), hit_pairs))
    
    # Concatenate all hit masks
    # map back to original ray indices
    order = np.argsort(inverse_indices, kind='stable')
    inv_sorted = inverse_indices[order]
    split = np.flatnonzero(np.diff(inv_sorted)) + 1
    unique_ray_idx_to_original = np.split(order, split)

    assert len(unique_ray_idx_to_original) == unique_origins.shape[0] == unique_directions.shape[0]

    all_ray_idxs = []
    all_voxel_idxs = []

    for group_mask, start, length, hit_pairs in hit_masks:
        if len(hit_pairs) == 0:
            continue
        group_positions = np.flatnonzero(group_mask)
        chunk_positions = group_positions[start:start+length]
        for v_idx, ru_idx in hit_pairs:
            orig_ray_idxs = unique_ray_idx_to_original[ru_idx]
            all_voxel_idxs.extend(np.repeat(chunk_positions[v_idx], orig_ray_idxs.size))
            all_ray_idxs.extend(orig_ray_idxs.tolist())
    
    assert len(all_voxel_idxs) == len(all_ray_idxs), \
        f"Length mismatch: voxels {len(all_voxel_idxs)} vs rays {len(all_ray_idxs)}"
    
    del hit_masks, group_mask, start, length, hit_pairs
    gc.collect()
    
    if len(all_voxel_idxs) == 0:
        del all_voxel_idxs, all_ray_idxs, voxel_ids, voxel_sizes, voxel_centres
        del ray_ids, origins, directions, ray_partition, unique_origins, unique_directions
        del inverse_indices
        gc.collect()
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)
    
    # Filter loaded data based on mask
    filtered_voxel_ids = voxel_ids[all_voxel_idxs]
    filtered_voxel_sizes = voxel_sizes[all_voxel_idxs]
    filtered_voxel_centres = voxel_centres[all_voxel_idxs]

    del voxel_ids, voxel_sizes, voxel_centres, all_voxel_idxs
    gc.collect()

    filtered_ray_ids = ray_ids[all_ray_idxs]
    filtered_origins = origins[all_ray_idxs]
    filtered_directions = directions[all_ray_idxs]

    del ray_ids, origins, directions
    gc.collect()

    # Load and filter remaining data one by one to save memory
    filtered_leg_ids = np.asarray(ray_partition['leg_id'].values)[all_ray_idxs]
    filtered_points = np.asarray(ray_partition[['point_x', 'point_y', 'point_z']].values)[all_ray_idxs]
    filtered_normals = np.asarray(ray_partition[['normal_x', 'normal_y', 'normal_z']].values)[all_ray_idxs]
    filtered_point_weights = np.asarray(ray_partition['point_weight'].values)[all_ray_idxs]
    filtered_echo_intensities = np.asarray(ray_partition['echo_intensity'].values)[all_ray_idxs]
    filtered_is_leaf = np.asarray(ray_partition['is_leaf'].values)[all_ray_idxs]
    filtered_return_numbers = np.asarray(ray_partition['return_number'].values)[all_ray_idxs]
    filtered_number_of_returns = np.asarray(ray_partition['number_of_returns'].values)[all_ray_idxs]

    del all_ray_idxs, ray_partition
    gc.collect()
    
    # Calculate viewing angles
    filtered_viewing_angles = find_viewing_angles(directions=filtered_directions)
    
    # Calculate entry/exit coordinates
    filtered_voxel_mins = filtered_voxel_centres - (filtered_voxel_sizes[:, np.newaxis] / 2 - epsilon)
    filtered_voxel_maxs = filtered_voxel_centres + (filtered_voxel_sizes[:, np.newaxis] / 2 + epsilon)
    
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
    
    del t_min, t_max, inv_filtered_directions
    gc.collect()
    
    filtered_exit_coords = filtered_origins + t_exit[:, np.newaxis] * filtered_directions
    filtered_entry_coords = filtered_origins + t_enter[:, np.newaxis] * filtered_directions
    
    del t_enter, t_exit
    gc.collect()

    # Classify hit types
    unbound = np.isnan(filtered_points).any(axis=1)
    in_voxel = np.all((filtered_points >= (filtered_voxel_mins - epsilon)) & (filtered_points <= (filtered_voxel_maxs + epsilon)), axis=1)
    dist_to_entry_sq = np.sum((filtered_origins - (filtered_entry_coords)) ** 2, axis=1)
    dist_to_exit_sq = np.sum((filtered_origins - (filtered_exit_coords)) ** 2, axis=1)
    dist_to_point_sq = np.sum((filtered_points - filtered_origins) ** 2, axis=1)
    before_voxel = (dist_to_entry_sq > dist_to_point_sq) & ~in_voxel & ~unbound
    after_voxel = (dist_to_exit_sq < dist_to_point_sq) & ~in_voxel & ~unbound
    
    del dist_to_entry_sq, dist_to_exit_sq, dist_to_point_sq
    gc.collect()

    hit_type = np.full(filtered_points.shape[0], -1, dtype=np.int32)
    hit_type[unbound] = 0
    hit_type[before_voxel] = 1
    hit_type[in_voxel] = 2
    hit_type[after_voxel] = 3
    
    del unbound, in_voxel, before_voxel, after_voxel
    gc.collect()

    # Calculate distance to voxel centre
    filtered_distances_to_voxel_centre = np.linalg.norm(filtered_origins - filtered_voxel_centres, axis=1)
    
    del filtered_voxel_mins, filtered_voxel_maxs, filtered_origins, filtered_directions
    gc.collect()

    # Build output dataframe
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
    
    del filtered_voxel_sizes, filtered_voxel_ids, filtered_voxel_centres
    del filtered_leg_ids, filtered_ray_ids, filtered_entry_coords, filtered_exit_coords
    del filtered_distances_to_voxel_centre, filtered_points, filtered_echo_intensities
    del filtered_return_numbers, filtered_number_of_returns, filtered_normals
    del filtered_point_weights, filtered_viewing_angles, hit_type, filtered_is_leaf
    gc.collect()
    
    data_df = pd.DataFrame(data_dict)
    del data_dict
    gc.collect()
    
    return data_df




# Function used for calculating voxel-ray intersections but using dask optimised code to ensure memory efficiency in highly paralleised computations
def voxel_ray_intersections_dask_initial_numba_test(valid_rays_dir, references_dir, temp_dir=None, cpus=None, mem=None, debug=True, epsilon=1e-6):
    # --------------------------
    # Utilities
    # --------------------------

    @njit(parallel=True, fastmath=True, cache=True)
    def process_ray_voxel_pairs_kernel(
        ray_cell_voxel_pairs_arr,
        origins_arr, directions_arr, points_arr, normals_arr,
        echo_intensity_arr, return_number_arr, number_of_returns_arr,
        point_weight_arr, is_leaf_arr, leg_ids_arr, ray_ids_arr,
        v_ids_arr, v_sizes_arr, v_centres_arr, vmins_arr, vmaxs_arr,
        epsilon_val
    ):
        print(f"num_threads = {get_num_threads()}")

        n = ray_cell_voxel_pairs_arr.shape[0]
        out_rows = []
        
        for pair_idx in prange(n):
            ray_idx = int(ray_cell_voxel_pairs_arr[pair_idx]['ray_idx'])
            voxel_idx = int(ray_cell_voxel_pairs_arr[pair_idx]['voxel_idx'])
            
            o = origins_arr[ray_idx]
            d = directions_arr[ray_idx]
            vmin = vmins_arr[voxel_idx]
            vmax = vmaxs_arr[voxel_idx]
            p = points_arr[ray_idx]
            
            # Safe direction to avoid division by zero
            safe_d = np.empty(3, dtype=np.float64)
            for axis in range(3):
                if np.abs(d[axis]) <= epsilon_val:
                    safe_d[axis] = epsilon_val if d[axis] == 0 else np.sign(d[axis]) * epsilon_val
                else:
                    safe_d[axis] = d[axis]
            
            # AABB intersection
            inv_d = 1.0 / safe_d
            t1 = (vmin - o) * inv_d
            t2 = (vmax - o) * inv_d
            
            t_enter = -np.inf
            t_exit = np.inf
            for axis in range(3):
                t_min_axis = min(t1[axis], t2[axis])
                t_max_axis = max(t1[axis], t2[axis])
                t_enter = max(t_enter, t_min_axis)
                t_exit = min(t_exit, t_max_axis)
            
            # Check intersection validity
            if not (t_enter <= t_exit + epsilon_val and t_exit >= -epsilon_val):
                continue
            
            # Compute entry and exit points
            entry = o + t_enter * d
            exit_pt = o + t_exit * d
            
            # Distance to voxel centre
            dist_to_centre = np.sqrt((o[0] - v_centres_arr[voxel_idx, 0])**2 +
                                    (o[1] - v_centres_arr[voxel_idx, 1])**2 +
                                    (o[2] - v_centres_arr[voxel_idx, 2])**2)
            
            # Hit classification
            unbound = (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]))
            in_voxel = (p[0] >= vmin[0] - epsilon_val and p[0] <= vmax[0] + epsilon_val and
                        p[1] >= vmin[1] - epsilon_val and p[1] <= vmax[1] + epsilon_val and
                        p[2] >= vmin[2] - epsilon_val and p[2] <= vmax[2] + epsilon_val)
            
            dist_to_entry_sq = (o[0] - entry[0])**2 + (o[1] - entry[1])**2 + (o[2] - entry[2])**2
            dist_to_exit_sq = (o[0] - exit_pt[0])**2 + (o[1] - exit_pt[1])**2 + (o[2] - exit_pt[2])**2
            dist_to_point_sq = (o[0] - p[0])**2 + (o[1] - p[1])**2 + (o[2] - p[2])**2
            
            before_voxel = (dist_to_entry_sq >= dist_to_point_sq) and (not in_voxel) and (not unbound)
            after_voxel = (dist_to_exit_sq <= dist_to_point_sq) and (not in_voxel) and (not unbound)
            
            if unbound:
                hit_type = 0
            elif before_voxel:
                hit_type = 1
            elif in_voxel:
                hit_type = 2
            elif after_voxel:
                hit_type = 3
            else:
                continue
            
            # Viewing angle calculation
            d_norm = np.sqrt(d[0]**2 + d[1]**2 + d[2]**2)
            if d_norm > 0:
                cos_theta = d[2] / d_norm
                cos_theta = max(-1.0, min(1.0, cos_theta))
                viewing_angle = np.degrees(np.arccos(cos_theta))
                if viewing_angle > 90.0:
                    viewing_angle = 180.0 - viewing_angle
            else:
                viewing_angle = 0.0
            
            # Append output row
            out_rows.append((
                float(v_sizes_arr[voxel_idx]),
                int(v_ids_arr[voxel_idx]),
                float(v_centres_arr[voxel_idx, 0]),
                float(v_centres_arr[voxel_idx, 1]),
                float(v_centres_arr[voxel_idx, 2]),
                int(leg_ids_arr[ray_idx]),
                int(ray_ids_arr[ray_idx]),
                float(entry[0]), float(entry[1]), float(entry[2]),
                float(exit_pt[0]), float(exit_pt[1]), float(exit_pt[2]),
                float(dist_to_centre),
                float(p[0]), float(p[1]), float(p[2]),
                float(echo_intensity_arr[ray_idx]),
                int(return_number_arr[ray_idx]) if not np.isnan(return_number_arr[ray_idx]) else 0,
                int(number_of_returns_arr[ray_idx]) if not np.isnan(number_of_returns_arr[ray_idx]) else 0,
                float(normals_arr[ray_idx, 0]), float(normals_arr[ray_idx, 1]), float(normals_arr[ray_idx, 2]),
                float(point_weight_arr[ray_idx]),
                float(viewing_angle),
                int(hit_type),
                bool(is_leaf_arr[ray_idx]),
            ))
        
        return out_rows

    # Get the correct temp_dir
    def _resolve_temp_dir(tmp_hint: str | None = None) -> str:
        """Resolve a suitable temporary directory, preferring HPC TMPDIR when available."""
        if tmp_hint is not None and os.path.isdir(tmp_hint):
            return tmp_hint

        hpc_tmp = os.environ.get("TMPDIR")
        if hpc_tmp and os.path.isdir(hpc_tmp):
            return hpc_tmp

        os_tmp = tempfile.gettempdir()
        if os_tmp and os.path.isdir(os_tmp):
            return os_tmp

        return "/tmp"

    # Compile voxel reference files
    def _compile_voxel_references(references_dir: str) -> pd.DataFrame:
        """
        Reads *.csv voxel reference files and returns a DataFrame with columns:
        ['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz', 'voxel_size'].
        If 'voxel_id' is missing, it is generated using create_voxel_id(...).
        Size is derived from filename suffix when not in data.
        """
        voxel_files = glob.glob(os.path.join(references_dir, '*.csv'))
        dfs: List[pd.DataFrame] = []

        for voxel_ref in voxel_files:
            df = pd.read_csv(voxel_ref, index_col=None, header=0)

            if 'voxel_id' not in df.columns:
                voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
                df['voxel_id'] = df.apply(
                    lambda row: create_voxel_id(
                        voxel_size=row['voxel_size'] if 'voxel_size' in row else voxel_size,
                        x=row['voxel_cx'],
                        y=row['voxel_cy'],
                        z=row['voxel_cz']
                    ),
                    axis=1
                )

            df = df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']].drop_duplicates()
            voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
            df['voxel_size'] = voxel_size
            dfs.append(df)

        if len(dfs) == 0:
            return pd.DataFrame(columns=['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz', 'voxel_size'])

        combined = pd.concat(dfs, ignore_index=True)
        return combined

    # Calculate avail_cpus, avail_mem, and return optimal worker/thread config
    def _determine_dask_resources(cpus: int | None, mem: int | None, optimal_threads: int = 8, mem_threshold: float = 0.7) -> tuple[str, int, int]:
        """
        Determine available CPUs and memory for Dask configuration.
        Returns (avail_cpus, avail_mem_string_for_dask, optimal_workers, threads_per_worker).
        """
        if cpus is not None:
            avail_threads = cpus
        else:
            if 'SLURM_CPUS_PER_TASK' in os.environ:
                avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
                threads = 2
            else:
                avail_cpus = psutil.cpu_count(logical=False)
                threads = psutil.cpu_count(logical=True) // avail_cpus

        if mem is not None:
            avail_mem = int(mem * mem_threshold)  # in MB
        else:
            avail_mem = int(float(os.environ.get('SLURM_MEM_PER_NODE', psutil.virtual_memory().available // (1024 * 1024))) * mem_threshold)  # in MB

        if optimal_threads > threads:
            n_workers = (avail_cpus * threads) // optimal_threads
            threads_per_worker = optimal_threads
        else:
            n_workers = avail_cpus
            threads_per_worker = threads

        memory_worker = avail_mem / n_workers
        avail_mem_string_for_dask = f"{int(memory_worker)}MB"

        return avail_mem_string_for_dask, n_workers, threads_per_worker

    def _save_group(valid_rays_dir: str, leg_id: int, df: pd.DataFrame) -> bool:
        """Save a single leg/voxel_size group to Parquet with the exact schema."""
        if df is None or len(df) == 0:
            print(f"No data to save for leg_id: {leg_id}.")
            return False

        voxel_size = round(float(df['voxel_size'].iloc[0]), 2)
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
        return True
    
    # --------------------------
    # Spatial index (sparse grid)
    # --------------------------

    class SparseGridIndex:
        """Sparse uniform grid that maps grid cell -> list of voxel indices.

        Grid cell size is chosen as the minimum voxel size, which ensures DDA steps
        are fine-grained enough to find candidates without broadcasting.
        """

        def __init__(self, cell_size: float, origin: np.ndarray, bbox_min: np.ndarray, bbox_max: np.ndarray):
            self.cell_size = float(cell_size)
            self.origin = np.asarray(origin, dtype=np.float64)
            self.bbox_min = np.asarray(bbox_min, dtype=np.float64)
            self.bbox_max = np.asarray(bbox_max, dtype=np.float64)
            self.map: Dict[Tuple[int, int, int], np.ndarray] = {}

        def _cell_index(self, p: np.ndarray) -> Tuple[int, int, int]:
            """Compute integer cell coordinates for a point p."""
            rel = (p - self.origin) / self.cell_size
            return (int(math.floor(rel[0])), int(math.floor(rel[1])), int(math.floor(rel[2])))

        def insert_voxel_aabb(self, vmin: np.ndarray, vmax: np.ndarray, voxel_idx: int):
            """Insert a voxel AABB into all overlapping grid cells."""
            start = self._cell_index(vmin)
            end = self._cell_index(vmax)
            # Iterate integer cell bounds
            for ix in range(start[0], end[0] + 1):
                for iy in range(start[1], end[1] + 1):
                    for iz in range(start[2], end[2] + 1):
                        key = (ix, iy, iz)
                        arr = self.map.get(key)
                        if arr is None:
                            self.map[key] = np.array([voxel_idx], dtype=np.int64)
                        else:
                            # append efficiently
                            self.map[key] = np.concatenate((arr, np.array([voxel_idx], dtype=np.int64)))

    def _build_sparse_grid(voxel_refs: pd.DataFrame, epsilon: float = 1e-6) -> Tuple[SparseGridIndex, Dict[str, np.ndarray]]:
        """
        Build a sparse grid and return (grid_index, voxel_data_dict) where voxel_data_dict contains
        arrays needed during traversal.
        """
        voxel_ids = voxel_refs['voxel_id'].to_numpy()
        voxel_sizes = voxel_refs['voxel_size'].to_numpy(dtype=np.float64)
        centres = voxel_refs[['voxel_cx','voxel_cy','voxel_cz']].to_numpy(dtype=np.float64)

        # Global bbox from voxel AABBs
        half = (voxel_sizes[:, None] / 2.0)
        vmins = centres - (half - epsilon)
        vmaxs = centres + (half + epsilon)
        bbox_min = np.min(vmins, axis=0)
        bbox_max = np.max(vmaxs, axis=0)

        # Choose cell_size as the minimum voxel size (finest grid)
        cell_size = max(1e-9, float(np.min(voxel_sizes)))
        origin = bbox_min.copy()
        grid = SparseGridIndex(cell_size=cell_size, origin=origin, bbox_min=bbox_min, bbox_max=bbox_max)

        for i in range(centres.shape[0]):
            grid.insert_voxel_aabb(vmins[i], vmaxs[i], i)

        voxel_data = {
            'ids': voxel_ids,
            'sizes': voxel_sizes,
            'centres': centres,
            'vmins': vmins,
            'vmaxs': vmaxs,
            'bbox_min': bbox_min,
            'bbox_max': bbox_max,
            'cell_size': np.array([cell_size], dtype=np.float64),
            'origin': origin,
        }
        return grid, voxel_data


    def _slab_intersections_batch(origins: np.ndarray, directions: np.ndarray,
                                vmins: np.ndarray, vmaxs: np.ndarray,
                                epsilon: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Vectorized slab intersection for a batch of candidate voxels vs a single ray.
        origins: (3,), directions: (3,), vmins/vmaxs: (M,3)
        Returns mask (M,), entry points (M,3), exit points (M,3)
        """
        # Make direction safe
        safe_dir = np.where(np.abs(directions) <= epsilon,
                            np.where(directions == 0, epsilon, np.sign(directions) * epsilon),
                            directions)
        inv_dir = 1.0 / safe_dir
        t1 = (vmins - origins) * inv_dir
        t2 = (vmaxs - origins) * inv_dir
        t_enter = np.max(np.minimum(t1, t2), axis=1)
        t_exit = np.min(np.maximum(t1, t2), axis=1)
        valid = (t_enter <= t_exit + epsilon) & (t_exit >= -epsilon)
        entry = origins + t_enter[:, None] * directions
        exitp = origins + t_exit[:, None] * directions
        return valid, entry, exitp


    def _dda_cells_for_rays_vectorized(grid: SparseGridIndex, origins: np.ndarray, directions: np.ndarray,
                                       epsilon: float) -> List[List[Tuple[int, int, int]]]:
        """Yield grid cell indices traversed by multiple rays using vectorized DDA.
        
        Parameters
        ----------
        grid : SparseGridIndex
            The sparse grid spatial index
        origins : np.ndarray
            Shape (N, 3) array of ray origins
        directions : np.ndarray
            Shape (N, 3) array of ray directions
        epsilon : float
            Tolerance for numerical comparisons
            
        Returns
        -------
        List[List[Tuple[int, int, int]]]
            List of cell lists, one per ray
        """
        N = origins.shape[0]
        bbox_min = grid.bbox_min
        bbox_max = grid.bbox_max
        cell_size = grid.cell_size
        
        # Safe directions (avoid division by zero)
        safe_dir = np.where(
            np.abs(directions) <= epsilon,
            np.where(directions == 0, epsilon, np.sign(directions) * epsilon),
            directions
        )
        inv_dir = 1.0 / safe_dir
        
        # Vectorized bbox intersection for all rays
        t1 = (bbox_min[np.newaxis, :] - origins) * inv_dir
        t2 = (bbox_max[np.newaxis, :] - origins) * inv_dir
        t_enter = np.max(np.minimum(t1, t2), axis=1)
        t_exit = np.min(np.maximum(t1, t2), axis=1)
        
        # Filter rays that intersect bbox
        valid_mask = (t_exit >= t_enter - epsilon)
        if not np.any(valid_mask):
            return [None] * N
        
        valid_indices = np.where(valid_mask)[0]
        valid_origins = origins[valid_mask]
        valid_directions = directions[valid_mask]
        valid_safe_dir = safe_dir[valid_mask]
        valid_t_enter = np.maximum(t_enter[valid_mask], 0.0)
        valid_t_exit = t_exit[valid_mask]
        
        N_valid = valid_origins.shape[0]
        
        # Compute entry and exit points for all valid rays
        entry_points = valid_origins + valid_t_enter[:, np.newaxis] * valid_directions
        exit_points = valid_origins + valid_t_exit[:, np.newaxis] * valid_directions
        
        # Compute cell indices at entry and exit
        entry_cells = np.floor((entry_points - grid.origin) / cell_size).astype(np.int64)
        exit_cells = np.floor((exit_points - grid.origin) / cell_size).astype(np.int64)
        
        # Ensure entry <= exit for each axis
        min_cells = np.minimum(entry_cells, exit_cells)
        max_cells = np.maximum(entry_cells, exit_cells)
        
        # Build result array with None for invalid rays
        result = [None] * N
        
        # Process each valid ray
        for ray_idx, orig_idx in enumerate(valid_indices):
            min_cell = min_cells[ray_idx]
            max_cell = max_cells[ray_idx]
            
            # Generate all cells in the bounding box
            x_range = np.arange(min_cell[0], max_cell[0] + 1, dtype=np.int64)
            y_range = np.arange(min_cell[1], max_cell[1] + 1, dtype=np.int64)
            z_range = np.arange(min_cell[2], max_cell[2] + 1, dtype=np.int64)
            
            # Create all combinations using meshgrid
            xx, yy, zz = np.meshgrid(x_range, y_range, z_range, indexing='ij')
            cells = np.stack([xx, yy, zz], axis=-1).reshape(-1, 3)
            
            # Convert to tuple list
            result[orig_idx] = [tuple(cell) for cell in cells]
        
        return result


    def _traverse_partition_no_broadcast(ray_partition: pd.DataFrame,
                                         grid: SparseGridIndex,
                                         voxel_data: Dict[str, np.ndarray],
                                         epsilon: float = 1e-6) -> pd.DataFrame:
        """Traverse a single ray partition using sparse grid DDA without broadcasting."""

        if ray_partition is None or len(ray_partition) == 0:
            return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        # Extract results from futures if needed
        if hasattr(voxel_data, 'result'):
            voxel_data = voxel_data.result()

        # Access voxel arrays
        v_ids = np.asarray(voxel_data['ids'])
        v_sizes = np.asarray(voxel_data['sizes'])
        v_centres = np.asarray(voxel_data['centres'])
        vmins = np.asarray(voxel_data['vmins'])
        vmaxs = np.asarray(voxel_data['vmaxs'])
    
        # Gather ray arrays from the partition
        ray_ids = ray_partition['ray_id'].to_numpy()
        leg_ids = ray_partition['leg_id'].to_numpy()
        origins = ray_partition[['origin_x','origin_y','origin_z']].to_numpy(dtype=np.float64)
        directions = ray_partition[['direction_x','direction_y','direction_z']].to_numpy(dtype=np.float64)
        points = ray_partition[['point_x','point_y','point_z']].to_numpy(dtype=np.float64)
        normals = ray_partition[['normal_x','normal_y','normal_z']].to_numpy(dtype=np.float64)
        echo_intensity = ray_partition['echo_intensity'].to_numpy()
        point_weight = ray_partition['point_weight'].to_numpy()
        is_leaf = ray_partition['is_leaf'].to_numpy()
        return_number = ray_partition['return_number'].to_numpy()
        number_of_returns = ray_partition['number_of_returns'].to_numpy()

        # Build ray-cell-voxel mapping during DDA traversal
        ray_cell_voxel_pairs = []  # List of (ray_idx, cell, voxel_idx) tuples

        # Process rays in batches
        batch_size = 10000  # Adjust based on memory constraints
        for batch_start in range(0, len(origins), batch_size):
            batch_end = min(batch_start + batch_size, len(origins))
            batch_indices = np.arange(batch_start, batch_end)
            o_batch = origins[batch_indices]
            d_batch = directions[batch_indices]

            all_cells = _dda_cells_for_rays_vectorized(grid, o_batch, d_batch, epsilon)
            assert len(all_cells) == o_batch.shape[0]

            # Process each ray's cells and voxels
            for ray_local_idx in range(len(all_cells)):
                ray_idx = batch_indices[ray_local_idx]
                cells = all_cells[ray_local_idx]
                if cells is not None:
                    for cell in cells:
                        voxel_indices = grid.map.get(cell)
                        if voxel_indices is not None:
                            for voxel_idx in voxel_indices:
                                ray_cell_voxel_pairs.append((ray_idx, cell, voxel_idx))

        del ray_partition, grid, voxel_data
        gc.collect()

        if not ray_cell_voxel_pairs:
            del ray_cell_voxel_pairs, v_ids, v_sizes, v_centres, vmins, vmaxs
            del ray_ids, leg_ids, origins, directions, points, normals
            del echo_intensity, point_weight, is_leaf, return_number, number_of_returns
            gc.collect()
            return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        # Convert to numpy arrays for numba
        ray_cell_voxel_array = np.empty(len(ray_cell_voxel_pairs), dtype=[
            ('ray_idx', np.int64),
            ('cell_x', np.int64),
            ('cell_y', np.int64),
            ('cell_z', np.int64),
            ('voxel_idx', np.int64)
        ])
        for i, (ray_idx, cell, voxel_idx) in enumerate(ray_cell_voxel_pairs):
            ray_cell_voxel_array[i] = (ray_idx, cell[0], cell[1], cell[2], voxel_idx)
        del ray_cell_voxel_pairs
        gc.collect()

        # Numba-compiled kernel for processing pre-computed ray-cell-voxel pairs
        # NOTE: Avoids global state, uses only local variables, and returns a plain list for thread safety.
        

        # Ensure nans are handled for incompatible fields before casting
        return_number = np.nan_to_num(return_number, nan=0.0)
        number_of_returns = np.nan_to_num(number_of_returns, nan=0.0)
        point_weight = np.nan_to_num(point_weight, nan=0.0)
        is_leaf = np.nan_to_num(is_leaf, nan=0).astype(np.bool_)


        # Convert input arrays to numba-compatible types matching voxel_ray_intersection_schema
        origins_nb = origins.astype(np.float64)
        directions_nb = directions.astype(np.float64)
        points_nb = points.astype(np.float64)
        normals_nb = normals.astype(np.float64)
        v_centres_nb = v_centres.astype(np.float64)
        vmins_nb = vmins.astype(np.float64)
        vmaxs_nb = vmaxs.astype(np.float64)
        v_ids_nb = v_ids.astype(np.uint64)
        v_sizes_nb = v_sizes.astype(np.float32)
        leg_ids_nb = leg_ids.astype(np.uint64)
        ray_ids_nb = ray_ids.astype(np.uint64)
        echo_intensity_nb = echo_intensity.astype(np.float64)
        return_number_nb = return_number.astype(np.int32)
        number_of_returns_nb = number_of_returns.astype(np.int32)
        point_weight_nb = point_weight.astype(np.float64)
        is_leaf_nb = is_leaf.astype(np.bool_)

        del origins, directions, points, normals, v_centres, vmins, vmaxs
        del v_ids, v_sizes, leg_ids, ray_ids, echo_intensity, return_number, number_of_returns, point_weight, is_leaf
        gc.collect()

        # Run numba kernel on pre-computed pairs
        out_data = process_ray_voxel_pairs_kernel(
            ray_cell_voxel_array,
            origins_nb, directions_nb, points_nb, normals_nb,
            echo_intensity_nb, return_number_nb, number_of_returns_nb,
            point_weight_nb, is_leaf_nb, leg_ids_nb, ray_ids_nb,
            v_ids_nb, v_sizes_nb, v_centres_nb, vmins_nb, vmaxs_nb,
            np.float64(epsilon)
        )
        print(f"Processed partition: found {len(out_data)} intersections.")
        print(f"Example data: {out_data[:5]}")

        del ray_cell_voxel_array
        del origins_nb, directions_nb, points_nb, normals_nb
        del echo_intensity_nb, return_number_nb, number_of_returns_nb
        del point_weight_nb, is_leaf_nb, leg_ids_nb, ray_ids_nb
        del v_ids_nb, v_sizes_nb, v_centres_nb, vmins_nb, vmaxs_nb
        gc.collect()

        # Convert output to DataFrame rows
        if not out_data:
            del out_data
            gc.collect()
            return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        df = pd.DataFrame(out_data, columns=voxel_ray_intersection_schema.names)
        del out_data
        gc.collect()
        return df

    # --------------------------
    # Main code: orchestrate Dask processing and saving
    # --------------------------

    # Resolve temp_dir
    temp_dir = _resolve_temp_dir(temp_dir)

    # Setup dask client
    mem_limit_str, n_workers, threads_per_worker = _determine_dask_resources(cpus=cpus, mem=mem, optimal_threads=5)

    # Pass n_workers into dask, and threads into numba
    os.environ['OMP_NUM_THREADS'] = str(threads_per_worker)
    os.environ['NUMBA_NUM_THREADS'] = str(threads_per_worker)
    set_num_threads(threads_per_worker)

    client = _start_dask_client(
        n_workers=n_workers, 
        threads_per_worker=1,
        memory_limit=mem_limit_str, 
        memory_target_fraction=0.8,
        memory_pause_fraction=0.9,
        memory_spill_fraction=0.95,
        temp_dir=temp_dir,
        processes=True
        )
    print(f"[voxel_ray_intersections] Starting Dask client with {n_workers} workers, "
        f"{threads_per_worker} threads/worker, {mem_limit_str} memory/worker.")
    
    # Build voxel references dataframe
    voxel_references = _compile_voxel_references(references_dir)
    print(f"Compiled {len(voxel_references)} voxel references.")

    # Build sparse grid index
    grid, voxel_data = _build_sparse_grid(voxel_references, epsilon=epsilon)
    print(f"Sparse grid built: {len(grid.map)} occupied cells; cell_size={grid.cell_size:.6f}")

    # Scatter read-only index to workers (broadcast=True ensures presence on all)
    grid_fut = client.scatter(grid, broadcast=True)
    voxel_fut = client.scatter(voxel_data, broadcast=True)

    # Process legs
    valid_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))
    print(f"Found {len(valid_files)} valid rays files.")

    meta = pd.DataFrame(columns=voxel_ray_intersection_schema.names)

    # Submit all tasks at once for concurrent processing
    futures_dict = {}
    for file in valid_files:
        base = os.path.basename(file)
        parts = os.path.splitext(base)[0].split('_')
        try:
            leg_id = int(parts[1])
        except Exception:
            leg_id = next((int(p) for p in parts if p.isdigit()), 0)

        ddf = dd.read_parquet(file, engine='pyarrow', blocksize='50MB')

        def _map_partition(ray_part: pd.DataFrame, grid_obj, voxel_obj) -> pd.DataFrame:
            return _traverse_partition_no_broadcast(ray_part, grid_obj, voxel_obj, epsilon=epsilon)

        result_ddf = ddf.map_partitions(_map_partition, grid_obj=grid, voxel_obj=voxel_data, meta=meta)
        fut = client.compute(result_ddf, sync=False)  # async submit
        futures_dict[fut] = leg_id
        print(f"Leg {leg_id}: submitted with {ddf.npartitions} ray partitions")

    # Save as they complete
    start_time = time.time()
    for fut in as_completed(futures_dict):
        leg_id = futures_dict[fut]
        result_df = fut.result()
        if result_df is None or len(result_df) == 0:
            print(f"Leg {leg_id}: no intersections")
            continue
        grouped = result_df.groupby('voxel_size', group_keys=True)
        for vox_size, grp in grouped:
            out_path = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{round(float(vox_size),2)}_intersections.parquet")
            grp.to_parquet(out_path, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
            del grp
        del result_df
        print(f"Leg {leg_id}: saved groups")
    
    end_time = time.time()
    total_time = end_time - start_time
    print(f"[voxel_ray_intersections] Completed in {total_time:.2f} seconds.")

    time.sleep(0.5)
    _close_dask_client(client)
    print("[voxel_ray_intersections] Dask client closed.")

# -*- coding: utf-8 -*-
"""
Scratch refactor: Dask for I/O (load/save), Numba for all numeric computations.
"""

def _resolve_temp_dir(tmp_hint: str | None = None) -> str:
    if tmp_hint is not None and os.path.isdir(tmp_hint):
        return tmp_hint
    hpc_tmp = os.environ.get("TMPDIR")
    if hpc_tmp and os.path.isdir(hpc_tmp):
        return hpc_tmp
    os_tmp = tempfile.gettempdir()
    if os_tmp and os.path.isdir(os_tmp):
        return os_tmp
    return "/tmp"


def _compile_voxel_references(references_dir: str) -> pd.DataFrame:
    voxel_files = glob.glob(os.path.join(references_dir, '*.csv'))
    dfs: List[pd.DataFrame] = []
    for voxel_ref in voxel_files:
        df = pd.read_csv(voxel_ref, index_col=None, header=0)
        if 'voxel_id' not in df.columns:
            voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
            df['voxel_id'] = df.apply(
                lambda row: create_voxel_id(
                    voxel_size=row['voxel_size'] if 'voxel_size' in row else voxel_size,
                    x=row['voxel_cx'], y=row['voxel_cy'], z=row['voxel_cz']
                ), axis=1
            )
        df = df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']].drop_duplicates()
        voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
        df['voxel_size'] = voxel_size
        dfs.append(df)
    if not dfs:
        return pd.DataFrame(columns=['voxel_id','voxel_cx','voxel_cy','voxel_cz','voxel_size'])
    return pd.concat(dfs, ignore_index=True)


# Calculate avail_cpus, avail_mem, and return optimal worker/thread config
def _determine_dask_resources(cpus: int | None, mem: int | None, optimal_threads: int = 8, mem_threshold: float = 0.7) -> tuple[str, int, int]:
    """
    Determine available CPUs and memory for Dask configuration.
    Returns (avail_cpus, avail_mem_string_for_dask, optimal_workers, threads_per_worker).
    """
    if cpus is not None:
        avail_threads = cpus
    else:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            threads = 2
        else:
            avail_cpus = psutil.cpu_count(logical=False)
            threads = psutil.cpu_count(logical=True) // avail_cpus

    if mem is not None:
        avail_mem = int(mem * mem_threshold)  # in MB
    else:
        avail_mem = int(float(os.environ.get('SLURM_MEM_PER_NODE', psutil.virtual_memory().available // (1024 * 1024))) * mem_threshold)  # in MB

    # Test for not oversubscribing
    threads = 1
    n_workers = (avail_cpus * threads) // optimal_threads
    threads_per_worker = optimal_threads

    memory_worker = avail_mem / n_workers
    avail_mem_string_for_dask = f"{int(memory_worker)}MB"

    return avail_mem_string_for_dask, n_workers, threads_per_worker


def _build_sparse_grid_arrays(voxel_refs: pd.DataFrame, epsilon: float = 1e-6):
    voxel_ids = voxel_refs['voxel_id'].to_numpy()
    voxel_sizes = voxel_refs['voxel_size'].to_numpy(dtype=np.float64)
    centres = voxel_refs[['voxel_cx','voxel_cy','voxel_cz']].to_numpy(dtype=np.float64)

    half = (voxel_sizes[:, None] / 2.0)
    vmins = centres - (half - epsilon)
    vmaxs = centres + (half + epsilon)

    bbox_min = np.min(vmins, axis=0)
    bbox_max = np.max(vmaxs, axis=0)

    cell_size = max(1e-9, float(np.min(voxel_sizes)))
    origin = bbox_min.copy()

    # Build dict cell -> list of voxel indices
    cell_map: Dict[Tuple[int,int,int], List[int]] = {}
    def cell_index(p):
        rel = (p - origin) / cell_size
        return (int(math.floor(rel[0])), int(math.floor(rel[1])), int(math.floor(rel[2])))

    for i in range(centres.shape[0]):
        start = cell_index(vmins[i]); end = cell_index(vmaxs[i])
        for ix in range(start[0], end[0] + 1):
            for iy in range(start[1], end[1] + 1):
                for iz in range(start[2], end[2] + 1):
                    key = (ix, iy, iz)
                    cell_map.setdefault(key, []).append(i)

    # Convert to CSR-like arrays for Numba (sorted keys)
    keys = list(cell_map.keys())
    keys.sort()
    K = len(keys)
    keys_ix = np.empty(K, dtype=np.int32)
    keys_iy = np.empty(K, dtype=np.int32)
    keys_iz = np.empty(K, dtype=np.int32)
    sizes = np.empty(K, dtype=np.int64)
    lists = []
    for idx, (ix,iy,iz) in enumerate(keys):
        keys_ix[idx] = ix; keys_iy[idx] = iy; keys_iz[idx] = iz
        lst = np.array(cell_map[(ix,iy,iz)], dtype=np.int64)
        sizes[idx] = lst.size
        lists.append(lst)
    offsets = np.empty(K+1, dtype=np.int64)
    offsets[0] = 0
    np.cumsum(sizes, out=offsets[1:])
    voxel_ids_flat = np.concatenate(lists) if lists else np.empty(0, dtype=np.int64)

    voxel_data = {
        'ids': voxel_ids,
        'sizes': voxel_sizes,
        'centres': centres,
        'vmins': vmins,
        'vmaxs': vmaxs,
        'bbox_min': bbox_min,
        'bbox_max': bbox_max,
        'cell_size': cell_size,
        'origin': origin,
        'keys_ix': keys_ix,
        'keys_iy': keys_iy,
        'keys_iz': keys_iz,
        'offsets': offsets,
        'voxel_ids_flat': voxel_ids_flat,
    }
    return voxel_data


@njit(cache=True, fastmath=True)
def _binary_search_key(ix, iy, iz, keys_ix, keys_iy, keys_iz):
    # keys_* are sorted lexicographically by (ix,iy,iz)
    lo = 0; hi = keys_ix.shape[0] - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        kx = keys_ix[mid]; ky = keys_iy[mid]; kz = keys_iz[mid]
        if ix < kx or (ix == kx and (iy < ky or (iy == ky and iz < kz))):
            hi = mid - 1
        elif ix > kx or (ix == kx and (iy > ky or (iy == ky and iz > kz))):
            lo = mid + 1
        else:
            return mid
    return -1


@njit(cache=True, fastmath=True)
def _safe_dir(d, eps):
    ds0 = d[0]; ds1 = d[1]; ds2 = d[2]
    if abs(ds0) <= eps: ds0 = eps if ds0 == 0 else (eps if ds0 > 0 else -eps)
    if abs(ds1) <= eps: ds1 = eps if ds1 == 0 else (eps if ds1 > 0 else -eps)
    if abs(ds2) <= eps: ds2 = eps if ds2 == 0 else (eps if ds2 > 0 else -eps)
    return ds0, ds1, ds2


@njit(cache=True, fastmath=True)
def _ray_box_entry_exit(o, d, bbox_min, bbox_max, eps):
    ds0, ds1, ds2 = _safe_dir(d, eps)
    inv0 = 1.0/ds0; inv1 = 1.0/ds1; inv2 = 1.0/ds2
    t1x = (bbox_min[0] - o[0]) * inv0; t2x = (bbox_max[0] - o[0]) * inv0
    t1y = (bbox_min[1] - o[1]) * inv1; t2y = (bbox_max[1] - o[1]) * inv1
    t1z = (bbox_min[2] - o[2]) * inv2; t2z = (bbox_max[2] - o[2]) * inv2
    tminx = t1x if t1x < t2x else t2x; tmaxx = t1x if t1x > t2x else t2x
    tminy = t1y if t1y < t2y else t2y; tmaxy = t1y if t1y > t2y else t2y
    tminz = t1z if t1z < t2z else t2z; tmaxz = t1z if t1z > t2z else t2z
    t_enter = tminx
    if tminy > t_enter: t_enter = tminy
    if tminz > t_enter: t_enter = tminz
    t_exit = tmaxx
    if tmaxy < t_exit: t_exit = tmaxy
    if tmaxz < t_exit: t_exit = tmaxz
    return t_enter, t_exit, ds0, ds1, ds2


@njit(cache=True, fastmath=True)
def _dda_mark_candidates(o, d, origin, cell_size, bbox_min, bbox_max,
                         keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
                         mask, eps):
    # Traverse only cells along the ray; mark candidate voxels in mask
    t_enter, t_exit, ds0, ds1, ds2 = _ray_box_entry_exit(o, d, bbox_min, bbox_max, eps)
    if not (t_exit >= t_enter - eps):
        return
    start_t = t_enter if t_enter > 0.0 else 0.0
    start_p0 = o[0] + start_t * d[0]
    start_p1 = o[1] + start_t * d[1]
    start_p2 = o[2] + start_t * d[2]
    # initial cell indices
    cx = int(math.floor((start_p0 - origin[0]) / cell_size))
    cy = int(math.floor((start_p1 - origin[1]) / cell_size))
    cz = int(math.floor((start_p2 - origin[2]) / cell_size))
    # step per axis
    stepx = 1 if d[0] > 0 else (-1 if d[0] < 0 else 0)
    stepy = 1 if d[1] > 0 else (-1 if d[1] < 0 else 0)
    stepz = 1 if d[2] > 0 else (-1 if d[2] < 0 else 0)
    # next boundary tMax per axis
    def next_plane(axis_coord, axis, step_axis, ds):
        if step_axis == 0:
            return 1e300
        plane = ( ( (cx if axis==0 else (cy if axis==1 else cz)) + (1 if step_axis>0 else 0) ) * cell_size ) + origin[axis]
        return (plane - (start_p0 if axis==0 else (start_p1 if axis==1 else start_p2))) / ds
    tMaxX = next_plane(cx, 0, stepx, ds0)
    tMaxY = next_plane(cy, 1, stepy, ds1)
    tMaxZ = next_plane(cz, 2, stepz, ds2)
    tDeltaX = (cell_size / abs(ds0)) if stepx != 0 else 1e300
    tDeltaY = (cell_size / abs(ds1)) if stepy != 0 else 1e300
    tDeltaZ = (cell_size / abs(ds2)) if stepz != 0 else 1e300
    t = start_t
    steps = 0
    max_steps = 1000000
    while t <= t_exit + eps and steps < max_steps:
        # mark candidates for this cell
        ki = _binary_search_key(cx, cy, cz, keys_ix, keys_iy, keys_iz)
        if ki >= 0:
            s = offsets[ki]; e = offsets[ki+1]
            for idx in range(s, e):
                vi = voxel_ids_flat[idx]
                mask[vi] = True
        # step to next cell
        steps += 1
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            cx += stepx; t = tMaxX; tMaxX += tDeltaX
        elif tMaxY <= tMaxX and tMaxY <= tMaxZ:
            cy += stepy; t = tMaxY; tMaxY += tDeltaY
        else:
            cz += stepz; t = tMaxZ; tMaxZ += tDeltaZ


@njit(cache=True, fastmath=True)
def _sphere_cull(o, d, centres, radius_sq, cand_idx, eps):
    M = cand_idx.shape[0]
    keep = np.zeros(M, np.bool_)
    for i in range(M):
        vi = cand_idx[i]
        ocx = o[0] - centres[vi,0]
        ocy = o[1] - centres[vi,1]
        ocz = o[2] - centres[vi,2]
        b = 2.0 * (ocx*d[0] + ocy*d[1] + ocz*d[2])
        c = ocx*ocx + ocy*ocy + ocz*ocz - radius_sq[vi]
        disc = b*b - 4.0*c
        keep[i] = disc >= -eps
    return keep


@njit(cache=True, fastmath=True)
def _slab_per_candidates(o, d, vmins, vmaxs, cand_idx, eps):
    # Two-pass: count then fill
    ds0, ds1, ds2 = _safe_dir(d, eps)
    inv0 = 1.0/ds0; inv1 = 1.0/ds1; inv2 = 1.0/ds2
    M = cand_idx.shape[0]
    count = 0
    for i in range(M):
        vi = cand_idx[i]
        t1x = (vmins[vi,0]-o[0])*inv0; t2x = (vmaxs[vi,0]-o[0])*inv0
        t1y = (vmins[vi,1]-o[1])*inv1; t2y = (vmaxs[vi,1]-o[1])*inv1
        t1z = (vmins[vi,2]-o[2])*inv2; t2z = (vmaxs[vi,2]-o[2])*inv2
        tminx = t1x if t1x < t2x else t2x; tmaxx = t1x if t1x > t2x else t2x
        tminy = t1y if t1y < t2y else t2y; tmaxy = t1y if t1y > t2y else t2y
        tminz = t1z if t1z < t2z else t2z; tmaxz = t1z if t1z > t2z else t2z
        t_enter = tminx
        if tminy > t_enter: t_enter = tminy
        if tminz > t_enter: t_enter = tminz
        t_exit = tmaxx
        if tmaxy < t_exit: t_exit = tmaxy
        if tmaxz < t_exit: t_exit = tmaxz
        ok = (t_enter <= t_exit + eps) and (t_exit >= -eps)
        if ok:
            count += 1
    # allocate outputs
    hit_idx = np.empty(count, np.int64)
    entry = np.empty((count,3), np.float64)
    exitp = np.empty((count,3), np.float64)
    k = 0
    for i in range(M):
        vi = cand_idx[i]
        t1x = (vmins[vi,0]-o[0])*inv0; t2x = (vmaxs[vi,0]-o[0])*inv0
        t1y = (vmins[vi,1]-o[1])*inv1; t2y = (vmaxs[vi,1]-o[1])*inv1
        t1z = (vmins[vi,2]-o[2])*inv2; t2z = (vmaxs[vi,2]-o[2])*inv2
        tminx = t1x if t1x < t2x else t2x; tmaxx = t1x if t1x > t2x else t2x
        tminy = t1y if t1y < t2y else t2y; tmaxy = t1y if t1y > t2y else t2y
        tminz = t1z if t1z < t2z else t2z; tmaxz = t1z if t1z > t2z else t2z
        t_enter = tminx
        if tminy > t_enter: t_enter = tminy
        if tminz > t_enter: t_enter = tminz
        t_exit = tmaxx
        if tmaxy < t_exit: t_exit = tmaxy
        if tmaxz < t_exit: t_exit = tmaxz
        ok = (t_enter <= t_exit + eps) and (t_exit >= -eps)
        if ok:
            hit_idx[k] = vi
            entry[k,0] = o[0] + t_enter*d[0]; entry[k,1] = o[1] + t_enter*d[1]; entry[k,2] = o[2] + t_enter*d[2]
            exitp[k,0] = o[0] + t_exit *d[0]; exitp[k,1] = o[1] + t_exit *d[1]; exitp[k,2] = o[2] + t_exit *d[2]
            k += 1
    return hit_idx, entry, exitp


@njit(cache=True, fastmath=True)
def _slab_count_only(o, d, vmins, vmaxs, cand_idx, eps):
    """Count slab intersections without allocating output arrays."""
    ds0, ds1, ds2 = _safe_dir(d, eps)
    inv0 = 1.0/ds0; inv1 = 1.0/ds1; inv2 = 1.0/ds2
    M = cand_idx.shape[0]
    count = 0
    for i in range(M):
        vi = cand_idx[i]
        t1x = (vmins[vi,0]-o[0])*inv0; t2x = (vmaxs[vi,0]-o[0])*inv0
        t1y = (vmins[vi,1]-o[1])*inv1; t2y = (vmaxs[vi,1]-o[1])*inv1
        t1z = (vmins[vi,2]-o[2])*inv2; t2z = (vmaxs[vi,2]-o[2])*inv2
        tminx = t1x if t1x < t2x else t2x; tmaxx = t1x if t1x > t2x else t2x
        tminy = t1y if t1y < t2y else t2y; tmaxy = t1y if t1y > t2y else t2y
        tminz = t1z if t1z < t2z else t2z; tmaxz = t1z if t1z > t2z else t2z
        t_enter = tminx
        if tminy > t_enter: t_enter = tminy
        if tminz > t_enter: t_enter = tminz
        t_exit = tmaxx
        if tmaxy < t_exit: t_exit = tmaxy
        if tmaxz < t_exit: t_exit = tmaxz
        ok = (t_enter <= t_exit + eps) and (t_exit >= -eps)
        if ok:
            count += 1
    return count


@njit(cache=True, fastmath=True)
def _slab_fill_candidates(o, d, vmins, vmaxs, cand_idx, eps,
                          hit_idx_buf, entry_buf, exit_buf):
    """Fill preallocated buffers and return the number of hits."""
    ds0, ds1, ds2 = _safe_dir(d, eps)
    inv0 = 1.0/ds0; inv1 = 1.0/ds1; inv2 = 1.0/ds2
    M = cand_idx.shape[0]
    k = 0
    for i in range(M):
        vi = cand_idx[i]
        t1x = (vmins[vi,0]-o[0])*inv0; t2x = (vmaxs[vi,0]-o[0])*inv0
        t1y = (vmins[vi,1]-o[1])*inv1; t2y = (vmaxs[vi,1]-o[1])*inv1
        t1z = (vmins[vi,2]-o[2])*inv2; t2z = (vmaxs[vi,2]-o[2])*inv2
        tminx = t1x if t1x < t2x else t2x; tmaxx = t1x if t1x > t2x else t2x
        tminy = t1y if t1y < t2y else t2y; tmaxy = t1y if t1y > t2y else t2y
        tminz = t1z if t1z < t2z else t2z; tmaxz = t1z if t1z > t2z else t2z
        t_enter = tminx
        if tminy > t_enter: t_enter = tminy
        if tminz > t_enter: t_enter = tminz
        t_exit = tmaxx
        if tmaxy < t_exit: t_exit = tmaxy
        if tmaxz < t_exit: t_exit = tmaxz
        ok = (t_enter <= t_exit + eps) and (t_exit >= -eps)
        if ok:
            hit_idx_buf[k]  = vi
            entry_buf[k,0]  = o[0] + t_enter*d[0]
            entry_buf[k,1]  = o[1] + t_enter*d[1]
            entry_buf[k,2]  = o[2] + t_enter*d[2]
            exit_buf[k,0]   = o[0] + t_exit *d[0]
            exit_buf[k,1]   = o[1] + t_exit *d[1]
            exit_buf[k,2]   = o[2] + t_exit *d[2]
            k += 1
    return k


@njit(cache=True, fastmath=False)
def _process_partition_numba(
    origins, directions, points, normals,
    echo_intensity, return_number, number_of_returns, point_weight, is_leaf,
    leg_ids, ray_ids,
    v_ids, v_sizes, v_centres, vmins, vmaxs,
    origin_grid, cell_size, bbox_min, bbox_max,
    keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
    eps
):
    n_rays = origins.shape[0]
    # precompute voxel radius^2 for sphere cull
    radius_sq = ((v_sizes * math.sqrt(3) * 0.5) + 0.05) ** 2
    # worst-case storage: assume small number of hits per ray; we accumulate in Python outside
    # Here we collect rows in fixed-size buffers per ray then append to a typed list
    total_hits = 0
    # first pass: count hits to preallocate output arrays once
    for r in range(n_rays):
        # candidate mask for voxels
        mask = np.zeros(v_sizes.shape[0], np.bool_)
        _dda_mark_candidates(origins[r], directions[r], origin_grid, cell_size,
                             bbox_min, bbox_max,
                             keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
                             mask, eps)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            continue
        keep = _sphere_cull(origins[r], directions[r], v_centres, radius_sq, cand_idx, eps)
        cand2 = cand_idx[keep]
        if cand2.size == 0:
            continue
        hits_idx, _, _ = _slab_per_candidates(origins[r], directions[r], vmins, vmaxs, cand2, eps)
        total_hits += hits_idx.size
    # allocate output arrays
    # 26 columns (per schema) -> create arrays; then Python will build DataFrame
    voxel_size_col      = np.empty(total_hits, np.float64)
    voxel_id_col        = np.empty(total_hits, np.int64)
    voxel_cx_col        = np.empty(total_hits, np.float64)
    voxel_cy_col        = np.empty(total_hits, np.float64)
    voxel_cz_col        = np.empty(total_hits, np.float64)
    leg_id_col          = np.empty(total_hits, np.int64)
    ray_id_col          = np.empty(total_hits, np.int64)
    t_entry_x_col       = np.empty(total_hits, np.float64)
    t_entry_y_col       = np.empty(total_hits, np.float64)
    t_entry_z_col       = np.empty(total_hits, np.float64)
    t_exit_x_col        = np.empty(total_hits, np.float64)
    t_exit_y_col        = np.empty(total_hits, np.float64)
    t_exit_z_col        = np.empty(total_hits, np.float64)
    distance_to_centre  = np.empty(total_hits, np.float64)
    point_x_col         = np.empty(total_hits, np.float64)
    point_y_col         = np.empty(total_hits, np.float64)
    point_z_col         = np.empty(total_hits, np.float64)
    echo_intensity_col  = np.empty(total_hits, np.float64)
    return_number_col   = np.empty(total_hits, np.int32)
    number_of_returns_col= np.empty(total_hits, np.int32)
    normal_x_col        = np.empty(total_hits, np.float64)
    normal_y_col        = np.empty(total_hits, np.float64)
    normal_z_col        = np.empty(total_hits, np.float64)
    point_weight_col    = np.empty(total_hits, np.float64)
    viewing_angle_col   = np.empty(total_hits, np.float64)
    hit_type_col        = np.empty(total_hits, np.int32)
    is_leaf_col         = np.empty(total_hits, np.bool_)

    k = 0
    for r in range(n_rays):
        mask = np.zeros(v_sizes.shape[0], np.bool_)
        _dda_mark_candidates(origins[r], directions[r], origin_grid, cell_size,
                             bbox_min, bbox_max,
                             keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
                             mask, eps)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            continue
        keep = _sphere_cull(origins[r], directions[r], v_centres, radius_sq, cand_idx, eps)
        cand2 = cand_idx[keep]
        if cand2.size == 0:
            continue
        hits_idx, entry, exitp = _slab_per_candidates(origins[r], directions[r], vmins, vmaxs, cand2, eps)
        for h in range(hits_idx.size):
            vi = hits_idx[h]
            # viewing angle: zenith angle normalized to 90 degrees
            dn = math.sqrt(directions[r,0]**2 + directions[r,1]**2 + directions[r,2]**2)
            if dn > 0.0:
                cos_theta = directions[r,2] / dn
                cos_theta = max(-1.0, min(1.0, cos_theta))  # Clamp to [-1, 1]
                viewing_angle = math.degrees(math.acos(cos_theta))
                if viewing_angle > 90.0:
                    viewing_angle = 180.0 - viewing_angle
            else:
                viewing_angle = 0.0
            va = 0.0
            if dn > 0.0:
                cth = directions[r,2] / dn
                if cth < -1.0: cth = -1.0
                if cth > 1.0: cth = 1.0
                ang = math.degrees(math.acos(cth))
                va = ang if ang <= 90.0 else 180.0 - ang
            # classification
            p = points[r]
            vmin = vmins[vi]; vmax = vmaxs[vi]
            in_voxel = (p[0] >= vmin[0]-eps and p[0] <= vmax[0]+eps and
                        p[1] >= vmin[1]-eps and p[1] <= vmax[1]+eps and
                        p[2] >= vmin[2]-eps and p[2] <= vmax[2]+eps)
            unbound = (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]))
            de = (origins[r,0]-entry[h,0])**2 + (origins[r,1]-entry[h,1])**2 + (origins[r,2]-entry[h,2])**2
            dx = (origins[r,0]-exitp[h,0])**2 + (origins[r,1]-exitp[h,1])**2 + (origins[r,2]-exitp[h,2])**2
            dp = (origins[r,0]-p[0])**2 + (origins[r,1]-p[1])**2 + (origins[r,2]-p[2])**2
            before = (de > dp) and (not in_voxel) and (not unbound)
            after  = (dx < dp) and (not in_voxel) and (not unbound)
            hit_type = -1
            if unbound: hit_type = 0
            elif before: hit_type = 1
            elif in_voxel: hit_type = 2
            elif after: hit_type = 3
            
            # fill columns
            voxel_size_col[k]     = v_sizes[vi]
            voxel_id_col[k]       = v_ids[vi]
            voxel_cx_col[k]       = v_centres[vi,0]
            voxel_cy_col[k]       = v_centres[vi,1]
            voxel_cz_col[k]       = v_centres[vi,2]
            leg_id_col[k]         = leg_ids[r]
            ray_id_col[k]         = ray_ids[r]
            t_entry_x_col[k]      = entry[h,0]
            t_entry_y_col[k]      = entry[h,1]
            t_entry_z_col[k]      = entry[h,2]
            t_exit_x_col[k]       = exitp[h,0]
            t_exit_y_col[k]       = exitp[h,1]
            t_exit_z_col[k]       = exitp[h,2]
            distance_to_centre[k] = math.sqrt( (origins[r,0]-v_centres[vi,0])**2 +
                                               (origins[r,1]-v_centres[vi,1])**2 +
                                               (origins[r,2]-v_centres[vi,2])**2 )
            point_x_col[k]        = p[0]
            point_y_col[k]        = p[1]
            point_z_col[k]        = p[2]
            echo_intensity_col[k] = echo_intensity[r]
            return_number_col[k]  = int(return_number[r]) if not np.isnan(return_number[r]) else 0
            number_of_returns_col[k]= int(number_of_returns[r]) if not np.isnan(number_of_returns[r]) else 0
            normal_x_col[k]       = normals[r,0]
            normal_y_col[k]       = normals[r,1]
            normal_z_col[k]       = normals[r,2]
            point_weight_col[k]   = point_weight[r]
            viewing_angle_col[k]  = va
            hit_type_col[k]       = hit_type
            is_leaf_col[k]        = bool(is_leaf[r])
            k += 1
    # build dict of columns, truncated to k
    data = (
        voxel_size_col[:k], voxel_id_col[:k], voxel_cx_col[:k], voxel_cy_col[:k], voxel_cz_col[:k],
        leg_id_col[:k], ray_id_col[:k],
        t_entry_x_col[:k], t_entry_y_col[:k], t_entry_z_col[:k],
        t_exit_x_col[:k],  t_exit_y_col[:k],  t_exit_z_col[:k],
        distance_to_centre[:k],
        point_x_col[:k], point_y_col[:k], point_z_col[:k],
        echo_intensity_col[:k], return_number_col[:k], number_of_returns_col[:k],
        normal_x_col[:k], normal_y_col[:k], normal_z_col[:k],
        point_weight_col[:k], viewing_angle_col[:k], hit_type_col[:k], is_leaf_col[:k]
    )
    return data, k


@njit(cache=True, fastmath=False)
def _process_partition_numba_new(
    origins, directions, points, normals,
    echo_intensity, return_number, number_of_returns, point_weight, is_leaf,
    leg_ids, ray_ids,
    v_ids, v_sizes, v_centres, vmins, vmaxs,
    origin_grid, cell_size, bbox_min, bbox_max,
    keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
    eps
):
    n_rays = origins.shape[0]

    # precompute voxel radius^2 for sphere cull
    radius_sq = ((v_sizes * math.sqrt(3) * 0.5) + 0.05) ** 2  # unchanged

    # ---------- PASS 1: count hits (no per-ray slabs, one reusable mask) ----------
    total_hits = 0
    max_cand2 = 0

    # one boolean mask for the whole partition (reused)
    mask = np.zeros(v_sizes.shape[0], np.bool_)

    for r in range(n_rays):
        _dda_mark_candidates(
            origins[r], directions[r], origin_grid, cell_size,
            bbox_min, bbox_max, keys_ix, keys_iy, keys_iz,
            offsets, voxel_ids_flat, mask, eps
        )  

        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            # clear nothing; continue
            continue

        keep = _sphere_cull(origins[r], directions[r], v_centres, radius_sq, cand_idx, eps)
        cand2 = cand_idx[keep]

        # reset only the touched bits (avoid re-allocating mask)
        for i in range(cand_idx.size):
            mask[cand_idx[i]] = False

        if cand2.size == 0:
            continue

        total_hits += _slab_count_only(origins[r], directions[r], vmins, vmaxs, cand2, eps)
        if cand2.size > max_cand2:
            max_cand2 = cand2.size

    # ---------- Allocate output columns once (same dtypes, same order) ----------
    voxel_size_col        = np.empty(total_hits, np.float64)
    voxel_id_col          = np.empty(total_hits, np.int64)
    voxel_cx_col          = np.empty(total_hits, np.float64)
    voxel_cy_col          = np.empty(total_hits, np.float64)
    voxel_cz_col          = np.empty(total_hits, np.float64)
    leg_id_col            = np.empty(total_hits, np.int64)
    ray_id_col            = np.empty(total_hits, np.int64)
    t_entry_x_col         = np.empty(total_hits, np.float64)
    t_entry_y_col         = np.empty(total_hits, np.float64)
    t_entry_z_col         = np.empty(total_hits, np.float64)
    t_exit_x_col          = np.empty(total_hits, np.float64)
    t_exit_y_col          = np.empty(total_hits, np.float64)
    t_exit_z_col          = np.empty(total_hits, np.float64)
    distance_to_centre    = np.empty(total_hits, np.float64)
    point_x_col           = np.empty(total_hits, np.float64)
    point_y_col           = np.empty(total_hits, np.float64)
    point_z_col           = np.empty(total_hits, np.float64)
    echo_intensity_col    = np.empty(total_hits, np.float64)
    return_number_col     = np.empty(total_hits, np.int32)
    number_of_returns_col = np.empty(total_hits, np.int32)
    normal_x_col          = np.empty(total_hits, np.float64)
    normal_y_col          = np.empty(total_hits, np.float64)
    normal_z_col          = np.empty(total_hits, np.float64)
    point_weight_col      = np.empty(total_hits, np.float64)
    viewing_angle_col     = np.empty(total_hits, np.float64)
    hit_type_col          = np.empty(total_hits, np.int32)
    is_leaf_col           = np.empty(total_hits, np.bool_)

    # reusable scratch buffers sized to worst-case candidate list
    hit_idx_tmp = np.empty(max(1, max_cand2), np.int64)
    entry_tmp   = np.empty((max(1, max_cand2), 3), np.float64)
    exit_tmp    = np.empty((max(1, max_cand2), 3), np.float64)

    k = 0

    # ---------- PASS 2: fill outputs using scratch buffers ----------
    for r in range(n_rays):
        _dda_mark_candidates(
            origins[r], directions[r], origin_grid, cell_size,
            bbox_min, bbox_max, keys_ix, keys_iy, keys_iz,
            offsets, voxel_ids_flat, mask, eps
        )  
        
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0:
            continue

        keep = _sphere_cull(origins[r], directions[r], v_centres, radius_sq, cand_idx, eps)
        cand2 = cand_idx[keep]

        # clear only touched bits
        for i in range(cand_idx.size):
            mask[cand_idx[i]] = False

        if cand2.size == 0:
            continue

        hits_k = _slab_fill_candidates(
            origins[r], directions[r], vmins, vmaxs, cand2, eps,
            hit_idx_tmp, entry_tmp, exit_tmp
        )

        # viewing angle: compute once
        dn = math.sqrt(directions[r,0]**2 + directions[r,1]**2 + directions[r,2]**2)
        va = 0.0
        if dn > 0.0:
            cth = directions[r,2] / dn
            if cth < -1.0: cth = -1.0
            if cth >  1.0: cth =  1.0
            ang = math.degrees(math.acos(cth))
            va = ang if ang <= 90.0 else 180.0 - ang

        p = points[r]
        for h in range(hits_k):
            vi = hit_idx_tmp[h]

            # classification (unchanged logic)
            vmin = vmins[vi]; vmax = vmaxs[vi]
            in_voxel = (p[0] >= vmin[0]-eps and p[0] <= vmax[0]+eps and
                        p[1] >= vmin[1]-eps and p[1] <= vmax[1]+eps and
                        p[2] >= vmin[2]-eps and p[2] <= vmax[2]+eps)
            unbound = (np.isnan(p[0]) or np.isnan(p[1]) or np.isnan(p[2]))
            de = (origins[r,0]-entry_tmp[h,0])**2 + (origins[r,1]-entry_tmp[h,1])**2 + (origins[r,2]-entry_tmp[h,2])**2
            dx = (origins[r,0]-exit_tmp[h,0])**2  + (origins[r,1]-exit_tmp[h,1])**2  + (origins[r,2]-exit_tmp[h,2])**2
            dp = (origins[r,0]-p[0])**2 + (origins[r,1]-p[1])**2 + (origins[r,2]-p[2])**2
            before = (de > dp) and (not in_voxel) and (not unbound)
            after  = (dx < dp) and (not in_voxel) and (not unbound)
            hit_type = -1
            if unbound:   hit_type = 0
            elif before:  hit_type = 1
            elif in_voxel:hit_type = 2
            elif after:   hit_type = 3

            # fill columns (same dtypes/order as your schema)
            voxel_size_col[k]     = v_sizes[vi]
            voxel_id_col[k]       = v_ids[vi]
            voxel_cx_col[k]       = v_centres[vi,0]
            voxel_cy_col[k]       = v_centres[vi,1]
            voxel_cz_col[k]       = v_centres[vi,2]
            leg_id_col[k]         = leg_ids[r]
            ray_id_col[k]         = ray_ids[r]
            t_entry_x_col[k]      = entry_tmp[h,0]
            t_entry_y_col[k]      = entry_tmp[h,1]
            t_entry_z_col[k]      = entry_tmp[h,2]
            t_exit_x_col[k]       = exit_tmp[h,0]
            t_exit_y_col[k]       = exit_tmp[h,1]
            t_exit_z_col[k]       = exit_tmp[h,2]
            distance_to_centre[k] = math.sqrt((origins[r,0]-v_centres[vi,0])**2 +
                                              (origins[r,1]-v_centres[vi,1])**2 +
                                              (origins[r,2]-v_centres[vi,2])**2)
            point_x_col[k]        = p[0]
            point_y_col[k]        = p[1]
            point_z_col[k]        = p[2]
            echo_intensity_col[k] = echo_intensity[r]
            return_number_col[k]  = int(return_number[r])      if not np.isnan(return_number[r])      else 0
            number_of_returns_col[k] = int(number_of_returns[r]) if not np.isnan(number_of_returns[r]) else 0
            normal_x_col[k]       = normals[r,0]
            normal_y_col[k]       = normals[r,1]
            normal_z_col[k]       = normals[r,2]
            point_weight_col[k]   = point_weight[r]
            viewing_angle_col[k]  = va
            hit_type_col[k]       = hit_type
            is_leaf_col[k]        = bool(is_leaf[r])
            k += 1

    # final tuple (same shape/order you expect)
    data = (
        voxel_size_col[:k], voxel_id_col[:k], voxel_cx_col[:k], voxel_cy_col[:k], voxel_cz_col[:k],
        leg_id_col[:k], ray_id_col[:k],
        t_entry_x_col[:k], t_entry_y_col[:k], t_entry_z_col[:k],
        t_exit_x_col[:k],  t_exit_y_col[:k],  t_exit_z_col[:k],
        distance_to_centre[:k],
        point_x_col[:k], point_y_col[:k], point_z_col[:k],
        echo_intensity_col[:k], return_number_col[:k], number_of_returns_col[:k],
        normal_x_col[:k], normal_y_col[:k], normal_z_col[:k],
        point_weight_col[:k], viewing_angle_col[:k], hit_type_col[:k], is_leaf_col[:k]
    )
    return data, k


def _map_partition_numba(ray_part: pd.DataFrame, voxel_data: Dict[str, np.ndarray], eps: float) -> pd.DataFrame:
    if ray_part is None or len(ray_part) == 0:
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

    # Extract numeric arrays (contiguous)
    origins        = ray_part[['origin_x','origin_y','origin_z']].to_numpy(dtype=np.float64)
    directions     = ray_part[['direction_x','direction_y','direction_z']].to_numpy(dtype=np.float64)
    points         = ray_part[['point_x','point_y','point_z']].to_numpy(dtype=np.float64)
    normals        = ray_part[['normal_x','normal_y','normal_z']].to_numpy(dtype=np.float64)
    echo_intensity = ray_part['echo_intensity'].to_numpy(dtype=np.float64)
    return_number  = ray_part['return_number'].to_numpy(dtype=np.float64)
    number_of_returns = ray_part['number_of_returns'].to_numpy(dtype=np.float64)
    point_weight   = ray_part['point_weight'].to_numpy(dtype=np.float64)
    is_leaf        = ray_part['is_leaf'].to_numpy(dtype=np.bool_)
    leg_ids        = ray_part['leg_id'].to_numpy(dtype=np.int64)
    ray_ids        = ray_part['ray_id'].to_numpy(dtype=np.int64)

    # Voxel arrays
    v_ids     = voxel_data['ids'].astype(np.int64)
    v_sizes   = voxel_data['sizes'].astype(np.float64)
    v_centres = voxel_data['centres'].astype(np.float64)
    vmins     = voxel_data['vmins'].astype(np.float64)
    vmaxs     = voxel_data['vmaxs'].astype(np.float64)
    origin_grid = voxel_data['origin'].astype(np.float64)
    cell_size   = float(voxel_data['cell_size'])
    bbox_min    = voxel_data['bbox_min'].astype(np.float64)
    bbox_max    = voxel_data['bbox_max'].astype(np.float64)
    keys_ix     = voxel_data['keys_ix'].astype(np.int32)
    keys_iy     = voxel_data['keys_iy'].astype(np.int32)
    keys_iz     = voxel_data['keys_iz'].astype(np.int32)
    offsets     = voxel_data['offsets'].astype(np.int64)
    voxel_ids_flat = voxel_data['voxel_ids_flat'].astype(np.int64)

    # Call Numba kernel (all numeric work inside)
    data, k = _process_partition_numba(
        origins, directions, points, normals,
        echo_intensity, return_number, number_of_returns, point_weight, is_leaf,
        leg_ids, ray_ids,
        v_ids, v_sizes, v_centres, vmins, vmaxs,
        origin_grid, cell_size, bbox_min, bbox_max,
        keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
        np.float64(eps)
    )

    if k == 0:
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

    # Build DataFrame in Python
    cols = voxel_ray_intersection_schema.names
    df = pd.DataFrame({
        cols[0]:  data[0],   # voxel_size
        cols[1]:  data[1],   # voxel_id
        cols[2]:  data[2],   # voxel_cx
        cols[3]:  data[3],   # voxel_cy
        cols[4]:  data[4],   # voxel_cz
        cols[5]:  data[5],   # leg_id
        cols[6]:  data[6],   # ray_id
        cols[7]:  data[7],   # t_entry_x
        cols[8]:  data[8],   # t_entry_y
        cols[9]:  data[9],   # t_entry_z
        cols[10]: data[10],  # t_exit_x
        cols[11]: data[11],  # t_exit_y
        cols[12]: data[12],  # t_exit_z
        cols[13]: data[13],  # distance_to_centre
        cols[14]: data[14],  # point_x
        cols[15]: data[15],  # point_y
        cols[16]: data[16],  # point_z
        cols[17]: data[17],  # echo_intensity
        cols[18]: data[18],  # return_number
        cols[19]: data[19],  # number_of_returns
        cols[20]: data[20],  # normal_x
        cols[21]: data[21],  # normal_y
        cols[22]: data[22],  # normal_z
        cols[23]: data[23],  # point_weight
        cols[24]: data[24],  # viewing_angle
        cols[25]: data[25],  # hit_type
        cols[26]: data[26],  # is_leaf
    })
    return df


@njit(cache=True, fastmath=False)
def _process_partition_pairs(
    origins, directions,
    leg_ids, ray_ids,
    v_ids, v_sizes, v_centres, vmins, vmaxs,
    origin_grid, cell_size, bbox_min, bbox_max,
    keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
    eps
):
    n_rays = origins.shape[0]
    radius_sq = ((v_sizes * math.sqrt(3) * 0.5) + 0.05) ** 2

    # Reusable mask
    mask = np.zeros(v_sizes.shape[0], np.bool_)

    # ---- PASS 1: count hits ----
    total_hits = 0
    for r in range(n_rays):
        _dda_mark_candidates(origins[r], directions[r], origin_grid, cell_size,
                             bbox_min, bbox_max, keys_ix, keys_iy, keys_iz,
                             offsets, voxel_ids_flat, mask, eps)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0: continue
        keep = _sphere_cull(origins[r], directions[r], v_centres, radius_sq, cand_idx, eps)
        cand2 = cand_idx[keep]
        for i in range(cand_idx.size):
            mask[cand_idx[i]] = False
        if cand2.size == 0: continue

        # slab test (count only)
        total_hits += _slab_count_only(origins[r], directions[r], vmins, vmaxs, cand2, eps)

    # ---- allocate compact outputs ----
    out_leg   = np.empty(total_hits, np.int64)
    out_ray   = np.empty(total_hits, np.int64)
    out_vox   = np.empty(total_hits, np.int64)
    out_vsize = np.empty(total_hits, np.float64)

    # ---- PASS 2: fill compact outputs ----
    k = 0
    # small scratch buffers
    hit_idx_tmp = np.empty(1024, np.int64)  # grows if needed
    entry_tmp   = np.empty((1024, 3), np.float64)
    exit_tmp    = np.empty((1024, 3), np.float64)

    for r in range(n_rays):
        _dda_mark_candidates(origins[r], directions[r], origin_grid, cell_size,
                             bbox_min, bbox_max, keys_ix, keys_iy, keys_iz,
                             offsets, voxel_ids_flat, mask, eps)
        cand_idx = np.nonzero(mask)[0]
        if cand_idx.size == 0: continue
        keep = _sphere_cull(origins[r], directions[r], v_centres, radius_sq, cand_idx, eps)
        cand2 = cand_idx[keep]
        for i in range(cand_idx.size):
            mask[cand_idx[i]] = False
        if cand2.size == 0: continue

        # ensure scratch capacity
        if cand2.size > hit_idx_tmp.shape[0]:
            hit_idx_tmp = np.empty(cand2.size, np.int64)
            entry_tmp   = np.empty((cand2.size, 3), np.float64)
            exit_tmp    = np.empty((cand2.size, 3), np.float64)

        hits_k = _slab_fill_candidates(
            origins[r], directions[r], vmins, vmaxs, cand2, eps,
            hit_idx_tmp, entry_tmp, exit_tmp
        )

        # write only IDs/sizes
        for h in range(hits_k):
            vi = hit_idx_tmp[h]
            out_leg[k]   = leg_ids[r]
            out_ray[k]   = ray_ids[r]
            out_vox[k]   = v_ids[vi]
            out_vsize[k] = v_sizes[vi]
            k += 1

    return (out_leg[:k], out_ray[:k], out_vox[:k], out_vsize[:k]), k



def _map_partition_pairs(ray_part: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    if ray_part is None or len(ray_part) == 0:
        return pd.DataFrame(columns=['leg_id', 'ray_id', 'voxel_id', 'voxel_size'])
    
    c = get_client()
    vox_future = c.get_dataset('voxel_data')
    voxel_data = c.gather(vox_future)

    # Extract only what kernel needs
    origins    = ray_part[['origin_x','origin_y','origin_z']].to_numpy(dtype=np.float64)
    directions = ray_part[['direction_x','direction_y','direction_z']].to_numpy(dtype=np.float64)
    leg_ids    = ray_part['leg_id'].to_numpy(dtype=np.int64)
    ray_ids    = ray_part['ray_id'].to_numpy(dtype=np.int64)

    v_ids     = voxel_data['ids'].astype(np.int64)
    v_sizes   = voxel_data['sizes'].astype(np.float64)
    v_centres = voxel_data['centres'].astype(np.float64)
    vmins     = voxel_data['vmins'].astype(np.float64)
    vmaxs     = voxel_data['vmaxs'].astype(np.float64)
    origin_grid = voxel_data['origin'].astype(np.float64)
    cell_size   = float(voxel_data['cell_size'])
    bbox_min    = voxel_data['bbox_min'].astype(np.float64)
    bbox_max    = voxel_data['bbox_max'].astype(np.float64)
    keys_ix     = voxel_data['keys_ix'].astype(np.int32)
    keys_iy     = voxel_data['keys_iy'].astype(np.int32)
    keys_iz     = voxel_data['keys_iz'].astype(np.int32)
    offsets     = voxel_data['offsets'].astype(np.int64)
    voxel_ids_flat = voxel_data['voxel_ids_flat'].astype(np.int64)

    data, k = _process_partition_pairs(
        origins, directions, leg_ids, ray_ids,
        v_ids, v_sizes, v_centres, vmins, vmaxs,
        origin_grid, cell_size, bbox_min, bbox_max,
        keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
        np.float64(eps)
    )

    if k == 0:
        return pd.DataFrame(columns=['leg_id', 'ray_id', 'voxel_id', 'voxel_size'])

    return pd.DataFrame({
        'leg_id':     data[0],
        'ray_id':     data[1],
        'voxel_id':   data[2],
        'voxel_size': data[3],
    })


def voxel_ray_intersections_dask_new(valid_rays_dir: str,
                                 references_dir: str,
                                 voxel_chunk_size: int = 10000,
                                 temp_dir: str | None = None,
                                 cpus: int | None = None,
                                 mem: int | None = None,
                                 debug: bool = True,
                                 epsilon: float = 1e-6) -> None:
    memory_limit_str, n_workers, threads_per_worker = _determine_dask_resources(
        cpus=cpus, mem=mem, optimal_threads=2
    )
    client = _start_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_str,
        memory_target_fraction=0.7,
        memory_pause_fraction=0.9,
        memory_spill_fraction=0.95,
        temp_dir=_resolve_temp_dir(temp_dir),
        processes=True
    )
    if debug:
        print(f"[voxel_ray_intersections_dask] Dask client: "
              f"workers={n_workers}, threads/worker={threads_per_worker}, mem/worker={memory_limit_str}")

    dask.config.set({"dataframe.shuffle.method": "p2p"})

    start_time = time.time()

    voxel_refs = _compile_voxel_references(references_dir)
    voxel_data = _build_sparse_grid_arrays(voxel_refs, epsilon=epsilon)
    
    
    
    # If voxel_data is big, publish a Future handle rather than the raw object
    vox_future = client.submit(lambda x: x, voxel_data)  # or client.scatter(voxel_data)
    client.replicate([vox_future])                       # (optional) copy to all workers
    client.publish_dataset(voxel_data=vox_future)        # named handle on the scheduler


    valid_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))
    if debug:
        print(f"Found {len(valid_files)} valid rays files.")

    def leg_from_filename(path: str) -> int:
        base = os.path.basename(path)
        parts = os.path.splitext(base)[0].split('_')
        try:
            return int(parts[1])
        except Exception:
            return next((int(p) for p in parts if p.isdigit()), 0)

    ddfs = []
    for file in valid_files:
        leg_id = leg_from_filename(file)
        ddf = dd.read_parquet(file, engine='pyarrow', split_row_groups=True)
        ddfs.append(ddf.assign(leg_id=leg_id))
    
    all_ddf = dd.concat(ddfs, interleave_partitions=True)

    meta_pairs = {
        'leg_id': str(all_ddf['leg_id'].dtype),
        'ray_id': str(all_ddf['ray_id'].dtype),
        'voxel_id': str(voxel_data['ids'].dtype),
        'voxel_size': str(voxel_data['sizes'].dtype)
    }
    pairs_ddf = all_ddf.map_partitions(
        _map_partition_pairs, meta=meta_pairs
    )

    # 2) Prepare rays DDF (select only needed columns)
    rays_ddf = all_ddf[['leg_id','ray_id',
                        'origin_x','origin_y','origin_z',
                        'direction_x','direction_y','direction_z',
                        'point_x','point_y','point_z',
                        'normal_x','normal_y','normal_z',
                        'echo_intensity','return_number','number_of_returns',
                        'point_weight','is_leaf']]

    # 3) Prepare voxel refs DDF
    voxel_df = voxel_refs[['voxel_id','voxel_cx','voxel_cy','voxel_cz','voxel_size']]
    voxels_ddf = dd.from_pandas(voxel_df, npartitions=256)

    # 4) Join pairs → rays → voxels
    pairs_rays = pairs_ddf.merge(
        rays_ddf,
        on=['leg_id','ray_id'],
        how='left',
        shuffle_method='p2p'
    )
    joined = pairs_rays.merge(
        voxels_ddf, 
        on=['voxel_id','voxel_size'], 
        how='left',
        shuffle_method='p2p'
    )

    del pairs_ddf
    del rays_ddf
    del voxels_ddf
    del pairs_rays
    client.run(lambda: __import__("gc").collect())

    # 5) Vectorized slab entry/exit using Dask DataFrame operations
    eps = 1e-6
    
    # Compute vmins/vmaxs from centres and voxel_size
    half = joined['voxel_size'] / 2.0
    joined['vmin_x'] = joined['voxel_cx'] - half
    joined['vmin_y'] = joined['voxel_cy'] - half
    joined['vmin_z'] = joined['voxel_cz'] - half
    joined['vmax_x'] = joined['voxel_cx'] + half
    joined['vmax_y'] = joined['voxel_cy'] + half
    joined['vmax_z'] = joined['voxel_cz'] + half
    
    # Safe inverse of direction components (avoid division by zero)
    joined['invx'] = 1.0 / joined['direction_x'].where(joined['direction_x'].abs() > eps, eps)
    joined['invy'] = 1.0 / joined['direction_y'].where(joined['direction_y'].abs() > eps, eps)
    joined['invz'] = 1.0 / joined['direction_z'].where(joined['direction_z'].abs() > eps, eps)
    
    # Slab intersection calculations (t1, t2 per axis)
    joined['t1x'] = (joined['vmin_x'] - joined['origin_x']) * joined['invx']
    joined['t2x'] = (joined['vmax_x'] - joined['origin_x']) * joined['invx']
    joined['t1y'] = (joined['vmin_y'] - joined['origin_y']) * joined['invy']
    joined['t2y'] = (joined['vmax_y'] - joined['origin_y']) * joined['invy']
    joined['t1z'] = (joined['vmin_z'] - joined['origin_z']) * joined['invz']
    joined['t2z'] = (joined['vmax_z'] - joined['origin_z']) * joined['invz']
    
    # Min/max per axis
    joined['tminx'] = joined[['t1x', 't2x']].min(axis=1)
    joined['tmaxx'] = joined[['t1x', 't2x']].max(axis=1)
    joined['tminy'] = joined[['t1y', 't2y']].min(axis=1)
    joined['tmaxy'] = joined[['t1y', 't2y']].max(axis=1)
    joined['tminz'] = joined[['t1z', 't2z']].min(axis=1)
    joined['tmaxz'] = joined[['t1z', 't2z']].max(axis=1)
    
    # Entry/exit t parameters
    joined['t_enter'] = joined[['tminx', 'tminy', 'tminz']].max(axis=1)
    joined['t_exit'] = joined[['tmaxx', 'tmaxy', 'tmaxz']].min(axis=1)
    
    # 6) Entry/exit coordinates
    entry_x = joined['origin_x'] + joined['t_enter'] * joined['direction_x']
    entry_y = joined['origin_y'] + joined['t_enter'] * joined['direction_y']
    entry_z = joined['origin_z'] + joined['t_enter'] * joined['direction_z']
    exit_x = joined['origin_x'] + joined['t_exit'] * joined['direction_x']
    exit_y = joined['origin_y'] + joined['t_exit'] * joined['direction_y']
    exit_z = joined['origin_z'] + joined['t_exit'] * joined['direction_z']
    
    # 7) Viewing angle (zenith normalized to ≤ 90°)
    dn = (joined['direction_x']**2 + joined['direction_y']**2 + joined['direction_z']**2).map_partitions(np.sqrt)
    cth = (joined['direction_z'] / dn.map_partitions(lambda x: x.clip(lower=eps))).map_partitions(lambda x: x.clip(-1.0, 1.0))
    ang = cth.map_partitions(np.arccos) * (180.0 / np.pi)
    viewing_angle = ang.map_partitions(lambda x: x.where(x <= 90.0, 180.0 - x))
    
    # 8) Distance to centre
    dist_c = ((joined['origin_x'] - joined['voxel_cx'])**2 + 
              (joined['origin_y'] - joined['voxel_cy'])**2 + 
              (joined['origin_z'] - joined['voxel_cz'])**2).map_partitions(np.sqrt)
    
    # 9) Classification
    in_voxel = ((joined['point_x'] >= (joined['vmin_x'] - eps)) & 
                (joined['point_x'] <= (joined['vmax_x'] + eps)) &
                (joined['point_y'] >= (joined['vmin_y'] - eps)) & 
                (joined['point_y'] <= (joined['vmax_y'] + eps)) &
                (joined['point_z'] >= (joined['vmin_z'] - eps)) & 
                (joined['point_z'] <= (joined['vmax_z'] + eps)))
    unbound = (joined['point_x'].isna() | joined['point_y'].isna() | joined['point_z'].isna())
    
    de = ((joined['origin_x'] - entry_x)**2 + 
          (joined['origin_y'] - entry_y)**2 + 
          (joined['origin_z'] - entry_z)**2)
    dx_ = ((joined['origin_x'] - exit_x)**2 + 
           (joined['origin_y'] - exit_y)**2 + 
           (joined['origin_z'] - exit_z)**2)
    dp = ((joined['origin_x'] - joined['point_x'])**2 + 
          (joined['origin_y'] - joined['point_y'])**2 + 
          (joined['origin_z'] - joined['point_z'])**2)
    
    before = (de > dp) & (~in_voxel) & (~unbound)
    after = (dx_ < dp) & (~in_voxel) & (~unbound)
    
    # Build hit_type using where() for conditional assignment
    hit_type = joined['origin_x'].map_partitions(lambda x: pd.Series(-1, index=x.index, dtype=np.int32))
    hit_type = hit_type.where(~unbound, 0)
    hit_type = hit_type.where(~before, 1)
    hit_type = hit_type.where(~in_voxel, 2)
    hit_type = hit_type.where(~after, 3)

    # 9) Assemble final DDF columns
    result = joined.assign(
        voxel_cx=joined['voxel_cx'], voxel_cy=joined['voxel_cy'], voxel_cz=joined['voxel_cz'],
        t_entry_x=entry_x, t_entry_y=entry_y, t_entry_z=entry_z,
        t_exit_x=exit_x,   t_exit_y=exit_y,   t_exit_z=exit_z,
        distance_to_centre=dist_c,
        viewing_angle=viewing_angle,
        hit_type=hit_type
    )

    # 10) Write (stream; avoid persist)
    result.to_parquet(
        os.path.join(valid_rays_dir, "voxel_ray_intersections"),
        engine='pyarrow', compression='snappy',
        write_index=False,
        partition_on=['leg_id','voxel_size'],
        schema=voxel_ray_intersection_schema,
        overwrite=True,
        write_metadata_file=True
    )

    ### OLD ###
    # # Build a single DDF with leg_id attached
    # ddfs = []
    # for file in valid_files:
    #     leg_id = leg_from_filename(file)
    #     ddf = dd.read_parquet(file, engine='pyarrow', split_row_groups=True)
    #     ddfs.append(ddf.assign(leg_id=leg_id))
    # all_ddf = dd.concat(ddfs, interleave_partitions=True)

    # # find published voxel chunk datasets
    # dataset_names = [name for name in client.list_datasets() if name.startswith("voxel_data_chunk_")]
    # if len(dataset_names) == 0:
    #     # fallback to single dataset name if present
    #     if "voxel_data" in client.list_datasets():
    #         dataset_names = ["voxel_data"]

    # mapped_parts = []

    # def _map_with_vox(ray_part: pd.DataFrame, vox) -> pd.DataFrame:
    #     vdata = vox.result() if hasattr(vox, "result") else vox
    #     return _map_partition_numba(ray_part, vdata, eps=epsilon)

    # for ds_name in dataset_names:
    #     vox = client.get_dataset(ds_name)
    #     mapped_parts.append(all_ddf.map_partitions(_map_with_vox, vox, meta=meta))

    # # concat mapped results across voxel chunks so each partition is scheduled per voxel chunk
    # mapped = dd.concat(mapped_parts, interleave_partitions=True)

    # persisted = mapped.persist()
    # wait(persisted)

    # # Write one dataset partitioned by leg_id and voxel_size_rounded
    # persisted.to_parquet(
    #     os.path.join(valid_rays_dir, "intersections"),
    #     engine='pyarrow',
    #     compression='snappy',
    #     write_index=False,
    #     partition_on=['leg_id', 'voxel_size'],
    #     schema=voxel_ray_intersection_schema,
    #     # overwrite=True,
    # )

    # del persisted
    # client.run(lambda: __import__("gc").collect())

    end_time = time.time()
    print(f"[voxel_ray_intersections_dask] Completed in {end_time - start_time:.2f} s")

    time.sleep(0.5)
    _close_dask_client(client)
    print("[voxel_ray_intersections_dask] Dask client closed.")


def voxel_ray_intersections_dask(valid_rays_dir: str,
                                 references_dir: str,
                                 temp_dir: str | None = None,
                                 cpus: int | None = None,
                                 mem: int | None = None,
                                 debug: bool = True,
                                 epsilon: float = 1e-6) -> None:
    # Configure dask settings
    memory_limit_str, n_workers, threads_per_worker = _determine_dask_resources(cpus=cpus, mem=mem, optimal_threads=2)

    # set_num_threads(max(1, threads_per_worker))
    client = _start_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_str,
        memory_target_fraction=0.7,
        memory_pause_fraction=0.9,
        memory_spill_fraction=0.95,
        temp_dir=_resolve_temp_dir(temp_dir),
        processes=True
    )
    if debug:
        print(f"[voxel_ray_intersections_dask] Dask client: workers={n_workers}, threads/worker={threads_per_worker}, mem/worker={memory_limit_str}")

    # Build voxel references and sparse grid arrays once
    voxel_refs = _compile_voxel_references(references_dir)
    voxel_data = _build_sparse_grid_arrays(voxel_refs, epsilon=epsilon)
    if debug:
        print(f"Sparse grid: {voxel_data['keys_ix'].shape[0]} occupied cells; cell_size={voxel_data['cell_size']:.6f}")

    # Scatter voxel_data (broadcast) to workers
    vox_fut = client.scatter(voxel_data, broadcast=True)

    valid_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))
    if debug:
        print(f"Found {len(valid_files)} valid rays files.")

    meta = pd.DataFrame(columns=voxel_ray_intersection_schema.names)
    futures_dict = {}

    for file in valid_files:
        base = os.path.basename(file)
        parts = os.path.splitext(base)[0].split('_')
        try:
            leg_id = int(parts[1])
        except Exception:
            leg_id = next((int(p) for p in parts if p.isdigit()), 0)
        ddf = dd.read_parquet(file, engine='pyarrow', split_row_groups=True) # blocksize='32MB')

        def _map(ray_part: pd.DataFrame, vox) -> pd.DataFrame:
            vdata = vox.result() if hasattr(vox, 'result') else vox
            return _map_partition_numba(ray_part, vdata, eps=epsilon)

        result_ddf = ddf.map_partitions(_map, vox=voxel_data, meta=meta)
        fut = client.compute(result_ddf, sync=False)
        futures_dict[fut] = leg_id
        if debug:
            print(f"Leg {leg_id}: submitted with {ddf.npartitions} partitions")

    start_time = time.time()
    for fut in as_completed(futures_dict):
        leg_id = futures_dict[fut]
        result_df = fut.result()
        if result_df is None or len(result_df) == 0:
            if debug:
                print(f"Leg {leg_id}: no intersections")
            continue
        grouped = result_df.groupby('voxel_size', group_keys=True)
        for vox_size, grp in grouped:
            out_path = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{round(float(vox_size),2)}_intersections.parquet")
            grp.to_parquet(out_path, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
            del grp
        del result_df
        if debug:
            print(f"Leg {leg_id}: saved groups")

    end_time = time.time()
    if debug:
        print(f"[voxel_ray_intersections_dask] Completed in {end_time - start_time:.2f} s")
    time.sleep(0.5)
    _close_dask_client(client)
    if debug:
        print("[voxel_ray_intersections_dask] Dask client closed.")






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
            yet_first_hit = yet_to_hit & first_hit
            Tk_first_lw = np.nansum(yet_first_hit)
            BWk_first = 1

            current_first_hit_leaf = current_hit & leaf_mask
            Tk_first_leaf = np.nansum(current_first_hit_leaf)

            # -- Equal Hit Weighting -- #
            echoes_before_lw = np.count_nonzero(previous_hit)
            echoes_during_lw = np.count_nonzero(current_hit)
            echoes_after_lw = np.count_nonzero(yet_to_hit)
            Tk_equal_lw = echoes_after_lw / np.clip((echoes_during_lw + echoes_after_lw), 1, None)
            BWk_equal_lw = (echoes_during_lw + echoes_after_lw) / np.clip((echoes_before_lw + echoes_during_lw + echoes_after_lw), 1, None)

            echoes_before_leaf = np.count_nonzero(previous_hit & leaf_mask)
            echoes_during_leaf = np.count_nonzero(current_hit & leaf_mask)
            echoes_after_leaf = np.count_nonzero(yet_to_hit & leaf_mask)
            Tk_equal_leaf = echoes_after_leaf / np.clip((echoes_during_leaf + echoes_after_leaf), 1, None)
            BWk_equal_leaf = (echoes_during_leaf + echoes_after_leaf) / np.clip((echoes_before_leaf + echoes_during_leaf + echoes_after_leaf), 1, None)

            # -- Intensity Hit Weighting -- #
            echo_intensities = voxel_df['echo_intensity'].values
            intensity_before_lw = np.nansum(echo_intensities[previous_hit])
            intensity_during_lw = np.nansum(echo_intensities[current_hit])
            intensity_after_lw = np.nansum(echo_intensities[yet_to_hit])

            denom_lw = intensity_during_lw + intensity_after_lw
            Tk_int_lw = intensity_after_lw / denom_lw if denom_lw != 0 else np.nan
            BWk_int_lw = (intensity_during_lw + intensity_after_lw) / (intensity_before_lw + intensity_during_lw + intensity_after_lw)

            intensity_before_leaf = np.nansum(echo_intensities[previous_hit & leaf_mask])
            intensity_during_leaf = np.nansum(echo_intensities[current_hit & leaf_mask])
            intensity_after_leaf = np.nansum(echo_intensities[yet_to_hit & leaf_mask])
            del echo_intensities
            denom_leaf = intensity_during_leaf + intensity_after_leaf
            Tk_int_leaf = intensity_after_leaf / denom_leaf if denom_leaf != 0 else np.nan
            BWk_int_leaf = (intensity_during_leaf + intensity_after_leaf) / (intensity_before_leaf + intensity_during_leaf + intensity_after_leaf)

            P_first_lw, P_equal_lw, P_int_lw, P_first_leaf, P_equal_leaf, P_int_leaf = (
                _collapse(*args) for args in [
                    (Tk_first_lw, BWk_first), 
                    (Tk_equal_lw, BWk_equal_lw),
                    (Tk_int_lw,   BWk_int_lw),
                    (Tk_first_leaf, BWk_first),
                    (Tk_equal_leaf, BWk_equal_leaf),
                    (Tk_int_leaf,   BWk_int_leaf)
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
    # Ensure 'voxel_id' remains a column and group by it directly to compute per-voxel metrics
    # (don't set 'voxel_id' as the DataFrame index, since subsequent grouping expects it as a column)
    # Map to per-voxel calculation via dask groupby.apply with the provided meta
    voxel_metrics_df = voxel_intersections_df.groupby('voxel_id').apply(
        calculate_voxel_metrics_per_voxel,
        meta=meta,
        include_groups=False
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