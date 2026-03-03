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
from numba import njit, prange


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
    leg_{scan_id}_voxel_{voxel_size}_ray_intersections.parquet

And contains the information outlined in the following schema.
Each index corresponds to a ray that intersects a voxel.
"""
voxel_ray_intersection_schema = pa.schema([
    pa.field('voxel_size', pa.float32()),
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_cx', pa.float64()),
    pa.field('voxel_cy', pa.float64()),
    pa.field('voxel_cz', pa.float64()),
    pa.field('scan_id', pa.uint64()),
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
    pa.field('viewing_angle', pa.float64()),
    pa.field('hit_type', pa.int32()),
    pa.field('is_leaf', pa.bool_())
])

# Voxel Metrics Schema
"""
This schema is used to store the metrics for each voxel, based on the selected legs and voxel size.
Since this one is only used to store to a csv file (for final output), it is not as important to be efficient.


"""
voxel_metrics_schema_singlereturn = pa.schema([
    pa.field('voxel_id', pa.uint64()),
    pa.field('voxel_cx', pa.float64()),
    pa.field('voxel_cy', pa.float64()),
    pa.field('voxel_cz', pa.float64()),
    pa.field('voxel_size', pa.float32()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('pgap_wood', pa.float64()),
    pa.field('I_lw', pa.float64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.float64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('I_wood', pa.float64()),  # num_wood_hits / num_rays (i.e. wood only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('G_wood', pa.float64()),               # G function calculated from wood hits only
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
    pa.field('voxel_size', pa.float32()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('pgap_wood', pa.float64()),
    pa.field('I_lw', pa.float64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.float64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('I_wood', pa.float64()),  # num_wood_hits / num_rays (i.e. wood only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('G_wood', pa.float64()),               # G function calculated from wood hits only
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
    pa.field('scan_id', pa.uint64()),
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

# ---- normals_weights.py (can live alongside your metrics code) ----
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from numba import njit

def compute_normals_weights_from_points_parallel(
    points: np.ndarray,
    *,
    voxel_size: float = 20.0,
    knn: int = 6,
    n_jobs: int = -1,
    eps: float = 1e-9
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parallel, memory-friendly version of your plane-fitting step.
      - Bins points by coarse 'normal-voxel' of size `voxel_size`
      - In each bin: build cKDTree, KNN, Numba PCA normals, weights = 1/(kth_distance+eps)
      - Parallelizes over bins (joblib); inside each bin cKDTree runs single-threaded to avoid oversubscription

    points: (N,3) float64; returns (normals(N,3), weights(N,))
    """
    points = np.asarray(points, dtype=np.float64)
    N = len(points)
    if N == 0:
        print("[compute_normals_weights] Empty input; returning empty arrays")
        return np.zeros((0,3), dtype=np.float64), np.zeros((0,), dtype=np.float64)
    if N < knn:
        print(f"[compute_normals_weights] Only {N} points (< knn={knn}); returning default normals/weights")
        return np.zeros((N,3), dtype=np.float64), np.ones((N,), dtype=np.float64)

    print(f"[compute_normals_weights] Processing {N:,} points with voxel_size={voxel_size}, knn={knn}")

    # Grid keys
    vox = np.floor(points / voxel_size).astype(np.int64)
    keys = (vox[:,0] * 73856093) ^ (vox[:,1] * 19349663) ^ (vox[:,2] * 83492791)  # simple hash
    order = np.argsort(keys, kind="stable")
    keys_sorted = keys[order]
    splits = np.flatnonzero(np.diff(keys_sorted)) + 1
    starts = np.r_[0, splits]; ends = np.r_[splits, N]
    
    num_bins = len(starts)
    print(f"  ✓ Partitioned into {num_bins} spatial bins")

    # Output buffers
    normals = np.zeros((N,3), dtype=np.float64)
    weights = np.ones((N,), dtype=np.float64)

    def _process_bin(s: int, e: int):
        idx = order[s:e]
        pts = points[idx]
        if len(pts) < knn:
            # leave zeros/ones defaults
            return idx, np.zeros((len(pts),3), dtype=np.float64), np.ones((len(pts),), dtype=np.float64)
        tree = cKDTree(pts)
        k = min(knn, len(pts))
        dists, nb = tree.query(pts, k=k, workers=1)
        # nb is (M, k) indices *within* pts; compute normals in this local frame
        nb = nb.astype(np.int64, copy=False)
        loc_normals = _compute_normals_vectorized(pts, nb)
        # simple inverse-of-farthest-distance weight
        w = 1.0 / (dists[:, -1] + eps)
        return idx, loc_normals, w

    # Run bins in parallel with progress bar
    if n_jobs == -1:
        n_jobs = max(1, num_bins // 4)  # Use 1/4 of bins per job for better parallelization
    print(f"  Computing normals & weights (n_jobs={n_jobs}):")
    jobs = [delayed(_process_bin)(s, e) for s, e in zip(starts, ends)]
    chunks = Parallel(n_jobs=n_jobs, prefer="processes", batch_size="auto", verbose=0, env_var='LOKY_DISABLE_RESOURCE_TRACKER=1')(
        tqdm(jobs, total=num_bins, desc="    Bins", unit=" bin", ncols=80, leave=True)
    )
    
    print(f"  ✓ Computed {len(chunks)} bins; assembling output...")
    for idx, nrm, w in chunks:
        normals[idx] = nrm
        weights[idx] = w

    print(f"  ✓ Normals & weights complete: {N:,} points processed")
    return normals, weights

def compute_normals_weights_from_points(points, voxel_size=20.0, knn=6):
    """
    Get normals and weights from points, optimized for Dask map_partitions.
    Uses numba JIT compilation and vectorized operations.
    
    INPUTS:
        points: A numpy array of points (N, 3)
        voxel_size: Size of voxels for spatial binning
        knn: The number of nearest neighbours to consider

    OUTPUTS:
        normals: The normals of the points (N, 3)
        weights: The weights of the points (N,)
    """
    import numpy as np
    from scipy.spatial import cKDTree
    
    
    points = np.asarray(points, dtype=np.float64)
    
    if len(points) < knn:
        return np.zeros((len(points), 3), dtype=np.float64), np.ones(len(points), dtype=np.float64)

    # Fast voxel-based spatial partitioning (no parallelization overhead)
    voxel_indices = (points / voxel_size).astype(np.int32)
    voxel_keys = (
        voxel_indices[:, 0].astype(np.int64) * 1000000 +
        voxel_indices[:, 1].astype(np.int64) * 1000 +
        voxel_indices[:, 2].astype(np.int64)
    )
    
    # Group by voxel using argsort (faster than dict for large arrays)
    sort_idx = np.argsort(voxel_keys)
    sorted_keys = voxel_keys[sort_idx]
    split_indices = np.where(np.diff(sorted_keys) != 0)[0] + 1
    
    normals = np.zeros((len(points), 3), dtype=np.float64)
    weights = np.ones(len(points), dtype=np.float64)
    
    # Process each voxel independently
    voxel_starts = np.concatenate(([0], split_indices))
    voxel_ends = np.concatenate((split_indices, [len(points)]))
    
    for start, end in zip(voxel_starts, voxel_ends):
        voxel_point_indices = sort_idx[start:end]
        voxel_points = points[voxel_point_indices]
        
        # Skip voxels with too few points
        if len(voxel_points) < knn:
            continue
        
        # Use scipy cKDTree for fast KNN within voxel
        tree = cKDTree(voxel_points)
        distances, neighbor_indices = tree.query(
            voxel_points, 
            k=min(knn, len(voxel_points)),
            workers=1  # Single-threaded within partition
        )
        
        # Compute PCA normals for each point using neighbors
        voxel_normals = _compute_normals_vectorized(voxel_points, neighbor_indices)
        voxel_weights = 1.0 / (distances[:, -1] + 1e-9)
        
        normals[voxel_point_indices] = voxel_normals
        weights[voxel_point_indices] = voxel_weights
    
    return normals, weights


@njit(parallel=False)
def _compute_normals_vectorized(points, neighbor_indices):
    """
    Compute normals using PCA on neighboring points.
    Numba JIT compiled for speed.
    
    INPUTS:
        points: Points array (N, 3)
        neighbor_indices: KNN neighbor indices (N, K)
    
    OUTPUTS:
        normals: Unit normal vectors (N, 3)
    """
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float64)
    
    for i in range(n_points):
        # Get neighbor points
        neighbor_pts = points[neighbor_indices[i]]
        
        # Compute centroid manually (Numba doesn't support np.mean with axis)
        centroid = np.zeros(3)
        for j in range(neighbor_pts.shape[0]):
            for k in range(3):
                centroid[k] += neighbor_pts[j, k]
        for k in range(3):
            centroid[k] /= neighbor_pts.shape[0]
        
        # Center points
        centered = neighbor_pts - centroid
        
        # Compute covariance matrix (3x3)
        cov = np.zeros((3, 3))
        for j in range(centered.shape[0]):
            for a in range(3):
                for b in range(3):
                    cov[a, b] += centered[j, a] * centered[j, b]
        cov /= centered.shape[0]
        
        # Eigenvalue decomposition via power iteration (fast for 3x3)
        # The normal is the eigenvector with smallest eigenvalue
        normal = _compute_smallest_eigenvector_3x3(cov)
        
        # Explicitly assign each component to avoid broadcasting issues
        for k in range(3):
            normals[i, k] = normal[k]
    
    return normals


@njit
def _compute_smallest_eigenvector_3x3(cov):
    """
    Compute the eigenvector of the smallest eigenvalue for 3x3 matrix.
    Uses power iteration for speed.
    """
    # Start with random vector
    v = np.array([1.0, 0.0, 0.0])
    
    # Power iteration to find largest eigenvector of (I - cov) = smallest of cov
    # Invert via: find largest eigenvalue of -cov
    for _ in range(10):  # Usually converges in 2-3 iterations
        v_new = np.zeros(3)
        for i in range(3):
            for j in range(3):
                v_new[i] -= cov[i, j] * v[j]  # Negate to find smallest
        
        # Normalize
        norm = np.sqrt(v_new[0]**2 + v_new[1]**2 + v_new[2]**2)
        if norm > 1e-10:
            v_new /= norm
        else:
            break
        
        v = v_new
    
    # Normalize final result
    norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm > 1e-10:
        v /= norm
    
    return v

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
    lambda_1 = float(average_leaf_area) / (float(voxel_size) ** 3)

    return lambda_1

def calculate_lambda_1_vec(average_leaf_areas, voxel_sizes):
    """
    Calculate lambda_1 for a given voxel size, vectorized for arrays.
    
    INPUTS:
        average_leaf_areas: A numpy array of average leaf areas or a single float
        voxel_sizes: A numpy array of voxel sizes
    
    OUTPUTS:
        lambda_1: A numpy array of lambda_1 values
    """
    lambda_1 = float(average_leaf_areas) / (voxel_sizes ** 3)
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

def effective_path_length_vec(free_path_lengths, lambda_1):
    """
    Calculate the effective path length z, vectorized for Dask Series or numpy arrays.
    
    INPUTS:
        free_path_lengths: A Dask Series, pandas Series, or numpy array of free path lengths
        lambda_1: A Dask Series, pandas Series, or numpy array of lambda_1 values

    OUTPUTS:
        z: A Dask Series or numpy array of effective path length z values
    """
    import dask.dataframe as dd
    
    # Check if inputs are Dask Series
    is_dask = isinstance(free_path_lengths, dd.Series) or isinstance(lambda_1, dd.Series)
    
    if is_dask:
        # Dask-native computation
        product = lambda_1 * free_path_lengths
        valid_mask = product < 1
        
        # Compute effective path length only for valid cases
        eff_path_length_zs = (-np.log(1 - product) / lambda_1).where(valid_mask, other=np.nan)
    else:
        # Numpy/pandas fallback
        with np.errstate(divide='ignore', invalid='ignore'):
            valid_mask = (lambda_1 * free_path_lengths) < 1
            eff_path_length_zs = np.full_like(free_path_lengths, fill_value=np.nan, dtype=np.float64)
            eff_path_length_zs[valid_mask] = -np.log(1 - lambda_1[valid_mask] * free_path_lengths[valid_mask]) / lambda_1[valid_mask]
    
    return eff_path_length_zs

def calculate_inclination_angle_distribution(normals, weights, num_bins=18):
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

def calculate_inclination_angle_distribution_old(points, knn=6, radius=0.1, max_nn=10, num_bins=18):
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

def calculate_G_vec(viewing_angles, LIAD_values, epsilon=1e-9):
    """
    Calculate the G function mean, vectorized for Dask Series or numpy arrays.
    
    INPUTS:
        viewing_angles: A Dask Series or numpy array of viewing angles
        LIAD_values: A Dask Series or numpy array of LIAD values
    
    OUTPUTS:
        G_mean: A Dask Series or numpy array of G function mean values
    """
    import dask.array as da
    import dask.dataframe as dd
    
    # Check for empty arrays (handle both numpy and dask)
    va_len = 0
    liad_len = 0
    
    try:
        va_len = len(viewing_angles) if viewing_angles is not None else 0
    except (TypeError, AttributeError):
        va_len = viewing_angles.size if hasattr(viewing_angles, 'size') else 0
    
    try:
        liad_len = len(LIAD_values) if LIAD_values is not None else 0
    except (TypeError, AttributeError):
        liad_len = LIAD_values.size if hasattr(LIAD_values, 'size') else 0
    
    if va_len == 0 or liad_len == 0:
        return np.nan
    
    # Normalise LIAD
    total_LIAD = LIAD_values.sum()
    LIAD_norm = LIAD_values / total_LIAD if total_LIAD > 0 else LIAD_values

    # Ensure angles are clipped
    viewing_angles = da.clip(viewing_angles, epsilon, 90)

    ### A(angle, leaf_angle)  ####
    theta_a = da.radians(viewing_angles)
    theta_b = da.radians(np.arange(0, 90, 5))  # Assuming bin centres at 5 degree intervals

    # Calculate the cotangent of the angles
    cos_theta_a = da.cos(theta_a)
    cot_theta_a = 1 / da.tan(theta_a)
    cos_theta_b = da.cos(theta_b)
    cot_theta_b = 1 / da.tan(theta_b)

    cos_outer = da.outer(cos_theta_a, cos_theta_b)
    cot_outer = da.outer(cot_theta_a, cot_theta_b)

    A = da.zeros_like(cos_outer)
    mask_greater_1 = da.abs(cot_outer) > 1

    A[mask_greater_1] = cos_outer[mask_greater_1]

    inside = da.clip(cot_outer[~mask_greater_1], -1, 1)
    psi = da.arccos(inside)
    factor = 1.0 + (2.0 / np.pi) * (da.tan(psi) - psi)

    A[~mask_greater_1] = factor * cos_outer[~mask_greater_1]

    # Calculate the G function mean for all angles
    delta_bin = np.radians(5)  # Assuming bin centres at 5 degree intervals
    G = A @ (LIAD_norm * delta_bin)

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
    scan_ids = ray_partition['scan_id'].values
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

    scan_ids = scan_ids[np.newaxis, :]
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
        del voxel_ids, voxel_sizes, voxel_centres, scan_ids, ray_ids, is_leaf, points, origins, directions
        gc.collect()
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)

    del mask, chunk_masks, chunk
    gc.collect()

    # Flatten all values to match the mask
    filtered_voxel_ids = voxel_ids[voxel_ref_idx].reshape(-1)
    filtered_voxel_sizes = voxel_sizes[voxel_ref_idx]
    filtered_voxel_centres = voxel_centres[voxel_ref_idx].reshape(-1, 3)

    filtered_scan_ids = scan_ids[:, ray_ref_idx].reshape(-1)
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
    del voxel_ids, voxel_sizes, voxel_centres, scan_ids, ray_ids, is_leaf, points, origins, directions, echo_intensities

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
        'scan_id': filtered_scan_ids.flatten(),
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

    del filtered_voxel_sizes, filtered_voxel_ids, filtered_scan_ids, filtered_ray_ids, filtered_entry_coords, filtered_exit_coords, filtered_distances_to_voxel_centre, filtered_points, filtered_viewing_angles, hit_type, filtered_is_leaf
    gc.collect()

    print(f"Process {process_id}. Returning results...")

    return data_df

# Function to traverse the voxels and find ray intersections
def traverse_voxels_oldcode(voxel_references, ray_partition, chunks_per_compute, temp_dir, epsilon=1e-6):

    if ray_partition.empty:
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
    
    # Prep ray information
    scan_ids = ray_partition['scan_id'].values
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

    scan_ids = scan_ids[np.newaxis, :]
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
        del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, scan_ids, ray_ids, is_leaf, points, origins, directions
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

    filtered_scan_ids = scan_ids[:, ray_ref_idx].reshape(-1)
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
    del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, scan_ids, ray_ids, is_leaf, points, origins, directions
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
    filtered_scan_ids = filtered_scan_ids[valid_ray_mask]
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
    filtered_scan_ids = np.nan_to_num(filtered_scan_ids, nan=-1).astype(np.int64)
    filtered_ray_ids = np.nan_to_num(filtered_ray_ids, nan=-1).astype(np.int64)
    filtered_voxel_ids = np.nan_to_num(filtered_voxel_ids, nan=-1).astype(np.int64)

    data = [
        pa.array(filtered_voxel_sizes),
        pa.array(filtered_voxel_ids),
        pa.array(filtered_scan_ids),
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

    del filtered_voxel_sizes, filtered_voxel_ids, filtered_scan_ids, filtered_ray_ids, filtered_t_entry_coords, filtered_t_exit_coords, filtered_t_entry_radii, filtered_t_exit_radii, filtered_points, filtered_viewing_angles, filtered_hit_rays, filtered_is_leaf
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
        elif os.path.splitext(voxel_ref)[0].split("_")[-1].replace('.', '', 1).isdigit():
            voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
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
            rays['scan_id'] = leg
            hit_object_key = 'hit_object_id' if not use_class else 'class'
            rays['is_leaf'] = rays[hit_object_key].isin(leaf_object_ids)
            # Filter out points with unknown object ids
            rays = rays[
                pd.isna(rays[hit_object_key]) |
                rays[hit_object_key].isin(wood_object_ids + leaf_object_ids)
            ]

            rays = rays.drop(columns=['hit_object_id', 'class'])
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
            scan_id = df['scan_id'].iloc[0] if 'scan_id' in df.columns else 0
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
            ax.set_title(f'Leaf and Wood Point Check - Leg {scan_id}')
            ax.legend()
            plt.show()
            plt.savefig(os.path.join(output_dir, f'leg_{scan_id}_leaf_wood_check.png'))

            # Save 3d .ply
            print("Saving leaf and wood point clouds...")
            pcd_leaf = pv.PolyData(leaf_points)
            pcd_leaf.save(os.path.join(output_dir, f'leg_{scan_id}_leaf_points_test.ply'))
            pcd_wood = pv.PolyData(wood_points)
            pcd_wood.save(os.path.join(output_dir, f'leg_{scan_id}_wood_points_test.ply'))

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
        scan_id = df['scan_id'].iloc[0]
        voxel_size = round(df['voxel_size'].iloc[0], 1)
        valid_rays = os.path.join(valid_rays_dir, f"leg_{scan_id}_valid_rays.parquet")
        
        reference = "/home/capheus/projects/51_tree_test/1001_etri_uniform_diamond/references/1001_etri_uniform_diamond_results_0.2.csv"

        if os.path.exists(valid_rays):
            print(f"Leg {scan_id}")
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
                            out_file = os.path.join(valid_rays_dir, f"leg_{scan_id}_vs_{voxel_size}_hits_not_type2_points.xyz")
                            points = points_meant_to_be_in_voxel
                        else:
                            print("All missing points are not in reference voxels.")
                            out_file = None
                else:
                    out_file = os.path.join(valid_rays_dir, f"leg_{scan_id}_vs_{voxel_size}_hits_not_type2_points.xyz")
                
                if out_file is not None:
                    np.savetxt(out_file, points, fmt="%.6f")


# Function used for taking valid_rays parquet files and references to establish voxel_ray intersections per valid_rays file
def voxel_ray_intersections(valid_rays_dir, references_dir, voxel_chunk_size=100, temp_dir=None, debug=False, epsilon=1e-6):
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
        nthreads = 2 # hard code this for your system
        mem_threshold = 0.9
    else:
        avail_cpu = psutil.cpu_count(logical=False)
        nthreads = psutil.cpu_count(logical=True)
        mem_threshold = 0.75
        print(f"No SLURM_CPUS_PER_TASK detected, using system CPU count: {avail_cpu} physical cores with {nthreads} threads.")

    optimal_workers = max(1, round(avail_cpu / 2))
    threads_per_worker = max(1, nthreads // optimal_workers)
    print(f"Optimal Dask configuration: {optimal_workers} workers with {threads_per_worker} threads each.")

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

    memory_worker = avail_mem / optimal_workers
    avail_mem_string_for_dask = f"{int(memory_worker)}MB"
    print(f"[voxel_ray_intersections] Starting Dask with memory_limit={avail_mem_string_for_dask}")

    client = _start_dask_client(
        memory_limit=avail_mem_string_for_dask,
        n_workers=optimal_workers,
        threads_per_worker=threads_per_worker,
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
        scan_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])
        print(f"[voxel_ray_intersections] Loading valid rays file for leg {scan_id}: {file}")
        df = dd.read_parquet(file, engine='pyarrow', blocksize="15MB")
        print(f"[voxel_ray_intersections] Leg {scan_id} partitions: {df.npartitions}")
        meta = pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        # # estimate memory per partition
        # num_rays = df.map_partitions(len).compute().max()
        # num_voxels = len(voxel_references)
        # estimated_memory_per_partition = estimate_broadcast_memory(num_rays=num_rays, num_voxels=num_voxels)
        # result = df.map_partitions(
        #     map_ray_partition_to_function,
        #     voxel_group=voxel_references,
        #     temp_dir=temp_dir,
        #     meta=meta
        # )


        chunk_results = []
        for start in range(0, voxel_references.shape[0], voxel_chunk_size):
            vchunk = voxel_references.iloc[start:start + voxel_chunk_size]
            r = df.map_partitions(
                map_ray_partition_to_function,
                voxel_group=vchunk,
                temp_dir=temp_dir,
                meta=meta
            )
            chunk_results.append(r)
        result = dd.concat(chunk_results, axis=0, interleave_partitions=True)
    
        # result = dd.concat(chunk_results, axis=0, interleave_partitions=True)
        voxel_ray_intersections[scan_id] = result
        print(f"[voxel_ray_intersections] Mapped partitions for leg {scan_id}")# with memory: {(estimated_memory_per_partition / (1024**2)):.2f} MB each.")

    def save_task(df, scan_id):
        if df.empty:
            print(f"No data to save for scan_id: {scan_id}.")
            return False
        voxel_size = round(float(df['voxel_size'].iloc[0]), 2)
        output_filename = os.path.join(valid_rays_dir, f"leg_{scan_id}_voxel_{voxel_size}_intersections.parquet")
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
        print(f"Saved intersections for scan_id: {scan_id} to {output_filename}.")
        return True

    print("[voxel_ray_intersections] Submitting Dask compute jobs...")
    # futures = []
    start_time = time.time()
    # for scan_id, results in voxel_ray_intersections.items():
    #     future = client.compute(results)
    #     futures.append((scan_id, future))

        # out_dir = os.path.join(valid_rays_dir, f"leg_{scan_id}_voxel_intersections")
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
    for scan_id, results in voxel_ray_intersections.items():
        future = client.compute(results)
        futures_dict[future] = scan_id
        futures.append((scan_id, future))

    # Process as they complete
    for future in as_completed(futures_dict):
        scan_id = futures_dict[future]
        results = future.result()
        # ... process and save
        grouped = results.groupby('voxel_size', group_keys=True)
        print(f"[voxel_ray_intersections] Leg {scan_id} grouped into {len(grouped)} voxel_size groups.")
        for voxel_size, group_df in grouped:
            print(f"[voxel_ray_intersections] Saving group voxel_size={voxel_size} (rows={len(group_df)}) for leg {scan_id}")
            save_task(group_df, scan_id)
            del group_df
        del results
        print(f"[voxel_ray_intersections] Completed save for leg {scan_id}")
    # for scan_id, future in futures:
    #     with ProgressBar():
    #         results = future.result()
    #     for voxel_size, group_df in results.groupby('voxel_size', group_keys=True):
    #         save_task(group_df, scan_id)
    #         del group_df
    #     del results

    # start_time = time.time()
    # print("[voxel_ray_intersections] Awaiting computation results...")
    # for scan_id, future in futures:
    #     print(f"[voxel_ray_intersections] Waiting on leg {scan_id} future...")
    #     with ProgressBar():
    #         results = future.result()
    #     print(f"[voxel_ray_intersections] Result received for leg {scan_id} (rows={len(results)})")
    #     grouped = results.groupby('voxel_size', group_keys=True)
    #     print(f"[voxel_ray_intersections] Leg {scan_id} grouped into {len(grouped)} voxel_size groups.")
    #     for voxel_size, group_df in grouped:
    #         print(f"[voxel_ray_intersections] Saving group voxel_size={voxel_size} (rows={len(group_df)}) for leg {scan_id}")
    #         save_task(group_df, scan_id)
    #         del group_df
    #     del results
    #     print(f"[voxel_ray_intersections] Completed save for leg {scan_id}")

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

    # Calculate memory usage more accurately
    # Ray-sphere intersection temporaries
    oc_memory = U * 3 * 8           # oc = origins - voxel_centres (before deletion)
    b_memory = U * 8                # b vector
    c_memory = U * 8                # c vector
    discriminant_memory = U * 8     # discriminant
    
    # Box intersection (when potential hits exist)
    # In worst case, assume ~50% of rays hit sphere, need AABB arrays
    potential_hit_ratio = 0.5
    t1_t2_memory = int(U * potential_hit_ratio * 3 * 8 * 2)  # t1 and t2 arrays
    t_enter_exit_memory = int(U * potential_hit_ratio * 8 * 2)  # t_enter, t_exit
    
    # Voxel bounds (for each chunk)
    voxel_bounds_memory = 3 * 8 * 2  # voxel_mins, voxel_maxs (3 coords each)
    
    # Direction safety check creates temp arrays
    direction_temp_memory = U * 3 * 8 * 2  # small_dir mask + potential_directions copy
    
    # Broadcast/indexing temporaries
    broadcast_memory = U * 3 * 8 * 3  # origins_b, directions_b, voxel_centres_b
    
    # Hit pair list (worst case: all rays hit)
    hit_pairs_memory = int(U * potential_hit_ratio * 16)  # List of (voxel_idx, ray_idx) tuples
    
    # Safety buffer for overhead and intermediate operations
    buffer = 1.5
    
    total_memory_per_chunk = (
        oc_memory +
        b_memory +
        c_memory +
        discriminant_memory +
        t1_t2_memory +
        t_enter_exit_memory +
        voxel_bounds_memory +
        direction_temp_memory +
        broadcast_memory +
        hit_pairs_memory
    ) * buffer
    
    optimal_chunk_size = max(
        min_chunk_size,
        min(
            int(memory_limit_bytes / total_memory_per_chunk),
            max_chunk_size
        )
    )

    if debug:
        print(f"[traverse_voxels] Memory diagnostics:")
        print(f"  - Number of unique rays (U): {U}")
        print(f"  - Number of voxels: {len(voxel_centres)}")
        print(f"  - Memory limit (bytes): {memory_limit_bytes}")
        print(f"  - Broadcast memory per voxel: {broadcast_memory} bytes")
        print(f"  - Optimal chunk size: {optimal_chunk_size} voxels")
        print(f"  - Min chunk size: {min_chunk_size}, Max chunk size: {max_chunk_size}")

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
    filtered_scan_ids = np.asarray(ray_partition['scan_id'].values)[all_ray_idxs]
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
        'scan_id': filtered_scan_ids.flatten(),
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
    del filtered_scan_ids, filtered_ray_ids, filtered_entry_coords, filtered_exit_coords
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
        point_weight_arr, is_leaf_arr, scan_ids_arr, ray_ids_arr,
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
                int(scan_ids_arr[ray_idx]),
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
                threads = avail_cpus * 2 # HARDCODED SYSTEM
            else:
                avail_cpus = psutil.cpu_count(logical=False)
                threads = psutil.cpu_count(logical=True)

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

    def _save_group(valid_rays_dir: str, scan_id: int, df: pd.DataFrame) -> bool:
        """Save a single leg/voxel_size group to Parquet with the exact schema."""
        if df is None or len(df) == 0:
            print(f"No data to save for scan_id: {scan_id}.")
            return False

        voxel_size = round(float(df['voxel_size'].iloc[0]), 2)
        output_filename = os.path.join(valid_rays_dir, f"leg_{scan_id}_voxel_{voxel_size}_intersections.parquet")
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
        scan_ids = ray_partition['scan_id'].to_numpy()
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
            del ray_ids, scan_ids, origins, directions, points, normals
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
        scan_ids_nb = scan_ids.astype(np.uint64)
        ray_ids_nb = ray_ids.astype(np.uint64)
        echo_intensity_nb = echo_intensity.astype(np.float64)
        return_number_nb = return_number.astype(np.int32)
        number_of_returns_nb = number_of_returns.astype(np.int32)
        point_weight_nb = point_weight.astype(np.float64)
        is_leaf_nb = is_leaf.astype(np.bool_)

        del origins, directions, points, normals, v_centres, vmins, vmaxs
        del v_ids, v_sizes, scan_ids, ray_ids, echo_intensity, return_number, number_of_returns, point_weight, is_leaf
        gc.collect()

        # Run numba kernel on pre-computed pairs
        out_data = process_ray_voxel_pairs_kernel(
            ray_cell_voxel_array,
            origins_nb, directions_nb, points_nb, normals_nb,
            echo_intensity_nb, return_number_nb, number_of_returns_nb,
            point_weight_nb, is_leaf_nb, scan_ids_nb, ray_ids_nb,
            v_ids_nb, v_sizes_nb, v_centres_nb, vmins_nb, vmaxs_nb,
            np.float64(epsilon)
        )
        print(f"Processed partition: found {len(out_data)} intersections.")
        print(f"Example data: {out_data[:5]}")

        del ray_cell_voxel_array
        del origins_nb, directions_nb, points_nb, normals_nb
        del echo_intensity_nb, return_number_nb, number_of_returns_nb
        del point_weight_nb, is_leaf_nb, scan_ids_nb, ray_ids_nb
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
            scan_id = int(parts[1])
        except Exception:
            scan_id = next((int(p) for p in parts if p.isdigit()), 0)

        ddf = dd.read_parquet(file, engine='pyarrow', blocksize='50MB')

        def _map_partition(ray_part: pd.DataFrame, grid_obj, voxel_obj) -> pd.DataFrame:
            return _traverse_partition_no_broadcast(ray_part, grid_obj, voxel_obj, epsilon=epsilon)

        result_ddf = ddf.map_partitions(_map_partition, grid_obj=grid, voxel_obj=voxel_data, meta=meta)
        fut = client.compute(result_ddf, sync=False)  # async submit
        futures_dict[fut] = scan_id
        print(f"Leg {scan_id}: submitted with {ddf.npartitions} ray partitions")

    # Save as they complete
    start_time = time.time()
    for fut in as_completed(futures_dict):
        scan_id = futures_dict[fut]
        result_df = fut.result()
        if result_df is None or len(result_df) == 0:
            print(f"Leg {scan_id}: no intersections")
            continue
        grouped = result_df.groupby('voxel_size', group_keys=True)
        for vox_size, grp in grouped:
            out_path = os.path.join(valid_rays_dir, f"leg_{scan_id}_voxel_{round(float(vox_size),2)}_intersections.parquet")
            grp.to_parquet(out_path, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
            del grp
        del result_df
        print(f"Leg {scan_id}: saved groups")
    
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
def _determine_dask_resources(
        cpus: int | None, 
        mem: int | None, 
        optimal_threads: int = 8, 
        mem_threshold: float = 0.7,
        partition_worker_ratio: float = 0.001
    ) -> tuple[str, int, int, str, str]:
    """
    Determine available CPUs and memory for Dask configuration.
    Returns (avail_cpus, avail_mem_string_for_dask, optimal_workers, threads_per_worker).
    """
    if cpus is not None:
        avail_cpus = cpus
    else:
        if 'SLURM_CPUS_PER_TASK' in os.environ:
            avail_cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            threads = avail_cpus * 2        # hardcoded for UQ Bunya
        else:
            avail_cpus = psutil.cpu_count(logical=False)
            threads = psutil.cpu_count(logical=True)

    if mem is not None:
        avail_mem = int(mem * mem_threshold)  # in MB
    else:
        avail_mem = int(float(os.environ.get('SLURM_MEM_PER_NODE', psutil.virtual_memory().available // (1024 * 1024))) * mem_threshold)  # in MB

    n_workers = threads // optimal_threads
    threads_per_worker = optimal_threads

    memory_worker = avail_mem / n_workers
    avail_mem_string_for_dask = f"{int(memory_worker)}MB"

    partition_size_mb = memory_worker * partition_worker_ratio
    partition_size_str = f"{int(partition_size_mb)}MB"

    # Establish best temp_dir
    if os.environ.get("TMPDIR") and os.path.isdir(os.environ["TMPDIR"]):
        temp_dir = os.environ["TMPDIR"]
    else:
        temp_dir = tempfile.gettempdir() if os.path.isdir(tempfile.gettempdir()) else "/tmp"

    return avail_mem_string_for_dask, n_workers, threads_per_worker, partition_size_str, temp_dir


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


@njit(nogil=True, cache=True, fastmath=True)
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


@njit(nogil=True, cache=True, fastmath=True)
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


@njit(nogil=True, cache=True, fastmath=True)
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
    scan_ids, ray_ids,
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
    scan_id_col          = np.empty(total_hits, np.int64)
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
            scan_id_col[k]         = scan_ids[r]
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
        scan_id_col[:k], ray_id_col[:k],
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
    scan_ids, ray_ids,
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
    scan_id_col            = np.empty(total_hits, np.int64)
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
            scan_id_col[k]         = scan_ids[r]
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
        scan_id_col[:k], ray_id_col[:k],
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
    scan_ids        = ray_part['scan_id'].to_numpy(dtype=np.int64)
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
        scan_ids, ray_ids,
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
        cols[5]:  data[5],   # scan_id
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


@njit(nogil=True, cache=True, fastmath=False)
def _process_partition_pairs(
    origins, directions,
    scan_ids, ray_ids,
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
            out_leg[k]   = scan_ids[r]
            out_ray[k]   = ray_ids[r]
            out_vox[k]   = v_ids[vi]
            out_vsize[k] = v_sizes[vi]
            k += 1

    return (out_leg[:k], out_ray[:k], out_vox[:k], out_vsize[:k]), k



def _map_partition_pairs(ray_part: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    if ray_part is None or len(ray_part) == 0:
        return pd.DataFrame(columns=['scan_id', 'ray_id', 'voxel_id', 'voxel_size'])
    
    c = get_client()
    vox_future = c.get_dataset('voxel_data')
    voxel_data = c.gather(vox_future)

    # Extract only what kernel needs
    origins    = ray_part[['origin_x','origin_y','origin_z']].to_numpy(dtype=np.float64)
    directions = ray_part[['direction_x','direction_y','direction_z']].to_numpy(dtype=np.float64)
    scan_ids    = ray_part['scan_id'].to_numpy(dtype=np.int64)
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
        origins, directions, scan_ids, ray_ids,
        v_ids, v_sizes, v_centres, vmins, vmaxs,
        origin_grid, cell_size, bbox_min, bbox_max,
        keys_ix, keys_iy, keys_iz, offsets, voxel_ids_flat,
        np.float64(eps)
    )

    # Check if k is empty and return empty DataFrame if so
    if k == 0:
        return pd.DataFrame(columns=['scan_id', 'ray_id', 'voxel_id', 'voxel_size'])

    return pd.DataFrame({
        'scan_id':     data[0],
        'ray_id':     data[1],
        'voxel_id':   data[2],
        'voxel_size': data[3],
    })


def voxel_ray_intersections_dask_new(valid_rays_dir: str,
                                 references_dir: str,
                                 output_path: str | None = None,
                                 temp_dir: str | None = None,
                                 cpus: int | None = None,
                                 mem: int | None = None,
                                 optimal_threads: int = 2,
                                 debug: bool = True,
                                 epsilon: float = 1e-6) -> None:
    memory_limit_str, n_workers, threads_per_worker, partition_size_str, temp_dir = _determine_dask_resources(
        cpus=cpus, mem=mem, optimal_threads=optimal_threads
    )
    client = _start_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_str,
        memory_target_fraction=0.7,
        memory_pause_fraction=0.9,
        memory_spill_fraction=0.8,
        temp_dir=temp_dir,
        processes=True
    )
    if debug:
        print(f"[voxel_ray_intersections_dask] Dask client: "
              f"workers={n_workers}, threads/worker={threads_per_worker}, mem/worker={memory_limit_str}, partition_size={partition_size_str}")

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
        scan_id = leg_from_filename(file)
        ddf = dd.read_parquet(file, engine='pyarrow', split_row_groups='adaptive', blocksize=partition_size_str)
        ddfs.append(ddf.assign(scan_id=scan_id))
    
    all_ddf = dd.concat(ddfs, interleave_partitions=True)

    meta_pairs = {
        'scan_id': str(all_ddf['scan_id'].dtype),
        'ray_id': str(all_ddf['ray_id'].dtype),
        'voxel_id': str(voxel_data['ids'].dtype),
        'voxel_size': str(voxel_data['sizes'].dtype)
    }
    pairs_ddf = all_ddf.map_partitions(
        _map_partition_pairs, meta=meta_pairs
    )

    # 2) Prepare rays DDF (select only needed columns)
    rays_ddf = all_ddf[['scan_id','ray_id',
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
        on=['scan_id','ray_id'],
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
    time.sleep(0.5) # Ensure memory is freed

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
    output_path = output_path if output_path is not None else os.path.join(valid_rays_dir, "voxel_ray_intersections")
    result.to_parquet(
        output_path,
        engine='pyarrow', compression='snappy',
        write_index=False,
        partition_on=['scan_id','voxel_size'],
        schema=voxel_ray_intersection_schema,
        overwrite=True,
        write_metadata_file=True
    )

    ### OLD ###
    # # Build a single DDF with scan_id attached
    # ddfs = []
    # for file in valid_files:
    #     scan_id = leg_from_filename(file)
    #     ddf = dd.read_parquet(file, engine='pyarrow', split_row_groups=True)
    #     ddfs.append(ddf.assign(scan_id=scan_id))
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

    # # Write one dataset partitioned by scan_id and voxel_size_rounded
    # persisted.to_parquet(
    #     os.path.join(valid_rays_dir, "intersections"),
    #     engine='pyarrow',
    #     compression='snappy',
    #     write_index=False,
    #     partition_on=['scan_id', 'voxel_size'],
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
                                 optimal_threads: int = 4,
                                 mem: int | None = None,
                                 debug: bool = True,
                                 epsilon: float = 1e-6) -> None:
    # Configure dask settings
    memory_limit_str, n_workers, threads_per_worker = _determine_dask_resources(cpus=cpus, mem=mem, optimal_threads=optimal_threads)

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
            scan_id = int(parts[1])
        except Exception:
            scan_id = next((int(p) for p in parts if p.isdigit()), 0)
        ddf = dd.read_parquet(file, engine='pyarrow', split_row_groups=True) # blocksize='32MB')

        def _map(ray_part: pd.DataFrame, vox) -> pd.DataFrame:
            vdata = vox.result() if hasattr(vox, 'result') else vox
            return _map_partition_numba(ray_part, vdata, eps=epsilon)

        result_ddf = ddf.map_partitions(_map, vox=voxel_data, meta=meta)
        fut = client.compute(result_ddf, sync=False)
        futures_dict[fut] = scan_id
        if debug:
            print(f"Leg {scan_id}: submitted with {ddf.npartitions} partitions")

    start_time = time.time()
    for fut in as_completed(futures_dict):
        scan_id = futures_dict[fut]
        result_df = fut.result()
        if result_df is None or len(result_df) == 0:
            if debug:
                print(f"Leg {scan_id}: no intersections")
            continue
        grouped = result_df.groupby('voxel_size', group_keys=True)
        for vox_size, grp in grouped:
            out_path = os.path.join(valid_rays_dir, f"leg_{scan_id}_voxel_{round(float(vox_size),2)}_intersections.parquet")
            grp.to_parquet(out_path, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)
            del grp
        del result_df
        if debug:
            print(f"Leg {scan_id}: saved groups")

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


        # Get scan_id from filename
        scan_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])

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

        voxel_ray_intersections[scan_id] = result

    def save_task(df, scan_id):
        if df.empty:
            return False
        
        # Get the voxel size from the dataframe
        voxel_size = round(float(df.name), 2)

        # Create the output filename
        output_filename = os.path.join(valid_rays_dir, f"leg_{scan_id}_voxel_{voxel_size}_intersections.parquet")

        # Save the dataframe to parquet
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema_old)

        return True


    with ProgressBar():

        for scan_id, results in voxel_ray_intersections.items():

            print(f"Processing leg {scan_id}...")

            results = results.compute()

            print(f"Saving results for leg {scan_id}...")
            results = results.groupby('voxel_size').apply(lambda x: save_task(x, scan_id))

            del results
            gc.collect()

            print(f"Completed leg {scan_id}!")


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

        bin_centres, LIAD_leaf_values, angles = calculate_inclination_angle_distribution(normals=valid_normals_leaf, weights=valid_weights_leaf)
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

def create_intersections_ddf(parquet_root: str, scan_ids: list[int | str], voxel_sizes: list[float | str], blocksize_str:str = "25MB") -> dask.dataframe.DataFrame:
    import os, glob
    all_files = []

    # Try and load a hive pattern input and apply filter before returning
    try:
        ddf = dd.read_parquet(parquet_root, engine="pyarrow", blocksize=blocksize_str)
        
        # Verify this is a voxel_ray_intersection dataset (not valid_rays)
        if 'origin_x' in ddf.columns or 'direction_x' in ddf.columns:
            raise ValueError("Detected valid_rays schema instead of voxel_ray_intersection schema. Ensure parquet_root contains only voxel_ray_intersection files.")
        
        if scan_ids is not None:
            # Check which column exists: scan_id or leg_id
            if 'scan_id' in ddf.columns:
                ddf = ddf[ddf['scan_id'].isin(scan_ids)]
            elif 'leg_id' in ddf.columns:
                ddf = ddf[ddf['leg_id'].isin(scan_ids)]
            else:
                raise ValueError("Neither 'scan_id' nor 'leg_id' column found in dataset.")
                
        if voxel_sizes is not None:
            voxel_size_strs = [f"{vs:.1f}" if isinstance(vs, float) else str(vs) for vs in voxel_sizes]
            ddf = ddf[ddf['voxel_size'].astype(str).isin(voxel_size_strs)]

        if ddf.npartitions > 0:
            print(f"Detected hive-partitioned dataset at {parquet_root}.")
            return ddf
    
    except Exception as e:
        print(f"Hive partition attempt failed: {e}. Falling back to legacy glob method.")

        # If no filters provided, collect all parquet files in the root.
        # This works for hive pattern and legacy
        if scan_ids is None and voxel_sizes is None:
            # Legacy method: glob only intersection parquet files (not valid_rays)
            all_files = glob.glob(os.path.join(parquet_root, "*_intersections.parquet"))
        
        elif scan_ids is None and voxel_sizes is not None:
            all_files = []
            for vs in voxel_sizes:
                # Legacy file matching
                vs_files = glob.glob(os.path.join(parquet_root, f"*voxel_size={vs}*_intersections.parquet"))
                if vs_files:
                    all_files.extend(vs_files)
        
        elif scan_ids is not None and voxel_sizes is None:
            all_files = []
            for scan_id in scan_ids:
                # legacy file matching
                scan_files = glob.glob(os.path.join(parquet_root, f"*leg_{scan_id}*_intersections.parquet"))
                if scan_files:
                    all_files.extend(scan_files)
        
        else:
            # Both scan_ids and voxel_sizes provided
            all_files = []
            for scan_id in scan_ids:
                for vs in voxel_sizes:
                    # legacy file matching
                    vs_scan_files = glob.glob(os.path.join(parquet_root, f"*leg_{scan_id}*voxel_size={vs}*_intersections.parquet"))
                    if vs_scan_files:
                        all_files.extend(vs_scan_files)
            
        if all_files:
            print(f"Found {len(all_files)} files with both filters.")
            ddfs = [dd.read_parquet(f, engine="pyarrow", blocksize=blocksize_str) for f in all_files]
            return dd.concat(ddfs, axis=0, ignore_index=True)
        else:
            return None

# get_voxel_metrics_nodask.py

import os, glob, math, time, warnings
from typing import Optional, List, Tuple, Dict
import numpy as np
import pandas as pd
import psutil
from joblib import Parallel, delayed

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# ------------------ internal helpers ------------------ #

# --- Add these helpers near the top of the module ---

import re
import pyarrow as pa
import pyarrow.parquet as pq

_SCAN_PATTERNS = [
    re.compile(r".*?leg[_-]?(\d+)", re.IGNORECASE),
    re.compile(r".*?scan[_-]?(\d+)", re.IGNORECASE),
]

_VOXEL_SIZE_PATTERNS = [
    re.compile(r".*?voxel[_-]?(\d+(?:\.\d+)?)", re.IGNORECASE),
    re.compile(r".*?voxel_size[_-]?(\d+(?:\.\d+)?)", re.IGNORECASE),
]

def _parse_scan_id_from_filename(path: str) -> int | None:
    base = os.path.basename(path)
    for rx in _SCAN_PATTERNS:
        m = rx.match(base)
        if m:
            try:
                return int(m.group(1))
            except Exception:
                pass
    return None

def _parse_voxel_size_from_filename(path: str) -> float | None:
    base = os.path.basename(path)
    for rx in _VOXEL_SIZE_PATTERNS:
        m = rx.match(base)
        if m:
            try:
                return float(m.group(1))
            except Exception:
                pass
    # also allow suffix patterns like ..._0.50.parquet
    tail = os.path.splitext(base)[0].split("_")[-1]
    if tail.replace(".", "", 1).isdigit():
        try:
            return float(tail)
        except Exception:
            pass
    return None

def _select_files_with_both_filters(
    intersections_folder: str,
    scan_ids: list[str] | None,
    voxel_sizes: list[float] | None
) -> list[tuple[str, int | None, float | None]]:
    """
    Returns [(file_path, inferred_scan_id, inferred_voxel_size), ...]
    Applies logical AND when both filter lists are provided.
    If no filters are provided, returns all files with inferred metadata (may be None if not parsable).
    """
    files = sorted(glob.glob(os.path.join(intersections_folder, "*_intersections.parquet")))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {intersections_folder}")

    scan_set = set(map(lambda s: str(int(s)), scan_ids)) if scan_ids else None
    vox_set  = set(voxel_sizes) if voxel_sizes else None

    out: list[tuple[str,int|None,float|None]] = []
    for f in files:
        sid = _parse_scan_id_from_filename(f)
        vs  = _parse_voxel_size_from_filename(f)

        # prefilter by file name if user supplied filters
        if scan_set is not None:
            if sid is None or (str(sid) not in scan_set):
                continue
        if vox_set is not None:
            if vs is None or (vs not in vox_set):
                continue
        out.append((f, sid, vs))

    # If user passed no filters, we still return all files with the parsed metadata
    return out

def _arrow_types_mapper(arrow_type):
    # keep UInt64/Int32 as pandas nullable dtypes when present in data
    if pa.types.is_uint64(arrow_type): return pd.UInt64Dtype()
    if pa.types.is_int32(arrow_type):  return pd.Int32Dtype()
    return None

def _load_intersections_with_injected_metadata(
    selected_files: list[tuple[str, int | None, float | None]],
    required_columns: list[str],
    *,
    must_have_scan_id: bool = True,
    must_have_voxel_size: bool = True
) -> tuple[pd.DataFrame, list[int]]:
    """
    Read each Parquet, inject 'scan_id' and 'voxel_size' columns derived from filename
    (if they are missing from the file). Returns a single concatenated DataFrame (sorted by voxel_id)
    and the list of included scan_ids for CSV header logging.
    
    Prints progress updates and uses TQDM for file reading.
    """
    
    dfs: list[pd.DataFrame] = []
    included_scan_ids: set[int] = set()
    
    print(f"\n[Loading intersections] Reading {len(selected_files)} Parquet file(s)...")

    for path, sid, vs in tqdm(selected_files, desc="  Files", unit=" file", ncols=90, leave=True):
        # Read schema first to determine which columns actually exist in this file
        table_schema = pq.read_schema(path)
        available_cols = set(table_schema.names)
        
        # Filter requested columns to only those that exist
        file_cols = [c for c in required_columns if c in available_cols and c not in ("scan_id", "voxel_size")]
        
        tbl = pq.read_table(path, columns=file_cols)
        df  = tbl.to_pandas(types_mapper=_arrow_types_mapper, split_blocks=True, self_destruct=True)

        # Inject scan_id
        if "scan_id" not in df.columns:
            if sid is None and must_have_scan_id:
                raise ValueError(
                    f"Could not infer scan_id from filename '{os.path.basename(path)}'. "
                    f"Please include 'leg_<id>'/ 'scan_<id>' in the filename or pass a filter list with a single id."
                )
            df["scan_id"] = np.uint64(0 if sid is None else sid)
        
        scan_id_val = int(df["scan_id"].iloc[0])
        included_scan_ids.add(scan_id_val)

        # Inject voxel_size
        if "voxel_size" not in df.columns:
            if vs is None and must_have_voxel_size:
                raise ValueError(
                    f"Could not infer voxel_size from filename '{os.path.basename(path)}'. "
                    f"Please include '_voxel_<size>' / 'voxel_size_<size>' in the filename or pass a filter list with a single size."
                )
            df["voxel_size"] = float(0.0 if vs is None else vs)

        dfs.append(df)

    if not dfs:
        print("  ⚠ No data loaded.")
        return pd.DataFrame(columns=required_columns), []

    print(f"\n[Concatenating] Combining {len(dfs)} file(s)...")
    big = pd.concat(dfs, axis=0, ignore_index=True)
    n_total_rows = len(big)
    print(f"  ✓ Concatenated: {n_total_rows:,} rows")

    # Ensure the two columns are the right dtypes
    print(f"[Formatting] Standardizing dtypes and sorting...")
    big["scan_id"]    = big["scan_id"].astype("UInt64")
    big["voxel_size"] = big["voxel_size"].astype("float64")

    # Sort by voxel_id for contiguous grouping
    if "voxel_id" not in big.columns:
        raise KeyError("Input intersections are missing 'voxel_id' — required for grouping.")
    
    big.sort_values(["voxel_id"], kind="stable", inplace=True)
    big.reset_index(drop=True, inplace=True)
    print(f"  ✓ Sorted by voxel_id and reset index")

    n_unique_voxels = big["voxel_id"].nunique()
    n_unique_scans = big["scan_id"].nunique()
    n_unique_sizes = big["voxel_size"].nunique()
    print(f"  ✓ Input summary: {n_unique_voxels:,} unique voxels | "
          f"{n_unique_scans} scan(s) | {n_unique_sizes} voxel_size(s)")

    return big, sorted(included_scan_ids)

def _list_intersection_parquets(intersections_folder: str,
                                scan_ids: Optional[List[str]],
                                voxel_sizes: Optional[List[float]]) -> List[str]:
    """
    Gathers all intersection parquet files under folder and filters by scan_id and voxel_size
    based on filename tokens if present. Otherwise the filtering will be applied after reading.
    """
    files = sorted(glob.glob(os.path.join(intersections_folder, "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No Parquet files found under {intersections_folder}")

    # Light prefilter based on common naming patterns (best effort — non-fatal)
    out = []
    for f in files:
        ok = True
        base = os.path.basename(f)
        if scan_ids is not None:
            # accept names like "leg_12_..." or "...scan_12..."
            tok = None
            if "leg_" in base:
                try: tok = base.split("leg_")[1].split("_")[0]
                except: tok = None
            elif "scan_" in base:
                try: tok = base.split("scan_")[1].split("_")[0]
                except: tok = None
            if tok is not None and (str(tok) not in set(map(str, scan_ids))):
                ok = False

        if voxel_sizes is not None:
            # accept names like "..._voxel_0.50_..." or "...voxel_size_0.5..."
            vtok = None
            if "_voxel_" in base:
                try: vtok = base.split("_voxel_")[1].split("_")[0]
                except: vtok = None
            elif "voxel_size_" in base:
                try: vtok = base.split("voxel_size_")[1].split("_")[0]
                except: vtok = None
            if vtok is not None:
                try:
                    vf = float(vtok)
                    if vf not in set(voxel_sizes):
                        ok = False
                except:
                    pass
        if ok:
            out.append(f)
    return out or files


def _read_dataset(intersections_paths: List[str],
                  scan_ids: Optional[List[str]],
                  voxel_sizes: Optional[List[float]],
                  columns: Optional[List[str]] = None) -> ds.Dataset:
    """
    Build a PyArrow Dataset with optional predicate pushdown for scan_id and voxel_size.
    """
    dataset = ds.dataset(intersections_paths, format="parquet")
    # Prepare filters
    filters = []
    if scan_ids is not None and len(scan_ids) > 0:
        # scan_id is uint64 in schema
        scan_ids_uint = [np.uint64(int(s)) for s in scan_ids]
        filters.append(("scan_id", "in", scan_ids_uint))
    if voxel_sizes is not None and len(voxel_sizes) > 0:
        # voxel_size stored as float (float32/64); use exact values passed by user
        filters.append(("voxel_size", "in", voxel_sizes))

    if filters:
        dataset = ds.dataset(intersections_paths, format="parquet", partitioning="hive")
        dataset = dataset.replace_schema_metadata(dataset.schema.metadata)  # no-op, keeps meta
        # We'll pass filters at scan time.
    return dataset


def _arrow_types_mapper(arrow_type):
    # Keep unsigned ints / ints as pandas nullable types when possible
    if pa.types.is_uint64(arrow_type): return pd.UInt64Dtype()
    if pa.types.is_int32(arrow_type):  return pd.Int32Dtype()
    return None


def _count_unique(vals: np.ndarray) -> int:
    """Fast unique count (np.unique on 1D array)."""
    if vals.size == 0:
        return 0
    return np.unique(vals).size


def _scan_to_pandas_sorted_by_voxel(dataset: ds.Dataset,
                                    filters,
                                    columns: List[str],
                                    target_mb: int) -> pd.DataFrame:
    """
    Stream the dataset in batches (by fragments/rowgroups) into pandas,
    concatenate, then sort by voxel_id for contiguous grouping.
    For very large inputs, you can implement external sort-by-hash here.
    """
    # Use to_table() with filter parameter instead of scan()
    if filters:
        tbl = dataset.to_table(columns=columns, filter=filters)
    else:
        tbl = dataset.to_table(columns=columns)
    
    # Convert to pandas in one go
    # PyArrow's Table.to_pandas is highly optimized in C++.
    df = tbl.to_pandas(types_mapper=_arrow_types_mapper, split_blocks=True, self_destruct=True)
    
    # Sort by voxel_id for contiguous ranges (important for parallel group spans)
    df.sort_values(["voxel_id"], kind="stable", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ---------------- Per-voxel metrics (faithful to your logic) ---------------- #

def _metrics_for_voxel_block(
    block: pd.DataFrame,
    *,
    average_leaf_area: float,
    is_multireturn: bool,
    is_leaf_true: bool = True,
    beam_divergence_mrad: float,
    epsilon: float
) -> pd.DataFrame:
    """
    Compute metrics for a single voxel_id (block is already filtered to that voxel).
    Returns a 1-row DataFrame matching whichever voxel_metrics schema your _gen_dataframe() builds.
    """
    # Select schema outside and construct a 1-row df to fill
    voxel_id = block["voxel_id"].iloc[0]
    vs = float(block["voxel_size"].iloc[0])

    # Build the result row using your schema factory (handles column order/dtypes)
    schema = voxel_metrics_schema_multireturn if is_multireturn else voxel_metrics_schema_singlereturn
    out = _gen_dataframe(schema)  # 0-row frame with all columns; we will fill loc[0]

    vx = float(block["voxel_cx"].iloc[0]); vy = float(block["voxel_cy"].iloc[0]); vz = float(block["voxel_cz"].iloc[0])

    hit_types = block["hit_type"].to_numpy()
    unbound      = (hit_types == 0)
    previous_hit = (hit_types == 1)
    current_hit  = (hit_types == 2)
    yet_to_hit   = (hit_types == 3)

    if is_leaf_true:
        leaf_mask = block["is_leaf"].to_numpy()
    else:
        leaf_mask = ~block["is_leaf"].to_numpy()

    current_leaf_mask = current_hit & leaf_mask

    # Rays considered valid for counting (as in your code)
    valid_ray_mask = unbound | current_hit | yet_to_hit
    # Unique ray count
    num_rays = _count_unique(block.loc[valid_ray_mask, "ray_id"].to_numpy(dtype=np.uint64))
    if num_rays <= 0:
        out.loc[0, 'voxel_id'] = voxel_id
        out.loc[0, 'voxel_cx'] = vx; out.loc[0, 'voxel_cy'] = vy; out.loc[0, 'voxel_cz'] = vz
        out.loc[0, 'voxel_size'] = vs
        out.loc[0, 'num_rays'] = 0
        return out

    # Basic tallies
    num_lw_hits = int(current_hit.sum())
    num_leaf_hits = int(current_leaf_mask.sum())
    num_wood_hits = num_lw_hits - num_leaf_hits

    # Mean viewing angles
    va = block["viewing_angle"].to_numpy()
    mean_angle_all  = float(np.nanmean(va[current_hit])) if num_lw_hits > 0 else np.nan
    mean_angle_leaf = float(np.nanmean(va[current_leaf_mask])) if num_leaf_hits > 0 else np.nan
    mean_angle_wood = float(np.nanmean(va[current_hit & ~leaf_mask])) if num_wood_hits > 0 else np.nan

    # PGAP and I
    pgap_lw   = (num_rays - num_lw_hits) / max(num_rays, 1)
    pgap_leaf = (num_rays - num_leaf_hits) / max(num_rays, 1)
    pgap_wood = (num_rays - num_wood_hits) / max(num_rays, 1)
    I_lw, I_leaf, I_wood = 1.0 - pgap_lw, 1.0 - pgap_leaf, 1.0 - pgap_wood

    # Path lengths: ||exit - entry|| for rays in valid_ray_mask; else NaN
    ent = block[["t_entry_x","t_entry_y","t_entry_z"]].to_numpy()
    ext = block[["t_exit_x","t_exit_y","t_exit_z"]].to_numpy()
    pl = np.full(len(block), np.nan, dtype=np.float64)
    d_ent_ext = ext - ent
    pl[valid_ray_mask] = np.linalg.norm(d_ent_ext[valid_ray_mask], axis=1)

    # Free-path lengths
    fpl = pl.copy()
    ray_ids = block["ray_id"].to_numpy(dtype=np.uint64)
    nrets  = block["number_of_returns"].to_numpy(dtype=np.int32, na_value=-2147483648)
    rnums  = block["return_number"].to_numpy(dtype=np.int32,  na_value=-2147483648)
    pts    = block[["point_x","point_y","point_z"]].to_numpy()

    # Unbound → fpl = pl
    fpl[unbound] = pl[unbound]

    # Single-return rays
    single_ret = (nrets == 1)
    if single_ret.any():
        # entry→point for "hit", pl for "yet_to_hit", NaN for "previous_hit"
        mask_hit  = single_ret & current_hit
        mask_yet  = single_ret & yet_to_hit
        mask_prev = single_ret & previous_hit
        if mask_hit.any():
            fpl[mask_hit] = np.linalg.norm(pts[mask_hit] - ent[mask_hit], axis=1)
        if mask_yet.any():
            fpl[mask_yet] = pl[mask_yet]
        if mask_prev.any():
            fpl[mask_prev] = np.nan

    # Multi-return rays: follow your exact logic
    if (nrets > 1).any():
        # Group indices by ray_id for "current_hit" and "yet_to_hit"
        from collections import defaultdict
        inside_ray_indices = defaultdict(list)
        next_hit_ray_indices = defaultdict(list)

        multi_mask = (nrets > 1)
        multi_inds = np.where(multi_mask)[0]
        for idx in multi_inds:
            rid = ray_ids[idx]
            if current_hit[idx]:
                inside_ray_indices[rid].append(idx)
            if yet_to_hit[idx]:
                next_hit_ray_indices[rid].append(idx)

        for rid, inside_indices in inside_ray_indices.items():
            # Sort by return_number
            sorted_indices = sorted(inside_indices, key=lambda i: rnums[i])
            if not sorted_indices:
                continue
            # first return: entry→point; subsequent: prev_point→point
            for k, idx in enumerate(sorted_indices):
                if k == 0:
                    fpl[idx] = np.linalg.norm(pts[idx] - ent[idx])
                else:
                    prev_idx = sorted_indices[k - 1]
                    fpl[idx] = np.linalg.norm(pts[idx] - pts[prev_idx])

            # Last return may have a "next" in yet_to_hit with return_number = last+1
            last_idx = sorted_indices[-1]
            last_rnum = rnums[last_idx]
            next_list = next_hit_ray_indices.get(rid, [])
            for ni in next_list:
                if rnums[ni] == last_rnum + 1:
                    fpl[ni] = np.linalg.norm(ext[last_idx] - pts[last_idx])
                    break  # only the immediately next

    # Effective path lengths using lambda_1
    lambda_1 = calculate_lambda_1(average_leaf_area, vs)
    eff_pl  = calculate_effective_path_length(path_lengths=pl,  lambda_1=lambda_1)
    eff_fpl = calculate_effective_path_length(path_lengths=fpl, lambda_1=lambda_1)

    # Aggregates
    mean_pl = float(np.nanmean(pl))
    sum_pl  = float(np.nansum(pl))
    mean_fpl = float(np.nanmean(fpl))
    sum_fpl  = float(np.nansum(fpl))

    sum_fpl_hit      = float(np.nansum(fpl[current_hit]))
    sum_fpl_exit     = float(np.nansum(fpl[yet_to_hit]))
    sum_fpl_hit_leaf = float(np.nansum(fpl[current_leaf_mask]))

    mean_eff_pl = float(np.nanmean(eff_pl))
    var_eff_pl  = float(np.nanvar(eff_pl))
    sum_eff_pl  = float(np.nansum(eff_pl))

    mean_eff_fpl = float(np.nanmean(eff_fpl))
    var_eff_fpl  = float(np.nanvar(eff_fpl))
    sum_eff_fpl  = float(np.nansum(eff_fpl))
    sum_eff_fpl_hit      = float(np.nansum(eff_fpl[current_hit]))
    sum_eff_fpl_exit     = float(np.nansum(eff_fpl[yet_to_hit]))
    sum_eff_fpl_hit_leaf = float(np.nansum(eff_fpl[current_leaf_mask]))

    # PIAD - all hits
    normals = block[["normal_x","normal_y","normal_z"]].to_numpy()
    weights = block["point_weight"].to_numpy()
    bins, piad_vals, _extra = calculate_inclination_angle_distribution(normals=normals, weights=weights)

    # LIAD - only leaf hits contribute to LIAD
    leaf_normals = block.loc[current_leaf_mask, ["normal_x","normal_y","normal_z"]].to_numpy()
    leaf_weights = block.loc[current_leaf_mask, "point_weight"].to_numpy()
    bins, liad_vals, _extra = calculate_inclination_angle_distribution(normals=leaf_normals, weights=leaf_weights)

    # WIAD - only wood hits contribute to WIAD
    wood_normals = block.loc[current_hit & ~leaf_mask, ["normal_x","normal_y","normal_z"]].to_numpy()
    wood_weights = block.loc[current_hit & ~leaf_mask, "point_weight"].to_numpy()
    bins, wiad_vals, _extra = calculate_inclination_angle_distribution(normals=wood_normals, weights=wood_weights)

    # Calculate all G values
    va_lw   = va[current_hit]
    va_leaf = va[current_leaf_mask]
    va_wood = va[current_hit & ~leaf_mask]
    G_leaf  = calculate_G(viewing_angles=va_leaf, bin_centres=bins, LIAD_values=liad_vals)
    G_lw    = calculate_G(viewing_angles=va_lw,   bin_centres=bins, LIAD_values=piad_vals)
    G_wood  = calculate_G(viewing_angles=va_wood, bin_centres=bins, LIAD_values=wiad_vals)
    # Reduce arrays to mean if needed
    G_leaf = float(np.nanmean(G_leaf)) if isinstance(G_leaf, np.ndarray) else (np.nan if G_leaf is None else float(G_leaf))
    G_lw   = float(np.nanmean(G_lw))   if isinstance(G_lw, np.ndarray)   else (np.nan if G_lw is None else float(G_lw))
    G_wood = float(np.nanmean(G_wood)) if isinstance(G_wood, np.ndarray) else (np.nan if G_wood is None else float(G_wood))

    # Multi-return probability-style metrics (optional block)
    LAD_first = LAD_equal = LAD_int = np.nan
    LAD_MLE_nocorr = LAD_MLE_lambda1 = LAD_MLE_bias = LAD_MLE_lambda1_bias = np.nan
    P_first = P_equal = P_int = np.nan
    P_first_leaf = P_equal_leaf = P_int_leaf = np.nan

    if is_multireturn:
        # Kent & Bailey style probabilities (your approach)
        def _collapse(T, W):
            tot = W if np.isscalar(W) else np.nansum(W)
            return float((T * W) / tot) if tot else np.nan

        # First-hit weighting
        first_hit = (block["return_number"].to_numpy(dtype=np.int32, na_value=-2147483648) == 1)
        yet_first_hit = yet_to_hit & first_hit
        Tk_first_lw = np.count_nonzero(yet_first_hit)
        BWk_first   = 1.0

        current_first_hit_leaf = current_hit & leaf_mask
        Tk_first_leaf = np.count_nonzero(current_first_hit_leaf)

        # Equal-hit weighting
        echoes_before_lw = int(previous_hit.sum())
        echoes_during_lw = int(current_hit.sum())
        echoes_after_lw  = int(yet_to_hit.sum())
        denom_lw = max(echoes_during_lw + echoes_after_lw, 1)
        Tk_equal_lw   = echoes_after_lw / denom_lw
        BWk_equal_lw  = (echoes_during_lw + echoes_after_lw) / max(echoes_before_lw + echoes_during_lw + echoes_after_lw, 1)

        echoes_before_leaf = int((previous_hit & leaf_mask).sum())
        echoes_during_leaf = int((current_hit & leaf_mask).sum())
        echoes_after_leaf  = int((yet_to_hit  & leaf_mask).sum())
        denom_leaf = max(echoes_during_leaf + echoes_after_leaf, 1)
        Tk_equal_leaf  = echoes_after_leaf / denom_leaf
        BWk_equal_leaf = (echoes_during_leaf + echoes_after_leaf) / max(echoes_before_leaf + echoes_during_leaf + echoes_after_leaf, 1)

        # Intensity weighting
        intens = block["echo_intensity"].to_numpy()
        intensity_before_lw = float(np.nansum(intens[previous_hit]))
        intensity_during_lw = float(np.nansum(intens[current_hit]))
        intensity_after_lw  = float(np.nansum(intens[yet_to_hit]))
        denom_int_lw = intensity_during_lw + intensity_after_lw
        Tk_int_lw  = (intensity_after_lw / denom_int_lw) if denom_int_lw != 0 else np.nan
        BWk_int_lw = (intensity_during_lw + intensity_after_lw) / max(intensity_before_lw + intensity_during_lw + intensity_after_lw, 1e-12)

        intensity_before_leaf = float(np.nansum(intens[previous_hit & leaf_mask]))
        intensity_during_leaf = float(np.nansum(intens[current_hit  & leaf_mask]))
        intensity_after_leaf  = float(np.nansum(intens[yet_to_hit   & leaf_mask]))
        denom_int_leaf = intensity_during_leaf + intensity_after_leaf
        Tk_int_leaf  = (intensity_after_leaf / denom_int_leaf) if denom_int_leaf != 0 else np.nan
        BWk_int_leaf = (intensity_during_leaf + intensity_after_leaf) / max(intensity_before_leaf + intensity_during_leaf + intensity_after_leaf, 1e-12)

        P_first, P_equal, P_int, P_first_leaf, P_equal_leaf, P_int_leaf = (
            _collapse(T, W) for (T, W) in [
                (Tk_first_lw,  BWk_first),
                (Tk_equal_lw,  BWk_equal_lw),
                (Tk_int_lw,    BWk_int_lw),
                (Tk_first_leaf,  BWk_first),
                (Tk_equal_leaf,  BWk_equal_leaf),
                (Tk_int_leaf,    BWk_int_leaf)
            ]
        )

        # LAD proxies (Pimont 2018 style, as per your comment)
        LAD_first = BL_pimont_2018(P=P_first,     mean_path_length=mean_pl, G=G_leaf, CI=1.0)
        LAD_equal = BL_pimont_2018(P=P_equal,     mean_path_length=mean_pl, G=G_leaf, CI=1.0)
        LAD_int   = BL_pimont_2018(P=P_int,       mean_path_length=mean_pl, G=G_leaf, CI=1.0)

        # Beam area and bias terms for Vincent 2021 MLE
        dist_to_centre = block["distance_to_centre"].to_numpy()
        ray_weights = 1.0 / np.clip(block["number_of_returns"].to_numpy(dtype=np.int32, na_value=-2147483648), 1, None)

        beam_div_rad = beam_divergence_mrad * 1e-3  # mrad → rad
        beam_surface_area_all = np.full(dist_to_centre.shape, np.nan, dtype=np.float64)
        beam_radius = dist_to_centre[valid_ray_mask] * beam_div_rad
        beam_surface_area_all[valid_ray_mask] = np.pi * (beam_radius ** 2)

        # Unique pulses (unique ray_ids among valid rays)
        uniq_mask_idx = np.unique(ray_ids, return_index=True)[1]
        unique_pulse_area = beam_surface_area_all[uniq_mask_idx]
        unique_ray_ids = ray_ids[uniq_mask_idx]
        sorter = np.argsort(unique_ray_ids)

        # Indices per hit type mapped to unique pulses
        def _map_to_unique(idx_mask):
            return np.searchsorted(unique_ray_ids, ray_ids[idx_mask], sorter=sorter)

        idx_current = _map_to_unique(current_hit)
        idx_yet     = _map_to_unique(yet_to_hit)
        idx_unbound = _map_to_unique(unbound)
        idx_c_or_y  = _map_to_unique(current_hit | yet_to_hit)

        w_current = ray_weights[current_hit]
        w_yet     = ray_weights[yet_to_hit]
        w_unbound = ray_weights[unbound]
        w_cy      = ray_weights[current_hit | yet_to_hit]

        # Needed path-length slices
        pl_yet     = pl[yet_to_hit]
        pl_unbound = pl[unbound]
        fpl_cur    = fpl[current_hit]
        eff_fpl_cur = eff_fpl[current_hit]
        eff_pl_yet  = eff_pl[yet_to_hit]
        eff_pl_unb  = eff_pl[unbound]

        # 1) sum_ba_hit = Σ_q S_q * Σ_j α_jq
        uniq_w_hit = np.bincount(idx_current, weights=w_current, minlength=unique_ray_ids.size)
        sum_ba_hit = float(np.nansum(unique_pulse_area * uniq_w_hit))

        # 2) unique_fpl_hit = Σ_j α_jq * FPL_jq ; 3) unique_pl_exit = α_out,q * pl_q
        uniq_fpl_hit = np.bincount(idx_current, weights=(fpl_cur * w_current), minlength=unique_ray_ids.size)
        sum_yet_exit = np.bincount(idx_yet, weights=(pl_yet * w_yet), minlength=unique_ray_ids.size)
        sum_unb_exit = np.bincount(idx_unbound, weights=(pl_unbound * 1.0), minlength=unique_ray_ids.size)
        uniq_pl_exit = sum_yet_exit + sum_unb_exit
        sum_pl_all   = float(np.nansum(unique_pulse_area * (uniq_fpl_hit + uniq_pl_exit)))

        # 4–5) with effective PL
        uniq_eff_fpl_hit = np.bincount(idx_current, weights=(eff_fpl_cur * w_current), minlength=unique_ray_ids.size)
        sum_yet_exit_eff = np.bincount(idx_yet, weights=(eff_pl_yet * w_yet), minlength=unique_ray_ids.size)
        sum_unb_exit_eff = np.bincount(idx_unbound, weights=(eff_pl_unb * 1.0), minlength=unique_ray_ids.size)
        uniq_eff_pl_exit = sum_yet_exit_eff + sum_unb_exit_eff
        sum_pl_all_eff   = float(np.nansum(unique_pulse_area * (uniq_eff_fpl_hit + uniq_eff_pl_exit)))

        # 6) sum of α_in,q (weights entering voxel)
        sum_cy = np.bincount(idx_c_or_y, weights=w_cy, minlength=unique_ray_ids.size)
        sum_unb_enter = np.bincount(idx_unbound, weights=np.ones_like(idx_unbound, dtype=np.float64), minlength=unique_ray_ids.size)
        uniq_w_enter = sum_cy + sum_unb_enter

        bias_pt_1 = float(np.nansum(unique_pulse_area * uniq_w_enter)) / max(num_rays, 1)
        bias_pt_2     = float(np.nansum(unique_pulse_area * uniq_fpl_hit)) / sum_pl_all if sum_pl_all != 0 else np.nan
        bias_pt_2_eff = float(np.nansum(unique_pulse_area * uniq_eff_fpl_hit)) / sum_pl_all_eff if sum_pl_all_eff != 0 else np.nan

        bias_corr     = bias_pt_1 * bias_pt_2
        bias_corr_eff = bias_pt_1 * bias_pt_2_eff

        LAD_MLE_nocorr       = MLE_vincent_2021(sum_ba_hit=sum_ba_hit,      sum_pl_all=sum_pl_all,      G=G_leaf, CI=1.0, bias_corr=None)
        LAD_MLE_lambda1      = MLE_vincent_2021(sum_ba_hit=sum_ba_hit,      sum_pl_all=sum_pl_all_eff,  G=G_leaf, CI=1.0, bias_corr=None)
        LAD_MLE_bias         = MLE_vincent_2021(sum_ba_hit=sum_ba_hit,      sum_pl_all=sum_pl_all,      G=G_leaf, CI=1.0, bias_corr=bias_corr)
        LAD_MLE_lambda1_bias = MLE_vincent_2021(sum_ba_hit=sum_ba_hit,      sum_pl_all=sum_pl_all_eff,  G=G_leaf, CI=1.0, bias_corr=bias_corr_eff)

    # ---------------- fill result row ----------------
    out.loc[0, 'voxel_id']   = voxel_id
    out.loc[0, 'voxel_cx']   = vx; out.loc[0, 'voxel_cy'] = vy; out.loc[0, 'voxel_cz'] = vz
    out.loc[0, 'voxel_size'] = vs
    out.loc[0, 'num_rays']   = int(num_rays)
    out.loc[0, 'num_hits']   = int(num_lw_hits)
    out.loc[0, 'num_leaf_hits'] = int(num_leaf_hits)

    out.loc[0, 'pgap_lw']   = float(pgap_lw)
    out.loc[0, 'pgap_leaf'] = float(pgap_leaf)
    out.loc[0, 'pgap_wood'] = float(pgap_wood)
    out.loc[0, 'I_lw']      = float(I_lw)
    out.loc[0, 'I_leaf']    = float(I_leaf)
    out.loc[0, 'I_wood']    = float(I_wood)
    out.loc[0, 'G_lw']      = float(G_lw) if np.isfinite(G_lw) else np.nan
    out.loc[0, 'G_leaf']    = float(G_leaf) if np.isfinite(G_leaf) else np.nan
    out.loc[0, 'G_wood']    = float(G_wood) if np.isfinite(G_wood) else np.nan
    out.loc[0, 'lambda_1']  = float(lambda_1)

    # LIAD bins (18 bins assumed like your code)
    # liad_vals may be shorter if no data; guard each access
    for k, centre in enumerate([
        2.5, 7.5, 12.5, 17.5, 22.5, 27.5, 32.5, 37.5, 42.5,
        47.5, 52.5, 57.5, 62.5, 67.5, 72.5, 77.5, 82.5, 87.5
    ]):
        val = np.float32(liad_vals[k]) if k < len(liad_vals) else np.nan
        out.loc[0, f'LIAD_leaf_bin_{centre}'] = val

    out.loc[0, 'mean_angle_leaf'] = np.float32(mean_angle_leaf)
    out.loc[0, 'mean_angle_all']  = np.float32(mean_angle_all)
    out.loc[0, 'mean_angle_wood']  = np.float32(mean_angle_wood)

    out.loc[0, 'mean_path_length'] = np.float64(mean_pl)
    out.loc[0, 'sum_path_length']  = np.float64(sum_pl)

    out.loc[0, 'mean_free_path_length'] = np.float64(mean_fpl)
    out.loc[0, 'sum_free_path_length']  = np.float64(sum_fpl)
    out.loc[0, 'sum_free_path_length_hit']      = np.float64(sum_fpl_hit)
    out.loc[0, 'sum_free_path_length_exit']     = np.float64(sum_fpl_exit)
    out.loc[0, 'sum_free_path_length_hit_leaf'] = np.float64(sum_fpl_hit_leaf)

    out.loc[0, 'mean_eff_path_length'] = np.float64(mean_eff_pl)
    out.loc[0, 'var_eff_path_length']  = np.float64(var_eff_pl)
    out.loc[0, 'sum_eff_path_length']  = np.float64(sum_eff_pl)

    out.loc[0, 'mean_eff_free_path_length'] = np.float64(mean_eff_fpl)
    out.loc[0, 'var_eff_free_path_length']  = np.float64(var_eff_fpl)
    out.loc[0, 'sum_eff_free_path_length']  = np.float64(sum_eff_fpl)
    out.loc[0, 'sum_eff_free_path_length_hit']      = np.float64(sum_eff_fpl_hit)
    out.loc[0, 'sum_eff_free_path_length_exit']     = np.float64(sum_eff_fpl_exit)
    out.loc[0, 'sum_eff_free_path_length_hit_leaf'] = np.float64(sum_eff_fpl_hit_leaf)

    if is_multireturn:
        out.loc[0, 'P_first']         = float(P_first)         if np.isfinite(P_first)         else np.nan
        out.loc[0, 'P_equal']         = float(P_equal)         if np.isfinite(P_equal)         else np.nan
        out.loc[0, 'P_intensity']     = float(P_int)           if np.isfinite(P_int)           else np.nan
        out.loc[0, 'P_first_leaf']    = float(P_first_leaf)    if np.isfinite(P_first_leaf)    else np.nan
        out.loc[0, 'P_equal_leaf']    = float(P_equal_leaf)    if np.isfinite(P_equal_leaf)    else np.nan
        out.loc[0, 'P_intensity_leaf']= float(P_int_leaf)      if np.isfinite(P_int_leaf)      else np.nan

        out.loc[0, 'LAD_first']           = float(LAD_first)           if np.isfinite(LAD_first)           else np.nan
        out.loc[0, 'LAD_equal']           = float(LAD_equal)           if np.isfinite(LAD_equal)           else np.nan
        out.loc[0, 'LAD_intensity']       = float(LAD_int)             if np.isfinite(LAD_int)             else np.nan
        out.loc[0, 'LAD_MLE_nocorr']      = float(LAD_MLE_nocorr)      if np.isfinite(LAD_MLE_nocorr)      else np.nan
        out.loc[0, 'LAD_MLE_lambda1']     = float(LAD_MLE_lambda1)     if np.isfinite(LAD_MLE_lambda1)     else np.nan
        out.loc[0, 'LAD_MLE_bias']        = float(LAD_MLE_bias)        if np.isfinite(LAD_MLE_bias)        else np.nan
        out.loc[0, 'LAD_MLE_lambda1_bias']= float(LAD_MLE_lambda1_bias)if np.isfinite(LAD_MLE_lambda1_bias)else np.nan

    return out


# ============================================================================
# Progress-aware public API
# ============================================================================

def get_voxel_metrics_nodask(
    intersections_folder: str,
    average_leaf_area: float,
    *,
    output_dir: Optional[str] = None,
    cpus: Optional[int] = None,
    mem: Optional[int] = None,
    scan_ids: Optional[List[str]] = None,
    voxel_sizes: Optional[List[float]] = None,
    optimal_threads: int = 2,
    beam_divergence: float = 0.35,   # mrad
    is_multireturn: bool = False,
    is_leaf_true: bool = True,       # same meaning as your flag
    debug: bool = True,
    epsilon: float = 1e-9,
    # Tuning
    voxel_block_rows_hint: int = 0,    # 0 -> auto, else compute blocks ~ this many rows each
    normal_calc_voxel_size: float = 10,       # voxel size for normal estimation (if normals not present)
) -> pd.DataFrame:
    """
    Non-Dask, resource-aware and parallel computation of voxel metrics.
    - Loads intersections via PyArrow.
    - Sorts rows by voxel_id.
    - Splits into contiguous voxel blocks and computes metrics in parallel.
    - Provides clear progress updates with live counters and progress bars.

    Returns:
        pd.DataFrame of voxel metrics (concatenated across all voxel_sizes),
        and writes CSV per voxel_size identical to your original routine.
    """

    # ========== Phase 1: Setup & Configuration ==========
    print("\n" + "=" * 80)
    print("  [get_voxel_metrics_nodask] Voxel Metrics Computation")
    print("=" * 80)

    res = detect_resources(target_threads_per_worker=optimal_threads)
    n_workers = cpus if (cpus and cpus > 0) else res.n_workers
    threads_per_worker = res.threads_per_worker
    os.environ.setdefault("OMP_NUM_THREADS", str(threads_per_worker))
    os.environ.setdefault("OPENBLAS_NUM_THREADS", str(threads_per_worker))
    os.environ.setdefault("MKL_NUM_THREADS", str(threads_per_worker))

    print(f"\n[1] Configuration:")
    print(f"    • Workers:           {n_workers}")
    print(f"    • Threads/worker:    {threads_per_worker}")
    print(f"    • Memory/worker:     ~{res.mem_per_worker_mb} MB")
    print(f"    • Multi-return:      {is_multireturn}")
    print(f"    • Normal calc voxel size: {normal_calc_voxel_size}")
    # ========== Phase 2: Dataset Discovery & Loading ==========
    print(f"\n[2] Loading dataset:")
    
    
# ---------------- dataset & filters ----------------
    selected = _select_files_with_both_filters(intersections_folder, scan_ids, voxel_sizes)

    # Columns we need (exactly those referenced in your calculations)
    cols = [
        'voxel_size','voxel_id','voxel_cx','voxel_cy','voxel_cz',
        'scan_id','ray_id',
        't_entry_x','t_entry_y','t_entry_z',
        't_exit_x','t_exit_y','t_exit_z',
        'distance_to_centre',
        'point_x','point_y','point_z',
        'echo_intensity','return_number','number_of_returns',
        'normal_x','normal_y','normal_z','point_weight',
        'viewing_angle','hit_type','is_leaf'
    ]

    df, included_scan_ids = _load_intersections_with_injected_metadata(
        selected_files=selected,
        required_columns=cols
    )

    if df.empty:
        if output_dir is None:
            output_dir = intersections_folder
        schema = voxel_metrics_schema_multireturn if is_multireturn else voxel_metrics_schema_singlereturn
        return _gen_dataframe(schema)

    n_voxels = df["voxel_id"].nunique()
    n_rows = len(df)
    print(f"    • Loaded {n_rows:,} rows across {n_voxels:,} unique voxels from {len(included_scan_ids)} scan(s)")

    
# ======================================================================
    # NEW: Compute plane-fitting normals & weights from the current dataset
    # ======================================================================
    # Only rows with actual hit points are usable for plane fitting
    print(f"\n[2b] Computing normals & weights for leaf/wood hits:")
    hit_mask = (df["hit_type"].to_numpy() == 2)
    finite_pts = np.isfinite(df[["point_x","point_y","point_z"]].to_numpy()).all(axis=1)
    usable = hit_mask & finite_pts

    if debug:
        log(f"[normals] usable hit points: {int(usable.sum())} / {len(df)}")

    # Split by leaf / wood (separate fits, as requested)
    leaf_mask = usable & (df["is_leaf"].to_numpy() == True)
    wood_mask = usable & (df["is_leaf"].to_numpy() == False)

    # Prepare output cols (ensure present)
    if "normal_x" not in df.columns: df["normal_x"] = np.nan
    if "normal_y" not in df.columns: df["normal_y"] = np.nan
    if "normal_z" not in df.columns: df["normal_z"] = np.nan
    if "point_weight" not in df.columns: df["point_weight"] = np.nan

    def _fit_and_assign(mask: np.ndarray):
        if mask.sum() == 0:
            return
        idx = np.nonzero(mask)[0]
        pts = df.loc[mask, ["point_x","point_y","point_z"]].to_numpy(np.float64, copy=False)

        # Use unique points (within distance threshold 0.001) and index normals back to original rows
        unique_pts, unique_indices = np.unique(pts, axis=0, return_index=True)
        pts = unique_pts
        idx = idx[unique_indices]

        # Use unique 
        normals, weights = compute_normals_weights_from_points_parallel(
            pts,
            voxel_size=normal_calc_voxel_size,
            n_jobs=-1
        )
        # Assign back
        df.loc[idx, "normal_x"] = normals[:,0]
        df.loc[idx, "normal_y"] = normals[:,1]
        df.loc[idx, "normal_z"] = normals[:,2]
        df.loc[idx, "point_weight"] = weights

    # Leaf and wood computed independently (so quality reflects scan coverage per subset)
    _fit_and_assign(leaf_mask)
    _fit_and_assign(wood_mask)

    if debug:
        # Print out head for leaf_mask and wood_mask, just points, normals, and point_weight
        print("\n    Sample of computed normals & weights for leaf hits:")
        print(df.loc[leaf_mask, ["point_x","point_y","point_z","normal_x","normal_y","normal_z","point_weight"]].head(10))
        print("\n    Sample of computed normals & weights for wood hits:")
        print(df.loc[wood_mask, ["point_x","point_y","point_z","normal_x","normal_y","normal_z","point_weight"]].head(10))

    # ========== Phase 3: Task Decomposition ==========
    print(f"\n[3] Decomposing into voxel blocks:")
    vox_ids = df["voxel_id"].to_numpy()
    boundaries = np.flatnonzero(np.diff(vox_ids)) + 1
    starts = np.r_[0, boundaries]
    ends   = np.r_[boundaries, len(df)]

    tasks: List[Tuple[int,int]] = []
    if voxel_block_rows_hint and voxel_block_rows_hint > 0:
        acc = 0; bstart = 0
        for i, (s,e) in enumerate(zip(starts, ends)):
            acc += (e - s)
            if acc >= voxel_block_rows_hint:
                tasks.append((bstart, i+1))
                bstart = i+1; acc = 0
        if bstart < len(starts):
            tasks.append((bstart, len(starts)))
    else:
        tasks = [(i, i+1) for i in range(len(starts))]

    n_tasks = len(tasks)
    print(f"    • Created {n_tasks} parallel task(s)")
    print(f"    • Voxel groups per task: 1–{max(e-a for a,e in tasks)} (average ~{n_voxels/n_tasks:.1f})")

    # ========== Phase 4: Parallel Computation ==========
    print(f"\n[4] Computing metrics (parallel, {n_workers} workers):")

    def _process_range(a_idx: int, b_idx: int) -> List[pd.DataFrame]:
        rows: List[pd.DataFrame] = []
        for gi in range(a_idx, b_idx):
            s, e = starts[gi], ends[gi]
            block = df.iloc[s:e]
            rows.append(
                _metrics_for_voxel_block(
                    block,
                    average_leaf_area=average_leaf_area,
                    is_multireturn=is_multireturn,
                    beam_divergence_mrad=beam_divergence,
                    epsilon=epsilon
                )
            )
        return rows

    prefer = "processes"
    with tqdm(
        total=n_tasks,
        desc="    Progress",
        unit=" task",
        ncols=90,
        leave=True,
        position=0
    ) as pbar:
        results_nested = Parallel(
            n_jobs=-1, prefer=prefer, batch_size="auto", verbose=0
        )(
            tqdm(
                (delayed(_process_range)(a,b) for (a,b) in tasks),
                total=n_tasks,
                leave=False,
            )
        )
        pbar.update(n_tasks)

    out_frames = [row for sub in results_nested for row in sub]
    voxel_metrics_df = pd.concat(out_frames, axis=0, ignore_index=True)

    n_computed = len(voxel_metrics_df)
    print(f"    ✓ Computed metrics for {n_computed:,} voxels")

    # ========== Phase 5: Output ==========
    print(f"\n[5] Writing output:")
    if output_dir is None:
        output_dir = intersections_folder
    os.makedirs(output_dir, exist_ok=True)

    ts = time.strftime('%Y%m%d_%H%M%S')
    n_files = 0
    for vs, g in voxel_metrics_df.groupby("voxel_size", sort=False):
        out_csv = os.path.join(output_dir, f"voxel_metrics_{round(vs,1)}m_{ts}.csv")
        header_comment = f"# Scan IDs: {', '.join(map(str, included_scan_ids))}\n"
        with open(out_csv, "w") as f:
            f.write(header_comment)
        g.to_csv(out_csv, mode="a", index=False)
        n_files += 1
        print(f"    • {os.path.basename(out_csv)} ({len(g):,} rows)")

    print(f"    ✓ Wrote {n_files} CSV file(s)")

    # ========== Summary ==========
    print(f"\n[6] Summary:")
    print(f"    • Total voxels:      {n_computed:,}")
    print(f"    • Total rows input:  {n_rows:,}")
    print(f"    • Unique scan IDs:   {len(included_scan_ids)}")
    print(f"    • Output directory:  {output_dir}")
    print("\n" + "=" * 80)
    print("  ✓ Voxel metrics computation complete\n")

    return voxel_metrics_df


def get_voxel_metrics(
        intersections_folder: str, 
        average_leaf_area: float, 
        output_dir: str | None = None,
        cpus: int | None = None,
        mem: int | None = None,
        scan_ids: list[str] | None = None,
        voxel_sizes: list[float] | None = None,
        optimal_threads: int = 2,
        beam_divergence: float = 0.35, 
        is_multireturn: bool = False, 
        is_leaf_true: bool = True, 
        debug: bool = True, 
        epsilon: float = 1e-9
    ):
    """
    This function will take the voxel_ray_intersection files and calculate the metrics for each voxel.
    It will save the results to a parquet file.

    Args:
        intersections_files (list): List of paths to voxel_ray_intersection files. It can be paths to single .parquet files or a single folder
        average_leaf_area (float): Average leaf area in square meters (m²) used for lambda_1 calculation.
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

    # Configure dask settings
    memory_limit_str, n_workers, threads_per_worker, partition_size_str, temp_dir = _determine_dask_resources(
        cpus=cpus, mem=mem, optimal_threads=optimal_threads
    )
    
    # _start_dask_client(
    #     n_workers=n_workers,
    #     threads_per_worker=threads_per_worker,
    #     memory_limit=memory_limit_str,
    #     memory_target_fraction=0.7,
    #     memory_spill_fraction=0.8,
    #     memory_pause_fraction=0.9,
    #     temp_dir=temp_dir,
    #     processes=True
    # )
    print(f"[get_voxel_metrics] Dask client: workers={n_workers}, threads/worker={threads_per_worker}, mem/worker={memory_limit_str}, partition_size={partition_size_str}")

    # Configure input files
    voxel_intersections_ddf = create_intersections_ddf(intersections_folder, scan_ids=scan_ids, voxel_sizes=voxel_sizes, blocksize_str=partition_size_str)
    if voxel_intersections_ddf is None:
        raise ValueError("No valid voxel_ray_intersection files found or filters excluded all data.")
    
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
        voxel_size = voxel_df['voxel_size'].values[0]

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
            statement= f"Voxel {voxel_id} has no rays."
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
        lambda_1 = calculate_lambda_1(average_leaf_area, voxel_df['voxel_size'].values[0])
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
        bins, liad_vals, _ = calculate_inclination_angle_distribution(
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
        data_df.loc[0, 'voxel_size'] = voxel_size
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

    # Extract requisite information from schema and ddf for density calculations
    schema = voxel_metrics_schema_singlereturn if not is_multireturn else voxel_metrics_schema_multireturn
    meta = _gen_dataframe(schema)
    scan_id_column = "scan_id" if "scan_id" in voxel_intersections_ddf.columns else ("leg_id" if "leg_id" in voxel_intersections_ddf.columns else None)
    included_scan_ids = sorted(voxel_intersections_ddf[scan_id_column].unique().compute().tolist()) if scan_id_column is not None else ["NA"]
    
    # Use map_partitions for faster per-partition processing instead of groupby().apply()
    voxel_metrics_df = voxel_intersections_ddf.map_partitions(
        lambda part: part.groupby('voxel_id', group_keys=True).apply(
            calculate_voxel_metrics_per_voxel,
            include_groups=False
        ),
        meta=meta
    )
    

    print("Computing voxel metrics with Dask (map_partitions)...")
    with ProgressBar():
        voxel_metrics_df = voxel_metrics_df.compute()
    
    voxel_metrics_df = voxel_metrics_df.reset_index(drop=True)

    # Save to csv output per voxel_size with filters listed in a comment in the header
    if output_dir is None:
        output_dir = os.path.dirname(intersections_folder)  # Save in same directory as input files if no output_dir provided

    for vs in voxel_metrics_df['voxel_size'].unique():
        output_file = os.path.join(
            output_dir, 
            f"voxel_metrics_{vs}m_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        )
        subset_df = voxel_metrics_df[voxel_metrics_df['voxel_size'] == vs]
        
        # Get unique scan_ids included in this subset
        header_comment = f"# Scan IDs: {', '.join(map(str, included_scan_ids))}\n"
        
        # Write header comment and dataframe to CSV
        with ProgressBar():
            with open(output_file, 'w') as f:
                f.write(header_comment)
            subset_df.to_csv(output_file, index=False, mode='a')

    _close_dask_client(client=DASK_CLIENT)

    return voxel_metrics_df


def get_voxel_metrics_dask(
    intersections_parquet: str | List[str],
    average_leaf_area: float,
    output_dir: str = None,
    scan_ids: List[int | str] = None,     
    voxel_sizes: List[float | str] = None,
    beam_divergence_mrad: float = 0.35,
    is_multireturn: bool = False,
    is_leaf_true: bool = True,
    epsilon: float = 1e-9,
    n_workers: int = None,
    cpus: int = None,
    mem: str = None,
    optimal_threads: int = 2,
    threads_per_worker: int = None,
    memory_limit: str = "auto",
    blocksize: str = "256MB",
    debug: bool = False
) -> pd.DataFrame:
    """
    Dask-first per-voxel metric computation from a single parquet dataset (or list of paths).

    Parameters
    ----------
    intersections_parquet : str | Sequence[str]
        Path to a parquet dataset/folder or a list of parquet files.
    average_leaf_area : float
        Average leaf area used for lambda_1 calculation. Your original docstring
        defines lambda_1 = average_leaf_area / voxel_size.  # <-- preserves your baseline
    output_dir : str | None
        Directory where per-voxel_size CSVs are written. Defaults to dataset's folder.
    scan_ids : list[str|int] | None
        If provided, filter to these scan_ids (or leg_ids if that's what's present).
    voxel_sizes : list[float] | None
        If provided, filter to these voxel sizes.
    beam_divergence_mrad : float
        Lidar beam divergence in mrad (default 0.35 mrad).
    is_multireturn : bool
        Enable multi-return MLE block.
    is_leaf_true : bool
        If True, use 'is_leaf' mask as-is, else invert it.
    epsilon : float
        Small value to avoid division by zero.
    n_workers, threads_per_worker, memory_limit, blocksize
        Dask cluster settings. Defaults are safe locally; tune for HPC.

    Returns
    -------
    pandas.DataFrame
        Per-voxel metrics. Also written to CSV(s) per voxel_size.
    """
    # Clean filter inputs first
    if scan_ids is not None:
        scan_ids = set(str(sid) for sid in scan_ids)  # Convert to strings for consistent comparison
    if voxel_sizes is not None:
        voxel_sizes = set(float(vs) for vs in voxel_sizes)  # Convert to floats for consistent comparison


    # --- Dask client ---
    memory_limit_str, n_workers, threads_per_worker, partition_size_str, temp_dir = _determine_dask_resources(
        cpus=cpus, mem=mem, optimal_threads=optimal_threads
    )
    client = _start_dask_client(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        memory_limit=memory_limit_str,
        memory_target_fraction=0.7,
        memory_pause_fraction=0.9,
        memory_spill_fraction=0.8,
        temp_dir=temp_dir,
        processes=True
    )

    # --- Read parquet dataset(s) ---
    if isinstance(intersections_parquet, list):
        if len(intersections_parquet) > 1:
            paths = [p for p in intersections_parquet if os.path.exists(p)]
            if len(paths) == 0:
                raise ValueError("No valid parquet files found in provided list.")
            ddfs = [dd.read_parquet(p, engine="pyarrow", blocksize=blocksize) for p in paths]
            ddf = dd.concat(ddfs, axis=0, interleave_partitions=True)
        else:
            if not os.path.exists(intersections_parquet[0]):
                raise ValueError(f"File not found: {intersections_parquet[0]}")
            ddf = dd.read_parquet(intersections_parquet[0], engine="pyarrow", blocksize=blocksize)
    elif isinstance(intersections_parquet, str):
        if not os.path.exists(intersections_parquet):
            raise ValueError(f"File not found: {intersections_parquet}")
        ddf = dd.read_parquet(intersections_parquet, engine="pyarrow", blocksize=blocksize)
    else:
        raise ValueError("intersections_parquet must be a string path or a list of string paths.")

    # --- Filters: scans / legs and voxel_sizes ---
    if scan_ids is not None:
        if "scan_id" in ddf.columns:
            ddf = ddf[ddf["scan_id"].isin(scan_ids)]
        elif "leg_id" in ddf.columns:
            ddf = ddf[ddf["leg_id"].isin(scan_ids)]
        # else: nothing to filter by

    if voxel_sizes is not None and "voxel_size" in ddf.columns:
        ddf = ddf[ddf["voxel_size"].isin(voxel_sizes)]

    # --- Ensure voxel_id exists (or construct a stable surrogate) ---
    if "voxel_id" not in ddf.columns:
        if "voxel_cx" in ddf.columns and "voxel_cy" in ddf.columns and "voxel_cz" in ddf.columns and "voxel_size" in ddf.columns:
            # Construct a stable string key; avoids unsafe hashing collisions across partitions
            ddf = ddf.assign(
                voxel_id=ddf.map_partitions(
                    lambda part: part.apply(
                        lambda row: create_voxel_id(
                            voxel_size=row["voxel_size"],
                            x=row["voxel_cx"],
                            y=row["voxel_cy"],
                            z=row["voxel_cz"]
                        ),
                        axis=1
                    ),
                    meta=("voxel_id", "int64")
                )
            )
        else:
            raise ValueError("Input must have 'voxel_id' or {voxel_cx, voxel_cy, voxel_cz, voxel_size} to derive one.")

    # --- Core boolean masks ---
    is_unbound = ddf["hit_type"].eq(0)
    is_prev    = ddf["hit_type"].eq(1)
    is_curr    = ddf["hit_type"].eq(2)
    is_yet     = ddf["hit_type"].eq(3)
    leaf_eff   = ddf["is_leaf"] if is_leaf_true else ~ddf["is_leaf"]

    valid_ray_mask = is_unbound | is_curr | is_yet  # rays that contribute to pgap/I

    # --- Geometric path length PL = ||exit - entry|| ---
    dx = ddf["t_exit_x"] - ddf["t_entry_x"]
    dy = ddf["t_exit_y"] - ddf["t_entry_y"]
    dz = ddf["t_exit_z"] - ddf["t_entry_z"]
    ddf = ddf.assign(path_length=(dx*dx + dy*dy + dz*dz)**0.5)

    # --- Free path length (FPL) ---
    # Start with NaN everywhere; fill by cases
    ddf = ddf.assign(free_path_length=np.nan)
    # Unbound: FPL = PL
    ddf["free_path_length"] = ddf["free_path_length"].where(~is_unbound, ddf["path_length"])

    # Single-return logic
    is_sr = ddf["number_of_returns"].eq(1)
    pex = ddf["point_x"] - ddf["t_entry_x"]
    pey = ddf["point_y"] - ddf["t_entry_y"]
    pez = ddf["point_z"] - ddf["t_entry_z"]
    fpl_sr_curr = (pex*pex + pey*pey + pez*pez)**0.5
    # Current hits (SR): entry -> point
    ddf["free_path_length"] = ddf["free_path_length"].where(~(is_sr & is_curr), fpl_sr_curr)
    # Yet-to-hit (SR): FPL = PL
    ddf["free_path_length"] = ddf["free_path_length"].where(~(is_sr & is_yet), ddf["path_length"])
    # Previous hit (SR): leave NaN

    # Multi-return logic: join current hits to previous current hit (rn-1),
    # and yet-to-hit to last current hit (rn-1).
    if is_multireturn:
        is_mr = ddf["number_of_returns"].gt(1)

        # Tag rows
        ddf = ddf.reset_index(drop=False).rename(columns={"index": "_row_id"})

        # Current (MR): distance to previous current point, else entry -> point
        df_curr = ddf[is_curr & is_mr][[
            "_row_id", "voxel_id", "ray_id", "return_number",
            "point_x", "point_y", "point_z",
            "t_entry_x", "t_entry_y", "t_entry_z"
        ]]
        curr_prevkey = df_curr.assign(rn_prev=df_curr["return_number"] - 1)[[
            "_row_id", "voxel_id", "ray_id", "rn_prev",
            "point_x", "point_y", "point_z",
            "t_entry_x", "t_entry_y", "t_entry_z"
        ]]
        prev_points = df_curr.rename(columns={
            "return_number": "rn_prev",
            "point_x": "prev_x",
            "point_y": "prev_y",
            "point_z": "prev_z",
        })[["voxel_id", "ray_id", "rn_prev", "prev_x", "prev_y", "prev_z"]]

        curr_joined = curr_prevkey.merge(prev_points, on=["voxel_id","ray_id","rn_prev"], how="left")
        has_prev = curr_joined["prev_x"].notnull()
        dpx = curr_joined["point_x"] - curr_joined["prev_x"]
        dpy = curr_joined["point_y"] - curr_joined["prev_y"]
        dpz = curr_joined["point_z"] - curr_joined["prev_z"]
        dist_prev = (dpx*dpx + dpy*dpy + dpz*dpz)**0.5
        dex = curr_joined["point_x"] - curr_joined["t_entry_x"]
        dey = curr_joined["point_y"] - curr_joined["t_entry_y"]
        dez = curr_joined["point_z"] - curr_joined["t_entry_z"]
        dist_entry = (dex*dex + dey*dey + dez*dez)**0.5
        fpl_curr_vals = dist_prev.where(has_prev, dist_entry)

        ddf = ddf.merge(curr_joined[["_row_id"]].assign(fpl_curr=fpl_curr_vals), on="_row_id", how="left")
        ddf["free_path_length"] = ddf["free_path_length"].where(~(is_curr & is_mr), ddf["fpl_curr"])
        ddf = ddf.drop(columns=["fpl_curr"])

        # Yet-to-hit (MR): distance from last current hit to exit
        df_yet = ddf[is_yet & is_mr][["_row_id","voxel_id","ray_id","return_number","t_exit_x","t_exit_y","t_exit_z"]]
        yet_prevkey = df_yet.assign(rn_prev=df_yet["return_number"] - 1)[["_row_id","voxel_id","ray_id","rn_prev","t_exit_x","t_exit_y","t_exit_z"]]
        prev_curr_points = df_curr.rename(columns={
            "return_number": "rn_prev",
            "point_x": "last_x",
            "point_y": "last_y",
            "point_z": "last_z",
        })[["voxel_id","ray_id","rn_prev","last_x","last_y","last_z"]]
        yet_joined = yet_prevkey.merge(prev_curr_points, on=["voxel_id","ray_id","rn_prev"], how="left")
        lex = yet_joined["t_exit_x"] - yet_joined["last_x"]
        ley = yet_joined["t_exit_y"] - yet_joined["last_y"]
        lez = yet_joined["t_exit_z"] - yet_joined["last_z"]
        fpl_yet_vals = (lex*lex + ley*ley + lez*lez)**0.5

        ddf = ddf.merge(yet_joined[["_row_id"]].assign(fpl_yet=fpl_yet_vals), on="_row_id", how="left")
        ddf["free_path_length"] = ddf["free_path_length"].where(~(is_yet & is_mr), ddf["fpl_yet"])
        ddf = ddf.drop(columns=["fpl_yet"])

    # --- Persist the expanded columns (recommended) ---
    try:
        ddf = ddf.persist()
    except Exception:
        pass

    # --- Per-voxel basic aggregates ---
    # num_rays among valid_ray_mask
    valid_cols = ddf[["voxel_id","ray_id","path_length","free_path_length"]][valid_ray_mask]
    num_rays = valid_cols.groupby("voxel_id")["ray_id"].nunique(split_every=64).rename("num_rays")

    # counts for hits and leaf hits
    curr_lw = ddf[["voxel_id","viewing_angle","free_path_length"]][is_curr]
    curr_leaf = ddf[["voxel_id","viewing_angle","free_path_length"]][is_curr & leaf_eff]

    num_hits = curr_lw.groupby("voxel_id").size().rename("num_hits")
    num_leaf_hits = curr_leaf.groupby("voxel_id").size().rename("num_leaf_hits")

    # mean viewing angles (all current vs leaf-only current)
    mean_angle_all  = curr_lw.groupby("voxel_id")["viewing_angle"].mean().rename("mean_angle_all")
    mean_angle_leaf = curr_leaf.groupby("voxel_id")["viewing_angle"].mean().rename("mean_angle_leaf")

    # path length aggregates
    mean_pl = valid_cols.groupby("voxel_id")["path_length"].mean().rename("mean_path_length")
    sum_pl  = valid_cols.groupby("voxel_id")["path_length"].sum().rename("sum_path_length")

    # free path length aggregates
    mean_fpl = valid_cols.groupby("voxel_id")["free_path_length"].mean().rename("mean_free_path_length")
    sum_fpl  = valid_cols.groupby("voxel_id")["free_path_length"].sum().rename("sum_free_path_length")
    sum_fpl_hit      = curr_lw.groupby("voxel_id")["free_path_length"].sum().rename("sum_free_path_length_hit")
    sum_fpl_hit_leaf = curr_leaf.groupby("voxel_id")["free_path_length"].sum().rename("sum_free_path_length_hit_leaf")
    sum_fpl_exit     = ddf[["voxel_id","free_path_length"]][is_yet].groupby("voxel_id")["free_path_length"].sum().rename("sum_free_path_length_exit")

    # voxel coords (first), voxel_size and scan id (if present)
    base_cols = ["voxel_id","voxel_cx","voxel_cy","voxel_cz","voxel_size"]
    first_ddf = ddf[[c for c in base_cols if c in ddf.columns]].drop_duplicates(subset=["voxel_id"])
    vox_cx = first_ddf.set_index("voxel_id")["voxel_cx"].rename("voxel_cx") if "voxel_cx" in first_ddf.columns else None
    vox_cy = first_ddf.set_index("voxel_id")["voxel_cy"].rename("voxel_cy") if "voxel_cy" in first_ddf.columns else None
    vox_cz = first_ddf.set_index("voxel_id")["voxel_cz"].rename("voxel_cz") if "voxel_cz" in first_ddf.columns else None
    vox_vs = first_ddf.set_index("voxel_id")["voxel_size"].astype("float32").rename("voxel_size") if "voxel_size" in first_ddf.columns else None

    scan_col = "scan_id" if "scan_id" in ddf.columns else ("leg_id" if "leg_id" in ddf.columns else None)
    vox_scan = None
    if scan_col:
        vox_scan = ddf[["voxel_id", scan_col]].drop_duplicates(subset=["voxel_id"]).set_index("voxel_id")[scan_col].rename(scan_col)

    # --- Per-voxel aggregates computed above ---
    # (We already have: num_rays, num_hits, num_leaf_hits, mean_angle_all, mean_angle_leaf,
    #  mean_pl, sum_pl, mean_fpl, sum_fpl, sum_fpl_hit, sum_fpl_hit_leaf, sum_fpl_exit,
    #  vox_cx, vox_cy, vox_cz, vox_vs, vox_scan [optional])

    def _to_df(obj, name=None):
        """Ensure a Dask DataFrame with a 'voxel_id' column for merging."""
        if isinstance(obj, dd.Series):
            df = obj.to_frame(name or obj.name)
        else:
            df = obj
        # index -> column
        return df.reset_index()  # creates 'voxel_id'

    # Start from num_rays (guaranteed to exist)
    voxel_tbl = _to_df(num_rays)  # columns: ["voxel_id", "num_rays"]

    # Progressive outer merges on 'voxel_id' (robust to unknown divisions)
    def _m(vox, other):
        return vox.merge(_to_df(other), on="voxel_id", how="outer")

    voxel_tbl = _m(voxel_tbl, num_hits)
    voxel_tbl = _m(voxel_tbl, num_leaf_hits)
    voxel_tbl = _m(voxel_tbl, mean_angle_all)
    voxel_tbl = _m(voxel_tbl, mean_angle_leaf)
    voxel_tbl = _m(voxel_tbl, mean_pl)
    voxel_tbl = _m(voxel_tbl, sum_pl)
    voxel_tbl = _m(voxel_tbl, mean_fpl)
    voxel_tbl = _m(voxel_tbl, sum_fpl)
    voxel_tbl = _m(voxel_tbl, sum_fpl_hit)
    voxel_tbl = _m(voxel_tbl, sum_fpl_hit_leaf)
    voxel_tbl = _m(voxel_tbl, sum_fpl_exit)

    # Optional geometry/metadata columns if present
    for opt in (vox_cx, vox_cy, vox_cz, vox_vs):
        if opt is not None:
            voxel_tbl = _m(voxel_tbl, opt)
    if vox_scan is not None:
        voxel_tbl = _m(voxel_tbl, vox_scan)

    # pgap & I (now as column ops)
    voxel_tbl["pgap_lw"]   = (voxel_tbl["num_rays"] - voxel_tbl["num_hits"]) / (voxel_tbl["num_rays"] + epsilon)
    voxel_tbl["pgap_leaf"] = (voxel_tbl["num_rays"] - voxel_tbl["num_leaf_hits"]) / (voxel_tbl["num_rays"] + epsilon)
    voxel_tbl["I_lw"]   = 1.0 - voxel_tbl["pgap_lw"]
    voxel_tbl["I_leaf"] = 1.0 - voxel_tbl["pgap_leaf"]

    # pgap & I
    voxel_tbl["pgap_lw"]   = (voxel_tbl["num_rays"] - voxel_tbl["num_hits"]) / (voxel_tbl["num_rays"] + epsilon)
    voxel_tbl["pgap_leaf"] = (voxel_tbl["num_rays"] - voxel_tbl["num_leaf_hits"]) / (voxel_tbl["num_rays"] + epsilon)
    voxel_tbl["I_lw"]   = 1.0 - voxel_tbl["pgap_lw"]
    voxel_tbl["I_leaf"] = 1.0 - voxel_tbl["pgap_leaf"]

    # --- lambda_1 and effective path lengths ---
    # Your docstring: lambda_1 = average_leaf_area / voxel_size  (keep exact semantics)
    if "voxel_size" in voxel_tbl.columns:
        voxel_tbl["lambda_1"] = average_leaf_area / (voxel_tbl["voxel_size"] + epsilon)  # per-voxel
    else:
        # Fallback if voxel_size missing; let user pass average lambda_1 via average_leaf_area
        voxel_tbl["lambda_1"] = float(average_leaf_area)

    # Broadcast lambda_1 to rows, then compute effective lengths using your validated helper
    d_lambda = voxel_tbl[["lambda_1"]].reset_index().rename(columns={"index": "voxel_id"})
    ddf = ddf.merge(d_lambda, on="voxel_id", how="left")

    # Use map_partitions to call your vectorized numpy helper across partitions
    # --- Prepare meta with the NEW columns declared ---
    meta = ddf._meta.assign(
        eff_path_length=np.float64(),
        eff_free_path_length=np.float64(),
    )

    def _add_effective_lengths(pdf: pd.DataFrame) -> pd.DataFrame:
        lam = pdf["lambda_1"].to_numpy()
        pl = pdf["path_length"].to_numpy(dtype=np.float64)
        fpl = pdf["free_path_length"].to_numpy(dtype=np.float64)
        pdf["eff_path_length"] = calculate_effective_path_length(pl, lam)
        pdf["eff_free_path_length"] = calculate_effective_path_length(fpl, lam)
        return pdf

    ddf = ddf.map_partitions(_add_effective_lengths, meta=meta)

    # Aggregate effective-length stats
    ecols = ddf[["voxel_id","eff_path_length","eff_free_path_length","hit_type"]]
    mean_eff_pl = ecols.groupby("voxel_id")["eff_path_length"].mean().rename("mean_eff_path_length")
    var_eff_pl  = ecols.groupby("voxel_id")["eff_path_length"].var().rename("var_eff_path_length")
    sum_eff_pl  = ecols.groupby("voxel_id")["eff_path_length"].sum().rename("sum_eff_path_length")
    mean_eff_fpl = ecols.groupby("voxel_id")["eff_free_path_length"].mean().rename("mean_eff_free_path_length")
    var_eff_fpl  = ecols.groupby("voxel_id")["eff_free_path_length"].var().rename("var_eff_free_path_length")
    sum_eff_fpl  = ecols.groupby("voxel_id")["eff_free_path_length"].sum().rename("sum_eff_free_path_length")

    eff_lw   = ddf[["voxel_id","eff_free_path_length"]][is_curr]
    eff_exit = ddf[["voxel_id","eff_free_path_length"]][is_yet]
    eff_leaf = ddf[["voxel_id","eff_free_path_length"]][is_curr & leaf_eff]
    sum_eff_fpl_hit      = eff_lw.groupby("voxel_id")["eff_free_path_length"].sum().rename("sum_eff_free_path_length_hit")
    sum_eff_fpl_exit     = eff_exit.groupby("voxel_id")["eff_free_path_length"].sum().rename("sum_eff_free_path_length_exit")
    sum_eff_fpl_hit_leaf = eff_leaf.groupby("voxel_id")["eff_free_path_length"].sum().rename("sum_eff_free_path_length_hit_leaf")

    voxel_tbl = voxel_tbl.join(dd.concat(
        [mean_eff_pl, var_eff_pl, sum_eff_pl,
         mean_eff_fpl, var_eff_fpl, sum_eff_fpl,
         sum_eff_fpl_hit, sum_eff_fpl_exit, sum_eff_fpl_hit_leaf], axis=1))

    # --- LIAD & G(·): keep semantics with a small groupby.apply ---
    # This mirrors your working function’s per-voxel logic for leaf angular distribution,
    # then computes G for (all hits) and (leaf hits).
    def _compute_liad_and_g(pdf: pd.DataFrame) -> pd.DataFrame:
        out = {
            "G_lw": np.nan,
            "G_leaf": np.nan,
        }
        # current hits
        curr_mask = (pdf["hit_type"] == 2)
        leaf_mask = curr_mask & (pdf["is_leaf"] if is_leaf_true else ~pdf["is_leaf"])

        # LIAD from leaf normals/weights (on current-leaf echoes)
        leaf_normals = pdf.loc[leaf_mask, ["normal_x","normal_y","normal_z"]].to_numpy(dtype=np.float64)
        leaf_weights = pdf.loc[leaf_mask, "point_weight"].to_numpy(dtype=np.float64)
        if leaf_normals.size == 0:
            # Fill LIAD bins with NaN
            for i, centre in enumerate(np.arange(2.5, 90.0, 5.0)):
                out[f"LIAD_leaf_bin_{centre:.1f}"] = np.nan
            return pd.DataFrame([out])

        bins, liad_values, _ = calculate_inclination_angle_distribution(normals=leaf_normals, weights=leaf_weights)
        # store liad bins
        for i, centre in enumerate(bins):
            out[f"LIAD_leaf_bin_{centre:.1f}"] = float(liad_values[i]) if i < len(liad_values) else np.nan

        view_angles_lw   = pdf.loc[curr_mask, "viewing_angle"].to_numpy(dtype=np.float64)
        view_angles_leaf = pdf.loc[leaf_mask, "viewing_angle"].to_numpy(dtype=np.float64)

        G_leaf = calculate_G(viewing_angles=view_angles_leaf, bin_centres=bins, LIAD_values=liad_values)
        G_lw   = calculate_G(viewing_angles=view_angles_lw,   bin_centres=bins, LIAD_values=liad_values)
        # both functions can return ndarray or scalar; use mean if array
        out["G_leaf"] = float(np.nanmean(G_leaf)) if hasattr(G_leaf, "__len__") else (np.nan if not np.isfinite(G_leaf) else float(G_leaf))
        out["G_lw"]   = float(np.nanmean(G_lw))   if hasattr(G_lw, "__len__")   else (np.nan if not np.isfinite(G_lw)   else float(G_lw))
        return pd.DataFrame([out])

    # meta for apply
    liad_meta = { "G_lw": "float64", "G_leaf": "float64" }
    for centre in np.arange(2.5, 90.0, 5.0):
        liad_meta[f"LIAD_leaf_bin_{centre:.1f}"] = "float32"

    # apply per voxel
    liad_g = ddf.groupby("voxel_id").apply(_compute_liad_and_g, meta=pd.DataFrame(columns=list(liad_meta.keys())).astype(liad_meta))
    # groupby.apply returns a frame-of-frames; normalize
    liad_g = liad_g.reset_index(drop=True)
    voxel_tbl = voxel_tbl.join(liad_g)

    # --- Multi-return MLE block (vectorized joins + groupby) ---
    if is_multireturn:
        beam_div_rad = beam_divergence_mrad * 1e-3  # mrad -> rad
        radius = ddf["distance_to_centre"].where(valid_ray_mask, np.nan) * beam_div_rad
        beam_area = math.pi * (radius ** 2)
        ddf = ddf.assign(_beam_area_all=beam_area)

        key = ["voxel_id", "ray_id"]
        ray_w = 1.0 / ddf["number_of_returns"].clip(lower=1)
        # sums per (voxel, ray)
        w_hit   = (ray_w.where(is_curr, 0.0)).groupby(key).sum().rename("sum_w_hit")
        w_fpl   = ((ddf["free_path_length"] * ray_w).where(is_curr, 0.0)).groupby(key).sum().rename("sum_w_fpl_hit")
        pl_exit = ((ddf["path_length"] * ray_w).where(is_yet, 0.0)).groupby(key).sum()
        pl_unbd = (ddf["path_length"].where(is_unbound, 0.0)).groupby(key).sum()
        sum_exit = (pl_exit + pl_unbd).rename("sum_exit")
        area_pr = ddf["_beam_area_all"].groupby(key).mean().rename("pulse_area")

        per_ray = dd.concat([area_pr, w_hit, w_fpl, sum_exit], axis=1)

        sum_ba_hit = (per_ray["pulse_area"] * per_ray["sum_w_hit"]).groupby("voxel_id").sum().rename("sum_ba_hit")
        sum_pl_all = (per_ray["pulse_area"] * (per_ray["sum_w_fpl_hit"] + per_ray["sum_exit"])).groupby("voxel_id").sum().rename("sum_pl_all")

        # lambda_1 correction with effective lengths
        eff_w_fpl = ((ddf["eff_free_path_length"] * ray_w).where(is_curr, 0.0)).groupby(key).sum().rename("sum_w_eff_fpl_hit")
        eff_exit_w = ((ddf["eff_path_length"] * ray_w).where(is_yet, 0.0)).groupby(key).sum()
        eff_exit_u = (ddf["eff_path_length"].where(is_unbound, 0.0)).groupby(key).sum()
        sum_eff_exit = (eff_exit_w + eff_exit_u).rename("sum_eff_exit")

        per_ray_eff = dd.concat([area_pr, eff_w_fpl, sum_eff_exit], axis=1)
        sum_pl_all_eff = (per_ray_eff["pulse_area"] * (per_ray_eff["sum_w_eff_fpl_hit"] + per_ray_eff["sum_eff_exit"])).groupby("voxel_id").sum().rename("sum_pl_all_eff")

        # bias corrections
        w_enter = (ray_w.where(is_curr | is_yet, 0.0)).groupby(key).sum() + (is_unbound.astype("int8")).groupby(key).sum()
        per_ray_enter = dd.concat([area_pr, w_enter.rename("sum_w_enter")], axis=1)
        bias_pt1 = (per_ray_enter["pulse_area"] * per_ray_enter["sum_w_enter"]).groupby("voxel_id").sum() / (num_rays + epsilon)

        uniq_sum_w_fpl_hit = (per_ray["pulse_area"] * per_ray["sum_w_fpl_hit"]).groupby("voxel_id").sum()
        bias_pt2 = uniq_sum_w_fpl_hit / (sum_pl_all + epsilon)
        bias_corr = (bias_pt1 * bias_pt2).rename("bias_corr")

        uniq_sum_w_eff_fpl_hit = (per_ray_eff["pulse_area"] * per_ray_eff["sum_w_eff_fpl_hit"]).groupby("voxel_id").sum()
        bias_pt2_eff = uniq_sum_w_eff_fpl_hit / (sum_pl_all_eff + epsilon)
        bias_corr_eff = (bias_pt1 * bias_pt2_eff).rename("bias_corr_eff")

        # Plug into your estimator using the per-voxel G_leaf we computed above
        # (voxel_tbl already has G_leaf joined)
        # We call MLE_vincent_2021 in a vector-friendly way by deferring to pandas at the end.
        voxel_tbl = voxel_tbl.join(dd.concat([
            sum_ba_hit, sum_pl_all, sum_pl_all_eff, bias_corr, bias_corr_eff
        ], axis=1))

        # Compute the four LAD variants at the very end on pandas (post-compute) to reuse your helper.
        compute_lads_at_end = True
    else:
        compute_lads_at_end = False

    # --- Compute to pandas ---
    voxel_df = voxel_tbl.reset_index().rename(columns={"index": "voxel_id"}).compute()
    voxel_df = voxel_df.reset_index(drop=True)

    # LAD (post-compute) if requested
    if compute_lads_at_end:
        def _lad_variants(row):
            try:
                lad_nocorr = MLE_vincent_2021(
                    sum_ba_hit=row["sum_ba_hit"],
                    sum_pl_all=row["sum_pl_all"],
                    G=row["G_leaf"],
                    CI=1.0,
                    bias_corr=None,
                )
            except Exception:
                lad_nocorr = np.nan
            try:
                lad_lambda1 = MLE_vincent_2021(
                    sum_ba_hit=row["sum_ba_hit"],
                    sum_pl_all=row["sum_pl_all_eff"],
                    G=row["G_leaf"],
                    CI=1.0,
                    bias_corr=None,
                )
            except Exception:
                lad_lambda1 = np.nan
            try:
                lad_bias = MLE_vincent_2021(
                    sum_ba_hit=row["sum_ba_hit"],
                    sum_pl_all=row["sum_pl_all"],
                    G=row["G_leaf"],
                    CI=1.0,
                    bias_corr=row["bias_corr"],
                )
            except Exception:
                lad_bias = np.nan
            try:
                lad_lambda1_bias = MLE_vincent_2021(
                    sum_ba_hit=row["sum_ba_hit"],
                    sum_pl_all=row["sum_pl_all_eff"],
                    G=row["G_leaf"],
                    CI=1.0,
                    bias_corr=row["bias_corr_eff"],
                )
            except Exception:
                lad_lambda1_bias = np.nan
            return pd.Series({
                "LAD_MLE_nocorr": lad_nocorr,
                "LAD_MLE_lambda1": lad_lambda1,
                "LAD_MLE_bias": lad_bias,
                "LAD_MLE_lambda1_bias": lad_lambda1_bias,
            })
        voxel_df = pd.concat([voxel_df, voxel_df.apply(_lad_variants, axis=1)], axis=1)

    # --- Save per-voxel_size CSVs ---
    if output_dir is None:
        # default to the folder of the first dataset path
        output_dir = os.path.dirname(paths[0]) if os.path.isdir(paths[0]) else os.path.dirname(paths[0])
    os.makedirs(output_dir, exist_ok=True)

    time_tag = time.strftime("%Y%m%d_%H%M%S")
    if "voxel_size" in voxel_df.columns:
        for vs, sub in voxel_df.groupby("voxel_size", dropna=False):
            if pd.isna(vs):
                name = f"voxel_metrics_{time_tag}.csv"
            else:
                name = f"voxel_metrics_{vs}m_{time_tag}.csv"
            out_path = os.path.join(output_dir, name)

            # Header with scan IDs if available
            if scan_col and scan_col in voxel_df.columns:
                ids = sorted({i for i in sub[scan_col].dropna().unique().tolist()})
                header = f"# Scan IDs: {', '.join(map(str, ids))}\n"
            else:
                header = "# Scan IDs: (not available)\n"

            with open(out_path, "w") as f:
                f.write(header)
            sub.to_csv(out_path, index=False, mode="a")
    else:
        # Single file if voxel_size unknown
        out_path = os.path.join(output_dir, f"voxel_metrics_{time_tag}.csv")
        voxel_df.to_csv(out_path, index=False)

    _close_dask_client(client=DASK_CLIENT)
    return voxel_df


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

    # voxel_grouped = voxel_intersections_df.groupby('voxel_id')
    # first_voxel_id = voxel_intersections_df['voxel_id'].values[0]
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
    
    if len(files) == 0:
        raise ValueError("No valid rays parquet files found.")
    
    print(f"Loading {len(files)} files...")
    valid_ray_dir = os.path.dirname(files[0])
    
    # Use lazy dask computation
    dfs = [dd.read_parquet(file, engine='pyarrow') for file in files]
    valid_rays_ddf = dd.concat(dfs, axis=0, ignore_index=True)
    
    # Define a function to process partitions lazily
    def add_normals_weights_to_partition(partition, knn=6):
        """Add normals and weights to leaf points in a partition."""
        # Select only leaf points
        leaf_mask = partition['is_leaf'] & ~partition['point_x'].isna()
        leaf_points = partition.loc[leaf_mask, ['point_x', 'point_y', 'point_z']].to_numpy()
        
        # Initialize columns with NaN
        partition['normal_x'] = np.nan
        partition['normal_y'] = np.nan
        partition['normal_z'] = np.nan
        partition['point_weight'] = np.nan
        
        # Only compute if there are leaf points
        if len(leaf_points) > 0:
            normals, weights = compute_normals_weights_from_points(points=leaf_points, knn=knn)
            partition.loc[leaf_mask, 'normal_x'] = normals[:, 0]
            partition.loc[leaf_mask, 'normal_y'] = normals[:, 1]
            partition.loc[leaf_mask, 'normal_z'] = normals[:, 2]
            partition.loc[leaf_mask, 'point_weight'] = weights
        
        return partition
    
    print(f"Calculating normals and weights for leaf points (lazy)...")
    # Apply the function to each partition lazily
    valid_rays_ddf = valid_rays_ddf.map_partitions(
        add_normals_weights_to_partition,
        knn=knn,
        meta=valid_rays_ddf._meta.assign(
            normal_x=np.float64(),
            normal_y=np.float64(),
            normal_z=np.float64(),
            point_weight=np.float64()
        )
    )
    
    print("Saving results to parquet files...")
    # Compute all normals and weights once, then save per-leg
    print("Computing normals and weights for all partitions...")
    with ProgressBar():
        valid_rays_computed = valid_rays_ddf.compute()
    
    # Group by scan_id and save each leg separately
    for scan_id in valid_rays_computed['scan_id'].unique():
        output_file = os.path.join(valid_ray_dir, f"leg_{scan_id}_valid_rays.parquet")
        leg_df = valid_rays_computed[valid_rays_computed['scan_id'] == scan_id]
        
        # Remove old file if it exists to allow overwrite
        backup_file = None
        if os.path.exists(output_file):
            backup_file = output_file + ".backup"
            shutil.copy(output_file, backup_file)
        
        try:
            print(f"Saving leg {scan_id} with normals and weights...")
            leg_df.to_parquet(
                output_file,
                engine='pyarrow',
                compression='snappy',
                index=False,
                schema=valid_rays_schema
            )
            # Remove backup if write was successful
            if backup_file and os.path.exists(backup_file):
                os.remove(backup_file)
            print(f"Saved {output_file}")
        except Exception as e:
            # Restore backup if write failed
            if backup_file and os.path.exists(backup_file):
                shutil.move(backup_file, output_file)
            print(f"Error writing {output_file}: {e}")
            raise

        # grouped_df = valid_rays_df.groupby('scan_id')
        # DEBUG_PRINT = False
        # def save_group(group):
        #     nonlocal DEBUG_PRINT
        #     scan_id = group['scan_id'].iloc[0]
        #     output_file = os.path.join(valid_ray_dir, f"leg_{scan_id}_valid_rays.parquet")

        #     if debug and not DEBUG_PRINT:
        #         print("Debugging enabled:")
        #         DEBUG_PRINT = True
        #         print(group[~group['point_x'].isna() & group['is_leaf'] == True].head())

        #     group.to_parquet(output_file, engine='pyarrow', index=False, schema=valid_rays_schema)
        #     output_files.append(output_file)
        #     print(f"Saved {output_file}")

        # grouped_df.apply(save_group, include_groups=True)

        # print(f"Saved {len(output_files)} valid rays files with normals and weights.")

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
        scan_id = df['scan_id'].iloc[0]
        voxel_size = df['voxel_size'].iloc[0]
        valid_rays_file = os.path.join(valid_rays_dir, f'leg_{scan_id}_valid_rays.parquet')
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


def test_helios_settings(helios_dir, use_class, leaf_object_ids, wood_object_ids, output_dir):
    """
    Test helios settings by plotting a sample of points from the helios files.
    
    Args:
        helios_dir (str): Directory containing helios .xyz files.
        use_class (bool): Whether to use classification or hit_object_id for identifying leaf/wood.
        leaf_object_ids (list): List of object IDs corresponding to leaf points.
        wood_object_ids (list): List of object IDs corresponding to wood points.
        valid_rays_dir (str): Directory to save the output plot.

    Returns:
        None

    User can check saved image to verify if leaf and wood points are set correctly.
    """
    import csv
    import glob
    import numpy as np
    import dask
    import matplotlib.pyplot as plt

    # Check classification and object_ids
    helios_files = glob.glob(os.path.join(helios_dir, '*.xyz'))
    if helios_files:
        test_file = helios_files[0]     # Just use the first file
        class_col = 9 if use_class else 8      # if not use_class, assume use hit_object_id
        num_test_points = 1000

        num_rows = 0
        leaf_points = []
        wood_points = []
        other_points = []
        with open(test_file, newline="") as f:
            reader = csv.reader(f, delimiter=' ')
            while num_rows < num_test_points:
                for row in reader:
                    x = float(row[0])
                    y = float(row[1])
                    z = float(row[2])

                    class_id = int(row[class_col])
                    if class_id in leaf_object_ids:
                        leaf_points.append([x,y,z])
                    elif class_id in wood_object_ids:
                        wood_points.append([x,y,z])
                    else:
                        other_points.append([x,y,z])
                    num_rows += 1

        if len(leaf_points) > 0 or len(wood_points) > 0 or len(other_points) > 0:
            # Convert to numpy
            leaf_points = np.array(leaf_points, dtype=np.float32)
            wood_points = np.array(wood_points, dtype=np.float32)
            other_points = np.array(other_points, dtype=np.float32)
            
            # Plot point cloud
            fig = plt.figure(figsize=(10, 6))
            ax = fig.add_subplot(111)

            # Plot leaf points in green
            if leaf_points.size > 0:
                ax.scatter(leaf_points[:, 0], leaf_points[:, 2], c='green', s=1, label='Leaf')

            # Plot wood points in brown
            if wood_points.size > 0:
                ax.scatter(wood_points[:, 0], wood_points[:, 2], c='saddlebrown', s=1, label='Wood')

            # Plot other points in blue
            if other_points.size > 0:
                ax.scatter(other_points[:, 0], other_points[:, 2], c='blue', s=1, label='Other')

            print("Plotting leaf and wood points to check classification...")
            ax.set_xlabel('X')
            ax.set_ylabel('Z')
            ax.set_title(f'Leaf and Wood Point Check - File {os.path.basename(test_file)}')
            ax.legend()

            try:
                os.makedirs(output_dir, exist_ok=True)
                plt.savefig(os.path.join(output_dir, f'file_{os.path.basename(test_file)}_leaf_wood_check.png'))
                plt.close()
                print(f"Saved leaf and wood check plot to {output_dir}")
                return True
            except Exception as e:
                print(f"Error saving plot: {e}")
                return False


#### --- Fix Memory Issues with Processing Helios Sims --- ####
## Shared Resources

# utils_voxel_ray.py
import os, math, shutil, tempfile, psutil, warnings
from dataclasses import dataclass
from typing import Optional, Tuple, List
import numpy as np
import pandas as pd

# ---- Progress helpers -------------------------------------------------------
def log(msg: str): print(msg, flush=True)

# ---- SLURM & local resource discovery --------------------------------------
@dataclass
class Resources:
    n_workers: int
    threads_per_worker: int
    mem_per_worker_mb: int
    partition_blocksize_mb: int
    temp_dir: str

def _find_tempdir() -> str:
    for env in ("TMPDIR", "TMP", "TEMP", "SCRATCH"):
        p = os.environ.get(env)
        if p and os.path.isdir(p):
            return p
    try:
        return tempfile.gettempdir()
    except Exception:
        return "/tmp"

def detect_resources(
    target_threads_per_worker: int = 2,
    mem_fraction: float = 0.70,
    partition_worker_ratio: float = 0.002   # partition ~0.2% of mem/worker
) -> Resources:
    # CPU
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm_cpus:
        phys = int(slurm_cpus)
        logical = max(phys, psutil.cpu_count(logical=True) or phys)
    else:
        phys = psutil.cpu_count(logical=False) or 1
        logical = psutil.cpu_count(logical=True) or phys

    # pick worker & threads layout
    n_workers = max(1, logical // max(1, target_threads_per_worker))
    threads_per_worker = max(1, min(target_threads_per_worker, logical))

    # Memory (MB)
    if os.environ.get("SLURM_MEM_PER_NODE"):
        node_mem_mb = int(float(os.environ["SLURM_MEM_PER_NODE"]))
    elif os.environ.get("SLURM_MEM_PER_CPU") and slurm_cpus:
        node_mem_mb = int(float(os.environ["SLURM_MEM_PER_CPU"])) * int(slurm_cpus)
    else:
        node_mem_mb = int(psutil.virtual_memory().total / (1024 * 1024))

    usable_mb = int(node_mem_mb * mem_fraction)
    mem_per_worker_mb = max(256, usable_mb // n_workers)  # floor

    # Dask blocksize / batch blocksize (heuristic)
    partition_blocksize_mb = max(8, int(mem_per_worker_mb * partition_worker_ratio))

    temp_dir = _find_tempdir()
    return Resources(
        n_workers=n_workers,
        threads_per_worker=threads_per_worker,
        mem_per_worker_mb=mem_per_worker_mb,
        partition_blocksize_mb=partition_blocksize_mb,
        temp_dir=temp_dir,
    )

# ---- Schema helper (names only; schema object is defined in your project) ---
OUTPUT_COLUMNS = [
    'voxel_size','voxel_id','voxel_cx','voxel_cy','voxel_cz',
    'scan_id','ray_id',
    't_entry_x','t_entry_y','t_entry_z',
    't_exit_x','t_exit_y','t_exit_z',
    'distance_to_centre',
    'point_x','point_y','point_z',
    'echo_intensity','return_number','number_of_returns',
    'viewing_angle','hit_type','is_leaf'
]

# ---- Ray utilities ----------------------------------------------------------
def viewing_angle_deg(dxyz: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # angle between +Z and direction, normalized to <= 90°
    dn = np.sqrt(np.sum(dxyz * dxyz, axis=1))
    dn = np.clip(dn, eps, None)
    cos_th = np.clip(dxyz[:, 2] / dn, -1.0, 1.0)
    ang = np.degrees(np.arccos(cos_th))
    return np.where(ang <= 90.0, ang, 180.0 - ang)

def ensure_small_nonzero(arr: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    # replace zeros/small values with signed eps to avoid divide-by-zero
    out = arr.copy()
    mask = np.abs(out) <= eps
    out[mask] = np.where(out[mask] == 0.0, eps, np.sign(out[mask]) * eps)
    return out

# ---- File helpers -----------------------------------------------------------
def list_parquet_files(valid_rays_dir: str) -> List[str]:
    import glob
    return sorted(glob.glob(os.path.join(valid_rays_dir, "*_valid_rays.parquet")))

def leg_from_filename(path: str) -> int:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0].split("_")
    for token in stem:
        if token.isdigit():
            return int(token)
    return 0

def compile_voxel_refs(references_dir: str) -> pd.DataFrame:
    import glob
    dfs = []
    for csvf in sorted(glob.glob(os.path.join(references_dir, "*.csv"))):
        df = pd.read_csv(csvf)
        if "voxel_id" not in df.columns:
            # create_voxel_id(voxel_size, x, y, z) existed in your code base; keep stable
            from hashlib import blake2b
            def create_voxel_id(vs, x, y, z):
                h = blake2b(digest_size=8)
                h.update(np.array([vs, x, y, z], dtype=np.float64).tobytes())
                return int.from_bytes(h.digest(), "little", signed=False)
            vs = float(os.path.splitext(csvf)[0].split("_")[-1])
            df["voxel_id"] = [create_voxel_id(vs, r.voxel_cx, r.voxel_cy, r.voxel_cz) for r in df.itertuples()]
        if "voxel_size" not in df.columns:
            vs = float(os.path.splitext(csvf)[0].split("_")[-1])
            df["voxel_size"] = vs
        dfs.append(df[["voxel_id","voxel_cx","voxel_cy","voxel_cz","voxel_size"]].drop_duplicates())
    if not dfs:
        raise FileNotFoundError(f"No voxel reference CSVs in {references_dir}")
    out = pd.concat(dfs, ignore_index=True)
    return out


### NO DASK VOXEL_RAY_INTERSECTION FUNCTION TO AVOID MEMORY ISSUES; SEE voxel_ray_intersection_dask INSTEAD ###

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import math
from typing import Dict, List, Tuple
import sys
from tqdm import tqdm
from joblib import Parallel, delayed
from multiprocessing import Manager
from threading import Thread

# ------------------------------------------------------------
# Utility: build regular grid metadata from voxel references
# ------------------------------------------------------------
def infer_grid_from_refs(voxel_refs: pd.DataFrame):
    """
    Build a structure:
        size → {
            "voxel_size": s,
            "grid_min": (x0,y0,z0),
            "grid_shape": (nx,ny,nz),
            "sorted_voxels": df sorted by (ix,iy,iz),
            "id_grid": array shaped (nx,ny,nz),
            "cx_grid", "cy_grid", "cz_grid": centers
        }
    """
    out = {}
    for s in sorted(voxel_refs.voxel_size.unique()):
        df = voxel_refs[voxel_refs.voxel_size == s].copy()
        # infer grid_min, grid extents
        xs = np.sort(df.voxel_cx.unique())
        ys = np.sort(df.voxel_cy.unique())
        zs = np.sort(df.voxel_cz.unique())

        nx, ny, nz = len(xs), len(ys), len(zs)

        # voxel center spacing = s
        # grid_min = center - s/2
        gx0 = xs[0] - s*0.5
        gy0 = ys[0] - s*0.5
        gz0 = zs[0] - s*0.5

        # build index mapping
        # compute indices for each voxel
        ix = ((df.voxel_cx - gx0) / s).round().astype(int)
        iy = ((df.voxel_cy - gy0) / s).round().astype(int)
        iz = ((df.voxel_cz - gz0) / s).round().astype(int)

        # Clip indices to valid range to handle rounding errors
        ix = np.clip(ix, 0, nx - 1)
        iy = np.clip(iy, 0, ny - 1)
        iz = np.clip(iz, 0, nz - 1)

        df["ix"] = ix
        df["iy"] = iy
        df["iz"] = iz

        # allocate grids
        id_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
        cx_grid = np.zeros((nx, ny, nz), dtype=np.float64)
        cy_grid = np.zeros((nx, ny, nz), dtype=np.float64)
        cz_grid = np.zeros((nx, ny, nz), dtype=np.float64)

        for r in df.itertuples():
            if 0 <= r.ix < nx and 0 <= r.iy < ny and 0 <= r.iz < nz:
                id_grid[r.ix, r.iy, r.iz] = r.voxel_id
                cx_grid[r.ix, r.iy, r.iz] = r.voxel_cx
                cy_grid[r.ix, r.iy, r.iz] = r.voxel_cy
                cz_grid[r.ix, r.iy, r.iz] = r.voxel_cz

        out[s] = {
            "voxel_size": s,
            "grid_min": np.array([gx0, gy0, gz0], dtype=np.float64),
            "grid_shape": (nx, ny, nz),
            "id_grid": id_grid,
            "cx_grid": cx_grid,
            "cy_grid": cy_grid,
            "cz_grid": cz_grid,
        }
    return out


# ------------------------------------------------------------
# DDA traversal for a single ray through a single grid
# ------------------------------------------------------------
def dda_traverse_single_grid(origin, direction, grid_min, voxel_size, grid_shape):
    """
    Returns list of (ix,iy,iz, t_entry, t_exit)
    direction is assumed normalized or not – traversal uses parametric t.
    """
    ox, oy, oz = origin
    dx, dy, dz = direction

    nx, ny, nz = grid_shape

    # Solve ray-box intersection: [grid_min, grid_max)
    gmin = grid_min
    gmax = grid_min + voxel_size * np.array(grid_shape)

    # Slab intersection
    tmin, tmax = -1e300, 1e300
    for i, (o, d, mn, mx) in enumerate(zip(origin, direction, gmin, gmax)):
        if abs(d) < 1e-12:
            if o < mn or o >= mx:
                return []  # no hit
        else:
            inv = 1.0 / d
            t1 = (mn - o) * inv
            t2 = (mx - o) * inv
            lo = min(t1, t2)
            hi = max(t1, t2)
            tmin = max(tmin, lo)
            tmax = min(tmax, hi)
            if tmax < tmin:
                return []

    if tmax < 0:
        return []

    # start traversal at t = max(tmin, 0)
    t = max(tmin, 0.0)
    start_point = origin + t * direction
    # which voxel?
    ix = int((start_point[0] - gmin[0]) // voxel_size)
    iy = int((start_point[1] - gmin[1]) // voxel_size)
    iz = int((start_point[2] - gmin[2]) // voxel_size)

    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return []

    # Step directions
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1

    # Compute boundary tMax and tDelta
    def axis_params(o, d, c, step):
        # next boundary coord
        if step > 0:
            nextb = gmin[c] + (eval_idx[c] + 1) * voxel_size
        else:
            nextb = gmin[c] + (eval_idx[c]) * voxel_size
        tMax = (nextb - o) / (d if abs(d) > 1e-12 else 1e-12)
        tDelta = voxel_size / (abs(d) if abs(d) > 1e-12 else 1e-12)
        return tMax, tDelta

    hits = []
    eval_idx = [ix, iy, iz]

    tMaxX, tDeltaX = axis_params(ox, dx, 0, step_x)
    tMaxY, tDeltaY = axis_params(oy, dy, 1, step_y)
    tMaxZ, tDeltaZ = axis_params(oz, dz, 2, step_z)

    # traverse until t > tmax
    while 0 <= eval_idx[0] < nx and 0 <= eval_idx[1] < ny and 0 <= eval_idx[2] < nz:
        # which boundary hit next?
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            t_next = tMaxX
            axis = 0
        elif tMaxY <= tMaxZ:
            t_next = tMaxY
            axis = 1
        else:
            t_next = tMaxZ
            axis = 2

        t_entry = t
        t_exit = min(t_next, tmax)

        # record hit
        hits.append((eval_idx[0], eval_idx[1], eval_idx[2], t_entry, t_exit))

        if t_next > tmax:
            break

        # advance
        t = t_next
        if axis == 0:
            eval_idx[0] += step_x
            tMaxX += tDeltaX
        elif axis == 1:
            eval_idx[1] += step_y
            tMaxY += tDeltaY
        else:
            eval_idx[2] += step_z
            tMaxZ += tDeltaZ

    return hits


# ------------------------------------------------------------
# Main function: traversal-first voxel-ray intersections
# ------------------------------------------------------------
def voxel_ray_intersections_nodask(valid_rays_dir: str,
                                            references_dir: str,
                                            *,
                                            epsilon: float = 1e-6,
                                            n_jobs: int = -1,
                                            debug: bool = True):
    """
    Produces a DataFrame with exactly OUTPUT_COLUMNS
    using DDA traversal first, then full per-pair field calculation.
    Processes each parquet file in parallel.
    
    Parameters
    ----------
    valid_rays_dir : str
        Directory containing *_valid_rays.parquet files
    references_dir : str
        Directory containing voxel reference CSVs
    epsilon : float
        Numerical tolerance
    n_jobs : int
        Number of parallel jobs (-1 = all CPUs)
    """
    
    log("=" * 70)
    log("[voxel_ray_intersections_nodask] Starting voxel-ray intersection computation")
    log("=" * 70)
    
    # Load voxel reference CSVs (shared across workers)
    log("\n[1/5] Loading voxel reference CSVs...")
    refdf = compile_voxel_refs(references_dir)
    log(f"  ✓ Loaded {len(refdf)} voxel references")

    # Infer grid geometry per voxel_size (shared across workers)
    log("\n[2/5] Building spatial grids per voxel size...")
    grids = infer_grid_from_refs(refdf)
    for s, g in grids.items():
        nx, ny, nz = g["grid_shape"]
        n_voxels = np.count_nonzero(g["id_grid"] >= 0)
        log(f"  ✓ Grid {s}m: shape {nx}×{ny}×{nz}, {n_voxels} active voxels")

    # Function to process a single file
    def process_file(pf, grids, epsilon, shared_progress=None):
        """Process a single parquet file and return DataFrame with shared progress tracking"""
        leg_id = leg_from_filename(pf)
        out_blocks = []
        file_rays = 0
        file_intersections = 0
        work_completed = 0
        
        pfh = pq.ParquetFile(pf)
        num_rgs = pfh.num_row_groups
        
        num_grids = len(grids)
        
        for rg in range(num_rgs):
            tbl = pfh.read_row_group(
                rg,
                columns=[
                    "scan_id","ray_id",
                    "origin_x","origin_y","origin_z",
                    "direction_x","direction_y","direction_z",
                    "point_x","point_y","point_z",
                    "echo_intensity",
                    "return_number","number_of_returns","is_leaf",
                ]
            ).to_pandas()

            if tbl.empty:
                work_completed += num_grids
                if shared_progress is not None:
                    shared_progress["completed"] += num_grids
                continue

            n_rays_in_rg = len(tbl)
            file_rays += n_rays_in_rg

            # Extract arrays
            scan_id = tbl["scan_id"].fillna(0).astype(np.int64).to_numpy()
            ray_id  = tbl["ray_id"].fillna(0).astype(np.int64).to_numpy()

            origins = tbl[["origin_x","origin_y","origin_z"]].to_numpy(np.float64)
            dirs    = tbl[["direction_x","direction_y","direction_z"]].to_numpy(np.float64)
            pts     = tbl[["point_x","point_y","point_z"]].to_numpy(np.float64)
            echo    = tbl["echo_intensity"].fillna(0.0).to_numpy(np.float64)
            ret_no  = tbl["return_number"].fillna(0).astype(np.int32).to_numpy()
            ret_cnt = tbl["number_of_returns"].fillna(0).astype(np.int32).to_numpy()
            is_leaf = tbl["is_leaf"].fillna(False).astype(bool).to_numpy()
            view_ang = viewing_angle_deg(dirs)

            # unique rays
            _, ur_first, ur_inv = np.unique(ray_id, return_index=True, return_inverse=True)
            uniq_orig = origins[ur_first]
            uniq_dirs = ensure_small_nonzero(dirs[ur_first])

            # Map unique -> originals
            order = np.argsort(ur_inv)
            inv_sorted = ur_inv[order]
            split = np.flatnonzero(np.diff(inv_sorted)) + 1
            uniq_to_orig = np.split(order, split)

            n_uniq_rays = len(uniq_orig)

            # Traverse all voxel size grids for this row group
            for (_, g) in grids.items():
                gs = g["grid_shape"]
                gmin = g["grid_min"]
                vs  = g["voxel_size"]

                idgrid = g["id_grid"]
                cxg = g["cx_grid"]
                cyg = g["cy_grid"]
                czg = g["cz_grid"]

                rg_intersections = 0
                
                for uidx in range(n_uniq_rays):
                    o = uniq_orig[uidx]
                    d = uniq_dirs[uidx]

                    hits = dda_traverse_single_grid(o, d, gmin, vs, gs)
                    if not hits:
                        continue

                    # expand to original rows
                    orig_rows = uniq_to_orig[uidx]

                    # For each voxel hit
                    rows = []
                    for (ix, iy, iz, t0, t1) in hits:
                        voxel_id = idgrid[ix, iy, iz]
                        if voxel_id < 0:
                            continue
                        cx = cxg[ix, iy, iz]
                        cy = cyg[ix, iy, iz]
                        cz = czg[ix, iy, iz]

                        entry_xyz = o + t0 * d
                        exit_xyz  = o + t1 * d

                        # expand per original row
                        for ridx in orig_rows:
                            px, py, pz = pts[ridx]
                            ox0, oy0, oz0 = origins[ridx]

                            # classify hit_type
                            s_half = vs * 0.5
                            vmin = np.array([cx - s_half, cy - s_half, cz - s_half])
                            vmax = np.array([cx + s_half, cy + s_half, cz + s_half])

                            unbound = np.isnan(px) or np.isnan(py) or np.isnan(pz)
                            in_voxel = (vmin[0] - epsilon <= px <= vmax[0] + epsilon and
                                        vmin[1] - epsilon <= py <= vmax[1] + epsilon and
                                        vmin[2] - epsilon <= pz <= vmax[2] + epsilon)

                            dist_entry_sq = (ox0 - entry_xyz[0])**2 + (oy0 - entry_xyz[1])**2 + (oz0 - entry_xyz[2])**2
                            dist_exit_sq  = (ox0 - exit_xyz[0])**2 + (oy0 - exit_xyz[1])**2 + (oz0 - exit_xyz[2])**2
                            dist_point_sq = (px - ox0)**2 + (py - oy0)**2 + (pz - oz0)**2

                            before = (dist_entry_sq > dist_point_sq) and (not in_voxel) and (not unbound)
                            after  = (dist_exit_sq  < dist_point_sq) and (not in_voxel) and (not unbound)

                            if unbound:
                                ht = 0
                            elif before:
                                ht = 1
                            elif in_voxel:
                                ht = 2
                            elif after:
                                ht = 3
                            else:
                                ht = -1

                            dist_to_center = math.sqrt((ox0 - cx)**2 + (oy0 - cy)**2 + (oz0 - cz)**2)

                            rows.append({
                                "voxel_size": float(vs),
                                "voxel_id": int(voxel_id),
                                "voxel_cx": float(cx),
                                "voxel_cy": float(cy),
                                "voxel_cz": float(cz),
                                "scan_id": int(scan_id[ridx]) if not np.isnan(scan_id[ridx]) else 0,
                                "ray_id": int(ray_id[ridx]) if not np.isnan(ray_id[ridx]) else 0,
                                "t_entry_x": float(entry_xyz[0]),
                                "t_entry_y": float(entry_xyz[1]),
                                "t_entry_z": float(entry_xyz[2]),
                                "t_exit_x":  float(exit_xyz[0]),
                                "t_exit_y":  float(exit_xyz[1]),
                                "t_exit_z":  float(exit_xyz[2]),
                                "distance_to_centre": float(dist_to_center),
                                "point_x": float(px) if not np.isnan(px) else np.nan,
                                "point_y": float(py) if not np.isnan(py) else np.nan,
                                "point_z": float(pz) if not np.isnan(pz) else np.nan,
                                "echo_intensity": float(echo[ridx]) if not np.isnan(echo[ridx]) else 0.0,
                                "return_number": int(ret_no[ridx]) if not np.isnan(ret_no[ridx]) else 0,
                                "number_of_returns": int(ret_cnt[ridx]) if not np.isnan(ret_cnt[ridx]) else 0,
                                "viewing_angle": float(view_ang[ridx]) if not np.isnan(view_ang[ridx]) else np.nan,
                                "hit_type": int(ht) if not np.isnan(ht) else -1,
                                "is_leaf": bool(is_leaf[ridx]) if not np.isnan(is_leaf[ridx]) else False
                            })

                    if rows:
                        rg_intersections += len(rows)
                        file_intersections += len(rows)
                        out_blocks.append(pd.DataFrame(rows)[OUTPUT_COLUMNS])
                
                # Update shared progress
                work_completed += 1
                if shared_progress is not None:
                    shared_progress["completed"] += 1
                    shared_progress["intersections"] += rg_intersections
                    shared_progress["rays"] += n_rays_in_rg

        if out_blocks:
            result = pd.concat(out_blocks, ignore_index=True)[OUTPUT_COLUMNS]
        else:
            result = pd.DataFrame({c: pd.Series(dtype="float64") for c in OUTPUT_COLUMNS})[OUTPUT_COLUMNS]
        
        # Update total rays processed at the end
        if shared_progress is not None:
            shared_progress["rays"] += file_rays
        
        return result, file_rays, file_intersections, leg_id

    # List input files
    files = list_parquet_files(valid_rays_dir)
    log(f"\n[3/5] Processing {len(files)} parquet file(s) in parallel...")
    
    # Use Manager to share progress across processes
    manager = Manager()
    shared_progress = manager.dict()
    shared_progress["completed"] = 0
    shared_progress["intersections"] = 0
    shared_progress["rays"] = 0
    
    num_grids = len(grids)
    total_work_units = num_grids * sum(
        pq.ParquetFile(pf).num_row_groups for pf in files
    )
    
    # Start aggregated progress bar in main thread
    def progress_monitor():
        """Monitor shared progress dict and update single pbar"""
        with tqdm(
            total=total_work_units,
            desc="Overall Progress",
            unit=" grid traversals",
            ncols=100,
            leave=True
        ) as pbar:
            last_completed = 0
            while last_completed < total_work_units:
                current = shared_progress.get("completed", 0)
                if current > last_completed:
                    pbar.update(current - last_completed)
                    intersections = shared_progress.get("intersections", 0)
                    rays = shared_progress.get("rays", 0)
                    elapsed = pbar.format_dict["elapsed"] + 1e-6
                    rayspersec = rays / elapsed
                    pbar.set_postfix({
                        "rays": f"{rays:,} ({rayspersec:,.0f} r/s)",
                        "hits": f"{intersections:,}"
                    })
                    last_completed = current
                time.sleep(0.1)
            # Final update
            pbar.n = total_work_units
            pbar.refresh()
    
    # Start monitor thread
    monitor_thread = Thread(target=progress_monitor, daemon=True)
    monitor_thread.start()
    
    # Process files in parallel
    # Remove one thread from max if n_jobs = -1 to account for monitoring
    if debug:
        if n_jobs == -1:
            n_jobs = max(1, os.cpu_count() - 1)
        log(f"  ✓ Using n_jobs={n_jobs} (adjusted for monitoring thread)")

    try:
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(process_file)(pf, grids, epsilon, shared_progress) for pf in files
        )
    finally:
        # Signal the monitor thread to stop by marking completion
        shared_progress["completed"] = total_work_units
        # Wait for monitor to finish (with short timeout)
        monitor_thread.join(timeout=2)

    log("\n[4/5] Concatenating results...")
    out_blocks = []
    total_rays_processed = 0
    total_intersections = 0
    
    for result_df, file_rays, file_intersections, leg_id in results:
        if not result_df.empty:
            out_blocks.append(result_df)
        total_rays_processed += file_rays
        total_intersections += file_intersections
        log(f"  ✓ leg_{leg_id}: {file_rays} rays → {file_intersections} intersections")

    if not out_blocks:
        log("  ⚠ No intersections found")
        return pd.DataFrame({c: pd.Series(dtype="float64") for c in OUTPUT_COLUMNS})[OUTPUT_COLUMNS]

    result = pd.concat(out_blocks, ignore_index=True)[OUTPUT_COLUMNS]
    log(f"  ✓ Concatenated {len(out_blocks)} files → {len(result)} rows")

    log("\n[5/5] Saving per-leg per-voxel_size parquet files...")
    
    # Group by scan_id and voxel_size, then save each group
    if not result.empty:
        for (scan_id, voxel_size), group_df in result.groupby(['scan_id', 'voxel_size'], observed=True):
            output_filename = os.path.join(
                valid_rays_dir,
                f"leg_{int(scan_id)}_voxel_{round(float(voxel_size), 2)}_intersections.parquet"
            )
            group_df.to_parquet(
                output_filename,
                engine='pyarrow',
                compression='snappy',
                index=False,
                schema=voxel_ray_intersection_schema
            )
            log(f"  ✓ Saved {output_filename} ({len(group_df)} rows)")

    log("\n[6/6] Summary")
    log(f"  Total rays processed: {total_rays_processed:,}")
    log(f"  Total intersections: {total_intersections:,}")
    log(f"  Output shape: {result.shape}")
    log("=" * 70)
    log("✓ voxel_ray_intersections_nodask complete\n")

    return result