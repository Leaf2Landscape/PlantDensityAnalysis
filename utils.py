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
from numba import njit, prange, set_num_threads

import os
import glob
import math
import time
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq

from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from multiprocessing import Manager
from threading import Thread

try:
    import psutil  # type: ignore
except Exception:  # pragma: no cover
    psutil = None


# Dask test modules

import os
import glob
import gc
import time
import math
import tempfile
from typing import Dict, Tuple, List

import json

import numpy as np
import pandas as pd

import psutil
import dask.dataframe as dd
from dask.distributed import as_completed, wait

from numba import typed, njit, prange, set_num_threads, get_num_threads
from collections.abc import Iterable
from collections import defaultdict

import tqdm_joblib


### Module Initialization: Configure Numba threading from HPC environment ###
def _setup_numba_threads():
    """
    Configure Numba threading on module load based on HPC environment.
    Respects SLURM_CPUS_PER_TASK, OMP_NUM_THREADS, or falls back to cpu_count.
    """
    # Detect number of CPUs from environment
    nthreads = 1
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        try:
            n = int(os.environ['SLURM_CPUS_PER_TASK'])
            if n > 0:
                nthreads = n
        except (ValueError, TypeError):
            pass
    elif 'OMP_NUM_THREADS' in os.environ:
        try:
            n = int(os.environ['OMP_NUM_THREADS'])
            if n > 0:
                nthreads = n
        except (ValueError, TypeError):
            pass
    else:
        # Fall back to os.cpu_count()
        n = os.cpu_count()
        if n and n > 0:
            nthreads = n
    
    # Clamp to Numba's actual maximum (e.g. physical cores, not hyperthreads)
    numba_max = get_num_threads()
    if nthreads > numba_max:
        nthreads = numba_max

    # Set OMP_NUM_THREADS environment variable (Numba respects this)
    os.environ['OMP_NUM_THREADS'] = str(nthreads)
    
    # Also explicitly set Numba threads
    set_num_threads(nthreads)

_setup_numba_threads()

# Per-process cache for warmup signatures to avoid repeated first-call warmups.
_VOXEL_RI_WARMED_KERNELS = set()

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
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('G_wood', pa.float64()),               # G function calculated from wood hits only
    pa.field('G_lw', pa.float64()),                # G function calculated from all hits
    pa.field('bins_json', pa.string()),  # Angle distribution bin centres as a JSON string
    pa.field('liad_json', pa.string()),         # LIAD histogram as JSON string
    pa.field('liad_dewit', pa.string()),        # LIAD De Wit classification
    pa.field('liad_dewit_rmse', pa.float64()),       # De Wit rmse for designated label
    pa.field('liad_dewit_l1', pa.float64()),         # De Wit l1 for designated label
    pa.field('wiad_json', pa.string()),         # WIAD histogram as JSON string
    pa.field('wiad_dewit', pa.string()),        # WIAD De Wit classification
    pa.field('wiad_dewit_rmse', pa.float64()),       # De Wit rmse for designated label
    pa.field('wiad_dewit_l1', pa.float64()),         # De Wit l1 for designated label
    pa.field('piad_json', pa.string()),         # PIAD histogram as JSON string
    pa.field('piad_dewit', pa.string()),        # PIAD De Wit classification
    pa.field('piad_dewit_rmse', pa.float64()),       # De Wit rmse for designated label
    pa.field('piad_dewit_l1', pa.float64()),         # De Wit l1 for designated label
    pa.field('lambda_1', pa.float64()),
    pa.field('mean_angle_leaf', pa.float32()), # Mean angle of leaf hits only
    pa.field('mean_angle_lw', pa.float32()), # Mean angle of all hits
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
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('G_wood', pa.float64()),               # G function calculated from wood hits only
    pa.field('G_lw', pa.float64()),                # G function calculated from all hits
    pa.field('bins_json', pa.string()), # Angle distribution bin centres as a JSON string
    pa.field('liad_json', pa.string()),         # LIAD histogram as JSON string
    pa.field('liad_dewit', pa.string()),         # LIAD de Wit classification as string
    pa.field('liad_dewit_rmse', pa.float64()),       # de Wit classification RMSE for chosen label
    pa.field('liad_dewit_l1', pa.float64()),         # de Wit l1 for chosen label
    pa.field('wiad_json', pa.string()),         # WIAD histogram as JSON string
    pa.field('wiad_dewit', pa.string()),         # WIAD de Wit classification as string
    pa.field('wiad_dewit_rmse', pa.float64()),       # de Wit classification RMSE for chosen label
    pa.field('wiad_dewit_l1', pa.float64()),         # de Wit l1 for chosen label
    pa.field('piad_json', pa.string()),         # PIAD histogram as JSON string
    pa.field('piad_dewit', pa.string()),         # PIAD de Wit classification as string
    pa.field('piad_dewit_rmse', pa.float64()),       # De Wit classification RMSE for chosen label
    pa.field('piad_dewit_l1', pa.float64()),         # De Wit l1 for chosen label
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
    pa.field('mean_angle_leaf', pa.float32()), # Mean angle of leaf hits only
    pa.field('mean_angle_lw', pa.float32()), # Mean angle of all hits
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


DEWIT_LABELS = np.array([
    "planophile",    # mostly horizontal
    "erectophile",   # mostly vertical
    "plagiophile",   # around 45°
    "uniform",       # flat
    "spherical",     # sin(2θ)
    "extremophile"   # steeper than erectophile
])


def _query_nvidia_gpus():
    """
    Returns list of dicts: [{'index': int, 'uuid': str, 'name': str}, ...]
    """
    import subprocess
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            return []
        gpus = []
        for line in r.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                gpus.append({"index": int(parts[0]), "uuid": parts[1], "name": parts[2]})
        return gpus
    except Exception:
        return []

def resolve_cuda_index(preferred_uuid: Optional[str] = None):
    """
    Map preferred UUID to Open3D CUDA logical index.
    If CUDA_VISIBLE_DEVICES is set, Open3D sees remapped logical indices [0..N-1].
    """
    gpus = _query_nvidia_gpus()
    if not gpus:
        return None, None

    cvd = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
    cvd_tokens = [t.strip() for t in cvd.split(",") if t.strip()] if cvd else []

    if preferred_uuid:
        preferred_uuid = preferred_uuid.strip()
        # Exact UUID match
        host_match = next((g for g in gpus if g["uuid"] == preferred_uuid), None)
        if host_match is None:
            # Allow prefix match without "GPU-"
            short = preferred_uuid.replace("GPU-", "")
            host_match = next((g for g in gpus if g["uuid"].replace("GPU-", "") == short), None)

        if host_match is None:
            log(f"[WARN] Requested OPEN3D_GPU_UUID not found: {preferred_uuid}")
            return None, None

        # If CUDA_VISIBLE_DEVICES is UUID list, map to logical index by position
        if cvd_tokens and cvd_tokens[0].startswith("GPU-"):
            if host_match["uuid"] in cvd_tokens:
                return cvd_tokens.index(host_match["uuid"]), host_match["uuid"]
            print(f"[WARN] UUID {host_match['uuid']} is not visible in CUDA_VISIBLE_DEVICES.")
            return None, None

        # If CUDA_VISIBLE_DEVICES is index list, map host index to logical position
        if cvd_tokens and re.fullmatch(r"[0-9, ]+", cvd):
            visible = [int(x) for x in cvd_tokens]
            if host_match["index"] in visible:
                return visible.index(host_match["index"]), host_match["uuid"]
            print(f"[WARN] GPU index {host_match['index']} (UUID {host_match['uuid']}) not visible.")
            return None, None

        # No CVD mask: host index == logical index
        return host_match["index"], host_match["uuid"]

    # No preferred UUID: first visible GPU is logical 0 if CVD is set, else first host index
    if cvd_tokens:
        return 0, None
    return gpus[0]["index"], gpus[0]["uuid"]

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


def _canonical_curves(theta_deg: np.ndarray, categories: list = ["planophile", "erectophile", "plagiophile", "uniform", "spherical", "extremophile"]) -> np.ndarray:
    """
    Build discretized, normalized canonical PDFs for the six de Wit categories at
    the provided bin centers (theta_deg in degrees).

    Inputs:
    - theta_deg: (n_bins,) array of bin centers in degrees (e.g., 2.5, 7.5, ..., 87.5)
    - categories: list of category names to generate (default to the classical 6 de Wit classes)
    """
    th = np.deg2rad(theta_deg)  # convert to radians
    # Raw (unnormalized) shapes
    raw = []
    for name in categories:
        if name == "planophile":
            y = (2.0 / np.pi) * (1.0 + np.cos(2.0 * th))  # ∝ (2/π)(1 + cos(2θ))
        elif name == "erectophile":
            y = (2.0 / np.pi) * (1.0 - np.cos(2.0 * th))  # ∝ (2/π)(1 - cos(2θ))
        elif name == "plagiophile":
            y = (2.0 / np.pi) * (1.0 - np.cos(4.0 * th))  # ∝ (2/π)(1 - cos(4θ))
        elif name == "uniform":
            y = (2.0 / np.pi) * np.ones_like(th)  # constant (uniform distribution)
        elif name == "spherical":
            y = np.sin(th)                     # ∝ sin(θ)
        elif name == "extremophile":
            y = (2.0 / np.pi) * (1.0 + np.cos(4.0 * th))  # ∝ (2/π)(1 + cos(4θ))
        else:
            raise ValueError(f"Unknown category: {name}")
        # clamp negatives (e.g., sin(2θ) can be tiny negative due to float noise near 0/90)
        y = np.maximum(y, 0.0)
        raw.append(y)

    raw = np.vstack(raw)  # (n_cat, n_bins)

    # Discrete normalization across bins so each shape sums to 1
    raw_sum = raw.sum(axis=1, keepdims=True)
    # If any shape sums ~0 (shouldn't happen), make it uniform as fallback
    raw_norm = np.divide(raw, np.maximum(raw_sum, 1e-12))
    return raw_norm  # (n_cat, n_bins)


def classify_liad_to_dewit(
    liad: np.ndarray,
    bin_centres_deg: np.ndarray = None,
    return_scores: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Classify each voxel's LIAD histogram to the closest de Wit category
    by RMSE to canonical curves evaluated at the provided bin centers.

    Parameters
    ----------
    liad : (n_voxels, n_bins) array
        Each row is a LIAD histogram over 0–90° (not necessarily normalized).
    bin_centres_deg : (n_bins,) array, optional
        Bin centers in degrees. If None, equal-width centers over [0,90] are assumed.
    return_scores : bool
        If True, also return the RMSE scores per voxel for the chosen label

    Returns
    -------
    labels : (n_voxels,) array of str
        Best-fit de Wit class for each voxel.
    scores : (n_voxels, ) array
        RMSE per voxel for best category (only if return_scores=True).
    NOTE: If only one voxel, just return floats
    """
    liad = np.atleast_2d(np.asarray(liad, dtype=float))
    if liad.ndim != 2:
        raise ValueError("liad must be a 1D or 2D array")
    if liad.shape[0] == 0 or liad.shape[1] == 0:
        raise ValueError("liad array cannot be empty")

    n_vox, n_bins = liad.shape
    if bin_centres_deg is None:
        edges = np.linspace(0, 90, n_bins + 1)
        bin_centres_deg = 0.5 * (edges[:-1] + edges[1:])
    else:
        bin_centres_deg = np.asarray(bin_centres_deg, dtype=float)
        if bin_centres_deg.shape != (n_bins,):
            raise ValueError("bin_centres_deg must have shape (n_bins,)")

    # Normalize LIAD rows (so comparisons are shape-only)
    # liad_norm = liad / (np.linalg.norm(liad, axis=1, keepdims=True) + 1e-10)  # (n_vox, n_bins)

    # Build canonical curves
    canon = _canonical_curves(bin_centres_deg)  # (6, n_bins)
    # print(f"canon row sums:", canon.sum(axis=1))
    # print(f"canon distinct rows:", np.linalg.matrix_rank(canon))

    row_sum = liad.sum(axis=1, keepdims=True)
    liad_norm = liad / np.maximum(row_sum, 1e-12)

    # Compute RMSE between each voxel and each canonical category
    # Expand dims for broadcasting: (n_vox, 1, n_bins) vs (1, n_cat, n_bins)
    diff = np.abs(liad_norm[:, None, :] - canon[None, :, :])
    rmse = np.sqrt(np.mean(diff ** 2, axis=2))  # (n_vox, n_cat)
    l1 = diff.sum(axis=2)

    # Best category = argmin RMSE
    best_idx = np.argmin(abs(l1), axis=1)          # (n_vox,)
    best_idx_r = np.argmin(abs(rmse), axis=1)
    rmse_best = rmse[np.arange(n_vox), best_idx]  # (n_vox,)
    l1_best = l1[np.arange(n_vox), best_idx]
    labels = DEWIT_LABELS[best_idx]

    if return_scores:
        if labels.size == 1 and rmse_best.size == 1:
            labels = labels[0]
            rmse_best = rmse_best[0]
            l1_best = l1_best[0]
        return labels, rmse_best, l1_best
    return labels, None, None



# ---- normals_weights.py (can live alongside your metrics code) ----
import numpy as np
from joblib import Parallel, delayed
from scipy.spatial import cKDTree
from numba import njit

def compute_normals_weights_from_points_parallel(
    points: np.ndarray,
    *,
    voxel_size: float = 20.0,
    knn: int = 10,
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
        loc_normals, loc_confidences = _compute_normals_vectorized(pts, nb)
        
        # weight based on area-proportional weight for planar surfaces
        r_k = np.maximum(dists[:, -1], eps)  # distance to kth neighbor
        w = r_k**2
        
        conf = np.clip(loc_confidences, 0.0, 1.0)
        w *= conf  # downweight points with low confidence (e.g., near edges)

        # Remove extreme weights to avoid outliers dominating metrics
        w = np.clip(w, 0.0, np.percentile(w, 99.5))

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
        confidences: Surface variation confidence (N,)
    """
    n_points = points.shape[0]
    normals = np.zeros((n_points, 3), dtype=np.float64)
    confidences = np.zeros(n_points, dtype=np.float64)

    for i in range(n_points):
        # Get neighbor points
        neighbor_pts = points[neighbor_indices[i]]

        # Compute centroid
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

        # Compute normal (smallest eigenvector)
        normal, confidence = _compute_smallest_eigenvector_3x3(cov)

        # Ensure normal orientation is consistent: point outward from centroid to point
        direction = points[i] - centroid
        dot = 0.0
        for k in range(3):
            dot += normal[k] * direction[k]
        if dot < 0:
            for k in range(3):
                normal[k] = -normal[k]

        # Normalize the normal vector to ensure unit length (important for all axes)
        norm = 0.0
        for k in range(3):
            norm += normal[k] * normal[k]
        norm = np.sqrt(norm)
        if norm > 0:
            for k in range(3):
                normal[k] /= norm

        # Assign
        for k in range(3):
            normals[i, k] = normal[k]
        confidences[i] = confidence

    return normals, confidences


@njit
def _compute_smallest_eigenvector_3x3(cov, eps=1e-12):
    """
    Compute the eigenvector of the smallest eigenvalue for a 3x3 matrix using direct eigen-decomposition.
    Returns:
        normal: The eigenvector corresponding to the smallest eigenvalue (unit vector)
        confidence: 1 - surface_variation in [0,1] as a measure of how planar the neighborhood is
    """
    # Symmetrize to remove numerical skew
    cov_s = 0.5 * (cov + cov.T)
    # Use numpy.linalg.eigh for symmetric matrices (guaranteed real eigenvalues)
    eigvals, eigvecs = np.linalg.eigh(cov_s)
    min_idx = np.argmin(eigvals)
    v = eigvecs[:, min_idx]
    # Normalize the normal vector
    norm = np.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if norm > 0.0:
        v = v / norm
    else:
        v = np.array([1.0, 0.0, 0.0])
    # Calculate the confidence of the surface planes used to attenuate the weights later on
    lam3 = eigvals[min_idx]
    trace = cov_s[0, 0] + cov_s[1, 1] + cov_s[2, 2]
    surf_var = 0.0
    if trace > 0.0 and lam3 >= 0.0:
        surf_var = lam3 / trace
    confidence = 1.0 - min(1.0, max(0.0, surf_var))  # Invert so that more planar = higher confidence
    return v, confidence

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

def _import_open3d():
    import open3d as o3d
    import open3d.core as o3c
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    return o3d, o3c

def calculate_inclination_angle_distribution_o3dmesh(mesh_legacy, num_bins=18, debug=False):
    if mesh_legacy is None:
        return np.array([]), np.array([]), float('nan')
    o3d, _ = _import_open3d()
    if len(mesh_legacy.triangles) == 0:
        return np.array([]), np.array([]), float('nan')
    verts = np.asarray(mesh_legacy.vertices)
    tris  = np.asarray(mesh_legacy.triangles)
    v0 = verts[tris[:,1]] - verts[tris[:,0]]
    v1 = verts[tris[:,2]] - verts[tris[:,0]]
    cp = np.cross(v0, v1)
    areas = 0.5 * np.linalg.norm(cp, axis=1)
    norms = np.linalg.norm(cp, axis=1, keepdims=True)
    normals = np.divide(cp, norms, where=(norms!=0), out=np.zeros_like(cp))
    if debug:
        ang_debug = np.arccos(np.clip(normals[:,2], -1, 1))
        print(f"Debug: angles (radians) = {ang_debug}")
    ang = np.degrees(np.arccos(np.clip(normals[:,2], -1, 1)))
    ang = np.where(ang > 90, 180-ang, ang)
    
    mean_angle = float(np.nanmean(ang)) if ang.size else float('nan')
    bin_edges = np.linspace(0, 90, num_bins+1)
    idx = np.digitize(ang, bin_edges) - 1
    idx = np.clip(idx, 0, num_bins-1)
    bin_counts = np.bincount(idx, weights=areas, minlength=num_bins)
    total_area = areas.sum()
    liad = bin_counts / total_area if total_area > 0 else np.zeros(num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    if debug:
        # Plot the leaf facets, and direction used to the calculate the liad_bins
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, projection='3d')
        centroids = verts[tris].mean(axis=1)
        ax1.scatter(centroids[:,0], centroids[:,1], centroids[:,2], s=1, alpha=0.5)
        ax1.quiver(centroids[:,0], centroids[:,1], centroids[:,2], normals[:,0], normals[:,1], normals[:,2], length=0.01, color='b', alpha=0.25)
        ax1.quiver(centroids[:,0], centroids[:,1], centroids[:,2], ang_debug[:,0], ang_debug[:,1], ang_debug[:,2], length=0.02, color='g', alpha=0.5)
        ax1.set_title('Mesh with Normals + inclination angle')
        ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
        ax2 = fig.add_subplot(122)
        ax2.bar(bin_centers, liad, width=90/num_bins*0.9)
        ax2.set_title('LIAD Histogram')
        ax2.set_xlabel('Inclination Angle (degrees)')
        ax2.set_ylabel('LIAD (area fraction)')
        plt.tight_layout()
        plt.show()

        return bin_centers.astype(np.float32), liad.astype(np.float32), mean_angle, fig

    return bin_centers.astype(np.float32), liad.astype(np.float32), mean_angle


def calculate_inclination_angle_distribution_weighted_points(normals, weights, num_bins=18):
    """
    Calculate the Leaf Angle Distribution (LAD) for a set of normals and weights.
    
    INPUTS:
        normals: A numpy array of normals
        weights: A numpy array of weights
        num_bins: The number of bins to use for the histogram
        
    OUTPUTS:
        bin_centres_deg: The bin centres
        LIAD_values: The LIAD values
        angles: The angles
    """
    # Normalise normals
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)

    # Compute inclination angle
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
    bin_centres_deg = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres_deg, LIAD_values, angles


# Calculate the G function mean
def calculate_G(viewing_angles, bin_centres_deg, LIAD_values, epsilon=1e-9):
    """
    Calculate the G function mean.
    
    INPUTS:
        viewing_angle: The viewing angles
        bin_centres_deg: The bin centres
        LIAD_values: The LIAD values
    
    OUTPUTS:
        G_mean: The G function mean
    """
    # Check for empty arrays
    if len(viewing_angles) == 0 or len(bin_centres_deg) == 0 or len(LIAD_values) == 0:
        return np.nan
    
    # # Normalise LIAD
    # total_LIAD = LIAD_values.sum()
    # LIAD_norm = LIAD_values / total_LIAD if total_LIAD > 0 else LIAD_values
    LIAD_norm = LIAD_values

    # Ensure angles are clipped
    viewing_angles = np.clip(viewing_angles, epsilon, 90)
    bin_centres_deg = np.clip(bin_centres_deg, epsilon, 90)

    ### A(angle, leaf_angle)  ####
    theta_a = np.radians(viewing_angles)
    theta_b = np.radians(bin_centres_deg)

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
    delta_bin = np.radians(bin_centres_deg[1] - bin_centres_deg[0])
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
    import re
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
        # Replace this with a voxelisation of the site (for non-reference comparison workflows)
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
    voxel_references = [os.path.join(references_dir, f) for f in os.listdir(references_dir)
                        if os.path.isfile(os.path.join(references_dir, f)) and f.endswith('.csv')]

    def _extract_voxel_size_from_filename(voxel_ref_path):
        """Extract voxel size from reference filename patterns such as *_results_0.5.csv."""
        stem = os.path.splitext(os.path.basename(voxel_ref_path))[0]

        if "voxel_size_" in stem:
            try:
                return float(stem.split("voxel_size_")[-1])
            except ValueError:
                pass

        # Fallback: expected format is *_results_{voxel_size}.csv
        match = re.search(r"_results_(\d+(?:\.\d+)?)$", stem)
        if match:
            return float(match.group(1))

        raise ValueError(f"Voxel size not found in {voxel_ref_path}. Please check the file name.")

    dfs = []
    for voxel_ref in tqdm(voxel_references, desc='Reading reference voxel files', unit='file'):

        df = pd.read_csv(voxel_ref, index_col=None, header=0)

        if "voxel_size" not in df.columns:
            voxel_size = _extract_voxel_size_from_filename(voxel_ref)
            df['voxel_size'] = voxel_size
            df.to_csv(voxel_ref, index=False)
            logger.info(f"Added missing voxel_size={voxel_size} to {voxel_ref}")
        else:
            voxel_size = float(df['voxel_size'].iloc[0])
        

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

            df.to_csv(voxel_ref, index=False)

            logger.info(f"Updated voxel_ids for {voxel_ref}")

        voxel_size = float(df["voxel_size"].iloc[0])
        df = df[['voxel_cx', 'voxel_cy', 'voxel_cz']].astype(float)
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

    ### START LEG RAY PROCESSING ###

    pulses = sorted(glob.glob(os.path.join(input_dir, '*_pulse.txt')))
    points = sorted(glob.glob(os.path.join(input_dir, '*_points.xyz')))
    leg_indices = [int(pf.split('leg')[1].split('_')[0]) for pf in pulses]

    def _process_single_leg(leg_idx, pulse_file, xyz_file):
        """Read, AABB-filter, classify, and write one scanner leg to parquet."""
        import pyarrow as pa
        import pyarrow.csv as pac
        import numpy as np
        import pandas as pd
        import shutil

        delimiter_opts = pac.ParseOptions(delimiter=' ')

        # --- Read pulse file (one row per emitted ray) ---
        pulses_df = pac.read_csv(
            pulse_file,
            read_options=pac.ReadOptions(
                column_names=['origin_x', 'origin_y', 'origin_z',
                              'direction_x', 'direction_y', 'direction_z',
                              'gps_time', 'ray_id', '_']
            ),
            parse_options=delimiter_opts,
            convert_options=pac.ConvertOptions(
                include_columns=['origin_x', 'origin_y', 'origin_z',
                                 'direction_x', 'direction_y', 'direction_z', 'ray_id'],
                column_types={'ray_id': pa.uint64()}
            )
        ).to_pandas()

        # --- AABB slab test (vectorised NumPy; IEEE 754 handles zero-dirs correctly) ---
        origins    = pulses_df[['origin_x',    'origin_y',    'origin_z'   ]].to_numpy()
        directions = pulses_df[['direction_x', 'direction_y', 'direction_z']].to_numpy()
        with np.errstate(divide='ignore', invalid='ignore'):
            t1 = (plot_min - origins) / directions
            t2 = (plot_max - origins) / directions
        t_enter = np.max(np.minimum(t1, t2), axis=1)
        t_exit  = np.min(np.maximum(t1, t2), axis=1)
        keep = (t_enter <= t_exit) & (t_exit >= 0.0)
        pulses_df = pulses_df[keep].reset_index(drop=True)
        del origins, directions, t1, t2, t_enter, t_exit, keep

        # --- Read hit-points file ---
        hits_df = pac.read_csv(
            xyz_file,
            read_options=pac.ReadOptions(
                column_names=['point_x', 'point_y', 'point_z', 'echo_intensity',
                              'echo_width', 'return_number', 'number_of_returns',
                              'ray_id', 'hit_object_id', 'class', 'gps_time']
            ),
            parse_options=delimiter_opts,
            convert_options=pac.ConvertOptions(
                include_columns=['point_x', 'point_y', 'point_z', 'echo_intensity',
                                 'return_number', 'number_of_returns',
                                 'ray_id', 'hit_object_id', 'class'],
                column_types={'ray_id': pa.uint64()}
            )
        ).to_pandas()

        # --- Left-join filtered rays with their hits ---
        rays = pulses_df.merge(hits_df, on='ray_id', how='left')
        del pulses_df, hits_df

        # --- Classify and remove unknown object ids ---
        rays['scan_id'] = np.uint64(leg_idx)
        hit_object_key = 'hit_object_id' if not use_class else 'class'
        rays['is_leaf'] = rays[hit_object_key].isin(leaf_object_ids)
        valid_ids = set(wood_object_ids) | set(leaf_object_ids)
        rays = rays[rays[hit_object_key].isna() | rays[hit_object_key].isin(valid_ids)]
        rays = rays.drop(columns=['hit_object_id', 'class'])

        # --- Write parquet ---
        rays_file = os.path.join(output_dir, valid_rays_template.format(leg=leg_idx))
        if os.path.exists(rays_file):
            shutil.rmtree(rays_file) if os.path.isdir(rays_file) else os.remove(rays_file)
        rays.to_parquet(rays_file, engine='pyarrow', compression='snappy', schema=valid_rays_schema)

        num_points = int((~rays['point_x'].isna()).sum())
        logger.info(f"Leg {leg_idx}: {len(rays)} rays, {num_points} points written.")
        return len(rays), num_points

    print(f"Processing {len(pulses)} scanner legs in parallel...")
    logger.info(f"Processing {len(pulses)} legs...")
    with tqdm_joblib.tqdm_joblib(tqdm(desc='Processing scanner legs', total=len(pulses), unit='leg')):
        results = Parallel(n_jobs=-1)(
            delayed(_process_single_leg)(leg_idx, pf, xf)
            for leg_idx, pf, xf in zip(leg_indices, pulses, points)
        )

    total_rays   = sum(r for r, _ in results)
    total_points = sum(p for _, p in results)
    logger.info(f"Total: {total_rays} valid rays, {total_points} hit points.")
    
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

def create_intersections_ddf(parquet_root: str, scan_ids: list[int | str], voxel_sizes: list[float | str], is_hive: bool = False, blocksize_str:str = "25MB") -> dask.dataframe.DataFrame:
    import os, glob
    all_files = []

    # Try and load a hive pattern input and apply filter before returning
    if is_hive:
        print("Using hive_folder parquet loading")
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
    
    else:
        print(f"Using legacy glob method for intersections.parquet files.")

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
    mean_angle_lw  = float(np.nanmean(va[current_hit])) if num_lw_hits > 0 else np.nan
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
    bins, piad_vals, _ = calculate_inclination_angle_distribution_weighted_points(normals=normals, weights=weights)
    if piad_vals is not None and len(piad_vals) > 0:
        piad_dewit, piad_dewit_rmse, piad_dewit_l1 = classify_liad_to_dewit(
            liad=piad_vals,
            bin_centres_deg=bins,
            return_scores=True
        )
    else:
        piad_dewit = ""
        piad_dewit_rmse = np.nan
        piad_dewit_l1 = np.nan

    # LIAD - only leaf hits contribute to LIAD
    leaf_normals = block.loc[current_leaf_mask, ["normal_x","normal_y","normal_z"]].to_numpy()
    leaf_weights = block.loc[current_leaf_mask, "point_weight"].to_numpy()
    bins, liad_vals, _ = calculate_inclination_angle_distribution_weighted_points(normals=leaf_normals, weights=leaf_weights)
    if liad_vals is not None and len(liad_vals) > 0:
        liad_dewit, liad_dewit_rmse, liad_dewit_l1 = classify_liad_to_dewit(
            liad=liad_vals,
            bin_centres_deg=bins,
            return_scores=True
        )
    else:
        liad_dewit = ""        
        liad_dewit_rmse = np.nan
        liad_dewit_l1 = np.nan

    # WIAD - only wood hits contribute to WIAD
    wood_normals = block.loc[current_hit & ~leaf_mask, ["normal_x","normal_y","normal_z"]].to_numpy()
    wood_weights = block.loc[current_hit & ~leaf_mask, "point_weight"].to_numpy()
    bins, wiad_vals, _ = calculate_inclination_angle_distribution_weighted_points(normals=wood_normals, weights=wood_weights)
    if wiad_vals is not None and len(wiad_vals) > 0:
        wiad_dewit, wiad_dewit_rmse, wiad_dewit_l1 = classify_liad_to_dewit(
            liad=wiad_vals,
            bin_centres_deg=bins,
            return_scores=True
        )
    else:
        wiad_dewit = ""
        wiad_dewit_rmse = np.nan
        wiad_dewit_l1 = np.nan

    # Calculate all G values
    va_lw   = va[current_hit]
    va_leaf = va[current_leaf_mask]
    va_wood = va[current_hit & ~leaf_mask]
    G_leaf  = calculate_G(viewing_angles=va_leaf, bin_centres_deg=bins, LIAD_values=liad_vals)
    G_lw    = calculate_G(viewing_angles=va_lw,   bin_centres_deg=bins, LIAD_values=piad_vals)
    G_wood  = calculate_G(viewing_angles=va_wood, bin_centres_deg=bins, LIAD_values=wiad_vals)
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

    out.loc[0, 'bins_json'] = json.dumps(bins.tolist()) if bins is not None else None
    out.loc[0, 'piad_json'] = json.dumps(piad_vals.tolist()) if piad_vals is not None else None
    out.loc[0, 'piad_dewit'] = piad_dewit if piad_dewit is not None else ""
    out.loc[0, 'piad_dewit_rmse'] = float(piad_dewit_rmse) if piad_dewit_rmse is not None else np.nan
    out.loc[0, 'piad_dewit_l1'] = float(piad_dewit_l1) if piad_dewit_l1 is not None else np.nan
    out.loc[0, 'liad_json'] = json.dumps(liad_vals.tolist()) if liad_vals is not None else None
    out.loc[0, 'liad_dewit'] = liad_dewit if liad_dewit is not None else ""
    out.loc[0, 'liad_dewit_rmse'] = float(liad_dewit_rmse) if liad_dewit_rmse is not None else np.nan
    out.loc[0, 'liad_dewit_l1'] = float(liad_dewit_l1) if liad_dewit_l1 is not None else np.nan
    out.loc[0, 'wiad_json'] = json.dumps(wiad_vals.tolist()) if wiad_vals is not None else None
    out.loc[0, 'wiad_dewit'] = wiad_dewit if wiad_dewit is not None else ""
    out.loc[0, 'wiad_dewit_rmse'] = float(wiad_dewit_rmse) if wiad_dewit_rmse is not None else np.nan
    out.loc[0, 'wiad_dewit_l1'] = float(wiad_dewit_l1) if wiad_dewit_l1 is not None else np.nan

    out.loc[0, 'mean_angle_leaf'] = np.float32(mean_angle_leaf)
    out.loc[0, 'mean_angle_lw']  = np.float32(mean_angle_lw)
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


def process_wood_volume_file(wood_mesh: str, wood_voxel_size: float=0.01, threshold: int=4) -> Optional[np.ndarray]:
    wood_inside_points = calculate_wood_volume(wood_mesh, voxel_size=wood_voxel_size, threshold=threshold)
    if wood_inside_points is not None:
        try:
            np.savetxt(wood_volume_file, wood_inside_points, fmt='%.3f')
            print(f"Wood volume file saved at {wood_volume_file}.")
            return np.round(wood_inside_points, 3)
        except Exception as e:
            print(f"Error saving wood volume file {wood_volume_file}: {e}")
            return None
    print(f"No wood volume file found at {wood_volume_file}.")
    return None

def compute_wood_volume_in_voxel(wood_volume, voxel_center, voxel_size, small_voxel_size=0.01):
    """
    Return estimates the volume of wood points within a voxel.
    This function assumes wood_volume_file is a numpy array of shape (N, 3).
    """

    if wood_volume is None or wood_volume.shape[0] == 0:
        return 0.0
    
    # Calculate number of points within the voxel
    half_size = voxel_size / 2.0
    min_bound = np.array(voxel_center) - half_size
    max_bound = np.array(voxel_center) + half_size
    in_voxel = np.all((wood_volume >= min_bound) & (wood_volume <= max_bound), axis=1)
    num_points_in_voxel = np.sum(in_voxel)

    wood_volume = small_voxel_size ** 3 * num_points_in_voxel
    # print(f"Computed wood volume in voxel centered at {voxel_center}: {wood_volume} (with {num_points_in_voxel} points).")

    return wood_volume


def calculate_wood_volume(wood_mesh: trimesh.Trimesh, voxel_size: float=0.01, threshold: int=4) -> np.ndarray:
    """
    Calculate the wood volume of a mesh by voxelizing it.
    Vectorized approach: batch all points per x-slice and cast rays in one operation.
    """
    if wood_mesh.is_empty:
        print("Wood mesh is empty, cannot calculate volume.")
        return None
    
    start_time = dt.datetime.now()
    
    # Convert to open3d
    o3d_wood_mesh = o3d.geometry.TriangleMesh()
    o3d_wood_mesh.vertices = o3d.utility.Vector3dVector(wood_mesh.vertices)
    o3d_wood_mesh.triangles = o3d.utility.Vector3iVector(wood_mesh.faces)
    o3d_wood_mesh.compute_vertex_normals()
    o3d_wood_mesh.remove_duplicated_vertices()
    o3d_wood_mesh.remove_duplicated_triangles()
    o3d_wood_mesh.remove_degenerate_triangles()
    
    # Get bounding box of the mesh
    aabb = o3d_wood_mesh.get_axis_aligned_bounding_box()
    print(f"Bounding box of wood mesh: {aabb}")
    offset = voxel_size * 0.01

    # Create grid coordinates
    x = np.arange(aabb.min_bound[0] - offset, aabb.max_bound[0] + offset, voxel_size)
    y = np.arange(aabb.min_bound[1] - offset, aabb.max_bound[1] + offset, voxel_size)
    z = np.arange(aabb.min_bound[2] - offset, aabb.max_bound[2] + offset, voxel_size)

    total_points = len(x) * len(y) * len(z)
    print(f"Grid dimensions: {len(x)} x {len(y)} x {len(z)} = {total_points} points.")

    # Setup raycasting scene
    scene = o3d.t.geometry.RaycastingScene()
    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(o3d_wood_mesh)
    scene.add_triangles(mesh_t)

    # Ray directions
    directions = np.array([
        [1.0001, 0.0001, 0.0001],
        [-1.0001, 0.0001, 0.0001],
        [0.0001, 1.0001, 0.0001],
        [0.0001, -1.0001, 0.0001],
        [0.0001, 0.0001, 1.0001],
        [0.0001, 0.0001, -1.0001]
    ], dtype=np.float32)
    
    inside_points = []
    n_directions = len(directions)

    for xi, x_val in enumerate(tqdm(x, desc="Processing X-slices")):
        # Generate all y,z points for this x-slice
        yy, zz = np.meshgrid(y, z, indexing='ij')
        xx = np.full_like(yy, x_val, dtype=np.float32)
        slice_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]).astype(np.float32)
        n_points = len(slice_points)
        
        # Count intersections across all directions
        inside_counts = np.zeros(n_points, dtype=np.int32)
        
        # Process all directions in one vectorized call
        # Repeat points for each direction
        points_repeated = np.repeat(slice_points, n_directions, axis=0)  # (n_points * n_dirs, 3)
        dirs_tiled = np.tile(directions, (n_points, 1))  # (n_points * n_dirs, 3)
        
        # Create rays tensor: concatenate origins and directions
        rays = np.column_stack([points_repeated, dirs_tiled]).astype(np.float32)
        rays_tensor = o3d.core.Tensor(rays, dtype=o3d.core.Dtype.Float32)
        
        # Get all intersections at once
        intersections = scene.count_intersections(rays_tensor).numpy()
        
        # Reshape back to (n_points, n_directions) and count odd intersections per point
        intersections = intersections.reshape(n_points, n_directions)
        inside_counts = (intersections % 2).sum(axis=1).astype(np.int32)
        
        # Identify inside points based on threshold
        mask = inside_counts >= threshold
        inside_points.append(slice_points[mask])
        
        # Progress reporting
        if (xi + 1) % max(1, len(x) // 10) == 0:
            elapsed = dt.datetime.now() - start_time
            percent_complete = (xi + 1) / len(x) * 100
            if percent_complete > 0:
                est_total_time = elapsed.total_seconds() / percent_complete * 100
                remaining = est_total_time - elapsed.total_seconds()
                print(f"Progress: {percent_complete:.1f}% complete, "
                      f"ETA: {remaining/60:.1f} minutes, "
                      f"Found {sum(len(p) for p in inside_points)} inside points so far")
    
    # Concatenate all inside points
    if inside_points:
        inside_points = np.vstack(inside_points)
    else:
        inside_points = np.array([], dtype=np.float32).reshape(0, 3)
    
    if len(inside_points) == 0:
        print(f"No inside points found. Try a different threshold or check mesh.")
        return None
    
    return inside_points


def process_leaf_area_file(scene_file: str, leaf_mesh: trimesh.Trimesh) -> None:
    """
    Process the leaf area file and save it as a CSV.
    The leaf area is calculated from the mesh and saved in a CSV file.
    """
    if leaf_mesh.is_empty:
        print("Leaf mesh is empty, cannot calculate area.")
        return
    
    # Find triangle clusters that are connected (i.e. a leaf)
    leaf_mesh = leaf_mesh.copy()

    # Use trimesh to compute connected components and their areas
    components = leaf_mesh.split(only_watertight=False)
    areas = [comp.area for comp in components if comp.faces.shape[0] > 0]

    if not areas:
        avg_area, min_area, max_area, num_leaves, total_leaf_area = 0.0, 0.0, 0.0, 0, 0.0
    else:
        avg_area = float(np.mean(areas))
        min_area = float(np.min(areas))
        max_area = float(np.max(areas))
        num_leaves = len(areas)
        total_leaf_area = float(np.nansum(areas))

    print(f"Leaf area stats: avg={avg_area:.3f}, min={min_area:.3f}, max={max_area:.3f}, num_leaves={num_leaves}, total_leaf_area={total_leaf_area}")

    output_path = os.path.join(os.path.dirname(scene_file), os.path.basename(scene_file).replace(".obj", "_leaf_area.csv"))

    df = pd.DataFrame({
        'tree_id': [os.path.basename(scene_file).replace(".obj", "")],
        'avg_leaf_area': [avg_area],
        'min_leaf_area': [min_area],
        'max_leaf_area': [max_area],
        'num_leaves': [num_leaves],
        'total_leaf_area': [total_leaf_area]
    })
    df.to_csv(output_path, index=False)
    print(f"Leaf area saved to {output_path}.")

    return avg_area, min_area, max_area, num_leaves, total_leaf_area

# ============================================================================
# Progress-aware public API
# ============================================================================
def get_voxel_metrics(
    intersections_folder: str,
    average_leaf_area: float,
    *,
    output_dir: Optional[str] = None,
    project_name: Optional[str] = None,
    cpus: Optional[int] = None,
    mem: Optional[int] = None,
    scan_ids: Optional[List[str]] = None,
    voxel_sizes: Optional[List[float]] = None,
    optimal_threads: int = 2,
    beam_divergence: float = 0.35,   # mrad
    is_multireturn: bool = False,
    is_leaf_true: bool = True,       # same meaning as your flag
    same_normals: bool = False,      # compute normals once and reuse across voxel_sizes
    debug: bool = True,
    epsilon: float = 1e-9,
    # Tuning
    voxel_block_rows_hint: int = 0,    # 0 -> auto, else compute blocks ~ this many rows each
    normal_calc_voxel_size: float = 10,       # voxel size for normal estimation (if normals not present)
) -> pd.DataFrame:
    """
    Non-Dask, resource-aware and parallel computation of voxel metrics.
    - Processes one voxel_size at a time to manage memory.
    - Optionally reuses normals/weights across voxel_sizes if same_normals=True.
    - Loads intersections via PyArrow.
    - Sorts rows by voxel_id.
    - Splits into contiguous voxel blocks and computes metrics in parallel.
    - Provides clear progress updates with live counters and progress bars.

    Parameters:
        same_normals : bool
            If True, compute normals & weights from the first voxel_size,
            then reuse them for all subsequent voxel_sizes by point (x,y,z) lookup.
            Reduces memory usage and redundant computation for multi-resolution datasets.

    Returns:
        pd.DataFrame of voxel metrics (concatenated across all voxel_sizes),
        and writes CSV per voxel_size identical to your original routine.
    """

    # ========== Phase 1: Setup & Configuration ==========
    print("\n" + "=" * 80)
    print("  [get_voxel_metrics] Voxel Metrics Computation")
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
    print(f"    • Same normals:      {same_normals}")
    print(f"    • Normal calc voxel size: {normal_calc_voxel_size}")

    # ========== Phase 2: Dataset Discovery & Initial Filtering ==========
    print(f"\n[2] Discovering dataset:")
    
    selected = _select_files_with_both_filters(intersections_folder, scan_ids, voxel_sizes)
    
    # Extract unique voxel_sizes from selected files
    unique_voxel_sizes = sorted(set(vs for _, _, vs in selected if vs is not None))
    if not unique_voxel_sizes:
        raise ValueError("No voxel_sizes could be determined from selected files.")
    
    print(f"    • Found {len(unique_voxel_sizes)} voxel_size(s): {unique_voxel_sizes}")
    print(f"    • Will process sequentially to manage memory")

    if project_name is None:
        project_name = os.path.basename(os.path.normpath(intersections_folder))

    if output_dir is None:
        output_dir = intersections_folder
    os.makedirs(output_dir, exist_ok=True)

    # ========== Phase 3: Compute Normals Once (if same_normals=True) ==========
    cached_normals = {}  # maps (x, y, z) -> (nx, ny, nz, weight) for reuse
    
    if same_normals:
        print(f"\n[2b] Pre-computing normals & weights (reusable across voxel_sizes):")
        
        # Load only the first voxel_size to compute normals
        first_vs_files = [f for f, _, vs in selected if vs == unique_voxel_sizes[0]]
        cols = [
            'voxel_size','voxel_id','voxel_cx','voxel_cy','voxel_cz',
            'scan_id','ray_id',
            't_entry_x','t_entry_y','t_entry_z',
            't_exit_x','t_exit_y','t_exit_z',
            'distance_to_centre',
            'point_x','point_y','point_z',
            'echo_intensity','return_number','number_of_returns',
            'viewing_angle','hit_type','is_leaf'
        ]
        
        df_first, _ = _load_intersections_with_injected_metadata(
            selected_files=[(f, None, unique_voxel_sizes[0]) for f in first_vs_files],
            required_columns=cols,
            must_have_scan_id=False,
            must_have_voxel_size=False
        )
        
        if not df_first.empty:
            hit_mask = (df_first["hit_type"].to_numpy() == 2)
            finite_pts = np.isfinite(df_first[["point_x","point_y","point_z"]].to_numpy()).all(axis=1)
            usable = hit_mask & finite_pts
            
            leaf_mask = usable & (df_first["is_leaf"].to_numpy() == True)
            wood_mask = usable & (df_first["is_leaf"].to_numpy() == False)
            
            print(f"    • Usable hit points: {int(usable.sum())} / {len(df_first)}")
            
            # Compute normals for leaf and wood separately
            def _compute_and_cache(mask, label):
                if mask.sum() == 0:
                    print(f"    • {label}: 0 points, skipping")
                    return
                
                pts = df_first.loc[mask, ["point_x","point_y","point_z"]].to_numpy(np.float64, copy=False)
                unique_pts, unique_indices = np.unique(pts, axis=0, return_index=True)
                
                normals, weights = compute_normals_weights_from_points_parallel(
                    unique_pts,
                    voxel_size=normal_calc_voxel_size,
                    n_jobs=-1
                )
                
                # Cache by (x, y, z) tuple
                for i, (pt, norm, w) in enumerate(zip(unique_pts, normals, weights)):
                    key = tuple(pt)
                    cached_normals[key] = (norm[0], norm[1], norm[2], w)
                
                print(f"    • {label}: cached {len(unique_pts)} unique points")
            
            _compute_and_cache(leaf_mask, "Leaf")
            _compute_and_cache(wood_mask, "Wood")
            
            del df_first
            gc.collect()
        
        print(f"    ✓ Cached {len(cached_normals)} unique point locations")

    # ========== Phase 4: Process Each Voxel Size Sequentially ==========
    all_voxel_metrics = []
    
    for vs_idx, target_vs in enumerate(unique_voxel_sizes):
        print(f"\n[3.{vs_idx+1}] Processing voxel_size = {target_vs}m:")
        
        # Filter selected files to this voxel_size
        vs_files = [f for f, _, vs in selected if vs == target_vs]
        
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
            selected_files=[(f, None, target_vs) for f in vs_files],
            required_columns=cols,
            must_have_scan_id=False,
            must_have_voxel_size=False
        )
        
        if df.empty:
            print(f"    ⚠ No data for voxel_size={target_vs}, skipping")
            continue
        
        n_voxels = df["voxel_id"].nunique()
        n_rows = len(df)
        print(f"    • Loaded {n_rows:,} rows across {n_voxels:,} unique voxels")

        # ========== Phase 4b: Normals & Weights ==========
        if same_normals and cached_normals:
            print(f"    • Mapping cached normals & weights by point location:")
            
            # Prepare output cols if missing
            for col in ["normal_x", "normal_y", "normal_z", "point_weight"]:
                if col not in df.columns:
                    df[col] = np.nan
            
            # Map by point (x, y, z)
            hit_mask = (df["hit_type"].to_numpy() == 2)
            finite_pts = np.isfinite(df[["point_x","point_y","point_z"]].to_numpy()).all(axis=1)
            usable = hit_mask & finite_pts
            
            n_mapped = 0
            for idx in np.where(usable)[0]:
                pt = tuple(df.loc[idx, ["point_x","point_y","point_z"]].values)
                if pt in cached_normals:
                    nx, ny, nz, w = cached_normals[pt]
                    df.loc[idx, "normal_x"] = nx
                    df.loc[idx, "normal_y"] = ny
                    df.loc[idx, "normal_z"] = nz
                    df.loc[idx, "point_weight"] = w
                    n_mapped += 1
            
            print(f"      ✓ Mapped {n_mapped} / {int(usable.sum())} points from cache")
            
        else:
            # Compute normals for this voxel_size independently
            print(f"    • Computing normals & weights for this voxel_size:")
            
            # Prepare output cols if missing
            for col in ["normal_x", "normal_y", "normal_z", "point_weight"]:
                if col not in df.columns:
                    df[col] = np.nan
            
            hit_mask = (df["hit_type"].to_numpy() == 2)
            finite_pts = np.isfinite(df[["point_x","point_y","point_z"]].to_numpy()).all(axis=1)
            usable = hit_mask & finite_pts
            
            if debug:
                print(f"      • Usable hit points: {int(usable.sum())} / {len(df)}")
            
            leaf_mask = usable & (df["is_leaf"].to_numpy() == True)
            wood_mask = usable & (df["is_leaf"].to_numpy() == False)
            
            def _fit_and_assign(mask: np.ndarray, label: str):
                if mask.sum() == 0:
                    print(f"      • {label}: 0 points")
                    return
                idx = np.nonzero(mask)[0]
                pts = df.loc[mask, ["point_x","point_y","point_z"]].to_numpy(np.float64, copy=False)

                unique_pts, unique_indices = np.unique(pts, axis=0, return_index=True)
                pts = unique_pts
                idx = idx[unique_indices]

                normals, weights = compute_normals_weights_from_points_parallel(
                    pts,
                    voxel_size=normal_calc_voxel_size,
                    n_jobs=-1
                )
                df.loc[idx, "normal_x"] = normals[:,0]
                df.loc[idx, "normal_y"] = normals[:,1]
                df.loc[idx, "normal_z"] = normals[:,2]
                df.loc[idx, "point_weight"] = weights
                print(f"      • {label}: computed {len(pts)} unique points")

            _fit_and_assign(leaf_mask, "Leaf")
            _fit_and_assign(wood_mask, "Wood")

        if debug:
            print("\n    Sample of computed normals & weights for leaf hits:")
            leaf_sample = df.loc[(df["hit_type"] == 2) & (df["is_leaf"]), ["point_x","point_y","point_z","normal_x","normal_y","normal_z","point_weight"]].head(10)
            print(leaf_sample)
            print("\n    Sample of computed normals & weights for wood hits:")
            wood_sample = df.loc[(df["hit_type"] == 2) & ~(df["is_leaf"]), ["point_x","point_y","point_z","normal_x","normal_y","normal_z","point_weight"]].head(10)
            print(wood_sample)

        # ========== Phase 4c: Task Decomposition ==========
        print(f"\n    [Decomposing] into voxel blocks:")
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
        print(f"      • Created {n_tasks} parallel task(s)")
        print(f"      • Voxel groups per task: 1–{max(e-a for a,e in tasks)} (average ~{n_voxels/n_tasks:.1f})")

        # ========== Phase 4d: Parallel Computation ==========
        print(f"\n    [Computing] metrics (parallel, {n_workers} workers):")

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
            desc="      Progress",
            unit=" task",
            ncols=90,
            leave=True,
            position=0
        ) as pbar:
            results_nested = Parallel(
                n_jobs=n_workers, prefer=prefer, batch_size="auto", verbose=0
            )(
                tqdm(
                    (delayed(_process_range)(a,b) for (a,b) in tasks),
                    total=n_tasks,
                    leave=False,
                )
            )
            pbar.update(n_tasks)

        out_frames = [row for sub in results_nested for row in sub]
        vs_metrics_df = pd.concat(out_frames, axis=0, ignore_index=True)

        n_computed = len(vs_metrics_df)
        print(f"      ✓ Computed metrics for {n_computed:,} voxels")
        
        all_voxel_metrics.append(vs_metrics_df)
        
        # Clean up before next iteration
        del df, vs_metrics_df
        gc.collect()

    # ========== Phase 5: Consolidate & Output ==========
    print(f"\n[4] Writing output:")
    
    if not all_voxel_metrics:
        print("    ⚠ No metrics computed")
        schema = voxel_metrics_schema_multireturn if is_multireturn else voxel_metrics_schema_singlereturn
        return _gen_dataframe(schema)
    
    voxel_metrics_df = pd.concat(all_voxel_metrics, axis=0, ignore_index=True)

    ts = time.strftime('%Y%m%d_%H%M%S')
    n_files = 0
    for vs, g in voxel_metrics_df.groupby("voxel_size", sort=False):
        basename = f"{project_name}_voxel_metrics_{round(vs,1)}m_{ts}.csv"
        out_csv = os.path.join(output_dir, basename)
        header_comment = f"# Scan IDs:, {', '.join(map(str, included_scan_ids))}\n"
        with open(out_csv, "w") as f:
            f.write(header_comment)
        g.to_csv(out_csv, mode="a", index=False)
        n_files += 1
        print(f"    • {os.path.basename(out_csv)} ({len(g):,} rows)")

    print(f"    ✓ Wrote {n_files} CSV file(s)")

    # ========== Summary ==========
    print(f"\n[5] Summary:")
    print(f"    • Total voxels:      {len(voxel_metrics_df):,}")
    print(f"    • Unique scan IDs:   {len(set(included_scan_ids)) if included_scan_ids else 0}")
    print(f"    • Voxel_sizes processed: {len(unique_voxel_sizes)}")
    print(f"    • Output directory:  {output_dir}")
    print("\n" + "=" * 80)
    print("  ✓ Voxel metrics computation complete\n")

    return voxel_metrics_df

    if debug:
        # Print out head for leaf_mask and wood_mask, just points, normals, and point_weight
        print("\n    Sample of computed normals & weights for leaf hits:")
        print(df.loc[leaf_mask, ["point_x","point_y","point_z","normal_x","normal_y","normal_z","point_weight"]].head(10))
        print("\n    Sample of computed normals & weights for wood hits:")
        print(df.loc[wood_mask, ["point_x","point_y","point_z","normal_x","normal_y","normal_z","point_weight"]].head(10))

        # Plot wood and leaf points in separate subplots, side by side, with normals as quivers
        try:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            sample_size = min(1000, len(df))
            sample_df = df[usable].sample(sample_size, random_state=42)

            fig = plt.figure(figsize=(16, 7))

            # Leaf subplot
            ax1 = fig.add_subplot(1, 2, 1, projection='3d')
            leaf_df = sample_df[sample_df["is_leaf"]]
            ax1.scatter(leaf_df["point_x"], leaf_df["point_y"], leaf_df["point_z"], c='g', s=5, alpha=0.6, label='Leaf')
            ax1.quiver(
            leaf_df["point_x"], leaf_df["point_y"], leaf_df["point_z"],
            leaf_df["normal_x"], leaf_df["normal_y"], leaf_df["normal_z"],
            length=0.1, normalize=True, color='blue', linewidth=0.5
            )
            ax1.set_title("Leaf Points with Normals")
            ax1.legend()

            # Wood subplot
            ax2 = fig.add_subplot(1, 2, 2, projection='3d')
            wood_df = sample_df[~sample_df["is_leaf"]]
            ax2.scatter(wood_df["point_x"], wood_df["point_y"], wood_df["point_z"], c='saddlebrown', s=5, alpha=0.6, label='Wood')
            ax2.quiver(
            wood_df["point_x"], wood_df["point_y"], wood_df["point_z"],
            wood_df["normal_x"], wood_df["normal_y"], wood_df["normal_z"],
            length=0.1, normalize=True, color='blue', linewidth=0.5
            )
            ax2.set_title("Wood Points with Normals")
            ax2.legend()

            plt.tight_layout()
            plt.show()
        except ImportError:
            print("Matplotlib not available, skipping normal visualization.")

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
        basename = f"{project_name}_voxel_metrics_{round(vs,1)}m_{ts}.csv"
        out_csv = os.path.join(output_dir, basename)
        header_comment = f"# Scan IDs:, {', '.join(map(str, included_scan_ids))}\n"
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
    csvs = sorted(glob.glob(os.path.join(references_dir, "*.csv")))
    for csvf in csvs:
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
    out = out.drop_duplicates(subset=["voxel_id"])
    return out


# Optional schema (if your environment defines it)
voxel_ray_intersection_schema = globals().get("voxel_ray_intersection_schema", None)


# -----------------------------------------------------------------------------
# File helpers
# -----------------------------------------------------------------------------

def list_parquet_files(valid_rays_dir: str) -> List[str]:
    return sorted(glob.glob(os.path.join(valid_rays_dir, "*_valid_rays.parquet")))


def leg_from_filename(path: str) -> int:
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0].split("_")
    for token in stem:
        if token.isdigit():
            return int(token)
    return 0


# Precompute neighbor offsets for Chebyshev radius k
def neighbor_offsets(k: int = 1):
    offs = []
    for dx in range(-k, k+1):
        for dy in range(-k, k+1):
            for dz in range(-k, k+1):
                # include center cell too (dx=dy=dz=0)
                offs.append((dx,dy,dz))
    return offs

def _circular_mode_mod(x, period, bins=720):
    m = np.mod(x, period)
    hist, edges = np.histogram(m, bins=bins, range=(0.0, period))
    k = int(np.argmax(hist))
    return 0.5 * (edges[k] + edges[k+1])

def _phase_lock_grid_min(df_s: pd.DataFrame, s: float) -> np.ndarray:
    cx = df_s["voxel_cx"].to_numpy(float)
    cy = df_s["voxel_cy"].to_numpy(float)
    cz = df_s["voxel_cz"].to_numpy(float)

    phx = _circular_mode_mod(cx, s); offx = (phx - 0.5*s) % s
    phy = _circular_mode_mod(cy, s); offy = (phy - 0.5*s) % s
    phz = _circular_mode_mod(cz, s); offz = (phz - 0.5*s) % s

    gx0 = (np.floor((cx.min() - 0.5*s - offx) / s) * s) + offx
    gy0 = (np.floor((cy.min() - 0.5*s - offy) / s) * s) + offy
    gz0 = (np.floor((cz.min() - 0.5*s - offz) / s) * s) + offz
    return np.array([gx0, gy0, gz0], dtype=np.float64)

def build_dense_grids_from_refs(voxel_refs: pd.DataFrame,
                                eps: float = 1e-9,
                                buffer_k: int = 1):
    """
    Returns: dict[size] -> {
        'voxel_size': s,
        'grid_min': gmin,
        'grid_shape': (nx,ny,nz),
        'id_grid': int64[nx,ny,nz] (-1 empty),
        'occ_min_idx'/'occ_max_idx': padded index AABB,
        'occ_min_xyz'/'occ_max_xyz': half-open world AABB for occupied (buffered)
    }
    """
    out = {}
    for s_val in sorted(voxel_refs["voxel_size"].unique()):
        s = float(s_val)
        df = voxel_refs[voxel_refs["voxel_size"] == s].copy()
        if df.empty:
            continue

        # 1) lock lattice phase from actual centres
        gmin = _phase_lock_grid_min(df, s)

        # 2) index centres with floor + tiny eps (robust to noise)
        cx = df["voxel_cx"].to_numpy(np.float64)
        cy = df["voxel_cy"].to_numpy(np.float64)
        cz = df["voxel_cz"].to_numpy(np.float64)

        ix = np.floor((cx - gmin[0]) / s + eps).astype(np.int64)
        iy = np.floor((cy - gmin[1]) / s + eps).astype(np.int64)
        iz = np.floor((cz - gmin[2]) / s + eps).astype(np.int64)

        nx = int(ix.max()) + 1
        ny = int(iy.max()) + 1
        nz = int(iz.max()) + 1

        if ix.min() < 0 or iy.min() < 0 or iz.min() < 0:
            raise ValueError(f"Negative index exists for size {s}.")

        # 3) build dense id grid
        id_grid = np.full((nx, ny, nz), -1, dtype=np.int64)
        vids = df["voxel_id"].to_numpy(np.int64)
        for i, j, k, vid in zip(ix, iy, iz, vids):
            if id_grid[i, j, k] != -1:
                raise ValueError(f"Duplicate index {(int(i),int(j),int(k))} for s={s}")
            id_grid[i, j, k] = int(vid)

        # 4) occupied AABB (+ buffer_k) in index space → world space half‑open box
        min_i = int(ix.min()); min_j = int(iy.min()); min_k = int(iz.min())
        max_i = int(ix.max()); max_j = int(iy.max()); max_k = int(iz.max())

        ax0 = max(0, min_i - buffer_k)
        ay0 = max(0, min_j - buffer_k)
        az0 = max(0, min_k - buffer_k)
        ax1 = min(nx - 1, max_i + buffer_k)
        ay1 = min(ny - 1, max_j + buffer_k)
        az1 = min(nz - 1, max_k + buffer_k)

        occ_min_idx = (ax0, ay0, az0)
        occ_max_idx = (ax1, ay1, az1)

        occ_min_xyz = gmin + s * np.array([ax0, ay0, az0], dtype=np.float64)
        # half-open upper face => +1 cell
        occ_max_xyz = gmin + s * np.array([ax1+1, ay1+1, az1+1], dtype=np.float64)

        out[s] = dict(
            voxel_size=s,
            grid_min=gmin,
            grid_shape=(nx, ny, nz),
            id_grid=id_grid,
            occ_min_idx=occ_min_idx,
            occ_max_idx=occ_max_idx,
            occ_min_xyz=occ_min_xyz,
            occ_max_xyz=occ_max_xyz,
        )
    return out

import numpy as np
from numba import njit, prange

@njit(cache=True, fastmath=False, nogil=True)
def _slab(o, d, gmin, gmax, eps):
    t_enter = -1.0e300
    t_exit  =  1.0e300
    for ax in (0, 1, 2):
        oa = o[ax]; da = d[ax]; mn = gmin[ax]; mx = gmax[ax]
        if abs(da) < eps:
            if (oa < mn - eps) or (oa > mx + eps):
                return 1.0, 0.0  # miss
        else:
            inv = 1.0 / da
            t1 = (mn - oa) * inv; t2 = (mx - oa) * inv
            if t1 > t2:
                t1, t2 = t2, t1
            if t1 > t_enter: t_enter = t1
            if t2 < t_exit:  t_exit  = t2
            if t_exit < t_enter - eps:
                return 1.0, 0.0  # miss
    if t_exit < -eps:
        return 1.0, 0.0
    return t_enter, t_exit

@njit(cache=True, fastmath=False, nogil=True)
def _dda_count_single(o, d, gmin, s, nx, ny, nz, id_grid,
                      occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                      eps):
    # normalize (optional)
    n = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]) ** 0.5
    if n > 0:
        d = np.array([d[0]/n, d[1]/n, d[2]/n])

    gmax = np.array([gmin[0] + s*nx, gmin[1] + s*ny, gmin[2] + s*nz], dtype=np.float64)
    tmin_g, tmax_g = _slab(o, d, gmin, gmax, eps)
    tmin_o, tmax_o = _slab(o, d, occ_min_xyz, occ_max_xyz, eps)
    if tmin_g > tmax_g or tmin_o > tmax_o:
        return 0
    tmin = tmin_g if tmin_g > tmin_o else tmin_o
    tmax = tmax_g if tmax_g < tmax_o else tmax_o
    if tmin > tmax:
        return 0

    # start inside [gmin,gmax)
    t = (tmin if tmin > 0.0 else 0.0) + 1e-9
    p0 = o[0] + t*d[0]; p1 = o[1] + t*d[1]; p2 = o[2] + t*d[2]

    if p0 < gmin[0]: p0 = gmin[0]
    if p1 < gmin[1]: p1 = gmin[1]
    if p2 < gmin[2]: p2 = gmin[2]
    gmaxm0 = np.nextafter(gmax[0], gmin[0])
    gmaxm1 = np.nextafter(gmax[1], gmin[1])
    gmaxm2 = np.nextafter(gmax[2], gmin[2])
    if p0 > gmaxm0: p0 = gmaxm0
    if p1 > gmaxm1: p1 = gmaxm1
    if p2 > gmaxm2: p2 = gmaxm2

    ix = int((p0 - gmin[0]) // s)
    iy = int((p1 - gmin[1]) // s)
    iz = int((p2 - gmin[2]) // s)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 0

    step_x = 0 if abs(d[0]) < 1e-12 else (1 if d[0] > 0 else -1)
    step_y = 0 if abs(d[1]) < 1e-12 else (1 if d[1] > 0 else -1)
    step_z = 0 if abs(d[2]) < 1e-12 else (1 if d[2] > 0 else -1)

    def axis_params(o_i, d_i, i0, axis, step):
        if step == 0:
            return 1e300, 1e300
        nextb = gmin[axis] + ((i0 + 1)*s if step > 0 else i0*s)
        tMax  = (nextb - o_i) / d_i
        if tMax < 0.0: tMax = 0.0
        tDelta = s / abs(d_i)
        return tMax, tDelta

    tMaxX, tDeltaX = axis_params(o[0], d[0], ix, 0, step_x)
    tMaxY, tDeltaY = axis_params(o[1], d[1], iy, 1, step_y)
    tMaxZ, tDeltaZ = axis_params(o[2], d[2], iz, 2, step_z)

    minx, miny, minz = occ_min_idx[0], occ_min_idx[1], occ_min_idx[2]
    maxx, maxy, maxz = occ_max_idx[0], occ_max_idx[1], occ_max_idx[2]

    count = 0
    while (ix >= minx and ix <= maxx and
           iy >= miny and iy <= maxy and
           iz >= minz and iz <= maxz):
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            t_next = tMaxX; axis = 0
        elif tMaxY <= tMaxZ:
            t_next = tMaxY; axis = 1
        else:
            t_next = tMaxZ; axis = 2

        # only occupied get counted
        if id_grid[ix, iy, iz] != -1:
            count += 1

        if t_next > tmax + eps:
            break

        if axis == 0:
            ix += step_x; tMaxX += tDeltaX
        elif axis == 1:
            iy += step_y; tMaxY += tDeltaY
        else:
            iz += step_z; tMaxZ += tDeltaZ

    return count

@njit(parallel=True, cache=False, fastmath=False, nogil=True)
def ray_count_kernel_parallel(origins, dirs, gmin, s, nx, ny, nz, id_grid,
                              occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                              eps, counts):
    n = origins.shape[0]
    for r in prange(n):
        counts[r] = _dda_count_single(origins[r], dirs[r], gmin, s, nx, ny, nz, id_grid,
                                      occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                      eps)


@njit(parallel=False, cache=False, fastmath=False, nogil=True)
def ray_count_kernel_serial(origins, dirs, gmin, s, nx, ny, nz, id_grid,
                            occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                            eps, counts):
    n = origins.shape[0]
    for r in range(n):
        counts[r] = _dda_count_single(origins[r], dirs[r], gmin, s, nx, ny, nz, id_grid,
                                      occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                      eps)

@njit(cache=True, fastmath=False, nogil=True)
def _dda_write_single(o, d, gmin, s, nx, ny, nz, id_grid,
                      occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                      eps, start, out_ix, out_iy, out_iz, out_t0, out_t1, out_ray):
    # (same traversal as count, but write occupied hits to flattened arrays)
    n = (d[0]*d[0] + d[1]*d[1] + d[2]*d[2]) ** 0.5
    if n > 0:
        d = np.array([d[0]/n, d[1]/n, d[2]/n])

    gmax = np.array([gmin[0] + s*nx, gmin[1] + s*ny, gmin[2] + s*nz], dtype=np.float64)
    tmin_g, tmax_g = _slab(o, d, gmin, gmax, eps)
    tmin_o, tmax_o = _slab(o, d, occ_min_xyz, occ_max_xyz, eps)
    if tmin_g > tmax_g or tmin_o > tmax_o:
        return 0
    tmin = tmin_g if tmin_g > tmin_o else tmin_o
    tmax = tmax_g if tmax_g < tmax_o else tmax_o
    if tmin > tmax:
        return 0

    t = (tmin if tmin > 0.0 else 0.0) + 1e-9
    p0 = o[0] + t*d[0]; p1 = o[1] + t*d[1]; p2 = o[2] + t*d[2]
    if p0 < gmin[0]: p0 = gmin[0]
    if p1 < gmin[1]: p1 = gmin[1]
    if p2 < gmin[2]: p2 = gmin[2]
    gmaxm0 = np.nextafter(gmax[0], gmin[0])
    gmaxm1 = np.nextafter(gmax[1], gmin[1])
    gmaxm2 = np.nextafter(gmax[2], gmin[2])
    if p0 > gmaxm0: p0 = gmaxm0
    if p1 > gmaxm1: p1 = gmaxm1
    if p2 > gmaxm2: p2 = gmaxm2

    ix = int((p0 - gmin[0]) // s)
    iy = int((p1 - gmin[1]) // s)
    iz = int((p2 - gmin[2]) // s)
    if ix < 0 or ix >= nx or iy < 0 or iy >= ny or iz < 0 or iz >= nz:
        return 0

    step_x = 0 if abs(d[0]) < 1e-12 else (1 if d[0] > 0 else -1)
    step_y = 0 if abs(d[1]) < 1e-12 else (1 if d[1] > 0 else -1)
    step_z = 0 if abs(d[2]) < 1e-12 else (1 if d[2] > 0 else -1)

    def axis_params(o_i, d_i, i0, axis, step):
        if step == 0:
            return 1e300, 1e300
        nextb = gmin[axis] + ((i0 + 1)*s if step > 0 else i0*s)
        tMax  = (nextb - o_i) / d_i
        if tMax < 0.0: tMax = 0.0
        tDelta = s / abs(d_i)
        return tMax, tDelta

    tMaxX, tDeltaX = axis_params(o[0], d[0], ix, 0, step_x)
    tMaxY, tDeltaY = axis_params(o[1], d[1], iy, 1, step_y)
    tMaxZ, tDeltaZ = axis_params(o[2], d[2], iz, 2, step_z)

    minx, miny, minz = occ_min_idx[0], occ_min_idx[1], occ_min_idx[2]
    maxx, maxy, maxz = occ_max_idx[0], occ_max_idx[1], occ_max_idx[2]

    wrote = 0
    t_curr = t
    while (ix >= minx and ix <= maxx and
           iy >= miny and iy <= maxy and
           iz >= minz and iz <= maxz):
        if tMaxX <= tMaxY and tMaxX <= tMaxZ:
            t_next = tMaxX; axis = 0
        elif tMaxY <= tMaxZ:
            t_next = tMaxY; axis = 1
        else:
            t_next = tMaxZ; axis = 2

        if id_grid[ix, iy, iz] != -1:
            out_ix[start + wrote] = ix
            out_iy[start + wrote] = iy
            out_iz[start + wrote] = iz
            out_t0[start + wrote] = t_curr
            out_t1[start + wrote] = (t_next if t_next < tmax else tmax)
            wrote += 1

        if t_next > tmax + eps:
            break

        t_curr = t_next
        if axis == 0:
            ix += step_x; tMaxX += tDeltaX
        elif axis == 1:
            iy += step_y; tMaxY += tDeltaY
        else:
            iz += step_z; tMaxZ += tDeltaZ

    return wrote

@njit(parallel=True, cache=False, fastmath=False, nogil=True)
def ray_write_kernel_parallel(origins, dirs, gmin, s, nx, ny, nz, id_grid,
                              occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                              eps, offsets,
                              out_ix, out_iy, out_iz, out_t0, out_t1, out_ray):
    n = origins.shape[0]
    for r in prange(n):
        start = offsets[r]
        wrote = _dda_write_single(origins[r], dirs[r], gmin, s, nx, ny, nz, id_grid,
                                  occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                  eps, start, out_ix, out_iy, out_iz, out_t0, out_t1, out_ray)
        # tag which ray each hit came from
        for k in range(wrote):
            out_ray[start + k] = r


@njit(parallel=False, cache=False, fastmath=False, nogil=True)
def ray_write_kernel_serial(origins, dirs, gmin, s, nx, ny, nz, id_grid,
                            occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                            eps, offsets,
                            out_ix, out_iy, out_iz, out_t0, out_t1, out_ray):
    n = origins.shape[0]
    for r in range(n):
        start = offsets[r]
        wrote = _dda_write_single(origins[r], dirs[r], gmin, s, nx, ny, nz, id_grid,
                                  occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                  eps, start, out_ix, out_iy, out_iz, out_t0, out_t1, out_ray)
        for k in range(wrote):
            out_ray[start + k] = r

def warmup_numba_kernels(grid, epsilon, kernel_mode: str):
    # tiny fake data
    origins = np.zeros((2,3), np.float64)
    dirs    = np.array([[1,0,0],[0,1,0]], np.float64)
    gmin = grid["grid_min"].astype(np.float64)
    s    = float(grid["voxel_size"])
    nx,ny,nz = grid["grid_shape"]
    idg  = grid["id_grid"]
    occ_min_xyz = grid["occ_min_xyz"].astype(np.float64)
    occ_max_xyz = grid["occ_max_xyz"].astype(np.float64)
    occ_min_idx = np.array(grid["occ_min_idx"], np.int64)
    occ_max_idx = np.array(grid["occ_max_idx"], np.int64)
    counts = np.zeros(origins.shape[0], np.int32)
    if kernel_mode == "parallel":
        ray_count_kernel_parallel(origins, dirs, gmin, s, nx, ny, nz, idg,
                                  occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                  epsilon, counts)
    else:
        ray_count_kernel_serial(origins, dirs, gmin, s, nx, ny, nz, idg,
                                occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                epsilon, counts)
    # second pass (zero-sized buffers)
    offsets = np.zeros_like(counts)
    out_ix = out_iy = out_iz = np.zeros(1, np.int32)
    out_t0 = out_t1 = np.zeros(1, np.float64)
    out_ray = np.zeros(1, np.int32)
    if kernel_mode == "parallel":
        ray_write_kernel_parallel(origins, dirs, gmin, s, nx, ny, nz, idg,
                                  occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                  epsilon, offsets, out_ix, out_iy, out_iz, out_t0, out_t1, out_ray)
    else:
        ray_write_kernel_serial(origins, dirs, gmin, s, nx, ny, nz, idg,
                                occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                                epsilon, offsets, out_ix, out_iy, out_iz, out_t0, out_t1, out_ray)

import pyarrow.parquet as pq
import pandas as pd
import numpy as np
import os, math

def process_files_numba_for_size(
        files, 
        grid, 
        epsilon, 
        output_dir,
        schema=None,
        numba_threads_override: Optional[int] = None,
    kernel_mode: str = "parallel",
        show_progress: bool = True,
        quiet: bool = False,
        verbose: bool = True
    ):
    """
    Drop-in replacement for Parallel(...) for a single voxel_size grid.
    Processes files row-group-by-row-group with Numba-parallel kernels,
    using chunked rays so tqdm can update between kernel calls.

    Parameters
    ----------
    files : List[str]
        Paths to *valid_rays.parquet files for this voxel_size run.
    grid : Dict
        Dense grid for this voxel size (from build_dense_grids_from_refs), with:
        'voxel_size', 'grid_min', 'grid_shape', 'id_grid',
        'occ_min_idx','occ_max_idx','occ_min_xyz','occ_max_xyz'.
    epsilon : float
        Slab tolerance (e.g., 1e-6 for meter-scale data).
    output_dir : str
        Directory to write per-leg outputs for this voxel size.
    schema : pyarrow.Schema or None
        Optional Arrow schema for saving parquet; pass-through to to_parquet.
    chunk_rays : int
        Number of unique rays per kernel call. Larger => fewer tqdm updates; smaller => smoother bars.
    show_progress : bool
        Show tqdm bars (row-groups, count stage, write stage).
    verbosebool : bool
        Print per-leg save summaries.

    Returns
    -------
    Dict[str, int]
        Summary counters with keys: rows_written, rays_traversed.
    """
    import os, math
    import numpy as np
    import pandas as pd
    import pyarrow.parquet as pq
    from tqdm.auto import tqdm

    kernel_mode = str(kernel_mode).strip().lower()
    if kernel_mode not in {"parallel", "serial"}:
        raise ValueError(f"Unsupported kernel_mode={kernel_mode!r}; expected 'parallel' or 'serial'")

    if numba_threads_override is not None:
        set_num_threads(max(1, int(numba_threads_override)))

    rays_traversed = 0
    rows_written = 0
    echos = 0
    samples = 0
    current_leg = None
    time_start = time.time()
    rays_per_second = 0

    # --- Helpers for consistent, non-interleaving output ---
    def log(msg: str):
        """Write a line without disrupting tqdm's rendering."""
        if quiet:
            return
        if overall_bar is not None:
            overall_bar.write(msg)
        else:
            print(msg)

    def _format_int(n: int) -> str:
        return f"{n:,}"

    def _postfix_str() -> str:
        # Keep it short to avoid wrapping; ~70–80 chars is usually safe.
        # Abbrev keys and compress large ints.
        return (
            f"leg={current_leg if current_leg is not None else '-'} | "
            f"echos={_format_int(echos)} | "
            f"samples={_format_int(samples)} | "
            f"{_format_int(rays_per_second)} rays/s"
        )

    def _viewing_angle_deg_single(dx: float, dy: float, dz: float, eps: float = 1e-9) -> float:
        """Angle between +Z and (dx,dy,dz) clamped to [0, 90] by folding >90 to 180-angle."""
        n = math.sqrt(dx*dx + dy*dy + dz*dz)
        if n < eps:
            return 0.0
        c = max(-1.0, min(1.0, dz / n))
        ang = math.degrees(math.acos(c))
        return ang if ang <= 90.0 else 180.0 - ang

    def _concat_and_save_leg(leg_id: int, blocks: list, s_val: float) -> int:
        """Concat row blocks for one leg and save a single parquet file."""
        if not blocks:
            return 0
        df = pd.concat(blocks, ignore_index=True)
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, f"leg_{int(leg_id)}_voxel_{s_val:.1f}_intersections.parquet")
        df.to_parquet(out_path, engine="pyarrow", compression="snappy",
                      index=False, schema=schema)
        if verbose:
            log(f"  ✓ Saved {out_path} ({len(df):,} rows)")
        return len(df)

    # --- unpack grid & warmup kernels -------------------------------------
    s = float(grid["voxel_size"])
    gmin = grid["grid_min"].astype(np.float64)
    nx, ny, nz = grid["grid_shape"]
    id_grid = grid["id_grid"]
    occ_min_xyz = grid["occ_min_xyz"].astype(np.float64)
    occ_max_xyz = grid["occ_max_xyz"].astype(np.float64)
    occ_min_idx = np.array(grid["occ_min_idx"], np.int64)
    occ_max_idx = np.array(grid["occ_max_idx"], np.int64)

    if kernel_mode == "parallel":
        count_kernel = ray_count_kernel_parallel
        write_kernel = ray_write_kernel_parallel
    else:
        count_kernel = ray_count_kernel_serial
        write_kernel = ray_write_kernel_serial

    # Avoid repeated warmup overhead when a worker handles many single-file tasks.
    warmup_key = (
        kernel_mode,
        int(nx),
        int(ny),
        int(nz),
        float(s),
    )
    if warmup_key not in _VOXEL_RI_WARMED_KERNELS:
        warmup_numba_kernels(grid, epsilon, kernel_mode=kernel_mode)
        _VOXEL_RI_WARMED_KERNELS.add(warmup_key)

    # --- Pre-scan Parquet files to know the overall total (row groups) ---
    pf_meta = []  # list of tuples: (pf_path, pfh, leg_id, num_row_groups)
    overall_total_rgs = 0
    if show_progress:
        for _pf in files:
            _pfh = pq.ParquetFile(_pf)
            _leg = leg_from_filename(_pf)
            _nrg = _pfh.num_row_groups
            pf_meta.append((_pf, _pfh, _leg, _nrg))
            overall_total_rgs += _nrg
    else:
        # Keep the original behavior: we will open lazily inside the loop
        pf_meta = [(pf, None, None, None) for pf in files]

    # --- NEW: Create the overall tqdm bar (one bar for the whole process) -----
    overall_bar = None
    def _create_overall_bar():
        if not show_progress:
            return None
        return tqdm(
            total=overall_total_rgs,
            desc=f"Overall (voxel={float(grid['voxel_size']):.1f} m)",
            position=0,
            leave=True,
            dynamic_ncols=True,
            smoothing=0.1,
        )

    if show_progress:
        overall_bar = _create_overall_bar()


    def update_process():
        """
        Update the overall tqdm bar's postfix with the current processing summary.
        Replaces the previous in-place terminal redraw, while keeping the same info.
        """
        # Build a compact postfix (post-text) for the bar
        if overall_bar is not None:
        # Note: set_postfix is efficient; refresh triggers a single re-render
            overall_bar.set_postfix_str(_postfix_str(), refresh=True)
        elif not quiet:
            log("Summary | " + _postfix_str())

    # --- main loop over files (legs) ---------------------------------------
    for pf, pfh_cached, leg_id_cached, nrg_cached in pf_meta:
        pfh = pfh_cached if pfh_cached is not None else pq.ParquetFile(pf)
        leg_id = leg_id_cached if leg_id_cached is not None else leg_from_filename(pf)
        current_leg = leg_id

        log(f"Processing leg {leg_id} with {pfh.num_row_groups} row groups...")
        log(f"  Grid: s={s:.3f}m, shape={nx}x{ny}x{nz}, occupied AABB idx {occ_min_idx} to {occ_max_idx}")

        out_blocks = []   # collect per-rowgroup blocks for this leg
        update_process()

        for rg in range(pfh.num_row_groups):
            # Read row group into pandas
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
                if overall_bar is not None:
                    overall_bar.update(1)
                    update_process()
                continue

            # --- Build unique rays mapping (as in your previous logic)
            ray_id = tbl["ray_id"].to_numpy(np.int64)
            _, ur_first, ur_inv = np.unique(ray_id, return_index=True, return_inverse=True)
            uniq_orig = tbl[["origin_x","origin_y","origin_z"]].to_numpy(np.float64)[ur_first]
            uniq_dirs = tbl[["direction_x","direction_y","direction_z"]].to_numpy(np.float64)[ur_first]
            N = len(uniq_orig)

            if N == 0:
                if overall_bar is not None:
                    overall_bar.update(1)
                    update_process()
                continue
            rays_traversed += N
            rays_per_second = rays_traversed // (time.time() - time_start)
            update_process()

            # -------- PASS 1: COUNT (chunked with tqdm) ---------------------
            counts = np.zeros(N, np.int32)
            count_kernel(
                uniq_orig, uniq_dirs,
                gmin, s, nx, ny, nz, id_grid,
                occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                epsilon, counts
            )

            total_hits = int(counts.sum())
            if total_hits == 0:
                if overall_bar is not None:
                    overall_bar.update(1)
                    update_process()
                continue

            # Offsets (prefix sum) for flattened hit arrays
            offsets = np.zeros_like(counts)
            if N > 1:
                np.cumsum(counts[:-1], out=offsets[1:])

            # Allocate flattened buffers for write pass
            out_ix = np.empty(total_hits, np.int32)
            out_iy = np.empty(total_hits, np.int32)
            out_iz = np.empty(total_hits, np.int32)
            out_t0 = np.empty(total_hits, np.float64)
            out_t1 = np.empty(total_hits, np.float64)
            out_ray = np.empty(total_hits, np.int32)

            # -------- PASS 2: WRITE (no chunking) ---------------------
            write_kernel(
                uniq_orig, uniq_dirs,
                gmin, s, nx, ny, nz, id_grid,
                occ_min_xyz, occ_max_xyz, occ_min_idx, occ_max_idx,
                epsilon, offsets,
                out_ix, out_iy, out_iz,
                out_t0, out_t1, out_ray
            )

            # -------- EXPAND flattened hits to rows (Python) ----------------
            scan_id = tbl["scan_id"].fillna(0).astype(np.int64).to_numpy()
            pts     = tbl[["point_x","point_y","point_z"]].to_numpy(np.float64)
            echo    = tbl["echo_intensity"].fillna(0.0).to_numpy(np.float64)
            ret_no  = tbl["return_number"].fillna(0).astype(np.int32).to_numpy()
            ret_cnt = tbl["number_of_returns"].fillna(0).astype(np.int32).to_numpy()
            is_leaf = tbl["is_ leaf".replace("_ ", "_")].fillna(False).astype(bool).to_numpy()  # typo guard

            # mapping unique->original rows
            order = np.argsort(ur_inv)
            inv_sorted = ur_inv[order]
            split = np.flatnonzero(np.diff(inv_sorted)) + 1
            uniq_to_orig = np.split(order, split)

            rows = []
            s_half = s * 0.5
            eps = float(epsilon)

            
            # 1) Per-hit values, vectorized
            hit_ray = out_ray                     # (H,)
            ix = out_ix.astype(np.int64); iy = out_iy.astype(np.int64); iz = out_iz.astype(np.int64)  # (H,)
            vid = id_grid[ix, iy, iz]             # (H,)
            cx = gmin[0] + (ix + 0.5) * s         # (H,)
            cy = gmin[1] + (iy + 0.5) * s
            cz = gmin[2] + (iz + 0.5) * s

            o_hits = uniq_orig[hit_ray]           # (H,3)
            d_hits = uniq_dirs[hit_ray]           # (H,3)
            entry = o_hits + out_t0[:,None] * d_hits    # (H,3)
            exit_  = o_hits + out_t1[:,None] * d_hits

            # 2) Group hits per ray (split indices once)
            H = len(out_ray); N = len(uniq_orig)
            hits_per_ray = np.bincount(hit_ray, minlength=N)     # (N,)
            # prefix for rays to split hit indices
            hit_prefix = np.zeros(N+1, np.int64); np.cumsum(hits_per_ray, out=hit_prefix[1:])
            # per-ray hit index ranges [hit_prefix[r]:hit_prefix[r+1])

            # 3) For each ray, we also need the original row indices that map to this ray
            # You already have uniq_to_orig: list of np arrays of row indices for each ray r
            orig_per_ray = np.array([len(g) for g in uniq_to_orig], dtype=np.int64)   # (N,)
            # total expanded rows:
            K = int(np.sum(hits_per_ray * orig_per_ray))

            # 4) Preallocate all output columns as NumPy arrays of length K (FAST)
            voxel_id_col  = np.empty(K, np.int64)
            voxel_cx_col  = np.empty(K, np.float64)
            voxel_cy_col  = np.empty(K, np.float64)
            voxel_cz_col  = np.empty(K, np.float64)
            scan_id_col   = np.empty(K, np.int64)
            ray_id_col    = np.empty(K, np.int64)
            t_entry_x_col = np.empty(K, np.float64); t_entry_y_col = np.empty(K, np.float64); t_entry_z_col = np.empty(K, np.float64)
            t_exit_x_col  = np.empty(K, np.float64); t_exit_y_col  = np.empty(K, np.float64); t_exit_z_col  = np.empty(K, np.float64)
            point_x_col   = np.empty(K, np.float64); point_y_col   = np.empty(K, np.float64); point_z_col   = np.empty(K, np.float64)
            echo_col      = np.empty(K, np.float64)
            ret_no_col    = np.empty(K, np.int32)
            ret_cnt_col   = np.empty(K, np.int32)
            is_leaf_col   = np.empty(K, np.bool_)
            view_ang_col  = np.empty(K, np.float64)
            hit_type_col  = np.empty(K, np.int32)
            dist_ctr_col  = np.empty(K, np.float64)

            # 5) We’ll also prep row-wise arrays ONCE (no .iat in loops)
            scan_id_arr = scan_id                    # (R,)
            ray_id_arr  = ray_id                     # (R,)
            orig_x = tbl["origin_x"].to_numpy(np.float64)
            orig_y = tbl["origin_y"].to_numpy(np.float64)
            orig_z = tbl["origin_z"].to_numpy(np.float64)
            dir_x  = tbl["direction_x"].to_numpy(np.float64)
            dir_y  = tbl["direction_y"].to_numpy(np.float64)
            dir_z  = tbl["direction_z"].to_numpy(np.float64)
            pt_x   = pts[:,0]; pt_y = pts[:,1]; pt_z = pts[:,2]

            # 6) Small helper: viewing angle vectorized for rows (one-time per row group)
            def viewing_angle_vec(dx, dy, dz, eps=1e-9):
                n = np.sqrt(dx*dx + dy*dy + dz*dz); n = np.where(n < eps, eps, n)
                c = np.clip(dz / n, -1.0, 1.0)
                ang = np.degrees(np.arccos(c))
                return np.where(ang <= 90.0, ang, 180.0 - ang)

            view_angle_rows = viewing_angle_vec(dir_x, dir_y, dir_z)  # (R,)

            # 7) Fill slices per ray (no inner hit‑loop)
            write_pos = 0
            s_half = s * 0.5; eps = float(epsilon)

            for r in range(N):
                n_hits = hits_per_ray[r]
                if n_hits == 0:
                    continue
                hit_lo = hit_prefix[r]; hit_hi = hit_prefix[r+1]
                hit_idx = np.arange(hit_lo, hit_hi, dtype=np.int64)

                # per-hit values for this ray
                vid_r = vid[hit_lo:hit_hi]          # (n_hits,)
                cx_r  = cx[hit_lo:hit_hi]
                cy_r  = cy[hit_lo:hit_hi]
                cz_r  = cz[hit_lo:hit_hi]
                ex_r  = entry[hit_lo:hit_hi, :]     # (n_hits,3)
                ee_r  = exit_[hit_lo:hit_hi, :]

                # original row indices for this ray
                orig_idx = uniq_to_orig[r]          # (m_r,)
                m = len(orig_idx)

                # target slice in the big K arrays
                L = n_hits * m
                sl = slice(write_pos, write_pos + L)

                # expand by Kronecker product style broadcasting
                # Repeat each hit m times; tile the row arrays n_hits times
                voxel_id_col[sl] = np.repeat(vid_r, m)
                voxel_cx_col[sl] = np.repeat(cx_r, m)
                voxel_cy_col[sl] = np.repeat(cy_r, m)
                voxel_cz_col[sl] = np.repeat(cz_r, m)

                # entry/exit xyz
                t_entry_x_col[sl] = np.repeat(ex_r[:,0], m)
                t_entry_y_col[sl] = np.repeat(ex_r[:,1], m)
                t_entry_z_col[sl] = np.repeat(ex_r[:,2], m)
                t_exit_x_col[sl]  = np.repeat(ee_r[:,0], m)
                t_exit_y_col[sl]  = np.repeat(ee_r[:,1], m)
                t_exit_z_col[sl]  = np.repeat(ee_r[:,2], m)

                # ray and scan meta: tile row arrays
                rr = np.tile(orig_idx, n_hits)      # (L,)
                scan_id_col[sl]  = scan_id_arr[rr]
                ray_id_col[sl]   = ray_id_arr[rr]
                point_x_col[sl]  = pt_x[rr]; point_y_col[sl] = pt_y[rr]; point_z_col[sl] = pt_z[rr]
                echo_col[sl]     = echo[rr]
                ret_no_col[sl]   = ret_no[rr]
                ret_cnt_col[sl]  = ret_cnt[rr]
                is_leaf_col[sl]  = is_leaf[rr]
                view_ang_col[sl] = view_angle_rows[rr]

                # distances
                ox = orig_x[rr]; oy = orig_y[rr]; oz = orig_z[rr]
                de = (ox - t_entry_x_col[sl])**2 + (oy - t_entry_y_col[sl])**2 + (oz - t_entry_z_col[sl])**2
                dx_ = (ox - t_exit_x_col[sl])**2  + (oy - t_exit_y_col[sl])**2  + (oz - t_exit_z_col[sl])**2
                dp = (point_x_col[sl] - ox)**2 + (point_y_col[sl] - oy)**2 + (point_z_col[sl] - oz)**2

                unbound = np.isnan(point_x_col[sl]) | np.isnan(point_y_col[sl]) | np.isnan(point_z_col[sl])

                # Classification of hit types
                cx_rep = np.repeat(cx_r, m)
                cy_rep = np.repeat(cy_r, m)
                cz_rep = np.repeat(cz_r, m)
                in_vox = (cx_rep - s_half - eps <= point_x_col[sl]) & (point_x_col[sl] <= cx_rep + s_half + eps) & \
                         (cy_rep - s_half - eps <= point_y_col[sl]) & (point_y_col[sl] <= cy_rep + s_half + eps) & \
                         (cz_rep - s_half - eps <= point_z_col[sl]) & (point_z_col[sl] <= cz_rep + s_half + eps)
                dp = (point_x_col[sl] - ox)**2 + (point_y_col[sl] - oy)**2 + (point_z_col[sl] - oz)**2
                before = (dp < de) & (~in_vox) & (~unbound)
                after  = (dp >= dx_) & (~in_vox) & (~unbound)

                ht = np.full(L, -1, np.int32)
                ht[unbound] = 0
                ht[before]  = 1
                ht[in_vox]  = 2
                ht[after]   = 3
                hit_type_col[sl] = ht

                # distance_to_centre
                cx_rep = np.repeat(cx_r, m)
                cy_rep = np.repeat(cy_r, m)
                cz_rep = np.repeat(cz_r, m)
                dist_ctr_col[sl] = np.sqrt((ox - cx_rep)**2 + (oy - cy_rep)**2 + (oz - cz_rep)**2)

                write_pos += L

            # 8) Finally build the DataFrame columnar (VERY fast)
            out_df = pd.DataFrame({
                "voxel_size":      np.full(K, s, np.float64),
                "voxel_id":        voxel_id_col,
                "voxel_cx":        voxel_cx_col, "voxel_cy": voxel_cy_col, "voxel_cz": voxel_cz_col,
                "scan_id":         scan_id_col,
                "ray_id":          ray_id_col,
                "t_entry_x":       t_entry_x_col, "t_entry_y": t_entry_y_col, "t_entry_z": t_entry_z_col,
                "t_exit_x":        t_exit_x_col,  "t_exit_y":  t_exit_y_col,  "t_exit_z":  t_exit_z_col,
                "distance_to_centre": dist_ctr_col,
                "point_x":         point_x_col, "point_y": point_y_col, "point_z": point_z_col,
                "echo_intensity":  echo_col,
                "return_number":   ret_no_col,
                "number_of_returns": ret_cnt_col,
                "viewing_angle":   view_ang_col,
                "hit_type":        hit_type_col,
                "is_leaf":         is_leaf_col,
            })

            samples += len(out_ray)
            echos += (hit_type_col == 2).sum()
            if overall_bar is not None:
                overall_bar.update(1)
            update_process()

            if not out_df.empty:
                out_blocks.append(out_df)

        # Save once per leg (to avoid clobbering across row groups)
        rows_written += _concat_and_save_leg(leg_id, out_blocks, s)
    
    if overall_bar is not None:
        overall_bar.close()

    return {
        "rows_written": int(rows_written),
        "rays_traversed": int(rays_traversed),
    }


def _process_single_leg_task(
    pf: str,
    grid,
    epsilon: float,
    valid_rays_dir: str,
    numba_threads_override: int,
    kernel_mode: str,
    verbose: bool,
):
    """Top-level worker entrypoint for process-based leg parallelism."""
    _threads = max(1, int(numba_threads_override))
    # Keep native thread pools aligned with per-worker budget.
    os.environ["OMP_NUM_THREADS"] = str(_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(_threads)
    os.environ["MKL_NUM_THREADS"] = str(_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(_threads)
    set_num_threads(_threads)

    return process_files_numba_for_size(
        [pf],
        grid,
        epsilon,
        valid_rays_dir,
        schema=None,
        numba_threads_override=numba_threads_override,
        kernel_mode=kernel_mode,
        show_progress=False,
        quiet=True,
        verbose=verbose,
    )
# ------------------------------------------------------------
# Helper: Detect CPU count from HPC environment
# ------------------------------------------------------------
def _detect_num_cpus() -> int:
    """
    Detect available CPUs for Numba parallelization.
    Checks SLURM environment first (HPC systems), then falls back to os.cpu_count().
    
    Returns
    -------
    int
        Number of CPUs to use for Numba parallelization.
    """
    # Check SLURM_CPUS_PER_TASK (set by Slurm job scheduler on HPC)
    if 'SLURM_CPUS_PER_TASK' in os.environ:
        try:
            cpus = int(os.environ['SLURM_CPUS_PER_TASK'])
            if cpus > 0:
                return cpus
        except (ValueError, TypeError):
            pass
    
    # Check OMP_NUM_THREADS (OpenMP standard)
    if 'OMP_NUM_THREADS' in os.environ:
        try:
            cpus = int(os.environ['OMP_NUM_THREADS'])
            if cpus > 0:
                return cpus
        except (ValueError, TypeError):
            pass
    
    # Fall back to os.cpu_count()
    cpus = os.cpu_count()
    if cpus and cpus > 0:
        return cpus
    
    # Final fallback
    return 1

# ------------------------------------------------------------
# Main function: traversal-first voxel-ray intersections
# ------------------------------------------------------------
def voxel_ray_intersections(valid_rays_dir: str,
                                   references_dir: str,
                                   *,
                                   epsilon: float = 1e-6,
                                   n_jobs: int = -1,
                                   leg_workers: Optional[int] = None,
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
    log("[voxel_ray_intersections] Starting voxel-ray intersection computation")
    log("=" * 70)

    # Load voxel reference CSVs (shared across workers)
    log("\n[1/5] Loading voxel reference CSVs...")
    refdf = compile_voxel_refs(references_dir)
    log(f"  ✓ Loaded {len(refdf)} voxel references")

    # Infer grid geometry per voxel_size (shared across workers)
    log("\n[2/5] Building spatial grids per voxel size...")
    grids = build_dense_grids_from_refs(refdf, eps=epsilon)
    for vs, g in grids.items():
        nx, ny, nz = g["grid_shape"]
        s = g["voxel_size"]
        n_voxels = (g["id_grid"] != -1).sum()
        log(f"  ✓ Grid {s}m: shape {nx}×{ny}×{nz}, {n_voxels} active voxels")

    # List input files
    files = list_parquet_files(valid_rays_dir)
    log(f"\n[3/5] Processing {len(files)} parquet file(s) in parallel...")

    # Fast metadata scan: capture row-groups per leg for thread-aware scheduling.
    row_groups_by_file = {}
    row_groups_per_leg = []
    if files:
        for pf in files:
            try:
                nrg = pq.ParquetFile(pf).num_row_groups
            except Exception:
                # Conservative fallback if metadata read fails for any leg.
                nrg = 1
            nrg = max(1, int(nrg))
            row_groups_by_file[pf] = nrg
            row_groups_per_leg.append(nrg)
        log(
            f"  ✓ Parquet metadata: max_row_groups_per_leg={max(row_groups_per_leg)}, "
            f"min={min(row_groups_per_leg)}, mean={float(np.mean(row_groups_per_leg)):.2f}"
        )

    # Configure Numba threads: use n_jobs if specified, otherwise detect from HPC environment
    if n_jobs >= 0:
        nthreads = max(1, n_jobs)
    else:
        nthreads = _detect_num_cpus()
    set_num_threads(nthreads)
    from numba import get_num_threads
    actual_threads = get_num_threads()
    log(f"  ✓ Configured Numba for {nthreads} threads (actual: {actual_threads})")
    log(f"  ℹ Environment: SLURM_CPUS_PER_TASK={os.environ.get('SLURM_CPUS_PER_TASK', 'not set')}, OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'not set')}")

    def _should_use_tqdm() -> bool:
        import sys

        try:
            return bool(getattr(sys.stderr, "isatty", lambda: False)())
        except Exception:
            return False

    try:
        import concurrent.futures
        import math

        if row_groups_by_file:
            _rg_values = np.array([max(1, int(v)) for v in row_groups_by_file.values()], dtype=np.int32)
            observed_p90_row_groups = int(max(1, math.ceil(float(np.percentile(_rg_values, 90)))))
        else:
            observed_p90_row_groups = 1

        n_files = len(files)
        if n_files >= observed_p90_row_groups:
            parallel_mode = "file"
        else:
            parallel_mode = "rowgroup"

        log(
            "  ✓ Parallel mode decision (fixed for run): "
            f"mode={parallel_mode}, files={n_files}, p90_row_groups={observed_p90_row_groups}"
        )

        env_max_workers_raw = os.environ.get("VOXEL_RI_MAX_WORKERS")
        env_max_workers = None
        if env_max_workers_raw:
            try:
                env_max_workers = max(1, int(env_max_workers_raw))
            except (TypeError, ValueError):
                env_max_workers = None

        for vs, grid in grids.items():
            import gc
            gc.collect()

            if parallel_mode == "file" and n_files > 1:
                cpu_budget = max(1, int(nthreads))
                if leg_workers is not None:
                    effective_leg_workers = max(1, int(leg_workers))
                else:
                    effective_leg_workers = cpu_budget

                if env_max_workers is not None:
                    effective_leg_workers = min(effective_leg_workers, env_max_workers)

                effective_leg_workers = max(1, min(effective_leg_workers, n_files, cpu_budget))

                # File-parallel mode uses serial kernels only. Never enable Numba parallelism here.
                threads_per_worker = 1

                log(
                    f"  ✓ File parallelism only: {effective_leg_workers} process worker(s) × "
                    f"{threads_per_worker} Numba thread(s)/worker"
                )

                total_legs = n_files
                desc = f"Legs (voxel={float(grid['voxel_size']):.1f}m, file parallel only)"
                use_tqdm = _should_use_tqdm()

                start_ts = time.time()
                done_legs = 0
                done_rays = 0
                report_every = max(1, total_legs // 20)
                pbar = None
                if use_tqdm:
                    pbar = tqdm(total=total_legs, desc=desc, dynamic_ncols=True, leave=True)
                else:
                    print(f"  -> {desc}: started {total_legs} legs on {effective_leg_workers} worker(s)", flush=True)

                try:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=effective_leg_workers,
                    ) as ex:
                        futs = [
                            ex.submit(
                                _process_single_leg_task,
                                pf,
                                grid,
                                epsilon,
                                valid_rays_dir,
                                threads_per_worker,
                                "serial",
                                debug,
                            )
                            for pf in files
                        ]

                        for fut in concurrent.futures.as_completed(futs):
                            res = fut.result()
                            done_legs += 1
                            if isinstance(res, dict):
                                done_rays += int(res.get("rays_traversed", 0))

                            elapsed = max(1e-9, time.time() - start_ts)
                            legs_rate = done_legs / elapsed
                            rays_rate = done_rays / elapsed

                            if pbar is not None:
                                pbar.update(1)
                                pbar.set_postfix_str(
                                    f"rays={done_rays:,} | {rays_rate:,.0f} rays/s",
                                    refresh=True,
                                )
                            elif done_legs >= total_legs or done_legs % report_every == 0:
                                pct = 100.0 * done_legs / max(1, total_legs)
                                print(
                                    f"  -> {desc}: {done_legs}/{total_legs} ({pct:.1f}%) | "
                                    f"{legs_rate:.2f} legs/s | {done_rays:,} rays | {rays_rate:,.0f} rays/s",
                                    flush=True,
                                )
                finally:
                    if pbar is not None:
                        pbar.close()
            else:
                # Row-group mode uses a single process with Numba parallel kernels only.
                if leg_workers is not None:
                    log("  ℹ leg_workers ignored in row-group-only mode")
                log(f"  ✓ Row-group parallelism only: 1 process × {nthreads} Numba thread(s)")

                process_files_numba_for_size(
                    files,
                    grid,
                    epsilon,
                    valid_rays_dir,
                    schema=voxel_ray_intersection_schema,
                    numba_threads_override=nthreads,
                    kernel_mode="parallel",
                    quiet=False,
                    verbose=debug,
                )
    except Exception as e:
        log(f"  ✗ Error during processing: {e}")
        raise Exception(f"Error in process_files_numba_for_size: {e}")

    finally:
        log("✓ voxel_ray_intersections complete\n")