import os
import csv
import argparse
import json
import math
import pandas as pd
import numpy as np
import traceback
import open3d as o3d
import open3d.core as o3c
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
from utils import compute_normals_weights_from_points_parallel, calculate_inclination_angle_distribution, create_voxel_id

import datetime as dt
from typing import Optional
import sys
from tqdm import tqdm
import psutil
import gc
from functools import partial 
import joblib
from joblib import Parallel, delayed, Memory
import trimesh
import contextlib
import subprocess
import pyvista as pv
from typing import List, Tuple, Union, Optional
from pathlib import Path
    
from numba import njit, prange

global _CUDA_DEVICE_ID

global GRID_VOXEL_SIZE
global INVALID_ID
GRID_VOXEL_SIZE = 0.05  # 5cm voxels for DDA traversal
INVALID_ID = np.uint32(0xFFFFFFFF)

temp_dir = os.getenv('TMPDIR', 'tmp')
# temp_dir = "/scratch/project/veg3d/uqjrivor/Raja_Tumba_Test/tmp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
memory = Memory(location=temp_dir, verbose=1)

def get_memory_usage():
    job_id = os.getenv('SLURM_JOB_ID', 'local')
    try:
        result = subprocess.run([
            'scontrol', 'show', 'job', job_id
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = result.stdout.decode()
        for line in output.split('\n'):
            if 'maxRSS' in line:
                print(f"Slurm... Used memory: {line}")
    except Exception as e:
        used_memory = psutil.virtual_memory().used / (1024 ** 2)  # Convert to MB
        print(f"Not Slurm... Used memory: {used_memory:.2f} MB")

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """
    Context manager to patch joblib to report progress updates to a tqdm bar.
    This is necessary because joblib.Parallel doesn't have a built-in progress bar.
    """
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

@memory.cache
def load_points(file_path: str, usecols: tuple=None, leaf_keys: List[int]=None, wood_keys: List[int]=None) -> Optional[np.ndarray]:
    """
    Load points from a text file using np.loadtxt, returning None if the file is empty or invalid.
    """
    try:
        points = np.loadtxt(file_path, usecols=usecols)
        points[3].astype(np.int32)  # Ensure class column is int for key matching
        if points.size == 0:
            print(f"Empty points file found in {file_path}.")
            return None, None
        if points.ndim == 1:
            points = points.reshape(-1, 3)

        leaf_points = points[np.isin(points[:, 3], leaf_keys)] if leaf_keys is not None else None
        wood_points = points[np.isin(points[:, 3], wood_keys)] if wood_keys is not None else None

        return leaf_points, wood_points
    
    except Exception as e:
        print(f"Error loading points from {file_path}: {e}")
        return None, None
    
# Global leaf and wood mesh cache
_LEAF_POINTS = None
_WOOD_POINTS = None
_LEAF_TREE = None
_WOOD_TREE = None
_LEAF_MIN = None
_WOOD_MIN = None
_LEAF_MAX = None
_WOOD_MAX = None

_LEAF_KEYS = None
_WOOD_KEYS = None
_USE_COLS = None

def _ensure_clip_worker_points(file_path, build_tree: bool=True):
    """Ensure that the global leaf and wood point clouds are loaded in the worker process.
    """
    global _LEAF_POINTS, _WOOD_POINTS
    global _LEAF_TREE, _WOOD_TREE
    global _LEAF_MIN, _LEAF_MAX
    global _WOOD_MIN, _WOOD_MAX

    if _LEAF_POINTS is None or _WOOD_POINTS is None:
        _LEAF_POINTS, _WOOD_POINTS = load_points(file_path, usecols=_USE_COLS, leaf_keys=_LEAF_KEYS, wood_keys=_WOOD_KEYS)
        if _LEAF_POINTS is not None:
            _LEAF_MIN = _LEAF_POINTS.min(axis=0)
            _LEAF_MAX = _LEAF_POINTS.max(axis=0)
            _LEAF_POINTS_MAX = _LEAF_POINTS.max(axis=0)

            if build_tree:
                _LEAF_TREE = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(_LEAF_POINTS[:, :3]))  # built once per worker

        if _WOOD_POINTS is not None:
            _WOOD_MIN = _WOOD_POINTS.min(axis=0)
            _WOOD_MAX = _WOOD_POINTS.max(axis=0)
    
            if build_tree:
                _WOOD_TREE = o3d.geometry.KDTreeFlann(o3d.utility.Vector3dVector(_WOOD_POINTS[:, :3]))  # built once per worker

def _clip_one_pointcloud_with_aabb(points, voxel_center, voxel_size):
    """Fast candidate selection via triangle AABB overlap; then clip via PyVista."""
    if points is None or points.size == 0:
        return np.empty((0, 8), np.float64)

    half = voxel_size / 2.0
    min_bound = np.asarray(voxel_center) - half
    max_bound = np.asarray(voxel_center) + half

    inside_points = (points[:, 0] >= min_bound[0]) & (points[:, 0] <= max_bound[0]) & \
              (points[:, 1] >= min_bound[1]) & (points[:, 1] <= max_bound[1]) & \
              (points[:, 2] >= min_bound[2]) & (points[:, 2] <= max_bound[2])
    idx = np.flatnonzero(inside_points)
    if idx.size == 0:
        return np.empty((0, 8), np.float64)

    return points[idx]

def ray_box_intersection_vectorized(orig, dirs, bmin, bmax, eps=1e-12):
    safe = np.where(np.abs(dirs) < eps,
                    np.where(dirs >= 0, eps, -eps),
                    dirs)
    t1 = (bmin - orig) / safe
    t2 = (bmax - orig) / safe
    t_near = np.maximum.reduce(np.minimum(t1, t2), axis=1)
    t_far  = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    return t_near, t_far

def compute_efpl_array(d_arr, lambda_1):
    d_arr = np.asarray(d_arr)
    out = np.zeros_like(d_arr)
    mask = d_arr > 0
    # Avoid division by zero for lambda_1 == 0
    if lambda_1 == 0:
        out[mask] = d_arr[mask]
    elif lambda_1 is None:
        out[mask] = np.nan
    else:
        out[mask] = -np.log(1.0 - lambda_1 * d_arr[mask]) / lambda_1
    return out


### Ray tracing functions ###
def _grid(voxel_size, ray_spacing):
    """
    Generate a grid covering a square of size `face_len` centered at the origin.
    For full coverage when rotating the voxel, use face_len = voxel_size * sqrt(2).
    This ensures the grid covers the diagonal of the voxel after rotation.
    """
    face_len = voxel_size * np.sqrt(2)
    s = np.arange(-face_len / 2, face_len / 2 + ray_spacing, ray_spacing)
    return np.meshgrid(s, s, indexing='xy')

def generate_face_rays_bottom(vc, vs, grid):
    # XX, YY = _grid(vs * 2, spc)
    XX, YY = grid
    zface = vc[2] - vs * 2
    org = np.column_stack([vc[0] + XX.ravel(),
                           vc[1] + YY.ravel(),
                           np.full(XX.size, zface)])
    return org, np.tile([0, 0, 1], (len(org), 1))

def generate_face_rays_top(vc, vs, grid, offset=0.0):
    # XX, YY = _grid(vs * 2, spc)
    XX, YY = grid
    zface = vc[2] + vs * 2 + offset
    org = np.column_stack([vc[0] + XX.ravel(),
                           vc[1] + YY.ravel(),
                           np.full(XX.size, zface)])
    return org, np.tile([0, 0, -1], (len(org), 1))

def generate_side_rays_xplus(vc, vs, grid):
    # YY, ZZ = _grid(vs * 2, spc)
    YY, ZZ = grid
    xface = vc[0] + vs * 2
    org = np.column_stack([np.full(YY.size, xface),
                           vc[1] + YY.ravel(),
                           vc[2] + ZZ.ravel()])
    return org, np.tile([-1, 0, 0], (len(org), 1))

def generate_side_rays_xminus(vc, vs, grid, offset=0.0):
    # YY, ZZ = _grid(vs * 2, spc)
    YY, ZZ = grid
    xface = vc[0] - vs * 2 - offset
    org = np.column_stack([np.full(YY.size, xface),
                           vc[1] + YY.ravel(),
                           vc[2] + ZZ.ravel()])
    return org, np.tile([1, 0, 0], (len(org), 1))

def generate_side_rays_yplus(vc, vs, grid):
    # XX, ZZ = _grid(vs * 2, spc)
    XX, ZZ = grid
    yface = vc[1] + vs * 2
    org = np.column_stack([vc[0] + XX.ravel(),
                           np.full(XX.size, yface),
                           vc[2] + ZZ.ravel()])
    return org, np.tile([0, -1, 0], (len(org), 1))

def generate_side_rays_yminus(vc, vs, grid, offset=0.0):
    # XX, ZZ = _grid(vs * 2, spc)
    XX, ZZ = grid
    yface = vc[1] - vs * 2 - offset
    org = np.column_stack([vc[0] + XX.ravel(),
                           np.full(XX.size, yface),
                           vc[2] + ZZ.ravel()])
    return org, np.tile([0, 1, 0], (len(org), 1))

def rotate_rays(orig, dirs, angle_deg, vc, axis="x"):
    ang = np.radians(angle_deg)
    if axis == "x":
        R = np.array([[1, 0, 0],
                      [0, np.cos(ang), -np.sin(ang)],
                      [0, np.sin(ang),  np.cos(ang)]])
    else:  # yaxis
        R = np.array([[ np.cos(ang), 0, np.sin(ang)],
                      [ 0,           1, 0           ],
                      [-np.sin(ang), 0, np.cos(ang)]])
    o = (orig - vc) @ R.T + vc
    d = dirs @ R.T
    return o, d


import os
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

# -------------------------
# Helpers (new)
# -------------------------
def _normalize(v, axis=-1, eps=1e-12):
    n = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / (n + eps)

def _weight_to_radius(weights, voxel_size, r_min_factor=0.25, r_max_factor=0.75, invert=True):
    """
    Map weight -> disk radius (meters).
      - invert=True: high weight => small radius.
    """
    r_min = float(r_min_factor) * float(voxel_size)
    r_max = float(r_max_factor) * float(voxel_size)
    w = np.asarray(weights, dtype=np.float32)
    w = np.clip(w, 0.0, 1.0)
    if invert:
        w = 1.0 - w
    return r_min + w * (r_max - r_min)

def _ray_disk_first_hit(ro, rd, P, N, R, tmin, tmax, t_eps=1e-6):
    """
    Ray against M disks (surfel):
      - each disk i lies in plane (P[i], N[i]) with radius R[i]
    Returns (t_hit, j_global) where j_global indexes into candidate rows,
    or (np.inf, -1) if no hit.

    Vectorized over candidates; rd is normalized.
    """
    # Avoid candidates with invalid normals
    nan_mask = np.any(~np.isfinite(N), axis=1)
    if np.all(nan_mask):
        return np.inf, -1
    if np.any(nan_mask):
        keep = ~nan_mask
        P, N, R = P[keep], N[keep], R[keep]
        idx_map = np.nonzero(keep)[0]
    else:
        idx_map = None

    denom = N @ rd  # (M,)
    valid = np.abs(denom) > 1e-8
    if not np.any(valid):
        return np.inf, -1

    if idx_map is not None:
        idx_valid = idx_map[valid]
    else:
        idx_valid = np.nonzero(valid)[0]

    Pv = P[valid]
    Nv = N[valid]
    Rv = R[valid]
    denom_v = denom[valid]

    # t where ray intersects each candidate plane
    t = ((Pv - ro) * Nv).sum(axis=1) / denom_v  # (M_valid,)

    # Restrict to the [tmin, tmax] segment (and positive)
    seg_mask = (t >= max(tmin, t_eps)) & (t <= tmax)
    if not np.any(seg_mask):
        return np.inf, -1

    t_seg = t[seg_mask]
    idx_seg = idx_valid[seg_mask]
    Pv_seg = Pv[seg_mask]
    Rv_seg = Rv[seg_mask]

    # In-plane radial test
    hit_pts = ro + t_seg[:, None] * rd
    radial = np.linalg.norm(hit_pts - Pv_seg, axis=1)
    ok = radial <= (Rv_seg + 1e-6)
    if not np.any(ok):
        return np.inf, -1

    t_hits = t_seg[ok]
    idx_hits = idx_seg[ok]

    j = np.argmin(t_hits)
    return float(t_hits[j]), int(idx_hits[j])

@njit(cache=True, fastmath=True)  # see §4
def ray_disk_first_hit_vec(ro, rd, Pxyz, Nn, Radii, tmin, tmax, t_eps=1e-6):
    # ro, rd: (3,), Pxyz:(M,3), Nn:(M,3), Radii:(M,)
    # compute t = dot(n, p - ro)/dot(n, rd)
    M = Pxyz.shape[0]
    out_t = np.inf
    out_j = -1

    # dot(n, rd)
    denom = Nn[:,0]*rd[0] + Nn[:,1]*rd[1] + Nn[:,2]*rd[2]
    # ignore nearly parallel
    valid = np.abs(denom) > 1e-8
    if not np.any(valid):
        return out_t, out_j

    num   = (Nn[:,0]*(Pxyz[:,0]-ro[0])
            +Nn[:,1]*(Pxyz[:,1]-ro[1])
            +Nn[:,2]*(Pxyz[:,2]-ro[2]))
    t     = np.where(valid, num/denom, np.inf)

    # t window
    mask = (t >= (tmin + t_eps)) & (t <= tmax)
    if not np.any(mask):
        return out_t, out_j

    hit_pts_x = ro[0] + t*rd[0]
    hit_pts_y = ro[1] + t*rd[1]
    hit_pts_z = ro[2] + t*rd[2]

    dx = hit_pts_x - Pxyz[:,0]
    dy = hit_pts_y - Pxyz[:,1]
    dz = hit_pts_z - Pxyz[:,2]
    radial2 = dx*dx + dy*dy + dz*dz

    mask &= (radial2 <= Radii*Radii)
    if not np.any(mask):
        return out_t, out_j

    # choose smallest positive t
    t_masked = np.where(mask, t, np.inf)
    j = np.argmin(t_masked)
    return t_masked[j], j

@njit(cache=True, fastmath=True)
def dda_traverse_fast(ro, rd, tmin, tmax, GRID_VOXEL_SIZE=0.05, gm_world=np.zeros(3, dtype=np.float32), inv_vox=20.0, nx=0, ny=0, nz=0, indptr=None, indices=None, P_data=None, N_data=None, R_data=None, G_data=None, eps=1e-6):
        """Optimized 3D-DDA with minimal allocations using sparse CSR grid."""
        t = float(max(tmin, 0.0))
        posx = ro[0] + t * rd[0]
        posy = ro[1] + t * rd[1]
        posz = ro[2] + t * rd[2]

        # Current cell
        cx = int(np.floor((posx - gm_world[0]) * inv_vox))
        cy = int(np.floor((posy - gm_world[1]) * inv_vox))
        cz = int(np.floor((posz - gm_world[2]) * inv_vox))

        # If outside grid, no cells to visit
        if cx < 0 or cx >= nx or cy < 0 or cy >= ny or cz < 0 or cz >= nz:
            return np.inf, INVALID_ID
        
        # Step direction signs
        sx = 1 if rd[0] > eps else (-1 if rd[0] < -eps else 0)
        sy = 1 if rd[1] > eps else (-1 if rd[1] < -eps else 0)
        sz = 1 if rd[2] > eps else (-1 if rd[2] < -eps else 0)
        
        # tDelta
        tDelta_x = GRID_VOXEL_SIZE / abs(rd[0]) if abs(rd[0]) > eps else np.inf
        tDelta_y = GRID_VOXEL_SIZE / abs(rd[1]) if abs(rd[1]) > eps else np.inf
        tDelta_z = GRID_VOXEL_SIZE / abs(rd[2]) if abs(rd[2]) > eps else np.inf
        
        # tMax
        if sx > 0:
            next_x = (cx + 1) * GRID_VOXEL_SIZE + gm_world[0]
            tMax_x = (next_x - posx) / rd[0] if abs(rd[0]) > eps else np.inf
        elif sx < 0:
            next_x = (cx * GRID_VOXEL_SIZE) + gm_world[0]
            tMax_x = (next_x - posx) / rd[0] if abs(rd[0]) > eps else np.inf
        else:
            tMax_x = np.inf
            
        if sy > 0:
            next_y = (cy + 1) * GRID_VOXEL_SIZE + gm_world[1]
            tMax_y = (next_y - posy) / rd[1] if abs(rd[1]) > eps else np.inf
        elif sy < 0:
            next_y = (cy * GRID_VOXEL_SIZE) + gm_world[1]
            tMax_y = (next_y - posy) / rd[1] if abs(rd[1]) > eps else np.inf
        else:
            tMax_y = np.inf
            
        if sz > 0:
            next_z = (cz + 1) * GRID_VOXEL_SIZE + gm_world[2]
            tMax_z = (next_z - posz) / rd[2] if abs(rd[2]) > eps else np.inf
        elif sz < 0:
            next_z = (cz * GRID_VOXEL_SIZE) + gm_world[2]
            tMax_z = (next_z - posz) / rd[2] if abs(rd[2]) > eps else np.inf
        else:
            tMax_z = np.inf
        
        t_closest = np.inf
        closest_gid = INVALID_ID
        
        while t < min(tmax, t_closest):
            if 0 <= cx < nx and 0 <= cy < ny and 0 <= cz < nz:
                voxel_idx = cx + nx * (cy + ny * cz)
                s, e = indptr[voxel_idx], indptr[voxel_idx + 1]
                if e > s:
                    cand = indices[s:e]  # Contiguous slice; zero overhead

                    t_hit, j = ray_disk_first_hit_vec(
                        ro, rd, P_data[cand], 
                        N_data[cand], R_data[cand], 
                        t, min(tmax, t_closest), 
                        t_eps=eps)
                    if np.isfinite(t_hit) and t_hit < t_closest:
                        t_closest = t_hit
                        if G_data is not None:
                            closest_gid = G_data[cand[j]]
            
            # Step to next face
            if tMax_x < tMax_y:
                if tMax_x < tMax_z:
                    if t + tMax_x >= min(tmax, t_closest):
                        break

                    t += tMax_x
                    tMax_y -= tMax_x
                    tMax_z -= tMax_x
                    tMax_x = tDelta_x
                    cx += sx
                else:
                    if t + tMax_z >= min(tmax, t_closest):
                        break

                    t += tMax_z
                    tMax_x -= tMax_z
                    tMax_y -= tMax_z
                    tMax_z = tDelta_z
                    cz += sz
            else:
                if tMax_y < tMax_z:
                    if t + tMax_y >= min(tmax, t_closest):
                        break

                    t += tMax_y
                    tMax_x -= tMax_y
                    tMax_z -= tMax_y
                    tMax_y = tDelta_y
                    cy += sy
                else:
                    if t + tMax_z >= min(tmax, t_closest):
                        break

                    t += tMax_z
                    tMax_x -= tMax_z
                    tMax_y -= tMax_z
                    tMax_z = tDelta_z
                    cz += sz

            # If we leave the grid, stop
            if cx < 0 or cx >= nx or cy < 0 or cy >= ny or cz < 0 or cz >= nz:
                break
        return t_closest, closest_gid

def raytrace_points_grouped(points,
                           voxel_center, voxel_size,
                           rays_FAR6, lambda_1,  # shape (F,A,R,6), float32
                           scene_name="scene"):
    """
    Ray traversal via 3D-DDA (Digital Differential Analyzer) over a voxelized point cloud.
    
    The point cloud is voxelized into 5cm voxels, then rays traverse the voxel grid
    using 3D-DDA to find the first hit in each voxel along the ray path.

    Inputs (same signature as before):
      - points: np.array of shape (N, 8): [x,y,z,class,nx,ny,nz,weight]
      - voxel_center, voxel_size: voxel region of interest
      - rays_FAR6: (F,A,R,6) rays with [O(3), D(3)]
      - scene_name: for debug filenames

    Returns:
      stats_comb, stats_leaf, stats_wood (same structure as before)
    """
    
    # --- Extract point data ---
    if isinstance(points, dict):
        P = np.asarray(points["xyz"], dtype=np.float32)
        G = np.asarray(points.get("class", None), dtype=np.uint32) if "class" in points else None
        N = np.asarray(points.get("normals", None), dtype=np.float32) if "normals" in points else None
        W = np.asarray(points.get("weights", None), dtype=np.float32) if "weights" in points else None
    else:
        arr = np.asarray(points)
        if arr.ndim != 2 or arr.shape[1] < 7:
            raise ValueError("points must be dict with xyz/normals/weights or an (N,7+) array.")
        P = arr[:, 0:3].astype(np.float32)
        G = arr[:, 3].astype(np.uint32)
        N = arr[:, 4:7].astype(np.float32)
        W = arr[:, 7].astype(np.float32)

    if N is None:
        raise ValueError("Normals are required for disk (surfel) intersections.")

    N = _normalize(N)
    if W is None:
        W = np.ones((P.shape[0],), dtype=np.float32)
    R = _weight_to_radius(W, voxel_size, r_min_factor=0.25, r_max_factor=0.75, invert=True)

    # C-Contiguous views for hot loop
    P_data = np.ascontiguousarray(P, dtype=np.float32)
    N_data = np.ascontiguousarray(N, dtype=np.float32)
    R_data = np.ascontiguousarray(R, dtype=np.float32)
    G_data = np.ascontiguousarray(G, dtype=np.uint32) if G is not None else None

    # --- Build CSR voxel grid from point cloud ---
    p_min = P.min(axis=0)
    p_max = P.max(axis=0)

    grid_min = np.floor(p_min / GRID_VOXEL_SIZE).astype(np.int32)
    grid_max = np.ceil(p_max / GRID_VOXEL_SIZE).astype(np.int32)
    grid_shape = (grid_max - grid_min + 1).astype(np.int32)

    nx,ny,nz = int(grid_shape[0]), int(grid_shape[1]), int(grid_shape[2])
    nv = nx * ny * nz
    if nv <= 0:
        # No grid (empty); return empty stats
        F, A, Rr, _ = rays_FAR6.shape
        empty = [[dict(N=0, n_hits=0, I=0.0,
                       mean_path_length=0.0, sum_path_length=0.0,
                       mean_free_path_length=0.0, sum_free_path_length=0.0,
                       mean_eff_free_path_length=0.0, var_eff_free_path_length=0.0,
                       sum_eff_free_path_length=0.0, sum_hits_eff_free_path_length=0.0)
                       for _ in range(A)] for _ in range(F)]
        return empty, empty, empty

    # Linear cell id per point
    ix = np.floor(P_data[:,0] / GRID_VOXEL_SIZE).astype(np.int32) - grid_min[0]
    iy = np.floor(P_data[:,1] / GRID_VOXEL_SIZE).astype(np.int32) - grid_min[1]
    iz = np.floor(P_data[:,2] / GRID_VOXEL_SIZE).astype(np.int32) - grid_min[2]
    ix = np.clip(ix, 0, nx - 1)
    iy = np.clip(iy, 0, ny - 1)
    iz = np.clip(iz, 0, nz - 1)

    cell_id = ix + nx * (iy + ny * iz)  # (N,)
    order = np.argsort(cell_id, kind="mergesort")
    cell_sorted = cell_id[order]
    counts = np.bincount(cell_sorted, minlength=nv)

    indptr = np.zeros(nv + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    indices = order.astype(np.int32, copy=False)

    # Grid world origin (min corner of cell [0,0,0])
    gm_world = grid_min.astype(np.float32) * GRID_VOXEL_SIZE
    inv_vox = np.float32(1.0 / GRID_VOXEL_SIZE)
    eps = np.float32(1e-6)

    # --- Unpack rays & voxel slab intersection ---
    F, A, Rr, _ = rays_FAR6.shape
    vc = np.asarray(voxel_center, dtype=np.float32)
    bmin = vc - 0.5 * float(voxel_size)
    bmax = vc + 0.5 * float(voxel_size)

    O = rays_FAR6[..., 0:3]  # (F,A,R,3)
    D = rays_FAR6[..., 3:6]  # (F,A,R,3)
    Dn = _normalize(D)

    t_near, t_far = ray_box_intersection_vectorized(
        O.reshape(-1, 3).astype(np.float32, copy=False), 
        Dn.reshape(-1, 3).astype(np.float32, copy=False), 
        bmin.astype(np.float32, copy=False), bmax.astype(np.float32, copy=False)
    )
    t_near = t_near.reshape(F, A, Rr)
    t_far = t_far.reshape(F, A, Rr)
    valid_rays_mask = (t_near <= t_far) & (t_far >= 0.0)

    # --- Output buffers ---
    dists = np.full((F, A, Rr), np.inf, dtype=np.float32)
    gids = np.full((F, A, Rr), INVALID_ID, dtype=np.uint32)

    # Flatten once for batch processing
    rays_flat = Dn.reshape(-1, 3).astype(np.float32, copy=False)
    origins_flat = O.reshape(-1, 3).astype(np.float32, copy=False)
    tnear_flat = t_near.reshape(-1).astype(np.float32, copy=False)
    tfar_flat = t_far.reshape(-1).astype(np.float32, copy=False)
    valid_flat = valid_rays_mask.reshape(-1)
    valid_indices = np.nonzero(valid_flat)[0]
    
    # Process valid rays
    for idx_in_batch in valid_indices:
        ro = origins_flat[idx_in_batch]
        rd = rays_flat[idx_in_batch]
        tmin = tnear_flat[idx_in_batch]
        tmax = tfar_flat[idx_in_batch]

        t_hit, gid = dda_traverse_fast(
            ro=ro, rd=rd, tmin=tmin, tmax=tmax,
            GRID_VOXEL_SIZE=GRID_VOXEL_SIZE,
            gm_world=gm_world,
            inv_vox=inv_vox,
            nx=nx, ny=ny, nz=nz,
            indptr=indptr, indices=indices,
            P_data=P_data, N_data=N_data, R_data=R_data, G_data=G_data)

        if np.isfinite(t_hit):
            f = idx_in_batch // (A * Rr)
            a = (idx_in_batch % (A * Rr)) // Rr
            r = idx_in_batch % Rr
            dists[f, a, r] = t_hit
            gids[f, a, r] = gid

    # --- Debug output (same as before) ---
    if DEBUG_MODE:
        hit_mask = valid_rays_mask & np.isfinite(dists)
        if np.any(hit_mask):
            f_idx, a_idx, r_idx = np.nonzero(hit_mask)
            O_hits = O[f_idx, a_idx, r_idx]
            D_hits = Dn[f_idx, a_idx, r_idx]
            d_hits = dists[f_idx, a_idx, r_idx]
            pts = O_hits + D_hits * d_hits[:, None]

            gids_hits = gids[f_idx, a_idx, r_idx]
            leaf_hit = np.isin(gids_hits, _LEAF_KEYS) if (G is not None and _LEAF_KEYS is not None) else np.zeros_like(gids_hits, dtype=bool)
            wood_hit = np.isin(gids_hits, _WOOD_KEYS) if (G is not None and _WOOD_KEYS is not None) else np.zeros_like(gids_hits, dtype=bool)

            classes = np.zeros(pts.shape[0], dtype=np.int8)
            classes[leaf_hit] = 1
            classes[wood_hit] = 0

            data = np.column_stack((pts, f_idx, a_idx, classes))

            debug_dir = os.path.join(DEBUG_PATH, f"voxel_size={voxel_size}",
                                     f"voxel_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}")
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"{scene_name}_hits_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.xyz")

            header_text = "x y z face_idx angle_idx class\n"
            df = pd.DataFrame(data, columns=["x", "y", "z", "face_idx", "angle_idx", "class"])
            chunk_size = 200_000
            with open(out_path, 'w', buffering=4*1024*1024) as f:
                f.write(header_text)
                for start in range(0, df.shape[0], chunk_size):
                    end = start + chunk_size
                    df.iloc[start:end].to_csv(f, sep=' ', header=False, index=False, float_format='%.6f')

    # --- Stats computation (unchanged) ---
    valid_hits_mask = np.isfinite(dists) & (dists >= t_near) & (dists <= t_far) & (dists >= 0) & valid_rays_mask

    dist_leaf = np.full_like(dists, np.inf, dtype=np.float32)
    dist_wood = np.full_like(dists, np.inf, dtype=np.float32)

    if (G is not None) and (_LEAF_KEYS is not None):
        m_leaf = np.isin(gids, _LEAF_KEYS) & valid_hits_mask
        dist_leaf[m_leaf] = dists[m_leaf]
        _is_leaf = True
    else:
        _is_leaf = False

    if (G is not None) and (_WOOD_KEYS is not None):
        m_wood = np.isin(gids, _WOOD_KEYS) & valid_hits_mask
        dist_wood[m_wood] = dists[m_wood]
        _is_wood = True
    else:
        _is_wood = False

    # Combined: take minimum distance (first hit) from either leaf or wood
    # Use dists directly for combined to capture all valid hits, not just leaf or wood
    comb_dist = dists.copy()
    comb_dist[~valid_hits_mask] = np.inf

    def mean_var_ddof1(x, N):
        mx = x.sum(axis=2)
        mean = np.divide(mx, N, out=np.zeros_like(mx), where=(N > 0))
        mx2 = (x**2).sum(axis=2)
        numer = mx2 - (mx * mx) / np.maximum(N, 1)
        denom = np.maximum(N - 1, 1)
        var = numer / denom
        var[(N <= 1)] = 0.0
        return mean, var

    N_rays = valid_rays_mask.sum(axis=2).astype(np.int32)
    
    # Hit counts for each component
    hit_leaf = np.isfinite(dist_leaf)
    hit_wood = np.isfinite(dist_wood)
    hit_any = np.isfinite(comb_dist)  # Hit either leaf or wood
    
    n_hits_leaf = hit_leaf.sum(axis=2).astype(np.int32)
    n_hits_wood = hit_wood.sum(axis=2).astype(np.int32)
    n_hits_comb = hit_any.sum(axis=2).astype(np.int32)

    path_lengths = np.zeros_like(dists, dtype=np.float32)
    path_lengths[valid_rays_mask] = (t_far[valid_rays_mask] - t_near[valid_rays_mask])
    mean_path_length = np.divide(path_lengths.sum(axis=2), N_rays, out=np.zeros_like(path_lengths.sum(axis=2)), where=(N_rays > 0))
    sum_path_length = path_lengths.sum(axis=2)

    # Helper function to compute stats for a distance array
    def compute_component_stats(dist_component, hit_component, name):
        free_path_length = path_lengths.copy()
        free_path_length[hit_component] = dist_component[hit_component] - t_near[hit_component]
        
        eff_free_path_length = np.zeros_like(free_path_length, dtype=np.float32)
        eff_free_path_length[valid_rays_mask] = compute_efpl_array(free_path_length[valid_rays_mask], lambda_1).astype(np.float32)
        
        n_hits = hit_component.sum(axis=2).astype(np.int32)
        sum_free_path_length = free_path_length.sum(axis=2)
        mean_free_path_length = np.divide(sum_free_path_length, N_rays, out=np.zeros_like(sum_free_path_length), where=(N_rays > 0))
        mean_eff_free_path_length, var_eff_free_path_length = mean_var_ddof1(eff_free_path_length, N_rays)
        sum_eff_free_path_length = np.sum(eff_free_path_length, axis=2)
        sum_hits_eff_free_path_length = (eff_free_path_length * hit_component).sum(axis=2)
        
        return {
            'n_hits': n_hits,
            'sum_free_path_length': sum_free_path_length,
            'mean_free_path_length': mean_free_path_length,
            'mean_eff_free_path_length': mean_eff_free_path_length,
            'var_eff_free_path_length': var_eff_free_path_length,
            'sum_eff_free_path_length': sum_eff_free_path_length,
            'sum_hits_eff_free_path_length': sum_hits_eff_free_path_length,
        }

    # Compute stats for leaf, wood, and combined
    leaf_stats = compute_component_stats(dist_leaf, hit_leaf, "leaf") if _is_leaf else None
    wood_stats = compute_component_stats(dist_wood, hit_wood, "wood") if _is_wood else None
    comb_stats = compute_component_stats(comb_dist, hit_any, "combined")

    # Build stats dictionaries
    def build_stats_dict(f, a, n_hits, sum_free, mean_free, mean_eff_free, var_eff_free, sum_eff_free, sum_hits_eff_free):
        return dict(
            N=int(N_rays[f, a]),
            n_hits=int(n_hits[f, a]),
            I=(float(n_hits[f, a]) / float(N_rays[f, a])) if N_rays[f, a] else 0.0,
            mean_path_length=float(mean_path_length[f, a]),
            sum_path_length=float(sum_path_length[f, a]),
            sum_free_path_length=float(sum_free[f, a]),
            mean_free_path_length=float(mean_free[f, a]),
            mean_eff_free_path_length=float(mean_eff_free[f, a]),
            var_eff_free_path_length=float(var_eff_free[f, a]),
            sum_eff_free_path_length=float(sum_eff_free[f, a]),
            sum_hits_eff_free_path_length=float(sum_hits_eff_free[f, a]),
        )

    stats_comb = [[None for _ in range(A)] for _ in range(F)]
    for f in range(F):
        for a in range(A):
            stats_comb[f][a] = build_stats_dict(
                f, a,
                comb_stats['n_hits'],
                comb_stats['sum_free_path_length'],
                comb_stats['mean_free_path_length'],
                comb_stats['mean_eff_free_path_length'],
                comb_stats['var_eff_free_path_length'],
                comb_stats['sum_eff_free_path_length'],
                comb_stats['sum_hits_eff_free_path_length'],
            )

    stats_leaf = [[None for _ in range(A)] for _ in range(F)]
    for f in range(F):
        for a in range(A):
            if leaf_stats is not None and _is_leaf:
                stats_leaf[f][a] = build_stats_dict(
                    f, a,
                    leaf_stats['n_hits'],
                    leaf_stats['sum_free_path_length'],
                    leaf_stats['mean_free_path_length'],
                    leaf_stats['mean_eff_free_path_length'],
                    leaf_stats['var_eff_free_path_length'],
                    leaf_stats['sum_eff_free_path_length'],
                    leaf_stats['sum_hits_eff_free_path_length'],
                )
            else:
                stats_leaf[f][a] = dict(
                    N=0, n_hits=0, I=0.0,
                    mean_path_length=0.0, sum_path_length=0.0,
                    sum_free_path_length=0.0, mean_free_path_length=0.0,
                    mean_eff_free_path_length=0.0, var_eff_free_path_length=0.0,
                    sum_eff_free_path_length=0.0, sum_hits_eff_free_path_length=0.0,
                )

    stats_wood = [[None for _ in range(A)] for _ in range(F)]
    for f in range(F):
        for a in range(A):
            if wood_stats is not None and _is_wood:
                stats_wood[f][a] = build_stats_dict(
                    f, a,
                    wood_stats['n_hits'],
                    wood_stats['sum_free_path_length'],
                    wood_stats['mean_free_path_length'],
                    wood_stats['mean_eff_free_path_length'],
                    wood_stats['var_eff_free_path_length'],
                    wood_stats['sum_eff_free_path_length'],
                    wood_stats['sum_hits_eff_free_path_length'],
                )
            else:
                stats_wood[f][a] = dict(
                    N=0, n_hits=0, I=0.0,
                    mean_path_length=0.0, sum_path_length=0.0,
                    sum_free_path_length=0.0, mean_free_path_length=0.0,
                    mean_eff_free_path_length=0.0, var_eff_free_path_length=0.0,
                    sum_eff_free_path_length=0.0, sum_hits_eff_free_path_length=0.0,
                )

    return stats_comb, stats_leaf, stats_wood

def compute_G_function_binwise(viewing_angles, leaf_bin_centers, LIAD_values):
    """
    Compute G(angle) from the leaf angle distribution (LIAD), binwise integration.
    """
    total_lad = LIAD_values.sum()
    lad_norm = LIAD_values / total_lad if total_lad > 0 else LIAD_values.copy()
    viewing_angles = np.clip(viewing_angles, 0.0001, 89.9999)
    leaf_bin_centers = np.clip(leaf_bin_centers, 0.0001, 89.9999)
    theta_rad = np.radians(viewing_angles)
    theta_lrad = np.radians(leaf_bin_centers)
    cos_theta = np.cos(theta_rad)
    cot_theta = 1 / np.tan(theta_rad)
    cos_tl = np.cos(theta_lrad)
    cot_tl = 1 / np.tan(theta_lrad)
    cot_mesh = np.outer(cot_theta, cot_tl)
    cos_mesh = np.outer(cos_theta, cos_tl)
    G_mat = np.zeros_like(cot_mesh)
    mask_gt1 = (np.abs(cot_mesh) > 1)
    mask_le1 = ~mask_gt1
    G_mat[mask_gt1] = cos_mesh[mask_gt1]
    inside = np.clip(cot_mesh[mask_le1], -1, 1)
    psi = np.arccos(inside)
    factor = 1.0 + (2.0 / np.pi) * (np.tan(psi) - psi)
    G_mat[mask_le1] = cos_mesh[mask_le1] * factor
    G_values = G_mat @ lad_norm
    return G_values

def generate_voxel_centers(voxel_size, bounds):
    """
    Placeholder function to generate voxel centers.
    Replace this with actual logic from your 040.py script.
    """
    # Generate voxel grid for the combined plot bounds of leaf_mesh and wood_mesh (trimesh version)
    minx, miny, minz, maxx, maxy, maxz = bounds
    x_centers = np.arange(minx + voxel_size / 2, maxx + voxel_size / 2, voxel_size)
    y_centers = np.arange(miny + voxel_size / 2, maxy + voxel_size / 2, voxel_size)
    z_centers = np.arange(minz + voxel_size / 2, maxz + voxel_size / 2, voxel_size)
    voxel_centers = np.array(np.meshgrid(x_centers, y_centers, z_centers, indexing='ij')).reshape(3, -1).T
    coords = ((voxel_centers * 11 + voxel_size * 73)*13).astype(int)
    voxel_ids = np.array([create_voxel_id(voxel_size, vc_x, vc_y, vc_z) for (vc_x, vc_y, vc_z) in voxel_centers])
    
    return voxel_centers, voxel_ids

def filter_voxel_centers(voxel_centers, voxel_size):
    """
    Filter voxel centers based on the bounds of leaf and wood meshes.
    """
    min_leaf = _CLIP_GLOBALS['leaf_min']
    max_leaf = _CLIP_GLOBALS['leaf_max']
    min_wood = _CLIP_GLOBALS['wood_min']
    max_wood = _CLIP_GLOBALS['wood_max']

    # Create masks for leaf and wood bounds
    leaf_mask = np.all((voxel_centers >= (min_leaf - voxel_size / 2)) & 
                       (voxel_centers <= (max_leaf + voxel_size / 2)), axis=1)
    wood_mask = np.all((voxel_centers >= (min_wood - voxel_size / 2)) & 
                       (voxel_centers <= (max_wood + voxel_size / 2)), axis=1)

    # Combine masks
    combined_mask = leaf_mask | wood_mask

    return voxel_centers[combined_mask]


# Precompute rotation matrices once (outside voxel loop)
def rotation_mx_x(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def rotation_mx_y(deg):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

# Precompute canonical face grids once (origins & directions relative to (0,0,0))
# Then translate to voxel_center each time; no trig inside inner loops
_GRIDS = {}
def _build_canonical_grids(voxel_sizes, ray_spacing):
    for vs in voxel_sizes:
        grid = _grid(voxel_size=vs, ray_spacing=ray_spacing)
        _GRIDS[vs] = {
            "bottom": generate_face_rays_bottom(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "top":    generate_face_rays_top(vc=np.array([0,0,0]), vs=vs, grid=grid, offset=ray_spacing/2),
            "xplus":  generate_side_rays_xplus(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "xminus": generate_side_rays_xminus(vc=np.array([0,0,0]), vs=vs, grid=grid, offset=ray_spacing/2),
            "yplus":  generate_side_rays_yplus(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "yminus": generate_side_rays_yminus(vc=np.array([0,0,0]), vs=vs, grid=grid, offset=ray_spacing/2),
        }

global FACE_ORDER
FACE_ORDER = ["bottom","top","xplus","xminus","yplus","yminus"]

global ANGLE_ORDER

# faces that rotate around y vs x (matches your current logic)
rot_axis = {"xplus":"y","xminus":"y","bottom":"x","top":"x","yplus":"x","yminus":"x"}

def build_ray_tensor_for_voxel(voxel_center, voxel_size, angles):
    rays_list = []
    labels = []  # (face_idx, angle_idx) per ray
    face_order = FACE_ORDER
    angles_sorted = sorted(angles)

    F = len(face_order)
    A = len(angles_sorted)
    R = _GRIDS[voxel_size][face_order[0]][0].shape[0]  # number of rays per face

    rays = np.empty((F, A, R, 6), dtype=np.float32)  # preallocate
    vc = voxel_center.astype(np.float32)

    # Precompute rotation matrices once per angle
    rot_x = {deg: rotation_mx_x(deg) for deg in angles_sorted}
    rot_y = {deg: rotation_mx_y(deg) for deg in angles_sorted}

    for f_idx, face in enumerate(face_order):
        O_base, D_base = _GRIDS[voxel_size][face]         # relative to origin
        O_trans = O_base + vc                           # translate to voxel_center (diagonal offset built into original grid)
        for a_idx, deg in enumerate(angles_sorted):
            Rmx = rot_y[deg] if rot_axis[face] == "y" else rot_x[deg]
            # rotate directions about center; origins rotate around center too if you need orientation shift
            D_rot = (D_base @ Rmx.T).astype(np.float32)
            # (optional) rotate O around center: (O - center) @ Rmx.T + center
            O_rot = ((O_trans - vc) @ Rmx.T + vc).astype(np.float32)

            # Write rotated rays into preallocated array
            rays[f_idx, a_idx, :, 0:3] = O_rot
            rays[f_idx, a_idx, :, 3:6] = D_rot

    return rays, face_order, angles_sorted

@njit(parallel=True, fastmath=True)
def group_counts(label_ids, t_hit, n_faces, n_angles):
    # returns per-group stats; you can extend with sums/means of z, delta_e, etc.
    counts = np.zeros((n_faces, n_angles), np.int32)
    hits_n = np.zeros((n_faces, n_angles), np.int32)
    for i in prange(label_ids.shape[0]):
        lab = label_ids[i]
        f_idx = (lab >> 16) & 0xFFFF
        a_idx = lab & 0xFFFF
        counts[f_idx, a_idx] += 1
        if np.isfinite(t_hit[i]) and t_hit[i] > 0.0:
            hits_n[f_idx, a_idx] += 1
    return counts, hits_n
    
def process_voxel(
    voxel_center,
    leaf_points,
    wood_points,
    scenes,
    voxel_size, 
    angles,
    lambda_1
    ):
    """
    Process a single voxel center with the given parameters.
    This function should contain the logic to process the voxel.
    """
    MIN_HITS = 10

    if leaf_points is None and "leaf" in scenes:
        print(f"[DEBUG] {voxel_center}: No leaf points for leaf-only scene. Skipping.")
        return [], []
    
    if wood_points is None and "wood" in scenes:
        print(f"[DEBUG] {voxel_center}: No wood points for wood-only scene. Skipping.")
        return [], []

    # Compute LIAD bins for voxel
    try:
        leaf_normals = leaf_points[:, 3:6] if leaf_points is not None else np.array([])
        leaf_weights = leaf_points[:, 6] if leaf_points is not None else np.array([])
        wood_normals = wood_points[:, 3:6] if wood_points is not None else np.array([])
        wood_weights = wood_points[:, 6] if wood_points is not None else np.array([])
        
        bin_leaf, liad, _ = calculate_inclination_angle_distribution(leaf_normals, leaf_weights) if leaf_points is not None and leaf_normals.size > 0 else (np.array([]), np.array([]), None)
        bin_wood, wiad, _ = calculate_inclination_angle_distribution(wood_normals, wood_weights) if wood_points is not None and wood_normals.size > 0 else (np.array([]), np.array([]), None)
        
        comb_normals = np.vstack([leaf_normals, wood_normals]) if (leaf_normals.size > 0 and wood_normals.size > 0) else (leaf_normals if leaf_normals.size > 0 else wood_normals)
        comb_weights = np.hstack([leaf_weights, wood_weights]) if (leaf_weights.size > 0 and wood_weights.size > 0) else (leaf_weights if leaf_weights.size > 0 else wood_weights)
        bin_comb, piad, _ = calculate_inclination_angle_distribution(comb_normals, comb_weights) if comb_normals.size > 0 else (np.array([]), np.array([]), None)

        # Store LIAD/WIAD/PIAD bins and values
        liad_dict = {f"LIAD_bin_{c:.1f}": float(v)
                for c, v in zip(bin_leaf, liad)}
        wiad_dict = {f"WIAD_bin_{c:.1f}": float(v)
                for c, v in zip(bin_wood, wiad)}
        piad_dict = {f"PIAD_bin_{c:.1f}": float(v)
                for c, v in zip(bin_comb, piad)}
        
    except Exception as e:
        raise RuntimeError(
            f"Error computing LIAD for voxel at {voxel_center}: {e}"
        ) from e

    # Build ray tensor for all rays in voxel
    rays_FAR6, face_order, angles_sorted = build_ray_tensor_for_voxel(
    voxel_center=voxel_center, 
    voxel_size=voxel_size, 
    angles=angles)
    
    ### For each scene in scenes, create a scene and raycast for outputs
    all_results = []

    try:
        for scene in scenes:
            if scene == "leaf":
                if leaf_points is None or leaf_points.size == 0:
                    print(f"[DEBUG] {voxel_center}: No leaf points for leaf-only scene. Skipping.")
                    all_results.append({})
                    continue
                points = leaf_points

            elif scene == "wood":
                if wood_points is None or wood_points.size == 0:
                    print(f"[DEBUG] {voxel_center}: No wood points for wood-only scene. Skipping.")
                    all_results.append({})
                    continue
                points = wood_points

            elif scene == "combined":
                if (leaf_points is None or leaf_points.size == 0) and (wood_points is None or wood_points.size == 0):
                    print(f"[DEBUG] {voxel_center}: No points for combined scene. Skipping.")
                    all_results.append({})
                    continue
                points = np.vstack([leaf_points, wood_points]) if (leaf_points is not None and wood_points is not None) else (leaf_points if leaf_points is not None else wood_points)
            else:
                raise ValueError(f"Unknown scene type: {scene}")

            stats_comb_grouped, stats_leaf_grouped, stats_wood_grouped = raytrace_points_grouped(
                points, voxel_center, voxel_size, rays_FAR6, lambda_1=lambda_1, scene_name=scene
            )

            try:
                for f_idx, face_lbl in enumerate(face_order):
                    for a_idx, angle in enumerate(angles_sorted):
                        dx = rays_FAR6[f_idx, a_idx, 0, 3]
                        dy = rays_FAR6[f_idx, a_idx, 0, 4]
                        dz = rays_FAR6[f_idx, a_idx, 0, 5]

                        comb_data = stats_comb_grouped[f_idx][a_idx]
                        leaf_data = stats_leaf_grouped[f_idx][a_idx]
                        wood_data = stats_wood_grouped[f_idx][a_idx]

                        if comb_data["N"] == 0:
                            comb_data = {k: np.nan for k in comb_data}
                        if leaf_data["N"] == 0:
                            leaf_data = {k: np.nan for k in leaf_data}
                        if wood_data["N"] == 0:
                            wood_data = {k: np.nan for k in wood_data}

                        ### G ###
                        # Convert dz to zenith angle in degrees
                        # Normalize direction vector and extract zenith angle
                        dir_norm = math.sqrt(dx**2 + dy**2 + dz**2)
                        zenith_angle = math.degrees(math.acos(abs(dz) / dir_norm)) if dir_norm > 0 else 0.0

                        G_leaf_est = np.nan
                        if bin_leaf.size > 0:
                            G_leaf_est = compute_G_function_binwise([zenith_angle], bin_leaf, liad)[0]

                        G_wood_est = np.nan
                        if bin_wood.size > 0:
                            G_wood_est = compute_G_function_binwise([zenith_angle], bin_wood, wiad)[0]

                        G_comb_est = np.nan
                        if bin_comb.size > 0:
                            G_comb_est = compute_G_function_binwise([zenith_angle], bin_comb, piad)[0]

                        ### pgap ###
                        pgap_leaf = 1.0 - leaf_data["I"]
                        pgap_wood = 1.0 - wood_data["I"]
                        pgap_comb = 1.0 - comb_data["I"]

                        ### Compute LAD using pgap, mean_path_length, and G estimate
                        LAD_est = -math.log(pgap_leaf) / (G_leaf_est * leaf_data["mean_path_length"]) if (0 < pgap_leaf < 1 and leaf_data["mean_path_length"] is not None and leaf_data["mean_path_length"] > 0 and G_leaf_est > 0) else np.nan
                        WAD_est = -math.log(pgap_wood) / (G_wood_est * wood_data["mean_path_length"]) if (0 < pgap_wood < 1 and wood_data["mean_path_length"] is not None and wood_data["mean_path_length"] > 0 and G_wood_est > 0) else np.nan
                        PAD_est = -math.log(pgap_comb) / (G_comb_est * comb_data["mean_path_length"]) if (0 < pgap_comb < 1 and comb_data["mean_path_length"] is not None and comb_data["mean_path_length"] > 0 and G_comb_est > 0) else np.nan
                        
                        ### data prep ###
                        leaf_fraction = (leaf_data["n_hits"] / comb_data["n_hits"]
                                if comb_data["n_hits"] else np.nan)
                        wood_fraction = (wood_data["n_hits"] / comb_data["n_hits"]
                                if comb_data["n_hits"] else np.nan)
                        
                        row = {
                            "voxel_cx": float(voxel_center[0]), "voxel_cy": float(voxel_center[1]), "voxel_cz": float(voxel_center[2]),
                            "face": face_lbl, "zenith_angle": zenith_angle,
                            "dx": dx, "dy": dy, "dz": dz,
                            # reference densities
                            "LAD_est": float(LAD_est) if LAD_est is not None else np.nan,
                            "WAD_est": float(WAD_est) if WAD_est is not None else np.nan,
                            "PAD_est": float(PAD_est) if PAD_est is not None else np.nan,
                            # ray and hit values
                            "total_num_rays": int(comb_data["N"]) if comb_data["N"] is not None else np.nan,
                            "total_num_hits": int(comb_data["n_hits"]) if comb_data["n_hits"] is not None else np.nan,
                            "total_missed_rays": int(comb_data["N"] - comb_data["n_hits"]) if comb_data["N"] is not None and comb_data["n_hits"] is not None else np.nan,
                            # hits per component
                            "n_hits_leaf": int(leaf_data["n_hits"]) if (leaf_data["n_hits"] is not None and not np.isnan(leaf_data["n_hits"])) else np.nan,
                            "n_hits_wood": int(wood_data["n_hits"]) if (wood_data["n_hits"] is not None and not np.isnan(wood_data["n_hits"])) else np.nan,
                            "n_hits_comb": int(comb_data["n_hits"]) if (comb_data["n_hits"] is not None and not np.isnan(comb_data["n_hits"])) else np.nan,
                            # observed I
                            "I_leaf": float(leaf_data["I"]) if leaf_data["I"] is not None else np.nan,
                            "I_wood": float(wood_data["I"]) if wood_data["I"] is not None else np.nan,
                            "I_comb": float(comb_data["I"]) if comb_data["I"] is not None else np.nan,
                            # observed pgap
                            "pgap_leaf": float(pgap_leaf) if not np.isnan(pgap_leaf) else np.nan,
                            "pgap_wood": float(pgap_wood) if not np.isnan(pgap_wood) else np.nan,
                            "pgap_comb": float(pgap_comb) if not np.isnan(pgap_comb) else np.nan,
                            # path length
                            "mean_path_length_leaf": float(leaf_data["mean_path_length"]) if leaf_data["mean_path_length"] is not None else np.nan,
                            "mean_path_length_wood": float(wood_data["mean_path_length"]) if wood_data["mean_path_length"] is not None else np.nan,
                            "mean_path_length_comb": float(comb_data["mean_path_length"]) if comb_data["mean_path_length"] is not None else np.nan,
                            # free path length
                            "mean_free_path_length_leaf": float(leaf_data["mean_free_path_length"]) if leaf_data["mean_free_path_length"] is not None else np.nan,
                            "mean_free_path_length_wood": float(wood_data["mean_free_path_length"]) if wood_data["mean_free_path_length"] is not None else np.nan,
                            "mean_free_path_length_comb": float(comb_data["mean_free_path_length"]) if comb_data["mean_free_path_length"] is not None else np.nan,
                            # effective free path length
                            "mean_eff_free_path_length_leaf": float(leaf_data["mean_eff_free_path_length"]) if leaf_data["mean_eff_free_path_length"] is not None else np.nan,
                            "mean_eff_free_path_length_wood": float(wood_data["mean_eff_free_path_length"]) if wood_data["mean_eff_free_path_length"] is not None else np.nan,
                            "mean_eff_free_path_length_comb": float(comb_data["mean_eff_free_path_length"]) if comb_data["mean_eff_free_path_length"] is not None else np.nan,
                            # per-angle G
                            "G_leaf_computed": float(G_leaf_est) if not np.isnan(G_leaf_est) else np.nan,
                            "G_wood_computed":  float(G_wood_est) if not np.isnan(G_wood_est) else np.nan,
                            "G_comb_computed":  float(G_comb_est) if not np.isnan(G_comb_est) else np.nan,
                            "leaf_fraction": float(leaf_fraction) if not np.isnan(leaf_fraction) else np.nan                            
                        }

                        row.update(liad_dict)
                        row.update(wiad_dict)
                        row.update(piad_dict)
                        row['scene'] = scene

                        all_results.append(row)

            except Exception as e:
                raise RuntimeError(
                    f"Error processing grouped stats for voxel at {voxel_center}: {e}"
                ) from e
    
    except Exception as e:
        raise RuntimeError(
            f"Error processing voxel at {voxel_center} with size {voxel_size}: {e}"
        ) from e

    # print(f"[DEBUG] Processed voxel at {voxel_center} with size {voxel_size}, generated {len(voxel_rows)} rows.")
    return all_results


# ---- Global cache for threading-based clipping ----
_CLIP_GLOBALS = {
    'leaf_points': None,
    'wood_points': None,
    'leaf_min': None,
    'leaf_max': None,
    'wood_min': None,
    'wood_max': None
}

def set_clip_globals(leaf_points, wood_points):
    """
    Populate module-level globals once in the parent process.
    Threads will read from these; no pickling or joblib.Memory in workers.
    """
    global _CLIP_GLOBALS
    _CLIP_GLOBALS['leaf_points'] = leaf_points
    _CLIP_GLOBALS['wood_points'] = wood_points

    # Precompute triangle AABB mins/maxs once
    if leaf_points is not None and not leaf_points.size == 0:
        _CLIP_GLOBALS['leaf_min'] = leaf_points[:, 0:3].min(axis=0)
        _CLIP_GLOBALS['leaf_max'] = leaf_points[:, 0:3].max(axis=0)
    else:
        _CLIP_GLOBALS['leaf_min'] = None
        _CLIP_GLOBALS['leaf_max'] = None

    if wood_points is not None and not wood_points.size == 0:
        _CLIP_GLOBALS['wood_min'] = wood_points[:, 0:3].min(axis=0)
        _CLIP_GLOBALS['wood_max'] = wood_points[:, 0:3].max(axis=0)
    else:
        _CLIP_GLOBALS['wood_min'] = None
        _CLIP_GLOBALS['wood_max'] = None

def clip_one_thread(voxel_center, voxel_size):
    """
    Thread worker: clip both leaf and wood using the shared globals.
    Returns (center, leaf_vertices, leaf_faces, wood_vertices, wood_faces).
    """
    g = _CLIP_GLOBALS
    leaf_points = _clip_one_pointcloud_with_aabb(
        g['leaf_points'],
        voxel_center, voxel_size
    )
    wood_points = _clip_one_pointcloud_with_aabb(
        g['wood_points'],
        voxel_center, voxel_size
    )
    return (voxel_center, leaf_points, wood_points)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process voxel batch.")
    parser.add_argument("scene_file", type=str, help="Path to the single helios points file. Use the class_col and leaf and wood classes arguments to set the correct classes for leaf and wood.")
    parser.add_argument("--class_col", type=int, default=9, help="Column index in the points file that contains class labels (default: 9). set to 8 for hitObjectId.")
    parser.add_argument("--leaf_keys", type=int, nargs='+', default=[1], help="List of keywords to identify leaf groups in the OBJ file (default: ['leaf']).")
    parser.add_argument("--wood_keys", type=int, nargs='+', default=[0], help="List of keywords to identify wood groups in the OBJ file (default: ['wood']).")
    parser.add_argument("--scene_formats", type=str, nargs='+', default=["combined"], help="Scene formats to process (default: ['combined']), use 'leaf' for leaf-only and 'wood' for wood-only.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0], help="Voxel sizes for processing (default: [0.2, 0.5, 1.0, 2.0]).")
    parser.add_argument("--num_angle_bins", type=int, default=18, help="Number of angle bins for ray tracing (default: 18).")
    parser.add_argument("--ray_spacing", type=float, default=0.005, help="Ray spacing for ray tracing (default: 0.005).")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel workers (default: 32) will max to num_cpus.")
    parser.add_argument("--normals_voxel_size", type=float, default=10, help="Voxel size for normal estimation (default: 10). Reduce for memory overload conditions.")
    parser.add_argument("--existing_csv", type=str, default=None, help="Path to existing CSV to extraxt voxel centers from instead of generating a new grid. Should have columns 'voxel_cx', 'voxel_cy', 'voxel_cz'. Only one file per process, so make sure you select the correct voxel size.")
    parser.add_argument("--debug", action='store_true', help="If set, debug outputs will be saved.")
    args = parser.parse_args()

    # Print a nice statement outlining chosen inputs
    print(f"Processing scene file: {args.scene_file}")
    print(f"Class column index: {args.class_col}")
    print(f"Leaf class keys: {args.leaf_keys}")
    print(f"Wood class keys: {args.wood_keys}")
    print(f"Scene formats: {args.scene_formats}")
    print(f"Voxel sizes: {args.voxel_sizes}")
    print(f"Number of Angle Bins: {args.num_angle_bins}")
    print(f"Ray spacing: {args.ray_spacing}")
    print(f"Max workers: {args.max_workers}")
    print(f"Normals voxel size: {args.normals_voxel_size}")
    print(f"Debug mode: {args.debug}")

    file_ext = os.path.splitext(args.scene_file)[1].lower()

    _LEAF_KEYS = args.leaf_keys
    _WOOD_KEYS = args.wood_keys
    _USE_COLS = (0, 1, 2, args.class_col)  # x, y, z, class columns

    # Clear the joblib.Memory cache to ensure any updates are applied:
    memory.clear(warn=True)

    # Establish joblib tempdir for any launching process
    os.environ['JOBLIB_TEMP_FOLDER'] = os.environ.get('TMPDIR', '/tmp') + '/joblib'
    os.makedirs(os.environ['JOBLIB_TEMP_FOLDER'], exist_ok=True)

    
    import os, sys, tempfile
    print("Python:", sys.executable)
    print("TMPDIR:", os.environ.get('TMPDIR'))
    print("JOBLIB_TEMP_FOLDER:", os.environ.get('JOBLIB_TEMP_FOLDER'))
    print("tempfile.gettempdir():", tempfile.gettempdir())

    import joblib, loky
    print("joblib:", joblib.__version__)
    print("loky:", loky.__version__)
    print("loky file", loky.__file__)

    # Store global settings
    global DEBUG_MODE
    global DEBUG_PATH
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        DEBUG_PATH = os.path.join(os.path.dirname(args.scene_file), "debug")
        if not os.path.exists(DEBUG_PATH):
            os.makedirs(DEBUG_PATH)

    voxel_sizes = args.voxel_sizes
    ray_spacing = args.ray_spacing
    # cross_section_area = args.cross_section_area
    # lambda_1 = cross_section_area / (voxel_size ** 3)
    
    log_file = os.path.basename(args.scene_file).replace(file_ext, '.log')
    if log_file:
        class Logger(object):
            def __init__(self, filename):
                self.terminal = sys.stdout
                self.log = open(filename, "a", buffering=1, encoding="utf-8")
            def write(self, message):
                self.terminal.write(message)
                self.log.write(message)
            def flush(self):
                self.terminal.flush()
                self.log.flush()
        sys.stdout = sys.stderr = Logger(log_file)

    # Get number of os.cpus
    if os.environ.get("SLURM_CPUS_PER_TASK") is None:
        num_cpus = psutil.cpu_count(logical=True)
    else:
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)) * 2) # hyperthreading
    num_cpus = max(1, num_cpus)
    n_workers = min(args.max_workers, num_cpus)

    # Cut 0 to 90 degrees into discrete bins from num_angle_bins
    angles = np.linspace(0, 90, args.num_angle_bins + 1)  # Create bins from 0 to 90 degrees
    angle_centers = np.round((angles[:-1] + angles[1:]) / 2, 1)  # Calculate the centers of the bins
    angles = set(angle_centers)
    if 0 in angles:
        angles.remove(0)
        angles.add(0.0001)
    if 90 in angles:
        angles.remove(90)
        angles.add(89.9999)
    # Sort angles and pass to global
    ANGLE_ORDER = sorted(angles)


    # Process each voxel center in parallel and collect results
    # Use joblib's Memory to cache mesh loading, and pass cached mesh objects to loky workers
    # Cache mesh loading using joblib.Memory
    cached_leaf_points, cached_wood_points = load_points(args.scene_file, _USE_COLS, _LEAF_KEYS, _WOOD_KEYS)
    set_clip_globals(cached_leaf_points, cached_wood_points)
    bounds = (cached_leaf_points[:, 0:3].min(axis=0), cached_leaf_points[:, 0:3].max(axis=0))
    bounds = tuple(np.concatenate([cached_leaf_points[:, 0:3].min(axis=0), cached_leaf_points[:, 0:3].max(axis=0)]))

    # add normals and weights to cached leaf and wood points for later use in ray tracing; this way we only compute them once per scene
    if cached_leaf_points is not None and cached_leaf_points.size != 0:
        normals, weights = compute_normals_weights_from_points_parallel(cached_leaf_points[:, 0:3], voxel_size=args.normals_voxel_size)
        cached_leaf_points = np.column_stack([cached_leaf_points, normals, weights])
    if cached_wood_points is not None and cached_wood_points.size != 0:
        normals, weights = compute_normals_weights_from_points_parallel(cached_wood_points[:, 0:3], voxel_size=args.normals_voxel_size)
        cached_wood_points = np.column_stack([cached_wood_points, normals, weights])

    # Update globals with new cached points that include normals and weights
    set_clip_globals(cached_leaf_points, cached_wood_points)

    # Warm up the clipping function
    _ = _clip_one_pointcloud_with_aabb(
        _CLIP_GLOBALS['leaf_points'],
        voxel_center=np.asarray([0,0,0], dtype=float), voxel_size=voxel_sizes[0])

    # Build canonical grids for all voxel sizes
    _build_canonical_grids(voxel_sizes, ray_spacing)

    voxel_centers = None
    if args.existing_csv is not None:
        existing_df = pd.read_csv(args.existing_csv)
        if not all(col in existing_df.columns for col in ['voxel_cx', 'voxel_cy', 'voxel_cz']):
            raise ValueError("Existing CSV must contain 'voxel_cx', 'voxel_cy', and 'voxel_cz' columns.")
        voxel_centers = existing_df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values
        print(f"Loaded {len(voxel_centers)} voxel centers from {args.existing_csv}.")

    for voxel_size in voxel_sizes:
        # Batch voxel centers into number of CPUs
        if voxel_centers is None:
            voxel_centers, voxel_ids = generate_voxel_centers(
                voxel_size=voxel_size,  # Example voxel size, adjust as needed
                bounds=bounds  # Use the bounds from the loaded meshes
            )

        total_voxels = len(voxel_centers)

        # Batch the voxel centers into groups based on the number of CPUs
        batches = []
        voxel_center_batches = [voxel_centers[i:i + num_cpus] for i in range(0, len(voxel_centers), num_cpus)]

        def process_voxel_wrapper(voxel_center, leaf_points, wood_points, scenes, voxel_size, angles, lambda_1):
            try:
                args = (voxel_center, leaf_points, wood_points, scenes, voxel_size, angles, lambda_1)
                result = process_voxel(*args)
                # print(f" Processed voxel {args[0]} successfully.")
                return result
            except Exception as e:
                print(f"Error processing voxel {args[0]}: {e}")
                traceback.print_exc()
                return []

        # For each voxel_center batch, process the voxels in parallel
        start = dt.datetime.now()

        # Filter out voxel centers that are outside the bounds of leaf and wood meshes
        voxel_centers = filter_voxel_centers(
            voxel_centers=voxel_centers,
            voxel_size=voxel_size
        )

        pbar = tqdm(total=len(voxel_centers), desc="Clipping meshes", unit="voxels") 
        with tqdm_joblib(pbar):
            results = Parallel(n_jobs=n_workers, backend='threading')(
                    delayed(clip_one_thread)(voxel_center, voxel_size) for voxel_center in voxel_centers
                )

        # results = [item for sublist in results for item in sublist]  # Flatten list of lists
        clipped_voxel_centres, clipped_leaf_points, clipped_wood_points = zip(*results)
        valid_indices = [i for i, (v, lp, wp) in enumerate(results) if lp.shape[0] != 0 or wp.shape[0] != 0]

        clipped_voxel_centres = [clipped_voxel_centres[i] for i in valid_indices]
        clipped_leaf_points = [clipped_leaf_points[i] for i in valid_indices]
        clipped_wood_points = [clipped_wood_points[i] for i in valid_indices]

        if len(clipped_voxel_centres) == 0:
            print(f"No valid clipped meshes found for voxel size {voxel_size}. Skipping processing.")
            continue 

        clip_time = dt.datetime.now() - start

        grid = _grid(voxel_size, ray_spacing)
        lambda_1 = None  # Setup lambda1 later. Needs user provided leaf cross section area

        worker = partial(
            process_voxel_wrapper,
            scenes=args.scene_formats,
            voxel_size=voxel_size,
            angles=angles,
            lambda_1=lambda_1
        )

        
        start02 = dt.datetime.now()
        print(f"Preprocessing time: {clip_time}")

        voxel_results = []
        with tqdm_joblib(tqdm(total=len(clipped_voxel_centres), desc="Processing voxels", unit="voxel")):
            voxel_results = Parallel(n_jobs=n_workers, backend='loky', env_var='LOKY_DISABLE_RESOURCE_TRACKER=1')(
            delayed(worker)(vc, lm, wm) 
            for vc, lm, wm in zip(clipped_voxel_centres, clipped_leaf_points, clipped_wood_points)
            )
        voxel_results = [item for sublist in voxel_results for item in sublist]

        # Create a dataframe of voxel_results and subsequently filter by scene column
        df = pd.DataFrame(voxel_results)
        assert "scene" in df.columns, "No 'scene' column found in voxel results."

        # Add voxel_size and voxel_id column for easier tracking of voxels across scenes and debugging
        df["voxel_size"] = voxel_size
        df["voxel_id"] = df.apply(lambda row: create_voxel_id(row["voxel_size"], row["voxel_cx"], row["voxel_cy"], row["voxel_cz"]), axis=1)

        # Aggregate results by voxel_id and scene, taking mean of pgap, mean_path_length, and G values for each voxel across all faces and angles. This will give us one pgap, mean_path_length, and G estimate per voxel per scene to use for Clumping Index calculation.
        df_agg = df.groupby(["voxel_id", "scene"]).agg({
            "pgap_leaf": "mean",
            "pgap_wood": "mean",
            "pgap_comb": "mean",
            "mean_path_length_leaf": "mean",
            "mean_path_length_wood": "mean",
            "mean_path_length_comb": "mean",
            "G_leaf_computed": "mean",
            "G_wood_computed": "mean",
            "G_comb_computed": "mean",
            "LAD_est": "mean",
            "WAD_est": "mean",
            "PAD_est": "mean",
            "n_hits_leaf": "sum",
            "n_hits_wood": "sum",
            "n_hits_comb": "sum",
            "total_num_rays": "sum",
            "mean_free_path_length_leaf": "mean",
            "mean_free_path_length_wood": "mean",
            "mean_free_path_length_comb": "mean",
            "mean_eff_free_path_length_leaf": "mean",
            "mean_eff_free_path_length_wood": "mean",
            "mean_eff_free_path_length_comb": "mean"
        }).reset_index()
        
        # Apply mean aggregation to all other columns not explicitly specified
        other_cols = [col for col in df.columns if col not in ["voxel_id", "scene"] and col not in df_agg.columns and pd.api.types.is_numeric_dtype(df[col])]
        for col in other_cols:
            df_agg[col] = df.groupby(["voxel_id", "scene"])[col].mean().values


        # Calculate Clumping Index for each voxel using all aggregate pgap, mean_path_length, and G values per voxel.
        ### CI from pgap results ###
        def compute_CI(pgap, L, G, mean_path_length):
            if not np.isnan(L) and L > 0 and pgap > 0 and pgap < 1 and G > 0 and mean_path_length > 0:
                CI = -np.log(pgap) / (L * G * mean_path_length)
                return CI
            else:
                return np.nan
            
        def add_CI_to_df(
                row,
                tol = 0.02, # tolerance
                max_iter = 10, # usually converges in 1-3 iterations
                eps = 1e-9
            ):

            pgap_leaf = row["pgap_leaf"]
            pgap_wood = row["pgap_wood"]
            pgap_comb = row["pgap_comb"]
            LAD_est = row["LAD_est"]
            WAD_est = row["WAD_est"]
            PAD_est = row["PAD_est"]
            G_leaf_est = row["G_leaf_computed"]
            G_wood_est = row["G_wood_computed"]
            G_comb_est = row["G_comb_computed"]
            mean_path_length_leaf = row["mean_path_length_leaf"]
            mean_path_length_wood = row["mean_path_length_wood"]
            mean_path_length_comb = row["mean_path_length_comb"]

            CI_leaf = compute_CI(pgap_leaf, LAD_est, G_leaf_est, mean_path_length_leaf)
            CI_wood = compute_CI(pgap_wood, WAD_est, G_wood_est, mean_path_length_wood)
            CI_comb = compute_CI(pgap_comb, PAD_est, G_comb_est, mean_path_length_comb)

            LAD_prev = LAD_est
            WAD_prev = WAD_est
            PAD_prev = PAD_est

            for _ in range(max_iter):
                CI_leaf = compute_CI(pgap_leaf, LAD_prev, G_leaf_est, mean_path_length_leaf)
                LAD_new = LAD_prev / CI_leaf if not np.isnan(CI_leaf) and CI_leaf > 0 else np.nan
                
                if (abs(LAD_new - LAD_prev) + eps) < tol:
                    break
                
                LAD_prev = LAD_new

            for _ in range(max_iter):
                CI_wood = compute_CI(pgap_wood, WAD_prev, G_wood_est, mean_path_length_wood)
                WAD_new = WAD_prev / CI_wood if not np.isnan(CI_wood) and CI_wood > 0 else np.nan
                
                if (abs(WAD_new - WAD_prev)) / (abs(WAD_prev) + eps) < tol:
                    break
                
                WAD_prev = WAD_new

            for _ in range(max_iter):
                CI_comb = compute_CI(pgap_comb, PAD_prev, G_comb_est, mean_path_length_comb)
                PAD_new = PAD_prev / CI_comb if not np.isnan(CI_comb) and CI_comb > 0 else np.nan
                
                if (abs(PAD_new - PAD_prev)) / (abs(PAD_prev) + eps) < tol:
                    break
                
                PAD_prev = PAD_new
            
            LAD_corr = LAD_new
            WAD_corr = WAD_new
            PAD_corr = PAD_new

            return pd.Series({
                "CI_leaf": CI_leaf,
                "CI_wood": CI_wood,
                "CI_combined": CI_comb,
                "LAD_corr": LAD_corr,
                "WAD_corr": WAD_corr,
                "PAD_corr": PAD_corr
            })
        df_agg[["CI_leaf", "CI_wood", "CI_combined", "LAD_corr", "WAD_corr", "PAD_corr"]] = df_agg.apply(add_CI_to_df, axis=1)

        scenes_present = df["scene"].dropna().unique().tolist()

        total_time = dt.datetime.now() - start
        raytrace_time = dt.datetime.now() - start02

        for scene in scenes_present:
            df_s = df[df["scene"] == scene].copy()
            output_basename = os.path.basename(args.scene_file).replace(file_ext, f'_{scene}_results_{voxel_size}.csv')
            output_path = os.path.join(os.path.dirname(args.scene_file), output_basename)
            df_s.to_csv(output_path, index=False)
            agg_output_basename = os.path.basename(args.scene_file).replace(file_ext, f'_{scene}_results_aggregated_{voxel_size}.csv')
            agg_output_path = os.path.join(os.path.dirname(args.scene_file), agg_output_basename)
            df_agg[df_agg["scene"] == scene].to_csv(agg_output_path, index=False)
            print(f"Saved {len(df_s)} rows for scene '{scene}' to {output_path}.")

        # Save performance results to a separate CSV file
        clip_time = clip_time.total_seconds()
        raytrace_time = raytrace_time.total_seconds()
        total_time = total_time.total_seconds()

        def per_voxel(time):
            return time / len(clipped_voxel_centres) if clipped_voxel_centres else 0
        
        perf_df = pd.DataFrame([{
            "clip_per_voxel": per_voxel(clip_time),
            "clipping": clip_time,
            "raytrace_per_voxel": per_voxel(raytrace_time),
            "raytracing": raytrace_time,
            "total": total_time
        }])
        perf_output_basename = os.path.basename(args.scene_file).replace(file_ext, f'_performance_{voxel_size}.csv')
        perf_output_path = os.path.join(os.path.dirname(args.scene_file), perf_output_basename)
        perf_df.to_csv(perf_output_path, index=False)

        print(f"Processed {args.scene_file} and saved results to {output_path} in {total_time} seconds.")

    


    

