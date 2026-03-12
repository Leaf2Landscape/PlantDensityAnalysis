import os
import argparse
import json
import math
import pandas as pd
import numpy as np
import traceback
import open3d as o3d
import open3d.core as o3c
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

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

from utils import classify_liad_to_dewit, calculate_G, resolve_cuda_index
    
from numba import njit, prange

# --- Warp setup (add near top of your module) ---
import warp as wp
wp.init()  # JIT/runtime init; safe to call once

import threading
_gpu_lock = threading.Lock()

# choose device; you can parameterize this too
_WARP_DEVICE = "cuda:0"  # fall back to "cpu" for debug

# ---------- Warp helpers & kernel ----------

@wp.func
def _ray_box_intersect(orig: wp.vec3, dir: wp.vec3, bmin: wp.vec3, bmax: wp.vec3) -> wp.bool:
    # Standard slab test
    eps = 1.0e-12
    dx = dir[0]; dy = dir[1]; dz = dir[2]
    tx1 = (bmin[0] - orig[0]) / (dx if wp.abs(dx) > eps else (eps if dx >= 0.0 else -eps))
    tx2 = (bmax[0] - orig[0]) / (dx if wp.abs(dx) > eps else (eps if dx >= 0.0 else -eps))
    ty1 = (bmin[1] - orig[1]) / (dy if wp.abs(dy) > eps else (eps if dy >= 0.0 else -eps))
    ty2 = (bmax[1] - orig[1]) / (dy if wp.abs(dy) > eps else (eps if dy >= 0.0 else -eps))
    tz1 = (bmin[2] - orig[2]) / (dz if wp.abs(dz) > eps else (eps if dz >= 0.0 else -eps))
    tz2 = (bmax[2] - orig[2]) / (dz if wp.abs(dz) > eps else (eps if dz >= 0.0 else -eps))

    tmin = wp.max(wp.min(tx1, tx2), wp.max(wp.min(ty1, ty2), wp.min(tz1, tz2)))
    tmax = wp.min(wp.max(tx1, tx2), wp.min(wp.max(ty1, ty2), wp.max(tz1, tz2)))

    valid = (tmax >= 0.0) and (tmin <= tmax)
    return valid

@wp.kernel
def _raycast_voxel_kernel(
    leaf_id: wp.uint64, 
    wood_id: wp.uint64,
    has_leaf: int, 
    has_wood: int,
    origins: wp.array(dtype=wp.vec3),
    dirs: wp.array(dtype=wp.vec3),
    bmin: wp.vec3,
    bmax: wp.vec3,
    max_hits: int,
    eps_advance: float,
    first_any: wp.array(dtype=wp.float32),
    first_leaf: wp.array(dtype=wp.float32),
    first_wood: wp.array(dtype=wp.float32),
    count_all: wp.array(dtype=wp.int32),
    count_leaf: wp.array(dtype=wp.int32),
    count_wood: wp.array(dtype=wp.int32)
):
    i = wp.tid()
    o = origins[i]
    d = dirs[i]

    # Compute ray-box intersection manually (inline version of _ray_box_intersect)
    eps = 1.0e-12
    dx = d[0]
    dy = d[1]
    dz = d[2]
    
    # Safe division avoiding division by zero
    tx1 = (bmin[0] - o[0]) / wp.where(wp.abs(dx) > eps, dx, wp.where(dx >= 0.0, eps, -eps))
    tx2 = (bmax[0] - o[0]) / wp.where(wp.abs(dx) > eps, dx, wp.where(dx >= 0.0, eps, -eps))
    ty1 = (bmin[1] - o[1]) / wp.where(wp.abs(dy) > eps, dy, wp.where(dy >= 0.0, eps, -eps))
    ty2 = (bmax[1] - o[1]) / wp.where(wp.abs(dy) > eps, dy, wp.where(dy >= 0.0, eps, -eps))
    tz1 = (bmin[2] - o[2]) / wp.where(wp.abs(dz) > eps, dz, wp.where(dz >= 0.0, eps, -eps))
    tz2 = (bmax[2] - o[2]) / wp.where(wp.abs(dz) > eps, dz, wp.where(dz >= 0.0, eps, -eps))

    tnear = wp.max(wp.min(tx1, tx2), wp.max(wp.min(ty1, ty2), wp.min(tz1, tz2)))
    tfar = wp.min(wp.max(tx1, tx2), wp.min(wp.max(ty1, ty2), wp.max(tz1, tz2)))

    ok = (tfar >= 0.0) and (tnear <= tfar)
    if not ok:
        first_any[i]  = wp.float32(1e32)
        first_leaf[i] = wp.float32(1e32)
        first_wood[i] = wp.float32(1e32)
        return

    # Clip segment to slab
    start = o + d * tnear
    remain = tfar - tnear

    # Init outputs
    first_any[i]  = wp.float32(1e32)
    first_leaf[i] = wp.float32(1e32)
    first_wood[i] = wp.float32(1e32)

    ca = int(0)
    cl = int(0)
    cw = int(0)

    t_accum = float(0.0)
    h = int(0)
    while h < max_hits and remain > 1.0e-6:
        leaf_found = False
        wood_found = False
        t_leaf = wp.float32(1e32)
        t_wood = wp.float32(1e32)

        if has_leaf == 1:
            ql = wp.mesh_query_ray(leaf_id, start, d, remain, -1)
            if ql.result:
                leaf_found = True
                t_leaf = ql.t

        if has_wood == 1:
            qw = wp.mesh_query_ray(wood_id, start, d, remain, -1)
            if qw.result:
                wood_found = True
                t_wood = qw.t

        if (not leaf_found) and (not wood_found):
            break

        hit_is_leaf = False
        hit_is_wood = False

        if leaf_found and wood_found:
            if t_leaf < t_wood:
                hit_is_leaf = True
                t_first = t_leaf
            else:
                hit_is_wood = True
                t_first = t_wood
        elif leaf_found:
            hit_is_leaf = True
            t_first = t_leaf
        elif wood_found:
            hit_is_wood = True
            t_first = t_wood
        else:
            break   # no hits, should not happen due to earlier check

        ca += 1
        if hit_is_leaf:
            cl += 1
        if hit_is_wood:
            cw += 1

        if not wp.isfinite(first_any[i]):
            first_any[i] = t_accum + t_first
        if leaf_found and not wp.isfinite(first_leaf[i]):
            first_leaf[i] = t_accum + t_leaf
        if wood_found and not wp.isfinite(first_wood[i]):
            first_wood[i] = t_accum + t_wood

        t_step = t_first + eps_advance
        start = start + d * t_step
        remain = remain - t_step
        t_accum = float(t_accum + t_step)
        h = int(h + 1)

    count_all[i] = ca
    count_leaf[i] = cl
    count_wood[i] = cw
    

global _DEVICE


# Initialise cached variables for PMF and lambda_1 calculations
global _LAM_GRID
_LAM_GRID = np.linspace(1e-3, 3.0, 60)
global _OMEGA_GRID
_OMEGA_GRID = np.linspace(1e-3, 0.999, 60)
global _PMF_CACHE
_PMF_CACHE = None
global _PMF_N_MAX
_PMF_N_MAX = 0

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

@memory.cache
def load_wood_volume_file(wood_volume_file: str) -> Optional[np.ndarray]:
    """
    Load the wood volume file if it exists.
    The file is expected to be in the same directory as the scene file.
    """
    if os.path.exists(wood_volume_file):
        try:
            return np.loadtxt(wood_volume_file)
        except Exception as err:
            print(f"Error loading wood volume file {wood_volume_file}: {err}")
            return None
    else:
        return None
    
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
def load_mesh_trimesh(file_path: str) -> Optional[trimesh.Trimesh]:
    """
    Load a mesh using trimesh, returning None if the mesh is empty or invalid.
    """
    try:
        mesh = trimesh.load_mesh(file_path)
        if isinstance(mesh, trimesh.Trimesh):
            return mesh
        else:
            print(f"Invalid or empty mesh found in {file_path}.")
            return None
    except Exception as err:
        print(f"Error loading mesh from {file_path}: {err}")
        return None
    
# Global leaf and wood mesh cache
_LEAF_MESH = None
_WOOD_MESH = None
_LEAF_TREE = None
_WOOD_TREE = None

_LEAF_TRI_MIN = None
_LEAF_TRI_MAX = None   
_WOOD_TRI_MIN = None
_WOOD_TRI_MAX = None


def _ensure_clip_worker_meshes(leaf_mesh_path: str, wood_mesh_path: str, build_tree: bool=True):
    """Ensure that the global leaf and wood meshes are loaded in the worker process.
    """
    global _LEAF_MESH, _WOOD_MESH
    global _LEAF_TREE, _WOOD_TREE
    global _LEAF_TRI_MIN, _LEAF_TRI_MAX
    global _WOOD_TRI_MIN, _WOOD_TRI_MAX

    if _LEAF_MESH is None:
        _LEAF_MESH = load_mesh_trimesh(leaf_mesh_path)

        if _LEAF_MESH is not None and not _LEAF_MESH.is_empty:
            tris = _LEAF_MESH.triangles  # (N, 3, 3)
            _LEAF_TRI_MIN = tris.min(axis=1)
            _LEAF_TRI_MAX = tris.max(axis=1)
            if build_tree:
                _LEAF_TREE = _LEAF_MESH.triangles_tree  # built once per worker

    if _WOOD_MESH is None:
        _WOOD_MESH = load_mesh_trimesh(wood_mesh_path)
        
        if _WOOD_MESH is not None and not _WOOD_MESH.is_empty:
            tris = _WOOD_MESH.triangles
            _WOOD_TRI_MIN = tris.min(axis=1)
            _WOOD_TRI_MAX = tris.max(axis=1)
            if build_tree:
                _WOOD_TREE = _WOOD_MESH.triangles_tree


def _clip_one_mesh_with_aabb(mesh, tri_min, tri_max, voxel_center, voxel_size):
    """Fast candidate selection via triangle AABB overlap; then clip via PyVista."""
    if mesh is None or mesh.is_empty:
        return (np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))

    half = voxel_size / 2.0
    min_bound = np.asarray(voxel_center) - half
    max_bound = np.asarray(voxel_center) + half
    bounds6 = (min_bound[0], max_bound[0], min_bound[1], max_bound[1], min_bound[2], max_bound[2])

    overlap = (tri_max >= min_bound).all(axis=1) & (tri_min <= max_bound).all(axis=1)
    idx = np.flatnonzero(overlap)
    if idx.size == 0:
        return (np.empty((0, 3), np.float64), np.empty((0, 3), np.int64))

    submesh = mesh.submesh([idx], append=True)
    p_sub = pv.wrap(submesh)

    # Prefer bounds param to avoid constructing a cube dataset
    try:
        clipped = p_sub.clip_box(bounds=bounds6, invert=False)
    except TypeError:
        cube = pv.Cube(center=voxel_center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
        clipped = p_sub.clip_box(cube, invert=False)

    if isinstance(clipped, pv.UnstructuredGrid):
        clipped = clipped.extract_surface(algorithm=None)
    if not clipped.is_all_triangles:
        clipped = clipped.triangulate()

    vertices = np.asarray(clipped.points)
    faces = np.asarray(clipped.faces.reshape((-1, 4))[:, 1:])
    return (vertices, faces)

    

def get_clipped_meshes(
        leaf_mesh,
        wood_mesh,
        voxel_center,
        voxel_size) -> trimesh.Trimesh:
    """
    Roughly clip a mesh using a bounding box defined by the voxel center and size.
    This function uses a KDTree for efficient point querying.
    """

    half_size = voxel_size / 2.0
    min_bound = np.array(voxel_center) - half_size
    max_bound = np.array(voxel_center) + half_size
    voxel_bounds = np.stack([min_bound, max_bound], axis=0)

    if leaf_mesh is None or leaf_mesh.is_empty:
        print(f"Leaf mesh is empty or None at voxel center {voxel_center}.")
        clipped_leaf_faces = np.empty((0, 3), dtype=np.int64)
        clipped_leaf_vertices = np.empty((0, 3), dtype=np.float64)
        leaf_mesh = None

    else:

        leaf_tree = leaf_mesh.triangles_tree

        candidate_triangle_indices = list(leaf_tree.intersection(voxel_bounds.flatten()))

        if not candidate_triangle_indices:
            # print(f"No triangles found within voxel bounds {voxel_bounds}.")
            del leaf_mesh, leaf_tree
            gc.collect()

            clipped_leaf_faces = np.empty((0, 3), dtype=np.int64)
            clipped_leaf_vertices = np.empty((0, 3), dtype=np.float64)

        else:
            # Extract triangles within the voxel bounds
            sub_mesh = leaf_mesh.submesh([candidate_triangle_indices], append=True)
            del leaf_mesh, leaf_tree

            # Create a PyVista mesh from the trimesh submesh
            sub_mesh = pv.wrap(sub_mesh)
            voxel = pv.Cube(center=voxel_center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
            clipped_leaf_mesh = sub_mesh.clip_box(voxel, invert=False)

            if isinstance(clipped_leaf_mesh, pv.UnstructuredGrid):
                clipped_leaf_mesh = clipped_leaf_mesh.extract_geometry()
            if not clipped_leaf_mesh.is_all_triangles:
                clipped_leaf_mesh = clipped_leaf_mesh.triangulate()
            clipped_leaf_vertices = np.asarray(clipped_leaf_mesh.points)
            clipped_leaf_faces = np.asarray(clipped_leaf_mesh.faces.reshape((-1, 4))[:, 1:])

            # print(f"No valid mesh found after clipping for voxel at {voxel_center}.")

            ### DEBUG ###
            # Save .ply files for debugging
            # cube = pv.Cube(center=voxel_center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
            # cube.save(f"voxel_cube_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.ply")
            # clipped_leaf_mesh.save(f"clipped_leaf_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.ply")

            del sub_mesh, clipped_leaf_mesh
            gc.collect()
            
    
    # ### DEBUG ####
    # # Save .ply files for debugging
    # project_dir = os.path.dirname(leaf_mesh_path)
    # surface_area = clipped_leaf_mesh.area if hasattr(clipped_leaf_mesh, 'area') else 0.0
    # debug_leaf_path = os.path.join(project_dir, f"clipped_leaf_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}_{surface_area}.ply")
    # debug_wood_path = os.path.join(project_dir, f"clipped_wood_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.ply")
    # clipped_leaf_mesh.save(debug_leaf_path)

    if wood_mesh is None or wood_mesh.is_empty:   
        clipped_wood_faces = np.empty((0, 3), dtype=np.int64)
        clipped_wood_vertices = np.empty((0, 3), dtype=np.float64)
        wood_mesh = None
    else:
        wood_tree = wood_mesh.triangles_tree

        candidate_triangle_indices = list(wood_tree.intersection(voxel_bounds.flatten()))

        if not candidate_triangle_indices:
            # print(f"No triangles found within voxel bounds {voxel_bounds}.")
            del wood_mesh, wood_tree
            gc.collect()

            clipped_wood_faces = np.empty((0, 3), dtype=np.int64)
            clipped_wood_vertices = np.empty((0, 3), dtype=np.float64)
        else:
            # Extract triangles within the voxel bounds
            sub_mesh = wood_mesh.submesh([candidate_triangle_indices], append=True)
            del wood_mesh, wood_tree

            # Create a PyVista mesh from the trimesh submesh
            sub_mesh = pv.wrap(sub_mesh)
            voxel = pv.Cube(center=voxel_center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
            clipped_wood_mesh = sub_mesh.clip_box(voxel, invert=False)
            if isinstance(clipped_wood_mesh, pv.UnstructuredGrid):
                clipped_wood_mesh = clipped_wood_mesh.extract_geometry()
            if not clipped_wood_mesh.is_all_triangles:
                clipped_wood_mesh = clipped_wood_mesh.triangulate()
            clipped_wood_vertices = np.asarray(clipped_wood_mesh.points)
            clipped_wood_faces = np.asarray(clipped_wood_mesh.faces.reshape((-1, 4))[:, 1:])

            ### DEBUG ###
            # clipped_wood_mesh.save(f"clipped_wood_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.ply")

    return voxel_center, clipped_leaf_vertices, clipped_leaf_faces, clipped_wood_vertices, clipped_wood_faces


def build_voxel_scene(o3d_leaf, o3d_wood):
    """Return (scene, leaf_id, wood_id)  either id may be None."""

    _load_leaf = True if o3d_leaf is not None and len(o3d_leaf.triangles) > 0 else False
    _load_wood = True if o3d_wood is not None and len(o3d_wood.triangles) > 0 else False
    leaf_id = wood_id = None
    
    try:
        scene = o3d.t.geometry.RaycastingScene(device=_DEV)

        if _load_leaf:
            leaves = o3d.t.geometry.TriangleMesh.from_legacy(o3d_leaf)
            leaves = leaves.to(_DEV)
            leaf_id = scene.add_triangles(leaves)
        
        if _load_wood:
            wood = o3d.t.geometry.TriangleMesh.from_legacy(o3d_wood)
            wood = wood.to(_DEV)
            wood_id = scene.add_triangles(wood)

        # print("[INFO] Using CUDA for raycasting.")
    except Exception as e:
        # Fallback to CPU
        dev = o3d.core.Device("CPU:0")
        scene = o3d.t.geometry.RaycastingScene(device=dev)

        if _load_leaf:
            leaves = o3d.t.geometry.TriangleMesh.from_legacy(o3d_leaf)
            leaf_id = scene.add_triangles(leaves)
        
        if _load_wood:
            wood = o3d.t.geometry.TriangleMesh.from_legacy(o3d_wood)
            wood_id = scene.add_triangles(wood)

        # print(f"[INFO] CUDA unavailable, falling back to CPU: {e}.")

    return scene, leaf_id, wood_id

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

def compute_LIAD_from_mesh(o3d_mesh, num_bins=18):
    """
    Compute area-weighted leaf inclination distribution.
    Returns: (bin centers, LIAD, mean angle)
    """
    if (o3d_mesh is None) or (len(o3d_mesh.triangles) == 0):
        return np.array([]), np.array([]), np.nan
    
    verts = np.asarray(o3d_mesh.vertices)
    tris = np.asarray(o3d_mesh.triangles)
    v0 = verts[tris[:, 1]] - verts[tris[:, 0]]
    v1 = verts[tris[:, 2]] - verts[tris[:, 0]]

    cross_prod = np.cross(v0, v1)
    areas = 0.5 * np.linalg.norm(cross_prod, axis=1)
    norms = np.linalg.norm(cross_prod, axis=1, keepdims=True)
    normals = np.divide(cross_prod, norms, where=(norms != 0), out=np.zeros_like(cross_prod))

    angle_facets = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
    angle_facets = np.where(angle_facets > 90, 180 - angle_facets, angle_facets)
    angle_facets = 90.0 - angle_facets  # Convert to plane such that vertical leaves are small angles and horizontal leaves are large

    mean_angle = angle_facets.mean() if angle_facets.size > 0 else np.nan
    bin_edges = np.linspace(0, 90, num_bins + 1)
    idx = np.digitize(angle_facets, bin_edges) - 1
    idx = np.clip(idx, 0, num_bins - 1)
    bin_counts = np.bincount(idx, weights=areas, minlength=num_bins)
    total_area = areas.sum()
    liad = bin_counts / total_area if total_area > 0 else np.zeros(num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers, liad, mean_angle

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
    else:
        # Ensure argument to log is positive: 1.0 - lambda_1 * d_arr[mask] > 0
        valid_mask = mask & (1.0 - lambda_1 * d_arr > 0)
        out[valid_mask] = -np.log(1.0 - lambda_1 * d_arr[valid_mask]) / lambda_1
    return out


### Ray tracing functions ###
def _grid(voxel_size, ray_spacing):
    """
    Generate a grid covering a square of size `face_len` centered at the origin.
    For full coverage when rotating the voxel, use face_len = voxel_size * sqrt(2).
    This ensures the grid covers the diagonal of the voxel after rotation.
    """
    face_len = (voxel_size * np.sqrt(2)) + 1e-6  # Add small epsilon to ensure all rays cover the diagonal corner
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

def generate_face_rays_top(vc, vs, grid):
    # XX, YY = _grid(vs * 2, spc)
    XX, YY = grid
    zface = vc[2] + vs * 2
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

def generate_side_rays_xminus(vc, vs, grid):
    # YY, ZZ = _grid(vs * 2, spc)
    YY, ZZ = grid
    xface = vc[0] - vs * 2
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

def generate_side_rays_yminus(vc, vs, grid):
    # XX, ZZ = _grid(vs * 2, spc)
    XX, ZZ = grid
    yface = vc[1] - vs * 2
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

def compute_lad_metrics(hits_leaf, N, G_leaf, delta_bar, mean_z, mean_delta_e, var_delta_e, lambda_1):
    """
    Compute LAD metrics from the leaf-only simulation.
    """
    results = {
        'LAD_BL': np.nan,
        'LAD_BL_EPL': np.nan,
        'LAD_BL_UEPL': np.nan,
        'LAD_MCF': np.nan,
        'LAD_MCF_Corrected': np.nan
    }
    eps = 1e-9
    I_leaf = hits_leaf / float(N) if N > 0 else 0.0
    if (N > 0) and (0 < I_leaf < 1) and (delta_bar > eps) and (G_leaf > eps):
        results['LAD_BL'] = -math.log(1.0 - I_leaf) / (G_leaf * delta_bar)
    if N > 0:
        bot = math.log(2.0 * N + 2.0)
    else:
        bot = np.nan
    if (N > 0) and (0 < I_leaf < 1) and (mean_delta_e > eps):
        top = math.log(1 - I_leaf) + (I_leaf / (2.0 * N * (1 - I_leaf)))
        bot = math.log(2.0 * N + 2.0)
        if abs(bot) > eps:
            val_epl = -(1.0 / mean_delta_e) * top
            results['LAD_BL_EPL'] = val_epl 
    elif I_leaf == 1.0:
        results['LAD_BL_EPL'] = bot / mean_delta_e 
    lad_bl_epl_val = results['LAD_BL_EPL']
    if (not np.isnan(lad_bl_epl_val) and lad_bl_epl_val > 0 and
        var_delta_e > eps and mean_delta_e > eps):
        a_e = (mean_delta_e ** 2) / var_delta_e
        inside = 1.0 - 2.0 * a_e * lad_bl_epl_val
        if inside >= 0.0:
            val_uepl = (1.0 / a_e) * (1.0 - math.sqrt(inside))
            results['LAD_BL_UEPL'] = val_uepl / G_leaf
    if (not np.isnan(lad_bl_epl_val) and lad_bl_epl_val > 0 and (G_leaf > eps)):
        results['LAD_BL_EPL'] = lad_bl_epl_val / G_leaf
    if (mean_z > eps) and (G_leaf > eps):
        results['LAD_MCF'] = I_leaf / (G_leaf * mean_z)
    if (lambda_1 > 0 and delta_bar > 0 and (0 < I_leaf < 1) and 
        (mean_z > eps) and (1 - lambda_1 * delta_bar) > 0):
        denom = math.log(1.0 - lambda_1 * delta_bar) * mean_z
        if abs(denom) > eps:
            val_corr = -1.0 * (lambda_1 * delta_bar * I_leaf) / denom
            results['LAD_MCF_Corrected'] = val_corr / G_leaf
    return results

def compute_pad_metrics(hits_lw, hits_leaf, N, G_lw, delta_bar, sum_z_lw,
                        sum_z_e_lw, sum_hits_z_e_leaf, sum_hits_z_e_lw,
                        alpha, mean_delta_e, var_delta_e, lambda_1, leaf_fraction):
    """
    Compute PAD metrics from the combined simulation.
    In a single-mesh scenario, wood and leaf data are identical.
    """
    results = {
        'PAD_BL': np.nan,
        'PAD_BL_EPL': np.nan,
        'PAD_BL_UEPL': np.nan,
        'PAD_MCF': np.nan,
        'PAD_MCF_Corrected': np.nan,
        'PAD_MLE_pimont_2018': np.nan,
        'LAD_MLE_pimont_2019': np.nan,
        'LAD_MLE_Soma_21': np.nan
    }
    eps = 1e-9
    I_lw = hits_lw / float(N) if N > 0 else 0.0
    if (N > 0) and (0 < I_lw < 1) and (delta_bar > eps) and (G_lw > eps):
        results['PAD_BL'] = -math.log(1.0 - I_lw) / (G_lw * delta_bar)
    if N > 0:
        bot = math.log(2.0 * N + 2.0)
    else:
        bot = np.nan
    if (N > 0) and (0 < I_lw < 1) and (mean_delta_e > eps):
        top = math.log(1 - I_lw) + (I_lw / (2.0 * N * (1 - I_lw)))
        bot = math.log(2.0 * N + 2.0)
        if abs(bot) > eps:
            val_epl = -(1.0 / mean_delta_e) * top 
            results['PAD_BL_EPL'] = val_epl
    elif I_lw == 1.0:
        results['PAD_BL_EPL'] = bot / mean_delta_e
    pad_bl_epl_val = results['PAD_BL_EPL']
    if (not np.isnan(pad_bl_epl_val) and pad_bl_epl_val > 0 and
        var_delta_e > eps and mean_delta_e > eps):
        a_e = (mean_delta_e ** 2) / var_delta_e
        inside = 1.0 - 2.0 * a_e * pad_bl_epl_val
        if inside >= 0.0:
            val_uepl = (1.0 / a_e) * (1.0 - math.sqrt(inside))
            results['PAD_BL_UEPL'] = val_uepl / G_lw
    if (not np.isnan(pad_bl_epl_val) and pad_bl_epl_val > 0 and (G_lw > eps)):
        results['PAD_BL_EPL'] = pad_bl_epl_val / G_lw
        
    mean_z_lw = sum_z_lw / N if N > 0 else 0.0  
    if (mean_z_lw > eps) and (G_lw > eps):
        results['PAD_MCF'] = I_lw / (G_lw * mean_z_lw)
    if (lambda_1 > 0 and delta_bar > 0 and (0 < I_lw < 1) and 
        (mean_z_lw > eps) and (1 - lambda_1 * delta_bar) > 0):
        denom = math.log(1.0 - lambda_1 * delta_bar) * mean_z_lw
        if abs(denom) > eps:
            val_corr = -1.0 * (lambda_1 * delta_bar * I_lw) / denom
            results['PAD_MCF_Corrected'] = val_corr / G_lw
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_z_e_lw > eps) and (alpha is not None):
        bracket = hits_lw - (sum_hits_z_e_lw / sum_z_e_lw)
        results['PAD_MLE_pimont_2018'] = (alpha * leaf_fraction / (G_lw * sum_z_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_z_e_lw > eps) and (alpha is not None):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_z_e_lw)
        results['LAD_MLE_pimont_2019'] = (alpha * leaf_fraction / (G_lw * sum_z_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_z_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_z_lw)
        results['LAD_MLE_Soma_21'] = (leaf_fraction / (G_lw * sum_z_lw)) * bracket
    return results

def load_and_split_by_group(scene_file: Union[str, Path], leaf_keys, wood_keys) -> Tuple[Optional[str], Optional[str], Tuple[float, float, float, float, float, float], Optional[trimesh.Trimesh], Optional[trimesh.Trimesh]]:
    """
    Load and split the scene file into leaf and wood meshes using trimesh.
    Returns paths to saved leaf and wood mesh files, bounds, and trimesh objects.
    """
    verts: List[List[float]] = []
    leaf_faces: List[List[int]] = []
    wood_faces: List[List[int]] = []

    leaf_mesh_path = str(scene_file).replace('.obj', '_leaf.obj')
    wood_mesh_path = str(scene_file).replace('.obj', '_wood.obj')

    leaf_mesh = None
    wood_mesh = None

    if os.path.exists(leaf_mesh_path):
        print(f"Leaf mesh already exists at {leaf_mesh_path}. Loading from file.")
        leaf_mesh = trimesh.load_mesh(leaf_mesh_path, process=False)
    if os.path.exists(wood_mesh_path):
        print(f"Wood mesh already exists at {wood_mesh_path}. Loading from file.")
        wood_mesh = trimesh.load_mesh(wood_mesh_path, process=False)

    if leaf_mesh is None or wood_mesh is None:
        current_tag = ""
        with open(scene_file, 'r', errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(coord) for coord in line.split()[1:4]])
                elif line.startswith(("g ", "o ")):
                    current_tag = line[2:].strip().lower()
                elif line.startswith("f "):
                    face = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                    if any(key in current_tag for key in leaf_keys):
                        leaf_faces.append(face)
                    elif any(key in current_tag for key in wood_keys):
                        wood_faces.append(face)

        verts = np.asarray(verts, dtype=np.float64)
        if leaf_faces:
            leaf_mesh = trimesh.Trimesh(vertices=verts, faces=leaf_faces, process=False)
            print(f"Saving leaf mesh to {leaf_mesh_path}.")
            leaf_mesh.export(leaf_mesh_path)
        else:
            print(f"No leaf mesh found in {scene_file}. Leaf mesh will not be saved.")
            leaf_mesh_path = ""
            leaf_mesh = None
        if wood_faces:
            wood_mesh = trimesh.Trimesh(vertices=verts, faces=wood_faces, process=False)
            print(f"Saving wood mesh to {wood_mesh_path}.")
            wood_mesh.export(wood_mesh_path)
        else:
            print(f"No wood mesh found in {scene_file}. Wood mesh will not be saved.")
            wood_mesh_path = ""
            wood_mesh = None

    # Use trimesh to get bounds
    # Compute combined bounds from valid meshes
    bounds_list = []
    if leaf_mesh is not None and hasattr(leaf_mesh, "bounds"):
        bounds_list.append(leaf_mesh.bounds)
    if wood_mesh is not None and hasattr(wood_mesh, "bounds"):
        bounds_list.append(wood_mesh.bounds)
    if bounds_list:
        min_bounds = np.min([b[0] for b in bounds_list], axis=0)
        max_bounds = np.max([b[1] for b in bounds_list], axis=0)
        bounds = tuple(np.concatenate([min_bounds, max_bounds]))
    else:
        # Fallback: load scene mesh and use its bounds
        scene_mesh = trimesh.load_mesh(scene_file, process=False)
        bounds = tuple(scene_mesh.bounds.flatten())  # (minx, miny, minz, maxx, maxy, maxz)

    return leaf_mesh_path, wood_mesh_path, bounds, leaf_mesh, wood_mesh

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
    voxel_ids = np.char.add(
        np.char.add(coords[:, 0].astype(str), '_'),
        np.char.add(coords[:, 1].astype(str), '_')
    )
    voxel_ids = np.char.add(voxel_ids, coords[:, 2].astype(str))
    
    return voxel_centers, voxel_ids

def filter_voxel_centers(voxel_centers, leaf_bounds, wood_bounds, voxel_size):
    """
    Filter voxel centers based on the bounds of leaf and wood meshes.
    """
    min_leaf, max_leaf = leaf_bounds[0:3], leaf_bounds[3:6]
    min_wood, max_wood = wood_bounds[0:3], wood_bounds[3:6]

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
            "top":    generate_face_rays_top(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "xplus":  generate_side_rays_xplus(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "xminus": generate_side_rays_xminus(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "yplus":  generate_side_rays_yplus(vc=np.array([0,0,0]), vs=vs, grid=grid),
            "yminus": generate_side_rays_yminus(vc=np.array([0,0,0]), vs=vs, grid=grid),
        }

global FACE_ORDER
FACE_ORDER = ["bottom","top","xplus","xminus","yplus","yminus"]

global ANGLE_ORDER

# faces that rotate around y vs x (matches your current logic)
rot_axis = {"xplus":"y","xminus":"y","bottom":"x","top":"x","yplus":"x","yminus":"x"}

def build_ray_tensor_for_voxel(voxel_center, voxel_size, angles):
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
    
### --- Function for simulation and metrics calculation --- ###
def simulate_voxel_grouped_warp(
    voxel_center: np.ndarray,
    voxel_size: float,
    rays_FAR6: np.ndarray,     # shape (F, A, R, 6) == [ox,oy,oz, dx,dy,dz]
    leaf_mesh: dict | None,    # {"vertices": (n,3) float64/32, "faces": (m,3) int64/32} or {}
    wood_mesh: dict | None,    # same as above or {}
    device: str = _WARP_DEVICE,
    max_hits: int = 8,
    eps_advance: float = 1e-5,
    max_rays_per_batch: int = 1000000  # Limit rays per batch to avoid OOM
):
    """
    Returns a dict of numpy arrays with shape (F, A, ...) matching your downstream metrics.
    Processes rays in batches to avoid GPU memory exhaustion.
    """

    # --------- 1) Build Warp meshes (BVH) for this voxel's clipped meshes ----------
    has_leaf = int(bool(leaf_mesh and len(leaf_mesh.get("faces", [])) > 0))
    has_wood = int(bool(wood_mesh and len(wood_mesh.get("faces", [])) > 0))

    def _make_wp_mesh(md: dict | None):
        if not md or len(md.get("faces", [])) == 0:
            return None
        v = np.asarray(md["vertices"], dtype=np.float32, order="C")
        f = np.asarray(md["faces"],    dtype=np.int32,   order="C").ravel()
        if len(v) == 0 or len(f) == 0:
            return None
        v_d = wp.array(v, dtype=wp.vec3,  device=device)
        f_d = wp.array(f, dtype=wp.int32, device=device)
        return wp.Mesh(points=v_d, indices=f_d)

    with _gpu_lock:
        leaf_wp = _make_wp_mesh(leaf_mesh)
        wood_wp = _make_wp_mesh(wood_mesh)

        leaf_id = leaf_wp.id if leaf_wp is not None else None
        wood_id = wood_wp.id if wood_wp is not None else None

        # --------- 2) Prepare rays ----------
        F, A, R, _ = rays_FAR6.shape
        n_rays = F * A * R
        O = rays_FAR6[..., 0:3].reshape(n_rays, 3).astype(np.float32, order="C")
        D = rays_FAR6[..., 3:6].reshape(n_rays, 3).astype(np.float32, order="C")

        # Voxel slab bounds
        vc   = np.asarray(voxel_center, dtype=np.float32)
        half = float(voxel_size) * 0.5
        bmin = wp.vec3(vc[0]-half, vc[1]-half, vc[2]-half)
        bmax = wp.vec3(vc[0]+half, vc[1]+half, vc[2]+half)

        # --------- 3) Allocate outputs ----------
        first_any = np.full(n_rays, np.inf, dtype=np.float32)
        first_leaf = np.full(n_rays, np.inf, dtype=np.float32)
        first_wood = np.full(n_rays, np.inf, dtype=np.float32)
        cnt_all = np.zeros(n_rays, dtype=np.int32)
        cnt_leaf = np.zeros(n_rays, dtype=np.int32)
        cnt_wood = np.zeros(n_rays, dtype=np.int32)

        # --------- 4) Process rays in batches ----------
        num_batches = (n_rays + max_rays_per_batch - 1) // max_rays_per_batch
        for batch_idx in range(num_batches):
            start_idx = batch_idx * max_rays_per_batch
            end_idx = min(start_idx + max_rays_per_batch, n_rays)
            batch_size = end_idx - start_idx

            O_batch = O[start_idx:end_idx]
            D_batch = D[start_idx:end_idx]

            # Upload batch to GPU
            origins_d = wp.array(O_batch, dtype=wp.vec3, device=device)
            dirs_d    = wp.array(D_batch, dtype=wp.vec3, device=device)

            # Allocate output arrays for this batch
            first_any_d  = wp.zeros(batch_size, dtype=float,     device=device)
            first_leaf_d = wp.zeros(batch_size, dtype=float,     device=device)
            first_wood_d = wp.zeros(batch_size, dtype=float,     device=device)
            count_all_d  = wp.zeros(batch_size, dtype=wp.int32,  device=device)
            count_leaf_d = wp.zeros(batch_size, dtype=wp.int32,  device=device)
            count_wood_d = wp.zeros(batch_size, dtype=wp.int32,  device=device)

            # Launch kernel on this batch
            # Only pass valid mesh IDs to the kernel
            leaf_id_arg = leaf_id if leaf_id is not None else wp.uint64(0)
            wood_id_arg = wood_id if wood_id is not None else wp.uint64(0)
            
            wp.launch(
                kernel=_raycast_voxel_kernel,
                dim=batch_size,
                inputs=[
                    leaf_id_arg, wood_id_arg,
                    has_leaf, has_wood,
                    origins_d, dirs_d,
                    bmin, bmax,
                    int(max_hits), float(eps_advance),
                    first_any_d, first_leaf_d, first_wood_d,
                    count_all_d, count_leaf_d, count_wood_d
                ],
                device=device
            )

            # Copy results back and store in output arrays
            first_any[start_idx:end_idx] = first_any_d.numpy()
            first_leaf[start_idx:end_idx] = first_leaf_d.numpy()
            first_wood[start_idx:end_idx] = first_wood_d.numpy()
            cnt_all[start_idx:end_idx] = count_all_d.numpy()
            cnt_leaf[start_idx:end_idx] = count_leaf_d.numpy()
            cnt_wood[start_idx:end_idx] = count_wood_d.numpy()

        # Clean up batch arrays
        del origins_d, dirs_d
        del first_any_d, first_leaf_d, first_wood_d
        del count_all_d, count_leaf_d, count_wood_d
        gc.collect()

    # --------- 5) Reshape to (F, A, R) ----------
    first_any  = first_any.reshape(F, A, R)
    first_leaf = first_leaf.reshape(F, A, R)
    first_wood = first_wood.reshape(F, A, R)
    cnt_all    = cnt_all.reshape(F, A, R)
    cnt_leaf   = cnt_leaf.reshape(F, A, R)
    cnt_wood   = cnt_wood.reshape(F, A, R)

    # --------- 6) Package results for your downstream code ----------
    out = {
        "first_hit_any":  first_any,
        "first_hit_leaf": first_leaf,
        "first_hit_wood": first_wood,
        "counts_all":     cnt_all,
        "counts_leaf":    cnt_leaf,
        "counts_wood":    cnt_wood,
        # Convenience for valid-ray mask (anything finite after slab clip)
        "valid_any":      np.isfinite(first_any),
        "valid_leaf":     np.isfinite(first_leaf),
        "valid_wood":     np.isfinite(first_wood),
        "F": F, "A": A, "R": R
    }
    return out

def process_voxel(
    voxel_center, 
    voxel_size, 
    leaf_mesh,
    wood_mesh, 
    angles, 
    wood_volume_points, 
    lambda_1
    ):
    """
    Process a single voxel center with the given parameters.
    This function should contain the logic to process the voxel.
    """    
    # Build meshes
    if leaf_mesh is not {}:
        o3d_leaf = o3d.geometry.TriangleMesh()
        o3d_leaf.vertices = o3d.utility.Vector3dVector(leaf_vertices)
        o3d_leaf.triangles = o3d.utility.Vector3iVector(leaf_faces)
    else:
        o3d_leaf = None

    if wood_mesh is not {}:
        o3d_wood = o3d.geometry.TriangleMesh()
        o3d_wood.vertices = o3d.utility.Vector3dVector(leaf_vertices)
        o3d_wood.triangles = o3d.utility.Vector3iVector(leaf_faces)

    # Extract surface areas from meshes
    try:
        leaf_area = o3d_leaf.get_surface_area() if o3d_leaf else 0.0
        wood_area = o3d_wood.get_surface_area() if o3d_wood else 0.0
        comb_area = leaf_area + wood_area
    except Exception as e:
        raise RuntimeError(
            f"Error computing surface area for voxel at {voxel_center}: {e}"
        ) from e
    
    # Calculate LAI, WAI, and PAI
    try:
        LAI = leaf_area / (voxel_size ** 2)
        WAI = wood_area / (voxel_size ** 2)
        PAI = comb_area / (voxel_size ** 2)
        leaf_fraction_ref = LAI / PAI if PAI > 0 else 0.0
    except Exception as e:
        raise RuntimeError(
            f"Error computing LAI, WAI, or PAI for voxel at {voxel_center}: {e}"
        ) from e
    
    # Calculate LAD, WAD, and PAD
    try:
        LAD = leaf_area / (voxel_size ** 3)
        WAD = wood_area / (voxel_size ** 3)
        PAD = comb_area / (voxel_size ** 3)
    except Exception as e:
        raise RuntimeError(
            f"Error computing LAD, WAD, or PAD for voxel at {voxel_center}: {e}"
        ) from e
    
    # Calculate wood volume (alpha) in voxel if points provided
    try:
        wood_vol = compute_wood_volume_in_voxel(wood_volume_points, voxel_center, voxel_size) if wood_volume_points is not None else 0.0
        if wood_vol != 0.0:
            alpha = (voxel_size ** 3 - wood_vol) / (voxel_size ** 3)
        else:
            alpha = None
    except Exception as e:
        raise RuntimeError(
            f"Error computing wood volume or alpha for voxel at {voxel_center}: {e}"
        ) from e
    
    # Compute LIAD bins for voxel
    try:
        bin_leaf, liad, _ = compute_LIAD_from_mesh(o3d_leaf) if o3d_leaf else (np.array([]), np.array([]), None)
        bin_wood, wiad, _ = compute_LIAD_from_mesh(o3d_wood) if o3d_wood else (np.array([]), np.array([]), None)
        # Calculate PIAD from aggregated liad and wiad and renormalising
        if liad.size > 0 and wiad.size > 0:
            piad = (liad * LAD + wiad * WAD) / (LAD + WAD) if (LAD + WAD) > 0 else np.array([])
            bin_all = bin_leaf  # Assuming same bins for combined; adjust if needed
        elif liad.size > 0 and not wiad.size > 0:
            piad = liad
            bin_all = bin_leaf
        elif wiad.size > 0 and not liad.size > 0:
            piad = wiad
            bin_all = bin_wood
        else:
            piad = np.array([])
            bin_all = np.array([])
        
        if liad.size > 0 and bin_leaf.size > 0:
            liad_dewit, liad_rmse, liad_l1 = classify_liad_to_dewit(
                liad=liad,
                bin_centres_deg=bin_leaf,
                return_scores=True
            )
            liad_rmse = np.nan if liad_rmse is None else liad_rmse
            liad_l1 = np.nan if liad_l1 is None else liad_l1
        else:
            liad_dewit = "NA"
            liad_rmse = np.nan
            liad_l1 = np.nan

        if wiad.size > 0 and bin_wood.size > 0:
            wiad_dewit, wiad_rmse, wiad_l1 = classify_liad_to_dewit(
                liad=wiad,
                bin_centres_deg=bin_wood,
                return_scores=True
            )
            wiad_rmse = np.nan if wiad_rmse is None else wiad_rmse
            wiad_l1 = np.nan if wiad_l1 is None else wiad_l1

        if piad.size > 0 and bin_all.size> 0:
            piad_dewit, piad_rmse, piad_l1 = classify_liad_to_dewit(
                liad=piad,
                bin_centres_deg=bin_all,
                return_scores=True
            )
            piad_rmse = np.nan if piad_rmse is None else piad_rmse
            piad_l1 = np.nan if piad_l1 is None else piad_l1
        else:
            piad_dewit = "NA"
            piad_rmse = np.nan
            piad_l1 = np.nan
        
    except Exception as e:
        raise RuntimeError(
            f"Error computing LIAD for voxel at {voxel_center}: {e}"
        ) from e

    # Build ray tensor for all rays in voxel
    rays_FAR6, _, _ = build_ray_tensor_for_voxel(
    voxel_center=voxel_center, 
    voxel_size=voxel_size, 
    angles=angles)

    ### WARP TEST ###
    rc = simulate_voxel_grouped_warp(
        voxel_center=voxel_center,
        voxel_size=voxel_size,
        rays_FAR6=rays_FAR6,
        leaf_mesh=leaf_mesh if leaf_mesh else None,
        wood_mesh=wood_mesh if wood_mesh else None,
        device=_WARP_DEVICE,
        max_hits=10,
    )
    F, A, R = rc["F"], rc["A"], rc["R"]

    
    ## 3) Recompute path-length stats over the slab (CPU vectorized) ----------
    # (reuse your existing vectorized slab lengths, identical to before)
    O = rays_FAR6[..., 0:3].reshape(-1, 3)
    D = rays_FAR6[..., 3:6].reshape(-1, 3)
    vc = np.asarray(voxel_center, dtype=np.float32)
    half = voxel_size / 2.0
    bmin = vc - half
    bmax = vc + half
    t_near, t_far = ray_box_intersection_vectorized(O, D, bmin, bmax)
    t_near = t_near.reshape(F, A, R)
    t_far  = t_far.reshape(F, A, R)
    valid_mask = (t_near <= t_far) & (t_far >= 0.0)

    # Potential path lengths (inside slab)
    path_lengths = np.zeros_like(t_far, dtype=np.float32)
    path_lengths[valid_mask] = (t_far[valid_mask] - t_near[valid_mask]).astype(np.float32)

    # First-hit free path lengths (any/leaf/wood)
    f_any  = rc["first_hit_any"]
    f_leaf = rc["first_hit_leaf"]
    f_wood = rc["first_hit_wood"]

    valid_any  = np.isfinite(f_any)  & valid_mask
    valid_leaf = np.isfinite(f_leaf) & valid_mask
    valid_wood = np.isfinite(f_wood) & valid_mask

    # Free path length = distance from slab entry to first hit; else full slab length
    free_any  = path_lengths.copy()
    free_leaf = path_lengths.copy()
    free_wood = path_lengths.copy()
    free_any [valid_any ] = (f_any [valid_any ]).astype(np.float32)
    free_leaf[valid_leaf] = (f_leaf[valid_leaf]).astype(np.float32)
    free_wood[valid_wood] = (f_wood[valid_wood]).astype(np.float32)

    # Effective free path length (apply your lambda_1 transform)
    eff_any  = np.zeros_like(free_any,  dtype=np.float32)
    eff_leaf = np.zeros_like(free_leaf, dtype=np.float32)
    eff_wood = np.zeros_like(free_wood, dtype=np.float32)
    mask_all = valid_mask

    eff_any [mask_all] = compute_efpl_array(free_any [mask_all],  lambda_1).astype(np.float32)
    eff_leaf[mask_all] = compute_efpl_array(free_leaf[mask_all], lambda_1).astype(np.float32)
    eff_wood[mask_all] = compute_efpl_array(free_wood[mask_all], lambda_1).astype(np.float32)

    # ---------- 4) Aggregate per-(F,A) ----------
    N = valid_mask.sum(axis=2).astype(np.int32)

    n_hits_all  = valid_any .sum(axis=2).astype(np.int32)
    n_hits_leaf = valid_leaf.sum(axis=2).astype(np.int32)
    n_hits_wood = valid_wood.sum(axis=2).astype(np.int32)

    sum_ppl = path_lengths.sum(axis=2)                           # potential
    mean_ppl = np.divide(sum_ppl, N, out=np.zeros_like(sum_ppl), where=(N>0))

    sum_fpl_all  = free_any .sum(axis=2)
    sum_fpl_leaf = free_leaf.sum(axis=2)
    sum_fpl_wood = free_wood.sum(axis=2)

    mean_fpl_all  = np.divide(sum_fpl_all,  N, out=np.zeros_like(sum_fpl_all ), where=(N>0))
    mean_fpl_leaf = np.divide(sum_fpl_leaf, N, out=np.zeros_like(sum_fpl_leaf), where=(N>0))
    mean_fpl_wood = np.divide(sum_fpl_wood, N, out=np.zeros_like(sum_fpl_wood), where=(N>0))

    sum_efpl_all   = eff_any .sum(axis=2)
    sum_efpl_leaf  = eff_leaf.sum(axis=2)
    sum_efpl_wood  = eff_wood.sum(axis=2)
    sum_efpl_hits_all  = (eff_any  * valid_any ).sum(axis=2)
    sum_efpl_hits_leaf = (eff_leaf * valid_leaf).sum(axis=2)
    sum_efpl_hits_wood = (eff_wood * valid_wood).sum(axis=2)

    # Multi-hit statistics from kernel (mean/var over rays)
    counts_all  = rc["counts_all"]
    counts_leaf = rc["counts_leaf"]
    counts_wood = rc["counts_wood"]

    safeN = np.maximum(N, 1)
    masked_all = np.where(valid_mask, counts_all, 0)

    sum_x_all = masked_all.sum(axis=2).astype(np.float64)
    sum_x2_all = (masked_all **2).sum(axis=2).astype(np.float64)
    mean_count_all = sum_x_all / safeN
    var_count_all  = np.zeros_like(mean_count_all, dtype=np.float64)
    has2 = (N > 1)
    var_count_all[has2] = (sum_x2_all[has2] - (sum_x_all[has2]**2)/N[has2]) / (N[has2]-1)

    # Repeat for leaf/wood if you need them; omitted here for brevity
    # mean_count_leaf, var_count_leaf, mean_count_wood, var_count_wood ...

    # pgap (first-hit based)
    pgap_all  = 1.0 - (n_hits_all  / np.maximum(N, 1))
    pgap_leaf = 1.0 - (n_hits_leaf / np.maximum(N, 1))
    pgap_wood = 1.0 - (n_hits_wood / np.maximum(N, 1))

    # ---------- 5) G-function on view zenith (as you had) ----------
    # Use the first ray's direction per (F,A) to define zenith; same as your code
    dirs0 = rays_FAR6[:, :, 0, 3:6]  # (F,A,3)
    norms = np.linalg.norm(dirs0, axis=2, keepdims=True)
    dnorm = dirs0 / np.maximum(norms, 1e-12)
    dz = dnorm[:, :, 2]
    viewing_angles = np.degrees(np.arccos(np.clip(np.abs(dz), 0.0, 1.0))).astype(np.float32)

    G_all = np.zeros_like(viewing_angles, dtype=np.float32)
    G_leaf = np.zeros_like(viewing_angles, dtype=np.float32)
    G_wood = np.zeros_like(viewing_angles, dtype=np.float32)
    if piad.size > 0 and bin_all.size > 0:
        G_all = calculate_G(viewing_angles.ravel(), bin_all, piad).reshape(viewing_angles.shape).astype(np.float32)
    if liad.size > 0 and bin_leaf.size > 0:
        G_leaf = calculate_G(viewing_angles.ravel(), bin_leaf, liad).reshape(viewing_angles.shape).astype(np.float32)
    if wiad.size > 0 and bin_wood.size > 0:
        G_wood = calculate_G(viewing_angles.ravel(), bin_wood, wiad).reshape(viewing_angles.shape).astype(np.float32)

    # ---------- 6) Compute CI (unchanged formulas) ----------
    CI_all  = np.full((F, A), np.nan, dtype=np.float32)
    CI_leaf = np.full((F, A), np.nan, dtype=np.float32)
    CI_wood = np.full((F, A), np.nan, dtype=np.float32)

    if PAD > 0:
        valid_ci = (pgap_all > 0) & (pgap_all < 1) & (G_all > 0) & (mean_ppl > 0)
        CI_all[valid_ci] = (-np.log(pgap_all[valid_ci]) / (G_all[valid_ci] * PAD * mean_ppl[valid_ci])).astype(np.float32)
    if LAD > 0:
        valid_ci = (pgap_leaf > 0) & (pgap_leaf < 1) & (G_leaf > 0) & (mean_ppl > 0)
        CI_leaf[valid_ci] = (-np.log(pgap_leaf[valid_ci]) / (G_leaf[valid_ci] * LAD * mean_ppl[valid_ci])).astype(np.float32)
    if WAD > 0:
        valid_ci = (pgap_wood > 0) & (pgap_wood < 1) & (G_wood > 0) & (mean_ppl > 0)
        CI_wood[valid_ci] = (-np.log(pgap_wood[valid_ci]) / (G_wood[valid_ci] * WAD * mean_ppl[valid_ci])).astype(np.float32)

    # ---------- 7) Assemble metadata + stats (same shape & keys) ----------
    metadata = {
        "voxel_cx": np.float32(voxel_center[0]),
        "voxel_cy": np.float32(voxel_center[1]),
        "voxel_cz": np.float32(voxel_center[2]),
        "voxel_size": voxel_size,
        "alpha": alpha if alpha is not None else np.nan,
        "LAI_ref": LAI if LAI is not None else np.nan,
        "WAI_ref": WAI if WAI is not None else np.nan,
        "PAI_ref": PAI if PAI is not None else np.nan,
        "LAD_ref": LAD if LAD is not None else np.nan,
        "WAD_ref": WAD if WAD is not None else np.nan,
        "PAD_ref": PAD if PAD is not None else np.nan,
        "leaf_fraction": leaf_fraction_ref if leaf_fraction_ref is not None else np.nan,
        "liad_json": json.dumps(liad.tolist()) if liad is not None else "NA",
        "liad_dewit": liad_dewit if liad_dewit is not None else "NA",
        "liad_rmse": liad_rmse if liad_rmse is not None else np.nan,
        "liad_l1":   liad_l1   if liad_l1   is not None else np.nan,
        "wiad_json": json.dumps(wiad.tolist()) if wiad is not None else "NA",
        "wiad_dewit": wiad_dewit if wiad_dewit is not None else "NA",
        "wiad_rmse": wiad_rmse if wiad_rmse is not None else np.nan,
        "wiad_l1":   wiad_l1   if wiad_l1   is not None else np.nan,
        "piad_json": json.dumps(piad.tolist()) if piad is not None else "NA",
        "piad_dewit": piad_dewit if piad_dewit is not None else "NA",
        "piad_rmse": piad_rmse if piad_rmse is not None else np.nan,
        "piad_l1":   piad_l1   if piad_l1   is not None else np.nan
    }

    stats = {
        "dx": dnorm[:, :, 0].astype(np.float32),
        "dy": dnorm[:, :, 1].astype(np.float32),
        "dz": dnorm[:, :, 2].astype(np.float32),
        "zenith_angle": viewing_angles.astype(np.float32),

        "num_rays": N.astype(np.int32),
        "sum_ppl":  sum_ppl.astype(np.float32),
        "mean_ppl": mean_ppl.astype(np.float32),

        "num_hits_all":  n_hits_all.astype(np.int32),
        "num_hits_leaf": n_hits_leaf.astype(np.int32),
        "num_hits_wood": n_hits_wood.astype(np.int32),

        "pgap_all":  pgap_all.astype(np.float32),
        "pgap_leaf": pgap_leaf.astype(np.float32),
        "pgap_wood": pgap_wood.astype(np.float32),

        "G_all":  G_all.astype(np.float32),
        "G_leaf": G_leaf.astype(np.float32),
        "G_wood": G_wood.astype(np.float32),

        "sum_fpl_all":  sum_fpl_all.astype(np.float32),
        "mean_fpl_all": mean_fpl_all.astype(np.float32),
        "sum_fpl_leaf": sum_fpl_leaf.astype(np.float32),
        "mean_fpl_leaf": mean_fpl_leaf.astype(np.float32),
        "sum_fpl_wood": sum_fpl_wood.astype(np.float32),
        "mean_fpl_wood": mean_fpl_wood.astype(np.float32),

        "sum_efpl_all":  sum_efpl_all.astype(np.float32),
        "sum_efpl_hits_all":  sum_efpl_hits_all.astype(np.float32),
        "sum_efpl_leaf": sum_efpl_leaf.astype(np.float32),
        "sum_efpl_hits_leaf": sum_efpl_hits_leaf.astype(np.float32),
        "sum_efpl_wood": sum_efpl_wood.astype(np.float32),
        "sum_efpl_hits_wood": sum_efpl_hits_wood.astype(np.float32),

        # Multi-hit (ANY); add leaf/wood if desired
        "mean_count_all": mean_count_all.astype(np.float32),
        "var_count_all":  var_count_all.astype(np.float32),
    }

    return metadata, stats

    
    ### For each scene in scenes, create a scene and raycast for outputs
    all_results = []

    def simulate_voxel_grouped_with_multi(scene, leaf_gids, wood_gids,
                                        include_all_points=False):
        """
        Vectorized per-(face, angle) aggregation that matches your stats dictionaries.
        
        Inputs:
            - scene: Open3D RaycastingScene with leaf and wood meshes added
            - leaf_gids, wood_gids: geometry IDs for leaf and wood in the scene
            - voxel_center, voxel_size: defines the voxel for ray-box intersection
            - rays_FAR6: pre-generated rays for all faces and angles, shape (F,A,R,6)
            - lambda_1: parameter for EFPL calculation
            - include_all_points: whether to include all points regardless of their classification. 
                                This will include non-specified classes (i.e. understorey non-leaf/wood or ground etc) in the combined results

        Returns:
            out: dict with numpy-only arrays for all the stats, including multi-hit counts and first hit distances for leaf, wood, and any hit.
        """

        import numpy as np
        import gc

        F, A, R, _ = rays_FAR6.shape
        vc = np.asarray(voxel_center, dtype=np.float32)
        bmin = vc - 0.5 * voxel_size
        bmax = vc + 0.5 * voxel_size

        O = rays_FAR6[..., 0:3]  # (F,A,R,3)
        D = rays_FAR6[..., 3:6]  # (F,A,R,3)

        # Compute t_near, t_far for ray-box intersection
        t_near, t_far = ray_box_intersection_vectorized(
            O.reshape(-1, 3), D.reshape(-1, 3), bmin, bmax
        )
        t_near = t_near.reshape(F, A, R)
        t_far = t_far.reshape(F, A, R)
        valid_rays_mask = (t_near <= t_far) & (t_far >= 0.0)  # (F,A,R)

        # --- Run multi-return list_intersections raytracing
        # Use the first_hit_any, first_hit_leaf, and first_hit_wood arrays to calculate pgap for the BL approach to CI
        # Use all returns to use the VoxLAD approach to CI
        counts_leaf_multi = None
        counts_wood_multi = None

        first_hit_any = np.full((F, A, R), np.inf, dtype=np.float32)
        first_hit_leaf = np.full((F, A, R), np.inf, dtype=np.float32)
        first_hit_wood = np.full((F, A, R), np.inf, dtype=np.float32)

        try:
            ans_list = scene.list_intersections(o3c.Tensor(rays_FAR6, dtype=o3c.float32))

            counts_leaf_multi = np.zeros((F, A, R), dtype=np.int32) if leaf_gids is not None else None
            counts_wood_multi = np.zeros((F, A, R), dtype=np.int32) if wood_gids is not None else None
            counts_all_multi = np.zeros((F, A, R), dtype=np.int32) 

            # -------------------------
            # Pattern 1: ragged dict
            # -------------------------
            if isinstance(ans_list, dict) and ("geometry_ids" in ans_list):
                geom_ids = ans_list["geometry_ids"].numpy().astype(np.uint32)  # (K,)
                ray_idx  = ans_list.get("ray_ids", None)
                t_hits   = ans_list.get("t_hit", None)

                if ray_idx is not None:
                    ray_idx = ray_idx.numpy().astype(np.int64)  # (K,)

                    # Map flat ray index -> (f,a,r)
                    Ar = A * R
                    f_idx = (ray_idx // Ar).astype(np.int64)
                    a_idx = ((ray_idx % Ar) // R).astype(np.int64)
                    r_idx = (ray_idx % R).astype(np.int64)

                    # First-hit extraction and multi-hit count
                    if t_hits is not None:
                        t_hits = t_hits.numpy().astype(np.float32)  # (K,)

                        if leaf_gids is not None:
                            m_leaf = np.isin(geom_ids, np.asarray(leaf_gids, dtype=np.uint32))
                            leaf_count = m_leaf.sum()
                            if leaf_count > 0:
                                # single
                                np.minimum.at(first_hit_leaf, (f_idx[m_leaf], a_idx[m_leaf], r_idx[m_leaf]), t_hits[m_leaf])
                                # multi
                                counts = np.bincount(ray_idx[m_leaf], minlength=F*A*R).astype(np.int32)
                                counts_leaf_multi[...] = counts.reshape(F, A, R)

                        if wood_gids is not None:
                            m_wood = np.isin(geom_ids, np.asarray(wood_gids, dtype=np.uint32))
                            wood_count = m_wood.sum()
                            if wood_count > 0:
                                np.minimum.at(first_hit_wood, (f_idx[m_wood], a_idx[m_wood], r_idx[m_wood]), t_hits[m_wood])
                                counts = np.bincount(ray_idx[m_wood], minlength=F*A*R).astype(np.int32)
                                counts_wood_multi[...] = counts.reshape(F, A, R)

                        # Combined first hit of either class:
                        if (leaf_gids is None and wood_gids is None) or include_all_points:
                            # No specified classification, use ALL hits for first_hit_any:
                            np.minimum.at(first_hit_any, (f_idx, a_idx, r_idx), t_hits)
                            counts = np.bincount(ray_idx, minlength=F*A*R).astype(np.int32)
                            counts_all_multi[...] = counts.reshape(F, A, R)
                        else:
                            any_gids = []
                            if leaf_gids is not None:
                                any_gids.extend(np.atleast_1d(np.asarray(leaf_gids, dtype=np.uint32)))
                            if wood_gids is not None:
                                any_gids.extend(np.atleast_1d(np.asarray(wood_gids, dtype=np.uint32)))
                            if any_gids:
                                m_any = np.isin(geom_ids, np.asarray(any_gids, dtype=np.uint32))
                                if m_any.sum() > 0:
                                    np.minimum.at(first_hit_any, (f_idx[m_any], a_idx[m_any], r_idx[m_any]), t_hits[m_any])
                                    counts = np.bincount(ray_idx[m_any], minlength=F*A*R).astype(np.int32)
                                    counts_all_multi[...] = counts.reshape(F, A, R)

                    # DEBUG
                    if DEBUG_MODE:
                        ray_splits = ans_list.get("ray_splits", None)
                        if ray_splits is not None:
                            rs = ray_splits.numpy().astype(np.int64)
                            gt_all = np.diff(rs).reshape(F, A, R)

                            mism = (counts_all_multi != gt_all)
                            if mism.any():
                                bad = np.argwhere(mism)
                                for f, a, r in bad[:5]:
                                    print(f"[DEBUG] mismatch @ (f={f}, a={a}, r-{r}: bincount={counts_all_multi[f,a,r]}, splits={gt_all[f,a,r]})")
                            else:
                                print(f"[DEBUG] bincount matches ray_splits across {gt_all.size} rays.")
                    # cleanup
                    del f_idx, a_idx, r_idx
                    gc.collect()

            # -------------------------
            # Pattern 2: list/tuple per-ray
            # -------------------------
            elif isinstance(ans_list, (list, tuple)) and (len(ans_list) == F * A * R):
                idx = 0
                for f in range(F):
                    for a in range(A):
                        for r in range(R):
                            item = ans_list[idx]
                            idx += 1

                            # Accept a few shapes
                            # Gather arrays (if missing, skip gracefully)
                            if isinstance(item, dict):
                                geom_ids = np.asarray(item.get("geometry_ids", []), dtype=np.uint32)
                                t_arr    = np.asarray(item.get("t_hits", []), dtype=np.float32)
                            else:
                                geom_ids = np.asarray(getattr(item, "geometry_ids", []), dtype=np.uint32)
                                t_arr    = np.asarray(getattr(item, "t_hits", []), dtype=np.float32)
                            
                            # First-hit extraction for this ray
                            if t_arr.size > 0:
                                if leaf_gids is not None:
                                    m_leaf = np.isin(geom_ids, np.asarray(leaf_gids, dtype=np.uint32))
                                    leaf_count = m_leaf.sum()
                                    if leaf_count > 0:
                                        first_hit_leaf[f, a, r] = min(first_hit_leaf[f, a, r], t_arr[m_leaf].min())
                                        counts_leaf_multi[f, a, r] = int(leaf_count)
                                        print(f"single: {first_hit_leaf[f, a, r]}, multi: {leaf_count}")
                                if wood_gids is not None:
                                    m_wood = np.isin(geom_ids, np.asarray(wood_gids, dtype=np.uint32))
                                    wood_count = m_wood.sum()
                                    if wood_count > 0:
                                        first_hit_wood[f, a, r] = min(first_hit_wood[f, a, r], t_arr[m_wood].min())
                                        counts_wood_multi[f, a, r] = int(wood_count)
                                        print(f"single: {first_hit_wood[f, a, r]}, multi: {wood_count}")
                                
                                # Combined first hit of either class:
                                if (leaf_gids is None and wood_gids is None) or include_all_points:
                                    # No specified classification, use ALL hits for first_hit_any:
                                    first_hit_any[f, a, r] = min(first_hit_any[f, a, r], t_arr.min())
                                    counts_all_multi[f, a, r] = int(t_arr.count())
                                else:
                                    m_any = np.isin(geom_ids, np.concatenate([
                                        np.asarray(leaf_gids, dtype=np.uint32) if leaf_gids is not None else np.array([], dtype=np.uint32),
                                        np.asarray(wood_gids, dtype=np.uint32) if wood_gids is not None else np.array([], dtype=np.uint32)
                                    ]))
                                    if m_any.sum() > 0:
                                        first_hit_any[f, a, r] = min(first_hit_any[f, a, r], t_arr[m_any].min())
                                        counts_all_multi[f, a, r] = int(m_any.sum())

                            # Multi-hit class counts for this ray (count ALL hits, not just first)
                            if counts_leaf_multi is not None and leaf_gids is not None and geom_ids.size:
                                counts_leaf_multi[f, a, r] = int(np.isin(geom_ids, leaf_gids).sum())
                            if counts_wood_multi is not None and wood_gids is not None and geom_ids.size:
                                counts_wood_multi[f, a, r] = int(np.isin(geom_ids, wood_gids).sum())

            else:
                # Unknown structure; keep defaults (no per-class counts; first_hits remain inf)
                pass

        except Exception as e:
            print(f"[warn] raycasting failed: -- {e}")
            raise e

        ## --- Shared statistics --- ##
        # Number of rays is the same for first hit and multi hit calculations
        N = valid_rays_mask.sum(axis=2).astype(np.int32)  # (F,A)

        ## --- First Hit Statistics --- ##
        # Valid hits (~np.inf) inside voxel slab (as before)
        valid_leaf_hits_mask = np.isfinite(first_hit_leaf) & valid_rays_mask
        valid_wood_hits_mask = np.isfinite(first_hit_wood) & valid_rays_mask
        valid_all_hits_mask = np.isfinite(first_hit_any) & valid_rays_mask

        # Count hits for each class (for BL approach to CI)
        n_hits_all = valid_all_hits_mask.sum(axis=2)  # (F,A)
        n_hits_leaf = valid_leaf_hits_mask.sum(axis=2)  # (F,A)
        n_hits_wood = valid_wood_hits_mask.sum(axis=2)  # (F,A)

        # Potential path length, free path length, and effective path length
        def mean_var_ddof1(x, N):
            mx = x.sum(axis=2)
            mean = np.divide(mx, N, out=np.zeros_like(mx), where=(N > 0))
            mx2 = (x**2).sum(axis=2)
            numer = mx2 - (mx * mx) / np.maximum(N, 1)
            denom = np.maximum(N - 1, 1)
            var = numer / denom
            var[(N <= 1)] = 0.0
            return mean, var

        # potential path lengths don't change for any first_hit condition
        path_lengths = np.zeros_like(t_far, dtype=np.float32)
        path_lengths[valid_rays_mask] = (t_far[valid_rays_mask] - t_near[valid_rays_mask])

        # free path length is path_length for no hit, otherwise it's the distance to the first hit and is done for each class separately
        free_path_length_all = path_lengths.copy()
        free_path_length_all[valid_all_hits_mask] = first_hit_any[valid_all_hits_mask] - t_near[valid_all_hits_mask]
        free_path_length_leaf = path_lengths.copy()
        free_path_length_leaf[valid_leaf_hits_mask] = first_hit_leaf[valid_leaf_hits_mask] - t_near[valid_leaf_hits_mask]
        free_path_length_wood = path_lengths.copy()
        free_path_length_wood[valid_wood_hits_mask] = first_hit_wood[valid_wood_hits_mask] - t_near[valid_wood_hits_mask]

        # effective free path length is the free path length corrected using lambda_1 (a function of average leaf size and voxel size)
        eff_free_path_length_all = np.zeros_like(free_path_length_all, dtype=np.float32)
        eff_free_path_length_all[valid_rays_mask] = compute_efpl_array(
            free_path_length_all[valid_rays_mask], lambda_1
        ).astype(np.float32)
        eff_free_path_length_leaf = np.zeros_like(free_path_length_leaf, dtype=np.float32)
        eff_free_path_length_leaf[valid_rays_mask] = compute_efpl_array(
            free_path_length_leaf[valid_rays_mask], lambda_1
        ).astype(np.float32)
        eff_free_path_length_wood = np.zeros_like(free_path_length_wood, dtype=np.float32)
        eff_free_path_length_wood[valid_rays_mask] = compute_efpl_array(
            free_path_length_wood[valid_rays_mask], lambda_1
        ).astype(np.float32)

        # Aggregate useful statistics for each path length approach
        sum_path_length = path_lengths.sum(axis=2)
        mean_path_length = np.divide(sum_path_length, N, out=np.zeros_like(sum_path_length), where=(N > 0))

        sum_free_path_length_all = free_path_length_all.sum(axis=2)
        mean_free_path_length_all = np.divide(sum_free_path_length_all, N, out=np.zeros_like(sum_free_path_length_all), where=(N > 0))
        sum_free_path_length_leaf = free_path_length_leaf.sum(axis=2)
        mean_free_path_length_leaf = np.divide(sum_free_path_length_leaf, N, out=np.zeros_like(sum_free_path_length_leaf), where=(N > 0))
        sum_free_path_length_wood = free_path_length_wood.sum(axis=2)
        mean_free_path_length_wood = np.divide(sum_free_path_length_wood, N, out=np.zeros_like(sum_free_path_length_wood), where=(N > 0))

        sum_eff_free_path_length_all = eff_free_path_length_all.sum(axis=2)
        mean_eff_free_path_length_all = np.divide(sum_eff_free_path_length_all, N, out=np.zeros_like(sum_eff_free_path_length_all), where=(N > 0))
        var_eff_free_path_length_all = np.zeros_like(mean_eff_free_path_length_all)
        _, var_eff_free_path_length_all = mean_var_ddof1(eff_free_path_length_all, N)
        sum_eff_free_path_length_leaf = eff_free_path_length_leaf.sum(axis=2)
        mean_eff_free_path_length_leaf = np.divide(sum_eff_free_path_length_leaf, N, out=np.zeros_like(sum_eff_free_path_length_leaf), where=(N > 0))
        var_eff_free_path_length_leaf = np.zeros_like(mean_eff_free_path_length_leaf)
        _, var_eff_free_path_length_leaf = mean_var_ddof1(eff_free_path_length_leaf, N)
        sum_eff_free_path_length_wood = eff_free_path_length_wood.sum(axis=2)
        mean_eff_free_path_length_wood = np.divide(sum_eff_free_path_length_wood, N, out=np.zeros_like(sum_eff_free_path_length_wood), where=(N > 0))
        var_eff_free_path_length_wood = np.zeros_like(mean_eff_free_path_length_wood)
        _, var_eff_free_path_length_wood = mean_var_ddof1(eff_free_path_length_wood, N)
        sum_hits_eff_free_path_length_all = (eff_free_path_length_all * valid_all_hits_mask).sum(axis=2)
        sum_hits_eff_free_path_length_leaf = (eff_free_path_length_leaf * valid_leaf_hits_mask).sum(axis=2)
        sum_hits_eff_free_path_length_wood = (eff_free_path_length_wood * valid_wood_hits_mask).sum(axis=2)

        # Build dx, dy, dz for each ray (for potential later use in angle-based analyses or debugging)
        dirs = D[:, :, 0, :] # (F,A,R,3) Use first ray for directions
        norms = np.linalg.norm(dirs, axis=2, keepdims=True)
        dirs = dirs / np.maximum(norms, 1e-12)
        dx = dirs[:, :, 0].astype(np.float32)
        dy = dirs[:, :, 1].astype(np.float32)
        dz = dirs[:, :, 2].astype(np.float32)
        viewing_angles = np.degrees(np.arccos(np.clip(np.abs(dz), 0.0, 1.0))).astype(np.float32)  # Zenith angle in degrees between 0 and 90

        # Calculate G function values for each class if bins and distributions are available; otherwise, set to None
        if liad is not None and liad.size > 0 and bin_leaf.size > 0:
            G_leaf = calculate_G(viewing_angles.ravel(), bin_leaf, liad).reshape(viewing_angles.shape).astype(np.float32, copy=False)
        if wiad is not None and wiad.size > 0:
            G_wood = calculate_G(viewing_angles.ravel(), bin_wood, wiad).reshape(viewing_angles.shape).astype(np.float32, copy=False)
        if piad is not None and piad.size > 0:
            G_all = calculate_G(viewing_angles.ravel(), bin_all, piad).reshape(viewing_angles.shape).astype(np.float32, copy=False)

        # Calculate first hit pgap
        pgap_all = 1.0 - (n_hits_all / np.maximum(N, 1))
        pgap_leaf = 1.0 - (n_hits_leaf / np.maximum(N, 1))
        pgap_wood = 1.0 - (n_hits_wood / np.maximum(N, 1))

        # Vectorized CI calculation for all (F, A) combinations        # We will add multi-hit statistics to these dictionaries if counts_leaf_multi and counts_wood_multi are available; otherwise, we return only first-hit-based statistics.
        CI_all = np.full((F, A), np.nan, dtype=np.float32)
        CI_leaf = np.full((F, A), np.nan, dtype=np.float32)
        CI_wood = np.full((F, A), np.nan, dtype=np.float32)

        # Compute CI where valid
        if PAD is not None and PAD > 0:
            valid_ci_all = (pgap_all > 0) & (pgap_all < 1) & (G_all > 0) & (mean_path_length > 0)
            CI_all[valid_ci_all] = (-np.log(pgap_all[valid_ci_all]) / (G_all[valid_ci_all] * PAD * mean_path_length[valid_ci_all])).astype(np.float32)

        if LAD is not None and LAD > 0:
            valid_ci_leaf = (pgap_leaf > 0) & (pgap_leaf < 1) & (G_leaf > 0) & (mean_path_length > 0)
            CI_leaf[valid_ci_leaf] = (-np.log(pgap_leaf[valid_ci_leaf]) / (G_leaf[valid_ci_leaf] * LAD * mean_path_length[valid_ci_leaf])).astype(np.float32)

        if WAD is not None and WAD > 0:
            valid_ci_wood = (pgap_wood > 0) & (pgap_wood < 1) & (G_wood > 0) & (mean_path_length > 0)
            CI_wood[valid_ci_wood] = (-np.log(pgap_wood[valid_ci_wood]) / (G_wood[valid_ci_wood] * WAD * mean_path_length[valid_ci_wood])).astype(np.float32)
        # ------------------------------------------------------------
        # Create return arrays for stats
        
        # insert common metrics for leaf, wood, and all conditions
        # stats = dict(

        # )

        # stats_comb = [[None for _ in range(A)] for _ in range(F)]
        # for f in range(F):
        #     for a in range(A):
        #         stats_comb[f][a] = dict(
        #             N=int(N[f, a]),
        #             n_hits=int(n_hits_all[f, a]),
        #             I=(float(n_hits_all[f, a]) / float(N[f, a])) if N[f, a] else 0.0,
        #             mean_path_length=float(mean_path_length[f, a]),
        #             sum_path_length=float(sum_path_length[f, a]),
        #             sum_free_path_length=float(sum_free_path_length_all[f, a]),
        #             mean_free_path_length=float(mean_free_path_length_all[f, a]),
        #             mean_eff_free_path_length=float(mean_eff_free_path_length_all[f, a]),
        #             var_eff_free_path_length=float(var_eff_free_path_length_all[f, a]),
        #             sum_eff_free_path_length=float(np.sum(eff_free_path_length_all[f, a, :])),
        #             sum_hits_eff_free_path_length=float(sum_hits_eff_free_path_length_all[f, a]),
        #         )

        # stats_leaf = [[None for _ in range(A)] for _ in range(F)]
        # for f in range(F):
        #     for a in range(A):
        #         leaf_dict = dict(
        #             N=int(N[f, a]),
        #             n_hits=int(n_hits_leaf[f, a]),
        #             I=(float(n_hits_leaf[f, a]) / float(N[f, a])) if N[f, a] else 0.0,
        #             mean_path_length=float(mean_path_length[f, a]),
        #             sum_path_length=float(sum_path_length[f, a]),
        #             sum_free_path_length=float(sum_free_path_length_leaf[f, a]),
        #             mean_free_path_length=float(mean_free_path_length_leaf[f, a]),
        #             mean_eff_free_path_length=float(mean_eff_free_path_length_leaf[f, a]),
        #             var_eff_free_path_length=float(var_eff_free_path_length_leaf[f, a]),
        #             sum_eff_free_path_length=float(sum_eff_free_path_length_leaf[f, a]),
        #             sum_hits_eff_free_path_length=float(sum_hits_eff_free_path_length_leaf[f, a]),
        #         )
        #         stats_leaf[f][a] = leaf_dict


        # stats_wood = [[None for _ in range(A)] for _ in range(F)]
        # for f in range(F):
        #     for a in range(A):
        #         wood_dict = dict(
        #             N=int(N[f, a]),
        #             n_hits=int(n_hits_wood[f, a]),
        #             I=(float(n_hits_wood[f, a]) / float(N[f, a])) if N[f, a] else 0.0,
        #             mean_path_length=float(mean_path_length[f, a]),
        #             sum_path_length=float(sum_path_length[f, a]),
        #             sum_free_path_length=float(sum_free_path_length_wood[f, a]),
        #             mean_free_path_length=float(mean_free_path_length_wood[f, a]),
        #             mean_eff_free_path_length=float(mean_eff_free_path_length_wood[f, a]),
        #             var_eff_free_path_length=float(var_eff_free_path_length_wood[f, a]),
        #             sum_eff_free_path_length=float(np.sum(eff_free_path_length_wood[f, a, :])),
        #             sum_hits_eff_free_path_length=float(sum_hits_eff_free_path_length_wood[f, a]),
        #             mean_eff_free_path_length_free=float(mean_eff_free_path_length_wood[f, a]),
        #             var_eff_free_path_length_free=float(var_eff_free_path_length_wood[f, a])
        #         )
        #         stats_wood[f][a] = wood_dict

        # ------------------------------------------------------------
        # MULTI-HIT (Option A/B) per (face, angle) — unchanged wiring
        # (Now uses counts_all_multi / counts_leaf_multi / counts_wood_multi if present)
        # ------------------------------------------------------------
        
        ## --- Helper Functions --- ##
        # Combinatorial bases (no factorials in hot loops)
        def _build_pascal_table(n_max: int) -> np.ndarray:
            C = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
            C[0, 0] = 1.0
            for n in range(1, n_max + 1):
                C[n, 0] = 1.0
                for k in range(1, n + 1):
                    C[n, k] = C[n - 1, k - 1] + (C[n - 1, k] if k <= n - 1 else 0.0)
            return C

        # Precompute B[j, n, k] = C(n, k) * (1-ω_j)^(n-k) for all ω_j in the grid and n,k up to n_max.
        def _precompute_B_omegas(omega_grid: np.ndarray, n_max: int, C: np.ndarray) -> np.ndarray:
            O = omega_grid.size
            B = np.zeros((O, n_max + 1, n_max + 1), dtype=np.float64)
            for j, om in enumerate(omega_grid):
                one_minus = 1.0 - om
                powers = one_minus ** np.arange(0, n_max + 1)  # 0..n_max
                for n in range(1, n_max + 1):
                    k_idx = np.arange(1, n + 1)
                    B[j, n, k_idx] = C[n - 1, k_idx - 1] * powers[n - k_idx]
            return B

        # Precompute T[k] = ( (ωλ)^k / k! ) for k=0..n_max for given ol=ωλ; T[0]=0.
        def _compute_T_vec(ol: float, n_max: int) -> np.ndarray:
            """T[k] = ( (ωλ)^k / k! ), k>=1; T[0]=0."""
            T = np.zeros(n_max + 1, dtype=np.float64)
            if n_max >= 1:
                T[1] = ol
                for k in range(2, n_max + 1):
                    T[k] = T[k - 1] * ol / k
            return T

        # PMF table (log-space)
        class LogPMFCache:
            """
            Cache of log P(n | λ_i, ω_j) for n=0..n_max and the entire (λ,ω) grid.
            Reuse this across combined / leaf / wood to avoid recomputation.
            """
            def __init__(self, lam_grid: np.ndarray, omega_grid: np.ndarray, n_max: int):
                self.lam_grid = lam_grid.astype(np.float64)
                self.omega_grid = omega_grid.astype(np.float64)
                self.n_max = int(n_max)
                self.logP_all = None  # shape (n_max+1, G) where G = L*O

                self._build()

            def _build(self):
                L = self.lam_grid.size
                O = self.omega_grid.size
                G = L * O
                C = _build_pascal_table(self.n_max)
                B = _precompute_B_omegas(self.omega_grid, self.n_max, C)

                logP_all = np.empty((self.n_max + 1, G), dtype=np.float64)
                eps = 1e-300
                idx = 0
                for i in range(L):
                    lam = float(self.lam_grid[i])
                    e_neg = np.exp(-lam)
                    for j in range(O):
                        om = float(self.omega_grid[j])
                        ol = om * lam
                        T = _compute_T_vec(ol, self.n_max)
                        ssum = B[j] @ T  # length (n_max+1); ssum[0] = 0
                        p = np.empty(self.n_max + 1, dtype=np.float64)
                        p[0] = e_neg
                        if self.n_max >= 1:
                            p[1:] = e_neg * ssum[1:]
                        logP_all[:, idx] = np.log(np.maximum(p, eps))
                        idx += 1
                self.logP_all = logP_all

        # Histogram builder
        def build_histograms(counts_FAR: np.ndarray,
                            valid_mask_FAR: np.ndarray,
                            n_max: int) -> np.ndarray:
            F, A, R = counts_FAR.shape
            M = F * A
            counts = counts_FAR.reshape(M, R)
            mask = valid_mask_FAR.reshape(M, R)
            H = np.zeros((M, n_max + 1), dtype=np.int64)
            flat_indices = np.arange(M)[:, None] * (n_max + 1) + counts
            flat_indices = flat_indices[mask]
            flat_counts = np.bincount(flat_indices, minlength=M * (n_max + 1))
            H = flat_counts.reshape(M, n_max + 1)
            return H

        # Batched MLE (one matmul + argmax)
        def mle_clustered_batched_np(counts_FAR: np.ndarray,
                                    valid_mask_FAR: np.ndarray,
                                    lam_grid: np.ndarray,
                                    omega_grid: np.ndarray,
                                    precomputed: LogPMFCache | None = None):
            # max observed count across valid rays
            n_max = int(np.max(np.where(valid_mask_FAR, counts_FAR, 0)))
            n_max = max(n_max, 0)

            cache = precomputed
            if (cache is None or
                cache.n_max != n_max or
                cache.lam_grid.shape != lam_grid.shape or
                cache.omega_grid.shape != omega_grid.shape or
                np.any(cache.lam_grid != lam_grid) or
                np.any(cache.omega_grid != omega_grid)):
                cache = LogPMFCache(lam_grid, omega_grid, n_max)

            H = build_histograms(counts_FAR.astype(np.int32), valid_mask_FAR, n_max)     # (M, n_max+1)
            LL_all = H.astype(np.float64).dot(cache.logP_all)                            # (M, G)

            best_idx = np.argmax(LL_all, axis=1)
            L = lam_grid.size
            O = omega_grid.size
            lam_idx = best_idx // O
            omg_idx = best_idx % O

            best_lambda = lam_grid[lam_idx]
            best_omega  = omega_grid[omg_idx]
            best_ll     = LL_all[np.arange(LL_all.shape[0]), best_idx]

            return best_lambda, best_omega, best_ll, cache
        
        # Main function for multi-hit stats computation per (F,A) using the above helpers
        def _compute_multihit_stats(counts_FAR, valid_rays_mask_FA, F, A,
                                    lam_grid=None, omega_grid=None):
            global _PMF_CACHE, _PMF_N_MAX

            if lam_grid is None:
                lam_grid = np.linspace(1e-3, 3.0, 60)
            if omega_grid is None:
                omega_grid = np.linspace(1e-3, 0.999, 60)

            # Basic per-(F,A) stats (vectorized)
            n_rays_multihit = valid_rays_mask_FA.sum(axis=2).astype(np.int32)
            safeN = np.maximum(n_rays_multihit, 1)
            masked = np.where(valid_rays_mask_FA, counts_FAR, 0)

            sum_x  = masked.sum(axis=2).astype(np.float64)
            sum_x2 = (masked**2).sum(axis=2).astype(np.float64)

            mean_count = sum_x / safeN
            var_count = np.zeros_like(mean_count, dtype=np.float64)
            has2 = n_rays_multihit > 1
            var_count[has2] = (sum_x2[has2] - (sum_x[has2]**2) / n_rays_multihit[has2]) / (n_rays_multihit[has2] - 1)

            pgap_multi = np.zeros_like(mean_count, dtype=np.float64)
            nz = n_rays_multihit > 0
            pgap_multi[nz] = (np.where(valid_rays_mask_FA[nz], counts_FAR[nz] == 0, False).sum(axis=1)) / n_rays_multihit[nz]

            lambda_eff = mean_count.copy()

            # ---- Batched MLE for all directions (no loops) ----
            lam_hat_flat, omg_hat_flat, ll_hat_flat, cache = mle_clustered_batched_np(
                counts_FAR, valid_rays_mask_FA, lam_grid, omega_grid, precomputed=_PMF_CACHE
            )
            _PMF_CACHE = cache
            _PMF_N_MAX = max(_PMF_N_MAX, cache.n_max)
            lambda_B = lam_hat_flat.reshape(F, A)
            omega_B  = omg_hat_flat.reshape(F, A)
            loglik_B = ll_hat_flat.reshape(F, A)

            # Assemble result in your existing dict shape
            multi_stats = {
                "pgap_multi": pgap_multi,
                "mean_count": mean_count,
                "var_count": var_count,
                "lambda_eff": lambda_eff,
                "lambda_B": lambda_B,
                "omega_B": omega_B,
                "loglik_B": loglik_B
            }

            return multi_stats

        ## --- Initialise grids --- ##
        lam_grid = _LAM_GRID
        omega_grid = _OMEGA_GRID

        if counts_all_multi is not None:
            multihit_stats_all = _compute_multihit_stats(
                counts_all_multi, valid_rays_mask, F, A, lam_grid, omega_grid
            )

        if counts_leaf_multi is not None:
            multihit_stats_leaf = _compute_multihit_stats(
                counts_leaf_multi, valid_rays_mask, F, A, lam_grid, omega_grid
            )

        if counts_wood_multi is not None:
            multihit_stats_wood = _compute_multihit_stats(
                counts_wood_multi, valid_rays_mask, F, A, lam_grid, omega_grid
            )

        # Build and return numpy-only dict of results
        def _as32(a, dtype):
            x = np.asarray(a, dtype=dtype)
            return np.ascontiguousarray(x)

        FA = F * A

        out = {
            "dx": _as32(dx, np.float32), "dy": _as32(dy, np.float32), "dz": _as32(dz, np.float32),
            "zenith_angle": _as32(viewing_angles, np.float32),
            "num_rays": _as32(N, np.int32),
            "mean_ppl": _as32(mean_path_length, np.float32),
            "sum_ppl": _as32(sum_path_length, np.float32),

            # First-hit stats
            "num_hits_all": _as32(n_hits_all, np.int32),
            "num_hits_leaf": _as32(n_hits_leaf, np.int32),
            "num_hits_wood": _as32(n_hits_wood, np.int32),

            # pgap/G and other gap related metrics
            "pgap_all": _as32(pgap_all, np.float32),
            "pgap_leaf": _as32(pgap_leaf, np.float32),
            "pgap_wood": _as32(pgap_wood, np.float32),      
            "G_all": _as32(G_all, np.float32),
            "G_leaf": _as32(G_leaf, np.float32),
            "G_wood": _as32(G_wood, np.float32),

            # path length related metrics
            "sum_fpl_all": _as32(sum_free_path_length_all, np.float32),
            "mean_fpl_all": _as32(mean_free_path_length_all, np.float32),
            "sum_fpl_leaf": _as32(sum_free_path_length_leaf, np.float32),
            "mean_fpl_leaf": _as32(mean_free_path_length_leaf, np.float32),
            "sum_fpl_wood": _as32(sum_free_path_length_wood, np.float32),
            "mean_fpl_wood": _as32(mean_free_path_length_wood, np.float32),

            "sum_efpl_all": _as32(sum_eff_free_path_length_all, np.float32),
            "sum_efpl_hits_all": _as32(sum_hits_eff_free_path_length_all, np.float32),
            "mean_efpl_all": _as32(mean_eff_free_path_length_all, np.float32),
            "var_efpl_all": _as32(var_eff_free_path_length_all, np.float32),
            "sum_efpl_leaf": _as32(sum_eff_free_path_length_leaf, np.float32),
            "sum_efpl_hits_leaf": _as32(sum_hits_eff_free_path_length_leaf, np.float32),
            "mean_efpl_leaf": _as32(mean_eff_free_path_length_leaf, np.float32),
            "var_efpl_leaf": _as32(var_eff_free_path_length_leaf, np.float32),
            "sum_efpl_wood": _as32(sum_eff_free_path_length_wood, np.float32),
            "sum_efpl_hits_wood": _as32(sum_hits_eff_free_path_length_wood, np.float32),
            "mean_efpl_wood": _as32(mean_eff_free_path_length_wood, np.float32),
            "var_efpl_wood": _as32(var_eff_free_path_length_wood, np.float32)
        }

        def _attach_multihit(suffix, stats_dict):
            if stats_dict is None:
                return
            
            out[f"pgap_multi_{suffix}"] = _as32(stats_dict["pgap_multi"], np.float32)
            out[f"mean_count_{suffix}"] = _as32(stats_dict["mean_count"], np.float32),
            out[f"var_count_{suffix}"] = _as32(stats_dict["var_count"], np.float32),
            out[f"lambda_eff_{suffix}"] = _as32(stats_dict["lambda_eff"], np.float32),
            out[f"lambda_B_{suffix}"] = _as32(stats_dict["lambda_B"], np.float32),
            out[f"Omega_B_{suffix}"] = _as32(stats_dict["Omega_B"], np.float32),
            out[f"loglik_B_{suffix}"] = _as32(stats_dict["loglik_B"], np.float32)

        _attach_multihit("all", multihit_stats_all if 'multi_stats_all' in locals() else None)
        _attach_multihit("leaf", multihit_stats_leaf if 'multi_stats_leaf' in locals() else None)
        _attach_multihit("wood", multihit_stats_wood if 'multi_stats_wood' in locals() else None)

        return out

    try:

        voxel_scene, leaf_gid, wood_gid = build_voxel_scene(o3d_leaf, o3d_wood)
    
        # Use simulate_voxel_grouped for vectorized stats computation
        stats = simulate_voxel_grouped_with_multi(
        voxel_scene, leaf_gid, wood_gid)

        # Generate metadata
        metadata = {
            "voxel_cx": voxel_center[0].astype(np.float32),
            "voxel_cy": voxel_center[1].astype(np.float32),
            "voxel_cz": voxel_center[2].astype(np.float32),
            "voxel_size": voxel_size,
            "alpha": alpha if alpha is not None else np.nan,
            "LAI_ref": LAI if LAI is not None else np.nan,
            "WAI_ref": WAI if WAI is not None else np.nan,
            "PAI_ref": PAI if PAI is not None else np.nan,
            "LAD_ref": LAD if LAD is not None else np.nan,
            "WAD_ref": WAD if WAD is not None else np.nan,
            "PAD_ref": PAD if PAD is not None else np.nan,
            "leaf_fraction": leaf_fraction_ref if leaf_fraction_ref is not None else np.nan,
            "liad_json": json.dumps(liad.tolist()) if liad is not None else "NA",
            "liad_dewit": liad_dewit if liad_dewit is not None else "NA",
            "liad_rmse": liad_rmse if liad_rmse is not None else np.nan,
            "liad_l1": liad_l1 if liad_l1 is not None else np.nan,
            "wiad_json": json.dumps(wiad.tolist()) if wiad is not None else "NA",
            "wiad_dewit": wiad_dewit if wiad_dewit is not None else "NA",
            "wiad_rmse": wiad_rmse if wiad_rmse is not None else np.nan,
            "wiad_l1": wiad_l1 if wiad_l1 is not None else np.nan,
            "piad_json": json.dumps(piad.tolist()) if piad is not None else "NA",
            "piad_dewit": piad_dewit if piad_dewit is not None else "NA",
            "piad_rmse": piad_rmse if piad_rmse is not None else np.nan,
            "piad_l1": piad_l1 if piad_l1 is not None else np.nan
        }
    
    except Exception as e:
        raise RuntimeError(
            f"Error processing voxel at {voxel_center} with size {voxel_size}: {e}"
        ) from e

    return metadata, stats

# NEW Clipping parallel logic

# ---- Global cache for threading-based clipping ----
_CLIP_GLOBALS = {
    'leaf_mesh': None,
    'wood_mesh': None,
    'leaf_tri_min': None,
    'leaf_tri_max': None,
    'wood_tri_min': None,
    'wood_tri_max': None,
}

def set_clip_globals(leaf_mesh, wood_mesh):
    """
    Populate module-level globals once in the parent process.
    Threads will read from these; no pickling or joblib.Memory in workers.
    """
    g = _CLIP_GLOBALS
    g['leaf_mesh'] = leaf_mesh
    g['wood_mesh'] = wood_mesh

    # Precompute triangle AABB mins/maxs once
    if leaf_mesh is not None and not leaf_mesh.is_empty:
        tris = leaf_mesh.triangles  # (N, 3, 3)
        g['leaf_tri_min'] = tris.min(axis=1)
        g['leaf_tri_max'] = tris.max(axis=1)
    else:
        g['leaf_tri_min'] = None
        g['leaf_tri_max'] = None

    if wood_mesh is not None and not wood_mesh.is_empty:
        tris = wood_mesh.triangles
        g['wood_tri_min'] = tris.min(axis=1)
        g['wood_tri_max'] = tris.max(axis=1)
    else:
        g['wood_tri_min'] = None
        g['wood_tri_max'] = None

def clip_one_thread(voxel_center, voxel_size):
    """
    Thread worker: clip both leaf and wood using the shared globals.
    Returns (center, leaf_vertices, leaf_faces, wood_vertices, wood_faces).
    """
    g = _CLIP_GLOBALS
    leaf_v, leaf_f = _clip_one_mesh_with_aabb(
        g['leaf_mesh'], g['leaf_tri_min'], g['leaf_tri_max'],
        voxel_center, voxel_size
    )
    wood_v, wood_f = _clip_one_mesh_with_aabb(
        g['wood_mesh'], g['wood_tri_min'], g['wood_tri_max'],
        voxel_center, voxel_size
    )
    return (voxel_center, leaf_v, leaf_f, wood_v, wood_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process voxel batch.")
    parser.add_argument("scene_file", type=str, help="Path to the single .obj scene file. This will automatically extract leaf and wood meshes.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0], help="Voxel sizes for processing (default: [0.2, 0.5, 1.0, 2.0]).")
    parser.add_argument("--num_angle_bins", type=int, default=18, help="Number of angle bins for ray tracing (default: 18).")
    parser.add_argument("--ray_spacing", type=float, default=0.005, help="Ray spacing for ray tracing (default: 0.005).")
    parser.add_argument("--wood_volume_voxel_size", type=float, default=0.01, help="Voxel size for wood volume calculation (default: 0.01).")
    parser.add_argument("--wood_volume_threshold", type=int, default=4, help="Threshold for wood volume calculation (default: 4).")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel workers (default: 32) will max to num_cpus.")
    parser.add_argument("--debug", action='store_true', help="If set, debug outputs will be saved.")
    args = parser.parse_args()

    # Print a nice statement outlining chosen inputs
    print(f"Processing scene file: {args.scene_file}")
    print(f"Voxel sizes: {args.voxel_sizes}")
    print(f"Number of Angle Bins: {args.num_angle_bins}")
    print(f"Ray spacing: {args.ray_spacing}")
    print(f"Wood volume voxel size: {args.wood_volume_voxel_size}")
    print(f"Wood volume threshold: {args.wood_volume_threshold}")
    print(f"Max workers: {args.max_workers}")
    print(f"Debug mode: {args.debug}")

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


    # Check for CUDA device availability with Open3D
    cuda_id = None
    preferred_uuid = os.getenv("OPEN3D_GPU_UUID", None)
    cuda_id, selected_uuid = resolve_cuda_index(preferred_uuid)

    if cuda_id is not None:
        print(f"Using CUDA logical device {cuda_id}" + (f" (UUID: {selected_uuid})" if selected_uuid else ""))
    else:
        print("No usable CUDA device found; will try SYCL/CPU")

    try:
        _DEV = o3d.core.Device(f"CUDA:{cuda_id}")
    except Exception as e:
        _DEV = o3d.core.Device("CPU:0")

    # Store a global setting for leaf-off
    global DEBUG_MODE
    global DEBUG_PATH
    DEBUG_MODE = args.debug
    if DEBUG_MODE:
        DEBUG_PATH = os.path.join(os.path.dirname(args.scene_file), "debug")
        if not os.path.exists(DEBUG_PATH):
            os.makedirs(DEBUG_PATH)

    voxel_sizes = args.voxel_sizes
    ray_spacing = args.ray_spacing
    
    log_file = os.path.basename(args.scene_file).replace('.obj', '.log')
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

    # Load the scene file and extract leaf and wood meshes
    leaf_keys = ["leaf", "leaves", "leafs"]
    wood_keys = ["wood", "trunk", "branch", "stem"]
    leaf_mesh_file, wood_mesh_file, bounds, leaf_mesh, wood_mesh = load_and_split_by_group(
        args.scene_file, leaf_keys, wood_keys
    )

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
    cached_leaf_mesh = memory.cache(load_mesh_trimesh)(leaf_mesh_file)
    cached_wood_mesh = memory.cache(load_mesh_trimesh)(wood_mesh_file)
    set_clip_globals(cached_leaf_mesh, cached_wood_mesh)

    
    # Warm up the clipping function
    _ = _clip_one_mesh_with_aabb(_CLIP_GLOBALS['leaf_mesh'],
                             _CLIP_GLOBALS['leaf_tri_min'],
                             _CLIP_GLOBALS['leaf_tri_max'],
                             voxel_center=np.asarray([0,0,0], dtype=float),
                             voxel_size=voxel_sizes[0])


    wood_volume_file = os.path.join(os.path.dirname(args.scene_file), os.path.basename(args.scene_file).replace(".obj", f"_inside_voxels_size{args.wood_volume_voxel_size}_thresh{args.wood_volume_threshold}.txt"))
    if not os.path.exists(wood_volume_file):
        print(f"Wood volume file {wood_volume_file} does not exist. Generating wood volume data.")
        process_wood_volume_file(
            scene_file=args.scene_file,
            wood_mesh=wood_mesh,
            wood_voxel_size=args.wood_volume_voxel_size,
            threshold=args.wood_volume_threshold
        )

    leaf_area_csv = os.path.join(os.path.dirname(args.scene_file), os.path.basename(args.scene_file).replace(".obj", "_leaf_area.csv"))
    if not os.path.exists(leaf_area_csv):
        print(f"Leaf area CSV {leaf_area_csv} does not exist. Generating leaf area data.")
        avg_leaf_area, min_leaf_area, max_leaf_area, num_leaves, total_leaf_area = process_leaf_area_file(
            scene_file=args.scene_file,
            leaf_mesh=leaf_mesh
        )
        print(f"Average leaf area: {avg_leaf_area}, Min leaf area: {min_leaf_area}, Max leaf area: {max_leaf_area}")
    else:
        print(f"Leaf area CSV {leaf_area_csv} already exists. Skipping leaf area calculation.")
        df = pd.read_csv(leaf_area_csv)
        avg_leaf_area = df['avg_leaf_area'][0]
        min_leaf_area = df['min_leaf_area'][0]
        max_leaf_area = df['max_leaf_area'][0]
        num_leaves = df['num_leaves'][0]
        total_leaf_area = df['total_leaf_area'][0]

    # Build canonical grids for all voxel sizes
    _build_canonical_grids(voxel_sizes, ray_spacing)

    for voxel_size in voxel_sizes:
        # Batch voxel centers into number of CPUs
        voxel_centers, voxel_ids = generate_voxel_centers(
            voxel_size=voxel_size,  # Example voxel size, adjust as needed
            bounds=bounds  # Use the bounds from the loaded meshes
        )

        total_voxels = len(voxel_centers)

        # Batch the voxel centers into groups based on the number of CPUs
        batches = []
        voxel_center_batches = [voxel_centers[i:i + num_cpus] for i in range(0, len(voxel_centers), num_cpus)]

        lambda_1 = avg_leaf_area / (voxel_size ** 3)



        def process_voxel_wrapper(voxel_center, leaf_mesh, wood_mesh, voxel_size, angles, wood_volume_points, lambda_1):
            try:
                args = (voxel_center, voxel_size, leaf_mesh, wood_mesh, angles, wood_volume_points, lambda_1)
                metadata, stats = process_voxel(*args)
                # print(f" Processed voxel {args[0]} successfully.")
                return metadata, stats
            except Exception as e:
                print(f"Error processing voxel {args[0]}: {e}")
                traceback.print_exc()
                return {}, {}
            
        def clip_mesh_wrapper(
                voxel_centers, 
                voxel_size,
                leaf_mesh_file,
                wood_mesh_file,
            ):
            
            _ensure_clip_worker_meshes(leaf_mesh_file, wood_mesh_file)

            out = []
            for c in voxel_centers:
                leaf_v, leaf_f = _clip_one_mesh_with_aabb(_LEAF_MESH, _LEAF_TRI_MIN, _LEAF_TRI_MAX, c, voxel_size)
                wood_v, wood_f = _clip_one_mesh_with_aabb(_WOOD_MESH, _WOOD_TRI_MIN, _WOOD_TRI_MAX, c, voxel_size)
                out.append((c, leaf_v, leaf_f, wood_v, wood_f))

            
            return out

        # For each voxel_center batch, process the voxels in parallel
        start = dt.datetime.now()

        # Filter out voxel centers that are outside the bounds of leaf and wood meshes
        leaf_bounds = leaf_mesh.bounds.flatten() if leaf_mesh is not None else (0, 0, 0, 0, 0, 0)
        wood_bounds = wood_mesh.bounds.flatten() if wood_mesh is not None else (0, 0, 0, 0, 0, 0)
        voxel_centers = filter_voxel_centers(
            voxel_centers=voxel_centers,
            leaf_bounds=leaf_bounds,
            wood_bounds=wood_bounds,
            voxel_size=voxel_size
        )


        pbar = tqdm(total=len(voxel_centers // 2), desc="Clipping meshes", unit="voxels") 
        with tqdm_joblib(pbar):
            results = Parallel(n_jobs=n_workers // 2, backend='threading')(
                    delayed(clip_one_thread)(voxel_center, voxel_size) for voxel_center in voxel_centers
                )
            pbar.close()

        # results = [item for sublist in results for item in sublist]  # Flatten list of lists
        clipped_voxel_centres, clipped_leaf_vertices, clipped_leaf_faces, clipped_wood_vertices, clipped_wood_faces = zip(*results)
        valid_indices = [i for i, (_, lv, lf, wv, wf) in enumerate(results) if lv.shape[0] != 0 or lf.shape[0] != 0 or wv.shape[0] != 0 or wf.shape[0] != 0]

        clipped_voxel_centres = [clipped_voxel_centres[i] for i in valid_indices]
        clipped_leaf_vertices = [clipped_leaf_vertices[i] for i in valid_indices]
        clipped_leaf_faces = [clipped_leaf_faces[i] for i in valid_indices]
        clipped_wood_vertices = [clipped_wood_vertices[i] for i in valid_indices]
        clipped_wood_faces = [clipped_wood_faces[i] for i in valid_indices]

        if len(clipped_leaf_vertices) == 0 and len(clipped_wood_vertices) == 0:
            print(f"No valid clipped meshes found for voxel size {voxel_size}. Skipping processing.")
            continue 

        # Convert clipped vertices and faces to Open3D meshes
        clipped_leaf_meshes = []
        clipped_wood_meshes = []
        for vc, leaf_vertices, leaf_faces in zip(clipped_voxel_centres, clipped_leaf_vertices, clipped_leaf_faces):
            
            if leaf_vertices is not None and leaf_faces is not None:
                clipped_leaf_meshes.append(
                    {
                        "vertices": leaf_vertices,
                        "faces": leaf_faces
                    }
                )
                # o3d_leaf_mesh = o3d.geometry.TriangleMesh()
                # o3d_leaf_mesh.vertices = o3d.utility.Vector3dVector(leaf_vertices)
                # o3d_leaf_mesh.triangles = o3d.utility.Vector3iVector(leaf_faces)
                # clipped_leaf_meshes.append(o3d_leaf_mesh)

                ## DEBUG ##
                # if DEBUG_MODE:
                #     debug_dir = os.path.join(DEBUG_PATH, f"voxel_size={voxel_size}", f"voxel_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}")
                #     os.makedirs(debug_dir, exist_ok=True)

                #     test_mesh_path = os.path.join(debug_dir, f"leaf_mesh_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}.ply")
                #     o3d.io.write_triangle_mesh(test_mesh_path, o3d_leaf_mesh)
            else:
                clipped_leaf_meshes.append({})

        for vc, wood_vertices, wood_faces in zip(clipped_voxel_centres, clipped_wood_vertices, clipped_wood_faces):
            if wood_vertices is not None and wood_faces is not None:
                clipped_wood_meshes.append({
                    "vertices": wood_vertices,
                    "faces": wood_faces
                })
                # o3d_wood_mesh = o3d.geometry.TriangleMesh()
                # o3d_wood_mesh.vertices = o3d.utility.Vector3dVector(wood_vertices)
                # o3d_wood_mesh.triangles = o3d.utility.Vector3iVector(wood_faces)
                # clipped_wood_meshes.append(o3d_wood_mesh)

                ## DEBUG ##
                # if DEBUG_MODE:
                #     debug_dir = os.path.join(DEBUG_PATH, f"voxel_size={voxel_size}", f"voxel_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}")
                #     os.makedirs(debug_dir, exist_ok=True)
                #     test_mesh_path = os.path.join(debug_dir, f"wood_mesh_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}.ply")
                #     o3d.io.write_triangle_mesh(test_mesh_path, o3d_wood_mesh)
            else:
                clipped_wood_meshes.append({})

        clip_time = dt.datetime.now() - start

        grid = _grid(voxel_size, ray_spacing)

        wood_volume_points = memory.cache(load_wood_volume_file)(wood_volume_file)

        worker = partial(
            process_voxel_wrapper,
            voxel_size=voxel_size,
            angles=angles,
            wood_volume_points=wood_volume_points,
            lambda_1=lambda_1
        )

        
        start02 = dt.datetime.now()
        print(f"Preprocessing time: {clip_time}")

        # Run sequentially if the _DEV is CPU, otherwise run in parallel to maximise open3d's raytracing
        if _DEV.get_type() == o3d.core.Device.DeviceType.CPU:
            print("Running sequentially on CPU...")
            pbar = tqdm(total=len(clipped_voxel_centres), desc="Processing voxels", unit="voxel")
            voxel_results = []
            for vc, lm, wm in zip(clipped_voxel_centres, clipped_leaf_meshes, clipped_wood_meshes):
                metadata, stats = worker(vc, lm, wm)
                voxel_results.append((metadata, stats))
                pbar.update(1)
            pbar.close()
        else:
            print(f"Running in parallel with {n_workers} workers on {_DEV}...")
            pbar = tqdm(total=len(clipped_voxel_centres), desc="Processing voxels", unit="voxel")
            with tqdm_joblib(pbar):
                voxel_results = Parallel(n_jobs=n_workers, backend='threading')(
                    delayed(worker)(vc, lm, wm) for vc, lm, wm in zip(clipped_voxel_centres, clipped_leaf_meshes, clipped_wood_meshes)
                )
            pbar.close()

        # Efficiently concatenate all metadata + stats into a single DataFrame
        # Each voxel_result is (metadata_dict, stats_dict)
        # metadata: single values per voxel
        # stats: arrays of shape (F, A) per voxel
        
        rows = []
        for metadata, stats in voxel_results:
            if not metadata or not stats:
                continue

            # Infer from first (F, A) shaped array
            for v in stats.values():
                if isinstance(v, np.ndarray) and v.ndim == 2:
                    F, A = v.shape
                    break
            
            if F is None or A is None:
                continue  # Skip if we can't determine shape
            
            # Flatten all stats arrays from (F, A) -> (F*A,)
            FA = F * A
            flat_stats = {}
            for key, val in stats.items():
                if isinstance(val, np.ndarray):
                    flat_stats[key] = val.reshape(FA)
                else:
                    flat_stats[key] = val
            
            # Build a single row per voxel with metadata
            base_row = metadata.copy()

            # Broadcast metadata across all F*A rows
            for i in range(FA):
                row = base_row.copy()
                for key, val in flat_stats.items():
                    row[key] = val[i]
                rows.append(row)
        
        df = pd.DataFrame(rows) if rows else pd.DataFrame()
        
        # Reorder columns: metadata first, then stats
        if not df.empty:
            metadata_cols = list(metadata.keys())
            stats_cols = [col for col in df.columns if col not in metadata_cols]
            df = df[metadata_cols + stats_cols]
        
        total_time = dt.datetime.now() - start
        raytrace_time = dt.datetime.now() - start02

        output_basename = os.path.basename(args.scene_file).replace('.obj', f'_results_{voxel_size}_{dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.csv')
        output_path = os.path.join(os.path.dirname(args.scene_file), output_basename)
        df.to_csv(output_path, index=False)
        print(f"Saved {len(df)} rows to {output_path}.")

        ### DEBUG TOTAL LEAF AREA ###
        # if DEBUG_MODE:
        #     # Compute total leaf area only for unique voxel centers
        #     unique_voxels = df.drop_duplicates(subset=['voxel_cx', 'voxel_cy', 'voxel_cz'])
        #     total_leaf_area = unique_voxels['lai'].sum() * (voxel_size ** 2)
        #     total_wood_area = unique_voxels['wai'].sum() * (voxel_size ** 2)
        #     leaf_area_test_path = os.path.join(DEBUG_PATH, os.path.basename(args.scene_file).replace('.obj', f'_leaf_area_test.csv'))
        #     debug_df = pd.DataFrame([{
        #         "voxel_size": voxel_size,
        #         "measured_leaf_area": total_leaf_area
        #     }])
            
        #     if os.path.exists(leaf_area_test_path):
        #         df_exist = pd.read_csv(leaf_area_test_path)
        #         df_exist = pd.concat([df_exist, debug_df], ignore_index=True)
        #         df_exist.to_csv(leaf_area_test_path, index=False)
        #     else:
        #         debug_df.to_csv(leaf_area_test_path, index=False)

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
        perf_output_basename = os.path.basename(args.scene_file).replace('.obj', f'_performance_{voxel_size}.csv')
        perf_output_path = os.path.join(os.path.dirname(args.scene_file), perf_output_basename)
        perf_df.to_csv(perf_output_path, index=False)

        print(f"Processed {args.scene_file} and saved results to {output_path} in {total_time} seconds.")

    


    

