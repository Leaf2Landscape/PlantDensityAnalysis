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

def calculate_wood_volume(wood_mesh: trimesh.Trimesh, voxel_size: float=0.01, threshold: int=4, cache_size=10000) -> str:
    """
    Calculate the wood volume of a mesh by voxelizing it.
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
    print (f"Bounding box of wood mesh: {aabb}")
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

    # Prepare rays
    directions = np.array([
        [1.0001, 0.0001, 0.0001],
            [-1.0001, 0.0001, 0.0001],
            [0.0001, 1.0001, 0.0001],
            [0.0001, -1.0001, 0.0001],
            [0.0001, 0.0001, 1.0001],
            [0.0001, 0.0001, -1.0001]
        ], dtype=np.float32)
    
    inside_points = []

    for xi, x_val in enumerate(tqdm(x, desc="Processing X-slices")):
        slice_points = []
        slice_indices = []

        for yi, y_val in enumerate(y):
            for zi, z_val in enumerate(z):
                point = np.array([x_val, y_val, z_val], dtype=np.float32)
                slice_points.append(point)
                slice_indices.append((xi, yi, zi))

        for i in range(0, len(slice_points), cache_size):
            end_idx = min(i + cache_size, len(slice_points))
            chunk_points = slice_points[i:end_idx]
                
            # Check each direction for each point
            for direction in directions:
                # Create rays for all points in this direction
                rays = []
                for point in chunk_points:
                    ray = np.concatenate([point, direction])
                    rays.append(ray)
                
                # Convert to tensor for batch processing
                rays_tensor = o3d.core.Tensor(np.array(rays), dtype=o3d.core.Dtype.Float32)
                
                # Get intersection counts for all rays at once
                intersection_counts = scene.count_intersections(rays_tensor).numpy()
                
                # Update inside counts for each point
                if 'inside_counts' not in locals():
                    inside_counts = np.zeros(len(chunk_points), dtype=np.int32)
                
                # Add odd intersections (indicates inside)
                inside_counts += (intersection_counts % 2).astype(np.int32)
            
            # Identify inside points based on threshold
            for j in range(len(chunk_points)):
                if inside_counts[j] >= threshold:
                    inside_points.append(chunk_points[j])
            
            # Reset for next batch
            if 'inside_counts' in locals():
                del inside_counts
        
        # Print progress every 10 x-slices
        if xi % 10 == 0 and xi > 0:
            elapsed = dt.datetime.now() - start_time
            percent_complete = (xi + 1) / len(x) * 100
            est_total_time = elapsed / percent_complete * 100
            remaining = (est_total_time - elapsed).total_seconds()
            
            print(f"Progress: {percent_complete:.1f}% complete, " +
                    f"ETA: {remaining/60:.1f} minutes, " + 
                    f"Found {len(inside_points)} inside points so far")
    
    # Convert to numpy array
    inside_points = np.array(inside_points)
    
    if len(inside_points) == 0:
        print(f"No inside points found. Try a different threshold or check mesh.")
        return
    
    return inside_points

def process_wood_volume_file(scene_file: str, wood_mesh: str, wood_voxel_size: float=0.01, threshold: int=4) -> Optional[np.ndarray]:
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
def load_wood_volume_file(wood_volume_file: str, wood_voxel_size: float=0.01, threshold: int=4) -> Optional[np.ndarray]:
    """
    Load the wood volume file if it exists.
    The file is expected to be in the same directory as the scene file.
    """
    if os.path.exists(wood_volume_file):
        try:
            return np.loadtxt(wood_volume_file)
        except Exception as e:
            print(f"Error loading wood volume file {wood_volume_file}: {e}")
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
    except Exception as e:
        print(f"Error loading mesh from {file_path}: {e}")
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
        clipped = clipped.extract_geometry()
    if not clipped.is_all_triangles:
        clipped = clipped.triangulate()

    vertices = np.asarray(clipped.points)
    faces = np.asarray(clipped.faces.reshape((-1, 4))[:, 1:])
    return (vertices, faces)

    
def _faces_to_poly(vertices, faces):
    """Convert trimesh faces to a format compatible with PyVista.
    """
    if not faces:
        return None
    faces_arr = np.asarray(faces, dtype=np.int64)
    n_per_face = np.full((faces_arr.shape[0], 1), faces_arr.shape[1], np.int64)
    faces_flat = np.hstack([n_per_face, faces_arr])
    return pv.PolyData(vertices, faces_flat)

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

    _load_leaf = True if o3d_leaf is not None and len(o3d_leaf.triangles) > 0 and LEAF_OFF is False else False
    _load_wood = True if o3d_wood is not None and len(o3d_wood.triangles) > 0 else False
    leaf_id = wood_id = None
    
    try:
        dev = o3d.core.Device(f"SYCL:{_CUDA_DEVICE_ID}" if _CUDA_DEVICE_ID is not None else "SYCL:0")
        scene = o3d.t.geometry.RaycastingScene(device=dev)

        if _load_leaf:
            leaves = o3d.t.geometry.TriangleMesh.from_legacy(o3d_leaf)
            leaves = leaves.to(dev)
            leaf_id = scene.add_triangles(leaves)
        
        if _load_wood:
            wood = o3d.t.geometry.TriangleMesh.from_legacy(o3d_wood)
            wood = wood.to(dev)
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
    normals = np.divide(cross_prod, norms, where=(norms != 0))
    angle_facets = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
    angle_facets = np.where(angle_facets > 90, 180 - angle_facets, angle_facets)
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

def simulate_combined_mesh_with_points(scene, leaf_gid, wood_gid,
                                       voxel_center, voxel_size,
                                       ray_origins, ray_dirs, lambda_1):
    hit_details = []
    voxel_center = np.asarray(voxel_center, dtype=np.float32)
    bmin = voxel_center - 0.5 * voxel_size
    bmax = voxel_center + 0.5 * voxel_size
    t_near, t_far = ray_box_intersection_vectorized(
        ray_origins, ray_dirs, bmin, bmax)
    valid_rays_mask = (t_near <= t_far) & (t_far >= 0)
    N = int(valid_rays_mask.sum())
    if N == 0:
        return {k: 0.0 for k in (
            "N","n_hits","I","delta_bar","sum_delta","sum_z","mean_z",
            "mean_delta_e","var_delta_e","sum_z_e","sum_hits_z_e",
            "mean_efpl_free","var_efpl_free"
        )}, hit_details

    rays_np = np.hstack([ray_origins, ray_dirs]).astype(np.float32)
    # print(f"Simulating {N} rays for voxel at {voxel_center} with size {voxel_size}.")
    hits = scene.cast_rays(o3c.Tensor(rays_np, dtype=o3c.float32))
    # print(f"Ray casting completed for voxel at {voxel_center}.")
    gids = hits["geometry_ids"].numpy()
    dists = hits["t_hit"].numpy()

    # Mask infinite t_hits
    valid_hits_mask = np.isfinite(dists) & (dists >= t_near) & (dists <= t_far) & (dists >= 0) & valid_rays_mask

    dist_leaf = np.full_like(dists, np.inf)
    dist_wood = np.full_like(dists, np.inf)
    if leaf_gid is not None:
        m = (gids == leaf_gid) & valid_hits_mask
        dist_leaf[m] = dists[m]
    if wood_gid is not None:
        m = (gids == wood_gid) & valid_hits_mask
        dist_wood[m] = dists[m]

    comb_dist = np.minimum(dist_leaf, dist_wood)
    hit_any = np.isfinite(comb_dist) # & (comb_dist < np.inf)
    n_hits_lw = int(hit_any.sum())
    hit_leaf = np.isfinite(dist_leaf)
    n_hits_leaf = int(hit_leaf.sum())

    delta = np.zeros_like(dists)
    delta[valid_rays_mask] = t_far[valid_rays_mask] - t_near[valid_rays_mask]
    z_arr = delta.copy()
    z_arr[hit_any] = comb_dist[hit_any] - t_near[hit_any]

    efpl_d = compute_efpl_array(delta[valid_rays_mask], lambda_1)
    efpl_f = compute_efpl_array(z_arr[valid_rays_mask],  lambda_1)

    stats_lw = dict(
        N=N,
        n_hits=n_hits_lw,
        I=n_hits_lw / N if N else 0.0,
        delta_bar=delta[valid_rays_mask].mean() if N else 0.0,
        sum_delta=delta[valid_rays_mask].sum(),
        sum_z=z_arr[valid_rays_mask].sum(),
        mean_z=z_arr[valid_rays_mask].mean() if N else 0.0,
        mean_delta_e=efpl_d.mean() if N else 0.0,
        var_delta_e=efpl_d.var(ddof=1) if N > 1 else 0.0,
        sum_z_e=efpl_f.sum(),
        sum_hits_z_e=efpl_f[hit_any[valid_rays_mask]].sum() if n_hits_lw else 0.0,
        mean_efpl_free=efpl_f.mean() if N else 0.0,
        var_efpl_free=efpl_f.var(ddof=1) if N > 1 else 0.0,
    )

    delta = np.zeros_like(dists)
    delta[valid_rays_mask] = t_far[valid_rays_mask] - t_near[valid_rays_mask]
    z_arr = delta.copy()
    z_arr[hit_leaf] = dists[hit_leaf] - t_near[hit_leaf]

    efpl_d = compute_efpl_array(delta[valid_rays_mask], lambda_1)
    efpl_f = compute_efpl_array(z_arr[valid_rays_mask],  lambda_1)

    stats_leaf = dict(
        N=N,
        n_hits=n_hits_leaf,
        I=n_hits_leaf / N if N else 0.0,
        delta_bar=delta[valid_rays_mask].mean() if N else 0.0,
        sum_delta=delta[valid_rays_mask].sum(),
        sum_z=z_arr[valid_rays_mask].sum(),
        mean_z=z_arr[valid_rays_mask].mean() if N else 0.0,
        mean_delta_e=efpl_d.mean() if N else 0.0,
        var_delta_e=efpl_d.var(ddof=1) if N > 1 else 0.0,
        sum_z_e=efpl_f.sum(),
        sum_hits_z_e=efpl_f[hit_leaf[valid_rays_mask]].sum() if n_hits_leaf else 0.0,
        mean_efpl_free=efpl_f.mean() if N else 0.0,
        var_efpl_free=efpl_f.var(ddof=1) if N > 1 else 0.0,
    )
    return stats_lw, stats_leaf, hit_details


def simulate_voxel_grouped(scene, leaf_gids, wood_gids,
                           voxel_center, voxel_size,
                           rays_FAR6,  # shape (Faces,Angles,Rays,6), float32
                           lambda_1,
                           scene_name="scene"):
    """
    Vectorized per-(face, angle) aggregation that matches your stats dictionaries.
    Returns:
      stats_lw[F][A], stats_leaf[F][A], each a dict like your original.
    """
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
    t_far  = t_far.reshape(F, A, R)
    valid_rays_mask = (t_near <= t_far) & (t_far >= 0.0)  # (F,A,R)

    # Single cast_rays call
    hits = scene.cast_rays(o3c.Tensor(rays_FAR6, dtype=o3c.float32))

    # Each field returns with leading dims (F,A,R)
    gids   = hits["geometry_ids"].numpy()      # (F,A,R), uint32 or int64
    dists  = hits["t_hit"].numpy()             # (F,A,R), float32; inf for miss

    # If debugging, save xyz to ASCII .xyz, with scalar for Face, Angle, and Leaf/Wood hit
    if DEBUG_MODE:
        # Filter to valid hits before computing points to avoid NaN from inf * direction
        hit_mask = valid_rays_mask & np.isfinite(dists)
        if np.any(hit_mask):
            f_idx, a_idx, r_idx = np.nonzero(hit_mask)

            O_hits = O[f_idx, a_idx, r_idx]  # (N_hits, 3)
            D_hits = D[f_idx, a_idx, r_idx]  # (N_hits, 3)
            dists_hits = dists[f_idx, a_idx, r_idx]  # (N_hits,)
            pts = O_hits + D_hits * dists_hits[:, np.newaxis]  # (N_hits, 3)

            gids_hits = gids[f_idx, a_idx, r_idx]  # (N_hits,)
            leaf_hit = (np.isin(gids_hits, leaf_gids))
            wood_hit = (np.isin(gids_hits, wood_gids))

            classes = np.zeros(pts.shape[0], dtype=np.int8)
            classes[leaf_hit] = 1
            classes[wood_hit] = 0

            data = np.column_stack((pts, f_idx, a_idx, classes))  # (N_hits, 6)

            debug_dir = os.path.join(DEBUG_PATH, f"voxel_size={voxel_size}", f"voxel_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}")
            os.makedirs(debug_dir, exist_ok=True)
            out_path = os.path.join(debug_dir, f"{scene_name}_hits_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.xyz")

            # Write separate class mapping file
            class_map_path = os.path.join(debug_dir, "class_mapping.txt")
            face_idx_to_label = {i: label for i, label in enumerate(FACE_ORDER)}
            angle_idx_to_label = {i: label for i, label in enumerate(ANGLE_ORDER)}
            with open(class_map_path, 'w') as f:
                f.write(f"# face_idx mapping: {','.join([f'{i}={label}' for i, label in face_idx_to_label.items()])}\n")
                f.write(f"# angle_idx mapping: {','.join([f'{i}={label}' for i, label in angle_idx_to_label.items()])}\n")
                f.write(f"# class: 0=wood, 1=leaf\n")

            # Write hits to .xyz file in chunks to avoid large memory usage
            header_text = f"x y z face_idx angle_idx class\n"
            df = pd.DataFrame(data, columns=["x", "y", "z", "face_idx", "angle_idx", "class"])

            chunk_size = 200000  # 200 thousand rows per chunk
            with open(out_path, 'w', buffering=4*1024*1024) as f:
                f.write(header_text)
                for start in range(0, df.shape[0], chunk_size):
                    end = start + chunk_size
                    df.iloc[start:end].to_csv(f, sep=' ', header=False, index=False, float_format='%.2f')

    # Valid hits inside the voxel slab
    valid_hits_mask = np.isfinite(dists) & (dists >= t_near) & (dists <= t_far) & (dists >= 0) & valid_rays_mask

    # Separate leaf vs wood by geometry_id
    # Open3D returns INVALID_ID (typically 0xFFFFFFFF) on miss.
    # We only consider valid_hits_mask anyway.
    dist_leaf = np.full_like(dists, np.inf, dtype=np.float32)
    dist_wood = np.full_like(dists, np.inf, dtype=np.float32)
    if leaf_gids is not None:
        m_leaf = np.isin(gids, leaf_gids) & valid_hits_mask
        dist_leaf[m_leaf] = dists[m_leaf]
        _is_leaf = True
    else:
        # No leaf in voxel scene
        _is_leaf = False

    if wood_gids is not None:
        m_wood = np.isin(gids, wood_gids) & valid_hits_mask
        dist_wood[m_wood] = dists[m_wood]
        _is_wood = True
    else:
        # No wood in voxel scene
        _is_wood = False

    ### Combined Stats First ###
    # Use some shared variables later for leaf and wood statistics if present

    # eff_free_path_length means/vars (ddof=1); compute sums and sums of squares
    def mean_var_ddof1(x, N):
        mx = x.sum(axis=2)
        mean = np.divide(mx, N, out=np.zeros_like(mx), where=(N > 0))
        mx2 = (x**2).sum(axis=2)
        # var = (sum(x^2) - sum(x)^2 / N) / (N-1)
        numer = mx2 - (mx * mx) / np.maximum(N, 1)
        denom = np.maximum(N - 1, 1)
        var = numer / denom
        var[(N <= 1)] = 0.0
        return mean, var

    # Combined (leaf+wood): first hit distance among the two
    if _is_leaf is True or _is_wood is True:
        comb_dist = np.minimum(dist_leaf, dist_wood)  # (F,A,R)
    else:
        comb_dist = np.full_like(dists, np.inf, dtype=np.float32)
        comb_dist[valid_hits_mask] = dists[valid_hits_mask]

    N = valid_rays_mask.sum(axis=2).astype(np.int32)                 # (F,A)
    hit_any   = np.isfinite(comb_dist)            # (F,A,R)
    n_hits_lw   = hit_any.sum(axis=2).astype(np.int32)               # (F,A)
    
    path_lengths = np.zeros_like(dists, dtype=np.float32)
    path_lengths[valid_rays_mask] = (t_far[valid_rays_mask] - t_near[valid_rays_mask])
    mean_path_length = np.divide(path_lengths.sum(axis=2), N, out=np.zeros_like(path_lengths.sum(axis=2)), where=(N > 0))  # (F,A)
    sum_path_length  = path_lengths.sum(axis=2)                  # (F,A)
    free_path_length_comb = path_lengths.copy()
    free_path_length_comb[hit_any] = comb_dist[hit_any] - t_near[hit_any]
    eff_free_path_length_comb = np.zeros_like(free_path_length_comb, dtype=np.float32)
    eff_free_path_length_comb[valid_rays_mask] = compute_efpl_array(free_path_length_comb[valid_rays_mask], lambda_1).astype(np.float32)
    sum_eff_free_path_length_comb = np.sum(eff_free_path_length_comb, axis=2)
    
    sum_path_lengths   = path_lengths.sum(axis=2)                                  # (F,A)
    sum_free_path_length_comb    = free_path_length_comb.sum(axis=2)            # (F,A)
    mean_free_path_length_comb   = np.divide(sum_free_path_length_comb, N, out=np.zeros_like(sum_free_path_length_comb), where=(N > 0))  # (F,A)
    mean_eff_free_path_length_comb, var_eff_free_path_length_comb = mean_var_ddof1(eff_free_path_length_comb, N)
    sum_hits_eff_free_path_length_comb   = (eff_free_path_length_comb * hit_any).sum(axis=2)

    stats_comb = [[None for _ in range(A)] for _ in range(F)]
    for f in range(F):
        for a in range(A):
            stats_comb[f][a] = dict(
                N=int(N[f, a]),
                n_hits=int(n_hits_lw[f, a]),
                I=(float(n_hits_lw[f, a]) / float(N[f, a])) if N[f, a] else 0.0,
                mean_path_length=float(mean_path_length[f, a]),
                sum_path_length=float(sum_path_length[f, a]),
                sum_free_path_length=float(sum_free_path_length_comb[f, a]),
                mean_free_path_length=float(mean_free_path_length_comb[f, a]),
                mean_eff_free_path_length=float(mean_eff_free_path_length_comb[f, a]),
                var_eff_free_path_length=float(var_eff_free_path_length_comb[f, a]),
                sum_eff_free_path_length=float(sum_eff_free_path_length_comb[f, a]),
                sum_hits_eff_free_path_length=float(sum_hits_eff_free_path_length_comb[f, a]),
            )

    ### Leaf-only Stats ###
    if _is_leaf:
        hit_leaf  = np.isfinite(dist_leaf)            # (F,A,R)

        free_path_length_leaf = path_lengths.copy()
        free_path_length_leaf[hit_leaf] = dists[hit_leaf] - t_near[hit_leaf]
        sum_free_path_length_leaf = free_path_length_leaf.sum(axis=2)
        eff_free_path_length_leaf = np.zeros_like(free_path_length_leaf, dtype=np.float32)
        eff_free_path_length_leaf[valid_rays_mask] = compute_efpl_array(free_path_length_leaf[valid_rays_mask], lambda_1).astype(np.float32)
        sum_eff_free_path_length_leaf = np.sum(eff_free_path_length_leaf, axis=2)
        n_hits_leaf = hit_leaf.sum(axis=2).astype(np.int32)

        mean_free_path_length_leaf = np.nanmean(free_path_length_leaf, axis=2)
        mean_eff_free_path_length_leaf, var_eff_free_path_length_leaf = mean_var_ddof1(eff_free_path_length_leaf, N)
        sum_hits_eff_free_path_length_leaf = (eff_free_path_length_leaf * hit_leaf).sum(axis=2)

        stats_leaf = [[None for _ in range(A)] for _ in range(F)]
        for f in range(F):
            for a in range(A):
                stats_leaf[f][a] = dict(
                    N=int(N[f, a]),
                    n_hits=int(n_hits_leaf[f, a]),
                    I=(float(n_hits_leaf[f, a]) / float(N[f, a])) if N[f, a] else 0.0,
                    mean_path_length=float(mean_path_length[f, a]),
                    sum_path_length=float(sum_path_length[f, a]),
                    sum_free_path_length=float(sum_free_path_length_leaf[f, a]),
                    mean_free_path_length=float(mean_free_path_length_leaf[f, a]),
                    mean_eff_free_path_length=float(mean_eff_free_path_length_leaf[f, a]),
                    var_eff_free_path_length=float(var_eff_free_path_length_leaf[f, a]),
                    sum_eff_free_path_length=float(sum_eff_free_path_length_leaf[f, a]),
                    sum_hits_eff_free_path_length=float(sum_hits_eff_free_path_length_leaf[f, a]),
                )
        
    else:
        stats_leaf = [[None for _ in range(A)] for _ in range(F)]
        for f in range(F):
            for a in range(A):
                stats_leaf[f][a] = dict(
                    N=0,
                    n_hits=0,
                    I=0.0,
                    mean_path_length=0.0,
                    sum_path_length=0.0,
                    sum_free_path_length=0.0,
                    mean_free_path_length=0.0,
                    mean_eff_free_path_length=0.0,
                    var_eff_free_path_length=0.0,
                    sum_eff_free_path_length=0.0,
                    sum_hits_eff_free_path_length=0.0,
                )

    ### Wood-only Stats ###
    if _is_wood:
        hit_wood  = np.isfinite(dist_wood)            # (F,A,R)
        # free_path_length for wood-only: distance from t_near to wood intersection
        free_path_length_wood = path_lengths.copy()
        free_path_length_wood[hit_wood] = dists[hit_wood] - t_near[hit_wood]

        # Effective free path length transforms (you already have compute_efpl_array)
        # Compute only on valid rays to avoid NaNs
        eff_free_path_length_path = np.zeros_like(path_lengths, dtype=np.float32)
        eff_free_path_length_wood = np.zeros_like(free_path_length_wood, dtype=np.float32)

        eff_free_path_length_wood[valid_rays_mask]   = compute_efpl_array(free_path_length_wood[valid_rays_mask], lambda_1).astype(np.float32)

        n_hits_wood = hit_wood.sum(axis=2).astype(np.int32)              # (F,A)

        sum_free_path_length_wood  = free_path_length_wood.sum(axis=2)                                 # (F,A)
        mean_free_path_length_wood = np.divide(sum_free_path_length_wood, N, out=np.zeros_like(sum_free_path_length_wood), where=(N > 0))
        mean_eff_free_path_length_wood, var_eff_free_path_length_wood = mean_var_ddof1(eff_free_path_length_wood, N)

        sum_hits_eff_free_path_length_wood = (eff_free_path_length_wood * hit_wood).sum(axis=2)

        stats_wood= [[None for _ in range(A)] for _ in range(F)]

        for f in range(F):
            for a in range(A):
                stats_wood[f][a] = dict(
                    N=int(N[f, a]),
                    n_hits=int(n_hits_wood[f, a]),
                    I=(float(n_hits_wood[f, a]) / float(N[f, a])) if N[f, a] else 0.0,
                    mean_path_length=float(mean_path_length[f, a]),
                    sum_path_length=float(sum_path_length[f, a]),
                    sum_free_path_length=float(sum_free_path_length_wood[f, a]),
                    mean_free_path_length=float(mean_free_path_length_wood[f, a]),
                    mean_eff_free_path_length=float(mean_eff_free_path_length_wood[f, a]),
                    var_eff_free_path_length=float(var_eff_free_path_length_wood[f, a]),
                    sum_eff_free_path_length=float((eff_free_path_length_wood).sum(axis=2)[f, a]),
                    sum_hits_eff_free_path_length=float(sum_hits_eff_free_path_length_wood[f, a]),
                    mean_eff_free_path_length_free=float(mean_eff_free_path_length_wood[f, a]),
                    var_eff_free_path_length_free=float(var_eff_free_path_length_wood[f, a])
                )

    else:
        stats_wood = [[None for _ in range(A)] for _ in range(F)]
        for f in range(F):
            for a in range(A):
                stats_wood[f][a] = dict(
                    N=0,
                    n_hits=0,
                    I=0.0,
                    mean_path_length=0.0,
                    sum_path_length=0.0,
                    sum_free_path_length=0.0,
                    mean_free_path_length=0.0,
                    mean_eff_free_path_length=0.0,
                    var_eff_free_path_length=0.0,
                    sum_eff_free_path_length=0.0,
                    sum_hits_eff_free_path_length=0.0,
                    mean_eff_free_path_length_free=0.0,
                    var_eff_free_path_length_free=0.0
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
    voxel_size, 
    o3d_leaf,
    o3d_wood, 
    scenes,
    angles, 
    wood_volume_points, 
    lambda_1
    ):
    """
    Process a single voxel center with the given parameters.
    This function should contain the logic to process the voxel.
    """
    MIN_HITS = 10

    if o3d_leaf is None and "leaf" in scenes:
        print(f"[DEBUG] {voxel_center}: No leaf mesh after clipping for leaf-only scene. Skipping.")
        return [], []
    
    if o3d_wood is None and "wood" in scenes:
        print(f"[DEBUG] {voxel_center}: No wood mesh after clipping for wood-only scene. Skipping.")
        return [], []
    
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
        bin_comb, piad, _ = compute_LIAD_from_mesh(o3d_leaf + o3d_wood) if (o3d_leaf and o3d_wood) else (np.array([]), np.array([]), None)

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
                if o3d_leaf is None:
                    print(f"[DEBUG] {voxel_center}: No leaf mesh after clipping for leaf-only scene. Skipping.")
                    all_results.append({})
                    continue

                voxel_scene, leaf_gid, wood_gid = build_voxel_scene(o3d_leaf, None)
                # print(f"leaf --> {leaf_gid}, {wood_gid}")
                
                
                

            elif scene == "wood":
                if o3d_wood is None:
                    print(f"[DEBUG] {voxel_center}: No wood mesh after clipping for wood-only scene. Skipping.")
                    all_results.append({})
                    continue

                voxel_scene, leaf_gid, wood_gid = build_voxel_scene(None, o3d_wood)
                # print(f"wood --> {leaf_gid}, {wood_gid}")
                

            elif scene == "combined":
                if o3d_leaf is None and o3d_wood is None:
                    print(f"[DEBUG] {voxel_center}: No meshes after clipping for combined scene. Skipping.")
                    all_results.append({})
                    continue

                voxel_scene, leaf_gid, wood_gid = build_voxel_scene(o3d_leaf, o3d_wood)

            else:
                raise ValueError(f"Unknown scene type: {scene}")
    
            # Use simulate_voxel_grouped for vectorized stats computation
            stats_comb_grouped, stats_leaf_grouped, stats_wood_grouped = simulate_voxel_grouped(
            voxel_scene, leaf_gid, wood_gid,
            voxel_center, voxel_size,
            rays_FAR6, lambda_1, scene_name=scene)

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

                        ### CI from G ###
                        CI_leaf = (-math.log(pgap_leaf) / (G_leaf_est * LAD * leaf_data["mean_path_length"])
                            if (LAD is not None and LAD > 0 and G_leaf_est > 0 and
                                0 < pgap_leaf < 1 and leaf_data["mean_path_length"] is not None and leaf_data["mean_path_length"] > 0) else np.nan)
                        CI_wood = (-math.log(pgap_wood) / (G_wood_est * WAD * wood_data["mean_path_length"])
                            if (WAD is not None and WAD > 0 and G_wood_est > 0 and
                                0 < pgap_wood < 1 and wood_data["mean_path_length"] is not None and wood_data["mean_path_length"] > 0) else np.nan)
                        CI_comb = (-math.log(pgap_comb) / (G_comb_est * PAD * comb_data["mean_path_length"])
                            if (PAD is not None and PAD > 0 and G_comb_est > 0 and
                                0 < pgap_comb < 1 and comb_data["mean_path_length"] is not None and comb_data["mean_path_length"] > 0) else np.nan)
                        
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
                            "LAI_ref": float(LAI) if LAI is not None else np.nan, 
                            "WAI_ref": float(WAI) if WAI is not None else np.nan,
                            "PAI_ref": float(PAI) if PAI is not None else np.nan,
                            "LAD_ref": float(LAD) if LAD is not None else np.nan,
                            "WAD_ref": float(WAD) if WAD is not None else np.nan,
                            "PAD_ref": float(PAD) if PAD is not None else np.nan,
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
                            # CI from true G(θ) 
                            "CI_Leaf": float(CI_leaf) if not np.isnan(CI_leaf) else np.nan,
                            "CI_Wood": float(CI_wood) if not np.isnan(CI_wood) else np.nan,
                            "CI_Comb": float(CI_comb) if not np.isnan(CI_comb) else np.nan,
                            "alpha":   float(alpha) if alpha is not None else np.nan,
                            "leaf_fraction": float(leaf_fraction) if not np.isnan(leaf_fraction) else np.nan,
                            "wood_fraction": float(wood_fraction) if not np.isnan(wood_fraction) else np.nan,

                            
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
    parser.add_argument("--scene_formats", type=str, nargs='+', default=["combined"], help="Scene formats to process (default: ['combined']), use 'leaf' for leaf-only and 'wood' for wood-only.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0], help="Voxel sizes for processing (default: [0.2, 0.5, 1.0, 2.0]).")
    parser.add_argument("--num_angle_bins", type=int, default=18, help="Number of angle bins for ray tracing (default: 18).")
    parser.add_argument("--ray_spacing", type=float, default=0.005, help="Ray spacing for ray tracing (default: 0.1).")
    parser.add_argument("--wood_volume_voxel_size", type=float, default=0.01, help="Voxel size for wood volume calculation (default: 0.01).")
    parser.add_argument("--wood_volume_threshold", type=int, default=4, help="Threshold for wood volume calculation (default: 4).")
    parser.add_argument("--cross_section_area", type=float, default=0.003, help="Cross section area for wood volume calculation (default: 0.01).")
    parser.add_argument("--leaf_off", action='store_true', help="If set, leaf mesh will not be included in raytracing.")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel workers (default: 32).")
    parser.add_argument("--debug", action='store_true', help="If set, debug outputs will be saved.")
    args = parser.parse_args()

    # Print a nice statement outlining chosen inputs
    print(f"Processing scene file: {args.scene_file}")
    print(f"Scene formats: {args.scene_formats}")
    print(f"Voxel sizes: {args.voxel_sizes}")
    print(f"Number of Angle Bins: {args.num_angle_bins}")
    print(f"Ray spacing: {args.ray_spacing}")
    print(f"Wood volume voxel size: {args.wood_volume_voxel_size}")
    print(f"Wood volume threshold: {args.wood_volume_threshold}")
    print(f"Leaf off: {args.leaf_off}")
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
    _CUDA_DEVICE_ID = None
    try:
        # Try to detect NVIDIA GPU using nvidia-smi
        try:
            result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                gpu_lines = result.stdout.strip().split('\n')
                print(f"Detected NVIDIA GPUs:")
                for line in gpu_lines:
                    print(f"  {line}")
                _CUDA_DEVICE_ID = 0  # Use first GPU
            else:
                print("nvidia-smi not found or no NVIDIA GPU detected. Will try CUDA devices.")
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("nvidia-smi command failed. Attempting to initialize CUDA devices directly.")
        
    except Exception as e:
        print(f"Error checking CUDA availability: {e}")
        print("Code will use CPU for ray tracing.")

    # Store a global setting for leaf-off
    LEAF_OFF = args.leaf_off
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



        def process_voxel_wrapper(voxel_center, leaf_mesh, wood_mesh, scenes, voxel_size, angles, wood_volume_points, lambda_1):
            try:
                args = (voxel_center, voxel_size, leaf_mesh, wood_mesh, scenes, angles, wood_volume_points, lambda_1)
                result = process_voxel(*args)
                # print(f" Processed voxel {args[0]} successfully.")
                return result
            except Exception as e:
                print(f"Error processing voxel {args[0]}: {e}")
                traceback.print_exc()
                return []
            
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


        pbar = tqdm(total=len(voxel_centers), desc="Clipping meshes", unit="voxels") 
        with tqdm_joblib(pbar):
            results = Parallel(n_jobs=n_workers, backend='threading')(
                    delayed(clip_one_thread)(voxel_center, voxel_size) for voxel_center in voxel_centers
                )
            pbar.close()

        # results = [item for sublist in results for item in sublist]  # Flatten list of lists
        clipped_voxel_centres, clipped_leaf_vertices, clipped_leaf_faces, clipped_wood_vertices, clipped_wood_faces = zip(*results)
        valid_indices = [i for i, (v, lv, lf, wv, wf) in enumerate(results) if lv.shape[0] != 0 or lf.shape[0] != 0 or wv.shape[0] != 0 or wf.shape[0] != 0]

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
                o3d_leaf_mesh = o3d.geometry.TriangleMesh()
                o3d_leaf_mesh.vertices = o3d.utility.Vector3dVector(leaf_vertices)
                o3d_leaf_mesh.triangles = o3d.utility.Vector3iVector(leaf_faces)
                clipped_leaf_meshes.append(o3d_leaf_mesh)

                ## DEBUG ##
                if DEBUG_MODE:
                    debug_dir = os.path.join(DEBUG_PATH, f"voxel_size={voxel_size}", f"voxel_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}")
                    os.makedirs(debug_dir, exist_ok=True)

                    test_mesh_path = os.path.join(debug_dir, f"leaf_mesh_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}.ply")
                    o3d.io.write_triangle_mesh(test_mesh_path, o3d_leaf_mesh)
            else:
                clipped_leaf_meshes.append(None)

        for vc, wood_vertices, wood_faces in zip(clipped_voxel_centres, clipped_wood_vertices, clipped_wood_faces):
            if wood_vertices is not None and wood_faces is not None:
                o3d_wood_mesh = o3d.geometry.TriangleMesh()
                o3d_wood_mesh.vertices = o3d.utility.Vector3dVector(wood_vertices)
                o3d_wood_mesh.triangles = o3d.utility.Vector3iVector(wood_faces)
                clipped_wood_meshes.append(o3d_wood_mesh)

                ## DEBUG ##
                if DEBUG_MODE:
                    debug_dir = os.path.join(DEBUG_PATH, f"voxel_size={voxel_size}", f"voxel_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}")
                    os.makedirs(debug_dir, exist_ok=True)
                    test_mesh_path = os.path.join(debug_dir, f"wood_mesh_{vc[0]:.2f}_{vc[1]:.2f}_{vc[2]:.2f}.ply")
                    o3d.io.write_triangle_mesh(test_mesh_path, o3d_wood_mesh)
            else:
                clipped_wood_meshes.append(None)

        clip_time = dt.datetime.now() - start

        grid = _grid(voxel_size, ray_spacing)

        wood_volume_points = memory.cache(load_wood_volume_file)(wood_volume_file)

        worker = partial(
            process_voxel_wrapper,
            scenes=args.scene_formats,
            voxel_size=voxel_size,
            angles=angles,
            wood_volume_points=wood_volume_points,
            lambda_1=lambda_1
        )

        
        start02 = dt.datetime.now()
        print(f"Preprocessing time: {clip_time}")

        voxel_results = []
        for i, (vc, lm, wm) in enumerate(tqdm(zip(clipped_voxel_centres, clipped_leaf_meshes, clipped_wood_meshes), total=len(clipped_voxel_centres), desc="Processing voxels", unit="voxel")):
            all_results = worker(vc, lm, wm)
            voxel_results.extend(all_results)

        # Create a dataframe of voxel_results and subsequently filter by scene column
        df = pd.DataFrame(voxel_results)
        assert "scene" in df.columns, "No 'scene' column found in voxel results."

        scenes_present = df["scene"].dropna().unique().tolist()

        total_time = dt.datetime.now() - start
        raytrace_time = dt.datetime.now() - start02

        for scene in scenes_present:
            df_s = df[df["scene"] == scene].copy()
            output_basename = os.path.basename(args.scene_file).replace('.obj', f'_{scene}_results_{voxel_size}.csv') if LEAF_OFF is False else os.path.basename(args.scene_file).replace('.obj', f'_{scene}_results_{voxel_size}_leaf_off.csv')
            output_path = os.path.join(os.path.dirname(args.scene_file), output_basename)
            df_s.to_csv(output_path, index=False)
            print(f"Saved {len(df_s)} rows for scene '{scene}' to {output_path}.")

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
        perf_output_basename = os.path.basename(args.scene_file).replace('.obj', f'_performance_{voxel_size}.csv') if LEAF_OFF is False else os.path.basename(args.scene_file).replace('.obj', f'_performance_{voxel_size}_leaf_off.csv')
        perf_output_path = os.path.join(os.path.dirname(args.scene_file), perf_output_basename)
        perf_df.to_csv(perf_output_path, index=False)

        print(f"Processed {args.scene_file} and saved results to {output_path} in {total_time} seconds.")

    


    

