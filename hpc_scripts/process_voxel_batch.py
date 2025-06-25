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

temp_dir = os.getenv('TMPDIR', 'tmp')
# temp_dir = "/scratch/project/veg3d/uqjrivor/Raja_Tumba_Test/tmp"
if not os.path.exists(temp_dir):
    os.makedirs(temp_dir, exist_ok=True)
memory = Memory(location=temp_dir, verbose=1)

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
        leaf_mesh_path,
        wood_mesh_path,
        voxel_center,
        voxel_size) -> trimesh.Trimesh:
    """
    Roughly clip a mesh using a bounding box defined by the voxel center and size.
    This function uses a KDTree for efficient point querying.
    """
    # get_memory_usage()
    leaf_mesh = load_mesh_trimesh(leaf_mesh_path)
    leaf_tree = leaf_mesh.triangles_tree

    half_size = voxel_size / 2.0
    min_bound = np.array(voxel_center) - half_size
    max_bound = np.array(voxel_center) + half_size
    voxel_bounds = np.stack([min_bound, max_bound], axis=0)

    candidate_triangle_indices = list(leaf_tree.intersection(voxel_bounds.flatten()))

    if not candidate_triangle_indices:
        print(f"No triangles found within voxel bounds {voxel_bounds}.")
        del leaf_mesh, leaf_tree
        gc.collect()
        return voxel_center, None, None, None, None
    # get_memory_usage()
    # Extract triangles within the voxel bounds
    sub_mesh = leaf_mesh.submesh([candidate_triangle_indices], append=True)
    del leaf_mesh, leaf_tree

    ### NEW CODE ###
    # This code uses pyvista's clip_box function to clip the mesh.

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
    if clipped_leaf_faces.size == 0:
        print(f"No valid mesh found after clipping for voxel at {voxel_center}.")
        del sub_mesh, clipped_leaf_mesh
        gc.collect()
        return voxel_center, None, None, None, None
    
    ### DEBUG ####
    # Save .ply files for debugging
    project_dir = os.path.dirname(leaf_mesh_path)
    surface_area = clipped_leaf_mesh.area if hasattr(clipped_leaf_mesh, 'area') else 0.0
    debug_leaf_path = os.path.join(project_dir, f"clipped_leaf_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}_{surface_area}.ply")
    debug_wood_path = os.path.join(project_dir, f"clipped_wood_{voxel_center[0]:.2f}_{voxel_center[1]:.2f}_{voxel_center[2]:.2f}.ply")
    clipped_leaf_mesh.save(debug_leaf_path)
    
    del sub_mesh, clipped_leaf_mesh

    wood_mesh = load_mesh_trimesh(wood_mesh_path) if wood_mesh_path else None
    # get_memory_usage()
    if wood_mesh is None:   
        gc.collect()
        return voxel_center, clipped_leaf_vertices, clipped_leaf_faces, None, None
    
    ### NEW CODE ###
    # Create a PyVista mesh from the wood mesh
    wood_mesh = pv.wrap(wood_mesh)
    clipped_wood_mesh = wood_mesh.clip_box(voxel, invert=False)

    if isinstance(clipped_wood_mesh, pv.UnstructuredGrid):
        clipped_wood_mesh = clipped_wood_mesh.extract_geometry()
    if not clipped_wood_mesh.is_all_triangles:
        clipped_wood_mesh = clipped_wood_mesh.triangulate()
    clipped_wood_vertices = np.asarray(clipped_wood_mesh.points)
    clipped_wood_faces = np.asarray(clipped_wood_mesh.faces.reshape((-1, 4))[:, 1:])
    if clipped_leaf_faces.size == 0:
        print(f"No valid wood mesh found after clipping for voxel at {voxel_center}.")
        del sub_mesh, clipped_leaf_mesh
        gc.collect()
        return voxel_center, clipped_leaf_vertices, clipped_leaf_faces, None, None

    del wood_mesh, clipped_wood_mesh, voxel
    
    ### OLD CODE ###
    # See above.

    # Convert to open3d
    # o3d_wood_mesh = o3d.geometry.TriangleMesh()
    # o3d_wood_mesh.vertices = o3d.utility.Vector3dVector(wood_mesh.vertices)
    # o3d_wood_mesh.triangles = o3d.utility.Vector3iVector(wood_mesh.faces)
    # o3d_wood_mesh.compute_vertex_normals()
    # del wood_mesh

    # # Clip the wood mesh using the same voxel bounding box
    # clipped_wood_mesh = o3d_wood_mesh.crop(aabb)

    # clipped_wood_vertices = np.asarray(clipped_wood_mesh.vertices)
    # clipped_wood_faces = np.asarray(clipped_wood_mesh.triangles)
    # get_memory_usage()
    # del clipped_wood_mesh, o3d_wood_mesh
    gc.collect()
    
    # Ensure the mesh is valid
    return voxel_center, clipped_leaf_vertices, clipped_leaf_faces, clipped_wood_vertices, clipped_wood_faces


def build_voxel_scene(o3d_leaf, o3d_wood):
    """Return (scene, leaf_id, wood_id)  either id may be None."""
    scene = o3d.t.geometry.RaycastingScene()
    leaf_id = wood_id = None
    if (o3d_leaf is not None) and (len(o3d_leaf.triangles) > 0):
        leaf_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_leaf))

    if (o3d_wood is not None) and (len(o3d_wood.triangles) > 0):
        wood_id = scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_wood))

    return scene, leaf_id, wood_id

def compute_wood_volume_in_voxel(wood_volume_file, voxel_center, voxel_size, small_voxel_size=0.01):
    """
    Return estimates the volume of wood points within a voxel.
    This function assumes wood_volume_file is a numpy array of shape (N, 3).
    """

    if wood_volume_file is None or wood_volume_file.size == 0:
        return 0.0
    
    # Calculate number of points within the voxel
    half_size = voxel_size / 2.0
    min_bound = np.array(voxel_center) - half_size
    max_bound = np.array(voxel_center) + half_size
    in_voxel = np.all((wood_volume_file >= min_bound) & (wood_volume_file <= max_bound), axis=1)
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
def _grid(face_len, spacing):
    s = np.arange(-face_len / 2, face_len / 2 +spacing, spacing)
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
    valid = (t_near <= t_far) & (t_far >= 0)
    N = int(valid.sum())
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

    dist_leaf = np.full_like(dists, np.inf)
    dist_wood = np.full_like(dists, np.inf)
    if leaf_gid is not None:
        m = (gids == leaf_gid) & valid & (dists >= t_near) & (dists <= t_far)
        dist_leaf[m] = dists[m]
    if wood_gid is not None:
        m = (gids == wood_gid) & valid & (dists >= t_near) & (dists <= t_far)
        dist_wood[m] = dists[m]
    
    comb_dist = np.minimum(dist_leaf, dist_wood)
    hit_any = np.isfinite(comb_dist) & (comb_dist < np.inf)
    n_hits = int(hit_any.sum())

    delta = np.zeros_like(dists)
    delta[valid] = t_far[valid] - t_near[valid]
    z_arr = delta.copy()
    z_arr[hit_any] = comb_dist[hit_any] - t_near[hit_any]

    efpl_d = compute_efpl_array(delta[valid], lambda_1)
    efpl_f = compute_efpl_array(z_arr[valid],  lambda_1)

    stats_lw = dict(
        N=N,
        n_hits=n_hits,
        I=n_hits / N if N else 0.0,
        delta_bar=delta[valid].mean() if N else 0.0,
        sum_delta=delta[valid].sum(),
        sum_z=z_arr[valid].sum(),
        mean_z=z_arr[valid].mean() if N else 0.0,
        mean_delta_e=efpl_d.mean() if N else 0.0,
        var_delta_e=efpl_d.var(ddof=1) if N > 1 else 0.0,
        sum_z_e=efpl_f.sum(),
        sum_hits_z_e=efpl_f[hit_any[valid]].sum() if n_hits else 0.0,
        mean_efpl_free=efpl_f.mean() if N else 0.0,
        var_efpl_free=efpl_f.var(ddof=1) if N > 1 else 0.0,
    )

    hit_m = (gids == leaf_gid) & valid & (dists >= t_near) & (dists <= t_far)

    delta = np.zeros_like(dists)
    delta[valid] = t_far[valid] - t_near[valid]
    z_arr = delta.copy()
    z_arr[hit_m] = dists[hit_m] - t_near[hit_m]

    efpl_d = compute_efpl_array(delta[valid], lambda_1)
    efpl_f = compute_efpl_array(z_arr[valid],  lambda_1)

    n_hits = int(hit_m.sum())
    stats_leaf = dict(
        N=N,
        n_hits=n_hits,
        I=n_hits / N if N else 0.0,
        delta_bar=delta[valid].mean() if N else 0.0,
        sum_delta=delta[valid].sum(),
        sum_z=z_arr[valid].sum(),
        mean_z=z_arr[valid].mean() if N else 0.0,
        mean_delta_e=efpl_d.mean() if N else 0.0,
        var_delta_e=efpl_d.var(ddof=1) if N > 1 else 0.0,
        sum_z_e=efpl_f.sum(),
        sum_hits_z_e=efpl_f[hit_m[valid]].sum() if n_hits else 0.0,
        mean_efpl_free=efpl_f.mean() if N else 0.0,
        var_efpl_free=efpl_f.var(ddof=1) if N > 1 else 0.0,
    )
    return stats_lw, stats_leaf, hit_details

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
    
def process_voxel(
        voxel_center, 
        voxel_size, 
        o3d_leaf,
        o3d_wood, 
        ray_spacing, 
        grid,
        angles, 
        wood_volume_file, 
        lambda_1
    ):
    """
    Process a single voxel center with the given parameters.
    This function should contain the logic to process the voxel.
    """
    MIN_HITS = 10

    if o3d_leaf is None:
        print(f"[DEBUG] {voxel_center}: No leaf mesh after clipping. Skipping.")
        # print(f"Voxel center {voxel_center} has no leaf mesh after clipping. Skipping.")
        return [], []

    # Build voxel scene
    try:
        voxel_scene, leaf_gid, wood_gid = build_voxel_scene(o3d_leaf, o3d_wood)
    except Exception as e:
        raise RuntimeError(
            f"Error building voxel scene for voxel at {voxel_center}: {e}"
        ) from e

    # Reference areas / densities
    try:
        leaf_area = o3d_leaf.get_surface_area()
        wood_area = o3d_wood.get_surface_area() if o3d_wood else 0.0
    except Exception as e:
        raise RuntimeError(
            f"Error computing surface area for voxel at {voxel_center}: {e}"
        ) from e

    try:
        LAI_leaf = leaf_area / (voxel_size ** 2)
        LAI_lw = (leaf_area + wood_area) / (voxel_size ** 2)
        LAD_ref = leaf_area / (voxel_size ** 3)
        PAD_ref = (leaf_area + wood_area) / (voxel_size ** 3)
        if wood_volume_file is not None:
            wood_vol = compute_wood_volume_in_voxel(wood_volume_file, voxel_center, voxel_size, 0.02) if wood_volume_file is not None else 0.0
            alpha = (voxel_size ** 3 - wood_vol) / (voxel_size ** 3)
        else:
            alpha = None
        # print(f"Processing voxel at {voxel_center} with LAI_leaf: {LAI_leaf}, LAI_lw: {LAI_lw}, LAD_ref: {LAD_ref}, PAD_ref: {PAD_ref}, alpha: {alpha}")

        # LIAD bins for voxel
        bin_leaf, liad_leaf, _ = compute_LIAD_from_mesh(o3d_leaf)
        comb_mesh = o3d_leaf if o3d_wood is None else o3d_leaf + o3d_wood
        bin_lw, liad_lw, _ = compute_LIAD_from_mesh(comb_mesh)

        # Store the LIAD bins repeated on every row
        liad_dict = {f"LIAD_bin_{c:.1f}": float(v)
                    for c, v in zip(bin_leaf, liad_leaf)}
    except Exception as e:
        raise RuntimeError(
            f"Error computing LAI, LIAD, or alpha for voxel at {voxel_center}: {e}"
        ) from e
    
    voxel_rows = []
    faces = [
        ("bottom", generate_face_rays_bottom),
        ("top", generate_face_rays_top),
        ("xplus", generate_side_rays_xplus),
        ("xminus", generate_side_rays_xminus),
        ("yplus", generate_side_rays_yplus),
        ("yminus", generate_side_rays_yminus)
    ]
    try:
        for face_lbl, face_fn in faces:
            # print(f"[DEBUG] {voxel_center}: Processing face {face_lbl}")
            o_base, d_base = face_fn(voxel_center, voxel_size, grid)
            for angle in sorted(angles):
                # print(f"[DEBUG] {voxel_center}: {face_lbl}: Processing angle {angle}")
                ### Rotate rays ###
                if face_lbl in ("xplus", "xminus"):
                    rot_o, rot_d = rotate_rays(o_base, d_base, angle, voxel_center, axis='y')
                else:
                    rot_o, rot_d = rotate_rays(o_base, d_base, angle, voxel_center, axis='x')

                ### Ray tracing ###
                lw_data, leaf_data, _ = simulate_combined_mesh_with_points(voxel_scene, leaf_gid, wood_gid, voxel_center, voxel_size, rot_o, rot_d, lambda_1)

                if leaf_data["N"] == 0:
                    leaf_data = {k: np.nan for k in leaf_data}
                if lw_data["N"] == 0:
                    lw_data = {k: np.nan for k in lw_data}

                ### G ###
                G_leaf_est = np.nan
                if leaf_data ["n_hits"] >= MIN_HITS and bin_leaf.size > 0:
                    G_leaf_est = compute_G_function_binwise([angle], bin_leaf, liad_leaf)[0]

                G_lw_est = np.nan
                if lw_data["n_hits"] >= MIN_HITS and bin_lw.size > 0:
                    G_lw_est = compute_G_function_binwise([angle], bin_lw, liad_lw)[0]

                ### pgap ###
                pgap_leaf = 1.0 - leaf_data["I"]
                pgap_lw = 1.0 - lw_data["I"]

                ### CI from G ###
                CI_leaf = (-math.log(pgap_lw) / (G_leaf_est * LAI_leaf)
                        if (LAI_leaf > 0 and G_leaf_est > 0 and
                            0 < pgap_lw < 1) else np.nan)

                CI_lw   = (-math.log(pgap_lw) / (G_lw_est * LAI_leaf)
                        if (LAI_leaf > 0 and G_lw_est > 0 and
                            0 < pgap_lw < 1) else np.nan)
                
                ### LAD/PAD metrics 
                lad_leaf = compute_lad_metrics(
                    leaf_data["n_hits"], leaf_data["N"], G_leaf_est,
                    leaf_data["delta_bar"], leaf_data["mean_z"],
                    leaf_data["mean_delta_e"], leaf_data["var_delta_e"],
                    lambda_1)

                pad_lw = compute_pad_metrics(
                    lw_data["n_hits"], leaf_data["n_hits"],
                    lw_data["N"], G_lw_est, lw_data["delta_bar"],
                    lw_data["sum_z"], lw_data["sum_z_e"],
                    leaf_data["sum_hits_z_e"], lw_data["sum_hits_z_e"],
                    alpha, lw_data["mean_delta_e"], lw_data["var_delta_e"],
                    lambda_1,
                    leaf_data["n_hits"]/lw_data["n_hits"]
                    if lw_data["n_hits"] else 0.0)
                
                ### data prep ###
                dx, dy, dz = rot_d[0]
                leaf_fraction = (leaf_data["n_hits"] / lw_data["n_hits"]
                                if lw_data["n_hits"] else np.nan)
                
                row = {
                    "voxel_cx": voxel_center[0], "voxel_cy": voxel_center[1], "voxel_cz": voxel_center[2],
                    "face": face_lbl, "angle_deg": angle,
                    "dx": dx, "dy": dy, "dz": dz,

                    # per-angle G
                    "G_leaf_computed": G_leaf_est,
                    "G_lw_computed":   G_lw_est,

                    # reference densities
                    "LAI_Leaf": LAI_leaf, "LAI_lw": LAI_lw,
                    "LAD_ref":  LAD_ref,  "PAD_ref": PAD_ref,

                    # CI from true G(ÃÂ¸)
                    "CI_leaf": CI_leaf,
                    "CI_lw":   CI_lw,
                    "alpha":   alpha,
                    "leaf_fraction": leaf_fraction,
                    # LAD metrics
                    "LAD_BL":          lad_leaf.get("LAD_BL", np.nan),
                    "LAD_BL_EPL":      lad_leaf.get("LAD_BL_EPL", np.nan),
                    "LAD_BL_UEPL":     lad_leaf.get("LAD_BL_UEPL", np.nan),
                    "LAD_MCF":         lad_leaf.get("LAD_MCF", np.nan),
                    "LAD_MCF_Corr":    lad_leaf.get("LAD_MCF_Corrected", np.nan),

                    # PAD metrics
                    "PAD_BL":          pad_lw.get("PAD_BL", np.nan),
                    "PAD_BL_EPL":      pad_lw.get("PAD_BL_EPL", np.nan),
                    "PAD_BL_UEPL":     pad_lw.get("PAD_BL_UEPL", np.nan),
                    "PAD_MCF":         pad_lw.get("PAD_MCF", np.nan),
                    "PAD_MCF_Corr":    pad_lw.get("PAD_MCF_Corrected", np.nan),
                    "PAD_MLE_pimont_2018": pad_lw.get("PAD_MLE_pimont_2018", np.nan),
                    # extra LAD estimates that lived in the PAD dict
                    "LAD_MLE_pimont_2019": pad_lw.get("LAD_MLE_pimont_2019", np.nan),
                    "LAD_MLE_soma":        pad_lw.get("LAD_MLE_Soma_21",   np.nan),
                    # raw stats
                    # raw ray statistics
                    "Total_number_of_rays": lw_data.get("N", np.nan),
                    "sum_path_length":      lw_data.get("sum_delta", np.nan),
                    "mean_path_length":     lw_data.get("delta_bar", np.nan),

                    # hits
                    "num_leaf_hits": leaf_data.get("n_hits", np.nan),
                    "num_lw_hits":   lw_data.get("n_hits", np.nan),
                    "num_hits":      lw_data.get("N",      np.nan),   # kept for legacy parity

                    # interception / pgap
                    "I_leaf": leaf_data.get("I", np.nan),
                    "I_lw":   lw_data.get("I", np.nan),
                    "pgap_leaf": 1.0 - leaf_data.get("I", np.nan),
                    "pgap_lw":   1.0 - lw_data.get("I", np.nan),

                    # freepath sums & means
                    "sum_free_path_length":           lw_data.get("sum_z",             np.nan),
                    "sum_effective_free_path_length": lw_data.get("sum_z_e",           np.nan),
                    "sum_effective_free_path_length_hit":
                        lw_data.get("sum_hits_z_e", np.nan),
                    "sum_effective_free_path_length_hit_leaf":
                        leaf_data.get("sum_hits_z_e", np.nan),

                    "mean_free_path_length":              lw_data.get("mean_z",        np.nan),
                    "mean_effective_path_length":         lw_data.get("mean_delta_e",  np.nan),
                    "var_effective_path_length":          lw_data.get("var_delta_e",   np.nan),
                    "mean_effective_free_path_length":    lw_data.get("mean_efpl_free",np.nan),
                    "var_effective_free_path_length":     lw_data.get("var_efpl_free", np.nan),
                }

                # Add LIAD bins to the row
                row.update(liad_dict)

                voxel_rows.append(row)
    
        # print(f"[DEBUG] {voxel_center}: Finished processing all faces and angles")
    
    except Exception as e:
        raise RuntimeError(
            f"Error processing voxel at {voxel_center} with size {voxel_size}: {e}"
        ) from e

            
    del o3d_leaf, o3d_wood, voxel_scene, rot_o, rot_d, lw_data, leaf_data
    gc.collect()
    return voxel_rows, []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process voxel batch.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file containing voxel batch information.")
    parser.add_argument('--index', type=int, required=True, help='The index for batch processing')
    parser.add_argument('--log_file', type=str, help='File path to use for log information.')
    args = parser.parse_args()

    csv_path = args.csv_path
    index = args.index
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} does not exist. Please check the path.")
    original_stdout = sys.stdout
    log_file = args.log_file
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
    
    # Read the CSV file and process the batches starting from index
    with open(csv_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        csv.field_size_limit(10**7)
        batches = list(reader)
        if index >= len(batches):
            raise IndexError(f"Start index {index} is out of range for the number of batches {len(batches)}.")
        
    # Parse variables from the csv index
    batch = batches[index]
    files = json.loads(batch['files'])
    voxel_size = float(batch['voxel_size'])
    lambda_1 = float(batch['lambda_1'])
    ray_spacing = float(batch['ray_spacing'])
    voxel_centers = json.loads(batch['voxel_centers'])
    # voxel_ids = json.loads(batch['voxel_ids'])
    angles = json.loads(batch['angles'])

    # Extract the leaf and wood meshes and wood points from the scene_file
    leaf_mesh_file = files.get('leaf_mesh_file')
    wood_mesh_file = files.get('wood_mesh_file')

    try:
        wood_volume_file = np.loadtxt(files['wood_volume_file'])
        wood_volume_file = wood_volume_file.copy() if files.get('wood_volume_file') else None  # Ensure we work with a copy
    except FileNotFoundError:
        wood_volume_file = None  # Handle case where wood volume file is not provided

    # Process each voxel center in parallel and collect results
    # Queue up all the futures first (they are submitted but not started until executor runs)
    def process_voxel_wrapper(voxel_center, leaf_mesh, wood_mesh, voxel_size, ray_spacing, grid, angles, wood_volume_file, lambda_1):
        try:
            args = (voxel_center, voxel_size, leaf_mesh, wood_mesh, ray_spacing, grid, angles, wood_volume_file, lambda_1)
            result, placeholder = process_voxel(*args)
            print(f" Processed voxel {args[0]} successfully.")
            return result, placeholder
        except Exception as e:
            print(f"Error processing voxel {args[0]}: {e}")
            traceback.print_exc()
            return [], []
    
    def parallel_process_voxels(voxel_centers, voxel_size, leaf_mesh_file, wood_mesh_file, ray_spacing, angles, wood_volume_file, lambda_1):
        num_worker = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)))
        num_worker = max(1, num_worker)
        # available_memory = psutil.virtual_memory().available
        # estimated_per_worker_mem = (os.path.getsize(leaf_mesh_file) + (os.path.getsize(wood_mesh_file))) * 4
        # num_worker = min(1, num_worker, available_memory // estimated_per_worker_mem)
        print(f"Using {num_worker} workers for processing voxels.")

        # Load mesh into cache once, to avoid race conditions
        leaf_mesh = load_mesh_trimesh(leaf_mesh_file)
        wood_mesh = load_mesh_trimesh(wood_mesh_file)

        # Calculate the grid to pass to worker function
        grid = _grid(voxel_size * 2, ray_spacing)

        # Get clipped meshes for each voxel center
        pbar = tqdm(total=len(voxel_centers), desc="Clipping meshes", unit="voxel")
        with tqdm_joblib(pbar):
            results = Parallel(n_jobs=num_worker, backend='loky', prefer="processes")(
                delayed(get_clipped_meshes)(leaf_mesh_file, wood_mesh_file, voxel_center, voxel_size)
                for voxel_center in voxel_centers
            )
        pbar.close()

        # Unzip the results
        voxel_centers, clipped_leaf_vertices, clipped_leaf_faces, clipped_wood_vertices, clipped_wood_faces = zip(*results)

        # Convert clipped vertices and faces to Open3D meshes
        clipped_leaf_meshes = []
        clipped_wood_meshes = []
        for leaf_vertices, leaf_faces in zip(clipped_leaf_vertices, clipped_leaf_faces):
            if leaf_vertices is not None and leaf_faces is not None:
                o3d_leaf_mesh = o3d.geometry.TriangleMesh()
                o3d_leaf_mesh.vertices = o3d.utility.Vector3dVector(leaf_vertices)
                o3d_leaf_mesh.triangles = o3d.utility.Vector3iVector(leaf_faces)
                clipped_leaf_meshes.append(o3d_leaf_mesh)
            else:
                clipped_leaf_meshes.append(None)

        for wood_vertices, wood_faces in zip(clipped_wood_vertices, clipped_wood_faces):
            if wood_vertices is not None and wood_faces is not None:
                o3d_wood_mesh = o3d.geometry.TriangleMesh()
                o3d_wood_mesh.vertices = o3d.utility.Vector3dVector(wood_vertices)
                o3d_wood_mesh.triangles = o3d.utility.Vector3iVector(wood_faces)
                clipped_wood_meshes.append(o3d_wood_mesh)
            else:
                clipped_wood_meshes.append(None)

        # Remove None meshes from the lists
        valid_indices = [i for i, mesh in enumerate(clipped_leaf_meshes) if mesh is not None]
        voxel_centers = [voxel_centers[i] for i in valid_indices]
        clipped_leaf_meshes = [clipped_leaf_meshes[i] for i in valid_indices]
        clipped_wood_meshes = [clipped_wood_meshes[i] for i in valid_indices]

        # Clean up memory
        del clipped_leaf_vertices, clipped_leaf_faces, clipped_wood_vertices, clipped_wood_faces
        gc.collect()

        worker = partial(
            process_voxel_wrapper,
            voxel_size=voxel_size,
            ray_spacing=ray_spacing,
            grid=grid,
            angles=angles,
            wood_volume_file=wood_volume_file,
            lambda_1=lambda_1
        )

        clip_time = dt.datetime.now() - start
        start02 = dt.datetime.now()
        print(f"Preprocessing time: {clip_time}")
        ### PROCESSING VOXELS ###
        results = []
        for i, (voxel_center, leaf_mesh, wood_mesh) in enumerate(tqdm(zip(voxel_centers, clipped_leaf_meshes, clipped_wood_meshes), total=len(voxel_centers), desc="Processing voxels", unit="voxel")):
            result, _ = worker(voxel_center, leaf_mesh, wood_mesh)
            df = pd.DataFrame(result)
            mesh_surface_area = leaf_mesh.get_surface_area() if leaf_mesh else 0.0
            LAI = (df['LAI_Leaf'].mean()/voxel_size) if not df.empty else 0.0
            print(f"Voxel {i+1}/{len(voxel_centers)}: Center {voxel_center}, Surface Area: {mesh_surface_area:.2f}, LAI: {LAI:.2f}")
            results.append(df)

        total_time = dt.datetime.now() - start
        raytrace_time = dt.datetime.now() - start02

        return results, clip_time, total_time, raytrace_time
    
    start = dt.datetime.now()
    
    # # Call the parallel processing function
    results, clip_time, total_time, raytrace_time = parallel_process_voxels(
        voxel_centers, voxel_size, leaf_mesh_file, wood_mesh_file, ray_spacing, angles, wood_volume_file, lambda_1
    )

    # Convert results to a DataFrame and save to CSV
    # This csv will ne save in a subfolder to csv_path for preliminary results
    df = pd.concat(results, ignore_index=True)
    preliminary_output_path = os.path.join(os.path.dirname(csv_path), "preliminary_results")
    os.makedirs(preliminary_output_path, exist_ok=True)
    output_csv = os.path.join(preliminary_output_path, f"results_batch_{index}_vs_{voxel_size}.csv")
    df.to_csv(output_csv, index=False)

    # Save time outputs
    time_output = {
        'cpus': psutil.cpu_count(logical=False),
        'num_voxels': len(voxel_centers),
        'clip_time': clip_time.total_seconds(),
        'raytrace_time': raytrace_time.total_seconds(),
        'total_time': total_time.total_seconds(),
        's/voxel': raytrace_time.total_seconds() / len(voxel_centers) if len(voxel_centers) > 0 else 0.0,
    }
    time_output_csv = os.path.join(preliminary_output_path, f"processing_times_{index}_vs_{voxel_size}.csv")
    pd.DataFrame([time_output]).to_csv(time_output_csv, index=False)
    sys.stdout = original_stdout  # Reset stdout to original


    print(f"Processed voxel batch {index} and saved results to {output_csv}.")

    


    

