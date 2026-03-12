# -*- coding: utf-8 -*-
"""
Threaded voxel processing (CPU/GPU pipelined) with GPU stability fixes
---------------------------------------------------------------------
- CPU (Open3D/Embree) raycasting: single worker (one voxel at a time) so Embree can parallelize internally.
- GPU (Warp/CUDA) raycasting: multi-threaded workers feeding the device using per-thread CUDA streams.
- Fix: hit counts (`num_hits_*`) derived from counts arrays, not `isfinite(first_hit_*)`.
- Fix: Warp allocations/launches happen inside `wp.ScopedDevice/ScopedStream`, 
       all device arrays use correct dtypes (wp.float32 / wp.int32 / wp.vec3), and we synchronize stream before host copies.
"""
from __future__ import annotations

import os
import sys
import gc
import csv
import json
import time
import argparse
import traceback
import threading
from queue import Queue
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil

# Lazy heavy imports

def _import_open3d():
    import open3d as o3d
    import open3d.core as o3c
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    return o3d, o3c

def _import_trimesh_pv():
    import trimesh
    import pyvista as pv
    return trimesh, pv

# External utilities (unchanged API expected)
from utils import classify_liad_to_dewit, calculate_G, resolve_cuda_index, create_voxel_id

# Optional Warp
_HAS_WARP = False
try:
    import warp as wp
    wp.init()
    _HAS_WARP = True
except Exception:
    _HAS_WARP = False

# -------------------------
# CSV Writer (single consumer thread)
# -------------------------
class CSVWriter:
    def __init__(self, path: str, metadata_keys_order: Optional[Sequence[str]] = None, overwrite: bool = False):
        self.path = path
        self.metadata_keys_order = list(metadata_keys_order) if metadata_keys_order else None
        self.overwrite = overwrite
        self._file = None
        self._writer = None
        self._fieldnames: List[str] = []
        self._header_written = False
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        mode = "w" if overwrite else "a"
        self._file = open(self.path, mode, encoding="utf-8", newline="")
        self._header_written = (not overwrite) and os.path.exists(self.path) and os.path.getsize(self.path) > 0

    def _prepare_fieldnames(self, rows: List[Dict[str, Any]]):
        keys = set()
        for r in rows:
            keys.update(r.keys())
        ordered: List[str] = []
        if self.metadata_keys_order:
            for k in self.metadata_keys_order:
                if k in keys:
                    ordered.append(k)
        rest = sorted([k for k in keys if k not in ordered])
        self._fieldnames = ordered + rest
        self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
        if not self._header_written:
            self._writer.writeheader()
            self._file.flush()
            self._header_written = True

    def write_rows(self, rows: List[Dict[str, Any]]):
        if not rows:
            return
        if self._writer is None:
            self._prepare_fieldnames(rows)
        new_cols = set().union(*[r.keys() for r in rows]) - set(self._fieldnames)
        if new_cols:
            raise RuntimeError(
                f"New columns appeared after header was written: {sorted(new_cols)}. "
                "Ensure the first batch contains all columns."
            )
        for r in rows:
            self._writer.writerow(r)
        self._file.flush()

    def close(self):
        if self._file:
            try:
                self._file.flush()
            finally:
                self._file.close()
            self._file = None


def _to_py_scalar(v: Any) -> Any:
    if isinstance(v, (np.floating, np.integer, np.bool_)):
        return v.item()
    if isinstance(v, np.ndarray):
        try:
            return json.dumps(v.tolist())
        except Exception:
            return v.tolist()
    return v


def build_rows(
    metadata: Dict[str, Any],
    stats: Dict[str, np.ndarray],
    *,
    face_order: Optional[Sequence[str]] = None,
    angles: Optional[Sequence[float]] = None,
    voxel_id: Optional[Union[int, str]] = None,
) -> List[Dict[str, Any]]:
    F = A = None
    for v in stats.values():
        if isinstance(v, np.ndarray) and v.ndim == 2:
            F, A = v.shape
            break
    if F is None or A is None:
        return []
    face_order = list(face_order) if face_order is not None else [f"face_{i}" for i in range(F)]
    angles = list(angles) if angles is not None else [float(i) for i in range(A)]
    rows: List[Dict[str, Any]] = []
    meta = {k: _to_py_scalar(v) for k, v in metadata.items()}
    if voxel_id is not None:
        meta["voxel_id"] = voxel_id
    for f in range(F):
        for a in range(A):
            row = dict(meta)
            row["face_index"] = f
            row["face_name"] = face_order[f] if f < len(face_order) else f"face_{f}"
            row["angle_index"] = a
            row["angle_center_deg"] = float(angles[a]) if a < len(angles) else float(a)
            for key, arr in stats.items():
                if isinstance(arr, np.ndarray):
                    if arr.ndim == 2 and arr.shape == (F, A):
                        row[key] = _to_py_scalar(arr[f, a])
                    else:
                        try:
                            row[key] = _to_py_scalar(np.asarray(arr)[f, a])
                        except Exception:
                            row[key] = _to_py_scalar(arr)
                else:
                    row[key] = _to_py_scalar(arr)
            rows.append(row)
    return rows

# Geometry helpers
FACE_ORDER = ["bottom", "top", "xplus", "xminus", "yplus", "yminus"]
_ROT_AXIS = {"xplus": "y", "xminus": "y", "bottom": "x", "top": "x", "yplus": "x", "yminus": "x"}

def _grid(face_extent: float, ray_spacing: float):
    s = np.arange(-face_extent / 2.0, face_extent / 2.0 + ray_spacing, ray_spacing)
    return np.meshgrid(s, s, indexing="xy")

def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]], dtype=np.float32)

def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]], dtype=np.float32)

class DeviceManager:
    def __init__(self, prefer_cuda: bool = True):
        self.prefer_cuda = prefer_cuda
        self.cuda_index: Optional[int] = None
        self.cuda_uuid: Optional[str] = None
        self.using_warp: bool = False
        self.device_str: str = "cpu"
        self._detect()
    def _detect(self):
        if self.prefer_cuda and _HAS_WARP:
            try:
                prefer_uuid = os.getenv("OPEN3D_GPU_UUID", None)
                idx, uuid = resolve_cuda_index(prefer_uuid)
                if idx is not None:
                    self.cuda_index, self.cuda_uuid = idx, uuid
                    self.using_warp = True
                    self.device_str = f"cuda:{idx}"
                    print(f"Using CUDA device {idx}{' (UUID: ' + uuid + ')' if uuid else ''} via Warp")
                    return
            except Exception as e:
                print(f"[warn] CUDA resolve failed, falling back to CPU: {e}")
        print("CUDA/Warp unavailable -> using CPU raycasting (Open3D)")
        self.using_warp = False
        self.device_str = "cpu"

class RayGrid:
    def __init__(self, voxel_sizes: Sequence[float], ray_spacing: float):
        self.cache: Dict[float, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
        for vs in voxel_sizes:
            face_len = float(vs) * np.sqrt(2) + 1e-6
            XX, YY = _grid(face_len, ray_spacing)
            self.cache[float(vs)] = {
                "bottom": self._face_bottom(vs, XX, YY),
                "top": self._face_top(vs, XX, YY),
                "xplus": self._face_xplus(vs, XX, YY),
                "xminus": self._face_xminus(vs, XX, YY),
                "yplus": self._face_yplus(vs, XX, YY),
                "yminus": self._face_yminus(vs, XX, YY),
            }
    @staticmethod
    def _face_bottom(vs, XX, YY):
        z = -vs * 2
        org = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z, dtype=np.float32)])
        dir = np.tile([0,0,1], (len(org),1)).astype(np.float32)
        return org.astype(np.float32), dir
    @staticmethod
    def _face_top(vs, XX, YY):
        z = +vs * 2
        org = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z, dtype=np.float32)])
        dir = np.tile([0,0,-1], (len(org),1)).astype(np.float32)
        return org.astype(np.float32), dir
    @staticmethod
    def _face_xplus(vs, XX, YY):
        x = +vs * 2
        org = np.column_stack([np.full(YY.size, x, dtype=np.float32), XX.ravel(), YY.ravel()])
        dir = np.tile([-1,0,0], (len(org),1)).astype(np.float32)
        return org.astype(np.float32), dir
    @staticmethod
    def _face_xminus(vs, XX, YY):
        x = -vs * 2
        org = np.column_stack([np.full(YY.size, x, dtype=np.float32), XX.ravel(), YY.ravel()])
        dir = np.tile([1,0,0], (len(org),1)).astype(np.float32)
        return org.astype(np.float32), dir
    @staticmethod
    def _face_yplus(vs, XX, YY):
        y = +vs * 2
        org = np.column_stack([XX.ravel(), np.full(XX.size, y, dtype=np.float32), YY.ravel()])
        dir = np.tile([0,-1,0], (len(org),1)).astype(np.float32)
        return org.astype(np.float32), dir
    @staticmethod
    def _face_yminus(vs, XX, YY):
        y = -vs * 2
        org = np.column_stack([XX.ravel(), np.full(XX.size, y, dtype=np.float32), YY.ravel()])
        dir = np.tile([0,1,0], (len(org),1)).astype(np.float32)
        return org.astype(np.float32), dir
    def build(self, voxel_center: np.ndarray, voxel_size: float, angles: Sequence[float]):
        grids = self.cache[float(voxel_size)]
        face_order = FACE_ORDER
        angles_sorted = list(sorted(angles))
        F = len(face_order); R = grids[face_order[0]][0].shape[0]; A = len(angles_sorted)
        rays = np.empty((F,A,R,6), dtype=np.float32)
        vc = voxel_center.astype(np.float32)
        rot_x = {deg: _rot_x(deg) for deg in angles_sorted}
        rot_y = {deg: _rot_y(deg) for deg in angles_sorted}
        for f_idx, face in enumerate(face_order):
            O_base, D_base = grids[face]
            O_trans = O_base + vc
            axis = _ROT_AXIS[face]
            for a_idx, deg in enumerate(angles_sorted):
                Rmx = rot_y[deg] if axis == "y" else rot_x[deg]
                D_rot = (D_base @ Rmx.T).astype(np.float32)
                O_rot = ((O_trans - vc) @ Rmx.T + vc).astype(np.float32)
                rays[f_idx, a_idx, :, 0:3] = O_rot
                rays[f_idx, a_idx, :, 3:6] = D_rot
        return rays, face_order, angles_sorted

class MeshClipper:
    def __init__(self, leaf_mesh: Optional["trimesh.Trimesh"], wood_mesh: Optional["trimesh.Trimesh"]):
        trimesh, _ = _import_trimesh_pv()
        self.leaf_mesh = leaf_mesh
        self.wood_mesh = wood_mesh
        self.leaf_tri_min = None
        self.leaf_tri_max = None
        self.wood_tri_min = None
        self.wood_tri_max = None
        if self.leaf_mesh is not None and not self.leaf_mesh.is_empty:
            tris = self.leaf_mesh.triangles
            self.leaf_tri_min = tris.min(axis=1)
            self.leaf_tri_max = tris.max(axis=1)
        if self.wood_mesh is not None and not self.wood_mesh.is_empty:
            tris = self.wood_mesh.triangles
            self.wood_tri_min = tris.min(axis=1)
            self.wood_tri_max = tris.max(axis=1)
    @staticmethod
    def _clip_one(mesh, tri_min, tri_max, voxel_center, voxel_size):
        if mesh is None or mesh.is_empty or tri_min is None or tri_max is None:
            return np.empty((0,3), np.float32), np.empty((0,3), np.int32)
        _, pv = _import_trimesh_pv()
        half = float(voxel_size)/2.0
        minb = np.asarray(voxel_center) - half
        maxb = np.asarray(voxel_center) + half
        overlap = (tri_max >= minb).all(axis=1) & (tri_min <= maxb).all(axis=1)
        idx = np.flatnonzero(overlap)
        if idx.size == 0:
            return np.empty((0,3), np.float32), np.empty((0,3), np.int32)
        submesh = mesh.submesh([idx], append=True)
        pv_sub = pv.wrap(submesh)
        bounds6 = (minb[0], maxb[0], minb[1], maxb[1], minb[2], maxb[2])
        try:
            clipped = pv_sub.clip_box(bounds=bounds6, invert=False)
        except TypeError:
            cube = pv.Cube(center=voxel_center, x_length=voxel_size, y_length=voxel_size, z_length=voxel_size)
            clipped = pv_sub.clip_box(cube, invert=False)
        if hasattr(clipped, 'extract_surface'):
            clipped = clipped.extract_surface(algorithm=None)
        if getattr(clipped, 'is_all_triangles', True) is False:
            clipped = clipped.triangulate()
        V = np.asarray(clipped.points, dtype=np.float32)
        F = np.asarray(clipped.faces.reshape((-1,4))[:,1:], dtype=np.int32)
        return V, F
    def clip(self, voxel_center, voxel_size):
        lv, lf = self._clip_one(self.leaf_mesh, self.leaf_tri_min, self.leaf_tri_max, voxel_center, voxel_size)
        wv, wf = self._clip_one(self.wood_mesh, self.wood_tri_min, self.wood_tri_max, voxel_center, voxel_size)
        leaf = {"vertices": lv, "faces": lf} if lv.size and lf.size else {}
        wood = {"vertices": wv, "faces": wf} if wv.size and wf.size else {}
        return leaf, wood

# math helpers

def ray_box_intersection_vectorized(orig, dirs, bmin, bmax, eps=1e-12):
    safe = np.where(np.abs(dirs) < eps, np.where(dirs >= 0, eps, -eps), dirs)
    t1 = (bmin - orig) / safe
    t2 = (bmax - orig) / safe
    t_near = np.maximum.reduce(np.minimum(t1, t2), axis=1)
    t_far  = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    return t_near, t_far

def compute_efpl_array(d_arr, lambda_1):
    out = np.zeros_like(d_arr, dtype=np.float32)
    mask = d_arr > 0
    if lambda_1 == 0:
        out[mask] = d_arr[mask].astype(np.float32)
    else:
        valid = mask & (1.0 - lambda_1 * d_arr > 0)
        out[valid] = (-np.log(1.0 - lambda_1 * d_arr[valid]) / lambda_1).astype(np.float32)
    return out

# --- Multi-hit ω (omega) + λ MLE helpers ------------------------------------
def _build_pascal_table(n_max: int) -> np.ndarray:
    C = np.zeros((n_max + 1, n_max + 1), dtype=np.float64)
    C[0, 0] = 1.0
    for n in range(1, n_max + 1):
        C[n, 0] = 1.0
        for k in range(1, n + 1):
            C[n, k] = C[n - 1, k - 1] + (C[n - 1, k] if k <= n - 1 else 0.0)
    return C

def _precompute_B_omegas(omega_grid: np.ndarray, n_max: int, C: np.ndarray) -> np.ndarray:
    O = omega_grid.size
    B = np.zeros((O, n_max + 1, n_max + 1), dtype=np.float64)
    for j, om in enumerate(omega_grid):
        one_minus = 1.0 - om
        powers = one_minus ** np.arange(0, n_max + 1)  # power vector 0..n_max
        for n in range(1, n_max + 1):
            k_idx = np.arange(1, n + 1)
            # B[j, n, k] = C(n-1, k-1) * (1-ω)^(n-k)
            B[j, n, k_idx] = C[n - 1, k_idx - 1] * powers[n - k_idx]
    return B

def _compute_T_vec(ol: float, n_max: int) -> np.ndarray:
    """T[k] = ((ωλ)^k / k!) for k>=1; T[0] = 0."""
    T = np.zeros(n_max + 1, dtype=np.float64)
    if n_max >= 1:
        T[1] = ol
        for k in range(2, n_max + 1):
            T[k] = T[k - 1] * ol / k
    return T

class _LogPMFCache:
    """
    Cache of log P(n | λ_i, ω_j) for n=0..n_max over the (λ, ω) grid.
    Thread-safe build; reuse across calls.
    """
    def __init__(self, lam_grid: np.ndarray, omega_grid: np.ndarray, n_max: int):
        self.lam_grid = lam_grid.astype(np.float64)
        self.omega_grid = omega_grid.astype(np.float64)
        self.n_max = int(n_max)
        self.logP_all = None  # shape (n_max+1, L*O)
        self._built = False
        self._lock = threading.Lock()

    def build(self):
        with self._lock:
            if self._built:
                return
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
                    T = _compute_T_vec(ol, self.n_max)  # len n_max+1, T[0]=0
                    ssum = B[j] @ T  # (n_max+1,) with ssum[0]=0
                    p = np.empty(self.n_max + 1, dtype=np.float64)
                    p[0] = e_neg
                    if self.n_max >= 1:
                        p[1:] = e_neg * ssum[1:]
                    logP_all[:, idx] = np.log(np.maximum(p, eps))
                    idx += 1
            self.logP_all = logP_all
            self._built = True

def _build_histograms(counts_FAR: np.ndarray,
                      valid_mask_FAR: np.ndarray,
                      n_max: int) -> np.ndarray:
    F, A, R = counts_FAR.shape
    M = F * A
    counts = counts_FAR.reshape(M, R)
    mask = valid_mask_FAR.reshape(M, R)
    # Bin counts 0..n_max per (F,A)
    # Using one big bincount on flattened (row_offset + n) indices
    flat_indices = np.arange(M)[:, None] * (n_max + 1) + np.clip(counts, 0, n_max)
    flat_indices = flat_indices[mask]
    H = np.bincount(flat_indices, minlength=M * (n_max + 1)).reshape(M, n_max + 1)
    return H

def _mle_clustered_batched_np(counts_FAR: np.ndarray,
                              valid_mask_FAR: np.ndarray,
                              lam_grid: np.ndarray,
                              omega_grid: np.ndarray,
                              cache: _LogPMFCache | None = None):
    # Determine max observed count within valid rays
    n_max = int(np.max(np.where(valid_mask_FAR, counts_FAR, 0)))
    n_max = max(n_max, 0)
    if n_max == 0:
        # trivial: all 0 hits -> λ=0, any ω; choose ω mid-range for stability
        M = counts_FAR.shape[0] * counts_FAR.shape[1]
        return (np.zeros(M), np.full(M, float(np.median(omega_grid))), np.zeros(M), None)

    # Prepare cache
    if cache is None or cache.n_max != n_max \
       or cache.lam_grid.shape != lam_grid.shape or cache.omega_grid.shape != omega_grid.shape \
       or np.any(cache.lam_grid != lam_grid) or np.any(cache.omega_grid != omega_grid):
        cache = _LogPMFCache(lam_grid, omega_grid, n_max)
    cache.build()

    # Build histograms H[(F*A), n_max+1]
    H = _build_histograms(counts_FAR.astype(np.int32), valid_mask_FAR, n_max)   # (M, n_max+1)
    # Log-likelihood for all grid points: H dot logP
    LL_all = H.astype(np.float64) @ cache.logP_all                               # (M, L*O)
    best_idx = np.argmax(LL_all, axis=1)
    L = lam_grid.size
    O = omega_grid.size
    lam_idx = best_idx // O
    omg_idx = best_idx % O
    best_lambda = lam_grid[lam_idx]
    best_omega  = omega_grid[omg_idx]
    best_ll     = LL_all[np.arange(LL_all.shape[0]), best_idx]
    return best_lambda, best_omega, best_ll, cache

def compute_LIAD_from_o3d_mesh(mesh_legacy, num_bins=18):
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
    return bin_centers.astype(np.float32), liad.astype(np.float32), mean_angle

def _warp_smoke(device="cuda:0"):
    # one voxel centered at origin (1m box)
    vc = np.array([0,0,0], np.float32); vs = 1.0
    # small triangle inside the voxel
    tri_v = np.array([[0.0, 0.0, 0.2],
                    [0.1, 0.0, 0.2],
                    [0.0, 0.1, 0.2]], dtype=np.float32)
    tri_f = np.array([[0,1,2]], dtype=np.int32)

    caster = WarpRaycaster(device)
    # a single near-normal direction from +z face
    rays_FAR6, _, _ = RayGrid([vs], 0.05).build(vc, vs, [0.0001])
    rc = caster.raycast(vc, vs, rays_FAR6, {"vertices": tri_v, "faces": tri_f}, {})
    print("smoke counts_all sum:", rc["counts_all"].sum())  # should be > 0

# Raycasters
if _HAS_WARP:
    @wp.kernel
    def _raycast_voxel_kernel(
        leaf_id: wp.uint64,
        wood_id: wp.uint64,
        has_leaf: int, has_wood: int,
        origins: wp.array(dtype=wp.vec3),
        dirs: wp.array(dtype=wp.vec3),
        bmin: wp.vec3, bmax: wp.vec3,
        max_hits: int, eps_advance: float,
        first_any: wp.array(dtype=wp.float32),
        first_leaf: wp.array(dtype=wp.float32),
        first_wood: wp.array(dtype=wp.float32),
        count_all: wp.array(dtype=wp.int32),
        count_leaf: wp.array(dtype=wp.int32),
        count_wood: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        o = origins[i]; d = dirs[i]
        eps = 1.0e-12
        dx,dy,dz = d[0],d[1],d[2]
        tx1 = (bmin[0]-o[0]) / wp.where(wp.abs(dx)>eps, dx, wp.where(dx>=0.0, eps, -eps))
        tx2 = (bmax[0]-o[0]) / wp.where(wp.abs(dx)>eps, dx, wp.where(dx>=0.0, eps, -eps))
        ty1 = (bmin[1]-o[1]) / wp.where(wp.abs(dy)>eps, dy, wp.where(dy>=0.0, eps, -eps))
        ty2 = (bmax[1]-o[1]) / wp.where(wp.abs(dy)>eps, dy, wp.where(dy>=0.0, eps, -eps))
        tz1 = (bmin[2]-o[2]) / wp.where(wp.abs(dz)>eps, dz, wp.where(dz>=0.0, eps, -eps))
        tz2 = (bmax[2]-o[2]) / wp.where(wp.abs(dz)>eps, dz, wp.where(dz>=0.0, eps, -eps))
        tnear = wp.max(wp.min(tx1,tx2), wp.max(wp.min(ty1,ty2), wp.min(tz1,tz2)))
        tfar  = wp.min(wp.max(tx1,tx2), wp.min(wp.max(ty1,ty2), wp.max(tz1,tz2)))
        ok = (tfar >= 0.0) and (tnear <= tfar)
        SENT = wp.float32(1e32)
        if not ok:
            first_any[i] = SENT
            first_leaf[i] = SENT
            first_wood[i] = SENT
            return
        start = o + d * tnear
        remain = tfar - tnear
        first_any[i] = SENT
        first_leaf[i] = SENT
        first_wood[i] = SENT
        ca = int(0); cl = int(0); cw = int(0)
        t_accum = float(0.0)
        h = int(0)
        while h < max_hits and remain > 1.0e-6:
            leaf_found = False; wood_found = False
            t_leaf = SENT; t_wood = SENT
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
            hit_is_leaf = False; hit_is_wood = False
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
                break
            ca += 1
            if hit_is_leaf: cl += 1
            if hit_is_wood: cw += 1
            if first_any[i] > 1.0e31: first_any[i] = t_accum + t_first
            if leaf_found and first_leaf[i] > 1.0e31: first_leaf[i] = t_accum + t_leaf
            if wood_found and first_wood[i] > 1.0e31: first_wood[i] = t_accum + t_wood
            t_step = t_first + eps_advance
            start = start + d * t_step
            remain = remain - t_step
            t_accum = float(t_accum + t_step)
            h = int(h + 1)
        count_all[i] = ca; count_leaf[i] = cl; count_wood[i] = cw

class WarpRaycaster:
    def __init__(self, device_str: str, max_hits=8, eps_advance=1e-5, max_rays_per_batch=250_000):
        self.device = device_str
        self.max_hits = int(max_hits)
        self.eps_advance = float(eps_advance)
        self.max_rays_per_batch = int(max_rays_per_batch)
        import threading
        self._tls = threading.local()
    def _get_stream(self):
        try:
            return self._tls.stream
        except AttributeError:
            self._tls.stream = wp.Stream(device=self.device)
            return self._tls.stream
    def _wp_mesh(self, md):
        if not md or len(md.get("faces", [])) == 0: return None
        v = np.asarray(md.get("vertices", []), dtype=np.float32, order="C")
        f = np.asarray(md.get("faces", []), dtype=np.int32, order="C")
        if v.size == 0 or f.size == 0: return None

        
        # faces must be triangles, 0-based, (T,3) or flat T*3
        if f.ndim == 2 and f.shape[1] == 3:
            f_flat = f.reshape(-1)                     # flatten to 1-D
        elif f.ndim == 1 and (f.size % 3 == 0):
            f_flat = f
        else:
            raise ValueError(f"faces has unexpected shape {f.shape}; expected (T,3) or (T*3,)")
        
        # Enforce contiguity & dtypes
        v = np.ascontiguousarray(v, dtype=np.float32)
        f_flat = np.ascontiguousarray(f_flat, dtype=np.int32)

        stream = self._get_stream()
        try:
            with wp.ScopedDevice(self.device), wp.ScopedStream(stream):
                v_d = wp.array(v, dtype=wp.vec3, device=self.device)
                f_d = wp.array(f_flat, dtype=wp.int32, device=self.device)
                mesh = wp.Mesh(points=v_d, indices=f_d)
                # Synchronize to ensure mesh is built before scope exits
                wp.synchronize_stream(stream)
            
            # keep device arrays alive so GC can't free them prematurely
            try:
                keep = self._tls.keep
            except AttributeError:
                self._tls.keep = keep = []
            keep.append((mesh, v_d, f_d))

            return mesh
        except RuntimeError as e:
            if "allocate" in str(e).lower() or "illegal memory" in str(e).lower():
                wp.synchronize()
                gc.collect()
                return None
            raise
    
    def raycast(self, voxel_center, voxel_size, rays_FAR6, leaf_mesh, wood_mesh):
        F,A,R,_ = rays_FAR6.shape
        n = F*A*R
        stream = self._get_stream()
        
        # Pre-build mesh objects outside loop to catch allocation errors early
        leaf_wp = self._wp_mesh(leaf_mesh)
        wood_wp = self._wp_mesh(wood_mesh)
        
        with wp.ScopedDevice(self.device), wp.ScopedStream(stream):
            O = rays_FAR6[...,0:3].reshape(n,3).astype(np.float32)
            D = rays_FAR6[...,3:6].reshape(n,3).astype(np.float32)
            vc = np.asarray(voxel_center, dtype=np.float32)
            half = float(voxel_size)/2.0
            bmin = wp.vec3(vc[0]-half, vc[1]-half, vc[2]-half)
            bmax = wp.vec3(vc[0]+half, vc[1]+half, vc[2]+half)
            first_any = np.full(n, np.inf, np.float32)
            first_leaf = np.full(n, np.inf, np.float32)
            first_wood = np.full(n, np.inf, np.float32)
            cnt_all = np.zeros(n, np.int32)
            cnt_leaf = np.zeros(n, np.int32)
            cnt_wood = np.zeros(n, np.int32)

            # print("leaf has V/F:", leaf_wp is not None,
            #     "wood has V/F:", wood_wp is not None,
            #     "rays:", n, "remain>0 count (first batch):", np.count_nonzero((D[:10] != 0).any(axis=1)))

            leaf_id = leaf_wp.id if leaf_wp is not None else wp.uint64(0)
            wood_id = wood_wp.id if wood_wp is not None else wp.uint64(0)
            has_leaf = 1 if leaf_wp is not None else 0
            has_wood = 1 if wood_wp is not None else 0
            nb = (n + self.max_rays_per_batch - 1)//self.max_rays_per_batch
            for bi in range(nb):
                s = bi*self.max_rays_per_batch; e = min(s+self.max_rays_per_batch, n); B = e-s
                origins_d = wp.array(O[s:e], dtype=wp.vec3, device=self.device)
                dirs_d    = wp.array(D[s:e], dtype=wp.vec3, device=self.device)
                first_any_d  = wp.zeros(B, dtype=wp.float32, device=self.device)
                first_leaf_d = wp.zeros(B, dtype=wp.float32, device=self.device)
                first_wood_d = wp.zeros(B, dtype=wp.float32, device=self.device)
                count_all_d  = wp.zeros(B, dtype=wp.int32, device=self.device)
                count_leaf_d = wp.zeros(B, dtype=wp.int32, device=self.device)
                count_wood_d = wp.zeros(B, dtype=wp.int32, device=self.device)
                wp.launch(_raycast_voxel_kernel, dim=B, inputs=[
                    leaf_id, wood_id, has_leaf, has_wood,
                    origins_d, dirs_d, bmin, bmax,
                    int(self.max_hits), float(self.eps_advance),
                    first_any_d, first_leaf_d, first_wood_d,
                    count_all_d, count_leaf_d, count_wood_d
                ], device=self.device)
                wp.synchronize_stream(stream)
                first_any[s:e] = first_any_d.numpy(); first_leaf[s:e] = first_leaf_d.numpy(); first_wood[s:e] = first_wood_d.numpy()
                cnt_all[s:e] = count_all_d.numpy(); cnt_leaf[s:e] = count_leaf_d.numpy(); cnt_wood[s:e] = count_wood_d.numpy()
                # print(f"Processed rays {s} to {e} / {n}: first_any [{first_any[s:e].min():.4f}, {first_any[s:e].max():.4f}], first_leaf [{first_leaf[s:e].min():.4f}, {first_leaf[s:e].max():.4f}], first_wood [{first_wood[s:e].min():.4f}, {first_wood[s:e].max():.4f}], counts_all [{cnt_all[s:e].min()}, {cnt_all[s:e].max()}], counts_leaf [{cnt_leaf[s:e].min()}, {cnt_leaf[s:e].max()}], counts_wood [{cnt_wood[s:e].min()}, {cnt_wood[s:e].max()}]")
                
                del origins_d, dirs_d, first_any_d, first_leaf_d, first_wood_d, count_all_d, count_leaf_d, count_wood_d
                gc.collect()
        return {
            "first_hit_any": first_any.reshape(F,A,R),
            "first_hit_leaf": first_leaf.reshape(F,A,R),
            "first_hit_wood": first_wood.reshape(F,A,R),
            "counts_all": cnt_all.reshape(F,A,R),
            "counts_leaf": cnt_leaf.reshape(F,A,R),
            "counts_wood": cnt_wood.reshape(F,A,R),
            "F": F, "A": A, "R": R,
        }


class O3DRaycaster:
    def __init__(self):
        pass
    @staticmethod
    def _to_o3d(md):
        if not md or len(md.get("faces", [])) == 0 or len(md.get("vertices", [])) == 0:
            return None
        o3d, _ = _import_open3d()
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(md["vertices"], dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(md["faces"], dtype=np.int32))
        return m
    def raycast(self, voxel_center, voxel_size, rays_FAR6, leaf_mesh, wood_mesh):
        o3d, o3c = _import_open3d()
        F,A,R,_ = rays_FAR6.shape
        scene = o3d.t.geometry.RaycastingScene(device=o3c.Device("CPU:0"))
        leaf_id = None; wood_id = None
        leaf_o3d = self._to_o3d(leaf_mesh); wood_o3d = self._to_o3d(wood_mesh)
        if leaf_o3d is not None and len(leaf_o3d.triangles) > 0:
            leaf_t = o3d.t.geometry.TriangleMesh.from_legacy(leaf_o3d)
            leaf_id = scene.add_triangles(leaf_t)
        if wood_o3d is not None and len(wood_o3d.triangles) > 0:
            wood_t = o3d.t.geometry.TriangleMesh.from_legacy(wood_o3d)
            wood_id = scene.add_triangles(wood_t)
        rays_t = o3c.Tensor(rays_FAR6.astype(np.float32), dtype=o3c.float32)
        ans = scene.list_intersections(rays_t)
        first_any = np.full((F,A,R), np.inf, np.float32)
        first_leaf = np.full((F,A,R), np.inf, np.float32)
        first_wood = np.full((F,A,R), np.inf, np.float32)
        counts_all = np.zeros((F,A,R), np.int32)
        counts_leaf = np.zeros((F,A,R), np.int32)
        counts_wood = np.zeros((F,A,R), np.int32)
        if isinstance(ans, dict) and ("geometry_ids" in ans):
            geom_ids = ans["geometry_ids"].numpy().astype(np.int64)
            ray_ids  = ans.get("ray_ids", None)
            t_hits   = ans.get("t_hit", None)
            if ray_ids is not None and t_hits is not None:
                ray_ids = ray_ids.numpy().astype(np.int64)
                t_hits  = t_hits.numpy().astype(np.float32)
                Ar = A*R
                f_idx = (ray_ids // Ar).astype(np.int64)
                a_idx = ((ray_ids % Ar)//R).astype(np.int64)
                r_idx = (ray_ids % R).astype(np.int64)
                np.minimum.at(first_any, (f_idx, a_idx, r_idx), t_hits)
                counts = np.bincount(ray_ids, minlength=F*A*R).astype(np.int32)
                counts_all[...] = counts.reshape(F,A,R)
                if leaf_id is not None:
                    m = (geom_ids == int(leaf_id))
                    if m.any():
                        np.minimum.at(first_leaf, (f_idx[m], a_idx[m], r_idx[m]), t_hits[m])
                        counts = np.bincount(ray_ids[m], minlength=F*A*R).astype(np.int32)
                        counts_leaf[...] = counts.reshape(F,A,R)
                if wood_id is not None:
                    m = (geom_ids == int(wood_id))
                    if m.any():
                        np.minimum.at(first_wood, (f_idx[m], a_idx[m], r_idx[m]), t_hits[m])
                        counts = np.bincount(ray_ids[m], minlength=F*A*R).astype(np.int32)
                        counts_wood[...] = counts.reshape(F,A,R)
        else:
            counts = scene.count_intersections(rays_t).numpy().astype(np.int32).reshape(F,A,R)
            counts_all = counts
        return {
            "first_hit_any": first_any,
            "first_hit_leaf": first_leaf,
            "first_hit_wood": first_wood,
            "counts_all": counts_all,
            "counts_leaf": counts_leaf,
            "counts_wood": counts_wood,
            "F": F, "A": A, "R": R,
        }

class StatComputer:
    def __init__(self, angles_order: Sequence[float],
                 lam_grid: np.ndarray | None = None,
                 omega_grid: np.ndarray | None = None):
        self.angles_order = list(angles_order)
        self.lam_grid = lam_grid if lam_grid is not None else np.linspace(1e-3, 3.0, 60)
        self.omega_grid = omega_grid if omega_grid is not None else np.linspace(1e-3, 0.999, 60)
        self._pmf_cache_all = None  # type: _LogPMFCache | None
        self._pmf_cache_leaf = None
        self._pmf_cache_wood = None
        self._cache_lock = threading.Lock()

    @staticmethod
    def _o3d_from_dict(md):
        if not md or len(md.get("faces", [])) == 0 or len(md.get("vertices", [])) == 0:
            return None
        o3d, _ = _import_open3d()
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(md["vertices"], dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(md["faces"], dtype=np.int32))
        return m
    def mesh_metrics(self, voxel_center, voxel_size, leaf_md, wood_md, num_angle_bins=18):
        leaf_o3d = self._o3d_from_dict(leaf_md)
        wood_o3d = self._o3d_from_dict(wood_md)
        leaf_area = float(leaf_o3d.get_surface_area()) if leaf_o3d is not None and len(leaf_o3d.triangles) else 0.0
        wood_area = float(wood_o3d.get_surface_area()) if wood_o3d is not None and len(wood_o3d.triangles) else 0.0
        all_area  = leaf_area + wood_area
        LAI = leaf_area/(voxel_size**2) if voxel_size > 0 else 0.0
        WAI = wood_area/(voxel_size**2) if voxel_size > 0 else 0.0
        PAI = all_area /(voxel_size**2) if voxel_size > 0 else 0.0
        LAD = leaf_area/(voxel_size**3) if voxel_size > 0 else 0.0
        WAD = wood_area/(voxel_size**3) if voxel_size > 0 else 0.0
        PAD = all_area /(voxel_size**3) if voxel_size > 0 else 0.0
        leaf_fraction_ref = (LAI/PAI) if PAI>0 else 0.0
        bin_leaf, liad, _ = compute_LIAD_from_o3d_mesh(leaf_o3d, num_bins=num_angle_bins)
        bin_wood, wiad, _ = compute_LIAD_from_o3d_mesh(wood_o3d, num_bins=num_angle_bins)
        if liad.size and wiad.size:
            piad = (liad*LAD + wiad*WAD)/(LAD+WAD) if (LAD+WAD)>0 else np.array([])
            bin_all = bin_leaf
        elif liad.size:
            piad = liad; bin_all = bin_leaf
        elif wiad.size:
            piad = wiad; bin_all = bin_wood
        else:
            piad = np.array([]); bin_all = np.array([])
        lambda_1 = 0.0
        meta = {
            "voxel_cx": float(voxel_center[0]),
            "voxel_cy": float(voxel_center[1]),
            "voxel_cz": float(voxel_center[2]),
            "voxel_size": float(voxel_size),
            "alpha": float('nan'),
            "LAI_ref": LAI, "WAI_ref": WAI, "PAI_ref": PAI,
            "LAD_ref": LAD, "WAD_ref": WAD, "PAD_ref": PAD,
            "leaf_fraction": leaf_fraction_ref,
            "liad_json": json.dumps(liad.tolist()) if liad.size else "NA",
            "wiad_json": json.dumps(wiad.tolist()) if wiad.size else "NA",
            "piad_json": json.dumps(piad.tolist()) if piad.size else "NA",
        }
        try:
            meta["liad_dewit"] = classify_liad_to_dewit(liad, bin_leaf, return_scores=True)[0] if (liad.size and bin_leaf.size) else "NA"
            meta["wiad_dewit"] = classify_liad_to_dewit(wiad, bin_wood, return_scores=True)[0] if (wiad.size and bin_wood.size) else "NA"
            meta["piad_dewit"] = classify_liad_to_dewit(piad, bin_all, return_scores=True)[0] if (piad.size and bin_all.size) else "NA"
        except Exception:
            meta.setdefault("liad_dewit", "NA"); meta.setdefault("wiad_dewit", "NA"); meta.setdefault("piad_dewit", "NA")
        aux = {"bin_leaf": bin_leaf, "bin_wood": bin_wood, "bin_all": bin_all, "liad": liad, "wiad": wiad, "piad": piad, "lambda_1": lambda_1}
        return meta, aux
    def compute(self, voxel_center, voxel_size, rays_FAR6, rc, mesh_meta, aux):
        F,A,R = rc["F"], rc["A"], rc["R"]
        f_any  = rc.get("first_hit_any"); f_leaf = rc.get("first_hit_leaf"); f_wood = rc.get("first_hit_wood")
        c_all  = rc.get("counts_all")
        c_leaf = rc.get("counts_leaf") if rc.get("counts_leaf") is not None else np.zeros_like(c_all)
        c_wood = rc.get("counts_wood") if rc.get("counts_wood") is not None else np.zeros_like(c_all)
        O = rays_FAR6[...,0:3].reshape(-1,3); D = rays_FAR6[...,3:6].reshape(-1,3)
        bmin = voxel_center - (voxel_size/2.0); bmax = voxel_center + (voxel_size/2.0)
        t_near, t_far = ray_box_intersection_vectorized(O,D,bmin,bmax)
        t_near = t_near.reshape(F,A,R); t_far = t_far.reshape(F,A,R)
        valid_mask = (t_near <= t_far) & (t_far >= 0.0)
        path_len = np.zeros_like(t_far, np.float32); path_len[valid_mask] = (t_far[valid_mask] - t_near[valid_mask]).astype(np.float32)
        # IMPORTANT: derive hits from counts arrays
        valid_any  = (c_all  > 0) & valid_mask if c_all  is not None else np.zeros_like(valid_mask, bool)
        valid_leaf = (c_leaf > 0) & valid_mask
        valid_wood = (c_wood > 0) & valid_mask
        free_any  = path_len.copy(); free_leaf = path_len.copy(); free_wood = path_len.copy()
        if f_any  is not None: free_any [valid_any ] = f_any [valid_any ]
        if f_leaf is not None: free_leaf[valid_leaf] = f_leaf[valid_leaf]
        if f_wood is not None: free_wood[valid_wood] = f_wood[valid_wood]
        lambda_1 = float(aux.get("lambda_1", 0.0))
        efpl_any  = np.zeros_like(free_any , np.float32); efpl_any [valid_mask] = compute_efpl_array(free_any [valid_mask], lambda_1)
        efpl_leaf = np.zeros_like(free_leaf, np.float32); efpl_leaf[valid_mask] = compute_efpl_array(free_leaf[valid_mask], lambda_1)
        efpl_wood = np.zeros_like(free_wood, np.float32); efpl_wood[valid_mask] = compute_efpl_array(free_wood[valid_mask], lambda_1)
        N      = valid_mask.sum(axis=2).astype(np.int32)
        n_all  = valid_any.sum(axis=2).astype(np.int32)
        n_leaf = valid_leaf.sum(axis=2).astype(np.int32)
        n_wood = valid_wood.sum(axis=2).astype(np.int32)
        sum_ppl = path_len.sum(axis=2)
        mean_ppl = np.divide(sum_ppl, N, out=np.zeros_like(sum_ppl), where=(N>0))
        sum_fpl_all  = free_any .sum(axis=2); mean_fpl_all  = np.divide(sum_fpl_all , N, out=np.zeros_like(sum_fpl_all ), where=(N>0))
        sum_fpl_leaf = free_leaf.sum(axis=2); mean_fpl_leaf = np.divide(sum_fpl_leaf, N, out=np.zeros_like(sum_fpl_leaf), where=(N>0))
        sum_fpl_wood = free_wood.sum(axis=2); mean_fpl_wood = np.divide(sum_fpl_wood, N, out=np.zeros_like(sum_fpl_wood), where=(N>0))
        sum_efpl_all  = efpl_any .sum(axis=2); sum_efpl_hits_all  = (efpl_any *valid_any ).sum(axis=2)
        sum_efpl_leaf = efpl_leaf.sum(axis=2); sum_efpl_hits_leaf = (efpl_leaf*valid_leaf).sum(axis=2)
        sum_efpl_wood = efpl_wood.sum(axis=2); sum_efpl_hits_wood = (efpl_wood*valid_wood).sum(axis=2)
        safeN = np.maximum(N,1)
        mean_count_all = np.divide(np.where(valid_mask, c_all, 0).sum(axis=2), safeN)
        sum_x2_all     = (np.where(valid_mask, c_all, 0)**2).sum(axis=2)
        var_count_all  = np.zeros_like(mean_count_all, np.float32)
        has2 = (N>1); var_count_all[has2] = (sum_x2_all[has2] - (mean_count_all[has2]*safeN[has2]))/(safeN[has2]-1)
        dirs0 = rays_FAR6[:, :, 0, 3:6]
        norms = np.linalg.norm(dirs0, axis=2, keepdims=True)
        dnorm = dirs0 / np.maximum(norms, 1e-12)
        dz = dnorm[:,:,2]
        viewing_angles = np.degrees(np.arccos(np.clip(np.abs(dz), 0.0, 1.0))).astype(np.float32)
        bin_all = aux.get("bin_all", np.array([])); liad = aux.get("liad", np.array([])); wiad = aux.get("wiad", np.array([])); piad = aux.get("piad", np.array([]))
        bin_leaf = aux.get("bin_leaf", np.array([])); bin_wood = aux.get("bin_wood", np.array([]))
        G_all = np.zeros_like(viewing_angles, np.float32); G_leaf = np.zeros_like(viewing_angles, np.float32); G_wood = np.zeros_like(viewing_angles, np.float32)
        if piad.size and bin_all.size:  G_all  = calculate_G(viewing_angles.ravel(), bin_all , piad).reshape(viewing_angles.shape).astype(np.float32)
        if liad.size and bin_leaf.size: G_leaf = calculate_G(viewing_angles.ravel(), bin_leaf, liad).reshape(viewing_angles.shape).astype(np.float32)
        if wiad.size and bin_wood.size: G_wood = calculate_G(viewing_angles.ravel(), bin_wood, wiad).reshape(viewing_angles.shape).astype(np.float32)
        pgap_all = 1.0 - (n_all / np.maximum(N,1))
        pgap_leaf = 1.0 - (n_leaf / np.maximum(N,1))
        pgap_wood = 1.0 - (n_wood / np.maximum(N,1))
        LAD_ref = mesh_meta.get("LAD_ref", 0.0); WAD_ref = mesh_meta.get("WAD_ref", 0.0); PAD_ref = mesh_meta.get("PAD_ref", 0.0)
        CI_all  = np.full((F,A), np.nan, np.float32)
        CI_leaf = np.full((F,A), np.nan, np.float32)
        CI_wood = np.full((F,A), np.nan, np.float32)
        valid_ci = (pgap_all>0)&(pgap_all<1)&(G_all>0)&(mean_ppl>0)
        if PAD_ref>0:  CI_all [valid_ci] = (-np.log(pgap_all [valid_ci])/(G_all [valid_ci]*PAD_ref*mean_ppl[valid_ci])).astype(np.float32)
        valid_ci = (pgap_leaf>0)&(pgap_leaf<1)&(G_leaf>0)&(mean_ppl>0)
        if LAD_ref>0: CI_leaf[valid_ci] = (-np.log(pgap_leaf[valid_ci])/(G_leaf[valid_ci]*LAD_ref*mean_ppl[valid_ci])).astype(np.float32)
        valid_ci = (pgap_wood>0)&(pgap_wood<1)&(G_wood>0)&(mean_ppl>0)
        if WAD_ref>0: CI_wood[valid_ci] = (-np.log(pgap_wood[valid_ci])/(G_wood[valid_ci]*WAD_ref*mean_ppl[valid_ci])).astype(np.float32)
        
        # Multi-hit approach stats
        def _reshape_FA(x_flat):
            return x_flat.reshape(F, A).astype(np.float32)

        # ALL
        lam_hat_all = np.zeros((F, A), np.float32)
        omg_hat_all = np.zeros((F, A), np.float32)
        ll_hat_all  = np.zeros((F, A), np.float32)
        if c_all is not None:
            lam_hat_flat, omg_hat_flat, ll_hat_flat, self._pmf_cache_all = _mle_clustered_batched_np(
                c_all, valid_mask, self.lam_grid, self.omega_grid, cache=self._pmf_cache_all
            )
            lam_hat_all = _reshape_FA(lam_hat_flat)
            omg_hat_all = _reshape_FA(omg_hat_flat)
            ll_hat_all  = _reshape_FA(ll_hat_flat)

        # LEAF
        lam_hat_leaf = np.zeros((F, A), np.float32)
        omg_hat_leaf = np.zeros((F, A), np.float32)
        ll_hat_leaf  = np.zeros((F, A), np.float32)
        if c_leaf is not None:
            lam_hat_flat, omg_hat_flat, ll_hat_flat, self._pmf_cache_leaf = _mle_clustered_batched_np(
                c_leaf, valid_mask, self.lam_grid, self.omega_grid, cache=self._pmf_cache_leaf
            )
            lam_hat_leaf = _reshape_FA(lam_hat_flat)
            omg_hat_leaf = _reshape_FA(omg_hat_flat)
            ll_hat_leaf  = _reshape_FA(ll_hat_flat)

        # WOOD
        lam_hat_wood = np.zeros((F, A), np.float32)
        omg_hat_wood = np.zeros((F, A), np.float32)
        ll_hat_wood  = np.zeros((F, A), np.float32)
        if c_wood is not None:
            lam_hat_flat, omg_hat_flat, ll_hat_flat, self._pmf_cache_wood = _mle_clustered_batched_np(
                c_wood, valid_mask, self.lam_grid, self.omega_grid, cache=self._pmf_cache_wood
            )
            lam_hat_wood = _reshape_FA(lam_hat_flat)
            omg_hat_wood = _reshape_FA(omg_hat_flat)
            ll_hat_wood  = _reshape_FA(ll_hat_flat)    
        
        stats = {
            "dx": dnorm[:,:,0].astype(np.float32), "dy": dnorm[:,:,1].astype(np.float32), "dz": dnorm[:,:,2].astype(np.float32),
            "zenith_angle": np.round(viewing_angles.astype(np.float32), 1), "num_rays": N.astype(np.int32),
            "num_hits_all": n_all.astype(np.int32), "num_hits_leaf": n_leaf.astype(np.int32), "num_hits_wood": n_wood.astype(np.int32),
            "pgap_all": pgap_all.astype(np.float32), "pgap_leaf": pgap_leaf.astype(np.float32), "pgap_wood": pgap_wood.astype(np.float32),
            "G_all": G_all.astype(np.float32), "G_leaf": G_leaf.astype(np.float32), "G_wood": G_wood.astype(np.float32),
            "CI_all": CI_all.astype(np.float32), "CI_leaf": CI_leaf.astype(np.float32), "CI_wood": CI_wood.astype(np.float32),
            "lambda_hat_all": lam_hat_all, "lambda_hat_leaf": lam_hat_leaf, "lambda_hat_wood": lam_hat_wood,
            "omega_all": omg_hat_all, "omega_leaf": omg_hat_leaf, "omega_wood": omg_hat_wood,
            "ll_all": ll_hat_all, "ll_leaf": ll_hat_leaf, "ll_wood": ll_hat_wood,
            "sum_ppl": sum_ppl.astype(np.float32), "mean_ppl": mean_ppl.astype(np.float32),
            "sum_fpl_all": sum_fpl_all.astype(np.float32), "mean_fpl_all": mean_fpl_all.astype(np.float32),
            "sum_fpl_leaf": sum_fpl_leaf.astype(np.float32), "mean_fpl_leaf": mean_fpl_leaf.astype(np.float32),
            "sum_fpl_wood": sum_fpl_wood.astype(np.float32), "mean_fpl_wood": mean_fpl_wood.astype(np.float32),
            "sum_efpl_all": sum_efpl_all.astype(np.float32), "sum_efpl_hits_all": sum_efpl_hits_all.astype(np.float32),
            "sum_efpl_leaf": sum_efpl_leaf.astype(np.float32), "sum_efpl_hits_leaf": sum_efpl_hits_leaf.astype(np.float32),
            "sum_efpl_wood": sum_efpl_wood.astype(np.float32), "sum_efpl_hits_wood": sum_efpl_hits_wood.astype(np.float32),
            "mean_count_all": mean_count_all.astype(np.float32), "var_count_all": var_count_all.astype(np.float32),
        }
        metadata = dict(mesh_meta)
        return metadata, stats

# Simple geometry loader (leaf/wood split by tags or existing *_leaf/_wood files)

def load_and_split_by_group(scene_file: str, leaf_keys: Sequence[str], wood_keys: Sequence[str]):
    trimesh, _ = _import_trimesh_pv()
    scene_file = str(scene_file)
    leaf_path = scene_file.replace(".obj", "_leaf.obj")
    wood_path = scene_file.replace(".obj", "_wood.obj")
    leaf_mesh = None; wood_mesh = None
    if os.path.exists(leaf_path):
        try:
            leaf_mesh = trimesh.load_mesh(leaf_path, process=False)
            print(f"Loaded existing leaf mesh: {leaf_path}")
        except Exception as e:
            print(f"[warn] Failed to load {leaf_path}: {e}")
    if os.path.exists(wood_path):
        try:
            wood_mesh = trimesh.load_mesh(wood_path, process=False)
            print(f"Loaded existing wood mesh: {wood_path}")
        except Exception as e:
            print(f"[warn] Failed to load {wood_path}: {e}")
    if leaf_mesh is None or wood_mesh is None:
        verts: List[List[float]] = []
        leaf_faces: List[List[int]] = []
        wood_faces: List[List[int]] = []
        leaf_keys_l = [k.lower() for k in leaf_keys]
        wood_keys_l = [k.lower() for k in wood_keys]
        current_tag = ""
        with open(scene_file, 'r', errors='ignore') as f:
            for line in f:
                if line.startswith('v '):
                    verts.append([float(c) for c in line.split()[1:4]])
                elif line.startswith(('g ', 'o ')):
                    current_tag = line[2:].strip().lower()
                elif line.startswith('f '):
                    face = [int(tok.split('/')[0]) - 1 for tok in line.split()[1:]]
                    if any(k in current_tag for k in leaf_keys_l):
                        leaf_faces.append(face)
                    elif any(k in current_tag for k in wood_keys_l):
                        wood_faces.append(face)
        verts = np.asarray(verts, dtype=np.float64)
        if leaf_mesh is None and leaf_faces:
            leaf_mesh = trimesh.Trimesh(vertices=verts, faces=leaf_faces, process=False)
            try: leaf_mesh.export(leaf_path)
            except Exception: pass
        if wood_mesh is None and wood_faces:
            wood_mesh = trimesh.Trimesh(vertices=verts, faces=wood_faces, process=False)
            try: wood_mesh.export(wood_path)
            except Exception: pass
    # bounds (union)
    bounds_list = []
    if leaf_mesh is not None and hasattr(leaf_mesh, 'bounds'): bounds_list.append(leaf_mesh.bounds)
    if wood_mesh is not None and hasattr(wood_mesh, 'bounds'): bounds_list.append(wood_mesh.bounds)
    if bounds_list:
        minb = np.min([b[0] for b in bounds_list], axis=0)
        maxb = np.max([b[1] for b in bounds_list], axis=0)
        bounds6 = tuple(np.concatenate([minb, maxb]).tolist())
    else:
        scene = trimesh.load_mesh(scene_file, process=False)
        bounds6 = tuple(scene.bounds.flatten().tolist())
    return leaf_mesh, wood_mesh, bounds6

# pipeline main

def generate_voxel_centers(voxel_size: float, bounds6):
    minx, miny, minz, maxx, maxy, maxz = bounds6
    xs = np.arange(minx + voxel_size/2.0, maxx + voxel_size/2.0, voxel_size)
    ys = np.arange(miny + voxel_size/2.0, maxy + voxel_size/2.0, voxel_size)
    zs = np.arange(minz + voxel_size/2.0, maxz + voxel_size/2.0, voxel_size)
    centers = np.array(np.meshgrid(xs, ys, zs, indexing='ij')).reshape(3,-1).T.astype(np.float32)
    coords = ((centers*11 + voxel_size*73)*13).astype(int)
    vids = np.char.add(np.char.add(coords[:,0].astype(str),'_'), np.char.add(coords[:,1].astype(str),'_'))
    vids = np.char.add(vids, coords[:,2].astype(str))
    return centers, vids

def filter_voxel_centers(centers, leaf_bounds6, wood_bounds6, voxel_size):
    min_leaf, max_leaf = np.array(leaf_bounds6[0:3]), np.array(leaf_bounds6[3:6])
    min_wood, max_wood = np.array(wood_bounds6[0:3]), np.array(wood_bounds6[3:6])
    leaf_mask = np.all((centers >= (min_leaf - voxel_size/2.0)) & (centers <= (max_leaf + voxel_size/2.0)), axis=1)
    wood_mask = np.all((centers >= (min_wood - voxel_size/2.0)) & (centers <= (max_wood + voxel_size/2.0)), axis=1)
    return centers[leaf_mask | wood_mask]

# ---- Worker threads ----
STOP = object()

class WriterThread(threading.Thread):
    def __init__(self, write_q: Queue, csv_path: str):
        super().__init__(daemon=True)
        self.write_q = write_q
        self.writer = CSVWriter(csv_path, metadata_keys_order=[
            "voxel_cx","voxel_cy","voxel_cz","voxel_size","voxel_id","face_index","face_name",
            "angle_index","angle_center_deg","alpha","LAI_ref","WAI_ref","PAI_ref","LAD_ref","WAD_ref","PAD_ref",
            "leaf_fraction","liad_dewit","wiad_dewit","piad_dewit","liad_json","wiad_json","piad_json",
        ])
        self.written_voxels = 0
    def run(self):
        while True:
            item = self.write_q.get()
            if item is STOP:
                self.write_q.task_done(); break
            rows = item
            try:
                self.writer.write_rows(rows)
                self.written_voxels += 1
            finally:
                self.write_q.task_done()
        self.writer.close()

class RaycastWorker(threading.Thread):
    def __init__(self, name: str, raycaster, in_q: Queue, out_q: Queue):
        super().__init__(daemon=True, name=name)
        self.raycaster = raycaster
        self.in_q = in_q
        self.out_q = out_q
    def run(self):
        while True:
            job = self.in_q.get()
            if job is STOP:
                self.in_q.task_done(); break
            try:
                vc, vs, leaf_md, wood_md, rays_FAR6, face_order, angles_sorted, mesh_meta, aux, voxel_id = job
                rc = self.raycaster.raycast(vc, vs, rays_FAR6, leaf_md, wood_md)
                self.out_q.put((job, rc))
            except Exception as e:
                print(f"[raycast:{self.name}] error: {e}\n{traceback.format_exc()}")
            finally:
                self.in_q.task_done()

class StatsWorker(threading.Thread):
    def __init__(self, name: str, stat_eng, in_q: Queue, write_q: Queue):
        super().__init__(daemon=True, name=name)
        self.stat_eng = stat_eng
        self.in_q = in_q
        self.write_q = write_q
    def run(self):
        while True:
            item = self.in_q.get()
            if item is STOP:
                self.in_q.task_done(); break
            try:
                (vc, vs, leaf_md, wood_md, rays_FAR6, face_order, angles_sorted, mesh_meta, aux, voxel_id), rc = item
                metadata, stats = self.stat_eng.compute(vc, vs, rays_FAR6, rc, mesh_meta, aux)
                rows = build_rows(metadata, stats, face_order=face_order, angles=angles_sorted, voxel_id=voxel_id)
                self.write_q.put(rows)
            except Exception as e:
                print(f"[stats:{self.name}] error: {e}\n{traceback.format_exc()}")
            finally:
                self.in_q.task_done()

# --------------
# Main
# --------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Process voxel batch.")
    parser.add_argument("scene_file", type=str, help="Path to the single .obj scene file. This will automatically extract leaf and wood meshes.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=[0.2,0.5,1.0,2.0], help="Voxel sizes for processing.")
    parser.add_argument("--num_angle_bins", type=int, default=18, help="Number of angle bins for ray tracing.")
    parser.add_argument("--ray_spacing", type=float, default=0.005, help="Ray spacing.")
    parser.add_argument("--wood_volume_voxel_size", type=float, default=0.01, help="(unused placeholder, preserved)")
    parser.add_argument("--wood_volume_threshold", type=int, default=4, help="(unused placeholder, preserved)")
    parser.add_argument("--max_workers", type=int, default=32, help="Max CPU workers for clipping/prep.")
    parser.add_argument("--gpu_workers", type=int, default=4, help="Number of concurrent Warp raycasting threads.")
    parser.add_argument("--stats_workers", type=int, default=0, help="Number of stats threads (default: CPUs//2).")
    parser.add_argument("--prep_workers", type=int, default=0, help="Workers for prep stage (0->CPUs//2)")
    parser.add_argument("--force_cpu", action='store_true', help="Force CPU raycasting (for testing/debugging).")
    parser.add_argument("--debug", action='store_true', help="Verbose errors.")
    args = parser.parse_args(argv)

    print(f"Scene: {args.scene_file}")
    print(f"Voxel sizes: {args.voxel_sizes}")
    print(f"Angle bins: {args.num_angle_bins}")
    print(f"Ray spacing: {args.ray_spacing}")
    print(f"Max workers (CPU clip/prep): {args.max_workers}")
    print(f"GPU workers: {args.gpu_workers}")
    print(f"Force CPU raycasting: {args.force_cpu}")
    # Device
    dev = DeviceManager(prefer_cuda=True)
    raycaster = WarpRaycaster(dev.device_str) if dev.using_warp and not args.force_cpu else O3DRaycaster()

    # Load meshes
    leaf_keys = ["leaf","leaves","leafs"]
    wood_keys = ["wood","trunk","branch","stem"]
    leaf_mesh, wood_mesh, bounds6 = load_and_split_by_group(args.scene_file, leaf_keys, wood_keys)
    leaf_bounds6 = tuple(leaf_mesh.bounds.flatten().tolist()) if leaf_mesh is not None else (0,0,0,0,0,0)
    wood_bounds6 = tuple(wood_mesh.bounds.flatten().tolist()) if wood_mesh is not None else (0,0,0,0,0,0)

    # CPU capacity
    if os.environ.get("SLURM_CPUS_PER_TASK") is None:
        num_cpus = psutil.cpu_count(logical=True)
    else:
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)))*2
    num_cpus = max(1, num_cpus)
    n_clip_workers = min(args.max_workers, num_cpus)
    n_stats_workers = args.stats_workers if args.stats_workers>0 else max(1, num_cpus//4)
    n_prep_workers = args.prep_workers if args.prep_workers>0 else max(1, num_cpus//2)

    for vs in args.voxel_sizes:
        # Angles
        edges = np.linspace(0, 90, args.num_angle_bins + 1)
        centers_deg = np.round((edges[:-1] + edges[1:]) / 2.0, 3)
        angle_set = set(centers_deg.tolist())
        if 0 in angle_set: angle_set.remove(0); angle_set.add(0.0001)
        if 90 in angle_set: angle_set.remove(90); angle_set.add(89.9999)
        ANGLE_ORDER = sorted(angle_set)
        # Rays helper
        raygrid = RayGrid([vs], args.ray_spacing)
        # Voxel centers
        centers, _ = generate_voxel_centers(vs, bounds6)
        centers = filter_voxel_centers(centers, leaf_bounds6, wood_bounds6, vs)
        if centers.size == 0:
            print(f"[skip] No voxel centers for size {vs}")
            continue
        # Clipping (parallel)
        clipper = MeshClipper(leaf_mesh, wood_mesh)
        def _clip(vc):
            lm, wm = clipper.clip(vc, vs)
            return (vc, lm, wm)
        from joblib import Parallel, delayed
        print(f"Voxel size {vs}: clipping {len(centers)} voxels ...")
        clipped = []
        with tqdm(total=len(centers), desc='Clipping', unit='voxel') as pbar:
            for result in Parallel(n_jobs=n_clip_workers, backend='threading')(delayed(_clip)(vc) for vc in centers):
                clipped.append(result)
                pbar.update(1)
        vox_data = [(vc,lm,wm) for (vc,lm,wm) in clipped if (lm or wm)]
        if not vox_data:
            print(f"[skip] No geometry within voxels for size {vs}")
            continue
        # Prep jobs (sequential small CPU): mesh metrics + rays
        stat_eng = StatComputer(angles_order=ANGLE_ORDER)
        jobs: List[tuple] = []
        print("Preparing jobs ...")
        def _prep_one(vc, lm, wm):
            mesh_meta, aux = stat_eng.mesh_metrics(vc, vs, lm, wm, num_angle_bins=args.num_angle_bins)
            rays_FAR6, face_order, angles_sorted = raygrid.build(vc, vs, ANGLE_ORDER)
            voxel_id = create_voxel_id(voxel_size=vs, x=vc[0], y=vc[1], z=vc[2])
            return (vc, vs, lm, wm, rays_FAR6, face_order, angles_sorted, mesh_meta, aux, voxel_id)
        jobs = []
        with tqdm(total=len(vox_data), desc="Prep", unit="voxel") as pbar:
            for result in Parallel(n_jobs=n_prep_workers, backend="threading")(
                delayed(_prep_one)(vc, lm, wm) for (vc, lm, wm) in vox_data
            ):
                jobs.append(result)
                pbar.update(1)
            
        # Queues and threads
        raycast_q: Queue = Queue(maxsize=2* (args.gpu_workers if dev.using_warp else 1))
        stats_q: Queue = Queue(maxsize=4*n_stats_workers)
        write_q: Queue = Queue(maxsize=16)
        import datetime as dt        
        out_csv = os.path.join(os.path.dirname(args.scene_file), os.path.basename(args.scene_file).replace('.obj', f'_results_{vs}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv'))
        writer_thr = WriterThread(write_q, out_csv)
        writer_thr.start()
        # Raycast workers: O3D -> single worker; Warp -> args.gpu_workers
        ray_workers: List[threading.Thread] = []
        n_ray_workers = 1 if not dev.using_warp else max(1, int(args.gpu_workers))
        for i in range(n_ray_workers):
            rw = RaycastWorker(name=f"ray-{i}", raycaster=raycaster, in_q=raycast_q, out_q=stats_q)
            rw.start(); ray_workers.append(rw)
        # Stats workers
        stat_workers: List[threading.Thread] = []
        for i in range(n_stats_workers):
            sw = StatsWorker(name=f"stat-{i}", stat_eng=stat_eng, in_q=stats_q, write_q=write_q)
            sw.start(); stat_workers.append(sw)
        # Progress
        pbar = tqdm(total=len(jobs), desc='Process', unit='voxel')
        # Feeder loop
        for job in jobs:
            raycast_q.put(job)
        # Drain loop: update when writer outputs a voxel
        produced = 0
        while produced < len(jobs):
            prev = writer_thr.written_voxels
            time.sleep(0.1)
            now = writer_thr.written_voxels
            if now > prev:
                pbar.update(now - prev)
                produced = now
        # Stop: signal queues
        for _ in ray_workers: raycast_q.put(STOP)
        for w in ray_workers: w.join()
        for _ in stat_workers: stats_q.put(STOP)
        for w in stat_workers: w.join()
        write_q.put(STOP)
        writer_thr.join()
        # Ensure progress bar reflects final count
        final_delta = writer_thr.written_voxels - produced
        if final_delta > 0:
            pbar = tqdm(total=len(jobs), desc='Process', unit='voxel')
            pbar.update(len(jobs))
        pbar.close()
        print(f"Saved results to {out_csv}.")
    print("All done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
