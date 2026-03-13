# -*- coding: utf-8 -*-
"""
Refactored voxel processing script (CPU/GPU raytracing with shared stats path)
-----------------------------------------------------------------------------
* Keeps the same CLI flags as the existing workflow.
* Uses clear classes for device selection, ray generation, clipping, raycasting, and
  statistics computation.
* Falls back cleanly to Open3D CPU raycasting if CUDA/Warp is unavailable.
* Writes one CSV per voxel_size (naming: <scene>_results_<voxel_size>.csv).
* Prints concise progress updates with progress bars.

Requirements (as in your environment):
    numpy, pandas, open3d, trimesh, pyvista, tqdm, psutil, joblib
    (optional) warp (NVIDIA CUDA). If not present/usable, CPU path is used.

External utils expected (unchanged):
    from utils import classify_liad_to_dewit, calculate_G, resolve_cuda_index, create_voxel_id

Author: Refactor by Copilot
Date: 2026-03-12
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from tqdm import tqdm
import psutil

# Defer heavy imports to runtime to allow packaging without env errors

def _import_open3d():
    import open3d as o3d
    import open3d.core as o3c
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)
    return o3d, o3c

def _import_trimesh_pv():
    import trimesh
    import pyvista as pv
    return trimesh, pv

# ---- External project utilities (kept the same) ----
try:
    from utils import classify_liad_to_dewit, calculate_G, resolve_cuda_index, create_voxel_id
except Exception as _e:
    print("[warn] Could not import project utils. Ensure utils.py is on PYTHONPATH.")
    raise

# ---- Optional NVIDIA Warp (for CUDA raytracing) ----
_HAS_WARP = False
try:
    import warp as wp  # type: ignore
    wp.init()
    _HAS_WARP = True
except Exception:
    _HAS_WARP = False

# -------------------------
# CSV Writer (simple)
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
        ordered.extend(rest)
        import csv as _csv
        self._fieldnames = ordered
        self._writer = _csv.DictWriter(self._file, fieldnames=self._fieldnames)
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
    import numpy as _np
    if isinstance(v, (_np.floating, _np.integer, _np.bool_)):
        return v.item()
    if isinstance(v, _np.ndarray):
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

FACE_ORDER = ["bottom", "top", "xplus", "xminus", "yplus", "yminus"]
_ROT_AXIS = {"xplus": "y", "xminus": "y", "bottom": "x", "top": "x", "yplus": "x", "yminus": "x"}


def _grid(face_extent: float, ray_spacing: float, offset: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    s = np.arange(-face_extent / 2.0, face_extent / 2.0 + ray_spacing, ray_spacing)
    if offset:
        s += ray_spacing / 2.0
    return np.meshgrid(s, s, indexing="xy")


def _rot_x(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float32)


def _rot_y(deg: float) -> np.ndarray:
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32)


class DeviceManager:
    def __init__(self, prefer_cuda: bool = True):
        self.prefer_cuda = prefer_cuda
        self.cuda_index: Optional[int] = None
        self.cuda_uuid: Optional[str] = None
        self.using_warp: bool = False
        self.device_str: str = "CPU:0"
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
            grids = {
                "bottom": self._face_bottom(vs, XX, YY),
                "xplus": self._face_xplus(vs, XX, YY),
                "yplus": self._face_yplus(vs, XX, YY)
            }
            XX, YY = _grid(face_len, ray_spacing, offset=True)
            grids.update({
                "top": self._face_top(vs, XX, YY),
                "xminus": self._face_xminus(vs, XX, YY),
                "yminus": self._face_yminus(vs, XX, YY)
            })
            self.cache[float(vs)] = grids

    @staticmethod
    def _face_bottom(vs, XX, YY):
        z = -vs * 2
        org = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z, dtype=np.float32)])
        dir = np.tile([0, 0, 1], (len(org), 1)).astype(np.float32)
        return org.astype(np.float32), dir

    @staticmethod
    def _face_top(vs, XX, YY):
        z = +vs * 2
        org = np.column_stack([XX.ravel(), YY.ravel(), np.full(XX.size, z, dtype=np.float32)])
        dir = np.tile([0, 0, -1], (len(org), 1)).astype(np.float32)
        return org.astype(np.float32), dir

    @staticmethod
    def _face_xplus(vs, XX, YY):
        x = +vs * 2
        org = np.column_stack([np.full(YY.size, x, dtype=np.float32), XX.ravel(), YY.ravel()])
        dir = np.tile([-1, 0, 0], (len(org), 1)).astype(np.float32)
        return org.astype(np.float32), dir

    @staticmethod
    def _face_xminus(vs, XX, YY):
        x = -vs * 2
        org = np.column_stack([np.full(YY.size, x, dtype=np.float32), XX.ravel(), YY.ravel()])
        dir = np.tile([1, 0, 0], (len(org), 1)).astype(np.float32)
        return org.astype(np.float32), dir

    @staticmethod
    def _face_yplus(vs, XX, YY):
        y = +vs * 2
        org = np.column_stack([XX.ravel(), np.full(XX.size, y, dtype=np.float32), YY.ravel()])
        dir = np.tile([0, -1, 0], (len(org), 1)).astype(np.float32)
        return org.astype(np.float32), dir

    @staticmethod
    def _face_yminus(vs, XX, YY):
        y = -vs * 2
        org = np.column_stack([XX.ravel(), np.full(XX.size, y, dtype=np.float32), YY.ravel()])
        dir = np.tile([0, 1, 0], (len(org), 1)).astype(np.float32)
        return org.astype(np.float32), dir

    def build(self, voxel_center: np.ndarray, voxel_size: float, angles: Sequence[float]) -> Tuple[np.ndarray, List[str], List[float]]:
        grids = self.cache[float(voxel_size)]
        face_order = FACE_ORDER
        angles_sorted = list(sorted(angles))
        F = len(face_order)
        R = grids[face_order[0]][0].shape[0]
        A = len(angles_sorted)
        rays = np.empty((F, A, R, 6), dtype=np.float32)
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
    def _clip_one(mesh: Optional["trimesh.Trimesh"], tri_min: Optional[np.ndarray], tri_max: Optional[np.ndarray],
                  voxel_center: np.ndarray, voxel_size: float) -> Tuple[np.ndarray, np.ndarray]:
        if mesh is None or mesh.is_empty or tri_min is None or tri_max is None:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
        _, pv = _import_trimesh_pv()
        half = float(voxel_size) / 2.0
        minb = np.asarray(voxel_center) - half
        maxb = np.asarray(voxel_center) + half
        overlap = (tri_max >= minb).all(axis=1) & (tri_min <= maxb).all(axis=1)
        idx = np.flatnonzero(overlap)
        if idx.size == 0:
            return np.empty((0, 3), np.float32), np.empty((0, 3), np.int32)
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
        F = np.asarray(clipped.faces.reshape((-1, 4))[:, 1:], dtype=np.int32)
        return V, F

    def clip(self, voxel_center: np.ndarray, voxel_size: float) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        lv, lf = self._clip_one(self.leaf_mesh, self.leaf_tri_min, self.leaf_tri_max, voxel_center, voxel_size)
        wv, wf = self._clip_one(self.wood_mesh, self.wood_tri_min, self.wood_tri_max, voxel_center, voxel_size)
        leaf = {"vertices": lv, "faces": lf} if lv.size and lf.size else {}
        wood = {"vertices": wv, "faces": wf} if wv.size and wf.size else {}
        return leaf, wood


def load_and_split_by_group(scene_file: Union[str, str], leaf_keys: Sequence[str], wood_keys: Sequence[str]) -> Tuple[Optional["trimesh.Trimesh"], Optional["trimesh.Trimesh"], Tuple[float, float, float, float, float, float]]:
    trimesh, _ = _import_trimesh_pv()
    scene_file = str(scene_file)
    leaf_path = scene_file.replace(".obj", "_leaf.obj")
    wood_path = scene_file.replace(".obj", "_wood.obj")
    leaf_mesh = None
    wood_mesh = None
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
        with open(scene_file, "r", errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(c) for c in line.split()[1:4]])
                elif line.startswith(("g ", "o ")):
                    current_tag = line[2:].strip().lower()
                elif line.startswith("f "):
                    face = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
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
    bounds_list = []
    if leaf_mesh is not None and hasattr(leaf_mesh, "bounds"):
        bounds_list.append(leaf_mesh.bounds)
    if wood_mesh is not None and hasattr(wood_mesh, "bounds"):
        bounds_list.append(wood_mesh.bounds)
    if bounds_list:
        minb = np.min([b[0] for b in bounds_list], axis=0)
        maxb = np.max([b[1] for b in bounds_list], axis=0)
        bounds6 = tuple(np.concatenate([minb, maxb]).tolist())
    else:
        scene = trimesh.load_mesh(scene_file, process=False)
        bounds6 = tuple(scene.bounds.flatten().tolist())
    return leaf_mesh, wood_mesh, bounds6


def ray_box_intersection_vectorized(orig: np.ndarray, dirs: np.ndarray, bmin: np.ndarray, bmax: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    safe = np.where(np.abs(dirs) < eps, np.where(dirs >= 0, eps, -eps), dirs)
    t1 = (bmin - orig) / safe
    t2 = (bmax - orig) / safe
    t_near = np.maximum.reduce(np.minimum(t1, t2), axis=1)
    t_far = np.minimum.reduce(np.maximum(t1, t2), axis=1)
    return t_near, t_far


def compute_efpl_array(d_arr: np.ndarray, lambda_1: float) -> np.ndarray:
    out = np.zeros_like(d_arr, dtype=np.float32)
    mask = d_arr > 0
    if lambda_1 == 0:
        out[mask] = d_arr[mask].astype(np.float32)
    else:
        valid = mask & (1.0 - lambda_1 * d_arr > 0)
        out[valid] = (-np.log(1.0 - lambda_1 * d_arr[valid]) / lambda_1).astype(np.float32)
    return out


def compute_LIAD_from_o3d_mesh(mesh_legacy: Optional["o3d.geometry.TriangleMesh"], num_bins: int = 18) -> Tuple[np.ndarray, np.ndarray, float]:
    if mesh_legacy is None:
        return np.array([]), np.array([]), float("nan")
    o3d, _ = _import_open3d()
    if len(mesh_legacy.triangles) == 0:
        return np.array([]), np.array([]), float("nan")
    verts = np.asarray(mesh_legacy.vertices)
    tris = np.asarray(mesh_legacy.triangles)
    v0 = verts[tris[:, 1]] - verts[tris[:, 0]]
    v1 = verts[tris[:, 2]] - verts[tris[:, 0]]
    cp = np.cross(v0, v1)
    areas = 0.5 * np.linalg.norm(cp, axis=1)
    norms = np.linalg.norm(cp, axis=1, keepdims=True)
    normals = np.divide(cp, norms, where=(norms != 0), out=np.zeros_like(cp))
    ang = np.degrees(np.arccos(np.clip(normals[:, 2], -1, 1)))
    ang = np.where(ang > 90, 180 - ang, ang)
    ang = 90.0 - ang
    mean_angle = float(np.nanmean(ang)) if ang.size else float("nan")
    bin_edges = np.linspace(0, 90, num_bins + 1)
    idx = np.digitize(ang, bin_edges) - 1
    idx = np.clip(idx, 0, num_bins - 1)
    bin_counts = np.bincount(idx, weights=areas, minlength=num_bins)
    total_area = areas.sum()
    liad = bin_counts / total_area if total_area > 0 else np.zeros(num_bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    return bin_centers.astype(np.float32), liad.astype(np.float32), mean_angle


if _HAS_WARP:
    import warp as wp
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
        count_wood: wp.array(dtype=wp.int32),
    ):
        i = wp.tid()
        o = origins[i]
        d = dirs[i]
        eps = 1.0e-12
        dx = d[0]; dy = d[1]; dz = d[2]
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
            first_any[i] = wp.float32(1e32)
            first_leaf[i] = wp.float32(1e32)
            first_wood[i] = wp.float32(1e32)
            return
        start = o + d * tnear
        remain = tfar - tnear
        first_any[i] = wp.float32(1e32)
        first_leaf[i] = wp.float32(1e32)
        first_wood[i] = wp.float32(1e32)
        ca = int(0); cl = int(0); cw = int(0)
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
            t_first = wp.float32(1e32)
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
            else:
                hit_is_wood = True
                t_first = t_wood
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

class WarpRaycaster:
    def __init__(self, device_str: str, max_hits: int = 8, eps_advance: float = 1e-5, max_rays_per_batch: int = 1_000_000):
        if not _HAS_WARP:
            raise RuntimeError("Warp is not available")
        self.device = device_str
        self.max_hits = int(max_hits)
        self.eps_advance = float(eps_advance)
        self.max_rays_per_batch = int(max_rays_per_batch)

    def _wp_mesh(self, md: Optional[Dict[str, np.ndarray]]):
        if not md or len(md.get("faces", [])) == 0:
            return None
        v = np.asarray(md["vertices"], dtype=np.float32, order="C")
        f = np.asarray(md["faces"], dtype=np.int32, order="C").ravel()
        if v.size == 0 or f.size == 0:
            return None
        v_d = wp.array(v, dtype=wp.vec3, device=self.device)
        f_d = wp.array(f, dtype=wp.int32, device=self.device)
        return wp.Mesh(points=v_d, indices=f_d)

    def raycast(self, voxel_center: np.ndarray, voxel_size: float, rays_FAR6: np.ndarray,
                leaf_mesh: Optional[Dict[str, np.ndarray]], wood_mesh: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        F, A, R, _ = rays_FAR6.shape
        n_rays = F * A * R
        O = rays_FAR6[..., 0:3].reshape(n_rays, 3).astype(np.float32)
        D = rays_FAR6[..., 3:6].reshape(n_rays, 3).astype(np.float32)
        vc = np.asarray(voxel_center, dtype=np.float32)
        half = float(voxel_size) * 0.5
        bmin = wp.vec3(vc[0] - half, vc[1] - half, vc[2] - half)
        bmax = wp.vec3(vc[0] + half, vc[1] + half, vc[2] + half)
        first_any = np.full(n_rays, np.inf, dtype=np.float32)
        first_leaf = np.full(n_rays, np.inf, dtype=np.float32)
        first_wood = np.full(n_rays, np.inf, dtype=np.float32)
        cnt_all = np.zeros(n_rays, dtype=np.int32)
        cnt_leaf = np.zeros(n_rays, dtype=np.int32)
        cnt_wood = np.zeros(n_rays, dtype=np.int32)
        leaf_wp = self._wp_mesh(leaf_mesh)
        wood_wp = self._wp_mesh(wood_mesh)
        leaf_id = leaf_wp.id if leaf_wp is not None else wp.uint64(0)
        wood_id = wood_wp.id if wood_wp is not None else wp.uint64(0)
        has_leaf = 1 if leaf_wp is not None else 0
        has_wood = 1 if wood_wp is not None else 0
        nb = (n_rays + self.max_rays_per_batch - 1) // self.max_rays_per_batch
        for bi in range(nb):
            s = bi * self.max_rays_per_batch
            e = min(s + self.max_rays_per_batch, n_rays)
            B = e - s
            origins_d = wp.array(O[s:e], dtype=wp.vec3, device=self.device)
            dirs_d = wp.array(D[s:e], dtype=wp.vec3, device=self.device)
            first_any_d = wp.zeros(B, dtype=float, device=self.device)
            first_leaf_d = wp.zeros(B, dtype=float, device=self.device)
            first_wood_d = wp.zeros(B, dtype=float, device=self.device)
            count_all_d = wp.zeros(B, dtype=wp.int32, device=self.device)
            count_leaf_d = wp.zeros(B, dtype=wp.int32, device=self.device)
            count_wood_d = wp.zeros(B, dtype=wp.int32, device=self.device)
            wp.launch(
                kernel=_raycast_voxel_kernel,
                dim=B,
                inputs=[leaf_id, wood_id, has_leaf, has_wood, origins_d, dirs_d, bmin, bmax, int(self.max_hits), float(self.eps_advance), first_any_d, first_leaf_d, first_wood_d, count_all_d, count_leaf_d, count_wood_d],
                device=self.device,
            )
            first_any[s:e] = first_any_d.numpy()
            first_leaf[s:e] = first_leaf_d.numpy()
            first_wood[s:e] = first_wood_d.numpy()
            cnt_all[s:e] = count_all_d.numpy()
            cnt_leaf[s:e] = count_leaf_d.numpy()
            cnt_wood[s:e] = count_wood_d.numpy()
            del origins_d, dirs_d, first_any_d, first_leaf_d, first_wood_d, count_all_d, count_leaf_d, count_wood_d
            gc.collect()
        return {
            "first_hit_any": first_any.reshape(F, A, R),
            "first_hit_leaf": first_leaf.reshape(F, A, R),
            "first_hit_wood": first_wood.reshape(F, A, R),
            "counts_all": cnt_all.reshape(F, A, R),
            "counts_leaf": cnt_leaf.reshape(F, A, R),
            "counts_wood": cnt_wood.reshape(F, A, R),
            "F": F, "A": A, "R": R,
        }

class O3DRaycaster:
    def __init__(self):
        pass
    @staticmethod
    def _to_o3d(md: Optional[Dict[str, np.ndarray]]) -> Optional["o3d.geometry.TriangleMesh"]:
        if not md or len(md.get("faces", [])) == 0 or len(md.get("vertices", [])) == 0:
            return None
        o3d, _ = _import_open3d()
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(md["vertices"], dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(md["faces"], dtype=np.int32))
        return m
    def raycast(self, voxel_center: np.ndarray, voxel_size: float, rays_FAR6: np.ndarray,
                leaf_mesh: Optional[Dict[str, np.ndarray]], wood_mesh: Optional[Dict[str, np.ndarray]]) -> Dict[str, np.ndarray]:
        o3d, o3c = _import_open3d()
        F, A, R, _ = rays_FAR6.shape
        scene = o3d.t.geometry.RaycastingScene(device=o3c.Device("CPU:0"))
        leaf_id = None
        wood_id = None
        leaf_o3d = self._to_o3d(leaf_mesh)
        wood_o3d = self._to_o3d(wood_mesh)
        if leaf_o3d is not None and len(leaf_o3d.triangles) > 0:
            leaf_t = o3d.t.geometry.TriangleMesh.from_legacy(leaf_o3d)
            leaf_id = scene.add_triangles(leaf_t)
        if wood_o3d is not None and len(wood_o3d.triangles) > 0:
            wood_t = o3d.t.geometry.TriangleMesh.from_legacy(wood_o3d)
            wood_id = scene.add_triangles(wood_t)
        rays_t = o3c.Tensor(rays_FAR6.astype(np.float32), dtype=o3c.float32)
        ans = scene.list_intersections(rays_t)
        first_any = np.full((F, A, R), np.inf, dtype=np.float32)
        first_leaf = np.full((F, A, R), np.inf, dtype=np.float32)
        first_wood = np.full((F, A, R), np.inf, dtype=np.float32)
        counts_all = np.zeros((F, A, R), dtype=np.int32)
        counts_leaf = np.zeros((F, A, R), dtype=np.int32)
        counts_wood = np.zeros((F, A, R), dtype=np.int32)
        if isinstance(ans, dict) and ("geometry_ids" in ans):
            geom_ids = ans["geometry_ids"].numpy().astype(np.int64)
            ray_ids = ans.get("ray_ids", None)
            t_hits = ans.get("t_hit", None)
            if ray_ids is not None and t_hits is not None:
                ray_ids = ray_ids.numpy().astype(np.int64)
                t_hits = t_hits.numpy().astype(np.float32)
                Ar = A * R
                f_idx = (ray_ids // Ar).astype(np.int64)
                a_idx = ((ray_ids % Ar) // R).astype(np.int64)
                r_idx = (ray_ids % R).astype(np.int64)
                np.minimum.at(first_any, (f_idx, a_idx, r_idx), t_hits)
                counts = np.bincount(ray_ids, minlength=F * A * R).astype(np.int32)
                counts_all[...] = counts.reshape(F, A, R)
                if leaf_id is not None:
                    m = (geom_ids == int(leaf_id))
                    if m.any():
                        np.minimum.at(first_leaf, (f_idx[m], a_idx[m], r_idx[m]), t_hits[m])
                        counts = np.bincount(ray_ids[m], minlength=F * A * R).astype(np.int32)
                        counts_leaf[...] = counts.reshape(F, A, R)
                if wood_id is not None:
                    m = (geom_ids == int(wood_id))
                    if m.any():
                        np.minimum.at(first_wood, (f_idx[m], a_idx[m], r_idx[m]), t_hits[m])
                        counts = np.bincount(ray_ids[m], minlength=F * A * R).astype(np.int32)
                        counts_wood[...] = counts.reshape(F, A, R)
        else:
            counts = scene.count_intersections(rays_t).numpy().astype(np.int32).reshape(F, A, R)
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
    def __init__(self, angles_order: Sequence[float]):
        self.angles_order = list(angles_order)
    @staticmethod
    def _o3d_from_dict(md: Optional[Dict[str, np.ndarray]]) -> Optional["o3d.geometry.TriangleMesh"]:
        o3d, _ = _import_open3d()
        if not md or len(md.get("faces", [])) == 0 or len(md.get("vertices", [])) == 0:
            return None
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(np.asarray(md["vertices"], dtype=np.float64))
        m.triangles = o3d.utility.Vector3iVector(np.asarray(md["faces"], dtype=np.int32))
        return m
    def mesh_metrics(self, voxel_center: np.ndarray, voxel_size: float,
                     leaf_md: Optional[Dict[str, np.ndarray]], wood_md: Optional[Dict[str, np.ndarray]],
                     num_angle_bins: int = 18) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        o3d, _ = _import_open3d()
        leaf_o3d = self._o3d_from_dict(leaf_md)
        wood_o3d = self._o3d_from_dict(wood_md)
        leaf_area = float(leaf_o3d.get_surface_area()) if leaf_o3d is not None and len(leaf_o3d.triangles) else 0.0
        wood_area = float(wood_o3d.get_surface_area()) if wood_o3d is not None and len(wood_o3d.triangles) else 0.0
        all_area = leaf_area + wood_area
        LAI = leaf_area / (voxel_size ** 2) if voxel_size > 0 else 0.0
        WAI = wood_area / (voxel_size ** 2) if voxel_size > 0 else 0.0
        PAI = all_area / (voxel_size ** 2) if voxel_size > 0 else 0.0
        LAD = leaf_area / (voxel_size ** 3) if voxel_size > 0 else 0.0
        WAD = wood_area / (voxel_size ** 3) if voxel_size > 0 else 0.0
        PAD = all_area / (voxel_size ** 3) if voxel_size > 0 else 0.0
        leaf_fraction_ref = (LAI / PAI) if PAI > 0 else 0.0
        bin_leaf, liad, _ = compute_LIAD_from_o3d_mesh(leaf_o3d, num_bins=num_angle_bins)
        bin_wood, wiad, _ = compute_LIAD_from_o3d_mesh(wood_o3d, num_bins=num_angle_bins)
        if liad.size and wiad.size:
            piad = (liad * LAD + wiad * WAD) / (LAD + WAD) if (LAD + WAD) > 0 else np.array([])
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
            "alpha": float("nan"),
            "LAI_ref": LAI, "WAI_ref": WAI, "PAI_ref": PAI,
            "LAD_ref": LAD, "WAD_ref": WAD, "PAD_ref": PAD,
            "leaf_fraction": leaf_fraction_ref,
            "liad_json": json.dumps(liad.tolist()) if liad.size else "NA",
            "wiad_json": json.dumps(wiad.tolist()) if wiad.size else "NA",
            "piad_json": json.dumps(piad.tolist()) if piad.size else "NA",
        }
        aux = {"bin_leaf": bin_leaf, "bin_wood": bin_wood, "bin_all": bin_all, "liad": liad, "wiad": wiad, "piad": piad, "lambda_1": lambda_1}
        try:
            if liad.size and bin_leaf.size:
                liad_dew, _, _ = classify_liad_to_dewit(liad, bin_leaf, return_scores=True); meta["liad_dewit"] = liad_dew
            else: meta["liad_dewit"] = "NA"
            if wiad.size and bin_wood.size:
                wiad_dew, _, _ = classify_liad_to_dewit(wiad, bin_wood, return_scores=True); meta["wiad_dewit"] = wiad_dew
            else: meta["wiad_dewit"] = "NA"
            if piad.size and bin_all.size:
                piad_dew, _, _ = classify_liad_to_dewit(piad, bin_all, return_scores=True); meta["piad_dewit"] = piad_dew
            else: meta["piad_dewit"] = "NA"
        except Exception:
            meta.setdefault("liad_dewit", "NA"); meta.setdefault("wiad_dewit", "NA"); meta.setdefault("piad_dewit", "NA")
        return meta, aux
    def compute(self, voxel_center: np.ndarray, voxel_size: float, rays_FAR6: np.ndarray,
                rc: Dict[str, np.ndarray], mesh_meta: Dict[str, Any], aux: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        F, A, R = rc["F"], rc["A"], rc["R"]
        f_any = rc.get("first_hit_any"); f_leaf = rc.get("first_hit_leaf"); f_wood = rc.get("first_hit_wood")
        c_all = rc.get("counts_all"); c_leaf = rc.get("counts_leaf"); c_wood = rc.get("counts_wood")
        O = rays_FAR6[..., 0:3].reshape(-1, 3); D = rays_FAR6[..., 3:6].reshape(-1, 3)
        bmin = voxel_center - (voxel_size / 2.0); bmax = voxel_center + (voxel_size / 2.0)
        t_near, t_far = ray_box_intersection_vectorized(O, D, bmin, bmax)
        t_near = t_near.reshape(F, A, R); t_far = t_far.reshape(F, A, R)
        valid_mask = (t_near <= t_far) & (t_far >= 0.0)
        path_len = np.zeros_like(t_far, dtype=np.float32); path_len[valid_mask] = (t_far[valid_mask] - t_near[valid_mask]).astype(np.float32)
        valid_any = (c_all > 0) & valid_mask
        valid_leaf = (c_leaf > 0) & valid_mask
        valid_wood = (c_wood > 0) & valid_mask
        free_any = path_len.copy(); free_leaf = path_len.copy(); free_wood = path_len.copy()
        if f_any is not None: free_any[valid_any] = f_any[valid_any]
        if f_leaf is not None: free_leaf[valid_leaf] = f_leaf[valid_leaf]
        if f_wood is not None: free_wood[valid_wood] = f_wood[valid_wood]
        lambda_1 = float(aux.get("lambda_1", 0.0))
        efpl_any = np.zeros_like(free_any, dtype=np.float32); efpl_leaf = np.zeros_like(free_leaf, dtype=np.float32); efpl_wood = np.zeros_like(free_wood, dtype=np.float32)
        efpl_any[valid_mask] = compute_efpl_array(free_any[valid_mask], lambda_1)
        efpl_leaf[valid_mask] = compute_efpl_array(free_leaf[valid_mask], lambda_1)
        efpl_wood[valid_mask] = compute_efpl_array(free_wood[valid_mask], lambda_1)
        N = valid_mask.sum(axis=2).astype(np.int32)
        n_all = valid_any.sum(axis=2).astype(np.int32); n_leaf = valid_leaf.sum(axis=2).astype(np.int32); n_wood = valid_wood.sum(axis=2).astype(np.int32)
        sum_ppl = path_len.sum(axis=2); mean_ppl = np.divide(sum_ppl, N, out=np.zeros_like(sum_ppl), where=(N>0))
        sum_fpl_all = free_any.sum(axis=2); sum_fpl_leaf = free_leaf.sum(axis=2); sum_fpl_wood = free_wood.sum(axis=2)
        mean_fpl_all = np.divide(sum_fpl_all, N, out=np.zeros_like(sum_fpl_all), where=(N>0))
        mean_fpl_leaf = np.divide(sum_fpl_leaf, N, out=np.zeros_like(sum_fpl_leaf), where=(N>0))
        mean_fpl_wood = np.divide(sum_fpl_wood, N, out=np.zeros_like(sum_fpl_wood), where=(N>0))
        sum_efpl_all = efpl_any.sum(axis=2); sum_efpl_leaf = efpl_leaf.sum(axis=2); sum_efpl_wood = efpl_wood.sum(axis=2)
        sum_efpl_hits_all = (efpl_any*valid_any).sum(axis=2); sum_efpl_hits_leaf = (efpl_leaf*valid_leaf).sum(axis=2); sum_efpl_hits_wood = (efpl_wood*valid_wood).sum(axis=2)
        safeN = np.maximum(N, 1)
        mean_count_all = np.divide(np.where(valid_mask, c_all, 0).sum(axis=2), safeN)
        sum_x2_all = (np.where(valid_mask, c_all, 0)**2).sum(axis=2)
        var_count_all = np.zeros_like(mean_count_all, dtype=np.float32)
        has2 = (N>1); var_count_all[has2] = (sum_x2_all[has2] - (mean_count_all[has2]*safeN[has2])) / (safeN[has2]-1)
        dirs0 = rays_FAR6[:, :, 0, 3:6]; norms = np.linalg.norm(dirs0, axis=2, keepdims=True); dnorm = dirs0/np.maximum(norms, 1e-12)
        dz = dnorm[:, :, 2]; viewing_angles = np.degrees(np.arccos(np.clip(np.abs(dz), 0.0, 1.0))).astype(np.float32)
        bin_all = aux.get("bin_all", np.array([])); liad = aux.get("liad", np.array([])); wiad = aux.get("wiad", np.array([])); piad = aux.get("piad", np.array([]))
        bin_leaf = aux.get("bin_leaf", np.array([])); bin_wood = aux.get("bin_wood", np.array([]))
        G_all = np.zeros_like(viewing_angles, dtype=np.float32); G_leaf = np.zeros_like(viewing_angles, dtype=np.float32); G_wood = np.zeros_like(viewing_angles, dtype=np.float32)
        if piad.size and bin_all.size: G_all[...] = calculate_G(viewing_angles.ravel(), bin_all, piad).reshape(viewing_angles.shape).astype(np.float32)
        if liad.size and bin_leaf.size: G_leaf[...] = calculate_G(viewing_angles.ravel(), bin_leaf, liad).reshape(viewing_angles.shape).astype(np.float32)
        if wiad.size and bin_wood.size: G_wood[...] = calculate_G(viewing_angles.ravel(), bin_wood, wiad).reshape(viewing_angles.shape).astype(np.float32)
        pgap_all = 1.0 - (n_all/np.maximum(N, 1)); pgap_leaf = 1.0 - (n_leaf/np.maximum(N, 1)); pgap_wood = 1.0 - (n_wood/np.maximum(N, 1))
        LAD_ref = mesh_meta.get("LAD_ref", 0.0); WAD_ref = mesh_meta.get("WAD_ref", 0.0); PAD_ref = mesh_meta.get("PAD_ref", 0.0)
        CI_all = np.full((F,A), np.nan, dtype=np.float32); CI_leaf = np.full((F,A), np.nan, dtype=np.float32); CI_wood = np.full((F,A), np.nan, dtype=np.float32)
        valid_ci = (pgap_all>0)&(pgap_all<1)&(G_all>0)&(mean_ppl>0)
        if PAD_ref>0: CI_all[valid_ci] = (-np.log(pgap_all[valid_ci])/(G_all[valid_ci]*PAD_ref*mean_ppl[valid_ci])).astype(np.float32)
        valid_ci = (pgap_leaf>0)&(pgap_leaf<1)&(G_leaf>0)&(mean_ppl>0)
        if LAD_ref>0: CI_leaf[valid_ci] = (-np.log(pgap_leaf[valid_ci])/(G_leaf[valid_ci]*LAD_ref*mean_ppl[valid_ci])).astype(np.float32)
        valid_ci = (pgap_wood>0)&(pgap_wood<1)&(G_wood>0)&(mean_ppl>0)
        if WAD_ref>0: CI_wood[valid_ci] = (-np.log(pgap_wood[valid_ci])/(G_wood[valid_ci]*WAD_ref*mean_ppl[valid_ci])).astype(np.float32)
        stats = {
            "dx": dnorm[:, :, 0].astype(np.float32),
            "dy": dnorm[:, :, 1].astype(np.float32),
            "dz": dnorm[:, :, 2].astype(np.float32),
            "zenith_angle": viewing_angles.astype(np.float32),
            "num_rays": N.astype(np.int32),
            "sum_ppl": sum_ppl.astype(np.float32),
            "mean_ppl": mean_ppl.astype(np.float32),
            "num_hits_all": n_all.astype(np.int32),
            "num_hits_leaf": n_leaf.astype(np.int32),
            "num_hits_wood": n_wood.astype(np.int32),
            "pgap_all": pgap_all.astype(np.float32),
            "pgap_leaf": pgap_leaf.astype(np.float32),
            "pgap_wood": pgap_wood.astype(np.float32),
            "G_all": G_all.astype(np.float32),
            "G_leaf": G_leaf.astype(np.float32),
            "G_wood": G_wood.astype(np.float32),
            "sum_fpl_all": sum_fpl_all.astype(np.float32),
            "mean_fpl_all": mean_fpl_all.astype(np.float32),
            "sum_fpl_leaf": sum_fpl_leaf.astype(np.float32),
            "mean_fpl_leaf": mean_fpl_leaf.astype(np.float32),
            "sum_fpl_wood": sum_fpl_wood.astype(np.float32),
            "mean_fpl_wood": mean_fpl_wood.astype(np.float32),
            "sum_efpl_all": sum_efpl_all.astype(np.float32),
            "sum_efpl_hits_all": sum_efpl_hits_all.astype(np.float32),
            "sum_efpl_leaf": sum_efpl_leaf.astype(np.float32),
            "sum_efpl_hits_leaf": sum_efpl_hits_leaf.astype(np.float32),
            "sum_efpl_wood": sum_efpl_wood.astype(np.float32),
            "sum_efpl_hits_wood": sum_efpl_hits_wood.astype(np.float32),
            "mean_count_all": mean_count_all.astype(np.float32),
            "var_count_all": var_count_all.astype(np.float32),
        }
        metadata = dict(mesh_meta)
        return metadata, stats


def generate_voxel_centers(voxel_size: float, bounds6: Tuple[float, float, float, float, float, float]) -> Tuple[np.ndarray, np.ndarray]:
    minx, miny, minz, maxx, maxy, maxz = bounds6
    xs = np.arange(minx + voxel_size / 2.0, maxx + voxel_size / 2.0, voxel_size)
    ys = np.arange(miny + voxel_size / 2.0, maxy + voxel_size / 2.0, voxel_size)
    zs = np.arange(minz + voxel_size / 2.0, maxz + voxel_size / 2.0, voxel_size)
    centers = np.array(np.meshgrid(xs, ys, zs, indexing="ij")).reshape(3, -1).T.astype(np.float32)
    coords = ((centers * 11 + voxel_size * 73) * 13).astype(int)
    vids = np.char.add(np.char.add(coords[:, 0].astype(str), "_"), np.char.add(coords[:, 1].astype(str), "_"))
    vids = np.char.add(vids, coords[:, 2].astype(str))
    return centers, vids

def filter_voxel_centers(centers: np.ndarray, leaf_bounds6: Tuple[float, float, float, float, float, float],
                         wood_bounds6: Tuple[float, float, float, float, float, float], voxel_size: float) -> np.ndarray:
    min_leaf, max_leaf = np.array(leaf_bounds6[0:3]), np.array(leaf_bounds6[3:6])
    min_wood, max_wood = np.array(wood_bounds6[0:3]), np.array(wood_bounds6[3:6])
    leaf_mask = np.all((centers >= (min_leaf - voxel_size / 2.0)) & (centers <= (max_leaf + voxel_size / 2.0)), axis=1)
    wood_mask = np.all((centers >= (min_wood - voxel_size / 2.0)) & (centers <= (max_wood + voxel_size / 2.0)), axis=1)
    return centers[leaf_mask | wood_mask]


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Process voxel batch.")
    parser.add_argument("scene_file", type=str, help="Path to the single .obj scene file. This will automatically extract leaf and wood meshes.")
    parser.add_argument("--voxel_sizes", type=float, nargs='+', default=[0.2, 0.5, 1.0, 2.0], help="Voxel sizes for processing (default: [0.2, 0.5, 1.0, 2.0]).")
    parser.add_argument("--num_angle_bins", type=int, default=18, help="Number of angle bins for ray tracing (default: 18).")
    parser.add_argument("--ray_spacing", type=float, default=0.01, help="Ray spacing for ray tracing (default: 0.01).")
    parser.add_argument("--wood_volume_voxel_size", type=float, default=0.01, help="Voxel size for wood volume calculation (default: 0.01). (not used in refactor)")
    parser.add_argument("--wood_volume_threshold", type=int, default=4, help="Threshold for wood volume calculation (default: 4). (not used in refactor)")
    parser.add_argument("--max_workers", type=int, default=32, help="Maximum number of parallel workers (default: 32). Will cap to num_cpus.")
    parser.add_argument("--force_cpu", action='store_true', help="If set, forces the use of CPU even if Warp is available.")
    parser.add_argument("--debug", action='store_true', help="If set, debug outputs will be printed.")
    args = parser.parse_args(argv)

    print(f"Scene: {args.scene_file}")
    print(f"Voxel sizes: {args.voxel_sizes}")
    print(f"Angle bins: {args.num_angle_bins}")
    print(f"Ray spacing: {args.ray_spacing}")
    print(f"Max workers: {args.max_workers}")
    print(f"Force CPU: {args.force_cpu}")
    print(f"Debug: {args.debug}")

    dev = DeviceManager(prefer_cuda=True)
    if dev.using_warp and not args.force_cpu:
        raycaster = WarpRaycaster(dev.device_str)
    else:
        raycaster = O3DRaycaster()

    leaf_keys = ["leaf", "leaves", "leafs"]
    wood_keys = ["wood", "trunk", "branch", "stem"]
    leaf_mesh, wood_mesh, bounds6 = load_and_split_by_group(args.scene_file, leaf_keys, wood_keys)

    leaf_bounds6 = tuple(leaf_mesh.bounds.flatten().tolist()) if (leaf_mesh is not None) else (0, 0, 0, 0, 0, 0)
    wood_bounds6 = tuple(wood_mesh.bounds.flatten().tolist()) if (wood_mesh is not None) else (0, 0, 0, 0, 0, 0)

    if os.environ.get("SLURM_CPUS_PER_TASK") is None:
        num_cpus = psutil.cpu_count(logical=True)
    else:
        num_cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", psutil.cpu_count(logical=False)))*2
    num_cpus = max(1, num_cpus)
    n_workers = min(args.max_workers, num_cpus)

    for vs in args.voxel_sizes:
        edges = np.linspace(0, 90, args.num_angle_bins + 1)
        centers = np.round((edges[:-1] + edges[1:]) / 2.0, 3)
        angle_set = set(centers.tolist())
        if 0 in angle_set: angle_set.remove(0); angle_set.add(0.0001)
        if 90 in angle_set: angle_set.remove(90); angle_set.add(89.9999)
        ANGLE_ORDER = sorted(angle_set)
        raygrid = RayGrid([vs], args.ray_spacing)
        centers, vids = generate_voxel_centers(vs, bounds6)
        centers = filter_voxel_centers(centers, leaf_bounds6, wood_bounds6, vs)
        if centers.size == 0:
            print(f"[skip] No voxel centers for size {vs}"); continue
        import datetime as dt
        base = os.path.basename(args.scene_file).replace(".obj", f"_results_vs_{vs}_{dt.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv")
        out_csv = os.path.join(os.path.dirname(args.scene_file), base)
        writer = CSVWriter(out_csv, metadata_keys_order=[
            "voxel_cx", "voxel_cy", "voxel_cz", "voxel_size", "voxel_id", "face_index", "face_name",
            "angle_index", "angle_center_deg", "alpha", "LAI_ref", "WAI_ref", "PAI_ref", "LAD_ref", "WAD_ref", "PAD_ref",
            "leaf_fraction", "liad_dewit", "wiad_dewit", "piad_dewit", "liad_json", "wiad_json", "piad_json",
        ])
        clipper = MeshClipper(leaf_mesh, wood_mesh)
        def _clip_one(vc: np.ndarray, vs: float):
            leaf_md, wood_md = clipper.clip(vc, vs); return vc, leaf_md, wood_md
        print(f"Voxel size {vs}: clipping meshes for {len(centers)} voxels ...")
        from joblib import Parallel, delayed
        results = list(tqdm(Parallel(n_jobs=n_workers, backend="threading")(delayed(_clip_one)(vc, vs) for vc in centers), total=len(centers), desc="Clipping", unit="vox"))
        vox_data = [(vc, lm, wm) for (vc, lm, wm) in results if (lm or wm)]
        if not vox_data:
            print(f"[skip] No geometry within voxels for size {vs}"); writer.close(); continue
        stat_eng = StatComputer(angles_order=ANGLE_ORDER)
        with tqdm(total=len(vox_data), desc="Process", unit="vox") as pbar:
            for vc, leaf_md, wood_md in vox_data:
                try:
                    mesh_meta, aux = stat_eng.mesh_metrics(vc, vs, leaf_md, wood_md, num_angle_bins=args.num_angle_bins)
                    rays_FAR6, face_order, angles_sorted = raygrid.build(vc, vs, ANGLE_ORDER)
                    rc = raycaster.raycast(vc, vs, rays_FAR6, leaf_md, wood_md)
                    metadata, stats = stat_eng.compute(vc, vs, rays_FAR6, rc, mesh_meta, aux)
                    voxel_id = create_voxel_id(voxel_size=vs, x=vc[0], y=vc[1], z=vc[2])
                    rows = build_rows(metadata, stats, face_order=face_order, angles=angles_sorted, voxel_id=voxel_id)
                    writer.write_rows(rows)
                except Exception as e:
                    if args.debug:
                        print(f"[error] Voxel {vc.tolist()} failed: {e}\n{traceback.format_exc()}")
                finally:
                    pbar.update(1)
        writer.close()
        print(f"Saved results to {out_csv}.")
    print("All done.")
    return 0

if __name__ == "__main__":
    sys.exit(main())
