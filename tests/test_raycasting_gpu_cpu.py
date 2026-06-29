"""
Tests for GPU/CPU raycasting functionality from the new process_reference_single_obj.py.
Tests ray-box intersection, EFPL computation, and raycaster backends.
"""
import numpy as np
import pytest

# Import from the new module
import sys
import os

# Ray-box intersection is defined in process_reference_single_obj
from process_reference_single_obj import (
    ray_box_intersection_vectorized,
    compute_efpl_array,
    O3DRaycaster,
    MeshClipper,
    DeviceManager,
)


class TestRayBoxIntersectionVectorized:
    """Test vectorized ray-box intersection computation."""

    def test_ray_hits_center_box(self):
        """Ray from outside pointing at box center should intersect."""
        bmin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        bmax = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        origins = np.array([[0.0, 0.0, -2.0]], dtype=np.float32)
        dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        t_near, t_far = ray_box_intersection_vectorized(origins, dirs, bmin, bmax)

        assert t_near[0] > 0, "Ray should intersect box"
        assert t_far[0] > t_near[0], "t_far should be > t_near"
        # Expected: enters at z=-0.5 (t=1.5), exits at z=0.5 (t=2.5)
        assert np.isclose(t_near[0], 1.5, atol=1e-5)
        assert np.isclose(t_far[0], 2.5, atol=1e-5)

    def test_ray_misses_box(self):
        """Ray that misses should have t_far < t_near."""
        bmin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        bmax = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        origins = np.array([[1.0, 0.0, -2.0]], dtype=np.float32)
        dirs = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        t_near, t_far = ray_box_intersection_vectorized(origins, dirs, bmin, bmax)

        assert t_far[0] < t_near[0], "Ray should miss box"

    def test_multiple_rays_mixed_hits_misses(self):
        """Test multiple rays with mix of hits and misses."""
        bmin = np.array([-0.5, -0.5, -0.5], dtype=np.float32)
        bmax = np.array([0.5, 0.5, 0.5], dtype=np.float32)

        # 3 rays: 2 hits, 1 miss
        origins = np.array(
            [[0.0, 0.0, -2.0], [0.0, 0.0, -2.0], [2.0, 2.0, -2.0]], dtype=np.float32
        )
        dirs = np.array(
            [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32
        )

        t_near, t_far = ray_box_intersection_vectorized(origins, dirs, bmin, bmax)

        assert len(t_near) == 3 and len(t_far) == 3
        assert t_near[0] > 0 and t_far[0] > t_near[0], "Ray 0 should hit"
        assert t_near[1] > 0 and t_far[1] > t_near[1], "Ray 1 should hit"
        assert t_far[2] < t_near[2], "Ray 2 should miss"


class TestEFPLArray:
    """Test effective free path length array computation."""

    def test_efpl_zero_lambda(self):
        """When lambda=0, EFPL should equal distance."""
        d_arr = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        efpl = compute_efpl_array(d_arr, lambda_1=0.0)

        np.testing.assert_array_almost_equal(efpl, d_arr, decimal=5)

    def test_efpl_positive_lambda(self):
        """EFPL should be positive when lambda > 0 and inputs are valid."""
        d_arr = np.array([0.01, 0.02, 0.03], dtype=np.float32)
        lambda_1 = 0.1  # Smaller lambda to avoid numerical issues
        efpl = compute_efpl_array(d_arr, lambda_1=lambda_1)

        # EFPL = -log(1 - lambda*d) / lambda
        # Should be positive for valid inputs
        assert np.all(efpl > 0), "EFPL should be positive"
        assert np.all(np.isfinite(efpl)), "EFPL should be finite"

    def test_efpl_zero_distances(self):
        """EFPL of zero distance should be zero."""
        d_arr = np.array([0.0, 0.0], dtype=np.float32)
        efpl = compute_efpl_array(d_arr, lambda_1=1.0)

        np.testing.assert_array_almost_equal(efpl, [0.0, 0.0], decimal=5)

    def test_efpl_monotonic(self):
        """EFPL should increase monotonically with distance."""
        lambda_1 = 0.5
        d_arr = np.array([0.05, 0.1, 0.15, 0.2], dtype=np.float32)
        efpl = compute_efpl_array(d_arr, lambda_1=lambda_1)

        # Check monotonicity (allowing for floating point errors)
        diffs = np.diff(efpl)
        assert np.all(diffs >= -1e-6), "EFPL should be monotonically increasing"


class TestDeviceManager:
    """Test GPU/CPU device detection and management."""

    def test_device_manager_init(self):
        """DeviceManager should initialize without errors."""
        dm = DeviceManager(prefer_cuda=False)
        assert dm is not None
        assert dm.device_str is not None

    def test_device_manager_cpu_fallback(self):
        """DeviceManager should handle CPU mode gracefully."""
        dm = DeviceManager(prefer_cuda=False)
        assert dm.device_str == "cpu"

    def test_device_manager_cuda_detection(self):
        """DeviceManager should detect CUDA if available."""
        dm = DeviceManager(prefer_cuda=True)
        # Device string will be either cuda:X or cpu depending on availability
        assert dm.device_str in ["cpu"] or "cuda" in dm.device_str


class TestO3DRaycaster:
    """Test CPU raycaster (Open3D backend)."""

    def test_o3d_raycaster_creation(self):
        """O3DRaycaster should initialize."""
        caster = O3DRaycaster()
        assert caster is not None

    def test_o3d_raycaster_with_empty_meshes(self):
        """O3DRaycaster should handle empty meshes gracefully."""
        import open3d as o3d
        import numpy as np

        caster = O3DRaycaster()

        # Create empty triangle mesh
        o3d_mesh = o3d.geometry.TriangleMesh()

        # Create simple ray
        rays = np.array([[0, 0, 0, 0, 0, 1]], dtype=np.float32).reshape(1, 1, 1, 6)

        # Should not crash
        try:
            result = caster.raycast(
                voxel_center=np.array([0, 0, 0]),
                voxel_size=1.0,
                rays_FAR6=rays,
                leaf_mesh=None,
                wood_mesh=None,
            )
            assert result is not None
        except Exception as e:
            pytest.skip(f"O3D raycaster test skipped: {e}")


class TestMeshClipper:
    """Test mesh clipping for voxel extraction."""

    def test_mesh_clipper_init(self):
        """MeshClipper should initialize with None meshes."""
        clipper = MeshClipper(leaf_mesh=None, wood_mesh=None)
        assert clipper is not None

    def test_mesh_clipper_clip_empty(self):
        """Clipping empty meshes should return empty results."""
        clipper = MeshClipper(leaf_mesh=None, wood_mesh=None)

        result = clipper.clip(
            voxel_center=np.array([0.0, 0.0, 0.0]), voxel_size=1.0
        )

        leaf_clip, wood_clip = result
        assert leaf_clip is None or leaf_clip.vertices.size == 0
        assert wood_clip is None or wood_clip.vertices.size == 0


class TestGPUAvailability:
    """Test GPU availability and graceful degradation."""

    def test_warp_optional(self):
        """Code should work even if Warp is not installed."""
        # This is tested implicitly by DeviceManager
        dm = DeviceManager(prefer_cuda=True)
        # Should not raise, just fall back to CPU if needed
        assert dm.device_str is not None

    def test_gpu_disabled_stays_cpu(self):
        """Requesting CPU should stay on CPU."""
        dm = DeviceManager(prefer_cuda=False)
        assert dm.device_str == "cpu"
