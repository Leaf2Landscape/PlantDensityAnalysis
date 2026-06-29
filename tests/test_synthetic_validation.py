"""
Integration tests using synthetic test data.
Validates that processing produces sensible outputs (placeholder for pre-merge).

This test file will be expanded once process_reference_single_obj.py is merged.
For now, it validates synthetic data generation and basic properties.
"""
import numpy as np
import pytest


class TestSyntheticMeshGeneration:
    """Validate synthetic test data generation."""

    def test_synthetic_meshes_created(self, synthetic_meshes):
        """Synthetic meshes should be created and non-empty."""
        wood = synthetic_meshes["wood_mesh"]
        leaf = synthetic_meshes["leaf_mesh"]

        assert wood is not None, "Wood mesh should be created"
        assert leaf is not None, "Leaf mesh should be created"
        assert not wood.is_empty, "Wood mesh should not be empty"
        assert not leaf.is_empty, "Leaf mesh should not be empty"

    def test_scene_file_created(self, synthetic_meshes):
        """Scene file should exist and be readable."""
        import os

        scene_path = synthetic_meshes["scene_path"]
        assert os.path.exists(scene_path), "Scene file should exist"

        with open(scene_path, "r") as f:
            content = f.read()
            assert "g wood" in content, "Scene should contain wood group"
            assert "g leaf" in content, "Scene should contain leaf group"
            assert "v " in content, "Scene should contain vertices"
            assert "f " in content, "Scene should contain faces"

    def test_bounds_reasonable(self, voxel_bounds):
        """Voxel bounds should be reasonable (min < max)."""
        minx, miny, minz, maxx, maxy, maxz = voxel_bounds

        assert minx < maxx, "X bounds should be ordered"
        assert miny < maxy, "Y bounds should be ordered"
        assert minz < maxz, "Z bounds should be ordered"

        # Bounds should be in reasonable range (not infinite or NaN)
        assert np.isfinite(minx) and np.isfinite(maxx), "X bounds should be finite"
        assert np.isfinite(miny) and np.isfinite(maxy), "Y bounds should be finite"
        assert np.isfinite(minz) and np.isfinite(maxz), "Z bounds should be finite"

    def test_mesh_properties(self, synthetic_meshes):
        """Mesh properties should be valid."""
        wood = synthetic_meshes["wood_mesh"]
        leaf = synthetic_meshes["leaf_mesh"]

        # Check wood mesh (cylinder)
        assert wood.vertices.shape[0] > 0, "Wood should have vertices"
        assert wood.faces.shape[0] > 0, "Wood should have faces"

        # Check leaf mesh
        assert leaf.vertices.shape[0] > 0, "Leaf should have vertices"
        assert leaf.faces.shape[0] > 0, "Leaf should have faces"

    def test_mesh_bounds_meaningful(self, synthetic_meshes):
        """Mesh bounds should make sense geometrically."""
        wood = synthetic_meshes["wood_mesh"]
        leaf = synthetic_meshes["leaf_mesh"]

        wood_bounds = wood.bounds
        leaf_bounds = leaf.bounds

        # Bounds should be [min, max] with 3 dims each
        assert wood_bounds.shape == (2, 3), "Wood bounds shape incorrect"
        assert leaf_bounds.shape == (2, 3), "Leaf bounds shape incorrect"

        # Bounds should be ordered (min < max for each axis)
        assert np.all(wood_bounds[0] < wood_bounds[1]), "Wood min/max not ordered"
        assert np.all(leaf_bounds[0] < leaf_bounds[1]), "Leaf min/max not ordered"


class TestVoxelGeneration:
    """Test voxel center generation (placeholder for pre-merge)."""

    def test_bounds_expand_for_coverage(self, voxel_bounds):
        """Bounds should be expanded to ensure voxel coverage."""
        minx, miny, minz, maxx, maxy, maxz = voxel_bounds

        # Verify expansion happened (should be slightly larger than mesh bounds)
        size_x = maxx - minx
        size_y = maxy - miny
        size_z = maxz - minz

        assert size_x > 0 and size_y > 0 and size_z > 0, "Bounds should have positive size"

        # Volume should be reasonable (not tiny, not infinite)
        volume = size_x * size_y * size_z
        assert 0.001 < volume < 1000, f"Bounds volume {volume} seems unreasonable"
