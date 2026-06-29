"""
Smoke tests for utils.py functions.
Validates that core computation functions work correctly with known inputs.
These tests run on the current main branch version of utils.py.
"""
import numpy as np
import pytest
from utils import (
    create_voxel_id,
    calculate_lambda_1,
)


class TestVoxelID:
    """Test voxel ID generation."""

    def test_voxel_id_deterministic(self):
        """Same inputs should produce same voxel ID."""
        vid1 = create_voxel_id(voxel_size=0.1, x=1.0, y=2.0, z=3.0)
        vid2 = create_voxel_id(voxel_size=0.1, x=1.0, y=2.0, z=3.0)

        assert vid1 == vid2, "Voxel IDs should be deterministic"

    def test_voxel_id_unique(self):
        """Different voxels should have different IDs."""
        vid1 = create_voxel_id(voxel_size=0.1, x=1.0, y=2.0, z=3.0)
        vid2 = create_voxel_id(voxel_size=0.1, x=1.1, y=2.0, z=3.0)

        assert vid1 != vid2, "Different voxels should have different IDs"


class TestLambdaCalculation:
    """Test lambda_1 calculation from leaf area."""

    def test_lambda_positive(self):
        """Lambda should be positive for positive leaf area and voxel size."""
        avg_leaf_area = 0.0146533
        voxel_size = 0.01

        lam = calculate_lambda_1(avg_leaf_area, voxel_size)

        assert lam > 0, "Lambda should be positive"

    def test_lambda_scales_with_area(self):
        """Lambda should scale proportionally with leaf area."""
        voxel_size = 0.01

        lam1 = calculate_lambda_1(0.01, voxel_size)
        lam2 = calculate_lambda_1(0.02, voxel_size)

        assert lam2 == 2 * lam1, "Lambda should scale linearly with leaf area"
