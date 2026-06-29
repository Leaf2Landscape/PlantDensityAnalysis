"""
Pytest configuration and fixtures for PlantDensityAnalysis tests.
Provides synthetic test data (cylinder + leaf mesh) and utilities.
"""
import os
import tempfile
import numpy as np
import trimesh
import pytest


@pytest.fixture
def test_data_dir():
    """Create a temporary directory for test data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def synthetic_meshes(test_data_dir):
    """
    Generate simple synthetic meshes for testing:
    - Wood: vertical cylinder centered at origin, radius=0.1, height=1.0
    - Leaf: simple triangle(s) representing leaf surfaces
    """
    # Wood mesh: cylinder
    radius = 0.1
    height = 1.0
    wood_mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=8)
    wood_mesh.apply_translation([0, 0, height / 2])  # Center at origin height-wise

    # Leaf mesh: simple planar triangles (represent horizontal leaves)
    leaf_verts = np.array([
        [0.0, 0.0, 0.5],
        [0.3, 0.0, 0.5],
        [0.15, 0.2, 0.5],
        [0.0, 0.3, 0.6],
        [0.25, 0.3, 0.6],
        [0.125, 0.5, 0.6],
    ], dtype=np.float64)
    leaf_faces = np.array([
        [0, 1, 2],  # Triangle 1
        [3, 4, 5],  # Triangle 2
    ], dtype=np.int64)
    leaf_mesh = trimesh.Trimesh(vertices=leaf_verts, faces=leaf_faces, process=False)

    # Save to disk as OBJ files with groups
    wood_path = os.path.join(test_data_dir, "wood.obj")
    leaf_path = os.path.join(test_data_dir, "leaf.obj")

    # Create a combined scene file with groups
    scene_path = os.path.join(test_data_dir, "scene.obj")

    with open(scene_path, "w") as f:
        # Write wood (cylinder)
        f.write("g wood\n")
        for v in wood_mesh.vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        wood_v_offset = len(wood_mesh.vertices)
        for face in wood_mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        # Write leaf (triangles)
        f.write("g leaf\n")
        for v in leaf_verts:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for face in leaf_faces:
            f.write(f"f {face[0]+wood_v_offset+1} {face[1]+wood_v_offset+1} {face[2]+wood_v_offset+1}\n")

    return {
        "scene_path": scene_path,
        "wood_mesh": wood_mesh,
        "leaf_mesh": leaf_mesh,
        "wood_path": wood_path,
        "leaf_path": leaf_path,
    }


@pytest.fixture
def voxel_bounds(synthetic_meshes):
    """
    Return voxel processing bounds based on synthetic meshes.
    Slightly expanded to ensure full coverage.
    """
    wood = synthetic_meshes["wood_mesh"]
    leaf = synthetic_meshes["leaf_mesh"]

    bounds_list = [wood.bounds, leaf.bounds]
    min_b = np.min([b[0] for b in bounds_list], axis=0)
    max_b = np.max([b[1] for b in bounds_list], axis=0)

    # Expand slightly for voxel coverage
    expand = 0.05
    min_b = min_b - expand
    max_b = max_b + expand

    return tuple(np.concatenate([min_b, max_b]))
