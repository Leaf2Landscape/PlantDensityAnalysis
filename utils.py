"""
A Python script of commonly shared utilities for other scripts.
Includes schemas for i/o data, functions, and helpers.
"""

from fnvhash import fnv1a_32
import pyarrow as pa
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import gc
import dask
import os
import tempfile
import uuid
import shutil
from scipy.sparse.csgraph import connected_components


### CONSTANTS ###
beam_divergence = np.float64(0.001) # Beam divergence in radians

### SCHEMAS ###

# Voxel Ray Intersection Schema
"""
This schema is used to store ray intersections for each voxel.
It leverages the pyarrow library to maximise efficiency of dask, pandas, and parquet.

It is saved in the format:
    leg_{leg_id}_voxel_{voxel_size}_ray_intersections.parquet

And contains the information outlined in the following schema.
Each index corresponds to a ray that intersects a voxel.
"""
voxel_ray_intersection_schema_old = pa.schema([
    pa.field('voxel_size', pa.float32()),
    pa.field('voxel_id', pa.int64()),
    pa.field('leg_id', pa.int64()),
    pa.field('ray_id', pa.int64()),
    pa.field('t_entry_x', pa.float32()),
    pa.field('t_entry_y', pa.float32()),
    pa.field('t_entry_z', pa.float32()),
    pa.field('t_exit_x', pa.float32()),
    pa.field('t_exit_y', pa.float32()),
    pa.field('t_exit_z', pa.float32()),
    pa.field('t_entry_radius', pa.float32()),
    pa.field('t_exit_radius', pa.float32()),
    pa.field('point_x', pa.float32()),
    pa.field('point_y', pa.float32()),
    pa.field('point_z', pa.float32()),
    pa.field('viewing_angle', pa.float32()),
    pa.field('hit_ray', pa.bool_()),
    pa.field('is_leaf', pa.bool_())
])

voxel_ray_intersection_schema = pa.schema([
    pa.field('voxel_size', pa.float32()),
    pa.field('voxel_id', pa.int64()),
    pa.field('leg_id', pa.int64()),
    pa.field('ray_id', pa.int64()),
    pa.field('t_entry_x', pa.float32()),
    pa.field('t_entry_y', pa.float32()),
    pa.field('t_entry_z', pa.float32()),
    pa.field('t_exit_x', pa.float32()),
    pa.field('t_exit_y', pa.float32()),
    pa.field('t_exit_z', pa.float32()),
    pa.field('t_entry_radius', pa.float32()),
    pa.field('t_exit_radius', pa.float32()),
    pa.field('point_x', pa.float32()),
    pa.field('point_y', pa.float32()),
    pa.field('point_z', pa.float32()),
    pa.field('normal_x', pa.float32()),
    pa.field('normal_y', pa.float32()),
    pa.field('normal_z', pa.float32()),
    pa.field('point_weight', pa.float32()),
    pa.field('viewing_angle', pa.float32()),
    pa.field('hit_ray', pa.bool_()),
    pa.field('is_leaf', pa.bool_())
])

# Voxel Metrics Schema
"""
This schema is used to store the metrics for each voxel, based on the selected legs and voxel size.
Since this one is only used to store to a csv file (for final output), it is not as important to be efficient.


"""
voxel_metrics_schema_old = pa.schema([
    pa.field('voxel_id', pa.uint32()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('I_lw', pa.int64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.int64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('mean_angle_lw', pa.float64()), # Mean angle of leaf and wood hits
    pa.field('mean_angle_leaf', pa.float64()), # Mean angle of leaf hits only
    pa.field('mean_angle_all', pa.float64()), # Mean angle of all hits
    pa.field('mean_path_length', pa.float64()),
    pa.field('sum_path_length', pa.float64()),
    pa.field('mean_free_path_length', pa.float64()),
    pa.field('sum_free_path_length', pa.float64()),
    pa.field('sum_free_path_length_hit', pa.float64()),
    pa.field('sum_free_path_length_hit_leaf', pa.float64()),
    pa.field('mean_eff_path_length', pa.float64()),
    pa.field('var_eff_path_length', pa.float64()),
    pa.field('mean_eff_free_path_length', pa.float64()),
    pa.field('var_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length_hit', pa.float64()),  # Sum of z for all hits
    pa.field('sum_eff_free_path_length_hit_leaf', pa.float64()) # Sum of z for leaf hits only    
])

voxel_metrics_schema = pa.schema([
    pa.field('voxel_id', pa.uint32()),
    pa.field('num_rays', pa.uint32()),
    pa.field('num_hits', pa.uint32()),
    pa.field('num_leaf_hits', pa.uint32()),
    pa.field('pgap_lw', pa.float64()),
    pa.field('pgap_leaf', pa.float64()),
    pa.field('I_lw', pa.int64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.int64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('G_lw', pa.float64()),                    # G function calculated from leaf and wood hits
    pa.field('G_leaf', pa.float64()),               # G function calculated from leaf hits only
    pa.field('lambda_1', pa.float64()), # Lambda_1 for the voxel
    pa.field('LIAD_leaf_bin_2.5', pa.float64()),
    pa.field('LIAD_leaf_bin_7.5', pa.float64()),
    pa.field('LIAD_leaf_bin_12.5', pa.float64()),
    pa.field('LIAD_leaf_bin_17.5', pa.float64()),
    pa.field('LIAD_leaf_bin_22.5', pa.float64()),
    pa.field('LIAD_leaf_bin_27.5', pa.float64()),
    pa.field('LIAD_leaf_bin_32.5', pa.float64()),
    pa.field('LIAD_leaf_bin_37.5', pa.float64()),
    pa.field('LIAD_leaf_bin_42.5', pa.float64()),
    pa.field('LIAD_leaf_bin_47.5', pa.float64()),
    pa.field('LIAD_leaf_bin_52.5', pa.float64()),
    pa.field('LIAD_leaf_bin_57.5', pa.float64()),
    pa.field('LIAD_leaf_bin_62.5', pa.float64()),
    pa.field('LIAD_leaf_bin_67.5', pa.float64()),
    pa.field('LIAD_leaf_bin_72.5', pa.float64()),
    pa.field('LIAD_leaf_bin_77.5', pa.float64()),
    pa.field('LIAD_leaf_bin_82.5', pa.float64()),
    pa.field('LIAD_leaf_bin_87.5', pa.float64()),
    # pa.field('LIAD_lw_bin_2.5', pa.float64()),
    # pa.field('LIAD_lw_bin_7.5', pa.float64()),
    # pa.field('LIAD_lw_bin_12.5', pa.float64()),
    # pa.field('LIAD_lw_bin_17.5', pa.float64()),
    # pa.field('LIAD_lw_bin_22.5', pa.float64()),
    # pa.field('LIAD_lw_bin_27.5', pa.float64()),
    # pa.field('LIAD_lw_bin_32.5', pa.float64()),
    # pa.field('LIAD_lw_bin_37.5', pa.float64()),
    # pa.field('LIAD_lw_bin_42.5', pa.float64()),
    # pa.field('LIAD_lw_bin_47.5', pa.float64()),
    # pa.field('LIAD_lw_bin_52.5', pa.float64()),
    # pa.field('LIAD_lw_bin_57.5', pa.float64()),
    # pa.field('LIAD_lw_bin_62.5', pa.float64()),
    # pa.field('LIAD_lw_bin_67.5', pa.float64()),
    # pa.field('LIAD_lw_bin_72.5', pa.float64()),
    # pa.field('LIAD_lw_bin_77.5', pa.float64()),
    # pa.field('LIAD_lw_bin_82.5', pa.float64()),
    # pa.field('LIAD_lw_bin_87.5', pa.float64()),
    pa.field('mean_angle_leaf', pa.float64()), # Mean angle of leaf hits only
    pa.field('mean_angle_all', pa.float64()), # Mean angle of all hits
    pa.field('mean_path_length', pa.float64()),
    pa.field('sum_path_length', pa.float64()),
    pa.field('mean_free_path_length', pa.float64()),
    pa.field('sum_free_path_length', pa.float64()),
    pa.field('sum_free_path_length_hit', pa.float64()),
    pa.field('sum_free_path_length_hit_leaf', pa.float64()),
    pa.field('mean_eff_path_length', pa.float64()),
    pa.field('var_eff_path_length', pa.float64()),
    pa.field('mean_eff_free_path_length', pa.float64()),
    pa.field('var_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length', pa.float64()),
    pa.field('sum_eff_free_path_length_hit', pa.float64()),  # Sum of z for all hits
    pa.field('sum_eff_free_path_length_hit_leaf', pa.float64()) # Sum of z for leaf hits only    
])

# Reference Schema
"""
This schema is used to store the reference data for each voxel.
"""
reference_schema = pa.schema([
    pa.field('voxel_id', pa.uint32()),
    pa.field('voxel_size', pa.float32()),
    pa.field('CI', pa.float32()),
    pa.field('woody_vol_proportion', pa.float32()),
    pa.field('G', pa.float32()),
    pa.field('G_leaf', pa.float32()),
    pa.field('LAD', pa.float32()),
    pa.field('PAD', pa.float32()),
])

# Valid Rays Schema
valid_rays_schema = pa.schema([
    pa.field('leg_id', pa.int64()),
    pa.field('ray_id', pa.int64()),
    pa.field('origin_x', pa.float32()),
    pa.field('origin_y', pa.float32()),
    pa.field('origin_z', pa.float32()),
    pa.field('direction_x', pa.float32()),
    pa.field('direction_y', pa.float32()),
    pa.field('direction_z', pa.float32()),
    pa.field('point_x', pa.float32()),
    pa.field('point_y', pa.float32()),
    pa.field('point_z', pa.float32()),
    pa.field('is_leaf', pa.bool_())
])

### HELPER FUNCTIONS ###
# Commonly used functions that offer small utilities for components of other scripts.

def compute_normals_weights_from_points(points, knn=6):
    import open3d as o3d
    """
    Get normals and weights from points.
    
    INPUTS:
        points: A numpy array of points
        knn: The number of nearest neighbours to consider

    OUTPUTS:
        normals: The normals of the points
        weights: The weights of the points
    """
    if len(points) < knn:
        return np.ones(len(points)), np.ones(len(points))

    # Compute the normals with open3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=5.0)

    normals = np.zeros((len(pcd.points), 3))
    weights = np.zeros(len(pcd.points))

    voxel_dict = {}

    def unique_key_from_voxel_centre(voxel_centre):
        return f"{voxel_centre[0]}_{voxel_centre[1]}_{voxel_centre[2]}"

    for idx, point in enumerate(np.asarray(pcd.points)):
        voxel_id = unique_key_from_voxel_centre(voxel_grid.get_voxel(point))
        if voxel_id not in voxel_dict.keys():
            voxel_dict[voxel_id] = []        
        voxel_dict[voxel_id].append((idx, point))

    for voxel, points in voxel_dict.items():
        idxs, points = zip(*points)

        voxel_pcd = o3d.geometry.PointCloud()
        voxel_pcd.points = o3d.utility.Vector3dVector(points)
        voxel_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
        distances = np.asarray(voxel_pcd.compute_nearest_neighbor_distance())
        voxel_normals = np.asarray(voxel_pcd.normals)
        voxel_weights = 1 / (distances + 1e-9)

        for i, idx in enumerate(idxs):
            normals[idx] = voxel_normals[i]
            weights[idx] = voxel_weights[i]

    return normals, weights


    # Compute the point density weights
    nbrs = NearestNeighbors(n_neighbors=knn).fit(points)
    distances, _ = nbrs.kneighbors(points)
    # Inverse of the distance to the k nearest neighbours as weight
    weights = 1 / (distances[:, -1] + 1e-9) # Add a small value to avoid division by zero

    return weights
    # Compute the point density weights
    if len(points) < knn:
        weights = np.ones(len(points))
    else:
        nbrs = NearestNeighbors(n_neighbors=knn).fit(points)
        distances, _ = nbrs.kneighbors(points)
        # Inverse of the distance to the k nearest neighbours as weight
        weights = 1 / (distances[:, -1] + 1e-9) # Add a small value to avoid division by zero
    
    return weights

# Create a unique ID for a voxel
def create_voxel_id(voxel_size, x, y, z):
    """
    Create a unique ID for a voxel.
    
    INPUTS:
        nd_array: containing [voxel_size, x, y, z]

    OUTPUTS:
        voxel_id: A unique ID for the voxel
    """
    # Create a string representation of the voxel parameters
    voxel_string = f'{voxel_size}_{x}_{y}_{z}'

    # Encode the string and hash it using FNV-1a
    voxel_id = fnv1a_32(voxel_string.encode())
    # print(f"Created unique voxel_id: {voxel_id} for voxel {voxel_string}")

    return voxel_id

# Create a pandas dataframe from a pyarrow schema
def create_df_from_schema(schema):
    """
    Create a pandas dataframe from a pyarrow schema.
    
    INPUTS:
        schema: A pyarrow schema
        
    OUTPUTS:
        df: A pandas dataframe with the same columns and dtypes as the schema
    """
    new_df = pd.DataFrame(columns=schema.names)

    for field in schema:
        new_df[field.name] = new_df[field.name].astype(field.type.to_pandas_dtype())

    return new_df

# Calculate lambda_1
def calculate_lambda_1(average_leaf_area, voxel_size):
    """
    Calculate lambda_1 for a given voxel size.
    """
    lambda_1 = average_leaf_area / (voxel_size ** 3)

    return lambda_1

# Calculate the effective path length z
def effective_path_length_z(z, lambda_1):
    """
    Calculate the effective path length z.
    
    INPUTS:
        free_path_lengths: The free path lengths
        lambda_1: The calculated lambda_1
    
    OUTPUTS:
        z: The effective path length z
    """
    z = z.copy()
    with np.errstate(divide='ignore', invalid='ignore'):
        valid_mask = (lambda_1 * z) < 1
        eff_path_length_zs = np.full_like(z, fill_value=np.nan, dtype=np.float64)
        z[valid_mask] = -np.log(1 - lambda_1 * z[valid_mask]) / lambda_1
        
    return eff_path_length_zs

def calculate_LIAD(normals, weights, num_bins=18):
    """
    Calculate the Leaf Angle Distribution (LAD) for a set of normals and weights.
    
    INPUTS:
        normals: A numpy array of normals
        weights: A numpy array of weights
        num_bins: The number of bins to use for the histogram
        
    OUTPUTS:
        bin_centres: The bin centres
        LIAD_values: The LIAD values
        angles: The angles
    """
    # Compute the angles
    angles = np.arccos(np.dot(normals, np.array([0, 0, 1])))
    angles = np.where(angles > np.pi / 2, np.pi - angles, angles)
    angles = np.degrees(angles)

    # Compute LIAD for each voxel
    if len(angles) == 0 or np.all(np.isnan(angles)):
        return np.array([]), np.array([]), np.array([])

    if len(weights) == 0:
        weights = np.ones_like(angles)

    # Remove NaN angles and align weights
    valid_mask = ~np.isnan(angles)
    angles = angles[valid_mask]
    weights = weights[valid_mask]

    if len(angles) == 0:
        return np.array([]), np.array([]), np.array([])

    # Compute the histogram
    hist, bin_edges = np.histogram(angles, bins=num_bins, range=(0, 90), weights=weights)
    total_weight = np.sum(hist)
    if total_weight > 0:
        LIAD_values = hist / total_weight
    else:
        LIAD_values = np.zeros(num_bins)

    # Compute the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres, LIAD_values, angles

def calculate_LIAD_old(points, knn=6, radius=0.1, max_nn=10, num_bins=18):
    """
    Calculate the (LIAD) for a set of points.
    
    INPUTS:
        points: A numpy array of points
        knn: The number of nearest neighbours to consider
        radius: The radius to consider
        max_nn: The maximum number of nearest neighbours to consider
        num_bins: The number of bins to use for the histogram

    OUTPUTS:
        bin_centres: The bin centres
        LIAD_values: The LIAD values
        angles: The angles

    NOTE:
    1) Get Normals
    2) Weight based on proximity to other points
    3) Compute angles against up ([0, 0, 1]).
        Normalise to max 90 degrees.
    4) Bin anlges into 5 degree bins, and find weight ratio per bin against total weight 
        This varies from mesh-based approaches, since we use weighted volume instead of area
    5) Return bin centres, LIAD values, and angles
    """

    # Convert points to numpy
    points_copy = np.copy(points)
    points_copy = points_copy[~np.isnan(points_copy).any(axis=1)]

    # Compute the point density weights
    if len(points_copy) < knn:
        weights = np.ones(len(points_copy))
    else:
        nbrs = NearestNeighbors(n_neighbors=knn).fit(points_copy)
        distances, _ = nbrs.kneighbors(points_copy)
        # Inverse of the distance to the k nearest neighbours as weight
        weights = 1 / (distances[:, -1] + 1e-9) # Add a small value to avoid division by zero
    
    # Compute the normals
    if len(points_copy) < max_nn:
        return np.array([]), np.array([]), np.array([])
    
    # Create a KDTree for efficient neighbour search
    tree = NearestNeighbors(radius=radius, n_neighbors=max_nn).fit(points_copy)

    # Precompute neighbors for all points
    neighbors_indices = tree.radius_neighbors(points_copy, return_distance=False)

    # Initialize normals array
    normals = np.full((len(points_copy), 3), np.nan)

    for i, indices in enumerate(neighbors_indices):
        if len(indices) < max_nn and len(indices) < 3 and len(points_copy) > 0:
            continue

        # Ensure indices are within the bounds of 'points'
        valid_indices = [idx for idx in indices if 0 <= idx < len(points_copy)]
        if len(valid_indices) < max_nn:
            continue
        
        neighbours = points_copy[valid_indices]


        # Compute the covariance matrix
        covariance_matrix = np.cov(neighbours, rowvar=False)

        # Compute the eigenvalues and eigenvectors
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

        # Get the normal
        normal = eigenvectors[:, np.argmin(eigenvalues)]
        normals[i] = normal
    
    # Compute the angles
    angles = np.arccos(np.dot(normals, np.array([0, 0, 1])))
    angles = np.where(angles > np.pi / 2, np.pi - angles, angles)
    angles = np.degrees(angles)

    # Compute LIAD for each voxel
    if len(angles) == 0 or np.all(np.isnan(angles)):
        pass
    
    if len(weights) == 0:
        weights = np.ones_like(angles)

    # Remove NaN angles and align weights
    valid_mask = ~np.isnan(angles)
    angles = angles[valid_mask]
    weights = weights[valid_mask]

    if len(angles) == 0:
        pass

    total_weights = np.sum(weights)
    if total_weights > 0:
        weights /= total_weights
    else:
        weights = np.ones_like(angles)

    # Compute the histogram
    hist, bin_edges = np.histogram(angles, bins=num_bins, range=(0, 90), weights=weights) # Switch 90 to np.pi to include upward angles
    total_weight = np.sum(hist)
    if total_weight > 0:
        LIAD_values = hist / total_weight
    else:
        LIAD_values = np.zeros(num_bins)

    # Compute the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres, LIAD_values, angles


# Calculate the G function mean
def calculate_G(viewing_angles, bin_centres, LIAD_values, epsilon=1e-9):
    """
    Calculate the G function mean.
    
    INPUTS:
        viewing_angle: The viewing angles
        bin_centres: The bin centres
        LIAD_values: The LIAD values
    
    OUTPUTS:
        G_mean: The G function mean
    """
    # Check for empty arrays
    if len(viewing_angles) == 0 or len(bin_centres) == 0 or len(LIAD_values) == 0:
        return np.nan
    
    # # Normalise LIAD
    # total_LIAD = LIAD_values.sum()
    # LIAD_norm = LIAD_values / total_LIAD if total_LIAD > 0 else LIAD_values
    LIAD_norm = LIAD_values

    # Ensure angles are clipped
    viewing_angles = np.clip(viewing_angles, epsilon, 90)
    bin_centres = np.clip(bin_centres, epsilon, 90)

    ### A(angle, leaf_angle)  ####
    theta_a = np.radians(viewing_angles)
    theta_b = np.radians(bin_centres)

    # Calculate the cotangent of the angles
    cos_theta_a = np.cos(theta_a)
    cot_theta_a = 1 / np.tan(theta_a)
    cos_theta_b = np.cos(theta_b)
    cot_theta_b = 1 / np.tan(theta_b)

    #
    cos_outer = np.outer(cos_theta_a, cos_theta_b)
    cot_outer = np.outer(cot_theta_a, cot_theta_b)

    A = np.zeros_like(cos_outer)
    mask_greater_1 = np.abs(cot_outer) > 1

    A[mask_greater_1] = cos_outer[mask_greater_1]

    inside = np.clip(cot_outer[~mask_greater_1], -1, 1)
    psi = np.arccos(inside)
    factor = 1.0 + (2.0 / np.pi) * (np.tan(psi) - psi)

    A[~mask_greater_1] = factor * cos_outer[~mask_greater_1]

    # Calculate the G function mean for all angles
    delta_bin = np.radians(bin_centres[1] - bin_centres[0])
    G = A @ LIAD_norm # (LIAD_norm * delta_bin)

    return G

### LAD/PAD Functions ###
def CI_adjusted(AD, CI):
    """
    This function takes an ADeff and CI and returns the AD.
    Where, AD = ADeff/CI
    """
    AD = AD/CI
    return AD

def nan_zero_to_default_G_CI(G, CI):
    """
    This function takes an array and a default value and returns the array with nans replaced by the default value.
    """
    if isinstance(G, np.ndarray):
        G = np.where(np.logical_or(np.isnan(G), G==0), 0.5, G)

    if isinstance(CI, np.ndarray):
        CI = np.where(np.logical_or(np.isnan(CI), CI==0), 1.0, CI)
    
    return G, CI

# Beer-Lambert Pimont et al. 2018, eq. 5
def BL_pimont_2018(I, mean_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate density using Beer-Lambert (Pimont et al. 2018), equation 5.
        BL = -log(1 - I) / δ̄


    Calculate PAD by passing I/G values that use all hits, 
    and LAD by passing I/G values that use leaf hits only

    INPUTS:
        I:                  Relative Density Index of voxel (num_hits/num_rays)
        G:                  A provided G_mean value or default 0.5
        CI:                 A provided CI value or default 1.0
        mean_path_length:   Provided mean path length of voxel
        epsilon:            A condition to avoid issues with zero division

    OUTPUTS:
        ADeff:                 The calculated Leaf/Plant Area Density without corrected for CI

    """
    ### CI IS NOT CURRENTLY USED, BUT COULD BE LATER ###

    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        ADeff = np.where(
            (~np.isnan(I) & ~np.isnan(mean_path_length)),
            -(np.log(1 - I) / (G * mean_path_length)),
            np.nan
        )  

        AD = np.where(
            (~np.isnan(ADeff) & (CI != 0)),
            ADeff / CI,
            np.nan
        )
    
    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def BL_EPL_UEPL_pimont_2018(I, mean_eff_path_length, var_eff_path_length, num_rays, G=0.5, epsilon=1e-9, CI=1.0):
    """
    Calculate density using Beer-Lambert (Pimont et al. 2018) with Effective Path Length, equation 25.
        Λ̂ = {
          -1 / δ̄ₑ * (log(1 - I) + I / (2N(1 - I)))      when I < 1
          log(2N + 2) / δ̄ₑ                              when I = 1
        }
    
    &

    Calculate the unbiased effective path length (UEPL) (Pimont et al. 2018, eq. 27), based on the shared EPL value before G correction
        Λ̅₂ = 1 / aₑ * (1 - sqrt(1 - 2 * aₑ * Λ̅))
        
        where:
        Λ̅₂ is the second Lambda with a bar over it
        aₑ is a subscripted 'a' with 'e'
        sqrt represents the square root

    Calculate PAD by passing I values that use all hits,
    and LAD by passing I values that use leaf hits only

    INPUTS:
        I:              A numpy array of Relative Density Indexes (num_hits/num_rays)
        mean_eff_path_length:   A numpy array of mean_eff_path_length
        num_rays:       A numpy array of num_rays
        epsilon:        A condition to avoid issues with zero division

    OUTPUTS:
        ADeff_EPL:          The calculated density, without correcting for CI from EPL
        ADeff_UEPL:         The calculated density, without correcting for CI from UEPL

    """
    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        # Check for nans in inputs
        valid_mask = (
            ~np.isnan(I) & 
            ~np.isnan(mean_eff_path_length) & 
            np.logical_and(~np.isnan(num_rays), num_rays > 0)
        )

        # Split I < 1 and I == 1 values to handle separate calculations
        I_lt_1_mask = I < 1
        I_eq_1_mask = I == 1

        # Calculate ADeff_EPL (L or P depending on inputs)
        ADeff_EPL = np.where(
            np.logical_and(I_lt_1_mask, valid_mask),    # I < 1
            -(1 / mean_eff_path_length) * (np.log(1 - I) + (I / (2 * num_rays * (1 - I)))),
            np.where(
                np.logical_and(I_eq_1_mask, valid_mask),    # I == 1
                np.log(2 * num_rays + 2) / mean_eff_path_length,
                np.nan          # Other
            )
        )

        # Calculate ADeff_UEPL (L or P depending on inputs)
        valid_UEPL_mask = (
            np.logical_and(~np.isnan(ADeff_EPL), (ADeff_EPL > 0)) &
            (mean_eff_path_length > 0) & 
            (var_eff_path_length > 0)
        ) 
        a_e = np.where(
            valid_UEPL_mask,
            var_eff_path_length / mean_eff_path_length,
            np.nan
        )
        ADeff_UEPL = np.where(
            valid_UEPL_mask,
            1 / a_e * (1 - np.sqrt(1 - 2 * a_e * ADeff_EPL)),
            np.nan
        )

        # Correct both ADeff values with G
        ADeff_EPL = np.where(
            ~np.isnan(ADeff_EPL) & (G > 0),
            ADeff_EPL / G,
            np.nan
        )
        ADeff_UEPL = np.where(
            ~np.isnan(ADeff_UEPL) & (G > 0),
            ADeff_UEPL / G,
            np.nan
        )

        AD_EPL = ADeff_EPL / CI
        AD_UEPL = ADeff_UEPL / CI

    except Exception as e:
        print(f"Error: {e}")
        return np.nan, np.nan
    
    return AD_EPL, AD_UEPL

def MCF_beland_2011(I, mean_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Modified Contact Frequency (MCF) using the formula from Pimont et al. 2018 (eq. 8).

    λ̃ = I / z̅  (See paper for more details about this simplification)
    and corrected for G (i.e. / G)
    
    INPUTS:
        mean_free_path_lengths: The mean z value
        I: The relative density index (num_hits/num_rays)
        G: The G function value
        epsilon: A condition to avoid issues with zero division

    OUTPUTS:
        ADeff: The calculated Mean Crown Fraction
    """
    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        # Calculate MCF
        AD = I / (mean_free_path_length * G) / CI

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def MCF_corrected_beland_2014(mean_free_path_length, I, lambda_1, mean_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the corrected Modified Contact Frequency (MCF) using the formula from Pimont et al. 2018 (eq. 9).

    λ̃ = I / z̅ * (1 + λ₁ * δ̄)  (See paper for more details about this simplification)
    and corrected for G (i.e. / G)
    
    INPUTS:
        mean_free_path_lengths: The mean z value
        I: The relative density index (num_hits/num_rays)
        lambda_1: The lambda_1 value
        mean_path_lengths: The mean path length
        G: The G function value
        epsilon: A condition to avoid issues with zero division

    OUTPUTS:
        ADeff: The calculated density from corrected Modified Contact Frequency 
    """
    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        valid_mask = (
            (mean_free_path_length > epsilon) & 
            (I > 0) & (I < 1) &
            (lambda_1 > 0) &
            (mean_path_length > 0)
        )

        ADeff = np.where(
            valid_mask,
            -1 * (lambda_1 * mean_path_length * I) / (np.log(1 - lambda_1 * mean_path_length) * mean_free_path_length),
            np.nan
        )

        # Correct for G
        ADeff = np.where(
            ~np.isnan(ADeff) & (G > 0),
            ADeff / G,
            np.nan
        )

        AD = np.where(
            ~np.isnan(ADeff) & ~np.isnan(CI) & (CI > 0),
            ADeff / CI,
            np.nan
        )

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def MLE_pimont_2019(woody_vol_proportion, num_hits, num_leaf_hits, sum_eff_free_path_length_hit_leaf, sum_eff_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Maximum Likelihood Estimation (MLE) using the formula from Pimont et al. 2018 (eq. 10).
    λ̃ = (1 - I) / (z̅ * G)  (See paper for more details about this simplification)

    For LAD, pass in the sum_hits_effective_path_length array for leaf only.
    
    """

    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        leaf_fraction = np.where(
            np.logical_and(num_hits > 0, num_leaf_hits > 0), 
            np.where(num_hits == num_leaf_hits, 1, num_leaf_hits / num_hits),
            0
        )

        valid_mask = (
            (woody_vol_proportion > 0) &
            (leaf_fraction > 0) &
            (G > 0) &
            (sum_eff_free_path_length_hit_leaf > 0) &
            (sum_eff_free_path_length > 0) &
            (num_hits > 0)
        )

        ADeff = np.where(
            valid_mask,
            (woody_vol_proportion * leaf_fraction / (G * sum_eff_free_path_length)) * (num_hits - sum_eff_free_path_length_hit_leaf / sum_eff_free_path_length),
            np.nan
        )

        AD = np.where(
            ~np.isnan(ADeff) & ~np.isnan(CI) & (CI > 0),
            ADeff / CI,
            np.nan
        )

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def MLE_soma_2021(num_hits, num_leaf_hits, sum_free_path_length_hit_leaf, sum_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Maximum Likelihood Estimation (MLE) using the formula from Soma et al. 2021 (eq. 10).
    λ̃ = (1 - I) / (z̅ * G)  (See paper for more details about this simplification)
    
    """

    try:
        G, CI = nan_zero_to_default_G_CI(G, CI)

        leaf_fraction = np.where(
            np.logical_and(num_hits > 0, num_leaf_hits > 0), 
            np.where(num_hits == num_leaf_hits, 1, num_leaf_hits / num_hits),
            0
        )

        valid_mask = (
            (leaf_fraction > 0) &
            (G > 0) &
            (sum_free_path_length_hit_leaf > 0) &
            (sum_free_path_length > 0) &
            (num_hits > 0)
        )

        ADeff = np.where(
            valid_mask,
            (leaf_fraction / (G * sum_free_path_length)) * (num_hits - sum_free_path_length_hit_leaf / sum_free_path_length),
            np.nan
        )

        AD = np.where(
            ~np.isnan(ADeff) & ~np.isnan(CI) & (CI > 0),
            ADeff / CI,
            np.nan
        )

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def PAD_MLE(value):
    ### INSERT ###
    pass


# Functions used for voxel ray intersections

# Find viewing angles of the rays in comparison with straight up
# Normalise between 0 and 90 degrees
def find_viewing_angles(directions, reference_vector=np.array([0, 0, 1])):
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized_directions = directions / dir_norms
    dot_products = np.dot(normalized_directions, reference_vector)
    cos_thetas = np.clip(dot_products, -1, 1)
    viewing_angle = np.degrees(np.arccos(cos_thetas))
    viewing_angle = np.where(viewing_angle > 90, 180 - viewing_angle, viewing_angle)  # Adjust angles over 90 degrees
    return viewing_angle

# Function to traverse the voxels and find ray intersections
def traverse_voxels(voxel_references, ray_partition, chunks_per_compute, temp_dir, epsilon=1e-6):

    if ray_partition.empty:
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
    
    # Prep ray information
    leg_ids = ray_partition['leg_id'].values
    ray_ids = ray_partition['ray_id'].values
    origins = np.asarray(ray_partition[['origin_x', 'origin_y', 'origin_z']].values)
    directions = np.asarray(ray_partition[['direction_x', 'direction_y', 'direction_z']].values)
    points = np.asarray(ray_partition[['point_x', 'point_y', 'point_z']].values)
    is_leaf = np.asarray(ray_partition['is_leaf'].values)

    num_rays = len(ray_partition)
    num_voxels = len(voxel_references)
    
    voxel_ids = voxel_references['voxel_id'].values
    voxel_sizes = voxel_references['voxel_size'].values
    voxel_mins = np.asarray(voxel_references[['voxel_min_x', 'voxel_min_y', 'voxel_min_z']].values) - epsilon
    voxel_maxs = np.asarray(voxel_references[['voxel_max_x', 'voxel_max_y', 'voxel_max_z']].values) + epsilon

    del voxel_references, ray_partition
    # gc.collect()

    voxel_ids = voxel_ids[:, np.newaxis]
    voxel_mins = voxel_mins[:, np.newaxis, :]
    voxel_maxs = voxel_maxs[:, np.newaxis, :]

    leg_ids = leg_ids[np.newaxis, :]
    ray_ids = ray_ids[np.newaxis, :]
    is_leaf = is_leaf[np.newaxis, :]
    origins = origins[np.newaxis, :, :]
    directions = directions[np.newaxis, :, :]
    points = points[np.newaxis, :, :]    

    ### TEST DATA ###
    # voxel_mins = np.array([
    #     [[0, 0, 0]],
    #     [[1, 1, 1]],
    #     [[2, 2, 2]]
    # ])  # Shape (3, 1, 3)

    # voxel_maxs = np.array([
    #     [[1, 1, 1]],
    #     [[2, 2, 2]],
    #     [[3, 3, 3]]
    # ])  # Shape (3, 1, 3)

    # origins = np.array([
    #     [[0.5, 0.5, -1],
    #      [1.5, 1.5, -1],
    #      [2.5, 2.5, -1]]
    # ])  # Shape (1, 3, 3)

    # directions = np.array([
    #     [[0, 0, 1],
    #      [0, 0, 1],
    #      [0, 0, 1]]
    # ])  # Shape (1, 3, 3)

    # Expected results:
    # - Ray 1 intersects voxel 1
    # - Ray 2 intersects voxel 2
    # - Ray 3 intersects voxel 3

    def broadcasted_ray_tracing(voxel_mins, voxel_maxs, origins, directions, beam_divergence = 0.001, epsilon=np.float64(1e-6)):
        # Calculate t_min and t_max for each dimension
        directions = np.where(
            np.abs(directions) <= epsilon,
            np.where(directions == 0, epsilon, np.sign(directions) * epsilon),
            directions
        )

        t_min = (voxel_mins - origins) / directions
        t_max = (voxel_maxs - origins) / directions

        t_enter = np.max(np.minimum(t_min, t_max), axis=2)
        t_exit = np.min(np.maximum(t_min, t_max), axis=2)

        # Check if t_enter is less than t_exit
        mask = (t_enter <= t_exit + epsilon) & (t_exit >= -epsilon)

        # Setup arrays for returned values
        # t_enter_coords = np.full((mask.shape[0], mask.shape[1], origins.shape[2]), np.nan, dtype=np.float64)
        # t_exit_coords = np.full((mask.shape[0], mask.shape[1], origins.shape[2]), np.nan, dtype=np.float64)
        # t_entry_radii = np.full_like(mask, np.nan, dtype=np.float64)
        # t_exit_radii = np.full_like(mask, np.nan, dtype=np.float64)

        # # If there are any true values in mask, run calculations
        # has_nonzero = np.any(mask, axis=(0,1))
        # if has_nonzero:
        #     # Calculate the entry and exit coordinates for valid rays
        #     origins = np.broadcast_to(origins, (mask.shape[0], mask.shape[1], origins.shape[2]))
        #     directions = np.broadcast_to(directions, (mask.shape[0], mask.shape[1], directions.shape[2]))
            
        #     origins = origins[mask]
        #     directions = directions[mask]
        #     t_enter = t_enter[mask]
        #     t_exit = t_exit[mask]
            
        #     t_enter_coords[mask] = origins + t_enter[:, np.newaxis] * directions
        #     t_exit_coords[mask] = origins + t_exit[:, np.newaxis] * directions

        #     # Calculate the radii from beam divergence using t_enter and t_exit as distances
        #     t_entry_radii[mask] = (t_enter * np.tan(beam_divergence)).astype(np.float64)
        #     t_exit_radii[mask] = (t_exit * np.tan(beam_divergence)).astype(np.float64)
        
        return mask # , t_enter_coords, t_exit_coords, t_entry_radii, t_exit_radii


        
    
    # Calculate mask, t_enter, and t_exit for max voxels that fit into memory
    masks = {}
    # t_enter_coords = {}
    # t_exit_coords = {}
    # t_entry_radiis = {}
    # t_exit_radiis = {}

    temp_dir = tempfile.mkdtemp(dir=temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Generate a unique ID for the process
    process_id = uuid.uuid4().hex

    print(f"Process {process_id}: Start {num_rays} rays, {num_voxels} voxels, in ({int(np.ceil(num_voxels / chunks_per_compute))}) chunks.")
    
    for i in range(0, voxel_mins.shape[0], chunks_per_compute):

        # Calculate the number of chunks to process in this iteration
        chunk_mask = broadcasted_ray_tracing( #, chunk_t_enter_coord, chunk_t_exit_coord, chunk_t_entry_radii, chunk_t_exit_radii = broadcasted_ray_tracing(
            voxel_mins[i:i+chunks_per_compute, :, :], 
            voxel_maxs[i:i+chunks_per_compute, :, :], 
            origins, 
            directions
        )

        # Save chunk_mask, t_enter, and t_exit to disk with unique filenames
        chunk_mask_filename = os.path.join(temp_dir, f"chunk_mask_{i}_{process_id}.npy")
        # t_enter_filename = os.path.join(temp_dir, f"t_enter_{i}_{process_id}.npy")
        # t_exit_filename = os.path.join(temp_dir, f"t_exit_{i}_{process_id}.npy")
        # t_entry_radii_filename = os.path.join(temp_dir, f"t_entry_radii_{i}_{process_id}.npy")
        # t_exit_radii_filename = os.path.join(temp_dir, f"t_exit_radii_{i}_{process_id}.npy")

        # Save arrays to disk
        np.save(chunk_mask_filename, chunk_mask)
        # np.save(t_enter_filename, chunk_t_enter_coord)
        # np.save(t_exit_filename, chunk_t_exit_coord)
        # np.save(t_entry_radii_filename, chunk_t_entry_radii)
        # np.save(t_exit_radii_filename, chunk_t_exit_radii)

        dtype = chunk_mask.dtype
        shape = chunk_mask.shape
        masks[i] = [chunk_mask_filename, dtype, shape]

        # dtype = chunk_t_enter_coord.dtype
        # shape = chunk_t_enter_coord.shape
        # t_enter_coords[i] = [t_enter_filename, dtype, shape]

        # dtype = chunk_t_exit_coord.dtype
        # shape = chunk_t_exit_coord.shape
        # t_exit_coords[i] = [t_exit_filename, dtype, shape]

        # dtype = chunk_t_entry_radii.dtype
        # shape = chunk_t_entry_radii.shape
        # t_entry_radiis[i] = [t_entry_radii_filename, dtype, shape]

        # dtype = chunk_t_exit_radii.dtype
        # shape = chunk_t_exit_radii.shape
        # t_exit_radiis[i] = [t_exit_radii_filename, dtype, shape]


        del chunk_mask #, chunk_t_enter_coord, chunk_t_exit_coord, chunk_t_entry_radii, chunk_t_exit_radii
        gc.collect()

    print(f"Process {process_id}: Finished {num_rays} rays, {num_voxels} voxels. Concatenating results...")

    # Combine masks, t_enters, and t_exits into single arrays for further processing
    mask = None
    chunk_masks = []
    for key in sorted(masks.keys()):
        # Use np.memmap to map the saved files
        filename = masks[key][0]
        dtype = masks[key][1]
        shape = masks[key][2]

        # with open(filename, 'rb') as f:
        #     chunk_mask = pickle.load(f)
        chunk = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
        chunk_masks.append(chunk)
        
    mask = np.concatenate(chunk_masks, axis=0)

    # Flatten mask and retrieve idx for rays and voxels
    voxel_ref_idx, ray_ref_idx = np.nonzero(mask)
    if len(voxel_ref_idx) == 0:
        # print("No valid rays found.")
        # Cleanup memory
        del mask, voxel_ref_idx, ray_ref_idx
        del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, leg_ids, ray_ids, is_leaf, points, origins, directions
        gc.collect()
        # Delete the temporary folder and its contents
        shutil.rmtree(temp_dir, ignore_errors=True)

        # Return an empty DataFrame with the same schema
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
    
    del mask, chunk_masks, chunk
    gc.collect()

    # # Combine t_enter and t_exit arrays
    # t_enters = []
    # for key in sorted(t_enter_coords.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_enter_coords[key][0]
    #     dtype = t_enter_coords[key][1]
    #     shape = t_enter_coords[key][2]

    #     chunk_t_enter = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_enters.append(chunk_t_enter)
    
    # filtered_t_entry_coords = np.concatenate(t_enters, axis=0)    
    # filtered_t_entry_coords = filtered_t_entry_coords[voxel_ref_idx, ray_ref_idx]
    # # Cleanup memory
    # del t_enter_coords, t_enters
    # gc.collect()

    # t_exits = []
    # for key in sorted(t_exit_coords.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_exit_coords[key][0]
    #     dtype = t_exit_coords[key][1]
    #     shape = t_exit_coords[key][2]

    #     chunk_t_exit = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_exits.append(chunk_t_exit)

    # filtered_t_exit_coords = np.concatenate(t_exits, axis=0)
    # filtered_t_exit_coords = filtered_t_exit_coords[voxel_ref_idx, ray_ref_idx]
    # # Cleanup memory
    # del t_exit_coords, t_exits
    # gc.collect()

    # # t_entry_radii
    # t_en_radiis = []
    # for key in sorted(t_entry_radiis.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_entry_radiis[key][0]
    #     dtype = t_entry_radiis[key][1]
    #     shape = t_entry_radiis[key][2]

    #     chunk_t_entry_radii = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_en_radiis.append(chunk_t_entry_radii)

    # filtered_t_entry_radii = np.concatenate(t_en_radiis, axis=0)
    # filtered_t_entry_radii = filtered_t_entry_radii[voxel_ref_idx, ray_ref_idx]
    # # Cleanup memory
    # del t_en_radiis, chunk_t_entry_radii, t_entry_radiis
    # gc.collect()

    # # t_exit_radii
    # t_ex_radiis = []
    # for key in sorted(t_exit_radiis.keys()):
    #     # Use np.memmap to map the saved files
    #     filename = t_exit_radiis[key][0]
    #     dtype = t_exit_radiis[key][1]
    #     shape = t_exit_radiis[key][2]

    #     chunk_t_exit_radii = np.lib.format.open_memmap(filename, mode='r', dtype=dtype, shape=shape)
    #     t_ex_radiis.append(chunk_t_exit_radii)

    # filtered_t_exit_radii = np.concatenate(t_ex_radiis, axis=0)
    # filtered_t_exit_radii = filtered_t_exit_radii[voxel_ref_idx, ray_ref_idx]

    # del t_ex_radiis, chunk_t_exit_radii, t_exit_radiis
    # gc.collect()

    # Delete the temporary folder and its contents
    shutil.rmtree(temp_dir, ignore_errors=True)
    

    # Flatten all values to match the mask
    filtered_voxel_ids = voxel_ids[voxel_ref_idx].reshape(-1)
    filtered_voxel_sizes = voxel_sizes[voxel_ref_idx]
    filtered_voxel_mins = voxel_mins[voxel_ref_idx].reshape(-1, 3)
    filtered_voxel_maxs = voxel_maxs[voxel_ref_idx].reshape(-1, 3)

    filtered_leg_ids = leg_ids[:, ray_ref_idx].reshape(-1)
    filtered_ray_ids = ray_ids[:, ray_ref_idx].reshape(-1)
    filtered_is_leaf = is_leaf[:, ray_ref_idx].reshape(-1)
    filtered_points = points[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_origins = origins[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_directions = directions[:, ray_ref_idx, :].reshape(-1, 3)

    # Find viewing angles based on the filtered directions
    filtered_viewing_angles = find_viewing_angles(directions=filtered_directions)

    # Cleanup memory
    del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, leg_ids, ray_ids, is_leaf, points, origins, directions
    # gc.collect()

    # Filter out points that are within each respective voxel  
    filtered_hit_rays = np.all((filtered_points >= filtered_voxel_mins) & (filtered_points <= filtered_voxel_maxs), axis=1)
    
    # Filter out rays which have previously hit a voxel (and therefore do not continue into this next one)
    # Also keep any NAN points (i.e. unbound rays)
    # distance_to_exit = np.linalg.norm(filtered_t_exit_coords - filtered_origins, axis=1)
    filtered_directions = np.where(
        np.abs(filtered_directions) <= epsilon,
        np.where(filtered_directions == 0, epsilon, np.sign(filtered_directions) * epsilon),
        filtered_directions
    )
    t_min = (filtered_voxel_mins - filtered_origins) / filtered_directions
    t_max = (filtered_voxel_maxs - filtered_origins) / filtered_directions
    t_enter = np.max(np.minimum(t_min, t_max), axis=1)
    t_exit = np.min(np.maximum(t_min, t_max), axis=1)
    del t_min, t_max
    gc.collect()

    t_exit_coords = filtered_origins + t_exit[:, np.newaxis] * filtered_directions
    t_enter_coords = filtered_origins + t_enter[:, np.newaxis] * filtered_directions
    distance_to_exit_squared = np.sum((filtered_origins + t_exit[:, np.newaxis] * filtered_directions - filtered_origins) ** 2, axis=1)
    distance_to_point_squared = np.sum((filtered_points - filtered_origins) ** 2, axis=1)
    yet_to_hit_rays = np.logical_or(distance_to_point_squared > distance_to_exit_squared, np.isnan(filtered_points).any(axis=1)) 
    valid_ray_mask = np.logical_or(filtered_hit_rays, yet_to_hit_rays)

    if not np.any(valid_ray_mask):
        print("No valid rays intersect these voxels.")
        return pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)
        
    del filtered_voxel_mins, filtered_voxel_maxs
    gc.collect()
    
    # Ensure only hit ray points are kept per voxel
    filtered_points = np.where(
        filtered_hit_rays[:, None],
        filtered_points,
        np.full(filtered_points.shape, np.nan)
    )

    # Remove any invalid rays
    filtered_voxel_sizes = filtered_voxel_sizes[valid_ray_mask]
    filtered_voxel_ids = filtered_voxel_ids[valid_ray_mask]
    filtered_leg_ids = filtered_leg_ids[valid_ray_mask]
    filtered_ray_ids = filtered_ray_ids[valid_ray_mask]
    # filtered_t_entry_coords = filtered_t_entry_coords[valid_ray_mask]
    # filtered_t_exit_coords = filtered_t_exit_coords[valid_ray_mask]
    # filtered_t_entry_radii = filtered_t_entry_radii[valid_ray_mask]
    # filtered_t_exit_radii = filtered_t_exit_radii[valid_ray_mask]
    filtered_points = filtered_points[valid_ray_mask]
    filtered_viewing_angles = filtered_viewing_angles[valid_ray_mask]
    filtered_hit_rays = filtered_hit_rays[valid_ray_mask]
    filtered_is_leaf = filtered_is_leaf[valid_ray_mask] 
    filtered_origins = filtered_origins[valid_ray_mask]
    filtered_directions = filtered_directions[valid_ray_mask]

    filtered_t_exit_coords = t_exit_coords[valid_ray_mask]
    filtered_t_entry_coords = t_enter_coords[valid_ray_mask]
    del t_exit_coords, t_enter_coords
    gc.collect()

    beam_divergence = 0.001
    filtered_t_entry_radii = t_enter[valid_ray_mask] * np.tan(beam_divergence)
    filtered_t_exit_radii = t_exit[valid_ray_mask] * np.tan(beam_divergence)
    del t_enter, t_exit, filtered_origins, filtered_directions
    gc.collect()

    data = [
        pa.array(filtered_voxel_sizes),
        pa.array(filtered_voxel_ids),
        pa.array(filtered_leg_ids),
        pa.array(filtered_ray_ids),
        pa.array(filtered_t_entry_coords[:, 0]),
        pa.array(filtered_t_entry_coords[:, 1]),
        pa.array(filtered_t_entry_coords[:, 2]),
        pa.array(filtered_t_exit_coords[:, 0]),
        pa.array(filtered_t_exit_coords[:, 1]),
        pa.array(filtered_t_exit_coords[:, 2]),
        pa.array(filtered_t_entry_radii),
        pa.array(filtered_t_exit_radii),
        pa.array(filtered_points[:, 0]),
        pa.array(filtered_points[:, 1]),
        pa.array(filtered_points[:, 2]),
        pa.array(filtered_viewing_angles),
        pa.array(filtered_hit_rays),
        pa.array(filtered_is_leaf)
    ]
    result = pa.Table.from_arrays(data, schema=voxel_ray_intersection_schema_old)
    result = result.to_pandas()

    del filtered_voxel_sizes, filtered_voxel_ids, filtered_leg_ids, filtered_ray_ids, filtered_t_entry_coords, filtered_t_exit_coords, filtered_t_entry_radii, filtered_t_exit_radii, filtered_points, filtered_viewing_angles, filtered_hit_rays, filtered_is_leaf
    gc.collect()

    print(f"Process {process_id}. Returning results...")

    return result


    








    







### LARGE FUNCTIONS ###
# Functions that are used to perform large operations, such as calculating metrics or processing data.

# Prepare data from helios simulations
def prepare_helios_data(input_dir, output_dir, references_dir, leaf_object_ids, debug=True, epsilon=1e-6):
    """
    Main function to process helios simulation data.
    
    Args:
        input_dir (str): Path to the input folder containing helios simulation data.
        output_dir (str): Path to the output folder where processed data will be saved.
    """
    # Import modules
    import os
    import glob
    import shutil
    import logging
    import dask.delayed
    from dask.diagnostics import ProgressBar
    import pandas as pd
    import numpy as np
    import dask.array as da
    import dask.dataframe as dd
    import dask

    # Check if the input folder exists
    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"The input folder '{input_dir}' does not exist.")
    if not os.path.exists(references_dir):
        raise FileNotFoundError(f"The references folder '{references_dir} does not exist.")

    # Check if the output folder exists, if not create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Output folder '{output_dir}' created.")
  
    log_file = os.path.join(output_dir, f"valid_rays.log")

    logger = logging.getLogger()
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(filename=log_file, encoding='utf-8', level=level)

    logger.info(f"Preparing data from '{input_dir}' to '{output_dir}'...")

    # Setup valid rays filename template
    valid_rays_template = "leg_{leg:d}_valid_rays.parquet"

    ### PLOT BOUNDARY CALCULATION ###

    # Establish the plot boundaries of the plot, regardless of voxel size
    logger.info("Finding all voxel references to establish plot boundary.")
    voxel_references = glob.glob(os.path.join(references_dir, '*.csv'))

    dfs =[]
    for voxel_ref in voxel_references:
        voxel_size = os.path.basename(voxel_ref)
        voxel_size = os.path.splitext(voxel_size)[0]
        if "voxel_size_" in voxel_size:
            voxel_size = float(voxel_size.split("voxel_size_")[1])
        else:
            raise ValueError(f"Voxel size not found in {voxel_ref}. Please check the file name.")

        df = pd.read_csv(voxel_ref, index_col=None, header=0)
        

        if 'voxel_id' not in df.columns:
            logger.warning(f"No voxel_id found in {voxel_ref}. Updating csv now.")
            new_df = df[['voxel_cx', 'voxel_cy', 'voxel_cz']]
            new_df['voxel_size'] = voxel_size

            def parallel_voxel_id(pd_series):
                voxel_size = pd_series['voxel_size']
                x = pd_series['voxel_cx']
                y = pd_series['voxel_cy']
                z = pd_series['voxel_cz']
                voxel_id = create_voxel_id(voxel_size=voxel_size, x=x, y=y, z=z)

                return voxel_id

            # Add unique voxel_ids back to csv.
            voxel_ids = new_df.apply(parallel_voxel_id, axis=1)
            df['voxel_id'] = voxel_ids

            df.to_csv(voxel_ref)

            logger.info(f"Updated voxel_ids for {voxel_ref}")

        df = df[['voxel_cx', 'voxel_cy', 'voxel_cz']].astype(float)
        voxel_size = float(voxel_size)
        df['min_x'] = df['voxel_cx'].min() - (voxel_size / 2 + epsilon)
        df['max_x'] = df['voxel_cx'].max() + (voxel_size / 2 + epsilon)
        df['min_y'] = df['voxel_cy'].min() - (voxel_size / 2 + epsilon)
        df['max_y'] = df['voxel_cy'].max() + (voxel_size / 2 + epsilon)
        df['min_z'] = df['voxel_cz'].min() - (voxel_size / 2 + epsilon)
        df['max_z'] = df['voxel_cz'].max() + (voxel_size / 2 + epsilon)

        df = df[['min_x', 'max_x', 'min_y', 'max_y', 'min_z', 'max_z']]
        dfs.append(df)

    plot_bounds = pd.concat(dfs, axis=0, ignore_index=True)
    plot_min = np.array([plot_bounds['min_x'].min(), plot_bounds['min_y'].min(), plot_bounds['min_z'].min()])
    plot_max = np.array([plot_bounds['max_x'].max(), plot_bounds['max_y'].max(), plot_bounds['max_z'].max()])

    logger.info(f"Plot boundaries calculated as min: {plot_min} and max: {plot_max}")

    # Cleanup memory
    del plot_bounds, df, dfs, voxel_references

    ### DEFINE FUNCTIONS REQUIRED FOR VALID RAYS PREPARATION ###

    # Function to traverse plot and remove any rays which do not intersect the voxel plot
    # Plot min and max are already calculated
    def is_in_plot(ray_origins, ray_directions):
        t_min = da.divide(plot_min - ray_origins, ray_directions, out=da.full_like(ray_origins, np.inf), where=da.all(ray_directions!=0))
        t_max = da.divide(plot_max - ray_origins, ray_directions, out=da.full_like(ray_origins, np.inf), where=da.all(ray_directions!=0))
        t1 = da.minimum(t_min, t_max)
        t2 = da.maximum(t_min, t_max)
        t_enter = da.max(t1, axis=1)
        t_exit = da.min(t2, axis=1)
        mask = (t_enter <= t_exit) & (t_exit >= 0)
        return mask
    
    # Function to enable map_partitions functionality
    def valid_mask(df):
        origins = da.array([df['origin_x'].values, df['origin_y'].values, df['origin_z'].values]).T
        directions = da.array([df['direction_x'].values, df['direction_y'].values, df['direction_z'].values]).T
        mask = is_in_plot(origins, directions).compute()
        return df[mask]
    
    @dask.delayed
    def import_helios_leg(pulse_file, xyz_file):
        # Import all rays into dask dataframe
        leg_rays = dd.read_csv(pulse_file, delimiter=' ', header=None, names=['origin_x', 'origin_y', 'origin_z', 'direction_x', 'direction_y', 'direction_z', 'gps_time', 'ray_id', '_'])
        leg_rays = leg_rays.drop(columns=['gps_time', '_'])

        # Check for valid rays on partitions
        leg_rays = leg_rays.map_partitions(valid_mask, meta=leg_rays._meta)

        # Import all hits into dask dataframe
        leg_hits = dd.read_csv(xyz_file, delimiter=' ', header=None, names=['point_x', 'point_y', 'point_z', 'intensity', 'echo_width', 'return_number', 'number_of_returns', 'ray_id', 'hit_object_id', 'class', 'gps_time'])
        leg_hits = leg_hits.drop(columns=['intensity', 'echo_width', 'return_number', 'number_of_returns', 'class', 'gps_time'])

        leg_rays = leg_rays.merge(leg_hits, on='ray_id', how='left')

        return leg_rays
    

    ### START LEG RAY PROCESSING ###
    pulses = glob.glob(os.path.join(input_dir, '*_pulse.txt'))
    points = glob.glob(os.path.join(input_dir, '*_points.xyz'))
    pulses.sort()
    points.sort()

    ray_processing_list = []
    for i, pulse_file in enumerate(pulses):
        xyz_file = points[i]

        leg = pulse_file.split("leg")[1].split("_")[0]

        delayed_result = import_helios_leg(pulse_file=pulse_file, xyz_file=xyz_file)
        ray_processing_list.append((leg, delayed_result))

    # Count number of rays and points in output
    total_rays = 0
    total_points = 0

    # Process legs
    with ProgressBar():
        statement = "Processing dask delayed functions..."
        print(statement)
        logger.info(statement)
        results = dask.compute(*[ray_processing_list[i][1] for i in range(len(ray_processing_list))])
        
        for leg, result in enumerate(results):
            statement = f"Processing leg {leg}..."
            logger.info(statement)
            print(statement)

            rays = result.compute()
            logger.info("Dask computation complete.")
            
            leg = int(leg)
            rays_file = os.path.join(output_dir, valid_rays_template.format(leg=leg))
            if os.path.exists(rays_file):
                if os.path.isfile(rays_file):
                    os.remove(rays_file)
                else:
                    shutil.rmtree(rays_file)
            
            logger.info(f"Saving valid rays for leg {leg} to {rays_file}")
            rays['leg_id'] = leg
            rays['is_leaf'] = rays['hit_object_id'].isin(leaf_object_ids)

            rays = rays.drop(columns=['hit_object_id'])
            rays.to_parquet(rays_file, engine='pyarrow', compression='snappy', schema=valid_rays_schema)

            logger.info("Counting points...")

            num_rays = len(rays)
            num_points = (~rays['point_x'].isna()).sum()
            logger.info(f"Leg {leg} has {num_rays} valid rays and {num_points} points.")

            total_rays += int(num_rays)
            total_points += int(num_points)
            logger.info(f"Updated totals: {total_rays} rays and {total_points} points.")
    
    statement= "Helios data preparation complete."
    print(statement)
    logger.info(statement)

# Function used for taking valid_rays parquet files and references to establish voxel_ray intersections per valid_rays file
def voxel_ray_intersections(valid_rays_dir, references_dir, temp_dir, cpus=None, mem=None, debug=True, epsilon=1e-6):
    import os
    import glob
    import pandas as pd
    import numpy as np
    import psutil
    import shutil
    import dask
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar

    # Compile the references files to establish a voxel dataframe of size and voxel_id
    voxel_references = glob.glob(os.path.join(references_dir, '*.csv'))

    dfs = []
    max_voxels = 0
    for voxel_ref in voxel_references:
        # Read the csv
        df = pd.read_csv(voxel_ref, index_col=None, header=0)

        # Filter out unnecessary columns and duplicates
        df = df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz']].drop_duplicates()

        # Add voxel_size to dataframe for later grouping
        voxel_size = float(os.path.splitext(voxel_ref)[0].split("_")[-1])
        df['voxel_size'] = voxel_size

        # Prep dataframe as bounds of each voxel rather than centre (maintaining size and unique id)
        df['voxel_min_x'] = df['voxel_cx'].astype(float) - (voxel_size / 2 + epsilon)
        df['voxel_max_x'] = df['voxel_cx'].astype(float) + (voxel_size / 2 + epsilon)
        df['voxel_min_y'] = df['voxel_cy'].astype(float) - (voxel_size / 2 + epsilon)
        df['voxel_max_y'] = df['voxel_cy'].astype(float) + (voxel_size / 2 + epsilon)
        df['voxel_min_z'] = df['voxel_cz'].astype(float) - (voxel_size / 2 + epsilon)
        df['voxel_max_z'] = df['voxel_cz'].astype(float) + (voxel_size / 2 + epsilon)
        df = df.drop(columns=['voxel_cx', 'voxel_cy', 'voxel_cz'])

        dfs.append(df)

    # Combine into single pandas dataframe for later grouping
    voxel_references = pd.concat(dfs)

    # Calculate max chunk_size based on number of voxels
    # Get available memory in bytes

    def total_mem_required_per_ray(elements_per_ray=3*8+4, memory_per_element=8):
        return elements_per_ray * memory_per_element
    
    def calculate_chunk_size_mb(num_voxels, available_memory, min_chunk_size=100):
        mem_required_per_ray = total_mem_required_per_ray()
        total_mem_per_ray_per_voxel = num_voxels * mem_required_per_ray
        chunk_size = max(min_chunk_size, available_memory / total_mem_per_ray_per_voxel)
        return chunk_size


    # Compile all valid_rays parquets
    valid_rays_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))

    def map_ray_partition_to_function(ray_partition, voxel_group, temp_dir, chunks_per_compute):



        result = traverse_voxels(ray_partition=ray_partition, voxel_references=voxel_group, chunks_per_compute=chunks_per_compute, temp_dir=temp_dir)
        return result

    voxel_ray_intersections = {}

    for file in valid_rays_files:
        # Read in parquet file
        df = dd.read_parquet(file, engine='pyarrow', blocksize=100 * 1024 * 1024)
        cpus = os.cpu_count() if cpus is None else cpus

        if df.npartitions < cpus:
            # Check for the size of resulting partition
            df_mem = df.memory_usage(deep=True).compute().sum()
            target_partition_mem = 25 * 1024 * 1024 # bytes
            if df_mem / cpus < target_partition_mem: # Avoid too small partitions
                cpus = int(df_mem / target_partition_mem)
            
            df = df.repartition(npartitions=cpus)


        # Get leg_id from filename
        leg_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])

        # Map partitions to traverse voxels
        meta = pd.DataFrame(columns=voxel_ray_intersection_schema_old.names)

        if mem == None:
            mem = psutil.virtual_memory().available
        
        available_mem = mem * 0.8
        partition_mem = df.memory_usage_per_partition(deep=True).compute().max()
        num_voxels = len(voxel_references)

        voxel_memory = num_voxels * 12 * 8  # Voxels are broadcast to (n, 1, 8), each element is 8 bytes (float64)
        mem_per_compute = (partition_mem + voxel_memory) * 2
        # Calculate the memory required for broadcasting a single voxel to all rays
        chunks_per_compute = int((available_mem / (mem_per_compute * df.npartitions)))

        if chunks_per_compute < 1:
            required_partitions = int(np.ceil(partition_mem / mem_per_compute))
            max_partitions = int(np.floor(available_mem / (required_partitions * partition_mem)))
            df = df.repartition(npartitions=max_partitions)
            chunks_per_compute = int((available_mem / (mem_per_compute * df.npartitions)))
            while (max_partitions * partition_mem) > available_mem:
                max_partitions -= 1
                df = df.repartition(npartitions=max_partitions)
                partition_mem = df.memory_usage_per_partition(deep=True).compute().max()
            
            chunks_per_compute = int((available_mem / (mem_per_compute * df.npartitions)))

        dd_results = []

        result = df.map_partitions(
            map_ray_partition_to_function,
            voxel_group=voxel_references,
            chunks_per_compute=chunks_per_compute,
            temp_dir=temp_dir,
            meta=meta
        )

        dd_results.append(result)

        voxel_ray_intersections[leg_id] = result

    def save_task(df, leg_id):
        if df.empty:
            return False
        
        # Get the voxel size from the dataframe
        voxel_size = round(float(df.name), 2)

        # Create the output filename
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")

        # Save the dataframe to parquet
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema_old)

        return True


    with ProgressBar():

        for leg_id, results in voxel_ray_intersections.items():

            print(f"Processing leg {leg_id}...")

            results = results.compute()

            print(f"Saving results for leg {leg_id}...")
            results = results.groupby('voxel_size').apply(lambda x: save_task(x, leg_id))

            del results
            gc.collect()

            print(f"Completed leg {leg_id}!")

            # # Save the results to parquet
            # save_task(results, leg_id, voxel_size)
        
        # for leg_id, voxel_sizes in voxel_ray_intersections.items():

        #     for voxel_size, results in voxel_sizes.items():

        #         # print(f"Processing leg {leg_id} with voxel size {voxel_size}...")
        #         # results = list(dask.compute(*results, scheduler='processes', num_workers=n_workers))

        #         # print(f"Saving {len(results)} results for leg {leg_id} with voxel size {voxel_size}...")
        #         # results = pd.concat(results, ignore_index=True)

        #         df_results = []
        #         for i, result in enumerate(results):
        #             # for r in result:

        #             print(f"Processing chunk {i+1} of {len(results)} for leg {leg_id} and voxel size {voxel_size}")
        #             result = result.compute()
        #             df_results.append(result)

        #         results = pd.concat(df_results, ignore_index=True)

        #         save_task(results, leg_id, voxel_size)

        #         gc.collect()


def get_voxel_metrics_old(intersections_files, lambda_1, is_leaf_true=True, debug=True, epsilon=1e-9):
    """old"""
#     """
#     This function will take the voxel_ray_intersection files and calculate the metrics for each voxel.
#     It will save the results to a parquet file.

#     Args:
#         intersections_files (list): List of paths to voxel_ray_intersection files.
#         lambda_1 (float): This is calculated using (average leaf area / voxel size) and will need to be calculated and passed in.
#         debug (bool): Whether to print debug information.
#         epsilon (float): Small value to avoid division by zero.

#     Returns:
#         None
#     """
#     import os
#     import glob
#     import pandas as pd
#     import numpy as np
#     import dask.dataframe as dd
#     from dask.diagnostics import ProgressBar
#     import logging
#     import psutil

    

#     # Setup logger
#     # logger = logging.getLogger()
#     # level = logging.DEBUG if debug else logging.WARNING
#     # logging.basicConfig(filename=os.path.join(intersections_files, 'voxel_metrics.log'), encoding='utf-8', level=level)

#     # Per voxel function
#     def calculate_voxel_metrics_per_voxel(voxel_df, min_rays=6, epsilon=1e-9):
#         """
#         Calculate voxel metrics for a given voxel dataframe.
        
#         INPUTS:
#             voxel_df: A pandas dataframe containing voxel data
#             min_rays: Minimum number of rays to consider a voxel valid
#             epsilon: A small value to avoid division by zero

#         OUTPUTS:
#             voxel_metrics: A pandas dataframe containing the calculated metrics for each voxel
#         """
        
#         # Check if the dataframe is empty
#         if len(voxel_df) == 0:
#             return pd.DataFrame(columns=voxel_metrics_schema.names)

#         # Get voxel_id name
#         voxel_id = voxel_df.name

#         # Calculate the number of rays in each voxel
#         num_rays = voxel_df['ray_id'].count()
#         if num_rays <= 0:
#             statement= f"Voxel {voxel_df['voxel_id'].values[0]} has no rays."
#             print(statement)
#             return pd.DataFrame(columns=voxel_metrics_schema.names)
        
#         num_hits = voxel_df['hit_ray'].sum()
#         num_leaf_hits = voxel_df[(voxel_df['hit_ray']) & (voxel_df['is_leaf'])].shape[0] if is_leaf_true else voxel_df[(voxel_df['hit_ray']) & ~(voxel_df['is_leaf'])].shape[0]

#         # Calculate pgap_lw and I
#         pgap_lw = (num_rays - num_hits) / num_rays
#         I_lw = num_hits / num_rays
#         pgap_leaf = (num_rays - num_leaf_hits) / num_rays
#         I_leaf = num_leaf_hits / num_rays

#         # Calcualte path lengths
#         path_lengths = np.linalg.norm(voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values, axis=1)
#         hit_mask = voxel_df['hit_ray'].values
#         free_path_lengths = np.where(
#             hit_mask,
#             np.linalg.norm(voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values - voxel_df[['point_x', 'point_y', 'point_z']].values, axis=1),
#             path_lengths
#         )

#         # Calculate the sums and means
#         sum_path_length = np.nansum(path_lengths)
#         mean_path_length = np.nanmean(path_lengths)
#         sum_free_path_length = np.nansum(free_path_lengths)
#         mean_free_path_length = np.nanmean(free_path_lengths)
#         sum_free_path_length_hit = np.nansum(free_path_lengths[voxel_df['hit_ray'].values])
#         sum_free_path_length_hit_leaf = np.nansum(free_path_lengths[voxel_df['is_leaf'].values]) if is_leaf_true else np.nansum(free_path_lengths[~voxel_df['is_leaf'].values])
#         # mean_free_path_length_leaf = np.nanmean(free_path_lengths_leaf)

#         # Calculate effective path lengths and free path lengths
#         def calculate_effective_path_length(path_lengths, lambda_1):
#             with np.errstate(divide='ignore', invalid='ignore'):
#                 mask = (lambda_1 * path_lengths) < 1
#                 effective_path_length = np.where(
#                     mask,
#                     -np.log(1 - lambda_1 * path_lengths) / lambda_1,
#                     np.nan
#                 )
#             return effective_path_length
#         eff_path_lengths = calculate_effective_path_length(path_lengths, lambda_1)
#         eff_free_path_lengths = calculate_effective_path_length(free_path_lengths, lambda_1)

#         # Calculated the mean and var of effective path lengths and free path lengths
#         mean_eff_path_length = np.nanmean(eff_path_lengths)
#         var_eff_path_length = np.nanvar(eff_path_lengths)
#         mean_eff_free_path_length = np.nanmean(eff_free_path_lengths)
#         var_eff_free_path_length = np.nanvar(eff_free_path_lengths)

#         # Calculate extra effective free path lengths values
#         sum_eff_free_path_length = np.nansum(eff_free_path_lengths)
#         sum_eff_free_path_lengths_hit = np.nansum(eff_free_path_lengths[voxel_df['hit_ray'].values])
#         sum_eff_free_path_lengths_hit_leaf = np.nansum(eff_free_path_lengths[voxel_df['is_leaf'].values]) if is_leaf_true else np.nansum(eff_free_path_lengths[~voxel_df['is_leaf'].values])

#         # Calculate LIAD and G for all points
#         mask = ~np.isnan(voxel_df[['point_x', 'point_y', 'point_z']].values).any(axis=1)
#         valid_points = voxel_df[['point_x', 'point_y', 'point_z']][mask]
#         viewing_angles = voxel_df['viewing_angle'].values
#         mean_angle_all = np.nanmean(viewing_angles)
#         viewing_angles = viewing_angles[mask]
#         num_valid_points = valid_points.shape[0]

#         if num_valid_points != num_hits:
#             statement = f"Voxel {voxel_df['voxel_id'].values[0]} has {num_valid_points} valid points but {num_hits} hits."
#             print(statement)
            
#         # Calcaulte LIAD for both leaf and wood
#         bin_centres, LIAD_values, angles = calculate_LIAD_old(valid_points)
#         mean_angle_lw = np.nanmean(viewing_angles)
#         G_lw = calculate_G(viewing_angles=viewing_angles, bin_centres=bin_centres, LIAD_values=LIAD_values)
#         # Calculate G_mean for all rays
#         G_lw = G_lw.mean() if isinstance(G_lw, np.ndarray) else G_lw
        
#         # Calcualte LIAD and G for leaf hits
#         mask = (voxel_df['is_leaf'].values if is_leaf_true else ~voxel_df['is_leaf'].values) 
#         leaf_points = voxel_df[['point_x', 'point_y', 'point_z']][mask]
#         leaf_viewing_angles = voxel_df['viewing_angle'].values[mask]
#         mean_angle_leaf = np.nanmean(leaf_viewing_angles)

#         bin_centres, LIAD_values, angles = calculate_LIAD_old(leaf_points)
#         G_leaf = calculate_G(viewing_angles=leaf_viewing_angles, bin_centres=bin_centres, LIAD_values=LIAD_values)
#         G_leaf = G_leaf.mean() if isinstance(G_leaf, np.ndarray) else G_leaf

#         data = {
#             'voxel_id': voxel_id,
#             'num_rays': num_rays,
#             'num_hits': num_hits,
#             'num_leaf_hits': num_leaf_hits,
#             'pgap_lw': pgap_lw,
#             'pgap_leaf': pgap_leaf,
#             'I_lw': I_lw,
#             'I_leaf': I_leaf,
#             'G_lw': G_lw,
#             'G_leaf': G_leaf,
#             'mean_angle_lw': mean_angle_lw,
#             'mean_angle_leaf': mean_angle_leaf,
#             'mean_angle_all': mean_angle_all,
#             'mean_path_length': mean_path_length,
#             'sum_path_length': sum_path_length,
#             'mean_free_path_length': mean_free_path_length,
#             'sum_free_path_length': sum_free_path_length,
#             'sum_free_path_length_hit': sum_free_path_length_hit,
#             'sum_free_path_length_hit_leaf': sum_free_path_length_hit_leaf,
#             'mean_eff_path_length': mean_eff_path_length,
#             'var_eff_path_length': var_eff_path_length,
#             'mean_eff_free_path_length': mean_eff_free_path_length,
#             'var_eff_free_path_length': var_eff_free_path_length,
#             'sum_eff_free_path_length': sum_eff_free_path_length,
#             'sum_eff_free_path_length_hit': sum_eff_free_path_lengths_hit,
#             'sum_eff_free_path_length_hit_leaf': sum_eff_free_path_lengths_hit_leaf
#         }
#         voxel_metrics = pd.DataFrame(data, index=[0], columns=voxel_metrics_schema.names)

#         return voxel_metrics
        


#     # # Find available memory
#     # available_memory = psutil.virtual_memory().available
#     # available_memory_mb = available_memory / (1024 * 1024)


#     # Read all parquets into dask dataframe
#     dfs = []
#     for file in intersections_files:
#         if os.path.exists(file):
#             df = dd.read_parquet(file, engine='pyarrow') # add later if needed: blocksize=None)

#             dfs.append(df)

#     if len(dfs) == 0:
#         raise ValueError("No valid voxel_ray_intersection files found.")
    
#     # Combine all dataframes into one
#     voxel_intersections_df = dd.concat(dfs, axis=0, ignore_index=True)
#     voxel_intersections_df = voxel_intersections_df.repartition(npartitions=1)
#     voxel_intersections_df = voxel_intersections_df.groupby('voxel_id')
#     unique_voxel_ids = voxel_intersections_df['voxel_id'].unique().compute()
#     num_voxels = len(unique_voxel_ids)

#     # Extract requisite information for density calculations
#     meta = pd.DataFrame(columns=voxel_metrics_schema.names)
#     voxel_metrics_df = voxel_intersections_df.apply(calculate_voxel_metrics_per_voxel, meta=meta)

#     # Return the calculated metrics
#     with ProgressBar():
#         voxel_metrics_df = voxel_metrics_df.compute()
#         voxel_metrics_df = voxel_metrics_df.reset_index(drop=True)
#     return voxel_metrics_df

    pass

def get_voxel_metrics(intersections_files, lambda_1, is_leaf_true=True, debug=True, epsilon=1e-9):
    """
    This function will take the voxel_ray_intersection files and calculate the metrics for each voxel.
    It will save the results to a parquet file.

    Args:
        intersections_files (list): List of paths to voxel_ray_intersection files.
        lambda_1 (float): This is calculated using (average leaf area / voxel size) and will need to be calculated and passed in.
        debug (bool): Whether to print debug information.
        epsilon (float): Small value to avoid division by zero.

    Returns:
        None
    """
    import os
    import glob
    import pandas as pd
    import numpy as np
    import dask.dataframe as dd
    from dask.diagnostics import ProgressBar
    import logging
    import psutil

    

    # Setup logger
    # logger = logging.getLogger()
    # level = logging.DEBUG if debug else logging.WARNING
    # logging.basicConfig(filename=os.path.join(intersections_files, 'voxel_metrics.log'), encoding='utf-8', level=level)

    # Per voxel function
    def calculate_voxel_metrics_per_voxel(voxel_df, min_rays=6, epsilon=1e-9):
        """
        Calculate voxel metrics for a given voxel dataframe.
        
        INPUTS:
            voxel_df: A pandas dataframe containing voxel data
            min_rays: Minimum number of rays to consider a voxel valid
            epsilon: A small value to avoid division by zero

        OUTPUTS:
            voxel_metrics: A pandas dataframe containing the calculated metrics for each voxel
        """
        
        # Check if the dataframe is empty
        if len(voxel_df) == 0:
            return pd.DataFrame(columns=voxel_metrics_schema.names)

        # Get voxel_id name
        voxel_id = voxel_df.name

        # Calculate the number of rays in each voxel
        num_rays = voxel_df['ray_id'].count()
        if num_rays <= 0:
            statement= f"Voxel {voxel_df['voxel_id'].values[0]} has no rays."
            print(statement)
            return pd.DataFrame(columns=voxel_metrics_schema.names)
        
        num_hits = voxel_df['hit_ray'].sum()
        num_leaf_hits = voxel_df[(voxel_df['hit_ray']) & (voxel_df['is_leaf'])].shape[0] if is_leaf_true else voxel_df[(voxel_df['hit_ray']) & ~(voxel_df['is_leaf'])].shape[0]

        # Prepare common masks
        hit_mask = voxel_df['hit_ray'].values
        leaf_mask = hit_mask & voxel_df['is_leaf'].values if is_leaf_true else hit_mask & ~voxel_df['is_leaf'].values

        # Calculate mean viewing angles
        mean_angle_all = np.nanmean(voxel_df['viewing_angle'][hit_mask].values)
        mean_angle_leaf = np.nanmean(voxel_df['viewing_angle'][leaf_mask].values)

        # Calculate pgap_lw and I
        pgap_lw = (num_rays - num_hits) / num_rays
        I_lw = num_hits / num_rays
        pgap_leaf = (num_rays - num_leaf_hits) / num_rays
        I_leaf = num_leaf_hits / num_rays

        # Calcualte path lengths
        path_lengths = np.linalg.norm(voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values, axis=1)
        
        free_path_lengths = np.where(
            hit_mask,
            np.linalg.norm(voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values - voxel_df[['point_x', 'point_y', 'point_z']].values, axis=1),
            path_lengths
        )

        # Calculate the sums and means
        sum_path_length = np.nansum(path_lengths)
        mean_path_length = np.nanmean(path_lengths)
        sum_free_path_length = np.nansum(free_path_lengths)
        mean_free_path_length = np.nanmean(free_path_lengths)
        sum_free_path_length_hit = np.nansum(free_path_lengths[hit_mask])
        sum_free_path_length_hit_leaf = np.nansum(free_path_lengths[leaf_mask])
        # mean_free_path_length_leaf = np.nanmean(free_path_lengths_leaf)

        # Calculate effective path lengths and free path lengths
        def calculate_effective_path_length(path_lengths, lambda_1):
            with np.errstate(divide='ignore', invalid='ignore'):
                mask = (lambda_1 * path_lengths) < 1
                effective_path_length = np.where(
                    mask,
                    -np.log(1 - lambda_1 * path_lengths) / lambda_1,
                    np.nan
                )
            return effective_path_length
        eff_path_lengths = calculate_effective_path_length(path_lengths, lambda_1)
        eff_free_path_lengths = calculate_effective_path_length(free_path_lengths, lambda_1)

        # Calculated the mean and var of effective path lengths and free path lengths
        mean_eff_path_length = np.nanmean(eff_path_lengths)
        var_eff_path_length = np.nanvar(eff_path_lengths)
        mean_eff_free_path_length = np.nanmean(eff_free_path_lengths)
        var_eff_free_path_length = np.nanvar(eff_free_path_lengths)

        # Calculate extra effective free path lengths values
        sum_eff_free_path_length = np.nansum(eff_free_path_lengths)
        sum_eff_free_path_lengths_hit = np.nansum(eff_free_path_lengths[hit_mask])
        sum_eff_free_path_lengths_hit_leaf = np.nansum(eff_free_path_lengths[leaf_mask])

        # Calculate LIAD and G using just leaf points
        valid_points_leaf = voxel_df[['point_x', 'point_y', 'point_z']][leaf_mask].values
        valid_normals_leaf = voxel_df[['normal_x', 'normal_y', 'normal_z']].values[leaf_mask]
        valid_weights_leaf = voxel_df['point_weight'].values[leaf_mask]
        valid_viewing_angles_leaf = voxel_df['viewing_angle'].values[leaf_mask]

        # Check for NAN points
        if np.isnan(valid_points_leaf).any():
            statement = f"Voxel {voxel_df['voxel_id'].values[0]} has NaN points."
            print(statement)

            mask = ~np.isnan(valid_points_leaf).any(axis=1)
            valid_points_leaf = valid_points_leaf[mask]
            valid_normals_leaf = valid_normals_leaf[mask]
            valid_weights_leaf = valid_weights_leaf[mask]
            valid_viewing_angles_leaf = valid_viewing_angles_leaf[mask]

        bin_centres, LIAD_leaf_values, angles = calculate_LIAD(normals=valid_normals_leaf, weights=valid_weights_leaf)
        G_leaf = calculate_G(viewing_angles=valid_viewing_angles_leaf, bin_centres=bin_centres, LIAD_values=LIAD_leaf_values)
        G_leaf = G_leaf.mean() if isinstance(G_leaf, np.ndarray) else G_leaf
        lambda_1 = lambda_1
        if len(LIAD_leaf_values) == 0:
            LIAD_leaf_values = [np.nan] * 18

        data = {
            'voxel_id': voxel_id,
            'num_rays': num_rays,
            'num_hits': num_hits,
            'num_leaf_hits': num_leaf_hits,
            'pgap_lw': pgap_lw,
            'pgap_leaf': pgap_leaf,
            'I_lw': I_lw,
            'I_leaf': I_leaf,
            'G_lw': np.nan,
            'G_leaf': G_leaf,
            'lamda_1': lambda_1,
            'LIAD_leaf_bin_2.5': LIAD_leaf_values[0],
            'LIAD_leaf_bin_7.5': LIAD_leaf_values[1],
            'LIAD_leaf_bin_12.5': LIAD_leaf_values[2],
            'LIAD_leaf_bin_17.5': LIAD_leaf_values[3],
            'LIAD_leaf_bin_22.5': LIAD_leaf_values[4],
            'LIAD_leaf_bin_27.5': LIAD_leaf_values[5],
            'LIAD_leaf_bin_32.5': LIAD_leaf_values[6],
            'LIAD_leaf_bin_37.5': LIAD_leaf_values[7],
            'LIAD_leaf_bin_42.5': LIAD_leaf_values[8],
            'LIAD_leaf_bin_47.5': LIAD_leaf_values[9],
            'LIAD_leaf_bin_52.5': LIAD_leaf_values[10],
            'LIAD_leaf_bin_57.5': LIAD_leaf_values[11],
            'LIAD_leaf_bin_62.5': LIAD_leaf_values[12],
            'LIAD_leaf_bin_67.5': LIAD_leaf_values[13],
            'LIAD_leaf_bin_72.5': LIAD_leaf_values[14],
            'LIAD_leaf_bin_77.5': LIAD_leaf_values[15],
            'LIAD_leaf_bin_82.5': LIAD_leaf_values[16],
            'LIAD_leaf_bin_87.5': LIAD_leaf_values[17],
            # 'LIAD_lw_bin_2.5': LIAD_lw_values[0],
            # 'LIAD_lw_bin_7.5': LIAD_lw_values[1],
            # 'LIAD_lw_bin_12.5': LIAD_lw_values[2],
            # 'LIAD_lw_bin_17.5': LIAD_lw_values[3],
            # 'LIAD_lw_bin_22.5': LIAD_lw_values[4],
            # 'LIAD_lw_bin_27.5': LIAD_lw_values[5],
            # 'LIAD_lw_bin_32.5': LIAD_lw_values[6],
            # 'LIAD_lw_bin_37.5': LIAD_lw_values[7],
            # 'LIAD_lw_bin_42.5': LIAD_lw_values[8],
            # 'LIAD_lw_bin_47.5': LIAD_lw_values[9],
            # 'LIAD_lw_bin_52.5': LIAD_lw_values[10],
            # 'LIAD_lw_bin_57.5': LIAD_lw_values[11],
            # 'LIAD_lw_bin_62.5': LIAD_lw_values[12],
            # 'LIAD_lw_bin_67.5': LIAD_lw_values[13],
            # 'LIAD_lw_bin_72.5': LIAD_lw_values[14],
            # 'LIAD_lw_bin_77.5': LIAD_lw_values[15],
            # 'LIAD_lw_bin_82.5': LIAD_lw_values[16],
            # 'LIAD_lw_bin_87.5': LIAD_lw_values[17],
            # 'mean_angle_lw': mean_angle_lw,
            'mean_angle_leaf': mean_angle_leaf,
            'mean_angle_all': mean_angle_all,
            'mean_path_length': mean_path_length,
            'sum_path_length': sum_path_length,
            'mean_free_path_length': mean_free_path_length,
            'sum_free_path_length': sum_free_path_length,
            'sum_free_path_length_hit': sum_free_path_length_hit,
            'sum_free_path_length_hit_leaf': sum_free_path_length_hit_leaf,
            'mean_eff_path_length': mean_eff_path_length,
            'var_eff_path_length': var_eff_path_length,
            'mean_eff_free_path_length': mean_eff_free_path_length,
            'var_eff_free_path_length': var_eff_free_path_length,
            'sum_eff_free_path_length': sum_eff_free_path_length,
            'sum_eff_free_path_length_hit': sum_eff_free_path_lengths_hit,
            'sum_eff_free_path_length_hit_leaf': sum_eff_free_path_lengths_hit_leaf
        }
        voxel_metrics = pd.DataFrame(data, index=[0], columns=voxel_metrics_schema.names)

        return voxel_metrics
        


    # # Find available memory
    # available_memory = psutil.virtual_memory().available
    # available_memory_mb = available_memory / (1024 * 1024)


    # Read all parquets into dask dataframe
    dfs = []
    for file in intersections_files:
        if os.path.exists(file):
            df = dd.read_parquet(file, engine='pyarrow') # add later if needed: blocksize=None)

            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    # Combine all dataframes into one
    voxel_intersections_df = dd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.repartition(npartitions=1)
    voxel_intersections_df = voxel_intersections_df.groupby('voxel_id')
    unique_voxel_ids = voxel_intersections_df['voxel_id'].unique().compute()
    num_voxels = len(unique_voxel_ids)

    # Extract requisite information for density calculations
    meta = pd.DataFrame(columns=voxel_metrics_schema.names)
    voxel_metrics_df = voxel_intersections_df.apply(calculate_voxel_metrics_per_voxel, meta=meta)

    # Return the calculated metrics
    with ProgressBar():
        voxel_metrics_df = voxel_metrics_df.compute()
        voxel_metrics_df = voxel_metrics_df.reset_index(drop=True)
    return voxel_metrics_df

def calculate_occlusion_metrics(intersections_files, reference_file,  debug=True, epsilon=1e-9):
    """
    This function will take the voxel_ray_intersection files and calculate the occlusion metrics for each voxel and group of points
    It will return a dataframe for voxel information:
        - Number of rays from each direction (i.e. North, South, East, West, Up, Down)
        - Total volume coverage percentave (i.e. using beam divergence, what percentage of voxel volume is explored)
        This can be used to create a voxel map .xyz which demonstrates all space of the chosen plot (i.e. not just explored voxels)
        
    It will also return a dataframe for point information, which is based on point groups:
        - Number of rays from each direction (i.e. North, South, East, West, Up, Down)
        This can be used to create a .laz file which includes extra classification information that demonstrates the points exploration metrics.
        """

    from sklearn.neighbors import NearestNeighbors
    
    dfs = []
    for file in intersections_files:
        if os.path.exists(file):
            df = pd.read_parquet(file, engine='pyarrow')
            dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    # Combine all dataframes into one
    voxel_intersections_df = pd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.reset_index(drop=True)

    # Retrieve reference information for voxel boundaries
    reference_df = pd.read_csv(reference_file, index_col=None, header=0)
    reference_df = reference_df[['voxel_id', 'voxel_cx', 'voxel_cy', 'voxel_cz', 'voxel_size']]
    reference_df = reference_df.drop_duplicates()
    reference_df = reference_df.set_index('voxel_id')
    
    voxel_mins = reference_df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values - (reference_df['voxel_size'].values / 2)
    voxel_maxs = reference_df[['voxel_cx', 'voxel_cy', 'voxel_cz']].values + (reference_df['voxel_size'].values / 2)

    def get_occlusion_per_voxel(voxel_df, voxel_min, voxel_max, epsilon=1e-9):
        # Calculate the planes which constitute each face of the voxel
        voxel_planes = np.array([
            [voxel_min[0], voxel_min[1], voxel_min[2]],
            [voxel_max[0], voxel_min[1], voxel_min[2]],
            [voxel_min[0], voxel_max[1], voxel_min[2]],
            [voxel_max[0], voxel_max[1], voxel_min[2]],
            [voxel_min[0], voxel_min[1], voxel_max[2]],
            [voxel_max[0], voxel_min[1], voxel_max[2]],
            [voxel_min[0], voxel_max[1], voxel_max[2]],
            [voxel_max[0], voxel_max[1], voxel_max[2]]
        ])

        points = voxel_df[['point_x', 'point_y', 'point_z']].values

        # Find which points land on each plane (plus tolerance)
        point_planes = np.array([
            np.abs(points[:, 0] - voxel_planes[0, 0]) < epsilon,
            np.abs(points[:, 1] - voxel_planes[0, 1]) < epsilon,
            np.abs(points[:, 2] - voxel_planes[0, 2]) < epsilon
        ]).T

        # Calculate the number of points on each plane
        num_points_per_plane = np.sum(point_planes, axis=0)

        # Calculate the total volume coverage percentage
        total_volume = np.prod(voxel_max - voxel_min)
        entry_coords = voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values
        exit_coords = voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values
        entry_radii = voxel_df[['t_entry_radii']].values
        exit_radii = voxel_df[['t_exit_radii']].values
        beam_volumes = ((1/3) * np.pi * np.linalg.norm(entry_coords - exit_coords, axis=1)) * (entry_radii ** 2 + entry_radii * exit_radii + exit_radii ** 2)
        total_volume_coverage = np.sum(beam_volumes) / total_volume

        # Calculate per point group (nearest neighbours) statistics
        # Group points using nearest neighbors
        def group_points_with_nearest_neighbors(points, radius=0.1):
            """
            Group points using nearest neighbors within a specified radius.

            Args:
                points (numpy.ndarray): Array of points (N x 3).
                radius (float): Radius for nearest neighbors.

            Returns:
                numpy.ndarray: Array of group labels for each point.
            """
            if len(points) == 0:
                return np.array([])

            # Fit NearestNeighbors model
            nbrs = NearestNeighbors(radius=radius).fit(points)
            adjacency_matrix = nbrs.radius_neighbors_graph(points, mode='connectivity')

            _, group_labels = connected_components(adjacency_matrix, directed=False)

            return group_labels

        # Apply grouping to points
        group_labels = group_points_with_nearest_neighbors(points)

        # Add group labels to the dataframe
        voxel_df['group_label'] = group_labels



def convert_parquet_to_csv(parquet_file, output_file):
    """
    Convert a parquet file to a csv file.
    """
    
    import pandas as pd
    import pyarrow as pa

    # Read the parquet file
    df = pd.read_parquet(parquet_file, engine='pyarrow')

    df.to_csv(output_file, index=False)

def add_normals_weights_from_intersection_files(files, knn=6):
    """
    Add normals and weights to the points in the intersection files.
    """
    import dask.dataframe as dd
    import numpy as np
    from sklearn.neighbors import NearestNeighbors
    import os

    # Read the intersection files
    dfs = []
    for file in files:
        df = dd.read_parquet(file, engine='pyarrow')
        dfs.append(df)

    if len(dfs) == 0:
        raise ValueError("No valid voxel_ray_intersection files found.")
    
    print(f"Adding normals and weights to {len(dfs)} files...")
    
    # Combine all dataframes into one
    voxel_intersections_df = dd.concat(dfs, axis=0, ignore_index=True)
    voxel_intersections_df = voxel_intersections_df.reset_index(drop=True)

    # Filter out leaf hits (that definitely hit something)
    leaf_df = voxel_intersections_df[(voxel_intersections_df['is_leaf'] == True) & (voxel_intersections_df['hit_ray'] == True)]
    leaf_df = leaf_df.compute()
    leaf_points = leaf_df[['point_x', 'point_y', 'point_z']].values

    # Calculate normals and weights on all leaf hits
    normals, weights = compute_normals_weights_from_points(points=leaf_points, knn=knn)
    del leaf_points

    leaf_df["normal_x"] = normals[:, 0]
    leaf_df["normal_y"] = normals[:, 1]
    leaf_df["normal_z"] = normals[:, 2]
    leaf_df["point_weight"] = weights
    del normals, weights

    voxel_intersections_df = voxel_intersections_df.compute()

    voxel_intersections_df = voxel_intersections_df.merge(
        leaf_df[['ray_id', 'point_x', 'point_y', 'point_z', 'normal_x', 'normal_y', 'normal_z', 'point_weight']],
        on=['ray_id', 'point_x', 'point_y', 'point_z'],
        how='left'
    )
    del leaf_df

    return voxel_intersections_df