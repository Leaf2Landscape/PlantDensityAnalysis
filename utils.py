"""
A Python script of commonly shared utilities for other scripts.
Includes schemas for i/o data, functions, and helpers.
"""

import dask.distributed
from fnvhash import fnv1a_32
import pyarrow as pa
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trimesh
import math
import gc
import dask
from distributed import Client

### CONSTANTS ###
beam_divergence = 0.001 # Beam divergence in radians

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
    pa.field('viewing_angle', pa.float32()),
    pa.field('hit_ray', pa.bool_()),
    pa.field('is_leaf', pa.bool_())
])

# Voxel Metrics Schema
"""
This schema is used to store the metrics for each voxel, based on the selected legs and voxel size.
Since this one is only used to store to a csv file (for final output), it is not as important to be efficient.


"""
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
    lambda_1 = average_leaf_area / voxel_size

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

def calculate_LIAD(points, knn=6, radius=0.1, max_nn=10, num_bins=18):
    """
    Calculate the Local Intensity Angular Distribution (LIAD) for a set of points.
    
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
    if total_weights > 0:
        LIAD_values = hist / total_weight
    else:
        LIAD_values = np.zeros(num_bins)

    # Compute the bin centres
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2

    return bin_centres, LIAD_values, angles

def calculate_liad_values(group):
    viewing_angle = group['viewing_angle'].values
    valid_points = group[['hit_ray', 'point_x', 'point_y', 'point_z', 'is_leaf']].values
    valid_points = valid_points[valid_points[:, 0] == True][:, 1:]
    leaf_points = valid_points[valid_points[:, -1] == True][:, :2]

    bins, LIADs, angles = calculate_LIAD(valid_points)
    bins_leaf, LIADs_leaf, angles_leaf = calculate_LIAD(leaf_points)

    return viewing_angle, bins, LIADs, angles, bins_leaf, LIADs_leaf, angles_leaf


# Compute the G function binwise
def compute_G_function_binwise_vectorized(cos_theta, cos_theta_leaf, cot_theta, cot_theta_leaf, LIAD_values, max_cot):
    """
    Compute the G function binwise using a vectorized approach.
    
    INPUTS:
        cos_theta: The cosine of the viewing angles
        cos_theta_leaf: The cosine of the leaf viewing angles
        cot_theta: The cotangent of the viewing angles
        cot_theta_leaf: The cotangent of the leaf viewing angles
        LIAD_values: The LIAD values
        max_cot: The maximum cotangent value
    
    OUTPUTS:
        G_values: The G function values
    """
    # Compute the outer product of cot_theta*cot_theta_leaf and cos_theta*cos_theta_leaf
    cp_cot = np.outer(cot_theta, cot_theta_leaf)
    cp_cos = np.outer(cos_theta, cos_theta_leaf)

    # Calculate A using broadcasting
    psi = np.arccos(np.clip(cp_cos, -1, 1))
    tan_psi = np.tan(psi)
    A = np.arctan(cp_cot * tan_psi)
    G_values = np.sum(A * LIAD_values, axis=1)
    
    return G_values

# Calculate the G function mean
def calculate_G_mean(viewing_angle, bin_centres, LIAD_values):
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
    if len(viewing_angle) == 0 or len(bin_centres) == 0 or len(LIAD_values) == 0:
        return np.nan
    
    # Convert degrees to radians
    theta = np.radians(viewing_angle)
    theta_leaf = np.radians(bin_centres)

    cos_theta = np.cos(theta)
    cos_theta_leaf = np.cos(theta_leaf)

    tan_theta = np.tan(theta)
    epsilon = 1e-9
    tan_theta[tan_theta == 0] = epsilon
    cot_theta = 1 / tan_theta
    max_cot = 1e3
    cot_theta = np.clip(cot_theta, -max_cot, max_cot)

    tan_theta_leaf = np.tan(theta_leaf)
    tan_theta_leaf[tan_theta_leaf == 0] = epsilon
    cot_theta_leaf = 1 / tan_theta_leaf
    cot_theta_leaf = np.clip(cot_theta_leaf, -max_cot, max_cot)

    # Normalise LIAD
    LIAD_values /= np.sum(LIAD_values) if np.sum(LIAD_values) > 0 else LIAD_values

    # Calculate G Values using the vectorized function
    G_values = compute_G_function_binwise_vectorized(
        cos_theta, cos_theta_leaf, cot_theta, cot_theta_leaf, LIAD_values, max_cot
    )

    # # Calculate the mean ### OLD
    G_mean = np.nan_to_num(np.nanmean(G_values), nan=0, posinf=0, neginf=0) if len(G_values) > 0 else np.nan
    
    return G_mean

### LAD/PAD Functions ###
def CI_adjusted(AD, CI):
    """
    This function takes an ADeff and CI and returns the AD.
    Where, AD = ADeff/CI
    """
    AD = AD/CI
    return AD

def nan_to_default_G_CI(G, CI):
    """
    This function takes an array and a default value and returns the array with nans replaced by the default value.
    """
    if isinstance(G, np.ndarray):
        G = np.where(np.isnan(G), 0.5, G)

    if isinstance(CI, np.ndarray):
        CI = np.where(np.isnan(CI), 1.0, CI)
    
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
        G, CI = nan_to_default_G_CI(G, CI)

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
        G, CI = nan_to_default_G_CI(G, CI)

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
        G, CI = nan_to_default_G_CI(G, CI)

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
        G, CI = nan_to_default_G_CI(G, CI)

        valid_mask = (
            (mean_free_path_length > epsilon) & 
            np.logical_and(I > 0, I < 1) &
            np.logical_and(np.logical_and((lambda_1 > 0), (mean_path_length > 0)), (1 - lambda_1 * mean_path_length > 0)) &
            ~np.isnan(G)
        )

        ADeff = np.where(
            valid_mask,
            -1 * (lambda_1 * mean_path_length * I) / ((1 - lambda_1 * mean_path_length) * mean_free_path_length * G),
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
        G, CI = nan_to_default_G_CI(G, CI)

        leaf_fraction = np.where(num_hits > 0, np.divide(num_leaf_hits, num_hits), np.nan)

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

def MLE_soma_2021(num_hits, num_leaf_hits, sum_free_path_length_hit, sum_free_path_length, G=0.5, CI=1.0, epsilon=1e-9):
    """
    Calculate the Maximum Likelihood Estimation (MLE) using the formula from Soma et al. 2021 (eq. 10).
    λ̃ = (1 - I) / (z̅ * G)  (See paper for more details about this simplification)
    
    """

    try:
        G, CI = nan_to_default_G_CI(G, CI)

        leaf_fraction = np.where(num_hits > 0, np.divide(num_leaf_hits, num_hits), np.nan)

        valid_mask = (
            (leaf_fraction > 0) &
            (G > 0) &
            (sum_free_path_length_hit > 0) &
            (sum_free_path_length > 0) &
            (num_hits > 0)
        )

        ADeff = np.where(
            valid_mask,
            (leaf_fraction / (G * sum_free_path_length)) * (num_hits - sum_free_path_length_hit / sum_free_path_length),
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
def find_viewing_angles(directions, reference_vector=np.array([0, 0, 1])):
    dir_norms = np.linalg.norm(directions, axis=1, keepdims=True)
    normalized_directions = directions / dir_norms
    dot_products = np.dot(normalized_directions, reference_vector)
    cos_thetas = np.clip(dot_products, -1, 1)
    viewing_angle = np.arccos(cos_thetas)
    return viewing_angle

def traverse_voxels(voxel_references, ray_partition, epsilon=1e-9):
    if ray_partition.empty:
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)
    
    # Prep ray information
    leg_ids = ray_partition['leg_id'].values
    ray_ids = ray_partition['ray_id'].values
    origins = np.asarray(ray_partition[['origin_x', 'origin_y', 'origin_z']].values)
    directions = np.asarray(ray_partition[['direction_x', 'direction_y', 'direction_z']].values)
    points = np.asarray(ray_partition[['point_x', 'point_y', 'point_z']].values)
    is_leaf = np.asarray(ray_partition['is_leaf'].values)
    viewing_angle = find_viewing_angles(directions=directions)
    
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
    viewing_angle = viewing_angle[np.newaxis, :]
    origins = origins[np.newaxis, :, :]
    directions = directions[np.newaxis, :, :]
    points = points[np.newaxis, :, :]    

    # voxel_ids = np.expand_dims(voxel_ids, axis=1)         # Shape becomes (n, 1)
    # voxel_sizes = np.expand_dims(voxel_sizes, axis=1)     # Shape becomes (n, 1)
    # voxel_mins = np.expand_dims(voxel_mins, axis=1)  # Shape becomes (n, 1, 3)
    # voxel_maxs = np.expand_dims(voxel_maxs, axis=1)  # Shape becomes (n, 1, 3)

    # leg_ids = np.expand_dims(leg_ids, axis=0).repeat(n, axis=0)         # Shape becomes (n, r)
    # ray_ids = np.expand_dims(ray_ids, axis=0).repeat(n, axis=0)         # Shape becomes (n, r)
    # is_leaf = np.expand_dims(is_leaf, axis=0).repeat(n, axis=0)           # Shape becomes (n, r)
    # viewing_angle = np.expand_dims(viewing_angle, axis=0).repeat(n, axis=0) # Shape becomes (n, r)
    # origins = np.expand_dims(origins, axis=0).repeat(n, axis=0)       # Shape becomes (n, r, 3)
    # directions = np.expand_dims(directions, axis=0).repeat(n, axis=0) # Shape becomes (n, r, 3)
    # points = np.expand_dims(points, axis=0).repeat(n, axis=0)         # Shape becomes (n, r, 3)

    t_min = np.divide(
        voxel_mins - origins,
        directions,
        out=np.full((voxel_mins.shape[0], origins.shape[1], origins.shape[2]), float(np.inf)),
        where=directions != 0
    )
    t_max = np.divide(
        voxel_maxs - origins, 
        directions, 
        out=np.full((voxel_mins.shape[0], origins.shape[1], origins.shape[2]), float(np.inf)), 
        where=directions != 0
    )

    t1 = np.minimum(t_min, t_max)
    t2 = np.maximum(t_min, t_max)
    t_enter = np.max(t1, axis=2)  # Take max along the coordinate axis (shape becomes (n, r))
    t_exit = np.min(t2, axis=2)   # Take min along the coordinate axis (shape becomes (n, r))
    mask = (t_enter <= t_exit + epsilon) & (t_exit >= -epsilon)

    if not mask.any():
        return pd.DataFrame(columns=voxel_ray_intersection_schema.names)
    
    del t_min, t_max, t1, t2
    # gc.collect()
    
    # Flatten mask and retrieve idx for rays and voxels
    voxel_ref_idx, ray_ref_idx = np.nonzero(mask)

    # Flatten all values to match the mask
    filtered_voxel_ids = voxel_ids[voxel_ref_idx].reshape(-1)
    filtered_voxel_sizes = voxel_sizes[voxel_ref_idx]
    filtered_voxel_mins = voxel_mins[voxel_ref_idx].reshape(-1, 3)
    filtered_voxel_maxs = voxel_maxs[voxel_ref_idx].reshape(-1, 3)

    filtered_leg_ids = leg_ids[:, ray_ref_idx].reshape(-1)
    filtered_ray_ids = ray_ids[:, ray_ref_idx].reshape(-1)
    filtered_is_leaf = is_leaf[:, ray_ref_idx].reshape(-1)
    filtered_viewing_angles = viewing_angle[:, ray_ref_idx].reshape(-1)
    filtered_points = points[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_origins = origins[:, ray_ref_idx, :].reshape(-1, 3)
    filtered_directions = directions[:, ray_ref_idx, :].reshape(-1, 3)

    filtered_t_enters = t_enter[voxel_ref_idx, ray_ref_idx]
    filtered_t_exits = t_exit[voxel_ref_idx, ray_ref_idx]

    # Cleanup memory
    del voxel_ids, voxel_sizes, voxel_mins, voxel_maxs, leg_ids, ray_ids, is_leaf, viewing_angle, points, origins, directions
    # gc.collect()

    # Filter points to only include ones within the respective voxel    
    filtered_hit_rays = np.all((filtered_points >= filtered_voxel_mins) & (filtered_points <= filtered_voxel_maxs), axis=1)
    filtered_points = np.where(
        filtered_hit_rays[:, None],
        filtered_points,
        np.full(filtered_points.shape, np.nan)
    )

    num_hits = filtered_hit_rays.sum()
    num_points = np.sum(~np.isnan(filtered_points).all(axis=1))


    # Split points for parquet format
    filtered_points_x = filtered_points[:, 0]
    filtered_points_y = filtered_points[:, 1]
    filtered_points_z = filtered_points[:, 2]

    # Cleanup memory
    del filtered_points
    # gc.collect()

    t_entry_coords = filtered_origins + filtered_t_enters[:, None] * filtered_directions
    t_exit_coords = filtered_origins + filtered_t_exits[:, None] * filtered_directions

    distance_to_entry = np.linalg.norm(t_entry_coords - filtered_origins, axis=1).astype(np.float32)
    t_entry_radii = (distance_to_entry * np.tan(beam_divergence)).astype(np.float32)
    distance_to_exit = np.linalg.norm(t_exit_coords - filtered_origins, axis=1).astype(np.float32)
    t_exit_radii = (distance_to_exit * np.tan(beam_divergence)).astype(np.float32)

    del filtered_origins, filtered_directions, filtered_t_enters, filtered_t_exits, distance_to_entry, distance_to_exit
    # gc.collect()

    data = [
        pa.array(filtered_voxel_sizes),
        pa.array(filtered_voxel_ids),
        pa.array(filtered_leg_ids),
        pa.array(filtered_ray_ids),
        pa.array(t_entry_coords[:, 0]),
        pa.array(t_entry_coords[:, 1]),
        pa.array(t_entry_coords[:, 2]),
        pa.array(t_exit_coords[:, 0]),
        pa.array(t_exit_coords[:, 1]),
        pa.array(t_exit_coords[:, 2]),
        pa.array(t_entry_radii),
        pa.array(t_exit_radii),
        pa.array(filtered_points_x),
        pa.array(filtered_points_y),
        pa.array(filtered_points_z),
        pa.array(filtered_viewing_angles),
        pa.array(filtered_hit_rays),
        pa.array(filtered_is_leaf)
    ]
    result = pa.Table.from_arrays(data, schema=voxel_ray_intersection_schema)
    result = result.to_pandas()

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
def voxel_ray_intersections(valid_rays_dir, references_dir, debug=True, epsilon=1e-6):
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
    
    num_voxels = len(voxel_references)
    n_workers = os.cpu_count() or 1
    available_memory = psutil.virtual_memory().available
    available_memory_per_worker = available_memory / n_workers
    mem_safety_factor = 0.7
    target_partition_mem = (10 * 1024 * 1024) # bytes


    available_memory_mb = available_memory / (1024 * 1024)
    chunk_size_mb = calculate_chunk_size_mb(num_voxels, available_memory_mb)
    chunk_size = int(chunk_size_mb * 1024 * 1024)

    # Compile all valid_rays parquets
    valid_rays_files = glob.glob(os.path.join(valid_rays_dir, '*_valid_rays.parquet'))

    # @dask.delayed
    def map_ray_partition_to_function(ray_partition, voxel_group):

        result = traverse_voxels(ray_partition=ray_partition, voxel_references=voxel_group)
        return result

    voxel_ray_intersections = {}

    for file in valid_rays_files:
        # Read in parquet file
        df = dd.read_parquet(file, engine='pyarrow', blocksize=(10 * 1024 * 1024))

        # df_mem = df.memory_usage(deep=True).compute().sum()


        # # Check for memory fit
        # num_rays = df.index.size.compute()
        # num_elements = len(df.columns)

        # # per partition
        # # df_mem_partition_max = df.memory_usage_per_partition(deep=True).compute().max()
        # # max_voxel_chunks = int(np.floor(np.sqrt(available_memory_per_worker * 0.8 / df_mem_partition_max)))

        # # per total
        # df_mem = df.memory_usage_per_partition(deep=True).compute().max()
        # max_voxel_chunks = int(np.floor(np.sqrt(available_memory_per_worker * 0.8 / df_mem)))
        # # print(f"Max voxel chunks: {max_voxel_chunks}")

        # Get leg_id from filename
        leg_id = int(os.path.splitext(os.path.basename(file))[0].split("_")[1])

        # Map partitions to traverse voxels
        meta = pd.DataFrame(columns=voxel_ray_intersection_schema.names)

        # client = Client()

        available_mem = psutil.virtual_memory().available
        partition_mem = df.memory_usage_per_partition(deep=True).compute().max()
        voxels_per_partition = 3
        for voxel_size, voxel_group in voxel_references.groupby('voxel_size'):
        
            # Chunk voxels to fit within max_voxel_chunk
            num_voxels = len(voxel_group)
            max_voxel_chunks_per_compute = int(np.floor(available_mem * 0.8 / (partition_mem * df.npartitions)))

            voxel_chunks = [voxel_group.iloc[i:i + voxels_per_partition] for i in range(0, len(voxel_group), voxels_per_partition)]

            results = []
            n = 0
            dd_results = []
            for voxel_chunk in voxel_chunks:
                
                result = df.map_partitions(
                    map_ray_partition_to_function,
                    voxel_group=voxel_chunk,
                    meta=meta
                )

                if n < max_voxel_chunks_per_compute:
                    dd_results.append(result)
                    n += 1
                else:
                    if len(dd_results) > 1:
                        comb_results = dd.concat(dd_results, axis=0)
                    elif dd_results == 1:
                        comb_results = dd_results[0] 
                
                    results.append(comb_results)
                    dd_results = []
                    n = 0
            
            if len(dd_results) > 1:
                comb_results = dd.concat(dd_results, axis=0)
                results.append(comb_results)
            elif len(dd_results) == 1:
                results.append(dd_results[0])

            if leg_id not in voxel_ray_intersections:
                voxel_ray_intersections[leg_id] = {}                
                
            if voxel_size not in voxel_ray_intersections[leg_id]:
                voxel_ray_intersections[leg_id][voxel_size] = []

            voxel_ray_intersections[leg_id][voxel_size] = results


    def save_task(df, leg_id, voxel_size):
        if df.empty:
            return False

        # Create the output filename
        output_filename = os.path.join(valid_rays_dir, f"leg_{leg_id}_voxel_{voxel_size}_intersections.parquet")

        # Save the dataframe to parquet
        df.to_parquet(output_filename, engine='pyarrow', compression='snappy', schema=voxel_ray_intersection_schema)

        return True


    with ProgressBar():
        
        for leg_id, voxel_sizes in voxel_ray_intersections.items():

            for voxel_size, results in voxel_sizes.items():

                # print(f"Processing leg {leg_id} with voxel size {voxel_size}...")
                # results = list(dask.compute(*results, scheduler='processes', num_workers=n_workers))

                # print(f"Saving {len(results)} results for leg {leg_id} with voxel size {voxel_size}...")
                # results = pd.concat(results, ignore_index=True)

                df_results = []
                for i, result in enumerate(results):
                    # for r in result:

                    print(f"Processing chunk {i+1} of {len(results)} for leg {leg_id} and voxel size {voxel_size}")
                    result = result.compute()
                    df_results.append(result)

                results = pd.concat(df_results, ignore_index=True)

                save_task(results, leg_id, voxel_size)

                gc.collect()


def get_voxel_metrics(intersections_files, lambda_1, debug=True, epsilon=1e-9):
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
        num_leaf_hits = voxel_df[(voxel_df['hit_ray']) & (voxel_df['is_leaf'])].shape[0]

        # Calculate pgap_lw and I
        pgap_lw = (num_rays - num_hits) / num_rays
        I_lw = num_hits / num_rays
        pgap_leaf = (num_rays - num_leaf_hits) / num_rays
        I_leaf = num_leaf_hits / num_rays

        # Calcualte path lengths
        path_lengths = np.linalg.norm(voxel_df[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel_df[['t_entry_x', 't_entry_y', 't_entry_z']].values, axis=1)
        hit_mask = voxel_df['hit_ray'].values
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
        sum_free_path_length_hit = np.nansum(free_path_lengths[voxel_df['hit_ray'].values])
        sum_free_path_length_hit_leaf = np.nansum(free_path_lengths[voxel_df['is_leaf'].values])
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
        sum_eff_free_path_lengths_hit = np.nansum(eff_free_path_lengths[voxel_df['hit_ray'].values])
        sum_eff_free_path_lengths_hit_leaf = np.nansum(eff_free_path_lengths[voxel_df['is_leaf'].values])

        viewing_angle = voxel_df['viewing_angle'].values

        # Calculate LIAD and G for all points
        mask = ~np.isnan(voxel_df[['point_x', 'point_y', 'point_z']].values).any(axis=1)
        valid_points = voxel_df[['point_x', 'point_y', 'point_z']][mask]
        num_valid_points = valid_points.shape[0]

        if num_valid_points != num_hits:
            statement = f"Voxel {voxel_df['voxel_id'].values[0]} has {num_valid_points} valid points but {num_hits} hits."
            print(statement)
            
        bin_centres, LIAD_values, angles = calculate_LIAD(valid_points)
        G_lw = calculate_G_mean(viewing_angle=angles, bin_centres=bin_centres, LIAD_values=LIAD_values)
        
        # Calcualte LIAD and G for leaf hits
        mask = voxel_df['is_leaf'].values
        valid_points = voxel_df[['point_x', 'point_y', 'point_z']][mask]

        bin_centres, LIAD_values, angles = calculate_LIAD(valid_points)
        G_leaf = calculate_G_mean(viewing_angle=angles, bin_centres=bin_centres, LIAD_values=LIAD_values)

        data = {
            'voxel_id': voxel_id,
            'num_rays': num_rays,
            'num_hits': num_hits,
            'num_leaf_hits': num_leaf_hits,
            'pgap_lw': pgap_lw,
            'pgap_leaf': pgap_leaf,
            'I_lw': I_lw,
            'I_leaf': I_leaf,
            'G_lw': G_lw,
            'G_leaf': G_leaf,
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






            


    

    

    
    
    





def calculate_voxel_metrics(
        voxel_df,
        min_rays=6,
        G=None,
        CI=None,
        woody_vol_proportion=None,
        PAD_BL=True,
        PAD_BL_EPL=True,
        PAD_BL_UEPL=True,
        PAD_MCF=True,
        PAD_MCF_Corr=True,
        PAD_MLE_pimont_2018=True,
        LAD_BL=True,
        LAD_BL_EPL=True,
        LAD_BL_UEPL=True,
        LAD_MCF=True,
        LAD_MCF_Corr=True,
        LAD_MLE_pimont_2019=True,
        LAD_MLE_soma_2021=True,
):
    """
    This function will calculate the metrics (both LAD and PAD) for all voxels in voxel_df.
    It is designed to handle pandas dataframe inputs containing multiple voxel_sizes and voxel_ids.

    If you pass one in (i.e. pre-filtered), it will still operate as expected.
    But you can also pass in a voxel_ray_intersection_df that fits into memory with multiple voxels information.
    
    INPUTS:
        voxel_df: A voxel_ray_intersection_schema dataframe containing the voxel information.
            Each index (row) corresponds to a unique ray that intersects a voxel.
            Columns: 
                voxel_size:         Size of voxel
                voxel_id:           ID of voxel
                leg_id:             ID of leg
                ray_id:             ID of ray
                t_entry_x:          Entry x of ray in voxel
                t_entry_y:          Entry y of ray in voxel
                t_entry_z:          Entry z of ray in voxel
                t_exit_x:           Exit x of ray in voxel
                t_exit_y:           Exit y of ray in voxel
                t_exit_z:           Exit z of ray in voxel
                t_entry_radius:     Entry radius of ray in voxel, used for beam divergence
                t_exit_radius:      Exit radius of ray in voxel, used for beam divergence
                point_x:       X points of ray
                point_y:       Y points of ray
                point_z:       Z points of ray
                viewing_angle:     Viewing angles of ray
                hit_ray:            Whether ray hit the voxel
                is_leaf:            Whether the hit is a leaf hit

        min_rays: Minimum number of rays to calculate metrics

    OUTPUTS:
        voxel_metrics_df: A dataframe containing the metrics for each voxel.
            Columns:
                voxel_size:         Size of voxel
                voxel_id:           ID of voxel
                leg_id:             ID of leg
                hits_leaf:          Number of hits classified as leaf
                hits_wood:          Number of hits classified as wood
                num_hits:           Number of hits total
                G_leaf:             G Function calculated from leaf hits
                G:                  G Function calculated from leaf and wood hits
                mean_path_length:          Mean of full path length
                mean_free_path_lengths:             Mean z
                mean_eff_path_length:       Mean of effective path length
                var_delta_e:        Variance of effective path length
                lambda_1:           Lambda 1
                PAD_BL:             PAD Basal Area
                PAD_BL_EPL:         PAD Basal Area Effective Path Length
                PAD_BL_UEPL:        PAD Basal Area Unbiased Effective Path Length
                PAD_MCF:            PAD Mean Crown Fraction
                PAD_MCF_Corr:  PAD Mean Crown Fraction Corrected
                PAD_MLE_pimont_2018: PAD Maximum Likelihood Estimation Pimont 2018
                LAD_BL:             LAD Basal Area
                LAD_BL_EPL:         LAD Basal Area Effective Path Length
                LAD_BL_UEPL:        LAD Basal Area Unbiased Effective Path Length
                LAD_MCF:            LAD Mean Crown Fraction
                LAD_MCF_Corr:  LAD Mean Crown Fraction Corrected
                LAD_MLE_pimont_2019: LAD Maximum Likelihood Estimation Pimont 2019
                LAD_MLE_Soma_21:    LAD Maximum Likelihood Estimation Soma 21            

    This function is ideal for working on a single partition of voxel_ray_intersection data.
        i.e. voxel_metrics_dfs = voxel_ray_intersection_df.apply(calculate_voxel_metrics)
    """
    # Group into unique voxel_size and voxel_id
    voxel_df = voxel_df.groupby(['voxel_size', 'voxel_id'])
    
    # The following values will now be arrays of values with dimensions for voxel_size and voxel_id
    # This will allow for vectorized calculations of metrics

    # Calculate overarching values
    # voxel_id = voxel_df['voxel_id'].iloc[0]
    # voxel_size = voxel_df['voxel_size'].iloc[0]

    lambda_1s = voxel_df.apply(calculate_lambda_1)

    num_hits = voxel_df['hit_ray'].sum()
    num_leaf_hits = voxel_df[voxel_df['hit_ray']]['is_leaf'].sum()
    num_rays = voxel_df['ray_id'].nunique()

    # Calculate pgap_lw and I (relative density index)
    pgap_lw = np.where(num_rays > 0, (num_rays - num_hits) / num_rays, np.nan)
    I = np.where(num_rays > 0, num_hits / num_rays, np.nan)
    I_leaf = np.where(num_rays > 0, num_leaf_hits / num_rays, np.nan)

    # Calculate path lengths
    path_lengths = voxel_df.apply(
        lambda voxel: np.linalg.norm(
            voxel[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel[['t_entry_x', 't_entry_y', 't_entry_z']].values,
            axis=1
        )
    )

    free_path_lengths = voxel_df.apply(
        lambda voxel: np.where(
            voxel['hit_ray'],
            np.linalg.norm(
                voxel[['point_x', 'point_y', 'point_z']].values - voxel[['t_entry_x', 't_entry_y', 't_entry_z']].values,
                axis=1
            ),
            path_lengths[voxel.name]
        )
    )

    # Calculate sums and means
    sum_path_lengths = path_lengths.apply(np.sum)           # sum_path_lengths = np.sum(path_lengths) if path_lengths.size > 0 else 0
    mean_path_lengths = path_lengths.apply(np.nanmean)      # mean_path_length = np.mean(path_lengths) if path_lengths.size > 0 else np.nan
    sum_free_path_lengths = free_path_lengths.apply(np.sum)         # sum_free_path_lengths = np.sum(free_path_lengths) if free_path_lengths.size > 0 else 0
    mean_free_path_lengths = free_path_lengths.apply(np.nanmean)    # mean_free_path_lengths = np.mean(free_path_lengths) if free_path_lengths.size > 0 else np.nan

    # Calculate effective path lengths
    eff_path_lengths = voxel_df.apply(
        lambda voxel: np.where(
            (lambda_1s[voxel.name] * np.linalg.norm(
                voxel[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel[['t_entry_x', 't_entry_y', 't_entry_z']].values,
                axis=1
            )) < 1,
            -np.log(1 - lambda_1s[voxel.name] * np.linalg.norm(
                voxel[['t_exit_x', 't_exit_y', 't_exit_z']].values - voxel[['t_entry_x', 't_entry_y', 't_entry_z']].values,
                axis=1
            )) / lambda_1s[voxel.name],
            path_lengths[voxel.name]
        )
    )

    # Calculate mean, variance, and based on z effective path lengths
    mean_eff_path_lengths = eff_path_lengths.apply(np.nanmean)
    non_nan_counts = eff_path_lengths.apply(lambda voxel: np.count_nonzero(~np.isnan(voxel)))
    var_eff_path_lengths = eff_path_lengths.apply(lambda voxel: np.nanvar(voxel, ddof=1) if np.count_nonzero(~np.isnan(voxel)) > 1 else np.nan)
    eff_path_length_zs = free_path_lengths.apply(lambda voxel: effective_path_length_z(voxel, lambda_1s[voxel.name]))

    # Calculate LIAD values
    viewing_angle, bins, LIADs, angles, bins_leaf, LIADs_leaf, angles_leaf = voxel_df.apply(calculate_liad_values)

    # Calculate the G functions, unless otherwise specified
    if G is None:
        G = calculate_G_mean(viewing_angle, bins, LIADs)
        G_leaf = calculate_G_mean(viewing_angle, bins_leaf, LIADs_leaf)

        # Filter invalid G_mean values for voxels
        valid_mask = np.isfinite(G) & (G > 0) & (num_rays >= min_rays)
        G = np.where(valid_mask, G, np.nan)
        G_leaf = np.where(valid_mask, G_leaf, np.nan)
    else:
        G = np.float(G)
        G_leaf = G

    invalid_ze_mask = eff_path_length_zs < 0
    if np.any(invalid_ze_mask):
        print(f"Warning: {np.count_nonzero(invalid_ze_mask)} invalid eff_path_length_zs values found, setting them to zero")
        eff_path_length_zs[invalid_ze_mask] = 0

    finite_mask = np.isfinite(eff_path_length_zs)
    hit_mask = voxel_df['hit_ray'].values
    leaf_hit_mask = voxel_df[hit_mask]['is_leaf'].values

    sum_free_path_lengths_e = np.sum(eff_path_length_zs[finite_mask]) if eff_path_length_zs.size > 0 else 0
    sum_hits_effective_path_lengths = np.sum(eff_path_length_zs[hit_mask]) if num_hits > 0 else 0 # old code used eff_path_length_zs[:num_hits] which did not ensure alignment of EPL
    sum_hits_effective_path_lengths_leaf = np.sum(eff_path_length_zs[leaf_hit_mask]) if num_leaf_hits > 0 else 0

    # Establish voxel_metrics dataframe
    voxel_metrics_df = create_df_from_schema(voxel_metrics_schema)

    ### CALCULATE ALL METRICS ###
    # These will all be effective density (i.e. not CI adjusted)

    # Calculate PAD Metrics
    if PAD_BL:
        PAD_BL = BL_pimont_2018(I=I, mean_path_length=mean_path_lengths, G=G)

    if PAD_BL_EPL or PAD_BL_UEPL:
        PAD_BL_EPL, PAD_BL_UEPL = BL_EPL_UEPL_pimont_2018(I=I, mean_eff_path_length=mean_eff_path_lengths, var_eff_path_length=var_eff_path_lengths, num_rays=num_rays, G=G)

    if PAD_MCF:
        PAD_MCF = MCF_pimont_2018(mean_free_path_lengths=mean_free_path_lengths, I=I, G=G)

    if PAD_MCF_Corr:
        PAD_MCF_Corr = MCF_corrected_pimont_2018(mean_free_path_lengths=mean_free_path_lengths, I=I, lambda_1=lambda_1s, mean_path_length=mean_path_lengths, G=G)

    if PAD_MLE_pimont_2018 and woody_vol_proportion is not None:
        PAD_MLE_pimont_2018 = MLE_pimont_2018(woody_vol_proportion=woody_vol_proportion, num_hits=num_hits, num_leaf_hits=num_leaf_hits, sum_hits_effective_path_length=sum_hits_effective_path_lengths, sum_effective_path_length=sum_free_path_lengths_e, G=G)

    if PAD_MLE_soma_2021 and woody_vol_proportion is not None:
        PAD_MLE_soma_2021 = MLE_soma_2021(num_hits=num_hits, num_leaf_hits=num_leaf_hits, sum_hits_effective_path_length=sum_hits_effective_path_lengths_leaf, sum_effective_path_length=sum_free_path_lengths_e, G=G)

    # Calculate LAD metrics
    if LAD_BL:
        LAD_BL = BL_pimont_2018(I=I_leaf, mean_path_length=mean_path_lengths, G=G_leaf)
    
    if LAD_BL_EPL or LAD_BL_UEPL:
        LAD_BL_EPL, LAD_BL_UEPL = BL_EPL_UEPL_pimont_2018(I=I_leaf, mean_eff_path_length=mean_eff_path_lengths, var_eff_path_length=var_eff_path_lengths, num_rays=num_rays, G=G_leaf)

    if LAD_MCF:
        LAD_MCF = MCF_pimont_2018(mean_free_path_lengths=mean_free_path_lengths, I=I_leaf, G=G_leaf)

    if LAD_MCF_Corr:
        LAD_MCF_Corr = MCF_corrected_pimont_2018(mean_free_path_lengths=mean_free_path_lengths, I=I_leaf, lambda_1=lambda_1s, mean_path_length=mean_path_lengths, G=G_leaf)

    if LAD_MLE_pimont_2018 and woody_vol_proportion is not None:
        LAD_MLE_pimont_2018 = MLE_pimont_2018(woody_vol_proportion=woody_vol_proportion, num_hits=num_hits, num_leaf_hits=num_leaf_hits, sum_hits_effective_path_length=sum_hits_effective_path_lengths_leaf, sum_effective_path_length=sum_free_path_lengths_e, G=G_leaf)

    if LAD_MLE_soma_2021 and woody_vol_proportion is not None:
        LAD_MLE_soma_2021 = MLE_soma_2021(num_hits=num_hits, num_leaf_hits=num_leaf_hits, sum_hits_effective_path_length=sum_hits_effective_path_lengths_leaf, sum_effective_path_length=sum_free_path_lengths_e, G=G_leaf)


    # Add metrics to the dataframe
    voxel_metrics_df['num_rays'] = num_rays
    voxel_metrics_df['num_hits'] = num_hits
    voxel_metrics_df['num_leaf_hits'] = num_leaf_hits
    voxel_metrics_df['I'] = I
    voxel_metrics_df['I_leaf'] = I_leaf
    voxel_metrics_df['pgap_lw'] = pgap_lw
    voxel_metrics_df['mean_path_length'] = mean_path_lengths
    voxel_metrics_df['sum_path_length'] = sum_path_lengths
    voxel_metrics_df['mean_free_path_length'] = mean_free_path_lengths
    voxel_metrics_df['sum_free_path_length'] = sum_free_path_lengths
    voxel_metrics_df['mean_eff_path_length'] = mean_eff_path_lengths
    voxel_metrics_df['sum_free_path_lengths_e'] = sum_free_path_lengths_e
    voxel_metrics_df['var_delta_e'] = var_delta_e
    voxel_metrics_df['sum_hits_z_e'] = sum_hits_z_e
    voxel_metrics_df['sum_hits_z_e_leaf'] = sum_hits_z_e_leaf
    voxel_metrics_df['G_mean'] = G_mean
    voxel_metrics_df['G_leaf'] = G_leaf
    voxel_metrics_df['mean_leaf_angle'] = mean_leaf_angle
    voxel_metrics_df['LAD_BL'] = LAD['LAD_BL']



        
    







def compute_LAD_metrics(
        hits_leaf,              # Number of hits classified as leaf
        num_hits,               # Number of hits total
        G_leaf,                 # G Function calculated from leaf hits
        mean_path_length,              # Mean of full path length
        mean_free_path_lengths,                 # mean z
        mean_eff_path_length,           # Mean of effective path length
        var_delta_e,            # Variance of effective path length
        lambda_1
):
    """
    Compute the LAD metrics.
    
    This script will return a dictionary of LAD metrics, per voxel
    """
    LAD_dict = {
        ''
    }


    pass

def compute_PAD_metrics():
    """
    Compute the PAD metrics.
    
    This script expects the following inputs:
    - A list of actual values

    It will return a dictionary of PAD metrics, per voxel
    """

    pass

def compute_wood_volume_from_file(file_path):
    """
    Compute the wood volume from a file.
    
    This script expects the following inputs:
    - A file path

    It will return the wood volume
    """

    pass


####################################################################################################
#                                       OLD CODE                                                   #
####################################################################################################

def compute_lad_metrics(
    hits_leaf, N, G_leaf, mean_path_length, mean_free_path_lengths, mean_eff_path_length, var_delta_e, lambda_1
):
    """
    Compute LAD metrics from the leaf-only simulation.
    Uses the full path length mean_path_length (mean of ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ) and the effective free path length (mean_eff_path_length from Z).
    """
    results = {
        'LAD_BL': np.nan,
        'LAD_BL_EPL': np.nan,
        'LAD_BL_UEPL': np.nan,
        'LAD_MCF': np.nan,
        'LAD_MCF_Corr': np.nan
    }
    eps = 1e-9
    I_leaf = hits_leaf / float(N) if N > 0 else 0.0
    if (N > 0) and (0 < I_leaf < 1) and (mean_path_length > eps) and (G_leaf > eps):
        results['LAD_BL'] = -math.log(1.0 - I_leaf) / (G_leaf * mean_path_length)  ## Pimont et al. 2018, eq. 5
    # Compute bot unconditionally when N > 0 so it can be used later
    if N > 0:
        bot = math.log(2.0 * N + 2.0)
    else:
        bot = np.nan
        
    if (N > 0) and (0 < I_leaf < 1) and (mean_eff_path_length > eps):
        top = math.log(1 - I_leaf) + (I_leaf / (2.0 * N * (1 - I_leaf)))
        bot = math.log(2.0 * N + 2.0)
        if abs(bot) > eps:
            val_epl = -(1.0 / mean_eff_path_length) * top
            results['LAD_BL_EPL'] = val_epl 
    elif I_leaf == 1.0:
        results['LAD_BL_EPL'] = bot / mean_eff_path_length 
    lad_bl_epl_val = results['LAD_BL_EPL']  ## Pimont et al. 2018, eq. 25
    if (not np.isnan(lad_bl_epl_val) and lad_bl_epl_val > 0 and
        var_delta_e > eps and mean_eff_path_length > eps):
        a_e = var_delta_e / mean_eff_path_length
        inside = 1.0 - 2.0 * a_e * lad_bl_epl_val
        if inside >= 0.0:
            val_uepl = (1.0 / a_e) * (1.0 - math.sqrt(inside))
            results['LAD_BL_UEPL'] = val_uepl / G_leaf  ## Pimont et al. 2018, eq. 27
    if (not np.isnan(lad_bl_epl_val) and lad_bl_epl_val > 0 and (G_leaf > eps)):
        results['LAD_BL_EPL'] = lad_bl_epl_val / G_leaf
        
    if (mean_free_path_lengths > eps) and (G_leaf > eps):
        results['LAD_MCF'] = I_leaf / (G_leaf * mean_free_path_lengths)  ## Pimont et al. 2018, eq. 8
    if (lambda_1 > 0 and mean_path_length > 0 and (0 < I_leaf < 1) and 
        (mean_free_path_lengths > eps) and (1 - lambda_1 * mean_path_length) > 0):
        denom = math.log(1.0 - lambda_1 * mean_path_length) * mean_free_path_lengths
        if abs(denom) > eps:
            val_corr = -1.0 * (lambda_1 * mean_path_length * I_leaf) / denom
            results['LAD_MCF_Corr'] = val_corr / G_leaf  ## Pimont et al. 2018, eq. 12
    return results

def compute_pad_metrics(
    hits_lw, hits_leaf, N, G_lw, mean_path_length, sum_free_path_lengths_lw, sum_free_path_lengths_e_lw, sum_hits_z_e_leaf,
    sum_hits_z_e_lw, woody_vol_proportion, mean_eff_path_length, var_delta_e, lambda_1, leaf_fraction
):
    """
    Compute PAD metrics from the combined (leaf+wood) simulation.
    """
    results = {
        'PAD_BL': np.nan,
        'PAD_BL_EPL': np.nan,
        'PAD_BL_UEPL': np.nan,
        'PAD_MCF': np.nan,
        'PAD_MCF_Corr': np.nan,
        'PAD_MLE_pimont_2018': np.nan,
        'LAD_MLE_pimont_2019': np.nan,
        'LAD_MLE_Soma_21': np.nan
    }
    eps = 1e-9
    I_lw = hits_lw / float(N) if N > 0 else 0.0
    if (N > 0) and (0 < I_lw < 1) and (mean_path_length > eps) and (G_lw > eps):
        results['PAD_BL'] = -math.log(1.0 - I_lw) / (G_lw * mean_path_length)
    # Compute bot unconditionally when N > 0 so it can be used later
    if N > 0:
        bot = math.log(2.0 * N + 2.0)
    else:
        bot = np.nan
    if (N > 0) and (0 < I_lw < 1) and (mean_eff_path_length > eps):
        top = math.log(1 - I_lw) + (I_lw / (2.0 * N * (1 - I_lw)))
        bot = math.log(2.0 * N + 2.0)
        if abs(bot) > eps:
            val_epl = -(1.0 / mean_eff_path_length) * top 
            results['PAD_BL_EPL'] = val_epl
    elif I_lw == 1.0:
        results['PAD_BL_EPL'] = bot / mean_eff_path_length
    pad_bl_epl_val = results['PAD_BL_EPL']
    if (not np.isnan(pad_bl_epl_val) and pad_bl_epl_val > 0 and
        var_delta_e > eps and mean_eff_path_length > eps):
        a_e = var_delta_e / mean_eff_path_length
        inside = 1.0 - 2.0 * a_e * pad_bl_epl_val
        if inside >= 0.0:
            val_uepl = (1.0 / a_e) * (1.0 - math.sqrt(inside))
            results['PAD_BL_UEPL'] = val_uepl / G_lw
    if (not np.isnan(pad_bl_epl_val) and pad_bl_epl_val > 0 and (G_lw > eps)):
        results['PAD_BL_EPL'] = pad_bl_epl_val / G_lw
        
    mean_z_lw = sum_free_path_lengths_lw / N if N > 0 else 0.0  
    if (mean_z_lw > eps) and (G_lw > eps):
        results['PAD_MCF'] = I_lw / (G_lw * mean_z_lw)
    if (lambda_1 > 0 and mean_path_length > 0 and (0 < I_lw < 1) and 
        (mean_z_lw > eps) and (1 - lambda_1 * mean_path_length) > 0):
        denom = math.log(1.0 - lambda_1 * mean_path_length) * mean_z_lw
        if abs(denom) > eps:
            val_corr = -1.0 * (lambda_1 * mean_path_length * I_lw) / denom
            results['PAD_MCF_Corr'] = val_corr / G_lw
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_free_path_lengths_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_lw / sum_free_path_lengths_e_lw)
        results['PAD_MLE_pimont_2018'] = (woody_vol_proportion * leaf_fraction / (G_lw * sum_free_path_lengths_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_free_path_lengths_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_free_path_lengths_e_lw)
        results['LAD_MLE_pimont_2019'] = (woody_vol_proportion * leaf_fraction / (G_lw * sum_free_path_lengths_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_free_path_lengths_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_free_path_lengths_lw)
        results['LAD_MLE_Soma_21'] = (leaf_fraction / (G_lw * sum_free_path_lengths_lw)) * bracket
    return results


def compute_wood_volume_in_voxel(wood_interior_points, voxel_center, voxel_size, small_voxel_size=0.02):
    """
    Approximate wood volume within the voxel by counting how many 'interior' points
    fall into the voxel, times the volume of each small voxel.
    """
    half = voxel_size * 0.5
    min_x = voxel_center[0] - half
    max_x = voxel_center[0] + half
    min_y = voxel_center[1] - half
    max_y = voxel_center[1] + half
    min_z = voxel_center[2] - half
    max_z = voxel_center[2] + half
    inside_mask = (
        (wood_interior_points[:, 0] >= min_x) &
        (wood_interior_points[:, 0] < max_x) &
        (wood_interior_points[:, 1] >= min_y) &
        (wood_interior_points[:, 1] < max_y) &
        (wood_interior_points[:, 2] >= min_z) &
        (wood_interior_points[:, 2] < max_z)
    )
    count_in = np.count_nonzero(inside_mask)
    small_vol = small_voxel_size ** 3
    return count_in * small_vol