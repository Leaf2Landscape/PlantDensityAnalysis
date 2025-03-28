"""
A Python script of commonly shared utilities for other scripts.
Includes schemas for i/o data, functions, and helpers.
"""

from fnvhash import fnv1a_32
import pyarrow as pa
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
import trimesh
import math

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
    pa.field('ray_points_x', pa.float32()),
    pa.field('ray_points_y', pa.float32()),
    pa.field('ray_points_z', pa.float32()),
    pa.field('viewing_angles', pa.float32()),
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
    pa.field('num_points', pa.uint32()),
    pa.field('I', pa.int64()),      # num_hits / num_rays (i.e. leaf and wood)
    pa.field('I_leaf', pa.int64()),  # num_leaf_hits / num_rays (i.e. leaf only)
    pa.field('pgap', pa.float64()),
    pa.field('mean_path_length', pa.float64()),
    pa.field('sum_path_length', pa.float64()),
    pa.field('mean_free_path_length', pa.float64()),
    pa.field('sum_free_path_length', pa.float64()),
    pa.field('mean_eff_path_length', pa.float64()),
    pa.field('sum_free_path_lengths_e', pa.float64()),
    pa.field('var_delta_e', pa.float64()),
    pa.field('sum_hits_z_e', pa.float64()),         # Sum of z_e for all hits
    pa.field('sum_hits_z_e_leaf', pa.float64()),    # Sum of z_e for leaf hits only
    pa.field('G_mean', pa.float64()),
    pa.field('G_leaf', pa.float64()),
    pa.field('mean_leaf_angle', pa.float64()),
    pa.field('LAD_BL', pa.float64()),
    pa.field('LAD_BL_EPL', pa.float64()),
    pa.field('LAD_BL_UEPL', pa.float64()),
    pa.field('LAD_MCF', pa.float64()),
    pa.field('LAD_MCF_Corr', pa.float64()),
    pa.field('LAD_MLE_pimont_2019', pa.float64()),
    pa.field('LAD_MLE_Soma_21', pa.float64()),
    pa.field('PAD_BL', pa.float64()),
    pa.field('PAD_BL_EPL', pa.float64()),
    pa.field('PAD_BL_UEPL', pa.float64()),
    pa.field('PAD_MCF', pa.float64()),
    pa.field('PAD_MCF_Corr', pa.float64()),
    pa.field('PAD_MLE_pimont_2018', pa.float64())
])

### HELPER FUNCTIONS ###
# Commonly used functions that offer small utilities for components of other scripts.

# Create a unique ID for a voxel
def create_voxel_id(voxel_size, x, y, z):
    """
    Create a unique ID for a voxel.
    
    INPUTS:
        voxel_size: Size of the voxel
        x: X coordinate of the voxel
        y: Y coordinate of the voxel
        z: Z coordinate of the voxel

    OUTPUTS:
        voxel_id: A unique ID for the voxel
    """
    # Create a string representation of the voxel parameters
    voxel_string = f'{voxel_size}_{x}_{y}_{z}'

    # Encode the string and hash it using FNV-1a
    voxel_id = fnv1a_32(voxel_string.encode())
    print(f"Created unique voxel_id: {voxel_id} for voxel {voxel_string}")

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
def calculate_lambda_1(voxel_size, r=0.05):
    """
    Calculate lambda_1 for a given voxel size.
    
    INPUTS:
        voxel_size: A numpy array of voxel_sizes or int
        r: Radius of the beam divergence
    
    OUTPUTS:
        lambda_1: The calculated lambda_1

    When running metrics on LiDAR data, the r should be your beam divergence in radians.
    """
    lambda_1 = r / (2 * voxel_size)

    return

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

    # Compute the point density weights
    if len(points) < knn:
        weights = np.ones(len(points))
    else:
        nbrs = NearestNeighbors(n_neighbors=knn).fit(points)
        distances, _ = nbrs.kneighbors(points)
        # Inverse of the distance to the k nearest neighbours as weight
        weights = 1 / (distances[:, -1] + 1e-9) # Add a small value to avoid division by zero
    
    # Compute the normals
    if len(points) < max_nn:
        return np.array([]), np.array([]), np.array([])
    
    # Create a KDTree for efficient neighbour search
    tree = NearestNeighbors(radius=radius, n_neighbors=max_nn).fit(points)

    # Precompute neighbors for all points
    neighbors_indices = tree.radius_neighbors(points, return_distance=False)

    # Initialize normals array
    normals = np.full((len(points), 3), np.nan)

    for i, indices in enumerate(neighbors_indices):
        if len(indices) < max_nn:
            continue
        neighbours = points[indices]

        if len(neighbours) < 3:
            continue

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
    viewing_angles = group['viewing_angles'].values
    valid_points = group[['hit_ray', 'ray_points_x', 'ray_points_y', 'ray_points_z', 'is_leaf']].values
    valid_points = valid_points[valid_points[:, 0] == True][:, 1:]
    leaf_points = valid_points[valid_points[:, -1] == True][:, :2]

    bins, LIADs, angles = calculate_LIAD(valid_points)
    bins_leaf, LIADs_leaf, angles_leaf = calculate_LIAD(leaf_points)

    return viewing_angles, bins, LIADs, angles, bins_leaf, LIADs_leaf, angles_leaf


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
def calculate_G_mean(viewing_angles, bin_centres, LIAD_values):
    """
    Calculate the G function mean.
    
    INPUTS:
        viewing_angles: The viewing angles
        bin_centres: The bin centres
        LIAD_values: The LIAD values
    
    OUTPUTS:
        G_mean: The G function mean
    """
    # Check for empty arrays
    if len(viewing_angles) == 0 or len(bin_centres) == 0 or len(LIAD_values) == 0:
        return np.nan
    
    # Convert degrees to radians
    theta = np.radians(viewing_angles)
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
        AD:                 The calculated Leaf/Plant Area Density corrected with provided CI.
                            If not provided, CI is 1 and will calculate an effective Leaf Area Index (CI*LAI)
    """
    try:
        # Check for nans in inputs
        if np.isnan(I) or np.isnan(mean_path_length) or np.isnan(G) or np.isnan(CI):
            raise ValueError(f"One or more inputs are NaN: I={I}, mean_path_length={mean_path_length}, G={G}, CI={CI}")        

        # Calculate D (LAD or PAD depending on inputs)
        AD = -(np.log(1 - I) / (CI * G * mean_path_length))
    
    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD

def BL_EPL_pimont_2018(I, mean_eff_path_length, num_rays, epsilon=1e-9):
    """
    Calculate density using Beer-Lambert (Pimont et al. 2018) with Effective Path Length, equation 25.
        Λ̂ = {
          -1 / δ̄ₑ * (log(1 - I) + I / (2N(1 - I)))      when I < 1
          log(2N + 2) / δ̄ₑ                              when I = 1
        }
    
    Calculate PAD by passing I values that use all hits,
    and LAD by passing I values that use leaf hits only

    INPUTS:
        I:              A numpy array of Relative Density Indexes (num_hits/num_rays)
        mean_eff_path_length:   A numpy array of mean_eff_path_length
        num_rays:       A numpy array of num_rays
        epsilon:        A condition to avoid issues with zero division

    OUTPUTS:
        AD:             The calculated density
    """
    try:
        # Check for nans in inputs
        valid_mask = np.where(~np.isnan(I) & ~np.isnan(mean_eff_path_length) & ~np.isnan(num_rays) & num_rays > 0)
        I = I[valid_mask]
        mean_eff_path_length = mean_eff_path_length[valid_mask]
        num_rays = num_rays[valid_mask]

        # Split I < 1 and I == 1 values to handle separate calculations
        I_lt_1_mask = I < 1
        I_eq_1_mask = I == 1

        # Calculate AD (L or P depending on inputs)
        AD = np.where(
            I_lt_1_mask,    # I < 1
            -(1 / mean_eff_path_length) * (np.log(1 - I) + (I / (2 * num_rays * (1 - I)))),
            np.where(
            I_eq_1_mask,    # I == 1
            np.log(2 * num_rays + 2) / mean_eff_path_length,
            np.nan          # Other
            )
        )
    
    except Exception as e:
        print(f"Error: {e}")
        return np.nan
    
    return AD







### LARGE FUNCTIONS ###
# Functions that are used to perform large operations, such as calculating metrics or processing data.

def calculate_voxel_metrics(
        voxel_df,
        min_rays=6,
        G=None,
        CI=None,
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
                ray_points_x:       X points of ray
                ray_points_y:       Y points of ray
                ray_points_z:       Z points of ray
                viewing_angles:     Viewing angles of ray
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
                G_wood:             G Function calculated from wood hits
                mean_path_length:          Mean of full path length
                mean_z:             Mean z
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

    # Calculate pgap and I (relative density index)
    pgap = np.where(num_rays > 0, (num_rays - num_hits) / num_rays, np.nan)
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
                voxel[['ray_points_x', 'ray_points_y', 'ray_points_z']].values - voxel[['t_entry_x', 't_entry_y', 't_entry_z']].values,
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
    viewing_angles, bins, LIADs, angles, bins_leaf, LIADs_leaf, angles_leaf = voxel_df.apply(calculate_liad_values)

    # Calculate the G functions, unless otherwise specified
    if G is None:
        G = calculate_G_mean(viewing_angles, bins, LIADs)
        G_leaf = calculate_G_mean(viewing_angles, bins_leaf, LIADs_leaf)
    else:
        G = G_leaf

    invalid_ze_mask = eff_path_length_zs < 0
    if np.any(invalid_ze_mask):
        print(f"Warning: {np.count_nonzero(invalid_ze_mask)} invalid eff_path_length_zs values found, setting them to zero")
        eff_path_length_zs[invalid_ze_mask] = 0

    finite_mask = np.isfinite(eff_path_length_zs)
    hit_mask = voxel_df['hit_ray'].values
    leaf_hit_mask = voxel_df[hit_mask]['is_leaf'].values

    sum_free_path_lengths_e = np.sum(eff_path_length_zs[finite_mask]) if eff_path_length_zs.size > 0 else 0
    sum_hits_z_e = np.sum(eff_path_length_zs[hit_mask]) if num_hits > 0 else 0 # old code used eff_path_length_zs[:num_hits] which did not ensure alignment of EPL
    sum_hits_z_e_leaf = np.sum(eff_path_length_zs[leaf_hit_mask]) if num_leaf_hits > 0 else 0

    if num_hits > 0 and num_rays >= min_rays and np.isnan(G_mean):
        print(f"Warning: G Mean is NaN for voxel {voxel_id}")
    elif num_rays <= min_rays:
        G_mean = np.nan
        if num_hits > 0:
            print(f"Warning: Voxel has hits, but there are not enough rays to calculate metrics for voxel {voxel_id}")
        else:
            print(f"Warning: No hits or rays in {voxel_id}")

    # Calculate CI values?
    # Using the G Function to calculate CI
    # Or read reference CI in and use that per voxel

    # Establish voxel_metrics dataframe
    voxel_metrics_df = create_df_from_schema(voxel_metrics_schema)

    # Calculate PAD Metrics
    if PAD_BL:
        PAD_BL = calculate_PAD_BL(I, G, mean_path_length)

    LAD = calculate_lad_metrics(
        num_leaf_hits,
        num_rays,
        G_leaf,
        mean_path_length,
        mean_z,
        mean_eff_path_length,
        var_delta_e,
        lambda_1
    )

    # Calculate PAD metrics
    PAD = calculate_pad_metrics(
        num_hits,
        num_leaf_hits,
        num_rays,
        G_lw,
        mean_path_length,
        sum_free_path_lengths_lw,
        sum_free_path_lengths_e_lw,
        sum_hits_z_e_leaf,
        sum_hits_z_e_lw,
        alpha,
        mean_eff_path_length,
        var_delta_e,
        lambda_1,
        leaf_fraction
    )

    # Add metrics to the dataframe
    voxel_metrics_df['voxel_id'] = voxel_id
    voxel_metrics_df['num_rays'] = num_rays
    voxel_metrics_df['num_hits'] = num_hits
    voxel_metrics_df['num_leaf_hits'] = num_leaf_hits
    voxel_metrics_df['I'] = I
    voxel_metrics_df['I_leaf'] = I_leaf
    voxel_metrics_df['pgap'] = pgap
    voxel_metrics_df['mean_path_length'] = mean_path_length
    voxel_metrics_df['sum_path_length'] = sum_path_lengths
    voxel_metrics_df['mean_free_path_length'] = mean_free_path_lengths
    voxel_metrics_df['sum_free_path_length'] = sum_free_path_lengths
    voxel_metrics_df['mean_eff_path_length'] = mean_eff_path_length
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
        mean_z,                 # mean z
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
    hits_leaf, N, G_leaf, mean_path_length, mean_z, mean_eff_path_length, var_delta_e, lambda_1
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
        
    if (mean_z > eps) and (G_leaf > eps):
        results['LAD_MCF'] = I_leaf / (G_leaf * mean_z)  ## Pimont et al. 2018, eq. 8
    if (lambda_1 > 0 and mean_path_length > 0 and (0 < I_leaf < 1) and 
        (mean_z > eps) and (1 - lambda_1 * mean_path_length) > 0):
        denom = math.log(1.0 - lambda_1 * mean_path_length) * mean_z
        if abs(denom) > eps:
            val_corr = -1.0 * (lambda_1 * mean_path_length * I_leaf) / denom
            results['LAD_MCF_Corr'] = val_corr / G_leaf  ## Pimont et al. 2018, eq. 12
    return results

def compute_pad_metrics(
    hits_lw, hits_leaf, N, G_lw, mean_path_length, sum_free_path_lengths_lw, sum_free_path_lengths_e_lw, sum_hits_z_e_leaf,
    sum_hits_z_e_lw, alpha, mean_eff_path_length, var_delta_e, lambda_1, leaf_fraction
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
        results['PAD_MLE_pimont_2018'] = (alpha * leaf_fraction / (G_lw * sum_free_path_lengths_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_free_path_lengths_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_free_path_lengths_e_lw)
        results['LAD_MLE_pimont_2019'] = (alpha * leaf_fraction / (G_lw * sum_free_path_lengths_e_lw)) * bracket
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