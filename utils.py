"""
A Python script of commonly shared utilities for other scripts.
"""

def compute_LAD_metrics():
    """
    Compute the LAD metrics.
    
    This script expects the following inputs:
    - A list of actual values

    It will return a dictionary of LAD metrics, per voxel
    """

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
    hits_leaf, N, G_leaf, delta_bar, mean_z, mean_delta_e, var_delta_e, lambda_1
):
    """
    Compute LAD metrics from the leaf-only simulation.
    Uses the full path length delta_bar (mean of ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ) and the effective free path length (mean_delta_e from Z).
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
        results['LAD_BL'] = -math.log(1.0 - I_leaf) / (G_leaf * delta_bar)  ## Pimont et al. 2018, eq. 5
    # Compute bot unconditionally when N > 0 so it can be used later
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
    lad_bl_epl_val = results['LAD_BL_EPL']  ## Pimont et al. 2018, eq. 25
    if (not np.isnan(lad_bl_epl_val) and lad_bl_epl_val > 0 and
        var_delta_e > eps and mean_delta_e > eps):
        a_e = var_delta_e / mean_delta_e
        inside = 1.0 - 2.0 * a_e * lad_bl_epl_val
        if inside >= 0.0:
            val_uepl = (1.0 / a_e) * (1.0 - math.sqrt(inside))
            results['LAD_BL_UEPL'] = val_uepl / G_leaf  ## Pimont et al. 2018, eq. 27
    if (not np.isnan(lad_bl_epl_val) and lad_bl_epl_val > 0 and (G_leaf > eps)):
        results['LAD_BL_EPL'] = lad_bl_epl_val / G_leaf
        
    if (mean_z > eps) and (G_leaf > eps):
        results['LAD_MCF'] = I_leaf / (G_leaf * mean_z)  ## Pimont et al. 2018, eq. 8
    if (lambda_1 > 0 and delta_bar > 0 and (0 < I_leaf < 1) and 
        (mean_z > eps) and (1 - lambda_1 * delta_bar) > 0):
        denom = math.log(1.0 - lambda_1 * delta_bar) * mean_z
        if abs(denom) > eps:
            val_corr = -1.0 * (lambda_1 * delta_bar * I_leaf) / denom
            results['LAD_MCF_Corrected'] = val_corr / G_leaf  ## Pimont et al. 2018, eq. 12
    return results

def compute_pad_metrics(
    hits_lw, hits_leaf, N, G_lw, delta_bar, sum_z_lw, sum_z_e_lw, sum_hits_z_e_leaf,
    sum_hits_z_e_lw, alpha, mean_delta_e, var_delta_e, lambda_1, leaf_fraction
):
    """
    Compute PAD metrics from the combined (leaf+wood) simulation.
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
    # Compute bot unconditionally when N > 0 so it can be used later
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
        a_e = var_delta_e / mean_delta_e
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
    if (G_lw > eps) and (sum_z_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_lw / sum_z_e_lw)
        results['PAD_MLE_pimont_2018'] = (alpha * leaf_fraction / (G_lw * sum_z_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_z_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_z_e_lw)
        results['LAD_MLE_pimont_2019'] = (alpha * leaf_fraction / (G_lw * sum_z_e_lw)) * bracket
    leaf_fraction = hits_leaf / hits_lw if hits_lw > 0 else 0.0
    if (G_lw > eps) and (sum_z_e_lw > eps):
        bracket = hits_lw - (sum_hits_z_e_leaf / sum_z_lw)
        results['LAD_MLE_Soma_21'] = (leaf_fraction / (G_lw * sum_z_lw)) * bracket
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