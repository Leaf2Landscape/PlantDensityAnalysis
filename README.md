# PlantDensityAnalysis: Voxel Reference Retrieval from 3D Mesh and HELIOS Simulation Analysis & Comparison

A complete workflow for analyzing plant canopy structure using HELIOS simulations and validating results against high-resolution 3D reference meshes.

## Overview

This project provides two complementary workflows:

1. **Generate Reference Voxel Maps** (`process_reference_single_obj.py`) - Create ground-truth optical metrics from 3D mesh models
2. **Analyze HELIOS Simulations** (`helios_analysis.ipynb`) - Process simulated ray-tracing data and compare with reference maps

---

## Part 1: Generate Reference Voxel Maps

The reference generation script processes a 3D plant model (OBJ file with leaf and wood geometry) to compute voxel-based optical metrics for validation.

### Prerequisites

- 3D model file (`model.obj`) with leaf and wood geometry separated by groups or objects
- Python environment with dependencies: `trimesh`, `open3d`, `pyvista`, `numpy`, `pandas`

### Input Format: OBJ File Structure

Your OBJ file should have groups or objects labeled to distinguish leaf and wood:

```
# Example structure
g leaf_group
v 0.1 0.2 0.3
v 0.4 0.5 0.6
...
f 1 2 3
f 2 3 4

g wood_group
v 1.1 1.2 1.3
...
f 5 6 7
```

The script searches for keywords (`leaf`, `wood`) in group/object names—adjust naming if needed.

### Basic Usage

```bash
python process_reference_single_obj.py /path/to/model.obj
```

### Output Files

For each voxel size specified, the script generates:

```
model/
├── model_leaf.obj                          # Extracted leaf mesh
├── model_wood.obj                          # Extracted wood mesh
├── model_leaf_area.csv                     # Leaf area statistics
├── model_voxel_size_0.2.csv                # Reference metrics at 0.2m voxel size
├── model_voxel_size_0.5.csv                # Reference metrics at 0.5m voxel size
└── ...                                     # Additional voxel sizes
```

### Reference Output Schema

Each `voxel_size_*.csv` contains:

| Column | Description |
|--------|-------------|
| `voxel_id` | Unique voxel identifier |
| `voxel_cx, voxel_cy, voxel_cz` | Voxel center coordinates |
| `LAD_ref` | Leaf Area Density (reference) |
| `PAD_ref` | Plant Area Density (reference) |
| `CI_leaf_corr_ref` | Leaf clumping index |
| `CI_lw_corr_ref` | Leaf-wood clumping index |
| `G_leaf_ref`, `G_wood_ref` | G-function values |
| `liad_bin_0` to `liad_bin_17` | Leaf Inclination Angle Distribution (18 bins, 5° each) |

### Advanced Options

```bash
# Use custom voxel sizes
python process_reference_single_obj.py model.obj \
  --voxel_sizes 0.1 0.2 0.5 1.0

# Use custom ray spacing (affects metric resolution)
python process_reference_single_obj.py model.obj \
  --ray_spacing 0.01

# Enable wood volume calculation with custom threshold
python process_reference_single_obj.py model.obj \
  --wood_volume_voxel_size 0.01 \
  --wood_volume_threshold 4

# Disable leaf raytracing (wood-only metrics)
python process_reference_single_obj.py model.obj \
  --leaf_off

# Enable debug mode to save intermediate raytracing results
python process_reference_single_obj.py model.obj \
  --debug

# Parallelize with custom worker count
python process_reference_single_obj.py model.obj \
  --max_workers 64
```

---

## Part 2: Analyze HELIOS Simulations & Compare with Reference

The `helios_analysis.ipynb` notebook provides a step-by-step workflow to process HELIOS simulation outputs and validate them against the reference voxel maps.

### Prerequisites

- HELIOS simulation output files (see expected structure below)
- Reference voxel maps generated in Part 1
- Jupyter notebook environment with dependencies from `environment.yml`

### Input Structure: HELIOS Data

Expected directory layout:

```
project_dir/
├── helios/
│   ├── leg000_points.xyz          # Ray hit points
│   ├── leg000_pulse.txt           # Pulse information
│   ├── leg000_fullwave.txt        # Full waveform data
│   ├── leg001_points.xyz
│   ├── leg001_pulse.txt
│   ├── leg001_fullwave.txt
│   └── ...                         # Additional legs
│
├── references/
│   ├── model_voxel_size_0.2.csv   # Reference from Part 1
│   ├── model_voxel_size_0.5.csv
│   └── ...
│
└── results/                        # Created automatically
```

### Notebook Workflow

The notebook runs as a **four-step** pipeline. Leaf normal vectors and ray weights are now computed **automatically inside `get_voxel_metrics`** (Step 4) — there is no longer a separate normals step.

#### Step 1: Configure Paths and Object IDs

Set the project paths and the object IDs that identify leaf vs. wood in your HELIOS data. The same cell runs a quick classification check via `test_helios_settings`, which plots a sample of points coloured by leaf/wood so you can confirm the IDs before processing:

```python
project_dir = '/path/to/project'
helios_dir = '/path/to/helios/data'
references_dir = '/path/to/references'

# Specify which object IDs in HELIOS correspond to leaf/wood
# (these must match the IDs used in your HELIOS simulation)
leaf_object_ids = [0]       # Adjust based on your HELIOS setup
wood_object_ids = [1]       # Adjust based on your HELIOS setup
use_class = False           # Set True if using class field instead of object IDs

# Classification preview — pass output_dir=None to display without saving
test_helios_settings(helios_dir, use_class, leaf_object_ids,
                     wood_object_ids, output_dir=project_dir)
```

#### Step 2: Extract Valid Rays from HELIOS

Convert HELIOS XYZ, pulse, and fullwave data into efficient Parquet format, keeping only rays that passed through reference voxel grids:

```python
from utils import prepare_helios_data

prepare_helios_data(
    input_dir=helios_dir, 
    output_dir=valid_rays_dir, 
    references_dir=references_dir, 
    leaf_object_ids=leaf_object_ids, 
    wood_object_ids=wood_object_ids, 
    use_class=use_class
)
```

**Output**: `valid_rays/leg_{leg_id}_valid_rays.parquet`

#### Step 3: Compute Ray-Voxel Intersections

For each valid ray and voxel size, determine ray-voxel intersections and record entry/exit points, leaf/wood classification, and ray parameters. Output is kept per-leg so metrics can later be computed from any combination of legs without recomputing intersections:

```python
from utils import voxel_ray_intersections

voxel_ray_intersections(
    valid_rays_dir=valid_rays_dir, 
    references_dir=references_dir,
    debug=False
)
```

**Output**: `leg_{leg_id}_voxel_{voxel_size}_intersections.parquet`

#### Step 4: Calculate Metrics and Merge with Reference

Aggregate the intersections into per-voxel optical metrics (LAD, PAD, gap probability, G-function, LIAD). **Leaf normals and ray weights are computed here automatically.** The results are then merged with the reference voxel maps for validation:

```python
from utils import calculate_lambda_1, get_voxel_metrics

# Select legs and voxel sizes to analyze
legs = 'all'  # or specify list: [0, 1, 2]
voxel_sizes = [0.2, 0.5, 1.0, 2.0]

# Set average leaf area (m²) for lambda calculation
# (adjust based on your plant species/model)
average_leaf_area = 0.003582

# For each voxel size: compute lambda_1, run get_voxel_metrics,
# merge with the matching references CSV, and save the result.
# Output: results/{project_name}_leg_{legs}_voxel_size_{size}.csv
```

### Output: Comparison Results

For each voxel size, results are saved as CSV with merged HELIOS and reference metrics:

| Column | Description |
|--------|-------------|
| `voxel_id` | Voxel identifier |
| `N` | Number of valid rays in voxel |
| `n_hits` | Ray hits in voxel |
| `I` | Hit rate (ray transmittance) |
| `G_leaf` | G-function for leaf (HELIOS) |
| `LAD` | Leaf Area Density (HELIOS) |
| `LAD_ref` | Leaf Area Density (Reference) |
| `PAD` | Plant Area Density (HELIOS) |
| `PAD_ref` | Plant Area Density (Reference) |
| `gap_probability` | Probability of ray gap (1 - I) |
| `CI_leaf_corr_ref` | Leaf clumping index |
| `liad_bin_*_ref` | Reference LIAD bins |

---

## Example Workflow: Start to Finish

### 1. Generate Reference (One-time)

```bash
cd /path/to/project

# Create reference voxel maps from your 3D model
# Assumes model.obj with "leaf" and "wood" groups
python process_reference_single_obj.py models/plant.obj \
  --voxel_sizes 0.2 0.5 1.0 2.0 \
  --ray_spacing 0.005 \
  --max_workers 32

# This creates:
# - models/plant_leaf.obj (extracted leaf mesh)
# - models/plant_wood.obj (extracted wood mesh)  
# - models/plant_voxel_size_0.2.csv
# - models/plant_voxel_size_0.5.csv
# - models/plant_voxel_size_1.0.csv
# - models/plant_voxel_size_2.0.csv

# Copy reference files to expected location
mkdir -p references
cp models/plant_voxel_size*.csv references/
```

### 2. Analyze HELIOS & Compare (Iterative)

```bash
# Open notebook
jupyter notebook helios_analysis.ipynb

# In the notebook:
# 1. Update paths: project_dir, helios_dir, references_dir
# 2. Update object IDs to match your HELIOS simulation
# 3. Run steps 1-4 in order
# 4. Check results in results/{project}_leg_*_voxel_size_*.csv
```

### 3. Validate Results

Compare HELIOS-derived metrics with reference values:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load results for a specific voxel size
results = pd.read_csv('results/project_leg_all_voxel_size_0.5.csv')

# Compare LAD between HELIOS and reference (drop voxels with no reference)
valid = results.dropna(subset=['LAD', 'LAD_ref'])
print(valid[['LAD', 'LAD_ref']].describe())

# Calculate RMSE for validation
rmse = np.sqrt(mean_squared_error(valid['LAD'], valid['LAD_ref']))
print(f"RMSE: {rmse:.4f}")
```

---

## Performance Tips

- **Memory Usage**: Set `TMPDIR` environment variable to control cache location
  ```bash
  export TMPDIR=/path/to/large/scratch
  python process_reference_single_obj.py model.obj
  ```

- **Parallelization**: Adjust `--max_workers` based on your CPU/memory
  ```bash
  python process_reference_single_obj.py model.obj --max_workers 64
  ```

- **GPU Acceleration**: Open3D raytracing automatically uses CUDA if available; falls back to CPU

- **Ray Spacing Tradeoff**: 
  - Smaller values (0.005m) = higher accuracy, longer computation
  - Larger values (0.01m) = faster, lower resolution

- **Voxel Size Selection**:
  - Smaller voxels (0.1m) = finer spatial detail, more data
  - Larger voxels (2.0m) = faster, broader patterns

---

## Troubleshooting

**"No leaf/wood mesh found"**
- Check OBJ file has groups/objects with names containing `leaf` or `wood` (case-insensitive)
- Verify group syntax: `g leaf_name` or `o object_name`

**"No intersection files found" in notebook**
- Confirm HELIOS directory path is correct
- Ensure files follow naming: `leg*_points.xyz`, `leg*_pulse.txt`, `leg*_fullwave.txt`

**Memory errors on large datasets**
- Reduce `--max_workers` in reference generation (e.g., 16 instead of 32)
- Process HELIOS data in smaller legs/voxel_sizes
- Use HPC batch scripts for massive datasets

**Mismatch between HELIOS and reference metrics**
- Verify leaf/wood object IDs match between HELIOS simulation and your `--leaf_object_ids` / `--wood_object_ids`
- Check `average_leaf_area` parameter matches your plant model
- Confirm reference voxel grids are for the same 3D model

---

## Citation

If you use this workflow in research, please cite:
- HELIOS ray-tracing simulator: [Qi et al., 2019](https://doi.org/10.1016/j.rse.2019.03.018)
- Optical metrics framework: [Pimont et al., 2018](https://doi.org/10.1016/j.rse.2018.04.041)

---

## Further Reading

- `helios_analysis.ipynb` — Interactive, documented notebook driving the full HELIOS analysis workflow
- `process_reference_single_obj.py` — Reference voxel-map generator (Part 1)
- `utils.py` — Shared library of core computation and optical-metric functions used by the notebook
- `hpc_scripts/` — Batch processing utilities for large datasets on HPC clusters (see `hpc_scripts/README.md`)
- `environment.yml` — Conda environment specification
