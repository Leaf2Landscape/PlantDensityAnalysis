# Voxel-Level Reference Clumping Index Retrieval and TLS Density Validation Framework

A reproducible workflow for retrieving voxel-level reference clumping indices and optical metrics from high-resolution 3D tree meshes, and for validating HELIOS++ simulated TLS density estimates against those references.

> **Note for peer review:** this repository is anonymised. The full manuscript title, author list, and citation will be added on acceptance.

## Overview

This project provides two complementary workflows:

1. **Generate Reference Voxel Maps** (`process_reference_single_obj.py`) — Create ground-truth optical metrics from 3D mesh models
2. **Analyze HELIOS Simulations** (`helios_analysis.ipynb`) — Process simulated ray-tracing data and compare with reference maps

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

The script searches group/object names (case-insensitive) for keywords. Leaf is matched by `leaf`, `leaves`, `leafs`; wood is matched by `wood`, `trunk`, `branch`, `stem`. Adjust the naming in your OBJ if needed.

### Basic Usage

```bash
python process_reference_single_obj.py /path/to/model.obj
```

### Output Files

All outputs are written to the **same directory as the input `.obj`** (not a subfolder). For a model at `models/plant.obj` with the default voxel sizes you get:

```
models/
├── plant_leaf.obj                              # Extracted leaf mesh
├── plant_wood.obj                              # Extracted wood mesh
├── plant_leaf_area.csv                         # Leaf area statistics
├── plant_inside_voxels_size0.01_thresh4.txt    # Wood interior points (volume calc)
├── plant_combined_results_0.2.csv              # Reference metrics at 0.2 m voxel size
├── plant_combined_results_0.5.csv              # Reference metrics at 0.5 m voxel size
├── plant_combined_results_1.0.csv              # ... one file per voxel size
├── plant_combined_results_2.0.csv
├── plant_performance_0.2.csv                   # Timing per voxel size
└── plant.log
```

> **Filename note:** the results CSV is named `{model}_{scene}_results_{voxel_size}.csv`, where `{scene}` comes from `--scene_formats` (default `combined`; can also be `leaf` or `wood`). With `--leaf_off`, a `_leaf_off` suffix is appended. So the default file for a 0.2 m grid is `plant_combined_results_0.2.csv`.

To stage the references for Part 2:

```bash
mkdir -p references
cp models/plant_combined_results_*.csv references/
```

### Reference Output Schema

Each `{model}_combined_results_{voxel_size}.csv` contains **one row per (voxel, face, zenith angle, scene)** — it is directional, not one row per voxel. Key columns:

| Column | Description |
|--------|-------------|
| `voxel_cx, voxel_cy, voxel_cz` | Voxel center coordinates |
| `face` | Voxel face the rays entered (`bottom`, `top`, `xplus`, `xminus`, `yplus`, `yminus`) |
| `zenith_angle`, `dx, dy, dz` | Ray zenith angle (deg) and direction vector |
| `LAD_ref`, `WAD_ref`, `PAD_ref` | Reference leaf / wood / plant area density (m²/m³) |
| `LAI_ref`, `WAI_ref`, `PAI_ref` | Reference leaf / wood / plant area index |
| `total_num_rays`, `total_num_hits`, `total_missed_rays` | Ray tallies for the combined mesh |
| `n_hits_leaf`, `n_hits_wood`, `n_hits_comb` | Hit counts per component |
| `I_leaf`, `I_wood`, `I_comb` | Interception (hit rate) per component |
| `pgap_leaf`, `pgap_wood`, `pgap_comb` | Gap probability per component (`1 − I`) |
| `mean_path_length_{leaf,wood,comb}` | Mean potential path length through the voxel |
| `mean_free_path_length_{leaf,wood,comb}` | Mean free path length |
| `mean_eff_free_path_length_{leaf,wood,comb}` | Mean effective free path length |
| `G_leaf_computed`, `G_wood_computed`, `G_comb_computed` | G-function per component for this zenith angle |
| `CI_Leaf`, `CI_Wood`, `CI_Comb` | Component clumping indices |
| `alpha`, `leaf_fraction`, `wood_fraction` | Woody-volume proportion and hit fractions |
| `LIAD_bin_2.5` … `LIAD_bin_87.5` | Leaf inclination angle distribution (18 bins, 5° each, keyed by bin centre) |
| `WIAD_bin_*`, `PIAD_bin_*` | Wood / plant inclination angle distributions |
| `scene` | Scene type for the row (`combined`, `leaf`, or `wood`) |

> `voxel_id` is **not** produced here. It is generated later (from the voxel centre and size) by `prepare_helios_data` in Part 2 if missing.

### Advanced Options

```bash
# Choose scene formats (default: combined). Affects the output filename.
python process_reference_single_obj.py model.obj \
  --scene_formats combined leaf wood

# Use custom voxel sizes
python process_reference_single_obj.py model.obj \
  --voxel_sizes 0.1 0.2 0.5 1.0

# Number of zenith angle bins (default: 18 → centres 2.5°…87.5°)
python process_reference_single_obj.py model.obj \
  --num_angle_bins 18

# Use custom ray spacing (default: 0.005 m)
python process_reference_single_obj.py model.obj \
  --ray_spacing 0.01

# Enable wood volume calculation with custom threshold
python process_reference_single_obj.py model.obj \
  --wood_volume_voxel_size 0.01 \
  --wood_volume_threshold 4

# Disable leaf raytracing (wood-only metrics; adds a _leaf_off filename suffix)
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

The `helios_analysis.ipynb` notebook provides a step-by-step workflow to process HELIOS++ simulation outputs and validate them against the reference voxel maps.

### Prerequisites

- HELIOS++ simulation output files (see expected structure below)
- Reference voxel maps generated in Part 1
- Jupyter notebook environment with dependencies from `environment.yml`

### Input Structure: HELIOS Data

Expected directory layout:

```
project_dir/
├── helios/
│   ├── leg000_points.xyz          # Ray hit points (read)
│   ├── leg000_pulse.txt           # Pulse / ray origins & directions (read)
│   ├── leg001_points.xyz
│   ├── leg001_pulse.txt
│   └── ...                         # Additional legs
│
├── references/
│   ├── plant_combined_results_0.2.csv   # Reference from Part 1
│   ├── plant_combined_results_0.5.csv
│   └── ...
│
└── results/                        # Created automatically (or set output_dir)
```

> **Input note:** the pipeline reads only `*_pulse.txt` and `*_points.xyz`. Full-waveform (`*_fullwave.txt`) files are not consumed by the single-return workflow. Leg numbers are parsed from the `leg<NNN>_` prefix.

### Notebook Workflow

The notebook runs as a **four-step** pipeline. Leaf normal vectors and ray weights are computed **automatically inside `get_voxel_metrics`** (Step 4) — there is no separate normals step.

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
use_class = False           # Set True to use the class field instead of hit_object_id

# Classification preview — pass output_dir=None to display without saving
test_helios_settings(helios_dir, use_class, leaf_object_ids,
                     wood_object_ids, output_dir=project_dir)
```

#### Step 2: Extract Valid Rays from HELIOS

Convert HELIOS XYZ and pulse data into efficient Parquet format, keeping only rays that pass through the reference voxel grids' bounding box:

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

For each valid ray and voxel size, determine ray-voxel intersections via DDA traversal and record entry/exit points, leaf/wood classification, and ray parameters. Output is kept per-leg so metrics can later be computed from any combination of legs without recomputing intersections:

```python
from utils import voxel_ray_intersections

voxel_ray_intersections(
    valid_rays_dir=valid_rays_dir,
    references_dir=references_dir,
    debug=False
)
```

**Output**: `leg_{leg_id}_voxel_{voxel_size}_intersections.parquet` (voxel size formatted to one decimal, e.g. `0.2`, `1.0`, `2.0`).

#### Step 4: Calculate Metrics (and Merge with Reference)

Aggregate the intersections into per-voxel optical metrics (gap probability, G-function, path lengths, LIAD/WIAD/PIAD and their de Wit classifications). **Leaf normals and ray weights are computed here automatically.**

```python
from utils import calculate_lambda_1, get_voxel_metrics

# Set average leaf area (m²) for the lambda_1 calculation
# (adjust based on your plant species/model)
average_leaf_area = 0.003582

voxel_metrics_df = get_voxel_metrics(
    intersections_folder=intersections_dir,
    average_leaf_area=average_leaf_area,
    output_dir=results_dir,        # defaults to intersections_folder if omitted
    project_name='plant',
    scan_ids=None,                 # None = all legs; or e.g. ['0','1','2']
    voxel_sizes=None,              # None = all sizes found; or e.g. [0.2, 0.5]
    is_multireturn=False
)
```

**Output of `get_voxel_metrics`**: one CSV per voxel size, named
`{project_name}_voxel_metrics_{voxel_size}m_{timestamp}.csv`
(e.g. `plant_voxel_metrics_0.5m_20260615_103000.csv`), written to `output_dir`. The first line is a `# Scan IDs:` comment listing the legs included.

> **Important:** `get_voxel_metrics` computes the HELIOS-side metrics only; it does **not** merge the reference CSVs or compute `LAD`/`PAD` itself. The density estimation (Beer–Lambert / MLE / contact frequency) and the merge against the reference maps are done in the notebook using the helper functions in `utils.py` (`BL_pimont_2018`, `BL_EPL_UEPL_pimont_2018`, `MCF_beland_2011`, `MCF_corrected_beland_2014`, `MLE_pimont_2019`, `MLE_soma_2021`, `MLE_vincent_2021`, `CI_adjusted`, …). The reference join key is `voxel_id`.

### `get_voxel_metrics` Output Schema (single-return)

Selected columns from `voxel_metrics_schema_singlereturn`:

| Column | Description |
|--------|-------------|
| `voxel_id` | Voxel identifier |
| `voxel_cx, voxel_cy, voxel_cz`, `voxel_size` | Voxel centre and size |
| `num_rays` | Number of valid rays in the voxel |
| `num_hits`, `num_leaf_hits` | Combined and leaf hit counts |
| `I_lw`, `I_leaf`, `I_wood` | Interception (hit rate) for plant / leaf / wood |
| `pgap_lw`, `pgap_leaf`, `pgap_wood` | Gap probabilities (`1 − I`) |
| `G_leaf`, `G_wood`, `G_lw` | G-function per component |
| `lambda_1` | λ₁ used for effective path lengths |
| `mean_angle_leaf`, `mean_angle_lw` | Mean viewing angle of hits |
| `mean_path_length`, `sum_path_length` | Potential path length stats |
| `mean_free_path_length`, `sum_free_path_length`, `sum_free_path_length_hit`, `sum_free_path_length_hit_leaf` | Free path length stats |
| `mean_eff_free_path_length`, `var_eff_free_path_length`, `sum_eff_free_path_length`, `sum_eff_free_path_length_hit`, `sum_eff_free_path_length_hit_leaf` | Effective free path length stats |
| `bins_json`, `liad_json`, `wiad_json`, `piad_json` | Bin centres and L/W/P-IAD histograms (JSON) |
| `liad_dewit`, `liad_dewit_rmse`, `liad_dewit_l1` | LIAD de Wit class and fit scores (same for `wiad_*`, `piad_*`) |

With `is_multireturn=True`, the `voxel_metrics_schema_multireturn` adds Kent & Bailey gap probabilities (`P_first`, `P_equal`, `P_intensity`, …) and AMAPVox-style estimators (`LAD_MLE_nocorr`, `LAD_MLE_lambda1`, `LAD_MLE_bias`, `LAD_MLE_lambda1_bias`).

---

## Example Workflow: Start to Finish

### 1. Generate Reference (One-time)

```bash
cd /path/to/project

# Create reference voxel maps from your 3D model
# Assumes plant.obj with "leaf"/"wood" (or trunk/branch/stem) groups
python process_reference_single_obj.py models/plant.obj \
  --voxel_sizes 0.2 0.5 1.0 2.0 \
  --ray_spacing 0.005 \
  --max_workers 32

# This creates (in models/):
# - plant_leaf.obj, plant_wood.obj
# - plant_leaf_area.csv
# - plant_combined_results_0.2.csv
# - plant_combined_results_0.5.csv
# - plant_combined_results_1.0.csv
# - plant_combined_results_2.0.csv

# Copy reference files to the expected location
mkdir -p references
cp models/plant_combined_results_*.csv references/
```

### 2. Analyze HELIOS & Compare (Iterative)

```bash
# Open notebook
jupyter notebook helios_analysis.ipynb

# In the notebook:
# 1. Update paths: project_dir, helios_dir, references_dir
# 2. Update object IDs to match your HELIOS simulation
# 3. Run steps 1–4 in order
# 4. Check the per-voxel-size CSVs written by get_voxel_metrics, plus the
#    merged/density-estimated outputs produced by the notebook glue cells
```

### 3. Validate Results

After the notebook has merged HELIOS metrics with the reference maps and computed density estimates, compare against reference values:

```python
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

# Load the merged results produced by the notebook (filename depends on how
# you save the merge; the get_voxel_metrics CSV alone does not contain LAD/LAD_ref)
results = pd.read_csv('results/plant_merged_voxel_0.5m.csv')

# Compare LAD between HELIOS and reference (drop voxels with no reference)
valid = results.dropna(subset=['LAD', 'LAD_ref'])
print(valid[['LAD', 'LAD_ref']].describe())

# Calculate RMSE for validation
rmse = np.sqrt(mean_squared_error(valid['LAD'], valid['LAD_ref']))
print(f"RMSE: {rmse:.4f}")
```

---

## Performance Tips

- **Memory Usage**: Set `TMPDIR` to control the cache/scratch location
  ```bash
  export TMPDIR=/path/to/large/scratch
  python process_reference_single_obj.py model.obj
  ```

- **Parallelization**: Adjust `--max_workers` based on your CPU/memory
  ```bash
  python process_reference_single_obj.py model.obj --max_workers 64
  ```

- **GPU Acceleration**: Open3D raytracing requests a SYCL device when available and falls back to CPU automatically. `nvidia-smi` is queried to select a device index.

- **Ray Spacing Tradeoff**:
  - Smaller values (0.005 m) = higher accuracy, longer computation
  - Larger values (0.01 m) = faster, lower resolution

- **Voxel Size Selection**:
  - Smaller voxels (0.1–0.2 m) = finer spatial detail, more data
  - Larger voxels (2.0 m) = faster, broader patterns

---

## Troubleshooting

**"No leaf/wood mesh found"**
- Check the OBJ has groups/objects whose names contain `leaf`/`leaves`/`leafs` or `wood`/`trunk`/`branch`/`stem` (case-insensitive).
- Verify group syntax: `g leaf_name` or `o object_name`.

**Reference CSVs not picked up in the notebook**
- Confirm the files were copied into `references/` and are named `*_results_{voxel_size}.csv` (or `*voxel_size_{size}.csv`); both patterns are parsed.

**"No intersection files found" in notebook**
- Confirm the HELIOS directory path is correct.
- Ensure files follow naming: `leg*_points.xyz` and `leg*_pulse.txt`.

**Memory errors on large datasets**
- Reduce `--max_workers` in reference generation (e.g. 16 instead of 32).
- Process HELIOS data in fewer legs / voxel sizes at a time (`scan_ids`, `voxel_sizes`).
- Use HPC batch scripts for massive datasets.

**Mismatch between HELIOS and reference metrics**
- Verify leaf/wood object IDs match between the HELIOS simulation and your `leaf_object_ids` / `wood_object_ids`.
- Check `average_leaf_area` matches your plant model.
- Confirm the reference voxel grids are for the same 3D model.

---

## Citation

If you use this workflow in research, please cite:
- HELIOS++ laser scanning simulator: [Winiwarter et al., 2022](https://doi.org/10.1016/j.rse.2021.112772)
- Optical metric / estimator frameworks: [Pimont et al., 2018](https://doi.org/10.1016/j.rse.2018.06.024); [Pimont et al., 2019](https://doi.org/10.3390/rs11131580)

---

## Further Reading

- `helios_analysis.ipynb` — Interactive, documented notebook driving the full HELIOS analysis workflow
- `process_reference_single_obj.py` — Reference voxel-map generator (Part 1)
- `utils.py` — Shared library of core computation and optical-metric functions used by the notebook
- `hpc_scripts/` — Batch processing utilities for large datasets on HPC clusters (see `hpc_scripts/README.md`)
- `environment.yml` — Conda environment specification
