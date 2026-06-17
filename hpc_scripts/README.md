# HPC Batch Processing Scripts

These scripts demonstrate batch processing patterns for large-scale plant density analysis on SLURM-scheduled HPC systems. They show how to distribute voxel metric computation across multiple compute nodes using job arrays.

**Note:** These are example implementations for reference and validation purposes. They are **not production-ready** and may require adaptation for your specific HPC environment, resource constraints, and data volumes.

## Scripts

- **`init_hpc_voxel_batches.py`** – Prepares batch configuration and splits work into discrete tasks
- **`process_voxel_batch.py`** – Processes a single batch on a compute node
- **`collate_results.py`** – Aggregates results from all batches into final output CSVs
- **`apptainer.def`** – Container definition for reproducible HPC environments

## Typical Workflow

```bash
# Initialize batches
python init_hpc_voxel_batches.py <config.yaml>

# Submit job array to SLURM (example)
sbatch --array=0-99 slurm_submit.sh

# After all jobs complete, collate results
python collate_results.py <results_dir>
```

## Customization Required

Before deploying on your HPC cluster, review and adapt:
- SLURM partition/resource requests
- Module loads and environment setup
- Batch size and memory allocation per task
- Temporary storage paths and cleanup
- Error handling and retry logic
- Logging and monitoring
