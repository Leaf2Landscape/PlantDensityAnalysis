import os
import glob
import pandas as pd
import argparse
from datetime import datetime as dt
from tqdm import tqdm
from joblib import Parallel, delayed
from collections import defaultdict
import re

def main(preliminary_output_path, csv_path):
    """
    Collate results from multiple CSV files in the specified directory and save to a single CSV file.
    
    Args:
        preliminary_output_path (str): Path to the directory containing preliminary output CSV files.
        csv_path (str): Path to save the final collated CSV file.
    """
    if not os.path.exists(preliminary_output_path):
        raise FileNotFoundError(f"The specified path {preliminary_output_path} does not exist.")
    
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    
    # Collect all processed CSV files from the preliminary output path
    try:
        processed_files = glob.glob(os.path.join(preliminary_output_path, "results_*.csv"))
        print(f"Found {len(processed_files)} processed files in {preliminary_output_path}.")
    except:
        print(f"Error accessing files in the directory: {preliminary_output_path}")
        raise ValueError(f"Error accessing files in the directory: {preliminary_output_path}")
    
    # Group processed files by voxel size extracted from filename

    voxel_size_groups = defaultdict(list)
    pattern = re.compile(r"results_batch_\d+_vs_([0-9.]+)\.csv")

    for f in processed_files:
        match = pattern.search(os.path.basename(f))
        if match:
            voxel_size = match.group(1)
            voxel_size_groups[voxel_size].append(f)
        else:
            print(f"Warning: Could not extract voxel size from filename: {f}")

    print(f"Grouped files into {len(voxel_size_groups)} voxel size groups.")
    
    # Read and concatenate all dataframes
    def read_csv_file(f):
        return pd.read_csv(f)
    
    for voxel_size, files in voxel_size_groups.items():
        print(f"Processing {len(files)} files for voxel size {voxel_size}.")
        all_processed_dfs = Parallel(n_jobs=-1)(
            delayed(read_csv_file)(f) for f in tqdm(files, desc=f"Reading files for voxel size {voxel_size}")
        )
        concat_df = pd.concat(all_processed_dfs, ignore_index=True)
        print(f"Concatenated {len(all_processed_dfs)} dataframes for voxel size {voxel_size} with total size of {concat_df.shape[0]} rows and {concat_df.shape[1]} columns.")
        
        # Save the concatenated dataframe to a CSV file
        output_csv_path = os.path.join(os.path.dirname(csv_path), f"collated_results_vs_{voxel_size}.csv")
        concat_df.to_csv(output_csv_path, index=False, engine="pyarrow")
        print(f"Results for voxel size {voxel_size} saved to {output_csv_path}.")

    # Collect all processing times CSV files in the preliminary output path
    try:
        processing_times_files = glob.glob(os.path.join(preliminary_output_path, "processing_times_*.csv"))
        print(f"Found {len(processing_times_files)} processing times files in {preliminary_output_path}.")
    except:
        print(f"Error accessing files in the directory: {preliminary_output_path}")
        raise ValueError(f"Error accessing files in the directory: {preliminary_output_path}")
    
    # Split into unique voxel sizes
    processing_times_groups = defaultdict(list)
    pattern = re.compile(r"processing_times_batch_\d+_vs_([0-9.]+)\.csv")
    for f in processing_times_files:
        match = pattern.search(os.path.basename(f))
        if match:
            voxel_size = match.group(1)
            processing_times_groups[voxel_size].append(f)
        else:
            print(f"Warning: Could not extract voxel size from filename: {f}")

    print(f"Grouped processing times files into {len(processing_times_groups)} voxel size groups.")

    for voxel_size, files in processing_times_groups.items():
        print(f"Processing {len(files)} processing times files for voxel size {voxel_size}.")
        all_processed_dfs = Parallel(n_jobs=-1)(
            delayed(read_csv_file)(f) for f in tqdm(files, desc=f"Reading processing times files for voxel size {voxel_size}")
        )
        concat_df = pd.concat(all_processed_dfs, ignore_index=True)
        print(f"Concatenated {len(all_processed_dfs)} processing times dataframes for voxel size {voxel_size} with total size of {concat_df.shape[0]} rows and {concat_df.shape[1]} columns.")
        
        # Save the concatenated processing times dataframe to a CSV file
        output_processing_times_csv_path = os.path.join(os.path.dirname(csv_path), f"processing_times_vs_{voxel_size}.csv")
        concat_df.to_csv(output_processing_times_csv_path, index=False, engine="pyarrow")
        print(f"Processing times for voxel size {voxel_size} saved to {output_processing_times_csv_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate results from multiple CSV files.")
    parser.add_argument("preliminary_output_path", type=str, help="Path to the directory containing preliminary output CSV files.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to save the final collated CSV file.")
    
    args = parser.parse_args()
    
    preliminary_output_path = args.preliminary_output_path
    csv_path = args.csv_path

    if not csv_path.endswith('.csv'):
        csv_path = os.path.join(csv_path, f"collated_results_{dt.now().strftime('%Y%m%d_%H%M%S')}.csv")
    else:
        timestamped_suffix = dt.now().strftime('%Y%m%d_%H%M%S')
        csv_path = csv_path.replace('.csv', f'_{timestamped_suffix}.csv')
    
    if not os.path.exists(preliminary_output_path):
        print(f"Warning: The specified path {preliminary_output_path} does not exist. Please check the path.")
        raise FileNotFoundError(f"The specified path {preliminary_output_path} does not exist.")
    
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    main(preliminary_output_path, csv_path)
    print("Collation completed successfully.")