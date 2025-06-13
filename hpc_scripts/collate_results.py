import os
import glob
import pandas as pd
import argparse
from datetime import datetime as dt

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
    
    # Collect all CSV files in the preliminary output path
    all_files = glob.glob(os.path.join(preliminary_output_path, "*.csv"))
    
    if not all_files:
        raise ValueError("No CSV files found in the specified directory.")
    
    # Read and concatenate all dataframes
    all_dfs = [pd.read_csv(f) for f in all_files]
    final_df = pd.concat(all_dfs, ignore_index=True)
    
    # Save the final dataframe to a CSV file
    final_df.to_csv(csv_path, index=False)
    print(f"Final results saved to {csv_path}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collate results from multiple CSV files.")
    parser.add_argument("preliminary_output_path", type=str, required=True, help="Path to the directory containing preliminary output CSV files.")
    parser.add_argument("--csv_path", type=str, required=True, help="Path to save the final collated CSV file.")
    
    args = parser.parse_args()
    
    preliminary_output_path = args.preliminary_output_path
    csv_path = args.csv_path
    
    if not os.path.exists(preliminary_output_path):
        raise FileNotFoundError(f"The specified path {preliminary_output_path} does not exist.")
    
    if not os.path.exists(csv_path):
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)