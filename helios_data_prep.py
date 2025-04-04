"""
Prepare helios simulation data for use in voxel_ray intersections.

This script requires an input folder (containing the helios simulation data), an output folder (where the processed data will be saved),
and a references folder that contains all the voxel_centers reference data (used to establish plot boundaries).

i.e. 
python helios_data_prep.py /path/to/input_dir /path/to/output_dir /path/to/references

"""

import argparse
import utils


if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Prepare helios simulation data for voxel_ray intersections.")
    parser.add_argument("input_dir", type=str, help="Path to the input folder containing helios simulation data.")
    parser.add_argument("output_dir", type=str, help="Path to the output folder where processed data will be saved.")
    parser.add_argument("references_dir", type=str, help="Path to the references folder.")

    # Parse the arguments
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    references_dir = args.references_dir

    # Start main process
    utils.prepare_helios_data(input_dir, output_dir, references_dir) # Optional debug=False for less logging information
