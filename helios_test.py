import utils

parquet_file = "/home/capheus/PlantDensityAnalysis_TOBEFIXED/leg_33_voxel_2.0_intersections.parquet"

if __name__ == "__main__":
    output_path = "/home/capheus/PlantDensityAnalysis_TOBEFIXED/leg_33_voxel_2.0_intersections.csv"
    print(f"Converting {parquet_file} to {output_path}")
    utils.convert_parquet_to_csv(parquet_file, output_path)
    print(f"Conversion completed. Output saved to {output_path}")