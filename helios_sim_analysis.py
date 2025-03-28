

### ROUGH CODE FOR HANDLING DATAFRAME PARTITIONS ###

# Group pandas dataframe by voxel_id
legs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]  # Example legs
voxel_sizes = [0.5]  # Example voxel sizes



for size in voxel_sizes:
    voxel_centers_file = f"200_Leaf_60/voxel_centers_{size}.parquet"
    voxel_centers = dd.read_parquet(voxel_centers_file).compute()
    voxel_centers = voxel_centers[['voxel_id', 'voxel_center_x', 'voxel_center_y', 'voxel_center_z']]

    lambda_1 = calculate_lambda1(size)
    print(f"Voxel size: {size}, lambda_1: {lambda_1}")

    # Load and concatenate dataframes
    dataframes = []
    for leg in legs:
        for voxel_size in voxel_sizes:
            file_path = f"200_Leaf_60/voxel_ray_intersections/leg_{leg}_voxel_{voxel_size}_ray_intersections.parquet"
            df = dd.read_parquet(file_path)
            dataframes.append(df)

    combined_df = dd.concat(dataframes)

    # Partition into memory-friendly chunks
    npartitions = combined_df.npartitions
    partition_sizes = combined_df.map_partitions(lambda df: df.memory_usage(deep=True).sum()).compute()
    max_partition_size = partition_sizes.max()
    available_memory = psutil.virtual_memory().available

    # Ensure all sizes are in bytes
    max_partition_size = int(max_partition_size)  # Convert to integer if not already
    available_memory = int(available_memory)  # Convert to integer if not already

    max_partitions_per_chunk = min(npartitions, available_memory // max_partition_size)
    chunk_size = max(1, npartitions // max_partitions_per_chunk)
    combined_df = combined_df.repartition(npartitions=chunk_size)

    # test
    def test(df):
        test = calculate_voxel_metrics_all_voxels(df)
        return test

    test = combined_df.map_partitions(test)
    test.compute()