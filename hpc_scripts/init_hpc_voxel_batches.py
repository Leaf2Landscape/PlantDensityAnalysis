import os
import csv
import json
from typing import List, Tuple, Union, Optional
from pathlib import Path
import pyvista as pv
import numpy as np
import argparse
import math


def _faces_to_poly(vertices: np.ndarray, faces: List[List[int]]) -> Optional[pv.PolyData]:
    """
    Convert vertices and faces to a PyVista PolyData object.
    """
    if not faces:
        return None
    
    faces_arr = np.asarray(faces, dtype=np.int64)
    n_per_face = np.full((faces_arr.shape[0], 1), faces_arr.shape[1], dtype=np.int64)
    faces_flat = np.hstack((n_per_face, faces_arr)).ravel()
    return pv.PolyData(vertices, faces_flat)

def load_and_split_by_group(scene_file: Union[str, Path], leaf_keys, wood_keys) -> Tuple[Optional[pv.PolyData], Optional[pv.PolyData], Tuple[float, float, float, float, float, float]]:
    """
    Placeholder function to load and split the scene file into leaf and wood meshes.
    Replace this with actual logic from your 040.py script.
    """
    verts: List[List[float]] = []
    leaf_faces: List[List[int]] = []
    wood_faces: List[List[int]] = []

    # Save leaf and wood meshes to files next to scene_file, and return the path
    leaf_mesh_path = scene_file.replace('.obj', '_leaf.obj')
    wood_mesh_path = scene_file.replace('.obj', '_wood.obj')

    if os.path.exists(leaf_mesh_path):
        print(f"Leaf mesh already exists at {leaf_mesh_path}. Loading from file.")
        leaf_mesh = pv.read(leaf_mesh_path)
    if os.path.exists(wood_mesh_path):
        print(f"Wood mesh already exists at {wood_mesh_path}. Loading from file.")
        wood_mesh = pv.read(wood_mesh_path)
    if not os.path.exists(leaf_mesh_path) and not os.path.exists(wood_mesh_path):
        current_tag = ""
        with Path(scene_file).open('r', errors="ignore") as f:
            for line in f:
                if line.startswith("v "):
                    verts.append([float(coord) for coord in line.split()[1:4]])
                elif line.startswith(("g ", "o ")):
                    # Reset current tag for new group or object
                    current_tag = line[2:].strip().lower()
                elif line.startswith("f "):
                    face = [int(tok.split("/")[0]) - 1 for tok in line.split()[1:]]
                    if any(key in current_tag for key in leaf_keys):
                        leaf_faces.append(face)
                    elif any(key in current_tag for key in wood_keys):
                        wood_faces.append(face)

        verts = np.asarray(verts, dtype=np.float64)
        leaf_mesh = _faces_to_poly(verts, leaf_faces)
        wood_mesh = _faces_to_poly(verts, wood_faces)       

        if leaf_mesh is not None:
            print(f"Saving leaf mesh to {leaf_mesh_path}.")
            leaf_mesh.save(leaf_mesh_path)
        else:
            print(f"No leaf mesh found in {scene_file}. Leaf mesh will not be saved.")
            leaf_mesh_path = ""
        if wood_mesh is not None:
            print(f"Saving wood mesh to {wood_mesh_path}.")
            wood_mesh.save(wood_mesh_path)
        else:
            print(f"No wood mesh found in {scene_file}. Wood mesh will not be saved.")
            wood_mesh_path = ""

    scene_mesh = pv.read(scene_file)
    bounds = scene_mesh.bounds

    
    return leaf_mesh_path, wood_mesh_path, bounds, leaf_mesh, wood_mesh

def generate_unique_id(center: np.ndarray, voxel_size: float) -> str:
    """
    Generate a unique ID for a voxel center based on its coordinates and voxel size.
    """
    x, y, z = center
    vs = int(voxel_size * 73)
    return f"{int(x*10 / voxel_size)}_{int(y*10 / voxel_size)}_{int(z*10 / voxel_size)}"

def p_tri_idx(tri, vertices, voxel_size, min_bound):
                """
                Wrapper function for processing a triangle in parallel.
                """
                return process_triangle(tri, vertices, voxel_size, min_bound)

def process_triangle(tri, vertices, voxel_size, min_bound):
    """
    Process a triangle to determine its voxel indices and occupied voxels.
    """
    tri_vertices = vertices[tri]
    tri_min = np.min(tri_vertices, axis=0)
    tri_max = np.max(tri_vertices, axis=0)

    min_idx = tuple(np.floor((tri_min - min_bound) / voxel_size).astype(int))
    max_idx = tuple(np.floor((tri_max - min_bound) / voxel_size).astype(int))

    occupied_voxels = set()
    for i in range(min_idx[0], max_idx[0] + 1):
        for j in range(min_idx[1], max_idx[1] + 1):
            for k in range(min_idx[2], max_idx[2] + 1):
                voxel_idx = (i, j, k)
                occupied_voxels.add(voxel_idx)

    return occupied_voxels

def generate_voxel_centers(voxel_size, leaf_mesh=None, wood_mesh=None):
    """
    Placeholder function to generate voxel centers.
    Replace this with actual logic from your 040.py script.
    """
    # Generate voxel grid for the combined plot bounds of leaf_mesh and wood_mesh
    if leaf_mesh is not None:
        min_bound = leaf_mesh.bounds[0:3]  # Get the minimum bounds of the leaf mesh
        min_bound = min_bound if wood_mesh is None else np.minimum(min_bound, wood_mesh.bounds[0:3])

        vertices = leaf_mesh.points
        vertices = vertices - min_bound  # Shift vertices to start from the minimum bound

        voxel_indices = np.floor(vertices / voxel_size).astype(int)
        occupied_voxels = np.unique(voxel_indices, axis=0)

        voxel_centers = min_bound + (occupied_voxels + 0.5) * voxel_size
        coords = ((voxel_centers * 11 + voxel_size * 73)*13).astype(int)
        voxel_ids = np.char.add(
            np.char.add(coords[:, 0].astype(str), '_'),
            np.char.add(coords[:, 1].astype(str), '_')
        )
        voxel_ids = np.char.add(voxel_ids, coords[:, 2].astype(str))

    else:
        raise ValueError("No leaf mesh provided. Cannot generate voxel centers without leaf mesh data.")
    
    return voxel_centers, voxel_ids

def split_into_batches(centers, batch_size):
    """
    Split valid_voxels (dictionary) into batches of a given size.
    Returns a list of dictionaries, each containing up to batch_size key-value pairs from valid_voxels.
    """
    # keys = list(valid_voxels.keys())
    # batches = []
    # for i in range(0, len(keys), batch_size):
    #     batch_keys = keys[i:i + batch_size]
    #     batch = {k: valid_voxels[k] for k in batch_keys}
    #     batches.append(batch)
    batches = [centers[i:i + batch_size] for i in range(0, len(centers), batch_size)]
    return batches

def write_csv(csv_path, scene_file, wood_volume_file, voxel_sizes, ray_spacing, cross_section_area, available_cpus, array_batch_size, angles, leaf_keys=["leaf"], wood_keys=["wood"]):
    """
    Write voxel batch metadata to a CSV file.
    """
    # Load leaf and wood meshes
    # This function will save the leaf and wood meshes separately and return their paths
    # This will enable the process_voxel_batch.py to access the files without reprocessing
    print(f"Loading and splitting scene file {scene_file} into leaf and wood meshes.")
    leaf_mesh_file, wood_mesh_file, bounds, leaf_mesh, wood_mesh = load_and_split_by_group(scene_file, leaf_keys, wood_keys)

    files = {
        'leaf_mesh_file': leaf_mesh_file,
        'wood_mesh_file': wood_mesh_file,
        'wood_volume_file': wood_volume_file if wood_volume_file else "",
    }

    memory_per_cpu = 5 # Tune this value based on your system's memory usage per CPU
    max_cpu_batches = max(1, int((16384/array_batch_size) // (available_cpus * memory_per_cpu)))  # Maximum memory per CPU in GB
    voxel_batch_size = max_cpu_batches * available_cpus  # Total batch size based on available CPUs and memory 
    min_mem_req = voxel_batch_size * memory_per_cpu  # Estimate memory requirement in GB

    
    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['files', 'voxel_size', 'lambda_1', 'ray_spacing', 'voxel_ids', 'voxel_centers', 'angles']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for voxel_size in voxel_sizes:
            print(f"Initialising voxel size: {voxel_size}m")

            voxel_centers, voxel_idxs = generate_voxel_centers(voxel_size, leaf_mesh, wood_mesh)
            print(f"Generated {len(voxel_centers)} valid voxel centers for voxel size {voxel_size}m.")
            
            voxel_centers_batches = [voxel_centers[i:i + voxel_batch_size] for i in range(0, len(voxel_centers), voxel_batch_size)]
            voxel_ids_batches = [voxel_idxs[i:i + voxel_batch_size] for i in range(0, len(voxel_idxs), voxel_batch_size)]
            print(f"Split voxel centers into {len(voxel_centers_batches)} batches of size {voxel_batch_size}.")
            print(f"Split voxel IDs into {len(voxel_ids_batches)} batches of size {voxel_batch_size}.")

            lambda_1 = cross_section_area / (voxel_size ** 3)

            for center_batch, id_batch in zip(voxel_centers_batches, voxel_ids_batches):   
                # Write each batch to the CSV file
                writer.writerow({
                    'files': json.dumps(files),
                    'voxel_size': voxel_size,
                    'lambda_1': lambda_1,
                    'ray_spacing': ray_spacing,
                    'voxel_ids': json.dumps(id_batch.tolist()),  # Convert IDs to list for JSON serialization
                    'voxel_centers': json.dumps(center_batch.tolist()),  # Convert centers to list for JSON serialization
                    'angles': json.dumps(angles)  
                })
            print(f"Wrote {len(voxel_centers_batches)} batches for voxel size {voxel_size}m to CSV.")

    return min_mem_req


def write_slurm_header(log_path, slurm_script_name, available_cpus, min_req_mem, time_per_batch, container_path, num_batches=100):
    """
    Write the SLURM header for the job script.
    """

    # Convert time_per_voxel from minutes to HH:MM:SS format
    days = time_per_batch // (60 * 24)
    time_per_batch %= (60 * 24)
    hours = time_per_batch // 60
    minutes = time_per_batch % 60

    time_string = f"{int(days):02d}-{int(hours):02d}:{int(minutes):02d}:00" if days > 0 else f"{int(hours):02d}:{int(minutes):02d}:00"

    slurm_header = f"""#!/bin/bash
#SBATCH --job-name={slurm_script_name}
#SBATCH --output={os.path.join(log_path, f'{slurm_script_name}.out')}
#SBATCH --error={os.path.join(log_path, f'{slurm_script_name}.err')}
#SBATCH --mem={min_req_mem}G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={available_cpus}
#SBATCH --time={time_string}
#SBATCH --partition=general
#SBATCH --account=a_l2l
#SBATCH --array=0-{num_batches-1}

"""
    
    ###OLD###
    ## Load necessary modules
# module load miniconda3
# source $EBROOTMINICONDA3/etc/profile.d/conda.sh
# conda activate {container_path}

    return slurm_header

def write_slurm_scripts(csv_path, process_path, available_cpus, min_req_mem, time_per_batch, container_path, batch_size):
    """
    Write SLURM scripts for each batch in the CSV file.
    """
    with open(csv_path, 'r') as csvfile:
        # Read the CSV file and create SLURM scripts for each batch
        project_name = os.path.basename(csv_path).split('_voxel_batches')[0]
        log_path = os.path.join(os.path.dirname(csv_path), 'logs')
        reader = csv.DictReader(csvfile)

        # Read all rows from the CSV
        csv.field_size_limit(10**7)
        rows = list(reader)
        # Split rows into chunks based on available_
        batch_chunks = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]

        # Initialise list of slurm scripts to return
        slurm_scripts = []

        log_path = os.path.join(os.path.dirname(csv_path), 'logs')
        if not os.path.exists(log_path):
            os.makedirs(log_path, exist_ok=True)

        for batch_index, batch in enumerate(batch_chunks):
            slurm_script_path = os.path.join(os.path.dirname(csv_path), f'slurm_{project_name}_batch_{batch_index}.sh')
            with open(slurm_script_path, 'w') as slurm_file:
                num_batches = len(batch)
                slurm_script_name = f"{project_name}_batch_{batch_index}"
                slurm_file.write(write_slurm_header(log_path, slurm_script_name, available_cpus, min_req_mem, time_per_batch, container_path, num_batches))

                start_index = batch_index * batch_size
                ### OLD ###
                # slurm_file.write(f"python {process_path} {csv_path} --index $(({start_index} + $SLURM_ARRAY_TASK_ID)) --log_file {os.path.join(log_path, str(start_index))}_$SLURM_ARRAY_TASK_ID.log\n")
                slurm_file.write(f"apptainer exec {container_path} python {process_path} {csv_path} --index $(({start_index} + $SLURM_ARRAY_TASK_ID)) --log_file {os.path.join(log_path, str(start_index))}_$SLURM_ARRAY_TASK_ID.log\n")

            slurm_scripts.append(slurm_script_path)

    return slurm_scripts

def submit_slurm_scripts(slurm_scripts, time_per_voxel_min):
    """
    Write controller SLURM scripts, each with a max walltime of 14 days.
    Each controller script submits a subset of SLURM scripts sequentially.
    Returns a list of controller script paths.
    """
    max_days = 14
    max_minutes = max_days * 24 * 60  # 20160 minutes

    # Calculate how many batches fit in one controller script
    batches_per_controller = max(int(max_minutes // (time_per_voxel_min * 1.25)), 1)
    controller_scripts = []

    for ctrl_idx, start in enumerate(range(0, len(slurm_scripts), batches_per_controller)):
        end = min(start + batches_per_controller, len(slurm_scripts))
        scripts_chunk = slurm_scripts[start:end]
        num_batches = len(scripts_chunk)
        total_minutes = int(time_per_voxel_min * num_batches * 1.25)
        days = total_minutes // (60 * 24)
        total_minutes -= days * 60 * 24
        hours = total_minutes // 60
        minutes = total_minutes % 60
        time_string = f"{days:02d}-{hours:02d}:{minutes:02d}:00" if days > 0 else f"{hours:02d}:{minutes:02d}:00"

        controller_script = os.path.join(
            os.path.dirname(slurm_scripts[0]),
            f"slurm_controller_{ctrl_idx}.sh"
        )
        with open(controller_script, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"#SBATCH --job-name=slurm_controller_{ctrl_idx}\n")
            f.write(f"#SBATCH --output={controller_script}.out\n")
            f.write(f"#SBATCH --error={controller_script}.err\n")
            f.write(f"#SBATCH --mem=256M\n")
            f.write(f"#SBATCH --cpus-per-task=1\n")
            f.write(f"#SBATCH --time={time_string}\n")
            f.write(f"#SBATCH --partition=general\n")
            f.write(f"#SBATCH --account=a_l2l\n\n")
            f.write("echo 'Submitting SLURM scripts sequentially...'\n")
            for i, script in enumerate(scripts_chunk):
                f.write(f"echo 'Submitting {script}'\n")
                f.write(f"jobid=$(sbatch {script} | awk '{{print $4}}')\n")
                f.write("echo \"Submitted batch as job $jobid. Waiting for it to finish...\"\n")
                f.write("sacct -j $jobid --format=State --noheader | grep -qE 'COMPLETED|FAILED|CANCELLED|TIMEOUT'\n")
                f.write("while ! sacct -j $jobid --format=State --noheader | grep -qE 'COMPLETED|FAILED|CANCELLED|TIMEOUT'; do\n")
                f.write("  sleep 60\n")
                f.write("done\n")
            f.write("echo 'All SLURM scripts in this controller submitted and completed.'\n")
        os.chmod(controller_script, 0o755)
        print(f"Controller script written to {controller_script} for batches {start} to {end-1}.")
        controller_scripts.append(controller_script)

    return controller_scripts



def main(scene_file, wood_volume_file, voxel_sizes, ray_spacing, cross_section_area, available_cpus, available_memory, time_per_voxel, csv_path, slurm_path, angles, leaf_keys, wood_keys, container_path):
    """
    Main function to generate the voxel batch CSV.
    """
    max_total_cpus = 1536
    array_batch_size = min(1000, max_total_cpus // available_cpus)

    # Calculate estimated time per batch (minutes)
    time_per_batch = int(math.ceil((time_per_voxel * array_batch_size) / 60))
    print(f"Estimated time per batch: {time_per_batch}.")
    
    # Write the CSV file for indexed process batches
    print(f"Writing CSV to {csv_path} with batch size {array_batch_size}.")
    min_req_mem = write_csv(csv_path, scene_file, wood_volume_file, voxel_sizes, ray_spacing, cross_section_area, available_cpus, array_batch_size, angles, leaf_keys, wood_keys)

    # Check csv was created
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file {csv_path} was not created successfully.")

    # Write the slurm scripts used to run each process as an array job
    process_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "process_voxel_batch.py")
    if not os.path.exists(process_path):
        raise FileNotFoundError(f"Process script {process_path} does not exist. Please ensure it is in the same directory as this script.")
    
    # Write the slurm scripts
    print(f"Writing SLURM scripts to {slurm_path} for processing voxel batches.")
    slurm_scripts = write_slurm_scripts(csv_path, process_path, available_cpus, min_req_mem, time_per_batch, container_path, array_batch_size)
    print(f"{len(slurm_scripts)} SLURM scripts created in {slurm_path} for processing voxel batches.")

    # Create the controller script to submit all SLURM scripts
    controller_script = submit_slurm_scripts(slurm_scripts, time_per_voxel)
    # os.system(f"bash {controller_script}")

    # Queue the first SLURM script
    # first_slurm_script = slurm_scripts[0]
    # print(f"Queuing the first SLURM script {first_slurm_script} for processing...")
    # os.system(f"sbatch {first_slurm_script}")

    print("Initialization complete. Check the logs for progress.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Generate voxel batch CSV for HPC processing.")
    parser.add_argument("scene_file", type=str, help="A merged .obj scene file to derive reference statistics from.")
    parser.add_argument("--wood_volume_file", type=str, default="", help="A wood volume .txt file to derive reference statistics from.")
    parser.add_argument("--cpus", type=int, default=32, help="Number of CPUs to allocate for the job.")
    parser.add_argument("--mem", type=int, default=350, help="Memory limit in GB for the job.")
    parser.add_argument("--time_per_voxel", type=int, default=20, help="Estimated time in seconds for each voxel to complete.")
    parser.add_argument("--voxel_sizes", type=float, nargs="+", default=[2.0, 1.0, 0.5], help="List of voxel sizes")
    parser.add_argument("--ray_spacing", type=float, default=0.005, help="Ray spacing")
    parser.add_argument("--cross_section_area", type=float, default=0.003582, help="Cross section area")
    parser.add_argument("--angles", type=float, nargs="+", default=[0.000001, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0,  89.999999], help="List of angles for processing")
    parser.add_argument("--leaf_keys", type=str, nargs="+", default=["leaf", "leaves"], help="List of leaf mesh keys to identify in the scene file.")
    parser.add_argument("--wood_keys", type=str, nargs="+", default=["wood", "bark", "stem", "trunk"], help="List of wood mesh keys to identify in the scene file.")
    parser.add_argument("--container_path", type=str, default="base", help="Conda environment to activate for processing.")

    args = parser.parse_args()

    scene_file = args.scene_file
    wood_volume_file = args.wood_volume_file
    voxel_sizes = args.voxel_sizes
    ray_spacing = args.ray_spacing
    cross_section_area = args.cross_section_area
    available_cpus = args.cpus
    available_memory = args.mem
    time_per_voxel = args.time_per_voxel
    angles = args.angles
    leaf_keys = args.leaf_keys
    wood_keys = args.wood_keys
    container_path = args.container_path
    
    # Derive csv path from scene file
    slurm_path = os.path.join(os.path.dirname(scene_file), "slurm_scripts")
    if not os.path.exists(slurm_path):
        os.makedirs(slurm_path, exist_ok=True)
    else:
        # Remove existing slurm scripts if they exist
        for file in os.listdir(slurm_path):
            if file.endswith(".sh"):
                os.remove(os.path.join(slurm_path, file))
    csv_path = os.path.join(slurm_path, f"{os.path.basename(scene_file)}_voxel_batches.csv")

    # Check scene_file exists
    if not os.path.exists(scene_file):
        raise FileNotFoundError(f"The scene file {scene_file} does not exist.")
    
    # Check if scene_file is on the archival system
    if scene_file.startswith('/QRISdata/'):
        raise ValueError("The scene file is on the archival system. Please move it to a local or scratch directory before running this script.")
    
    # Clear existing CSV file if it exists
    if os.path.exists(csv_path):
        os.remove(csv_path)

    print(f"Generating voxel batch CSV at {csv_path} with voxel sizes {voxel_sizes}, ray spacing {ray_spacing}, and cross section area {cross_section_area}.")
    main(scene_file, wood_volume_file, voxel_sizes, ray_spacing, cross_section_area, available_cpus, available_memory, time_per_voxel, csv_path, slurm_path, angles, leaf_keys, wood_keys, container_path)