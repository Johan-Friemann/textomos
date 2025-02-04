import sys
import json
import tifffile
from x_ray_simulation import generate_sinograms
from tomographic_reconstruction import perform_tomographic_reconstruction
from segmentation import segment_reconstruction
from hdf5_utils import save_data, get_database_shape

"""
This is a utility file that is called from batch_run.sh. It's purpose is to run
gvxr, reconstruct, segment, and save data to database.

This construction is necessary due to strange behavior when running gvxr many
times within the same python program. In order to get a clean simulator the 
gvxr singleton needs to  be created from scratch. This could be achieved through
launching a new process withing a python script. However, if new processes
that executes the entire textomos pipeline are launched multiple times within a
script, a memory leak appears. We therefore launch the script from bash instead.

The memory leak could be a bug in gvxr or an issue with textomos...
"""

if __name__ == "__main__":
    config_path = sys.argv[1]
    database_path = sys.argv[2]
    chunk_size = sys.argv[3]
    generate_until = sys.argv[4]

    with open(config_path) as f:
        config_dict = json.load(f)
    sinograms = generate_sinograms(config_dict)

    sinograms = generate_sinograms(config_dict)
    reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)
    tifffile.imwrite(
        config_dict["reconstruction_output_path"],
        reconstruction,
    )
    del sinograms, reconstruction

    segmentation = segment_reconstruction(config_dict)
    tifffile.imwrite(
        config_dict["segmentation_output_path"],
        segmentation,
    )
    del segmentation

    save_data(database_path, config_dict, chunk_size=chunk_size)

    num_datapoints = get_database_shape(database_path)[0]

    if num_datapoints == generate_until:
        with open("/textomos/input/finished", "wb") as f:
            f.write("FINISHED")
