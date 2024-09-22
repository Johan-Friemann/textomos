import sys
import json
import tifffile
from x_ray_simulation import generate_sinograms
from tomographic_reconstruction import perform_tomographic_reconstruction
from segmentation import segment_reconstruction
from hdf5_utils import save_data

if __name__ == "__main__":
    config_path = sys.argv[1]
    database_path = sys.argv[2]
    chunk_size = sys.argv[3]

    with open(config_path) as f:
        config_dict = json.load(f)
    sinograms = generate_sinograms(config_dict)

    sinograms = generate_sinograms(config_dict)
    reconstruction = perform_tomographic_reconstruction(
        sinograms, config_dict
    )
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

    save_data(
        database_path, config_dict, chunk_size=chunk_size
    )
