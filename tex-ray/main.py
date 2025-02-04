import tifffile
import sys
import json
from x_ray_simulation import generate_sinograms
from tomographic_reconstruction import perform_tomographic_reconstruction
from textile_generation import generate_woven_composite_sample
from segmentation import segment_reconstruction
from zeiss_xradia_410_versa import read_txrm_scan_data, read_txm_scan_data


if __name__ == "__main__":
    if len(sys.argv) == 1:
        input_path="./textomos/input/default_input.json"
    else:
        input_path = sys.argv[1]
    with open(input_path) as f:
        config_dict = json.load(f)

    generate_woven_composite_sample(config_dict)

    sinograms = generate_sinograms(config_dict)
    reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)
    tifffile.imwrite(config_dict["reconstruction_output_path"], reconstruction)
    del sinograms, reconstruction

    segmentation = segment_reconstruction(config_dict)
    tifffile.imwrite(config_dict["segmentation_output_path"], segmentation)
    del segmentation
