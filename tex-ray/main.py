import tifffile
import json
from x_ray_simulation import generate_sinograms
from tomographic_reconstruction import perform_tomographic_reconstruction
from textile_generation import generate_unit_cell
from zeiss_xradia_410_versa import read_txrm_scan_data, read_txm_scan_data


with open("./tex-ray/input/default_input.json") as f:
    config_dict = json.load(f)

generate_unit_cell(config_dict)

sinograms = generate_sinograms(config_dict)
reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)

tifffile.imwrite(config_dict["output_path"], reconstruction)
