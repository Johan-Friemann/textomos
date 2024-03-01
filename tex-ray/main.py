import tifffile
from x_ray import generate_sinograms
from tomographic_reconstruction import perform_tomographic_reconstruction
from textile_generation import generate_unit_cell
from zeiss_xradia_410_versa import read_txrm_scan_data, read_txm_scan_data

################################################################################
config_dict = {}

config_dict["weft_path"] = "./tex-ray/meshes/weft.stl"

config_dict["warp_path"] = "./tex-ray/meshes/warp.stl"

config_dict["matrix_path"] = "./tex-ray/meshes/matrix.stl"

config_dict["cut_matrix"] = True

config_dict["unit_cell_weft_length"] = 21.5

config_dict["unit_cell_warp_length"] = 6.5

config_dict["unit_cell_thickness"] = 4.0

config_dict["weft_yarns_per_layer"] = 4

config_dict["warp_yarns_per_layer"] = 8

config_dict["number_of_yarn_layers"] = 6

config_dict["yarn_width_to_spacing_ratio"] = 0.9

config_dict["weft_to_warp_ratio"] = 0.4

config_dict["weave_pattern"] = [
    [1, 0, 1],
    [3, 0, -1],
    [5, 1, 1],
    [7, 1, -1],
    [3, 2, 1],
    [1, 2, -1],
    [7, 3, 1],
    [5, 3, -1],
]

config_dict["deform"] = [
    10.0,
    10.0,
    10.0,
    0.1,
    0.1,
    0.1,
    5.0,
    5.0,
    10.0,
    0.05,
    0.05,
    0.05,
]

config_dict["weft_elements"] = [6]

config_dict["weft_ratios"] = [1.0]

config_dict["weft_density"] = 1.8

config_dict["warp_elements"] = [6]

config_dict["warp_ratios"] = [1.0]

config_dict["warp_density"] = 1.8

config_dict["matrix_elements"] = [6, 1, 17, 8]

config_dict["matrix_ratios"] = [0.404, 0.481, 0.019, 0.096]

config_dict["matrix_density"] = 1.0

config_dict["offset"] = [0.0, 0.0, 12.0]

config_dict["tilt"] = [-1.0, 2.0, -3.0]

config_dict["rot_axis"] = "x"

config_dict["tiling"] = [2, 3, 3]

config_dict["output_path"] = "./tex-ray/reconstructions/reconstruction.tif"

config_dict["sample_length_unit"] = "mm"

config_dict["scanner_length_unit"] = "mm"

config_dict["energy_unit"] = "keV"

# 13.5 um --> 0.0135 mm --> (LFOV) --> effective 0.0375 mm
config_dict["detector_pixel_size"] = 13.5 / 0.4 / 1000.0

config_dict["distance_source_origin"] = 60.0

config_dict["distance_origin_detector"] = 80.0

config_dict["detector_rows"] = 2048

config_dict["detector_columns"] = 2048

config_dict["number_of_projections"] = 801

config_dict["scanning_angle"] = 360.0

config_dict["threshold"] = 0.000000001

config_dict["display"] = True

config_dict["photonic_noise"] = True

config_dict["binning"] = 4

config_dict["anode_angle"] = 12.0

config_dict["tube_voltage"] = 40.0

config_dict["tube_power"] = 10.0

config_dict["filter_thickness"] = 2.0

config_dict["filter_material"] = "Al"

config_dict["target_material"] = "W"

config_dict["exposure_time"] = 5.0

config_dict["energy_bin_width"] = 0.5

config_dict["num_reference"] = 20

config_dict["reconstruction_algorithm"] = "FDK_CUDA"

# Typically 1 for sim, and -1 for real scan
config_dict["sample_rotation_direction"] = 1  

################################################################################

generate_unit_cell(config_dict)

sinograms = generate_sinograms(config_dict)
reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)

tifffile.imwrite(config_dict["output_path"], reconstruction)
