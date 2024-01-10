import tifffile
from x_ray.x_ray import *
from textile_generation.textile_generation import *

################################################################################
config_dict = {}

config_dict["weft_path"] = "./tex-ray/meshes/weft.stl"

config_dict["warp_path"] = "./tex-ray/meshes/warp.stl"

config_dict["matrix_path"] = "./tex-ray/meshes/matrix.stl"

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

config_dict["display"] = True

config_dict["fiber_path"] = "./tex-ray/meshes/fiber.stl"

config_dict["fiber_elements"] = [6]

config_dict["fiber_ratios"] = [1.0]

config_dict["fiber_density"] = 1.8

config_dict["matrix_elements"] = [6, 1, 17, 8]

config_dict["matrix_ratios"] = [0.404, 0.481, 0.019, 0.096]

config_dict["matrix_density"] = 1.0

config_dict["output_path"] = "./tex-ray/reconstructions/reconstruction.tif"

config_dict["sample_length_unit"] = "mm"

config_dict["scanner_length_unit"] = "cm"

config_dict["energy_unit"] = "keV"

config_dict["detector_pixel_size"] = 0.008

config_dict["scanning_angle"] = 360

config_dict["threshold"] = 0.000000001

config_dict["reconstruction_algorithm"] = "FDK_CUDA"
################################################################################

# Uncomment when weft-warp has been implemented in x_ray
#generate_unit_cell(config_dict)

sinograms = generate_sinograms(config_dict)
reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)

tifffile.imwrite(config_dict["output_path"], reconstruction)
