import tifffile
from x_ray.x_ray import *
from textile_generation.textile_generation import *

################################################################################
config_dict = {}

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

config_dict["matrix_path"] = "./tex-ray/meshes/matrix.stl"

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


Weft, Warp = create_layer2layer_unit_cell(
    config_dict["unit_cell_weft_length"],
    config_dict["unit_cell_warp_length"],
    config_dict["unit_cell_thickness"],
    config_dict["weft_yarns_per_layer"],
    config_dict["warp_yarns_per_layer"],
    config_dict["number_of_yarn_layers"],
    config_dict["yarn_width_to_spacing_ratio"],
    config_dict["weft_to_warp_ratio"],
    config_dict["weave_pattern"],
)

write_layer_to_layer_unit_cell_mesh(
    Weft,
    Warp,
    "./tex-ray/meshes/weft.stl",
    "./tex-ray/meshes/warp.stl",
    "./tex-ray/meshes/matrix.stl",
)

boolean_difference_post_processing(
    "./tex-ray/meshes/weft.stl", "./tex-ray/meshes/warp.stl"
)


sinograms = generate_sinograms(config_dict)
reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)

tifffile.imwrite(config_dict["output_path"], reconstruction)
