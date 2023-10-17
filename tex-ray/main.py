import tifffile
from x_ray.x_ray import *

################################################################################
config_dict = {}

config_dict['display'] = True

config_dict['fiber_path'] = "./tex-ray/meshes/fiber.stl"

config_dict['fiber_elements'] = [6]

config_dict['fiber_ratios'] = [1.0]

config_dict['fiber_density'] = 1.8

config_dict['matrix_path'] = "./tex-ray/meshes/matrix.stl"

config_dict['matrix_elements'] = [6, 1, 17, 8]

config_dict['matrix_ratios'] = [0.404, 0.481, 0.019, 0.096]

config_dict['matrix_density'] = 1.0

config_dict['output_path'] = "./tex-ray/reconstruction.tif"

config_dict['sample_length_unit'] = "mm"

config_dict['scanner_length_unit'] = "cm"

config_dict['energy_unit'] = "keV"

config_dict['detector_pixel_size'] = 0.008

config_dict['distance_source_origin'] = 10

config_dict['distance_origin_detector'] = 6

config_dict['detector_rows'] = 100

config_dict['detector_columns'] = 640

config_dict['x_ray_energies'] = [80]

config_dict['x_ray_counts'] = [100]

config_dict['number_of_projections'] = 102

config_dict['scanning_angle'] = 360

config_dict['threshold'] = 0.000000001

config_dict['reconstruction_algorithm'] = 'FDK_CUDA'
################################################################################


sinograms = generate_sinograms(config_dict)
reconstruction = perform_tomographic_reconstruction(sinograms, config_dict)

tifffile.imwrite(config_dict['output_path'], reconstruction)