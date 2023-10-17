from gvxrPython3 import gvxr

import numpy as np
import astra
import tifffile

from xray_util import *


def generate_sinograms(config_dict):
    """ Perform an X-Ray CT scan of a sample and return the sinograms.

    Args:
        config_dict (dictionary): A dictionary of options.

    Keyword args:
        -
    Returns:
        sinograms (numpy array[float]): The measured CT sinograms. The array has
                                        the shape (detector_rows,
                                        number_of_projections, detector_columns)
    """
    gvxr.createOpenGLContext()

    set_up_detector(
        config_dict['distance_origin_detector'],
        config_dict['detector_columns'],
        config_dict['detector_rows'],
        config_dict['detector_pixel_size'],
        length_unit=config_dict['scanner_length_unit']
    )
    
    set_up_xray_source(
        config_dict['distance_source_origin'],
        -1,
        config_dict['x_ray_energies'],
        config_dict['x_ray_counts'], 
        length_unit=config_dict['scanner_length_unit'],
        energy_unit=config_dict['energy_unit']
    )
    set_up_sample(
        config_dict['fiber_path'],
        config_dict['fiber_elements'],
        config_dict['fiber_ratios'],
        config_dict['fiber_density'],
        config_dict['matrix_path'],
        config_dict['matrix_elements'],
        config_dict['matrix_ratios'],
        config_dict['matrix_density'],
        length_unit=config_dict['sample_length_unit']
    )
    raw_projections = perform_tomographic_scan(
        config_dict['number_of_projections'],
        config_dict['scanning_angle'],
        display=config_dict['display']
    )
    flat_field_image = measure_flat_field()
    dark_field_image = measure_dark_field()
    corrected_projections = perform_flat_field_correction(raw_projections,
                                                          flat_field_image,
                                                          dark_field_image)
    neg_log_projections = neg_log_transform(
        corrected_projections, config_dict['threshold']
    )

    # Reformat the projections into a set of sinograms on the ASTRA form.
    sinograms = np.swapaxes(neg_log_projections, 0, 1)
    sinograms = np.array(sinograms).astype(np.single)

    return sinograms


def perform_tomographic_reconstruction(sinograms, config_dict):
    """Perform a tomographic reconstruction with ASTRA given a set of sinograms.
    
    Args:
        sinograms (numpy array[float]): The measured CT sinograms. The array has
                                        the shape (detector_rows,
                                        number_of_projections, detector_columns)
        config_dict (dictionary): A dictionary of options.

    Keyword args:
        -

    Returns:
        reconstruction (numpy array[float]): The reconstructed sample. The array
                                             has the shape (detector_rows,
                                             detector_columns, detector_columns)

    """

    projection_angles = np.linspace(
        0, config_dict['scanning_angle'], config_dict['number_of_projections']
    )

    scale_factor = compute_astra_scale_factor(
        config_dict['distance_source_origin'],
        config_dict['distance_origin_detector'],
        config_dict['detector_pixel_size']
    )
    
    vol_geo = astra.create_vol_geom(
        config_dict['detector_columns'],
        config_dict['detector_columns'],
        config_dict['detector_rows']
    )
    
    proj_geo = astra.creators.create_proj_geom(
        'cone',
        config_dict['detector_pixel_size']*scale_factor,
        config_dict['detector_pixel_size']*scale_factor,
        config_dict['detector_rows'],
        config_dict['detector_columns'],
        projection_angles,
        config_dict['distance_source_origin']*scale_factor,
        config_dict['distance_origin_detector']*scale_factor
    )
    proj_id = astra.data3d.create('-sino', proj_geo, data=sinograms)
    rec_id = astra.data3d.create('-vol', vol_geo, data=0)
    alg_cfg = astra.astra_dict(config_dict['reconstruction_algorithm'])
    alg_cfg['ReconstructionDataId'] = rec_id
    alg_cfg['ProjectionDataId'] = proj_id
    alg_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(alg_id)
    reconstruction = astra.data3d.get(rec_id)

    # Remove eventual erroneous negative values.
    reconstruction[reconstruction < 0] = 0

    # Rescale to get attenuation coefficient in scanner_length_unit^-1.
    reconstruction /= config_dict['detector_pixel_size'] * \
                      config_dict['distance_source_origin'] / (
                      config_dict['distance_source_origin'] +
                      config_dict['distance_origin_detector']
                    )
    
    return reconstruction


################################################################################
config_dict = {}

config_dict['display'] = True

config_dict['fiber_path'] = "./tex-ray/fiber.stl"

config_dict['fiber_elements'] = [6]

config_dict['fiber_ratios'] = [1.0]

config_dict['fiber_density'] = 1.8

config_dict['matrix_path'] = "./tex-ray/matrix.stl"

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
