from gvxrPython3 import gvxr

import numpy as np
import astra
import tifffile

from xray_util import *

################################################################################
display = True

fiber_path = "./tex-ray/fiber.stl"

fiber_elements = [6]

fiber_ratios = [1.0]

fiber_density = 1.8

matrix_path = "./tex-ray/matrix.stl"

matrix_elements = [6, 1, 17, 8]

matrix_ratios = [0.404, 0.481, 0.019, 0.096]

matrix_density = 1.0

output_path = "./tex-ray/reconstruction.tif"

sample_length_unit = "mm"

scanner_length_unit = "cm"

energy_unit = "keV"

detector_pixel_size = 0.008

distance_source_origin = 10

distance_origin_detector = 6

detector_rows = 100

detector_columns = 640

x_ray_energies = [80]

x_ray_counts = [100]

number_of_projections = 100

scanning_angle = 360

projection_angles = np.linspace(0,np.deg2rad(scanning_angle),
                                num=number_of_projections)

threshold = 0.000000001

reconstruction_algorithm = 'FDK_CUDA'
################################################################################

gvxr.createOpenGLContext()

set_up_detector(distance_origin_detector, detector_columns, detector_rows,
                detector_pixel_size, length_unit=scanner_length_unit)
set_up_xray_source(distance_source_origin, -1, x_ray_energies, x_ray_counts, 
                   length_unit=scanner_length_unit, energy_unit=energy_unit)
set_up_sample(fiber_path, fiber_elements, fiber_ratios, fiber_density,
              matrix_path, matrix_elements, matrix_ratios, matrix_density,
              length_unit=sample_length_unit)
raw_projections = perform_tomographic_scan(number_of_projections,
                                           scanning_angle, display=display)
flat_field_image = measure_flat_field()
dark_field_image = measure_dark_field()
corrected_projections = perform_flat_field_correction(raw_projections,
                                                      flat_field_image,
                                                      dark_field_image)
neg_log_projections = neg_log_transform(corrected_projections, threshold)

# Reformat the projections into a set of sinograms on the form that ASTRA needs.
sinograms = np.swapaxes(neg_log_projections, 0, 1)
sinograms = np.array(sinograms).astype(np.single)

# Not needed anymore so we remove... It is necessary because emory might run out
# if we have many detector pixels and a large number of projections.
raw_projections = None
flat_field_image = None
dark_field_image = None
corrected_projections = None 

scale_factor = compute_astra_scale_factor(distance_source_origin,
                                          distance_origin_detector,
                                          detector_pixel_size)
vol_geo = astra.create_vol_geom(detector_columns, detector_columns,
                                 detector_rows)
proj_geo = astra.creators.create_proj_geom('cone',
                                           detector_pixel_size*scale_factor,
                                           detector_pixel_size*scale_factor,
                                           detector_rows,
                                           detector_columns,
                                           projection_angles,
                                           distance_source_origin*scale_factor,
                                           distance_origin_detector*scale_factor
                                        )
proj_id = astra.data3d.create('-sino', proj_geo, data=sinograms)
rec_id = astra.data3d.create('-vol', vol_geo, data=0)
alg_cfg = astra.astra_dict(reconstruction_algorithm)
alg_cfg['ReconstructionDataId'] = rec_id
alg_cfg['ProjectionDataId'] = proj_id
alg_id = astra.algorithm.create(alg_cfg)
astra.algorithm.run(alg_id)
reconstruction = astra.data3d.get(rec_id)

# Remove eventual erroneous negative values.
reconstruction[reconstruction < 0] = 0

# Rescale to get attenuation coefficient in scanner_length_unit^-1.
reconstruction /= detector_pixel_size * distance_source_origin/(
    distance_source_origin + distance_origin_detector)

tifffile.imwrite(output_path, reconstruction)
