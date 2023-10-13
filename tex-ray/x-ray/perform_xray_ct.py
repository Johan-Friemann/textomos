from gvxrPython3 import gvxr

import numpy as np
import astra
import tifffile

from xray_util import *

################################################################################
detector_pixel_size = 0.08/1000

distance_source_origin = 10/100

distance_origin_detector = 6/100

detector_rows = 100

detector_columns = 640

number_of_projections = 100

scanning_angle = 360

threshold = 0.000000001
################################################################################

gvxr.createOpenGLContext()

set_up_detector(distance_origin_detector, detector_columns, detector_rows,
                detector_pixel_size, length_unit="m")
set_up_xray_source(distance_source_origin, -1, [80], [100])

set_up_sample("./tex-ray/fiber.stl" , [6], [1.0], 1.8, "./tex-ray/matrix.stl",
              [6, 1, 17, 8], [0.404, 0.481, 0.019, 0.096], 1.0,
              length_unit="mm")
raw_projections = perform_tomographic_scan(number_of_projections,
                                           scanning_angle, display=True)
flat_field_image = measure_flat_field()
dark_field_image = measure_dark_field()
corrected_projections = perform_flat_field_correction(raw_projections,
                                                      flat_field_image,
                                                      dark_field_image)
neg_log_projections = neg_log_transform(corrected_projections, threshold)
################################################################################

# Make sure the data is in single-precision floating-point numbers
neg_log_projections = np.array(neg_log_projections).astype(np.single)

# Reformat the projections into a set of sinograms
sinograms = np.swapaxes(neg_log_projections, 0, 1)
sinograms = np.array(sinograms).astype(np.single)

corrected_projections = None # Not needed anymore

angles = np.linspace(0,2*np.pi,num=number_of_projections)
scale_factor = 1 / (detector_pixel_size * distance_source_origin/(distance_source_origin + distance_origin_detector))
vol_geom = astra.create_vol_geom(detector_columns, detector_columns, detector_rows)
proj_geom = astra.creators.create_proj_geom('cone', detector_pixel_size*scale_factor, detector_pixel_size*scale_factor, detector_rows, detector_columns, angles, distance_source_origin*scale_factor, distance_origin_detector*scale_factor)

proj_id = astra.data3d.create('-sino', proj_geom, data=sinograms)

rec_id = astra.data3d.create('-vol', vol_geom, data=0)

alg_cfg = astra.astra_dict('FDK_CUDA')
alg_cfg['ReconstructionDataId'] = rec_id
alg_cfg['ProjectionDataId'] = proj_id

alg_id = astra.algorithm.create(alg_cfg)
print('Reconstructing... ')
astra.algorithm.run(alg_id)
print('done')
reconstruction = astra.data3d.get(rec_id)

reconstruction[reconstruction < 0] = 0

# Rescale to get attenuation per cm
reconstruction /= detector_pixel_size * distance_source_origin/(distance_source_origin + distance_origin_detector)
reconstruction /= 100


tifffile.imwrite('./tex-ray/test.tif', reconstruction)
