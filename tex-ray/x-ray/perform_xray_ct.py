from gvxrPython3 import gvxr

from xray_util import set_up_scanner_geometry, set_up_xray_source

################################################################################
detector_pixel_size = 0.08/1000

distance_source_origin = 10/100

distance_origin_detector = 10/100

detector_rows = 100

detector_columns = 640
################################################################################

gvxr.createOpenGLContext()

set_up_scanner_geometry(distance_source_origin, distance_origin_detector,
                        detector_columns      , detector_rows           ,
                        detector_pixel_size   , unit="m")
set_up_xray_source([80], [0])

