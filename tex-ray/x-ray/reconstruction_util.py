import astra

"""
This file consists of helper functions to be called by the main X-Ray CT script.

The helper functions are specifically related to the X-Ray image reconstruction.
"""

def compute_astra_scale_factor(distance_source_origin, distance_origin_detector,
                               detector_pixel_size):
    """ Compute the ASTRA scale factor for conical beam geometry. 
        The ASTRA toolbox requires the reconstruction volume to be set up such
        that the reconstruction voxels have size 1. This function computes the
        scale factor that all projection geometry distances need to be re-scaled
        by. It is important that all arguments are given in the same units.

    Args:
        distance_source (float): The distance from the X-Ray source to the
                                 reconstruction volume origin.
        distance_origin_detector (float): The distance from the reconstruction
                                          volume origin to the X-Ray detector.
        detector_pixel_size (float): The X-Ray detector pixel side length.

    Keyword args:
        -
    
    Returns:
        scale_factor (float): The ASTRA scale factor.

    """
    scale_factor = 1 / (detector_pixel_size * distance_source_origin / 
        (distance_source_origin + distance_origin_detector)
    )
    return scale_factor


