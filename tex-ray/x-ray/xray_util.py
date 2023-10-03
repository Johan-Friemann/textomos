from gvxrPython3 import gvxr

def set_up_scanner_geometry(distance_source_origin, distance_origin_detector,
                            detector_columns      , detector_rows           ,
                            detector_pixel_size   , unit="m"):
    """Set up the gVirtualXRay scanner geometry. The reconstruction volume is
       located at x=y=z=0.

    Args:
        distance_source_origin (float): The distance from X-Ray source to the
                                        origin of the reconstruction area.
        distance_origin_detector (float): The distance from the origin of the
                                          reconstruction area to the X-Ray
                                          detector.
        detector_columns (int): The number of pixels in the width direction of
                                the detector.
        detector_rows (int): The number of pixels in the height direction of the
                             detector.
    Keyword args:
        unit (string): The unit of length measurement (m, cm, mm, um).
                       Default unit is m (meter).

    Returns:
        -
    
    """

    gvxr.setSourcePosition(-distance_source_origin,  0.0, 0.0, unit)
    gvxr.setDetectorPosition(distance_origin_detector, 0.0, 0.0, unit)
    gvxr.setDetectorUpVector(0, 0, 1)
    gvxr.setDetectorNumberOfPixels(detector_columns, detector_rows)
    gvxr.setDetectorPixelSize(detector_pixel_size, detector_pixel_size, unit)


def set_up_xray_source(energies=[], counts=[], unit="keV"):
    """Set up the gVirtualXRay X-Ray source. Only supports point source.
       The spectrum of the source is built by specifying photon energies and
       corresponding photon counts. If lists of length 1 are given a
       monochromatic source is set up.

    Args:
        -

    Keyword args:
        energies (list[float]): A list of X-Ray photon energies.
        counts (list[int]): A list of X-Ray photon counts.
        unit (string): The unit of photon energy (eV, keV, MeV). The default is
                       keV (kilo electronvolt).
    
    Returns:
        -
    """

    if len(energies) != len(counts):
        raise ValueError("Bad arguments: 'energies' and 'counts'" +
                         " must be of the same length!")
    gvxr.resetBeamSpectrum()
    gvxr.usePointSource()
    if len(energies) == 1:
        gvxr.setMonoChromatic(energies[0], unit, counts[0])
    else:
        for energy, count in zip(energies, counts):
            gvxr.addEnergyBinToSpectrum(energy, unit, count)
    