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


def set_up_xray_source(energies, counts, unit="keV"):
    """Set up the gVirtualXRay X-Ray source. Only supports point source.
       The spectrum of the source is built by specifying photon energies and
       corresponding photon counts. If lists of length 1 are given a
       monochromatic source is set up.

    Args:
        energies (list[float]): A list of X-Ray photon energies.
        counts (list[int]): A list of X-Ray photon counts.

    Keyword args:
        unit (string): The unit of photon energy (eV, keV, MeV). The default is
                       keV (kilo electronvolt).
    
    Returns:
        -
    """

    if len(energies) != len(counts):
        raise ValueError("Bad arguments: 1st argument 'energies' and 2nd " +
                         "argument 'counts' must be of the same length!")
    
    gvxr.resetBeamSpectrum()
    gvxr.usePointSource()
    if len(energies) == 1:
        gvxr.setMonoChromatic(energies[0], unit, counts[0])
    else:
        for energy, count in zip(energies, counts):
            gvxr.addEnergyBinToSpectrum(energy, unit, count)


def set_up_sample(fiber_path , fiber_elements , fiber_ratios , fiber_density ,
                  matrix_path, matrix_elements, matrix_ratios, matrix_density,
                  unit="m"):
    """Load fiber and matrix geometry and X-Ray absorption properties. 

    Args:
        fiber_geometry_path (string): The path to the fiber geometry mesh file.
        fiber_elements (list(int)): The element numbers of the constituents of
                                    the fiber material.
        fiber_ratios (list(float)): The relative amount of the constituent
                                    elements.
        fiber_density (float): The density of the fiber material in g/cm^3.
        matrix_geometry_path (string): The path to the matrix geometry mesh
                                       file.
        matrix_elements (list(int)): The element numbers of the constituents of
                                     the matrix material.
        matrix_ratios (list(float)): The relative amount of the constituent
                                     elements.
        matrix_density (float): The density of the matrix material in g/cm^3.

    Keyword args:
        unit (string): The unit of length (m, cm, mm, um).
                       Default unit is m (meter).
    
    Returns:
        -
    """

    if len(fiber_elements) != len(fiber_ratios):
        raise ValueError("Bad arguments: number of fiber ratios must agree " +
                         "with the number of elements!")
    
    if sum(fiber_ratios) != 1.0:
        raise ValueError("Bad arguments: sum of fiber ratios must be 1.0.")

    gvxr.loadMeshFile("fiber", fiber_path, unit)
    gvxr.setMixture("fiber", fiber_elements, fiber_ratios)
    gvxr.setDensity("fiber", fiber_density, "g/cm3")
    gvxr.moveToCentre("fiber")

    if len(matrix_elements) != len(matrix_ratios):
        raise ValueError("Bad arguments: number of matrix ratios must agree " +
                         "with the number of elements!")
    
    if sum(matrix_ratios) != 1.0:
        raise ValueError("Bad arguments: sum of matrix ratios must be 1.0.")

    gvxr.loadMeshFile("matrix", matrix_path, unit, False)
    gvxr.addPolygonMeshAsOuterSurface("matrix")
    gvxr.setMixture("matrix", matrix_elements, matrix_ratios)
    gvxr.setDensity("matrix", matrix_density, "g/cm3")
    gvxr.moveToCentre("matrix")
