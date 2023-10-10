import numpy as np
from gvxrPython3 import gvxr

"""
This file consists of helper functions to be called by the main X-Ray CT script.

Important things to bear in mind:
    - The unit of energy in gVirtualXRay is MeV (mega electron volts), ergo
      gvxr.computeXRayImage(...) returns X-Ray images in MeV when in energy
      fluence mode. Additionally, gvxr.getUnitOfEnergy("MeV") returns 1.0.
    - The unit of length in gVirtualXRay is mm (milimeter). Therefore the
      function gvxr.getUnitOfLength("mm") returns 1.0.
    - The data extracted from the negative log of the ratio of measured
      intensity to initial intensitiy, will be in detector length units
      (in pixel units). In order to compute for example, linear attenuation,
      the extracted data needs to be rescaled by the detector pixel size.
"""

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


def perform_tomographic_scan(num_projections, scanning_angle, 
                             display=False  , integrate_energy=True):
    """Perform a tomographic scan consisting of a certain number of projections
       and sweeping a certain angle. The scan rotates the sample clockwise
       around the z-axis.

    Args:
        num_projections (int): The number of X-Ray projections.
        scanning_angle (float): The scanning angle in degrees.

    Keyword args:
        display (bool): Will display the scanning scene if true.
        integrate_energy (bool): If true the energy fluence is measured by the
                                 detector. Photon count is measured if false.

    Returns:
        raw_projections (numpy array[float]): A numpy array of all measured
                                              X-Ray projections. It has the
                                              shape (num_projections, 
                                              detector_rows, detector_columns).
    """
    raw_projections = []
    angular_step = scanning_angle/num_projections
    for angle_id in range(0, num_projections):

        # Compute an X-ray image    
        xray_image = np.array(gvxr.computeXRayImage(integrate_energy))

        # Add to the set of projections
        raw_projections.append(xray_image)

        # Update the rendering
        if display:
            gvxr.displayScene()

        # Rotate the sample
        gvxr.rotateScene(-angular_step, 0, 0, 1)

    raw_projections = np.array(raw_projections)
    return raw_projections

def measure_flat_field(integrate_energy=True):
    """Measure the flat field, i.e what the detector sees when the X-Ray source
       is on but there is no sample present. Can measure the energy fluence flat
       field, or the photon count flat field.

    Args:
        -
    
    Keyword args:
        integrate_energy (bool): If true the energy fluence is measured by the
                          detector. Photon count is measured if false.
    
    Returns:
        flat_field_image(numpy array[float]): A numpy array containing the flat
                                              field. It has the shape
                                              (detector_rows, detector_columns). 
    
    """
    detector_columns, detector_rows = gvxr.getDetectorNumberOfPixels()
    flat_field_image = np.ones((detector_rows, detector_columns))

    energy_bins = gvxr.getEnergyBins("MeV")
    photon_count_per_bin = gvxr.getPhotonCountEnergyBins()

    # If not measuring the energy fluence, set energy bins to 1 [arb units] in
    # order to return the total number of photons instead of total energy.
    if not integrate_energy:
        energy_bins = [1 for _ in range(len(energy_bins))]

    total_energy = 0.0
    for energy, count in zip(energy_bins, photon_count_per_bin):
        total_energy += energy * count
    flat_field_image *= total_energy

    return flat_field_image

def measure_dark_field(integrate_energy=True):
    """Measure the dark field, i.e what the detector sees when the X-Ray source
       is off. Can measure the energy fluence dark field, or the photon count
       dark field.

    Args:
        -
    
    Keyword args:
        integrate_energy (bool): If true the energy fluence is measured by the
                          detector. Photon count is measured if false.
    
    Returns:
        dark_field_image(numpy array[float]): A numpy array containing the dark
                                              field. It has the shape
                                              (detector_rows, detector_columns). 
    
    """
    detector_columns, detector_rows = gvxr.getDetectorNumberOfPixels()
    dark_field_image = np.zeros((detector_rows, detector_columns))

    return dark_field_image

def perform_flat_field_correction(raw_projections, integrate_energy=True):
    """Perform flat field correction on a given set of raw projections.

    Args:
        raw_projections(numpy array[float]): A numpy array of the raw X-Ray
                                             projections. It has the shape 
                                             (num_projections, detector_rows,
                                             detector_columns).
    
    Keyword args:
        integrate_energy (bool): If true the energy fluence is measured by the
                          detector. Photon count is measured if false.
    
    Returns:
        corrected_projections(numpy array[float]): A numpy array of all the 
                                                   corrected X-Ray projections.
                                                   It has the shape 
                                                   (num_projections, 
                                                    detector_rows,
                                                    detector_columns).
    """
    flat_field_image = measure_flat_field(integrate_energy)
    dark_field_image = measure_dark_field(integrate_energy)

    corrected_projections = (raw_projections - dark_field_image) / \
                            (flat_field_image - dark_field_image)
    
    return corrected_projections
