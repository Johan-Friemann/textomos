import numpy as np
from scipy.spatial.transform import Rotation as R
import spekpy as sp
from gvxrPython3 import gvxr

"""
This file consists of helper functions to be called by the main X-Ray CT script.

Important things to bear in mind:
    - The unit of energy in gVirtualXRay is MeV (mega electron volts), ergo
      gvxr.computeXRayImage(...) returns X-Ray images in MeV when in energy
      fluence mode. Additionally, gvxr.getUnitOfEnergy("MeV") returns 1.0.
    - The unit of length in gVirtualXRay is mm (milimeter). Therefore the
      function gvxr.getUnitOfLength("mm") returns 1.0.
    - The data measured by a sensor pixel is an integrated quantity, where the
      signal pertains all rays that pass through the corresponding voxel in the
      reconstruction volume. Specifically, if the 1D integral that equals to 
      -ln(I/I_0) is sampled, the sampled value will itself be an integral along
      the side of that pixel (or integral through the corresponding voxel). 
      Therefore, in order to get the attenuation per unit length the values in
      the reconstruction need to be rescaled by the voxel side length.
"""


def set_up_detector(
    distance_origin_detector,
    detector_columns,
    detector_rows,
    detector_pixel_size,
    binning=1,
    length_unit="m",
):
    """Set up the gVirtualXRay detector. Note that the reconstruction volume is
       located at x=y=z=0. Only supports square detector pixels.

    Args:
        distance_origin_detector (float): The distance from the origin of the
                                          reconstruction area to the X-Ray
                                          detector.
        detector_columns (int): The number of pixels in the width direction of
                                the detector.
        detector_rows (int): The number of pixels in the height direction of the
                             detector.
        detector_pixel_size (float): The side length of the detector pixels.

    Keyword args:
        binning (int): The binning number. It defines the side length of the
                       square of pixels to average over. Must be  a divisor of
                       both the detector rows and the detector columns.
        length_unit (string): The unit of length measurement (m, cm, mm, um).
                              Default unit is m (meter).

    Returns:
        -
    """
    if detector_rows % binning != 0.0 or detector_columns % binning != 0.0:
        raise ValueError(
            "Bad arguments: binning must be a divisor of both the dector "
            + "rows and the detector columns."
        )
    gvxr.setDetectorUpVector(0, 0, 1)
    gvxr.setDetectorPosition(distance_origin_detector, 0.0, 0.0, length_unit)
    gvxr.setDetectorNumberOfPixels(
        detector_columns // binning, detector_rows // binning
    )
    gvxr.setDetectorPixelSize(
        detector_pixel_size * binning,
        detector_pixel_size * binning,
        length_unit,
    )


def generate_xray_spectrum(
    anode_angle,
    energy_bin_width,
    tube_voltage,
    tube_power,
    exposure_time,
    distance_source_detector,
    offset,
    detector_pixel_size,
    binning=1,
    filter_thickness=0.0,
    filter_material="Al",
    target_material="W",
    length_unit="m",
):
    """Use SpekPy to generate an x-ray spectrum.

    Args:
        anode_angle (float): The effective x-ray tube anode angle given
                             in degrees.
        energy_bin_width (float): The width of the spectrum energy bins given
                                  in kilovolts (kV).
        tube_voltage (float): The voltage of the x-ray tube given in kilovolts
                              (kV).
        tube_power (float): The electrical power of the x-ray tube given in
                            watts (W).
        exposure_time (float): The x-ray exposure time given in seconds (s).
        distance_source_detector: The distance from x-ray source to detector.
        offset (list[float]): A list of length 3 that represent the sample
                              offset in global coordinates, measured from the
                              center of the sample (center of tiling).
                              [0,0,0] results in no offset.
        detector_pixel_size (float): The area of one detector pixel.
    Keyword args:
        binning (int): The binning number. It defines the side length of the
                       square of pixels to average over. Used to scale
                       flux appropriately.
        filter_thickness (float): The thickness of the x-ray filter.
                                  Default is no filter (=0.0).
        filter_material (string): The chemical symbol of the filter material.
                                  See SpekPy docs. Default is aluminium.
        target_material (string): The chemical symbol of the target material.
                                  See SpekPy docs. Default is tungsten.
        length_unit (string): The unit of length measurement (m, cm, mm).
                              Default unit is m (meter).
    Returns:
        (energy_bins, photon_flux) (numpy array[float]): Returns a tuple of the
                                                         energy bins, and the
                                                         photon flux per
                                                         detector pixel.
    """
    if length_unit == "mm":
        scale_factor = 1e-3
    elif length_unit == "cm":
        scale_factor = 1e-2
    elif length_unit == "m":
        scale_factor = 1.0
    else:
        raise ValueError(
            "Bad arguments: length_unit must be 'm', 'cm', or 'mm'."
        )
    # Spekpy uses cm as unit, so we must convert. length_unit --> m --> cm
    x = offset[0] * scale_factor * 100.0
    y = offset[1] * scale_factor * 100.0
    z = (distance_source_detector - offset[2]) * scale_factor * 100.0
    detector_area = (
        detector_pixel_size**2 * binning**2 * scale_factor**2 * 100.0**2
    )  # cm^2

    # for filter thickness SpekPy uses mm length_unit --> m --> mm
    filter_d = filter_thickness * scale_factor * 1000.0

    tube_current = tube_power / tube_voltage
    charge = tube_current * exposure_time

    spectrum = sp.Spek(
        kvp=tube_voltage,
        th=anode_angle,
        dk=energy_bin_width,
        mas=charge,
        targ=target_material,
        x=x,
        y=y,
        z=z,
    ).filter(filter_material, filter_d)
    bins, flux_per_area = spectrum.get_spectrum(diff=False)
    # diff=False: photons / cm^2 / bin
    return bins, flux_per_area * detector_area


def set_up_xray_source(
    distance_source_origin,
    focal_spot_size,
    energies,
    counts,
    sub_sources=2,
    energy_unit="keV",
    length_unit="m",
):
    """Set up the gVirtualXRay X-Ray source.
       The spectrum of the source is built by specifying photon energies and
       corresponding photon counts. If lists of length 1 are given a
       monochromatic source is set up.

    Args:
        distance_source_origin (float): The distance from the source to the
                                        origin of the reconstruction area.
        focal_spot_size (float): The side length of the X-Ray source focal spot.
                                 Will use a point source if non postive.
        energies (list[float]): A list of X-Ray photon energies.
        counts (list[int]): A list of X-Ray photon counts.

    Keyword args:
        energy_unit (string): The unit of photon energy (eV, keV, MeV).
                              The default is keV (kilo electronvolt).
        length_unit (string): The unit of length measurement (m, cm, mm, um).
                              Default unit is m (meter).
        sub_sources (int): The number of sub point sources per axis to use for
                           the focal spot. Note that a simulation is performed
                           sub_sources cubed times, so the computational time
                           can become long for many sources. However, a too
                           small number of sources can cause aliasing.
                           Default number is 2 sub sources.
    Returns:
        -
    """

    if len(energies) != len(counts):
        raise ValueError(
            "Bad arguments: 1st argument 'energies' and 2nd "
            + "argument 'counts' must be of the same length!"
        )

    gvxr.resetBeamSpectrum()

    if focal_spot_size <= 0:
        gvxr.setSourcePosition(-distance_source_origin, 0.0, 0.0, length_unit)
        gvxr.usePointSource()
    else:
        raise NotImplementedError(
            "gvxr currently does not have a correct "
            + "implementation available in the python pkg."
        )
        gvxr.setFocalSpot(
            -distance_source_origin,
            0,
            0,
            focal_spot_size,
            length_unit,
            sub_sources,
        )

    if len(energies) == 1:
        gvxr.setMonoChromatic(energies[0], energy_unit, counts[0])
    else:
        for energy, count in zip(energies, counts):
            gvxr.addEnergyBinToSpectrum(energy, energy_unit, count)


def set_up_sample(
    weft_path,
    weft_elements,
    weft_ratios,
    weft_density,
    warp_path,
    warp_elements,
    warp_ratios,
    warp_density,
    matrix_path,
    matrix_elements,
    matrix_ratios,
    matrix_density,
    rot_axis,
    tiling,
    offset,
    tilt,
    length_unit="m",
):
    """Load yarn (weft and warp) and matrix geometries and X-Ray absorption
       properties.

    Args:
        weft_path (string): The absolute path (including file name) to
                            the weft geometry mesh file.

        weft_elements (list(int)): The element numbers of the constituents of
                                   the weft material.

        weft_ratios (list(float)): The relative amount of the constituent
                                    elements of the weft material.

        weft_density (float): The density of the weft material in g/cm^3.

        warp_path (string): The absolute path (including file name) to
                            the warp geometry mesh file.

        warp_elements (list(int)): The element numbers of the constituents of
                                   the warp material.

        warp_ratios (list(float)): The relative amount of the constituent
                                   elements of the warp material.

        warp_density (float): The density of the warp material in g/cm^3.

        matrix_path (string): The absolute path (including file name) to
                              the matrix geometry mesh file.

        matrix_elements (list(int)): The element numbers of the constituents of
                                     the matrix material.

        matrix_ratios (list(float)): The relative amount of the constituent
                                     elements of the matrix material.

        matrix_density (float): The density of the matrix material in g/cm^3.

        rot_axis (str): An axis given in the sample's local coordinate fram.
                        The tomographic scan is performed around this axis,
                        i.e the axis that should point upwards in the global
                        coordinates. It can be "x", "y", or "z".

        tiling (list[int]): A list of length 3 that represents the tiling
                            pattern in the sample's local coordinates. Ex:
                            [2,3,3] will tile two cells in x and 3 in y and z.
                            [1,1,1] will result in no tiling (original sample).

        offset (list[float]): A list of length 3 that represent the sample
                              offset in global coordinates, measured from the
                              center of the sample (center of tiling).
                              [0,0,0] results in no offset.

        tilt (list[float]): A list of length 3 that represents a rotation.
                            The magnitude of the vector is the angle and the
                            direction is the axis of rotation. It is given in
                            degrees. [0,0,0] results in no tilt.

    Keyword args:
        length_unit (string): The unit of length measurement (m, cm, mm, um).
                              Default unit is m (meter).

    Returns:
        -
    """
    if len(weft_elements) != len(weft_ratios):
        raise ValueError(
            "Bad arguments: number of weft ratios must agree "
            + "with the number of elements!"
        )

    if sum(weft_ratios) != 1.0:
        raise ValueError("Bad arguments: sum of weft ratios must be 1.0.")

    if len(warp_elements) != len(warp_ratios):
        raise ValueError(
            "Bad arguments: number of warp ratios must agree "
            + "with the number of elements!"
        )

    if sum(warp_ratios) != 1.0:
        raise ValueError("Bad arguments: sum of warp ratios must be 1.0.")

    if len(matrix_elements) != len(matrix_ratios):
        raise ValueError(
            "Bad arguments: number of matrix ratios must agree "
            + "with the number of elements!"
        )

    if sum(matrix_ratios) != 1.0:
        raise ValueError("Bad arguments: sum of matrix ratios must be 1.0.")

    if rot_axis not in ["x", "y", "z"]:
        raise ValueError(
            "Bad arguments: Rotation axis must be 'x', 'y', or 'z'."
        )

    if len(offset) != 3:
        raise ValueError(
            "Bad arguments: offset should contain x, y, and z components."
        )

    if len(tilt) != 3:
        raise ValueError(
            "Bad arguments: tilt should contain x, y, and z components."
        )

    if len(tiling) != 3:
        raise ValueError(
            "Bad arguments: tiling should contain x, y, and z components."
        )

    # cast to numpy
    tilt = np.array(tilt)
    offset = np.array(offset)

    if rot_axis == "x":
        axis = np.array([0, -1, 0])
        angle = 90
    elif rot_axis == "y":
        axis = np.array([1, 0, 0])
        angle = 90
    else:
        # z is already up, so we just use this to not have to split by case.
        axis = np.array([0, 0, 1])
        angle = 0

    # gvxr rotates the local coords, so we need to map the tilt and offset
    # accordingly to apply them in the global system.
    rot_tilt = R.from_rotvec(-tilt * np.pi / 180)
    rot = R.from_rotvec(-angle * axis * np.pi / 180)
    tilt_angle = np.linalg.norm(tilt)
    tilt_axis = rot.apply(tilt / tilt_angle)
    offset = rot.apply(rot_tilt.apply(offset))

    # Turn inputs into lists for use in zipped loops.
    root_names = ["matrix_000", "weft_000", "warp_000"]
    paths = [matrix_path, weft_path, warp_path]
    elements = [matrix_elements, weft_elements, warp_elements]
    ratios = [matrix_ratios, weft_ratios, warp_ratios]
    densities = [matrix_density, weft_density, warp_density]
    tile_size = [0, 0, 0]

    # We load files for the first occurences here in order to load only once.
    for root_name, path, element, ratio, density in zip(
        root_names, paths, elements, ratios, densities
    ):
        gvxr.loadMeshFile(root_name, path, length_unit, True)
        gvxr.setMixture(root_name, element, ratio)
        gvxr.setDensity(root_name, density, "g/cm3")
        # We need to align the tiling to the matrix bounding box.
        if root_name == "matrix_000":
            bbox = gvxr.getNodeOnlyBoundingBox(root_name, length_unit)
            tile_size[0] = bbox[3] - bbox[0]
            tile_size[1] = bbox[4] - bbox[1]
            tile_size[2] = bbox[5] - bbox[2]
        # We must rotate before we translate since the translation
        # does not translate the axis of rotation.
        gvxr.rotateNode(root_name, angle, axis[0], axis[1], axis[2])
        # results in crazy behavior in gvxr if 0.
        if np.any(tilt):
            gvxr.rotateNode(
                root_name,
                tilt_angle,
                tilt_axis[0],
                tilt_axis[1],
                tilt_axis[2],
            )
        gvxr.translateNode(
            root_name,
            -tile_size[0] * (tiling[0] - 1) / 2 + offset[0],
            -tile_size[1] * (tiling[1] - 1) / 2 + offset[1],
            -tile_size[2] * (tiling[2] - 1) / 2 + offset[2],
            length_unit,
        )

    # Once mesh files are loaded we can keep re-using them for the tiling.
    for i in range(tiling[0]):
        for j in range(tiling[1]):
            for k in range(tiling[2]):
                # This is a hack so we skip the first one.
                if i == 0 and j == 0 and k == 0:
                    continue
                names = [
                    "matrix_" + str(i) + str(j) + str(k),
                    "weft_" + str(i) + str(j) + str(k),
                    "warp_" + str(i) + str(j) + str(k),
                ]

                for name, root_name, element, ratio, density in zip(
                    names, root_names, elements, ratios, densities
                ):
                    gvxr.emptyMesh(name)
                    gvxr.addMesh(name, root_name)
                    gvxr.addPolygonMeshAsInnerSurface(name)
                    gvxr.setMixture(name, element, ratio)
                    gvxr.setDensity(name, density, "g/cm3")
                    gvxr.rotateNode(
                        name,
                        angle,
                        axis[0],
                        axis[1],
                        axis[2],
                    )
                    # results in crazy behavior in gvxr if 0.
                    if np.any(tilt):
                        gvxr.rotateNode(
                            name,
                            tilt_angle,
                            tilt_axis[0],
                            tilt_axis[1],
                            tilt_axis[2],
                        )
                    gvxr.translateNode(
                        name,
                        tile_size[0] * (i - (tiling[0] - 1) / 2) + offset[0],
                        tile_size[1] * (j - (tiling[1] - 1) / 2) + offset[1],
                        tile_size[2] * (k - (tiling[2] - 1) / 2) + offset[2],
                        length_unit,
                    )


def add_photonic_noise(noise_free_projection, integrate_energy=True):
    """Add Poisson distributed photonic noise (shot noise) to the projection.

    Args:
        noise_free_projection numpy array[float]): A noise-free projection.

    Keyword args:
        integrate_energy (bool): If true the measured energy is rescaled to
                                 approximated number of photons counted before
                                 adding the noise. After the noise is added
                                 the resulting projection is scaled back to the
                                 energy domain.

    Returns:
        noisy_projection (numpy array[float]): A noisy projection.
    """
    if integrate_energy:
        expected_photon_count = np.sum(gvxr.getPhotonCountEnergyBins())
        # This returns the energy in MeV, but that is fine since the output
        # of gvxr.computeXRayImage(True) returns the image in MeV.
        expected_energy = gvxr.getTotalEnergyWithDetectorResponse()
        scale_factor = expected_photon_count / expected_energy
        photonic_projection = noise_free_projection * scale_factor
    else:
        photonic_projection = noise_free_projection
        scale_factor = 1.0

    noisy_projection = np.random.poisson(photonic_projection)

    return noisy_projection / scale_factor


def perform_tomographic_scan(
    num_projections,
    scanning_angle,
    display=False,
    integrate_energy=True,
    photonic_noise=True,
):
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
        photonic_noise (bool): If true photonic noise is added to projections.

    Returns:
        raw_projections (numpy array[float]): A numpy array of all measured
                                              X-Ray projections. It has the
                                              shape (num_projections,
                                              detector_rows, detector_columns).
    """
    # No need to correct for binning since it is taken care of during set-up.
    # We do not need to account for binning in the noise since a sum of
    # Poisson distributed variables is Poisson distributed.
    detector_columns, detector_rows = gvxr.getDetectorNumberOfPixels()
    raw_projections = np.empty(
        (num_projections, detector_rows, detector_columns)
    )

    angular_step = scanning_angle / num_projections
    for angle_id in range(0, num_projections):
        # Compute an X-ray image and add it to the set of projections.
        raw_projection = np.array(gvxr.computeXRayImage(integrate_energy))
        if photonic_noise:
            raw_projection = add_photonic_noise(raw_projection)
        raw_projections[angle_id] = raw_projection
        # Update the rendering if display.
        if display:
            gvxr.displayScene()

        # Rotate the sample
        gvxr.rotateScene(-angular_step, 0, 0, 1)

    return raw_projections


def measure_flat_field(
    integrate_energy=True, photonic_noise=True, num_reference=10
):
    """Measure the flat field, i.e what the detector sees when the X-Ray source
       is on but there is no sample present. Can measure the energy fluence flat
       field, or the photon count flat field.

    Args:
        -

    Keyword args:
        integrate_energy (bool): If true the energy fluence is measured by the
                          detector. Photon count is measured if false.
        photonic_noise (bool): If true photonic noise is added to flat field.
        num_reference (int): The number of reference images taken (and averaged)
                             to generate the flat field. Small number can result
                             in ring artefacts forming.

    Returns:
        flat_field_image(numpy array[float]): A numpy array containing the flat
                                              field. It has the shape
                                              (detector_rows, detector_columns).

    """
    # No need to correct for binning since it is taken care of during set-up.
    # We do not need to account for binning in the noise since a sum of
    # Poisson distributed variables is Poisson distributed.
    detector_columns, detector_rows = gvxr.getDetectorNumberOfPixels()
    flat_field_image = np.ones((detector_rows, detector_columns))

    # This returns the energy in MeV, but that is fine since the output
    # of gvxr.computeXRayImage(True) returns the image in MeV.
    total_energy = gvxr.getTotalEnergyWithDetectorResponse()
    total_photon_count = np.sum(gvxr.getPhotonCountEnergyBins())

    # If not measuring the energy fluence, set energy to photon count to prevent
    # repeating code.
    if not integrate_energy:
        total_energy = total_photon_count

    flat_field_image *= total_energy
    noisy_image = np.zeros(flat_field_image.shape)
    if photonic_noise:
        for i in range(num_reference):
            noisy_image += add_photonic_noise(
                flat_field_image, integrate_energy=integrate_energy
            )
    else:
        noisy_image = flat_field_image * num_reference
    noisy_image /= num_reference
    return noisy_image


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
    # No need to correct for binning since it is taken care of during set-up.
    detector_columns, detector_rows = gvxr.getDetectorNumberOfPixels()
    dark_field_image = np.zeros((detector_rows, detector_columns))

    return dark_field_image


def perform_flat_field_correction(
    raw_projections, flat_field_image, dark_field_image
):
    """Perform flat field correction on a given set of raw projections, based
       on the provided flat- and dark fields.

    Args:
        raw_projections(numpy array[float]): A numpy array of the raw X-Ray
                                             projections. It has the shape
                                             (num_projections, detector_rows,
                                             detector_columns).
        flat_field_image(numpy array[float]): The detector flat field. It has
                                              the shape (detector_rows,
                                              detector_columns).
        dark_field_image(numpy array[float]): The detector dark field. It has
                                              the shape (detector_rows,
                                              detector_columns).

    Keyword args:
        -

    Returns:
        corrected_projections(numpy array[float]): A numpy array of all the
                                                   corrected X-Ray projections.
                                                   It has the same shape as
                                                   raw_projections.
    """
    corrected_projections = (raw_projections - dark_field_image) / (
        flat_field_image - dark_field_image
    )

    return corrected_projections


def neg_log_transform(corrected_projections, threshold):
    """Perform the negative logarithm transform of flat field corrected
       projections.

    Args:
        corrected_projections(numpy array[float]): A numpy array of the
                                                   flat-field corrected X-Ray
                                                   projections. It has the shape
                                                   (num_projections,
                                                   detector_rows,
                                                   detector_columns).
        threshold (float): The threshold value. All elements less than threshold
                           inside corrected_projections will be set to
                           threshold before taking the negative logarithm.

    Keyword args:
        -

    Returns:
        neg_log_projections(numpy array[float]): A numpy array containing the
                                                 negative logarithm transformed
                                                 data. It has the same shape as
                                                 corrected_projection.
    """

    # make a copy in order to not modify the original projections when accessing
    # array elements.
    neg_log_projections = np.copy(corrected_projections)

    # Take the threshold to prevent taking log of negative or zero values.
    neg_log_projections[neg_log_projections < threshold] = threshold

    neg_log_projections = -np.log(neg_log_projections)

    return neg_log_projections


def compute_astra_scale_factor(
    distance_source_origin, distance_origin_detector, detector_pixel_size
):
    """Compute the ASTRA scale factor for conical beam geometry.
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
    scale_factor = 1 / (
        detector_pixel_size
        * distance_source_origin
        / (distance_source_origin + distance_origin_detector)
    )
    return scale_factor
