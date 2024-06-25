from multiprocessing import Process, Queue
import numpy as np
from scipy.spatial.transform import Rotation as R
import spekpy as sp
from gvxrPython3 import gvxr


"""
This file contains the main routines for generating sinograms of woven composite
material meshes.

Important things to bear in mind:
    - The unit of energy in gVirtualXRay is MeV (mega electron volts), ergo
      gvxr.computeXRayImage(...) returns X-Ray images in MeV when in energy
      fluence mode. Additionally, gvxr.getUnitOfEnergy("MeV") returns 1.0.
    - The unit of length in gVirtualXRay is mm (milimeter). Therefore the
      function gvxr.getUnitOfLength("mm") returns 1.0.
"""


class XrayConfigError(Exception):
    """Exception raised when missing a required config dictionary entry."""

    pass


def check_xray_config_dict(config_dict):
    """Check that a config dict pertaining an X-Ray CT scan simulation is valid.
      If invalid an appropriate exception is raised.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        xray_dict (dict): A dictionary consisting of relevant
                           X-Ray CT scan simulation parameters.

    """
    args = []
    req_keys = (
        "mesh_paths",
        "phase_elements",
        "phase_ratios",
        "phase_densities",
        "distance_source_origin",
        "distance_origin_detector",
        "detector_columns",
        "detector_rows",
        "detector_pixel_size",
        "anode_angle",
        "energy_bin_width",
        "tube_voltage",
        "tube_power",
        "exposure_time",
        "rot_axis",
        "offset",
        "tilt",
        "number_of_projections",
        "scanning_angle",
    )

    req_types = (
        list,
        list,
        list,
        list,
        float,
        float,
        int,
        int,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        list,
        list,
        int,
        float,
    )

    for req_key, req_type in zip(req_keys, req_types):
        args.append(config_dict.get(req_key))
        if args[-1] is None:
            raise XrayConfigError(
                "Missing required config entry: '"
                + req_key
                + "' of type "
                + str(req_type)
                + "."
            )
        if not isinstance(args[-1], req_type):
            raise TypeError(
                "Invalid type "
                + str(type(args[-1]))
                + " for required config entry '"
                + req_key
                + "'. Should be: "
                + str(req_type)
                + "."
            )
        if not req_type in (str, list):  # All basic numbers should be > 0.
            if not args[-1] > 0:
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be > 0."
                )
        elif req_type is list:
            if req_key == "phase_elements":
                for l in args[-1]:
                    if not isinstance(l, list):
                        raise ValueError(
                            "All entries of '"
                            + req_key
                            + "' must be lists of integers."
                        )
                    for i in l:
                        if not isinstance(i, int):
                            raise TypeError(
                                "All entries of '"
                                + req_key
                                + "' must be lists of integers."
                            )
                        if i < 0:
                            raise ValueError(
                                "All entries in the lists in '"
                                + req_key
                                + "' must > 0."
                            )
            # Ratios are always loaded after elements, so we can access [-2].
            if req_key == "phase_ratios":
                for l in args[-1]:
                    for d in l:
                        if not isinstance(d, float):
                            raise TypeError(
                                "All entries of '"
                                + req_key
                                + "' must be lists of floats."
                            )
                        if not d > 0:
                            raise ValueError(
                                "All entries in the lists in '"
                                + req_key
                                + "' must > 0."
                            )
                        if sum(l) != 1.0:
                            raise ValueError(
                                "The entries of the lists in'"
                                + req_key
                                + "' must sum to 1.0"
                            )
                        if len(args[-1]) != len(args[-2]):
                            raise ValueError(
                                "The length of '"
                                + req_key
                                + "' must equal the length of '"
                                + req_key.replace("ratios", "elements")
                                + "'."
                            )
                        for l1, l2 in zip(args[-1], args[-2]):
                            if len(l1) != len(l2):
                                raise ValueError(
                                    "The length of the entries in '"
                                    + req_key
                                    + "' must equal the length of the "
                                    + "corresponding entries in'"
                                    + req_key.replace("ratios", "elements")
                                    + "'."
                                )

            if req_key == "phase_densities":
                for d in args[-1]:
                    if not isinstance(d, float):
                        raise TypeError(
                            "All entries of '"
                            + req_key
                            + "' must be lists of floats."
                        )
                    if not d > 0:
                        raise ValueError(
                            "All entries of '" + req_key + "' must > 0."
                        )
                    if len(args[-1]) != len(args[-2]):
                        raise ValueError(
                            "The length of '"
                            + req_key
                            + "' must equal the length of '"
                            + req_key.replace("densities", "elements")
                            + "'."
                        )

            if req_key == "mesh_paths":
                for s in args[-1]:
                    if not isinstance(s, str):
                        raise TypeError(
                            "All entries of '" + req_key + "' must be strings."
                        )

            if req_key in ("offset", "tilt"):
                for d in args[-1]:
                    if not isinstance(d, float):
                        raise TypeError(
                            "All entries of '" + req_key + "' must be floats."
                        )
                if len(args[-1]) != 3:
                    raise ValueError(
                        "The entry '" + req_key + "' must have length 3."
                    )
            if req_key == "rot_axis":
                if not args[-1] in ("x", "y", "z"):
                    raise ValueError(
                        "The entry '"
                        + args[-1]
                        + "' of '"
                        + req_key
                        + "' is invalid. It should be 'x' ,'y', or 'z'"
                    )
    opt_keys = (
        "binning",
        "scanner_length_unit",
        "filter_thickness",
        "filter_material",
        "target_material",
        "energy_unit",
        "sample_length_unit",
        "display",
        "photonic_noise",
        "num_reference",
        "threshold",
    )
    opt_types = (int, str, float, str, str, str, str, bool, bool, int, float)
    def_vals = (1, "mm", 0.0, "Al", "W", "keV", "mm", True, True, 100, 1e-8)

    for opt_key, opt_type, def_val in zip(opt_keys, opt_types, def_vals):
        args.append(config_dict.get(opt_key, def_val))
        if not isinstance(args[-1], opt_type):
            raise TypeError(
                "Invalid type "
                + str(type(args[-1]))
                + " for optional config entry '"
                + opt_key
                + "'. Should be: "
                + str(opt_type)
                + "."
            )
        if not opt_type in (str, bool):  # All basic numbers should be > or >= 0
            if (not args[-1] > 0) and opt_key != "filter_thickness":
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be > 0."
                )
            elif (not args[-1] >= 0.0) and opt_key == "filter_thickness":
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be >= 0."
                )

        if opt_key in (
            "scanner_length_unit",
            "sample_length_unit",
        ) and not args[-1] in ("m", "cm", "mm"):
            raise ValueError(
                "The given value '"
                + args[-1]
                + "' of '"
                + req_key
                + "' is invalid. It should be 'm', 'cm', or 'mm'."
            )
        if opt_key == "energy_unit" and args[-1] not in (
            "eV",
            "keV",
            "MeV",
        ):
            raise ValueError(
                "The given value '"
                + args[-1]
                + "' of '"
                + req_key
                + "' is invalid. It should be 'eV', 'keV', or 'MeV'."
            )

    # Special exception check here since we mix req args and optional args.
    if (
        config_dict.get("detector_rows") % config_dict.get("binning", 1) != 0.0
        or config_dict.get("detector_columns") % config_dict.get("binning", 1)
        != 0.0
    ):
        raise ValueError(
            "Bad arguments: binning must be a divisor of both the detector "
            + "rows and the detector columns."
        )

    return dict(zip(req_keys + opt_keys, args))


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
    return None


def generate_xray_spectrum(
    anode_angle,
    energy_bin_width,
    tube_voltage,
    tube_power,
    exposure_time,
    distance_source_detector,
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

    # Spekpy uses cm as unit, so we must convert. length_unit --> m --> cm
    x = 0.0
    y = 0.0
    z = distance_source_detector * scale_factor * 100.0
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
    gvxr.resetBeamSpectrum()
    gvxr.setSourcePosition(-distance_source_origin, 0.0, 0.0, length_unit)

    if focal_spot_size <= 0:
        gvxr.usePointSource()
    else:
        gvxr.setFocalSpot(
            -distance_source_origin,
            0.0,
            0.0,
            focal_spot_size,
            length_unit,
            sub_sources,
        )

    if len(energies) == 1:
        gvxr.setMonoChromatic(energies[0], energy_unit, counts[0])
    else:
        for energy, count in zip(energies, counts):
            gvxr.addEnergyBinToSpectrum(energy, energy_unit, count)
    return None


def set_up_sample(
    mesh_paths,
    phase_elements,
    phase_ratios,
    phase_densities,
    rot_axis,
    offset,
    tilt,
    length_unit="m",
):
    """Load sample geometry and X-Ray absorption properties. The last entry in
    the lists defining the sample phases will be added as an outer surface, and
    is thus appropriate for matrix.

    Args:
        mesh_paths (lis[string]): The absolute paths (including file name)
                                      to sample geometry mesh files.

        phase_elements (list(list(int))): The element numbers of the
                                             constituents of the different
                                             sample material phases.

        phase_ratios (list(list(float))): The relative amount of the
                                             constituent elements of the
                                             different material phases.

        phase_densities (list(float)): The densities of the different
                                          material phases in g/cm^3.

        rot_axis (str): An axis given in the sample's local coordinate fram.
                        The tomographic scan is performed around this axis,
                        i.e the axis that should point upwards in the global
                        coordinates. It can be "x", "y", or "z".

        offset (list[float]): A list of length 3 that represent the sample
                              offset in global coordinates, measured from the
                              center of the sample. [0,0,0] results in 0 offset.

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

    phase_names = [
        "phase_" + str(i) for i in range(len(phase_densities))
    ]

    # We load files for the first occurences here in order to load only once.
    for name, path, element, ratio, density in zip(
        phase_names,
        mesh_paths,
        phase_elements,
        phase_ratios,
        phase_densities,
    ):
        gvxr.loadMeshFile(name, path, length_unit, False)
        if name == "phase_" + str(len(phase_densities) - 1):
            gvxr.addPolygonMeshAsOuterSurface(name)
        else:
            gvxr.addPolygonMeshAsInnerSurface(name)
        gvxr.setMixture(name, element, ratio)
        gvxr.setDensity(name, density, "g/cm3")

        # We must rotate before we translate since the translation
        # does not translate the axis of rotation.
        gvxr.rotateNode(name, angle, axis[0], axis[1], axis[2])
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
            offset[0],
            offset[1],
            offset[2],
            length_unit,
        )
    return None


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

    # Take the threshold again in order to prevent negative values.
    # This can occur if the noise happens to make I>I0.
    neg_log_projections[neg_log_projections < threshold] = threshold

    return neg_log_projections


def _generate_sinograms(xray_config_dict, queue):
    """Perform an X-Ray CT scan of a sample. Do not use this function directly,
    please use generate_sinograms instead.

    Args:
        xray_config_dict (dictionary): A (checked for X-Ray options) dictionary
                                      of tex_ray options.

        queue (Queue): A queue that allows reading data from a process.

    Keyword args:
        -
    Returns:
        -
    """
    gvxr.createOpenGLContext()

    set_up_detector(
        xray_config_dict["distance_origin_detector"],
        xray_config_dict["detector_columns"],
        xray_config_dict["detector_rows"],
        xray_config_dict["detector_pixel_size"],
        binning=xray_config_dict["binning"],
        length_unit=xray_config_dict["scanner_length_unit"],
    )

    energy_bins, photon_flux = generate_xray_spectrum(
        xray_config_dict["anode_angle"],
        xray_config_dict["energy_bin_width"],
        xray_config_dict["tube_voltage"],
        xray_config_dict["tube_power"],
        xray_config_dict["exposure_time"],
        xray_config_dict["distance_source_origin"]
        + xray_config_dict["distance_origin_detector"],
        xray_config_dict["detector_pixel_size"],
        binning=xray_config_dict["binning"],
        filter_thickness=xray_config_dict["filter_thickness"],
        filter_material=xray_config_dict["filter_material"],
        target_material=xray_config_dict["target_material"],
        length_unit=xray_config_dict["scanner_length_unit"],
    )

    set_up_xray_source(
        xray_config_dict["distance_source_origin"],
        -1,
        energy_bins,
        photon_flux,
        length_unit=xray_config_dict["scanner_length_unit"],
        energy_unit=xray_config_dict["energy_unit"],
    )
    set_up_sample(
        xray_config_dict["mesh_paths"],
        xray_config_dict["phase_elements"],
        xray_config_dict["phase_ratios"],
        xray_config_dict["phase_densities"],
        xray_config_dict["rot_axis"],
        xray_config_dict["offset"],
        xray_config_dict["tilt"],
        length_unit=xray_config_dict["sample_length_unit"],
    )
    raw_projections = perform_tomographic_scan(
        xray_config_dict["number_of_projections"],
        xray_config_dict["scanning_angle"],
        display=xray_config_dict["display"],
        photonic_noise=xray_config_dict["photonic_noise"],
    )
    # After finishing the tomographic constructions it is safe to close window.
    gvxr.destroyWindow()

    flat_field_image = measure_flat_field(
        photonic_noise=xray_config_dict["photonic_noise"],
        num_reference=xray_config_dict["num_reference"],
    )
    dark_field_image = measure_dark_field()
    corrected_projections = perform_flat_field_correction(
        raw_projections, flat_field_image, dark_field_image
    )
    neg_log_projections = neg_log_transform(
        corrected_projections, xray_config_dict["threshold"]
    )

    # Reformat the projections into a set of sinograms on the ASTRA form.
    sinograms = np.swapaxes(neg_log_projections, 0, 1)

    queue.put(sinograms)
    return None


def generate_sinograms(config_dict):
    """Perform an X-Ray CT scan of a sample and return the sinograms.

    This function wraps _generate_sinograms. This construction is needed for
    multiple sequential simulations due to the nature of the gvxr singleton.
    A new process must be spawned for each simulation to avoid loading into
    the previous simulation.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -
    Returns:
        sinograms (numpy array[float]): The measured CT sinograms. The array has
                                        the shape (detector_rows,
                                        number_of_projections, detector_columns)
    """
    # We must check options here because an exception in the child hangs queue.
    xray_config_dict = check_xray_config_dict(config_dict)

    q = Queue(1)
    p = Process(
        target=_generate_sinograms,
        args=(
            xray_config_dict,
            q,
        ),
    )
    p.start()
    sinograms = q.get()
    p.join()
    p.close()
    q.close()
    q.join_thread()

    return sinograms
