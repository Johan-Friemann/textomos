import numpy as np
import astra
from gvxrPython3 import gvxr

from x_ray.x_ray_util import *

"""
This file contains the main routines for generating sinograms of woven composite
material meshes and the tomographic reconstructions of the same. 
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
        "weft_path",
        "warp_path",
        "matrix_path",
        "distance_source_origin",
        "distance_origin_detector",
        "detector_columns",
        "detector_rows",
        "detector_pixel_size",
        "weft_elements",
        "weft_ratios",
        "weft_density",
        "warp_elements",
        "warp_ratios",
        "warp_density",
        "matrix_elements",
        "matrix_ratios",
        "matrix_density",
        "anode_angle",
        "energy_bin_width",
        "tube_voltage",
        "tube_power",
        "exposure_time",
        "rot_axis",
        "tiling",
        "offset",
        "tilt",
        "number_of_projections",
        "scanning_angle",
    )

    req_types = (
        str,
        str,
        str,
        float,
        float,
        int,
        int,
        float,
        list,
        list,
        float,
        list,
        list,
        float,
        list,
        list,
        float,
        float,
        float,
        float,
        float,
        float,
        str,
        list,
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
            if req_key in ("weft_elements", "warp_elements", "matrix_elements"):
                for i in args[-1]:
                    if not isinstance(i, int):
                        raise TypeError(
                            "All entries of '" + req_key + "' must be integers."
                        )
                    if i < 0:
                        raise ValueError(
                            "All entries of '" + req_key + "' must > 0."
                        )
            # Ratios are always loaded after elements, so we can access [-2].
            if req_key in ("weft_ratios", "warp_ratios", "matrix_ratios"):
                for d in args[-1]:
                    if not isinstance(d, float):
                        raise TypeError(
                            "All entries of '" + req_key + "' must be floats."
                        )
                    if not i > 0:
                        raise ValueError(
                            "All entries of '" + req_key + "' must > 0."
                        )
                if sum(args[-1]) != 1.0:
                    raise ValueError(
                        "The entries of '" + req_key + "' must sum to 1.0"
                    )
                if len(args[-1]) != len(args[-2]):
                    raise ValueError(
                        "The length of '"
                        + req_key
                        + "' must equal the length of '"
                        + req_key.replace("ratios", "elements")
                        + "'."
                    )
            if req_key == "tiling":
                for i in args[-1]:
                    if not isinstance(i, int):
                        raise TypeError(
                            "All entries of '" + req_key + "' must be integers."
                        )
                    if not i > 0:
                        raise ValueError(
                            "All entries of '" + req_key + "' must > 0."
                        )
                if len(args[-1]) != 3:
                    raise ValueError(
                        "The entry '" + req_key + "' must have length 3."
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

    return dict(zip(req_keys + opt_keys, args))


def check_reconstruction_config_dict(config_dict):
    """Check that a config dict pertaining an X-Ray CT reconstruction is valid.
      If invalid an appropriate exception is raised.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        args list[]: A list of required args to perform the reconstruction.
        opt_args list[]: A list of optional args used for the reconstruction.

    """


def generate_sinograms(config_dict):
    """Perform an X-Ray CT scan of a sample and return the sinograms.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -
    Returns:
        sinograms (numpy array[float]): The measured CT sinograms. The array has
                                        the shape (detector_rows,
                                        number_of_projections, detector_columns)
    """

    xray_config_dict = check_xray_config_dict(config_dict)

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
        xray_config_dict["offset"],
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
        xray_config_dict["weft_path"],
        xray_config_dict["weft_elements"],
        xray_config_dict["weft_ratios"],
        xray_config_dict["weft_density"],
        xray_config_dict["warp_path"],
        xray_config_dict["warp_elements"],
        xray_config_dict["warp_ratios"],
        xray_config_dict["warp_density"],
        xray_config_dict["matrix_path"],
        xray_config_dict["matrix_elements"],
        xray_config_dict["matrix_ratios"],
        xray_config_dict["matrix_density"],
        xray_config_dict["rot_axis"],
        xray_config_dict["tiling"],
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

    return sinograms


def perform_tomographic_reconstruction(
    sinograms, config_dict, align_coordinates=True
):
    """Perform a tomographic reconstruction with ASTRA given a set of sinograms.

    Args:
        sinograms (numpy array[float]): The measured CT sinograms. The array has
                                        the shape (detector_rows,
                                        number_of_projections, detector_columns)
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        align_coordinates (bool): If true the coordinate system of the
                                  reconstructed volume is rotated 90 degrees
                                  around the z-axis, such that the coordinate
                                  system is the same as for the input stl file.

    Returns:
        reconstruction (numpy array[float]): The reconstructed sample. The array
                                             has the shape (detector_rows,
                                             detector_columns, detector_columns)

    """
    # IMPORTANT: Since we scale everything by the binning parameter when we set
    # up our simulation, we must also scale by the binning parameter here!

    # ASTRA toolbox uses clockwise rotation as positive. If the scanner rotates
    # counter clockwise, we need to add a negative sign here.
    projection_angles = config_dict["sample_rotation_direction"] * np.linspace(
        0,
        np.deg2rad(config_dict["scanning_angle"]),
        config_dict["number_of_projections"],
    )

    scale_factor = compute_astra_scale_factor(
        config_dict["distance_source_origin"],
        config_dict["distance_origin_detector"],
        config_dict["detector_pixel_size"] * config_dict["binning"],
    )

    vol_geo = astra.create_vol_geom(
        config_dict["detector_columns"] // config_dict["binning"],
        config_dict["detector_columns"] // config_dict["binning"],
        config_dict["detector_rows"] // config_dict["binning"],
    )

    proj_geo = astra.creators.create_proj_geom(
        "cone",
        config_dict["detector_pixel_size"]
        * config_dict["binning"]
        * scale_factor,
        config_dict["detector_pixel_size"]
        * config_dict["binning"]
        * scale_factor,
        config_dict["detector_rows"] // config_dict["binning"],
        config_dict["detector_columns"] // config_dict["binning"],
        projection_angles,
        config_dict["distance_source_origin"] * scale_factor,
        config_dict["distance_origin_detector"] * scale_factor,
    )
    proj_id = astra.data3d.create("-sino", proj_geo, data=sinograms)
    rec_id = astra.data3d.create("-vol", vol_geo, data=0)
    alg_cfg = astra.astra_dict(config_dict["reconstruction_algorithm"])
    alg_cfg["ReconstructionDataId"] = rec_id
    alg_cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(alg_id)
    reconstruction = astra.data3d.get(rec_id)

    # Remove eventual erroneous negative values.
    reconstruction[reconstruction < 0] = 0

    # Rescale to get attenuation coefficient in scanner_length_unit^-1.
    reconstruction *= scale_factor

    # Since gvxr.getUnitOfLength("mm") returns 1.0 we scale from 1000.
    unit_scale = 1000 / gvxr.getUnitOfLength(config_dict["scanner_length_unit"])

    # Rescale to get attenuation coefficient in cm^-1.
    reconstruction *= unit_scale / 100

    # Rotate the coordinates 90 degrees around the positive z-axis.
    if align_coordinates:
        reconstruction = np.swapaxes(reconstruction, 1, 2)
        reconstruction = np.flip(reconstruction, axis=1)

    return reconstruction
