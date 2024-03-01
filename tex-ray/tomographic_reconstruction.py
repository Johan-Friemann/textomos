import numpy as np
import astra
from gvxrPython3 import gvxr

"""
This file contains the main routines related to the reconstruction of 
X-Ray CT scans.

Important things to bear in mind:
    - The data measured by a sensor pixel is an integrated quantity, where the
      signal pertains all rays that pass through the corresponding voxel in the
      reconstruction volume. Specifically, if the 1D integral that equals to 
      -ln(I/I_0) is sampled, the sampled value will itself be an integral along
      the side of that pixel (or integral through the corresponding voxel). 
      Therefore, in order to get the attenuation per unit length, the values in
      the reconstruction need to be rescaled by the voxel side length.
"""


class ReconConfigError(Exception):
    """Exception raised when missing a required config dictionary entry."""

    pass


def check_reconstruction_config_dict(config_dict):
    """Check that a config dict pertaining an X-Ray CT reconstruction is valid.
      If invalid an appropriate exception is raised.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        recon_dict (dict): A dictionary consisting of relevant X-Ray CT scan
                           reconstruction parameters.


    """
    args = []

    req_keys = (
        "sample_rotation_direction",
        "scanning_angle",
        "number_of_projections",
        "distance_source_origin",
        "distance_origin_detector",
        "detector_pixel_size",
        "detector_columns",
        "detector_rows",
    )
    req_types = (int, float, int, float, float, float, int, int)

    for req_key, req_type in zip(req_keys, req_types):
        args.append(config_dict.get(req_key))
        if args[-1] is None:
            raise ReconConfigError(
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
        if req_key != "sample_rotation_direction":
            if not args[-1] > 0:
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be > 0."
                )
        else:
            if not args[-1] in (-1, 1):
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be 1, or -1."
                )

    opt_keys = ("binning", "reconstruction_algorithm", "scanner_length_unit")
    opt_types = (int, str, str)
    def_vals = (1, "FDK_CUDA", "mm")

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
        if not opt_type is str:
            if not args[-1] > 0:
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be > 0."
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

    # Special exception check here since we mix req args and optional args.
    if (
        config_dict.get("detector_rows") % config_dict.get("detector_rows", 1)
        != 0.0
        or config_dict.get("detector_columns")
        % config_dict.get("detector_rows", 1)
        != 0.0
    ):
        raise ValueError(
            "Bad arguments: binning must be a divisor of both the detector "
            + "rows and the detector columns."
        )

    return dict(zip(req_keys + opt_keys, args))


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
    recon_config_dict = check_reconstruction_config_dict(config_dict)

    # IMPORTANT: Since we scale everything by the binning parameter when we set
    # up our simulation, we must also scale by the binning parameter here!

    # ASTRA toolbox uses clockwise rotation as positive. If the scanner rotates
    # counter clockwise, we need to add a negative sign here.
    projection_angles = recon_config_dict[
        "sample_rotation_direction"
    ] * np.linspace(
        0,
        np.deg2rad(recon_config_dict["scanning_angle"]),
        recon_config_dict["number_of_projections"],
    )

    scale_factor = compute_astra_scale_factor(
        recon_config_dict["distance_source_origin"],
        recon_config_dict["distance_origin_detector"],
        recon_config_dict["detector_pixel_size"] * recon_config_dict["binning"],
    )

    vol_geo = astra.create_vol_geom(
        recon_config_dict["detector_columns"] // recon_config_dict["binning"],
        recon_config_dict["detector_columns"] // recon_config_dict["binning"],
        recon_config_dict["detector_rows"] // recon_config_dict["binning"],
    )

    proj_geo = astra.creators.create_proj_geom(
        "cone",
        recon_config_dict["detector_pixel_size"]
        * recon_config_dict["binning"]
        * scale_factor,
        recon_config_dict["detector_pixel_size"]
        * recon_config_dict["binning"]
        * scale_factor,
        recon_config_dict["detector_rows"] // recon_config_dict["binning"],
        recon_config_dict["detector_columns"] // recon_config_dict["binning"],
        projection_angles,
        recon_config_dict["distance_source_origin"] * scale_factor,
        recon_config_dict["distance_origin_detector"] * scale_factor,
    )
    proj_id = astra.data3d.create("-sino", proj_geo, data=sinograms)
    rec_id = astra.data3d.create("-vol", vol_geo, data=0)
    alg_cfg = astra.astra_dict(recon_config_dict["reconstruction_algorithm"])
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
    unit_scale = 1000 / gvxr.getUnitOfLength(
        recon_config_dict["scanner_length_unit"]
    )

    # Rescale to get attenuation coefficient in cm^-1.
    reconstruction *= unit_scale / 100

    # Rotate the coordinates 90 degrees around the positive z-axis.
    if align_coordinates:
        reconstruction = np.swapaxes(reconstruction, 1, 2)
        reconstruction = np.flip(reconstruction, axis=1)

    return reconstruction
