import numpy as np
import astra
from gvxrPython3 import gvxr

from x_ray.x_ray_util import *

"""
This file contains the main routines for generating sinograms of woven composite
material meshes and the tomographic reconstructions of the same. 
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
    gvxr.createOpenGLContext()

    set_up_detector(
        config_dict["distance_origin_detector"],
        config_dict["detector_columns"],
        config_dict["detector_rows"],
        config_dict["detector_pixel_size"],
        binning=config_dict["binning"],
        length_unit=config_dict["scanner_length_unit"],
    )

    energy_bins, photon_flux = generate_xray_spectrum(
        config_dict["anode_angle"],
        config_dict["energy_bin_width"],
        config_dict["tube_voltage"],
        config_dict["tube_power"],
        config_dict["exposure_time"],
        config_dict["distance_source_origin"]
        + config_dict["distance_origin_detector"],
        config_dict["offset"],
        config_dict["detector_pixel_size"],
        filter_thickness=config_dict["filter_thickness"],
        filter_material=config_dict["filter_material"],
        target_material=config_dict["target_material"],
        length_unit=config_dict["scanner_length_unit"],
    )

    set_up_xray_source(
        config_dict["distance_source_origin"],
        -1,
        energy_bins,
        photon_flux,
        length_unit=config_dict["scanner_length_unit"],
        energy_unit=config_dict["energy_unit"],
    )
    set_up_sample(
        config_dict["weft_path"],
        config_dict["weft_elements"],
        config_dict["weft_ratios"],
        config_dict["weft_density"],
        config_dict["warp_path"],
        config_dict["warp_elements"],
        config_dict["warp_ratios"],
        config_dict["warp_density"],
        config_dict["matrix_path"],
        config_dict["matrix_elements"],
        config_dict["matrix_ratios"],
        config_dict["matrix_density"],
        config_dict["rot_axis"],
        config_dict["tiling"],
        config_dict["offset"],
        config_dict["tilt"],
        length_unit=config_dict["sample_length_unit"],
    )
    raw_projections = perform_tomographic_scan(
        config_dict["number_of_projections"],
        config_dict["scanning_angle"],
        display=config_dict["display"],
        photonic_noise=config_dict["photonic_noise"],
    )
    # After finishing the tomographic constructions it is safe to close window.
    gvxr.destroyWindow()

    flat_field_image = measure_flat_field(
        photonic_noise=config_dict["photonic_noise"],
        num_reference=config_dict["num_reference"],
    )
    dark_field_image = measure_dark_field()
    corrected_projections = perform_flat_field_correction(
        raw_projections, flat_field_image, dark_field_image
    )
    neg_log_projections = neg_log_transform(
        corrected_projections, config_dict["threshold"]
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
    projection_angles = np.linspace(
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
