import os
import sys
import json
import numpy as np
import random
from yarn_attenuation_util import *
from hdf5_utils import get_database_shape

"""
This files contains routines for generating textomos config files from a base
config. A base config file contains all possible textomos options and
corresponding values. Values that can be domain randomized are given as upper
and lower bounds. These values will be uniformly sampled within these limits.
"""


def generate_weave_pattern(
    weft_yarns_per_layer,
    warp_yarns_per_layer,
    weave_complexity,
):
    """Generate a random weave pattern.

    Args:
        weft_yarns_per_layer (int): The number of weft yarns per weave layer.

        warp_yarns_per_layer (int): The number of warp yarns per weave layer.

        weave_complexity (int): The number of cross overs per unit cell.
    Keyword args:
        -

    Returns:
        weave_pattern (list[list[int]]): A weave pattern compatible with the
                                         generate_woven_composite_sample.
    """
    # list all possible cross-over coordinates and shuffle for random weave
    possible_crossover = []
    for i in range(warp_yarns_per_layer):
        for j in range(weft_yarns_per_layer):
            possible_crossover.append([i, j])
    random.shuffle(possible_crossover)

    # We randomly select push down or up (-1 vs 1)
    weave_pattern = []
    for i in range(weave_complexity):
        possible_crossover[i].append([-1, 1][random.randint(0, 1)])
        weave_pattern.append(possible_crossover[i])

    return weave_pattern


def generate_layer2layer_geometry_config(**kwargs):
    """Generate options pertaining the geometry (weave, yarn size, etc.) and the
       material properties of a layer2layer composite sample with domain
       randomization.

    Args:
        -

    Keyword args:
        width_to_spacing_ratio (list[float]): Lower and upper bound for
                                              weft/warp width to spacing ratio.

        weft_to_warp_ratio (list[float]): Lower and upper bound for weft to warp
                                          ratio.

        yarns_per_layer (list[int]): Lower and upper bound for weft/warp number
                                     of yarns per layer.

        number_of_yarn_layers (list[int]): Lower and upper bound for number of
                                           yarn layers.

        unit_cell_side_length (list[float]): Lower and upper bound for the side
                                             length of unit cellin the yarn
                                             plane.

        unit_cell_thickness (list[float]): Lower and upper bound for unit cell
                                           thickness.

        weave_complexity (list[float]): Lower and upper bound for number of push
                                        up/down operations inside a unit cell.
                                        It is given as a percentage of yarn
                                        meeting points.

        tiling (list[int]): Lower and upper bound for number of unit cell
                            repetitions in any direction.

        shift_unit_cell (bool): Will randomly shift the unit cell between
                                z-layer tiles.

        deform_offset (list(float)): Lower and upper bound for cross section
                                      offset deformation.

        deform_scaling (list(float)): Lower and upper bound for cross section
                                      scaling deformation.

        deform_rotate (list(float)): Lower and upper bound for cross section
                                     rotation deformation.

        textile_resolution (int): Texgen mesh resolution.

        cut_mesh (float): The probability to cut warp out of weft (does the
                          opposite otherwise). Should be between 0 and 1.

    Returns:
        sample_config_dict (dict): A dict of the generated geometry options.
    """
    sample_config_dict = {}
    sample_config_dict["weave_type"] = kwargs["weave_type"]
    sample_config_dict["shift_unit_cell"] = kwargs["shift_unit_cell"]
    sample_config_dict["weft_yarns_per_layer"] = np.random.randint(
        kwargs["yarns_per_layer"][0], high=kwargs["yarns_per_layer"][1]
    )
    sample_config_dict["warp_yarns_per_layer"] = np.random.randint(
        kwargs["yarns_per_layer"][0], high=kwargs["yarns_per_layer"][1]
    )
    sample_config_dict["number_of_yarn_layers"] = np.random.randint(
        kwargs["number_of_yarn_layers"][0],
        high=kwargs["number_of_yarn_layers"][1],
    )
    sample_config_dict["weft_to_warp_ratio"] = kwargs["weft_to_warp_ratio"][
        0
    ] + np.random.rand() * (
        kwargs["weft_to_warp_ratio"][1] - kwargs["weft_to_warp_ratio"][0]
    )
    sample_config_dict["weft_width_to_spacing_ratio"] = kwargs[
        "width_to_spacing_ratio"
    ][0] + np.random.rand() * (
        kwargs["width_to_spacing_ratio"][1]
        - kwargs["width_to_spacing_ratio"][0]
    )
    sample_config_dict["warp_width_to_spacing_ratio"] = kwargs[
        "width_to_spacing_ratio"
    ][0] + np.random.rand() * (
        kwargs["width_to_spacing_ratio"][1]
        - kwargs["width_to_spacing_ratio"][0]
    )
    sample_config_dict["unit_cell_weft_length"] = kwargs[
        "unit_cell_side_length"
    ][0] + np.random.rand() * (
        kwargs["unit_cell_side_length"][1] - kwargs["unit_cell_side_length"][0]
    )
    sample_config_dict["unit_cell_warp_length"] = kwargs[
        "unit_cell_side_length"
    ][0] + np.random.rand() * (
        kwargs["unit_cell_side_length"][1] - kwargs["unit_cell_side_length"][0]
    )
    sample_config_dict["unit_cell_thickness"] = kwargs["unit_cell_thickness"][
        0
    ] + np.random.rand() * (
        kwargs["unit_cell_thickness"][1] - kwargs["unit_cell_thickness"][0]
    )
    sample_config_dict["weave_pattern"] = generate_weave_pattern(
        sample_config_dict["weft_yarns_per_layer"],
        sample_config_dict["warp_yarns_per_layer"],
        int(
            (
                kwargs["weave_complexity"][0]
                + np.random.rand()
                * (
                    kwargs["weave_complexity"][1]
                    - kwargs["weave_complexity"][0]
                )
            )
            * sample_config_dict["weft_yarns_per_layer"]
            * sample_config_dict["warp_yarns_per_layer"]
        ),
    )
    sample_config_dict["tiling"] = [
        np.random.randint(kwargs["tiling"][0], high=kwargs["tiling"][1]),
        np.random.randint(kwargs["tiling"][0], high=kwargs["tiling"][1]),
        np.random.randint(kwargs["tiling"][0], high=kwargs["tiling"][1]),
    ]
    sample_config_dict["deform"] = [
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_rotate"][0]
        + np.random.rand()
        * (kwargs["deform_rotate"][1] - kwargs["deform_rotate"][0]),
        kwargs["deform_offset"][0]
        + np.random.rand()
        * (kwargs["deform_offset"][1] - kwargs["deform_offset"][0]),
        kwargs["deform_offset"][0]
        + np.random.rand()
        * (kwargs["deform_offset"][1] - kwargs["deform_offset"][0]),
        kwargs["deform_offset"][0]
        + np.random.rand()
        * (kwargs["deform_offset"][1] - kwargs["deform_offset"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_rotate"][0]
        + np.random.rand()
        * (kwargs["deform_rotate"][1] - kwargs["deform_rotate"][0]),
        kwargs["deform_offset"][0]
        + np.random.rand()
        * (kwargs["deform_offset"][1] - kwargs["deform_offset"][0]),
        kwargs["deform_offset"][0]
        + np.random.rand()
        * (kwargs["deform_offset"][1] - kwargs["deform_offset"][0]),
        kwargs["deform_offset"][0]
        + np.random.rand()
        * (kwargs["deform_offset"][1] - kwargs["deform_offset"][0]),
    ]
    sample_config_dict["textile_resolution"] = kwargs["textile_resolution"]
    if np.random.rand() < kwargs["cut_mesh"]:
        sample_config_dict["cut_mesh"] = "weft"
    else:
        sample_config_dict["cut_mesh"] = "warp"
    return sample_config_dict


def generate_orthogonal_geometry_config(**kwargs):
    """Generate options pertaining the geometry (yarn size, shape, etc.) with
    domain randomization.

    Args:
        -

    Keyword args:
        warp_width_to_spacing_ratio (list[float]): Lower and upper bound for
                                                   warp width to spacing ratio.

        weft_width_to_spacing_ratio (list[float]): Lower and upper bound for
                                                   weft width to spacing ratio.

        binder_width_to_spacing_ratio (list[float]): Lower and upper bound for
                                                     binder width to spacing
                                                     ratio.

        weft_super_ellipse_power (list[float]): Lower and upper bound for base
                                                weft super ellipse power.

        warp_super_ellipse_power (list[float]): Lower and upper bound for base
                                                warp super ellipse power.

        binder_super_ellipse_power (list[float]): Lower and upper bound for base
                                                  binder super ellipse power.


        binder_thickness_to_spacing_ratio (list[float]): Lower and upper bound
                                                         for binder thickness to
                                                         spacing ratio.

        weft_to_warp_ratio (list[float]): Lower and upper bound for weft to warp
                                          ratio.

        yarns_per_layer (list[int]): Lower and upper bound for weft/warp number
                                     of yarns per layer.

        number_of_yarn_layers (list[int]): Lower and upper bound for number of
                                           yarn layers.

        unit_cell_side_length (list[float]): Lower and upper bound for the side
                                             length of unit cellin the yarn
                                             plane.

        unit_cell_thickness (list[float]): Lower and upper bound for unit cell
                                           thickness.

        tiling (list[int]): Lower and upper bound for number of unit cell
                            repetitions in any direction.

        compaction (list[float]): Lower and upper bound for for unit cell
                                  compaction/dilation in any direction.

        deform_scaling (list(float)): Lower and upper bound for cross section
                                      scaling deformation.

        deform_rotate (list(float)): Lower and upper bound for cross section
                                     rotation deformation.

        textile_resolution (int): Texgen mesh resolution.


    Returns:
        sample_config_dict (dict): A dict of the generated geometry options.
    """
    sample_config_dict = {}
    sample_config_dict["weft_yarns_per_layer"] = np.random.randint(
        kwargs["yarns_per_layer"][0], high=kwargs["yarns_per_layer"][1]
    )
    sample_config_dict["warp_yarns_per_layer"] = np.random.randint(
        kwargs["yarns_per_layer"][0], high=kwargs["yarns_per_layer"][1]
    )
    sample_config_dict["number_of_yarn_layers"] = np.random.randint(
        kwargs["number_of_yarn_layers"][0],
        high=kwargs["number_of_yarn_layers"][1],
    )
    sample_config_dict["weft_to_warp_ratio"] = kwargs["weft_to_warp_ratio"][
        0
    ] + np.random.rand() * (
        kwargs["weft_to_warp_ratio"][1] - kwargs["weft_to_warp_ratio"][0]
    )
    sample_config_dict["weft_width_to_spacing_ratio"] = kwargs[
        "weft_width_to_spacing_ratio"
    ][0] + np.random.rand() * (
        kwargs["weft_width_to_spacing_ratio"][1]
        - kwargs["weft_width_to_spacing_ratio"][0]
    )
    sample_config_dict["warp_width_to_spacing_ratio"] = kwargs[
        "warp_width_to_spacing_ratio"
    ][0] + np.random.rand() * (
        kwargs["warp_width_to_spacing_ratio"][1]
        - kwargs["warp_width_to_spacing_ratio"][0]
    )
    sample_config_dict["binder_width_to_spacing_ratio"] = kwargs[
        "binder_width_to_spacing_ratio"
    ][0] + np.random.rand() * (
        kwargs["binder_width_to_spacing_ratio"][1]
        - kwargs["binder_width_to_spacing_ratio"][0]
    )
    sample_config_dict["weft_super_ellipse_power"] = kwargs[
        "weft_super_ellipse_power"
    ][0] + np.random.rand() * (
        kwargs["weft_super_ellipse_power"][1]
        - kwargs["weft_super_ellipse_power"][0]
    )
    sample_config_dict["warp_super_ellipse_power"] = kwargs[
        "warp_super_ellipse_power"
    ][0] + np.random.rand() * (
        kwargs["warp_super_ellipse_power"][1]
        - kwargs["warp_super_ellipse_power"][0]
    )
    sample_config_dict["binder_super_ellipse_power"] = kwargs[
        "binder_super_ellipse_power"
    ][0] + np.random.rand() * (
        kwargs["binder_super_ellipse_power"][1]
        - kwargs["binder_super_ellipse_power"][0]
    )
    sample_config_dict["weft_to_warp_ratio"] = kwargs["weft_to_warp_ratio"][
        0
    ] + np.random.rand() * (
        kwargs["weft_to_warp_ratio"][1] - kwargs["weft_to_warp_ratio"][0]
    )
    sample_config_dict["binder_thickness_to_spacing_ratio"] = kwargs[
        "binder_thickness_to_spacing_ratio"
    ][0] + np.random.rand() * (
        kwargs["binder_thickness_to_spacing_ratio"][1]
        - kwargs["binder_thickness_to_spacing_ratio"][0]
    )
    sample_config_dict["unit_cell_weft_length"] = kwargs[
        "unit_cell_side_length"
    ][0] + np.random.rand() * (
        kwargs["unit_cell_side_length"][1] - kwargs["unit_cell_side_length"][0]
    )
    sample_config_dict["unit_cell_warp_length"] = kwargs[
        "unit_cell_side_length"
    ][0] + np.random.rand() * (
        kwargs["unit_cell_side_length"][1] - kwargs["unit_cell_side_length"][0]
    )
    sample_config_dict["unit_cell_thickness"] = kwargs["unit_cell_thickness"][
        0
    ] + np.random.rand() * (
        kwargs["unit_cell_thickness"][1] - kwargs["unit_cell_thickness"][0]
    )
    sample_config_dict["tiling"] = [
        np.random.randint(kwargs["tiling"][0], high=kwargs["tiling"][1]),
        np.random.randint(kwargs["tiling"][0], high=kwargs["tiling"][1]),
        np.random.randint(kwargs["tiling"][0], high=kwargs["tiling"][1]),
    ]
    sample_config_dict["deform"] = [
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_rotate"][0]
        + np.random.rand()
        * (kwargs["deform_rotate"][1] - kwargs["deform_rotate"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_rotate"][0]
        + np.random.rand()
        * (kwargs["deform_rotate"][1] - kwargs["deform_rotate"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_scaling"][0]
        + np.random.rand()
        * (kwargs["deform_scaling"][1] - kwargs["deform_scaling"][0]),
        kwargs["deform_rotate"][0]
        + np.random.rand()
        * (kwargs["deform_rotate"][1] - kwargs["deform_rotate"][0]),
    ]
    sample_config_dict["textile_resolution"] = kwargs["textile_resolution"]
    sample_config_dict["compaction"] = [
        kwargs["compaction"][0]
        + np.random.rand()
        * (kwargs["compaction"][1] - kwargs["compaction"][0]),
        kwargs["compaction"][0]
        + np.random.rand()
        * (kwargs["compaction"][1] - kwargs["compaction"][0]),
        kwargs["compaction"][0]
        + np.random.rand()
        * (kwargs["compaction"][1] - kwargs["compaction"][0]),
    ]
    return sample_config_dict


def generate_attenuation_properties_config(**kwargs):
    """Generate options pertaining the attenuation properties of the generated
       sample's different phases. Specifically, phase densities and constituent
       atomic mass fractions. Volume fraction of fibers in the yarns are
       estimated by computing fiber area coverage / yarn area. Where the yarn
       area is estimated from a CT-scan in number of voxels (pixels).

    Args:
        -

    Keyword args:
        elements list([int]): The element numbers of the atoms in the material
                              constitutents. The order of the list must
                              correspond to the order given in atomic_weights
                              and the compound lists.

        atomic_weights (list[float]): The weight in atomic units of the
                                      atoms present in the materials. The order
                                      of the list must correspond to the order
                                      given in elements and the compound lists.

        fiber_density (list[float]): Lower and upper bound for the density of
                                     the yarns' fibers.

        fiber_diameter (list[float]): Lower and upper bound for the diameter of
                                      the yarns' fibers.

        num_fibers (list[list[int]]): Lower and upper bound for the number of
                                      fibers per yarn. There is one row per yarn
                                      type.

        fiber_compound (list[int]): The number of atoms per type in the fiber
                                    material, corresponding to the elements in
                                    elements and the weights in and
                                    atomic_weights.

        matrix_density (list[float]): Lower and upper bound for the density of
                                      cured matrix material.

        matrix_compounds (list[list[int]]):
            The number of atoms per type for the matrix ingredients,
            corresponding to elements in elements and the weights in
            atomic_weights. Each row corresponds to one matrix ingredient.

        matrix_compounds_mixing_ratios (list[list[float]]):
            Lower and upper bounds for the mixing ratio by weight for the matrix
            constituent chemicals. Each row corresponds to a chemical.

        voxel_size (float): The side length of the voxels in the CT-scan used
                            to estimate the yarn areas.

        voxel_areas (list[list[int]]): Lower and upper bound for the number
                                      of voxels covered by a yarn cross
                                      section. There is one row per yarn type.

    Returns:
        attenuation_properties_config_dict (dict): A dict of the generated
                                                   sample material X-Ray
                                                   attenuation properties.
    """
    attenuation_properties_config_dict = {}

    yarn_fiber_volume_fractions = []
    for voxel_area, num_fiber in zip(
        kwargs["voxel_areas"], kwargs["num_fibers"]
    ):
        yarn_fiber_volume_fractions.append(
            estimate_fiber_volume_fraction(
                kwargs["voxel_size"],
                voxel_area[0]
                + np.random.rand() * (voxel_area[1] - voxel_area[0]),
                kwargs["fiber_diameter"][0]
                + np.random.rand()
                * (kwargs["fiber_diameter"][1] - kwargs["fiber_diameter"][0]),
                num_fiber[0] + np.random.rand() * (num_fiber[1] - num_fiber[0]),
            )
        )

    mixing_ratios = []
    for weight in kwargs["matrix_compounds_mixing_ratios"]:
        mixing_ratios.append(
            weight[0] + np.random.rand() * (weight[1] - weight[0])
        )

    matrix_atomic_mass_fractions = compute_matrix_atomic_mass_fractions(
        np.array(kwargs["atomic_weights"]),
        np.array(kwargs["matrix_compounds"]),
        np.array(mixing_ratios),
    )

    fiber_atomic_mass_fractions = compute_fiber_atomic_mass_fractions(
        np.array(kwargs["atomic_weights"]), np.array(kwargs["fiber_compound"])
    )

    sampled_fiber_density = kwargs["fiber_density"][0] + np.random.rand() * (
        kwargs["fiber_density"][1] - kwargs["fiber_density"][0]
    )
    sampled_matrix_density = kwargs["matrix_density"][0] + np.random.rand() * (
        kwargs["matrix_density"][1] - kwargs["matrix_density"][0]
    )

    phase_ratios = []
    phase_elements = []
    for yarn_fiber_volume_fraction in yarn_fiber_volume_fractions:
        phase_ratios.append(
            compute_yarn_atomic_mass_fractions(
                fiber_atomic_mass_fractions,
                sampled_fiber_density,
                yarn_fiber_volume_fraction,
                matrix_atomic_mass_fractions,
                sampled_matrix_density,
            ).tolist()
        )
        phase_elements.append(kwargs["elements"])
    phase_ratios.append(matrix_atomic_mass_fractions.tolist())
    phase_elements.append(kwargs["elements"])
    attenuation_properties_config_dict["phase_ratios"] = phase_ratios
    attenuation_properties_config_dict["phase_elements"] = phase_elements

    phase_densities = []
    for yarn_fiber_volume_fraction in yarn_fiber_volume_fractions:
        phase_densities.append(
            compute_yarn_density(
                sampled_fiber_density,
                yarn_fiber_volume_fraction,
                sampled_matrix_density,
            )
        )
    phase_densities.append(sampled_matrix_density)
    attenuation_properties_config_dict["phase_densities"] = phase_densities

    return attenuation_properties_config_dict


def generate_xray_config(**kwargs):
    """Generate options pertaining the X-Ray simulation (scanner properties,
       sample placement, etc.) with domain randomization.

    Args:
        -

    Keyword args:
        offset (list(list(float))): Lower and upper bounds of sample position
                                    offsets.

        tilt (list(list(float))): Lower and upper bounds of sample rotation
                                  offsets.

        detector_pixel_size (list(float)): Lower and upper bounds of detector
                                           pixel size.

        detector_rows (int): The number of detector pixels per row/column.

        distance_source_origin (list(float)): Lower and upper bound for the
                                              distance between X-Ray source and
                                              center of rotation.

        distance_source_origin (list(float)): Lower and upper bound for the
                                              distance between the center of
                                              rotation and the X-Ray detector.

        number_of_projections (list(int)): Lower and upper bounds of number of
                                           X-Ray projections.

        scanning_angle (list(float)): Lower and upper bounds of tomography
                                      scanning angle in degrees.

        anode_angle (list(float)):  Lower and upper bounds of X-Ray tube anode
                                    angle.

        tube_voltage (list(float)): Lower and upper bounds of X-Ray tube voltage
                                    in kV.

        tube_power (list(float)): Lower and upper bounds of X-Ray tube power in
                                  W.

        filter_thickness (list(float)): Lower and upper bounds of X-Ray filter
                                        thickness.

        point_spread (list(float)): Lower and upper bounds of system point
                                    spread kernel standard deviation.

        exposure_time (list(float)): Lower and upper bounds of X-Ray exposure
                                     time in seconds.

        num_reference ((int)): Number of reference images to use for white field
                               average.

        filter_material (str): The chemical symbol of the filter material.

        target_material (str): The chemical symbol of the target material.

        energy_bin_width (float): The width of the spectrum bins in keV.

        photonic_noie (Bool): Will add enable photonic noise if true.

        binning (int): The detector binning value.

        threshold (float): Threshold value to use while performing negative log
                           transform.

        rot_axis (str): Axis in mesh coordinates to rotate around while
                        scanning.

        sample_length_unit (str): The unit of length to use for setting up the
                                  sample (unit used in mesh). Should in
                                  principle be "mm".

        scanner_length_unit (str): The unit of length to use for setting up the
                                   scanner. Should in principle be "mm".

        energy_unit (str): The unit for X-Ray photon energy. Should in principle
                           be "keV" for spekpy compatibility.

        sample_rotation_direction (float): The probability to rotate the sample
                                           clockwise (opposite otherwise).
                                           Should be between 0 and 1.

        reconstruction_algorithm (str): The ASTRA tomographic reconstruction
                                        algorithm to use.


    Returns:
        x_ray_config_dict (dict): A dictionary of the generated X-Ray options.

    """
    x_ray_config_dict = {}
    x_ray_config_dict["offset"] = [
        kwargs["offset"][0][0]
        + np.random.rand() * (kwargs["offset"][0][1] - kwargs["offset"][0][0]),
        kwargs["offset"][1][0]
        + np.random.rand() * (kwargs["offset"][1][1] - kwargs["offset"][1][0]),
        kwargs["offset"][2][0]
        + np.random.rand() * (kwargs["offset"][2][1] - kwargs["offset"][2][0]),
    ]
    x_ray_config_dict["tilt"] = [
        kwargs["tilt"][0][0]
        + np.random.rand() * (kwargs["tilt"][0][1] - kwargs["tilt"][0][0]),
        kwargs["tilt"][1][0]
        + np.random.rand() * (kwargs["tilt"][1][1] - kwargs["tilt"][1][0]),
        kwargs["tilt"][2][0]
        + np.random.rand() * (kwargs["tilt"][2][1] - kwargs["tilt"][2][0]),
    ]
    x_ray_config_dict["detector_pixel_size"] = kwargs["detector_pixel_size"][
        0
    ] + np.random.rand() * (
        kwargs["detector_pixel_size"][1] - kwargs["detector_pixel_size"][0]
    )
    x_ray_config_dict["detector_rows"] = kwargs["detector_rows"]
    x_ray_config_dict["detector_columns"] = kwargs["detector_rows"]
    x_ray_config_dict["distance_source_origin"] = kwargs[
        "distance_source_origin"
    ][0] + np.random.rand() * (
        kwargs["distance_source_origin"][1]
        - kwargs["distance_source_origin"][0]
    )
    x_ray_config_dict["distance_origin_detector"] = kwargs[
        "distance_origin_detector"
    ][0] + np.random.rand() * (
        kwargs["distance_origin_detector"][1]
        - kwargs["distance_origin_detector"][0]
    )
    x_ray_config_dict["number_of_projections"] = np.random.randint(
        kwargs["number_of_projections"][0],
        high=kwargs["number_of_projections"][1],
    )
    x_ray_config_dict["scanning_angle"] = kwargs["scanning_angle"][
        0
    ] + np.random.rand() * (
        kwargs["scanning_angle"][1] - kwargs["scanning_angle"][0]
    )
    x_ray_config_dict["anode_angle"] = kwargs["anode_angle"][
        0
    ] + np.random.rand() * (kwargs["anode_angle"][1] - kwargs["anode_angle"][0])
    x_ray_config_dict["tube_voltage"] = kwargs["tube_voltage"][
        0
    ] + np.random.rand() * (
        kwargs["tube_voltage"][1] - kwargs["tube_voltage"][0]
    )
    x_ray_config_dict["tube_power"] = kwargs["tube_power"][
        0
    ] + np.random.rand() * (kwargs["tube_power"][1] - kwargs["tube_power"][0])
    x_ray_config_dict["filter_thickness"] = kwargs["filter_thickness"][
        0
    ] + np.random.rand() * (
        kwargs["filter_thickness"][1] - kwargs["filter_thickness"][0]
    )
    x_ray_config_dict["exposure_time"] = kwargs["exposure_time"][
        0
    ] + np.random.rand() * (
        kwargs["exposure_time"][1] - kwargs["exposure_time"][0]
    )
    x_ray_config_dict["point_spread"] = kwargs["point_spread"][
        0
    ] + np.random.rand() * (
        kwargs["point_spread"][1] - kwargs["point_spread"][0]
    )
    x_ray_config_dict["num_reference"] = kwargs["num_reference"]
    x_ray_config_dict["filter_material"] = kwargs["filter_material"]
    x_ray_config_dict["target_material"] = kwargs["target_material"]
    x_ray_config_dict["energy_bin_width"] = kwargs["energy_bin_width"]
    x_ray_config_dict["photonic_noise"] = kwargs["photonic_noise"]
    x_ray_config_dict["display"] = kwargs["display"]
    x_ray_config_dict["binning"] = kwargs["binning"]
    x_ray_config_dict["threshold"] = kwargs["threshold"]
    x_ray_config_dict["rot_axis"] = kwargs["rot_axis"]
    x_ray_config_dict["sample_length_unit"] = kwargs["sample_length_unit"]
    x_ray_config_dict["scanner_length_unit"] = kwargs["scanner_length_unit"]
    x_ray_config_dict["energy_unit"] = kwargs["energy_unit"]
    if np.random.rand() < kwargs["sample_rotation_direction"]:
        x_ray_config_dict["sample_rotation_direction"] = 1
    else:
        x_ray_config_dict["sample_rotation_direction"] = -1
    x_ray_config_dict["reconstruction_algorithm"] = kwargs[
        "reconstruction_algorithm"
    ]

    return x_ray_config_dict


def generate_config(
    sim_id,
    config_path,
    parameters={},
):
    """Generate a valid Textomos config associated with a sim-id and write to
       file.

    Args:
        sim_id (int): The simulation id to use for the config dictionary.

        config_path (str): The path to where to save the config.

        parameters (dict): Dictionary containing simulation parameters.
                       Domain randomizable parameters are given as an lower and
                       upper bound.

    Keyword args:
        -

    Returns:
        None
    """
    config_dict = {}
    tex_ray_path = __file__.rstrip("generate_config.py")
    # We load these paths here since they are identical for all weave types.
    config_dict["reconstruction_output_path"] = os.path.join(
        tex_ray_path,
        "reconstructions/" + "reconstruction_" + str(sim_id) + ".tiff",
    )
    config_dict["segmentation_output_path"] = os.path.join(
        tex_ray_path,
        "segmentations/" + "segmentation_" + str(sim_id) + ".tiff",
    )

    if parameters.get("weave_type", "layer2layer") == "layer2layer":
        phases = ["weft", "warp", "matrix"]
        mesh_paths = []
        for phase in phases:
            mesh_paths.append(
                os.path.join(
                    tex_ray_path, "meshes/" + phase + "_" + str(sim_id) + ".stl"
                )
            )
        config_dict["mesh_paths"] = mesh_paths
        config_dict.update(generate_layer2layer_geometry_config(**parameters))
    elif parameters["weave_type"] == "orthogonal":
        phases = ["weft", "warp", "binder", "matrix"]
        mesh_paths = []
        for phase in phases:
            mesh_paths.append(
                os.path.join(
                    tex_ray_path, "meshes/" + phase + "_" + str(sim_id) + ".stl"
                )
            )
        config_dict["mesh_paths"] = mesh_paths
        config_dict.update(generate_orthogonal_geometry_config(**parameters))

    config_dict.update(generate_attenuation_properties_config(**parameters))

    config_dict.update(generate_xray_config(**parameters))

    file_name = "input_" + str(sim_id) + ".json"

    with open(os.path.join(config_path, file_name), "w") as outfile:
        json.dump(config_dict, outfile)

    return None


if __name__ == "__main__":
    sim_id = sys.argv[1]
    path = sys.argv[2]
    base_input_path = sys.argv[3]
    database_path = sys.argv[4]
    generate_until = int(sys.argv[5])

    # If the database exists and is already at or above final size; terminate.
    if os.path.isdir(database_path):
        if get_database_shape(database_path)[0] >= generate_until:
            with open("/textomos/input/finished", "w") as f:
                f.write("FINISHED")
            sys.exit(0)

    with open(base_input_path) as f:
        parameters = json.load(f)
    generate_config(sim_id, path, parameters=parameters)
