import os
import signal
import time
import sys
from multiprocessing import Process
import numpy as np
import random
import tifffile
from x_ray_simulation import generate_sinograms
from tomographic_reconstruction import perform_tomographic_reconstruction
from textile_generation import generate_woven_composite_sample
from segmentation import segment_reconstruction
from hdf5_utils import save_data


"""
This files contains routines for running several simulations in batches
(including domain randomization). Data will be added to a database based on
hdf5 files. If a database already exists data is appended, otherwise the
database is created.
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


def generate_layer2layer_config(
    width_to_spacing_ratio=[0.85, 0.98],
    weft_to_warp_ratio=[0.2, 0.8],
    yarns_per_layer=[2, 10],
    number_of_yarn_layers=[4, 8],
    unit_cell_side_length=[5.0, 22.0],
    unit_cell_thickness=[2.0, 6.0],
    weave_complexity=[0.05, 0.45],
    tiling=[1, 3],
    shift_unit_cell=True,
    deform_offset=[0.025, 0.075],
    deform_scaling=[2.5, 7.5],
    deform_rotate=[2.5, 7.5],
    textile_resolution=12,
    cut_mesh=0.5,
    phase_elements=[[6], [6], [6, 1, 17, 8]],
    phase_ratios=[[1.0], [1.0], [0.404, 0.481, 0.019, 0.096]],
    phase_densities=[[1.78, 1.82], [1.78, 1.82], [1.06, 1.10]],
    **kwargs  # Needed to call by dict (will contain more than these args)
):
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

        phase_elements (list(list(int))): The element numbers of the
                                             constituents of the different
                                             sample material phases.

        phase_ratios (list(list(float))): The relative amount of the
                                             constituent elements of the
                                             different material phases.

        phase_densities (list(list(float))): The upper and lower bounds for the
                                             densities of the different phases.

    Returns:
        sample_config_dict (dict): A dict of the generated geometry options.
    """
    sample_config_dict = {}
    sample_config_dict["weave_type"] = "layer2layer"
    sample_config_dict["shift_unit_cell"] = shift_unit_cell
    sample_config_dict["weft_yarns_per_layer"] = np.random.randint(
        yarns_per_layer[0], high=yarns_per_layer[1]
    )
    sample_config_dict["warp_yarns_per_layer"] = np.random.randint(
        yarns_per_layer[0], high=yarns_per_layer[1]
    )
    sample_config_dict["number_of_yarn_layers"] = np.random.randint(
        number_of_yarn_layers[0], high=number_of_yarn_layers[1]
    )
    sample_config_dict["weft_to_warp_ratio"] = weft_to_warp_ratio[
        0
    ] + np.random.rand() * (weft_to_warp_ratio[1] - weft_to_warp_ratio[0])
    sample_config_dict["weft_width_to_spacing_ratio"] = width_to_spacing_ratio[
        0
    ] + np.random.rand() * (
        width_to_spacing_ratio[1] - width_to_spacing_ratio[0]
    )
    sample_config_dict["warp_width_to_spacing_ratio"] = width_to_spacing_ratio[
        0
    ] + np.random.rand() * (
        width_to_spacing_ratio[1] - width_to_spacing_ratio[0]
    )
    sample_config_dict["unit_cell_weft_length"] = unit_cell_side_length[
        0
    ] + np.random.rand() * (unit_cell_side_length[1] - unit_cell_side_length[0])
    sample_config_dict["unit_cell_warp_length"] = unit_cell_side_length[
        0
    ] + np.random.rand() * (unit_cell_side_length[1] - unit_cell_side_length[0])
    sample_config_dict["unit_cell_thickness"] = unit_cell_thickness[
        0
    ] + np.random.rand() * (unit_cell_thickness[1] - unit_cell_thickness[0])
    sample_config_dict["weave_pattern"] = generate_weave_pattern(
        sample_config_dict["weft_yarns_per_layer"],
        sample_config_dict["warp_yarns_per_layer"],
        int(
            (
                weave_complexity[0]
                + np.random.rand() * (weave_complexity[1] - weave_complexity[0])
            )
            * sample_config_dict["weft_yarns_per_layer"]
            * sample_config_dict["warp_yarns_per_layer"]
        ),
    )
    sample_config_dict["tiling"] = [
        np.random.randint(tiling[0], high=tiling[1]),
        np.random.randint(tiling[0], high=tiling[1]),
        np.random.randint(tiling[0], high=tiling[1]),
    ]
    sample_config_dict["deform"] = [
        deform_scaling[0]
        + np.random.rand() * (deform_scaling[1] - deform_scaling[0]),
        deform_scaling[0]
        + np.random.rand() * (deform_scaling[1] - deform_scaling[0]),
        deform_rotate[0]
        + np.random.rand() * (deform_rotate[1] - deform_rotate[0]),
        deform_offset[0]
        + np.random.rand() * (deform_offset[1] - deform_offset[0]),
        deform_offset[0]
        + np.random.rand() * (deform_offset[1] - deform_offset[0]),
        deform_offset[0]
        + np.random.rand() * (deform_offset[1] - deform_offset[0]),
        deform_scaling[0]
        + np.random.rand() * (deform_scaling[1] - deform_scaling[0]),
        deform_scaling[0]
        + np.random.rand() * (deform_scaling[1] - deform_scaling[0]),
        deform_rotate[0]
        + np.random.rand() * (deform_rotate[1] - deform_rotate[0]),
        deform_offset[0]
        + np.random.rand() * (deform_offset[1] - deform_offset[0]),
        deform_offset[0]
        + np.random.rand() * (deform_offset[1] - deform_offset[0]),
        deform_offset[0]
        + np.random.rand() * (deform_offset[1] - deform_offset[0]),
    ]
    sample_config_dict["textile_resolution"] = textile_resolution
    if np.random.rand() < cut_mesh:
        sample_config_dict["cut_mesh"] = "weft"
    else:
        sample_config_dict["cut_mesh"] = "warp"
    sample_config_dict["phase_elements"] = phase_elements
    sample_config_dict["phase_ratios"] = phase_ratios
    sample_config_dict["phase_densities"] = [
        phase_densities[0][0]
        + np.random.rand() * (phase_densities[0][1] - phase_densities[0][0]),
        phase_densities[1][0]
        + np.random.rand() * (phase_densities[1][1] - phase_densities[1][0]),
        phase_densities[2][0]
        + np.random.rand() * (phase_densities[2][1] - phase_densities[2][0]),
    ]
    return sample_config_dict


def generate_xray_config(
    offset=[[-0.2625, 0.2625], [-0.2625, 0.2625], [-0.2625, 0.2625]],
    tilt=[[-1.575, 1.575], [-1.575, 1.575], [-1.575, 1.575]],
    detector_pixel_size=[0.0320625, 0.0354375],
    detector_rows=2048,
    distance_source_origin=[76.0, 84.0],
    distance_origin_detector=[142.5, 157.5],
    number_of_projections=[1427, 1577],
    scanning_angle=[342, 378],
    anode_angle=[11.4, 12.6],
    tube_voltage=[38.0, 42.0],
    tube_power=[9.5, 10.5],
    filter_thickness=[0.95, 1.05],
    exposure_time=[4.75, 5.25],
    num_reference=30,
    filter_material="Al",
    target_material="W",
    energy_bin_width=0.5,
    photonic_noise=True,
    display=True,
    binning=4,
    threshold=0.000000001,
    rot_axis="x",
    sample_length_unit="mm",
    scanner_length_unit="mm",
    energy_unit="keV",
    sample_rotation_direction=0.5,
    reconstruction_algorithm="FDK_CUDA",
    **kwargs  # Needed to call by dict (will contain more than these args)
):
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

        exposure_time (list(float)): Lower and upper bounds of X-Ray exposure
                                     time in seconds.

        num_reference ((int)): Number of reference images to use for white field
                               average.

        filter_material (str): The chemical symbol of the filter material.

        target_material (str): The chemical symbol of the target material.

        energy_bin_width (float): The width of the spectrum bins in keV.

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
        offset[0][0] + np.random.rand() * (offset[0][1] - offset[0][0]),
        offset[1][0] + np.random.rand() * (offset[1][1] - offset[1][0]),
        offset[2][0] + np.random.rand() * (offset[2][1] - offset[2][0]),
    ]
    x_ray_config_dict["tilt"] = [
        tilt[0][0] + np.random.rand() * (tilt[0][1] - tilt[0][0]),
        tilt[1][0] + np.random.rand() * (tilt[1][1] - tilt[1][0]),
        tilt[2][0] + np.random.rand() * (tilt[2][1] - tilt[2][0]),
    ]
    x_ray_config_dict["detector_pixel_size"] = detector_pixel_size[
        0
    ] + np.random.rand() * (detector_pixel_size[1] - detector_pixel_size[0])
    x_ray_config_dict["detector_rows"] = detector_rows
    x_ray_config_dict["detector_columns"] = detector_rows
    x_ray_config_dict["distance_source_origin"] = distance_source_origin[
        0
    ] + np.random.rand() * (
        distance_source_origin[1] - distance_source_origin[0]
    )
    x_ray_config_dict["distance_origin_detector"] = distance_origin_detector[
        0
    ] + np.random.rand() * (
        distance_origin_detector[1] - distance_origin_detector[0]
    )
    x_ray_config_dict["number_of_projections"] = np.random.randint(
        number_of_projections[0], high=number_of_projections[1]
    )
    x_ray_config_dict["scanning_angle"] = scanning_angle[
        0
    ] + np.random.rand() * (scanning_angle[1] - scanning_angle[0])
    x_ray_config_dict["anode_angle"] = anode_angle[0] + np.random.rand() * (
        anode_angle[1] - anode_angle[0]
    )
    x_ray_config_dict["tube_voltage"] = tube_voltage[0] + np.random.rand() * (
        tube_voltage[1] - tube_voltage[0]
    )
    x_ray_config_dict["tube_power"] = tube_power[0] + np.random.rand() * (
        tube_power[1] - tube_power[0]
    )
    x_ray_config_dict["filter_thickness"] = filter_thickness[
        0
    ] + np.random.rand() * (filter_thickness[1] - filter_thickness[0])
    x_ray_config_dict["exposure_time"] = exposure_time[0] + np.random.rand() * (
        exposure_time[1] - exposure_time[0]
    )
    x_ray_config_dict["num_reference"] = num_reference
    x_ray_config_dict["filter_material"] = filter_material
    x_ray_config_dict["target_material"] = target_material
    x_ray_config_dict["energy_bin_width"] = energy_bin_width
    x_ray_config_dict["photonic_noise"] = photonic_noise
    x_ray_config_dict["display"] = display
    x_ray_config_dict["binning"] = binning
    x_ray_config_dict["threshold"] = threshold
    x_ray_config_dict["rot_axis"] = rot_axis
    x_ray_config_dict["sample_length_unit"] = sample_length_unit
    x_ray_config_dict["scanner_length_unit"] = scanner_length_unit
    x_ray_config_dict["energy_unit"] = energy_unit
    if np.random.rand() < sample_rotation_direction:
        x_ray_config_dict["sample_rotation_direction"] = 1
    else:
        x_ray_config_dict["sample_rotation_direction"] = -1
    x_ray_config_dict["reconstruction_algorithm"] = reconstruction_algorithm

    return x_ray_config_dict


def generate_config_dict(
    sim_id,
    parameters={},
):
    """Generate a valid Tex-Ray config dictionary associated with a sim-id.

    Args:
        sim_id (int): The simulation id to use for the config dictionary.

        parameters (dict): Dictionary containing simulation parameters.
                       Domain randomizable parameters are given as an lower and
                       upper bound.

    Keyword args:
        -

    Returns:
        config_dict (dict): A dictionary of Tex-Ray options.
    """
    config_dict = {}
    tex_ray_path = __file__.rstrip("batch_run.py")
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
        config_dict.update(generate_layer2layer_config(**parameters))

    config_dict.update(generate_xray_config(**parameters))

    return config_dict


def generate_woven_composite_sample_batch(num_samples, parameters={}):
    """Perform a batch run of generate_woven_composite_sample.

    Args:
        num_samples (int): The number of samples to generate.

    Keyword args:
        parameters (dict): Dictionary containing simulation parameters.
                       Domain randomizable parameters are given as an lower and
                       upper bound.

    Returns:
        configs (list[dict]): A list of the generated samples config dicts.

        exit_codes (list[int]): The generation process exit codes, 0 is success.
    """
    # Ignore sigint in children
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    configs = []
    jobs = []
    exit_codes = []

    for i in range(num_samples):
        config = generate_config_dict(i, parameters=parameters)
        configs.append(config)
        job = Process(target=generate_woven_composite_sample, args=(config,))
        job.start()
        jobs.append(job)

    # Let parent deal with sigint
    signal.signal(signal.SIGINT, original_sigint_handler)
    try:
        for job in jobs:
            job.join()
            exit_codes.append(job.exitcode)
    except KeyboardInterrupt:
        print("\nKeyboard interrupt, terminating all processes!")
        for job in jobs:
            job.terminate()
            job.join()
            job.close()
        clean_up_batch_files(num_samples)
        sys.exit(0)

    for job in jobs:
        job.close()

    return configs, exit_codes


def clean_up_batch_files(num_samples):
    """Clean up batch result and mesh files.
    DON'T CALL THIS BEFORE ADDING RESULTS TO DATABASE!

    Args:
        num_samples (int): The number of samples used in the batch.

    Keyword args:

    Returns:
        -
    """
    base_path = __file__.rstrip("batch_run.py")
    phases = ["weft", "warp", "matrix"]
    for sim_id in range(num_samples):
        for phase in phases:
            path = os.path.join(
                base_path, "meshes/" + phase + "_" + str(sim_id) + ".stl"
            )
            if os.path.exists(path):
                os.remove(path)

        path = os.path.join(
            base_path,
            "reconstructions/" + "reconstruction_" + str(sim_id) + ".tiff",
        )
        if os.path.exists(path):
            os.remove(path)
        path = os.path.join(
            base_path,
            "segmentations/" + "segmentation_" + str(sim_id) + ".tiff",
        )
        if os.path.exists(path):
            os.remove(path)
    return None


def run_batch(
    data_base_path,
    num_process=1,
    chunk_size=10,
    parameters={},
):
    """Run a batch of tex-ray simulations based on a config dictionary, and
    store the results in a hdf5 database.

    Args:
        master_config (dict): A valid Tex-Ray config dict that will be used as
                              the template for generating the configuration
                              dict.

        data_base_path (str): The path to the hdf5 database folder. If it does
                              not exist it is created.

    Keyword args:
        num_process (int): The number of processes to run in the batch. Decides
                           the number of samples that is produced per batch.

        parameters (dict): Dictionary containing simulation parameters.
                       Domain randomizable parameters are given as an lower and
                       upper bound.

        chunk_size (int): The chunk size. This argument is only used when
                          creating a new database.

    Returns:
        -
    """
    print("\nStarting to process a batch of size " + str(num_process) + ".")
    t0 = time.time()

    config_dicts, exit_codes = generate_woven_composite_sample_batch(
        num_process, parameters=parameters
    )

    num_success = 0
    try:
        for exit_code, sim_id in zip(exit_codes, range(num_process)):
            if exit_code == 0:
                num_success += 1
                sinograms = generate_sinograms(config_dicts[sim_id])
                reconstruction = perform_tomographic_reconstruction(
                    sinograms, config_dicts[sim_id]
                )
                tifffile.imwrite(
                    config_dicts[sim_id]["reconstruction_output_path"],
                    reconstruction,
                )
                del sinograms, reconstruction

                segmentation = segment_reconstruction(config_dicts[sim_id])
                tifffile.imwrite(
                    config_dicts[sim_id]["segmentation_output_path"],
                    segmentation,
                )
                del segmentation

                save_data(
                    data_base_path, config_dicts[sim_id], chunk_size=chunk_size
                )

    except KeyboardInterrupt:
        print("\nKeyboard interrupt, terminating all processes!")
        clean_up_batch_files(num_process)
        sys.exit(0)

    clean_up_batch_files(num_process)
    t1 = time.time()
    print(
        "\nSucessfully processed "
        + str(num_success)
        + " out of "
        + str(num_process)
        + " samples."
    )
    print("\nAverage time per sample: " + str((t1 - t0) / num_success) + " s.")

    return None


if __name__ == "__main__":
    database_path = "./tex-ray/dbase"
    num_process = 10
    chunk_size = 10
    parameters = {}

    while True:
        job = (
            Process(  # We wrap each batch in a process to prevent memory leak.
                target=run_batch,
                args=(database_path,),  # Comma inside parenthesis is important!
                kwargs={
                    "num_process": num_process,
                    "chunk_size": chunk_size,
                    "parameters": parameters,
                },
            )
        )
        job.start()
        job.join()
        job.close()
