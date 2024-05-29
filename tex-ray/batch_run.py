import os
import copy
import signal
import time
import sys
from multiprocessing import Process
import numpy as np
import random
import tifffile
import json
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
    base_pattern,
    domain_randomization=0.0,
):
    """Generate a random weave pattern.

    Args:
        weft_yarns_per_layer (int): The number of weft yarns per weave layer.

        warp_yarns_per_layer (int): The number of warp yarns per weave layer.

        base_pattern (list[list[int]]): The base config weave pattern used to
                                        gauge the weave complexity.

    Keyword args:
        domain_randomization (float): A number larger or equal to  0 and less
                                      than 1 that determines the amount of
                                      domain randomization to be appl. to base
                                      weave complexity.

    Returns:
        weave_pattern (list[list[int]]): A weave pattern compatible with the
                                         generate_woven_composite_sample.
    """
    weave_complexity = int(
        np.clip(
            len(base_pattern)
            * (1 - domain_randomization * (1 - 2 * np.random.rand())),
            1,
            None,
        )
    )

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


def generate_config_dict(
    sim_id,
    base_config,
    domain_randomization=0.0,
):
    """Generate a valid Tex-Ray config dictionary associated with a sim-id.
    The dictionary is generated based on a given config file, and domain
    randomization can be added.

    Args:
        sim_id (int): The simulation id to use for the config dictionary.

        base_config (dict): A valid Tex-Ray config dict that will be used as the
                            template for generating the configuration dict.

    Keyword args:
        domain_randomization (float): A number larger or equal to  0 and less
                                      than 1 that determines the amount of
                                      domain randomization to be appl. to base
                                      config.

    Returns:
        config_dict (dict): A list of the generated samples config dicts.
    """
    config_dict = copy.deepcopy(base_config)

    tex_ray_path = __file__.rstrip("batch_run.py")

    phases = ["weft", "warp", "matrix"]
    for phase in phases:
        config_dict[phase + "_path"] = os.path.join(
            tex_ray_path, "meshes/" + phase + "_" + str(sim_id) + ".stl"
        )
    config_dict["reconstruction_output_path"] = os.path.join(
        tex_ray_path,
        "reconstructions/" + "reconstruction_" + str(sim_id) + ".tiff",
    )
    config_dict["segmentation_output_path"] = os.path.join(
        tex_ray_path,
        "segmentations/" + "segmentation_" + str(sim_id) + ".tiff",
    )

    if domain_randomization == 0.0:
        return config_dict

    bounded_floats = [
        "weft_width_to_spacing_ratio",
        "warp_width_to_spacing_ratio",
        "weft_to_warp_ratio",
    ]
    skipped_floats = ["scanning_angle", "energy_bin_width", "threshold"]
    skipped_ints = [
        "sample_rotation_direction",
        "textile_resolution",
        "detector_rows",
        "detector_columns",
        "binning",
        "num_reference",
    ]
    # We avoid the possibility of negative numbers by scaling positive numbers.
    for key in config_dict.keys():
        if type(config_dict[key]) is float and key not in skipped_floats:
            if key in bounded_floats:
                config_dict[key] = np.clip(
                    config_dict[key]
                    * (1 - domain_randomization * (1 - 2 * np.random.rand())),
                    0.01,
                    0.99,
                )
            else:
                config_dict[key] = config_dict[key] * (
                    1 - domain_randomization * (1 - 2 * np.random.rand())
                )
        elif type(config_dict[key]) is int and key not in skipped_ints:
            config_dict[key] = int(
                np.clip(
                    config_dict[key]
                    * (1 - domain_randomization * (1 - 2 * np.random.rand())),
                    1,
                    None,
                )
            )

    config_dict["weave_pattern"] = generate_weave_pattern(
        config_dict["weft_yarns_per_layer"],
        config_dict["warp_yarns_per_layer"],
        config_dict["weave_pattern"],
        domain_randomization=domain_randomization,
    )

    deform = config_dict["deform"]
    randomized_deform = []
    for param in deform:
        randomized_deform.append(
            param * (1 - domain_randomization * (1 - 2 * np.random.rand()))
        )
    config_dict["deform"] = randomized_deform

    # Subtact three to make [1,1,1] correspond to zero tiling complexity.
    tiling_complexity = int(
        (sum(config_dict["tiling"]) - 3)
        * (1 - domain_randomization * (1 - 2 * np.random.rand()))
    )
    tiling = [1, 1, 1]
    while tiling_complexity > 0:
        idx = random.randint(0, 2)
        tiling[idx] += 1
        tiling_complexity -= 1
    config_dict["tiling"] = tiling

    # We cant treat tilt and offset the same way as the other parameters during
    # domain randomization.
    config_dict["tilt"] = [
        (np.random.rand() * 2 - 1) * 3,
        (np.random.rand() * 2 - 1) * 3,
        (np.random.rand() * 2 - 1) * 3,
    ]
    config_dict["offset"] = [
        (np.random.rand() * 2 - 1) * 3,
        (np.random.rand() * 2 - 1) * 3,
        (np.random.rand() * 2 - 1) * 3,
    ]

    return config_dict


def generate_woven_composite_sample_batch(
    num_samples, base_config, domain_randomization=0.0
):
    """Perform a batch run of generate_woven_composite_sample.

    Args:
        num_samples (int): The number of samples to generate.

        base_config (dict): A valid Tex-Ray config dict that will be used as the
                            template for generating samples.

    Keyword args:
        domain_randomization (float): A number larger or equal to  0 and less
                                      than 1 that determines the amount of
                                      domain randomization to be appl. to base
                                      config.

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
        config = generate_config_dict(i, base_config, domain_randomization)
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
    master_config,
    data_base_path,
    num_process=1,
    domain_randomization=0.0,
    chunk_size=10,
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

        domain_randomization (float): A number between 0.0 and 1.0 that
                                      determines the percentual domain
                                      randomization to be applied to the sample.

        chunk_size (int): The chunk size. This argument is only used when
                          creating a new database.

    Returns:
        -
    """
    print("\nStarting to process a batch of size " + str(num_process) + ".")
    t0 = time.time()

    config_dicts, exit_codes = generate_woven_composite_sample_batch(
        num_process, master_config, domain_randomization
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

    config_path = "./tex-ray/input/default_input.json"
    database_path = "./tex-ray/dbase"
    num_process = 4
    domain_randomization = 0.2

    with open(config_path) as f:
        default_config = json.load(f)

    while True:
        job = (
            Process(  # We wrap each batch in a process to prevent memory leak.
                target=run_batch,
                args=(default_config, database_path),
                kwargs={
                    "num_process": num_process,
                    "domain_randomization": domain_randomization,
                },
            )
        )
        job.start()
        job.join()
        job.close()
