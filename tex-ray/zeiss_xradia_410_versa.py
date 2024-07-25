"""
The name "ZEISS XRadia 410 Versa" is owned by Carl ZEISS AG.

Big thanks to Martin S. Andersen and txrmtools:
https://gitlab.gbar.dtu.dk/mskan/txrmtools for inspiration.
The mentioned package is not used here due to some small
differences between txrm and txm files.
"""

import numpy as np
import astra
from olefile import isOleFile, OleFileIO
import tifffile

from x_ray_simulation import neg_log_transform


def read_txm_scan_data(in_path):
    """Read Zeiss txm file data.
    Args:
        in_path (str): The absolute path to the txm file.

    Keyword args:
        -
    Returns:
        data_out (numpy array[numpy array[numpy array[int]]]):
            Tomographic reconstruction as a numpy array.
    """
    if not isOleFile(in_path):
        raise ValueError("%s is not an OLE file!" % (in_path))

    ole = OleFileIO(in_path)
    stream = ole.openstream("ImageInfo/ImageHeight")
    buffer = stream.read()
    stream.close()
    image_height = np.frombuffer(buffer, np.uint32)[0]

    stream = ole.openstream("ImageInfo/ImageWidth")
    buffer = stream.read()
    stream.close()
    image_width = np.frombuffer(buffer, np.uint32)[0]

    stream = ole.openstream("ImageInfo/ImagesTaken")
    buffer = stream.read()
    stream.close()
    images_taken = np.frombuffer(buffer, np.uint32)[0]

    data_out = np.ndarray(
        (images_taken, image_height, image_width), dtype=np.uint16
    )
    # Images are stored as chunks of 100. ImageData1 contains
    # Image1 to Image100, ImageData2 contains Image201 to Image300 and so on...
    for image_id in range(1, images_taken + 1):
        image_data_id = (image_id - 1) // 100 + 1
        formatted_str = "ImageData%i/Image%i" % (image_data_id, image_id)

        stream = ole.openstream(formatted_str)
        buffer = stream.read()
        stream.close()
        image_data = np.frombuffer(buffer, np.uint16)
        data_out[image_id - 1, :, :] = np.reshape(
            image_data, (image_height, image_width)
        )

    ole.close()
    return data_out


def read_txrm_scan_data(in_path, step=1, binning=1):
    """Read Zeiss txrm file data and perform flat field correction.

    Args:
        in_path (str): The absolute path to the txrm file.

    Keyword args:
        step (int): Will include every step:th projection.

        binning (int): Binning number, sum bin by bin pixels across projections.

    Returns:
        projections_out (numpy array[numpy array[numpy array[float]]]):
            Flat field corrected projections as a numpy array.

        angles (numpy array[float]): The angles in degrees that the projections
                                     where taken at.

        shifts (numpy array[numpy array[float]]): An array of sample shifts in
                                                  terms of number of pixel
                                                  displacements on the detector.
                                                  First row is x-shifts and
                                                  second row is y-shifts.


    """
    if not isOleFile(in_path):
        raise ValueError("%s is not an OLE file!" % (in_path))

    ole = OleFileIO(in_path)

    stream = ole.openstream("ImageInfo/ImageHeight")
    buffer = stream.read()
    stream.close()
    image_height = np.frombuffer(buffer, np.uint32)[0]

    stream = ole.openstream("ImageInfo/ImageWidth")
    buffer = stream.read()
    stream.close()
    image_width = np.frombuffer(buffer, np.uint32)[0]

    stream = ole.openstream("ImageInfo/ImagesTaken")
    buffer = stream.read()
    stream.close()
    images_taken = np.frombuffer(buffer, np.uint32)[0]

    stream = ole.openstream("ImageInfo/Angles")
    buffer = stream.read()
    stream.close()
    angles = np.frombuffer(buffer, np.float32)[::step]

    stream = ole.openstream("ReconSettings/CenterShift")
    buffer = stream.read()
    stream.close()
    center_shift = np.frombuffer(buffer, np.float32)[0]

    stream = ole.openstream("Alignment/X-Shifts")
    buffer = stream.read()
    stream.close()
    x_shifts = np.frombuffer(buffer, np.float32)[::step]

    stream = ole.openstream("Alignment/Y-Shifts")
    buffer = stream.read()
    stream.close()
    y_shifts = np.frombuffer(buffer, np.float32)[::step]

    # center_shift acts like a "DC" bias on the x_shifts
    shifts = np.array((x_shifts + center_shift, y_shifts))

    stream = ole.openstream("ReferenceData/Image")
    buffer = stream.read()
    stream.close()
    reference = np.frombuffer(buffer, np.uint16)
    reference_image = np.reshape(reference, (image_height, image_width))

    # Weird hack that makes num_images = len(np.arange(images_taken)[::step])
    num_images = (images_taken - 1) // step + 1 
    projections_out = np.ndarray(
        (num_images, image_height // binning, image_width // binning),
        dtype=np.float32,
    )
    # Images are stored as chunks of 100. ImageData1 contains
    # Image1 to Image100, ImageData2 contains Image201 to Image300 and so on...
    for image_id in range(1, images_taken + 1, step):
        image_data_id = (image_id - 1) // 100 + 1
        formatted_str = "ImageData%i/Image%i" % (image_data_id, image_id)

        stream = ole.openstream(formatted_str)
        buffer = stream.read()
        stream.close()
        image_data = np.frombuffer(buffer, np.uint16)
        image = np.reshape(image_data, (image_height, image_width))

        binned_image = (
            image.reshape(
                image_height // binning,
                binning,
                image_width // binning,
                binning,
            )
            .sum(-1)
            .sum(1)
        )
        binned_reference = (
            reference_image.reshape(
                image_height // binning,
                binning,
                image_width // binning,
                binning,
            )
            .sum(-1)
            .sum(1)
        )
        projections_out[(image_id - 1) // step] = (
            binned_image / binned_reference
        )

    return projections_out, angles, shifts


def read_txrm_scan_info(in_path):
    """Read Zeiss txrm file metadata.
    Args:
        in_path (str): The absolute path to the txrm file.

    Keyword args:
        -
    Returns:
        output_dict (dictionary): A dictionary that contains scan metadata, such
                                  as voltage, current, and detector pixel size.
    """
    output_dict = {}

    # float32 data
    paths = [
        "Voltage",
        "Current",
        "StoRADistance",
        "DtoRADistance",
        "PixelSize",
        "CamPixelSize",
        "ExpTimes",
        "OpticalMagnification",
        "ConeAngle",
    ]
    keys = [
        "voltage_kV",
        "current_microA",
        "d_source_object_mm",
        "d_object_detector_mm",
        "image_pixel_size_microm",
        "detector_pixel_size_microm",
        "exposure_time_sec",
        "optical_magnification",
        "cone_angle",
    ]
    for path, key in zip(paths, keys):
        ole = OleFileIO(in_path)
        full_path = "ImageInfo/" + path
        stream = ole.openstream(full_path)
        buffer = stream.read()
        value = np.frombuffer(buffer, np.float32)[0]
        output_dict[key] = value
        stream.close()

    # uint32 data
    paths = ["CamFullHeight", "CamFullWidth", "CameraBinning", "ImagesTaken"]
    keys = [
        "detector_num_pixel_height",
        "detector_num_pixel_width",
        "detector_binning_num",
        "images_taken",
    ]
    for path, key in zip(paths, keys):
        ole = OleFileIO(in_path)
        full_path = "ImageInfo/" + path
        stream = ole.openstream(full_path)
        buffer = stream.read()
        value = np.frombuffer(buffer, np.uint32)[0]
        output_dict[key] = value
        stream.close()

    # TODO Handle str data
    ole.close()

    return output_dict


def convert_txm_to_tiff(in_path, out_path, prune=[]):
    """Convert Zeiss txm data file to a tiff stack.
    Args:
        in_path (str): The absolute path to the txm file.
        out_path (str): The desired absolute path (including file name) to the
                        tifffile.

    Keyword args:
        prune (list[int]): If given prune the data. List has the form
                           [x_min, x_max, y_min, y_max, z_min, z_max].
                           If the entire range is desired for one dimension, put
                           the max value to -1.
    Returns:
        -
    """

    data_in = read_txm_scan_data(in_path)
    if prune:
        if len(prune) != 6:
            raise ValueError("prune must have length 6.")
        for element in prune:
            if type(element) is not int:
                raise ValueError("The elements of prune must be integers.")
        data_out = data_in[
            prune[0] : prune[1], prune[2] : prune[3], prune[4] : prune[5]
        ]
    else:
        data_out = data_in
    tifffile.imwrite(out_path, data_out)


def reconstruct_txrm_to_tiff(
    in_path, out_path, step=1, binning=1, threshold=1e-16, save_uint16=False
):
    """Perform a tomographic reconstruction on projection data saved in a Zeiss
        txrm file and save it as a tiff stack.

    Args:
        in_path (str): The absolute path to the txrm file.

        out_path (str): The absolute path to where to save the tiff file.

    Keyword args:
        step (int): Will include every step:th projection.

        binning (int): Binning number, sum bin by bin pixels across projections.
                       Note that the binning is applied in addition to the
                       binning that was performed by the machine.

        threshold (double): The thresholding value for negative log transform.

        save_uint16 (Bool): Will save the data as min-max scaled uint16 data if
                            true. This is used for comparison with
                            reconstructions performed by the machine.
                            Saves the data as float32 representing cm-1
                            if false (default).

    Returns:
        -
    """
    meta_data = read_txrm_scan_info(in_path)
    projections, angles, shifts = read_txrm_scan_data(
        in_path, step=step, binning=binning
    )
    neg_log_projections = neg_log_transform(projections, threshold)
    del projections
    sinograms = np.swapaxes(neg_log_projections, 0, 1)
    del neg_log_projections

    # Binning is treated as actual detector pixel size as ASTRA does not handle.
    # We need to take .item() from metadata entries because ASTRA can't handle
    # numpy data types sometimes.
    detector_rows = meta_data["detector_num_pixel_height"].item() // (
        meta_data["detector_binning_num"].item() * binning
    )
    detector_columns = meta_data["detector_num_pixel_width"].item() // (
        meta_data["detector_binning_num"].item() * binning
    )
    detector_pixel_size = (
        meta_data["detector_pixel_size_microm"].item()
        * (meta_data["detector_binning_num"].item() * binning)
        / 1000.0  # convert to mm
        / meta_data[
            "optical_magnification"
        ].item()  # account for optical magnification (not handled by ASTRA)
    )
    distance_source_origin = abs(meta_data["d_source_object_mm"].item())
    distance_origin_detector = meta_data["d_object_detector_mm"].item()
    voxel_size = detector_pixel_size / (
        (distance_origin_detector + distance_source_origin)
        / distance_source_origin
    )
    # Weird hack that makes num_images = len(np.arange(images_taken)[::step])
    num_images = (meta_data["images_taken"].item() - 1) // step + 1
    angles = -np.deg2rad(angles)  # Astra uses radians with cw being positive
    # The shifts are applied in terms of number of pixel displacements, we must
    # therefore scale by the binning since we have scaled the pixel size above.
    shifts /= binning

    cone_vecs = np.zeros((num_images, 12))
    cone_vecs[:, 0] = np.sin(angles) * distance_source_origin
    cone_vecs[:, 1] = -np.cos(angles) * distance_source_origin
    cone_vecs[:, 3] = (
        -np.sin(angles) * distance_origin_detector
        + np.cos(angles) * detector_pixel_size * shifts[0]
    )
    cone_vecs[:, 4] = (
        np.cos(angles) * distance_origin_detector
        + np.sin(angles) * detector_pixel_size * shifts[0]
    )
    cone_vecs[:, 5] = shifts[1] * detector_pixel_size
    cone_vecs[:, 6] = np.cos(angles) * detector_pixel_size
    cone_vecs[:, 7] = np.sin(angles) * detector_pixel_size
    cone_vecs[:, 11] = detector_pixel_size

    proj_geo = astra.create_proj_geom(
        "cone_vec", detector_rows, detector_columns, cone_vecs
    )
    vol_geo = astra.create_vol_geom(
        detector_columns, detector_columns, detector_rows
    )
    for coord in ("X", "Y", "Z"):
        vol_geo["option"]["WindowMin" + coord] *= voxel_size
        vol_geo["option"]["WindowMax" + coord] *= voxel_size

    proj_id = astra.data3d.create("-sino", proj_geo, data=sinograms)
    rec_id = astra.data3d.create("-vol", vol_geo, data=0)
    alg_cfg = astra.astra_dict("FDK_CUDA")
    alg_cfg["ReconstructionDataId"] = rec_id
    alg_cfg["ProjectionDataId"] = proj_id
    alg_id = astra.algorithm.create(alg_cfg)
    astra.algorithm.run(alg_id, 1)
    reconstruction = astra.data3d.get(rec_id)
    del sinograms
    # Remove eventual erroneous negative values.
    reconstruction[reconstruction < 0] = 0.0

    if save_uint16:
        reconstruction = (
            (reconstruction - np.min(reconstruction))
            / (np.max(reconstruction) - np.min(reconstruction))
            * 65535
        ).astype(np.uint16)
        # Flipping like puts the reconstruction in the same frame of reference
        # as the reconstructions from the machine.
        reconstruction = np.flip(reconstruction, axis=1)
    else:
        reconstruction *= 10  # this will convert to cm-1
        # Flipping like puts the reconstruction in the same frame of reference
        # as the reconstructions from the simulations.
        reconstruction = np.flip(reconstruction, axis=1)
        reconstruction = np.flip(reconstruction, axis=2)

    tifffile.imwrite(out_path, reconstruction)

    return None
