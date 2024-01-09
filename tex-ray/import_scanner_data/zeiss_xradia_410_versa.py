"""
The name "ZEISS XRadia 410 Versa" is owned by Carl ZEISS AG.

Big thanks to Martin S. Andersen and txrmtools:
https://gitlab.gbar.dtu.dk/mskan/txrmtools for inspiration.
The mentioned package is not used here due to some small
differences between txrm and txm files.
"""

import numpy as np
from olefile import isOleFile, OleFileIO
import tifffile


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


def read_txm_scan_info(in_path):
    """Read Zeiss txm file metadata.
    Args:
        in_path (str): The absolute path to the txm file.

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
        "d_source_object_microm",
        "d_object_detector_microm",
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
    paths = [
        "CamFullHeight",
        "CamFullWidth",
        "CameraBinning",
    ]
    keys = [
        "detector_num_pixel_height",
        "detector_num_pixel_width",
        "detector_binning_num",
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
