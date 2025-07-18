import numpy as np
import cupy as cp
import pickle as pk
import meshio
from scipy.spatial.transform import Rotation as R

"""
This file contains the routines for automatic labeling of a woven composite.
The labeling is carried out through voxelization of the constituent meshes.

Voxelization is performed in parallel on the GPU.
"""


class SegmentationConfigError(Exception):
    """Exception raised when missing a required config dictionary entry."""

    pass


def check_segmentation_config_dict(config_dict):
    """Check that a config dict pertaining segmentation is valid.
      If invalid an appropriate exception is raised.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        segmentation_dict (dict): A dictionary consisting of relevant
                                  segmentation parameters.

    """
    args = []
    req_keys = (
        "mesh_paths",
        "distance_source_origin",
        "distance_origin_detector",
        "detector_columns",
        "detector_rows",
        "detector_pixel_size",
        "rot_axis",
        "offset",
        "tilt",
    )

    req_types = (
        list,
        float,
        float,
        int,
        int,
        float,
        str,
        list,
        list,
    )

    for req_key, req_type in zip(req_keys, req_types):
        args.append(config_dict.get(req_key))
        if args[-1] is None:
            raise SegmentationConfigError(
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
        "sample_length_unit",
    )
    opt_types = (int, str, str)
    def_vals = (1, "mm", "mm")

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
        config_dict.get("detector_rows") % config_dict.get("binning", 1) != 0.0
        or config_dict.get("detector_columns") % config_dict.get("binning", 1)
        != 0.0
    ):
        raise ValueError(
            "Bad arguments: binning must be a divisor of both the detector "
            + "rows and the detector columns."
        )

    return dict(zip(req_keys + opt_keys, args))


def bresenham3D(volume, start, end, value):
    """Modifies an 3D volume by drawing a line from start to end using the
       Bresenham 3D algorithm. Sets each voxel along the line to value.

    Args:
        volume (numpy array[int]): An n by n by n numpy array to be modified.

        start (numpy array[int]): A numpy array of length 3 representing the
                                  voxel coordinate to where to start drawing
                                  the line (from).

        end (numpy array[int]): A numpy array of length 3 representing the
                                  voxel coordinate to where to end drawing
                                  the line (to).

        value (int): The value to set the modified voxels to.


    Keyword args:
        -

    Returns:
        None
    """
    z1, y1, x1 = start
    z2, y2, x2 = end

    dx = abs(x2 - x1)
    dy = abs(y2 - y1)
    dz = abs(z2 - z1)

    xs = 1 if x2 > x1 else -1
    ys = 1 if y2 > y1 else -1
    zs = 1 if z2 > z1 else -1

    if dx >= dy and dx >= dz:
        p1 = 2 * dy - dx
        p2 = 2 * dz - dx
        while x1 != x2:
            if (
                0 <= x1 < volume.shape[0]
                and 0 <= y1 < volume.shape[1]
                and 0 <= z1 < volume.shape[2]
            ):
                volume[x1, y1, z1] = value
            x1 += xs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dx
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dx
            p1 += 2 * dy
            p2 += 2 * dz

    elif dy >= dx and dy >= dz:
        p1 = 2 * dx - dy
        p2 = 2 * dz - dy
        while y1 != y2:
            if (
                0 <= x1 < volume.shape[0]
                and 0 <= y1 < volume.shape[1]
                and 0 <= z1 < volume.shape[2]
            ):
                volume[x1, y1, z1] = value
            y1 += ys
            if p1 >= 0:
                x1 += xs
                p1 -= 2 * dy
            if p2 >= 0:
                z1 += zs
                p2 -= 2 * dy
            p1 += 2 * dx
            p2 += 2 * dz

    else:
        p1 = 2 * dy - dz
        p2 = 2 * dx - dz
        while z1 != z2:
            if (
                0 <= x1 < volume.shape[0]
                and 0 <= y1 < volume.shape[1]
                and 0 <= z1 < volume.shape[2]
            ):
                volume[x1, y1, z1] = value
            z1 += zs
            if p1 >= 0:
                y1 += ys
                p1 -= 2 * dz
            if p2 >= 0:
                x1 += xs
                p2 -= 2 * dz
            p1 += 2 * dy
            p2 += 2 * dx

    if (
        0 <= x2 < volume.shape[0]
        and 0 <= y2 < volume.shape[1]
        and 0 <= z2 < volume.shape[2]
    ):
        volume[x2, y2, z2] = value


def segment_reconstruction(config_dict):
    """Label the reconstructed volume of a virtual x-ray tomographic scan.
       The scanned meshes are voxelized in the same reference frame as the
       reconstruction. Air is labelled as 0. Other phases are labeled with
       increasing numbers in the order they appear in the config.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        labelled_volume (numpy array[int]): A 3D numpy array containing the
                                            labels in the form of voxel values.

    """
    with open("./textomos/voxelizationKernel.cu", "r") as f:
        kernel_as_string = f.read()
    vox_kernel = cp.RawKernel(kernel_as_string, "vox_kernel")

    segmentation_config_dict = check_segmentation_config_dict(config_dict)

    if segmentation_config_dict["sample_length_unit"] == "mm":
        sample_scale_factor = 1e-3
    elif segmentation_config_dict["sample_length_unit"] == "cm":
        sample_scale_factor = 1e-2
    elif segmentation_config_dict["sample_length_unit"] == "m":
        sample_scale_factor = 1.0

    if segmentation_config_dict["scanner_length_unit"] == "mm":
        scanner_scale_factor = 1e-3
    elif segmentation_config_dict["scanner_length_unit"] == "cm":
        scanner_scale_factor = 1e-2
    elif segmentation_config_dict["scanner_length_unit"] == "m":
        scanner_scale_factor = 1.0

    tilt = np.array(segmentation_config_dict["tilt"])
    offset = np.array(segmentation_config_dict["offset"]) * sample_scale_factor

    if segmentation_config_dict["rot_axis"] == "x":
        axis = np.array([0, -1, 0])
        angle = 90
    elif segmentation_config_dict["rot_axis"] == "y":
        axis = np.array([1, 0, 0])
        angle = 90
    else:
        # z is already up, so we just use this to not have to split by case.
        axis = np.array([0, 0, 1])
        angle = 0

    num_voxels = (
        segmentation_config_dict["detector_rows"]
        // segmentation_config_dict["binning"]
    )

    voxel_size = (
        segmentation_config_dict["binning"]
        * segmentation_config_dict["detector_pixel_size"]
        * segmentation_config_dict["distance_source_origin"]
        / (
            segmentation_config_dict["distance_source_origin"]
            + segmentation_config_dict["distance_origin_detector"]
        )
        * scanner_scale_factor
    )

    rot = R.from_rotvec(angle * axis * np.pi / 180)
    rot_tilt = R.from_rotvec(tilt * np.pi / 180)

    triangles = []
    num_triangles = []
    phase_center_lines = []
    for path in segmentation_config_dict["mesh_paths"]:
        mesh = meshio.read(path)
        vertex_coords = mesh.points
        connectivity = mesh.cells[0].data
        triangle = (
            vertex_coords[connectivity].reshape((-1, 3)) * sample_scale_factor
        )
        triangles.append(triangle)
        num_triangles.append(len(triangle) // 3)
        try:
            with open(path.replace(".stl", ".pkl"), "rb") as file:
                phase_center_lines.append(pk.load(file))
        except FileNotFoundError:
            continue  # We skip matrix without having to check a bunch this way.

    vox = cp.zeros(num_voxels * num_voxels * num_voxels, dtype=cp.int32)
    # Flags are powers of 2: 1, 2, 4, ... one for each phase (excluding air).
    bit_flags = [
        2**i for i in range(len(segmentation_config_dict["mesh_paths"]))
    ]
    for i in range(len(bit_flags)):
        rotated_tri = rot.apply(triangles[i])
        tilted_tri = rot_tilt.apply(rotated_tri)
        offset_tri = (
            tilted_tri + offset
        ).flatten()  # Flatten here, not on gpu.
        gpu_triangles = cp.asarray(offset_tri, dtype=cp.float32)
        vox_kernel(
            (num_triangles[i],),
            (1,),
            (
                gpu_triangles,
                cp.int32(num_voxels),
                cp.float64(voxel_size),
                cp.int32(bit_flags[i]),
                vox,
            ),
        )
    vox = vox & -vox  # Get least significant bit: remove outer overwrite.
    out = cp.zeros_like(vox)

    # Find position of most significant bit: from powers of 2 to 0, 1, 2, 3, ...
    while cp.size(cp.nonzero(vox)[0]):
        out[cp.nonzero(vox)[0]] += 1
        vox >>= 1

    out = out.reshape(num_voxels, num_voxels, num_voxels)

    for phase, center_lines in enumerate(phase_center_lines):
        for center_line in center_lines:
            rotated_center_line = rot.apply(center_line * sample_scale_factor)
            tilted_center_line = rot_tilt.apply(rotated_center_line)
            offset_center_line = tilted_center_line + offset
            # Inverse of logic in voxelizationKernel.cu lines 57 or 61
            voxelized_yarn_path = (
                np.rint(offset_center_line / voxel_size + num_voxels /2 - 0.5)
            )
            for idx in range(len(voxelized_yarn_path) - 1):
                bresenham3D(
                    out,
                    voxelized_yarn_path[idx],
                    voxelized_yarn_path[idx + 1],
                    phase + len(segmentation_config_dict["mesh_paths"]) + 1,
                )

    return out.get().astype(np.uint8)
