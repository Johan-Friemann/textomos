import numpy as np
import cupy as cp
import meshio
from scipy.spatial.transform import Rotation as R

"""
This file contains the routines for automatic labeling of a woven composite.
The labeling is carried out through voxelization of the weft, warp, and and
matrix meshes. Two types of voxelizations can be performed. The first one is
performed on the meshes in the same sample configuration (including tiling,
offset etc) as the virtual x-ray scan. The second type contains only one highly
resolved unit-cell.

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
        "weft_path",
        "warp_path",
        "matrix_path",
        "distance_source_origin",
        "distance_origin_detector",
        "detector_columns",
        "detector_rows",
        "detector_pixel_size",
        "rot_axis",
        "tiling",
        "offset",
        "tilt",
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
        str,
        list,
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


def segment_reconstruction(config_dict):
    """Label the reconstructed volume of a virtual x-ray tomographic scan.
       The scanned meshes are voxelized in the same reference frame as the
       reconstruction. Matrix material is labelled as 1, weft material is
       labelled as 2, and warp material is labelled as 3.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        labelled_volume (numpy array[int]): A 3D numpy array containing the
                                            labels in the form of voxel values.

    """
    with open("./tex-ray/voxelizationKernel.cu", "r") as f:
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
    tiling = segmentation_config_dict["tiling"]

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

    paths = [
        segmentation_config_dict["matrix_path"],
        segmentation_config_dict["weft_path"],
        segmentation_config_dict["warp_path"],
    ]

    triangles = []
    num_triangles = []
    for path in paths:
        mesh = meshio.read(path)
        vertex_coords = mesh.points
        connectivity = mesh.cells[0].data
        triangle = (
            vertex_coords[connectivity].reshape((-1, 3)) * sample_scale_factor
        )
        triangles.append(triangle)
        num_triangles.append(len(triangle) // 3)

    tile_size = np.max(triangles[0], axis=0) - np.min(triangles[0], axis=0)

    vox = cp.zeros(num_voxels * num_voxels * num_voxels).astype(cp.int32)
    for i in range(tiling[0]):
        for j in range(tiling[1]):
            for k in range(tiling[2]):
                shift = np.array(
                    [
                        tile_size[0] * (i - (tiling[0] - 1) / 2),
                        tile_size[1] * (j - (tiling[1] - 1) / 2),
                        tile_size[2] * (k - (tiling[2] - 1) / 2),
                    ]
                )
                bit_flags = [5, 3, 2] # Results in 5, 6, 7 after kernel bit xor.
                for l in range(3):
                    shifted_tri = triangles[l] + shift
                    rotated_tri = rot.apply(shifted_tri)
                    tilted_tri = rot_tilt.apply(rotated_tri)
                    offset_tri = tilted_tri + offset
                    gpu_triangles = cp.asarray(offset_tri)
                    vox_kernel(
                        (num_triangles[l],),
                        (1,),
                        (
                            gpu_triangles.flatten().astype(cp.float32),
                            cp.int32(num_voxels),
                            cp.float64(voxel_size),
                            cp.int32(bit_flags[l]),
                            vox,
                        ),
                    )

    vox = vox.reshape(num_voxels, num_voxels, num_voxels) & 3 # Makes it 1, 2, 3
    return vox.get().astype(np.uint8)
