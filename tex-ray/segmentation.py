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
    return config_dict


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

    tilt = np.array(segmentation_config_dict["tilt"])
    offset = np.array(segmentation_config_dict["offset"])
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
        triangle = vertex_coords[connectivity].reshape((-1, 3))
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
                            num_voxels,
                            cp.float64(voxel_size),
                            cp.int32(l + 1),
                            vox,
                        ),
                    )

    vox = vox.reshape(num_voxels, num_voxels, num_voxels)
    return vox.get().astype(np.uint8)
