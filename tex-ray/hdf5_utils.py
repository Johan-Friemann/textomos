import tifffile
import h5py
import meshio
import numpy as np


def save_mesh_to_hdf5(
    chunk_path,
    local_idx,
    config_dict,
):
    """
    DOCSTRING
    """
    weft_mesh = meshio.read(config_dict["weft_path"])
    weft_vertex_coords = weft_mesh.points
    weft_triangle_vertex_connectivity = weft_mesh.cells[0].data

    warp_mesh = meshio.read(config_dict["warp_path"])
    warp_vertex_coords = warp_mesh.points
    warp_triangle_vertex_connectivity = warp_mesh.cells[0].data

    matrix_mesh = meshio.read(config_dict["matrix_path"])
    matrix_vertex_coords = matrix_mesh.points
    matrix_triangle_vertex_connectivity = matrix_mesh.cells[0].data

    with h5py.File(chunk_path, "a") as f:
        group = f.create_group("mesh_" + str(local_idx))
        group.create_dataset("weft_vertex_coords", data=weft_vertex_coords)
        group.create_dataset(
            "weft_triangle_vertex_connectivity",
            data=weft_triangle_vertex_connectivity,
        )
        group.create_dataset("warp_vertex_coords", data=warp_vertex_coords)
        group.create_dataset(
            "warp_triangle_vertex_connectivity",
            data=warp_triangle_vertex_connectivity,
        )
        group.create_dataset("matrix_vertex_coords", data=matrix_vertex_coords)
        group.create_dataset(
            "matrix_triangle_vertex_connectivity",
            data=matrix_triangle_vertex_connectivity,
        )


def save_reconstruction_to_hdf5(chunk_path, local_idx, config_dict):
    """
    DOCSTRING
    """
    reconstruction = tifffile.imread(config_dict["reconstruction_output_path"])
    with h5py.File(chunk_path, "a") as f:
        data_set = f.create_dataset(
            "reconstruction_" + str(local_idx), data=reconstruction
        )


def save_segmentation_to_hdf5(chunk_path, local_idx, config_dict):
    """
    DOCSTRING
    """
    segmentation = tifffile.imread(config_dict["segmentation_output_path"])
    with h5py.File(chunk_path, "a") as f:
        data_set = f.create_dataset(
            "segmentation_" + str(local_idx), data=segmentation
        )


def save_data(
    map_path,
    mesh_root_path,
    reconstruction_root_path,
    segmentation_root_path,
    config_dict,
):
    """
    DOCSTRING
    """
    chunk_size = 2

    with h5py.File(map_path, "a") as f:
        # We always increase by 1, -1 hack to make first instance go to 0
        global_idx = f.attrs.get("current_global_index", -1) + 1
        local_idx = f.attrs.get("current_local_index", -1) + 1
        chunk_idx = f.attrs.get("current_chunk_index", 0)
        if local_idx >= chunk_size:  # Go to new chunk
            local_idx = 0
            chunk_idx += 1
        # no big deal to overwrite if nothing changes
        f.attrs["current_global_index"] = global_idx
        f.attrs["current_local_index"] = local_idx
        f.attrs["current_chunk_index"] = chunk_idx

        group_name = "data_" + str(global_idx)
        group = f.create_group(group_name)
        group.attrs["chunk_index"] = chunk_idx
        group.attrs["local_index"] = local_idx
        sub_group = group.create_group("metadata")
        sub_group.attrs.update(config_dict)

    mesh_chunk_path = mesh_root_path + "mesh_data_" + str(chunk_idx) + ".hdf5"
    save_mesh_to_hdf5(mesh_chunk_path, local_idx, config_dict)

    reconstruction_chunk_path = (
        reconstruction_root_path + "reconstruction_data_" + str(chunk_idx) + ".hdf5"
    )
    save_reconstruction_to_hdf5(
        reconstruction_chunk_path, local_idx, config_dict
    )
    segmentation_chunk_path = (
        segmentation_root_path + "segmentation_data_" + str(chunk_idx) + ".hdf5"
    )
    save_segmentation_to_hdf5(segmentation_chunk_path, local_idx, config_dict)


import json

with open("./tex-ray/input/default_input.json") as f:
    config_dict = json.load(f)

save_data(
    "./tex-ray/foo.hdf5",
    "./tex-ray/",
    "./tex-ray/",
    "./tex-ray/",
    config_dict,
)
