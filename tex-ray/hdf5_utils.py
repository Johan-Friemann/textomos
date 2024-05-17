import os
import tifffile
import h5py
import meshio

"""
This file contains routines for saving simulation results to a database of hdf5
files. The design of the database structure is aimed to counteract two
anti-patterns of machine learning data: 1 monolithic file and a large amount of
files containing 1 data point each.

Therefore, the data is split into "chunks" of 100 data points per file. The
global id of each data point is recorded in a database map file, together
with the data points metadata.
"""


def save_mesh_to_hdf5(
    chunk_path,
    local_idx,
    config_dict,
):
    """Save meshes to an hdf5 file.

    Args:
        chunk_path (str): Path to mesh chunk file.

        local_idx (int): The local index of the data inside the chunk.

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        -

    Returns:
        -
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
    """Save a reconstruction to an hdf5 file.

    Args:
        chunk_path (str): Path to reconstruction chunk file.

        local_idx (int): The local index of the data inside the chunk.

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        -

    Returns:
        -
    """
    reconstruction = tifffile.imread(config_dict["reconstruction_output_path"])
    with h5py.File(chunk_path, "a") as f:
        data_set = f.create_dataset(
            "reconstruction_" + str(local_idx), data=reconstruction
        )


def save_segmentation_to_hdf5(chunk_path, local_idx, config_dict):
    """Save a segmentation to an hdf5 file.

    Args:
        chunk_path (str): Path to segmentation chunk file.

        local_idx (int): The local index of the data inside the chunk.

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        -

    Returns:
        -
    """
    segmentation = tifffile.imread(config_dict["segmentation_output_path"])
    with h5py.File(chunk_path, "a") as f:
        data_set = f.create_dataset(
            "segmentation_" + str(local_idx), data=segmentation
        )


def save_data(
    database_path,
    config_dict,
):
    """
    Save a virtual scan and its metadata to an hdf5 database. The database is
    arranged like:

    /database_folder/
                     database_map (contains metadata and ids of all data points)
                     mesh_data/
                               mesh_data_0.hdf5
                               ...
                     reconstruction_data/
                                        reconstruction_data_0.hdf5
                                        ...
                     segmentation_data/
                                       segmentation_data_0.hdf5
                                       ...


    Args:
        data_base_path (str): The absolute path to the database folder. If it
                              doesn't exist it is created (create new database).

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        -

    Returns:
        -
    """
    chunk_size = 100  # Magic number, a reasonable chunk size.

    map_path = os.path.join(database_path, "database_map.hdf5")
    if not os.path.exists(database_path):
        os.makedirs(database_path)

    mesh_path = os.path.join(database_path, "mesh_data/")
    if not os.path.exists(mesh_path):
        os.makedirs(mesh_path)

    reconstruction_path = os.path.join(database_path, "reconstruction_data/")
    if not os.path.exists(reconstruction_path):
        os.makedirs(reconstruction_path)

    segmentation_path = os.path.join(database_path, "segmentation_data/")
    if not os.path.exists(segmentation_path):
        os.makedirs(segmentation_path)

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

        # We exclude path information from database as it is not relevant.
        for key in config_dict.keys():
            if not "path" in key:
                sub_group.attrs[key] = config_dict[key]

    mesh_chunk_path = mesh_path + "mesh_data_" + str(chunk_idx) + ".hdf5"
    save_mesh_to_hdf5(mesh_chunk_path, local_idx, config_dict)

    reconstruction_chunk_path = (
        reconstruction_path + "reconstruction_data_" + str(chunk_idx) + ".hdf5"
    )
    save_reconstruction_to_hdf5(
        reconstruction_chunk_path, local_idx, config_dict
    )
    segmentation_chunk_path = (
        segmentation_path + "segmentation_data_" + str(chunk_idx) + ".hdf5"
    )
    save_segmentation_to_hdf5(segmentation_chunk_path, local_idx, config_dict)
