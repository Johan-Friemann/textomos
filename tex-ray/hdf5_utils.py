import os
import tifffile
import h5py
import meshio
import numpy as np

"""
This file contains routines for saving/getting simulation results to/from a 
database of hdf5 files. The design of the database structure is aimed to
counteract two anti-patterns of machine learning data: 1 monolithic file and a
large amount of files containing 1 data point each.

Therefore, the data is split into "chunks" with several data points per file.
The global id of each data point is recorded in a database map file, together
with the data point's metadata.
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
    vertex_coords = []
    triangle_vertex_connectivity = []
    for path in config_dict["mesh_paths"]:
        mesh = meshio.read(path)
        vertex_coords.append(mesh.points)
        triangle_vertex_connectivity.append(mesh.cells[0].data)

    with h5py.File(chunk_path, "a") as f:
        group = f.create_group("mesh_" + str(local_idx))
        for phase in range(len(vertex_coords)):
            group.create_dataset(
                "vertex_coords_" + str(phase), data=vertex_coords[phase]
            )
            group.create_dataset(
                "triangle_vertex_connectivity_" + str(phase),
                data=triangle_vertex_connectivity[phase],
            )

    return None


def save_reconstruction_to_hdf5(
    chunk_path, local_idx, config_dict, slice_axis="x"
):
    """Save a reconstruction to an hdf5 file.

    Args:
        chunk_path (str): Path to reconstruction chunk file.

        local_idx (int): The local index of the data inside the chunk.

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        slice_axis (str): The axis along which the data will be quickest
                          to access. Can be "x", "y", or "z".

    Returns:
        -
    """
    reconstruction = tifffile.imread(config_dict["reconstruction_output_path"])
    if slice_axis == "x":
        reconstruction = np.transpose(reconstruction, (2, 0, 1))
    elif slice_axis == "y":
        reconstruction = np.transpose(reconstruction, (1, 0, 2))
    elif slice_axis != "z":
        raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")

    with h5py.File(chunk_path, "a") as f:
        data_set = f.create_dataset(
            "reconstruction_" + str(local_idx), data=reconstruction
        )
        data_set.attrs["slice_axis"] = slice_axis
    return None


def save_segmentation_to_hdf5(
    chunk_path, local_idx, config_dict, slice_axis="x"
):
    """Save a segmentation to an hdf5 file.

    Args:
        chunk_path (str): Path to segmentation chunk file.

        local_idx (int): The local index of the data inside the chunk.

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        slice_axis (str): The axis along which the data will be quickest
                          to access. Can be "x", "y", or "z".

    Returns:
        -
    """
    segmentation = tifffile.imread(config_dict["segmentation_output_path"])
    if slice_axis == "x":
        segmentation = np.transpose(segmentation, (2, 0, 1))
    elif slice_axis == "y":
        segmentation = np.transpose(segmentation, (1, 0, 2))
    elif slice_axis != "z":
        raise ValueError("slice_axis can only be 'x', 'y', or 'z'.")

    with h5py.File(chunk_path, "a") as f:
        data_set = f.create_dataset(
            "segmentation_" + str(local_idx), data=segmentation
        )
        data_set.attrs["slice_axis"] = slice_axis
    return None


def save_data(
    database_path,
    config_dict,
    chunk_size=10,
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
        database_path (str): The absolute path to the database folder. If it
                              doesn't exist it is created (create new database).

        config_dict (dict): A dictionary of Tex-Ray options pertaining the
                            simulation that is about to be saved.

    Keyword args:
        chunk_size (int): The chunk size. This argument is only used when
                          creating a new database.

    Returns:
        -
    """

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
        # chunk_size argument is used during creation only (else: comes from db)
        if f.attrs.get("chunk_size") is None:
            f.attrs["chunk_size"] = chunk_size
        if local_idx >= f.attrs["chunk_size"]:
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
        # We treat list of varying length differently due to hdf5 restriction.
        for key in config_dict.keys():
            if not "path" in key:
                if key == "phase_elements" or key == "phase_ratios":
                    for phase in range(len(config_dict[key])):
                        sub_group.attrs[key + "_" + str(phase)] = config_dict[
                            key
                        ][phase]
                else:
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

    return None


def get_reconstruction_chunk_handle(database_path, chunk_index):
    """Get a reconstruction chunk file handle from database.

    Args:
        database_path (str): The absolute path to the database folder.

        chunk_index (int): The chunk index of the file to get.

    Keyword args:
        -

    Returns:
        chunk_handle (HDF5 file): A file handle that points to reconstruction
                                  data stored in chunk with chunk_index.
    """
    reconstruction_path = os.path.join(
        database_path,
        "reconstruction_data/reconstruction_data_" + str(chunk_index) + ".hdf5",
    )
    return h5py.File(reconstruction_path, "r")


def get_segmentation_chunk_handle(database_path, chunk_index):
    """Get a segmentation chunk file handle from database.

    Args:
        database_path (str): The absolute path to the database folder.

        chunk_index (int): The chunk index of the file to get.

    Keyword args:
        -

    Returns:
        chunk_handle (HDF5 file): A file handle that points to segmentation
                                  data stored in chunk with chunk_index.
    """
    segmentation_path = os.path.join(
        database_path,
        "segmentation_data/segmentation_data_" + str(chunk_index) + ".hdf5",
    )
    return h5py.File(segmentation_path, "r")


def get_database_shape(database_path):
    """Get the size of the database.

    Args:
        database_path (str): The absolute path to the database folder.

    Keyword args:
        -

    Returns:
        num_samples (int): The number of items in the database.
        num_chunks (int): The number of chunks (files/field) in the database.
        chunk_size (int): The database chunk size.
    """
    map_path = os.path.join(database_path, "database_map.hdf5")
    if not os.path.exists(database_path):
        raise FileNotFoundError(
            "There exists no database at '" + database_path + "'."
        )

    with h5py.File(map_path, "r") as f:
        num_samples = f.attrs.get("current_global_index") + 1
        num_chunks = f.attrs.get("current_chunk_index") + 1
        chunk_size = f.attrs.get("chunk_size")

    return num_samples, num_chunks, chunk_size


def get_metadata(database_path, global_idx, field_name):
    """Get the metadata with name field_name from the datapoint with global_idx.

    Args:
        database_path (str): The absolute path to the database folder.

        global_idx (int): The global index of the data to access.

        field_name (str): The name of the field to access.

    Keyword args:
        -

    Returns:
        metadata (various types): The value stored at metadata.
    """
    map_path = os.path.join(database_path, "database_map.hdf5")
    if not os.path.exists(database_path):
        raise FileNotFoundError(
            "There exists no database at '" + database_path + "'."
        )

    with h5py.File(map_path, "r") as f:
        current_global_index = f.attrs.get("current_global_index")
        if global_idx > current_global_index:
            raise ValueError(
                "No data point with index " + str(global_idx) + " exists."
            )
        metadata = f["data_" + str(global_idx)]["metadata"].attrs.get(
            field_name
        )
        if metadata is None:
            raise ValueError(
                "No metadata point with name " + field_name + " exists."
            )

    return metadata


def global_to_local_index(database_path, global_idx):
    """Find the chunk and local index of a data point from its global index.

    Args:
        database_path (str): The absolute path to the database folder.

        global_idx (int): The global index of the data to access.

    Keyword args:
        -

    Returns:
        -
    """
    map_path = os.path.join(database_path, "database_map.hdf5")
    if not os.path.exists(database_path):
        raise FileNotFoundError(
            "There exists no database at '" + database_path + "'."
        )

    with h5py.File(map_path, "r") as f:
        current_global_index = f.attrs.get("current_global_index", 0)
        if global_idx > current_global_index:
            raise ValueError(
                "No data point with index " + str(global_idx) + " exists."
            )
        data = f["data_" + str(global_idx)]
        chunk_idx = data.attrs.get("chunk_index")
        local_idx = data.attrs.get("local_index")

    return chunk_idx, local_idx


def get_reconstruction_from_database(database_path, global_idx):
    """Get a reconstruction from an hdf5 database.

    Args:
        database_path (str): The absolute path to the database folder.

        global_idx (int): The global index of the data to access.

    Keyword args:
        -

    Returns:
        reconstruction (np array[float]): The reconstruction found at
                                          global_idx.
    """
    # global_to_local raises exception if database doesn't exist.
    chunk_idx, local_idx = global_to_local_index(database_path, global_idx)

    f = get_reconstruction_chunk_handle(database_path, chunk_idx)
    reconstruction_dataset = f["reconstruction_" + str(local_idx)]
    slice_axis = reconstruction_dataset.attrs.get("slice_axis")

    if slice_axis == "x":
        reconstruction = np.transpose(reconstruction_dataset[:], (1, 2, 0))
    elif slice_axis == "y":
        reconstruction = np.transpose(reconstruction_dataset[:], (1, 0, 2))

    f.close()

    return reconstruction


def get_segmentation_from_database(database_path, global_idx):
    """Get a segmentation from an hdf5 database.

    Args:
        database_path (str): The absolute path to the database folder.

        global_idx (int): The global index of the data to access.

    Keyword args:
        -

    Returns:
        segmentation (np array[int]): The segmentation found at global_idx.
    """
    # global_to_local raises exception if database doesn't exist.
    chunk_idx, local_idx = global_to_local_index(database_path, global_idx)

    f = get_segmentation_chunk_handle(database_path, chunk_idx)
    segmentation_dataset = f["segmentation_" + str(local_idx)]
    slice_axis = segmentation_dataset.attrs.get("slice_axis")

    if slice_axis == "x":
        segmentation = np.transpose(segmentation_dataset[:], (1, 2, 0))
    elif slice_axis == "y":
        segmentation = np.transpose(segmentation_dataset[:], (1, 0, 2))

    f.close()

    return segmentation


def get_meshes_from_database(database_path, global_idx):
    """Get a set of meshes from an hdf5 database.

    Args:
        database_path (str): The absolute path to the database folder.

        global_idx (int): The global index of the data to access.

    Keyword args:
        -

    Returns:
        
        vertex_coords (list(np array[float])): Vertex coords for each
                                               phase.

        triangle_vertex_connectivity (list(np array[float])): Connectivity
                                                              matrix for each
                                                              phase.
    """
    # global_to_local raises exception if database doesn't exist.
    chunk_idx, local_idx = global_to_local_index(database_path, global_idx)
    mesh_path = os.path.join(
        database_path,
        "mesh_data/mesh_data_" + str(chunk_idx) + ".hdf5",
    )

    vertex_coords = []
    triangle_vertex_connectivity = []
    with h5py.File(mesh_path, "r") as f:
        mesh = f["mesh_" + str(local_idx)]
        keys = list(mesh.keys())
        for key in keys:
            if "vertex_coords_" in key:
                vertex_coords.append(mesh[key])
            elif "triangle_vertex_connectivity_" in key:
                triangle_vertex_connectivity.append(mesh(key))
    return vertex_coords, triangle_vertex_connectivity


def delete_data(database_path, global_idx, i_know_what_i_am_doing=False):
    """
    Delete a virtual scan and its metadata from an hdf5 database.

    ARE YOU SURE YOU WANT TO CALL THIS FUNCTION?

    Args:
        database_path (str): The absolute path to the database folder.

        global_idx (int): The global index of the data to delete.

    Keyword args:
        -

    Returns:
        -
    """
    if not i_know_what_i_am_doing:
        raise ValueError(
            "You don't seem to know what you are doing... "
            + "Deleting things is dangerous!"
        )
    if not os.path.exists(database_path):
        raise ValueError("There exists no database at '" + database_path + "'.")

    map_path = os.path.join(database_path, "database_map.hdf5")
    reconstruction_path = os.path.join(
        database_path, "reconstruction_data/reconstruction_data_"
    )
    segmentation_path = os.path.join(
        database_path, "segmentation_data/segmentation_data_"
    )
    mesh_path = os.path.join(database_path, "mesh_data/mesh_data_")

    with h5py.File(map_path, "r") as f:
        current_global_idx = f.attrs.get("current_global_index")
        current_local_idx = f.attrs.get("current_local_index")
        current_chunk_idx = f.attrs.get("current_chunk_index")
        chunk_size = f.attrs.get("chunk_size")

    if global_idx > current_global_idx:
        raise ValueError(
            "No data point with index " + str(global_idx) + " exists."
        )

    for idx in range(global_idx, current_global_idx + 1):
        chunk_idx, local_idx = global_to_local_index(database_path, idx)
        end_in = str(chunk_idx) + ".hdf5"
        end_out = str(chunk_idx - 1) + ".hdf5"

        # Special case when just deleting item instead of moving.
        if idx == global_idx:
            with h5py.File(map_path, "a") as f:
                del f["data_" + str(global_idx)]

            with h5py.File(reconstruction_path + end_in, "a") as f:
                del f["reconstruction_" + str(local_idx)]

            with h5py.File(segmentation_path + end_in, "a") as f:
                del f["segmentation_" + str(local_idx)]

            with h5py.File(mesh_path + end_in, "a") as f:
                del f["mesh_" + str(local_idx)]

            continue

        # Move data within a chunk.
        if local_idx > 0:
            with h5py.File(map_path, "a") as f:
                f.move("data_" + str(idx), "data_" + str(idx - 1))
                f["data_" + str(idx - 1)].attrs["local_index"] = local_idx - 1

            with h5py.File(reconstruction_path + end_in, "a") as f:
                f.move(
                    "reconstruction_" + str(local_idx),
                    "reconstruction_" + str(local_idx - 1),
                )

            with h5py.File(segmentation_path + end_in, "a") as f:
                f.move(
                    "segmentation_" + str(local_idx),
                    "segmentation_" + str(local_idx - 1),
                )

            with h5py.File(mesh_path + end_in, "a") as f:
                f.move("mesh_" + str(local_idx), "mesh_" + str(local_idx - 1))

        # Move data from one chunk to another chunk
        else:
            recon = get_reconstruction_from_database(database_path, idx)
            seg = get_segmentation_from_database(database_path, idx)

            vertex_coords, triangle_vertex_conenctivity = (
                get_meshes_from_database(database_path, idx)
            )

            # When we swap chunks we need to delete to make room for next move.
            with h5py.File(reconstruction_path + end_in, "a") as f:
                del f["reconstruction_0"]
            with h5py.File(segmentation_path + end_in, "a") as f:
                del f["segmentation_0"]
            with h5py.File(mesh_path + end_in, "a") as f:
                del f["mesh_0"]

            with h5py.File(map_path, "a") as f:
                f.move("data_" + str(idx), "data_" + str(idx - 1))
                f["data_" + str(idx - 1)].attrs["local_index"] = chunk_size - 1
                f["data_" + str(idx - 1)].attrs["chunk_index"] = chunk_idx - 1

            with h5py.File(reconstruction_path + end_out, "a") as f:
                f["reconstruction_" + str(chunk_size - 1)] = recon

            with h5py.File(segmentation_path + end_out, "a") as f:
                f["segmentation_" + str(chunk_size - 1)] = seg

            with h5py.File(mesh_path + end_out, "a") as f:
                mesh = f.create_group("mesh_" + str(chunk_size - 1))
                for phase in len(vertex_coords):
                    mesh["vertex_coords_" + str(phase)] = vertex_coords[phase]
                    mesh["triangle_vertex_conenctivity_" + str(phase)] = (
                        triangle_vertex_conenctivity[phase]
                    )
    # If we remove last item in chunk, delete chunk
    if current_local_idx == 0:
        os.remove(reconstruction_path + end_in)
        os.remove(segmentation_path + end_in)
        os.remove(mesh_path + end_in)

    # We have to update maps local and global index attributes at the end.
    with h5py.File(map_path, "a") as f:
        f.attrs["current_global_index"] = current_global_idx - 1
        if current_local_idx == 0:
            f.attrs["current_local_index"] = chunk_size - 1
            f.attrs["current_chunk_index"] = current_chunk_idx - 1
        else:
            f.attrs["current_local_index"] = current_local_idx - 1

    # If we remove the last item we delete the database.
    if current_global_idx - 1 < 0:
        os.rmdir(database_path + "/mesh_data")
        os.rmdir(database_path + "/reconstruction_data")
        os.rmdir(database_path + "/segmentation_data")
        os.remove(map_path)
        os.rmdir(database_path)
    return None
