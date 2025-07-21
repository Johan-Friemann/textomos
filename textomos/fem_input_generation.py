import numpy as np
import tifffile
import meshio
from skimage.filters import gaussian
from scipy import ndimage as ndi
from structure_tensor import eig_special_2d, structure_tensor_2d


def create_nodes(n_x, n_y, n_z, voxel_size):
    """Create the nodes of a structured mesh with linear hexahedral elements.

    Args:
        n_x (int): Number of elements in the x-direction.

        n_y (int): Number of elements in the y-direction.

        n_z (int): Number of elements in the z-direction.

        voxel_size (float): The voxel side length.

    Keyword args:
        -

    Returns:
        nodes (np array[float]): A (n_x+1)*(n_y+1)*(n_z+1) by 3 array with the
                                 nodal coordinates.
    """
    k, j, i = np.meshgrid(
        np.arange(n_z + 1),
        np.arange(n_y + 1),
        np.arange(n_x + 1),
        indexing="ij",
    )

    i = i.ravel()
    j = j.ravel()
    k = k.ravel()

    x = i * voxel_size
    y = j * voxel_size
    z = k * voxel_size
    return np.column_stack((x, y, z))


def create_elements(n_x, n_y, n_z):
    """Create the connectivity for a structured mesh with linear hexahedral
       elements.

    Args:
        n_x (int): Number of elements in the x-direction.

        n_y (int): Number of elements in the y-direction.

        n_z (int): Number of elements in the z-direction.

    Keyword args:
        -

    Returns:
        elements (np array[int]): An n_x*n_y*n_z by 8 array with the element
                                  node connectivities.
    """
    n_layer = (n_x + 1) * (n_y + 1)

    k, j, i = np.meshgrid(
        np.arange(n_z), np.arange(n_y), np.arange(n_x), indexing="ij"
    )

    i = i.ravel()
    j = j.ravel()
    k = k.ravel()

    element_bottom_left_corners = i + j * (n_x + 1) + k * (n_x + 1) * (n_y + 1)

    offsets = np.array(
        [
            0,
            1,
            1 + (n_x + 1),
            (n_x + 1),
            n_layer,
            n_layer + 1,
            n_layer + 1 + (n_x + 1),
            n_layer + (n_x + 1),
        ]
    )

    return element_bottom_left_corners[:, None] + offsets[None, :]


def create_boundary_node_pairs(n_x, n_y, n_z):
    """Create the boundary node pairs needed for periodic boundary condtion
       constraints.


           +--------+     z
          /        /|     |  y
         /        / |     | /
        +--------+  |     |/____x
        |        |  |
        |        |  +
        |        | /
        |        |/
        +--------+

    Face naming along increasing coordinates:
        x-axis: Left -> Right
        y-axis: Front -> Back
        z axis: Bottom -> Top

    Edges and vertices are named according to what faces they belong to.

    Args:
        n_x (int): Number of elements in the x-direction.

        n_y (int): Number of elements in the y-direction.

        n_z (int): Number of elements in the z-direction.

    Keyword args:
        -

    Returns:
        nodal_pairs (dict[np array[int]]): ---
    """
    nodes = np.arange((n_x + 1) * (n_y + 1) * (n_z + 1)).reshape(
        (n_z + 1), (n_y + 1), (n_x + 1)
    )
    nodal_pairs = {}

    # Faces
    top = nodes[-1, 1:-1, 1:-1]
    bottom = nodes[0, 1:-1, 1:-1]
    front = nodes[1:-1, 0, 1:-1]
    back = nodes[1:-1, -1, 1:-1]
    right = nodes[1:-1, 1:-1, -1]
    left = nodes[1:-1, 1:-1, 0]
    nodal_pairs["top_to_bottom"] = np.column_stack(
        (top.flatten(), bottom.flatten())
    )
    nodal_pairs["right_to_left"] = np.column_stack(
        (right.flatten(), left.flatten())
    )
    nodal_pairs["back_to_front"] = np.column_stack(
        (back.flatten(), front.flatten())
    )

    # Edges
    back_right = nodes[1:-1, -1, -1]
    top_back = nodes[-1, -1, 1:-1]
    top_right = nodes[-1, 1:-1, -1]
    bottom_left = nodes[0, 1:-1, 0]
    top_left = nodes[-1, 1:-1, 0]
    bottom_right = nodes[0, 1:-1, -1]
    top_front = nodes[-1, 0, 1:-1]
    bottom_back = nodes[0, -1, 1:-1]
    bottom_front = nodes[0, 0, 1:-1]
    front_left = nodes[1:-1, 0, 0]
    front_right = nodes[1:-1, 0, -1]
    back_left = nodes[1:-1, -1, 0]
    nodal_pairs["back_right_to_back_left"] = np.column_stack(
        (back_right.flatten(), back_left.flatten())
    )
    nodal_pairs["back_right_to_front_right"] = np.column_stack(
        (back_right.flatten(), front_right.flatten())
    )
    nodal_pairs["back_right_to_front_left"] = np.column_stack(
        (back_right.flatten(), front_left.flatten())
    )
    nodal_pairs["top_back_to_bottom_back"] = np.column_stack(
        (top_back.flatten(), bottom_back.flatten())
    )
    nodal_pairs["top_back_to_top_front"] = np.column_stack(
        (top_back.flatten(), top_front.flatten())
    )
    nodal_pairs["top_back_to_bottom_front"] = np.column_stack(
        (top_back.flatten(), bottom_front.flatten())
    )
    nodal_pairs["top_right_to_top_left"] = np.column_stack(
        (top_right.flatten(), top_left.flatten())
    )
    nodal_pairs["top_right_to_bottom_right"] = np.column_stack(
        (top_right.flatten(), bottom_right.flatten())
    )
    nodal_pairs["top_right_to_bottom_left"] = np.column_stack(
        (top_right.flatten(), bottom_left.flatten())
    )

    # Vertices
    top_back_right = nodes[-1, -1, -1]
    top_front_right = nodes[-1, 0, -1]
    bottom_back_left = nodes[0, -1, 0]
    bottom_front_left = nodes[0, 0, 0]
    top_front_left = nodes[-1, 0, 0]
    bottom_back_right = nodes[0, -1, -1]
    top_back_left = nodes[-1, -1, 0]
    bottom_front_right = nodes[0, 0, -1]
    nodal_pairs["top_back_right_to_top_front_right"] = np.column_stack(
        (top_back_right.flatten(), top_front_right.flatten())
    )
    nodal_pairs["top_back_right_to_top_back_left"] = np.column_stack(
        (top_back_right.flatten(), top_back_left.flatten())
    )
    nodal_pairs["top_back_right_to_bottom_back_right"] = np.column_stack(
        (top_back_right.flatten(), bottom_back_right.flatten())
    )
    nodal_pairs["top_back_right_to_bottom_back_left"] = np.column_stack(
        (top_back_right.flatten(), bottom_back_left.flatten())
    )
    nodal_pairs["top_back_right_to_top_front_left"] = np.column_stack(
        (top_back_right.flatten(), top_front_left.flatten())
    )
    nodal_pairs["top_back_right_to_bottom_front_right"] = np.column_stack(
        (top_back_right.flatten(), bottom_front_right.flatten())
    )
    nodal_pairs["top_back_right_to_bottom_front_left"] = np.column_stack(
        (top_back_right.flatten(), bottom_front_left.flatten())
    )

    return nodal_pairs


def structure_tensor_analysis(
    segmentation, material_classes, slice_axes, rho=4.0, sigma=0.25
):
    """Perform a slice wise structure tensor analysis to infer material
       orientation. The analisis is performed per material class in
       different directions (can be the same).

    Args:
        segmentation (np array[int]): The segmentation to perform analysis on.

        material_classes (list [int]): A list of ints corresponding to the
                                       material class indices to analyze.

        slice_axes (list [int]): A list of ints corresponding to the directions
                                 to slice the segmentation while performing
                                 analyses. It should have the same length as
                                 material_classes.

    Keyword args:
        rho (float): The rho parameter in the structure tensor analyis code.

        sigma (float): The sigma parameter in the structure tensor analyis code.

    Returns:
        orientations (np array[float]): An n_x*n_y*n_z by 3 array containing the
                                        orientation vectors per voxel.
                                        Unassigned voxels (matrix etc) are
                                        zero vectors.
    """
    dims = segmentation.shape
    x = np.zeros(dims)
    y = np.zeros(dims)
    z = np.zeros(dims)

    for material_class, slice_axis in zip(material_classes, slice_axes):
        material = segmentation == material_class
        for idx in range(dims[slice_axis]):
            if slice_axis == 0:
                s = np.index_exp[idx, :, :]
            elif slice_axis == 1:
                s = np.index_exp[:, idx, :]
            else:
                s = np.index_exp[:, :, idx]
            slice = material[s]
            distance_field = ndi.distance_transform_edt(slice)
            distance_field = gaussian(distance_field)
            S = structure_tensor_2d(distance_field, sigma, rho)
            _, vec = eig_special_2d(S)

            # += because we dont want to zero the previous yarn type
            if slice_axis == 0:
                x[s] += slice * vec[1]
                y[s] += slice * vec[0]
            elif slice_axis == 1:
                x[s] += slice * vec[1]
                z[s] += slice * vec[0]
            else:
                y[s] += slice * vec[1]
                z[s] += slice * vec[0]
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))
