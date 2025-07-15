import numpy as np


def create_nodes(n_x, n_y, n_z, voxel_size):
    """DOCSTRING"""
    nodes = np.zeros(((n_x + 1) * (n_y + 1) * (n_z + 1), 3))


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

    offsets = np.array([
        0,
        1,
        1 + (n_x + 1),
        (n_x + 1),
        n_layer,
        n_layer + 1,
        n_layer + 1 + (n_x + 1),
        n_layer + (n_x + 1),
    ])
    
    return element_bottom_left_corners[:, None] + offsets[None, :]

def create_boundary_conditions(n_x, n_y, n_z):
    """DOCSTRING"""
