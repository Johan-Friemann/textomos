import numpy as np
import tifffile
from skimage.filters import gaussian
from scipy import ndimage as ndi
from structure_tensor import eig_special_2d, structure_tensor_2d
from fem_input_LSDYNA import *


def chamis_micromechanical_model(
    E_f11, E_f22, G_f12, G_f23, v_f12, E_m, v_m, k_f
):
    """Compute homogenized composite properties with the Chamis micromechanical
       model.

    Args:
       E_f11 (float): Longitudinal Young's modulus of the fibers.

       E_f22 (float): Transverse Young's modulus of the fibers.

       G_f12 (float): Longitudinal-Transverse shear modulus of the fibers.

       G_f23 (float): Transverse-Transverse shear modulus of the fibers.

       v_f12 (float): Longitudinal-Transverse Poisson's ratio of the fibers.

       E_m (float): Young's modulus of the matrix.

       v_m (float): Poisson's ratio of the matrix.

       k_f (float): Fiber volume fraction.

    Keyword args:
        -

    Returns:
        properties (np array[float]): An array with the homogenized properties.
    """
    G_m = E_m / (2 * (1 + v_m))
    E_11 = k_f * E_f11 + (1 - k_f) * E_m
    E_22 = E_33 = E_m / (1 - np.sqrt(k_f) * (1 - E_m / E_f22))
    G_12 = G_13 = G_m / (1 - np.sqrt(k_f) * (1 - G_m / G_f12))
    G_23 = G_m / (1 - np.sqrt(k_f) * (1 - G_m / G_f23))
    v_12 = v_13 = k_f * v_f12 + (1 - k_f) * v_m
    v_23 = E_22 / (2 * G_23) - 1

    v_21 = v_12 * E_22 / E_11
    v_31 = v_13 * E_33 / E_11
    v_32 = v_23 * E_33 / E_22
    G_31 = G_13

    return np.array([E_11, E_22, E_33, v_21, v_31, v_32, G_12, G_23, G_31]).T


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
        nodal_pairs (dict[np array[int]]): A dictionary containing arrays of
                                           nodal constraint pairs for faces
                                           and edges. The shapes are num_pairs
                                           (per face/edge) by 2. The corners are
                                           stored as points.
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
    nodal_pairs["top_left_to_bottom_left"] = np.column_stack(
        (top_left.flatten(), bottom_left.flatten())
    )
    nodal_pairs["bottom_back_to_bottom_front"] = np.column_stack(
        (bottom_back.flatten(), bottom_front.flatten())
    )
    nodal_pairs["front_right_to_front_left"] = np.column_stack(
        (front_right.flatten(), front_left.flatten())
    )
    nodal_pairs["back_left_to_front_left"] = np.column_stack(
        (back_left.flatten(), front_left.flatten())
    )
    nodal_pairs["bottom_right_to_bottom_left"] = np.column_stack(
        (bottom_right.flatten(), bottom_left.flatten())
    )
    nodal_pairs["top_front_to_bottom_front"] = np.column_stack(
        (top_front.flatten(), bottom_front.flatten())
    )
    nodal_pairs["top_right_to_bottom_left"] = np.column_stack(
        (top_right.flatten(), bottom_left.flatten())
    )
    nodal_pairs["top_back_to_bottom_front"] = np.column_stack(
        (top_back.flatten(), bottom_front.flatten())
    )
    nodal_pairs["back_right_to_front_left"] = np.column_stack(
        (back_right.flatten(), front_left.flatten())
    )

    # Vertices

    nodal_pairs["top_back_right"] = nodes[-1, -1, -1]
    nodal_pairs["top_front_right"] = nodes[-1, 0, -1]
    nodal_pairs["bottom_back_left"] = nodes[0, -1, 0]
    nodal_pairs["bottom_front_left"] = nodes[0, 0, 0]
    nodal_pairs["top_front_left"] = nodes[-1, 0, 0]
    nodal_pairs["bottom_back_right"] = nodes[0, -1, -1]
    nodal_pairs["top_back_left"] = nodes[-1, -1, 0]
    nodal_pairs["bottom_front_right"] = nodes[0, 0, -1]

    return nodal_pairs


def structure_tensor_analysis(
    segmentation, material_classes, slice_axes, rho=2.0, sigma=0.25
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
            distance_field += (
                np.random.random_sample(size=distance_field.shape) * 0.1
            )
            distance_field = gaussian(distance_field)
            S = structure_tensor_2d(distance_field, sigma, rho)
            _, vec = eig_special_2d(S)

            # += because we dont want to zero the previous yarn type
            if slice_axis == 0:  # We align vectors with + y-axis
                x[s] += slice * vec[1] * np.sign(vec[0])
                y[s] += np.abs(slice * vec[0])
            elif slice_axis == 1:  # We align vectors with + z-axis
                x[s] += slice * vec[1] * np.sign(vec[0])
                z[s] += np.abs(slice * vec[0])
            else:
                y[s] += slice * vec[1]
                z[s] += slice * vec[0]
    return np.column_stack((x.flatten(), y.flatten(), z.flatten()))


def volume_fraction_analysis(
    segmentation,
    orientations,
    material_classes,
    slice_axes,
    expected_areas,
    voxel_area,
):
    """Perform a slice wise analysis of yarn fiber volume fraction. Perform
       the analyis per material class.

    Args:
        segmentation (np array[int]): The segmentation to perform analysis on.

        orientation (np array[float]): The fiber orientations.

        material_classes (list [int]): A list of ints corresponding to the
                                       material class indices to analyze.

        slice_axes (list [int]): A list of ints corresponding to the directions
                                 to slice the segmentation while performing
                                 analyses. It should have the same length as
                                 material_classes.

        expected_areas (list [float]): The expected area of all the fibers in a
                                       slice. It should be the same length
                                       as material_classes. Typically computed:
                                       num_yarns*fiber_area*num_fiber_per_yarn.
                                       Should be given in the same unit as
                                       voxel area.

        voxel area (float): The area of a voxel (pixel).

    Keyword args:
        -

    Returns:
        volume_fractions (np array[float]): An array with the same shape as the
                                            input segmenation. The entires are
                                            the fiber volume fractions.
                                            Unassigned voxels are assigned 0.0.
    """
    dims = segmentation.shape
    volume_fractions = np.zeros(dims)
    for material_class, slice_axis, expected_area in zip(
        material_classes, slice_axes, expected_areas
    ):
        material = segmentation == material_class
        for idx in range(dims[slice_axis]):
            if slice_axis == 0:
                s = np.index_exp[idx, :, :]
                scale = orientations[:, 2].reshape(dims[0], dims[1], dims[2])[s]
            elif slice_axis == 1:
                s = np.index_exp[:, idx, :]
                scale = orientations[:, 1].reshape(dims[0], dims[1], dims[2])[s]
            else:
                s = np.index_exp[:, :, idx]
                scale = orientations[:, 0].reshape(dims[0], dims[1], dims[2])[s]
            slice = material[s]
            num_voxels = np.sum(slice * scale)
            # += because we dont want to zero the previous yarn type
            volume_fractions[s] += slice * (
                expected_area / (num_voxels * voxel_area)
            )
    return volume_fractions


def fem_input_from_tiff(
    in_path,
    voxel_size,
    constituent_properties,
    out_path,
    num_bins=20,
    load_magnitude=1.0,
    load_case="epsilon_22",
    code="LSDYNA",
):
    """Build a unit cell analysis input file from a segmentation TIFF-file.

    Args:
        in_path (str): The absolute path to the TIFF to build the mesh from.

        voxel_size (float): The reconstruction voxel size corresponding to the
                            segmentation.

        constituent_properties (list [float]): A list of floats corresponding to
                                               the fiber and matrix constituent
                                               mechanical properties.

        out_path (str): The absolute path to where to save the FE-code input.

    Keyword args:
        num_bins (int): The number of distinct volume fraction bins to use. This
                        is used to avoid creating too many material "cards"
                        in the input files.

        load_magnitude (float): The strain magnitude to apply in the unit cell
                                analysis.

        load_case (str): What load case to run. Can be: "epsilon_11",
                         "epsilon_22", "epsilon_33", "epsilon_12", "epsilon_23",
                         or "epsilon_13".

        code (str): For what FE-code should the input file be created. Currently
                    only supports "LSDYNA".

    Returns:
        None
    """
    segmentation = tifffile.imread(in_path)
    segmentation[segmentation == 0] = 3  # We set spurious air to matrix
    dims = segmentation.shape  # We have to bear in mind that z is stored first
    rve_shape = [
        dims[2] * voxel_size,
        dims[1] * voxel_size,
        dims[0] * voxel_size,
    ]

    elements = create_elements(dims[2], dims[1], dims[0])
    points = create_nodes(dims[2], dims[1], dims[0], voxel_size)
    nodal_pairs = create_boundary_node_pairs(dims[2], dims[1], dims[0])
    orientation = structure_tensor_analysis(segmentation, (1, 2), (1, 0))
    vol_fraction = volume_fraction_analysis(
        segmentation,
        orientation,
        (1, 2),
        (0, 1),
        (
            4 * 7 * 12000 * 5.2e-6**2 * np.pi / 4,
            8 * 6 * 24000 * 5.2e-6**2 * np.pi / 4,
        ),
        voxel_size * voxel_size,
    )

    lower = np.min(vol_fraction[np.nonzero(vol_fraction)])
    upper = np.max(vol_fraction[np.nonzero(vol_fraction)])
    bins = np.zeros(num_bins)
    bins[1:] = np.linspace(lower, upper, num_bins - 1)
    bin_centers = np.zeros(num_bins - 1)
    bin_centers[1:] = (bins[2:] + bins[1:-1]) / 2
    mat_props = chamis_micromechanical_model(
        constituent_properties[0],
        constituent_properties[1],
        constituent_properties[2],
        constituent_properties[3],
        constituent_properties[4],
        constituent_properties[5],
        constituent_properties[6],
        bin_centers,
    )
    bins[
        -1
    ] += 0.1  # We want both lower and upper inclusivity so we shift right
    vol_fraction_bin_ids = np.digitize(vol_fraction.flatten(), bins)

    if code == "LSDYNA":
        file = open(out_path, "w")
        write_header_LSDYNA(1.0, file)
        write_materials_LSDYNA(mat_props, file)
        write_nodes_LSDYNA(points, file)
        write_elements_LSDYNA(elements, orientation, vol_fraction_bin_ids, file)
        write_periodic_constraints_LSDYNA(nodal_pairs, file)
        write_load_constraints_LSDYNA(
            nodal_pairs, rve_shape, load_magnitude, load_case, file
        )
        write_footer_LSDYNA(file)
        file.close()
    else:
        raise NotImplementedError("Only LSDYNA implemented!")

    return None
