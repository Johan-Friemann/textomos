import numpy as np
import pickle as pk
import trimesh
import scipy.interpolate as inp
from scipy.spatial.transform import Rotation as R


"""
This file contains a program for generating orthogonal-type 3D-textile meshes
with domain randomization.
"""


def generate_yarn_topology(num_nodes, points_per_node):
    """Generate the topology for one yarn in the form of triangle connectivity
    between points. The topology pertains an identical number of points per
    node along the yarn.

    Args:


    Keyword args:
        -

    Returns:
        triangles (numpy array[int]): An array of shape with three columns and
                                      the number of triangle rows. The entries
                                      correspond to the point indices in each
                                      triangle.
    """

    tube_idx = np.arange(0, (num_nodes - 1) * points_per_node)
    tube = np.reshape(
        np.stack(
            (
                tube_idx,
                (tube_idx + 1) % points_per_node
                + points_per_node * (tube_idx // points_per_node)
                + points_per_node,
                tube_idx + points_per_node,
                tube_idx,
                (tube_idx + 1) % points_per_node
                + points_per_node * (tube_idx // points_per_node),
                (tube_idx + 1) % points_per_node
                + points_per_node * (tube_idx // points_per_node)
                + points_per_node,
            ),
            axis=1,
        ),
        (-1, 3),
    )

    # Odd number of number of points make one more "bottom heavy" triangle.
    cap_idx_1 = np.arange(
        0, (points_per_node - 2) // 2 + 1 * (points_per_node % 2 != 0)
    )
    cap_idx_2 = np.arange(0, (points_per_node - 2) // 2)
    start_cap = np.vstack(
        (
            np.stack(
                (cap_idx_1, points_per_node - 1 - cap_idx_1, cap_idx_1 + 1),
                axis=1,
            ),
            np.stack(
                (
                    points_per_node - 1 - cap_idx_2,
                    points_per_node - 2 - cap_idx_2,
                    cap_idx_2 + 1,
                ),
                axis=1,
            ),
        ),
    )
    end_cap = np.vstack(
        (
            np.stack(
                (
                    points_per_node * (num_nodes - 1) + cap_idx_1,
                    points_per_node * (num_nodes - 1) + cap_idx_1 + 1,
                    points_per_node * num_nodes - cap_idx_1 - 1,
                ),
                axis=1,
            ),
            np.stack(
                (
                    points_per_node * num_nodes - 1 - cap_idx_2,
                    points_per_node * (num_nodes - 1) + cap_idx_2 + 1,
                    points_per_node * num_nodes - 2 - cap_idx_2,
                ),
                axis=1,
            ),
        ),
    )

    triangles = np.vstack((tube, start_cap, end_cap))
    return triangles


def generate_yarn_spline(
    sample_points,
    interpolation_parameter,
    points_per_node,
    horizontal_half_axis,
    vertical_half_axis,
    super_ellipse_power,
    deform=[],
    sampling_step=20,
    direction=0,
    smoothing=0.0,
    flat_top=False,
    flat_bottom=False,
):
    """Generate points that make up a yarn with a bspline. Can deform the cross
    sections as an option.

    Args:
        sample_points (numpy array[float]): An array with num of interpolation
                                            nodes rows and 3 columns. Each row
                                            correspodns to the coordinates of
                                            the interpolation nodes.

        interpolation_parameter (numpy array[float]): A monotonically increasing
                                                      list between 0.0 and 1.0
                                                      that decides at what
                                                      parameter value to
                                                      evaluate the spline.

        points_per_node (int): The number of points per yarn node
                              (points per cross section), n.b. notinterpolation
                              nodes). These points are the ones that make up the
                              triangles.

        super_ellipse_power (float): The exponent of the super ellipse.

        horizontal_half_axis (float): The horizontal half axis of the cross
                                      section super ellipse.

        vertical_half_axis (float): The vertical half axis of the cross
                                      section super ellipse.

    Keyword args:
        deform (list[float]): A list of length 4 where the entries refer to the
                              maximum possible:
                              1: percentual deformation in horizontal direction.
                              2: percentual deformation in vertical direction.
                              3: percentual change in super ellipse exponent.
                              4: rotation in degrees around spline tangent.

        sampling_step (int): Randomly sample the deformation at every
                             sampling_step node (n.b. not interpolation nodes)
                             and interpolate deformation linearly between these.

        direction (int): If 0 the yarn is in the x-direction, if 1 it is in the
                         y-direction.

        smoothing (float): B-spline smoothing parameter.


        flat_top (bool): Will make the top of a yarn flat if True.

        flat_bottom (bool): Will make the bottom of a yarn flat if True.

    Returns:
        points (numpy array[float]): An array of shape number of points rows
                                     and 3 columns containing the coordinates
                                     of all points making up the triangles in
                                     the yarn.

        center_line (numpy array[float]): An array of shape number of yarn
                                          cross section rows, and 3 columns
                                          containing the coordinates of all
                                          cross section centers along the yarn.
    """
    if deform == []:
        deform = 4 * [0.0]

    num_nodes = len(interpolation_parameter)
    t = np.linspace(0, 1, len(sample_points))
    x = sample_points[:, 0]
    y = sample_points[:, 1]
    z = sample_points[:, 2]

    spline = inp.make_splprep([t, x, y, z], s=smoothing)[0]
    dspline = spline.derivative()

    rs = spline(interpolation_parameter).T[:, 1:]
    drs = dspline(interpolation_parameter).T[:, 1:]
    ts = drs / np.linalg.norm(drs, axis=1)[:, np.newaxis]
    bs = np.tile((direction, 1 - direction, 0), (num_nodes, 1))
    ns = np.linalg.cross(bs, ts, axis=1)
    # Below needed to prevent inversion when snapping to bounding box.
    sgn = np.sign(ns[0, 2])
    ns[0] = np.array([0, 0, sgn])
    ns[-1] = np.array([0, 0, sgn])

    Rs = np.repeat(rs, points_per_node, axis=0)
    Ts = np.repeat(ts, points_per_node, axis=0)
    Ns = np.repeat(ns, points_per_node, axis=0)
    Bs = np.repeat(bs, points_per_node, axis=0)
    if deform[0] != 0.0:
        interpolated_steps = np.linspace(0, 1, num_nodes)
        sampled_steps = np.linspace(0, 1, num_nodes // sampling_step)
        sampled_deforms = (
            2
            * (np.random.rand(num_nodes // sampling_step) - 0.5)
            * horizontal_half_axis
            * deform[0]
            / 100.0
        )
        horizontal_deforms = np.interp(
            interpolated_steps, sampled_steps, sampled_deforms
        )

        tiled_horizontal_deforms = np.repeat(
            horizontal_deforms, points_per_node
        )
    else:
        tiled_horizontal_deforms = np.zeros(num_nodes * points_per_node)
    if deform[1] != 0.0:
        interpolated_steps = np.linspace(0, 1, num_nodes)
        sampled_steps = np.linspace(0, 1, num_nodes // sampling_step)
        sampled_deforms = (
            2
            * (np.random.rand(num_nodes // sampling_step) - 0.5)
            * horizontal_half_axis
            * deform[1]
            / 100.0
        )
        vertical_deforms = np.interp(
            interpolated_steps, sampled_steps, sampled_deforms
        )
        tiled_vertical_deforms = np.repeat(vertical_deforms, points_per_node)
    else:
        tiled_vertical_deforms = np.zeros(num_nodes * points_per_node)
    if deform[2] != 0.0:
        interpolated_steps = np.linspace(0, 1, num_nodes)
        sampled_steps = np.linspace(0, 1, num_nodes // sampling_step)
        sampled_deforms = (
            2
            * (np.random.rand(num_nodes // sampling_step) - 0.5)
            * super_ellipse_power
            * deform[2]
            / 100.0
        )
        super_ellipse_power_deforms = np.interp(
            interpolated_steps, sampled_steps, sampled_deforms
        )
        tiled_super_ellipse_power_deforms = np.repeat(
            super_ellipse_power_deforms, points_per_node
        )
    else:
        tiled_super_ellipse_power_deforms = np.zeros(
            num_nodes * points_per_node
        )
    if deform[3] != 0.0:
        interpolated_steps = np.linspace(0, 1, num_nodes)
        sampled_steps = np.linspace(0, 1, num_nodes // sampling_step)
        sampled_rotations = (
            2
            * (np.random.rand(num_nodes // sampling_step) - 0.5)
            * deform[3]
            * np.pi
            / 180.0
        )
        rotation_deforms = np.interp(
            interpolated_steps, sampled_steps, sampled_rotations
        )
        tiled_rotation_deforms = np.repeat(rotation_deforms, points_per_node)
        Rots = R.from_rotvec(tiled_rotation_deforms[:, np.newaxis] * Ts)
        Ns = Rots.apply(Ns)
        Bs = Rots.apply(Bs)

    vertical_asymmetry = np.ones(points_per_node)
    # We can safely shift r here since it has already been used for R
    # We need to shift by half the half axis since the the flattening will
    # affect the center line.
    if flat_top:
        rs[:,2] -= vertical_half_axis/2
        vertical_asymmetry[: points_per_node // 2] = 0.0
    if flat_bottom:
        rs[:,2] += vertical_half_axis/2
        vertical_asymmetry[points_per_node // 2 :] = 0.0

    tiled_vertical_asymmetry = np.tile(vertical_asymmetry, num_nodes)

    angs = -np.linspace(0, 2 * np.pi, points_per_node, endpoint=False)
    tiled_angs = np.tile(angs, num_nodes)
    points = (
        Rs
        + np.power(
            np.abs(np.cos(tiled_angs)[:, np.newaxis]),
            super_ellipse_power
            * (1 + tiled_super_ellipse_power_deforms[:, np.newaxis]),
        )
        * np.sign(np.cos(tiled_angs)[:, np.newaxis])
        * horizontal_half_axis
        * (1 + tiled_horizontal_deforms[:, np.newaxis])
        * Bs
        + np.power(
            np.abs(np.sin(tiled_angs)[:, np.newaxis]),
            super_ellipse_power
            * (1 + tiled_super_ellipse_power_deforms[:, np.newaxis]),
        )
        * np.sign(np.sin(tiled_angs))[:, np.newaxis]
        * vertical_half_axis
        * tiled_vertical_asymmetry[:, np.newaxis]
        * (1 + tiled_vertical_deforms[:, np.newaxis])
        * Ns
    )

    return points, rs


def generate_in_plane_sample_points(
    cell_shape,
    start_coord,
    num_crossing,
    crossing_width,
    direction=0,
    crimp=0.0,
):
    """Generate the sample points (interpolation nodes) for an in-plane yarn.

    Args:
        cell_shape (list[float]): A list of length 3 that determines the size
                                  in the x, y, and z-directions of the textile
                                  bounding box.

        start_coord (list[float]): A list of length 2 that determines the
                                   starting coordinates in the  directions
                                   orthogonal to the yarn. (y-yarn would be
                                   x and z etc.)

        num_crossing (int): The number of yarns in the orthogonal direction that
                            the yarn will cross over.

        crossing_width (float): The width of the yarns to cross over in the
                                orthogonal direction.

    Keyword args:
        direction (int): If 0 the yarn is in the x-direction, if 1 it is in the
                         y-direction.

        crimp (float): Determines how much to push the cross over nodes up (+)
                       or down (-) to generate crimp.

    Returns:
        sample_points (numpy array[float]): An array with the number of
                                            interpolation nodes rows and 3
                                            columns containing the interpolation
                                            node coordinates.
    """
    sample_points = np.zeros((num_crossing * 4 + 1, 3), dtype=float)

    spacing = cell_shape[direction] / num_crossing
    key_points = np.empty(num_crossing * 4 + 1, dtype=float)
    key_points[0::4] = np.linspace(
        -cell_shape[direction] / 2,
        cell_shape[direction] / 2,
        num_crossing + 1,
    )
    key_points[1::4] = np.linspace(
        -cell_shape[direction] / 2 + (spacing - crossing_width) / 2,
        cell_shape[direction] / 2 - spacing + (spacing - crossing_width) / 2,
        num_crossing,
    )
    key_points[2::4] = np.linspace(
        -cell_shape[direction] / 2 + spacing / 2,
        cell_shape[direction] / 2 - spacing / 2,
        num_crossing,
    )
    key_points[3::4] = np.linspace(
        -cell_shape[direction] / 2 + (spacing + crossing_width) / 2,
        cell_shape[direction] / 2 + (crossing_width - spacing) / 2,
        num_crossing,
    )

    sample_points[:, direction] = key_points
    sample_points[:, 1 - direction] = start_coord[0]
    sample_points[:, 2] = start_coord[1]
    sample_points[::8, 2] -= crimp
    sample_points[4::8, 2] += crimp

    return sample_points


def generate_out_of_plane_sample_points(
    cell_shape,
    position,
    num_crossing,
    binder_thickness,
    roundness=0.8,
    direction=0,
):
    """Generate the sample points (interpolation nodes) for an out-of-plane
       yarn.

    Args:
        cell_shape (list[float]): A list of length 3 that determines the size
                                  in the x, y, and z-directions of the textile
                                  bounding box.

        position (list[float]): A list of length 2 that determines the
                                starting coordinates in the  directions
                                orthogonal to the yarn. The first coordinate
                                determines the position in the side ways
                                direction and the second coordinate is 1 for
                                yarns starting at the top and -1 for yarns
                                starting at the bottom.

        num_crossing (int): The number of yarns in the orthogonal direction that
                            the yarn will cross over.

        binder_thickness (float): The thickness of the out of plane yarn.

    Keyword args:
        direction (int): If 0 the yarn is in the x-direction, if 1 it is in the
                         y-direction.

        roundness (float): A parameter that determines the roundness of the
                           out of plane yarns where they change their
                           vertical direction.

    Returns:
        sample_points (numpy array[float]): An array with the number of
                                            interpolation nodes rows and 3
                                            columns containing the interpolation
                                            node coordinates.
    """
    spacing = cell_shape[direction] / num_crossing
    key_points = np.zeros(4 * num_crossing - 1, dtype=float)
    key_points[::4] = np.linspace(
        -cell_shape[direction] / 2,
        cell_shape[direction] / 2 - spacing,
        num_crossing,
    )
    key_points[1::4] = np.linspace(
        -cell_shape[direction] / 2 + spacing / 2,
        cell_shape[direction] / 2 - spacing / 2,
        num_crossing,
    )
    key_points[2::4] = np.linspace(
        -cell_shape[direction] / 2 + spacing,
        cell_shape[direction] / 2,
        num_crossing,
    )
    key_points[3::4] = np.linspace(
        -cell_shape[direction] / 2 + spacing,
        cell_shape[direction] / 2 - spacing,
        num_crossing - 1,
    )

    sample_points = np.zeros((4 * num_crossing - 1, 3), dtype=float)
    sample_points[:, direction] = key_points
    sample_points[:, 1 - direction] = position[0]
    sample_points[::8, 2] = (
        -roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[1::8, 2] = (
        -(cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[2::8, 2] = -(
        roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[4::8, 2] = (
        roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[5::8, 2] = (
        cell_shape[2] / 2 - binder_thickness / 2
    ) * position[1]
    sample_points[6::8, 2] = (
        roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )

    return sample_points


def generate_weft_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layers,
    warp_width,
    weft_width,
    weft_thickness,
    binder_thickness,
    super_ellipse_power,
    num_stacks,
    deform=[],
    internal_crimp=True,
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):
    """Generate the weft yarn meshes.

    Args:
        cell_shape (list[float]): A list of length 3 that determines the size
                                  in the x, y, and z-directions of the textile
                                  bounding box.

        num_warp_per_layer (int): The number of warp yarns per layer.

        num_weft_per_layer (int): The number of weft yarns per layer.

        num_layers (int): The number of yarn layers in the textile (per stack).
                          The weft yarns have one additional layer (top layer).

        warp_width (float): The width of the warp yarns.

        weft_width (float): The width of the weft yarns.

        weft_thickness (float): The thickness of the weft yarns.

        binder_thickness (float): The thickness of the binder yarns.

        super_ellipse_power (float): The weft yarn cross section super ellipse
                                     exponent.

        num_stacks (int): The number of times to stack the textile in the
                          out-of-plane direction.

    Keyword args:
        deform (list[float]): A list of length 4 where the entries refer to the
                              maximum possible:
                              1: percentual deformation in horizontal direction.
                              2: percentual deformation in vertical direction.
                              3: percentual change in super ellipse exponent.
                              4: rotation in degrees around spline tangent.

        internal_crimp (bool): Will add crimp to the internal weft yarns if
                               True. Otherwise only the top and bottom yarns
                               have crimp.

        nodes_per_yarn (int): The number of nodes per yarn (number of cross
                              sections).

        points_per_node (int): The number of points per yarn node
                              (points per cross section), n.b. notinterpolation
                              nodes). These points are the ones that make up the
                              triangles.

        smoothing (float): B-spline smoothing parameter.

    Returns:
        points (list[numpy array[float]]): A list where each entry corresponds
                                           to the point array of one yarn.

        triangles (list[numpy array[int]]): A list where each entry corresponds
                                            to the triangle array of one yarn.

        center_lines (list[numpy array[int]]): A list where each entry
                                               corresponds to the center line
                                               array of one yarn.
    """
    start_ys = np.linspace(
        -cell_shape[1] / 2 + cell_shape[1] / num_weft_per_layer / 2,
        cell_shape[1] / 2 - cell_shape[1] / num_weft_per_layer / 2,
        num_weft_per_layer,
    )
    start_zs = np.linspace(
        -cell_shape[2] / 2 + binder_thickness / 2 + weft_thickness / 2,
        cell_shape[2] / 2 - binder_thickness / 2 - weft_thickness / 2,
        num_layers + 1,
    )
    start_zs[0] += weft_thickness / 2
    start_zs[-1] -= weft_thickness / 2
    interpolation_parameter = np.linspace(0, 1, nodes_per_yarn)

    points = []
    triangles = []
    center_lines = []
    offset = cell_shape[2] * (1 - num_stacks) / 2
    for stack in range(num_stacks):
        for idxz, z in enumerate(start_zs):
            for idxy, y in enumerate(start_ys):
                sample_points = generate_in_plane_sample_points(
                    cell_shape,
                    [y, z + offset],
                    num_warp_per_layer,
                    warp_width,
                    direction=0,
                    crimp=((idxz == 0 or idxz == num_layers) or internal_crimp)
                    * (1 - 2 * (idxy % 2))
                    * binder_thickness
                    / 2,
                )
                point, center_line = generate_yarn_spline(
                    sample_points,
                    interpolation_parameter,
                    points_per_node,
                    weft_width / 2,
                    weft_thickness / 2
                    + (weft_thickness / 2) * (idxz == 0 or idxz == num_layers),
                    (
                        1
                        if (idxz == 0 or idxz == num_layers)
                        else super_ellipse_power
                    ),
                    direction=0,
                    deform=deform,
                    smoothing=smoothing,
                    flat_top=idxz == 0,
                    flat_bottom=idxz == num_layers,
                )
                triangle = generate_yarn_topology(
                    nodes_per_yarn, points_per_node
                )
                points.append(point)
                triangles.append(triangle)
                center_lines.append(center_line)
        offset += cell_shape[2]

    return points, triangles, center_lines


def generate_warp_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layers,
    warp_width,
    warp_thickness,
    weft_width,
    weft_thickness,
    binder_thickness,
    super_ellipse_power,
    num_stacks,
    deform=[],
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):
    """Generate the warp yarn meshes.

    Args:
        cell_shape (list[float]): A list of length 3 that determines the size
                                  in the x, y, and z-directions of the textile
                                  bounding box.

        num_warp_per_layer (int): The number of warp yarns per layer.

        num_weft_per_layer (int): The number of weft yarns per layer.

        num_layers (int): The number of yarn layers in the textile (per stack).
                          The weft yarns have one additional layer (top layer).

        warp_width (float): The width of the warp yarns.

        warp_thickness (float): The thickness of the weft yarns.

        weft_width (float): The width of the weft yarns.

        weft_thickness (float): The thickness of the weft yarns.

        binder_thickness (float): The thickness of the binder yarns.

        super_ellipse_power (float): The weft yarn cross section super ellipse
                                     exponent.

        num_stacks (int): The number of times to stack the textile in the
                          out-of-plane direction.

    Keyword args:
        deform (list[float]): A list of length 4 where the entries refer to the
                              maximum possible:
                              1: percentual deformation in horizontal direction.
                              2: percentual deformation in vertical direction.
                              3: percentual change in super ellipse exponent.
                              4: rotation in degrees around spline tangent.

        internal_crimp (bool): Will add crimp to the internal weft yarns if
                               True. Otherwise only the top and bottom yarns
                               have crimp.

        nodes_per_yarn (int): The number of nodes per yarn (number of cross
                              sections).

        points_per_node (int): The number of points per yarn node
                              (points per cross section), n.b. notinterpolation
                              nodes). These points are the ones that make up the
                              triangles.

        smoothing (float): B-spline smoothing parameter.

    Returns:
        points (list[numpy array[float]]): A list where each entry corresponds
                                           to the point array of one yarn.

        triangles (list[numpy array[int]]): A list where each entry corresponds
                                            to the triangle array of one yarn.

        center_lines (list[numpy array[int]]): A list where each entry
                                               corresponds to the center line
                                               array of one yarn.
    """
    start_xs = np.linspace(
        -cell_shape[0] / 2 + cell_shape[0] / num_warp_per_layer / 2,
        cell_shape[0] / 2 - cell_shape[0] / num_warp_per_layer / 2,
        num_warp_per_layer,
    )
    start_zs = np.linspace(
        -cell_shape[2] / 2
        + binder_thickness / 2
        + weft_thickness
        + warp_thickness / 2,
        cell_shape[2] / 2
        - binder_thickness / 2
        - weft_thickness
        - warp_thickness / 2,
        num_layers,
    )

    interpolation_parameter = np.linspace(0, 1, nodes_per_yarn)

    points = []
    triangles = []
    center_lines = []

    offset = cell_shape[2] * (1 - num_stacks) / 2
    for stack in range(num_stacks):
        for z in start_zs:
            for x in start_xs:
                sample_points = generate_in_plane_sample_points(
                    cell_shape,
                    [x, z + offset],
                    num_weft_per_layer,
                    weft_width,
                    direction=1,
                    crimp=0.0,
                )
                point, center_line = generate_yarn_spline(
                    sample_points,
                    interpolation_parameter,
                    points_per_node,
                    warp_width / 2,
                    warp_thickness / 2,
                    super_ellipse_power,
                    direction=1,
                    deform=deform,
                    smoothing=smoothing,
                )
                triangle = generate_yarn_topology(
                    nodes_per_yarn, points_per_node
                )
                points.append(point)
                triangles.append(triangle)
                center_lines.append(center_line)
        offset += cell_shape[2]

    return points, triangles, center_lines


def generate_binder_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    binder_width,
    binder_thickness,
    super_ellipse_power,
    num_stacks,
    deform=[],
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):
    """Generate the binder yarn meshes.

    Args:
        cell_shape (list[float]): A list of length 3 that determines the size
                                  in the x, y, and z-directions of the textile
                                  bounding box.

        num_warp_per_layer (int): The number of warp yarns per layer.

        num_weft_per_layer (int): The number of weft yarns per layer.

        binder_width (float): The width of the binder yarns.

        binder_thickness (float): The thickness of the binder yarns.

        super_ellipse_power (float): The weft yarn cross section super ellipse
                                     exponent.

        num_stacks (int): The number of times to stack the textile in the
                          out-of-plane direction.

    Keyword args:
        deform (list[float]): A list of length 4 where the entries refer to the
                              maximum possible:
                              1: percentual deformation in horizontal direction.
                              2: percentual deformation in vertical direction.
                              3: percentual change in super ellipse exponent.
                              4: rotation in degrees around spline tangent.

        internal_crimp (bool): Will add crimp to the internal weft yarns if
                               True. Otherwise only the top and bottom yarns
                               have crimp.

        nodes_per_yarn (int): The number of nodes per yarn (number of cross
                              sections).

        points_per_node (int): The number of points per yarn node
                              (points per cross section), n.b. notinterpolation
                              nodes). These points are the ones that make up the
                              triangles.

        smoothing (float): B-spline smoothing parameter.

    Returns:
        points (list[numpy array[float]]): A list where each entry corresponds
                                           to the point array of one yarn.

        triangles (list[numpy array[int]]): A list where each entry corresponds
                                            to the triangle array of one yarn.

        center_lines (list[numpy array[int]]): A list where each entry
                                               corresponds to the center line
                                               array of one yarn.
    """
    positions = np.ones((num_warp_per_layer + 1, 2))
    positions[0::2, 1] = -1
    positions[:, 0] = np.linspace(
        -cell_shape[0] / 2,
        cell_shape[0] / 2,
        num_warp_per_layer + 1,
    )

    interpolation_parameter = np.linspace(0, 1, nodes_per_yarn)
    points = []
    triangles = []
    center_lines = []

    offset = cell_shape[2] * (1 - num_stacks) / 2
    for stack in range(num_stacks):
        for position in positions:
            sample_points = generate_out_of_plane_sample_points(
                cell_shape,
                position,
                num_weft_per_layer,
                binder_thickness,
                roundness=0.6,
                direction=1,
            )
            sample_points[:, 2] += offset
            point, center_line = generate_yarn_spline(
                sample_points,
                interpolation_parameter,
                points_per_node,
                binder_width / 2,
                binder_thickness / 2,
                super_ellipse_power,
                direction=1,
                deform=deform,
                smoothing=smoothing,
            )
            triangle = generate_yarn_topology(nodes_per_yarn, points_per_node)
            points.append(point)
            triangles.append(triangle)
            center_lines.append(center_line)
        offset += cell_shape[2]
    return points, triangles, center_lines


def generate_matrix(cell_shape):
    """Generate the matrix meshe.

    Args:
        cell_shape (list[float]): A list of length 3 that determines the size
                                  in the x, y, and z-directions of the textile
                                  bounding box.

    Keyword args:
        -

    Returns:
        points (numpy array[float]): The point array of the matrix.

        triangles (numpy array[int]): The triangle array of the matrix.
    """
    points = np.array(
        [
            [-cell_shape[0] / 2, -cell_shape[1] / 2, -cell_shape[2] / 2],
            [cell_shape[0] / 2, -cell_shape[1] / 2, -cell_shape[2] / 2],
            [cell_shape[0] / 2, cell_shape[1] / 2, -cell_shape[2] / 2],
            [-cell_shape[0] / 2, cell_shape[1] / 2, -cell_shape[2] / 2],
            [-cell_shape[0] / 2, -cell_shape[1] / 2, cell_shape[2] / 2],
            [cell_shape[0] / 2, -cell_shape[1] / 2, cell_shape[2] / 2],
            [cell_shape[0] / 2, cell_shape[1] / 2, cell_shape[2] / 2],
            [-cell_shape[0] / 2, cell_shape[1] / 2, cell_shape[2] / 2],
        ]
    )
    triangles = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 5],
            [2, 6, 5],
            [2, 3, 6],
            [3, 7, 6],
            [0, 4, 3],
            [4, 7, 3],
            [4, 5, 6],
            [4, 6, 7],
        ]
    )

    return points, triangles


def aggregate_yarns(points, triangles):
    """Convert a list of yarn meshes to one singular mesh

    Args:
        points (list[numpy array[float]]): A list where each entry corresponds
                                           to the point array of one yarn.

        triangles (list[numpy array[int]]): A list where each entry corresponds
                                            to the triangle array of one yarn.

    Keyword args:
        -

    Returns:
        points (numpy array[float]): The point array of the yarns.

        triangles (numpy array[int]): The triangle array of the yarns.
    """
    total_num_points = 0
    shifted_triangles = []
    for point, triangle in zip(points, triangles):
        triangle += total_num_points
        shifted_triangles.append(triangle)
        total_num_points += point.shape[0]

    aggregate_points = np.concatenate(points, axis=0)
    aggregate_triangles = np.concatenate(shifted_triangles, axis=0)

    return aggregate_points, aggregate_triangles


def create_orthogonal_sample(
    unit_cell_weft_length,
    unit_cell_warp_length,
    unit_cell_thickness,
    weft_yarns_per_layer,
    warp_yarns_per_layer,
    number_of_yarn_layers,
    weft_width_to_spacing_ratio,
    weft_super_ellipse_power,
    warp_width_to_spacing_ratio,
    warp_super_ellipse_power,
    weft_to_warp_ratio,
    binder_width_to_spacing_ratio,
    binder_thickness_to_spacing_ratio,
    binder_super_ellipse_power,
    compaction,
    tiling,
    deform,
    internal_crimp,
    weft_path,
    warp_path,
    binder_path,
    matrix_path,
    textile_resolution,
):
    """Generate weft, warp, binder, and matrix meshes for a orthogonal textile
       sample with or without domain randomization

    Args:
        unit_cell_weft_length (float): The size of the unit cell in the
                                       weft-direction.

        unit_cell_warp_length (float): The size of the unit cell in the
                                       warp-direction.

        unit_cell_thickness (float): The size of the unit cell in the
                                     out-of-plane-direction.

        weft_yarns_per_layer (int): The number of weft yarns per layer
                                    (per unit cell).

        warp_yarns_per_layer (int): The number of warp yarns per layer
                                    (per unit cell).

        number_of_yarn_layers (int): The number of layers (weft yarns have one
                                     additional layer) (per unit cell).

        weft_width_to_spacing_ratio (float): A number between 0 and 1 that
                                             determines how wide the weft yarns
                                             are in relation to the yarn
                                             spacing.

        weft_super_ellipse_power (float): The weft yarn cross section
                                          super ellipse exponent.

        warp_width_to_spacing_ratio (float): A number between 0 and 1 that
                                             determines how wide the warp yarns
                                             are in relation to the yarn
                                             spacing.

        warp_super_ellipse_power (float): The warp yarn cross section
                                          super ellipse exponent.

        weft_to_warp_ratio (float): A number between 0 and 1 that determines how
                                   thick weft yarns are in relation to the warp
                                   yarns.

        binder_width_to_spacing_ratio (float): A number between 0 and 1 that
                                               determines how wide the binder
                                               yarns are in relation to the yarn
                                               spacing.

        binder_thickness_to_spacing_ratio (float): A number between 0 and 1 that
                                                  determines how thick binder
                                                  yarns are in relation to the
                                                  gaps between the weft yarns.


        binder_super_ellipse_power (float): The binder yarn cross section
                                            super ellipse exponent.

        compaction (list[float]): A list of 3 numbers larger than 0 that
                                  determines how much the final textile will be
                                  compressed (or dilated for >1) in each
                                  direction. The yarn shapes are determined by
                                  the original size and their position is
                                  altered to fit in the compacted box.

        tiling (list[int]): A list of integers that determine the repeats of the
                            defined unit cell. The entries determines the number
                            of repeats in the x-, y-, and z-directions
                            respectively.


        deform (list[float]): A list of length 12 that contains deformation
                              parameters. Deformation will be applied randomly
                              at every 20th node. This is by scaling the
                              respective quantities. The parameters refer to
                              the maximum allowed alteration and are:
                              1: weft horizontal half axis scaling (%)
                              2: weft vertical half axis scaling (%)
                              3: weft super ellipse exponent (%)
                              4: weft crossection rotation (degrees)
                              5: warp horizontal half axis scaling (%)
                              6: warp vertical half axis scaling (%)
                              7: warp super ellipse exponent (%)
                              8: warp crossection rotation (degrees)
                              9: binder horizontal half axis scaling (%)
                              10: binder vertical half axis scaling (%)
                              11: binder super ellipse exponent (%)
                              12: binder crossection rotation (degrees)


        internal_crimp (bool): Will add crimp to the internal weft yarns if
                               True. Otherwise only the top and bottom yarns
                               have crimp.

        weft_path (str): The absolute path (including file name) to the weft
                         mesh.

        warp_path (str): The absolute path (including file name) to the warp
                         mesh.

        binder_path (str): The absolute path (including file name) to the binder
                         mesh.

        matrix_path (str): The absolute path (including file name) to the matrix
                         mesh.

        textile_resolution (int): The number of points per cross sections. The
                                  number of cross sections is computed to
                                  create a reasonable traingle aspect ratio.

    Keyword args:
        -

    Returns:
        None
    """
    num_weft_per_layer = weft_yarns_per_layer * tiling[1]
    num_warp_per_layer = warp_yarns_per_layer * tiling[0]

    cell_shape = [
        unit_cell_weft_length * tiling[0],
        unit_cell_warp_length * tiling[1],
        unit_cell_thickness,
    ]

    weft_spacing = cell_shape[1] / num_weft_per_layer
    weft_width = weft_width_to_spacing_ratio * weft_spacing
    warp_spacing = cell_shape[0] / num_warp_per_layer
    warp_width = warp_width_to_spacing_ratio * warp_spacing
    binder_spacing = (
        cell_shape[0] - num_warp_per_layer * warp_width
    ) / num_warp_per_layer
    binder_width = binder_spacing * binder_width_to_spacing_ratio
    binder_thickness = (
        (cell_shape[1] - num_weft_per_layer * weft_width)
        / num_weft_per_layer
        * binder_thickness_to_spacing_ratio
    )
    weft_thickness = (
        (unit_cell_thickness - 2 * binder_thickness)
        / (number_of_yarn_layers + 1)
        * weft_to_warp_ratio
    )
    warp_thickness = (
        (unit_cell_thickness - 2 * binder_thickness)
        / (number_of_yarn_layers)
        * (1 - weft_to_warp_ratio)
    )

    for idx in range(3):
        cell_shape[idx] *= compaction[idx]

    points, triangles, center_lines = generate_weft_yarns(
        cell_shape,
        num_warp_per_layer,
        num_weft_per_layer,
        number_of_yarn_layers,
        warp_width,
        weft_width,
        weft_thickness,
        binder_thickness,
        weft_super_ellipse_power,
        tiling[2],
        internal_crimp=internal_crimp,
        deform=deform[0:4],
        nodes_per_yarn=round(
            textile_resolution
            * cell_shape[1]
            / (2 * weft_thickness + 2 * weft_width)
        ),
        points_per_node=textile_resolution,
    )
    aggregated_points, aggregated_triangles = aggregate_yarns(points, triangles)
    weft_mesh = trimesh.Trimesh(
        vertices=aggregated_points, faces=aggregated_triangles
    )
    with open(weft_path.replace(".stl", ".pkl"), "wb") as file:
        pk.dump(center_lines, file)

    points, triangles, center_lines = generate_warp_yarns(
        cell_shape,
        num_warp_per_layer,
        num_weft_per_layer,
        number_of_yarn_layers,
        warp_width,
        warp_thickness,
        weft_width,
        weft_thickness,
        binder_thickness,
        warp_super_ellipse_power,
        tiling[2],
        deform=deform[4:8],
        nodes_per_yarn=round(
            textile_resolution
            * cell_shape[0]
            / (2 * warp_thickness + 2 * warp_width)
        ),
        points_per_node=textile_resolution,
    )
    aggregated_points, aggregated_triangles = aggregate_yarns(points, triangles)
    warp_mesh = trimesh.Trimesh(
        vertices=aggregated_points, faces=aggregated_triangles
    )
    with open(warp_path.replace(".stl", ".pkl"), "wb") as file:
        pk.dump(center_lines, file)

    points, triangles, center_lines = generate_binder_yarns(
        cell_shape,
        num_warp_per_layer,
        num_weft_per_layer,
        binder_width,
        binder_thickness,
        binder_super_ellipse_power,
        tiling[2],
        deform=deform[8:12],
        nodes_per_yarn=round(
            textile_resolution
            * cell_shape[0]
            / (2 * binder_thickness + 2 * binder_width)
        ),
        points_per_node=textile_resolution,
    )
    aggregated_points, aggregated_triangles = aggregate_yarns(points, triangles)
    binder_mesh = trimesh.Trimesh(
        vertices=aggregated_points, faces=aggregated_triangles
    )
    with open(binder_path.replace(".stl", ".pkl"), "wb") as file:
        pk.dump(center_lines, file)

    cell_shape[2] *= tiling[2]
    points, triangles = generate_matrix(cell_shape)
    matrix_mesh = trimesh.Trimesh(vertices=points, faces=triangles)

    # Cut yarns into cell shape
    intersected_weft_mesh = trimesh.boolean.intersection(
        (weft_mesh, matrix_mesh)
    )
    intersected_warp_mesh = trimesh.boolean.intersection(
        (warp_mesh, matrix_mesh)
    )
    intersected_binder_mesh = trimesh.boolean.intersection(
        (binder_mesh, matrix_mesh)
    )

    # Cut warp and binder out of weft
    cut_weft = trimesh.boolean.difference(
        (intersected_weft_mesh, intersected_warp_mesh)
    )
    cut_weft = trimesh.boolean.difference((cut_weft, intersected_binder_mesh))

    # Cut binder out of warp
    cut_warp = trimesh.boolean.difference(
        (intersected_warp_mesh, intersected_binder_mesh)
    )

    cut_weft.export(weft_path)
    cut_warp.export(warp_path)
    intersected_binder_mesh.export(binder_path)
    matrix_mesh.export(matrix_path)

    return None
