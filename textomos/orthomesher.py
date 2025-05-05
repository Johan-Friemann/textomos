import numpy as np
import meshio as me
import scipy.interpolate as inp


def generate_yarn_topology(num_nodes, points_per_node):
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
    direction=0,
    smoothing=0.0,
    flat_top=False,
    flat_bottom=False,
    flat_left=False,
    flat_right=False,
):
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

    Rs = np.repeat(rs, points_per_node, axis=0)
    Ns = np.repeat(ns, points_per_node, axis=0)
    Bs = np.repeat(bs, points_per_node, axis=0)

    vertical_asymmetry = np.ones(points_per_node)
    horizontal_asymmetry = np.ones(points_per_node)
    if flat_bottom:
        vertical_asymmetry[: points_per_node // 2] = 0.0
    if flat_top:
        vertical_asymmetry[points_per_node // 2 :] = 0.0
    if flat_left:
        horizontal_asymmetry[points_per_node // 4: 3*(points_per_node // 2)] = 0.0
    if flat_right:
        horizontal_asymmetry[0 :points_per_node // 4] = 0.0
        horizontal_asymmetry[3*(points_per_node // 4) :] = 0.0
        
    tiled_vertical_asymmetry = np.tile(vertical_asymmetry, num_nodes)
    tiled_horizontal_asymmetry = np.tile(horizontal_asymmetry, num_nodes)

    angs = -np.linspace(0, 2 * np.pi, points_per_node, endpoint=False)
    tiled_angs = np.tile(angs, num_nodes)
    points = (
        Rs
        + np.power(
            np.abs(np.cos(tiled_angs)[:, np.newaxis]),
            super_ellipse_power,
        )
        * np.sign(np.cos(tiled_angs)[:, np.newaxis])
        * horizontal_half_axis
        * tiled_horizontal_asymmetry[:, np.newaxis]
        * Bs
        + np.power(
            np.abs(np.sin(tiled_angs)[:, np.newaxis]),
            super_ellipse_power,
        )
        * np.sign(np.sin(tiled_angs))[:, np.newaxis]
        * vertical_half_axis
        * tiled_vertical_asymmetry[:, np.newaxis]
        * Ns
    )

    return points


def get_points_in_aabb(points, aabb):
    x_conditions = np.logical_and(
        points[:, 0] >= aabb[0, 0], points[:, 0] <= aabb[0, 1]
    )
    y_conditions = np.logical_and(
        points[:, 1] >= aabb[1, 0], points[:, 1] <= aabb[1, 1]
    )
    z_conditions = np.logical_and(
        points[:, 2] >= aabb[2, 0], points[:, 2] <= aabb[2, 1]
    )
    indices = np.logical_and(
        np.logical_and(x_conditions, y_conditions), z_conditions
    ).nonzero()[0]

    return indices


def generate_in_plane_sample_points(
    cell_shape,
    start_coord,
    num_crossing,
    crossing_width,
    direction=0,
    crimp=0.0,
):
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
    spacing = cell_shape[direction] / num_crossing
    key_points = np.zeros(4 * num_crossing + 1, dtype=float)
    key_points[::4] = np.linspace(
        -cell_shape[direction] / 2,
        cell_shape[direction] / 2,
        num_crossing + 1,
    )
    key_points[1::4] = np.linspace(
        -cell_shape[direction] / 2,
        cell_shape[direction] / 2 - spacing,
        num_crossing,
    )
    key_points[2::4] = np.linspace(
        -cell_shape[direction] / 2 + spacing / 2,
        cell_shape[direction] / 2 - spacing / 2,
        num_crossing,
    )
    key_points[3::4] = np.linspace(
        -cell_shape[direction] / 2 + spacing,
        cell_shape[direction] / 2,
        num_crossing,
    )

    sample_points = np.zeros((4 * num_crossing + 1, 3), dtype=float)
    sample_points[:, direction] = key_points
    sample_points[:, 1 - direction] = position[0]
    sample_points[1::8, 2] = (
        -roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[2::8, 2] = (
        -(cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[3::8, 2] = -(
        roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[5::8, 2] = (
        roundness * (cell_shape[2] / 2 - binder_thickness / 2) * position[1]
    )
    sample_points[6::8, 2] = (
        cell_shape[2] / 2 - binder_thickness / 2
    ) * position[1]
    sample_points[7::8, 2] = (
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
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):
    start_xs = np.linspace(
        -cell_shape[0] / 2 + cell_shape[0] / num_weft_per_layer / 2,
        cell_shape[0] / 2 - cell_shape[0] / num_weft_per_layer / 2,
        num_weft_per_layer,
    )
    start_zs = np.linspace(
        -cell_shape[2] / 2 + weft_thickness / 2,
        cell_shape[2] / 2 - weft_thickness / 2,
        num_layers + 1,
    )
    start_zs[0] += weft_thickness / 2
    start_zs[-1] -= weft_thickness / 2
    interpolation_parameter = np.linspace(0, 1, nodes_per_yarn)

    points = []
    triangles = []
    for idxz, z in enumerate(start_zs):
        for idxx, x in enumerate(start_xs):
            sample_points = generate_in_plane_sample_points(
                cell_shape,
                [x, z],
                num_warp_per_layer,
                warp_width,
                direction=1,
                crimp=binder_thickness
                * (1 - 2 * (idxx % 2))
                * (idxz == 0 or idxz == num_layers),
            )
            point = generate_yarn_spline(
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
                direction=1,
                smoothing=smoothing,
                flat_top=idxz == 0,
                flat_bottom=idxz == num_layers,
            )
            triangle = generate_yarn_topology(nodes_per_yarn, points_per_node)
            points.append(point)
            triangles.append(triangle)

    return points, triangles


def generate_warp_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layers,
    warp_width,
    warp_thickness,
    weft_width,
    weft_thickness,
    super_ellipse_power,
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):
    start_ys = np.linspace(
        -cell_shape[1] / 2 + cell_shape[1] / num_warp_per_layer / 2,
        cell_shape[1] / 2 - cell_shape[1] / num_warp_per_layer / 2,
        num_warp_per_layer,
    )
    start_zs = np.linspace(
        -cell_shape[2] / 2 + weft_thickness + warp_thickness / 2,
        cell_shape[2] / 2 - weft_thickness - warp_thickness / 2,
        num_layers,
    )

    interpolation_parameter = np.linspace(0, 1, nodes_per_yarn)

    points = []
    triangles = []
    for z in start_zs:
        for y in start_ys:
            sample_points = generate_in_plane_sample_points(
                cell_shape,
                [y, z],
                num_weft_per_layer,
                weft_width,
                direction=0,
                crimp=0.0,
            )
            point = generate_yarn_spline(
                sample_points,
                interpolation_parameter,
                points_per_node,
                warp_width / 2,
                warp_thickness / 2,
                super_ellipse_power,
                direction=0,
                smoothing=smoothing,
            )
            triangle = generate_yarn_topology(nodes_per_yarn, points_per_node)
            points.append(point)
            triangles.append(triangle)

    return points, triangles


def generate_binder_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    binder_width,
    binder_thickness,
    super_ellipse_power,
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):

    positions = np.ones((num_warp_per_layer + 1, 2))
    positions[1::2, 1] = -1
    positions[:, 0] = np.linspace(
        -cell_shape[0] / 2,
        cell_shape[0] / 2 ,
        num_warp_per_layer + 1,
    )

    interpolation_parameter = np.linspace(0, 1, nodes_per_yarn)
    points = []
    triangles = []

    for idx, position in enumerate(positions):
        sample_points = generate_out_of_plane_sample_points(
            cell_shape,
            position,
            num_weft_per_layer,
            binder_thickness,
            roundness=0.75,
            direction=0,
        )
        point = generate_yarn_spline(
            sample_points,
            interpolation_parameter,
            points_per_node,
            binder_width / 2,
            binder_thickness / 2,
            super_ellipse_power,
            direction=0,
            smoothing=smoothing,
            flat_left=idx==0,
            flat_right=idx==(len(positions)-1)
        )
        triangle = generate_yarn_topology(nodes_per_yarn, points_per_node)
        points.append(point)
        triangles.append(triangle)

    return points, triangles


def aggregate_yarns(points, triangles):
    total_num_points = 0
    shifted_triangles = []
    for point, triangle in zip(points, triangles):
        triangle += total_num_points
        shifted_triangles.append(triangle)
        total_num_points += point.shape[0]

    aggregate_points = np.concatenate(points, axis=0)
    aggregate_triangles = np.concatenate(shifted_triangles, axis=0)

    return aggregate_points, aggregate_triangles


################################################################################

super_ellipse_power = 0.5
cell_shape = [20, 20, 4.0]
weft_thickness = 0.5
warp_thickness = 0.4
warp_width = 3.0
weft_width = 2.0
binder_width = 2.0
binder_thickness = 0.4

points, triangles = generate_weft_yarns(
    cell_shape,
    4,
    8,
    5,
    warp_width,
    weft_width,
    weft_thickness,
    -binder_thickness / 2,
    super_ellipse_power,
)
ap1, at1 = aggregate_yarns(points, triangles)

points, triangles = generate_warp_yarns(
    cell_shape,
    4,
    8,
    5,
    warp_width,
    warp_thickness,
    weft_width,
    weft_thickness,
    super_ellipse_power,
)
ap2, at2 = aggregate_yarns(points, triangles)

points, triangles = generate_binder_yarns(
    cell_shape,
    4,
    8,
    binder_width,
    binder_thickness,
    super_ellipse_power,
)
ap3, at3 = aggregate_yarns(points, triangles)

mesh = me.Mesh(ap1, [("triangle", at1)])
mesh.write("./textomos/bar.stl")
mesh = me.Mesh(ap2, [("triangle", at2)])
mesh.write("./textomos/foo.stl")
mesh = me.Mesh(ap3, [("triangle", at3)])
mesh.write("./textomos/baz.stl")
