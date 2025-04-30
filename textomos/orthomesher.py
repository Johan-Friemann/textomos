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
    smoothing=0,
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
    bs = np.tile((0, 1, 0), (num_nodes, 1))
    ns = np.linalg.cross(bs, ts, axis=1)

    Rs = np.repeat(rs, points_per_node, axis=0)
    Ns = np.repeat(ns, points_per_node, axis=0)
    Bs = np.repeat(bs, points_per_node, axis=0)
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
        * Bs
        + np.power(
            np.abs(np.sin(tiled_angs)[:, np.newaxis]),
            super_ellipse_power,
        )
        * np.sign(np.sin(tiled_angs))[:, np.newaxis]
        * vertical_half_axis
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
    sample_points[::8, 2] += crimp
    sample_points[4::8, 2] -= crimp

    return sample_points


def generate_out_of_plane_sample_points(
    cell_shape, num_crossing, vertical_half_axis, roundness=0.8, direction=0
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
    sample_points[:, 0] = key_points
    sample_points[1::8, 2] = -roundness * (
        cell_shape[2] / 2 - vertical_half_axis
    )
    sample_points[2::8, 2] = -cell_shape[2] / 2 + vertical_half_axis
    sample_points[3::8, 2] = -roundness * (
        cell_shape[2] / 2 - vertical_half_axis
    )
    sample_points[5::8, 2] = roundness * (
        cell_shape[2] / 2 - vertical_half_axis
    )
    sample_points[6::8, 2] = cell_shape[2] / 2 - vertical_half_axis
    sample_points[7::8, 2] = roundness * (
        cell_shape[2] / 2 - vertical_half_axis
    )

    print(sample_points)
    return sample_points


################################################################################


def get_contact_points(node_range, point_range, points_per_node):
    zone_width = point_range[1] - point_range[0]
    idx = np.tile(
        np.arange(point_range[0], point_range[1]),
        node_range[1] - node_range[0] + 1,
    ) + np.repeat(
        np.arange(node_range[0], node_range[1] + 1) * points_per_node,
        zone_width,
    )
    # We treat the last point on a section differently.
    shifted_idx = idx[zone_width - 1 :: zone_width] + (
        -(points_per_node - 1) if point_range[1] == points_per_node else 1
    )
    point_idx = np.ravel(
        np.hstack(
            (
                np.reshape(idx, (-1, zone_width)),
                np.reshape(shifted_idx, (-1, 1)),
            ),
        ),
        order="C",
    )

    return point_idx


def compute_key_points(yarn_length, crossing_width, num_crossing):
    spacing = yarn_length / num_crossing

    keypoints = np.empty((num_crossing + 1) * 2, dtype=float)
    keypoints[0] = 0.0
    keypoints[-1] = yarn_length
    keypoints[1:-1:2] = (
        np.linspace(spacing / 2, yarn_length - spacing / 2, num_crossing)
        - crossing_width / 2
    )
    keypoints[2:-1:2] = (
        np.linspace(spacing / 2, yarn_length - spacing / 2, num_crossing)
        + crossing_width / 2
    )
    return keypoints


def generate_yarn(
    start_coord,
    cell_shape,
    nodes_per_crossing,
    nodes_per_non_crossing,
    num_crossing,
    crossing_width,
    points_per_node,
    direction=0,
    horizontal_half_axis=2,
    vertical_half_axis=0.2,
    super_ellipse_power=1,
    flat_first_half=False,
    flat_second_half=False,
):
    keypoints = compute_key_points(
        cell_shape[direction], crossing_width, num_crossing
    )
    num_nodes = (
        1
        + (nodes_per_crossing - 1) * num_crossing
        + (nodes_per_non_crossing - 1) * (num_crossing - 1)
        + 2 * (nodes_per_non_crossing // 2 - 1)
    )
    points = np.empty((num_nodes * points_per_node, 3), dtype=float)
    points[:, direction] = -cell_shape[direction] / 2

    nodes_per_segment = np.concatenate(
        (
            (nodes_per_non_crossing // 2 - 1,),
            np.ravel(
                np.vstack(
                    (
                        np.array([nodes_per_crossing] * (num_crossing - 1)) - 1,
                        np.array([nodes_per_non_crossing] * (num_crossing - 1))
                        - 1,
                    )
                ),
                order="F",
            ),
            (nodes_per_crossing - 1, nodes_per_non_crossing // 2 - 1),
        )
    )
    crossing_idx = np.concatenate(((0,), np.cumsum(nodes_per_segment)))
    nodes_per_segment += 1

    for idx in range(2 * num_crossing + 1):
        points[
            (points_per_node + points_per_node * crossing_idx[idx]) : (
                points_per_node + points_per_node * crossing_idx[idx + 1]
            ),
            direction,
        ] += np.repeat(
            np.linspace(
                keypoints[idx],
                keypoints[idx + 1],
                nodes_per_segment[idx],
            )[1:],
            points_per_node,
        )
    # Guarantee two side points, no matter spacing
    angle_parameter = np.concatenate(
        (
            np.linspace(
                0,
                np.pi,
                points_per_node // 2 + points_per_node % 2,
                endpoint=False,
            ),
            np.linspace(np.pi, 2 * np.pi, points_per_node // 2, endpoint=False),
        )
    )
    angle_parameter = np.tile(angle_parameter, num_nodes) * (
        -1 if direction == 1 else 1
    )

    points[:, (direction + 1) % 2] = (
        start_coord[0]
        + np.power(
            np.abs(np.cos(angle_parameter)),
            super_ellipse_power,
        )
        * np.sign(np.cos(angle_parameter))
        * horizontal_half_axis
    )
    points[:, 2] = start_coord[1] + np.power(
        np.abs(np.sin(angle_parameter)),
        super_ellipse_power,
    ) * np.sign(np.sin(angle_parameter)) * vertical_half_axis * (
        np.abs(angle_parameter) > np.pi if flat_second_half else 1
    ) * (
        np.abs(angle_parameter) < np.pi if flat_first_half else 1
    )

    crossing_ranges = np.reshape(crossing_idx[1:-1], (-1, 2))
    triangles = generate_yarn_topology(num_nodes, points_per_node)

    return points, triangles, crossing_ranges


def generate_warp_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layers,
    warp_width,
    warp_height,
    weft_width,
    weft_height,
    super_ellipse_power,
    nodes_per_crossing=20,
    nodes_per_non_crossing=5,
    points_per_node=20,
):
    start_ys = (
        np.mean(
            np.reshape(
                compute_key_points(
                    cell_shape[1], warp_width, num_warp_per_layer
                )[1:-1],
                (-1, 2),
            ),
            axis=1,
        )
        - cell_shape[1] / 2
    )
    start_zs = np.linspace(
        -cell_shape[2] / 2 + weft_height + warp_height / 2,
        cell_shape[2] / 2 - weft_height - warp_height / 2,
        num_layers,
    )

    points = []
    triangles = []
    crossing_ranges = []
    for z in start_zs:
        for y in start_ys:
            point, triangle, crossing_range = generate_yarn(
                [y, z],
                cell_shape,
                nodes_per_crossing,
                nodes_per_non_crossing,
                num_weft_per_layer,
                weft_width,
                points_per_node,
                direction=0,
                horizontal_half_axis=warp_width / 2,
                vertical_half_axis=warp_height / 2,
                super_ellipse_power=super_ellipse_power,
            )
            points.append(point)
            triangles.append(triangle)
            crossing_ranges.append(crossing_range)

    return points, triangles, crossing_ranges


def generate_weft_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layers,
    warp_width,
    warp_height,
    weft_width,
    weft_height,
    super_ellipse_power,
    nodes_per_crossing=20,
    nodes_per_non_crossing=5,
    points_per_node=20,
):
    start_xs = (
        np.mean(
            np.reshape(
                compute_key_points(
                    cell_shape[0], warp_width, num_weft_per_layer
                )[1:-1],
                (-1, 2),
            ),
            axis=1,
        )
        - cell_shape[1] / 2
    )
    start_zs = np.linspace(
        -cell_shape[2] / 2 + weft_height / 2,
        cell_shape[2] / 2 - weft_height / 2,
        num_layers + 1,
    )
    start_zs[0] += weft_height / 2
    start_zs[-1] -= weft_height / 2

    points = []
    triangles = []
    crossing_ranges = []
    for idx, z in enumerate(start_zs):
        for x in start_xs:
            point, triangle, crossing_range = generate_yarn(
                [x, z],
                cell_shape,
                nodes_per_crossing,
                nodes_per_non_crossing,
                num_warp_per_layer,
                warp_width,
                points_per_node,
                direction=1,
                horizontal_half_axis=weft_width / 2,
                vertical_half_axis=weft_height / 2
                + (weft_height / 2) * (idx == 0 or idx == num_layers),
                super_ellipse_power=(
                    1
                    if (idx == 0 or idx == num_layers)
                    else super_ellipse_power
                ),
                flat_first_half=idx == 0,
                flat_second_half=idx == num_layers,
            )
            points.append(point)
            triangles.append(triangle)
            crossing_ranges.append(crossing_range)

    return points, triangles, crossing_ranges


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


def handle_intersections(
    warp_points,
    warp_crossing_ranges,
    weft_points,
    weft_crossing_ranges,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layer,
    points_per_node,
    correction_eps=1e-3,
):
    for idx in range(num_warp_per_layer * num_layer):
        for jdx in range(num_weft_per_layer):
            weft_intersect_idx = idx % num_warp_per_layer
            weft_idx_top = jdx + num_weft_per_layer * (
                idx // num_warp_per_layer + 1
            )
            weft_idx_bottom = jdx + num_weft_per_layer * (
                idx // num_warp_per_layer
            )
            warp_contact_idx_top = get_contact_points(
                warp_crossing_ranges[idx][jdx],
                [0, points_per_node // 2],
                points_per_node,
            )
            warp_top_coords = warp_points[idx][warp_contact_idx_top]

            warp_contact_idx_bottom = get_contact_points(
                warp_crossing_ranges[idx][jdx],
                [points_per_node // 2, points_per_node],
                points_per_node,
            )
            warp_bottom_coords = warp_points[idx][warp_contact_idx_bottom]
            weft_contact_idx_top = get_contact_points(
                weft_crossing_ranges[weft_idx_top][weft_intersect_idx],
                [0, points_per_node // 2],
                points_per_node,
            )
            weft_contact_idx_bottom = get_contact_points(
                weft_crossing_ranges[weft_idx_top][weft_intersect_idx],
                [points_per_node // 2, points_per_node],
                points_per_node,
            )
            weft_top_coords = weft_points[weft_idx_top][weft_contact_idx_top]
            weft_bottom_coords = weft_points[weft_idx_bottom][
                weft_contact_idx_bottom
            ]
            eps = 1e-3
            center_of_mass_top = (
                np.mean(weft_top_coords, axis=0)
                + np.mean(warp_top_coords, axis=0)
            ) / 2
            warp_top_coords[
                np.where(warp_top_coords[:, 2] > center_of_mass_top[2])[0], 2
            ] = (center_of_mass_top[2] - correction_eps)
            weft_top_coords[
                np.where(weft_top_coords[:, 2] < center_of_mass_top[2])[0], 2
            ] = (center_of_mass_top[2] + correction_eps)

            center_of_mass_bottom = (
                np.mean(weft_bottom_coords, axis=0)
                + np.mean(warp_bottom_coords, axis=0)
            ) / 2

            warp_bottom_coords[
                np.where(warp_bottom_coords[:, 2] < center_of_mass_bottom[2])[
                    0
                ],
                2,
            ] = (
                center_of_mass_bottom[2] + correction_eps
            )
            weft_bottom_coords[
                np.where(weft_bottom_coords[:, 2] > center_of_mass_bottom[2])[
                    0
                ],
                2,
            ] = (
                center_of_mass_bottom[2] - correction_eps
            )
            warp_points[idx][warp_contact_idx_top] = warp_top_coords
            weft_points[weft_idx_top][weft_contact_idx_top] = weft_top_coords
            warp_points[idx][warp_contact_idx_bottom] = warp_bottom_coords
            weft_points[weft_idx_bottom][
                weft_contact_idx_bottom
            ] = weft_bottom_coords


def generate_binder_yarns(
    cell_shape,
    num_warp_per_layer,
    num_weft_per_layer,
    num_layers,
    warp_width,
    warp_height,
    weft_width,
    weft_height,
):
    start_ys = (
        np.mean(
            np.reshape(
                compute_key_points(
                    cell_shape[1], warp_width, num_warp_per_layer
                )[2:-2],
                (-1, 2),
            ),
            axis=1,
        )
        - cell_shape[1] / 2
    )
    binder_width = (cell_shape[1] - num_warp_per_layer * warp_width) / (
        num_warp_per_layer
    )

    # generate_yarn_topology(num_nodes, points_per_node):


def experiment():
    points_per_node = 40
    horizontal_half_axis = 1.0
    vertical_half_axis = 0.2
    super_ellipse_power = 0.5

    interpolation_parameter = np.linspace(0, 1, 200)
    num_nodes = len(interpolation_parameter)

    # X = generate_in_plane_sample_points([20, 20, 4], [0, 0], 4, 2.0)
    X = generate_out_of_plane_sample_points(
        [20, 20, 4.4], 8, vertical_half_axis
    )
    # X = np.vstack((x, y, z)).T
    points = generate_yarn_spline(
        X,
        interpolation_parameter,
        points_per_node,
        horizontal_half_axis,
        vertical_half_axis,
        super_ellipse_power,
        smoothing=0.0,
    )
    # get_points_in_aabb(points, np.array([[-10, -9], [-10, 10], [-2, -1]]))
    triangles = generate_yarn_topology(num_nodes, points_per_node)
    mesh = me.Mesh(points, [("triangle", triangles)])
    mesh.write("./textomos/baz.stl")


"""
cell_shape = [20, 20, 4.4]
weft_thickness = 0.6
warp_thickness = 0.4

ps, ts, cs = generate_warp_yarns(
    cell_shape,
    4,
    8,
    5,
    3.0,
    warp_thickness,
    2.0,
    weft_thickness,
    0.8,
    nodes_per_crossing=20,
)


ps1, ts1, cs1 = generate_weft_yarns(
    cell_shape,
    4,
    8,
    5,
    3.0,
    warp_thickness,
    2.0,
    weft_thickness,
    0.5,
    nodes_per_crossing=20,
)

handle_intersections(
    ps,
    cs,
    ps1,
    cs1,
    4,
    8,
    5,
    20,
)

ap, at = aggregate_yarns(ps, ts)
ap1, at1 = aggregate_yarns(ps1, ts1)

generate_binder_yarns(
    cell_shape,
    4,
    8,
    5,
    3.0,
    warp_thickness,
    2.0,
    weft_thickness,
)"""

experiment()

"""
mesh = me.Mesh(ap, [("triangle", at)])
mesh.write("./textomos/foo.stl")
mesh = me.Mesh(ap1, [("triangle", at1)])
mesh.write("./textomos/bar.stl")
"""
