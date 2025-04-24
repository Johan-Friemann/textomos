import numpy as np
import meshio as me


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
    angle_parameter = np.tile(angle_parameter, num_nodes)

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
        angle_parameter > np.pi if flat_second_half else 1
    ) * (
        angle_parameter < np.pi if flat_first_half else 1
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
):

    nodes_per_crossing = 30
    nodes_per_non_crossing = 5
    points_per_node = 20
    super_ellipse_power = 1.2

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
):

    nodes_per_crossing = 20
    nodes_per_non_crossing = 5
    points_per_node = 20
    super_ellipse_power = 0.5

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
                flat_first_half=idx == num_layers,
                flat_second_half=idx == 0,
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


"""
p, t, c = generate_yarn(
    [5.0, 0.0],
    [30, 10, 10],
    20,
    4,
    7,
    3,
    20,
    direction=0,
    super_ellipse_power=0.5,
)
"""
ps, ts, cs = generate_warp_yarns(
    [20, 20, 5.2],
    4,
    8,
    5,
    3.0,
    0.5,
    2.0,
    0.5,
)

ap, at = aggregate_yarns(ps, ts)

ps1, ts1, cs1 = generate_weft_yarns(
    [20, 20, 5.2],
    4,
    8,
    5,
    3.0,
    0.5,
    2.0,
    0.5,
)

ap1, at1 = aggregate_yarns(ps1, ts1)


# print(pp)
# t = np.delete(t,tt,axis=0)
mesh = me.Mesh(ap, [("triangle", at)])
mesh.write("./textomos/foo.stl")
mesh = me.Mesh(ap1, [("triangle", at1)])
mesh.write("./textomos/bar.stl")
