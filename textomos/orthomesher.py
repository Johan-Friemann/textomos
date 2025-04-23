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


def set_contact_surface(
    points, point_range, nodes_per_crossing, points_per_node, idx, surface
):
    points[
        get_contact_points(
            [
                (nodes_per_crossing - 1) * (2 * idx) + (nodes_per_crossing - 1),
                (nodes_per_crossing - 1) * (2 * idx)
                + 2 * (nodes_per_crossing - 1),
            ],
            point_range,
            points_per_node,
        )
    ] = surface


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
):

    keypoints = compute_key_points(
        cell_shape[direction], crossing_width, num_crossing
    )
    num_nodes = (
        1
        + nodes_per_crossing * num_crossing
        + nodes_per_non_crossing * (num_crossing - 1)
        + 2 * (nodes_per_non_crossing // 2)
    )
    points = np.empty((num_nodes * points_per_node, 3), dtype=float)
    # We must set end here since we dont include ends in linspaces.
    points[-points_per_node:, direction] = cell_shape[direction]

    nodes_per_segment = np.concatenate(
        (
            (0, nodes_per_non_crossing // 2),
            np.ravel(
                np.vstack(
                    (
                        np.array([nodes_per_crossing] * (num_crossing - 1)),
                        np.array([nodes_per_non_crossing] * (num_crossing - 1)),
                    )
                ),
                order="F",
            ),
            (nodes_per_crossing, nodes_per_non_crossing // 2),
        )
    )
    crossing_idx = np.cumsum(nodes_per_segment)

    for idx in range(2 * num_crossing + 1):
        points[
            (points_per_node * crossing_idx[idx]) : (
                points_per_node * crossing_idx[idx + 1]
            ),
            direction,
        ] = np.repeat(
            np.linspace(
                keypoints[idx],
                keypoints[idx + 1],
                nodes_per_segment[idx + 1],
                endpoint=False,
            ),
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

    points[:, (direction + 1) % 2] = start_coord[0] + np.power(
        np.abs(np.cos(angle_parameter)) * horizontal_half_axis,
        super_ellipse_power,
    ) * np.sign(np.cos(angle_parameter))
    points[:, 2] = start_coord[1] + np.power(
        np.abs(np.sin(angle_parameter)) * vertical_half_axis,
        super_ellipse_power,
    ) * np.sign(np.sin(angle_parameter))

    triangles = generate_yarn_topology(num_nodes, points_per_node)

    return points, triangles


# Curve params
p, t = generate_yarn(
    [5.0, 0.0],
    [30, 10, 10],
    20,
    3,
    7,
    3,
    20,
    direction=0,
    super_ellipse_power=0.5,
)

# pp = get_contact_zone([1, 3], [1, 9], 20)


# print(pp)
# t = np.delete(t,tt,axis=0)
mesh = me.Mesh(p, [("triangle", t)])
mesh.write("./textomos/foo.stl")
