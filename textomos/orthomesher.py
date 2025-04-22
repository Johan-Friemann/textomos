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


def get_indices(node_range, point_range, points_per_node):
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
    triangle_idx = np.ravel(
        np.stack(
            (
                2 * idx[:-zone_width],
                2 * idx[:-zone_width] + 1,
            )
        ),
        order="F",
    )

    return point_idx, triangle_idx


def generate_warp_yarn(start_coord, cell_shape, num_nodes, points_per_node):
    param_a = 2
    param_b = 1

    angle_parameter = np.linspace(0, 2 * np.pi, points_per_node, endpoint=False)
    angle_parameter = np.tile(angle_parameter, num_nodes)

    points = np.zeros((num_nodes * points_per_node, 3))
    points[:, 0] = np.repeat(
        np.linspace(start_coord[0], start_coord[0] + cell_shape[0], num_nodes),
        points_per_node,
    )
    points[:, 1] = start_coord[1] + np.cos(angle_parameter) * param_a
    points[:, 2] = start_coord[2] + np.sin(angle_parameter) * param_b

    triangles = generate_yarn_topology(num_nodes, points_per_node)

    return points, triangles


# Curve params
p, t = generate_warp_yarn([0.0, 0.0, 0.0], [30, 10, 10], 20, 20)
mesh = me.Mesh(p, [("triangle", t)])
mesh.write("./textomos/foo.stl")
