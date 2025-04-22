import numpy as np
import meshio as me


def generate_yarn_topology(num_nodes, points_per_node):
    tube_idx = np.arange(0, (num_nodes - 1) * points_per_node)
    tube_1 = np.stack(
        (
            tube_idx,
            (tube_idx + 1) % points_per_node
            + points_per_node * (tube_idx // points_per_node)
            + points_per_node,
            tube_idx + points_per_node,
        ),
        axis=1,
    )

    tube_2 = np.stack(
        (
            tube_idx,
            (tube_idx + 1) % points_per_node
            + points_per_node * (tube_idx // points_per_node),
            (tube_idx + 1) % points_per_node
            + points_per_node * (tube_idx // points_per_node)
            + points_per_node,
        ),
        axis=1,
    )

    # Odd number of number of points make one more "bottom heavy" triangle.
    if points_per_node % 2 != 0:
        cap_idx_1 = np.arange(0, (points_per_node - 2) // 2 + 1)
        cap_idx_2 = np.arange(0, (points_per_node - 2) // 2)
    else:
        cap_idx_1 = np.arange(0, (points_per_node - 2) // 2)
        cap_idx_2 = np.arange(0, (points_per_node - 2) // 2)

    start_cap_1 = np.stack(
        (cap_idx_1, points_per_node - 1 - cap_idx_1, cap_idx_1 + 1), axis=1
    )
    start_cap_2 = np.stack(
        (
            points_per_node - 1 - cap_idx_2,
            points_per_node - 2 - cap_idx_2,
            cap_idx_2 + 1,
        ),
        axis=1,
    )
    start_cap = np.empty((points_per_node - 2, 3), dtype=start_cap_1.dtype)
    start_cap[0::2] = start_cap_1
    start_cap[1::2] = start_cap_2

    end_cap = np.empty((points_per_node - 2, 3), dtype=start_cap_1.dtype)
    end_cap_1 = np.stack(
        (
            points_per_node * (num_nodes - 1) + cap_idx_1,
            points_per_node * (num_nodes - 1)  +  cap_idx_1 + 1,
            points_per_node * num_nodes - cap_idx_1 - 1,
        ),
        axis=1,
    )
    end_cap_2 = np.stack(
        (
            points_per_node * num_nodes - 1 - cap_idx_2,
            points_per_node * (num_nodes - 1) + cap_idx_2 + 1,
            points_per_node * num_nodes - 2 - cap_idx_2,
        ),
        axis=1,
    )
    end_cap[0::2] = end_cap_1
    end_cap[1::2] = end_cap_2

    triangles = np.empty(
        (
            tube_1.shape[0]
            + tube_2.shape[0]
            + start_cap.shape[0]
            + end_cap.shape[0],
            3,
        ),
        dtype=tube_1.dtype,
    )
    triangles[0 : -(start_cap.shape[0] + end_cap.shape[0]) : 2] = tube_1
    triangles[1 : -(start_cap.shape[0] + end_cap.shape[0]) : 2] = tube_2
    triangles[-(start_cap.shape[0] + end_cap.shape[0]) : -end_cap.shape[0]] = (
        start_cap
    )
    triangles[-end_cap.shape[0] :] = end_cap

    return triangles


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
