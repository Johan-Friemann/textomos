import numpy as np
import meshio as me
from scipy.spatial import Delaunay


def generate_warp_topology(start_coord, cell_shape, num_nodes, points_per_node):
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

    point_idx = np.arange(0, (num_nodes - 1) * points_per_node)
    triangles_1 = np.stack(
        (
            point_idx,
            (point_idx + 1) % points_per_node
            + points_per_node * (point_idx // points_per_node)
            + points_per_node,
            point_idx + points_per_node,
        ),
        axis=1,
    )
    triangles_2 = np.stack(
        (
            point_idx,
            (point_idx + 1) % points_per_node
            + points_per_node * (point_idx // points_per_node),
            (point_idx + 1) % points_per_node
            + points_per_node * (point_idx // points_per_node)
            + points_per_node,
        ),
        axis=1,
    )

    start_cap = Delaunay(points[:points_per_node, 1:]).simplices
    start_cap = start_cap[:, [0, 2, 1]]  # Needed for outwards normal
    end_cap = (
        Delaunay(points[(num_nodes - 1) * points_per_node :, 1:]).simplices
        + (num_nodes - 1) * points_per_node
    )

    triangles = np.empty(
        (
            triangles_1.shape[0]
            + triangles_2.shape[0]
            + start_cap.shape[0]
            + end_cap.shape[0],
            3,
        ),
        dtype=triangles_1.dtype,
    )
    triangles[0 : -(start_cap.shape[0] + end_cap.shape[0]) : 2] = triangles_1
    triangles[1 : -(start_cap.shape[0] + end_cap.shape[0]) : 2] = triangles_2
    triangles[-(start_cap.shape[0] + end_cap.shape[0]) : -end_cap.shape[0]] = (
        start_cap
    )
    triangles[-end_cap.shape[0] :] = end_cap

    return points, triangles


# Curve params
p, t = generate_warp_topology([0.0, 0.0, 0.0], [30, 10, 10], 100, 20)
mesh = me.Mesh(p, [("triangle", t)])
mesh.write("./textomos/foo.stl")
