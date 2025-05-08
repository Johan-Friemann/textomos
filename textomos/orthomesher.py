import numpy as np
import trimesh
import scipy.interpolate as inp
from scipy.spatial.transform import Rotation as R


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
    deform=[],
    sampling_step=20,
    direction=0,
    smoothing=0.0,
    flat_top=False,
    flat_bottom=False,
):  
    if deform == []:
        deform = 4*[0.0]

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
    if flat_top:
        vertical_asymmetry[: points_per_node // 2] = 0.0
    if flat_bottom:
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

    return points


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
        offset += cell_shape[2]

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
    binder_thickness,
    super_ellipse_power,
    num_stacks,
    deform=[],
    nodes_per_yarn=100,
    points_per_node=20,
    smoothing=0.0,
):
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
                point = generate_yarn_spline(
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
        offset += cell_shape[2]

    return points, triangles


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
            point = generate_yarn_spline(
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
        offset += cell_shape[2]
    return points, triangles


def generate_matrix(cell_shape):
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

    points, triangles = generate_weft_yarns(
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

    points, triangles = generate_warp_yarns(
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

    points, triangles = generate_binder_yarns(
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
