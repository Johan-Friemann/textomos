import numpy as np


def write_periodic_constraints_LSDYNA(nodal_pairs, file):
    """DOCSTRING"""
    constraint_id = 1
    # Faces
    for pair in nodal_pairs["top_to_bottom"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["top_front_left"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["right_to_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_right"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["back_to_front"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_back_left"] + 1, dof, -1.0
                )
            )

    # Edges
    for pair in nodal_pairs["top_left_to_bottom_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["top_front_left"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["bottom_back_to_bottom_front"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_back_left"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["front_right_to_front_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_right"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["back_left_to_front_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_back_left"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["bottom_right_to_bottom_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_right"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["top_front_to_bottom_front"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["top_front_left"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["top_right_to_bottom_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 2.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["top_front_left"] + 1, dof, -1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_right"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["top_back_to_bottom_front"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 2.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_back_left"] + 1, dof, -1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["top_front_left"] + 1, dof, -1.0
                )
            )
    for pair in nodal_pairs["back_right_to_front_left"]:
        for dof in (1, 2, 3):
            file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
            file.write("{:>10}\n".format(constraint_id))
            constraint_id += 1
            file.write("{:>10}{:>10}{:>10}\n".format(pair[0] + 1, dof, 1.0))
            file.write("{:>10}{:>10}{:>10}\n".format(pair[1] + 1, dof, -1.0))
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_left"] + 1, dof, 2.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_back_left"] + 1, dof, -1.0
                )
            )
            file.write(
                "{:>10}{:>10}{:>10}\n".format(
                    nodal_pairs["bottom_front_right"] + 1, dof, -1.0
                )
            )

    # Vertices
    for dof in (1, 2, 3):
        file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
        file.write("{:>10}\n".format(constraint_id))
        constraint_id += 1
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_right"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_left"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, dof, -1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, dof, -1.0
            )
        )
    for dof in (1, 2, 3):
        file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
        file.write("{:>10}\n".format(constraint_id))
        constraint_id += 1
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_right"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_left"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, dof, -1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, dof, -1.0
            )
        )
    for dof in (1, 2, 3):
        file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
        file.write("{:>10}\n".format(constraint_id))
        constraint_id += 1
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_back_left"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_left"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, dof, -1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, dof, -1.0
            )
        )
    for dof in (1, 2, 3):
        file.write("*CONSTRAINED_LINEAR_GLOBAL\n")
        file.write("{:>10}\n".format(constraint_id))
        constraint_id += 1
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_back_right"] + 1, dof, 1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_left"] + 1, dof, 2.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, dof, -1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, dof, -1.0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, dof, -1.0
            )
        )


def write_load_constraints_LSDYNA(
    nodal_pairs, rve_shape, strain_magnitude, load_case, file
):
    """DOCSTRING"""
    file.write("*DEFINE_CURVE\n")
    file.write(  # Same for all load cases.
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
            1, 0, 1.0, 1.0, 0.0, 0.0, 0, 2
        )
    )
    file.write(
        "{:>20}{:>20}\n{:>20}{:>20}\n".format(0.0, 0.0, 1.0, strain_magnitude)
    )

    file.write("*BOUNDARY_SPC_NODE\n")
    file.write(  # We always lock this node since it is the reference node!
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
            nodal_pairs["bottom_front_left"] + 1, 0, 1, 1, 1, 0, 0, 0
        )
    )
    if load_case == "epsilon_11":
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, 0, 0, 1, 1, 0, 0, 0
            )
        )
        file.write("*BOUNDARY_PRESCRIBED_MOTION_NODE\n")
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1,
                1,
                2,
                1,
                rve_shape[0],
                0,
                1.0e6,
                0.0,
            )
        )
    if load_case == "epsilon_22":
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, 0, 1, 0, 1, 0, 0, 0
            )
        )
        file.write("*BOUNDARY_PRESCRIBED_MOTION_NODE\n")
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1,
                2,
                2,
                1,
                rve_shape[1],
                0,
                1.0e6,
                0.0,
            )
        )
    if load_case == "epsilon_33":
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, 0, 1, 1, 0, 0, 0, 0
            )
        )
        file.write("*BOUNDARY_PRESCRIBED_MOTION_NODE\n")
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1,
                3,
                2,
                1,
                rve_shape[2],
                0,
                1.0e6,
                0.0,
            )
        )
    if load_case == "epsilon_12":
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, 0, 1, 0, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, 0, 0, 1, 1, 0, 0, 0
            )
        )
        file.write("*BOUNDARY_PRESCRIBED_MOTION_NODE\n")
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1,
                2,
                2,
                1,
                rve_shape[0],
                0,
                1.0e6,
                0.0,
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1,
                1,
                2,
                1,
                rve_shape[1],
                0,
                1.0e6,
                0.0,
            )
        )
    if load_case == "epsilon_13":
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, 0, 1, 1, 0, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, 0, 0, 1, 1, 0, 0, 0
            )
        )
        file.write("*BOUNDARY_PRESCRIBED_MOTION_NODE\n")
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1,
                3,
                2,
                1,
                rve_shape[0],
                0,
                1.0e6,
                0.0,
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1,
                1,
                2,
                1,
                rve_shape[2],
                0,
                1.0e6,
                0.0,
            )
        )
    if load_case == "epsilon_23":
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_front_right"] + 1, 0, 1, 1, 1, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1, 0, 1, 1, 0, 0, 0, 0
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1, 0, 1, 0, 1, 0, 0, 0
            )
        )
        file.write("*BOUNDARY_PRESCRIBED_MOTION_NODE\n")
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["bottom_back_left"] + 1,
                3,
                2,
                1,
                rve_shape[1],
                0,
                1.0e6,
                0.0,
            )
        )
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10}{:>10}\n".format(
                nodal_pairs["top_front_left"] + 1,
                2,
                2,
                1,
                rve_shape[2],
                0,
                1.0e6,
                0.0,
            )
        )


def write_nodes_LSDYNA(nodes, file):
    """DOCSTRING"""
    file.write("*NODE\n")
    for idx, node in enumerate(nodes):
        file.write(
            "{:>8}{:>16.8g}{:>16.8g}{:>16.8g}{:>8}{:>8}\n".format(
                idx + 1, node[0], node[1], node[2], 0, 0
            )
        )


def write_elements_LSDYNA(elements, orientations, materials, file):
    file.write("*ELEMENT_SOLID_ORTHO\n")
    for idx, (element, orientation, material) in enumerate(
        zip(elements, orientations, materials)
    ):
        if orientation[1] == 0.0 and orientation[2] != 0.0:
            coordinate_vector = np.array([0.0, 1.0, 0.0])
        elif orientation[2] == 0.0 and orientation[1] != 0.0:
            coordinate_vector = np.array([0.0, 0.0, 1.0])
        else:
            coordinate_vector = np.array([1.0, 0.0, 0.0])
            orientation = np.array([0.0, 0.0, 1.0])
        file.write(
            ("{:>8}{:>8}{:>8}{:>8}{:>8}" "{:>8}{:>8}{:>8}{:>8}{:>8}\n").format(
                idx + 1,
                material,
                element[0] + 1,
                element[1] + 1,
                element[2] + 1,
                element[3] + 1,
                element[4] + 1,
                element[5] + 1,
                element[6] + 1,
                element[7] + 1,
            )
        )
        file.write(
            "{:>16.8g}{:>16.8g}{:>16.8g}\n".format(
                orientation[0],
                orientation[1],
                orientation[2],
            )
        )
        file.write(
            "{:>16.8g}{:>16.8g}{:>16.8g}\n".format(
                coordinate_vector[0],
                coordinate_vector[1],
                coordinate_vector[2],
            )
        )


def write_materials_LSDYNA(mat_props, file):
    """DOCSTRING"""
    for idx, mat_prop in enumerate(mat_props):
        file.write("*MAT_ORTHOTROPIC_ELASTIC_TITLE\n")
        file.write("material_ortho_{}\n".format(idx + 1))
        file.write(
            (
                "{:>10}{:>10.5g}{:>10.5g}{:>10.5g}"
                "{:>10.5g}{:>10.5g}{:>10.5g}{:>10.5g}\n"
            ).format(
                idx + 1,
                1.0,
                mat_prop[0],
                mat_prop[1],
                mat_prop[2],
                mat_prop[3],
                mat_prop[4],
                mat_prop[5],
            )
        )
        file.write(
            (
                "{:>10.5g}{:>10.5g}{:>10.5g}{:>10}"
                "{:>10.5g}{:>10.5g}{:>10}{:>10}\n\n\n"
            ).format(
                mat_prop[6],
                mat_prop[7],
                mat_prop[8],
                0.0,
                1.0,
                1.0,
                "",
                "",
            )
        )
        file.write("*PART\n")
        file.write("part_solid_{}\n".format(idx + 1))
        file.write(
            "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
                idx + 1, idx + 1, idx + 1, 0, 0, 0, 0, 0
            )
        )
        file.write("*SECTION_SOLID_TITLE\n")
        file.write("section_solid_{}\n".format(idx + 1))
        file.write("{:>10}{:>10}{:>10}\n".format(idx + 1, 1, 0))


def write_header_LSDYNA(file, dt):
    """DOCSTRING"""
    file.write("*KEYWORD\n")
    file.write("*CONTROL_IMPLICIT_GENERAL\n")
    file.write(
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
            1, dt, 2, 1, 2, 0, 0, 0
        )
    )
    file.write("*CONTROL_IMPLICIT_SOLVER\n")
    file.write(
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
            23, 1, 2, 0, 4, 1.0, 1, 0.00000001
        )
    )
    file.write("*CONTROL_IMPLICIT_SOLUTION\n")
    file.write(
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10.5g}{:>10}{:>10.5g}\n".format(
            1, 11, 15, 0.001, 0.01, 1e10, 0.9, 1e-10
        )
    )
    file.write("*CONTROL_TERMINATION\n")
    file.write(
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
            1.0, 0, 0.0, 0.0, 1e8, 0, "", ""
        )
    )
    file.write("*DATABASE_BINARY_D3PLOT\n")
    file.write(
        "{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}{:>10}\n".format(
            dt, 0, 0, 0, 0, "", "", ""
        )
    )


def write_footer_LSDYNA(file):
    """DOCSTRING"""
    file.write("*END\n")
