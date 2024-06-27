import pymeshlab as pml
import numpy as np
from numpy.random import rand
from TexGen.Core import *


class TextileConfigError(Exception):
    """Exception raised when missing a required config dictionary entry."""

    pass


def check_layer2layer_config_dict(config_dict):
    """Check that a config dict pertaining a layer2layer unit cell is valid.
    If invalid an appropriate exception is raised.

     Args:
         config_dict (dictionary): A dictionary of tex_ray options.

     Keyword args:
         -

     Returns:
        layer2layer_dict (dict): A dictionary consisting of relevant layer2layer
                                 UC generation parameters.

    """
    args = []
    req_keys = (
        "mesh_paths",
        "unit_cell_weft_length",
        "unit_cell_warp_length",
        "unit_cell_thickness",
        "weft_yarns_per_layer",
        "warp_yarns_per_layer",
        "number_of_yarn_layers",
        "weft_width_to_spacing_ratio",
        "warp_width_to_spacing_ratio",
        "weft_to_warp_ratio",
    )

    req_types = (
        list,
        float,
        float,
        float,
        int,
        int,
        int,
        float,
        float,
        float,
    )

    for req_key, req_type in zip(req_keys, req_types):
        args.append(config_dict.get(req_key))
        if args[-1] is None:
            raise TextileConfigError(
                "Missing required config entry: '"
                + req_key
                + "' of type "
                + str(req_type)
                + "."
            )
        if not isinstance(args[-1], req_type):
            raise TypeError(
                "Invalid type "
                + str(type(args[-1]))
                + " for required config entry '"
                + req_key
                + "'. Should be: "
                + str(req_type)
                + "."
            )

        if req_key == "mesh_paths":
            for s in args[-1]:
                if not isinstance(s, str):
                    raise TypeError(
                        "All entries of 'mesh_paths' must be strings."
                    )
        else:
            if not args[-1] > 0:
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be > 0."
                )
            if (  # Two of the entries also have upper bounds.
                req_key == "weft_width_to_spacing_ratio"
                or req_key == "warp_width_to_spacing_ratio"
                or req_key == "weft_to_warp_ratio"
            ) and not args[-1] < 1:
                raise ValueError(
                    "The given value "
                    + str(args[-1])
                    + " of '"
                    + req_key
                    + "' is invalid. It should be < 1."
                )

    opt_keys = (
        "deform",
        "tiling",
        "shift_unit_cell",
        "textile_resolution",
        "weave_pattern",
    )
    def_vals = ([], [1, 1, 1], False, 20, [])
    opt_types = (list, list, bool, int, list)

    for opt_key, opt_type, def_val in zip(opt_keys, opt_types, def_vals):
        args.append(config_dict.get(opt_key, def_val))
        if not isinstance(args[-1], opt_type):
            raise TypeError(
                "Invalid type "
                + str(type(args[-1]))
                + " for optional config entry '"
                + opt_key
                + "'. Should be: "
                + str(opt_type)
                + "."
            )

        if opt_key == "textile_resolution":
            if args[-1] < 1:
                raise ValueError("The entry 'textile_resolution' must >= 1.")

        if opt_key == "deform":  # Special exception raising for 'deform'
            if not len(args[-1]) in [0, 12]:
                raise ValueError("The entry 'deform' must have length 0 or 12.")
            for i in range(len(args[-1])):
                if not isinstance(args[-1][i], float):
                    raise TypeError("All entries of 'deform' must be floats.")
                if args[-1][i] < 0.0:
                    raise ValueError("All entries of 'deform' must >= 0.")
                if i in [0, 1, 6, 7] and args[-1][i] > 100.0:
                    raise ValueError(
                        "All scaling entries (0,1,6, and 7) of 'deform' must "
                        + " <= 100.0."
                    )
        if opt_key == "tiling":
            if not len(args[-1]) == 3:
                raise ValueError("The entry 'tiling' must have length 3.")
            for i in range(len(args[-1])):
                if not isinstance(args[-1][i], int):
                    raise TypeError("All entries of 'tiling' must be ints.")
                if args[-1][i] < 1:
                    raise ValueError("All entries of 'deform' must >= 1.")

        # Special exception raising for weave_pattern, and for paths.
        if opt_key == "weave_pattern":
            for l in args[-1]:
                if len(l) != 3:
                    raise ValueError(
                        "Each row of 'weave_pattern' must have length 3."
                    )
                for i in l:
                    if not isinstance(i, int):
                        raise TypeError(
                            "All entries of 'weave_pattern' must be integers."
                        )
                    if l[0] < 0:
                        raise ValueError(
                            "The entries in first column of 'weave_pattern' "
                            + "must >= 0."
                        )
                    if l[1] < 0:
                        raise ValueError(
                            "The entries in second column of 'weave_pattern' "
                            + "must >= 0."
                        )
                    if l[0] >= config_dict.get("warp_yarns_per_layer"):
                        raise ValueError(
                            "The entries in first column of 'weave_pattern' "
                            + "must < the value of 'warp_yarns_per_layer."
                        )
                    if l[1] >= config_dict.get("weft_yarns_per_layer"):
                        raise ValueError(
                            "The entries in second column of 'weave_pattern' "
                            + "must < the value of 'weft_yarns_per_layer'."
                        )
                    if not l[2] in (-1, 1):
                        raise ValueError(
                            "The entries in the third column of '"
                            + "weave_pattern' must be either 1 or -1."
                        )

    return dict(zip(req_keys + opt_keys, args))


def create_layer2layer_sample(
    cell_x_size,
    cell_y_size,
    cell_z_size,
    num_weft,
    num_warp,
    num_layers,
    weft_spacing_ratio,
    warp_spacing_ratio,
    weft_to_warp_ratio,
    weave_pattern,
    tiling,
    deform,
    shift_unit_cell,
    textile_resolution,
):
    """Generate a layer to layer fabric sample using TexGen. The sample is
       generated by defining a unit cell, and then repeating that cell to make
       a fabric sample.

    Args:
        cell_x_size (float): The size of the unit cell in the x-direction.

        cell_y_size (float): The size of the unit cell in the y-direction.

        cell_z_size (float): The size of the unit cell in the z-direction.

        num_weft (int): The number of weft yarns per layer (per unit cell).

        num_warp (int): The number of warp yarns per layer (per unit cell).

        num_layers (int): The number of layers (weft yarns have one additional
                         layer) (per unit cell).

        weft_width_to_spacing_ratio (float): A number between 0 and 1 that
                                             determines how wide the weft yarns
                                             are in relation to the yarn
                                             spacing.

        warp_width_to_spacing_ratio (float): A number between 0 and 1 that
                                             determines how wide the warp yarns
                                             are in relation to the yarn
                                             spacing.

        weft_to_warp_ratio (float): A number between 0 and 1 that determines how
                                   thick weft yarns are in relation to the warp
                                   yarns.

        weave_pattern (list[list[int]]): A list containing lists of length 3.
            The list decides which crossing points to "push up" or "push down"
            in order to generate the textile weave pattern. The first element in
            the list of length 3 refers to the weft number, the second element
            refers to the warp number and the third element is 1 for "push up",
            or -1 for "push down". A "push" will move all the warp yarns at the
            selected location either one layer up or down.

        tiling (list[int]): A list of integers that determine the repeats of the
                            defined unit cell. The entries determines the number
                            of repeats in the x-, y-, and z-directions
                            respectively.


        deform (list[float]): A list of length 12 that contains deformation
                              parameters. Deformation will be applied randomly
                              at each node. This is done by multiplying the
                              parameter with a uniform random variable between
                              -1 and 1 for translation and rotation, and 0 to 1
                              for scaling. If an empty list is given no
                              deformation is applied. The parameters are:
                                    1: weft crossection x-scaling (%)
                                    2: weft crossection y-scaling (%)
                                    3: weft crossection rotation (degrees)
                                    4: weft node x displacement (length units)
                                    5: weft node y displacement (length units)
                                    6: weft node z displacement (length units)
                                    7: warp crossection x-scaling (%)
                                    8: warp crossection y-scaling (%)
                                    9: warp crossection rotation (degrees)
                                    10: warp node x displacement (length units)
                                    11: warp node y displacement (length units)
                                    12: warp node z displacement (length units)

        shift_unit_cell (bool): Will randomly shift the unit cells in the tiling
                                x- and y-directions (differently for each layer,
                                the z-direction, if applicable).

        textile_resolution (int): Sets the number of mesh nodes around a yarn
                                  cross section. Number of nodes along the yarn
                                  direction is calculated such that the nodal
                                  distance is the same as between the cross
                                  section nodes.

    Keyword args:
        -

    Returns:
        Weft (CTextile): A TexGen object that describes the weft yarns.

        Warp (CTextile): A TexGen object that describes the warp yarns.
    """
    x_yarn_spacing = cell_y_size / num_weft
    y_yarn_spacing = cell_x_size / num_warp
    x_yarn_width = x_yarn_spacing * weft_spacing_ratio
    y_yarn_width = y_yarn_spacing * warp_spacing_ratio
    x_yarn_thickness = cell_z_size * weft_to_warp_ratio / (num_layers + 1)
    y_yarn_thickness = cell_z_size * (1.0 - weft_to_warp_ratio) / num_layers

    # As the z-tiling is not interlocking we need to layer the textile.
    LayeredTextile = CTextileLayered()
    # Pre-allocate the XYZ (they are overwritten in the loop)
    offset = 0
    baseline = 0
    Min = XYZ()
    Max = XYZ()
    for i in range(tiling[2]):
        # Put dummy variable at spacing, since we overwrite it manually.
        TextileLayer = CTextileWeave3D(
            num_warp * tiling[0], num_weft * tiling[1], 1.0, cell_z_size
        )

        TextileLayer.SetResolution(textile_resolution)

        TextileLayer.SetXYarnSpacings(x_yarn_spacing)
        TextileLayer.SetYYarnSpacings(y_yarn_spacing)
        TextileLayer.SetXYarnHeights(x_yarn_thickness)
        TextileLayer.SetYYarnHeights(y_yarn_thickness)
        TextileLayer.SetXYarnWidths(x_yarn_width)
        TextileLayer.SetYYarnWidths(y_yarn_width)

        for _ in range(num_layers):
            TextileLayer.AddXLayers()
            TextileLayer.AddYLayers()
        # Add one extra to create the top.
        TextileLayer.AddXLayers()

        # We need to transform from unit cell level to the tiled structure.
        # We also need to offset the weave from layer to layer if enabled.
        tiled_weave_pattern = []
        if shift_unit_cell:
            x_shift = np.random.randint(0, high=num_warp)
            y_shift = x_shift = np.random.randint(0, high=num_weft)
        for j in range(tiling[0]):
            for k in range(tiling[1]):
                for push in weave_pattern:
                    push_tile = [
                        (push[0] + j * num_warp + x_shift)
                        % (num_warp * tiling[0]),
                        (push[1] + k * num_weft + y_shift)
                        % (num_weft * tiling[1]),
                        push[2],
                    ]
                    tiled_weave_pattern.append(push_tile)

        for push in tiled_weave_pattern:
            if push[2] == 1:
                TextileLayer.PushUp(push[0], push[1])
            if push[2] == -1:
                TextileLayer.PushDown(push[0], push[1])
        LayeredTextile.AddLayer(TextileLayer, XYZ(0, 0, offset))
        Domain = TextileLayer.GetDefaultDomain()
        Domain.GetBoxLimits(Min, Max)
        offset += Max.z - Min.z
        baseline = min(baseline, Min.z)

    DomainPlanes = CDomainPlanes(Min, XYZ(Max.x, Max.y, baseline + offset))
    Weft = CTextile()
    Weft.AssignDomain(DomainPlanes)
    Warp = CTextile()
    Warp.AssignDomain(DomainPlanes)

    for i in range(LayeredTextile.GetNumYarns()):
        Yarn = LayeredTextile.GetYarn(i)
        num_nodes = Yarn.GetNumNodes()
        # Decide if weft or warp. Modulo hack due to layering.
        if i % (
            num_weft * tiling[1] * (num_layers + 1)
            + num_warp * tiling[0] * num_layers
        ) < num_weft * tiling[1] * (num_layers + 1):
            idx = 0
        else:
            idx = 1

        if deform:
            Yarn.ConvertToInterpNodes()
            YarnSection = Yarn.GetYarnSection()
            InterpNode = YarnSection.GetSectionInterpNode()
            # In order to ensure continuity in the unit cell, first and last
            # nodes must have the same section. Thus assign outside the loop.
            first_section = CSectionRotated(
                CSectionScaled(
                    InterpNode.GetNodeSection(0),
                    # We only allow scaling down to not go outside bbox.
                    XY(
                        1.0 - rand() * deform[6 * idx] / 100.0,
                        1.0 - rand() * deform[6 * idx + 1] / 100.0,
                    ),
                ),
                np.deg2rad(deform[6 * idx + 2]) * (2 * rand() - 1),
            )
            for j in range(num_nodes):
                if j == 0 or j == num_nodes - 1:
                    modified_section = first_section
                else:
                    original_section = InterpNode.GetNodeSection(j)
                    modified_section = CSectionRotated(
                        CSectionScaled(
                            original_section,
                            # We only allow scaling down to not go outside bbox.
                            XY(
                                1.0 - rand() * deform[6 * idx] / 100.0,
                                1.0 - rand() * deform[6 * idx + 1] / 100.0,
                            ),
                        ),
                        np.deg2rad(deform[6 * idx + 2]) * (2 * rand() - 1),
                    )
                    # Only translate nodes that are not at the ends
                    Node = Yarn.GetNode(j)
                    Node.Translate(
                        XYZ(
                            (2 * rand() - 1) * deform[6 * idx + 3],
                            (2 * rand() - 1) * deform[6 * idx + 4],
                            (2 * rand() - 1) * deform[6 * idx + 5],
                        )
                    )
                InterpNode.ReplaceSection(j, modified_section)

        if idx == 0:
            Weft.AddYarn(Yarn)
        else:
            Warp.AddYarn(Yarn)

    return Weft, Warp


def write_weave_mesh(weft, warp, weft_path, warp_path, matrix_path):
    """Write meshes for the weft, warp, and matrix from TexGen objects
       representing the weft and warp yarns. The meshes are saved as stl files.

    Args:
        weft (CTextile): A TexGen Textile object that represents the weft yarns.

        warp (CTextile): A TexGen Textile object that represents the warp yarns.

        weft_path (str): The absolute path (including file name) of where to
                        write the weft mesh to.

        warp_path (str): The absolute path (including file name) of where to
                        write the warp mesh to.

        matrix_path(str): The absolute path (including file name) of where to
                        write the matrix mesh to.

    Keyword args:
        -

    Returns:
        -
    """
    weft_mesh = CMesh()
    warp_mesh = CMesh()
    matrix_mesh = CMesh()

    weft.AddSurfaceToMesh(weft_mesh, True)
    warp.AddSurfaceToMesh(warp_mesh, True)

    matrix_mesh.InsertMesh(weft.GetDomain().GetMesh())

    weft_mesh.SaveToSTL(weft_path, True)
    warp_mesh.SaveToSTL(warp_path, True)
    matrix_mesh.SaveToSTL(matrix_path, True)
    return None


def boolean_difference_post_processing(weft_path, warp_path, cut_mesh="weft"):
    """Use PyMeshLab's boolean difference method to cut one yarn type out of
       the other to resolve mesh overlap. Will overwrite the mesh that is cut.

    Args:
        weft_path (str): The absolute path (including file name) to the weft
                         mesh.

        warp_path (str): The absolute path (including file name) to the warp
                         mesh.


    Keyword args:
        cut_mesh (str): Will cut the warp out of the weft if "weft", and cut
                        weft out of the warp if "warp".

    Returns:
        -
    """
    mesh_set = pml.MeshSet()
    mesh_set.load_new_mesh(warp_path)
    mesh_set.load_new_mesh(weft_path)
    if cut_mesh == "weft":
        mesh_set.generate_boolean_difference(first_mesh=1, second_mesh=0)
        mesh_set.save_current_mesh(weft_path)
    elif cut_mesh == "warp":
        mesh_set.generate_boolean_difference(first_mesh=0, second_mesh=1)
        mesh_set.save_current_mesh(warp_path)
    return None


def set_origin_to_barycenter(weft_path, warp_path, matrix_path):
    """Use PyMeshLab's "Transform: Translate, Center, set Origin" method to set
       the  mesh's origin to its barycenter. This overwrites the original files.

    Args:
        weft_path (str): The absolute path (including file name) to the weft
                         mesh.

        warp_path (str): The absolute path (including file name) to the warp
                         mesh.

        matrix_path (str): The absolute path (including file name) to the matrix
                           mesh.


    Keyword args:
        -

    Returns:
        -
    """
    # We know that the matrix is the bounding box, so we use it to compute
    # the barycenter.
    mesh_set = pml.MeshSet()
    mesh_set.load_new_mesh(matrix_path)
    barycenter = mesh_set.get_geometric_measures()["barycenter"]
    mesh_set.compute_matrix_from_translation(
        traslmethod="Set new Origin", neworigin=barycenter
    )
    mesh_set.save_current_mesh(matrix_path)

    mesh_set = pml.MeshSet()
    mesh_set.load_new_mesh(weft_path)
    mesh_set.compute_matrix_from_translation(
        traslmethod="Set new Origin", neworigin=barycenter
    )
    mesh_set.save_current_mesh(weft_path)

    mesh_set = pml.MeshSet()
    mesh_set.load_new_mesh(warp_path)
    mesh_set.compute_matrix_from_translation(
        traslmethod="Set new Origin", neworigin=barycenter
    )
    mesh_set.save_current_mesh(warp_path)
    return None


def generate_woven_composite_sample(config_dict):
    """Generate a woven composite sample and create a mesh for weft yarns,
    warp yarns, and matrix respectively.

    Users are expected to interact with this function only and not use any
    of the internals directly.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        -
    """
    # Get type here since we use different dict checkers depending on type.
    weave_type = config_dict.get("weave_type")

    if weave_type == "layer2layer":
        weave_config_dict = check_layer2layer_config_dict(config_dict)

        Weft, Warp = create_layer2layer_sample(
            weave_config_dict["unit_cell_weft_length"],
            weave_config_dict["unit_cell_warp_length"],
            weave_config_dict["unit_cell_thickness"],
            weave_config_dict["weft_yarns_per_layer"],
            weave_config_dict["warp_yarns_per_layer"],
            weave_config_dict["number_of_yarn_layers"],
            weave_config_dict["weft_width_to_spacing_ratio"],
            weave_config_dict["warp_width_to_spacing_ratio"],
            weave_config_dict["weft_to_warp_ratio"],
            weave_config_dict["weave_pattern"],
            weave_config_dict["tiling"],
            weave_config_dict["deform"],
            weave_config_dict["shift_unit_cell"],
            weave_config_dict["textile_resolution"],
        )
    else:
        raise NotImplementedError(
            "The weave type '" + str(weave_type) + "' is not available."
        )

    write_weave_mesh(
        Weft,
        Warp,
        weave_config_dict["mesh_paths"][0],
        weave_config_dict["mesh_paths"][1],
        weave_config_dict["mesh_paths"][2],
    )

    boolean_difference_post_processing(
        weave_config_dict["mesh_paths"][0],
        weave_config_dict["mesh_paths"][1],
        cut_mesh=weave_config_dict["cut_mesh"]
    )

    set_origin_to_barycenter(
        weave_config_dict["mesh_paths"][0],
        weave_config_dict["mesh_paths"][1],
        weave_config_dict["mesh_paths"][2],
    )
    return None
