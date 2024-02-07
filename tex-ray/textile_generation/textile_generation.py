import pymeshlab as pml
from TexGen.Core import *


def create_layer2layer_unit_cell(
    cell_x_size,
    cell_y_size,
    cell_z_size,
    num_weft,
    num_warp,
    num_layers,
    spacing_ratio,
    weft_to_warp_ratio,
    weave_pattern,
):
    """Generate a layer to layer fabric unit cell using TexGen.

    Args:
        cell_x_size (float): The size of the unit cell in the x-direction.

        cell_y_size (float): The size of the unit cell in the y-direction.

        cell_z_size (float): The size of the unit cell in the z-direction.

        num_weft (int): The number of weft yarns per layer.

        num_warp (int): The number of warp yarns per layer.

        num_layers (int): The number of layers (weft yarns have one additional
                         layer).

        spacing ratio (float): A number between 0 and 1 that determines how wide
                              the yarns are in relation to the yarn spacing.

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

    Keyword args:
        -

    Returns:
        Weft (CTextile): A TexGen object that describes the weft yarns.

        Warp (CTextile): A TexGen object that describes the warp yarns.
    """
    x_yarn_spacing = cell_y_size / num_weft
    y_yarn_spacing = cell_x_size / num_warp
    x_yarn_width = x_yarn_spacing * spacing_ratio
    y_yarn_width = y_yarn_spacing * spacing_ratio
    x_yarn_thickness = cell_z_size * weft_to_warp_ratio / (num_layers + 1)
    y_yarn_thickness = cell_z_size * (1.0 - weft_to_warp_ratio) / num_layers

    # Put dummy variable at spacing, since we overwrite it manually.
    Textile = CTextileWeave3D(num_warp, num_weft, 1.0, cell_z_size)

    Textile.SetXYarnSpacings(x_yarn_spacing)
    Textile.SetYYarnSpacings(y_yarn_spacing)
    Textile.SetXYarnHeights(x_yarn_thickness)
    Textile.SetYYarnHeights(y_yarn_thickness)
    Textile.SetXYarnWidths(x_yarn_width)
    Textile.SetYYarnWidths(y_yarn_width)

    for _ in range(num_layers):
        Textile.AddXLayers()
        Textile.AddYLayers()
    # Add one extra to create the top.
    Textile.AddXLayers()

    for push in weave_pattern:
        if push[2] == 1:
            Textile.PushUp(push[0], push[1])
        if push[2] == -1:
            Textile.PushDown(push[0], push[1])

    Textile.SetGapSize(1)

    Domain = Textile.GetDefaultDomain()
    Weft = CTextile()
    Weft.AssignDomain(Domain)
    Warp = CTextile()
    Warp.AssignDomain(Domain)

    for i in range(Textile.GetNumYarns()):
        Yarn = Textile.GetYarn(i)
        if i < num_weft * (num_layers + 1):
            Weft.AddYarn(Yarn)
        else:
            Warp.AddYarn(Yarn)

    return Weft, Warp


def write_layer_to_layer_unit_cell_mesh(
    weft, warp, weft_path, warp_path, matrix_path
):
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


def boolean_difference_post_processing(weft_path, warp_path, matrix_path):
    """Use PyMeshLab's boolean difference method to cut the warp yarns out of
       the weft yarns to resolve mesh overlap. Therafter the resulting weft and
       warp are cut out from the matrix. This function overwrites the weft and
       matrix meshes.

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
    mesh_set = pml.MeshSet()
    mesh_set.load_new_mesh(warp_path)
    mesh_set.load_new_mesh(weft_path)
    mesh_set.load_new_mesh(matrix_path)
    mesh_set.generate_boolean_difference(first_mesh=1, second_mesh=0)
    mesh_set.save_current_mesh(weft_path)
    mesh_set.generate_boolean_union(first_mesh=1, second_mesh=0)
    mesh_set.generate_boolean_difference(first_mesh=2, second_mesh=4)
    mesh_set.save_current_mesh(matrix_path)


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


def generate_unit_cell(config_dict):
    """Generate a woven composite unit cell and create a mesh for weft yarns,
    warp yarns, and matrix respectively.

    Args:
        config_dict (dictionary): A dictionary of tex_ray options.

    Keyword args:
        -

    Returns:
        -
    """
    Weft, Warp = create_layer2layer_unit_cell(
        config_dict["unit_cell_weft_length"],
        config_dict["unit_cell_warp_length"],
        config_dict["unit_cell_thickness"],
        config_dict["weft_yarns_per_layer"],
        config_dict["warp_yarns_per_layer"],
        config_dict["number_of_yarn_layers"],
        config_dict["yarn_width_to_spacing_ratio"],
        config_dict["weft_to_warp_ratio"],
        config_dict["weave_pattern"],
    )

    write_layer_to_layer_unit_cell_mesh(
        Weft,
        Warp,
        config_dict["weft_path"],
        config_dict["warp_path"],
        config_dict["matrix_path"],
    )

    boolean_difference_post_processing(
        config_dict["weft_path"],
        config_dict["warp_path"],
        config_dict["matrix_path"],
    )

    set_origin_to_barycenter(
        config_dict["weft_path"],
        config_dict["warp_path"],
        config_dict["matrix_path"],
    )
