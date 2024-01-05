from TexGen.Core import *


def create_layer2layer_unit_cell(
    num_yarns_x,
    num_yarns_y,
    num_layers,
    weave_pattern,
    cell_thickness,
    x_yarn_spacing,
    y_yarn_spacing,
    x_yarn_width,
    y_yarn_width,
    x_yarn_thickness,
    y_yarn_thickness,
):
    """
    TEMP
    """
    # Put dummy variable at spacing, since we overwrite it manually.
    Textile = CTextileWeave3D(num_yarns_x, num_yarns_y, 1.0, cell_thickness)

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
        if i < num_yarns_y * (num_layers + 1):
            Weft.AddYarn(Yarn)
        else:
            Warp.AddYarn(Yarn)

    return Weft, Warp


def write_layer_to_layer_unit_cell_mesh(
    weft, warp, weft_path, warp_path, matrix_path
):
    """
    TEMP
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


weave_pattern = [
    [1, 0, 1],
    [3, 0, -1],
    [5, 1, 1],
    [7, 1, -1],
    [3, 2, 1],
    [1, 2, -1],
    [7, 3, 1],
    [5, 3, -1],
]
Weft, Warp = create_layer2layer_unit_cell(
    8,
    4,
    6,
    weave_pattern,
    4.0,
    1.625,
    2.6875,
    1.625 * 0.9,
    2.6875 * 0.9,
    0.3,
    0.3,
)

write_layer_to_layer_unit_cell_mesh(
    Weft,
    Warp,
    "./tex-ray/weft.stl",
    "./tex-ray/warp.stl",
    "./tex-ray/matrix.stl",
)
