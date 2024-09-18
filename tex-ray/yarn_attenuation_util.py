import numpy as np


def compute_matrix_atomic_mass_fractions(
    atomic_weights,
    matrix_compounds,
    matrix_compounds_mixing_ratios,
):
    """DOCSTRING

    Args:
        -

    Keyword args:
        -

    Returns:
        -
    """

    matrix_compounds_total_masses = np.sum(
        matrix_compounds * atomic_weights, axis=1
    )

    matrix_compounds_mass_fractions = matrix_compounds_mixing_ratios / np.sum(
        matrix_compounds_mixing_ratios
    )

    matrix_atomic_mass_fractions = np.sum(
        matrix_compounds
        * atomic_weights
        * matrix_compounds_mass_fractions[:, np.newaxis]
        / matrix_compounds_total_masses[:, np.newaxis],
        axis=0,
    )

    return matrix_atomic_mass_fractions


def compute_fiber_atomic_mass_fractions(
    atomic_weights,
    fiber_compound,
):
    """DOCSTRING

    Args:
        -

    Keyword args:
        -

    Returns:
        -
    """
    fiber_atomic_mass_fractions = (
        fiber_compound
        * atomic_weights
        / np.sum(fiber_compound * atomic_weights)
    )

    return fiber_atomic_mass_fractions


def compute_yarn_atomic_mass_fractions(
    fiber_atomic_mass_fractions,
    fiber_density,
    fiber_volume_fraction,
    matrix_atomic_mass_fractions,
    matrix_density,
):
    """DOCSTRING

    Args:
        -

    Keyword args:
        -

    Returns:
        -
    """
    yarn_atomic_mass_fractions = (
        matrix_atomic_mass_fractions
        * matrix_density
        * (1.0 - fiber_volume_fraction)
        + fiber_atomic_mass_fractions * fiber_volume_fraction * fiber_density
    ) / (
        fiber_volume_fraction * fiber_density
        + (1.0 - fiber_volume_fraction) * matrix_density
    )

    return yarn_atomic_mass_fractions


def compute_yarn_density(fiber_density, fiber_volume_fraction, matrix_density):
    """DOCSTRING

    Args:
        -

    Keyword args:
        -

    Returns:
        -
    """
    yarn_density = (
        fiber_volume_fraction * fiber_density
        + (1.0 - fiber_volume_fraction) * matrix_density
    )
    return yarn_density


def compute_fiber_volume_fraction(
    voxel_size, yarn_voxel_area, fiber_diameter, num_fibers_per_yarn
):
    """DOCSTRING

    Args:
        -

    Keyword args:
        -

    Returns:
        -
    """
    measured_yarn_area = voxel_size**2 * yarn_voxel_area
    total_fiber_area = num_fibers_per_yarn * (fiber_diameter / 2) ** 2 * np.pi

    return total_fiber_area / measured_yarn_area


if __name__ == "__main__":
    # [C, H, N, O], atomic units
    atomic_weights = np.array([12.011, 1.008, 14.007, 15.999])
    # number of [C, H, N, O], -
    matrix_compounds = np.array(
        [[25, 30, 2, 4], [17, 22, 2, 0], [21, 30, 2, 0]]
    )
    fiber_compound = np.array([1, 0, 0, 0])
    # Amounts of compounds in g
    matrix_compounds_mixing_ratios = np.array([100.0, 34.05, 34.05])
    # density in g/cm^3
    matrix_density = 1.14
    fiber_density = 1.78
    # µm
    voxel_size = 46.958218
    fiber_diameter = 5.2
    # num fibers per yarn
    num_fiber_weft = 12000.0
    num_fiber_warp = 24000.0
    # number of voxels covered by yarn
    weft_voxel_area = 186
    warp_voxel_area = 320

    fiber_volume_fraction_weft = compute_fiber_volume_fraction(
        voxel_size, weft_voxel_area, fiber_diameter, num_fiber_weft
    )
    fiber_volume_fraction_warp = compute_fiber_volume_fraction(
        voxel_size, warp_voxel_area, fiber_diameter, num_fiber_weft
    )
    matrix_atomic_mass_fractions = compute_matrix_atomic_mass_fractions(
        atomic_weights, matrix_compounds, matrix_compounds_mixing_ratios
    )
    fiber_atomic_mass_fractions = compute_fiber_atomic_mass_fractions(
        atomic_weights, fiber_compound
    )
    weft_atomic_mass_fractions = compute_yarn_atomic_mass_fractions(
        fiber_atomic_mass_fractions,
        fiber_density,
        fiber_volume_fraction_weft,
        matrix_atomic_mass_fractions,
        matrix_density,
    )
    weft_density = compute_yarn_density(
        fiber_density, fiber_volume_fraction_weft, matrix_density
    )
    warp_density = compute_yarn_density(
        fiber_density, fiber_volume_fraction_weft, matrix_density
    )

    print(yarn_atomic_mass_fractions)

    # [0.92979529 0.02244698 0.02246452 0.02529321]
