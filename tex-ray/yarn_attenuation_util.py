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
    """Compute the yarn density through the rule of mixtures. All given
       densitites must be the same unit.

    Args:
        fiber_density (float): The density of the fiber material.

        fiber_volume_fraction (float): The yarn fiber volume fraction.

        matrix_density (float): The density of the matrix material.

    Keyword args:
        -

    Returns:
        yarn_density (float): The density of the entire yarn.
    """
    yarn_density = (
        fiber_volume_fraction * fiber_density
        + (1.0 - fiber_volume_fraction) * matrix_density
    )
    return yarn_density


def estimate_fiber_volume_fraction(
    voxel_size, yarn_voxel_area, fiber_diameter, num_fibers_per_yarn
):
    """Estimate the fiber volume fraction inside a yarn from the number of
       voxels that the yarn cross section covers in a CT-slice. All length units
       of given arguments must be the same.

    Args:
        voxel_size (float): The voxel side length.

        yarn_voxel_area (int): The number of voxels (pixels) that make up the
                               yarn cross section.
        
        fiber_diameter (float): The diameter of an individual fiber.

        num_fibers_per_yarn (int): The number of fibers per yarn.

    Keyword args:
        -

    Returns:
        fiber_volume_fraction (float): The estimated fiber volume fraction.
    """
    measured_yarn_area = voxel_size**2 * yarn_voxel_area
    total_fiber_area = num_fibers_per_yarn * (fiber_diameter / 2) ** 2 * np.pi

    return total_fiber_area / measured_yarn_area


if __name__ == "__main__":
    atomic_weights = np.array([12.011, 1.008, 14.007, 15.999]) # atomic units
    matrix_compounds = np.array(
        [[25, 30, 2, 4], [17, 22, 2, 0], [21, 30, 2, 0]] 
    ) # number of [C, H, N, O]
    fiber_compound = np.array([1, 0, 0, 0]) # number of [C, H, N, O]
    matrix_compounds_mixing_ratios = np.array([100.0, 34.05, 34.05]) # g per
    matrix_density = 1.14 # g/cm^3
    fiber_density = 1.78 # g/cm^3
    voxel_size = 46.958218 # µm
    fiber_diameter = 5.2 # µm
    num_fiber_weft = 12000.0
    num_fiber_warp = 24000.0
    weft_voxel_area = 186
    warp_voxel_area = 320

    fiber_volume_fraction_weft = estimate_fiber_volume_fraction(
        voxel_size, weft_voxel_area, fiber_diameter, num_fiber_weft
    )
    fiber_volume_fraction_warp = estimate_fiber_volume_fraction(
        voxel_size, warp_voxel_area, fiber_diameter, num_fiber_warp
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
    warp_atomic_mass_fractions = compute_yarn_atomic_mass_fractions(
        fiber_atomic_mass_fractions,
        fiber_density,
        fiber_volume_fraction_warp,
        matrix_atomic_mass_fractions,
        matrix_density,
    )
    warp_density = compute_yarn_density(
        fiber_density, fiber_volume_fraction_warp, matrix_density
    )

    print("Weft atomic mass fractions of [C, H, N, O]:")
    print(weft_atomic_mass_fractions)
    print("Weft density in g/cm^3:")
    print(weft_density)
    print("Warp atomic mass fractions of [C, H, N, O]:")
    print(warp_atomic_mass_fractions)
    print("Warp density in g/cm^3:")
    print(warp_density)
    print("Matrix atomic mass fractions [C, H, N, O]:")
    print(matrix_atomic_mass_fractions)
