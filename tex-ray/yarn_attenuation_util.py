import numpy as np


def compute_weight_fraction(
    atomic_weights,
    matrix_compounds,
    matrix_compounds_mixing_ratios,
    matrix_density,
    fiber_compound,
    fiber_volume_fraction,
    fiber_density,
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

    matrix_atomic_weight_ratios = np.sum(
        matrix_compounds
        * atomic_weights
        * matrix_compounds_mass_fractions[:, np.newaxis]
        / matrix_compounds_total_masses[:, np.newaxis],
        axis=0,
    )

    fiber_compound_total_mass = np.sum(fiber_compound * atomic_weights)

    yarn_atomic_weight_ratios = (
        np.sum(
            matrix_compounds
            * atomic_weights
            * matrix_compounds_mass_fractions[:, np.newaxis]
            * matrix_density
            * (1.0 - fiber_volume_fraction)
            / matrix_compounds_total_masses[:, np.newaxis],
            axis=0,
        )
        + fiber_compound
        * atomic_weights
        * fiber_density
        * fiber_volume_fraction
        / fiber_compound_total_mass
    ) / (
        fiber_volume_fraction * fiber_density
        + (1.0 - fiber_volume_fraction) * matrix_density
    )

    return yarn_atomic_weight_ratios, matrix_atomic_weight_ratios


# [C, H, N, O], atomic units
atomic_weights = [12.011, 1.008, 14.007, 15.999]

# number of [C, H, N, O], -
compound_A = [25, 30, 2, 4]
compound_B1 = [17, 22, 2, 0]
compound_B2 = [21, 30, 2, 0]
compound_C = [1, 0, 0, 0]

# density, g/cm^3
density_AB = 1.14
density_C = 1.78

# fractions, -
mass_fraction_A_in_AB = 100 / (100 + 68.1)
mass_fraction_B_in_AB = 68.1 / (100 + 68.1)

# Assume for now
mass_fraction_B1_in_B = 0.5
mass_fraction_B2_in_B = 0.5

#############################
# Compute vol fraction fiber in yarn
voxel_size = 46.958218  # µm
a_weft = 7 / 2  # Num pixels
b_weft = 32 / 2  # Num pixels

a_warp = 7 / 2
b_warp = 45 / 2

weft_area = 186 * voxel_size**2  # From imageJ
warp_area = 320 * voxel_size**2  # From imageJ
fiber_radius = 5.2 / 2  # µm
fiber_area = fiber_radius * fiber_radius * np.pi
num_fiber_weft = 12000.0
num_fiber_warp = 24000.0

volume_fraction_C_in_ABC = (num_fiber_weft * fiber_area) / weft_area

# volume_fraction_C_in_ABC = (num_fiber_warp * fiber_area) / warp_area
print(volume_fraction_C_in_ABC)
volume_fraction_AB_in_ABC = 1.0 - volume_fraction_C_in_ABC
#############################

# Atomic units
total_mass_A = sum([atomic_weights[i] * compound_A[i] for i in range(4)])
total_mass_B1 = sum([atomic_weights[i] * compound_B1[i] for i in range(4)])
total_mass_B2 = sum([atomic_weights[i] * compound_B2[i] for i in range(4)])

ratios_matrix = [0, 0, 0, 0]
ratios_yarn = [0, 0, 0, 0]
for i in range(4):
    ratios_matrix[i] = (
        compound_A[i] * atomic_weights[i] * mass_fraction_A_in_AB / total_mass_A
        + compound_B1[i]
        * atomic_weights[i]
        * mass_fraction_B_in_AB
        * mass_fraction_B1_in_B
        / total_mass_B1
        + compound_B2[i]
        * atomic_weights[i]
        * mass_fraction_B_in_AB
        * mass_fraction_B2_in_B
        / total_mass_B2
    )

for i in range(4):
    ratios_yarn[i] = (
        compound_A[i]
        * atomic_weights[i]
        * mass_fraction_A_in_AB
        * density_AB
        * volume_fraction_AB_in_ABC
        / total_mass_A
        + compound_B1[i]
        * atomic_weights[i]
        * mass_fraction_B_in_AB
        * mass_fraction_B1_in_B
        * density_AB
        * volume_fraction_AB_in_ABC
        / total_mass_B1
        + compound_B2[i]
        * atomic_weights[i]
        * mass_fraction_B_in_AB
        * mass_fraction_B2_in_B
        * density_AB
        * volume_fraction_AB_in_ABC
        / total_mass_B2
        + compound_C[i] * density_C * volume_fraction_C_in_ABC
    ) / (
        density_AB * volume_fraction_AB_in_ABC
        + density_C * volume_fraction_C_in_ABC
    )

density_yarn = (
    density_C * volume_fraction_C_in_ABC
    + density_AB * volume_fraction_AB_in_ABC
)

print("Ratios matrix old:")
print(ratios_yarn)

print("Ratios matrix new:")
print(
    compute_weight_fraction(
        np.array(atomic_weights),
        np.array([compound_A, compound_B1, compound_B2]),
        np.array([100, 34.05, 34.05]),
        1.14,
        np.array([1,0,0,0]),
        volume_fraction_C_in_ABC,
        1.78,
    )
)
