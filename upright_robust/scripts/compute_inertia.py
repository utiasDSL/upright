import numpy as np

import upright_core as core

import IPython


def cuboid_vertices(half_extents):
    x, y, z = half_extents
    return np.array(
        [
            [x, y, z],
            [x, y, -z],
            [x, -y, z],
            [x, -y, -z],
            [-x, y, z],
            [-x, y, -z],
            [-x, -y, z],
            [-x, -y, -z],
        ]
    )

def point_mass_system_inertia(masses, points):
    """Inertia matrix corresponding to a finite set of point masses."""
    H = np.zeros((3, 3))
    for m, p in zip(masses, points):
        H += m * np.outer(p, p)
    return H, np.trace(H) * np.eye(3) - H


def point_mass_system_com(masses, points):
    """Inertia matrix corresponding to a finite set of point masses."""
    return np.sum(masses[:, None] * points, axis=0) / np.sum(masses)


def main():
    np.set_printoptions(precision=10, suppress=True)

    I_slice_stack4 = core.math.cuboid_inertia_matrix(mass=0.25, side_lengths=[0.1, 0.02, 0.1])
    I_slice_tall = core.math.cuboid_inertia_matrix(mass=1.0, side_lengths=[0.1, 0.02, 0.4])

    # using `repr` gets us commas
    print("Inertia diagonals")
    print(f"stack4 = {repr(np.diag(I_slice_stack4))}")
    print(f"tall = {repr(np.diag(I_slice_tall))}")
    return

    mass = 0.5
    masses = np.concatenate((1 * np.ones(4), 1 * np.ones(4)))
    masses *= mass / np.sum(masses)
    half_extents = np.array([0.05, 0.05, 0.2])
    vertices = cuboid_vertices(half_extents)
    H, I = point_mass_system_inertia(masses, vertices)
    print(np.diag(I))
    # print(c)
    # Hc = H - mass * np.outer(c, c)
    # Ic = np.trace(Hc) * np.eye(3) - Hc
    # print(Ic)
    #
    # Sc = core.math.skew3(c + [0, 0, 0.05])
    # I = Ic - mass * Sc @ Sc
    # print(I)

    # IPython.embed()

main()
