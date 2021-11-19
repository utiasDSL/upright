import numpy as np

from .util import (
    skew3,
    equilateral_triangle_inscribed_radius,
    cylinder_inertia_matrix,
)


class TrayBalanceParameters:
    """Parameters for the tray balancing task."""
    def __init__(self):
        # default parameters
        self.gravity = 9.81

        self.ee_side_length = 0.3

        self.tray_radius = 0.25
        self.tray_mass = 0.5
        self.tray_mu = 0.5
        self.tray_inscribed_radius = equilateral_triangle_inscribed_radius(
            self.ee_side_length
        )
        self.tray_com_height = 0.1  # height of center of mass from bottom of tray

        tray_total_height = 2 * self.tray_com_height
        self.tray_inertia = cylinder_inertia_matrix(
            self.tray_mass, self.tray_radius, tray_total_height
        )

        # position of tray COM relative to EE
        self.r_te_e = (0, 0, 0.067)


def inequality_constraints(C_we, ω_ew_w, a_ew_w, α_ew_w, np=np, params=None):
    """Calculate inequality constraints for a single timestep.

        Parameters:
            C_we:   3x3 array representing EE orientation as a rotation matrix
            ω_ew_w: length-3 array representing EE angular velocity
            a_ew_w: length-6 array representing EE linear acceleration
            α_ew_w: length-6 array representing EE angular acceleration

        Returns:
            Length-4 vector of constraint values. If the values are
            non-negative, then the constraints are satisfied.
    """
    if params is None:
        params = TrayBalanceParameters()

    C_ew = C_we.T
    Sω_ew_w = skew3(ω_ew_w, np=np)
    ddC_we = (skew3(α_ew_w, np=np) + Sω_ew_w @ Sω_ew_w) @ C_we

    g = np.array([0, 0, -params.gravity])

    α = params.tray_mass * C_ew @ (a_ew_w + ddC_we @ params.r_te_e - g)

    # rotational
    Iw = C_we @ params.tray_inertia @ C_we.T
    β = C_ew @ Sω_ew_w @ Iw @ ω_ew_w + params.tray_inertia @ C_ew @ α_ew_w
    S = np.array([[0, 1], [-1, 0]])

    rz = -params.tray_com_height
    r = params.tray_inscribed_radius

    γ = rz * S.T @ α[:2] - β[:2]

    # NOTE: these constraints are currently written to be >= 0, in
    # constraint to the notes which have everything <= 0.
    # NOTE the addition of a small term in the square root to ensure
    # derivative is well-defined at 0
    ε2 = 0.01

    # Friction cone with rotational component: this is always a tighter
    # bound than when the rotational component isn't considered (which
    # makes sense).
    # Splitting the absolute value into two constraints appears to be
    # better numerically for the solver

    h1 = params.tray_mu * α[2] - np.sqrt(α[0] ** 2 + α[1] ** 2 + ε2)
    h1a = h1 + β[2] / r
    h1b = h1 - β[2] / r

    h2 = α[2]  # α3 >= 0
    h3 = r ** 2 * α[2] ** 2 - γ[0] ** 2 - γ[1] ** 2

    return np.array([h1a, h1b, h2, h3])
