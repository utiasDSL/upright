import numpy as np
import seaborn
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import matplotlib as mpl
import sys

import upright_core as core

import IPython


# gravity
g = 9.81
G = np.array([0, 0, -g])


LINEAR_FIGURE_PATH = "planar_example_linear.pdf"
ANGULAR_FIGURE_PATH = "planar_example_angular.pdf"


class ContactPoint:
    def __init__(self, position, normal, μ):
        self.position = np.array(position)
        self.normal = np.array(normal)
        self.tangent = np.cross(normal, [0, 1, 0])
        self.μ = μ


class ObjectParams:
    def __init__(self, m, h, L, δ, μ):
        self.m = m
        self.h = h
        self.L = L
        self.δ = δ
        self.μ = μ

    @property
    def r_com_e(self):
        return np.array([0, 0, self.h])

    def contacts(self, fixture=False):
        if fixture:
            normal = np.array([1, 0, 1])
            normal = normal / np.linalg.norm(normal)
            C1 = ContactPoint(
                position=[-self.δ, 0, -self.h],
                normal=normal,
                μ=np.tan(np.arctan(self.μ) + np.pi / 4),
            )

            C2 = ContactPoint(position=[self.δ, 0, -self.h], normal=[0, 0, 1], μ=self.μ)
            C3 = ContactPoint(
                position=[-self.δ, 0, self.L - self.h], normal=[1, 0, 0], μ=self.μ
            )
            return [C1, C2, C3]
        else:
            C1 = ContactPoint(
                position=[-self.δ, 0, -self.h], normal=[0, 0, 1], μ=self.μ
            )
            C2 = ContactPoint(position=[self.δ, 0, -self.h], normal=[0, 0, 1], μ=self.μ)
            return [C1, C2]


def gravito_inertial_wrench(dxdtt, θ, dθdt, dθdtt, params):
    C_we = core.math.roty(-θ)
    C_ew = C_we.T

    ω = np.array([0, dθdt, 0])
    dωdt = np.array([0, dθdtt, 0])
    a_ew_w = np.array([dxdtt, 0, 0])

    S1 = core.math.skew3(dωdt)
    S2 = core.math.skew3(ω)
    dC_we_dtt = (S1 + S2 @ S2) @ C_we
    J_e = core.math.cylinder_inertia_matrix(params.m, params.δ, 2 * params.h)
    J_w = C_we @ J_e @ C_ew

    f = params.m * C_ew @ (dC_we_dtt @ params.r_com_e + a_ew_w - G)
    τ = C_ew @ (S2 @ J_w @ ω + J_w @ dωdt)

    return np.array([f[0], f[2], τ[1]])


def contact_wrench(fs_xz, contacts):
    nc = len(contacts)
    f = np.zeros(3)
    τ = np.zeros(3)
    for i in range(nc):
        fi = np.array([fs_xz[2 * i], 0, fs_xz[2 * i + 1]])
        ri = contacts[i].position
        f += fi
        τ += np.cross(ri, fi)
    return np.array([f[0], f[2], τ[1]])


def friction_cones(fs_xz, contacts):
    nc = len(contacts)
    constraints = np.zeros(3 * nc)
    for i in range(nc):
        fi = np.array([fs_xz[2 * i], 0, fs_xz[2 * i + 1]])
        fi_n = contacts[i].normal @ fi
        fi_t = contacts[i].tangent @ fi
        μi = contacts[i].μ
        constraints[i * 3 : (i + 1) * 3] = np.array(
            [fi_n, μi * fi_n - fi_t, μi * fi_n + fi_t]
        )
    return constraints


def linear_acceleration_bounds_vs_delta(δs, θ, dθdt, dθdtt, fixture=False, bound=100):
    n = δs.shape[0]
    as_min = np.zeros(n)
    as_max = np.zeros(n)

    # initial guess
    nc = 3 if fixture else 2
    x0 = np.zeros(2 * nc + 1)
    x0[2] = m * g / 2
    x0[4] = m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

    for i in range(n):
        δ = δs[i]

        # contacts
        if fixture:
            contacts = build_fixtured_contact_points(δ, h, L, μ)
        else:
            contacts = build_nominal_contact_points(δ, h, μ)

        def cost_min(x):
            a = x[0]
            return a

        def cost_max(x):
            a = x[0]
            return -a

        def eq_con(x):
            a = x[0]

            # gravito inertia wrench and contact wrench must balance (Newton-Euler
            # equations)
            giw = gravito_inertial_wrench(a, θ, dθdt, dθdtt, δ)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
            bounds=bounds,
        )
        as_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
            bounds=bounds,
        )
        as_max[i] = res_max.x[0]
    return as_min, as_max


def linear_acceleration_bounds_vs_angular_acceleration(
    dθdtts, θ, dθdt, δ, fixture=False, bound=100
):
    n = dθdtts.shape[0]
    as_min = np.zeros(n)
    as_max = np.zeros(n)

    # initial guess
    nc = 3 if fixture else 2
    x0 = np.zeros(2 * nc + 1)
    x0[2] = m * g / 2
    x0[4] = m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

    # contacts
    if fixture:
        contacts = build_fixtured_contact_points(δ, h, L, μ)
    else:
        contacts = build_nominal_contact_points(δ, h, μ)

    for i in range(n):
        dθdtt = dθdtts[i]

        def cost_min(x):
            a = x[0]
            return a

        def cost_max(x):
            a = x[0]
            return -a

        def eq_con(x):
            a = x[0]

            # gravito inertia wrench and contact wrench must balance (Newton-Euler
            # equations)
            giw = gravito_inertial_wrench(a, θ, dθdt, dθdtt, δ)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
            bounds=bounds,
        )
        as_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
            bounds=bounds,
        )
        as_max[i] = res_max.x[0]
    return as_min, as_max


def angular_acceleration_bounds_vs_delta(δs, θ, dθdt, fixture=False, bound=100):
    n = δs.shape[0]
    dωdts_min = np.zeros(n)
    dωdts_max = np.zeros(n)
    a_aligned = np.zeros(n)

    # initial guess
    nc = 3 if fixture else 2
    x0 = np.zeros(2 * nc + 1)
    x0[2] = m * g / 2
    x0[4] = m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

    for i in range(n):
        dxdtt = -g * np.tan(θ)
        a_aligned[i] = dxdtt

        # contacts
        δ = δs[i]
        if fixture:
            contacts = build_fixtured_contact_points(δ, h, L, μ)
        else:
            contacts = build_nominal_contact_points(δ, h, μ)

        def cost_min(x):
            dθdtt = x[0]
            return dθdtt  # + x[1:] @ x[1:]

        def cost_max(x):
            dθdtt = x[0]
            return -dθdtt  # + x[1:] @ x[1:]

        def eq_con(x):
            dθdtt = x[0]

            # gravito inertia wrench and contact wrench must balance (Newton-Euler
            # equations)
            giw = gravito_inertial_wrench(dxdtt, θ, dθdt, dθdtt, δ)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
        )
        if not res_min.success:
            print(
                f"Angular acceleration minimization for θ = {θ} and ω={dθdt} not successful."
            )
        dωdts_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
        )
        if not res_max.success:
            print(
                f"Angular acceleration maximization for θ = {θ} and ω={dθdt} not successful."
            )
        dωdts_max[i] = res_max.x[0]

    return dωdts_min, dωdts_max, a_aligned


def angular_acceleration_bounds_vs_angular_velocity(
    dθdts, θ, params, fixture=False, bound=100
):
    n = dθdts.shape[0]

    # initial guess
    nc = 3 if fixture else 2
    x0 = np.zeros(2 * nc + 1)
    x0[2] = params.m * g / 2
    x0[4] = params.m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

    contacts = params.contacts(fixture)
    dxdtt = -g * np.tan(θ)

    valid_dθdts = []
    dωdts_min = []
    dωdts_max = []

    for i in range(n):
        dθdt = dθdts[i]

        def cost_min(x):
            dθdtt = x[0]
            return dθdtt  # + x[1:] @ x[1:]

        def cost_max(x):
            dθdtt = x[0]
            return -dθdtt  # + x[1:] @ x[1:]

        def eq_con(x):
            dθdtt = x[0]

            # gravito inertia wrench and contact wrench must balance (Newton-Euler
            # equations)
            giw = gravito_inertial_wrench(dxdtt, θ, dθdt, dθdtt, params)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
        )
        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
        )
        if res_min.success and res_max.success:
            valid_dθdts.append(dθdt)
            dωdts_min.append(res_min.x[0])
            dωdts_max.append(res_max.x[0])

    return dωdts_min, dωdts_max, valid_dθdts


def linear_acceleration_bounds_vs_angle(
    angles, dθdt, dθdtt, params, fixture=False, bound=100
):
    n = angles.shape[0]
    as_min = np.zeros(n)
    as_max = np.zeros(n)

    nc = 3 if fixture else 2
    x0 = np.zeros(2 * nc + 1)
    x0[2] = params.m * g / 2
    x0[4] = params.m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

    contacts = params.contacts(fixture)

    for i in range(n):
        θ = angles[i]

        def cost_min(x):
            a = x[0]
            return a

        def cost_max(x):
            a = x[0]
            return -a

        def eq_con(x):
            a = x[0]

            # gravito inertia wrench and contact wrench must balance (Newton-Euler
            # equations)
            giw = gravito_inertial_wrench(a, θ, dθdt, dθdtt, params)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
            bounds=bounds,
        )
        as_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
            bounds=bounds,
        )
        as_max[i] = res_max.x[0]

    return as_min, as_max


def angular_acceleration_bounds_vs_angle(angles, dθdt, contacts, bound=100):
    n = angles.shape[0]
    dωdts_min = np.zeros(n)
    dωdts_max = np.zeros(n)
    a_aligned = np.zeros(n)

    nc = len(contacts)
    x0 = np.zeros(2 * nc + 1)
    x0[2] = m * g / 2
    x0[4] = m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

    for i in range(n):
        θ = angles[i]
        dxdtt = -g * np.tan(θ)
        a_aligned[i] = dxdtt

        def cost_min(x):
            dθdtt = x[0]
            return dθdtt  # + x[1:] @ x[1:]

        def cost_max(x):
            dθdtt = x[0]
            return -dθdtt  # + x[1:] @ x[1:]

        def eq_con(x):
            dθdtt = x[0]

            # gravito inertia wrench and contact wrench must balance (Newton-Euler
            # equations)
            giw = gravito_inertial_wrench(dxdtt, θ, dθdt, dθdtt)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
        )
        if not res_min.success:
            print(
                f"Angular acceleration minimization for θ = {θ} and ω={dθdt} not successful."
            )
        dωdts_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con},
                {"type": "ineq", "fun": ineq_con},
            ],
        )
        if not res_max.success:
            print(
                f"Angular acceleration maximization for θ = {θ} and ω={dθdt} not successful."
            )
        dωdts_max[i] = res_max.x[0]

    return dωdts_min, dωdts_max, a_aligned


def plot_linear_acceleration_vs_angle():
    face_palette = seaborn.color_palette("pastel")
    edge_palette = seaborn.color_palette("deep")

    n = 20
    dθdt = 0
    dθdtt = 0

    params5 = ObjectParams(m=0.5, h=0.1, L=0.05, δ=0.03, μ=0.3)
    params2 = ObjectParams(m=0.5, h=0.1, L=0.03, δ=0.03, μ=0.3)

    angle_lims = np.deg2rad([-22, 22])
    angles = np.linspace(angle_lims[0], angle_lims[1], n)
    as_aligned = -g * np.tan(angles)
    as_no_fix_min, as_no_fix_max = linear_acceleration_bounds_vs_angle(
        angles, dθdt, dθdtt, params5, fixture=False
    )
    as_fix5_min, as_fix5_max = linear_acceleration_bounds_vs_angle(
        angles, dθdt, dθdtt, params5, fixture=True
    )
    as_fix2_min, as_fix2_max = linear_acceleration_bounds_vs_angle(
        angles, dθdt, dθdtt, params2, fixture=True
    )

    fig = plt.figure(figsize=(3.25, 1.35))
    ax = plt.gca()

    plt.fill_between(
        angles,
        as_fix5_min,
        as_fix5_max,
        label=r"$\ell=\SI{5}{cm}$",
        fc=face_palette[0],
        ec=edge_palette[0],
    )
    plt.fill_between(
        angles,
        as_fix2_min,
        as_fix2_max,
        label=r"$\ell=\SI{3}{cm}$",
        fc=face_palette[3],
        ec=edge_palette[3],
    )
    plt.fill_between(
        angles,
        as_no_fix_min,
        as_no_fix_max,
        label=r"$\ell=\SI{0}{cm}$",
        fc=face_palette[2],
        ec=edge_palette[2],
    )
    plt.plot(angles, as_aligned, label="Aligned", color="k")

    xtick_degrees = np.array([-20, -10, 0, 10, 20])
    xtick_rads = np.deg2rad(xtick_degrees)
    ax.set_xticks(xtick_rads)
    ax.set_xticklabels(xtick_degrees)
    plt.xlabel(r"$\theta$ [deg]", labelpad=2)
    plt.xlim(angle_lims)

    # ax.set_yticks([0, 20, 40])
    plt.ylabel(r"$\ddot{r}^{ew}_x$ [\si{m/s\squared}]", labelpad=2)

    plt.legend()
    plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)
    fig.tight_layout(pad=0.1, w_pad=0.5)
    fig.savefig(LINEAR_FIGURE_PATH)


def plot_angular_acceleration_vs_angular_velocity():
    face_palette = seaborn.color_palette("pastel")
    edge_palette = seaborn.color_palette("deep")

    n = 50
    θ = 0
    dθdts = np.linspace(-10, 10, n)

    params10 = ObjectParams(m=0.5, h=0.1, L=0.05, δ=0.03, μ=0.3)
    params20 = ObjectParams(m=0.5, h=0.2, L=0.05, δ=0.03, μ=0.3)
    params30 = ObjectParams(m=0.5, h=0.3, L=0.05, δ=0.03, μ=0.3)

    (
        dωdts_min_h10,
        dωdts_max_h10,
        valid_dθdts_h10,
    ) = angular_acceleration_bounds_vs_angular_velocity(
        dθdts, θ, params10, fixture=False
    )
    (
        dωdts_min_h20,
        dωdts_max_h20,
        valid_dθdts_h20,
    ) = angular_acceleration_bounds_vs_angular_velocity(
        dθdts, θ, params20, fixture=False
    )
    (
        dωdts_min_h30,
        dωdts_max_h30,
        valid_dθdts_h30,
    ) = angular_acceleration_bounds_vs_angular_velocity(
        dθdts, θ, params30, fixture=False
    )

    fig = plt.figure(figsize=(3.25, 1.35))
    ax = plt.gca()

    plt.fill_between(
        valid_dθdts_h10,
        dωdts_min_h10,
        dωdts_max_h10,
        label=r"$h=\SI{10}{cm}$",
        fc=face_palette[0],
        ec=edge_palette[0],
    )
    plt.fill_between(
        valid_dθdts_h20,
        dωdts_min_h20,
        dωdts_max_h20,
        label=r"$h=\SI{20}{cm}$",
        fc=face_palette[3],
        ec=edge_palette[3],
    )
    plt.fill_between(
        valid_dθdts_h30,
        dωdts_min_h30,
        dωdts_max_h30,
        label=r"$h=\SI{30}{cm}$",
        fc=face_palette[2],
        ec=edge_palette[2],
    )

    ax.set_xticks([-10, -5, 0, 5, 10])
    plt.xlabel(r"$\dot{\theta}$ [rad/s]", labelpad=2)

    ax.set_yticks([-20, 0, 20])
    plt.ylabel(r"$\ddot{\theta}$ [\si{rad/s\squared}]", labelpad=-5)

    plt.legend()
    plt.grid(color=(0.75, 0.75, 0.75), alpha=0.5, linewidth=0.5)
    fig.tight_layout(pad=0.1, w_pad=0.1)
    fig.savefig(ANGULAR_FIGURE_PATH)


def main():
    mpl.use("pgf")
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "font.size": 6,
            "font.family": "serif",
            "font.serif": "Palatino",
            "font.sans-serif": "DejaVu Sans",
            "font.weight": "normal",
            "text.usetex": True,
            "legend.fontsize": 6,
            "axes.titlesize": 6,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage[utf8]{inputenc}",
                    r"\usepackage[T1]{fontenc}",
                    r"\usepackage{siunitx}",
                ]
            ),
        }
    )

    plot_linear_acceleration_vs_angle()
    plot_angular_acceleration_vs_angular_velocity()

    # angles = np.linspace(-np.pi/3, np.pi/3, n)
    # dωdts_min0, dωdts_max0, _ = angular_acceleration_bounds(angles, dθdt=0, contacts=[C1, C2])
    # dωdts_min5, dωdts_max5, _ = angular_acceleration_bounds(angles, dθdt=5, contacts=[C1, C2])
    # dωdts_min10, dωdts_max10, _ = angular_acceleration_bounds(angles, dθdt=10, contacts=[C1, C2])
    #
    # plt.figure()
    # plt.fill_between(angles, dωdts_min0, dωdts_max0, label="ω=0", color=(1, 0, 0, 0.5))
    # plt.fill_between(angles, dωdts_min5, dωdts_max5, label="ω=5", color=(0, 1, 0, 0.5))
    # plt.fill_between(angles, dωdts_min10, dωdts_max10, label="ω=10", color=(0, 0, 1, 0.5))
    # # plt.plot(angles, dωdts_min, label="min", color="r")
    # # plt.plot(angles, dωdts_max, label="max", color="r")
    # plt.legend()
    # plt.grid()
    # plt.title("Angular acceleration vs. angle.")

    # dxdtt = -g * np.tan(θ)
    # print(f"w = {gravito_inertial_wrench(dxdtt, θ, dθdt=1, dθdtt=0, δ=0.001)}")
    # sys.exit()

    # PLOT: dxdtt vs dωdt
    # dθdtts = np.linspace(-5, 5, n)
    # a_min_δ3, a_max_δ3 = linear_acceleration_bounds_vs_angular_acceleration(dθdtts, θ, dθdt, δ=0.03, fixture=False)
    # a_min_δ1, a_max_δ1 = linear_acceleration_bounds_vs_angular_acceleration(dθdtts, θ, dθdt, δ=0.001, fixture=False)
    #
    # plt.figure()
    # plt.fill_between(dθdtts, a_min_δ3, a_max_δ3, label="δ=3", color=(1, 0, 0, 0.5))
    # plt.fill_between(dθdtts, a_min_δ1, a_max_δ1, label="δ=1", color=(0, 1, 0, 0.5))
    # plt.axhline(dxdtt, color="k")
    # plt.legend()
    # plt.grid()
    # plt.title("Linear acceleration vs. angular acceleration")
    # plt.show()
    #
    # sys.exit()

    # PLOT: dωdt vs. ω

    # angles = np.linspace(-np.pi/4, np.pi/4, n)
    # _, _, as_aligned = angular_acceleration_bounds(angles, dθdt, [C1, C2])
    # as_no_fix_min, as_no_fix_max = linear_acceleration_bounds(angles, dθdt, dθdtt, [C1, C2])
    # as_fix_min, as_fix_max = linear_acceleration_bounds(angles, dθdt, dθdtt, [C1_big, C2, C3])
    # δs = np.linspace(0, 0.03, n)

    # NOTE: this assumes the pendulum model
    # dωdts_min, dωdts_max, _ = angular_acceleration_bounds_vs_delta(δs, θ, dθdt, fixture=False)
    # plt.figure()
    # plt.plot(δs, dωdts_max)
    # plt.grid()
    # plt.title("Angular acceleration vs. delta.")

    # plt.show()


main()
