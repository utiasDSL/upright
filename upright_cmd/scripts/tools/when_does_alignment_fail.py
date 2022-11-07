import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import upright_core as core

import IPython


m = 0.5
δ = 0.03
h = 0.1
L = 0.05  # fixture height
μ = 0.3

# CoM and inertia
r_com_e = np.array([0, 0, h])
J_e = core.math.cylinder_inertia_matrix(m, δ, 2*h)

# gravity
g = 9.81
G = np.array([0, 0, -g])


class ContactPoint:
    def __init__(self, r, normal, μ):
        self.r = np.array(r)
        self.normal = np.array(normal)
        self.tangent = np.cross(normal, [0, 1, 0])
        self.μ = μ


def gravito_inertial_wrench(dxdtt, θ, dθdt, dθdtt):
    C_we = core.math.roty(-θ)
    C_ew = C_we.T

    ω = np.array([0, dθdt, 0])
    dωdt = np.array([0, dθdtt, 0])
    a_ew_w = np.array([dxdtt, 0, 0])

    S1 = core.math.skew3(dωdt)
    S2 = core.math.skew3(ω)
    dC_we_dtt = (S1 + S2 @ S2) @ C_we
    J_w = C_we @ J_e @ C_ew

    if dθdt > 0:
        IPython.embed()

    f = m * C_ew @ (dC_we_dtt @ r_com_e + a_ew_w - G)
    τ = C_ew @ (S2 @ J_w @ ω + J_w @ dωdt)

    return np.array([f[0], f[2], τ[1]])


def contact_wrench(fs_xz, contacts):
    nc = len(contacts)
    f = np.zeros(3)
    τ = np.zeros(3)
    for i in range(nc):
        fi = np.array([fs_xz[2*i], 0, fs_xz[2*i+1]])
        ri = contacts[i].r
        f += fi
        τ += np.cross(ri, fi)
    return np.array([f[0], f[2], τ[1]])


def friction_cones(fs_xz, contacts):
    nc = len(contacts)
    constraints = np.zeros(3 * nc)
    for i in range(nc):
        fi = np.array([fs_xz[2*i], 0, fs_xz[2*i+1]])
        fi_n = contacts[i].normal @ fi
        fi_t = contacts[i].tangent @ fi
        μi = contacts[i].μ
        constraints[i*3:(i+1)*3] = np.array([fi_n, μi * fi_n - fi_t, μi * fi_n + fi_t])
    return constraints


def linear_acceleration_bounds(angles, dθdt, dθdtt, contacts, bound=100):
    n = angles.shape[0]
    as_min = np.zeros(n)
    as_max = np.zeros(n)

    nc = len(contacts)
    x0 = np.zeros(2 * nc + 1)
    x0[2] = m * g / 2
    x0[4] = m * g / 2

    bounds = [(None, None) for _ in range(2 * nc + 1)]
    bounds[0] = (-bound, bound)

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
            giw = gravito_inertial_wrench(a, θ, dθdt, dθdtt)
            cw = contact_wrench(x[1:], contacts)
            return giw - cw

        def ineq_con(x):
            return friction_cones(x[1:], contacts)

        res_min = minimize(
            cost_min,
            x0=x0,
            method="slsqp",
            constraints=[{"type": "eq", "fun": eq_con}, {"type": "ineq", "fun": ineq_con}],
            bounds=bounds,
        )
        as_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[{"type": "eq", "fun": eq_con}, {"type": "ineq", "fun": ineq_con}],
            bounds=bounds,
        )
        as_max[i] = res_max.x[0]

    return as_min, as_max


def angular_acceleration_bounds(angles, dθdt, contacts, bound=100):
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
            return dθdtt #+ x[1:] @ x[1:]

        def cost_max(x):
            dθdtt = x[0]
            return -dθdtt #+ x[1:] @ x[1:]

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
            constraints=[{"type": "eq", "fun": eq_con}, {"type": "ineq", "fun": ineq_con}],
        )
        if not res_min.success:
            print(f"Angular acceleration minimization for θ = {θ} and ω={dθdt} not successful.")
        dωdts_min[i] = res_min.x[0]

        res_max = minimize(
            cost_max,
            x0=x0,
            method="slsqp",
            constraints=[{"type": "eq", "fun": eq_con}, {"type": "ineq", "fun": ineq_con}],
        )
        if not res_max.success:
            print(f"Angular acceleration maximization for θ = {θ} and ω={dθdt} not successful.")
        dωdts_max[i] = res_max.x[0]

    return dωdts_min, dωdts_max, a_aligned


n = 20
dθdt = 0
C1 = ContactPoint(r=[-δ, 0, -h], normal=[0, 0, 1], μ=μ)
C2 = ContactPoint(r=[δ, 0, -h], normal=[0, 0, 1], μ=μ)

angles = np.linspace(-np.pi/3, np.pi/3, n)
dωdts_min0, dωdts_max0, _ = angular_acceleration_bounds(angles, dθdt=0, contacts=[C1, C2])
dωdts_min5, dωdts_max5, _ = angular_acceleration_bounds(angles, dθdt=5, contacts=[C1, C2])
dωdts_min10, dωdts_max10, _ = angular_acceleration_bounds(angles, dθdt=10, contacts=[C1, C2])

plt.figure()
plt.fill_between(angles, dωdts_min0, dωdts_max0, label="ω=0", color=(1, 0, 0, 0.5))
plt.fill_between(angles, dωdts_min5, dωdts_max5, label="ω=5", color=(0, 1, 0, 0.5))
plt.fill_between(angles, dωdts_min10, dωdts_max10, label="ω=10", color=(0, 0, 1, 0.5))
# plt.plot(angles, dωdts_min, label="min", color="r")
# plt.plot(angles, dωdts_max, label="max", color="r")
plt.legend()
plt.grid()
plt.title("Angular acceleration vs. angle.")


# C1 is expanded by the fixture
normal = np.array([1, 0, 1])
normal = normal / np.linalg.norm(normal)
C1_big = ContactPoint(r=[-δ, 0, -h], normal=normal, μ=np.tan(np.arctan(μ) + np.pi/4))

# C3 is added by the fixture
C3 = ContactPoint(r=[-δ, 0, L-h], normal=[1, 0, 0], μ=μ)

dθdtt = 0
angles = np.linspace(-np.pi/4, np.pi/4, n)
_, _, as_aligned = angular_acceleration_bounds(angles, dθdt, [C1, C2])
as_no_fix_min, as_no_fix_max = linear_acceleration_bounds(angles, dθdt, dθdtt, [C1, C2])
as_fix_min, as_fix_max = linear_acceleration_bounds(angles, dθdt, dθdtt, [C1_big, C2, C3])

plt.figure()
plt.plot(angles, as_aligned, label="aligned")
plt.plot(angles, as_no_fix_min, label="no_fix", color="r")
plt.plot(angles, as_no_fix_max, label="no_fix", color="r")
plt.plot(angles, as_fix_min, label="fix", color="g")
plt.plot(angles, as_fix_max, label="fix", color="g")
plt.legend()
plt.grid()
plt.title("Linear acceleration vs. angle")

plt.show()

# IPython.embed()
