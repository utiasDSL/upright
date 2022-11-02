import numpy as np
import qpsolvers
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import upright_core as core

import IPython


m = 0.5
δ = 0.03
h = 0.1
μ = 0.3

# CoM and inertia
r_com_e = np.array([0, 0, h])
J_e = core.math.cylinder_inertia_matrix(m, δ, 2*h)

# contact points
r_c1_e = np.array([-δ, 0, -h])
r_c2_e = np.array([δ, 0, -h])

# gravity
g = 9.81
G = np.array([0, 0, -g])

z_e = np.array([0, 0, 1])
x_e = np.array([1, 0, 0])

dθdt = 0
ω = np.array([0, dθdt, 0])


def gravito_inertial_wrench(θ, dxdtt, dθdtt):
    C_we = core.math.roty(-θ)
    C_ew = C_we.T

    dωdt = np.array([0, dθdtt, 0])
    a_ew_w = np.array([dxdtt, 0, 0])

    S1 = core.math.skew3(dωdt)
    S2 = core.math.skew3(ω)
    dC_we_dtt = (S1 + S2 @ S2) @ C_we
    J_w = C_we @ J_e @ C_ew

    f = m * C_ew @ (dC_we_dtt @ r_com_e + a_ew_w - G)
    τ = C_ew @ (S2 @ J_w @ ω + J_w @ dωdt)

    return np.array([f[0], f[2], τ[1]])


def contact_wrench(f1_xz, f2_xz):
    f1 = np.array([f1_xz[0], 0, f1_xz[1]])
    f2 = np.array([f2_xz[0], 0, f2_xz[1]])
    f = f1 + f2
    τ = np.cross(r_c1_e, f1) + np.cross(r_c2_e, f2)
    return np.array([f[0], f[2], τ[1]])


def friction_cones(f1_xz, f2_xz):
    return np.array(
        [
            f1_xz[1],
            f2_xz[1],
            μ * f1_xz[1] - f1_xz[0],
            μ * f1_xz[1] + f1_xz[0],
            μ * f2_xz[1] - f2_xz[0],
            μ * f2_xz[1] + f2_xz[0],
        ]
    )


n = 20
angles = np.linspace(-np.pi/3, np.pi/3, n)

dωdts_min = np.zeros(n)
dωdts_max = np.zeros(n)
dxdtts_aligned = np.zeros(n)

# TODO for the next thing, I want to optimize over allowable acceleration

for i in range(n):
    θ = angles[i]
    dxdtt = -g * np.tan(θ)
    dxdtts_aligned[i] = dxdtt

    # print(f"θ = {θ}")
    # print(f"dxdtt = {dxdtt}")

    def cost_min(x):
        dθdtt = x[0]
        return dθdtt #+ x[1:] @ x[1:]

    def cost_max(x):
        dθdtt = x[0]
        return -dθdtt #+ x[1:] @ x[1:]

    def eq_con(x):
        dθdtt = x[0]
        f1_xz = x[1:3]
        f2_xz = x[3:]

        # gravito inertia wrench and contact wrench must balance (Newton-Euler
        # equations)
        giw = gravito_inertial_wrench(θ, dxdtt, dθdtt)
        cw = contact_wrench(f1_xz, f2_xz)
        return giw - cw

    def ineq_con(x):
        f1_xz = x[1:3]
        f2_xz = x[3:]
        return friction_cones(f1_xz, f2_xz)

    x0 = np.array([0, 0, 0.5 * m * g, 0, 0.5 * m * g])
    res_min = minimize(
        cost_min,
        x0=x0,
        method="slsqp",
        constraints=[{"type": "eq", "fun": eq_con}, {"type": "ineq", "fun": ineq_con}],
    )
    dωdts_min[i] = res_min.x[0]

    res_max = minimize(
        cost_max,
        x0=x0,
        method="slsqp",
        constraints=[{"type": "eq", "fun": eq_con}, {"type": "ineq", "fun": ineq_con}],
    )
    dωdts_max[i] = res_max.x[0]

plt.figure()
plt.plot(angles, dωdts_min, label="min")
plt.plot(angles, dωdts_max, label="max")
plt.legend()
plt.grid()
plt.title("Angular acceleration vs. angle.")

plt.figure()
plt.plot(angles, dxdtts_aligned)
plt.grid()
plt.title("Linear acceleration vs. angle")

plt.show()

# IPython.embed()
