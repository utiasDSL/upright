import numpy as np
from scipy.optimize import minimize

import upright_core as core

import IPython


# gravity
g = 9.81
G = np.array([0, 0, -g])


def lift(x):
    # fmt: off
    return np.array([
        [x[0], x[1], x[2], 0, 0, 0],
        [0, x[0], 0, x[1], x[2], 0],
        [0, 0, x[0], 0, x[1], x[2]]
    ])
    # fmt: on


def skew6(V):
    v, ω = V[:3], V[3:]
    Sv = core.math.skew3(v)
    Sω = core.math.skew3(ω)
    return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


def vec(J):
    return np.array([J[0, 0], J[0, 1], J[0, 2], J[1, 1], J[1, 2], J[2, 2]])


class ContactPoint:
    def __init__(self, position, normal, μ):
        self.position = np.array(position)
        self.normal = np.array(normal)
        self.span = core.math.plane_span(self.normal)
        self.μ = μ

        # matrix to convert contact force into body contact wrench
        self.W = np.vstack((np.eye(3), core.math.skew3(position)))

        # matrix to enforce friction cone constraint F @ f >= 0 ==> inside FC
        # fmt: off
        self.F = np.array([
            [1,  0,  0],
            [μ, -1, -1],
            [μ,  1, -1],
            [μ, -1,  1],
            [μ,  1,  1],
        ]) @ np.vstack((self.normal, self.span))
        # fmt: on


class BalancedObject:
    def __init__(self, m, h, δ, μ, h0=0, x0=0):
        self.m = m
        self.h = h  # height of CoM above base of object
        self.δ = δ
        self.μ = μ

        side_lengths = 2 * np.array([δ, δ, h])
        self.J = core.math.cuboid_inertia_matrix(m, side_lengths)

        self.h0 = h0  # height of base above tray
        self.x0 = x0

        self.origin = np.array([self.x0, 0, self.h + self.h0])
        S = core.math.skew3(self.origin)
        self.M = np.block([[m * np.eye(3), -m * S], [m * S, self.J]])

    def contacts(self):
        # contacts are in the body frame w.r.t. to the origin
        C1 = ContactPoint(
            position=[-self.δ, -self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        C2 = ContactPoint(
            position=[self.δ, -self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        C3 = ContactPoint(
            position=[self.δ, self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        C4 = ContactPoint(
            position=[-self.δ, self.δ, -self.h], normal=[0, 0, 1], μ=self.μ
        )
        return [C1, C2, C3, C4]


def body_gravito_inertial_wrench(C, V, A, obj):
    """Gravito-inertial wrench in the body frame.

    The supplied velocity twist V and acceleration A must also be in the body
    frame.
    """
    Ag = np.concatenate((C @ G, np.zeros(3)))
    return obj.M @ (A - Ag) + skew6(V) @ obj.M @ V


def body_contact_wrench(forces, contacts):
    """Contact wrench in the body frame.

    forces is an (n, 3) array of contact forces
    contacts is the list of contact points
    """
    return np.sum([c.W @ f for c, f in zip(contacts, forces)], axis=0)


def friction_cone_constraints(forces, contacts):
    return np.concatenate([c.F @ f for c, f in zip(contacts, forces)])
    # nc = len(contacts)
    # constraints = np.zeros(3 * nc)
    # for i in range(nc):
    #     fi = np.array([fs_xz[2 * i], 0, fs_xz[2 * i + 1]])
    #     fi_n = contacts[i].normal @ fi
    #     fi_t = contacts[i].tangent @ fi
    #     μi = contacts[i].μ
    #     constraints[i * 3 : (i + 1) * 3] = np.array(
    #         [fi_n, μi * fi_n - fi_t, μi * fi_n + fi_t]
    #     )
    # return constraints


def body_regressor(V, A):
    v, ω = V[:3], V[3:]
    a, α = A[:3], A[3:]

    Sω = core.math.skew3(ω)
    Sv = core.math.skew3(v)
    Sa = core.math.skew3(a)
    Sα = core.math.skew3(α)
    Lω = lift(ω)
    Lα = lift(α)

    # fmt: off
    return np.block([
        [(v + Sω @ v)[:, None], Sα + Sω @ Sω, np.zeros((3, 6))],
        [np.zeros((3, 1)), -Sa - skew(Sω @ v), Lα + Sω @ Lω]
    ])
    # fmt: on


def optimize_acceleration(C, V, ad, obj, a_bound=10, α_bound=10):
    # first six decision variables are the acceleration; the remaining ones are
    # contact forces
    nc = 4  # number of contact points
    nv = 6 + 3 * nc  # number of decision variables
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad
    f0 = -obj.m * C @ G / nc
    for i in range(nc):
        x0[6 + i * 3 : 6 + (i + 1) * 3] = f0

    # acceleration bounds
    bounds = [(None, None) for _ in range(nv)]
    for i in range(3):
        bounds[i] = (-a_bound, a_bound)
        bounds[i + 3] = (-α_bound, α_bound)

    contacts = obj.contacts()

    def cost(x):
        # try to match desired (linear) acceleration as much as possible
        a = x[:3]
        e = ad - a
        return 0.5 * e @ e + 0.005 * x[3:6] @ x[3:6]

    def eq_con(x):
        A = x[:6]
        forces = x[6:].reshape((nc, 3))

        # gravito inertia wrench and contact wrench must balance (Newton-Euler
        # equations) for each object
        giw = body_gravito_inertial_wrench(C, V, A, obj)
        cw = body_contact_wrench(forces, contacts)
        # IPython.embed()
        return giw - cw

    def ineq_con(x):
        forces = x[6:].reshape((nc, 3))
        return friction_cone_constraints(forces, contacts)

    res = minimize(
        cost,
        x0=x0,
        method="slsqp",
        constraints=[
            {"type": "eq", "fun": eq_con},
            {"type": "ineq", "fun": ineq_con},
        ],
        bounds=bounds,
    )
    A = res.x[:6]
    return A


def main():
    np.set_printoptions(precision=5, suppress=True)

    C = np.eye(3)  # C_bw
    V = np.array([0, 0, 0, 0, 0, 0])
    ad_world = np.array([1, 0, 0])  # TODO we could construct a control law here
    ad_body = C @ ad_world

    # A = np.array([0, 0, 1, 0, 0, 0])
    # Y = body_regressor(V, A)
    obj = BalancedObject(m=1, h=0.1, δ=0.05, μ=0.2, h0=0, x0=0)
    A = optimize_acceleration(C, V, ad_body, obj)
    IPython.embed()


if __name__ == "__main__":
    main()
