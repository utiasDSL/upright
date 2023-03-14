import numpy as np
from scipy.optimize import minimize
import cdd

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
    """6D cross product matrix"""
    v, ω = V[:3], V[3:]
    Sv = core.math.skew3(v)
    Sω = core.math.skew3(ω)
    return np.block([[Sω, np.zeros((3, 3))], [Sv, Sω]])


def vec(J):
    """Vectorize the inertia matrix"""
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
        # (this is the negative of the face form)
        # fmt: off
        P = np.vstack((self.normal, self.span))
        self.F = np.array([
            [1,  0,  0],
            [μ, -1, -1],
            [μ,  1, -1],
            [μ, -1,  1],
            [μ,  1,  1],
        ]) @ P
        # fmt: on

        # span (generator) form matrix FC = {Sz | z >= 0}
        # fmt: off
        self.S = P @ np.array([
            [1,  1, 1,  1],
            [μ, -μ, 0,  0],
            [0,  0, μ, -μ]
        ])
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

        # mass matrix
        self.M = np.block([[m * np.eye(3), -m * S], [m * S, self.J]])

        # polytopic constraints on the inertial parameters
        # Pθ >= p
        Jvec = vec(self.J)
        self.θ = np.concatenate(([m], m * self.origin, Jvec))
        Δθ = np.concatenate(([0.1, 0.5 * δ, 0.5 * δ, 0.5 * h], 0.1 * Jvec))
        θ_min = self.θ - Δθ
        θ_max = self.θ + Δθ
        self.P = np.vstack((np.eye(self.θ.shape[0]), -np.eye(self.θ.shape[0])))
        self.p = np.concatenate((θ_min, -θ_max))

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
    """Constraints are non-negative if all contact forces are inside their friction cones."""
    return np.concatenate([c.F @ f for c, f in zip(contacts, forces)])


def body_regressor(C, V, A):
    """Compute regressor matrix Y given body frame velocity V and acceleration A.

    The regressor maps the inertial parameters to the body inertial wrench: w = Yθ.
    """
    v, ω = V[:3], V[3:]
    a, α = A[:3], A[3:]

    # account for gravity
    a = a - C @ G

    Sω = core.math.skew3(ω)
    Sv = core.math.skew3(v)
    Sa = core.math.skew3(a)
    Sα = core.math.skew3(α)
    Lω = lift(ω)
    Lα = lift(α)

    # fmt: off
    return np.block([
        [(a + Sω @ v)[:, None], Sα + Sω @ Sω, np.zeros((3, 6))],
        [np.zeros((3, 1)), -Sa - core.math.skew3(Sω @ v), Lα + Sω @ Lω]
    ])
    # fmt: on


def body_regressor_components(C, V):
    """Compute components {Yi} of the regressor matrix Y such that
    Y = sum(Yi * Ai forall i)
    """
    Ys = []
    for i in range(6):
        A = np.zeros(6)
        A[i] = 1
        Ys.append(body_regressor(C, V, A))
    return np.array(Ys)


def body_regressor_by_vector_matrix(C, V, z):
    """Compute a matrix D such that D @ A == Y.T @ z for some vector z."""
    Ys = body_regressor_components(C, V)
    return np.hstack([Y.T @ z for Y in Ys])


def span_to_face_form(S):
    # span form
    # we have generators as columns but cdd wants it as rows, hence the transpose
    Smat = cdd.Matrix(np.hstack((np.zeros((S.shape[1], 1)), S.T)))
    Smat.rep_type = cdd.RepType.GENERATOR

    # polyhedron
    poly = cdd.Polyhedron(Smat)

    # face form: Ax <= b
    Fmat = poly.get_inequalities()
    F = np.array([Fmat[i] for i in range(Fmat.row_size)])
    b = F[:, 0]
    A = -F[:, 1:]
    return A, b


def cwc(contacts):
    """Build the (face form of the) contact wrench cone from contact points of an object."""
    # combine span form of each contact wrench cone to get the overall CWC in
    # span form
    S = np.hstack([c.W @ c.S for c in contacts])

    # convert to face form
    A, b = span_to_face_form(S)
    assert np.allclose(b, 0)

    # Fw >= 0 ==> there exist feasible contact forces to support wrench w
    return A


def optimize_acceleration(C, V, ad, obj, a_bound=10, α_bound=10):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
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


def optimize_acceleration_face_form(C, V, ad, obj, a_bound=10, α_bound=10):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    # decision variables are the acceleration
    nv = 6
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # acceleration bounds
    bounds = [(None, None) for _ in range(nv)]
    for i in range(3):
        bounds[i] = (-a_bound, a_bound)
        bounds[i + 3] = (-α_bound, α_bound)

    contacts = obj.contacts()
    F = cwc(contacts)

    def cost(x):
        # try to match desired (linear) acceleration as much as possible
        a = x[:3]
        e = ad - a
        return 0.5 * e @ e + 0.005 * x[3:6] @ x[3:6]

    def ineq_con(x):
        A = x
        giw = body_gravito_inertial_wrench(C, V, A, obj)
        return F @ giw

    res = minimize(
        cost,
        x0=x0,
        method="slsqp",
        constraints=[
            {"type": "ineq", "fun": ineq_con},
        ],
        bounds=bounds,
    )
    A = res.x
    return A


def optimize_acceleration_robust(C, V, ad, obj, a_bound=10, α_bound=10):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    contacts = obj.contacts()
    F = cwc(contacts)

    # first 6 decision variables are the acceleration, then duals λ
    # corresponding to the CWC constraints, and finally duals z on the
    # parameter constraints
    nλ = F.shape[0]
    nz = obj.P.shape[0]
    nv = 6 + nλ + nz
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # bounds
    # all duals must be non-negative
    bounds = [(0, None) for _ in range(nv)]
    for i in range(3):
        bounds[i] = (-a_bound, a_bound)
        bounds[i + 3] = (-α_bound, α_bound)

    def cost(x):
        # try to match desired (linear) acceleration as much as possible
        a = x[:3]
        e = ad - a
        return 0.5 * e @ e + 0.005 * x[3:6] @ x[3:6]

    def eq_con(x):
        A, λ, z = x[:6], x[6 : 6 + nλ], x[-nz:]
        Y = body_regressor(C, V, A)
        return np.append(-Y.T @ F.T @ λ + obj.P.T @ z, np.sum(λ) - 1)

    def ineq_con(x):
        z = x[-nz:]
        return np.array([-z @ obj.p])

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
    giw = body_gravito_inertial_wrench(C, V, A, obj)
    IPython.embed()
    return A


# def inner_optimization(C, V, A, obj):
#     contacts = obj.contacts()
#     F = cwc(contacts)
#
#     # first opt var is y, the rest are the inertial parameters θ
#     nv = 11
#     x0 = np.zeros(nv)
#
#     # bounds
#     bounds = [(None, None) for _ in range(nv)]
#
#     def cost(x):
#         # max y
#         return x[0]
#
#     def ineq_con(x):
#         y, θ = x[0], x[1:]
#         Y = body_regressor(C, V, A)
#         return np.concatenate((obj.P @ θ - obj.p, -F @ Y @ θ + y * np.ones(F.shape[0])))
#
#     res = minimize(
#         cost,
#         x0=x0,
#         method="slsqp",
#         constraints=[
#             {"type": "ineq", "fun": ineq_con},
#         ],
#         bounds=bounds,
#     )
#     y, θ = res.x[0], res.x[1:]
#     print(f"y = {y}")
#     print(f"θ = {θ}")
#
#     # the GIW is not feasible, which means y should be able go negative
#     giw = body_gravito_inertial_wrench(C, V, A, obj)
#
#     IPython.embed()
#     return y, θ


def main():
    np.set_printoptions(precision=5, suppress=True)

    C = np.eye(3)  # C_bw
    V = np.array([0, 0, 0, 0, 0, 0])
    ad_world = np.array([5, 0, 0])  # TODO we could construct a control law here
    ad_body = C @ ad_world

    obj = BalancedObject(m=1, h=0.1, δ=0.05, μ=0.2, h0=0, x0=0)

    # test the body regressor
    # A = np.array([1, 2, 3, 4, 5, 6])
    # Y = body_regressor(V, A)
    # giw = body_gravito_inertial_wrench(C, V, A, obj)
    # IPython.embed()
    # return

    A = np.array([6, 0, 0, 0, 0, 0])
    inner_optimization(C, V, A, obj)

    # A1 = optimize_acceleration(C, V, ad_body, obj)
    # A2 = optimize_acceleration_face_form(C, V, ad_body, obj)
    # Ar = optimize_acceleration_robust(C, V, ad_body, obj)
    # IPython.embed()


if __name__ == "__main__":
    main()
