import numpy as np
import scipy
from scipy.optimize import minimize
from scipy import sparse
import cdd
from qpsolvers import solve_qp
import osqp
import time

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
        Δθ = np.concatenate(([0.1, 0.01 * δ, 0.01 * δ, 0.01 * h], 0 * Jvec))
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
    # a = a - C @ G

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
    # velocity + gravity component
    Ag = np.concatenate((C @ G, np.zeros(3)))
    Y0 = body_regressor(C, V, -Ag)

    # acceleration component
    Ys = []
    V = np.zeros(6)
    for i in range(6):
        A = np.zeros(6)
        A[i] = 1
        Ys.append(body_regressor(C, V, A))
    return Y0, Ys


def body_regressor_by_vector_matrix(C, V, z):
    """Compute a matrix D such that d0 + D @ A == Y.T @ z for some vector z."""
    Y0, Ys = body_regressor_components(C, V)
    d0 = Y0.T @ z
    D = np.vstack([Y.T @ z for Y in Ys]).T
    return d0, D


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

    IPython.embed()

    # Fw >= 0 ==> there exist feasible contact forces to support wrench w
    return A


def optimize_acceleration(C, V, ad, obj, a_bound=5, α_bound=1):
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
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then a ton of duals
    # {λ_i}
    nf = F.shape[0]
    nλ = obj.p.shape[0]
    nv = 6 + nf * nλ
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # bounds
    # all duals must be non-negative
    bounds = [(0, None) for _ in range(nv)]
    for i in range(3):
        bounds[i] = (-a_bound, a_bound)
        bounds[i + 3] = (-α_bound, α_bound)

    # pre-compute Jacobians
    # equality Jacobian
    n_eq = obj.P.shape[1]  # dimension of one set of equality constraints
    N_eq = obj.P.shape[1] * nf  # dim of all equality constraints
    J_eq = np.zeros((N_eq, nv))
    d_eq = np.zeros(N_eq)
    for i in range(nf):
        d0, D = body_regressor_by_vector_matrix(C, V, F[i, :])
        d_eq[i * n_eq : (i + 1) * n_eq] = d0

        # Jacobian w.r.t. A
        J_eq[i * n_eq : (i + 1) * n_eq, :6] = D

        # Jacobian w.r.t. λi
        J_eq[i * n_eq : (i + 1) * n_eq, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.P.T

    # inequality Jacobian
    J_ineq = np.zeros((nf, nv))
    for i in range(nf):
        J_ineq[i, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.p

    # λ = np.arange(20)
    # z = obj.P.T @ λ
    # Pinv = np.linalg.pinv(obj.P.T)
    # Z = scipy.linalg.null_space(obj.P.T)
    # IPython.embed()
    # return

    def cost(x):
        # try to match desired (linear) acceleration as much as possible
        a = x[:3]
        α = x[3:6]
        e = ad - a
        return 0.5 * e @ e + 0.005 * α @ α  # + 0.005 * x[:6] @ x[:6]

    def jac(x):
        a = x[:3]
        α = x[3:6]
        e = ad - a

        J = np.zeros(nv)
        J[:3] = -e
        J[3:6] = 0.01 * α
        return J

    def eq_con(x):
        return J_eq @ x + d_eq

    def eq_con_jac(x):
        return J_eq

    def ineq_con(x):
        return J_ineq @ x

    def ineq_con_jac(x):
        return J_ineq

    # res = minimize(
    #     cost,
    #     x0=x0,
    #     jac=jac,
    #     method="slsqp",
    #     constraints=[
    #         {"type": "eq", "fun": eq_con, "jac": eq_con_jac},
    #         {"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac},
    #     ],
    #     bounds=bounds,
    # )
    # A = res.x[:6]

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)
    q = np.zeros(nv)
    q[:3] = -ad

    lb = np.zeros(nv)
    ub = np.inf * np.ones(nv)

    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    # NOTE that we don't actually *need* a high-accuracy solution: any feasible
    # solution is acceptable. Warm-starting also will likely help a lot.
    t0 = time.time()
    x = solve_qp(
        P=sparse.csc_matrix(P),
        q=q,
        G=sparse.csc_matrix(-J_ineq),
        h=np.zeros(J_ineq.shape[0]),
        A=sparse.csc_matrix(J_eq),
        b=-d_eq,
        lb=lb,
        ub=ub,
        initvals=x0,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=10000,
        solver="osqp",
        # polish=True,
    )
    t1 = time.time()
    print(f"solve took {t1 - t0} seconds")
    A = x[:6]

    R = np.linalg.pinv(J_eq)
    N = scipy.linalg.null_space(J_eq)

    IPython.embed()

    return A


def optimize_acceleration_robust_osqp(C, V, ad, obj, a_bound=5, α_bound=1):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    contacts = obj.contacts()
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then a ton of duals
    # {λ_i}
    nf = F.shape[0]
    nλ = obj.p.shape[0]
    nv = 6 + nf * nλ
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # bounds
    # all duals must be non-negative
    bounds = [(0, None) for _ in range(nv)]
    for i in range(3):
        bounds[i] = (-a_bound, a_bound)
        bounds[i + 3] = (-α_bound, α_bound)

    # pre-compute Jacobians
    # equality Jacobian
    n_eq = obj.P.shape[1]  # dimension of one set of equality constraints
    N_eq = obj.P.shape[1] * nf  # dim of all equality constraints
    J_eq = np.zeros((N_eq, nv))
    d_eq = np.zeros(N_eq)
    for i in range(nf):
        d0, D = body_regressor_by_vector_matrix(C, V, F[i, :])
        d_eq[i * n_eq : (i + 1) * n_eq] = d0

        # Jacobian w.r.t. A
        J_eq[i * n_eq : (i + 1) * n_eq, :6] = D

        # Jacobian w.r.t. λi
        J_eq[i * n_eq : (i + 1) * n_eq, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.P.T

    # inequality Jacobian
    J_ineq = np.zeros((nf, nv))
    for i in range(nf):
        J_ineq[i, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.p

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)
    q = np.zeros(nv)
    q[:3] = -ad

    lb = np.zeros(nv)
    ub = np.inf * np.ones(nv)

    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    n_eq = J_eq.shape[0]
    n_ineq = J_ineq.shape[0]
    A = np.vstack((J_eq, J_ineq, np.eye(nv)))
    L = np.concatenate((-d_eq, np.zeros(n_ineq), lb))
    U = np.concatenate((-d_eq, np.inf * np.ones(n_ineq), ub))

    m = osqp.OSQP()
    m.setup(
        P=sparse.csc_matrix(P),
        q=q,
        A=sparse.csc_matrix(A),
        l=L,
        u=U,
        verbose=False,
        eps_abs=1e-6,
        eps_rel=1e-6,
        max_iter=10000,
    )
    m.warm_start(x=x0)

    # TODO we'd like to be able to re-solve again---does this cheapen things?
    t0 = time.time()
    res = m.solve()
    t1 = time.time()
    print(f"solve took {t1 - t0} seconds")
    A1 = res.x[:6]

    ad = np.array([0, -2, 0])  # some other reference
    q = np.zeros(nv)
    q[:3] = -ad
    m.update(q=q)
    t0 = time.time()
    res = m.solve()
    t1 = time.time()
    A2 = res.x[:6]
    print(f"update and solve took {t1 - t0} seconds")

    IPython.embed()

    return A1


def optimize_acceleration_robust2(C, V, ad, obj, a_bound=10, α_bound=10):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    contacts = obj.contacts()
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then a ton of duals
    # {λ_i}
    nf = F.shape[0]
    nλ = obj.p.shape[0]
    nv = 6 + nf * nλ
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # pre-compute Jacobians
    # equality Jacobian
    n_eq = obj.P.shape[1]  # dimension of one set of equality constraints
    N_eq = obj.P.shape[1] * nf  # dim of all equality constraints
    J_eq = np.zeros((N_eq, nv))
    d_eq = np.zeros(N_eq)
    for i in range(nf):
        d0, D = body_regressor_by_vector_matrix(C, V, F[i, :])
        d_eq[i * n_eq : (i + 1) * n_eq] = d0

        # Jacobian w.r.t. A
        J_eq[i * n_eq : (i + 1) * n_eq, :6] = D

        # Jacobian w.r.t. λi
        J_eq[i * n_eq : (i + 1) * n_eq, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.P.T

    # inequality Jacobian
    J_ineq = np.zeros((nf, nv))
    for i in range(nf):
        J_ineq[i, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.p

    # re-formulation using a decomposition: x = Y @ b + Z @ z, where z is now
    # our optimization variable. This appears to be extremely slow.
    Y = np.linalg.pinv(J_eq)
    Z = scipy.linalg.null_space(J_eq)

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)

    q = np.zeros(nv)
    q[:3] = -ad
    q = Z.T @ q + Z.T @ P @ Y @ -d_eq
    P = Z.T @ P @ Z

    lb = np.zeros(nv)
    ub = np.inf * np.ones(nv)

    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    Gm = -J_ineq @ Z
    h = J_ineq @ Y @ -d_eq

    Ga = np.vstack((Gm, Z, -Z))
    ha = np.concatenate((h, ub + Y @ d_eq, -lb - Y @ d_eq))

    z0, _, _, _ = np.linalg.lstsq(Z, x0 + Y @ d_eq, rcond=None)

    t0 = time.time()
    z = solve_qp(
        P=sparse.csc_matrix(P),
        q=q,
        G=sparse.csc_matrix(Ga),
        h=ha,
        initvals=z0,
        # eps_abs=1e-6,
        # eps_rel=1e-6,
        # max_iter=10000,
        solver="proxqp",
    )
    t1 = time.time()
    print(f"solve took {t1 - t0} seconds")
    x = -Y @ d_eq + Z @ z
    A = x[:6]

    IPython.embed()

    return A


def optimize_acceleration_sequential(C, V, ad, obj, a_bound=5, α_bound=1):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    contacts = obj.contacts()
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then one set of duals
    nf = F.shape[0]
    nλ = obj.p.shape[0]
    nz = 10
    nv = 6 + nz
    x0 = np.zeros(nv)

    # bounds on acceleration
    lb = -np.inf * np.ones(nv)
    ub = np.inf * np.ones(nv)
    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    # initial guess
    x0[:3] = ad

    # range and nullspace of P.T
    Pr = np.linalg.pinv(obj.P.T)
    Pn = scipy.linalg.null_space(obj.P.T)

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)
    P_sparse = sparse.csc_matrix(P)

    q = np.zeros(nv)
    q[:3] = -ad

    zs = np.zeros((nf, nz))
    Gs = np.zeros((0, nv))
    hs = []

    # solve sequence of QPs
    solve_time_total = 0
    for i in range(nf):
        d0, D = body_regressor_by_vector_matrix(C, V, F[i, :])

        G1 = np.concatenate((obj.p @ Pr @ D, -obj.p @ Pn))
        G2 = np.hstack((Pr @ D, -Pn))
        Gi = np.vstack((G1, G2))
        h = np.concatenate(([-obj.p @ Pr @ d0], -Pr @ d0))

        G_total = np.vstack((Gs, Gi))
        h_total = np.concatenate((hs, h))
        # print(Gi.shape)

        t0 = time.time()
        x = solve_qp(
            P=P_sparse,
            q=q,
            G=sparse.csc_matrix(G_total),
            h=h_total,
            lb=lb,
            ub=ub,
            initvals=x0,
            solver="osqp",
        )
        t1 = time.time()
        solve_time_total += t1 - t0
        A = x[:6]
        z = x[6:]
        G_save = Gi.copy()
        G_save[:, 6:] = 0
        Gs = np.vstack((Gs, G_save))
        hs = np.concatenate((hs, h - Gi[:, 6:] @ z))
    print(f"solve took {solve_time_total} seconds")
    return A


    # inequality Jacobian
    J_ineq = np.zeros((nf, nv))
    for i in range(nf):
        J_ineq[i, 6 + i * nλ : 6 + (i + 1) * nλ] = obj.p

    # re-formulation using a decomposition: x = Y @ b + Z @ z, where z is now
    # our optimization variable. This appears to be extremely slow.
    Y = np.linalg.pinv(J_eq)
    Z = scipy.linalg.null_space(J_eq)

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)

    q = np.zeros(nv)
    q[:3] = -ad
    q = Z.T @ q + Z.T @ P @ Y @ -d_eq
    P = Z.T @ P @ Z

    lb = np.zeros(nv)
    ub = np.inf * np.ones(nv)

    lb[:3] = -a_bound
    lb[3:6] = -α_bound
    ub[:3] = a_bound
    ub[3:6] = α_bound

    Gm = -J_ineq @ Z
    h = J_ineq @ Y @ -d_eq

    Ga = np.vstack((Gm, Z, -Z))
    ha = np.concatenate((h, ub + Y @ d_eq, -lb - Y @ d_eq))

    z0, _, _, _ = np.linalg.lstsq(Z, x0 + Y @ d_eq, rcond=None)

    t0 = time.time()
    z = solve_qp(
        P=sparse.csc_matrix(P),
        q=q,
        G=sparse.csc_matrix(Ga),
        h=ha,
        initvals=z0,
        # eps_abs=1e-6,
        # eps_rel=1e-6,
        # max_iter=10000,
        solver="proxqp",
    )
    t1 = time.time()
    print(f"solve took {t1 - t0} seconds")
    x = -Y @ d_eq + Z @ z
    A = x[:6]

    IPython.embed()

    return A


def inner_optimization(C, V, A, obj):
    # TODO we want to optimize over all of the Fs separately here, then take
    # the max one

    contacts = obj.contacts()
    F = -cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))  # body frame gravity

    # first opt var is y, the rest are the inertial parameters θ
    nf = F.shape[0]
    nθ = obj.P.shape[1]  # num parameters
    nλ = obj.P.shape[0]

    # compute primals
    ys = np.zeros(nf)
    for i in range(nf):
        x0 = np.zeros(nθ)

        def cost_and_jac(x):
            Y = body_regressor(C, V, A - Ag)
            # negative since we are maximizing
            cost = -F[i, :] @ Y @ x
            jac = -F[i, :] @ Y
            return cost, jac

        def ineq_con(x):
            return obj.P @ x - obj.p

        def ineq_con_jac(x):
            return obj.P

        res = minimize(
            cost_and_jac,
            x0=x0,
            jac=True,
            method="slsqp",
            constraints=[
                {"type": "ineq", "fun": ineq_con, "jac": ineq_con_jac},
            ],
        )
        ys[i], _ = cost_and_jac(res.x)

    # compute duals
    ds = np.zeros(nf)
    λs = np.zeros((nf, nλ))
    bounds = [(0, None) for _ in range(nλ)]
    for i in range(nf):
        x0 = np.zeros(nλ)

        def cost_and_jac(x):
            cost = -obj.p @ x
            jac = -obj.p
            return cost, jac

        def eq_con(x):
            Y = body_regressor(C, V, A - Ag)
            return Y.T @ F[i, :] + obj.P.T @ x

        def eq_con_jac(x):
            return obj.P.T

        res = minimize(
            cost_and_jac,
            x0=x0,
            jac=True,
            method="slsqp",
            constraints=[
                {"type": "eq", "fun": eq_con, "jac": eq_con_jac},
            ],
            bounds=bounds,
        )
        ds[i], _ = cost_and_jac(res.x)
        λs[i, :] = res.x

    # the GIW is not feasible, which means y should be able go negative
    # giw = body_gravito_inertial_wrench(C, V, A, obj)

    # IPython.embed()
    return ys, ds, λs


def main():
    np.set_printoptions(precision=5, suppress=True)

    C = np.eye(3)  # C_bw
    V = np.array([0, 0, 0, 0, 0, 0])
    ad_world = np.array([3, 0, 0])  # TODO we could construct a control law here
    ad_body = C @ ad_world

    obj = BalancedObject(m=1, h=0.1, δ=0.05, μ=0.2, h0=0, x0=0)

    # test the body regressor
    # Ag = np.concatenate((C @ G, np.zeros(3)))  # body frame gravity
    # A = np.array([1, 2, 3, 4, 5, 6])
    # Y = body_regressor(C, V, A - Ag)
    # Y0, Ys = body_regressor_components(C, V)
    # Y_other = Y0.copy()
    # for i in range(6):
    #     Y_other += Ys[i] * A[i]
    #
    # z = np.arange(6)
    # one = Y.T @ z
    # d0, D = body_regressor_by_vector_matrix(C, V, z)
    # two = d0 + D @ A

    cwc(obj.contacts())
    IPython.embed()
    return

    # A_body = np.array([3, 0, 0, 0, 0, 0])
    # A1 = optimize_acceleration(C, V, ad_body, obj)
    # ys, ds, λs = inner_optimization(C, V, A1, obj)
    # Ar1 = optimize_acceleration_robust_osqp(C, V, ad_body, obj)
    # Ar2 = optimize_acceleration_sequential(C, V, ad_body, obj)
    # print(Ar2)



if __name__ == "__main__":
    main()
