import numpy as np
import scipy
from scipy.optimize import minimize
from scipy import sparse
from qpsolvers import solve_qp
import osqp
import time

import upright_robust as rob

import IPython


# gravity
g = 9.81
G = np.array([0, 0, -g])


def optimize_acceleration(C, V, ad, obj, a_bound=5, α_bound=1):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    # TODO write this as a QP
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
        giw = rob.body_gravito_inertial_wrench(C, V, A, obj)
        cw = rob.body_contact_wrench(forces, contacts)
        return giw - cw

    def ineq_con(x):
        forces = x[6:].reshape((nc, 3))
        return rob.friction_cone_constraints(forces, contacts)

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
    F = rob.cwc(contacts)

    def cost(x):
        # try to match desired (linear) acceleration as much as possible
        a = x[:3]
        e = ad - a
        return 0.5 * e @ e + 0.005 * x[3:6] @ x[3:6]

    def ineq_con(x):
        A = x
        giw = rob.body_gravito_inertial_wrench(C, V, A, obj)
        return F @ giw  # NOTE this is positive

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
    F = rob.cwc(contacts)
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
        d0, D = rob.body_regressor_by_vector_matrix(C, V, F[i, :])
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

    return A

def optimize_acceleration_robust_face(C, V, ad, obj, a_bound=10, α_bound=10):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    contacts = obj.contacts()
    F = rob.cwc(contacts)
    Ag = np.concatenate((C @ G, np.zeros(3)))

    # first 6 decision variables are the acceleration, then a ton of duals
    # {λ_i}
    nf = F.shape[0]
    nv = 6
    x0 = np.zeros(nv)

    # initial guess
    x0[:3] = ad

    # fmt: off
    P_tilde = np.block([
        [obj.P, obj.p[:, None]],
        [np.zeros((1, obj.P.shape[1])), np.array([[-1]])]])
    # fmt: on
    R = rob.span_to_face_form(P_tilde.T)[0]

    # pre-compute Jacobians
    n_ineq = R.shape[0]  # dimension of one set of equality constraints
    N_ineq = n_ineq * nf  # dim of all equality constraints
    J_ineq = np.zeros((N_ineq, nv))
    d_ineq = np.zeros(N_ineq)
    for i in range(nf):
        d, D = rob.body_regressor_by_vector_matrix(C, V, F[i, :])
        d_tilde = np.append(d, 0)
        D_tilde = np.vstack((D, np.zeros((1, D.shape[1]))))
        d_ineq[i * n_ineq : (i + 1) * n_ineq] = R @ d_tilde

        # Jacobian w.r.t. A
        J_ineq[i * n_ineq : (i + 1) * n_ineq, :] = R @ D_tilde

    P = np.zeros((nv, nv))
    P[:3, :3] = np.eye(3)
    P[3:6, 3:6] = 0.01 * np.eye(3)
    q = np.zeros(nv)
    q[:3] = -ad

    ub = np.concatenate((a_bound * np.ones(3), α_bound * np.ones(3)))
    lb = -ub

    # NOTE that we don't actually *need* a high-accuracy solution: any feasible
    # solution is acceptable. Warm-starting also will likely help a lot.
    t0 = time.time()
    x = solve_qp(
        P=sparse.csc_matrix(P),
        q=q,
        G=sparse.csc_matrix(-J_ineq),
        h=d_ineq,
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

    return A


def optimize_acceleration_robust_osqp(C, V, ad, obj, a_bound=5, α_bound=1):
    """Optimize acceleration to best matched desired subject to balancing constraints."""
    contacts = obj.contacts()
    F = rob.cwc(contacts)
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
        d0, D = rob.body_regressor_by_vector_matrix(C, V, F[i, :])
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


def inner_optimization(C, V, A, obj):
    # TODO we want to optimize over all of the Fs separately here, then take
    # the max one

    contacts = obj.contacts()
    F = rob.cwc(contacts)
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
            Y = rob.body_regressor(C, V, A - Ag)
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
            Y = rob.body_regressor(C, V, A - Ag)
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

    return ys, ds, λs


def main():
    np.set_printoptions(precision=5, suppress=True)

    C = np.eye(3)  # C_bw
    V = np.array([0, 0, 0, 0, 0, 0])
    ad_world = np.array([3, 0, 0])  # TODO we could construct a control law here
    ad_body = C @ ad_world

    obj = rob.BalancedObject(m=1, h=0.05, δ=0.05, μ=0.2, h0=0, x0=0)

    # test the body regressor
    Ag = np.concatenate((C @ G, np.zeros(3)))  # body frame gravity
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

    # L = rob.lift6_matrices()
    # IPython.embed()

    # contacts = obj.contacts()
    F = rob.cwc(obj.contacts())
    # giw = rob.body_gravito_inertial_wrench(C, V, np.zeros(6), obj)
    IPython.embed()
    return

    # A_body = np.array([3, 0, 0, 0, 0, 0])
    # A1 = optimize_acceleration(C, V, ad_body, obj)
    # ys, ds, λs = inner_optimization(C, V, A1, obj)
    Ar1 = optimize_acceleration_robust(C, V, ad_body, obj)
    Ar2 = optimize_acceleration_robust_face(C, V, ad_body, obj)
    # Ar2 = optimize_acceleration_sequential(C, V, ad_body, obj)
    print(Ar1)
    print(Ar2)


if __name__ == "__main__":
    main()
