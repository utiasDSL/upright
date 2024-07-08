"""Constraint elimination code.

The goal is to prove that certain constraints will never be active over a set
of parameter values and object states (orientation, velocity, acceleration).
"""
import numpy as np
import cvxpy as cp
import time
from scipy.optimize import minimize

import upright_core as core
import upright_robust as rob
import rigeo as rg

import IPython


# gravity constant
g = 9.81


def solve_global_relaxed_dual(
    obj,
    f,
    other_constr_idx=None,
    v_max=1,
    ω_max=0.25,
    a_max=1,
    α_max=0.25,
    max_tilt_angle=np.deg2rad(30),
):
    """Global convex problem based on the face form of the dual constraint formulation."""
    if other_constr_idx is None:
        other_constr_idx = []

    Z = rob.body_regressor_by_vector_velocity_matrix(f)
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]

    nv = 6
    ng = 3
    nz = Z.shape[0]

    # fmt: off
    P_tilde = rob.compute_P_tilde_matrix(obj.P, obj.p)
    R = rob.span_to_face_form(P_tilde.T)
    # R = R / np.max(np.abs(R))

    # gravity constraints
    z_normal = np.array([0, 0, 1])

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    Λ = cp.Variable((6, 6), PSD=True)

    values = []
    Vs = []

    for i in range(R.shape[0]):
        objective = cp.Maximize(R[i, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A))
        # fmt: off
        constraints = [
            z == cp.vec(Λ),

            # we constrain z completely through Λ
            cp.diag(Λ[:3, :3]) <= v_max**2,
            cp.diag(Λ[3:, 3:]) <= ω_max**2,

            # gravity constraints
            cp.norm(G) <= g,
            z_normal @ G <= -g * np.cos(max_tilt_angle),

            # acceleration constraints
            A[:3] >= -a_max,
            A[:3] <= a_max,
            A[3:] >= -α_max,
            A[3:] <= α_max,
        ]
        # fmt: on
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
        values.append(problem.value)
    return np.min(values)


def solve_global_relaxed_dual_approx_inertia(
    obj1,
    obj2,
    f,
    F2,
    idx,
    v_max=1,
    ω_max=0.25,
    a_max=1,
    α_max=0.25,
):
    """Global convex problem based on the face form of the dual constraint formulation.

    Minimize the constraints for obj1 subject to all constraints on obj2, which
    are typically approximations of those on obj1.
    """
    Z = rob.body_regressor_by_vector_velocity_matrix(f)
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]

    # construct for every constraint
    Zs = []
    Ds = []
    Dgs = []
    for i in range(F2.shape[0]):
        Zs.append(rob.body_regressor_by_vector_velocity_matrix(F2[i, :]))
        Ds.append(rob.body_regressor_by_vector_acceleration_matrix(F2[i, :]))
        Dgs.append(-Ds[-1][:3, :])

    nv = 6
    ng = 3
    nz = Z.shape[0]

    P1_tilde = rob.compute_P_tilde_matrix(obj1.P, obj1.p)
    R1 = rob.span_to_face_form(P1_tilde.T)
    # R1 = R1 / np.max(np.abs(R1))

    P2_tilde = rob.compute_P_tilde_matrix(obj2.P, obj2.p)
    R2 = rob.span_to_face_form(P2_tilde.T)
    # R2 = R2 / np.max(np.abs(R2))

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    Λ = cp.Variable((6, 6), PSD=True)

    values = []
    Vs = []

    for i in range(R1.shape[0]):
        objective = cp.Maximize(R1[i, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A))
        # fmt: off
        constraints = [
            z == cp.vec(Λ),

            # we constrain z completely through Λ
            cp.diag(Λ[:3, :3]) <= v_max**2,
            cp.diag(Λ[3:, 3:]) <= ω_max**2,

            # gravity constraints
            cp.norm(G) <= g,
            z_normal @ G <= -g * np.cos(max_tilt_angle),

            # aligned approach
            # (A[:3] - G) @ [1, 0, 0] == 0,
            # (A[:3] - G) @ [0, 1, 0] == 0,

            # adaptive tilting
            # A[3:] == 0.1 * core.math.skew3(z_normal) @ (A[:3] - G),

            # acceleration constraints
            A[:3] >= -a_max,
            A[:3] <= a_max,
            A[3:] >= -α_max,
            A[3:] <= α_max,
        ] + [
            # all of the approximate constraints
            R2[:, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A) <= 0.0
            for (Z, D, Dg) in zip(Zs, Ds, Dgs)
        ]
        # fmt: on
        problem = cp.Problem(objective, constraints)
        # t0 = time.time()
        problem.solve(solver=cp.MOSEK)
        # t1 = time.time()
        # print(f"Δt = {t1 - t0}")
        values.append(problem.value)
        # if problem.value > 0:
        #     print(f"eigvals = {np.linalg.eigvals(Λ.value)}")

    n_neg = np.sum(np.array(values) < 0)
    print(
        f"min={np.min(values)}, max={np.max(values)}, ({n_neg} / {len(values)} negative)"
    )
    return np.min(values)


def solve_local_dual(obj, F, idx, other_constr_idx, v_max=1, ω_max=1):
    """Local solution of the face form of the dual problem."""

    # A, g, V, θ
    A0 = np.zeros(6)
    G0 = np.array([0, 0, -9.81])
    V0 = np.array([v_max, 0, 0, ω_max, 0, 0])

    # limits
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # fmt: off
    P_tilde = np.block([
        [obj.P, obj.p[:, None]],
        [np.zeros((1, obj.P.shape[1])), np.array([[1]])]])
    # fmt: on
    R = rob.span_to_face_form(P_tilde.T)
    R = R / np.max(np.abs(R))

    a_max = 1
    α_max = 1

    f = F[idx, :]

    values = []
    Vs = []

    for i in range(R.shape[0]):
        x0 = np.concatenate((A0, G0, V0))

        def cost(x):
            A, G, V = x[:6], x[6:9], x[9:15]
            Ag = np.concatenate((G, np.zeros(3)))

            d = rob.body_regressor_VG_by_vector(V, Ag, f)
            D = rob.body_regressor_A_by_vector(f)

            # negative to maximize
            return -R[i, :-1] @ (d + D @ A)

        def eq_cons(x):
            G = x[6:9]
            # return np.linalg.norm(G) - g
            return G @ G - g**2

        def ineq_cons(x):
            A, G, V = x[:6], x[6:9], x[9:15]
            # squared formulation is numerically better for the solver (it is
            # differentiable)
            return np.concatenate(
                (
                    [
                        a_max**2 - np.linalg.norm(A[:3]) ** 2,
                        α_max**2 - np.linalg.norm(A[3:]) ** 2,
                    ],
                    [
                        v_max**2 - np.linalg.norm(V[:3]) ** 2,
                        ω_max**2 - np.linalg.norm(V[3:]) ** 2,
                    ],
                    [-g * np.cos(max_tilt_angle) - G[2]],
                )
            )

        res = minimize(
            cost,
            x0=x0,
            method="slsqp",
            constraints=[
                {"type": "ineq", "fun": ineq_cons},
                {"type": "eq", "fun": eq_cons},
            ],
        )
        A, G, V = res.x[:6], res.x[6:9], res.x[9:15]
        value = cost(res.x)
        if not res.success:
            print("failed to solve local problem!")
            IPython.embed()
        # else:
        #     print("success")
        values.append(value)
        Vs.append(V)

    min_idx = np.argmin(values)
    print(f"local value = {values[min_idx]}")
    print(f"local V = {Vs[min_idx]}")
    return np.min(values)


def main_inertia_approx():
    np.set_printoptions(precision=5, suppress=True)

    box = rg.Box(half_extents=[0.05, 0.05, 0.2], center=[0, 0, 0.2])
    ell = box.minimum_bounding_ellipsoid()
    obj1 = rob.BalancedObject(
        m=1,
        h=0.2,
        δ=0.05,
        μ=0.2,
        h0=0,
        x0=0,
        Q=ell.Q,
        approx_inertia=False,
    )
    obj2 = rob.BalancedObject(
        m=1,
        h=0.2,
        δ=0.05,
        μ=0.2,
        h0=0,
        x0=0,
        approx_inertia=True,
    )

    F1 = rob.cwc(obj1.contacts())
    F2 = rob.cwc(obj2.contacts())
    # norm = np.max(np.abs(F1))
    # F1 /= norm
    # F2 /= norm

    print("We want all constraints negative.")

    for i in range(F1.shape[0]):
        print(i + 1)
        f = F1[i, :]
        relaxed = solve_global_relaxed_dual_approx_inertia(obj1, obj2, f, F2, i)


def main_elimination():
    np.set_printoptions(precision=5, suppress=True)

    box = rg.Box(half_extents=[0.05, 0.05, 0.2], center=[0, 0, 0.2])
    ell = box.minimum_bounding_ellipsoid()
    obj = rob.BalancedObject(
        m=1,
        h=0.2,
        δ=0.05,
        μ=0.5,
        h0=0,
        x0=0,
        Q=ell.Q,
        approx_inertia=False,
    )
    F = rob.cwc(obj.contacts())

    IPython.embed()

    for i in range(F.shape[0]):
        print(i + 1)
        f = F[i, :]
        relaxed = solve_global_relaxed_dual(obj, f)
        print(relaxed)


if __name__ == "__main__":
    main_inertia_approx()
    # main_elimination()
