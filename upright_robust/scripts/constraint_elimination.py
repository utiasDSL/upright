"""Constraint elimination code.

The goal is to prove that certain constraints will never be active over a set
of parameter values and object states (orientation, velocity, acceleration).
"""
import numpy as np
import cvxpy as cp
import time
from scipy.optimize import minimize

import upright_robust as rob

import IPython


# gravity constant
g = 9.81


def outer(x):
    y = cp.reshape(x, (x.shape[0], 1))
    return y @ y.T


def schur(X, x):
    y = cp.reshape(x, (x.shape[0], 1))
    return cp.bmat([[X, y], [y.T, [[1]]]])


def vech(X):
    return cp.hstack((X[i, i:] for i in range(X.shape[0]))).T


def vech_np(X):
    arrs = [X[i, i:] for i in range(X.shape[0])]
    return np.concatenate(arrs)


def extract_z(Λ):
    z = vech(Λ)
    idx = [4, 5, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20]
    return z[idx]


def extract_z_np(Λ):
    z = vech_np(Λ)
    idx = [4, 5, 8, 10, 12, 13, 15, 16, 17, 18, 19, 20]
    return z[idx]


def compute_Q_matrix(f):
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]
    # Z = rob.body_regressor_by_vector_velocity_matrix_half(f)
    Z = rob.body_regressor_by_vector_velocity_matrix(f)

    nv = 6
    ng = 3
    nθ = 10
    nz = Z.shape[0]

    Q = np.block([
        [np.zeros((nv, nv + ng + nz)), D],
        [np.zeros((ng, nv + ng +  nz)), Dg],
        [np.zeros((nz, nv + ng +  nz)), Z],
        [D.T, Dg.T, Z.T, np.zeros((nθ, nθ))]])
    return Q


def solve_global_relaxed(obj, F, idx, other_constr_idx):
    f = F[idx, :]
    Q = compute_Q_matrix(f)
    # Z = rob.body_regressor_by_vector_velocity_matrix_half(f)
    Z = rob.body_regressor_by_vector_velocity_matrix(f)

    nv = 6
    ng = 3
    nz = Z.shape[0]
    nθ = 10

    ny = nv + ng + nθ + nz

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # velocity and acceleration constraints
    A_min = -np.ones(nv)
    A_max = np.ones(nv)
    V_max = 0.5 * np.array([1, 1, 1, 1, 1, 1])
    V_min = -V_max
    Λ_max = np.outer(V_max, V_max)

    v_max = 0.5
    ω_max = 0.5

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    θ = cp.Variable(nθ)
    V = cp.Variable(nv)

    y = cp.hstack((A, G, z, θ))
    X = cp.Variable((ny, ny), PSD=True)

    Λ = cp.Variable((6, 6), PSD=True)

    # Xv = X[:nv, :nv]  # acceleration block
    Xa = X[:nv, :nv]  # acceleration block
    Xg = X[nv : nv + ng, nv : nv + ng]  # gravity block
    Xz = X[nv + ng : nv + ng + nz, nv + ng : nv + ng + nz]  # z block
    Xθ = X[-nθ:, -nθ:]  # θ block

    # notice that we don't need to include V in the optimization problem at all
    objective = cp.Maximize(0.5 * cp.trace(Q @ X))
    constraints = [
        z == cp.vec(Λ),
        # z == vech(Λ),
        # z == extract_z(Λ),

        # we constrain z completely through Λ
        # Λ <= Λ_max,
        cp.trace(Λ[:3, :3]) == v_max**2,  # NOTE this constraint is tight
        cp.trace(Λ[3:, 3:]) <= ω_max**2,

        schur(Λ, V) >> 0,
        # cp.diag(Λ) <= np.diag(Λ_max),
        # constraints on X
        schur(X, y) >> 0,
        cp.trace(Xa[:3, :3]) == 1,
        cp.trace(Xa[3:, 3:]) == 1,
        # cp.diag(Xa) <= np.maximum(A_max**2, A_min**2),
        cp.diag(Xθ) <= np.maximum(obj.θ_max**2, obj.θ_min**2),
        # Xa <= np.maximum(np.outer(A_max,A_max), np.outer(A_min, A_min)),
        # Xθ <= np.maximum(np.outer(obj.θ_max, obj.θ_max), np.outer(obj.θ_min, obj.θ_min)),
        cp.trace(Xg) == g**2,
        Xg[2, 2] >= (g * np.cos(max_tilt_angle)) ** 2,
        # consistency between Xz and Λ
        # Xz <= cp.kron(Λ_max, Λ),
        # Xz <= vech_np(Λ_max) @ vech(Λ).T,
        # Xz <= extract_z_np(Λ_max) @ extract_z(Λ).T,
        # Xv == Λ,
        cp.trace(Xz) == (v_max**2 + ω_max**2) * cp.trace(Λ),

        # X[nv + ng : nv + ng + nz, :] == 0,
        # X[:, nv + ng : nv + ng + nz] == 0,

        # TODO can we make this tighter by depending directly on Λ?
        # note that this is still conservative regardless because we are not
        # reasoning directly about the norm inequalities we have

        # cp.diag(Xz) <= extract_z_np(Λ_max)**2,
        # cp.trace(Xz) <= np.linalg.norm(extract_z_np(Λ_max))**2,
        # cp.trace(Xz) <= np.linalg.norm(extract_z_np(Λ_max)) * cp.norm1(extract_z(Λ)),
        # cp.trace(Xz) <= extract_z_np(Λ_max) @ extract_z(Λ),

        # TODO we can also include physical realizability on J
        # constraints on y
        cp.norm(G) <= g,
        z_normal @ G <= -g * np.cos(max_tilt_angle),
        # A >= A_min,
        # A <= A_max,
        cp.norm(A[:3]) <= 1,
        cp.norm(A[3:]) <= 1,
        θ >= obj.θ_min,
        θ <= obj.θ_max,
        cp.norm(V[:3]) <= v_max,
        cp.norm(V[3:]) <= ω_max,
        # V >= V_min,
        # V <= V_max,
    ]

    for i in other_constr_idx:
        fi = F[i, :]
        Qi = compute_Q_matrix(fi)
        constraints.append(0.5 * cp.trace(Qi @ X) <= 0)

    # off-diagonal blocks of z @ z.T are symmetric, which helps tighten the
    # relaxation
    for i in range(0, 5):
        for j in range(i + 1, 5):
            Xz_ij = Xz[i * 6 : (i + 1) * 6, j * 6 : (j + 1) * 6]

            constraints.append(Xz_ij == Xz_ij.T)
            constraints.append(cp.trace(Xz_ij[:3, :3]) == Λ[i, j] * v_max**2)
            constraints.append(cp.trace(Xz_ij[3:, 3:]) == Λ[i, j] * ω_max**2)
            # constraints.append(Xz_ij >> V_max[i] * V_min[j] * Λ)

    for i in range(6):
        Xz_ii = Xz[i * 6 : (i + 1) * 6, i * 6 : (i + 1) * 6]
        constraints.append(cp.trace(Xz_ii[:3, :3]) == Λ[i, i] * v_max**2)
        constraints.append(cp.trace(Xz_ii[3:, 3:]) == Λ[i, i] * ω_max**2)

    problem = cp.Problem(objective, constraints)
    # problem.solve(solver=cp.MOSEK, verbose=True)
    try:
        problem.solve(solver=cp.MOSEK)
    except cp.error.SolverError:
        print("failed to solve relaxed problem")
    print(problem.status)
    # print(np.linalg.eigvals(X.value))

    IPython.embed()
    return problem.value


def solve_global_relaxed_dual(obj, F, idx, other_constr_idx, v_max=1, ω_max=1):
    """Global convex problem based on the face form of the dual constraint formulation."""
    f = F[idx, :]

    Z = rob.body_regressor_by_vector_velocity_matrix(f)
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]

    nv = 6
    ng = 3
    nz = Z.shape[0]

    # fmt: off
    P_tilde = np.block([
        [obj.P, obj.p[:, None]],
        [np.zeros((1, obj.P.shape[1])), np.array([[-1]])]])
    # fmt: on
    R = rob.span_to_face_form(P_tilde.T)[0]
    R = R / np.max(np.abs(R))

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # velocity and acceleration constraints
    a_max = 1
    α_max = 1

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    Λ = cp.Variable((6, 6), PSD=True)

    values = []
    Vs = []

    for i in range(R.shape[0]):
        objective = cp.Minimize(R[i, :-1] @ (Z.T @ z + Dg.T @ G + D.T @ A))
        constraints = [
            z == cp.vec(Λ),

            # we constrain z completely through Λ
            cp.trace(Λ[:3, :3]) <= v_max**2,
            cp.trace(Λ[3:, 3:]) <= ω_max**2,

            # gravity constraints
            cp.norm(G) <= g,
            z_normal @ G <= -g * np.cos(max_tilt_angle),

            # acceleration constraints
            cp.norm(A[:3]) <= 1,
            cp.norm(A[3:]) <= 1,
        ]
        problem = cp.Problem(objective, constraints)
        t0 = time.time()
        problem.solve(solver=cp.MOSEK)
        t1 = time.time()
        # print(f"Δt = {t1 - t0}")
        values.append(problem.value)

        e, v = np.linalg.eig(Λ.value)
        Vs.append(np.sqrt(e[0]) * v[:, 0])

    min_idx = np.argmin(values)
    print(f"relaxed value = {values[min_idx]}")
    print(f"relaxed V = {Vs[min_idx]}")
    return np.min(values)


def solve_local(obj, F, idx, other_constr_idx, v_max=1, ω_max=1):

    # A, g, V, θ
    A0 = np.zeros(6)
    G0 = np.array([0, 0, -9.81])
    V0 = np.array([v_max, 0, 0, ω_max, 0, 0])
    θ0 = obj.θ
    x0 = np.concatenate((A0, G0, V0, θ0))

    # limits
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # velocity and acceleration constraints
    A_min = -np.ones(6)
    A_max = np.ones(6)
    V_min = -np.ones(6)
    V_max = np.ones(6)

    f = F[idx, :]

    def cost(x):
        A, G, V, θ = x[:6], x[6:9], x[9:15], x[-10:]
        Ag = np.concatenate((G, np.zeros(3)))
        Y = rob.body_regressor(V, A - Ag)
        return -f @ Y @ θ

    def eq_cons(x):
        G = x[6:9]
        return np.linalg.norm(G) - g

    def compute_extra_inequalities(x):
        A, G, V, θ = x[:6], x[6:9], x[9:15], x[-10:]
        Ag = np.concatenate((G, np.zeros(3)))
        Y = rob.body_regressor(V, A - Ag)
        other_cons = []
        for i in other_constr_idx:
            other_cons.append(-F[i, :] @ Y @ θ)
        return other_cons

    def ineq_cons(x):
        A, G, V, θ = x[:6], x[6:9], x[9:15], x[-10:]
        return np.concatenate(
            (
                # A_max - A,
                # A - A_min,
                # V_max - V,
                # V - V_min,
                [1 - np.linalg.norm(A[:3]), 1 - np.linalg.norm(A[3:])],
                [v_max - np.linalg.norm(V[:3]), ω_max - np.linalg.norm(V[3:])],
                obj.θ_max - θ,
                θ - obj.θ_min,
                [-g * np.cos(max_tilt_angle) - G[2]],
                compute_extra_inequalities(x),
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
        # tol=1e-3,
        # options={"maxiter": 1000},
    )
    A, G, V, θ = res.x[:6], res.x[6:9], res.x[9:15], res.x[-10:]
    value = -cost(res.x)
    print(f"local solved = {res.success}")
    print(f"local obj = {value}")
    # print(f"||v|| = {np.linalg.norm(V[:3])}, ||ω|| = {np.linalg.norm(V[3:])}")
    print(f"local V = {V}")
    if value <= 0:
        print("^^^^ NEGATIVE")
    if not res.success:
        IPython.embed()
    # IPython.embed()
    return value


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
        [np.zeros((1, obj.P.shape[1])), np.array([[-1]])]])
    # fmt: on
    R = rob.span_to_face_form(P_tilde.T)[0]
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

            # NOTE bit of a hack by settings C == 0
            d, D = rob.body_regressor_by_vector_matrix(np.zeros((3, 3)), V, f)

            return R[i, :-1] @ (d + D @ (A - Ag))

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
                    [a_max**2 - np.linalg.norm(A[:3])**2, α_max**2 - np.linalg.norm(A[3:])**2],
                    [v_max**2 - np.linalg.norm(V[:3])**2, ω_max**2 - np.linalg.norm(V[3:])**2],
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


def main():
    np.set_printoptions(precision=5, suppress=True)

    obj = rob.BalancedObject(m=1, h=0.1, δ=0.05, μ=0.5, h0=0, x0=0)
    F = rob.cwc(obj.contacts())
    F = F / np.max(np.abs(F))

    for i in range(F.shape[0]):
        print(i)
        # solve_global_relaxed(obj, F, i, list(range(i)))
        # solve_local(obj, F, i, list(range(i)))

        # relaxed = solve_global_relaxed(obj, F, i, [])
        relaxed = solve_global_relaxed_dual(obj, F, i, [])
        local = solve_local_dual(obj, F, i, [])
        if local >= 0:
            print("CONSTRAINT NEVER ACTIVE!")
        print(f"gap = {relaxed - local}\n")
        # return


if __name__ == "__main__":
    main()
