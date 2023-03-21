import numpy as np
import cvxpy as cp
import time

import upright_robust as rob

import IPython


# gravity
g = 9.81
# G = np.array([0, 0, -g])


def outer(x):
    y = cp.reshape(x, (x.shape[0], 1))
    return y @ y.T


def schur(X, x):
    y = cp.reshape(x, (x.shape[0], 1))
    return cp.bmat([[X, y], [y.T, [[1]]]])



def solve_problem(obj, f):
    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = D[:3, :]
    Z = rob.body_regressor_by_vector_velocity_matrix(f)

    nv = 6
    ng = 3
    nz = Z.shape[0]
    nθ = 10

    ny = nv + ng + nθ + nz

    # Sz = np.hstack((np.zeros((nz, nv + ng)), np.eye(nz), np.zeros((nz, nθ))))

    # y = (A, G, z, θ)
    # fmt: off
    Q = np.block([
        [np.zeros((nv, nv + ng + nz)), D],
        [np.zeros((ng, nv + ng +  nz)), Dg],
        [np.zeros((nz, nv + ng +  nz)), Z],
        [D.T, Dg.T, Z.T, np.zeros((nθ, nθ))]])
    # fmt: on

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # velocity and acceleration constraints
    A_min = -np.ones(nv)
    A_max = 0.5 * np.ones(nv)
    V_min = -np.ones(nv)
    V_max = np.ones(nv)
    Λ_max = np.outer(V_max, V_max)

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    θ = cp.Variable(nθ)

    y = cp.hstack((A, G, z, θ))
    X = cp.Variable((ny, ny), PSD=True)

    Λ = cp.Variable((6, 6), PSD=True)

    Xa = X[:nv, :nv]  # acceleration block
    Xg = X[nv:nv+3,nv:nv+3]  # gravity block
    Xz = X[nv+ng:nv+ng+nz, nv+ng:nv+ng+nz]  # z block
    Xθ = X[-nθ:, -nθ:]  # θ block

    # notice that we don't need to include V in the optimization problem at all
    objective = cp.Maximize(0.5 * cp.trace(Q @ X))
    constraints = [
        z == cp.vec(Λ),

        # we constrain z completely through Λ
        Λ <= Λ_max,

        # constraints on X
        schur(X, y) >> 0,
        cp.diag(Xa) <= A_max**2,
        cp.diag(Xθ) <= obj.θ_max**2,
        cp.trace(Xg) == g**2,
        Xg[2, 2] >= (g * np.cos(max_tilt_angle))**2,

        # consistency between Xz and Λ
        Xz << cp.kron(Λ_max, Λ),

        # TODO we can also include physical realizability on J

        # constraints on y
        cp.norm(G) <= g,
        z_normal @ G <= -g * np.cos(max_tilt_angle),
        A >= A_min,
        A <= A_max,
        θ >= obj.θ_min,
        θ <= obj.θ_max,
    ]

    problem = cp.Problem(objective, constraints)
    # value = problem.solve(solver=cp.SCS, verbose=True, max_iters=int(1e6))
    problem.solve(solver=cp.MOSEK, verbose=True)
    print(problem.status)
    print(problem.value)

    # TODO this is an undervalue of the true objective that contains V directly: f @ Y(C, V, A) @ θ
    print(0.5 * y.value @ Q @ y.value)

    IPython.embed()


def main():
    np.set_printoptions(precision=5, suppress=True)

    obj = rob.BalancedObject(m=1, h=0.1, δ=0.05, μ=0.5, h0=0, x0=0)
    F = -rob.cwc(obj.contacts())

    for i in range(F.shape[0]):
        solve_problem(obj, F[i, :])
        return



if __name__ == "__main__":
    main()
