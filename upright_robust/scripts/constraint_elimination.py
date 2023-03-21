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
    # Z = rob.body_regressor_by_vector_velocity_matrix(f)

    nv = 6
    ng = 3
    # nz = Z.shape[0]
    nθ = 10

    ny = nv + ng + nθ

    # Sz = np.hstack((np.zeros((nz, nv + ng)), np.eye(nz), np.zeros((nz, nθ))))

    # y = (A, G, z, θ)
    # fmt: off
    Q = np.block([
        [np.zeros((nv, nv + ng)), D],
        [np.zeros((ng, nv + ng)), Dg],
        [D.T, Dg.T, np.zeros((nθ, nθ))]])
    # fmt: on

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # velocity and acceleration constraints
    A_min = -np.ones(nv)
    A_max = np.ones(nv)
    # V_min = -np.ones(nv)
    # V_max = np.ones(nv)

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    # z = cp.Variable(nz)
    θ = cp.Variable(nθ)

    y = cp.hstack((A, G, θ))
    X = cp.Variable((ny, ny), PSD=True)

    # V = cp.Variable(6)
    # Λ = cp.Variable((6, 6), PSD=True)
    x_max = np.concatenate((A_max, [0, 0, -9.81], obj.θ_max))
    x_min = np.concatenate((A_min, [0, 0, -9.81], obj.θ_min))

    Xg = X[nv:nv+3,nv:nv+3]

    objective = cp.Maximize(0.5 * cp.trace(Q @ X))
    constraints = [
        # Sz @ y == cp.vec(Λ),
        schur(X, y) >> 0,
        # X << np.outer(x_max, x_max),
        # X << x_max @ x_max * np.eye(X.shape[0]),
        # cp.trace(X) <= A_max @ A_max + g**2 + obj.θ_max @ obj.θ_max,  # NOTE w/o this the problem is unbounded
        # cp.diag(X) <= x_max**2,
        # cp.diag(X) >= x_min**2,
        # schur(Λ, V) >> 0,
        # Λ == np.zeros((6, 6)),
        # z == np.zeros(36),

        cp.diag(X[:nv, :nv]) <= A_max**2,
        cp.diag(X[-nθ:, -nθ:]) <= obj.θ_max**2,
        cp.trace(Xg) == g**2,
        Xg[2, 2] >= (g * np.cos(max_tilt_angle))**2,

        cp.norm(G) <= g,
        z_normal @ G <= -g * np.cos(max_tilt_angle),
        # G == np.array([0, 0, -9.81]),
        # y >= x_min,
        # y <= x_max,

        # V >= V_min,
        # V <= V_max,
        A >= A_min,
        A <= A_max,
        θ >= obj.θ_min,
        θ <= obj.θ_max,
        # obj.P @ θ >= obj.p,
        # X << 101 * np.eye(X.shape[0]),
        # Λ >> 0,
    ]
    problem = cp.Problem(objective, constraints)
    # value = problem.solve(solver=cp.SCS, verbose=True, max_iters=int(1e6))
    problem.solve(solver=cp.MOSEK)
    print(problem.status)
    print(problem.value)
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
