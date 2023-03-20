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


def main():
    np.set_printoptions(precision=5, suppress=True)

    obj = rob.BalancedObject(m=1, h=0.1, δ=0.05, μ=0.2, h0=0, x0=0)
    F = -rob.cwc(obj.contacts())

    f = F[0, :]

    D = rob.body_regressor_by_vector_acceleration_matrix(f)
    Dg = D[:3, :]
    Z = rob.body_regressor_by_vector_velocity_matrix(f)

    nv = 6
    ng = 3
    nz = Z.shape[0]
    nθ = 10

    ny = nv + ng + nz + nθ

    Sz = np.hstack((np.zeros((nz, nv + ng)), np.eye(nz), np.zeros((nz, nθ))))

    # y = (A, G, z, θ)
    # fmt: off
    Q = np.block([
        [np.zeros((nv, nv + ng + nz)), D],
        [np.zeros((ng, nv + ng + nz)), Dg],
        [np.zeros((nz, nv + ng + nz)), Z],
        [D.T, Dg.T, Z.T, np.zeros((nθ, nθ))]])
    # fmt: on

    # gravity constraints
    z_normal = np.array([0, 0, 1])
    max_tilt_angle = np.deg2rad(30)

    # velocity and acceleration constraints
    A_min = -np.ones(nv)
    A_max = np.ones(nv)
    V_min = -np.ones(nv)
    V_max = np.ones(nv)

    A = cp.Variable(nv)
    G = cp.Variable(ng)
    z = cp.Variable(nz)
    θ = cp.Variable(nθ)

    y = cp.hstack((A, G, z, θ))
    X = cp.Variable((ny, ny), PSD=True)

    V = cp.Variable(6)
    Λ = cp.Variable((6, 6), PSD=True)

    objective = cp.Maximize(0.5 * cp.trace(Q @ X))
    constraints = [
        Sz @ y == cp.vec(Λ),
        schur(X, y) >> 0,
        schur(Λ, V) >> 0,
        Λ == np.zeros((6, 6)),
        z == np.zeros(36),
        cp.norm(G) <= g,
        z_normal @ G <= -g * np.cos(max_tilt_angle),
        V >= V_min,
        V <= V_max,
        A >= A_min,
        A <= A_max,
        obj.P @ θ >= obj.p,
        # X << 101 * np.eye(X.shape[0]),
        # Λ >> 0,
    ]
    problem = cp.Problem(objective, constraints)
    value = problem.solve()
    print(problem.status)

    IPython.embed()


if __name__ == "__main__":
    main()
