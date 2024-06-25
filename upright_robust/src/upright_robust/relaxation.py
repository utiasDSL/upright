import numpy as np
import cvxpy as cp

from .utils import (
    body_regressor_by_vector_acceleration_matrix,
    body_regressor_by_vector_velocity_matrix,
)


# TODO can I use the implementation from rigeo?
def schur(X, x):
    y = cp.reshape(x, (x.shape[0], 1))
    return cp.bmat([[X, y], [y.T, [[1]]]])


def compute_Q_matrix(f):
    D = body_regressor_by_vector_acceleration_matrix(f)
    Dg = -D[:3, :]
    Z = body_regressor_by_vector_velocity_matrix(f)

    nv = 6
    ng = 3
    nθ = 10
    nz = Z.shape[0]

    Q = np.block(
        [
            [np.zeros((nv, nv + ng + nz)), D],
            [np.zeros((ng, nv + ng + nz)), Dg],
            [np.zeros((nz, nv + ng + nz)), Z],
            [D.T, Dg.T, Z.T, np.zeros((nθ, nθ))],
        ]
    )
    return Q
