import numpy as np
import cvxpy as cp

from .utils import (
    body_regressor_by_vector_acceleration_matrix,
    body_regressor_by_vector_velocity_matrix,
)


def compute_Q_matrix(f):
    """Compute Q matrix of the primal SDP relaxation, such that the objective
    is ``minimize 0.5 * tr(Q @ X)``."""
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
