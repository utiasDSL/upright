"""Tests related to the body regressor Y, which maps inertial parameters to the
body frame wrench for a rigid body."""
import pytest
import numpy as np
import pinocchio

import upright_robust as rob


def test_regressor():
    """Test body regressor implementation."""
    np.random.seed(0)

    V = np.random.random(6)
    A = np.random.random(6)

    Y = rob.body_regressor(V, A)

    # compare to pinocchio's implementation
    # pinocchio orders the inertia matrix parameters with I_xz and I_yy swapped
    # compared to our implementation, so we have to manually correct that
    Z = pinocchio.bodyRegressor(pinocchio.Motion(V), pinocchio.Motion(A))
    Z_swapped = Z.copy()
    Z_swapped[:, 6] = Z[:, 7]
    Z_swapped[:, 7] = Z[:, 6]

    assert np.allclose(Y, Z_swapped)


def test_regressor_decomposition():
    """Test the decomposition Y.T @ f == D @ A + d"""
    np.random.seed(0)

    C_ew = np.eye(3)
    G = rob.body_gravity6(C_ew)

    V = np.random.random(6)
    A = np.random.random(6)
    f = np.random.random(6)

    Y = rob.body_regressor(V, A - G)
    D = rob.body_regressor_A_by_vector(f)
    d = rob.body_regressor_VG_by_vector(V, G, f)

    assert np.allclose(Y.T @ f, D @ A + d)


def test_regressor_decomposition_multi():
    """Test the decomposition for multiple objects"""
    np.random.seed(0)

    C_ew = np.eye(3)
    G = rob.body_gravity6(C_ew)

    V = np.random.random(6)
    A = np.random.random(6)

    f1 = np.random.random(6)
    f2 = np.random.random(6)
    f3 = np.random.random(6)
    f = np.concatenate((f1, f2, f3))

    Y = rob.body_regressor(V, A - G)
    D = rob.body_regressor_A_by_vector(f)
    d = rob.body_regressor_VG_by_vector(V, G, f)

    assert np.allclose(np.concatenate((Y.T @ f1, Y.T @ f2, Y.T @ f3)), D @ A + d)


def test_regressor_VG_vectorized():
    """Test the vectorized velocity-gravity part of the decomposition (i.e.
    compute the decomposition vector d for multiple vectors f at once)."""
    np.random.seed(0)

    C_ew = np.eye(3)
    G = rob.body_gravity6(C_ew)

    n = 3
    V = np.random.random(6)
    A = np.random.random(6)
    F = np.random.random((n, 6))

    M = rob.body_regressor_VG_by_vector_vectorized(V, G, F)
    M_tilde = rob.body_regressor_VG_by_vector_tilde_vectorized(V, G, F)

    for i in range(n):
        d = rob.body_regressor_VG_by_vector(V, G, F[i, :])
        assert np.allclose(M[:, i], d)
        assert np.allclose(M_tilde[:, i], np.append(d, 0))
