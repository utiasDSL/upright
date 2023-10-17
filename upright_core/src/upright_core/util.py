import numpy as np
from upright_core import math


def sort_canonical(A):
    """Helper to sort an nd-array into a canonical order, column by column."""
    B = np.copy(A)
    for i in range(len(B.shape)):
        B.sort(axis=-i - 1)
    return B


def allclose_unordered(A, B):
    """Helper to compare two nd-arrays where each array should have the same
    rows, but they may be in different orders.

    Returns True if the arrays are the same (but possibly with rows in a
    different order), False otherwise.
    """
    assert A.shape == B.shape
    n = A.shape[0]
    B_checked = np.zeros(n, dtype=bool)
    for i in range(n):
        a = A[i, :]
        residuals = np.linalg.norm(B - a, axis=1)

        # False where residual = 0, True otherwise
        mask = ~np.isclose(residuals, 0)

        # False where residual = 0 AND B has not been checked yet
        test = np.logical_or(mask, B_checked)

        # check to see if we have any cases where the test passes
        idx = np.argmin(test)
        if not test[idx]:
            B_checked[idx] = True
        else:
            return False
    return True


def support_area_distance(ctrl_object, Q_we):
    """Compute distance outside of SA at current EE orientation Q_we."""
    C_we = math.quat_to_rot(Q_we)
    normal = ctrl_object.support_area.normal()

    # position of CoM relative to center of SA
    r_com_e = ctrl_object.com_height * normal
    r_com_w = C_we @ r_com_e

    # solve for the intersection point of r_com_w with the SA (in the
    # SA frame) knowing that:
    # * intersection point in world frame has same (x, y) as CoM
    # * intersection point in EE frame dotted with normal = 0
    A = np.eye(3)
    A[:2, :] = C_we[:2, :]
    A[2, :] = normal
    b = np.zeros(3)
    b[:2] = r_com_w[:2]
    c = np.linalg.solve(A, b)

    d = ctrl_object.support_area.distance(c)
    return d
