import numpy as np
from upright_core import math


def sort_canonical(A):
    """Helper to sort an nd-array into a canonical order, column by column."""
    B = np.copy(A)
    for i in range(len(B.shape)):
        B.sort(axis=-i - 1)
    return B


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
