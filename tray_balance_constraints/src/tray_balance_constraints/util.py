import numpy as np
import liegroups


def quaternion_to_matrix(Q, normalize=True):
    """Convert quaternion to rotation matrix."""
    if normalize:
        if np.allclose(Q, 0):
            Q = np.array([0, 0, 0, 1])
        else:
            Q = Q / np.linalg.norm(Q)
    try:
        return liegroups.SO3.from_quaternion(Q, ordering="xyzw").as_matrix()
    except ValueError as e:
        IPython.embed()


def transform_point(r_ba_a, Q_ab, r_cb_b):
    """Transform point r_cb_b to r_ca_a.

    This is equivalent to r_ca_a = T_ab @ r_cb_b, where T_ab is the homogeneous
    transformation matrix from A to B (and I've abused notation for homogeneous
    vs. non-homogeneous points).
    """
    C_ab = quaternion_to_matrix(Q_ab)
    return r_ba_a + C_ab @ r_cb_b


def rotate_point(Q, r):
    """Rotate a point r using quaternion Q."""
    return transform_point(np.zeros(3), Q, r)


def support_area_distance(ctrl_object, Q_we):
    """Compute distance outside of SA at current EE orientation Q_we."""
    C_we = quaternion_to_matrix(Q_we)

    # position of CoM relative to center of SA
    r_com_o = np.array([0, 0, ctrl_object.com_height])
    r_com_w = C_we @ r_com_o

    # solve for the intersection point of r_com_w with the SA (in the
    # SA frame) knowing that:
    # * intersection point in world frame has same (x, y) as CoM
    # * intersection point in object frame has z = 0
    A = np.eye(3)
    A[:2, :] = C_we[:2, :]
    b = np.zeros(3)
    b[:2] = r_com_w[:2]
    c = np.linalg.solve(A, b)

    d = ctrl_object.support_area_min.distance_outside(c[:2])
    return d
