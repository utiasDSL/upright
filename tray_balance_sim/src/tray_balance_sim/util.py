import pybullet as pyb
import numpy as np
import liegroups
from scipy.linalg import expm

import IPython


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
    """Transform point r_cb_b to r_ca_a."""
    C_ab = quaternion_to_matrix(Q_ab)
    return r_ba_a + C_ab @ r_cb_b


def rotate_point(q, r):
    """Rotate a point r using quaternion q."""
    return transform_point(np.zeros(3), q, r)


def dhtf(q, a, d, α):
    """Constuct a transformation matrix from D-H parameters."""
    cα = np.cos(α)
    sα = np.sin(α)
    cq = np.cos(q)
    sq = np.sin(q)
    return np.array(
        [
            [cq, -sq * cα, sq * sα, a * cq],
            [sq, cq * cα, -cq * sα, a * sq],
            [0, sα, cα, d],
            [0, 0, 0, 1],
        ]
    )


def zoh(A, B, dt):
    """Compute discretized system matrices assuming zero-order hold on input."""
    ra, ca = A.shape
    rb, cb = B.shape

    assert ra == ca  # A is square
    assert ra == rb  # B has same number of rows as A

    ch = ca + cb
    rh = ch

    H = np.block([[A, B], [np.zeros((rh - ra, ch))]])
    Hd = expm(dt * H)
    Ad = Hd[:ra, :ca]
    Bd = Hd[:rb, ca : ca + cb]

    return Ad, Bd


def calc_r_te_e(r_ew_w, Q_we, r_tw_w):
    """Calculate position of tray relative to the EE."""
    # C_{ew} @ (r^{tw}_w - r^{ew}_w)
    r_te_w = r_tw_w - r_ew_w
    C_ew = quaternion_to_matrix(Q_we).T
    return C_ew @ r_te_w


def calc_Q_et(Q_we, Q_wt):
    """Calculate orientation of tray relative to the EE."""
    SO3_we = liegroups.SO3.from_quaternion(Q_we, ordering="xyzw")
    SO3_wt = liegroups.SO3.from_quaternion(Q_wt, ordering="xyzw")
    # SO3_we = SO3.from_quaternion_xyzw(Q_we)
    # SO3_wt = SO3.from_quaternion_xyzw(Q_wt)
    return SO3_we.inv().dot(SO3_wt).to_quaternion(ordering="xyzw")


def draw_curve(waypoints, rgb=(1, 0, 0), dist=0.05, linewidth=1, dashed=False):
    # process waypoints to space them (roughly) evenly
    visual_points = [waypoints[0, :]]
    for i in range(1, len(waypoints)):
        d = np.linalg.norm(waypoints[i, :] - visual_points[-1])
        if d >= dist:
            visual_points.append(waypoints[i, :])

    step = 2 if dashed else 1
    for i in range(0, len(visual_points) - 1, step):
        start = visual_points[i]
        end = visual_points[i + 1]
        pyb.addUserDebugLine(
            list(start),
            list(end),
            lineColorRGB=rgb,
            lineWidth=linewidth,
        )
