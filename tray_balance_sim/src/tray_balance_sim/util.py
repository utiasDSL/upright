import pybullet as pyb
import numpy as np
import liegroups
from scipy.linalg import expm
import yaml

import tray_balance_constraints as core

import IPython


# This is from <https://github.com/Maples7/dict-recursive-update/blob/07204cdab891ac4123b19fe3fa148c3dd1c93992/dict_recursive_update/__init__.py>
def recursive_dict_update(default, custom):
    """Return a dict merged from default and custom"""
    if not isinstance(default, dict) or not isinstance(custom, dict):
        raise TypeError("Params of recursive_update should be dicts")

    for key in custom:
        if isinstance(custom[key], dict) and isinstance(default.get(key), dict):
            default[key] = recursive_dict_update(default[key], custom[key])
        else:
            default[key] = custom[key]

    return default


def load_config(path, depth=0, max_depth=5):
    """Load configuration file located at `path`.

    `depth` and `max_depth` arguments are provided to protect against
    unexpectedly deep or infinite recursion through included files.
    """
    if depth > max_depth:
        raise Exception(f"Maximum inclusion depth {max_depth} exceeded.")

    with open(path) as f:
        d = yaml.safe_load(f)

    # get the includes while also removing them from the dict
    includes = d.pop("include", [])

    # construct a dict of everything included
    includes_dict = {}
    for include in includes:
        path = core.parsing.parse_ros_path(include)
        include_dict = load_config(path, depth=depth + 1)

        # nest the include under `key` if specified
        if "key" in include:
            include_dict = {include["key"]: include_dict}

        # update the includes dict and reassign
        includes_dict = recursive_dict_update(includes_dict, include_dict)

    # now add in the info from this file
    d = recursive_dict_update(includes_dict, d)
    return d


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
    """Draw debug lines along a curve represented by waypoints in PyBullet."""
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
