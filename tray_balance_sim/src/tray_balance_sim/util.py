import numpy as np
import jax.numpy as jnp
from jaxlie import SO3
import liegroups
from scipy.linalg import expm


# TODO these quaternion etc. transforms can be handled from elegantly (i.e.,
# extracted away better)


def quaternion_to_matrix(Q):
    """Convert quaternion to rotation matrix."""
    return liegroups.SO3.from_quaternion(Q, ordering="xyzw").as_matrix()


def transform_point(r_ba_a, Q_ab, r_cb_b):
    """Transform point r_cb_b to r_ca_a."""
    C_ab = quaternion_to_matrix(Q_ab)
    return r_ba_a + C_ab @ r_cb_b


def rot2d(θ, np=np):
    """2D rotation matrix: rotates points counter-clockwise."""
    c = np.cos(θ)
    s = np.sin(θ)
    return np.array([[c, -s], [s, c]])


def quat_error(q):
    xyz = q[:3]
    w = q[3]
    # this is just the angle part of an axis-angle
    return 2 * np.arctan2(np.linalg.norm(xyz), w)


def quat_multiply(q0, q1, normalize=True):
    """Hamilton product of two quaternions."""
    if normalize:
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
    C0 = liegroups.SO3.from_quaternion(q0, ordering="xyzw")
    C1 = liegroups.SO3.from_quaternion(q1, ordering="xyzw")
    return C0.dot(C1).to_quaternion(ordering="xyzw")


def skew1(x):
    """2D skew-symmetric operator."""
    return np.array([[0, -x], [x, 0]])


def skew3(x, np=jnp):
    """3D skew-symmetric operator."""
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def dhtf(q, a, d, α, np=jnp):
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
    return SO3.from_quaternion_xyzw(Q_we).inverse() @ r_te_w


def calc_Q_et(Q_we, Q_wt):
    """Calculate orientation of tray relative to the EE."""
    SO3_we = SO3.from_quaternion_xyzw(Q_we)
    SO3_wt = SO3.from_quaternion_xyzw(Q_wt)
    return SO3_we.inverse().multiply(SO3_wt).as_quaternion_xyzw()


def quat_inverse(Q):
    return np.append(-Q[:3], Q[3])
