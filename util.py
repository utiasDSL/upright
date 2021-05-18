import numpy as np
import pybullet as pyb
import jax.numpy as jnp
from jaxlie import SO3
from scipy.linalg import expm


def debug_frame(size, obj_uid, link_index):
    """Attach at a frame to a link for debugging purposes."""
    pyb.addUserDebugLine(
        [0, 0, 0],
        [size, 0, 0],
        lineColorRGB=[1, 0, 0],
        parentObjectUniqueId=obj_uid,
        parentLinkIndex=link_index,
    )
    pyb.addUserDebugLine(
        [0, 0, 0],
        [0, size, 0],
        lineColorRGB=[0, 1, 0],
        parentObjectUniqueId=obj_uid,
        parentLinkIndex=link_index,
    )
    pyb.addUserDebugLine(
        [0, 0, 0],
        [0, 0, size],
        lineColorRGB=[0, 0, 1],
        parentObjectUniqueId=obj_uid,
        parentLinkIndex=link_index,
    )


def rot2d(θ, np=np):
    """2D rotation matrix: rotates points counter-clockwise."""
    c = np.cos(θ)
    s = np.sin(θ)
    return np.array([[c, -s],
                     [s,  c]])


def pose_to_pos_quat(P):
    return P[:3], P[3:]


def pose_from_pos_quat(r, Q):
    return jnp.concatenate((r, Q))


def pose_error(Pd, P):
    rd, Qd = pose_to_pos_quat(Pd)
    Cd = SO3.from_quaternion_xyzw(Qd)

    r, Q = pose_to_pos_quat(P)
    C = SO3.from_quaternion_xyzw(Q)

    r_err = rd - r
    Q_err = Cd.multiply(C.inverse()).as_quaternion_xyzw()
    return jnp.concatenate((r_err, Q_err[:3]))  # exclude w term


def pitch_from_quat(Q):
    return SO3.from_quaternion_xyzw(Q).compute_pitch_radians()


def skew1(x):
    """2D skew-symmetric operator."""
    return np.array([[0, -x], [x, 0]])


def skew3(x, np=jnp):
    """3D skew-symmetric operator."""
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], ]])


def dhtf(q, a, d, α, np=jnp):
    """Constuct a transformation matrix from D-H parameters."""
    cα = np.cos(α)
    sα = np.sin(α)
    cq = np.cos(q)
    sq = np.sin(q)
    return np.array([
        [cq, -sq*cα,  sq*sα, a*cq],
        [sq,  cq*cα, -cq*sα, a*sq],
        [0,      sα,     cα,    d],
        [0,       0,      0,   1]])


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
    Bd = Hd[:rb, ca:ca+cb]

    return Ad, Bd