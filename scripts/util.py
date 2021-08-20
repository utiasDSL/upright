import numpy as np
import pybullet as pyb
import jax
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
    return np.array([[c, -s], [s, c]])


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


def state_error(xd, x):
    Pd, Vd = xd[:7], xd[7:]
    P, V = x[:7], x[7:]
    return jnp.concatenate((pose_error(Pd, P), Vd - V))


def pitch_from_quat(Q):
    """Get pitch from a quaternion."""
    return SO3.from_quaternion_xyzw(Q).compute_pitch_radians()


# def quat_from_axis_angle(a, np=np):
#     """Compute quaternion from an axis-angle."""
#     # NOTE: this is not written for jax: use jaxlie instead
#     angle = np.linalg.norm(a)
#     if np.isclose(angle, 0):
#         return np.array([0, 0, 0, 1])
#     axis = a / angle
#     c = np.cos(angle / 2)
#     s = np.sin(angle / 2)
#     return np.append(axis * s, c)


def quat_multiply(q0, q1):
    """Hamilton product of two quaternions."""
    R0 = SO3.from_quaternion_xyzw(q0)
    R1 = SO3.from_quaternion_xyzw(q1)
    return R0.multiply(R1).as_quaternion_xyzw()


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


def equilateral_triangle_inscribed_radius(side_length):
    """Compute radius of the inscribed circle of equilateral triangle."""
    return side_length / (2 * np.sqrt(3))


def cylinder_inertia_matrix(mass, radius, height):
    """Inertia matrix for cylinder aligned along z-axis."""
    xx = yy = mass * (3 * radius ** 2 + height ** 2) / 12
    zz = 0.5 * mass * radius ** 2
    return np.diag([xx, yy, zz])


def cuboid_inertia_matrix(mass, side_lengths):
    """Inertia matrix for a rectangular cuboid with side_lengths in (x, y, z)
    dimensions."""
    lx, ly, lz = side_lengths
    return (
        mass * np.diag([ly ** 2 + lz ** 2, lx ** 2 + lz ** 2, lx ** 2 + ly ** 2]) / 12.0
    )


class Body:
    """Rigid body parameters."""

    def __init__(self, mass, inertia, com):
        self.mass = mass
        self.inertia = np.array(inertia)
        self.com = np.array(com)  # relative to some reference point


def compose_bodies(bodies):
    """Compute dynamic parameters for a system of multiple rigid bodies."""
    mass = sum([body.mass for body in bodies])
    com = sum([body.mass * body.com for body in bodies]) / mass

    # parallel axis theorem to compute new inertia matrix
    inertia = np.zeros((3, 3))
    for body in bodies:
        r = body.com - com  # direction doesn't actually matter: it cancels out
        R = skew3(r, np=np)
        I_new = body.inertia - body.mass * R @ R
        inertia += I_new

    return Body(mass, inertia, com)


def circle_zmp_constraints(zmp, center, radius):
    """ZMP constraint for a circular support area."""
    e = zmp - center
    return radius ** 2 - e @ e


def edge_zmp_constraint(zmp, v1, v2):
    S = np.array([[0, 1], [-1, 0]])
    normal = S @ (v2 - v1)  # inward-facing
    return -(zmp - v1) @ normal  # negative because g >= 0


def polygon_zmp_constraints(zmp, vertices, np=jnp):
    """ZMP constraint for a polygonal support area.

    vertices are an N*2 array of vertices arranged in order, counter-clockwise.
    """
    def scan_func(v0, v1):
        return v1, edge_zmp_constraint(zmp, v0, v1)

    _, g = jax.lax.scan(scan_func, vertices[-1, :], vertices)
    return g


def polygon_zmp_constraints_np(zmp, vertices):
    """ZMP constraint for a polygonal support area.

    vertices are an N*2 array of vertices arranged in order, counter-clockwise.
    """
    N = vertices.shape[0]

    g = np.zeros(N)
    for i in range(N - 1):
        v1 = vertices[i, :]
        v2 = vertices[i+1, :]
        g[i] = edge_zmp_constraint(zmp, v1, v2)
    g[-1] = edge_zmp_constraint(zmp, vertices[-1, :], vertices[0, :])
    return g


class Polygon:
    def __init__(self, vertices):
        self.vertices = np.array(vertices)

    def zmp_constraints(self, zmp):
        pass
