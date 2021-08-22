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


def circle_r_tau(radius):
    return 2.0 * radius / 3


class PolygonSupportArea:
    """Polygonal support area

    vertices: N*2 array of vertices arranged in order, counter-clockwise.
    offset: the 2D vector pointing from the projection of the CoM on the
    support plane to the center of the support area
    """

    def __init__(self, vertices, offset=(0, 0)):
        self.vertices = np.array(vertices)
        self.offset = np.array(offset)

    @staticmethod
    def edge_zmp_constraint(zmp, v1, v2):
        """ZMP constraint for a single edge of a polygon.

        zmp is the zero-moment point
        v1 and v2 are the endpoints of the segment.
        """
        S = np.array([[0, 1], [-1, 0]])
        normal = S @ (v2 - v1)  # inward-facing
        return -(zmp - v1) @ normal  # negative because g >= 0

    def zmp_constraints(self, zmp):
        def scan_func(v0, v1):
            return v1, PolygonSupportArea.edge_zmp_constraint(zmp, v0, v1)

        _, g = jax.lax.scan(scan_func, self.vertices[-1, :], self.vertices)
        return g

    # TODO margin
    def zmp_constraints_numpy(self, zmp):
        N = self.vertices.shape[0]

        g = np.zeros(N)
        for i in range(N - 1):
            v1 = self.vertices[i, :]
            v2 = self.vertices[i + 1, :]
            g[i] = PolygonSupportArea.edge_zmp_constraint(zmp, v1, v2)
        g[-1] = PolygonSupportArea.edge_zmp_constraint(
            zmp, self.vertices[-1, :], self.vertices[0, :]
        )
        return g


class CircleSupportArea:
    """Circular support area

    offset: the 2D vector pointing from the projection of the CoM on the
    support plane to the center of the support area
    """

    def __init__(self, radius, offset=(0, 0), margin=0):
        self.radius = radius
        self.offset = np.array(offset)
        self.margin = margin

    def zmp_constraints(self, zmp):
        """Generate ZMP stability constraint.


        zmp: 2D point to check for inclusion in the support area
        margin: minimum distance from edge of support area to be considered inside

        Returns a value g, where g >= 0 satisfies the ZMP constraint
        """
        e = zmp - self.offset
        return self.radius ** 2 - self.margin ** 2 - e @ e
