import numpy as np
from scipy.linalg import null_space
from spatialmath.base import q2r, r2q, qunit, rotx, roty, rotz


QUAT_ORDER = "xyzs"


def skew3(v):
    """Form a skew-symmetric matrix out of 3-dimensional vector v."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def equilateral_triangle_inscribed_radius(side_length):
    """Compute radius of the inscribed circle of equilateral triangle."""
    return side_length / (2 * np.sqrt(3))


def circle_r_tau(radius):
    """r_tau for a circular support area with uniform friction."""
    return 2.0 * radius / 3


def _alpha_rect(w, h):
    # alpha_rect for half of the rectangle
    d = np.sqrt(h * h + w * w)
    return (w * h * d + w * w * w * (np.log(h + d) - np.log(w))) / 12.0


def rectangle_r_tau(w, h):
    """r_tau for a rectangular support area with uniform friction."""
    # see pushing notes
    return (_alpha_rect(w, h) + _alpha_rect(h, w)) / (w * h)


def equilateral_triangle_area(side_length):
    """Area of an equilateral triangle."""
    return np.sqrt(3) * side_length ** 2 / 4


def equilateral_triangle_r_tau(side_length):
    """r_tau for equilateral triangle."""
    h = equilateral_triangle_inscribed_radius(side_length)
    θ = np.pi / 3.0
    sec = 1.0 / np.cos(θ)
    tan = np.tan(θ)
    area = equilateral_triangle_area(side_length)
    return h ** 3 * (tan * sec + np.log(tan + sec)) / area


def quat_to_rot(q):
    """Convert quaternion q to rotation matrix."""
    return q2r(q, order=QUAT_ORDER)


def rot_to_quat(C):
    """Convert rotation matrix C to quaternion."""
    return r2q(C, order=QUAT_ORDER)


def quat_multiply(q0, q1, normalize=True):
    """Hamilton product of two quaternions."""
    order = "xyzs"
    if normalize:
        q0 = qunit(q0)
        q1 = qunit(q1)
    C0 = quat_to_rot(q0)
    C1 = quat_to_rot(q1)
    return rot_to_quat(C0 @ C1)


def quat_rotate(q, r):
    """Rotate point r by rotation represented by quaternion q."""
    return quat_to_rot(q) @ r


def quat_transform(r_ba_a, q_ab, r_cb_b):
    """Transform point r_cb_b by rotating by q_ab and translating by r_ba_a."""
    return quat_rotate(q_ab, r_cb_b) + r_ba_a


def quat_angle(q):
    """Get the scalar angle represented by a quaternion."""
    xyz = q[:3]
    w = q[3]
    # this is just the angle part of an axis-angle
    return 2 * np.arctan2(np.linalg.norm(xyz), w)


def quat_inverse(q):
    """Inverse of quaternion q.

    Such that quat_multiply(q, quat_inverse(q)) = [0, 0, 0, 1].
    """
    return np.append(-q[:3], q[3])


def cylinder_inertia_matrix(mass, radius, height):
    """Inertia matrix for cylinder aligned along z-axis."""
    xx = yy = mass * (3 * radius ** 2 + height ** 2) / 12
    zz = 0.5 * mass * radius ** 2
    return np.diag([xx, yy, zz])


def cuboid_inertia_matrix(mass, side_lengths):
    """Inertia matrix for a rectangular cuboid with side_lengths in (x, y, z)
    dimensions."""
    lx, ly, lz = side_lengths
    xx = ly ** 2 + lz ** 2
    yy = lx ** 2 + lz ** 2
    zz = lx ** 2 + ly ** 2
    return mass * np.diag([xx, yy, zz]) / 12.0


def inset_vertex(v, inset):
    """Move a vertex v closer to the origin by inset distance to the origin

    Raises a ValueError if `inset` is larger than `v`'s norm.

    Returns the inset vertex.
    """
    d = np.linalg.norm(v)
    if d <= inset:
        raise ValueError(f"Inset of {inset} is too large for the support area.")
    return (d - inset) * v / d


def inset_vertex_abs(v, inset):
    if (np.abs(v) <= inset).any():
        raise ValueError(f"Inset of {inset} is too large for the support area.")
    return v - np.sign(v) * inset


def plane_span(normal):
    """Computes the span of a plane defined by `normal` and going through the origin.

    Parameters:
        normal: a unit vector of shape (n,)

    Returns:
        An array S of shape(n - 1, n) such that S spans the plane: each row is
        a basis vector for the plane. In other words, this array is the
        transpose of a basis for the null space of the `normal`.

    Notes:
        Project a vector v, v.shape == (n,), into the plane using:
        >>> projection = S @ v
    """
    return null_space(normal[None, :]).T


# TODO
# def project_points_on_axes(vertices, point, axes):
#     return (axes @ (vertices - point).T).T
#
# def unproject_points_on_axes(vertices, point, axes):
#     pass
