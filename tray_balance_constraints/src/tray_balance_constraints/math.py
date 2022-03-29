import numpy as np
import liegroups


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


def quat_multiply(q0, q1, normalize=True):
    """Hamilton product of two quaternions."""
    if normalize:
        q0 = q0 / np.linalg.norm(q0)
        q1 = q1 / np.linalg.norm(q1)
    C0 = liegroups.SO3.from_quaternion(q0, ordering="xyzw")
    C1 = liegroups.SO3.from_quaternion(q1, ordering="xyzw")
    return C0.dot(C1).to_quaternion(ordering="xyzw")


def quat_error(q):
    xyz = q[:3]
    w = q[3]
    # this is just the angle part of an axis-angle
    return 2 * np.arctan2(np.linalg.norm(xyz), w)


def quat_inverse(q):
    """Inverse of quaternion q.

    Such that quat_multiply(q, quat_inverse(q)) = [0, 0, 0, 1].
    """
    return np.append(-q[:3], q[3])
