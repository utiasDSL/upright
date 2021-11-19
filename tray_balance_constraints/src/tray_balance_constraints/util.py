import numpy as np


def skew1(x):
    """2D skew-symmetric operator."""
    return np.array([[0, -x], [x, 0]])


def skew3(x, np=np):
    """3D skew-symmetric operator."""
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def equilateral_triangle_inscribed_radius(side_length):
    """Compute radius of the inscribed circle of equilateral triangle."""
    return side_length / (2 * np.sqrt(3))


def cylinder_inertia_matrix(m, r, h, np=np):
    """Inertia matrix for cylinder aligned along z-axis."""
    xx = yy = m * (3 * r**2 + h**2) / 12
    zz = 0.5 * m * r**2
    return np.diag([xx, yy, zz])
