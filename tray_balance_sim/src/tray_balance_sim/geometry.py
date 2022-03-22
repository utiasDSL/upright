import numpy as np


def equilateral_triangle_inscribed_radius(side_length):
    """Compute radius of the inscribed circle of equilateral triangle."""
    return side_length / (2 * np.sqrt(3))


def circle_r_tau(radius):
    """r_tau for a circular support area with uniform friction."""
    return 2.0 * radius / 3


def alpha_rect(w, h):
    # alpha_rect for half of the rectangle
    d = np.sqrt(h * h + w * w)
    return (w * h * d + w * w * w * (np.log(h + d) - np.log(w))) / 12.0


def rectangle_r_tau(w, h):
    """r_tau for a rectangular support area with uniform friction."""
    # see pushing notes
    return (alpha_rect(w, h) + alpha_rect(h, w)) / (w * h)


def equilateral_triangle_area(side_length):
    """Area of an equilateral triangle."""
    return np.sqrt(3) * side_length**2 / 4


def equilateral_triangle_r_tau(side_length):
    """r_tau for equilateral triangle."""
    h = equilateral_triangle_inscribed_radius(side_length)
    θ = np.pi / 3.0
    sec = 1. / np.cos(θ)
    tan = np.tan(θ)
    area = equilateral_triangle_area(side_length)
    return h**3 * (tan * sec + np.log(tan + sec)) / area
