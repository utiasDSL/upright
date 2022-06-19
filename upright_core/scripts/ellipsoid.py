import sys
import numpy as np
import scipy.linalg

import upright_core as con

import IPython


N = 100

center = np.array([0, 0, 0])
half_lengths = np.array([0.1, 0.1, 0.2])
directions = np.eye(3)
ell1 = con.Ellipsoid(center, half_lengths, directions)
# ell1 = con.Ellipsoid.point(center)
radii1 = 0.1 * np.ones(3)

center = np.array([1, 1, 0])
half_lengths = np.array([0.1, 0.1, 0])
directions = np.eye(3)
ell2 = con.Ellipsoid(center, half_lengths, directions)
# ell2 = con.Ellipsoid.point(center)
radii2 = 0.1 * np.array([1, 1, 2])

body1 = con.BoundedRigidBody(
    mass_min=0.25, mass_max=0.5, radii_of_gyration=radii1, com_ellipsoid=ell1
)
body2 = con.BoundedRigidBody(
    mass_min=0.25, mass_max=0.5, radii_of_gyration=radii2, com_ellipsoid=ell2
)

composite = con.BoundedRigidBody.compose([body1, body2])

IPython.embed()
