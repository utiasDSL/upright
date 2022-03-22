import sys
import numpy as np
import scipy.linalg

import tray_balance_constraints as con

import IPython

N = 100


center = np.array([0, 0, 0])
half_lengths = np.array([0.1, 0.1, 0.2])
directions = np.eye(3)
ell1 = con.Ellipsoid(center, half_lengths, directions, rank=3)
# ell1 = con.Ellipsoid.point(center)
radii1 = 0.1 * np.ones(3)

center = np.array([1, 1, 0])
half_lengths = np.array([0.1, 0.1, 0])
directions = np.eye(3)
ell2 = con.Ellipsoid(center, half_lengths, directions, rank=2)
# ell2 = con.Ellipsoid.point(center)
radii2 = 0.1 * np.ones(3)

body1 = con.BoundedRigidBody(
    mass_min=0.25, mass_max=0.5, radii_of_gyration=radii1, com_ellipsoid=ell1
)
body2 = con.BoundedRigidBody(
    mass_min=0.25, mass_max=0.5, radii_of_gyration=radii2, com_ellipsoid=ell2
)

composite = con.BoundedRigidBody.compose([body1, body2])

IPython.embed()

# ell = con.Ellipsoid.bounding_ellipsoid(P, 0.01)
# R = scipy.linalg.orth(P.T)
#
# # project points into R basis
# PR = P @ R
#
# c, hl, V = bounding_ellipsoid(PR, 0.01)
#
# # project the bounding ellipsoid back into the original space
# directions = R @ V
# center = R @ c
# half_lengths = hl
#
# ell = con.Ellipsoid(center, half_lengths, directions, rank=R.shape[1])

# IPython.embed()
