import sys
import numpy as np
import scipy.linalg

import tray_balance_constraints as con

import IPython

N = 100


center = np.array([0, 0, 0])
half_lengths = np.array([0.1, 0.1, 0.2])
directions = np.eye(3)
rank = 3
ell1 = con.Ellipsoid.point(center)
# ell3 = con.Ellipsoid(center, half_lengths, directions, rank)

center = np.array([1, 1, 0])
half_lengths = np.array([0.1, 0.1, 0])
directions = np.eye(3)
# ell2 = con.Ellipsoid(center, half_lengths, directions, rank)
ell2 = con.Ellipsoid.point(center)

body1 = con.RigidBodyBounds(0.25, 0.5, 0.1, ell1)
body2 = con.RigidBodyBounds(0.25, 0.5, 0.1, ell2)

P = np.zeros((N, 3))
for i in range(N):
    m1, r1 = body1.sample(boundary=True)
    m2, r2 = body2.sample(boundary=True)
    P[i, :] = (m1 * r1 + m2 * r2) / (m1 + m2)

# IPython.embed()
# sys.exit()

ell = con.Ellipsoid.bounding_ellipsoid(P, 0.01)
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

IPython.embed()
