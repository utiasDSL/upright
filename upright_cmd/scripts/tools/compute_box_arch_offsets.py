#!/usr/bin/env python3
import numpy as np
from spatialmath.base import rotx, rotz
import IPython


side_lengths = np.array([0.103, 0.09, 0.038])

Rx = rotx(0.5*np.pi)
Rz = rotz(0.5*np.pi)
box1_dims = box2_dims = np.abs(Rx @ side_lengths)
box3_dims = np.abs(Rz @ side_lengths)

y_length = box3_dims[1]

# compute offsets of each block
# x offsets are all zero (centered on the origin in x-direction)
y_offset1 = 0.5 * (y_length - box1_dims[1])
y_offset2 = -y_offset1
y_offset3 = -y_offset2  # relative to block 2

IPython.embed()
