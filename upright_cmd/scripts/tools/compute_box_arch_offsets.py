#!/usr/bin/env python3
import numpy as np
from spatialmath.base import rotx, roty, rotz
import IPython


side_lengths = np.array([0.103, 0.09, 0.038])

Rx = rotx(0.5*np.pi)
Ry = roty(0.5*np.pi)
Rz = rotz(0.5*np.pi)

box1_dims = box2_dims = np.abs(Ry @ side_lengths)
box3_dims = side_lengths

x_offset_1 = 0.5 * box1_dims[0]
x_offset_2 = box3_dims[0] - 0.5 * box2_dims[0]
x_offset_3 = -0.5 * (box3_dims[0] - box2_dims[0])

# compute offsets of each block
# y_offset1 = 0.5 * (box3_dims[1] - box1_dims[1])
# y_offset2 = -y_offset1
# y_offset3 = -y_offset2  # relative to block 2

print(f"x offset 1 = {x_offset_1}")
print(f"x offset 2 = {x_offset_2}")
print(f"x offset 3 = {x_offset_3}")
# print(f"y offset 1 = {y_offset1}")
# print(f"y offset 2 = {y_offset2}")
# print(f"y offset 3 = {y_offset3}")

IPython.embed()
