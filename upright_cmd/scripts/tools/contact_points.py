#!/usr/bin/env python3
import numpy as np
from spatialmath.base import rotz
from upright_core import geometry
import IPython


half_extents = [0.1, 0.1, 0.1]

rotation = rotz(np.pi / 4)

box1 = geometry.Box3d(half_extents)
box2 = geometry.Box3d(
    half_extents,
    position=np.array([0.1 + np.sqrt(2 * 0.1 ** 2), 0, 0]),
    rotation=rotation,
)

V = geometry.box_box_axis_aligned_contact(box1, box2)

IPython.embed()
