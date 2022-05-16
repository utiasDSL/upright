#!/usr/bin/env python3
import os
import numpy as np
import sys

import tray_balance_constraints as core
from tray_balance_ocs2.robot import PinocchioRobot

import IPython

np.random.seed(0)
np.set_printoptions(precision=3, suppress=True)

# get config path from command line argument
config_path = sys.argv[1]
config = core.parsing.load_config(config_path)["controller"]["robot"]

robot = PinocchioRobot(config)

q = np.array([0, -0.7854, 0, -1.5708, 0, 2.3562, -0.7854])
#v = np.zeros(robot.dims.v)
v = np.random.random(robot.dims.v) - 0.5
a = np.zeros(robot.dims.v)

x = np.concatenate((q, v, a))
u = np.zeros(robot.dims.u)

robot.forward(x, u)

r, Q = robot.link_pose()
dr, w = robot.link_velocity()
ddr, α = robot.link_acceleration()
ddr_spatial, _ = robot.link_spatial_acceleration()
J = robot.jacobian(q)

robot.forward_derivatives(x, u)

dVdq, dVdv = robot.link_velocity_derivatives()
dAdq, dAdv, dAda = robot.link_acceleration_derivatives()

# J == dVdv == dAda
np.testing.assert_allclose(J, dVdv, atol=1e-5)
np.testing.assert_allclose(J, dAda, atol=1e-5)

# h = 1e-5
# hq1 = h * np.eye(q.shape[0])[:, 0]
# q = q + hq1
# #v = v + hq1
# x = np.concatenate((q, v, a))
# robot.forward(x, u)
#
# ddr2, α2 = robot.link_acceleration()
# ddr2_spatial, _ = robot.link_spatial_acceleration()
#
# dAdq1_approx = (ddr2 - ddr) / h
# dAdq1_approx_spatial = (ddr2_spatial - ddr_spatial) / h
#
# IPython.embed()
