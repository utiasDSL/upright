#!/usr/bin/env python3
import os
import numpy as np
import sys

import tray_balance_constraints as core
from tray_balance_ocs2.robot import PinocchioRobot

import IPython

# get config path from command line argument
config_path = sys.argv[1]
config = core.parsing.load_config(config_path)["controller"]

robot = PinocchioRobot(config)

q = np.array([0, -0.7854, 0, -1.5708, 0, 2.3562, -0.7854])
v = np.zeros(robot.dims.v)
a = np.zeros(robot.dims.v)

x = np.concatenate((q, v, a))
u = np.zeros(robot.dims.u)

robot.forward(x, u)

r, Q = robot.link_pose()
v, w = robot.link_velocity()
a, α = robot.link_acceleration()
J = robot.jacobian(q)

IPython.embed()
