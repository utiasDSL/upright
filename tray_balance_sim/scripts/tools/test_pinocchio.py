#!/usr/bin/env python3
import os
import numpy as np

from tray_balance_sim.robot import PinocchioRobot

import IPython


q = np.array([2, 0, 0, 0, -0.75 * np.pi, -0.5 * np.pi, -0.75 * np.pi, -0.5 * np.pi, 0.5 * np.pi])
v = np.zeros(9)

x = np.concatenate((q, v))
u = np.zeros(9)

robot = PinocchioRobot()
robot.forward(x, u)

r, Q = robot.link_pose()
v, w = robot.link_velocity()
a, α = robot.link_acceleration()
J = robot.jacobian(q)

IPython.embed()
