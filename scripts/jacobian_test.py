#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data

from mm_pybullet_sim.simulation import ROBOT_HOME
from mm_pybullet_sim.robot import SimulatedRobot, RobotModel

import IPython


SIM_DT = 0.001
CTRL_DT = 0.1


def main():
    np.set_printoptions(precision=3, suppress=True)

    pyb.connect(pyb.GUI)
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(SIM_DT)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    robot = SimulatedRobot(SIM_DT)
    robot.reset_joint_configuration(ROBOT_HOME)

    q = np.array(ROBOT_HOME)
    model = RobotModel(0.1, q)
    # model.jacobian(q)

    q, v = robot.joint_states()

    IPython.embed()


if __name__ == "__main__":
    main()
