"""Tray balancing simulation in pybullet."""

# Friction
# ========
# Bullet calculates friction between two objects by multiplying the
# coefficients of friction for each object*. To deal with this, I set the the
# coefficient for the EE to 1, and then vary the object value to achieve the
# desired result.
#
# Coulomb friction cone is the default, but this can be disabled to use a
# considerably faster linearized pyramid model, which apparently isn't much
# less accurate, using:
# pyb.setPhysicsEngineParameter(enableConeFriction=0)
#
# *see https://pybullet.org/Bullet/BulletFull/btManifoldResult_8cpp_source.html

import time

import numpy as np
import pybullet as pyb
import pybullet_data

import IPython


SIM_DT = 0.001


def main():
    np.set_printoptions(precision=3, suppress=True)

    pyb.connect(pyb.GUI)

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(SIM_DT)

    # setup ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    # setup robot
    mm = SimulatedRobot()
    mm.reset_joint_configuration(ROBOT_HOME)

    # simulate briefly to let the robot settle down after being positioned
    t = 0
    while t < 1.0:
        pyb.stepSimulation()
        t += SIM_DT

    # arm gets bumped by the above settling, so we reset it again
    mm.reset_arm_joints(UR10_HOME_TRAY_BALANCE)

    # setup tray
    tray = Tray()
    ee_pos, _ = mm.link_pose()
    tray.reset_pose(position=ee_pos + [0, 0, 0.05])

    # main simulation loop
    t = 0
    while True:
        # open-loop command
        # u = [0.1, 0, 0, 0.1, 0, 0, 0, 0, 0]
        # mm.command_velocity(u)

        pyb.stepSimulation()

        t += SIM_DT
        # TODO smart sleep a la ROS - is there a standalone package for this?
        time.sleep(SIM_DT)


if __name__ == "__main__":
    main()
