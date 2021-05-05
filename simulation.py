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

import sqp

import IPython


SIM_DT = 0.001
MU_LATERAL = 0.5


# TODO: we need to get the forward kinematics for the robot in terms jax can
# understand, then autodiff to get everything else---will likely be slow
# let's start with a severely restricted planar model:
#   base x, ur10_arm_shoulder_lift_joint, ur10_arm_elbow_joint, ur10_arm_wrist_1_joint
# this should make it about the same complexity as the mm2d approach
#
# we can further restrict the tray to be stuck in the 2D plane via a constraint


class Tray:
    def __init__(self, mass=0.5, radius=0.25, height=0.01):

        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=[0, 0, 1, 1],
        )
        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=[0, 0, 2],
            baseOrientation=[0, 0, 0, 1],
        )

        # set friction
        pyb.changeDynamics(self.uid, -1, lateralFriction=MU_LATERAL)

    def get_pose(self):
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)

    def reset_pose(self, position=None, orientation=None):
        current_pos, current_orn = self.get_pose()
        if position is None:
            position = current_pos
        if orientation is None:
            orientation = current_orn
        pyb.resetBasePositionAndOrientation(self.uid, list(position), list(orientation))


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
