import numpy as np
import pybullet as pyb
import time
from liegroups import SO3
import pybullet_data

import IPython


SIM_DT = 0.001

UR10_JOINT_NAMES = [
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]

UR10_HOME = [0.0, -2.3562, -1.5708, -2.3562, -1.5708, 1.5708]


class SimulatedRobot:
    def __init__(self, position=(0, 0, 0), orientation=(0, 0, 0, 1), joint_angles=None):
        self.robot = pyb.loadURDF(
            "assets/urdf/mm.urdf",
            position,
            orientation,
            flags=pyb.URDF_MERGE_FIXED_LINKS,
        )

        # build a dict of all joints, keyed by name
        self.joints = {}
        for i in range(pyb.getNumJoints(self.robot)):
            info = pyb.getJointInfo(self.robot, i)
            name = info[1].decode("utf-8")
            self.joints[name] = info

        # get the indices for the UR10 joints
        self.ur10_joint_indices = []
        for name in UR10_JOINT_NAMES:
            idx = self.joints[name][0]
            self.ur10_joint_indices.append(idx)

        # set the UR10 to the home position
        if joint_angles is None:
            joint_angles = UR10_HOME

        for idx, angle in zip(self.ur10_joint_indices, joint_angles):
            pyb.resetJointState(self.robot, idx, angle)

    def _command_arm_velocity(self, ua):
        """Command arm joint velocities."""
        pyb.setJointMotorControlArray(
            self.robot,
            self.ur10_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=ua,
        )

    def _command_base_velocity(self, ub):
        """Command base velocity.

        The input ub = [vx, vy, wz] is in body coordinates.
        """
        # map from body coordinates to world coordinates for pybullet
        C_wb = SO3.rotz(ub[2])
        linear = C_wb.dot([ub[0], ub[1], 0])
        angular = [0, 0, ub[2]]
        pyb.resetBaseVelocity(self.robot, linear, angular)

    def command_velocity(self, u):
        """Command the velocity of the robot's joints."""
        self._command_base_velocity(u[:3])
        self._command_arm_velocity(u[3:])

    def _base_state(self):
        """Get the state of the base.

        Returns a tuple (q, v), where q is the 3-dim 2D pose of the base and
        v is the 3-dim twist of joint velocities.
        """
        position, quaternion = pyb.getBasePositionAndOrientation(self.robot)
        linear_vel, angular_vel = pyb.getBaseVelocity(self.robot)

        yaw = pyb.getEulerFromQuaternion(quaternion)[2]
        pose2d = [position[0], position[1], yaw]
        twist2d = [linear_vel[0], linear_vel[1], angular_vel[2]]

        return pose2d, twist2d

    def _arm_state(self):
        """Get the state of the arm.

        Returns a tuple (q, v), where q is the 6-dim array of joint angles and
        v is the 6-dim array of joint velocities.
        """
        states = pyb.getJointStates(self.robot, self.ur10_joint_indices)
        ur10_positions = [state[0] for state in states]
        ur10_velocities = [state[1] for state in states]
        return ur10_positions, ur10_velocities

    def joint_states(self):
        """Get the current state of the joints.

        Return a tuple (q, v), where q is the n-dim array of positions and v is
        the n-dim array of velocities.
        """
        qb, vb = self._base_state()
        qa, va = self._arm_state()
        return np.concatenate((qb, qa)), np.concatenate((vb, va))

    def jacobian(self, q):
        pass


def main():
    clid = pyb.connect(pyb.GUI)

    # NOTE: Coulomb friction cone is the default, but this can be disabled to
    # use a considerably faster linearized pyramid model, which isn't much less
    # accurate.
    # pyb.setPhysicsEngineParameter(enableConeFriction=0)

    # ground plane
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    pyb.loadURDF("plane.urdf", [0, 0, 0])

    # robot
    mm = SimulatedRobot()

    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(SIM_DT)

    t = 0

    # simulation loop
    while True:
        u = [0.1, 0, 0, 0.1, 0, 0, 0, 0, 0]
        mm.command_velocity(u)
        q, v = mm.joint_states()
        IPython.embed()

        pyb.stepSimulation()

        t += SIM_DT
        # TODO smart sleep a la ROS - is there a standalone package for this?
        time.sleep(SIM_DT)


if __name__ == "__main__":
    main()
