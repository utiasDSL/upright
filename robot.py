import numpy as np
import pybullet as pyb
from liegroups import SO3


UR10_JOINT_NAMES = [
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]

TOOL_JOINT_NAME = "tool0_tcp_fixed_joint"


class SimulatedRobot:
    def __init__(self, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        # NOTE: passing the flag URDF_MERGE_FIXED_LINKS is good for performance
        # but messes up the origins of the merged links, so this is not
        # recommended. Instead, if performance is an issue, consider using the
        # base_simple.urdf model instead of the Ridgeback.
        self.uid = pyb.loadURDF(
            "assets/urdf/mm.urdf",
            position,
            orientation,
        )

        # build a dict of all joints, keyed by name
        self.joints = {}
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            name = info[1].decode("utf-8")
            self.joints[name] = info

        # get the indices for the UR10 joints
        self.ur10_joint_indices = []
        for name in UR10_JOINT_NAMES:
            idx = self.joints[name][0]
            self.ur10_joint_indices.append(idx)

        # Link index (of the tool, in this case) is the same as the joint
        self.tool_idx = self.joints[TOOL_JOINT_NAME][0]

        # TODO may need to also set spinningFriction
        pyb.changeDynamics(self.uid, self.tool_idx, lateralFriction=1.0)

    def reset_base_pose(self, qb):
        base_pos = [qb[0], qb[1], 0]
        base_orn = pyb.getQuaternionFromEuler([0, 0, qb[2]])
        pyb.resetBasePositionAndOrientation(self.uid, base_pos, base_orn)

    def reset_arm_joints(self, qa):
        for idx, angle in zip(self.ur10_joint_indices, qa):
            pyb.resetJointState(self.uid, idx, angle)

    def reset_joint_configuration(self, q):
        """Reset the robot to a particular configuration.

        It is best not to do this during a simulation, as this overrides are
        dynamic effects.
        """
        self.reset_base_pose(q[:3])
        self.reset_arm_joints(q[3:])

    def _command_arm_velocity(self, ua):
        """Command arm joint velocities."""
        pyb.setJointMotorControlArray(
            self.uid,
            self.ur10_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=ua,
        )

    def _command_base_velocity(self, ub):
        """Command base velocity.

        The input ub = [vx, vy, wz] is in body coordinates.
        """
        # map from body coordinates to world coordinates for pybullet
        pose, _ = self._base_state()
        yaw = pose[2]
        C_wb = SO3.rotz(yaw)

        linear = C_wb.dot([ub[0], ub[1], 0])
        angular = [0, 0, ub[2]]
        pyb.resetBaseVelocity(self.uid, linear, angular)

    def command_velocity(self, u):
        """Command the velocity of the robot's joints."""
        self._command_base_velocity(u[:3])
        self._command_arm_velocity(u[3:])

    def _base_state(self):
        """Get the state of the base.

        Returns a tuple (q, v), where q is the 3-dim 2D pose of the base and
        v is the 3-dim twist of joint velocities.
        """
        position, quaternion = pyb.getBasePositionAndOrientation(self.uid)
        linear_vel, angular_vel = pyb.getBaseVelocity(self.uid)

        yaw = pyb.getEulerFromQuaternion(quaternion)[2]
        pose2d = [position[0], position[1], yaw]
        twist2d = [linear_vel[0], linear_vel[1], angular_vel[2]]

        return pose2d, twist2d

    def _arm_state(self):
        """Get the state of the arm.

        Returns a tuple (q, v), where q is the 6-dim array of joint angles and
        v is the 6-dim array of joint velocities.
        """
        states = pyb.getJointStates(self.uid, self.ur10_joint_indices)
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

    def link_pose(self, link_idx=None):
        """Get the pose of a particular link in the world frame.

        If no link_idx is provided, defaults to that of the tool.
        """
        if link_idx is None:
            link_idx = self.tool_idx
        state = pyb.getLinkState(self.uid, link_idx, computeForwardKinematics=True)
        pos, orn = state[0], state[1]
        return np.array(pos), np.array(orn)

    def jacobian(self, q=None):
        """Get the end effector Jacobian at the current configuration."""
        # Don't allow querying of arbitrary configurations, because the pose in
        # the world (i.e. base pose) cannot be specified.

        if q is None:
            q, _ = self.joint_states()
        z = [0.0] * 6
        qa = list(q[3:])

        # Only actuated joints are used for computing the Jacobian (i.e. just
        # the arm)
        tool_offset = [0, 0, 0]
        Jv, Jw = pyb.calculateJacobian(self.uid, self.tool_idx, tool_offset, qa, z, z)

        # combine, reorder, and remove columns for base z, roll, and pitch (the
        # full 6-DOF of the base is included in pybullet's Jacobian, but we
        # don't want all of it)
        J = np.vstack((Jv, Jw))
        J = np.hstack((J[:, 3:5], J[:, 2, np.newaxis], J[:, 6:]))

        # pybullet calculates the Jacobian w.r.t. the base link, so we need to
        # rotate everything but the first two columns into the world frame
        # (note first two columns are constant)
        yaw = q[2]
        C_wb = SO3.rotz(yaw)
        R = np.kron(np.eye(2), C_wb.as_matrix())
        J = np.hstack((J[:, :2], R @ J[:, 2:]))

        return J
