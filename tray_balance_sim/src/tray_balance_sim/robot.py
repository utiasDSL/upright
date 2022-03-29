import os

import numpy as np
import pybullet as pyb
import liegroups

from tray_balance_constraints import parsing

import IPython


class SimulatedRobot:
    def __init__(self, config, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        # NOTE: passing the flag URDF_MERGE_FIXED_LINKS is good for performance
        # but messes up the origins of the merged links, so this is not
        # recommended. Instead, if performance is an issue, consider using the
        # base_simple.urdf model instead of the Ridgeback.
        if config["static_obstacles"]["enabled"]:
            urdf_path = parsing.parse_urdf_path(config["urdf"]["robot_obstacles"])
        else:
            urdf_path = parsing.parse_urdf_path(config["urdf"]["robot"])

        self.uid = pyb.loadURDF(urdf_path, position, orientation)

        # home position
        self.base_home = parsing.parse_array(config["robot"]["home"]["base"])
        self.arm_home = parsing.parse_array(config["robot"]["home"]["arm"])
        self.home = np.concatenate((self.base_home, self.arm_home))

        self.ns = 18  # num state
        self.ni = 9  # num inputs

        self.cmd_vel = np.zeros(9)
        self.cmd_acc = np.zeros_like(self.cmd_vel)

        # noise
        self.q_meas_std_dev = config["robot"]["noise"]["measurement"]["q_std_dev"]
        self.v_meas_std_dev= config["robot"]["noise"]["measurement"]["v_std_dev"]
        self.v_cmd_std_dev = config["robot"]["noise"]["process"]["v_std_dev"]

        # build a dict of all joints, keyed by name
        self.joints = {}
        self.links = {}
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")
            self.joints[joint_name] = info
            self.links[link_name] = info

        # get the indices for the UR10 joints
        self.robot_joint_indices = []
        for name in config["robot"]["joint_names"]:
            idx = self.joints[name][0]
            self.robot_joint_indices.append(idx)

        # Link index (of the tool, in this case) is the same as the joint
        self.tool_idx = self.joints[config["robot"]["tool_joint_name"]][0]

        pyb.changeDynamics(self.uid, self.tool_idx, lateralFriction=1.0)

    def reset_arm_joints(self, qa):
        for idx, angle in zip(self.robot_joint_indices[3:], qa):
            pyb.resetJointState(self.uid, idx, angle)

    def reset_joint_configuration(self, q):
        """Reset the robot to a particular configuration.

        It is best not to do this during a simulation, as this overrides are
        dynamic effects.
        """
        for idx, angle in zip(self.robot_joint_indices, q):
            pyb.resetJointState(self.uid, idx, angle)

    def _base_rotation_matrix(self):
        """Get rotation matrix for the base.

        This is just the rotation about the z-axis by the yaw angle.
        """
        state = pyb.getJointState(self.uid, self.robot_joint_indices[2])
        yaw = state[0]
        C_wb = liegroups.SO3.rotz(yaw).as_matrix()
        return C_wb

    def command_velocity(self, u, bodyframe=True):
        """Command the velocity of the robot's joints."""
        u = np.copy(u)
        if bodyframe:
            # u[1] = 0  # nonholonomic
            C_wb = self._base_rotation_matrix()
            u[:3] = C_wb @ u[:3]

        # add process noise
        u_noisy = u + np.random.normal(scale=self.v_cmd_std_dev, size=u.shape)

        pyb.setJointMotorControlArray(
            self.uid,
            self.robot_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=list(u_noisy),
        )

    def command_acceleration(self, cmd_acc):
        """Command acceleration of the robot's joints."""
        self.cmd_acc = cmd_acc

    def step(self, secs):
        """Step the robot kinematics forward by `secs` seconds."""
        # input (acceleration) and velocity are both in the body frame
        self.cmd_vel += secs * self.cmd_acc
        self.command_velocity(self.cmd_vel, bodyframe=True)

    def joint_states(self, add_noise=False):
        """Get the current state of the joints.

        Return a tuple (q, v), where q is the n-dim array of positions and v is
        the n-dim array of velocities.
        """
        states = pyb.getJointStates(self.uid, self.robot_joint_indices)
        q = np.array([state[0] for state in states])
        v = np.array([state[1] for state in states])
        if add_noise:
            q += np.random.normal(scale=self.q_meas_std_dev, size=q.shape)
            v += np.random.normal(scale=self.v_meas_std_dev, size=v.shape)
        return q, v

    def link_pose(self, link_idx=None):
        """Get the pose of a particular link in the world frame.

        It is the pose of origin of the link w.r.t. the world. The origin of
        the link is the location of its parent joint.

        If no link_idx is provided, defaults to that of the tool.
        """
        if link_idx is None:
            link_idx = self.tool_idx
        state = pyb.getLinkState(self.uid, link_idx, computeForwardKinematics=True)
        pos, orn = state[4], state[5]
        return np.array(pos), np.array(orn)

    def link_velocity(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        state = pyb.getLinkState(
            self.uid,
            link_idx,
            computeLinkVelocity=True,
        )
        return np.array(state[-2]), np.array(state[-1])

    def jacobian(self, q=None):
        """Get the end effector Jacobian at the current configuration."""

        if q is None:
            q, _ = self.joint_states()
        z = list(np.zeros_like(q))
        q = list(q)

        tool_offset = [0, 0, 0]
        Jv, Jw = pyb.calculateJacobian(self.uid, self.tool_idx, tool_offset, q, z, z)
        J = np.vstack((Jv, Jw))
        return J
