import os

import numpy as np
import pybullet as pyb

import upright_core as core

import IPython


class FixedBaseMapping:
    @staticmethod
    def forward(q, v, bodyframe=False):
        return q.copy(), v.copy()

    @staticmethod
    def inverse(q_pyb, v_pyb, bodyframe=False):
        return q_pyb.copy(), v_pyb.copy()


class NonholonomicBaseMapping:
    @staticmethod
    def forward(q, v, bodyframe=False):
        yaw = q[2]
        C_wb = core.math.rotz(yaw)
        v_pyb = np.copy(v)
        v_pyb[1] = 0  # nonholonomic constraint: cannot move sideways
        v_pyb[:3] = C_wb @ v[:3]
        return q.copy(), v_pyb

    @staticmethod
    def inverse(q_pyb, v_pyb, bodyframe=False):
        yaw = q_pyb[2]
        C_wb = core.math.rotz(yaw)
        v = np.copy(v_pyb)
        v[:3] = C_wb.T @ v_pyb[:3]
        v[1] = 0
        return q_pyb.copy(), v


class OmnidirectionalBaseMapping:
    @staticmethod
    def forward(q, v, bodyframe=False):
        if bodyframe:
            yaw = q[2]
            C_wb = core.math.rotz(yaw)
            v_pyb = np.copy(v)
            v_pyb[:3] = C_wb @ v[:3]
            return q.copy(), v_pyb
        else:
            return q.copy(), v.copy()

    @staticmethod
    def inverse(q_pyb, v_pyb, bodyframe=False):
        if bodyframe:
            yaw = q_pyb[2]
            C_wb = core.math.rotz(yaw)
            v = np.copy(v_pyb)
            v[:3] = C_wb.T @ v_pyb[:3]
            return q_pyb.copy(), v
        else:
            return q_pyb.copy(), v_pyb.copy()


class PyBulletInputMapping:
    """Mappings between our coordinates and PyBullet coordinates.

    Each class provides two functions:
        forward(q, v) -> (q_pyb, v_pyb)
        inverse(q_pyb, v_pyb) -> (q, v)
    """
    @staticmethod
    def from_string(s):
        s = s.lower()
        if s == "fixed":
            return FixedBaseMapping
        elif s == "nonholonomic":
            return PyBulletInputMapping.nonholonomic
        elif s == "omnidirectional":
            return OmnidirectionalBaseMapping
        elif s == "floating":
            raise NotImplementedError("Floating base not yet implemented.")
        else:
            raise ValueError(f"Cannot create base type from string {s}.")


class SimulatedRobot:
    def __init__(self, config, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        # NOTE: passing the flag URDF_MERGE_FIXED_LINKS is good for performance
        # but messes up the origins of the merged links, so this is not
        # recommended. Instead, if performance is an issue, consider using the
        # base_simple.urdf model instead of the Ridgeback.
        urdf_path = core.parsing.parse_and_compile_urdf(config["robot"]["urdf"])
        self.uid = pyb.loadURDF(urdf_path, position, orientation, useFixedBase=True)

        # home position
        self.home = core.parsing.parse_array(config["robot"]["home"])

        # map from (q, v) to PyBullet input
        self.pyb_mapping = PyBulletInputMapping.from_string(
            config["robot"]["base_type"]
        )

        # dimensions
        self.nq = config["robot"]["dims"]["q"]  # num positions
        self.nv = config["robot"]["dims"]["v"]  # num velocities
        self.nx = config["robot"]["dims"]["x"]  # num states
        self.nu = config["robot"]["dims"]["u"]  # num inputs

        # noise
        self.q_meas_std_dev = config["robot"]["noise"]["measurement"]["q_std_dev"]
        self.v_meas_std_dev = config["robot"]["noise"]["measurement"]["v_std_dev"]
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

        # get the indices for the actuated joints
        self.robot_joint_indices = []
        for name in config["robot"]["joint_names"]:
            idx = self.joints[name][0]
            self.robot_joint_indices.append(idx)

        # Link index (of the tool, in this case) is the same as the joint
        self.tool_idx = self.joints[config["robot"]["tool_joint_name"]][0]

        # set tool to have friction coefficient μ=1 for convenience
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

    def command_velocity(self, cmd_vel, bodyframe=False):
        """Command the velocity of the robot's joints."""
        # self.cmd_vel = cmd_vel
        q, _ = self.joint_states()

        # convert to PyBullet coordinates
        _, v_pyb = self.pyb_mapping.forward(q, cmd_vel, bodyframe=bodyframe)

        # add process noise
        v_pyb_noisy = v_pyb + np.random.normal(scale=self.v_cmd_std_dev, size=v_pyb.shape)

        pyb.setJointMotorControlArray(
            self.uid,
            self.robot_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=list(v_pyb_noisy),
        )

    def joint_states(self, add_noise=False, bodyframe=False):
        """Get the current state of the joints.

        Return a tuple (q, v), where q is the n-dim array of positions and v is
        the n-dim array of velocities.
        """
        states = pyb.getJointStates(self.uid, self.robot_joint_indices)
        q_pyb = np.array([state[0] for state in states])
        v_pyb = np.array([state[1] for state in states])

        # convert from PyBullet coordinates
        q, v = self.pyb_mapping.inverse(q_pyb, v_pyb, bodyframe=bodyframe)

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
