import os

import numpy as np
import pybullet as pyb

import upright_core as core
from mobile_manipulation_central.simulation import BulletSimulatedRobot

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


class UprightSimulatedRobot(BulletSimulatedRobot):
    def __init__(self, config, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        urdf_path = core.parsing.parse_and_compile_urdf(config["robot"]["urdf"])

        locked_joints = config["robot"].get("locked_joints", {})
        locked_joints = {
            name: core.parsing.parse_number(value)
            for name, value in locked_joints.items()
        }

        super().__init__(
            urdf_path,
            tool_joint_name=config["robot"]["tool_joint_name"],
            position=position,
            orientation=orientation,
            actuated_joints=config["robot"]["joint_names"],
            locked_joints=locked_joints,
        )

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

        # set tool to have friction coefficient μ=1 for convenience
        pyb.changeDynamics(self.uid, self.tool_idx, lateralFriction=1.0)

    def reset_arm_joints(self, qa):
        """Reset the configuration of the arm only."""
        for idx, angle in zip(self.robot_joint_indices[3:], qa):
            pyb.resetJointState(self.uid, idx, angle)

    def command_velocity(self, cmd_vel, bodyframe=False):
        """Command the velocity of the robot's joints."""
        # self.cmd_vel = cmd_vel
        q, _ = self.joint_states()

        # convert to PyBullet coordinates
        _, v_pyb = self.pyb_mapping.forward(q, cmd_vel, bodyframe=bodyframe)

        # add process noise
        v_pyb_noisy = v_pyb + np.random.normal(
            scale=self.v_cmd_std_dev, size=v_pyb.shape
        )

        super().command_velocity(v_pyb_noisy)

        # return the actual commanded velocity
        return v_pyb_noisy

    def joint_states(self, add_noise=False, bodyframe=False):
        """Get the current state of the joints.

        Return a tuple (q, v), where q is the n-dim array of positions and v is
        the n-dim array of velocities.
        """
        q_pyb, v_pyb = super().joint_states()

        # convert from PyBullet coordinates
        q, v = self.pyb_mapping.inverse(q_pyb, v_pyb, bodyframe=bodyframe)

        if add_noise:
            q += np.random.normal(scale=self.q_meas_std_dev, size=q.shape)
            v += np.random.normal(scale=self.v_meas_std_dev, size=v.shape)
        return q, v
