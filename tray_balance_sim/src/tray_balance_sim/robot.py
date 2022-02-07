import os

import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
import pybullet as pyb
import jaxlie
import rospkg

import IPython

from tray_balance_sim.util import (
    dhtf,
    rot2d,
    pose_from_pos_quat,
    pose_to_pos_quat,
    skew3,
)

rospack = rospkg.RosPack()
ROBOT_URDF_PATH = os.path.join(rospack.get_path("tray_balance_assets"), "urdf", "mm_pyb.urdf")

BASE_JOINT_NAMES = ["x_to_world_joint", "y_to_x_joint", "base_to_y_joint"]

UR10_JOINT_NAMES = [
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]

ROBOT_JOINT_NAMES = BASE_JOINT_NAMES + UR10_JOINT_NAMES

TOOL_JOINT_NAME = "tool0_tcp_fixed_joint"

# DH parameters
PX = 0.27
PY = 0.01
PZ = 0.653
D1 = 0.1273
A2 = -0.612
A3 = -0.5723
D4 = 0.163941
D5 = 0.1157
D6 = 0.0922
D7 = 0.290


class SimulatedRobot:
    def __init__(self, dt, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        # NOTE: passing the flag URDF_MERGE_FIXED_LINKS is good for performance
        # but messes up the origins of the merged links, so this is not
        # recommended. Instead, if performance is an issue, consider using the
        # base_simple.urdf model instead of the Ridgeback.
        self.uid = pyb.loadURDF(ROBOT_URDF_PATH, position, orientation)

        self.dt = dt
        self.ns = 18  # num state
        self.ni = 9  # num inputs

        self.cmd_vel = np.zeros(9)
        self.cmd_acc = np.zeros_like(self.cmd_vel)

        # build a dict of all joints, keyed by name
        self.joints = {}
        for i in range(pyb.getNumJoints(self.uid)):
            info = pyb.getJointInfo(self.uid, i)
            name = info[1].decode("utf-8")
            self.joints[name] = info

        # get the indices for the UR10 joints
        self.robot_joint_indices = []
        for name in ROBOT_JOINT_NAMES:
            idx = self.joints[name][0]
            self.robot_joint_indices.append(idx)

        # Link index (of the tool, in this case) is the same as the joint
        self.tool_idx = self.joints[TOOL_JOINT_NAME][0]

        # pyb.changeDynamics(self.uid, -1, mass=0)
        # NOTE: this just makes the robot unable to move apparently
        # for i in range(pyb.getNumJoints(self.uid)):
        #     pyb.changeDynamics(self.uid, i, mass=0)
        # for i in range(pyb.getNumJoints(self.uid)):
        #     pyb.changeDynamics(self.uid, i, linearDamping=0, angularDamping=0)

        # TODO may need to also set spinningFriction
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
        C_wb = np.array(
            [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
        )
        return C_wb

    def command_velocity(self, u, bodyframe=True):
        """Command the velocity of the robot's joints."""
        if bodyframe:
            C_wb = self._base_rotation_matrix()
            ub = u[:3]
            u[:3] = C_wb @ ub
        pyb.setJointMotorControlArray(
            self.uid,
            self.robot_joint_indices,
            controlMode=pyb.VELOCITY_CONTROL,
            targetVelocities=list(u),
        )

    def command_acceleration(self, cmd_acc):
        """Command acceleration of the robot's joints."""
        # TODO for some reason feeding back v doesn't work
        # _, v = self.joint_states()
        # self.cmd_vel = v
        C_wb = self._base_rotation_matrix()
        base_acc = C_wb.dot(cmd_acc[:3])
        self.cmd_acc = np.concatenate((base_acc, cmd_acc[3:]))

    def step(self):
        """One step of the physics engine."""
        self.cmd_vel += self.dt * self.cmd_acc
        # acceleration is already in the world frame, so no need to rotate the
        # velocity
        self.command_velocity(self.cmd_vel, bodyframe=False)

    def joint_states(self):
        """Get the current state of the joints.

        Return a tuple (q, v), where q is the n-dim array of positions and v is
        the n-dim array of velocities.
        """
        states = pyb.getJointStates(self.uid, self.robot_joint_indices)
        q = np.array([state[0] for state in states])
        v = np.array([state[1] for state in states])
        return q, v

    def link_pose(self, link_idx=None):
        """Get the pose of a particular link in the world frame.

        If no link_idx is provided, defaults to that of the tool.
        """
        if link_idx is None:
            link_idx = self.tool_idx
        state = pyb.getLinkState(self.uid, link_idx, computeForwardKinematics=True)
        pos, orn = state[0], state[1]
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


class KinematicChain:
    """All transforms on the robot kinematic chain for a given configuration."""

    # world to base
    T0 = dhtf(np.pi / 2, 0, 0.01, np.pi / 2)
    T_w_0 = T0

    # base to arm
    T4 = dhtf(0, PX, PZ, -np.pi / 2)
    T5 = dhtf(0, 0, PY, np.pi / 2)

    # tool offset
    T12 = dhtf(np.pi, 0, D7, 0)

    def __init__(self, q):
        self.T1 = dhtf(np.pi / 2, 0, q[0], np.pi / 2)
        self.T2 = dhtf(np.pi / 2, 0, q[1], np.pi / 2)
        self.T3 = dhtf(q[2], 0, 0, 0)

        self.T6 = dhtf(q[3], 0, D1, np.pi / 2)
        self.T7 = dhtf(q[4], A2, 0, 0)
        self.T8 = dhtf(q[5], A3, 0, 0)
        self.T9 = dhtf(q[6], 0, D4, np.pi / 2)
        self.T10 = dhtf(q[7], 0, D5, -np.pi / 2)
        self.T11 = dhtf(q[8], 0, D6, 0)

        self.T_w_1 = self.T_w_0 @ self.T1
        self.T_w_2 = self.T_w_1 @ self.T2
        self.T_w_3 = self.T_w_2 @ self.T3

        self.T_w_5 = self.T_w_3 @ self.T4 @ self.T5
        self.T_w_6 = self.T_w_5 @ self.T6
        self.T_w_7 = self.T_w_6 @ self.T7
        self.T_w_8 = self.T_w_7 @ self.T8
        self.T_w_9 = self.T_w_8 @ self.T9
        self.T_w_10 = self.T_w_9 @ self.T10
        self.T_w_11 = self.T_w_10 @ self.T11

        self.T_w_12 = self.T_w_11 @ self.T12
        self.T_w_tool = self.T_w_12


class RobotModel:
    def __init__(self, dt):
        self.dt = dt
        self.ni = 9
        self.ns = 9 + 9

        self.dJdq = jax.jit(jax.jacfwd(self.jacobian))

    def jacobian(self, q):
        """Compute geometric Jacobian."""

        def rotation(T):
            return T[:3, :3]

        def translation(T):
            return T[:3, 3]

        chain = KinematicChain(q)
        z = np.array([0, 0, 1])  # Unit vector along z-axis

        # axis for each joint's angular velocity is the z-axis of the previous
        # transform
        z0 = rotation(chain.T_w_0) @ z
        z1 = rotation(chain.T_w_1) @ z
        z2 = rotation(chain.T_w_2) @ z

        z5 = rotation(chain.T_w_5) @ z
        z6 = rotation(chain.T_w_6) @ z
        z7 = rotation(chain.T_w_7) @ z
        z8 = rotation(chain.T_w_8) @ z
        z9 = rotation(chain.T_w_9) @ z
        z10 = rotation(chain.T_w_10) @ z

        # Angular Jacobian
        # joints xb and yb are prismatic, and so cause no angular velocity.
        Jo = jnp.vstack((np.zeros(3), np.zeros(3), z2, z5, z6, z7, z8, z9, z10)).T

        pe = translation(chain.T_w_tool)
        Jp = jnp.vstack(
            (
                z0,
                z1,
                jnp.cross(z2, pe - translation(chain.T_w_2)),
                jnp.cross(z5, pe - translation(chain.T_w_5)),
                jnp.cross(z6, pe - translation(chain.T_w_6)),
                jnp.cross(z7, pe - translation(chain.T_w_7)),
                jnp.cross(z8, pe - translation(chain.T_w_8)),
                jnp.cross(z9, pe - translation(chain.T_w_9)),
                jnp.cross(z10, pe - translation(chain.T_w_10)),
            )
        ).T

        # Full Jacobian
        return jnp.vstack((Jp, Jo))

    def tool_pose_matrix(self, q):
        """Tool pose as 4x4 homogeneous transformation matrix."""
        return KinematicChain(q).T_w_tool

    def tool_pose(self, q):
        """Tool pose as position and quaternion."""
        T = jaxlie.SE3.from_matrix(self.tool_pose_matrix(q))
        r = T.translation()
        Q = T.rotation().as_quaternion_xyzw()
        return r, Q
        # return pose_from_pos_quat(r, Q)

    def tool_velocity(self, q, v):
        """Calculate velocity at the tool with given joint state.

        x = [q, dq] is the joint state.
        """
        # NOTE it is much faster to let pybullet do this, so do that unless
        # there is good reason not too
        return pose_to_pos_quat(self.jacobian(q) @ v)

    def tool_acceleration(self, x, u):
        """Calculate acceleration at the tool with given joint state.

        x = [q, dq] is the joint state.
        """
        q, v = x[: self.ni], x[self.ni :]
        return self.jacobian(q) @ u + v @ self.dJdq(q) @ v

    def tangent(self, x, u):
        """Tangent vector dx = f(x, u)."""
        B = block_diag(rot2d(x[2], np=jnp), jnp.eye(7))
        return jnp.concatenate((x[self.ni :], B @ u))

    def simulate(self, x, u):
        """Forward simulate the model."""
        # TODO not sure if I can somehow use RK4 for part and not for
        # all---we'd have to split base and arm
        k1 = self.tangent(x, u)
        k2 = self.tangent(x + self.dt * k1 / 2, u)
        k3 = self.tangent(x + self.dt * k2 / 2, u)
        k4 = self.tangent(x + self.dt * k3, u)
        return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
