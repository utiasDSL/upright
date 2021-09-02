import numpy as np
import jax
import jax.numpy as jnp
from jax.scipy.linalg import block_diag
import pybullet as pyb
from liegroups import SO3
import jaxlie

from util import dhtf, rot2d, pose_from_pos_quat, pose_to_pos_quat, skew3

# TODO ideally we wouldn't be using both jaxlie and liegroups


UR10_JOINT_NAMES = [
    "ur10_arm_shoulder_pan_joint",
    "ur10_arm_shoulder_lift_joint",
    "ur10_arm_elbow_joint",
    "ur10_arm_wrist_1_joint",
    "ur10_arm_wrist_2_joint",
    "ur10_arm_wrist_3_joint",
]

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
        self.uid = pyb.loadURDF(
            "../assets/urdf/mm.urdf",
            position,
            orientation,
        )

        self.dt = dt

        self.cmd_vel = np.zeros(9)
        self.cmd_acc = np.zeros_like(self.cmd_vel)

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

    def _base_rotation_matrix(self):
        """Get rotation matrix for the base.

        This is just the rotation about the z-axis by the yaw angle.
        """
        base_pose, _ = self._base_state()
        yaw = base_pose[2]
        C_wb = SO3.rotz(yaw)
        return C_wb

    def _command_base_velocity(self, ub, bodyframe=True):
        """Command base velocity.

        The input ub = [vx, vy, wz] is in body coordinates, unless
        bodyframe=False. Then it is in world coordinates.
        """
        # map from body coordinates to world coordinates for pybullet
        if bodyframe:
            C_wb = self._base_rotation_matrix()
            linear = C_wb.dot([ub[0], ub[1], 0])
        else:
            linear = [ub[0], ub[1], 0]

        angular = [0, 0, ub[2]]
        pyb.resetBaseVelocity(self.uid, linear, angular)

    def command_velocity(self, u, bodyframe=True):
        """Command the velocity of the robot's joints."""
        self._command_base_velocity(u[:3], bodyframe=bodyframe)
        self._command_arm_velocity(u[3:])

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
        self.command_velocity(self.cmd_vel, bodyframe=False)

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

    def tool_velocity(self):
        q, v = self.joint_states()
        J = self.jacobian(q)
        V = J @ v
        return V[:3], V[3:]

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


class KinematicChain:
    """All transforms on the robot kinematic chain for a given configuration."""

    T_w_b = dhtf(np.pi / 2, 0, 0, np.pi / 2)
    T_θb_θ1 = dhtf(0, PX, PZ, -np.pi / 2) @ dhtf(0, 0, PY, np.pi / 2)
    T_θ6_tool = dhtf(0, 0, D7, 0)

    def __init__(self, q):
        self.T_xb = dhtf(np.pi / 2, 0, q[0], np.pi / 2)
        self.T_yb = dhtf(np.pi / 2, 0, q[1], np.pi / 2)
        self.T_θb = dhtf(q[2], 0, 0, 0)

        self.T_θ1 = dhtf(q[3], 0, D1, np.pi / 2)
        self.T_θ2 = dhtf(q[4], A2, 0, 0)
        self.T_θ3 = dhtf(q[5], A3, 0, 0)
        self.T_θ4 = dhtf(q[6], 0, D4, np.pi / 2)
        self.T_θ5 = dhtf(q[7], 0, D5, -np.pi / 2)
        self.T_θ6 = dhtf(q[8], 0, D6, 0)

        self.T_w_xb = self.T_w_b @ self.T_xb
        self.T_w_yb = self.T_w_xb @ self.T_yb
        self.T_w_θb = self.T_w_yb @ self.T_θb

        self.T_w_θ1 = self.T_w_θb @ self.T_θb_θ1 @ self.T_θ1
        self.T_w_θ2 = self.T_w_θ1 @ self.T_θ2
        self.T_w_θ3 = self.T_w_θ2 @ self.T_θ3
        self.T_w_θ4 = self.T_w_θ3 @ self.T_θ4
        self.T_w_θ5 = self.T_w_θ4 @ self.T_θ5
        self.T_w_θ6 = self.T_w_θ5 @ self.T_θ6

        self.T_w_tool = self.T_w_θ6 @ self.T_θ6_tool


class RobotModel:
    def __init__(self, dt, qd):
        self.dt = dt
        self.ni = 9

        self.dJdq = jax.jit(jax.jacfwd(self.jacobian))

    def jacobian(self, q):
        """Compute geometric Jacobian."""

        def rotation(T):
            return T[:3, :3]

        def translation(T):
            return T[:3, 3]

        chain = KinematicChain(q)
        z0 = jnp.array([0, 0, 1])  # Unit vector along z-axis

        # axis for each joint's angular velocity is the z-axis of the previous
        # transform
        z_xb = rotation(chain.T_w_xb) @ z0
        z_yb = rotation(chain.T_w_yb) @ z0
        z_θb = rotation(chain.T_w_θb) @ z0
        z_θ1 = rotation(chain.T_w_θ1) @ z0
        z_θ2 = rotation(chain.T_w_θ2) @ z0
        z_θ3 = rotation(chain.T_w_θ3) @ z0
        z_θ4 = rotation(chain.T_w_θ4) @ z0
        z_θ5 = rotation(chain.T_w_θ5) @ z0
        z_θ6 = rotation(chain.T_w_θ6) @ z0

        # Angular Jacobian
        # joints xb and yb are prismatic, and so cause no angular velocity.
        Jo = jnp.vstack((
            jnp.zeros(3), jnp.zeros(3), z_θb, z_θ1, z_θ2, z_θ3, z_θ4, z_θ5, z_θ6
        )).T

        # Linear Jacobian
        pe = translation(chain.T_w_tool)
        Jp = jnp.vstack((
            z_xb,
            z_yb,
            jnp.cross(z_θb, pe - translation(chain.T_w_θb)),
            jnp.cross(z_θ1, pe - translation(chain.T_w_θ1)),
            jnp.cross(z_θ2, pe - translation(chain.T_w_θ2)),
            jnp.cross(z_θ3, pe - translation(chain.T_w_θ3)),
            jnp.cross(z_θ4, pe - translation(chain.T_w_θ4)),
            jnp.cross(z_θ5, pe - translation(chain.T_w_θ5)),
            jnp.cross(z_θ6, pe - translation(chain.T_w_θ6)),
        )).T

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
        # q, dq = x[: self.ni], x[self.ni :]
        # J = self.jacobian(q)
        # print(J.shape)
        # print(dq.shape)
        # TODO this would be much faster if we let pybullet do it
        return pose_to_pos_quat(self.jacobian(q) @ v)

    def tool_acceleration(self, x, u):
        """Calculate acceleration at the tool with given joint state.

        x = [q, dq] is the joint state.
        """
        q, dq = x[: self.ni], x[self.ni :]
        return self.jacobian(q) @ u + dq @ self.dJdq(q) @ dq

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
