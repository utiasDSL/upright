import os

import numpy as np
import pybullet as pyb
import pinocchio
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

        self.dt = config["timestep"]
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

    def step(self):
        """One step of the physics engine."""
        # input (acceleration) and velocity are both in the body frame
        self.cmd_vel += self.dt * self.cmd_acc
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


class PinocchioRobot:
    def __init__(self):
        rospack = rospkg.RosPack()
        urdf_path = os.path.join(
            rospack.get_path("tray_balance_assets"), "urdf", "mm_ocs2.urdf"
        )

        root_joint = pinocchio.JointModelComposite(3)
        root_joint.addJoint(pinocchio.JointModelPX())
        root_joint.addJoint(pinocchio.JointModelPY())
        root_joint.addJoint(pinocchio.JointModelRZ())
        self.model = pinocchio.buildModelFromUrdf(urdf_path, root_joint)

        self.nq = self.model.nq
        self.nv = self.model.nv
        self.ns = self.nq + self.nv
        self.ni = self.nv

        self.data = self.model.createData()

        self.tool_idx = self.model.getBodyId("thing_tool")

    def forward_qva(self, q, v=None, a=None):
        """Forward kinematics using (q, v, a) all in the world frame (i.e.,
        corresponding directly to the Pinocchio model."""
        if v is None:
            v = np.zeros(self.nv)
        if a is None:
            a = np.zeros(self.ni)

        pinocchio.forwardKinematics(self.model, self.data, q, v, a)
        pinocchio.updateFramePlacements(self.model, self.data)

    def forward(self, x, u=None):
        """Forward kinematics. Must be called before the link pose, velocity,
        or acceleration methods."""
        q, v = x[: self.nq], x[self.nq :]
        assert v.shape[0] == self.nv
        if u is None:
            u = np.zeros(self.ni)

        # rotate input into the world frame
        a = np.copy(u)
        C_wb = liegroups.SO3.rotz(q[2]).as_matrix()
        a[:3] = C_wb @ u[:3]

        self.forward_qva(q, v, a)

    def link_pose(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        pose = self.data.oMf[link_idx]
        r = pose.translation
        Q = liegroups.SO3(pose.rotation).to_quaternion(ordering="xyzw")
        return r.copy(), Q.copy()

    def link_velocity(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        V = pinocchio.getFrameVelocity(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return V.linear, V.angular

    def link_acceleration(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        A = pinocchio.getFrameClassicalAcceleration(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return A.linear, A.angular

    def jacobian(self, q):
        return pinocchio.computeFrameJacobian(self.model, self.data, q, self.tool_idx)

    # TODO need to update these methods from previous iteration of the model
    # def tangent(self, x, u):
    #     """Tangent vector dx = f(x, u)."""
    #     B = block_diag(rot2d(x[2], np=jnp), jnp.eye(7))
    #     return jnp.concatenate((x[self.ni :], B @ u))
    #
    # def simulate(self, x, u):
    #     """Forward simulate the model."""
    #     # TODO not sure if I can somehow use RK4 for part and not for
    #     # all---we'd have to split base and arm
    #     k1 = self.tangent(x, u)
    #     k2 = self.tangent(x + self.dt * k1 / 2, u)
    #     k3 = self.tangent(x + self.dt * k2 / 2, u)
    #     k4 = self.tangent(x + self.dt * k3, u)
    #     return x + self.dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
