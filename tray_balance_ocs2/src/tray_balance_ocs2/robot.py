import numpy as np
import pinocchio

import tray_balance_constraints as core
from tray_balance_ocs2 import bindings


class PinocchioRobot:
    def __init__(self, config):
        # dimensions
        self.dims = bindings.RobotDimensions()
        self.dims.q = config["dims"]["q"]  # num positions
        self.dims.v = config["dims"]["v"]  # num velocities
        self.dims.x = config["dims"]["x"]  # num states
        self.dims.u = config["dims"]["u"]  # num inputs

        # create the model
        urdf_path = core.parsing.parse_ros_path(config["urdf"])
        base_type = config["base_type"]
        if base_type == "fixed":
            self.model = pinocchio.buildModelFromUrdf(urdf_path)
            self.mapping = bindings.FixedBasePinocchioMapping(self.dims)
        elif base_type == "omnidirectional":
            root_joint = pinocchio.JointModelComposite(3)
            root_joint.addJoint(pinocchio.JointModelPX())
            root_joint.addJoint(pinocchio.JointModelPY())
            root_joint.addJoint(pinocchio.JointModelRZ())
            self.model = pinocchio.buildModelFromUrdf(urdf_path, root_joint)
            self.mapping = bindings.OmnidirectionalPinocchioMapping(self.dims)
        else:
            raise ValueError(f"Invalid base type {base_type}.")

        self.data = self.model.createData()

        self.tool_idx = self.model.getBodyId(config["tool_link_name"])

    def forward_qva(self, q, v=None, a=None):
        """Forward kinematics using (q, v, a) all in the world frame (i.e.,
        corresponding directly to the Pinocchio model."""
        if v is None:
            v = np.zeros(self.dims.v)
        if a is None:
            a = np.zeros(self.dims.v)

        assert q.shape == (self.dims.q,)
        assert v.shape == (self.dims.v,)
        assert a.shape == (self.dims.v,)

        pinocchio.forwardKinematics(self.model, self.data, q, v, a)
        pinocchio.updateFramePlacements(self.model, self.data)

    def forward_derivatives_qva(self, q, v, a=None):
        """Compute derivatives of the forward kinematics using (q, v, a) all in
        the world frame (i.e., corresponding directly to the Pinocchio
        model."""
        if a is None:
            a = np.zeros(self.dims.v)

        assert q.shape == (self.dims.q,)
        assert v.shape == (self.dims.v,)
        assert a.shape == (self.dims.v,)

        pinocchio.computeForwardKinematicsDerivatives(self.model, self.data, q, v, a)

    def forward(self, x, u=None):
        """Forward kinematics. Must be called before the link pose, velocity,
        or acceleration methods."""
        if u is None:
            u = np.zeros(self.dims.u)

        assert x.shape == (self.dims.x,)
        assert u.shape == (self.dims.u,)

        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward_qva(q, v, a)

    def forward_derivatives(self, x, u=None):
        if u is None:
            u = np.zeros(self.dims.u)

        assert x.shape == (self.dims.x,)
        assert u.shape == (self.dims.u,)

        # get values in Pinocchio coordinates
        q = self.mapping.get_pinocchio_joint_position(x)
        v = self.mapping.get_pinocchio_joint_velocity(x, u)
        a = self.mapping.get_pinocchio_joint_acceleration(x, u)

        self.forward_derivatives_qva(q, v, a)

    def link_pose(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        pose = self.data.oMf[link_idx]
        r = pose.translation
        Q = core.math.rot_to_quat(pose.rotation)
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

    def link_spatial_acceleration(self, link_idx=None):
        if link_idx is None:
            link_idx = self.tool_idx
        A = pinocchio.getFrameAcceleration(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return A.linear, A.angular

    def jacobian(self, q):
        # TODO is it correct to do it in this frame?
        return pinocchio.computeFrameJacobian(
            self.model,
            self.data,
            q,
            self.tool_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

    def link_velocity_derivatives(self, link_idx=None):
        """Compute derivative of link velocity with respect to q and v."""
        if link_idx is None:
            link_idx = self.tool_idx
        dVdq, dVdv = pinocchio.getFrameVelocityDerivatives(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return dVdq, dVdv

    def link_acceleration_derivatives(self, link_idx=None):
        """Compute derivative of link classical acceleration with respect to q, v, a."""
        dr, ω = self.link_velocity(link_idx=link_idx)
        dVdq, dVdv = self.link_velocity_derivatives(link_idx=link_idx)
        dAdq, dAdv, dAda = self.link_spatial_acceleration_derivatives(link_idx=link_idx)

        # derivative of the coriolis term
        ddrdq, dwdq = dVdq[:3, :], dVdq[3:, :]
        ddrdv, dwdv = dVdv[:3, :], dVdv[3:, :]
        dcdq = (np.cross(dwdq.T, dr) + np.cross(ω, ddrdq.T)).T
        dcdv = (np.cross(dwdv.T, dr) + np.cross(ω, ddrdv.T)).T

        # add the coriolis term to the spatial acceleration
        dAs_dq = dAdq + np.vstack((dcdq, np.zeros((3, self.dims.q))))
        dAs_dv = dAdv + np.vstack((dcdv, np.zeros((3, self.dims.v))))
        dAs_da = dAda

        return dAs_dq, dAs_dv, dAs_da

    def link_spatial_acceleration_derivatives(self, link_idx=None):
        """Compute derivative of link spatial acceleration with respect to q, v, a."""
        if link_idx is None:
            link_idx = self.tool_idx
        _, dAdq, dAdv, dAda = pinocchio.getFrameAccelerationDerivatives(
            self.model,
            self.data,
            link_idx,
            pinocchio.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )
        return dAdq, dAdv, dAda

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
