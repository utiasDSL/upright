import os

import numpy as np
import pinocchio
import rospack
import liegroups


# TODO build this from config
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
