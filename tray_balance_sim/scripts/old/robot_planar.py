import numpy as np
import jax
import jax.numpy as jnp

from util import dhtf, zoh
from robot import SimulatedRobot


class SimulatedPlanarRobot(SimulatedRobot):
    """Planar robot restricted to 4 of 9 DOFs, in the x-z plane."""

    def __init__(self, dt, qd, position=(0, 0, 0), orientation=(0, 0, 0, 1)):
        super().__init__(dt, position, orientation)

        self.input_mask = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0]).astype(bool)
        self.ni = np.sum(self.input_mask)
        self.nz = len(self.input_mask) - self.ni

        self.qd = qd
        self.K = np.eye(self.nz)

    def joint_states(self):
        q, v = super().joint_states()
        return q[self.input_mask], v[self.input_mask]

    def command_velocity(self, u):
        """Command the velocity of the robot's joints.

        Do not call this directly if using acceleration commands. It will
        automatically be called by the step function.
        """
        u9 = np.zeros(9)
        u9[self.input_mask] = u

        # ignored joints have feedback to keep them at desired position
        q, _ = super().joint_states()
        Δq = self.qd - q
        u9[~self.input_mask] = self.K @ Δq[~self.input_mask]

        super().command_velocity(u9)


class PlanarRobotModel:
    def __init__(self, dt, qd):
        self.dt = dt
        self.ni = 4
        self.qd = qd

        Z = np.zeros((self.ni, self.ni))
        A = np.block([[Z, np.eye(self.ni)], [Z, Z]])
        B = np.block([[Z], [np.eye(self.ni)]])

        # compute discretized matrices assuming zero-order hold
        self.Ad, self.Bd = zoh(A, B, dt)

        self._init_kinematics(qd)

    def _init_kinematics(self, qd):
        px = 0.27
        py = 0.01
        pz = 0.653
        d1 = 0.1273
        d5 = 0.1157
        d6 = 0.0922
        d7 = 0.290

        self.T_w_0 = dhtf(np.pi / 2, 0, 0, np.pi / 2)
        self.T_0_1 = (
            dhtf(np.pi / 2, 0, qd[1], np.pi / 2)
            @ dhtf(qd[2], 0, 0, 0)
            @ dhtf(0, px, pz, -np.pi / 2)
            @ dhtf(0, 0, py, np.pi / 2)
            @ dhtf(qd[3], 0, d1, np.pi / 2)
        )
        self.T_3_tool = (
            dhtf(qd[7], 0, d5, -np.pi / 2)
            @ dhtf(qd[8], 0, d6, 0)
            @ dhtf(0, 0, d7, 0)
        )

        # auto-diff to get Jacobian
        self.jacobian = jax.jit(jax.jacrev(self.tool_pose))
        self.dJdq = jax.jit(jax.jacfwd(self.jacobian))

    def tool_pose(self, q):
        a2 = -0.612
        a3 = -0.5723
        d4 = 0.163941

        T_q0 = dhtf(np.pi / 2, 0, q[0], np.pi / 2)
        T_q1 = dhtf(q[1], a2, 0, 0)
        T_q2 = dhtf(q[2], a3, 0, 0)
        T_q3 = dhtf(q[3], 0, d4, np.pi / 2)

        T_w_tool = self.T_w_0 @ T_q0 @ self.T_0_1 @ T_q1 @ T_q2 @ T_q3 @ self.T_3_tool

        x = T_w_tool[0, 3]
        z = T_w_tool[2, 3]

        # TODO not sure why positive angle works
        # angle is negative pitch since y-axis points into the plane
        # see: https://github.com/utiasSTARS/liegroups/blob/master/liegroups/numpy/so3.py#L310
        pitch = jnp.arctan2(-T_w_tool[2, 0],
                            jnp.sqrt(T_w_tool[0, 0]**2 + T_w_tool[1, 0]**2))
        θ = pitch

        return jnp.array([x, z, θ])

    def tool_velocity(self, x):
        """Calculate velocity at the tool with given joint state.

        x = [q, dq] is the joint state.
        """
        q, dq = x[:self.ni], x[self.ni:]
        return self.jacobian(q) @ dq

    def tool_acceleration(self, x, u):
        """Calculate acceleration at the tool with given joint state.

        x = [q, dq] is the joint state.
        """
        q, dq = x[:self.ni], x[self.ni:]
        return self.jacobian(q) @ u + dq @ self.dJdq(q) @ dq

    # def tool_state(self, x):
    #     """Calculate the state [p, v] of the tool."""
    #     return jnp.concatenate((self.tool_pose(x), self.tool_velocity(x)))

    def simulate(self, x, u):
        """Forward simulate the model."""
        x_new = self.Ad @ x + self.Bd @ u
        return x_new
