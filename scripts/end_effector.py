import numpy as np
import pybullet as pyb
from util import quat_multiply, quat_from_axis_angle, zoh


COLOR = (1, 0, 0, 1)


class EndEffector:
    """End effector without a robot."""

    def __init__(
        self, position=(0, 0, 0), orientation=(0, 0, 0, 1), radius=0.03
    ):

        # positions of each finger relative to center
        # TODO calculate this directly instead of magic numbers
        shift1 = [0.1732, 0, 0]
        shift2 = [-0.0866, 0.15, 0]
        shift3 = [-0.0866, -0.15, 0]

        collision_uid = pyb.createCollisionShapeArray(
            shapeTypes=[pyb.GEOM_SPHERE, pyb.GEOM_SPHERE, pyb.GEOM_SPHERE],
            radii=[radius, radius, radius],
            collisionFramePositions=[shift1, shift2, shift3],
        )
        visual_uid = pyb.createVisualShapeArray(
            shapeTypes=[pyb.GEOM_SPHERE, pyb.GEOM_SPHERE, pyb.GEOM_SPHERE],
            radii=[radius, radius, radius],
            rgbaColors=[COLOR] * 3,
            visualFramePositions=[shift1, shift2, shift3],
        )

        # mass is zero to specify non-dynamic object
        self.uid = pyb.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=list(position),
            baseOrientation=list(orientation),
        )

        # set friction
        pyb.changeDynamics(self.uid, -1, lateralFriction=1.0)

    def get_pose(self):
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)

    def get_pose_planar(self):
        """Pose in the vertical x-z plane."""
        pos, orn = self.get_pose()
        pitch = pyb.getEulerFromQuaternion(orn)[1]
        return np.array([pos[0], pos[2], pitch])  # x, y, pitch

    def reset_pose(self, position=None, orientation=None):
        current_pos, current_orn = self.get_pose()
        if position is None:
            position = current_pos
        if orientation is None:
            orientation = current_orn
        pyb.resetBasePositionAndOrientation(self.uid, list(position), list(orientation))

    def get_velocity(self):
        v, ω = pyb.getBaseVelocity(self.uid)
        return v, ω

    def get_state(self):
        p, q = self.get_pose()
        v, ω = self.get_velocity()
        return np.concatenate((p, q, v, ω))

    def command_velocity(self, V):
        """Command the EE velocity twist."""
        pyb.resetBaseVelocity(self.uid, V[:3], V[3:])

    def command_acceleration(self, A):
        """Command the EE acceleration twist."""
        _, V = self.joint_states()
        self.cmd_vel = V
        self.cmd_acc = A

    def step(self):
        """One step of the physics engine."""
        self.cmd_vel += self.dt * self.cmd_acc
        self.command_velocity(self.cmd_vel)


class EndEffectorModel:
    """Model of floating end effector"""
    def __init__(self, dt):
        self.dt = dt

        Z = np.zeros((3, 3))
        A = np.block([[Z, np.eye(3)], [Z, Z]])
        B = np.block([[Z], [np.eye(3)]])

        # compute discretized matrices assuming zero-order hold for the linear
        # component
        self.Adp, self.Bdp = zoh(A, B, dt)

    def simulate(self, x, u):
        """Forward simulate the model.

        The state is x = [P, V] and input is acceleration.
        """
        P0, V0 = x[:7], x[7:]
        r0, q0 = P0[:3], P0[3:]
        v0, ω0 = V0[:3], V0[3:]
        a, α = u[:3], u[3:]

        # Linear part is integrated analytically (since it is linear!)
        xp0 = np.concatenate((r0, v0))
        xpf = self.Adp @ xp0 + self.Bdp @ a
        rf, vf = xpf[:3], xpf[3:]

        # Rotational part is approximated with a single Magnus expansion term,
        # which is essentially midpoint rule
        ωf = ω0 + self.dt * α
        aa = 0.5 * self.dt * (ω0 + ωf)
        Δq = quat_from_axis_angle(aa)
        qf = quat_multiply(Δq, q0)

        return np.concatenate((rf, qf, vf, ωf))
