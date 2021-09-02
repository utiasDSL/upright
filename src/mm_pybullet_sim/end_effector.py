import numpy as np
import jax.numpy as jnp
import pybullet as pyb
from jaxlie import SO3

from mm_pybullet_sim.util import zoh
from mm_pybullet_sim.geometry import equilateral_triangle_inscribed_radius


COLOR = (1, 0, 0, 1)


def ee_points(side_length):
    """Compute points of the end effector fingers."""
    in_radius = equilateral_triangle_inscribed_radius(side_length)
    out_radius = 2 * in_radius  # radius of circumscribed circle

    p1 = [out_radius, 0, 0]
    p2 = [-in_radius, 0.5 * side_length, 0]
    p3 = [-in_radius, -0.5 * side_length, 0]

    return p1, p2, p3


class EndEffector:
    """End effector without a robot."""

    def __init__(
        self,
        dt,
        position=(0, 0, 0),
        orientation=(0, 0, 0, 1),
        side_length=0.3,
        radius=0.03,
    ):
        self.dt = dt

        # positions of each finger relative to center
        shift1, shift2, shift3 = ee_points(side_length)

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
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)  # xyzw
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
        v, ω = self.get_velocity()
        self.cmd_vel = np.concatenate((v, ω))
        self.cmd_acc = A

    def step(self):
        """One step of the physics engine."""
        self.cmd_vel += self.dt * self.cmd_acc
        self.command_velocity(self.cmd_vel)


class EndEffectorModel:
    """Model of floating end effector"""

    def __init__(self, dt):
        self.dt = dt
        self.ni = 6
        self.ns = 7 + 6

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
        xp0 = jnp.concatenate((r0, v0))
        xpf = self.Adp @ xp0 + self.Bdp @ a
        rf, vf = xpf[:3], xpf[3:]

        # Rotational part is approximated with a single Magnus expansion term,
        # which is essentially midpoint rule
        ωf = ω0 + self.dt * α
        aa = 0.5 * self.dt * (ω0 + ωf)

        R0 = SO3.from_quaternion_xyzw(q0)
        ΔR = SO3.exp(aa)
        qf = ΔR.multiply(R0).as_quaternion_xyzw()

        # The above is equivalent to (but safer than, and written for jax):
        # Δq = quat_from_axis_angle(aa, np=jnp)
        # qf = quat_multiply(Δq, q0, np=jnp)

        return jnp.concatenate((rf, qf, vf, ωf))
