import copy

import numpy as np
import pybullet as pyb

from util import skew3


def cylinder_inertia_matrix(mass, radius, height):
    """Inertia matrix for cylinder aligned along z-axis."""
    xx = yy = mass * (3 * radius ** 2 + height ** 2) / 12
    zz = 0.5 * mass * radius ** 2
    return np.diag([xx, yy, zz])


def cuboid_inertia_matrix(mass, side_lengths):
    """Inertia matrix for a rectangular cuboid with side_lengths in (x, y, z)
    dimensions."""
    lx, ly, lz = side_lengths
    xx = ly ** 2 + lz ** 2
    yy = lx ** 2 + lz ** 2
    zz = lx ** 2 + ly ** 2
    return mass * np.diag([xx, yy, zz]) / 12.0


class RigidBody:
    """Rigid body parameters."""

    def __init__(self, mass, inertia, com):
        self.mass = mass
        self.inertia = np.array(inertia)
        self.com = np.array(com)  # relative to some reference point


def compose_bodies(bodies):
    """Compute dynamic parameters for a system of multiple rigid bodies."""
    mass = sum([body.mass for body in bodies])
    com = sum([body.mass * body.com for body in bodies]) / mass

    # parallel axis theorem to compute new inertia matrix
    inertia = np.zeros((3, 3))
    for body in bodies:
        r = body.com - com  # direction doesn't actually matter: it cancels out
        R = skew3(r, np=np)
        I_new = body.inertia - body.mass * R @ R
        inertia += I_new

    return RigidBody(mass, inertia, com)


class BulletBody:
    def __init__(self, mass, mu, collision_uid, visual_uid, position, orientation):
        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=position,
            baseOrientation=orientation,
        )

        # set friction
        pyb.changeDynamics(self.uid, -1, lateralFriction=mu)

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


class BalancedBody:
    """Rigid body balanced by the EE.

    com_height: height of the CoM from the bottom of the object
    r_tau: distance value used for max frictional torque calculation
    """

    def __init__(self, body, com_height, r_tau, support_area, mu):
        # dynamic parameters
        self.body = body

        # geometry
        self.com_height = com_height
        self.support_area = support_area

        # friction
        self.r_tau = r_tau
        self.mu = mu

    def copy(self):
        return copy.deepcopy(self)


class Cylinder(BalancedBody):
    def __init__(
        self,
        r_tau,
        support_area,
        mass,
        radius,
        height,
        mu,
        bullet_mu,
        color=(0, 0, 1, 1),
    ):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            height=height,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=radius,
            length=height,
            rgbaColor=color,
        )
        self.bullet = BulletBody(
            mass, bullet_mu, collision_uid, visual_uid, [0, 0, 2], [0, 0, 0, 1]
        )

        inertia = cylinder_inertia_matrix(mass, radius, height)
        body = RigidBody(mass, inertia, None)
        super().__init__(body, 0.5 * height, r_tau, support_area, mu)
