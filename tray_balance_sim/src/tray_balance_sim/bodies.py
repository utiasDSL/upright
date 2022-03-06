import numpy as np
import pybullet as pyb

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


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


class BulletBody:
    def __init__(
        self, mass, mu, r_tau, collision_uid, visual_uid, position, orientation
    ):
        self.uid = pyb.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=position,
            baseOrientation=orientation,
        )

        # set friction
        self.mu = mu
        pyb.changeDynamics(self.uid, -1, lateralFriction=mu, spinningFriction=r_tau)

    def get_pose(self):
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)

    def get_pose_planar(self):
        """Pose in the vertical x-z plane."""
        pos, orn = self.get_pose()
        pitch = pyb.getEulerFromQuaternion(orn)[1]
        return np.array([pos[0], pos[2], pitch])  # x, z, pitch

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

    def __init__(self, mass, inertia, com_height, r_tau, support_area, mu):
        # dynamic parameters
        self.mass = mass
        self.inertia = inertia
        self.com = None

        # geometry
        self.com_height = com_height
        self.support_area = support_area

        # friction
        self.r_tau = r_tau
        self.mu = mu

        # EXPERIMENTAL: add extra mass to the controller that is not really
        # there in simulation
        self.mass_error = 0
        self.r_tau_error = 0
        self.mu_error = 0

        # TODO these should be dependent
        self.com_error = np.zeros(3)
        self.com_height_error = 0

        self.children = []

    def convert_to_ocs2(self):
        body = ocs2.RigidBody(
            self.mass + self.mass_error, self.inertia, self.com + self.com_error
        )
        return ocs2.BalancedObject(
            body,
            self.com_height + self.com_height_error,
            self.support_area,
            self.r_tau + self.r_tau_error,
            self.mu + self.mu_error,  # - 0.01,
        )


class Cylinder(BalancedBody):
    """Balanced cylindrical object.

    mu and bullet_mu may be different in general. mu is the coefficient of
    friction betweenthis object and its support surface that we want to
    simulate. bullet_mu is the value to set in Bullet to make that true, since
    Bullet calculates the coefficient for a contact by multiplying those for
    each object.
    """

    def __init__(self, r_tau, support_area, mass, radius, height, mu):
        self.radius = radius
        self.height = height
        inertia = cylinder_inertia_matrix(mass, radius, height)

        super().__init__(mass, inertia, 0.5 * height, r_tau, support_area, mu)

    def add_to_sim(self, bullet_mu, color=(0, 0, 1, 1)):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=self.radius,
            height=self.height,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_CYLINDER,
            radius=self.radius,
            length=self.height,
            rgbaColor=color,
        )
        self.bullet = BulletBody(
            self.mass,
            bullet_mu,
            self.r_tau,
            collision_uid,
            visual_uid,
            [0, 0, 2],
            [0, 0, 0, 1],
        )


class Cuboid(BalancedBody):
    def __init__(self, r_tau, support_area, mass, side_lengths, mu):
        self.side_lengths = np.array(side_lengths)
        inertia = cuboid_inertia_matrix(mass, side_lengths)
        super().__init__(mass, inertia, 0.5 * side_lengths[2], r_tau, support_area, mu)

    def add_to_sim(self, bullet_mu, color=(0, 0, 1, 1)):
        half_extents = tuple(0.5 * self.side_lengths)
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )
        self.bullet = BulletBody(
            self.mass,
            bullet_mu,
            self.r_tau,
            collision_uid,
            visual_uid,
            [0, 0, 2],
            [0, 0, 0, 1],
        )
