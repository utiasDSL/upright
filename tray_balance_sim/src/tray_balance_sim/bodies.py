import numpy as np
import pybullet as pyb

import tray_balance_constraints as con
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
    def __init__(self, mass, mu, height, collision_uid, visual_uid):
        self.mass = mass
        self.mu = mu
        self.height = height
        self.collision_uid = collision_uid
        self.visual_uid = visual_uid

    def add_to_sim(self, position, orientation=(0, 0, 0, 1)):
        """Actually add the object to the simulation."""
        self.uid = pyb.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=self.collision_uid,
            baseVisualShapeIndex=self.visual_uid,
            basePosition=position,
            baseOrientation=orientation,
        )

        # set friction
        # I do not set a spinning friction coefficient here directly, but let
        # Bullet handle this internally
        pyb.changeDynamics(self.uid, -1, lateralFriction=self.mu)

    def get_pose(self):
        """Get the pose of the object in the simulation."""
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)

    def reset_pose(self, position=None, orientation=None):
        """Reset the pose of the object in the simulation."""
        current_pos, current_orn = self.get_pose()
        if position is None:
            position = current_pos
        if orientation is None:
            orientation = current_orn
        pyb.resetBasePositionAndOrientation(self.uid, list(position), list(orientation))

    @staticmethod
    def cylinder(mass, mu, radius, height, color=(0, 0, 1, 1)):
        """Construct a cylinder object."""
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
        return BulletBody(
            mass=mass,
            mu=mu,
            height=height,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
        )

    @staticmethod
    def cuboid(mass, mu, side_lengths, color=(0, 0, 1, 1)):
        """Construct a cuboid object."""
        half_extents = tuple(0.5 * np.array(side_lengths))
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
        )
        return BulletBody(
            mass=mass,
            mu=mu,
            height=side_lengths[2],
            collision_uid=collision_uid,
            visual_uid=visual_uid,
        )

    @staticmethod
    def fromdict(d):
        """Construct the object from a dictionary."""
        if d["shape"] == "cylinder":
            return BulletBody.cylinder(
                mass=d["mass"],
                mu=d["mu"],
                radius=d["radius"],
                height=d["height"],
                color=d["color"],
            )
        elif d["shape"] == "cuboid":
            return BulletBody.cuboid(
                mass=d["mass"],
                mu=d["mu"],
                side_lengths=d["side_lengths"],
                color=d["color"],
            )
        else:
            raise ValueError(f"Unrecognized object shape {d['shape']}")
