import numpy as np
import pybullet as pyb


class BalancedObject:
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


class Cylinder(BalancedObject):
    def __init__(self, mass=0.5, radius=0.25, height=0.01, mu=0.5, color=(0, 0, 1, 1)):
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
        super().__init__(mass, mu, collision_uid, visual_uid, [0, 0, 2], [0, 0, 0, 1])
