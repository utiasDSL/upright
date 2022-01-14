import pybullet as pyb
import liegroups


def debug_frame_world(size, origin, orientation=(0, 0, 0, 1), line_width=1):
    C = liegroups.SO3.from_quaternion(orientation, ordering="xyzw")

    dx = C.dot([size, 0, 0])
    dy = C.dot([0, size, 0])
    dz = C.dot([0, 0, size])

    pyb.addUserDebugLine(
        origin,
        list(dx + origin),
        lineColorRGB=[1, 0, 0],
        lineWidth=line_width,
    )
    pyb.addUserDebugLine(
        origin,
        list(dy + origin),
        lineColorRGB=[0, 1, 0],
        lineWidth=line_width,
    )
    pyb.addUserDebugLine(
        origin,
        list(dz + origin),
        lineColorRGB=[0, 0, 1],
        lineWidth=line_width,
    )


def debug_frame(size, obj_uid, link_index):
    """Attach at a frame to a link for debugging purposes."""
    pyb.addUserDebugLine(
        [0, 0, 0],
        [size, 0, 0],
        lineColorRGB=[1, 0, 0],
        parentObjectUniqueId=obj_uid,
        parentLinkIndex=link_index,
    )
    pyb.addUserDebugLine(
        [0, 0, 0],
        [0, size, 0],
        lineColorRGB=[0, 1, 0],
        parentObjectUniqueId=obj_uid,
        parentLinkIndex=link_index,
    )
    pyb.addUserDebugLine(
        [0, 0, 0],
        [0, 0, size],
        lineColorRGB=[0, 0, 1],
        parentObjectUniqueId=obj_uid,
        parentLinkIndex=link_index,
    )


class GhostSphere:
    def __init__(self, radius, position, color):
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        self.uid = pyb.createMultiBody(
            baseMass=0,  # non-dynamic body has mass = 0
            baseVisualShapeIndex=visual_uid,
            basePosition=list(position),
            baseOrientation=(0, 0, 0, 1),
        )

    def set_position(self, position):
        pyb.resetBasePositionAndOrientation(self.uid, list(position), (0, 0, 0, 1))


# def draw_sphere(radius, position, color, mass=0):
#     visual_uid = pyb.createVisualShape(
#         shapeType=pyb.GEOM_SPHERE,
#         radius=radius,
#         rgbaColor=color,
#     )
#     uid = pyb.createMultiBody(
#         baseMass=mass,  # non-dynamic body has mass = 0
#         baseVisualShapeIndex=visual_uid,
#         basePosition=list(position),
#         baseOrientation=(0, 0, 0, 1),
#     )
#     return uid
