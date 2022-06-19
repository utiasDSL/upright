import os
from collections import deque
import numpy as np
import pybullet as pyb
import pybullet_data
from pyb_utils.frame import debug_frame_world

from upright_core import parsing, math
from upright_sim.robot import SimulatedRobot
from upright_sim.camera import VideoManager

import IPython


# TODO rename to something like BulletObject
class BulletBody:
    def __init__(self, mass, mu, height, collision_uid, visual_uid, com_offset=None):
        self.mass = mass
        self.mu = mu
        self.height = height
        self.collision_uid = collision_uid
        self.visual_uid = visual_uid
        self.com_offset = com_offset if com_offset is not None else np.zeros(3)

    def add_to_sim(self, position, orientation=(0, 0, 0, 1)):
        """Actually add the object to the simulation."""
        # baseInertialFramePosition is an offset of the inertial frame origin
        # (i.e., center of mass) from the centroid of the object
        # see <https://github.com/erwincoumans/bullet3/blob/d3b4c27db4f86e1853ff7d84185237c437dc8485/examples/pybullet/examples/shiftCenterOfMass.py>
        self.uid = pyb.createMultiBody(
            baseMass=self.mass,
            baseInertialFramePosition=tuple(self.com_offset),
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

    def change_color(self, rgba):
        pyb.changeVisualShape(self.uid, -1, rgbaColor=list(rgba))

    @staticmethod
    def cylinder(mass, mu, radius, height, com_offset=None, color=(0, 0, 1, 1)):
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
            com_offset=com_offset,
        )

    @staticmethod
    def cuboid(mass, mu, side_lengths, com_offset=None, color=(0, 0, 1, 1)):
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
            com_offset=com_offset,
        )

    @staticmethod
    def fromdict(d):
        """Construct the object from a dictionary."""
        com_offset = np.array(d["com_offset"]) if "com_offset" in d else np.zeros(3)
        if d["shape"] == "cylinder":
            return BulletBody.cylinder(
                mass=d["mass"],
                mu=d["mu"],
                radius=d["radius"],
                height=d["height"],
                color=d["color"],
                com_offset=com_offset,
            )
        elif d["shape"] == "cuboid":
            return BulletBody.cuboid(
                mass=d["mass"],
                mu=d["mu"],
                side_lengths=d["side_lengths"],
                color=d["color"],
                com_offset=com_offset,
            )
        else:
            raise ValueError(f"Unrecognized object shape {d['shape']}")


# TODO replace below dynamic obstacle implementation with this one
class BulletDynamicObstacle:
    def __init__(self, initial_position, radius=0.1, velocity=None):
        self.initial_position = initial_position
        self.velocity = velocity if velocity is not None else np.zeros(3)

        self.body = BulletBody.sphere(radius)
        self.body.add_to_sim()

        pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(self.velocity))

    def sample_position(self, t):
        """Sample the position of the object at a given time."""
        # assume constant velocity
        return self.initial_position + t * self.velocity

    def reset_pose(self, position=None, orientation=None):
        self.body.reset_pose(position=position, orientation=orientation)

    def reset_velocity(self, v):
        self.velocity = v
        pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(v))

    def step(self):
        # velocity needs to be reset at each step of the simulation to negate
        # the effects of gravity
        pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(self.velocity))


class DynamicObstacle:
    def __init__(self, initial_position, radius=0.1, velocity=None):
        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=(1, 0, 0, 1),
        )
        self.uid = pyb.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=collision_uid,
            baseVisualShapeIndex=visual_uid,
            basePosition=list(initial_position),
            baseOrientation=(0, 0, 0, 1),
        )
        self.initial_position = initial_position

        self.velocity = velocity
        if self.velocity is None:
            self.velocity = np.zeros(3)
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(self.velocity))

    def sample_position(self, t):
        """Sample the position of the object at a given time."""
        # assume constant velocity
        return self.initial_position + t * self.velocity

    def reset_pose(self, r, Q):
        pyb.resetBasePositionAndOrientation(self.uid, list(r), list(Q))

    def reset_velocity(self, v):
        self.velocity = v
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(v))

    def step(self):
        # velocity needs to be reset at each step of the simulation to negate
        # the effects of gravity
        pyb.resetBaseVelocity(self.uid, linearVelocity=list(self.velocity))


def sim_object_setup(r_ew_w, config):
    arrangement_name = config["arrangement"]
    arrangement = config["arrangements"][arrangement_name]
    object_configs = config["objects"]
    ee = object_configs["ee"]

    objects = {}
    for d in arrangement:
        obj_type = d["type"]
        obj_config = config["objects"][obj_type]
        obj = BulletBody.fromdict(obj_config)

        if "parent" in d:
            parent = objects[d["parent"]]
            parent_position, _ = parent.get_pose()
            position = parent_position
            position[2] += 0.5 * parent.height + 0.5 * obj.height

            # PyBullet calculates coefficient of friction between two
            # bodies by multiplying them. Thus, to achieve our actual
            # desired friction at the support we need to divide the desired
            # value by the parent value to get the simulated value.
            obj.mu = obj.mu / parent.mu
        else:
            position = r_ew_w + [0, 0, 0.5 * ee["height"] + 0.5 * obj.height]
            obj.mu = obj.mu / ee["mu"]

        if "offset" in d:
            position[:2] += parsing.parse_support_offset(d["offset"])

        obj.add_to_sim(position)
        obj_name = d["name"]
        if obj_name in objects:
            raise ValueError(f"Multiple simulation objects named {obj_name}.")
        objects[obj_name] = obj

    return objects


class BulletSimulation:
    def __init__(self, config, timestamp, cli_args):
        # convert milliseconds to seconds
        self.timestep = config["timestep"]
        self.duration = config["duration"]

        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
        pyb.setGravity(*config["gravity"])
        pyb.setTimeStep(self.timestep)

        pyb.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=42,
            cameraPitch=-35.8,
            cameraTargetPosition=[1.28, 0.045, 0.647],
        )

        # get rid of extra parts of the GUI
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        # setup ground plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.loadURDF("plane.urdf", [0, 0, 0])

        # setup obstacles
        if config["static_obstacles"]["enabled"]:
            obstacles_uid = pyb.loadURDF(
                parsing.parse_ros_path(config["urdf"]["obstacles"])
            )
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object

        self.robot = SimulatedRobot(config)
        self.robot.reset_joint_configuration(self.robot.home)

        # simulate briefly to let the robot settle down after being positioned
        self.settle(1.0)

        # setup balanced objects
        r_ew_w, Q_we = self.robot.link_pose()
        self.objects = sim_object_setup(r_ew_w, config)

        # mark frame at the initial position
        debug_frame_world(0.2, list(r_ew_w), orientation=Q_we, line_width=3)

        # video recording
        self.video_manager = VideoManager.from_config(
            video_name=cli_args.video, config=config, timestamp=timestamp, r_ew_w=r_ew_w
        )

        # ghost objects
        self.ghosts = []

        # used to change color when object goes non-statically stable
        self.static_stable = True

    def object_poses(self):
        """Get the pose (position and orientation) of every balanced object at
        the current instant.

        Useful for logging purposes."""
        n = len(self.objects)
        r_ow_ws = np.zeros((n, 3))
        Q_wos = np.zeros((n, 4))
        for i, obj in enumerate(self.objects.values()):
            r_ow_ws[i, :], Q_wos[i, :] = obj.get_pose()
        return r_ow_ws, Q_wos

    def settle(self, duration):
        """Run simulation while doing nothing.

        Useful to let objects settle to rest before applying control.
        """
        t = 0
        while t < 1.0:
            pyb.stepSimulation()
            t += self.timestep

    def step(self, t, step_robot=True):
        """Step the simulation forward one timestep."""
        if step_robot:
            self.robot.step(secs=self.timestep)

        for ghost in self.ghosts:
            ghost.update()

        self.video_manager.record(t)

        pyb.stepSimulation()

        return t + self.timestep
