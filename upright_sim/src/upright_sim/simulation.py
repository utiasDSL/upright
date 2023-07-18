import os
from collections import deque
import numpy as np
import pybullet as pyb
import pybullet_data
from pyb_utils.frame import debug_frame_world
from mobile_manipulation_central.simulation import BulletSimulation

import upright_core as core
from upright_core import parsing, math, polyhedron
from upright_sim.robot import UprightSimulatedRobot
from upright_sim.camera import camera_from_dict, VideoManager
from upright_sim.util import wedge_mesh

import IPython


# TODO rename to something like BulletObject
class BulletBody:
    def __init__(
        self,
        mass,
        mu,
        box,
        collision_uid,
        visual_uid,
        position=None,
        orientation=None,
        com_offset=None,
        inertial_orientation=None,
        local_inertia_diagonal=None,
    ):
        self.mass = mass
        self.mu = mu
        self.box = box
        self.collision_uid = collision_uid
        self.visual_uid = visual_uid

        if position is None:
            position = np.zeros(3)
        self.r0 = position

        if orientation is None:
            orientation = np.array([0, 0, 0, 1])
        self.q0 = orientation

        if com_offset is None:
            com_offset = np.zeros(3)
        self.com_offset = com_offset

        # we need to get the box's orientation correct here for accurate height
        # computation
        self.box = self.box.transform(rotation=math.quat_to_rot(self.q0))

        if inertial_orientation is None:
            inertial_orientation = np.array([0, 0, 0, 1])
        self.inertial_orientation = inertial_orientation
        self.local_inertia_diagonal = local_inertia_diagonal

    @property
    def height(self):
        return self.box.height()

    def add_to_sim(self):
        """Actually add the object to the simulation."""
        # baseInertialFramePosition is an offset of the inertial frame origin
        # (i.e., center of mass) from the centroid of the object
        # see <https://github.com/erwincoumans/bullet3/blob/d3b4c27db4f86e1853ff7d84185237c437dc8485/examples/pybullet/examples/shiftCenterOfMass.py>
        self.uid = pyb.createMultiBody(
            baseMass=self.mass,
            baseInertialFramePosition=tuple(self.com_offset),
            baseInertialFrameOrientation=tuple(self.inertial_orientation),
            baseCollisionShapeIndex=self.collision_uid,
            baseVisualShapeIndex=self.visual_uid,
            basePosition=tuple(self.r0),
            baseOrientation=tuple(self.q0),
        )

        # update bounding polyhedron with actual position
        self.box = self.box.transform(translation=self.r0)

        # set friction
        # I do not set a spinning friction coefficient here directly, but let
        # Bullet handle this internally
        pyb.changeDynamics(self.uid, -1, lateralFriction=self.mu)

        # set local inertia if needed (required for objects built from meshes)
        if self.local_inertia_diagonal is not None:
            pyb.changeDynamics(
                self.uid, -1, localInertiaDiagonal=self.local_inertia_diagonal
            )

    def get_pose(self):
        """Get the pose of the object in the simulation."""
        pos, orn = pyb.getBasePositionAndOrientation(self.uid)
        return np.array(pos), np.array(orn)

    def get_velocity(self):
        v, ω = pyb.getBaseVelocity(self.uid)
        return np.array(v), np.array(ω)

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
    def cylinder(
        mass, mu, radius, height, orientation=None, com_offset=None, color=(0, 0, 1, 1)
    ):
        """Construct a cylinder object."""
        if orientation is None:
            orientation = np.array([0, 0, 0, 1])

        # for the cylinder, we rotate by 45 deg about z so that contacts occur
        # aligned with x-y axes
        qz = math.rot_to_quat(math.rotz(np.pi / 4))
        q = math.quat_multiply(orientation, qz)

        w = np.sqrt(2) * radius
        half_extents = 0.5 * np.array([w, w, height])
        box = polyhedron.ConvexPolyhedron.box(half_extents)

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
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            orientation=q,
            com_offset=com_offset,
        )

    @staticmethod
    def cuboid(
        mass,
        mu,
        side_lengths,
        orientation=None,
        com_offset=None,
        local_inertia_diagonal=None,
        color=(0, 0, 1, 1),
    ):
        """Construct a cuboid object."""
        half_extents = 0.5 * np.array(side_lengths)
        box = polyhedron.ConvexPolyhedron.box(half_extents)

        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_BOX,
            halfExtents=tuple(half_extents),
            rgbaColor=color,
        )
        return BulletBody(
            mass=mass,
            mu=mu,
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            orientation=orientation,
            com_offset=com_offset,
            local_inertia_diagonal=local_inertia_diagonal,
        )

    @staticmethod
    def sphere(
        mass,
        mu,
        radius,
        orientation=None,
        com_offset=None,
        local_inertia_diagonal=None,
        color=(0, 0, 1, 1),
    ):
        """Construct a cylinder object."""
        half_extents = np.ones(3) * radius / 2
        box = polyhedron.ConvexPolyhedron.box(half_extents)

        collision_uid = pyb.createCollisionShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
        )
        visual_uid = pyb.createVisualShape(
            shapeType=pyb.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
        )
        return BulletBody(
            mass=mass,
            mu=mu,
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            com_offset=com_offset,
            local_inertia_diagonal=local_inertia_diagonal,
        )

    @staticmethod
    def wedge(
        mass, mu, side_lengths, orientation=None, com_offset=None, color=(0, 0, 1, 1)
    ):
        half_extents = 0.5 * np.array(side_lengths)
        vertices, indices = wedge_mesh(half_extents)
        box = polyhedron.ConvexPolyhedron.wedge(half_extents)

        collision_uid = pyb.createCollisionShape(
            pyb.GEOM_MESH, vertices=vertices, indices=indices
        )
        visual_uid = pyb.createVisualShape(
            pyb.GEOM_MESH, vertices=vertices, indices=indices, rgbaColor=color
        )

        # compute local inertial frame position
        if com_offset is None:
            com_offset = np.zeros(3)
        hx, hy, hz = half_extents
        com_offset = com_offset + np.array([-hx / 3, 0, -hz / 3])

        # compute inertia and inertial frame orientation
        D, C = core.math.wedge_inertia_matrix(mass, side_lengths)
        local_inertia_diagonal = np.diag(D)
        inertial_orientation = core.math.rot_to_quat(C)

        return BulletBody(
            mass=mass,
            mu=mu,
            box=box,
            collision_uid=collision_uid,
            visual_uid=visual_uid,
            com_offset=com_offset,
            inertial_orientation=inertial_orientation,
            local_inertia_diagonal=local_inertia_diagonal,
        )

    @staticmethod
    def from_config(d, mu, orientation=None):
        """Construct the object from a dictionary."""
        com_offset = np.array(d.get("com_offset", (0, 0, 0)))
        local_inertia_diagonal = d.get("inertia_diag", None)

        if d["shape"] == "cylinder":
            return BulletBody.cylinder(
                mass=d["mass"],
                mu=mu,
                radius=d["radius"],
                height=d["height"],
                color=d["color"],
                orientation=orientation,
                com_offset=com_offset,
                local_inertia_diagonal=local_inertia_diagonal,
            )
        elif d["shape"] == "cuboid":
            return BulletBody.cuboid(
                mass=d["mass"],
                mu=mu,
                side_lengths=d["side_lengths"],
                color=d["color"],
                orientation=orientation,
                com_offset=com_offset,
                local_inertia_diagonal=local_inertia_diagonal,
            )
        elif d["shape"] == "wedge":
            if local_inertia_diagonal is not None:
                raise NotImplementedError(
                    "Manually setting inertia diagonal not supported for wedge."
                )
            return BulletBody.wedge(
                mass=d["mass"],
                mu=mu,
                side_lengths=d["side_lengths"],
                color=d["color"],
                orientation=orientation,
                com_offset=com_offset,
            )
        else:
            raise ValueError(f"Unrecognized object shape {d['shape']}")


class BulletDynamicObstacle:
    def __init__(
        self,
        times,
        positions,
        velocities,
        accelerations,
        radius=0.1,
        controlled=False,
        collides=True,
        color=(0, 0, 1, 1),
    ):
        self.start_time = None
        self._mode_idx = 0

        self.times = times
        self.positions = positions
        self.velocities = velocities
        self.accelerations = accelerations

        self.controlled = controlled
        self.collides = collides
        self.K = 10 * np.eye(3)  # position gain

        self.body = BulletBody.sphere(mass=1, mu=1, radius=radius, color=color)
        self.body.r0 = np.array(positions[0])

    @classmethod
    def from_config(cls, config, offset=None):
        """Parse obstacle properties from a config dict."""
        relative = config["relative"]
        if relative and offset is not None:
            offset = np.array(offset)
        else:
            offset = np.zeros(3)

        controlled = config["controlled"]
        collides = config.get("collides", True)
        color = config.get("color", (0, 0, 1, 1))

        times = []
        positions = []
        velocities = []
        accelerations = []
        for mode in config["modes"]:
            times.append(mode["time"])
            positions.append(np.array(mode["position"]) + offset)
            velocities.append(np.array(mode["velocity"]))
            accelerations.append(np.array(mode["acceleration"]))

        return cls(
            times=times,
            positions=positions,
            velocities=velocities,
            accelerations=accelerations,
            radius=config["radius"],
            controlled=controlled,
            collides=collides,
            color=color,
        )

    def _initial_mode_values(self):
        t = self.times[self._mode_idx]
        if self.start_time is not None:
            t += self.start_time
        r = self.positions[self._mode_idx]
        v = self.velocities[self._mode_idx]
        a = self.accelerations[self._mode_idx]
        return t, r, v, a

    def start(self, t0):
        """Add the obstacle to the simulation."""
        self.start_time = t0
        self.body.add_to_sim()

        if not self.collides:
            # make the obstacle not collide with anything else
            pyb.setCollisionFilterGroupMask(self.body.uid, -1, 0, 0)

        v0 = self._initial_mode_values()[2]
        pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(v0))

    def _desired_state(self, t):
        t0, r0, v0, a0 = self._initial_mode_values()
        dt = t - t0
        rd = r0 + dt * v0 + 0.5 * dt**2 * a0
        vd = v0 + dt * a0
        return rd, vd

    def joint_state(self):
        """Get the joint state (position, velocity) of the obstacle.

        If the obstacle has not yet been started, the nominal starting state is
        given.
        """
        if self.start_time is None:
            _, r, v, a = self._initial_mode_values()
        else:
            r = self.body.get_pose()[0]
            v = self.body.get_velocity()[0]
            a = self.accelerations[self._mode_idx]
        return r, v, a

    def step(self, t):
        """Step the object forward in time."""
        # no-op if obstacle hasn't been started
        reset = False
        if self.start_time is None:
            return reset

        # reset the obstacle if we've stepped into a new mode
        if self._mode_idx < len(self.times) - 1:
            if t - self.start_time >= self.times[self._mode_idx + 1]:
                self._mode_idx += 1
                _, r0, v0, _ = self._initial_mode_values()
                pyb.resetBasePositionAndOrientation(
                    self.body.uid, list(r0), [0, 0, 0, 1]
                )
                pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(v0))
                reset = True

        # velocity needs to be reset at each step of the simulation to negate
        # the effects of gravity
        if self.controlled:
            rd, vd = self._desired_state(t)
            r, _ = self.body.get_pose()
            cmd_vel = self.K @ (rd - r) + vd
            pyb.resetBaseVelocity(self.body.uid, linearVelocity=list(cmd_vel))
        return reset


class EEObject:
    def __init__(self, position, orientation, side_lengths):
        self.r0 = position
        self.side_lengths = side_lengths
        self.mu = 1.0
        self.box = polyhedron.ConvexPolyhedron.box(0.5 * self.side_lengths).transform(
            translation=self.r0,
            rotation=orientation,
        )

    @property
    def height(self):
        return self.side_lengths[2]


def balanced_object_setup(r_ew_w, Q_we, config, robot):
    arrangement_name = config["arrangement"]
    arrangement = config["arrangements"][arrangement_name]

    C_we = core.math.quat_to_rot(Q_we)

    # make "fake" EE object
    ee_config = config["objects"]["ee"]
    ee_position = r_ew_w + C_we @ ee_config["position"]
    ee_side_lengths = np.array(ee_config["side_lengths"])
    objects = {"ee": EEObject(ee_position, C_we, ee_side_lengths)}

    mus = parsing.parse_mu_dict(arrangement["contacts"], apply_margin=False)

    for obj_instance_conf in arrangement["objects"]:
        obj_name = obj_instance_conf["name"]
        if obj_name in objects:
            raise ValueError(f"Multiple simulation objects named {obj_name}.")

        obj_type = obj_instance_conf["type"]
        obj_type_conf = config["objects"][obj_type]

        Q_eo = obj_instance_conf.get("orientation", np.array([0, 0, 0, 1]))
        Q_eo = Q_eo / np.linalg.norm(Q_eo)
        Q_wo = math.quat_multiply(Q_we, Q_eo)

        parent_name = obj_instance_conf["parent"]
        parent = objects[parent_name]

        # fixtures are objects rigidly attached to the tray
        # they can provide additional support to the balanced objects
        fixture = "fixture" in obj_instance_conf and obj_instance_conf["fixture"]
        if fixture and parent_name != "ee":
            raise ValueError("Only objects with parent 'ee' can be fixtures.")

        # PyBullet calculates coefficient of friction between two bodies by
        # multiplying them. Thus, to achieve our actual desired friction at the
        # support we need to divide the desired value by the parent value to
        # get the simulated value.
        if fixture:
            pyb_mu = objects["ee"].mu
        else:
            real_mu = mus[parent_name][obj_name]
            pyb_mu = real_mu / parent.mu

        obj = BulletBody.from_config(obj_type_conf, mu=pyb_mu, orientation=Q_wo)

        # parse offset from parent (in EE frame)
        z = C_we @ np.array([0, 0, 1])
        d1 = parent.box.distance_from_centroid_to_boundary(z)
        d2 = obj.box.distance_from_centroid_to_boundary(-z)

        r_op_e_z = d1 + d2
        r_op_e_xy = np.zeros(2)
        if "offset" in obj_instance_conf:
            r_op_e_xy = parsing.parse_support_offset(obj_instance_conf["offset"])

        r_op_e = np.append(r_op_e_xy, r_op_e_z)

        # convert offset to the world frame
        obj.r0 = parent.r0 + C_we @ r_op_e

        obj.add_to_sim()
        objects[obj_name] = obj
        obj.fixture = fixture

    # for debugging, generate and show contact points
    if config.get("show_contact_points", False):
        boxes = {name: obj.box for name, obj in objects.items()}

        contact_points = []
        for contact in arrangement["contacts"]:
            name1 = contact["first"]
            name2 = contact["second"]
            points, _ = polyhedron.axis_aligned_contact(
                boxes[name1], boxes[name2], tol=1e-7
            )
            if points is None:
                raise ValueError(
                    f"No contact points found between {name1} and {name2}."
                )
            contact_points.append(points)

        contact_points = np.vstack(contact_points)
        colors = [[1, 1, 1] for _ in contact_points]
        pyb.addUserDebugPoints([v for v in contact_points], colors, pointSize=10)

    # get rid of "fake" EE object before returning
    objects.pop("ee")

    return objects


class UprightSimulation(BulletSimulation):
    def __init__(self, config, timestamp, video_name=None, extra_gui=False):
        super().__init__(
            timestep=config["timestep"], gravity=config["gravity"], extra_gui=extra_gui
        )

        self.config = config
        self.duration = config["duration"]

        # setup robot
        self.robot = UprightSimulatedRobot(config)
        self.robot.reset_joint_configuration(self.robot.home)

        # simulate briefly to let the robot settle down after being positioned
        self.settle(1.0)

        # setup obstacles
        if config["static_obstacles"]["enabled"]:
            obstacles_uid = pyb.loadURDF(
                parsing.parse_and_compile_urdf(config["static_obstacles"]["urdf"])
            )
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object

        r_ew_w, Q_we = self.robot.link_pose()

        self.dynamic_obstacles = []
        if self.config["dynamic_obstacles"]["enabled"]:
            for c in self.config["dynamic_obstacles"]["obstacles"]:
                obstacle = BulletDynamicObstacle.from_config(c, offset=r_ew_w)
                self.dynamic_obstacles.append(obstacle)

        # setup balanced objects
        self.objects = balanced_object_setup(r_ew_w, Q_we, config, self.robot)

        # mark frame at the initial position
        if config.get("show_debug_frames", False):
            debug_frame_world(0.2, list(r_ew_w), orientation=Q_we, line_width=3)

        # static cameras
        self.cameras = {
            k: camera_from_dict(v) for k, v in config.get("cameras", {}).items()
        }

        # video recording
        self.video_manager = VideoManager.from_config(
            video_name=video_name, config=config, timestamp=timestamp, r_ew_w=r_ew_w
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

    def launch_dynamic_obstacles(self, t0=0):
        """Start the dynamic obstacles.

        This adds each obstacle to the simulation at its initial state.
        """
        for obstacle in self.dynamic_obstacles:
            obstacle.start(t0=t0)

    def dynamic_obstacle_state(self):
        """Get the state vector of all dynamics obstacles."""
        if len(self.dynamic_obstacles) == 0:
            return np.array([])

        xs = []
        for obs in self.dynamic_obstacles:
            r, v, a = obs.joint_state()
            x = np.concatenate((r, v, a))
            xs.append(x)
        return np.concatenate(xs)

    def fixture_objects(self):
        # rigidly attach fixtured objects to the tray
        r_ew_w, _ = self.robot.link_pose()
        for name, obj in self.objects.items():
            print(f"{name}.fixture = {obj.fixture}")
            if obj.fixture:
                pyb.createConstraint(
                    self.robot.uid,
                    self.robot.tool_idx,
                    obj.uid,
                    -1,
                    pyb.JOINT_FIXED,
                    jointAxis=[0, 0, 1],  # doesn't matter
                    parentFramePosition=[0, 0, 0],
                    childFramePosition=list(r_ew_w - obj.r0),
                )

    def step(self, t, step_robot=True):
        """Step the simulation forward one timestep."""
        if step_robot:
            self.robot.step(secs=self.timestep)

        obstacle_reset = False
        for ghost in self.ghosts:
            ghost.update()
        for obstacle in self.dynamic_obstacles:
            obstacle_reset = obstacle.step(t) or obstacle_reset

        self.video_manager.record(t)

        pyb.stepSimulation()

        return t + self.timestep, obstacle_reset
