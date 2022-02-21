import os
from collections import deque
import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

from tray_balance_sim.end_effector import EndEffector
from tray_balance_sim.robot import SimulatedRobot
import tray_balance_sim.util as util
import tray_balance_sim.geometry as geometry
import tray_balance_sim.bodies as bodies

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2

import IPython


# set to true to add parameter error such that stack and cups configurations
# fail with nominal constraints
USE_STACK_ERROR = False
USE_CUPS_ERROR = False

CUPS_OFFSET_SIM = np.array([0, 0, 0])
# CUPS_OFFSET_SIM = np.array([0, 0.07, 0])

if USE_CUPS_ERROR:
    CUPS_OFFSET_CONTROL = -CUPS_OFFSET_SIM
else:
    CUPS_OFFSET_CONTROL = np.zeros(3)


rospack = rospkg.RosPack()
OBSTACLES_URDF_PATH = os.path.join(
    rospack.get_path("tray_balance_assets"), "urdf", "obstacles.urdf"
)

EE_SIDE_LENGTH = 0.2
EE_INSCRIBED_RADIUS = geometry.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

GRAVITY_MAG = 9.81
GRAVITY_VECTOR = np.array([0, 0, -GRAVITY_MAG])

# coefficient of friction for the EE
EE_MU = 1.0

# need at least some margin here to avoid objects falling
OBJ_ZMP_MARGIN = 0.01

# colors used by matplotlib: nice to use for object colors as well
PLT_COLOR1 = (0.122, 0.467, 0.706, 1)
PLT_COLOR2 = (1, 0.498, 0.055, 1)
PLT_COLOR3 = (0.173, 0.627, 0.173, 1)
PLT_COLOR4 = (0.839, 0.153, 0.157, 1)

### tray ###

TRAY_RADIUS = 0.2
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_COM_HEIGHT = 0.01
TRAY_MU_BULLET = TRAY_MU / EE_MU
TRAY_COLOR = PLT_COLOR1

### short cuboid ###

CUBOID_SHORT_MASS = 0.5
CUBOID_SHORT_TRAY_MU = 0.5
CUBOID_SHORT_COM_HEIGHT = 0.075
CUBOID_SHORT_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID_SHORT_COM_HEIGHT)
CUBOID_SHORT_R_TAU = geometry.rectangle_r_tau(*CUBOID_SHORT_SIDE_LENGTHS[:2])
CUBOID_SHORT_COLOR = PLT_COLOR2

# controller things μ is CUBOID_SHORT_TRAY_MU + CUBOID_SHORT_MU_ERROR, when it
# is actually just CUBOID_SHORT_TRAY_MU
CUBOID_SHORT_MU_CONTROL = CUBOID_SHORT_TRAY_MU
CUBOID_SHORT_MU_ERROR = CUBOID_SHORT_MU_CONTROL - CUBOID_SHORT_TRAY_MU

CUBOID_SHORT_R_TAU_CONTROL = 10 * CUBOID_SHORT_R_TAU
CUBOID_SHORT_R_TAU_ERROR = CUBOID_SHORT_R_TAU_CONTROL - CUBOID_SHORT_R_TAU

### tall cuboid ###

CUBOID_TALL_MASS = 0.5
CUBOID_TALL_TRAY_MU = 0.5
CUBOID_TALL_COM_HEIGHT = 0.25
CUBOID_TALL_SIDE_LENGTHS = (0.1, 0.1, 2 * CUBOID_TALL_COM_HEIGHT)
CUBOID_TALL_R_TAU = geometry.rectangle_r_tau(*CUBOID_TALL_SIDE_LENGTHS[:2])
CUBOID_TALL_COLOR = PLT_COLOR2

CUBOID_TALL_MU_CONTROL = CUBOID_TALL_TRAY_MU
CUBOID_TALL_MU_ERROR = CUBOID_TALL_MU_CONTROL - CUBOID_TALL_TRAY_MU

CUBOID_TALL_R_TAU_CONTROL = 10 * CUBOID_TALL_R_TAU
CUBOID_TALL_R_TAU_ERROR = CUBOID_TALL_R_TAU_CONTROL - CUBOID_TALL_R_TAU

### stack of boxes ###

CYLINDER_BASE_STACK_MASS = 0.75
CYLINDER_BASE_STACK_MU = 0.5
CYLINDER_BASE_STACK_MU_BULLET = CYLINDER_BASE_STACK_MU / EE_MU
CYLINDER_BASE_STACK_COM_HEIGHT = 0.05
CYLINDER_BASE_STACK_RADIUS = 0.15
CYLINDER_BASE_STACK_COLOR = PLT_COLOR1

# CYLINDER_BASE_STACK_CONTROL_MASS = CYLINDER_BASE_STACK_MASS
CYLINDER_BASE_STACK_CONTROL_MASS = 1.0 if USE_STACK_ERROR else CYLINDER_BASE_STACK_MASS
CYLINDER_BASE_STACK_MASS_ERROR = (
    CYLINDER_BASE_STACK_CONTROL_MASS - CYLINDER_BASE_STACK_MASS
)

# CUBOID_BASE_STACK_MASS = 0.75
# CUBOID_BASE_STACK_MU = 0.5
# CUBOID_BASE_STACK_MU_BULLET = CUBOID_BASE_STACK_MU / EE_MU
# CUBOID_BASE_STACK_COM_HEIGHT = 0.05
# CUBOID_BASE_STACK_SIDE_LENGTHS = (0.3, 0.3, 2 * CUBOID_BASE_STACK_COM_HEIGHT)
# CUBOID_BASE_STACK_COLOR = PLT_COLOR1
#
# CUBOID_BASE_STACK_CONTROL_MASS = CUBOID_BASE_STACK_MASS
# # CUBOID_BASE_STACK_CONTROL_MASS = 1.0
# CUBOID_BASE_STACK_MASS_ERROR = CUBOID_BASE_STACK_CONTROL_MASS - CUBOID_BASE_STACK_MASS

CUBOID1_STACK_MASS = 0.75
CUBOID1_STACK_TRAY_MU = 0.25
CUBOID1_STACK_COM_HEIGHT = 0.075
CUBOID1_STACK_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID1_STACK_COM_HEIGHT)
CUBOID1_STACK_COLOR = PLT_COLOR2

# CUBOID1_STACK_CONTROL_MASS = CUBOID1_STACK_MASS
CUBOID1_STACK_CONTROL_MASS = 1.0 if USE_STACK_ERROR else CUBOID1_STACK_MASS
CUBOID1_STACK_MASS_ERROR = CUBOID1_STACK_CONTROL_MASS - CUBOID1_STACK_MASS

CUBOID2_STACK_MASS = 1.25
CUBOID2_STACK_TRAY_MU = 0.25
CUBOID2_STACK_COM_HEIGHT = 0.075
CUBOID2_STACK_SIDE_LENGTHS = (0.1, 0.1, 2 * CUBOID2_STACK_COM_HEIGHT)
CUBOID2_STACK_COLOR = PLT_COLOR3
# horizontal offset of CoM relative to parent (CUBOID1_STACK)
CUBOID2_STACK_OFFSET = 0.5 * (
    np.array(CUBOID1_STACK_SIDE_LENGTHS[:2]) - CUBOID2_STACK_SIDE_LENGTHS[:2]
)

# CUBOID2_STACK_CONTROL_MASS = CUBOID2_STACK_MASS
CUBOID2_STACK_CONTROL_MASS = 1.0 if USE_STACK_ERROR else CUBOID2_STACK_MASS
CUBOID2_STACK_MASS_ERROR = CUBOID2_STACK_CONTROL_MASS - CUBOID2_STACK_MASS

CYLINDER3_STACK_MASS = 1.25
CYLINDER3_STACK_SUPPORT_MU = 0.25
CYLINDER3_STACK_RADIUS = 0.04
CYLINDER3_STACK_COM_HEIGHT = 0.05
CYLINDER3_STACK_COLOR = PLT_COLOR4
CYLINDER3_STACK_OFFSET = (
    0.5 * np.array(CUBOID2_STACK_SIDE_LENGTHS[:2]) - CYLINDER3_STACK_RADIUS
)

# CYLINDER3_STACK_CONTROL_MASS = CYLINDER3_STACK_MASS
CYLINDER3_STACK_CONTROL_MASS = 1.0 if USE_STACK_ERROR else CYLINDER3_STACK_MASS
CYLINDER3_STACK_MASS_ERROR = CYLINDER3_STACK_CONTROL_MASS - CYLINDER3_STACK_MASS

### set of cups ###

CYLINDER_CUP_MASS = 0.5
CYLINDER_CUP_SUPPORT_MU = 0.3
CYLINDER_CUP_RADIUS = 0.04
CYLINDER_CUP_COM_HEIGHT = 0.075
CYLINDER_CUP_COLORS = [PLT_COLOR2, PLT_COLOR3, PLT_COLOR4]


### robot starting configurations ###

BASE_HOME = [0, 0, 0]
UR10_HOME_STANDARD = [
    0.0,
    -0.75 * np.pi,
    -0.5 * np.pi,
    -0.75 * np.pi,
    -0.5 * np.pi,
    0.5 * np.pi,
]
UR10_HOME_TRAY_BALANCE = [
    0.0,
    -0.75 * np.pi,
    -0.5 * np.pi,
    -0.25 * np.pi,
    -0.5 * np.pi,
    0.5 * np.pi,
]
ROBOT_HOME = BASE_HOME + UR10_HOME_TRAY_BALANCE


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


class Simulation:
    def __init__(self, dt):
        self.dt = dt  # simulation timestep (s)
        self.video_file_name = None

    def settle(self, duration):
        """Run simulation while doing nothing.

        Useful to let objects settle to rest before applying control.
        """
        t = 0
        while t < 1.0:
            pyb.stepSimulation()
            t += self.dt

    def step(self, step_robot=True):
        """Step the simulation forward one timestep."""
        if step_robot:
            self.robot.step()
        pyb.stepSimulation()

    def record_video(self, file_name):
        """Record a video of the simulation to the given file."""
        self.video_file_name = str(file_name)

    def basic_setup(self, load_static_obstacles):
        # pyb.connect(pyb.GUI)
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")

        pyb.setGravity(0, 0, -GRAVITY_MAG)
        pyb.setTimeStep(self.dt)

        # default
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=4.6,
        #     cameraYaw=5.2,
        #     cameraPitch=-27,
        #     cameraTargetPosition=[1.18, 0.11, 0.05],
        # )

        # for close-ups shots of the EE / objects
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=1.8,
        #     cameraYaw=14,
        #     cameraPitch=-39,
        #     cameraTargetPosition=(1.028, 0.075, 0.666),
        # )

        # for taking pictures of the dynamic obstacle avoidance task
        # also dynamic obstacle POV #1
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=1.8,
        #     cameraYaw=147.6,
        #     cameraPitch=-29,
        #     cameraTargetPosition=[1.28, 0.045, 0.647],
        # )

        # dynamic obstacle POV #2
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=2.6,
        #     cameraYaw=-3.2,
        #     cameraPitch=-20.6,
        #     cameraTargetPosition=[1.28, 0.045, 0.647],
        # )

        # static obstacle course POV #1
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=3.6,
        #     cameraYaw=-39.6,
        #     cameraPitch=-38.2,
        #     cameraTargetPosition=[1.66, -0.31, 0.03],
        # )

        # static obstacle course POV #2
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=3.4,
        #     cameraYaw=10.0,
        #     cameraPitch=-23.4,
        #     cameraTargetPosition=[2.77, 0.043, 0.142],
        # )

        # static obstacle course POV #3
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=4.8,
        #     cameraYaw=87.6,
        #     cameraPitch=-13.4,
        #     cameraTargetPosition=[2.77, 0.043, 0.142],
        # )

        # single- and multi-object POV
        # pyb.resetDebugVisualizerCamera(
        #     cameraDistance=3,
        #     cameraYaw=26,
        #     cameraPitch=-30.6,
        #     cameraTargetPosition=[1.28, 0.045, 0.647],
        # )
        pyb.resetDebugVisualizerCamera(
            cameraDistance=4,
            cameraYaw=42,
            cameraPitch=-35.8,
            cameraTargetPosition=[1.28, 0.045, 0.647],
        )

        # get rid of extra parts of the GUI
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)
        # pyb.configureDebugVisualizer(pyb.COV_ENABLE_SINGLE_STEP_RENDERING, 1)

        # record video
        if self.video_file_name is not None:
            pyb.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4, self.video_file_name)

        # setup ground plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.loadURDF("plane.urdf", [0, 0, 0])

        # setup obstacles
        if load_static_obstacles:
            obstacles_uid = pyb.loadURDF(OBSTACLES_URDF_PATH)
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object
            # pyb.setCollisionFilterGroupMask(obstacles_uid, -1, 0, 0)

    def compute_cylinder_xy_positions(self, L=0.08):
        """L is distance along vector toward each vertex."""
        s = 0.2
        h = s / (2 * np.sqrt(3))
        r = 2 * h

        # triangle support area vertices
        vertices = np.array([[r, 0], [-h, 0.5 * s], [-h, -0.5 * s]])

        # unit normals to each of the vertices
        n0 = vertices[0, :] / np.linalg.norm(vertices[0, :])
        n1 = vertices[1, :] / np.linalg.norm(vertices[1, :])
        n2 = vertices[2, :] / np.linalg.norm(vertices[2, :])

        c0 = L * n0
        c1 = L * n1
        c2 = L * n2

        return (c0, c1, c2)

    def object_setup(self, r_ew_w, obj_names, controller_obj_names=None):
        # controller objects are the ones the controller thinks are there
        objects = {}

        # tray

        name = "tray"
        if name in obj_names:
            tray = bodies.Cylinder(
                r_tau=EE_INSCRIBED_RADIUS,
                support_area=ocs2.PolygonSupportArea.equilateral_triangle(
                    EE_SIDE_LENGTH
                ),
                mass=TRAY_MASS,
                radius=TRAY_RADIUS,
                height=2 * TRAY_COM_HEIGHT,
                mu=TRAY_MU,
            )
            tray.add_to_sim(bullet_mu=TRAY_MU_BULLET, color=TRAY_COLOR)

            # add 0.05 to account for EE height; this is fixed when the sim is
            # settled later
            r_tw_w = r_ew_w + [0, 0, TRAY_COM_HEIGHT + 0.05]
            tray.bullet.reset_pose(position=r_tw_w)

            objects[name] = tray

        name = "cylinder_base_stack"
        if name in obj_names:
            objects[name] = bodies.Cylinder(
                r_tau=EE_INSCRIBED_RADIUS,
                support_area=ocs2.PolygonSupportArea.equilateral_triangle(
                    EE_SIDE_LENGTH
                ),
                mass=CYLINDER_BASE_STACK_MASS,
                radius=CYLINDER_BASE_STACK_RADIUS,
                height=2 * CYLINDER_BASE_STACK_COM_HEIGHT,
                mu=CYLINDER_BASE_STACK_MU,
            )
            objects[name].mass_error = CYLINDER_BASE_STACK_MASS_ERROR
            objects[name].add_to_sim(
                bullet_mu=CYLINDER_BASE_STACK_MU_BULLET, color=CYLINDER_BASE_STACK_COLOR
            )

            r_tw_w = r_ew_w + [0, 0, CYLINDER_BASE_STACK_COM_HEIGHT + 0.05]
            objects[name].bullet.reset_pose(position=r_tw_w)

        name = "cuboid_base_stack"
        if name in obj_names:
            objects[name] = bodies.Cuboid(
                r_tau=EE_SIDE_LENGTH,
                support_area=ocs2.PolygonSupportArea.equilateral_triangle(
                    EE_SIDE_LENGTH
                ),
                mass=CUBOID_BASE_STACK_MASS,
                side_lengths=CUBOID_BASE_STACK_SIDE_LENGTHS,
                mu=CUBOID_BASE_STACK_MU,
            )
            objects[name].mass_error = CUBOID_BASE_STACK_MASS_ERROR

            objects[name].add_to_sim(
                bullet_mu=CUBOID_BASE_STACK_MU_BULLET, color=CUBOID_BASE_STACK_COLOR
            )
            r_tw_w = r_ew_w + [0, 0, 0.5 * CUBOID_BASE_STACK_SIDE_LENGTHS[2] + 0.05]
            objects[name].bullet.reset_pose(position=r_tw_w)

        def add_obj_to_sim(obj, name, color, parent, offset_xy=(0, 0)):
            bullet_mu = obj.mu / parent.bullet.mu
            obj.add_to_sim(bullet_mu=bullet_mu, color=color)

            r_ow_w = parent.bullet.get_pose()[0] + [
                offset_xy[0],
                offset_xy[1],
                parent.com_height + obj.com_height,
            ]
            obj.bullet.reset_pose(position=r_ow_w)
            objects[name] = obj
            parent.children.append(name)

        # flat and tall cuboids

        name = "cuboid_short"
        if name in obj_names:
            support_area = ocs2.PolygonSupportArea.axis_aligned_rectangle(
                CUBOID_SHORT_SIDE_LENGTHS[0],
                CUBOID_SHORT_SIDE_LENGTHS[1],
                margin=OBJ_ZMP_MARGIN,
            )
            objects[name] = bodies.Cuboid(
                r_tau=CUBOID_SHORT_R_TAU,
                support_area=support_area,
                mass=CUBOID_SHORT_MASS,
                side_lengths=CUBOID_SHORT_SIDE_LENGTHS,
                mu=CUBOID_SHORT_TRAY_MU,
            )
            objects[name].mu_error = CUBOID_SHORT_MU_ERROR
            objects[name].r_tau_error = CUBOID_SHORT_R_TAU_ERROR
            add_obj_to_sim(
                obj=objects[name],
                name=name,
                color=CUBOID_SHORT_COLOR,
                parent=objects["tray"],
            )

        name = "cuboid_tall"
        if name in obj_names:
            support_area = ocs2.PolygonSupportArea.axis_aligned_rectangle(
                CUBOID_TALL_SIDE_LENGTHS[0],
                CUBOID_TALL_SIDE_LENGTHS[1],
                margin=OBJ_ZMP_MARGIN,
            )
            objects[name] = bodies.Cuboid(
                r_tau=CUBOID_TALL_R_TAU,
                support_area=support_area,
                mass=CUBOID_TALL_MASS,
                side_lengths=CUBOID_TALL_SIDE_LENGTHS,
                mu=CUBOID_TALL_TRAY_MU,
            )
            objects[name].mu_error = CUBOID_TALL_MU_ERROR
            objects[name].r_tau_error = CUBOID_TALL_R_TAU_ERROR
            add_obj_to_sim(
                obj=objects[name],
                name=name,
                color=CUBOID_TALL_COLOR,
                parent=objects["tray"],
            )

        # stack of boxes

        name = "cuboid1_stack"
        if name in obj_names:
            support_area = ocs2.PolygonSupportArea.axis_aligned_rectangle(
                CUBOID1_STACK_SIDE_LENGTHS[0],
                CUBOID1_STACK_SIDE_LENGTHS[1],
                margin=OBJ_ZMP_MARGIN,
            )
            objects[name] = bodies.Cuboid(
                r_tau=geometry.rectangle_r_tau(*CUBOID1_STACK_SIDE_LENGTHS[:2]),
                support_area=support_area,
                mass=CUBOID1_STACK_MASS,
                side_lengths=CUBOID1_STACK_SIDE_LENGTHS,
                mu=CUBOID1_STACK_TRAY_MU,
            )
            objects[name].mass_error = CUBOID1_STACK_MASS_ERROR
            # objects[name].mu_error = -0.05
            objects[name].mu_error = 0
            add_obj_to_sim(
                obj=objects[name],
                name=name,
                color=CUBOID1_STACK_COLOR,
                # parent=objects["cuboid_base_stack"],
                parent=objects["cylinder_base_stack"],
            )

        name = "cuboid2_stack"
        if name in obj_names:
            support_area = ocs2.PolygonSupportArea.axis_aligned_rectangle(
                CUBOID2_STACK_SIDE_LENGTHS[0],
                CUBOID2_STACK_SIDE_LENGTHS[1],
                margin=OBJ_ZMP_MARGIN,
            )
            objects[name] = bodies.Cuboid(
                r_tau=geometry.rectangle_r_tau(*CUBOID2_STACK_SIDE_LENGTHS[:2]),
                support_area=support_area,
                mass=CUBOID2_STACK_MASS,
                side_lengths=CUBOID2_STACK_SIDE_LENGTHS,
                mu=CUBOID2_STACK_TRAY_MU,
            )
            objects[name].mass_error = CUBOID2_STACK_MASS_ERROR
            # objects[name].mu_error = -0.05
            objects[name].mu_error = 0
            add_obj_to_sim(
                obj=objects[name],
                name=name,
                color=CUBOID2_STACK_COLOR,
                parent=objects["cuboid1_stack"],
                offset_xy=CUBOID2_STACK_OFFSET,
            )

        name = "cylinder3_stack"
        if name in obj_names:
            objects[name] = bodies.Cylinder(
                r_tau=geometry.circle_r_tau(CYLINDER3_STACK_RADIUS),
                support_area=ocs2.PolygonSupportArea.circle(
                    CYLINDER3_STACK_RADIUS, margin=OBJ_ZMP_MARGIN
                ),
                mass=CYLINDER3_STACK_MASS,
                radius=CYLINDER3_STACK_RADIUS,
                height=2 * CYLINDER3_STACK_COM_HEIGHT,
                mu=CYLINDER3_STACK_SUPPORT_MU,
            )
            objects[name].mass_error = CYLINDER3_STACK_MASS_ERROR
            # objects[name].mu_error = -0.05
            objects[name].mu_error = 0
            add_obj_to_sim(
                obj=objects[name],
                name=name,
                color=CYLINDER3_STACK_COLOR,
                parent=objects["cuboid2_stack"],
                offset_xy=CYLINDER3_STACK_OFFSET,
            )

        # set of cups

        cup_positions = self.compute_cylinder_xy_positions(L=0.08)
        for i, name in enumerate(["cylinder1_cup", "cylinder2_cup", "cylinder3_cup"]):
            if name in obj_names:
                objects[name] = bodies.Cylinder(
                    r_tau=geometry.circle_r_tau(CYLINDER_CUP_RADIUS),
                    support_area=ocs2.PolygonSupportArea.circle(
                        CYLINDER_CUP_RADIUS, margin=OBJ_ZMP_MARGIN
                    ),
                    mass=CYLINDER_CUP_MASS,
                    radius=CYLINDER_CUP_RADIUS,
                    height=2 * CYLINDER_CUP_COM_HEIGHT,
                    mu=CYLINDER_CUP_SUPPORT_MU,
                )
                objects[name].com_error = CUPS_OFFSET_CONTROL

                add_obj_to_sim(
                    obj=objects[name],
                    name=name,
                    color=CYLINDER_CUP_COLORS[i],
                    parent=objects["tray"],
                    offset_xy=cup_positions[i] + CUPS_OFFSET_SIM[:2],
                )

        # name = "stacked_cylinder2"
        # if name in obj_names:
        #     objects[name] = bodies.Cylinder(
        #         r_tau=geometry.circle_r_tau(CYLINDER2_RADIUS),
        #         support_area=ocs2.PolygonSupportArea.circle(
        #             CYLINDER2_RADIUS, margin=OBJ_ZMP_MARGIN
        #         ),
        #         mass=CYLINDER2_MASS,
        #         radius=CYLINDER2_RADIUS,
        #         height=2 * CYLINDER2_COM_HEIGHT,
        #         mu=CYLINDER2_SUPPORT_MU,
        #     )
        #     objects[name].mass_error = 0
        #     # objects[name].com_error = np.array([0, 0, 0.075])
        #     # objects[name].com_height_error = 0.075
        #
        #     add_obj_to_sim(
        #         obj=objects[name],
        #         name=name,
        #         color=CYLINDER2_COLOR,
        #         parent=objects["stacked_cylinder1"],
        #     )
        #
        # name = "flat_cylinder2"
        # if name in obj_names:
        #     objects[name] = bodies.Cylinder(
        #         r_tau=geometry.circle_r_tau(CYLINDER2_RADIUS),
        #         support_area=ocs2.PolygonSupportArea.circle(
        #             CYLINDER2_RADIUS, margin=OBJ_ZMP_MARGIN
        #         ),
        #         mass=CYLINDER2_MASS,
        #         radius=CYLINDER2_RADIUS,
        #         height=2 * CYLINDER2_COM_HEIGHT,
        #         mu=CYLINDER2_SUPPORT_MU,
        #     )
        #     objects[name].mass_error = 0
        #     objects[name].com_error = FLAT_OFFSET
        #
        #     add_obj_to_sim(
        #         obj=objects[name],
        #         name=name,
        #         color=CYLINDER2_COLOR,
        #         parent=objects["tray"],
        #         offset_xy=c2 - FLAT_OFFSET[:2],
        #     )
        #
        # name = "stacked_cylinder3"
        # if name in obj_names:
        #     objects[name] = bodies.Cylinder(
        #         r_tau=geometry.circle_r_tau(CYLINDER3_RADIUS),
        #         support_area=ocs2.PolygonSupportArea.circle(
        #             CYLINDER3_RADIUS, margin=OBJ_ZMP_MARGIN
        #         ),
        #         mass=CYLINDER3_MASS,
        #         radius=CYLINDER3_RADIUS,
        #         height=2 * CYLINDER3_COM_HEIGHT,
        #         mu=CYLINDER3_SUPPORT_MU,
        #     )
        #     objects[name].mass_error = 0
        #     # objects[name].com_error = np.array([0, 0, 0.075])
        #     # objects[name].com_height_error = 0.075
        #
        #     add_obj_to_sim(
        #         obj=objects[name],
        #         name=name,
        #         color=CYLINDER3_COLOR,
        #         parent=objects["stacked_cylinder2"],
        #     )
        #
        # name = "flat_cylinder3"
        # if name in obj_names:
        #     objects[name] = bodies.Cylinder(
        #         r_tau=geometry.circle_r_tau(CYLINDER3_RADIUS),
        #         support_area=ocs2.PolygonSupportArea.circle(
        #             CYLINDER3_RADIUS, margin=OBJ_ZMP_MARGIN
        #         ),
        #         mass=CYLINDER3_MASS,
        #         radius=CYLINDER3_RADIUS,
        #         height=2 * CYLINDER3_COM_HEIGHT,
        #         mu=CYLINDER3_SUPPORT_MU,
        #     )
        #     objects[name].mass_error = 0
        #     objects[name].com_error = FLAT_OFFSET
        #
        #     add_obj_to_sim(
        #         obj=objects[name],
        #         name=name,
        #         color=CYLINDER3_COLOR,
        #         parent=objects["tray"],
        #         offset_xy=c3 - FLAT_OFFSET[:2],
        #     )

        # name = "rod"
        # if name in obj_names:
        #     objects[name] = bodies.Cylinder(
        #         r_tau=0,
        #         support_area=None,
        #         mass=1.0,
        #         radius=0.01,
        #         height=1.0,
        #         mu=1.0,
        #     )
        #     objects[name].add_to_sim(bullet_mu=1.0, color=(0.839, 0.153, 0.157, 1))
        #
        #     r_ow_w = r_ew_w + [0, 0, 2 * TRAY_COM_HEIGHT + 0.5 + 0.01]
        #     objects[name].bullet.reset_pose(position=r_ow_w)
        #     objects["tray"].children.append(name)

        return objects

    def composite_setup(self, objects):
        # composites are only used by the controller, so we only need to
        # compose the controller objects
        composites = []
        for obj in objects.values():
            # all descendants compose the new object
            descendants = []
            queue = deque([obj])
            while len(queue) > 0:
                desc = queue.popleft()
                descendants.append(desc.convert_to_ocs2())
                for name in desc.children:
                    queue.append(objects[name])

            # descendants have already been converted to C++ objects
            composites.append(ocs2.BalancedObject.compose(descendants))
        return composites


# class FloatingEESimulation(Simulation):
#     def __init__(self, dt=0.001):
#         super().__init__(dt)
#
#     def setup(self, obj_names=None):
#         """Setup pybullet simulation."""
#         super().basic_setup()
#
#         # setup floating end effector
#         robot = EndEffector(self.dt, side_length=EE_SIDE_LENGTH, position=(0, 0, 1))
#         self.robot = robot
#         # util.debug_frame(0.1, robot.uid, -1)
#
#         r_ew_w, Q_we = robot.get_pose()
#         objects = super().object_setup(r_ew_w, obj_names)
#
#         self.settle(1.0)
#
#         # need to set the CoM after the sim has been settled, so objects are in
#         # their proper positions
#         r_ew_w, Q_we = robot.get_pose()
#         for obj in objects.values():
#             r_ow_w, _ = obj.bullet.get_pose()
#             obj.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)
#
#         composites = super().composite_setup(objects)
#
#         return robot, objects, composites


class MobileManipulatorSimulation(Simulation):
    def __init__(self, dt=0.001):
        super().__init__(dt)

    def setup(self, tray_balance_settings, load_static_obstacles=False):
        """Setup pybullet simulation."""
        super().basic_setup(load_static_obstacles)

        robot = SimulatedRobot(
            self.dt, load_static_collision_objects=load_static_obstacles
        )
        self.robot = robot
        robot.reset_joint_configuration(ROBOT_HOME)

        # simulate briefly to let the robot settle down after being positioned
        self.settle(1.0)

        # arm gets bumped by the above settling, so we reset it again
        robot.reset_arm_joints(UR10_HOME_TRAY_BALANCE)
        # robot.reset_joint_configuration(ROBOT_HOME)

        # self.settle(1.0)

        r_ew_w, _ = robot.link_pose()
        objects = super().object_setup(r_ew_w, tray_balance_settings)

        self.settle(1.0)

        # need to set the CoM after the sim has been settled, so objects are in
        # their proper positions
        r_ew_w, Q_we = robot.link_pose()
        for obj in objects.values():
            r_ow_w, _ = obj.bullet.get_pose()
            obj.com = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

        composites = super().composite_setup(objects)
        # ocs2_objects = {}
        # for name, obj in objects.items():
        #     ocs2_objects[name] = obj.convert_to_ocs2()
        # big_cylinder = ocs2.BalancedObject.compose(
        #     [
        #         ocs2_objects[name]
        #         for name in [
        #             "cuboid1",
        #             "stacked_cylinder1",
        #             "stacked_cylinder2",
        #         ]
        #     ]
        # )
        # tray_big_cylinder = ocs2.BalancedObject.compose(
        #     [ocs2_objects["tray"], big_cylinder]
        # )
        # composites = [tray_big_cylinder, big_cylinder]

        return robot, objects, composites
