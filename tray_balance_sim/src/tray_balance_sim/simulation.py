import os
from collections import deque
import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

from tray_balance_sim.robot import SimulatedRobot
import tray_balance_sim.util as util
import tray_balance_sim.bodies as bodies

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2
from tray_balance_constraints import parsing, math

import IPython

# Naming:
# - config = raw dict
# - config_wrapper = object somehow containing the raw config
# - arrangement = the particular set of objects in use


# set to true to add parameter error such that stack and cups configurations
# fail with nominal constraints (only affects nominal constraints)
USE_STACK_ERROR = False
USE_CUPS_ERROR = False

# use this with robust
# CUPS_OFFSET_SIM = np.array([0, 0, 0])

# use this with nominal
CUPS_OFFSET_SIM = np.array([0, 0.07, 0])

if USE_CUPS_ERROR:
    CUPS_OFFSET_CONTROL = -CUPS_OFFSET_SIM
else:
    CUPS_OFFSET_CONTROL = np.zeros(3)


rospack = rospkg.RosPack()
OBSTACLES_URDF_PATH = os.path.join(
    rospack.get_path("tray_balance_assets"), "urdf", "obstacles.urdf"
)

EE_SIDE_LENGTH = 0.2
EE_HEIGHT = 0.04
EE_INSCRIBED_RADIUS = math.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

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
CUBOID_SHORT_R_TAU = math.rectangle_r_tau(*CUBOID_SHORT_SIDE_LENGTHS[:2])
CUBOID_SHORT_COLOR = PLT_COLOR2

# controller things μ is CUBOID_SHORT_TRAY_MU + CUBOID_SHORT_MU_ERROR, when it
# is actually just CUBOID_SHORT_TRAY_MU
CUBOID_SHORT_MU_CONTROL = CUBOID_SHORT_TRAY_MU
CUBOID_SHORT_MU_ERROR = CUBOID_SHORT_MU_CONTROL - CUBOID_SHORT_TRAY_MU

CUBOID_SHORT_R_TAU_CONTROL = CUBOID_SHORT_R_TAU
CUBOID_SHORT_R_TAU_ERROR = CUBOID_SHORT_R_TAU_CONTROL - CUBOID_SHORT_R_TAU

### tall cuboid ###

CUBOID_TALL_MASS = 0.5
CUBOID_TALL_TRAY_MU = 0.5
CUBOID_TALL_COM_HEIGHT = 0.25
CUBOID_TALL_SIDE_LENGTHS = (0.1, 0.1, 2 * CUBOID_TALL_COM_HEIGHT)
CUBOID_TALL_R_TAU = math.rectangle_r_tau(*CUBOID_TALL_SIDE_LENGTHS[:2])
CUBOID_TALL_COLOR = PLT_COLOR2

CUBOID_TALL_MU_CONTROL = CUBOID_TALL_TRAY_MU
CUBOID_TALL_MU_ERROR = CUBOID_TALL_MU_CONTROL - CUBOID_TALL_TRAY_MU

CUBOID_TALL_R_TAU_CONTROL = CUBOID_TALL_R_TAU
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

CUBOID1_STACK_MASS = 0.75
CUBOID1_STACK_TRAY_MU = 0.25
CUBOID1_STACK_COM_HEIGHT = 0.075
CUBOID1_STACK_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID1_STACK_COM_HEIGHT)
CUBOID1_STACK_COLOR = PLT_COLOR2
CUBOID1_STACK_MU_BULLET = CUBOID1_STACK_TRAY_MU / CYLINDER_BASE_STACK_MU_BULLET

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

# TODO want to move the home stuff into config as well
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


class PyBulletSimulation:
    def __init__(self, sim_config):
        self.dt = sim_config["timestep"]

        pyb.connect(pyb.GUI, options="--width=1280 --height=720")
        pyb.setGravity(*sim_config["gravity"])
        pyb.setTimeStep(sim_config["timestep"])

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
        if sim_config["static_obstacles"]["enabled"]:
            obstacles_uid = pyb.loadURDF(
                config_utils.urdf_path(sim_config["urdf"]["obstacles"])
            )
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object

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


def sim_object_setup(r_ew_w, config):
    # controller objects are the ones the controller thinks are there
    arrangement_name = config["arrangement"]
    arrangement = config["arrangements"][arrangement_name]
    object_configs = config["objects"]
    ee = object_configs["ee"]

    objects = {}
    for d in arrangement:
        name = d["name"]
        obj_config = config["objects"][name]
        obj = bodies.BulletBody.fromdict(obj_config)

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
        objects[name] = obj

    return objects


class MobileManipulatorSimulation(PyBulletSimulation):
    def __init__(self, sim_config):
        super().__init__(sim_config)

        self.robot = SimulatedRobot(sim_config)
        self.robot.reset_joint_configuration(ROBOT_HOME)

        # simulate briefly to let the robot settle down after being positioned
        self.settle(1.0)

        # arm gets bumped by the above settling, so we reset it again
        self.robot.reset_arm_joints(UR10_HOME_TRAY_BALANCE)

        self.settle(1.0)
