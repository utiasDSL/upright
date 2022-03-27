import os
from collections import deque
import numpy as np
import pybullet as pyb
import pybullet_data
import rospkg

from tray_balance_sim.robot import SimulatedRobot
import tray_balance_sim.util as util
import tray_balance_sim.geometry as geometry
import tray_balance_sim.bodies as bodies

import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2
import tray_balance_constraints as con

import IPython


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

CUBOID_SHORT_R_TAU_CONTROL = CUBOID_SHORT_R_TAU
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


# class BalancedObject:
#     def __init__(self, ctrl_obj, sim_obj, parent):
#         self.ctrl_obj = ctrl_obj
#         self.sim_obj = sim_obj
#         self.parent = parent
#         self.children = []


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


def compute_offset(d):
    x = d["x"] if "x" in d else 0
    y = d["y"] if "y" in d else 0
    if "r" in d and "θ" in d:
        r = d["r"]
        θ = d["θ"]
        x += r * np.cos(θ)
        y += r * np.sin(θ)
    return np.array([x, y])


class Simulation:
    def __init__(self, dt):
        self.dt = dt  # simulation timestep (s)

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

    def basic_setup(self, config):
        pyb.connect(pyb.GUI, options="--width=1280 --height=720")

        pyb.setGravity(0, 0, -GRAVITY_MAG)
        pyb.setTimeStep(self.dt)

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
                config_utils.urdf_path(config["urdf"]["obstacles"])
            )
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object


def sim_object_setup(self, r_ew_w, config):
    # controller objects are the ones the controller thinks are there
    objects = {}

    arrangement_name = config["arrangement"]
    arrangement = config["arrangements"][arrangement_name]
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
            position = r_ew_w + [0, 0, 0.02 + 0.5 * obj.height]
            obj.mu = obj.mu / EE_MU

        if "offset" in d:
            position[:2] += compute_offset(d["offset"])

        obj.add_to_sim(position)
        objects[name] = obj

    return objects


class ControlObjectConfigWrapper:
    def __init__(self, d):
        self.d = d
        self.children = []
        self.position = None

    @property
    def height(self):
        return self.d["height"]

    @property
    def parent_name(self):
        # No parent means the object is directly supported by the EE
        return self.d["parent"] if "parent" in d else None

    @property
    def offset(self):
        if "offset" in self.d:
            return np.array(self.d["offset"])
        return np.zeros(2)

    def bounded_balanced_object(self):
        """Generate a BoundedBalancedObject for this object."""
        # TODO parse SA
        # TODO parse r_tau
        # TODO make body
        # TODO make balanced object
        return con.BoundedBalancedObject()


# TODO need some naming reforms here:
# - config = configuration for *something*, but can currently be either the
#   raw dict or a special object, and is also used to refer to the specific
#   arrangement of objects being balanced
# Proposal:
# - config = raw dict
# - config_wrapper = object somehow containing the raw config
# - arrangement = the particular set of objects in use

def control_object_setup(config):
    # TODO need to pass in the control config here
    wrappers = {}
    arrangement_name = config["arrangement"]
    arrangement = config["arrangements"][arrangement_name]
    for conf in arrangement:
        name = conf["name"]
        wrapper = ControlObjectConfigWrapper(config["objects"][name])

        # compute position of the object
        if wrapper.parent_name is not None:
            parent = object_configs[wrapper.parent_name]
            dz = 0.5 * parent.height + 0.5 * wrapper.height
            wrapper.position = parent.position + [0, 0, dz]
        else:
            dz = 0.5 * EE_HEIGHT + 0.5 * wrapper.height
            wrapper.position = r_ew_w + [0, 0, dz]

        # add offset in the x-y (support) plane
        wrapper.position[:2] += wrapper.offset

        wrappers[name] = wrapper

    # find the direct children of each object
    for name, wrapper in wrappers.items():
        if wrapper.parent_name is not None:
            wrappers[wrapper.parent_name].children.append(name)

    # convert wrappers to BoundedBalancedObjects as required by the controller
    # and compose them as needed
    composites = []
    for wrapper in wrappers.values():
        # all descendants compose the new object
        descendants = []
        queue = deque([wrapper])
        while len(queue) > 0:
            wrapper = queue.popleft()
            descendants.append(wrapper.bounded_balanced_object())
            for name in wrapper.children:
                queue.append(wrappers[name])

        # descendants have already been converted to C++ objects
        composites.append(con.BoundedBalancedObject.compose(descendants))

    return composites


class MobileManipulatorSimulation(Simulation):
    def __init__(self, dt=0.001):
        super().__init__(dt)

    def setup(self, config):
        """Setup pybullet simulation."""
        super().basic_setup(config)

        self.robot = SimulatedRobot(config)
        self.robot.reset_joint_configuration(ROBOT_HOME)

        # simulate briefly to let the robot settle down after being positioned
        self.settle(1.0)

        # arm gets bumped by the above settling, so we reset it again
        self.robot.reset_arm_joints(UR10_HOME_TRAY_BALANCE)

        r_ew_w, _ = self.robot.link_pose()
        sim_objects = sim_object_setup(r_ew_w, config)

        # TODO this should be done elsewhere
        ctrl_objects = control_object_setup(config)

        self.settle(1.0)

        # need to set the CoM after the sim has been settled, so objects are in
        # their proper positions
        # r_ew_w, Q_we = robot.link_pose()
        # for obj in objects.values():
        #     r_ow_w, _ = obj.sim_obj.get_pose()
        #     # TODO none of these are write access!
        #     obj.ctrl_obj.body.com_ellipsoid.center = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

        return self.robot, sim_objects, ctrl_objects
