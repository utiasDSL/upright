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


class BalancedObject:
    def __init__(self, ctrl_obj, sim_obj, parent):
        self.ctrl_obj = ctrl_obj
        self.sim_obj = sim_obj
        self.parent = parent
        self.children = []


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

    def basic_setup(self, config):
        load_static_obstacles = False  # TODO
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
            obstacles_uid = pyb.loadURDF(config_utils.urdf_path(config["urdf"]["obstacles"]))
            pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object

    # TODO this should go elsewhere probably
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

    def object_setup(self, r_ew_w, config):
        # controller objects are the ones the controller thinks are there
        objects = {}

        def compute_offset(d):
            x = d["x"] if "x" in d else 0
            y = d["y"] if "y" in d else 0
            if "r" in d and "θ" in d:
                r = d["r"]
                θ = d["θ"]
                x += r * np.cos(θ)
                y += r * np.sin(θ)
            return np.array([x, y])

        object_data = config["object_configs"][config["object_config_name"]]
        for d in object_data:
            name = d["name"]
            properties = config["objects"][name]
            obj = bodies.BulletBody.fromdict(properties)

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


        # tray

        name = "tray"
        if name in obj_names:
            tray = bodies.Cylinder(
                r_tau=geometry.equilateral_triangle_r_tau(EE_SIDE_LENGTH),
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
            # TODO these will have to specified elsewhere
            # com_ellipsoid = con.Ellipsoid.point(obj.com)
            com_center = np.array([0, 0, CYLINDER_BASE_STACK_COM_HEIGHT + 0.02])
            com_half_lengths = 0.05 * np.array([1, 1, 1])
            com_ellipsoid = con.Ellipsoid(com_center, com_half_lengths, np.eye(3))
            r_gyr = CYLINDER_BASE_STACK_RADIUS * np.array([1, 1, 1])

            ctrl_body = con.BoundedRigidBody(
                mass_min=CYLINDER_BASE_STACK_MASS,
                mass_max=CYLINDER_BASE_STACK_MASS,
                radii_of_gyration=r_gyr,
                com_ellipsoid=com_ellipsoid,
            )
            ctrl_obj = con.BoundedBalancedObject(
                ctrl_body,
                com_height=CYLINDER_BASE_STACK_COM_HEIGHT,
                support_area_min=ocs2.PolygonSupportArea.equilateral_triangle(
                    EE_SIDE_LENGTH
                ),
                r_tau_min=geometry.equilateral_triangle_r_tau(EE_SIDE_LENGTH),
                mu_min=CYLINDER_BASE_STACK_MU,
            )

            # add the object to the bullet sim
            sim_obj = bodies.BulletBody.cylinder(
                mass=CYLINDER_BASE_STACK_MASS,
                mu=CYLINDER_BASE_STACK_MU_BULLET,
                position=r_ew_w + com_center,
                radius=CYLINDER_BASE_STACK_RADIUS,
                height=2*CYLINDER_BASE_STACK_COM_HEIGHT,
            )

            # it is convenient to wrap the control and sim objects
            objects[name] = BalancedObject(ctrl_obj, sim_obj, parent=None)

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

            # TODO avoid manually specifying
            com_center = np.array([0, 0, CUBOID1_STACK_COM_HEIGHT + 2 * CYLINDER_BASE_STACK_COM_HEIGHT + 0.02])
            com_half_lengths = 0.05 * np.array([1, 1, 1])
            com_ellipsoid = con.Ellipsoid(com_center, com_half_lengths, np.eye(3))
            r_gyr = 0.5 * np.array(CUBOID1_STACK_SIDE_LENGTHS)

            ctrl_body = con.BoundedRigidBody(
                mass_min=CUBOID1_STACK_MASS,
                mass_max=CUBOID1_STACK_MASS,
                radii_of_gyration=r_gyr,
                com_ellipsoid=com_ellipsoid,
            )
            ctrl_obj = con.BoundedBalancedObject(
                ctrl_body,
                com_height=CUBOID1_STACK_COM_HEIGHT,
                support_area_min=support_area,
                r_tau_min=geometry.rectangle_r_tau(*CUBOID1_STACK_SIDE_LENGTHS[:2]),
                mu_min=CUBOID1_STACK_TRAY_MU,
            )

            # add the object to the bullet sim
            sim_obj = bodies.BulletBody.cuboid(
                mass=CUBOID1_STACK_MASS,
                mu=CUBOID1_STACK_MU_BULLET,  # TODO avoid manually computation
                position=r_ew_w + com_center,
                side_lengths=np.array(CUBOID1_STACK_SIDE_LENGTHS),
            )

            # it is convenient to wrap the control and sim objects
            objects[name] = BalancedObject(ctrl_obj, sim_obj, parent="cylinder_base_stack")

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

        return objects

    def composite_setup(self, objects):
        # find the direct children of each object
        for name, obj in objects.items():
            if obj.parent is not None:
                objects[obj.parent].children.append(name)

        composites = []
        for obj in objects.values():
            # all descendants compose the new object
            descendants = []
            queue = deque([obj])
            while len(queue) > 0:
                desc = queue.popleft()
                descendants.append(desc.ctrl_obj)
                for name in desc.children:
                    queue.append(objects[name])

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
        objects = super().object_setup(r_ew_w, config)

        IPython.embed()

        self.settle(1.0)

        # need to set the CoM after the sim has been settled, so objects are in
        # their proper positions
        # r_ew_w, Q_we = robot.link_pose()
        # for obj in objects.values():
        #     r_ow_w, _ = obj.sim_obj.get_pose()
        #     # TODO none of these are write access!
        #     obj.ctrl_obj.body.com_ellipsoid.center = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

        composites = super().composite_setup(objects)
        return self.robot, objects, composites
