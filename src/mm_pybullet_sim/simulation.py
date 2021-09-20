import numpy as np
import pybullet as pyb
import pybullet_data

from mm_pybullet_sim.end_effector import EndEffector
from mm_pybullet_sim.robot import SimulatedRobot
import mm_pybullet_sim.util as util
import mm_pybullet_sim.geometry as geometry
import mm_pybullet_sim.bodies as bodies

import IPython


OBSTACLES_URDF_PATH = "/home/adam/phd/code/mm/ocs2_noetic/catkin_ws/src/ocs2_mobile_manipulator_modified/urdf/obstacles.urdf"


EE_SIDE_LENGTH = 0.2
EE_INSCRIBED_RADIUS = geometry.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

GRAVITY_MAG = 9.81
GRAVITY_VECTOR = np.array([0, 0, -GRAVITY_MAG])

EE_MU = 1.0

# tray parameters
TRAY_RADIUS = 0.2
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_COM_HEIGHT = 0.01
TRAY_MU_BULLET = TRAY_MU / EE_MU

CUBOID1_MASS = 0.5
CUBOID1_TRAY_MU = 0.5
CUBOID1_MU_BULLET = CUBOID1_TRAY_MU / TRAY_MU_BULLET
CUBOID1_COM_HEIGHT = 0.075
CUBOID1_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID1_COM_HEIGHT)

CUBOID2_MASS = 0.5
CUBOID2_TRAY_MU = 0.5
CUBOID2_MU_BULLET = CUBOID2_TRAY_MU / CUBOID1_MU_BULLET
CUBOID2_COM_HEIGHT = 0.075
CUBOID2_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID2_COM_HEIGHT)

CUBOID3_MASS = 0.5
CUBOID3_TRAY_MU = 0.5
CUBOID3_MU_BULLET = CUBOID3_TRAY_MU / CUBOID2_MU_BULLET
CUBOID3_COM_HEIGHT = 0.075
CUBOID3_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID3_COM_HEIGHT)

CYLINDER1_MASS = 0.5
CYLINDER1_SUPPORT_MU = 0.5
CYLINDER1_MU_BULLET = CYLINDER1_SUPPORT_MU / TRAY_MU_BULLET
CYLINDER1_RADIUS = 0.05
CYLINDER1_COM_HEIGHT = 0.075

CYLINDER2_MASS = 0.5
CYLINDER2_SUPPORT_MU = 0.5
CYLINDER2_MU_BULLET = CYLINDER2_SUPPORT_MU / TRAY_MU_BULLET  # for flat
# CYLINDER2_MU_BULLET = CYLINDER2_SUPPORT_MU / CYLINDER1_MU_BULLET  # for stacked
CYLINDER2_RADIUS = 0.05
CYLINDER2_COM_HEIGHT = 0.075

CYLINDER3_MASS = 0.5
CYLINDER3_SUPPORT_MU = 0.5
CYLINDER3_MU_BULLET = CYLINDER3_SUPPORT_MU / TRAY_MU_BULLET  # for flat
# CYLINDER3_MU_BULLET = CYLINDER3_SUPPORT_MU / CYLINDER2_MU_BULLET  # for stacked
CYLINDER3_RADIUS = 0.05
CYLINDER3_COM_HEIGHT = 0.075

OBJ_ZMP_MARGIN = 0

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
        if step_robot:
            self.robot.step()
        pyb.stepSimulation()

    def record_video(self, file_name):
        self.video_file_name = str(file_name)

    def basic_setup(self):
        if EE_INSCRIBED_RADIUS < TRAY_MU * TRAY_COM_HEIGHT:
            print("warning: w < μh")

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
        pyb.resetDebugVisualizerCamera(
            cameraDistance=1.8,
            cameraYaw=147.6,
            cameraPitch=-29,
            cameraTargetPosition=[1.28, 0.045, 0.647],
        )

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
        # obstacles_uid = pyb.loadURDF(OBSTACLES_URDF_PATH)
        # pyb.changeDynamics(obstacles_uid, -1, mass=0)  # change to static object

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

    def object_setup(self, r_ew_w, obj_names):
        # setup balanced objects
        objects = {}

        if "tray" in obj_names:
            objects["tray"] = bodies.Cylinder(
                r_tau=EE_INSCRIBED_RADIUS,
                support_area=geometry.CircleSupportArea(EE_INSCRIBED_RADIUS),
                mass=TRAY_MASS,
                radius=TRAY_RADIUS,
                height=2 * TRAY_COM_HEIGHT,
                mu=TRAY_MU,
            )
            objects["tray"].add_to_sim(
                bullet_mu=TRAY_MU_BULLET, color=(0.122, 0.467, 0.706, 1)
            )
            r_tw_w = r_ew_w + [0, 0, TRAY_COM_HEIGHT + 0.05]
            objects["tray"].bullet.reset_pose(position=r_tw_w)

        c1, c2, c3 = self.compute_cylinder_xy_positions(L=0.08)

        if "cylinder1" in obj_names:
            objects["cylinder1"] = bodies.Cylinder(
                r_tau=geometry.circle_r_tau(CYLINDER1_RADIUS),
                support_area=geometry.CircleSupportArea(
                    CYLINDER1_RADIUS, margin=OBJ_ZMP_MARGIN
                ),
                mass=CYLINDER1_MASS,
                radius=CYLINDER1_RADIUS,
                height=2 * CYLINDER1_COM_HEIGHT,
                mu=CYLINDER1_SUPPORT_MU,
            )
            objects["cylinder1"].add_to_sim(
                bullet_mu=CYLINDER1_MU_BULLET, color=(1, 0.498, 0.055, 1)
            )

            # flat
            r_ow_w = r_ew_w + [
                c1[0],
                c1[1],
                2 * TRAY_COM_HEIGHT + CYLINDER1_COM_HEIGHT + 0.05,
            ]

            # stacked
            # r_ow_w = r_ew_w + [
            #     0,
            #     0,
            #     2 * TRAY_COM_HEIGHT
            #     + CYLINDER1_COM_HEIGHT
            #     + 0.05,
            # ]
            objects["cylinder1"].bullet.reset_pose(position=r_ow_w)
            objects["tray"].children.append("cylinder1")

        if "cylinder2" in obj_names:
            objects["cylinder2"] = bodies.Cylinder(
                r_tau=geometry.circle_r_tau(CYLINDER2_RADIUS),
                support_area=geometry.CircleSupportArea(
                    CYLINDER2_RADIUS, margin=OBJ_ZMP_MARGIN
                ),
                mass=CYLINDER2_MASS,
                radius=CYLINDER2_RADIUS,
                height=2 * CYLINDER2_COM_HEIGHT,
                mu=CYLINDER2_SUPPORT_MU,
            )
            objects["cylinder2"].add_to_sim(
                bullet_mu=CYLINDER2_MU_BULLET, color=(0.173, 0.627, 0.173, 1)
            )

            # flat
            r_ow_w = r_ew_w + [
                c2[0],
                c2[1],
                2 * TRAY_COM_HEIGHT + CYLINDER2_COM_HEIGHT + 0.05,
            ]

            # stacked
            # r_ow_w = r_ew_w + [
            #     0,
            #     0,
            #     2 * TRAY_COM_HEIGHT
            #     + 2 * CYLINDER1_COM_HEIGHT
            #     + CYLINDER2_COM_HEIGHT
            #     + 0.05,
            # ]
            objects["cylinder2"].bullet.reset_pose(position=r_ow_w)
            objects["tray"].children.append("cylinder2")

        if "cylinder3" in obj_names:
            objects["cylinder3"] = bodies.Cylinder(
                r_tau=geometry.circle_r_tau(CYLINDER3_RADIUS),
                support_area=geometry.CircleSupportArea(
                    CYLINDER3_RADIUS, margin=OBJ_ZMP_MARGIN
                ),
                mass=CYLINDER3_MASS,
                radius=CYLINDER3_RADIUS,
                height=2 * CYLINDER3_COM_HEIGHT,
                mu=CYLINDER3_SUPPORT_MU,
            )
            objects["cylinder3"].add_to_sim(
                bullet_mu=CYLINDER3_MU_BULLET, color=(0.839, 0.153, 0.157, 1)
            )

            # flat
            r_ow_w = r_ew_w + [
                c3[0],
                c3[1],
                2 * TRAY_COM_HEIGHT + CYLINDER3_COM_HEIGHT + 0.05,
            ]

            # stacked
            # r_ow_w = r_ew_w + [
            #     0,
            #     0,
            #     2 * TRAY_COM_HEIGHT
            #     + 2 * CYLINDER1_COM_HEIGHT
            #     + 2 * CYLINDER2_COM_HEIGHT
            #     + CYLINDER3_COM_HEIGHT
            #     + 0.05,
            # ]
            objects["cylinder3"].bullet.reset_pose(position=r_ow_w)
            objects["tray"].children.append("cylinder3")

        if "cuboid1" in obj_names:
            support = geometry.PolygonSupportArea(
                geometry.cuboid_support_vertices(CUBOID1_SIDE_LENGTHS),
                margin=OBJ_ZMP_MARGIN,
            )
            objects["cuboid1"] = bodies.Cuboid(
                r_tau=geometry.circle_r_tau(CUBOID1_SIDE_LENGTHS[0] * 0.5),  # TODO
                support_area=support,
                mass=CUBOID1_MASS,
                side_lengths=CUBOID1_SIDE_LENGTHS,
                mu=CUBOID1_TRAY_MU,
            )
            objects["cuboid1"].add_to_sim(
                bullet_mu=CUBOID1_MU_BULLET, color=(0, 1, 0, 1)
            )
            r_ow_w = r_ew_w + [
                0,
                0,
                2 * TRAY_COM_HEIGHT + 0.5 * CUBOID1_SIDE_LENGTHS[2] + 0.05,
            ]
            objects["cuboid1"].bullet.reset_pose(position=r_ow_w)
            objects["tray"].children.append("cuboid1")

        if "cuboid2" in obj_names:
            support = geometry.PolygonSupportArea(
                geometry.cuboid_support_vertices(CUBOID2_SIDE_LENGTHS),
                margin=OBJ_ZMP_MARGIN,
            )
            objects["cuboid2"] = bodies.Cuboid(
                r_tau=geometry.circle_r_tau(CUBOID2_SIDE_LENGTHS[0] * 0.5),  # TODO
                support_area=support,
                mass=CUBOID2_MASS,
                side_lengths=CUBOID2_SIDE_LENGTHS,
                mu=CUBOID2_TRAY_MU,
            )
            objects["cuboid2"].add_to_sim(
                bullet_mu=CUBOID2_MU_BULLET, color=(1, 0, 0, 1)
            )
            r_ow_w = r_ew_w + [
                0,
                0,
                2 * TRAY_COM_HEIGHT
                + CUBOID1_SIDE_LENGTHS[2]
                + 0.5 * CUBOID2_SIDE_LENGTHS[2]
                + 0.05,
            ]
            objects["cuboid2"].bullet.reset_pose(position=r_ow_w)
            objects["tray"].children.append("cuboid2")

        if "cuboid3" in obj_names:
            support = geometry.PolygonSupportArea(
                geometry.cuboid_support_vertices(CUBOID3_SIDE_LENGTHS),
                margin=OBJ_ZMP_MARGIN,
            )
            objects["cuboid3"] = bodies.Cuboid(
                r_tau=geometry.circle_r_tau(CUBOID3_SIDE_LENGTHS[0] * 0.5),  # TODO
                support_area=support,
                mass=CUBOID3_MASS,
                side_lengths=CUBOID3_SIDE_LENGTHS,
                mu=CUBOID3_TRAY_MU,
            )
            objects["cuboid3"].add_to_sim(
                bullet_mu=CUBOID3_MU_BULLET, color=(1, 0, 1, 1)
            )
            r_ow_w = r_ew_w + [
                0,
                0,
                2 * TRAY_COM_HEIGHT
                + CUBOID1_SIDE_LENGTHS[2]
                + CUBOID2_SIDE_LENGTHS[2]
                + 0.5 * CUBOID3_SIDE_LENGTHS[2]
                + 0.05,
            ]
            objects["cuboid3"].bullet.reset_pose(position=r_ow_w)
            objects["tray"].children.append("cuboid3")

        return objects

    def composite_setup(self, objects):
        tray = objects["tray"]
        assert len(tray.children) <= 1
        if len(tray.children) > 0:
            # TODO this would be straightforward to extend to many objects on
            # the tray
            obj = objects[tray.children[0]]
            obj_tray_composite = tray.copy()
            obj_tray_composite.body = bodies.compose_bodies([tray.body, obj.body])
            delta = tray.body.com - obj_tray_composite.body.com
            obj_tray_composite.support_area.offset = delta[:2]
            obj_tray_composite.com_height = tray.com_height - delta[2]
            composites = [obj_tray_composite, obj]
        else:
            composites = [tray]
        return composites


class FloatingEESimulation(Simulation):
    def __init__(self, dt=0.001):
        super().__init__(dt)

    def setup(self, obj_names=None):
        """Setup pybullet simulation."""
        super().basic_setup()

        # setup floating end effector
        robot = EndEffector(self.dt, side_length=EE_SIDE_LENGTH, position=(0, 0, 1))
        self.robot = robot
        util.debug_frame(0.1, robot.uid, -1)

        r_ew_w, Q_we = robot.get_pose()
        objects = super().object_setup(r_ew_w, obj_names)

        self.settle(1.0)

        # need to set the CoM after the sim has been settled, so objects are in
        # their proper positions
        r_ew_w, Q_we = robot.get_pose()
        for obj in objects.values():
            r_ow_w, _ = obj.bullet.get_pose()
            obj.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

        composites = super().composite_setup(objects)

        return robot, objects, composites


class MobileManipulatorSimulation(Simulation):
    def __init__(self, dt=0.001):
        super().__init__(dt)

    def setup(self, obj_names=None):
        """Setup pybullet simulation."""
        super().basic_setup()

        # setup floating end effector
        # robot = EndEffector(self.dt, side_length=EE_SIDE_LENGTH, position=(0, 0, 1))
        robot = SimulatedRobot(self.dt)
        self.robot = robot
        util.debug_frame(0.1, robot.uid, -1)
        robot.reset_joint_configuration(ROBOT_HOME)

        # simulate briefly to let the robot settle down after being positioned
        self.settle(1.0)

        # arm gets bumped by the above settling, so we reset it again
        robot.reset_arm_joints(UR10_HOME_TRAY_BALANCE)
        # robot.reset_joint_configuration(ROBOT_HOME)

        # self.settle(1.0)

        r_ew_w, _ = robot.link_pose()
        objects = super().object_setup(r_ew_w, obj_names)

        self.settle(1.0)

        # need to set the CoM after the sim has been settled, so objects are in
        # their proper positions
        r_ew_w, Q_we = robot.link_pose()
        for obj in objects.values():
            r_ow_w, _ = obj.bullet.get_pose()
            obj.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

        # composites = super().composite_setup(objects)
        composites = None

        return robot, objects, composites
