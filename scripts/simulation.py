import numpy as np
import pybullet as pyb
import pybullet_data

from end_effector import EndEffector
import util
import geometry
import bodies


EE_SIDE_LENGTH = 0.3
EE_INSCRIBED_RADIUS = geometry.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

GRAVITY_MAG = 9.81
GRAVITY_VECTOR = np.array([0, 0, -GRAVITY_MAG])

# tray parameters
TRAY_RADIUS = 0.25
TRAY_MASS = 0.5
TRAY_MU = 0.5
TRAY_COM_HEIGHT = 0.01
TRAY_NUM_ZMP_CONSTRAINTS = 1
TRAY_NUM_CONSTRAINTS = 2 + TRAY_NUM_ZMP_CONSTRAINTS

OBJ_MASS = 1
OBJ_TRAY_MU = 0.5
OBJ_TRAY_MU_BULLET = OBJ_TRAY_MU / TRAY_MU
OBJ_RADIUS = 0.1
OBJ_SIDE_LENGTHS = (0.2, 0.2, 0.4)
OBJ_COM_HEIGHT = 0.2
OBJ_ZMP_MARGIN = 0.01
OBJ_NUM_ZMP_CONSTRAINTS = 4
OBJ_NUM_CONSTRAINTS = 2 + OBJ_NUM_ZMP_CONSTRAINTS

NUM_OBJECTS = 2

# SIM_DT = 0.001


class Simulation:
    def __init__(self, dt=0.001):
        self.dt = dt  # simulation timestep (s)

    def settle(self, duration):
        """Run simulation while doing nothing.

        Useful to let objects settle to rest before applying control.
        """
        t = 0
        while t < 1.0:
            pyb.stepSimulation()
            t += self.dt

    # TODO: obj_names could be uses to get rid of some objects that I don't
    # want in the sim
    def setup(self, obj_names=None):
        """Setup pybullet simulation."""
        if EE_INSCRIBED_RADIUS < TRAY_MU * TRAY_COM_HEIGHT:
            print("warning: w < μh")

        pyb.connect(pyb.GUI)

        pyb.setGravity(0, 0, -GRAVITY_MAG)
        pyb.setTimeStep(self.dt)

        pyb.resetDebugVisualizerCamera(
            cameraDistance=4.6,
            cameraYaw=5.2,
            cameraPitch=-27,
            cameraTargetPosition=[1.18, 0.11, 0.05],
        )

        # get rid of extra parts of the GUI
        pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

        # record video
        # pyb.startStateLogging(pyb.STATE_LOGGING_VIDEO_MP4, "no_robot.mp4")

        # setup ground plane
        pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
        pyb.loadURDF("plane.urdf", [0, 0, 0])

        # setup floating end effector
        ee = EndEffector(self.dt, side_length=EE_SIDE_LENGTH, position=(0, 0, 1))

        util.debug_frame(0.1, ee.uid, -1)

        # setup tray
        tray = bodies.Cylinder(
            r_tau=EE_INSCRIBED_RADIUS,
            support_area=geometry.CircleSupportArea(EE_INSCRIBED_RADIUS),
            mass=TRAY_MASS,
            radius=TRAY_RADIUS,
            height=2 * TRAY_COM_HEIGHT,
            mu=TRAY_MU,
            bullet_mu=TRAY_MU,
        )
        ee_pos, _ = ee.get_pose()
        tray.bullet.reset_pose(position=ee_pos + [0, 0, TRAY_COM_HEIGHT + 0.05])

        # object on tray
        # obj = Cylinder(
        #     r_tau=geometry.circle_r_tau(OBJ_RADIUS),
        #     support_area=geometry.CircleSupportArea(OBJ_RADIUS, margin=OBJ_ZMP_MARGIN),
        #     mass=OBJ_MASS,
        #     radius=OBJ_RADIUS,
        #     height=2 * OBJ_COM_HEIGHT,
        #     mu=OBJ_TRAY_MU,
        #     bullet_mu=OBJ_TRAY_MU_BULLET,
        #     color=(0, 1, 0, 1),
        # )
        # obj.bullet.reset_pose(position=ee_pos + [0, 0, 2 * TRAY_COM_HEIGHT + OBJ_COM_HEIGHT + 0.05])

        obj_support = geometry.PolygonSupportArea(
            geometry.cuboid_support_vertices(OBJ_SIDE_LENGTHS), margin=OBJ_ZMP_MARGIN
        )
        obj = bodies.Cuboid(
            r_tau=geometry.circle_r_tau(OBJ_RADIUS),
            support_area=obj_support,
            mass=OBJ_MASS,
            side_lengths=OBJ_SIDE_LENGTHS,
            mu=OBJ_TRAY_MU,
            bullet_mu=OBJ_TRAY_MU_BULLET,
            color=(0, 1, 0, 1),
        )
        obj.bullet.reset_pose(
            position=ee_pos
            + [0.05, 0, 2 * TRAY_COM_HEIGHT + 0.5 * OBJ_SIDE_LENGTHS[2] + 0.05]
        )

        self.settle(1.0)

        # set CoMs of the objects relative to EE
        r_ew_w, Q_we = ee.get_pose()
        r_tw_w, Q_wt = tray.bullet.get_pose()
        r_ow_w, Q_wo = obj.bullet.get_pose()

        tray.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_tw_w)
        obj.body.com = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)

        obj_tray_composite = tray.copy()
        obj_tray_composite.body = bodies.compose_bodies([tray.body, obj.body])
        delta = tray.body.com - obj_tray_composite.body.com
        obj_tray_composite.support_area.offset = delta[:2]
        obj_tray_composite.com_height = tray.com_height - delta[2]

        objects = {"tray": tray, "obj": obj}
        composites = [obj_tray_composite, obj]

        return ee, objects, composites
