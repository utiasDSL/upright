#!/usr/bin/env python3
import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data
import liegroups

from tray_balance_sim import geometry, bodies, util

import IPython


# simulation parameters
SIM_DT = 0.001
DURATION = 20.0  # duration of trajectory (s)

EE_SIDE_LENGTH = 0.2
EE_INSCRIBED_RADIUS = geometry.equilateral_triangle_inscribed_radius(EE_SIDE_LENGTH)

TRAY_RADIUS = 0.2
TRAY_MASS = 0.5
TRAY_MU = 1.0
TRAY_COM_HEIGHT = 0.01
TRAY_MU_BULLET = TRAY_MU
TRAY_R_TAU = EE_INSCRIBED_RADIUS
TRAY_R_TAU_BULLET = TRAY_R_TAU

CUBOID_SHORT_MASS = 0.5
CUBOID_SHORT_TRAY_MU = 0.5
CUBOID_SHORT_COM_HEIGHT = 0.075
CUBOID_SHORT_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID_SHORT_COM_HEIGHT)
CUBOID_SHORT_R_TAU = geometry.rectangle_r_tau(*CUBOID_SHORT_SIDE_LENGTHS[:2])
CUBOID_SHORT_MU_BULLET = CUBOID_SHORT_TRAY_MU / TRAY_MU_BULLET

# compute given pybullet's spinning friction combination rule
CUBOID_SHORT_R_TAU_BULLET = (
    CUBOID_SHORT_R_TAU - CUBOID_SHORT_MU_BULLET * TRAY_R_TAU_BULLET
) / TRAY_MU_BULLET


def main():
    np.set_printoptions(precision=3, suppress=True)

    N = int(DURATION / SIM_DT)
    t = 0.0
    ts = SIM_DT * np.arange(N)

    pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    # pyb.setPhysicsEngineParameter(enableConeFriction=0)
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(SIM_DT)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    # TODO friction between ground and cube seems wrong/small
    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_uid = pyb.loadURDF("plane.urdf", [0, 0, 0])
    # pyb.changeDynamics(
    #     ground_uid,
    #     -1,
    #     lateralFriction=0.0,
    #     anisotropicFriction=0.0,
    #     spinningFriction=0.0,
    # )

    tray = bodies.Cylinder(
        r_tau=TRAY_R_TAU_BULLET,
        support_area=None,
        mass=TRAY_MASS,
        radius=TRAY_RADIUS,
        height=2 * TRAY_COM_HEIGHT,
        mu=TRAY_MU,
    )
    tray.add_to_sim(bullet_mu=TRAY_MU_BULLET)
    tray.bullet.reset_pose(position=[0, 0, 0.1])

    cube = bodies.Cuboid(
        r_tau=CUBOID_SHORT_R_TAU_BULLET,
        support_area=None,
        mass=CUBOID_SHORT_MASS,
        side_lengths=CUBOID_SHORT_SIDE_LENGTHS,
        mu=CUBOID_SHORT_TRAY_MU,
    )
    cube.add_to_sim(bullet_mu=CUBOID_SHORT_MU_BULLET, color=(1, 0, 0, 1))
    cube.bullet.reset_pose(position=[0, 0, 0.2])

    ω = 0
    dω = 2.0

    cube_positions = np.zeros((N, 3))
    # cube_orientations = np.zeros((N, 4))
    cube_yaws = np.zeros(N)

    # simulation loop
    for i in range(N):

        ω += dω * SIM_DT
        pyb.resetBaseVelocity(tray.bullet.uid, [0, 0, 0], [0, 0, ω])

        r_tw_w, Q_wt = pyb.getBasePositionAndOrientation(tray.bullet.uid)
        r_cw_w, Q_wc = pyb.getBasePositionAndOrientation(cube.bullet.uid)
        # cube_positions[i, :] = r
        # cube_orientations[i, :] = Q

        r_ct_t = util.calc_r_te_e(np.array(r_tw_w), np.array(Q_wt), np.array(r_cw_w))
        Q_tc = util.calc_Q_et(Q_wt, Q_wc)

        cube_positions[i, :] = r_ct_t
        cube_yaws[i] = liegroups.SO3.from_quaternion(Q_tc, ordering="xyzw").to_rpy()[2]

        pyb.stepSimulation()
        t += SIM_DT
        # time.sleep(SIM_DT)

    plt.figure()
    plt.plot(ts, cube_positions[:, 0], label="x")
    plt.plot(ts, cube_positions[:, 1], label="y")
    plt.plot(ts, cube_yaws, label="yaw")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
