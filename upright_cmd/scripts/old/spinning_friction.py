#!/usr/bin/env python3
import time
import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import pybullet_data
import liegroups

from upright_sim import geometry, bodies, util

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
CUBOID_SHORT_TRAY_MU = 0.1
CUBOID_SHORT_COM_HEIGHT = 0.075
CUBOID_SHORT_SIDE_LENGTHS = (0.15, 0.15, 2 * CUBOID_SHORT_COM_HEIGHT)
# CUBOID_SHORT_R_TAU = 1 * geometry.rectangle_r_tau(*CUBOID_SHORT_SIDE_LENGTHS[:2])
CUBOID_SHORT_R_TAU = 1 * np.linalg.norm(CUBOID_SHORT_SIDE_LENGTHS[:2])
CUBOID_SHORT_MU_BULLET = CUBOID_SHORT_TRAY_MU / TRAY_MU_BULLET

# compute given pybullet's spinning friction combination rule
# see btManifoldResult.cpp line 43 in Bullet source code
# CUBOID_SHORT_R_TAU_BULLET = (
#     CUBOID_SHORT_R_TAU - CUBOID_SHORT_MU_BULLET * TRAY_R_TAU_BULLET
# ) / TRAY_MU_BULLET

CUBOID_SHORT_R_TAU_BULLET = CUBOID_SHORT_R_TAU - TRAY_R_TAU_BULLET


def main():
    np.set_printoptions(precision=3, suppress=True)

    N = int(DURATION / SIM_DT)
    ts = SIM_DT * np.arange(N)

    pyb.connect(pyb.GUI, options="--width=1280 --height=720")
    # pyb.setPhysicsEngineParameter(enableConeFriction=0)
    pyb.setGravity(0, 0, -9.81)
    pyb.setTimeStep(SIM_DT)
    pyb.configureDebugVisualizer(pyb.COV_ENABLE_GUI, 0)

    pyb.setAdditionalSearchPath(pybullet_data.getDataPath())
    ground_uid = pyb.loadURDF("plane.urdf", [0, 0, 0])
    pyb.changeDynamics(
        ground_uid,
        -1,
        lateralFriction=0.0,
    )

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

    # IPython.embed()

    torque_fric = CUBOID_SHORT_TRAY_MU * CUBOID_SHORT_R_TAU * 9.81 * CUBOID_SHORT_MASS
    torque_app = 0.2
    rel_acc = np.linalg.solve(cube.inertia, [0, 0, torque_app - torque_fric])[2]
    rel_vel = 0
    rel_ang = 0

    print(torque_fric)
    # print(rel_acc)

    ω = 0
    dω = 20.0

    cube_positions = np.zeros((N, 3))
    # cube_orientations = np.zeros((N, 4))
    cube_yaws = np.zeros(N)

    # settle things
    t = 0
    while t < 1.0:
        pyb.stepSimulation()
        t += SIM_DT

    rel_angs = np.zeros(N)

    # simulation loop
    t = 0
    for i in range(N):
        # ω += dω * SIM_DT
        # pyb.resetBaseVelocity(tray.bullet.uid, [0, 0, 0], [0, 0, ω])
        rel_vel += SIM_DT * rel_acc
        rel_ang += SIM_DT * rel_vel
        rel_angs[i] = rel_ang

        pyb.applyExternalTorque(tray.bullet.uid, -1, [0, 0, torque_app], flags=pyb.LINK_FRAME)

        r_tw_w, Q_wt = pyb.getBasePositionAndOrientation(tray.bullet.uid)
        r_cw_w, Q_wc = pyb.getBasePositionAndOrientation(cube.bullet.uid)
        # cube_positions[i, :] = r
        # cube_orientations[i, :] = Q

        # compute beta and compare with μ * normal force
        C_ew = util.quaternion_to_matrix(Q_wc).T
        ω_ew_e = C_ew @ [0, 0, ω]
        β = np.cross(ω_ew_e, cube.inertia @ ω_ew_e) + cube.inertia @ C_ew @ [0, 0, dω]
        fn = cube.mu * 9.81 * cube.mass
        r_tau_act = np.abs(β[2]) / fn
        # print(r_tau_act)
        # print(β[2])

        # IPython.embed()
        # return

        r_ct_t = util.calc_r_te_e(np.array(r_tw_w), np.array(Q_wt), np.array(r_cw_w))
        Q_tc = util.calc_Q_et(Q_wt, Q_wc)

        cube_positions[i, :] = r_ct_t
        cube_yaws[i] = liegroups.SO3.from_quaternion(Q_tc, ordering="xyzw").to_rpy()[2]

        pyb.stepSimulation()
        t += SIM_DT
        # time.sleep(SIM_DT)

        if t >= 1.0:
            res = pyb.getContactPoints(bodyA=tray.bullet.uid, bodyB=cube.bullet.uid)
            fric_torques = np.zeros(len(res))
            for i, point in enumerate(res):
                fn = point[-5]
                ff1 = point[-4] * np.array(point[-3])
                ff2 = point[-2] * np.array(point[-1])
                ff = ff1 + ff2

                ff_norm = np.linalg.norm(ff)
                ff_pred = fn * cube.mu

                contact_w = np.array(point[6])
                contact_c = contact_w - r_cw_w + [0, 0, CUBOID_SHORT_COM_HEIGHT]
                torque = np.cross(contact_c, ff)
                torque_pred = cube.r_tau * cube.mu * fn  # TODO doesn't make sense...

                fric_torques[i] = torque[2]

                print(f"normal force = {fn}")
                print(f"fric force = {ff_norm}")
                print(f"fric force (pred) = {ff_pred}")
                print(f"fric torque = {torque[2]}")
                print(f"fric torque (pred) = {torque_pred}")
                print(f"r_tau = {np.linalg.norm(contact_c[:2])}")

            print("---")
            print(f"total fn = {np.sum([p[-5] for p in res])}")
            print(f"total fric torque = {np.sum(fric_torques)}")

            IPython.embed()

    plt.figure()
    plt.plot(ts, cube_positions[:, 0], label="x")
    plt.plot(ts, cube_positions[:, 1], label="y")
    plt.plot(ts, cube_yaws, label="yaw")
    plt.plot(ts, rel_angs, label="predicted yaw")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
