#!/usr/bin/env python
"""Baseline tray balancing formulation."""
import time
import datetime
import sys

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb

import mm_pybullet_sim.util as util
import mm_pybullet_sim.balancing as balancing
from mm_pybullet_sim.simulation import MobileManipulatorSimulation, ROBOT_HOME
from mm_pybullet_sim.recording import Recorder
import mm_pybullet_sim.geometry as geometry

from mm_pybullet_sim.robot import RobotModel

from ocs2_mobile_manipulator_modified import (
    mpc_interface,
    scalar_array,
    vector_array,
    matrix_array,
    TargetTrajectories,
)

from liegroups import SO3
import IPython


# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 30  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)


class Obstacle:
    def __init__(self, initial_position, collision_uid, visual_uid, velocity=None):
        self.uid = pyb.createMultiBody(
            baseMass=0,  # non-dynamic body
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

    # def update_velocity(self, t):
    #     v = self.velocity_func(self.initial_position, t)
    #     pyb.resetBaseVelocity(self.uid, linearVelocity=v)

    def sample_position(self, t):
        """Sample the position of the object at a given time."""
        # assume constant velocity
        return self.initial_position + t * self.velocity
        # positions = np.array([self.initial_position + t * self.velocity for t in ts])
        # return positions


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=SIM_DT)

    N = int(DURATION / sim.dt)

    # simulation objects and model
    robot, objects, composites = sim.setup(obj_names=["tray", "cuboid1"])
    tray = objects["tray"]
    obj = objects["cuboid1"]

    robot_model = RobotModel(0.3, ROBOT_HOME)
    s = 0.2
    r = geometry.equilateral_triangle_inscribed_radius(s)
    support_vertices = np.array([[2 * r, 0], [-r, 0.5 * s], [-r, -0.5 * s]])
    support = geometry.PolygonSupportArea(support_vertices)

    q, v = robot.joint_states()
    r_ew_w, Q_we = robot.link_pose()
    v_ew_w, ω_ew_w = robot.link_velocity()
    r_tw_w, Q_wt = tray.bullet.get_pose()
    r_ow_w, Q_wo = obj.bullet.get_pose()

    # data recorder and plotter
    recorder = Recorder(
        sim.dt,
        DURATION,
        RECORD_PERIOD,
        ns=robot.ns,
        ni=robot.ni,
        control_period=CTRL_PERIOD,
        n_balance_con=5,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    for name, obj in objects.items():
        print(f"{name} CoM = {obj.body.com}")
    IPython.embed()

    r_obs0 = np.array(r_ew_w) + [0, -10, 0]
    v_obs = np.array([0, 0.4, 0])
    obs_radius = 0.1
    obs_collision_uid = pyb.createCollisionShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=obs_radius,
    )
    obs_visual_uid = pyb.createVisualShape(
        shapeType=pyb.GEOM_SPHERE,
        radius=obs_radius,
        rgbaColor=(1, 0, 0, 1),
    )
    obstacle = Obstacle(r_obs0, -1, obs_visual_uid, velocity=v_obs)

    # r_obs5 = obstacle.sample_position(5)

    # initial time, state, and input
    t = 0
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    # target_times = [0, 2, 4, 6, 8, 10]
    target_times = [0]

    # setup MPC and initial EE target pose
    mpc = mpc_interface("mpc")
    t_target = scalar_array()
    for target_time in target_times:
        t_target.push_back(target_time)

    input_target = vector_array()
    for _ in target_times:
        input_target.push_back(u)

    state_target = vector_array()
    # r_ew_w_d = np.array(r_ew_w) + [2, 0, -0.5]
    # r_ew_w_d = np.array(r_ew_w) + [0, 2, 0.5]
    r_ew_w_d = np.array(r_ew_w) + [0, -2, 0]
    # r_ew_w_d = np.array([0, -3, 1])
    Qd = util.quat_multiply(Q_we, np.array([0, 0, 1, 0]))
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [1, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [2, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [3, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [4, 0, 0], Qd, r_obs0)))
    # state_target.push_back(np.concatenate((r_ew_w_d + [4, 0, 0], Qd, r_obs0)))

    target_trajectories = TargetTrajectories(t_target, state_target, input_target)
    mpc.reset(target_trajectories)

    target_idx = 0

    assert len(state_target) == len(target_times)
    assert len(t_target) == len(target_times)
    assert len(input_target) == len(target_times)

    for i in range(N):
        q, v = robot.joint_states()
        x = np.concatenate((q, v))
        mpc.setObservation(t, x, u)

        # TODO this should be set to reflect the MPC time step
        # we can increase it if the MPC rate is faster
        if i % CTRL_PERIOD == 0:
            # robot.cmd_vel = v  # NOTE
            t0 = time.time()
            mpc.advanceMpc()
            t1 = time.time()
            recorder.control_durations[i // CTRL_PERIOD] = t1 - t0
            # print(f"dt = {t1 - t0}")

        # evaluate the current MPC policy (returns an entire trajectory of
        # waypoints, starting from the current time)
        t_result = scalar_array()
        x_result = vector_array()
        u_result = vector_array()
        mpc.getMpcSolution(t_result, x_result, u_result)

        u = u_result[0]
        robot.command_acceleration(u)

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()
            recorder.ineq_cons[idx, :] = mpc.stateInputInequalityConstraint(
                "trayBalance", t, x, u
            )
            # recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
            #     "obstacleAvoidance", t, x
            # )
            # recorder.collision_pair_distance[idx, :] = mpc.stateInequalityConstraint(
            #     "selfCollision", t, x
            # )

            r_ew_w_d = state_target[target_idx][:3]
            Q_we_d = state_target[target_idx][3:7]

            # record
            recorder.us[idx, :] = u
            recorder.xs[idx, :] = x
            recorder.r_ew_wds[idx, :] = r_ew_w_d
            recorder.r_ew_ws[idx, :] = r_ew_w
            recorder.Q_wes[idx, :] = Q_we
            recorder.Q_weds[idx, :] = Q_we_d
            recorder.v_ew_ws[idx, :] = v_ew_w
            recorder.ω_ew_ws[idx, :] = ω_ew_w

            r_tw_w, Q_wt = tray.bullet.get_pose()
            recorder.r_tw_ws[idx, :] = r_tw_w
            recorder.Q_wts[idx, :] = Q_wt

            if len(objects) > 1:
                r_ow_w, Q_wo = obj.bullet.get_pose()
                recorder.r_ow_ws[idx, :] = r_ow_w
                recorder.Q_wos[idx, :] = Q_wo

            # TODO it would be best to avoid recording these directly, but
            # calculate them as needed from recorded data later. Otherwise, you
            # cannot easily recover from weird issues like rotation error
            # changing the frame and blowing up position error
            # recorder.r_te_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_tw_w)
            # recorder.r_oe_es[idx, :] = util.calc_r_te_e(r_ew_w, Q_we, r_ow_w)
            # recorder.r_ot_ts[idx, :] = util.calc_r_te_e(r_tw_w, Q_wt, r_ow_w)
            # recorder.Q_ets[idx, :] = util.calc_Q_et(Q_we, Q_wt)
            # recorder.Q_eos[idx, :] = util.calc_Q_et(Q_we, Q_wo)
            # recorder.Q_tos[idx, :] = util.calc_Q_et(Q_wt, Q_wo)

            recorder.cmd_vels[idx, :] = robot.cmd_vel

        # print(f"cmd_vel before step = {robot.cmd_vel}")
        sim.step(step_robot=True)
        # _, v_test = robot.joint_states()
        # print(f"v after step = {v_test}")
        t += sim.dt
        if t >= target_times[target_idx] and target_idx < len(target_times) - 1:
            target_idx += 1

        # if t > 9:
        #     IPython.embed()
        #     return
        # if i % 100 == 0:
        #     x2 = np.array(
        #         [
        #             -0.00254792,
        #             0.00192868,
        #             0.00280538,
        #             -0.00306004,
        #             -2.35072,
        #             -1.56611,
        #             -0.805204,
        #             -1.55025,
        #             1.5504,
        #             -0.0317851,
        #             0.00444487,
        #             0.0561903,
        #             -0.0399788,
        #             0.0906232,
        #             0.118043,
        #             -0.379447,
        #             0.393679,
        #             -0.391596,
        #         ]
        #     )
        #     u2 = np.array(
        #         [
        #             0.189066,
        #             -0.522982,
        #             0.684831,
        #             -0.227833,
        #             1.16463,
        #             1.79072,
        #             -3.8395,
        #             3.91105,
        #             -3.90864,
        #         ]
        #     )
        #     q2, v2 = x2[:9], x2[9:]
        #     T_we2 = robot_model.tool_pose_matrix(q2)
        #     V_ew_w2 = robot_model.tool_velocity(q2, v2)
        #     A_ew_w2 = robot_model.tool_acceleration(x2, u2)
        #     J2 = robot.jacobian(q2)
        #     V_J = J2 @ v2
        #
        #     r_ew_w, Q_we = robot.link_pose()
        #     C_we = SO3.from_quaternion(Q_we, ordering="xyzw").as_matrix()
        #     v_ew_w, ω_ew_w = robot.link_velocity()
        #     A = robot_model.tool_acceleration(x, u)
        #     zmp = balancing.object_balance_constraints(tray, C_we, ω_ew_w, A[:3], A[3:])
        #     IPython.embed()

    print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    if "--save" in sys.argv:
        fname = "balance_data_" + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        recorder.save(fname)

    last_sim_index = i
    recorder.plot_ee_position(last_sim_index)
    recorder.plot_ee_orientation(last_sim_index)
    recorder.plot_ee_velocity(last_sim_index)
    recorder.plot_r_te_e_error(last_sim_index)
    # recorder.plot_r_oe_e_error(last_sim_index)
    # recorder.plot_r_ot_t_error(last_sim_index)
    recorder.plot_balancing_constraints(last_sim_index)
    recorder.plot_commands(last_sim_index)
    recorder.plot_control_durations()
    recorder.plot_cmd_vs_real_vel(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
