#!/usr/bin/env python3
"""Baseline tray balancing formulation."""
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
from PIL import Image

from tray_balance_sim import util, ocs2_util
from tray_balance_sim.simulation import MobileManipulatorSimulation
from tray_balance_sim.recording import Recorder, VideoRecorder

from ocs2_mobile_manipulator_modified import (
    scalar_array,
    vector_array,
    matrix_array,
    TargetTrajectories,
)

import IPython


# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 100  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 10.0  # duration of trajectory (s)

TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_DIR = Path("/media/adam/Data/PhD/Videos/ICRA22/")
VIDEO_PATH = VIDEO_DIR / ("dynamic_obstacle_" + TIMESTAMP)
VIDEO_PERIOD = 40  # 25 frames per second with 1000 steps per second
RECORD_VIDEO = False


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
            baseMass=0,  # non-dynamic body
            baseCollisionShapeIndex=-1,  # NOTE
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


def main():
    np.set_printoptions(precision=3, suppress=True)

    sim = MobileManipulatorSimulation(dt=SIM_DT)
    settings = ocs2_util.load_ocs2_task_settings()

    if RECORD_VIDEO:
        # dynamic obstacle course POV #1
        # video = VideoRecorder(
        #     path=VIDEO_PATH,
        #     distance=1.8,
        #     roll=0,
        #     pitch=-29,
        #     yaw=147.6,
        #     target_position=[1.28, 0.045, 0.647],
        # )

        # dynamic obstacle course POV #2
        video = VideoRecorder(
            path=VIDEO_PATH,
            distance=2.6,
            roll=0,
            pitch=-20.6,
            yaw=-3.2,
            target_position=[1.28, 0.045, 0.647],
        )

        # static obstacle course POV #3
        # video = VideoRecorder(
        #     path=VIDEO_PATH,
        #     distance=4.8,
        #     roll=0,
        #     pitch=-13.4,
        #     yaw=87.6,
        #     target_position=[2.77, 0.043, 0.142],
        # )

    N = int(DURATION / sim.dt)

    # simulation objects and model
    robot, objects, _ = sim.setup(settings.tray_balance_settings)

    q, v = robot.joint_states()
    r_ew_w, Q_we = robot.link_pose()
    v_ew_w, ω_ew_w = robot.link_velocity()

    n_balance_con_tray = 5
    n_balance_con_obj = 6

    # data recorder and plotter
    recorder = Recorder(
        sim.dt,
        DURATION,
        RECORD_PERIOD,
        ns=robot.ns,
        ni=robot.ni,
        n_objects=len(objects),
        control_period=CTRL_PERIOD,
        n_balance_con=3*1,
        n_collision_pair=settings.num_collision_pairs,
        n_dynamic_obs=settings.num_dynamic_obstacle_pairs,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    for name, obj in objects.items():
        print(f"{name} CoM = {obj.body.com}")

    # create the dynamic obstacle
    r_obs0 = np.array(r_ew_w) + [0, -1, 0]
    v_obs = np.array([0, 1.0, 0])
    obstacle = DynamicObstacle(r_obs0, radius=0.1, velocity=v_obs)

    # initial time, state, and input
    t = 0.0
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    target_times = [0, 5]  # TODO

    # setup MPC and initial EE target pose
    mpc = ocs2_util.setup_ocs2_mpc_interface()
    t_target = scalar_array()
    for target_time in target_times:
        t_target.push_back(target_time)

    input_target = vector_array()
    for _ in target_times:
        input_target.push_back(u)

    # stationary
    r_ew_w_d = r_ew_w
    Qd = Q_we

    # start with the obstacle out of the way
    r_obs_out_of_the_way = r_ew_w + [0, -10, 0]
    state_target = vector_array()
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs_out_of_the_way)))
    state_target.push_back(np.concatenate((r_ew_w_d, Qd, r_obs_out_of_the_way)))

    target_times_obs1 = [0, 5]
    target_times_obs2 = [2, 7]
    r_obsf = obstacle.sample_position(target_times_obs1[-1])

    t_target_obs1 = scalar_array()
    t_target_obs1.push_back(target_times_obs1[0])
    t_target_obs1.push_back(target_times_obs1[1])

    t_target_obs2 = scalar_array()
    t_target_obs2.push_back(target_times_obs2[0])
    t_target_obs2.push_back(target_times_obs2[1])

    state_target_obs1 = vector_array()
    state_target_obs1.push_back(np.concatenate((r_ew_w_d, Qd, r_obs0)))
    state_target_obs1.push_back(np.concatenate((r_ew_w_d, Qd, r_obsf)))

    state_target_obs2 = state_target_obs1

    # obstacle appears at t = 0
    target_trajectories_obs1 = TargetTrajectories(
        t_target_obs1, state_target_obs1, input_target
    )

    # obstacle appears at t = 2
    target_trajectories_obs2 = TargetTrajectories(
        t_target_obs2, state_target_obs2, input_target
    )

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

        # this should be set to reflect the MPC time step
        # we can increase it if the MPC rate is faster
        if i % CTRL_PERIOD == 0:
            # robot.cmd_vel = v  # NOTE
            try:
                t0 = time.time()
                mpc.advanceMpc()
                t1 = time.time()
            except RuntimeError as e:
                print(e)
                IPython.embed()
                i -= 1  # for the recorder
                break
            recorder.control_durations[i // CTRL_PERIOD] = t1 - t0

        # As far as I can tell, evaluateMpcSolution actually computes the input
        # for the particular time and state (the input is often at least
        # state-varying in DDP, with linear feedback on state error). OTOH,
        # getMpcSolution just gives the current MPC policy trajectory over the
        # entire time horizon, without accounting for the given state. So it is
        # like doing feedforward input only, which is bad.
        x_opt = np.zeros(robot.ns)
        u = np.zeros(robot.ni)
        mpc.evaluateMpcSolution(t, x, x_opt, u)

        robot.command_acceleration(u)

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()

            if settings.tray_balance_settings.enabled:
                recorder.ineq_cons[idx, :] = mpc.softStateInputInequalityConstraint(
                    "trayBalance", t, x, u
                )
            if settings.dynamic_obstacle_enabled:
                recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
                    "dynamicObstacleAvoidance", t, x
                )
            if settings.collision_avoidance_enabled:
                recorder.collision_pair_distance[idx, :] = mpc.stateInequalityConstraint(
                    "collisionAvoidance", t, x
                )

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

            for j, obj in enumerate(objects.values()):
                r, Q = obj.bullet.get_pose()
                recorder.r_ow_ws[j, idx, :] = r
                recorder.Q_wos[j, idx, :] = Q

            recorder.cmd_vels[idx, :] = robot.cmd_vel

        sim.step(step_robot=True)
        t += sim.dt

        # set the target trajectories to make controller aware of dynamic
        # obstacles
        if i == 0:
            mpc.setTargetTrajectories(target_trajectories_obs1)
        elif i == 2000:
            # reset the obstacle to use again
            pyb.resetBasePositionAndOrientation(obstacle.uid, list(r_obs0), (0, 0, 0, 1))
            pyb.resetBaseVelocity(obstacle.uid, linearVelocity=list(obstacle.velocity))
            mpc.setTargetTrajectories(target_trajectories_obs2)

        if RECORD_VIDEO and i % VIDEO_PERIOD == 0:
            video.save_frame()

    if recorder.ineq_cons.shape[1] > 0:
        print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    # save logged data
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        if len(sys.argv) > 2:
            prefix = sys.argv[2]
        else:
            prefix = "data"
        fname = prefix + "_" + TIMESTAMP
        recorder.save(fname)

    # last_sim_index = i
    # recorder.plot_ee_position(last_sim_index)
    # recorder.plot_ee_orientation(last_sim_index)
    # recorder.plot_ee_velocity(last_sim_index)
    # for j in range(len(objects)):
    #     recorder.plot_object_error(last_sim_index, j)
    # recorder.plot_balancing_constraints(last_sim_index)
    # recorder.plot_commands(last_sim_index)
    # recorder.plot_control_durations(last_sim_index)
    # recorder.plot_cmd_vs_real_vel(last_sim_index)
    # recorder.plot_joint_config(last_sim_index)
    # recorder.plot_dynamic_obs_dist(last_sim_index)
    #
    # plt.show()


if __name__ == "__main__":
    main()