#!/usr/bin/env python3
"""Testing of the robust balancing constraints"""
import enum
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import rospkg

from tray_balance_sim import util, ocs2_util, pyb_util, robustness
from tray_balance_sim.simulation import MobileManipulatorSimulation, DynamicObstacle
from tray_balance_sim.recording import Recorder, VideoRecorder

import IPython

# hook into the bindings from the OCS2-based controller
import tray_balance_ocs2.MobileManipulatorPythonInterface as ocs2


@enum.unique
class SimType(enum.Enum):
    POSE_TO_POSE = 1
    DYNAMIC_OBSTACLE = 2
    STATIC_OBSTACLE = 3


SIM_TYPE = SimType.DYNAMIC_OBSTACLE


# simulation parameters
SIM_DT = 0.001
CTRL_PERIOD = 20  # generate new control signal every CTRL_PERIOD timesteps
RECORD_PERIOD = 10
DURATION = 6.0  # duration of trajectory (s)

# state noise
Q_STDEV = 0.0
V_STDEV = 0.0
V_CMD_STDEV = 0.0

# video recording parameters
TIMESTAMP = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
VIDEO_DIR = Path("/media/adam/Data/PhD/Videos/tray-balance/robust/")
VIDEO_PATH = VIDEO_DIR / ("robust_stack3_" + TIMESTAMP)
VIDEO_PERIOD = 40  # 25 frames per second with 1000 steps per second
RECORD_VIDEO = False

# robust bounding spheres
NUM_BOUNDING_SPHERES = 1

# goal 1
# POSITION_GOAL = np.array([2, 0, -0.5])
# ORIENTATION_GOAL = np.array([0, 0, 0, 1])

# goal 2
# POSITION_GOAL = np.array([0, 2, 0.5])
# ORIENTATION_GOAL = np.array([0, 0, 0, 1])

# goal 3
POSITION_GOAL = np.array([0, -2, 0])
ORIENTATION_GOAL = np.array([0, 0, 1, 0])


def main():
    np.set_printoptions(precision=3, suppress=True)
    N = int(DURATION / SIM_DT)

    # simulation, objects, and model
    sim = MobileManipulatorSimulation(dt=SIM_DT)
    robot, objects, composites = sim.setup(
        # ["tray", "cuboid1", "stacked_cylinder1", "stacked_cylinder2"]
        ["tray", "flat_cylinder1", "flat_cylinder2", "flat_cylinder3"]
        # ["tray", "cuboid1"]
        # ["tray"]
    )

    # initial time, state, input
    t = 0.0
    q, v = robot.joint_states()
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    settings_wrapper = ocs2_util.TaskSettingsWrapper(composites, x)

    if settings_wrapper.settings.tray_balance_settings.robust:
        robustness.set_bounding_spheres(
            robot,
            objects,
            settings_wrapper.settings,
            target=objects["tray"].bullet.get_pose()[0],
            sim_timestep=SIM_DT,
            plot_point_cloud=True,
            k=NUM_BOUNDING_SPHERES,
        )
        robust_spheres = robustness.RobustSpheres(
            robot, settings_wrapper.settings.tray_balance_settings.robust_params
        )
        IPython.embed()

    # set process noise after initial routine to get robust spheres
    robot.v_cmd_stdev = V_CMD_STDEV

    # data recorder and plotter
    recorder = Recorder(
        sim.dt,
        DURATION,
        RECORD_PERIOD,
        ns=robot.ns,
        ni=robot.ni,
        n_objects=len(objects),
        control_period=CTRL_PERIOD,
        n_balance_con=settings_wrapper.get_num_balance_constraints(),
        n_collision_pair=settings_wrapper.get_num_collision_avoidance_constraints(),
        n_dynamic_obs=settings_wrapper.get_num_dynamic_obstacle_constraints(),
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    if RECORD_VIDEO:
        video = VideoRecorder(
            path=VIDEO_PATH,
            distance=4,
            roll=0,
            pitch=-35.8,
            yaw=42,
            target_position=[1.28, 0.045, 0.647],
        )

    r_ew_w, Q_we = robot.link_pose()

    if SIM_TYPE == SimType.POSE_TO_POSE:
        target_times = [0]
        target_inputs = [u]

        # goal pose
        r_ew_w_d = r_ew_w + POSITION_GOAL
        Qd = util.quat_multiply(Q_we, ORIENTATION_GOAL)
        r_obs0 = np.array(r_ew_w) + [0, -10, 0]

        target_states = [np.concatenate((r_ew_w_d, Qd, r_obs0))]

        # visual indicator for target
        # NOTE: debug frame doesn't show up in the recording
        pyb_util.debug_frame_world(0.2, list(r_ew_w_d), orientation=Qd, line_width=3)
        pyb_util.GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1))

    elif SIM_TYPE == SimType.DYNAMIC_OBSTACLE:
        target_times = [0, 5]  # TODO
        target_inputs = [u, u]

        # create the dynamic obstacle
        r_obs0 = np.array(r_ew_w) + [0, -1, 0]
        v_obs = np.array([0, 1.0, 0])
        obstacle = DynamicObstacle(r_obs0, radius=0.1, velocity=v_obs)

        # stationary pose
        r_ew_w_d = r_ew_w
        Qd = Q_we

        # start with the obstacle out of the way
        r_obs_out_of_the_way = r_ew_w + [0, -10, 0]

        target_states = [
            np.concatenate((r_ew_w_d, Qd, r_obs_out_of_the_way)),
            np.concatenate((r_ew_w_d, Qd, r_obs_out_of_the_way)),
        ]

        # trajectories are modified to introduce dynamic obstacles
        target_times_obs1 = [0, 5]
        target_times_obs2 = [2, 7]

        r_obsf = obstacle.sample_position(target_times_obs1[-1])
        target_states_obs = [
            np.concatenate((r_ew_w_d, Qd, r_obs0)),
            np.concatenate((r_ew_w_d, Qd, r_obsf)),
        ]

        # obstacle appears at t = 0
        target_trajectories_obs1 = ocs2_util.make_target_trajectories(
            target_times_obs1, target_states_obs, target_inputs
        )

        # obstacle appears again at t = 2
        target_trajectories_obs2 = ocs2_util.make_target_trajectories(
            target_times_obs2, target_states_obs, target_inputs
        )

    mpc = ocs2_util.setup_ocs2_mpc_interface(
        settings_wrapper.settings, target_times, target_states, target_inputs
    )

    target_idx = 0

    x_opt = np.copy(x)

    # simulation loop
    for i in range(N):
        q, v = robot.joint_states()
        x = np.concatenate((q, v))

        # add noise to state variables
        q_noisy = q + np.random.normal(scale=Q_STDEV, size=q.shape)
        v_noisy = v + np.random.normal(scale=V_STDEV, size=v.shape)
        x_noisy = np.concatenate((q_noisy, v_noisy))

        mpc.setObservation(t, x_opt, u)

        # by using x_opt, we're basically just doing pure open-loop planning,
        # since the state never deviates from the optimal trajectory (at least
        # due to noise)
        # this does have the benefit of smoothing out the state used for
        # computation, which is important for constraint handling
        # mpc.setObservation(t, x_opt, u)

        # TODO this should be set to reflect the MPC time step
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
        # x_opt_new = np.zeros(robot.ns)
        # u = np.zeros(robot.ni)
        mpc.evaluateMpcSolution(t, x_noisy, x_opt, u)
        robot.command_acceleration(u)

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()

            if settings_wrapper.settings.tray_balance_settings.enabled:
                if (
                    settings_wrapper.settings.tray_balance_settings.constraint_type
                    == ocs2.ConstraintType.Hard
                ):
                    recorder.ineq_cons[idx, :] = mpc.stateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                else:
                    con = mpc.softStateInputInequalityConstraint("trayBalance", t, x, u)
                    recorder.ineq_cons[idx, :] = con
            if settings_wrapper.settings.dynamic_obstacle_settings.enabled:
                recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
                    "dynamicObstacleAvoidance", t, x
                )
            if settings_wrapper.settings.collision_avoidance_settings.enabled:
                recorder.collision_pair_distance[
                    idx, :
                ] = mpc.stateInequalityConstraint("collisionAvoidance", t, x)

            r_ew_w_d = target_states[target_idx][:3]
            Q_we_d = target_states[target_idx][3:7]

            # record
            recorder.us[idx, :] = u
            recorder.xs[idx, :] = x_noisy
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
        if settings_wrapper.settings.tray_balance_settings.robust:
            robust_spheres.update()
        t += sim.dt
        time.sleep(sim.dt)

        # set the target trajectories to make controller aware of dynamic
        # obstacles
        if SIM_TYPE == SimType.DYNAMIC_OBSTACLE:
            if i == 0:
                mpc.setTargetTrajectories(target_trajectories_obs1)
            elif i == 2000:
                # reset the obstacle to use again
                obstacle.reset_pose(r_obs0, (0, 0, 0, 1))
                obstacle.reset_velocity(obstacle.velocity)
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

    last_sim_index = i
    recorder.plot_ee_position(last_sim_index)
    recorder.plot_ee_orientation(last_sim_index)
    recorder.plot_ee_velocity(last_sim_index)
    for j in range(len(objects)):
        recorder.plot_object_error(last_sim_index, j)
    recorder.plot_balancing_constraints(last_sim_index)
    recorder.plot_commands(last_sim_index)
    recorder.plot_control_durations(last_sim_index)
    recorder.plot_cmd_vs_real_vel(last_sim_index)
    recorder.plot_joint_config(last_sim_index)

    if recorder.dynamic_obs_distance.shape[1] > 0:
        print(f"Min dynamic obstacle distance = {np.min(recorder.dynamic_obs_distance, axis=0)}")
        recorder.plot_dynamic_obs_dist(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
