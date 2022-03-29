#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import time
import datetime
import sys
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pybullet as pyb
import rospkg
import yaml
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from tray_balance_sim import util, camera, simulation
from tray_balance_sim.recording import Recorder

import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    # load configuration
    rospack = rospkg.RosPack()
    config_path = Path(rospack.get_path("tray_balance_assets")) / "config"
    with open(config_path / "controller.yaml") as f:
        ctrl_config = yaml.safe_load(f)
    with open(config_path / "simulation.yaml") as f:
        sim_config = yaml.safe_load(f)

    # timing
    duration_millis = sim_config["duration"]
    timestep_millis = sim_config["timestep"]
    timestep_secs = 0.001 * timestep_millis
    duration_secs = 0.001 * duration_millis
    num_timesteps = int(duration_millis / timestep_millis)
    ctrl_period = ctrl_config["control_period"]

    # start the simulation
    sim = simulation.MobileManipulatorSimulation(sim_config)
    robot = sim.robot

    # setup objects for sim and controller
    r_ew_w, Q_we = robot.link_pose()
    sim_objects = simulation.sim_object_setup(r_ew_w, sim_config)

    # TODO this can be moved into the ctrl.parsing module
    ctrl_objects = core.parsing.parse_control_objects(ctrl_config)

    # initial time, state, input
    t = 0.0
    q, v = robot.joint_states()
    x = np.concatenate((q, v))
    u = np.zeros(robot.ni)

    # video recording
    now = datetime.datetime.now()
    video_manager = camera.VideoManager.from_config_dict(
        config=sim_config, timestamp=now, r_ew_w=r_ew_w
    )

    # TODO want a function to populate this from the config dict
    # better yet, we may not even need the object
    # TODO this should be the ControlConfigWrapper
    settings_wrapper = ctrl.parsing.ControllerSettingsWrapper(ctrl_config)
    settings_wrapper.settings.tray_balance_settings.enabled = True
    settings_wrapper.settings.tray_balance_settings.bounded = True
    settings_wrapper.settings.initial_state = x

    ghosts = []  # ghost (i.e., pure visual) objects
    settings_wrapper.settings.tray_balance_settings.bounded_config.objects = (
        ctrl_objects
    )

    # data recorder and plotter
    log_dir = Path(ctrl_config["logging"]["log_dir"])
    recorder = Recorder(
        timestep_secs,
        duration_secs,
        ctrl_config["logging"]["timestep"],
        ns=robot.ns,
        ni=robot.ni,
        n_objects=len(sim_objects),
        control_period=ctrl_period,
        n_balance_con=settings_wrapper.get_num_balance_constraints(),
        n_collision_pair=settings_wrapper.get_num_collision_avoidance_constraints(),
        n_dynamic_obs=settings_wrapper.get_num_dynamic_obstacle_constraints() + 1,
    )
    recorder.cmd_vels = np.zeros((recorder.ts.shape[0], robot.ni))

    target_times = [0]
    target_inputs = [u]
    target_idx = 0

    # goal pose
    r_ew_w_d = r_ew_w + ctrl_config["goal"]["position"]
    Qd = util.quat_multiply(Q_we, ctrl_config["goal"]["orientation"])
    r_obs0 = np.array(r_ew_w) + [0, -10, 0]

    target_states = [np.concatenate((r_ew_w_d, Qd, r_obs0))]

    mpc = ctrl.parsing.setup_ctrl_interface(
        settings_wrapper.settings, target_times, target_states, target_inputs
    )

    # visual indicator for target
    # NOTE: debug frame doesn't show up in the recording
    debug_frame_world(0.2, list(r_ew_w_d), orientation=Qd, line_width=3)
    ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))

    x_opt = np.copy(x)

    # simulation loop
    for i in range(num_timesteps):
        q, v = robot.joint_states()
        x = np.concatenate((q, v))

        # add noise to state variables
        q_noisy, v_noisy = robot.joint_states(add_noise=True)
        x_noisy = np.concatenate((q_noisy, v_noisy))

        # by using x_opt, we're basically just doing pure open-loop planning,
        # since the state never deviates from the optimal trajectory (at least
        # due to noise)
        # this does have the benefit of smoothing out the state used for
        # computation, which is important for constraint handling
        if ctrl_config["use_noisy_state_to_plan"]:
            mpc.setObservation(t, x_noisy, u)
        else:
            mpc.setObservation(t, x_opt, u)

        # this should be set to reflect the MPC time step
        # we can increase it if the MPC rate is faster
        if i % ctrl_period == 0:
            try:
                t0 = time.time()
                mpc.advanceMpc()
                t1 = time.time()
            except RuntimeError as e:
                print(e)
                print("exit the interpreter to proceed to plots")
                IPython.embed()
                i -= 1  # for the recorder
                break
            recorder.control_durations[i // ctrl_period] = t1 - t0

        # As far as I can tell, evaluateMpcSolution actually computes the input
        # for the particular time and state (the input is often at least
        # state-varying in DDP, with linear feedback on state error). OTOH,
        # getMpcSolution just gives the current MPC policy trajectory over the
        # entire time horizon, without accounting for the given state. So it is
        # like doing feedforward input only, which is bad.
        mpc.evaluateMpcSolution(t, x_noisy, x_opt, u)
        robot.command_acceleration(u)

        if recorder.now_is_the_time(i):
            idx = recorder.record_index(i)

            r_ew_w, Q_we = robot.link_pose()
            v_ew_w, ω_ew_w = robot.link_velocity()

            if settings_wrapper.settings.tray_balance_settings.enabled:
                if (
                    settings_wrapper.settings.tray_balance_settings.constraint_type
                    == ctrl.bindings.ConstraintType.Hard
                ):
                    recorder.ineq_cons[idx, :] = mpc.stateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
                else:
                    recorder.ineq_cons[idx, :] = mpc.softStateInputInequalityConstraint(
                        "trayBalance", t, x, u
                    )
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
            recorder.xs[idx, :] = x
            recorder.xs_noisy[idx, :] = x_noisy
            recorder.r_ew_wds[idx, :] = r_ew_w_d
            recorder.r_ew_ws[idx, :] = r_ew_w
            recorder.Q_wes[idx, :] = Q_we
            recorder.Q_weds[idx, :] = Q_we_d
            recorder.v_ew_ws[idx, :] = v_ew_w
            recorder.ω_ew_ws[idx, :] = ω_ew_w

            for j, obj in enumerate(sim_objects.values()):
                r, Q = obj.get_pose()
                recorder.r_ow_ws[j, idx, :] = r
                recorder.Q_wos[j, idx, :] = Q

            recorder.cmd_vels[idx, :] = robot.cmd_vel

        sim.step(step_robot=True)
        if settings_wrapper.settings.dynamic_obstacle_settings.enabled:
            obstacle.step()
        for ghost in ghosts:
            ghost.update()
        t += timestep_secs

        # if we have multiple targets, step through them
        if t >= target_times[target_idx] and target_idx < len(target_times) - 1:
            target_idx += 1

        video_manager.record(i)

    if recorder.ineq_cons.shape[1] > 0:
        print(f"Min constraint value = {np.min(recorder.ineq_cons)}")

    # save logged data
    if len(sys.argv) > 1 and sys.argv[1] == "--save":
        if len(sys.argv) > 2:
            prefix = sys.argv[2]
        else:
            prefix = "data"
        data_file_name = prefix + "_" + now.strftime("%Y-%m-%d_%H-%M-%S")
        recorder.save(log_dir / data_file_name)

    last_sim_index = i
    recorder.plot_ee_position(last_sim_index)
    recorder.plot_ee_orientation(last_sim_index)
    recorder.plot_ee_velocity(last_sim_index)
    for j in range(len(sim_objects)):
        recorder.plot_object_error(last_sim_index, j)
    recorder.plot_balancing_constraints(last_sim_index)
    recorder.plot_commands(last_sim_index)
    recorder.plot_control_durations(last_sim_index)
    recorder.plot_cmd_vs_real_vel(last_sim_index)
    recorder.plot_joint_config(last_sim_index)

    if settings_wrapper.settings.dynamic_obstacle_settings.enabled:
        print(
            f"Min dynamic obstacle distance = {np.min(recorder.dynamic_obs_distance, axis=0)}"
        )
        recorder.plot_dynamic_obs_dist(last_sim_index)

    plt.show()


if __name__ == "__main__":
    main()
