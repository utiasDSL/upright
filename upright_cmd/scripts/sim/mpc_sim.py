#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import datetime

import numpy as np
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from upright_sim import simulation
from upright_core.logging import DataLogger, DataPlotter
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # start the simulation
    timestamp = datetime.datetime.now()
    sim = simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, cli_args=cli_args
    )

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x = np.concatenate((q, v, a))
    u = np.zeros(sim.robot.nu)

    # controller
    integrator = ctrl.trajectory.DoubleIntegrator(v.shape[0])

    ctrl_manager = ctrl.manager.ControllerManager.from_config(ctrl_config, x0=x)
    model = ctrl_manager.model
    ref = ctrl_manager.ref
    Kp = model.settings.Kp
    mapping = ctrl.trajectory.StateInputMapping(model.settings.dims)

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", sim.timestep)
    logger.add("duration", sim.duration)
    logger.add("ctrl_timestep", ctrl_manager.timestep)
    logger.add("object_names", [str(name) for name in sim.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ref.poses():
        # sim.ghosts.append(GhostSphere(radius=0.05, position=r_ew_w_d, color=(0, 1, 0, 1)))
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    ctrl_manager.warmstart()

    v_ff = v.copy()
    a_ff = a.copy()

    use_direct_velocity_command = False
    use_velocity_feedback = False

    # simulation loop
    while t <= sim.duration:
        # get the true robot feedback
        q, v = sim.robot.joint_states(add_noise=False)
        x = np.concatenate((q, v, a_ff))

        # now get the noisy version for use in the controller
        # we can choose to use v_ff rather than v_noisy if we can to avoid
        # noisy velocity feedback
        q_noisy, v_noisy = sim.robot.joint_states(add_noise=True)
        if use_velocity_feedback:
            x_noisy = np.concatenate((q_noisy, v_noisy, a_ff))
        else:
            x_noisy = np.concatenate((q_noisy, v_ff, a_ff))

        # compute policy - MPC is re-optimized automatically when the internal
        # MPC timestep has been exceeded
        try:
            xd, u = ctrl_manager.step(t, x_noisy)
            u_cmd = u[:model.settings.dims.v]
        except RuntimeError as e:
            print(e)
            print("exit the interpreter to proceed to plots")
            IPython.embed()
            break

        if np.isnan(u).any():
            print("nan value in input!")
            IPython.embed()
            break

        # TODO why is this better than using the zero-order hold?
        # here we use the input u to generate the feedforward signal---using
        # the jerk level ensures smoothness at the velocity level
        qd, vd, _ = mapping.xu2qva(xd)

        if use_direct_velocity_command:
            v_ff, a_ff = integrator.integrate_approx(v_ff, a_ff, u, sim.timestep)
            v_cmd = Kp @ (qd - q_noisy) + vd
        else:
            ud = Kp @ (qd - q_noisy) + u_cmd
            v_ff, a_ff = integrator.integrate_approx(v_ff, a_ff, ud, sim.timestep)
            v_cmd = v_ff

        sim.robot.command_velocity(v_cmd)

        # TODO more logger reforms to come
        if logger.ready(t):
            # if ctrl_manager.settings.dynamic_obstacle_settings.enabled:
            #     recorder.dynamic_obs_distance[idx, :] = mpc.stateInequalityConstraint(
            #         "dynamic_obstacle_avoidance", t, x
            #     )
            # if model.settings.static_obstacle_settings.enabled:
            #     ds = ctrl_manager.mpc.stateInequalityConstraint("static_obstacle_avoidance", t, x)
            #     logger.append("collision_pair_distances", ds)
            # limits = ctrl_manager.mpc.stateInputInequalityConstraint("joint_state_input_limits", t, x, u)
            # logger.append("limits", limits)

            # log sim stuff
            r_ew_w, Q_we = sim.robot.link_pose()
            v_ew_w, ω_ew_w = sim.robot.link_velocity()
            r_ow_ws, Q_wos = sim.object_poses()
            logger.append("ts", t)
            logger.append("us", u)
            logger.append("xs", x)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", sim.robot.cmd_vel.copy())
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            r_ew_w_d, Q_we_d = ctrl_manager.ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            ctrl_manager.model.update(x, u_cmd)
            logger.append("ddC_we_norm", model.ddC_we_norm())
            logger.append("balancing_constraints", model.balancing_constraints()[0])
            logger.append("sa_dists", model.support_area_distances())
            logger.append("orn_err", model.angle_between_acc_and_normal())

        t = sim.step(t, step_robot=False)
        # if ctrl_manager.settings.dynamic_obstacle_settings.enabled:
        #     obstacle.step()

    try:
        print(f"Min constraint value = {np.min(logger.data['balancing_constraints'])}")
    except:
        pass

    logger.add("replanning_times", ctrl_manager.replanning_times)
    logger.add("replanning_durations", ctrl_manager.replanning_durations)

    # save logged data
    if cli_args.log is not None:
        logger.save(timestamp, name=cli_args.log)

    if sim.video_manager.save:
        print(f"Saved video to {sim.video_manager.path}")

    # visualize data
    DataPlotter.from_logger(logger).plot_all(show=True)


if __name__ == "__main__":
    main()
