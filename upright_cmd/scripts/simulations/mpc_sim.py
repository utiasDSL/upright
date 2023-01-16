#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet."""
import datetime

import numpy as np
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world

from upright_core.logging import DataLogger, DataPlotter
import upright_sim as sim
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
    env = sim.simulation.BulletSimulation(
        config=sim_config, timestamp=timestamp, video_name=cli_args.video, extra_gui=True
    )

    # settle sim to make sure everything is touching comfortably
    env.settle(5.0)
    env.launch_dynamic_obstacles()
    env.fixture_objects()

    for name in ["wedge_init", "wedge_init_side"]:
        if name in env.cameras:
            env.cameras[name].save_frame(f"{name}.png")

    # initial time, state, input
    t = 0.0
    q, v = env.robot.joint_states()
    a = np.zeros(env.robot.nv)
    x_obs = env.dynamic_obstacle_state()
    x = np.concatenate((q, v, a, x_obs))
    u = np.zeros(env.robot.nu)

    # controller
    integrator = ctrl.trajectory.DoubleIntegrator(v.shape[0])

    ctrl_manager = ctrl.manager.ControllerManager.from_config(ctrl_config, x0=x)
    model = ctrl_manager.model
    dims = model.settings.dims
    ref = ctrl_manager.ref
    mapping = ctrl.trajectory.StateInputMapping(model.settings.dims.robot)
    gravity = model.settings.gravity

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", env.timestep)
    logger.add("duration", env.duration)
    logger.add("ctrl_timestep", ctrl_manager.timestep)
    logger.add("object_names", [str(name) for name in env.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    # frames for desired waypoints
    if sim_config.get("show_debug_frames", False):
        for r_ew_w_d, Q_we_d in ref.poses():
            debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    # ctrl_manager.warmstart()

    print("Ready to start.")
    IPython.embed()

    v_ff = v.copy()
    a_ff = a.copy()

    # simulation loop
    while t <= env.duration:
        # get the true robot feedback
        q, v = env.robot.joint_states(add_noise=False)
        x_obs = env.dynamic_obstacle_state()
        x = np.concatenate((q, v, a_ff, x_obs))

        # now get the noisy version for use in the controller
        # we can choose to use v_ff rather than v_noisy if we can to avoid
        # noisy velocity feedback
        q_noisy, v_noisy = env.robot.joint_states(add_noise=True)
        x_noisy = np.concatenate((q_noisy, v_noisy, a_ff, x_obs))

        # compute policy - MPC is re-optimized automatically when the internal
        # MPC timestep has been exceeded
        try:
            xd, u = ctrl_manager.step(t, x_noisy)
            xd_robot = x[: dims.robot.x]
            u_robot = u[: dims.robot.u]
            f = u[-dims.f() :]
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
        qd, vd, ad = mapping.xu2qva(xd_robot)
        # v_ff, a_ff = integrator.integrate_approx(v_ff, ad, u_robot, env.timestep)
        v_ff, a_ff = integrator.integrate_approx(v_ff, a_ff, u_robot, env.timestep)
        v_cmd = v_ff

        # generated velocity is in the world frame
        env.robot.command_velocity(v_cmd, bodyframe=False)

        # TODO more logger reforms to come
        if logger.ready(t):
            # log sim stuff
            r_ew_w, Q_we = env.robot.link_pose()
            v_ew_w, ω_ew_w = env.robot.link_velocity()
            r_ow_ws, Q_wos = env.object_poses()
            logger.append("ts", t)
            logger.append("us", u)
            logger.append("xs", x)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", v_cmd)
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            r_ew_w_d, Q_we_d = ctrl_manager.ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

            model.update(x, u)
            logger.append("ddC_we_norm", model.ddC_we_norm())
            logger.append("sa_dists", model.support_area_distances())
            logger.append("orn_err", model.angle_between_acc_and_normal())
            logger.append("balancing_constraints", model.balancing_constraints())

            if model.settings.inertial_alignment_settings.constraint_enabled:
                alignment_constraints = (
                    ctrl_manager.mpc.getStateInputInequalityConstraintValue(
                        "inertial_alignment_constraint", t, x, u
                    )
                )
                logger.append("alignment_constraints", alignment_constraints)

            if model.settings.inertial_alignment_settings.cost_enabled:
                alignment_constraints = ctrl_manager.mpc.getCostValue(
                    "inertial_alignment_cost", t, x, u
                )
                logger.append("alignment_cost", alignment_constraints)

            if model.settings.obstacle_settings.enabled:
                if (
                    model.settings.obstacle_settings.constraint_type
                    == ctrl.bindings.ConstraintType.Soft
                ):
                    obs_constraints = (
                        ctrl_manager.mpc.getSoftStateInequalityConstraintValue(
                            "obstacle_avoidance", t, x
                        )
                    )
                else:
                    obs_constraints = (
                        ctrl_manager.mpc.getStateInputInequalityConstraintValue(
                            "obstacle_avoidance", t, x, u
                        )
                    )
                logger.append("collision_pair_distances", obs_constraints)

            # TODO eventually it would be nice to also compute this directly
            # via the core library
            if model.is_using_force_constraints():
                # if (
                #     model.settings.balancing_settings.constraint_type
                #     == ctrl.bindings.ConstraintType.Soft
                # ):
                #     contact_force_constraints = (
                #         ctrl_manager.mpc.getSoftStateInputInequalityConstraintValue(
                #             "contact_forces", t, x, u
                #         )
                #     )
                # else:
                #     contact_force_constraints = (
                #         ctrl_manager.mpc.getStateInputInequalityConstraintValue(
                #             "contact_forces", t, x, u
                #         )
                #     )
                object_dynamics_constraints = (
                    ctrl_manager.mpc.getStateInputEqualityConstraintValue(
                        "object_dynamics", t, x, u
                    )
                )
                logger.append("cost", ctrl_manager.mpc.cost(t, x, u))

                # logger.append("contact_force_constraints", contact_force_constraints)
                logger.append("contact_forces", f)
                logger.append(
                    "object_dynamics_constraints", object_dynamics_constraints
                )

        t = env.step(t, step_robot=False)[0]

    if "wedge_final_side" in env.cameras:
        env.cameras["wedge_final_side"].save_frame("wedge_final_side.png")

    try:
        print(f"Min constraint value = {np.min(logger.data['balancing_constraints'])}")
    except:
        pass

    logger.add("replanning_times", ctrl_manager.replanning_times)
    logger.add("replanning_durations", ctrl_manager.replanning_durations)

    # save logged data
    if cli_args.log is not None:
        logger.save(timestamp, name=cli_args.log)

    if env.video_manager.save:
        print(f"Saved video to {env.video_manager.path}")

    # visualize data
    DataPlotter.from_logger(logger).plot_all(show=True)


if __name__ == "__main__":
    main()