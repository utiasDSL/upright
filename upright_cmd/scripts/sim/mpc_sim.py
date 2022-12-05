#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet."""
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

    # settle sim to make sure everything is touching comfortably
    sim.settle(5.0)
    sim.launch_dynamic_obstacles()
    sim.fixture_objects()

    # initial time, state, input
    t = 0.0
    q, v = sim.robot.joint_states()
    a = np.zeros(sim.robot.nv)
    x_obs = sim.dynamic_obstacle_state()
    x = np.concatenate((q, v, a, x_obs))
    u = np.zeros(sim.robot.nu)

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

    # ctrl_manager.warmstart()

    print("Ready to start.")
    IPython.embed()

    v_ff = v.copy()
    a_ff = a.copy()

    use_direct_velocity_command = False
    use_velocity_feedback = True

    num_obs_resets = 0

    # simulation loop
    while t <= sim.duration:
        # get the true robot feedback
        q, v = sim.robot.joint_states(add_noise=False)
        x_obs = sim.dynamic_obstacle_state()
        x = np.concatenate((q, v, a_ff, x_obs))

        # now get the noisy version for use in the controller
        # we can choose to use v_ff rather than v_noisy if we can to avoid
        # noisy velocity feedback
        q_noisy, v_noisy = sim.robot.joint_states(add_noise=True)
        if use_velocity_feedback:
            x_noisy = np.concatenate((q_noisy, v_noisy, a_ff, x_obs))
        else:
            x_noisy = np.concatenate((q_noisy, v_ff, a_ff, x_obs))

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

        # IPython.embed()
        # return

        # TODO why is this better than using the zero-order hold?
        # here we use the input u to generate the feedforward signal---using
        # the jerk level ensures smoothness at the velocity level
        qd, vd, ad = mapping.xu2qva(xd_robot)

        # u_cmd = Kp @ (qd - q_noisy) + (vd - v_ff) + (ad - a_ff) + u_robot
        # v_ff, a_ff = integrator.integrate_approx(v_ff, ad, u_robot, sim.timestep)
        v_ff, a_ff = integrator.integrate_approx(v_ff, a_ff, u_robot, sim.timestep)
        v_cmd = v_ff

        sim.robot.command_velocity(v_cmd, bodyframe=False)

        # TODO more logger reforms to come
        if logger.ready(t):
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

        # if num_obs_resets == 0 and t >= 1.0:
        #     num_obs_resets += 1
        #     ctrl_manager.mpc.reset(ctrl_manager.ref)

        # if len(sim.dynamic_obstacles) > 0 and t >= (num_obs_resets + 1) * 2.0:
        #     num_obs_resets += 1
        #     obs = sim.dynamic_obstacles[0]
        #     # obs.reset(t, r=r_ew_w + [0, -2, 0])
        #
        #     Δt = 0.75
        #     r_target = r_ew_w + [0.1, 0.13, 0]
        #     r_obs0 = r_target + [0, -2, 0]
        #     v_obs0 = (r_target - r_obs0 - 0.5 * Δt ** 2 * gravity) / Δt
        #     obs.reset(t, r=r_obs0, v=v_obs0)
        #
        #     # TODO: Ideally, we could remain stable despite large resets
        #     # ctrl_manager.mpc.reset(ctrl_manager.ref)

        t = sim.step(t, step_robot=False)

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
