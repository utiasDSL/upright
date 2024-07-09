#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet."""
import copy
import datetime

import numpy as np
import pybullet as pyb
from pyb_utils.frame import debug_frame_world
import matplotlib.pyplot as plt
import rigeo as rg
import cvxpy as cp

from upright_core.logging import DataLogger, DataPlotter
import upright_sim as sim
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd

import IPython


LOG = True
PLOT = False


def run_simulation(config, video, logname):
    sim_config = config["simulation"]
    ctrl_config = config["controller"]

    # start the simulation
    print("starting sim")
    timestamp = datetime.datetime.now()
    env = sim.simulation.UprightSimulation(
        config=sim_config,
        timestamp=timestamp,
        video_name=video,
        extra_gui=sim_config.get("extra_gui", False),
    )

    # settle sim to make sure everything is touching comfortably
    env.settle(5.0)
    env.fixture_objects()

    # initial time, state, input
    t = 0.0
    q, v = env.robot.joint_states()
    a = np.zeros(env.robot.nv)
    x_obs = env.dynamic_obstacle_state()
    x = np.concatenate((q, v, a, x_obs))
    u = np.zeros(env.robot.nu)

    print("starting controller")

    # controller
    ctrl_manager = ctrl.manager.ControllerManager.from_config(ctrl_config, x0=x)
    model = ctrl_manager.model
    dims = model.settings.dims
    ref = ctrl_manager.ref

    # r_pyb, Q_pyb = env.robot.link_pose()
    # r_pin, Q_pin = model.robot.link_pose()
    # IPython.embed()
    # return

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

    # log variable quantities in the sim loop
    logger.add("waypoint", ctrl_config["waypoints"][0]["position"])
    logger.add("mass", sim_config["objects"]["tall_block_0"]["mass"])
    logger.add("com_offset", sim_config["objects"]["tall_block_0"]["com_offset"])
    logger.add("inertia", sim_config["objects"]["tall_block_0"]["inertia"])

    # frames for desired waypoints
    if sim_config.get("show_debug_frames", False):
        for r_ew_w_d, Q_we_d in ref.poses():
            debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    v_cmd = np.zeros_like(v)
    a_est = np.zeros_like(a)

    T = model.settings.tracking
    Kx = np.hstack(
        (
            T.kp * np.eye(dims.robot.q),
            T.kv * np.eye(dims.robot.v),
            T.ka * np.eye(dims.robot.v),
        )
    )

    print("Ready to start.")
    # IPython.embed()

    # simulation loop
    while t <= env.duration:
        # get the true robot feedback
        # instead of a proper estimator (e.g. Kalman filter) we're being lazy
        # assuming the model tracks well enough that real acceleration is
        # roughly perfect
        q, v = env.robot.joint_states(add_noise=False)
        x_obs = env.dynamic_obstacle_state()
        x = np.concatenate((q, v, a_est, x_obs))

        # now get the noisy version for use in the controller
        q_noisy, v_noisy = env.robot.joint_states(add_noise=True)
        x_noisy = np.concatenate((q_noisy, v_noisy, a_est, x_obs))

        # compute policy - MPC is re-optimized automatically when the internal
        # MPC timestep has been exceeded
        try:
            xd, u = ctrl_manager.step(t, x_noisy)
            xd_robot = xd[: dims.robot.x]
            u_robot = u[: dims.robot.u]
            f = u[-dims.f() :]

            # check out the gain matrix if desired
            # K = ctrl_manager.mpc.getLinearFeedbackGain(t)
        except RuntimeError as e:
            print(e)
            print("Exit the interpreter to proceed to plots.")
            IPython.embed()
            break

        if np.isnan(u).any():
            print("NaN value in input!")
            IPython.embed()
            break

        u_cmd = Kx @ (xd - x)[: dims.robot.x] + u_robot

        # integrate the command
        # it appears to be desirable to open-loop integrate velocity like this
        # to avoid PyBullet not handling velocity commands accurately at very
        # small values
        v_cmd = v_cmd + env.timestep * a_est + 0.5 * env.timestep**2 * u_cmd
        a_est = a_est + env.timestep * u_cmd

        # generated velocity is in the world frame
        env.robot.command_velocity(v_cmd, bodyframe=False)

        # TODO more logger reforms to come
        if logger.ready(t):
            # log sim stuff
            r_ew_w, Q_we = env.robot.link_pose()
            v_ew_w, ω_ew_w = env.robot.link_velocity()
            r_ow_ws, Q_wos = env.object_poses()
            logger.append("ts", t)
            logger.append("us", u_cmd)
            logger.append("xs", x)
            logger.append("xds", xd)
            logger.append("uds", u)
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
                object_dynamics_constraints = (
                    ctrl_manager.mpc.getStateInputEqualityConstraintValue(
                        "object_dynamics", t, x, u
                    )
                )
                logger.append("cost", ctrl_manager.mpc.cost(t, x, u))

                # if not frictionless, get the constraint values
                # if we are frictionless, then the forces just all need to be
                # non-negative
                if dims.nf == 3:
                    contact_force_constraints = (
                        ctrl_manager.mpc.getStateInputInequalityConstraintValue(
                            "contact_forces", t, x, u
                        )
                    )
                    logger.append(
                        "contact_force_constraints", contact_force_constraints
                    )

                logger.append("contact_forces", f)
                logger.append(
                    "object_dynamics_constraints", object_dynamics_constraints
                )

        t = env.step(t, step_robot=False)[0]

    try:
        print(f"Min constraint value = {np.min(logger.data['balancing_constraints'])}")
    except:
        pass

    logger.add("replanning_times", ctrl_manager.replanning_times)
    logger.add("replanning_durations", ctrl_manager.replanning_durations)

    # save logged data
    if logname is not None:
        logger.save(timestamp, name=logname)

    if env.video_manager.save:
        print(f"Saved video to {env.video_manager.path}")

    pyb.disconnect(env.client_id)

    # visualize data
    if PLOT:
        DataPlotter.from_logger(logger).plot_all(show=False)

        # plotting for desired trajectories
        prop_cycle = plt.rcParams["axes.prop_cycle"]
        colors = prop_cycle.by_key()["color"]

        ts = np.array(logger.data["ts"])
        xds = np.array(logger.data["xds"])
        xs = np.array(logger.data["xs"])
        uds = np.array(logger.data["uds"])
        us = np.array(logger.data["us"])

        plt.figure()
        for i in range(env.robot.nq):
            plt.plot(ts, xds[:, i], label=f"qd_{i}", linestyle="--")
        for i in range(env.robot.nq):
            plt.plot(ts, xs[:, i], label=f"q_{i}", color=colors[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Joint position")
        plt.title("Desired vs. Actual Joint Positions")
        plt.legend()
        plt.grid()

        plt.figure()
        for i in range(env.robot.nv):
            plt.plot(ts, xds[:, env.robot.nq + i], label=f"vd_{i}", linestyle="--")
        for i in range(env.robot.nv):
            plt.plot(ts, xs[:, env.robot.nq + i], label=f"v_{i}", color=colors[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Joint velocity")
        plt.title("Desired vs. Actual Joint Velocities")
        plt.legend()
        plt.grid()

        plt.figure()
        for i in range(env.robot.nv):
            plt.plot(
                ts,
                xds[:, env.robot.nq + env.robot.nv + i],
                label=f"ad_{i}",
                linestyle="--",
            )
        for i in range(env.robot.nv):
            plt.plot(
                ts,
                xs[:, env.robot.nq + env.robot.nv + i],
                label=f"a_{i}",
                color=colors[i],
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Joint acceleration")
        plt.title("Desired vs. Actual Joint Acceleration")
        plt.legend()
        plt.grid()

        plt.figure()
        for i in range(env.robot.nv):
            plt.plot(ts, uds[:, i], label=f"ud_{i}", linestyle="--")
        for i in range(env.robot.nv):
            plt.plot(ts, us[:, i], label=f"u_{i}", color=colors[i])
        plt.xlabel("Time [s]")
        plt.ylabel("Joint jerk")
        plt.title("Desired vs. Actual Joint Jerk Input")
        plt.legend()
        plt.grid()

        plt.show()


def box_face_centers(box):
    x, y, z = box.half_extents
    return [[x, 0, 0], [-x, 0, 0], [0, y, 0], [0, -y, 0], [0, 0, z], [0, 0, -z]]


def max_min_eig_inertia(box, com):
    """Find the inertia matrix with the maximum smallest eigenvalue."""
    μ = cp.Variable(8)
    I = cp.Variable((3, 3))
    H = cp.Variable((3, 3))
    λ = cp.Variable(1)
    objective = cp.Maximize(λ)
    constraints = [
        λ * np.eye(3) << I,
        I == cp.trace(H) * np.eye(3) - H,
        H == cp.sum([m * np.outer(v, v) for m, v in zip(μ, box.vertices)]),
        com == cp.sum([m * v for m, v in zip(μ, box.vertices)]),
        1 == cp.sum(μ),
        μ >= 0,
    ]
    problem = cp.Problem(objective, constraints)
    problem.solve(cp.MOSEK)
    assert problem.status == "optimal"
    return I.value


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    master_config = core.parsing.load_config(cli_args.config)

    # waypoints
    z = np.array([0, 0, 0.4])
    w1 = [[0, -2, 0], [1.5, -1.5, 0], [2, 0, 0], [1.5, 1.5, 0], [0, 2, 0]]
    w2 = [list(w + z) for w in w1]
    w3 = [list(w - z) for w in w1]
    waypoints = w1 + w2 + w3

    # CoMs
    obj_config = master_config["simulation"]["objects"]["tall_block_0"]
    box = rg.Box.from_side_lengths(obj_config["side_lengths"])
    com_box = rg.Box(half_extents=0.6 * box.half_extents)
    com_offsets = [[0, 0, 0]] + box_face_centers(com_box) + [list(v) for v in com_box.vertices]

    # inertias
    inertias = [max_min_eig_inertia(box, c) for c in com_offsets]
    inertia_scales = [1.0, 0.5, 0.1]

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    dirname = f"robust_sims_{timestamp}"

    run = 1
    for com_offset, inertia in zip(com_offsets, inertias):
        for s in inertia_scales:
            for i, waypoint in enumerate(waypoints):
                print(f"run = {run}")
                config = copy.deepcopy(master_config)
                config["controller"]["waypoints"] = [
                    {"time": 0, "position": waypoint, "orientation": [0, 0, 0, 1]}
                ]

                config["simulation"]["objects"]["tall_block_0"]["mass"] = 1.0
                config["simulation"]["objects"]["tall_block_0"]["com_offset"] = com_offset
                config["simulation"]["objects"]["tall_block_0"]["inertia"] = s * inertia

                IPython.embed()
                return

                # no need to recompile each time a waypoint is changed
                # (but we do need to recompile for changes to the inertial
                # parameters, at the moment)
                if i > 0:
                    config["controller"]["recompile_libraries"] = False

                if LOG:
                    name = f"{dirname}/run_{run}"
                else:
                    name = None

                run_simulation(config=config, video=None, logname=name)
                run += 1


if __name__ == "__main__":
    main()
