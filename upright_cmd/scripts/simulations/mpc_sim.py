#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet."""
import datetime

import numpy as np
import pybullet as pyb
from pyb_utils.ghost import GhostSphere
from pyb_utils.frame import debug_frame_world
import matplotlib.pyplot as plt

from upright_core.logging import DataLogger, DataPlotter
import upright_sim as sim
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd

import IPython


def take_photos(sim_config, env, when):
    """Save images at start or end of the simulation."""
    assert when == "start" or when == "end"
    if "photos" in sim_config:
        for photo in sim_config["photos"].get(when, []):
            cam_name = photo["camera"]
            img_name = photo["name"]
            env.cameras[cam_name].save_frame(f"{img_name}.png")
            print(f"Saved photo to {img_name}.png")


def solve_projectile_height(r0, v0, h, g):
    """Solve for (x, y) and time t when projectile reaches a given height (in
    the future)."""
    z0 = r0[2]
    vz = v0[2]

    # solve for intersection time
    t = (-vz - np.sqrt(vz**2 - 2 * (z0 - h) * g)) / g

    # solve for intersection point
    r = r0 + t * v0 + 0.5 * t**2 * np.array([0, 0, g])

    return t, r


def perp2d(a):
    return np.array([-a[1], a[0]])


def angle_between(a, b):
    θ = np.arccos(a @ b)
    c = perp2d(a)
    if b @ c > 0:
        θ = -θ
    return θ


def rot2d(θ):
    c = np.cos(θ)
    s = np.sin(θ)
    return np.array([[c, -s], [s, c]])


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
    env = sim.simulation.UprightSimulation(
        config=sim_config,
        timestamp=timestamp,
        video_name=cli_args.video,
        extra_gui=sim_config.get("extra_gui", False),
    )

    # settle sim to make sure everything is touching comfortably
    env.settle(5.0)
    env.launch_dynamic_obstacles()
    env.fixture_objects()

    take_photos(sim_config, env, when="start")

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

    # 1. solve for (x, y)
    r_ew_w, Q_we = env.robot.link_pose()
    r_obs, v_obs = x_obs[:3], x_obs[3:6]
    t_int, r_int = solve_projectile_height(r_obs, v_obs, h=r_ew_w[2], g=-9.81)

    # normal vector of the plane
    n_obs = np.cross(v_obs, [0, 0, 1])
    n_obs = n_obs / np.linalg.norm(n_obs)

    Δ = r_ew_w - r_int
    Δ = Δ / np.linalg.norm(Δ)

    if n_obs @ Δ < 0:
        n_obs *= -1

    # TODO bit of a hack
    yaw = q[2]
    n_ee = np.array([np.cos(yaw), np.sin(yaw), 0])
    θ = angle_between(n_obs[:2], n_ee[:2])

    if θ > 0:
        θ = min(max(θ, 0.5 * np.pi), 0.75 * np.pi)
    else:
        θ = -min(max(-θ, 0.5 * np.pi), 0.75 * np.pi)

    n_goal = core.math.rotz(θ) @ n_ee

    # now we want to move in direction n_obs as fast as possible, subject to
    # feasibility convenience of the robot. In principle there are a couple
    # ways to do this:
    # 1. waypoint that moves some arbitrary distance in this direction (not fast enough!)
    # 2. obstacle that forces us to move out of this region
    # 3. cost that encourages moving in a particular direction

    L = 1
    δ = r_int - r_ew_w
    δ_perp = δ - (δ @ n_goal) * n_goal
    w = np.linalg.norm(δ_perp)
    d = np.sqrt(L**2 - w**2)
    goal = r_int - δ_perp + d * n_goal

    pyb.addUserDebugPoints([goal], [(1, 0, 0)], pointSize=20)

    # fixed x_obs
    # x_obs = np.concatenate((r_int, np.zeros(6)))

    # ref = ctrl.wrappers.TargetTrajectories(
    #     [0], [np.concatenate((goal, Q_we))], [np.zeros(dims.robot.u)]
    # )
    # ctrl_manager.update(ref)

    print("Ready to start.")
    IPython.embed()

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

    take_photos(sim_config, env, when="end")

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
            ts, xds[:, env.robot.nq + env.robot.nv + i], label=f"ad_{i}", linestyle="--"
        )
    for i in range(env.robot.nv):
        plt.plot(
            ts, xs[:, env.robot.nq + env.robot.nv + i], label=f"a_{i}", color=colors[i]
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


if __name__ == "__main__":
    main()
