#!/usr/bin/env python3
"""Closed-loop upright simulation using Pybullet over ROS."""
import time
import datetime

import rospy
import numpy as np
from pyb_utils.frame import debug_frame_world
from std_msgs.msg import Empty
from ocs2_msgs.msg import mpc_observation
import matplotlib.pyplot as plt

from upright_core.logging import DataLogger, DataPlotter
import upright_sim as sim
import upright_core as core
import upright_control as ctrl
import upright_cmd as cmd

from mobile_manipulation_central import (
    SimulatedUR10ROSInterface,
    SimulatedMobileManipulatorROSInterface,
    SimulatedViconObjectInterface,
)

import IPython


class MPCObservationListener:
    """Listens to and records published MPC observations"""

    def __init__(self, topic_name):
        self.mpc_obs_sub = rospy.Subscriber(topic_name, mpc_observation, self._obs_cb)
        self.ts = []
        self.xs = []
        self.us = []

    def _obs_cb(self, msg):
        self.ts.append(msg.time)
        self.xs.append(msg.state.value)
        self.us.append(msg.input.value)


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args, _ = cmd.cli.sim_arg_parser().parse_known_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    log_config = config["logging"]

    # start the simulation
    timestamp = datetime.datetime.now()
    env = sim.simulation.UprightSimulation(
        config=sim_config, timestamp=timestamp, video_name=cli_args.video
    )

    # settle sim to make sure everything is touching comfortably
    env.settle(5.0)

    # initial time, state, input
    t = 0.0
    q, v = env.robot.joint_states()
    a = np.zeros(env.robot.nv)
    x_obs = env.dynamic_obstacle_state()
    x = np.concatenate((q, v, a, x_obs))
    u = np.zeros(env.robot.nu)

    Q_obs = np.array([0, 0, 0, 1])

    # we simulate a Vicon end point for the projectile if (1) we are using a
    # projectile at all and (2) we are not using the real Vicon system
    use_projectile_interface = len(env.dynamic_obstacles) > 0

    # data logging
    logger = DataLogger(config)

    logger.add("sim_timestep", env.timestep)
    logger.add("duration", env.duration)
    logger.add("object_names", [str(name) for name in env.objects.keys()])

    logger.add("nq", ctrl_config["robot"]["dims"]["q"])
    logger.add("nv", ctrl_config["robot"]["dims"]["v"])
    logger.add("nx", ctrl_config["robot"]["dims"]["x"])
    logger.add("nu", ctrl_config["robot"]["dims"]["u"])

    model = ctrl.manager.ControllerModel.from_config(ctrl_config)

    # reference pose trajectory
    model.update(x=model.settings.initial_state)
    r_ew_w, Q_we = model.robot.link_pose()
    ref = ctrl.wrappers.TargetTrajectories.from_config(ctrl_config, r_ew_w, Q_we, u)

    # frames and ghost (i.e., pure visual) objects
    for r_ew_w_d, Q_we_d in ref.poses():
        debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_d, line_width=3)

    # setup the ROS interface
    rospy.init_node("mpc_sim_ros")
    if model.settings.robot_base_type == ctrl.bindings.RobotBaseType.Fixed:
        ros_interface = SimulatedUR10ROSInterface()
    elif model.settings.robot_base_type == ctrl.bindings.RobotBaseType.Omnidirectional:
        ros_interface = SimulatedMobileManipulatorROSInterface()
    else:
        raise ValueError("Unsupported robot base type.")
    if use_projectile_interface:
        projectile_ros_interface = SimulatedViconObjectInterface("ThingProjectile")
    ros_interface.publish_time(t)

    # publisher for reset command to estimator
    reset_projectile_pub = rospy.Publisher(
        "reset_projectile_estimate", Empty, queue_size=1
    )

    # listen to state estimates
    mpc_obs_listener = MPCObservationListener("/mobile_manipulator_mpc_observation")

    # listen to MPC planned optimal trajectory
    mpc_plan_listener = MPCObservationListener("/mobile_manipulator_mpc_plan")

    # listen to commanded velocity (and acceleration) actually sent to the
    # robot
    cmd_listener = MPCObservationListener("/mobile_manipulator_cmds")

    # wait until a command has been received
    # note that we use real time here since this sim directly controls sim time
    print("Waiting for a command to be received...")
    while not ros_interface.ready():
        ros_interface.publish_feedback(t, q, v)
        ros_interface.publish_time(t)
        t += env.timestep
        time.sleep(env.timestep)
        if rospy.is_shutdown():
            return

    print("Command received. Executing...")
    t0 = t

    # add dynamic obstacles and start them moving
    env.launch_dynamic_obstacles(t0=t0)

    # simulation loop
    while not rospy.is_shutdown() and t - t0 <= env.duration:
        # feedback is in the world frame
        v_prev = v
        q, v = env.robot.joint_states(add_noise=True, bodyframe=False)
        ros_interface.publish_feedback(t, q, v)

        # publish feedback on dynamic obstacles if using
        if use_projectile_interface:
            x_obs = env.dynamic_obstacle_state()
            r_obs = x_obs[:3]
            projectile_ros_interface.publish_pose(t, r_obs, Q_obs)
            projectile_ros_interface.publish_ground_truth(t, r_obs, x_obs[3:6])

        # commands are always in the body frame (to match the real robot)
        cmd_vel_world = env.robot.command_velocity(
            ros_interface.cmd_vel, bodyframe=True
        )

        if logger.ready(t):
            x = np.concatenate((q, v, a, x_obs))

            # log sim stuff
            r_ew_w, Q_we = env.robot.link_pose()
            v_ew_w, ω_ew_w = env.robot.link_velocity()
            r_ow_ws, Q_wos = env.object_poses()
            logger.append("ts", t - t0)
            logger.append("xs", x)
            logger.append("r_ew_ws", r_ew_w)
            logger.append("Q_wes", Q_we)
            logger.append("v_ew_ws", v_ew_w)
            logger.append("ω_ew_ws", ω_ew_w)
            logger.append("cmd_vels", cmd_vel_world)
            logger.append("r_ow_ws", r_ow_ws)
            logger.append("Q_wos", Q_wos)

            # log controller stuff
            r_ew_w_d, Q_we_d = ref.get_desired_pose(t)
            logger.append("r_ew_w_ds", r_ew_w_d)
            logger.append("Q_we_ds", Q_we_d)

        t, obs_reset = env.step(t, step_robot=False)

        # if obstacle is reset (i.e. instanstantly reset to a new state), we
        # have to reset the estimator
        if obs_reset:
            reset_projectile_pub.publish(Empty())

        ros_interface.publish_time(t)

    try:
        print(f"Min constraint value = {np.min(logger.data['balancing_constraints'])}")
    except:
        pass

    # save logged data
    if cli_args.log is not None:
        logger.save(timestamp, name=cli_args.log)

    if env.video_manager.save:
        print(f"Saved video to {env.video_manager.path}")

    # visualize data
    DataPlotter.from_logger(logger).plot_all(show=False)

    prop_cycle = plt.rcParams["axes.prop_cycle"]
    colors = prop_cycle.by_key()["color"]

    mpc_obs_ts = np.array(mpc_obs_listener.ts)
    mpc_obs_xs = np.array(mpc_obs_listener.xs)
    mpc_obs_us = np.array(mpc_obs_listener.us)

    mpc_plan_ts = np.array(mpc_plan_listener.ts)
    mpc_plan_xs = np.array(mpc_plan_listener.xs)
    mpc_plan_us = np.array(mpc_plan_listener.us)

    cmd_ts = np.array(cmd_listener.ts)
    cmd_xs = np.array(cmd_listener.xs)

    plt.figure()
    for i in range(env.robot.nq):
        plt.plot(mpc_plan_ts, mpc_plan_xs[:, i], label=f"qd_{i}", linestyle="--")
    for i in range(env.robot.nq):
        plt.plot(mpc_obs_ts, mpc_obs_xs[:, i], label=f"q_{i}", color=colors[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Joint position")
    plt.title("Planned vs. Estimated Joint Positions")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(env.robot.nv):
        plt.plot(
            mpc_plan_ts,
            mpc_plan_xs[:, env.robot.nq + i],
            label=f"vd_{i}",
            linestyle="--",
        )
    for i in range(env.robot.nv):
        plt.plot(
            mpc_obs_ts, mpc_obs_xs[:, env.robot.nq + i], label=f"v_{i}", color=colors[i]
        )
    for i in range(env.robot.nv):
        plt.plot(
            cmd_ts,
            cmd_xs[:, env.robot.nq + i],
            label=f"vc_{i}",
            color=colors[i],
            linestyle=":",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Joint velocity")
    plt.title("Planned vs. Estimated Joint Velocities")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(env.robot.nv):
        plt.plot(
            mpc_plan_ts,
            mpc_plan_xs[:, env.robot.nq + env.robot.nv + i],
            label=f"ad_{i}",
            linestyle="--",
        )
    for i in range(env.robot.nv):
        plt.plot(
            mpc_obs_ts,
            mpc_obs_xs[:, env.robot.nq + env.robot.nv + i],
            label=f"a_{i}",
            color=colors[i],
        )
    for i in range(env.robot.nv):
        plt.plot(
            cmd_ts,
            cmd_xs[:, env.robot.nq + env.robot.nv + i],
            label=f"ac_{i}",
            color=colors[i],
            linestyle=":",
        )
    plt.xlabel("Time [s]")
    plt.ylabel("Joint acceleration")
    plt.title("Planned vs. Estimated Joint Acceleration")
    plt.legend()
    plt.grid()

    plt.figure()
    for i in range(env.robot.nu):
        plt.plot(mpc_plan_ts, mpc_plan_us[:, i], label=f"ud_{i}", linestyle="--")
    for i in range(env.robot.nu):
        plt.plot(mpc_obs_ts, mpc_obs_us[:, i], label=f"u_{i}", color=colors[i])
    plt.xlabel("Time [s]")
    plt.ylabel("Joint jerk")
    plt.title("Planned vs. Commanded Input Jerk")
    plt.legend()
    plt.grid()

    plt.show()


if __name__ == "__main__":
    main()
