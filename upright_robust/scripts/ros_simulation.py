#!/usr/bin/env python3
"""Closed-loop upright reactive simulation using Pybullet."""
import datetime
import signal
import sys
import time

import rospy
import numpy as np
import pybullet as pyb
from pyb_utils.frame import debug_frame_world
import matplotlib.pyplot as plt

import mobile_manipulation_central as mm
import upright_sim as sim
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob

import IPython


def sigint_handler(sig, frame):
    print("Ctrl-C pressed: exiting.")
    pyb.disconnect()
    sys.exit(0)


def main():
    np.set_printoptions(precision=6, suppress=True)
    signal.signal(signal.SIGINT, sigint_handler)

    cli_args = cmd.cli.sim_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    sim_config = config["simulation"]
    ctrl_config = config["controller"]
    # log_config = config["logging"]

    timestamp = datetime.datetime.now()
    env = sim.simulation.UprightSimulation(
        config=sim_config,
        timestamp=timestamp,
        video_name=cli_args.video,
        extra_gui=sim_config.get("extra_gui", False),
    )
    env.settle(5.0)

    # goal position
    # TODO need initial position
    # model = rob.RobustControllerModel(ctrl_config, env.timestep)
    # model.robot.forward(q, v)
    # r_ew_w_0, Q_we_0 = robot.link_pose()
    # r_ew_w_d = r_ew_w_0 + ctrl_config["waypoints"][0]["position"]
    # debug_frame_world(0.2, list(r_ew_w_d), orientation=Q_we_0, line_width=3)

    rospy.init_node("upright_ros_simulation")
    robot_interface = mm.SimulatedMobileManipulatorROSInterface()

    # TODO
    robot_interface.base.cmd_vel = np.zeros(3)
    robot_interface.arm.cmd_vel = np.zeros(6)

    t = 0
    robot_interface.publish_time(t)
    while not rospy.is_shutdown():
        q, v = env.robot.joint_states(add_noise=True, bodyframe=False)
        robot_interface.publish_feedback(t, q, v)

        # send along the commanded the velocity
        cmd_vel_world = env.robot.command_velocity(
            robot_interface.cmd_vel, bodyframe=True
        )

        t = env.step(t, step_robot=False)[0]
        robot_interface.publish_time(t)

        # NOTE: we can't use the rate here since the sim is in charge of time
        time.sleep(env.timestep)


if __name__ == "__main__":
    main()
