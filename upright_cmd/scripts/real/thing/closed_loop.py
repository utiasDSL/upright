#!/usr/bin/env python3
"""PyBullet simulation using the bounded balancing constraints"""
import argparse
import time
import datetime
import sys
import os
from pathlib import Path

import rospy
import numpy as np
import tray_balance_constraints as core
import tray_balance_ocs2 as ctrl
import upright_cmd as cmd
from upright_ros_interface.real import ROSRealInterface

import IPython


def main():
    np.set_printoptions(precision=3, suppress=True)

    cli_args = cmd.cli.basic_arg_parser().parse_args()

    # load configuration
    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]

    model = ctrl.manager.ControllerModel.from_config(ctrl_config)
    x0 = model.settings.initial_state
    u0 = np.zeros(model.robot.dims.u)

    # reference pose trajectory
    model.update(x=model.settings.initial_state)
    r_ew_w, Q_we = model.robot.link_pose()
    ref = ctrl.wrappers.TargetTrajectories.from_config(ctrl_config, r_ew_w, Q_we, u0)

    # setup the ROS interface
    ros_interface = ROSRealInterface("mobile_manipulator", "scaled_vel_joint_traj_controller")
    ros_interface.reset_mpc(ref)

    # give the MPC the initial conditions
    t0 = rospy.Time.now().to_sec()
    ros_interface.publish_observation(t0, x0, u0)

    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        ros_interface.send_trajectory()
        rate.sleep()

    # rospy.spin()


if __name__ == "__main__":
    main()
