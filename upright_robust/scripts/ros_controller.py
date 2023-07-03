#!/usr/bin/env python3
import os
import resource

import numpy as np
import rospy

import mobile_manipulation_central as mm
import upright_core as core
import upright_cmd as cmd
import upright_robust as rob


VERBOSE = False

RATE = 125  # Hz
TIMESTEP = 1. / RATE

# TODO adjust this
MAX_JOINT_VELOCITY = np.array([0.5, 0.5, 0.5, 1, 1, 1, 1, 1, 1])


def main():
    # os.nice(-10)
    # resource.setrlimit(resource.RLIMIT_RTPRIO, (98, 98))

    np.set_printoptions(precision=6, suppress=True)

    # load configuration
    cli_args = cmd.cli.sim_arg_parser().parse_args()
    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]

    # parse controller model
    model = rob.RobustControllerModel(ctrl_config, TIMESTEP)
    kp, kv = model.kp, model.kv
    controller = model.controller
    robot_model = model.robot

    # start ROS
    rospy.init_node("upright_robust_controller", disable_signals=True)
    robot_interface = mm.MobileManipulatorROSInterface()
    signal_handler = mm.RobotSignalHandler(robot_interface)

    # wait until robot feedback has been received
    rate = rospy.Rate(RATE)
    while not rospy.is_shutdown() and not robot_interface.ready():
        rate.sleep()

    # desired waypoint
    q, v = robot_interface.q, robot_interface.v
    robot_model.forward(q, v)
    r_ew_w_0, _ = robot_model.link_pose()
    rd = r_ew_w_0 + ctrl_config["waypoints"][0]["position"]
    vd = np.zeros(3)
    ad = np.zeros(3)

    # control loop
    t = rospy.Time.now().to_sec()
    while not rospy.is_shutdown():
        last_t = t
        t = rospy.Time.now().to_sec()
        dt = t - last_t

        # robot feedback
        q, v = robot_interface.q, robot_interface.v

        # current EE state
        robot_model.forward(q, v)
        r_ew_w, C_we = robot_model.link_pose(rotation_matrix=True)
        v_ew_w, _ = robot_model.link_velocity()

        # commanded EE acceleration
        a_ew_w_cmd = kp * (rd - r_ew_w) + kv * (vd - v_ew_w) + ad

        # compute command taking balancing into account
        t0 = rospy.Time.now().to_sec()
        u, A_e = controller.solve(q, v, a_ew_w_cmd)
        t1 = rospy.Time.now().to_sec()
        rospy.loginfo(f"controller took {1000 * (t1 - t0)} ms")

        # integrate acceleration command to get new commanded velocity from the
        # current velocity
        cmd_vel = v + dt * u
        cmd_vel = mm.bound_array(cmd_vel, lb=-MAX_JOINT_VELOCITY, ub=MAX_JOINT_VELOCITY)
        robot_interface.publish_cmd_vel(cmd_vel)

        if VERBOSE:
            A_w = block_diag(C_we, C_we) @ A_e

        rate.sleep()

    robot_interface.brake()


if __name__ == "__main__":
    main()
