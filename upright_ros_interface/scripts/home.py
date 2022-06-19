#!/usr/bin/env python2
"""Send the robot to the home configuration specified in the config file."""
import rospy
import numpy as np

import upright_core as core
from upright_core.logging import DataLogger, DataPlotter
import upright_cmd as cmd
from upright_ros_interface import ThingRobot

import IPython


# TODO: move to config file
HZ = 125


def main():
    rospy.init_node("home_node")

    args = cmd.cli.basic_arg_parser().parse_args()
    config = core.parsing.load_config(args.config)

    robot = ThingRobot(config["controller"]["robot"])
    rate = rospy.Rate(HZ)

    # wait until the robot is ready to go
    rospy.loginfo("Waiting for joint feedback.")
    while not robot.ready() and not rospy.is_shutdown():
        rate.sleep()

    rospy.loginfo("Robot is ready.")

    nq = nv = 9
    K = 1 * np.eye(nq)
    ε = 1e-2

    # control loop: run until converged to home position
    while not rospy.is_shutdown():
        q, _ = robot.joint_states()
        Δq = robot.home - q

        # error is small: we are done
        if (np.abs(Δq) < ε).all():
            break

        v_cmd = K @ Δq
        robot.command_velocity(v_cmd)
        rate.sleep()

    # alternative tracking version
    # T = 10
    # amp = 0.2
    # freq = 2
    # t = 0
    #
    # start = rospy.Time.now().to_sec()
    # while not rospy.is_shutdown():
    #     q, _ = robot.joint_states()
    #
    #     qd = np.copy(robot.home)
    #     vd = np.zeros(nv)
    #     qd[-1] = amp * (1 - np.cos(freq * t))
    #     vd[-1] = amp * freq * np.sin(freq * t)
    #
    #     v_cmd = K @ (qd - q) + vd
    #     robot.command_velocity(v_cmd)
    #
    #     t = rospy.Time.now().to_sec() - start
    #     if t > T:
    #         break
    #     rate.sleep()

    rospy.loginfo("Done.")


if __name__ == "__main__":
    main()
