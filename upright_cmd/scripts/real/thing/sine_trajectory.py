#!/usr/bin/env python3
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

import upright_cmd as cmd
import upright_control as ctrl
import upright_core as core
from upright_ros_interface import TrajectoryClient, UR10_JOINT_NAMES


def sinusoid(amplitude, frequency, time):
    f = 2 * np.pi * frequency

    q = amplitude * (1 - np.cos(f * time))
    v = amplitude * f * np.sin(f * time)
    a = amplitude * f ** 2 * np.cos(f * time)

    return q, v, a


if __name__ == "__main__":
    rospy.init_node("sine")

    argparser = cmd.cli.basic_arg_parser()
    cli_args = argparser.parse_args()

    config = core.parsing.load_config(cli_args.config)
    settings = ctrl.wrappers.ControllerSettings(config["controller"])
    mapping = ctrl.trajectory.StateInputMapping(settings.dims)

    q0, _, _ = mapping.xu2qva(settings.initial_state)

    duration = 10.0
    amplitude = 0.2
    frequency = 0.2
    times = np.linspace(0, duration, 100)

    # build the trajectory
    trajectory = JointTrajectory()
    trajectory.joint_names = UR10_JOINT_NAMES
    for i in range(len(times)):
        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(times[i])

        q, v, a = sinusoid(amplitude, frequency, times[i])
        point.positions = list(q0 + [0, 0, q, 0, 0, 0])
        point.velocities = [0, 0, v, 0, 0, 0]
        point.accelerations = [0, 0, a, 0, 0, 0]

        trajectory.points.append(point)

    # send the trajectory
    client = TrajectoryClient("scaled_vel_joint_traj_controller")
    client.send_joint_trajectory(trajectory)
