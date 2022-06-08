#!/usr/bin/env python3
import numpy as np
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from upright_ros_interface import TrajectoryClient, UR10_JOINT_NAMES


def sinusoid(amplitude, frequency, time):
    f = 2 * np.pi * frequency

    q = amplitude * (1 - np.cos(f * time))
    v = amplitude * f * np.sin(f * time)
    a = amplitude * f ** 2 * np.cos(f * time)

    return q, v, a


if __name__ == "__main__":
    # initial position
    q0 = np.array([1.57, -0.785, 1.57, -0.785, 1.57, 0])

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
