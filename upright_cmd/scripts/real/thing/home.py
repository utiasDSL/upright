#!/usr/bin/env python3
import sys
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from upright_ros_interface import TrajectoryClient, UR10_JOINT_NAMES
import tray_balance_constraints as core
import upright_cmd as cmd
import tray_balance_ocs2 as ctrl


if __name__ == "__main__":
    rospy.init_node("home")

    argparser = cmd.cli.basic_arg_parser()
    argparser.add_argument(
        "duration", type=float, help="Duration of homing trajectory."
    )
    cli_args = argparser.parse_args()

    if cli_args.duration < 3.0:
        print("Home trajectory duration should be at least 3 seconds.")
        sys.exit(1)

    config = core.parsing.load_config(cli_args.config)
    settings = ctrl.wrappers.ControllerSettings(config["controller"])
    mapping = ctrl.trajectory.StateInputMapping(settings.dims)

    q0, _, _ = mapping.xu2qva(settings.initial_state)

    trajectory = JointTrajectory()
    trajectory.joint_names = UR10_JOINT_NAMES
    point = JointTrajectoryPoint()
    point.time_from_start = rospy.Duration(cli_args.duration)
    point.positions = q0
    trajectory.points.append(point)

    client = TrajectoryClient("scaled_vel_joint_traj_controller")
    client.send_joint_trajectory(trajectory)
