# Load a trajectory from an NPZ file and send it using the action client.
import rospy
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from upright_ros_interface import TrajectoryClient, UR10_JOINT_NAMES
import upright_core as core
import upright_cmd as cmd
import upright_control as ctrl


if __name__ == "__main__":
    argparser = cmd.cli.basic_arg_parser()
    argparser.add_argument(
        "trajectory_file", help="NPZ file to load the trajectory from."
    )
    cli_args = argparser.parse_args()

    config = core.parsing.load_config(cli_args.config)
    settings = ctrl.wrappers.ControllerSettings(config["controller"])
    mapping = ctrl.trajectory.StateInputMapping(settings.dims)
    trajectory = ctrl.trajectory.StateInputTrajectory.load(cli_args.trajectory_file)

    msg = JointTrajectory()
    msg.joint_names = UR10_JOINT_NAMES
    for i in range(len(trajectory)):
        t, x, u = trajectory[i]
        q, v, a = mapping.xu2qva(x, u)

        point = JointTrajectoryPoint()
        point.time_from_start = rospy.Duration(t)
        point.positions = q
        point.velocities = v
        point.accelerations = a

        # point.effort = u  # jerk

        msg.points.append(point)

    client = TrajectoryClient("scaled_vel_joint_traj_controller")
    client.send_joint_trajectory(msg)
