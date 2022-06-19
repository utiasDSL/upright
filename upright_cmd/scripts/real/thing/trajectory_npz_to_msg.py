import rospy
import upright_core as core
import upright_cmd as cmd
import upright_control as ctrl

from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint


def main():
    argparser = cmd.cli.sim_arg_parser()
    argparser.add_argument(
        "trajectory_file", help="NPZ file to load the trajectory from."
    )
    argparser.add_argument("topic", help="ROS topic to publish to.")
    cli_args = argparser.parse_args()

    config = core.parsing.load_config(cli_args.config)
    settings = ctrl.wrappers.ControllerSettings(config["controller"])
    mapping = ctrl.trajectory.StateInputMapping(settings.dims)
    trajectory = ctrl.trajectory.StateInputTrajectory.load(cli_args.trajectory_file)

    # convert to message
    msg = JointTrajectory()
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

    # publish the message
    rospy.init_node("trajectory_publisher")
    pub = rospy.Publisher(cli_args.topic, JointTrajectory, queue_size=1)
    rospy.sleep(1.0)
    pub.publish(msg)

    print(f"Published trajectory to {cli_args.topic}.")


if __name__ == "__main__":
    main()
