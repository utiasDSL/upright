import rospy

import upright_cmd as cmd

from upright_ros_interface.srv import ParseControlSettings


def main():
    cli_args = cmd.cli.sim_arg_parser().parse_args()

    rospy.wait_for_service("parse_control_settings")
    parse_control_settings_service = rospy.ServiceProxy(
        "parse_control_settings", ParseControlSettings
    )

    try:
        resp = parse_control_settings_service(cli_args.config)
    except rospy.ServiceException as e:
        print("Service did not process request: " + str(e))
        return 1

    print(resp)


if __name__ == "__main__":
    main()
