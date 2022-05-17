import rospy

import tray_balance_constraints as core

import upright_ros_interface.srv

import IPython


def parse_control_settings_cb(req):
    print("Received request.")

    config = core.parsing.load_config(cli_args.config)
    ctrl_config = config["controller"]
    ctrl_wrapper = ctrl.parsing.ControllerConfigWrapper(ctrl_config)

    resp = upright_ros_interface.srv.ParseControlSettingsResponse()
    return resp


def main():
    rospy.init_node("parsing_server")
    service = rospy.Service(
        "parse_control_settings",
        upright_ros_interface.srv.ParseControlSettings,
        parse_control_settings_cb,
    )
    print("Spinning service.")
    service.spin()


if __name__ == "__main__":
    main()
