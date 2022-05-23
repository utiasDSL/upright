#!/usr/bin/env python3
import rospy
import ros_numpy
from ros_numpy.registry import converts_from_numpy
import numpy as np

import tray_balance_ocs2 as ctrl
import tray_balance_constraints as core

import upright_ros_interface.srv
from upright_msgs.msg import FloatArray, BoundedBalancedObject
from geometry_msgs.msg import Vector3

import IPython


@converts_from_numpy(FloatArray)
def convert(arr):
    msg = FloatArray()
    msg.shape = arr.shape
    msg.data = arr.flatten()
    return msg


def control_settings_to_response(settings):
    """convert the config to a response to send back"""
    resp = upright_ros_interface.srv.ParseControlSettingsResponse()

    resp.initial_state = settings.initial_state
    resp.gravity = ros_numpy.msgify(Vector3, settings.gravity)

    # cost weights
    resp.input_weight = ros_numpy.msgify(FloatArray, settings.input_weight)
    resp.state_weight = ros_numpy.msgify(FloatArray, settings.state_weight)
    resp.end_effector_weight = ros_numpy.msgify(
        FloatArray, settings.end_effector_weight
    )

    # input limits
    resp.input_limit_lower = settings.input_limit_lower
    resp.input_limit_upper = settings.input_limit_upper
    resp.input_limit_mu = settings.input_limit_mu
    resp.input_limit_delta = settings.input_limit_delta

    # state limits
    resp.state_limit_lower = settings.state_limit_lower
    resp.state_limit_upper = settings.state_limit_upper
    resp.state_limit_mu = settings.state_limit_mu
    resp.state_limit_delta = settings.state_limit_delta

    # operating points
    resp.use_operating_points = settings.use_operating_points
    operating_states = []
    operating_inputs = []
    for i in range(len(settings.operating_times)):
        resp.operating_times.append(settings.operating_times[i])
        operating_states.append(settings.operating_states[i])
        operating_inputs.append(settings.operating_inputs[i])

    resp.operating_states = ros_numpy.msgify(FloatArray, np.array(operating_inputs))
    resp.operating_inputs = ros_numpy.msgify(FloatArray, np.array(operating_states))

    # URDF paths
    resp.robot_urdf_path = settings.robot_urdf_path
    resp.obstacle_urdf_path = settings.obstacle_urdf_path

    # OCS2 paths
    resp.ocs2_config_path = settings.ocs2_config_path
    resp.lib_folder = settings.lib_folder

    # robot settings
    resp.robot_base_type = ctrl.bindings.robot_base_type_to_string(
        settings.robot_base_type
    )
    resp.end_effector_link_name = settings.end_effector_link_name
    resp.dims.q = settings.dims.q
    resp.dims.v = settings.dims.v
    resp.dims.x = settings.dims.x
    resp.dims.u = settings.dims.u

    # tray balance settings
    resp.tray_balance_settings.enabled = settings.tray_balance_settings.enabled
    resp.tray_balance_settings.normal_constraints_enabled = (
        settings.tray_balance_settings.constraints_enabled.normal
    )
    resp.tray_balance_settings.friction_constraints_enabled = (
        settings.tray_balance_settings.constraints_enabled.friction
    )
    resp.tray_balance_settings.zmp_constraints_enabled = (
        settings.tray_balance_settings.constraints_enabled.zmp
    )

    for obj in settings.tray_balance_settings.objects:
        obj_msg = BoundedBalancedObject()
        obj_msg.parameters = obj.get_parameters()
        resp.tray_balance_settings.objects.append(obj_msg)

    # resp.tray_balance_settings.constraint_type =  # TODO
    resp.tray_balance_settings.mu = settings.tray_balance_settings.mu
    resp.tray_balance_settings.delta = settings.tray_balance_settings.delta

    # inertial alignment settings
    resp.inertial_alignment_settings.enabled = (
        settings.inertial_alignment_settings.enabled
    )
    resp.inertial_alignment_settings.use_angular_acceleration = (
        settings.inertial_alignment_settings.use_angular_acceleration
    )
    resp.inertial_alignment_settings.weight = (
        settings.inertial_alignment_settings.weight
    )
    resp.inertial_alignment_settings.r_oe_e = ros_numpy.msgify(
        Vector3, settings.inertial_alignment_settings.r_oe_e
    )

    return resp


def parse_control_settings_cb(req):
    print("Received request.")

    # parse the config file
    config = core.parsing.load_config(req.config_path)
    ctrl_config = config["controller"]
    ctrl_wrapper = ctrl.parsing.ControllerConfigWrapper(ctrl_config)

    return control_settings_to_response(ctrl_wrapper.settings)


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
