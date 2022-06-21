#pragma once

#include <Eigen/Eigen>
#include <vector>

#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>

#include <upright_control/ControllerSettings.h>
#include <upright_control/types.h>
#include <upright_msgs/FloatArray.h>
#include <upright_ros_interface/ParseControlSettings.h>

namespace upright {

Vec3d parse_vec3(const geometry_msgs::Vector3& vec) {
    return Vec3d(vec.x, vec.y, vec.z);
}

VecXd parse_vector(std::vector<double>& vec) {
    return Eigen::Map<VecXd>(vec.data(), vec.size(), 1);
}

MatXd parse_matrix(upright_msgs::FloatArray& msg) {
    return Eigen::Map<MatXd>(msg.data.data(), msg.shape[0], msg.shape[1]);
}

ocs2::vector_array_t parse_vector_array(upright_msgs::FloatArray& msg) {
    ocs2::vector_array_t vec_array;
    size_t n = msg.shape[1];
    for (int i = 0; i < msg.shape[0]; ++i) {
        vec_array.push_back(Eigen::Map<VecXd>(&msg.data[i * n], n, 1));
    }
    return vec_array;
}

ControllerSettings parse_control_settings(
    upright_ros_interface::ParseControlSettings::Response& resp) {
    ControllerSettings settings;
    settings.gravity = parse_vec3(resp.gravity);
    settings.initial_state = parse_vector(resp.initial_state);

    // cost weights
    settings.input_weight = parse_matrix(resp.input_weight);
    settings.state_weight = parse_matrix(resp.state_weight);
    settings.end_effector_weight = parse_matrix(resp.end_effector_weight);

    // input limits
    settings.input_limit_lower = parse_vector(resp.input_limit_lower);
    settings.input_limit_upper = parse_vector(resp.input_limit_upper);
    settings.input_limit_mu = resp.input_limit_mu;
    settings.input_limit_delta = resp.input_limit_delta;

    // state limits
    settings.state_limit_lower = parse_vector(resp.state_limit_lower);
    settings.state_limit_upper = parse_vector(resp.state_limit_upper);
    settings.state_limit_mu = resp.state_limit_mu;
    settings.state_limit_delta = resp.state_limit_delta;

    // tracking gain
    settings.Kp = parse_matrix(resp.Kp);

    // operating points
    settings.use_operating_points = resp.use_operating_points;
    settings.operating_times = resp.operating_times;
    settings.operating_states = parse_vector_array(resp.operating_states);
    settings.operating_inputs = parse_vector_array(resp.operating_inputs);

    // URDF paths
    settings.robot_urdf_path = resp.robot_urdf_path;
    settings.obstacle_urdf_path = resp.obstacle_urdf_path;

    // OCS2 paths
    settings.ocs2_config_path = resp.ocs2_config_path;
    settings.lib_folder = resp.lib_folder;

    // robot settings
    settings.robot_base_type =
        robot_base_type_from_string(resp.robot_base_type);
    settings.end_effector_link_name = resp.end_effector_link_name;
    settings.dims.q = resp.dims.q;
    settings.dims.v = resp.dims.v;
    settings.dims.x = resp.dims.x;
    settings.dims.u = resp.dims.u;

    // tray balance settings
    settings.tray_balance_settings.enabled = resp.tray_balance_settings.enabled;
    settings.tray_balance_settings.constraints_enabled.normal =
        resp.tray_balance_settings.normal_constraints_enabled;
    settings.tray_balance_settings.constraints_enabled.friction =
        resp.tray_balance_settings.friction_constraints_enabled;
    settings.tray_balance_settings.constraints_enabled.zmp =
        resp.tray_balance_settings.zmp_constraints_enabled;
    for (auto& obj_msg : resp.tray_balance_settings.objects) {
        VecXd parameters = parse_vector(obj_msg.parameters);
        BoundedBalancedObject<double> obj =
            BoundedBalancedObject<double>::from_parameters(parameters);
        settings.tray_balance_settings.objects.push_back(obj);
    }
    settings.tray_balance_settings.mu = resp.tray_balance_settings.mu;
    settings.tray_balance_settings.delta = resp.tray_balance_settings.delta;

    // inertial alignment settings
    settings.inertial_alignment_settings.enabled =
        resp.inertial_alignment_settings.enabled;
    settings.inertial_alignment_settings.use_angular_acceleration =
        resp.inertial_alignment_settings.use_angular_acceleration;
    settings.inertial_alignment_settings.weight =
        resp.inertial_alignment_settings.weight;
    settings.inertial_alignment_settings.r_oe_e =
        parse_vec3(resp.inertial_alignment_settings.r_oe_e);

    return settings;
}

}  // namespace upright
