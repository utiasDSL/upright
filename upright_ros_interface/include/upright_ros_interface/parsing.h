#pragma once

#include <Eigen/Eigen>
#include <vector>

#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>

#include <pybind11/pybind11.h>
#include <pybind11/embed.h>

#include <upright_control/controller_settings.h>
#include <upright_control/types.h>
#include <upright_msgs/FloatArray.h>
#include <upright_ros_interface/ParseControlSettings.h>

namespace py = pybind11;

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

// TODO: ideally, we could just parse everything on C++, instead of doing it in
// Python and casting the object back like this
ControllerSettings parse_control_settings(const std::string& config_path) {
    // Note pybind11_catkin uses an older version of pybind, so we use
    // py::module rather than py::module_
    py::scoped_interpreter guard{};
    py::object upright_control = py::module::import("upright_control");
    py::object PyControllerSettings = upright_control.attr("wrappers").attr("ControllerSettings");
    return PyControllerSettings.attr("cpp")(config_path).cast<ControllerSettings>();
}

ocs2::TargetTrajectories parse_target_trajectory(const std::string& config_path, const VecXd& x0) {
    py::scoped_interpreter guard{};
    py::object upright_control = py::module::import("upright_control");
    py::object PyTargetTrajectories = upright_control.attr("wrappers").attr("TargetTrajectories");
    return PyTargetTrajectories.attr("from_config_file")(config_path, x0).cast<ocs2::TargetTrajectories>();
}

ControllerSettings parse_control_settings_old(
    upright_ros_interface::ParseControlSettings::Response& resp) {
    ControllerSettings settings;

    settings.solver_method =
        ControllerSettings::solver_method_from_string(resp.solver_method);
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

    // tracking rate
    settings.rate = resp.rate;

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
    settings.balancing_settings.enabled = resp.balancing_settings.enabled;
    settings.balancing_settings.constraints_enabled.normal =
        resp.balancing_settings.normal_constraints_enabled;
    settings.balancing_settings.constraints_enabled.friction =
        resp.balancing_settings.friction_constraints_enabled;
    settings.balancing_settings.constraints_enabled.zmp =
        resp.balancing_settings.zmp_constraints_enabled;
    for (auto& obj_msg : resp.balancing_settings.objects) {
        VecXd parameters = parse_vector(obj_msg.parameters);
        BoundedBalancedObject<double> obj =
            BoundedBalancedObject<double>::from_parameters(parameters);
        settings.balancing_settings.objects.insert({obj_msg.name, obj});
    }
    settings.balancing_settings.mu = resp.balancing_settings.mu;
    settings.balancing_settings.delta = resp.balancing_settings.delta;

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
