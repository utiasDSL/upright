/******************************************************************************
Copyright (c) 2020, Farbod Farshidian. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 * Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

 * Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

 * Neither the name of the copyright holder nor the names of its
  contributors may be used to endorse or promote products derived from
  this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 ******************************************************************************/

#include <vector>

#include <geometry_msgs/Vector3.h>
#include <ros/ros.h>
#include <upright_msgs/FloatArray.h>

#include <ocs2_mpc/MPC_DDP.h>
#include <ocs2_ros_interfaces/mpc/MPC_ROS_Interface.h>
#include <ocs2_ros_interfaces/synchronized_module/RosReferenceManager.h>

#include <tray_balance_ocs2/ControllerSettings.h>
#include <tray_balance_ocs2/MobileManipulatorInterface.h>

#include <upright_ros_interface/ParseControlSettings.h>

#include "upright_ros_interface/mpc_node.h"

using namespace ocs2;
using namespace mobile_manipulator;

Eigen::Vector3d parse_vec3(const geometry_msgs::Vector3& vec) {
    return Eigen::Vector3d(vec.x, vec.y, vec.z);
}

Eigen::VectorXd parse_vector(std::vector<double>& vec) {
    return Eigen::Map<Eigen::VectorXd>(vec.data(), vec.size(), 1);
}

Eigen::MatrixXd parse_matrix(upright_msgs::FloatArray& msg) {
    return Eigen::Map<Eigen::MatrixXd>(msg.data.data(), msg.shape[0],
                                       msg.shape[1]);
}

vector_array_t parse_vector_array(upright_msgs::FloatArray& msg) {
    vector_array_t vec_array;
    size_t n = msg.shape[1];
    for (int i = 0; i < msg.shape[0]; ++i) {
        vec_array.push_back(
            Eigen::Map<Eigen::VectorXd>(&msg.data[i * n], n, 1));
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
        Eigen::VectorXd parameters = parse_vector(obj_msg.parameters);
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

// void print_controller_settings(const ControllerSettings& settings) {
//     std::cout << "gravity = " << settings.gravity.transpose() << std::endl
//               << "x0 = " << settings.initial_state.transpose() << std::endl
//               << "input_weight = " << settings.input_weight << std::endl
//               << "state_weight = " << settings.state_weight << std::endl
//               << "end_effector_weight = " << settings.end_effector_weight << std::endl
//               << "input_limit_lower = " << settings.input_limit_lower.transpose() << std::endl
//               << "input_limit_upper = " << settings.input_limit_upper.transpose() << std::endl
//               << "input_limit_mu = " << settings.input_limit_mu << std::endl
//               << "input_limit_delta = " << settings.input_limit_delta << std::endl;
// }

int main(int argc, char** argv) {
    const std::string robotName = "mobile_manipulator";

    if (argc < 2) {
        throw std::runtime_error("Config path is required.");
    }

    // Initialize ros node
    ros::init(argc, argv, "mpc_node");
    ros::NodeHandle nodeHandle;

    ros::service::waitForService("parse_control_settings");
    upright_ros_interface::ParseControlSettings settings_srv;
    settings_srv.request.config_path = std::string(argv[1]);
    if (!ros::service::call("parse_control_settings", settings_srv)) {
        throw std::runtime_error("Service call for control settings failed.");
    }

    // Robot interface
    ControllerSettings settings = parse_control_settings(settings_srv.response);
    std::cout << settings << std::endl;
    MobileManipulatorInterface interface(settings);

    // ROS ReferenceManager
    std::shared_ptr<ocs2::RosReferenceManager> rosReferenceManagerPtr(
        new ocs2::RosReferenceManager(robotName,
                                      interface.getReferenceManagerPtr()));
    rosReferenceManagerPtr->subscribe(nodeHandle);

    // MPC
    ocs2::MPC_DDP mpc(interface.mpcSettings(), interface.ddpSettings(),
                      interface.getRollout(),
                      interface.getOptimalControlProblem(),
                      interface.getInitializer());
    mpc.getSolverPtr()->setReferenceManager(rosReferenceManagerPtr);

    // Launch MPC ROS node
    MPC_ROS_Interface mpcNode(mpc, robotName);
    mpcNode.launchNodes(nodeHandle);

    // Successful exit
    return 0;
}
