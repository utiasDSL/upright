#pragma once

#include <ostream>

#include <ocs2_core/reference/TargetTrajectories.h>

#include <upright_control/cost/InertialAlignmentCost.h>
#include <upright_control/dynamics/BaseType.h>
#include <upright_control/dynamics/Dimensions.h>
#include "upright_control/constraint/BoundedBalancingConstraints.h"
#include "upright_control/constraint/CollisionAvoidanceConstraint.h"
#include "upright_control/constraint/ObstacleConstraint.h"

namespace upright {

struct ControllerSettings {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    enum class Method {
        DDP,
        SQP,
    };

    Method method = Method::DDP;
    VecXd initial_state;
    Vec3d gravity;

    // Weights
    MatXd input_weight;
    MatXd state_weight;
    MatXd end_effector_weight;

    // Limits
    VecXd input_limit_lower;
    VecXd input_limit_upper;
    ocs2::scalar_t input_limit_mu = 1e-2;
    ocs2::scalar_t input_limit_delta = 1e-3;

    VecXd state_limit_lower;
    VecXd state_limit_upper;
    ocs2::scalar_t state_limit_mu = 1e-2;
    ocs2::scalar_t state_limit_delta = 1e-3;

    // We can linearize around a set of operating points instead of just using
    // a stationary trajectory.
    bool use_operating_points = false;
    // TODO this should be wrapped in a TargetTrajectories class
    // TargetTrajectories operating_trajectory;
    ocs2::scalar_array_t operating_times;
    ocs2::vector_array_t operating_states;
    ocs2::vector_array_t operating_inputs;

    // URDFs
    std::string robot_urdf_path;
    std::string obstacle_urdf_path;

    // OCS2 settings
    std::string ocs2_config_path;
    std::string lib_folder;

    // Robot settings
    RobotBaseType robot_base_type = RobotBaseType::Fixed;
    RobotDimensions dims;
    std::string end_effector_link_name;

    // Additional settings for constraints
    TrayBalanceSettings tray_balance_settings;
    InertialAlignmentSettings inertial_alignment_settings;
    DynamicObstacleSettings dynamic_obstacle_settings;
    CollisionAvoidanceSettings collision_avoidance_settings;
};

std::ostream& operator<<(std::ostream& out,
                         const ControllerSettings& settings) {
    out << "gravity = " << settings.gravity.transpose() << std::endl
        << "x0 = " << settings.initial_state.transpose() << std::endl
        << "input_weight = " << settings.input_weight << std::endl
        << "state_weight = " << settings.state_weight << std::endl
        << "end_effector_weight = " << settings.end_effector_weight << std::endl
        << "input_limit_lower = " << settings.input_limit_lower.transpose()
        << std::endl
        << "input_limit_upper = " << settings.input_limit_upper.transpose()
        << std::endl
        << "input_limit_mu = " << settings.input_limit_mu << std::endl
        << "input_limit_delta = " << settings.input_limit_delta << std::endl
        << "state_limit_lower = " << settings.state_limit_lower.transpose()
        << std::endl
        << "state_limit_upper = " << settings.state_limit_upper.transpose()
        << std::endl
        << "state_limit_mu = " << settings.state_limit_mu << std::endl
        << "state_limit_delta = " << settings.state_limit_delta << std::endl
        << "use_operating_points = " << settings.use_operating_points
        << std::endl
        << "robot_urdf_path = " << settings.robot_urdf_path << std::endl
        << "robot_base_type = "
        << robot_base_type_to_string(settings.robot_base_type) << std::endl
        << "end_effector_link_name = " << settings.end_effector_link_name
        << std::endl
        << "dims" << std::endl
        << settings.dims << std::endl
        << "tray_balance_settings" << std::endl
        << settings.tray_balance_settings << std::endl
        << "inertial_alignment_settings" << std::endl
        << settings.inertial_alignment_settings << std::endl;
    return out;
}

}  // namespace upright
