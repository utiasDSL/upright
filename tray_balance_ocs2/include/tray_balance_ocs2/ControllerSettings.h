#pragma once

#include <ocs2_core/initialization/OperatingPoints.h>

#include <tray_balance_ocs2/cost/InertialAlignmentCost.h>
#include <tray_balance_ocs2/dynamics/BaseType.h>
#include <tray_balance_ocs2/dynamics/Dimensions.h>
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"
#include "tray_balance_ocs2/constraint/ObstacleConstraint.h"
#include "tray_balance_ocs2/constraint/balancing/BalancingSettings.h"

namespace ocs2 {
namespace mobile_manipulator {

struct ControllerSettings {
    enum class Method {
        DDP,
        SQP,
    };

    void set_gravity(const Vec3<scalar_t>& gravity) {
        tray_balance_settings.bounded_config.gravity = gravity;
        inertial_alignment_settings.gravity = gravity;
    }

    Method method = Method::DDP;
    vector_t initial_state;

    // Weights
    matrix_t input_weight;
    matrix_t state_weight;
    matrix_t end_effector_weight;

    // Limits
    vector_t input_limit_lower;
    vector_t input_limit_upper;
    scalar_t input_limit_mu = 1e-2;
    scalar_t input_limit_delta = 1e-3;

    vector_t state_limit_lower;
    vector_t state_limit_upper;
    scalar_t state_limit_mu = 1e-2;
    scalar_t state_limit_delta = 1e-3;

    // We can linearize around a set of operating points instead of just using
    // a stationary trajectory.
    bool use_operating_points = false;
    scalar_array_t operating_times;
    vector_array_t operating_states;
    vector_array_t operating_inputs;

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

}  // namespace mobile_manipulator
}  // namespace ocs2
