#pragma once

#include "tray_balance_ocs2/constraint/balancing/BalancingSettings.h"
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"
#include "tray_balance_ocs2/constraint/ObstacleConstraint.h"

namespace ocs2 {
namespace mobile_manipulator {

struct ControllerSettings {
    enum class Method {
        DDP,
        SQP,
    };

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

    // URDFs
    std::string robot_urdf_path;
    std::string obstacle_urdf_path;

    // OCS2 settings
    std::string ocs2_config_path;

    // Additional settings for constraints
    TrayBalanceSettings tray_balance_settings;
    DynamicObstacleSettings dynamic_obstacle_settings;
    CollisionAvoidanceSettings collision_avoidance_settings;
};

}  // namespace mobile_manipulator
}  // namespace ocs
