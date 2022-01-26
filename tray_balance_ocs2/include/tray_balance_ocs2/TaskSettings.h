#pragma once

#include "tray_balance_ocs2/constraint/tray_balance/TrayBalanceSettings.h"
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"
#include "tray_balance_ocs2/constraint/ObstacleConstraint.h"

namespace ocs2 {
namespace mobile_manipulator {

struct TaskSettings {
    enum class Method {
        DDP,
        SQP,
    };

    Method method = Method::DDP;
    vector_t initial_state;

    TrayBalanceSettings tray_balance_settings;
    DynamicObstacleSettings dynamic_obstacle_settings;
    CollisionAvoidanceSettings collision_avoidance_settings;
};

}  // namespace mobile_manipulator
}  // namespace ocs
