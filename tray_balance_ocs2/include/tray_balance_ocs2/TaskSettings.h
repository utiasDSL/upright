#pragma once

#include "tray_balance_ocs2/constraint/tray_balance/TrayBalanceSettings.h"
#include "tray_balance_ocs2/constraint/CollisionAvoidanceConstraint.h"

namespace ocs2 {
namespace mobile_manipulator {

struct TaskSettings {
    enum class Method {
        DDP,
        SQP,
    };

    Method method = Method::DDP;

    bool dynamic_obstacle_enabled = false;

    TrayBalanceSettings tray_balance_settings;
    CollisionAvoidanceSettings collision_avoidance_settings;
};

}  // namespace mobile_manipulator
}  // namespace ocs
