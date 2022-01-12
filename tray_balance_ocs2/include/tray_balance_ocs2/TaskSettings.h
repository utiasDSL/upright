#pragma once

namespace ocs2 {
namespace mobile_manipulator {

struct TaskSettings {
    bool tray_balance_enabled = false;
    bool dynamic_obstacle_enabled = false;
    bool collision_avoidance_enabled = false;
};

}  // namespace mobile_manipulator
}  // namespace ocs
