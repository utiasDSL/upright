#pragma once

#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

struct DynamicObstacleMode {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    ocs2::scalar_t time;
    Vec3d position;
    Vec3d velocity;
    Vec3d acceleration;

    VecXd state() const {
        VecXd state(9);
        state << position, velocity, acceleration;
        return state;
    }
};

struct DynamicObstacle {
    std::string name;
    ocs2::scalar_t radius = 0;
    std::vector<DynamicObstacleMode> modes;
};

struct ObstacleSettings {
    bool enabled = false;

    // List of pairs of collision objects to check
    std::vector<std::pair<std::string, std::string>> collision_link_pairs;

    // Minimum distance allowed between collision objects
    ocs2::scalar_t minimum_distance = 0;

    // URDF of static obstacles
    std::string obstacle_urdf_path;

    // List of dynamic obstacles
    std::vector<DynamicObstacle> dynamic_obstacles;
};

}  // namespace upright
