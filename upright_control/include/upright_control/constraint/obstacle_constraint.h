#pragma once

#include <upright_control/constraint/constraint_type.h>
#include <upright_control/reference_trajectory.h>
#include <upright_control/types.h>

namespace upright {

// TODO deprecated
template <typename Scalar>
struct CollisionSphere {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    // Empty constructor required for binding as opaque vector type.
    CollisionSphere() {
        name = "";
        parent_frame_name = "";
        offset = Vec3<Scalar>::Zero();
        radius = 0;
    }

    CollisionSphere(const std::string& name,
                    const std::string& parent_frame_name,
                    const Vec3<Scalar>& offset, const Scalar radius)
        : name(name),
          parent_frame_name(parent_frame_name),
          offset(offset),
          radius(radius) {}

    // Name of this collision sphere.
    std::string name;

    // Name of the robot joint this collision sphere is attached to.
    std::string parent_frame_name;

    // Offset from that joint (in the joint's frame).
    Vec3<Scalar> offset;

    // Radius of this collision sphere.
    Scalar radius;
};

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

    ConstraintType constraint_type = ConstraintType::Soft;

    // Relaxed barrier function parameters
    ocs2::scalar_t mu = 1e-2;
    ocs2::scalar_t delta = 1e-3;

    // URDF of static obstacles
    std::string obstacle_urdf_path;

    // List of dynamic obstacles
    std::vector<DynamicObstacle> dynamic_obstacles;

    // Extra collision spheres to attach to the robot body for collision
    // avoidance.
    std::vector<CollisionSphere<ocs2::scalar_t>> extra_spheres;
};

}  // namespace upright
