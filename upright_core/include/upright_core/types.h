#pragma once

#include <Eigen/Eigen>

namespace upright {

const double NEAR_ZERO = 1e-8;

template <typename Scalar>
using VecX = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using MatX = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename Scalar>
using Vec2 = Eigen::Matrix<Scalar, 2, 1>;

template <typename Scalar>
using Mat2 = Eigen::Matrix<Scalar, 2, 2>;

template <typename Scalar>
using Vec3 = Eigen::Matrix<Scalar, 3, 1>;

template <typename Scalar>
using Mat3 = Eigen::Matrix<Scalar, 3, 3>;

template <typename Scalar>
using Mat23 = Eigen::Matrix<Scalar, 2, 3>;

template <typename Scalar>
struct Pose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Mat3<Scalar> orientation;
    Vec3<Scalar> position;

    static Pose<Scalar> Zero() {
        Pose<Scalar> pose;
        pose.orientation = Mat3<Scalar>::Identity();
        pose.position = Vec3<Scalar>::Zero();
        return pose;
    }
};

template <typename Scalar>
struct Twist {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3<Scalar> linear;
    Vec3<Scalar> angular;

    static Twist<Scalar> Zero() {
        Twist<Scalar> twist;
        twist.linear = Vec3<Scalar>::Zero();
        twist.angular = Vec3<Scalar>::Zero();
        return twist;
    }
};

template <typename Scalar>
struct Wrench {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Vec3<Scalar> force;
    Vec3<Scalar> torque;

    static Wrench<Scalar> Zero() {
        Wrench<Scalar> wrench;
        wrench.force = Vec3<Scalar>::Zero();
        wrench.torque = Vec3<Scalar>::Zero();
        return wrench;
    }
};

template <typename Scalar>
struct RigidBodyState {
    Pose<Scalar> pose;
    Twist<Scalar> velocity;
    Twist<Scalar> acceleration;

    static RigidBodyState<Scalar> Zero() {
        RigidBodyState<Scalar> state;
        state.pose = Pose<Scalar>::Zero();
        state.velocity = Twist<Scalar>::Zero();
        state.acceleration = Twist<Scalar>::Zero();
        return state;
    }
};

}  // namespace upright

