#pragma once

#include <Eigen/Eigen>

namespace upright {

const double NEAR_ZERO = 1e-8;

struct BalanceConstraintsEnabled {
    bool normal = true;
    bool friction = true;
    bool zmp = true;
};

template <typename Scalar>
using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

// TODO: replace above with below versions
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
struct Pose {
    Mat3<Scalar> orientation;
    Vec3<Scalar> position;
};

template <typename Scalar>
struct Twist {
    Vec3<Scalar> linear;
    Vec3<Scalar> angular;
};

template <typename Scalar>
struct Wrench {
    Vec3<Scalar> force;
    Vec3<Scalar> torque;
};

template <typename Scalar>
struct RigidBodyState {
    Pose<Scalar> pose;
    Twist<Scalar> velocity;
    Twist<Scalar> acceleration;
};

}  // namespace upright

