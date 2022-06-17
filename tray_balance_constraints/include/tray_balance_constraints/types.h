#pragma once

#include <Eigen/Eigen>

namespace upright {

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

const double NEAR_ZERO = 1e-8;

}  // namespace upright

