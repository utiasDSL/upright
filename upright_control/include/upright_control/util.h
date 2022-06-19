#pragma once

#include <Eigen/Eigen>

namespace upright {

// TODO probably move directly to dynamics
template <typename Scalar>
Eigen::Matrix<Scalar, 2, 2> base_rotation_matrix(
    const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& state) {
    // clang-format off
    const auto theta = state(2);
    Eigen::Matrix<Scalar, 2, 2> C_wb;
    C_wb << cos(theta), -sin(theta),
            sin(theta),  cos(theta);
    // clang-format on
    return C_wb;
}

}  // namespace upright
