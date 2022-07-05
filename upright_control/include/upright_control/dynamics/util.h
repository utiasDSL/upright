#pragma once

#include <Eigen/Eigen>

#include <upright_control/types.h>

namespace upright {

template <typename Scalar>
Mat2<Scalar> base_rotation_matrix(const VecX<Scalar>& state) {
    // clang-format off
    const auto theta = state(2);
    Mat2<Scalar> C_wb;
    C_wb << cos(theta), -sin(theta),
            sin(theta),  cos(theta);
    // clang-format on
    return C_wb;
}

}  // namespace upright
